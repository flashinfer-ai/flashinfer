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
#define SMEM_TREE_OFF 0
#define SMEM_TREE_STAGE_BYTES 544
#define SMEM_TREE_STRIDE 544
#define SMEM_QCOL_OFF 544
#define SMEM_QCOL_STAGE_BYTES 32
#define SMEM_QCOL_STRIDE 32
#define SMEM_FLAGW_OFF 576
#define SMEM_FLAGW_STAGE_BYTES 16
#define SMEM_FLAGW_STRIDE 16
#define SMEM_TOTAL 640
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

extern "C" {

__global__ __launch_bounds__(32) void
kernel_cake_hopper_msa_1a31475a824014e52fd6(unsigned int* __restrict__ S, int* __restrict__ nvp, int* __restrict__ out, int tiles, int total_q, int num_heads, int num_chunks, int fb, int fe)
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
    float* tree = reinterpret_cast<float*>(smem_raw + 0);
    const int tree_addr = smem + 0;
    unsigned int* qcol = reinterpret_cast<unsigned int*>(smem_raw + 544);
    const int qcol_addr = smem + 544;
    unsigned int* flagw = reinterpret_cast<unsigned int*>(smem_raw + 576);
    const int flagw_addr = smem + 576;

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
    int t0_1 = w * 16;
    int t0_2 = t0_1;
    float sc = __uint_as_float(cb[0]);
    float _fmax_0 = fmaxf(sc, -1.7014118346046923e+38f);
    sc = _fmax_0;
    float _min_0 = fminf(sc, 1.7014118346046923e+38f);
    sc = _min_0;
    sc = sc;
    float sc_3 = sc;
    unsigned int key = __as_u32(sc_3) & 4294967232u | (unsigned int)t0_2;
    kb[0] = __uint_as_float(key);
    float sc_4 = __uint_as_float(cb[1]);
    float _fmax_1 = fmaxf(sc_4, -1.7014118346046923e+38f);
    sc_4 = _fmax_1;
    float _min_1 = fminf(sc_4, 1.7014118346046923e+38f);
    sc_4 = _min_1;
    sc_4 = sc_4;
    float sc_5 = sc_4;
    unsigned int key_6 = __as_u32(sc_5) & 4294967232u | (unsigned int)(t0_2 + 1);
    kb[1] = __uint_as_float(key_6);
    float sc_7 = __uint_as_float(cb[2]);
    float _fmax_2 = fmaxf(sc_7, -1.7014118346046923e+38f);
    sc_7 = _fmax_2;
    float _min_2 = fminf(sc_7, 1.7014118346046923e+38f);
    sc_7 = _min_2;
    sc_7 = sc_7;
    float sc_8 = sc_7;
    unsigned int key_9 = __as_u32(sc_8) & 4294967232u | (unsigned int)(t0_2 + 2);
    kb[2] = __uint_as_float(key_9);
    float sc_10 = __uint_as_float(cb[3]);
    float _fmax_3 = fmaxf(sc_10, -1.7014118346046923e+38f);
    sc_10 = _fmax_3;
    float _min_3 = fminf(sc_10, 1.7014118346046923e+38f);
    sc_10 = _min_3;
    sc_10 = sc_10;
    float sc_11 = sc_10;
    unsigned int key_12 = __as_u32(sc_11) & 4294967232u | (unsigned int)(t0_2 + 3);
    kb[3] = __uint_as_float(key_12);
    float sc_13 = __uint_as_float(cb[4]);
    float _fmax_4 = fmaxf(sc_13, -1.7014118346046923e+38f);
    sc_13 = _fmax_4;
    float _min_4 = fminf(sc_13, 1.7014118346046923e+38f);
    sc_13 = _min_4;
    sc_13 = sc_13;
    float sc_14 = sc_13;
    unsigned int key_15 = __as_u32(sc_14) & 4294967232u | (unsigned int)(t0_2 + 4);
    kb[4] = __uint_as_float(key_15);
    float sc_16 = __uint_as_float(cb[5]);
    float _fmax_5 = fmaxf(sc_16, -1.7014118346046923e+38f);
    sc_16 = _fmax_5;
    float _min_5 = fminf(sc_16, 1.7014118346046923e+38f);
    sc_16 = _min_5;
    sc_16 = sc_16;
    float sc_17 = sc_16;
    unsigned int key_18 = __as_u32(sc_17) & 4294967232u | (unsigned int)(t0_2 + 5);
    kb[5] = __uint_as_float(key_18);
    float sc_19 = __uint_as_float(cb[6]);
    float _fmax_6 = fmaxf(sc_19, -1.7014118346046923e+38f);
    sc_19 = _fmax_6;
    float _min_6 = fminf(sc_19, 1.7014118346046923e+38f);
    sc_19 = _min_6;
    sc_19 = sc_19;
    float sc_20 = sc_19;
    unsigned int key_21 = __as_u32(sc_20) & 4294967232u | (unsigned int)(t0_2 + 6);
    kb[6] = __uint_as_float(key_21);
    float sc_22 = __uint_as_float(cb[7]);
    float _fmax_7 = fmaxf(sc_22, -1.7014118346046923e+38f);
    sc_22 = _fmax_7;
    float _min_7 = fminf(sc_22, 1.7014118346046923e+38f);
    sc_22 = _min_7;
    sc_22 = sc_22;
    float sc_23 = sc_22;
    unsigned int key_24 = __as_u32(sc_23) & 4294967232u | (unsigned int)(t0_2 + 7);
    kb[7] = __uint_as_float(key_24);
    float sc_25 = __uint_as_float(cb[8]);
    float _fmax_8 = fmaxf(sc_25, -1.7014118346046923e+38f);
    sc_25 = _fmax_8;
    float _min_8 = fminf(sc_25, 1.7014118346046923e+38f);
    sc_25 = _min_8;
    sc_25 = sc_25;
    float sc_26 = sc_25;
    unsigned int key_27 = __as_u32(sc_26) & 4294967232u | (unsigned int)(t0_2 + 8);
    kb[8] = __uint_as_float(key_27);
    float sc_28 = __uint_as_float(cb[9]);
    float _fmax_9 = fmaxf(sc_28, -1.7014118346046923e+38f);
    sc_28 = _fmax_9;
    float _min_9 = fminf(sc_28, 1.7014118346046923e+38f);
    sc_28 = _min_9;
    sc_28 = sc_28;
    float sc_29 = sc_28;
    unsigned int key_30 = __as_u32(sc_29) & 4294967232u | (unsigned int)(t0_2 + 9);
    kb[9] = __uint_as_float(key_30);
    float sc_31 = __uint_as_float(cb[10]);
    float _fmax_10 = fmaxf(sc_31, -1.7014118346046923e+38f);
    sc_31 = _fmax_10;
    float _min_10 = fminf(sc_31, 1.7014118346046923e+38f);
    sc_31 = _min_10;
    sc_31 = sc_31;
    float sc_32 = sc_31;
    unsigned int key_33 = __as_u32(sc_32) & 4294967232u | (unsigned int)(t0_2 + 10);
    kb[10] = __uint_as_float(key_33);
    float sc_34 = __uint_as_float(cb[11]);
    float _fmax_11 = fmaxf(sc_34, -1.7014118346046923e+38f);
    sc_34 = _fmax_11;
    float _min_11 = fminf(sc_34, 1.7014118346046923e+38f);
    sc_34 = _min_11;
    sc_34 = sc_34;
    float sc_35 = sc_34;
    unsigned int key_36 = __as_u32(sc_35) & 4294967232u | (unsigned int)(t0_2 + 11);
    kb[11] = __uint_as_float(key_36);
    float sc_37 = __uint_as_float(cb[12]);
    float _fmax_12 = fmaxf(sc_37, -1.7014118346046923e+38f);
    sc_37 = _fmax_12;
    float _min_12 = fminf(sc_37, 1.7014118346046923e+38f);
    sc_37 = _min_12;
    sc_37 = sc_37;
    float sc_38 = sc_37;
    unsigned int key_39 = __as_u32(sc_38) & 4294967232u | (unsigned int)(t0_2 + 12);
    kb[12] = __uint_as_float(key_39);
    float sc_40 = __uint_as_float(cb[13]);
    float _fmax_13 = fmaxf(sc_40, -1.7014118346046923e+38f);
    sc_40 = _fmax_13;
    float _min_13 = fminf(sc_40, 1.7014118346046923e+38f);
    sc_40 = _min_13;
    sc_40 = sc_40;
    float sc_41 = sc_40;
    unsigned int key_42 = __as_u32(sc_41) & 4294967232u | (unsigned int)(t0_2 + 13);
    kb[13] = __uint_as_float(key_42);
    float sc_43 = __uint_as_float(cb[14]);
    float _fmax_14 = fmaxf(sc_43, -1.7014118346046923e+38f);
    sc_43 = _fmax_14;
    float _min_14 = fminf(sc_43, 1.7014118346046923e+38f);
    sc_43 = _min_14;
    sc_43 = sc_43;
    float sc_44 = sc_43;
    unsigned int key_45 = __as_u32(sc_44) & 4294967232u | (unsigned int)(t0_2 + 14);
    kb[14] = __uint_as_float(key_45);
    float sc_46 = __uint_as_float(cb[15]);
    float _fmax_15 = fmaxf(sc_46, -1.7014118346046923e+38f);
    sc_46 = _fmax_15;
    float _min_15 = fminf(sc_46, 1.7014118346046923e+38f);
    sc_46 = _min_15;
    sc_46 = sc_46;
    float sc_47 = sc_46;
    unsigned int key_48 = __as_u32(sc_47) & 4294967232u | (unsigned int)(t0_2 + 15);
    kb[15] = __uint_as_float(key_48);
    float _fmax_16 = fmaxf(kb[0], kb[13]);
    float hi = _fmax_16;
    float _min_16 = fminf(kb[0], kb[13]);
    float lo = _min_16;
    kb[0] = hi;
    kb[13] = lo;
    float _fmax_17 = fmaxf(kb[1], kb[12]);
    float hi_49 = _fmax_17;
    float _min_17 = fminf(kb[1], kb[12]);
    float lo_50 = _min_17;
    kb[1] = hi_49;
    kb[12] = lo_50;
    float _fmax_18 = fmaxf(kb[2], kb[15]);
    float hi_51 = _fmax_18;
    float _min_18 = fminf(kb[2], kb[15]);
    float lo_52 = _min_18;
    kb[2] = hi_51;
    kb[15] = lo_52;
    float _fmax_19 = fmaxf(kb[3], kb[14]);
    float hi_53 = _fmax_19;
    float _min_19 = fminf(kb[3], kb[14]);
    float lo_54 = _min_19;
    kb[3] = hi_53;
    kb[14] = lo_54;
    float _fmax_20 = fmaxf(kb[4], kb[8]);
    float hi_55 = _fmax_20;
    float _min_20 = fminf(kb[4], kb[8]);
    float lo_56 = _min_20;
    kb[4] = hi_55;
    kb[8] = lo_56;
    float _fmax_21 = fmaxf(kb[5], kb[6]);
    float hi_57 = _fmax_21;
    float _min_21 = fminf(kb[5], kb[6]);
    float lo_58 = _min_21;
    kb[5] = hi_57;
    kb[6] = lo_58;
    float _fmax_22 = fmaxf(kb[7], kb[11]);
    float hi_59 = _fmax_22;
    float _min_22 = fminf(kb[7], kb[11]);
    float lo_60 = _min_22;
    kb[7] = hi_59;
    kb[11] = lo_60;
    float _fmax_23 = fmaxf(kb[9], kb[10]);
    float hi_61 = _fmax_23;
    float _min_23 = fminf(kb[9], kb[10]);
    float lo_62 = _min_23;
    kb[9] = hi_61;
    kb[10] = lo_62;
    float _fmax_24 = fmaxf(kb[0], kb[5]);
    float hi_63 = _fmax_24;
    float _min_24 = fminf(kb[0], kb[5]);
    float lo_64 = _min_24;
    kb[0] = hi_63;
    kb[5] = lo_64;
    float _fmax_25 = fmaxf(kb[1], kb[7]);
    float hi_65 = _fmax_25;
    float _min_25 = fminf(kb[1], kb[7]);
    float lo_66 = _min_25;
    kb[1] = hi_65;
    kb[7] = lo_66;
    float _fmax_26 = fmaxf(kb[2], kb[9]);
    float hi_67 = _fmax_26;
    float _min_26 = fminf(kb[2], kb[9]);
    float lo_68 = _min_26;
    kb[2] = hi_67;
    kb[9] = lo_68;
    float _fmax_27 = fmaxf(kb[3], kb[4]);
    float hi_69 = _fmax_27;
    float _min_27 = fminf(kb[3], kb[4]);
    float lo_70 = _min_27;
    kb[3] = hi_69;
    kb[4] = lo_70;
    float _fmax_28 = fmaxf(kb[6], kb[13]);
    float hi_71 = _fmax_28;
    float _min_28 = fminf(kb[6], kb[13]);
    float lo_72 = _min_28;
    kb[6] = hi_71;
    kb[13] = lo_72;
    float _fmax_29 = fmaxf(kb[8], kb[14]);
    float hi_73 = _fmax_29;
    float _min_29 = fminf(kb[8], kb[14]);
    float lo_74 = _min_29;
    kb[8] = hi_73;
    kb[14] = lo_74;
    float _fmax_30 = fmaxf(kb[10], kb[15]);
    float hi_75 = _fmax_30;
    float _min_30 = fminf(kb[10], kb[15]);
    float lo_76 = _min_30;
    kb[10] = hi_75;
    kb[15] = lo_76;
    float _fmax_31 = fmaxf(kb[11], kb[12]);
    float hi_77 = _fmax_31;
    float _min_31 = fminf(kb[11], kb[12]);
    float lo_78 = _min_31;
    kb[11] = hi_77;
    kb[12] = lo_78;
    float _fmax_32 = fmaxf(kb[0], kb[1]);
    float hi_79 = _fmax_32;
    float _min_32 = fminf(kb[0], kb[1]);
    float lo_80 = _min_32;
    kb[0] = hi_79;
    kb[1] = lo_80;
    float _fmax_33 = fmaxf(kb[2], kb[3]);
    float hi_81 = _fmax_33;
    float _min_33 = fminf(kb[2], kb[3]);
    float lo_82 = _min_33;
    kb[2] = hi_81;
    kb[3] = lo_82;
    float _fmax_34 = fmaxf(kb[4], kb[5]);
    float hi_83 = _fmax_34;
    float _min_34 = fminf(kb[4], kb[5]);
    float lo_84 = _min_34;
    kb[4] = hi_83;
    kb[5] = lo_84;
    float _fmax_35 = fmaxf(kb[6], kb[8]);
    float hi_85 = _fmax_35;
    float _min_35 = fminf(kb[6], kb[8]);
    float lo_86 = _min_35;
    kb[6] = hi_85;
    kb[8] = lo_86;
    float _fmax_36 = fmaxf(kb[7], kb[9]);
    float hi_87 = _fmax_36;
    float _min_36 = fminf(kb[7], kb[9]);
    float lo_88 = _min_36;
    kb[7] = hi_87;
    kb[9] = lo_88;
    float _fmax_37 = fmaxf(kb[10], kb[11]);
    float hi_89 = _fmax_37;
    float _min_37 = fminf(kb[10], kb[11]);
    float lo_90 = _min_37;
    kb[10] = hi_89;
    kb[11] = lo_90;
    float _fmax_38 = fmaxf(kb[12], kb[13]);
    float hi_91 = _fmax_38;
    float _min_38 = fminf(kb[12], kb[13]);
    float lo_92 = _min_38;
    kb[12] = hi_91;
    kb[13] = lo_92;
    float _fmax_39 = fmaxf(kb[14], kb[15]);
    float hi_93 = _fmax_39;
    float _min_39 = fminf(kb[14], kb[15]);
    float lo_94 = _min_39;
    kb[14] = hi_93;
    kb[15] = lo_94;
    float _fmax_40 = fmaxf(kb[0], kb[2]);
    float hi_95 = _fmax_40;
    float _min_40 = fminf(kb[0], kb[2]);
    float lo_96 = _min_40;
    kb[0] = hi_95;
    kb[2] = lo_96;
    float _fmax_41 = fmaxf(kb[1], kb[3]);
    float hi_97 = _fmax_41;
    float _min_41 = fminf(kb[1], kb[3]);
    float lo_98 = _min_41;
    kb[1] = hi_97;
    kb[3] = lo_98;
    float _fmax_42 = fmaxf(kb[4], kb[10]);
    float hi_99 = _fmax_42;
    float _min_42 = fminf(kb[4], kb[10]);
    float lo_100 = _min_42;
    kb[4] = hi_99;
    kb[10] = lo_100;
    float _fmax_43 = fmaxf(kb[5], kb[11]);
    float hi_101 = _fmax_43;
    float _min_43 = fminf(kb[5], kb[11]);
    float lo_102 = _min_43;
    kb[5] = hi_101;
    kb[11] = lo_102;
    float _fmax_44 = fmaxf(kb[6], kb[7]);
    float hi_103 = _fmax_44;
    float _min_44 = fminf(kb[6], kb[7]);
    float lo_104 = _min_44;
    kb[6] = hi_103;
    kb[7] = lo_104;
    float _fmax_45 = fmaxf(kb[8], kb[9]);
    float hi_105 = _fmax_45;
    float _min_45 = fminf(kb[8], kb[9]);
    float lo_106 = _min_45;
    kb[8] = hi_105;
    kb[9] = lo_106;
    float _fmax_46 = fmaxf(kb[12], kb[14]);
    float hi_107 = _fmax_46;
    float _min_46 = fminf(kb[12], kb[14]);
    float lo_108 = _min_46;
    kb[12] = hi_107;
    kb[14] = lo_108;
    float _fmax_47 = fmaxf(kb[13], kb[15]);
    float hi_109 = _fmax_47;
    float _min_47 = fminf(kb[13], kb[15]);
    float lo_110 = _min_47;
    kb[13] = hi_109;
    kb[15] = lo_110;
    float _fmax_48 = fmaxf(kb[1], kb[2]);
    float hi_111 = _fmax_48;
    float _min_48 = fminf(kb[1], kb[2]);
    float lo_112 = _min_48;
    kb[1] = hi_111;
    kb[2] = lo_112;
    float _fmax_49 = fmaxf(kb[3], kb[12]);
    float hi_113 = _fmax_49;
    float _min_49 = fminf(kb[3], kb[12]);
    float lo_114 = _min_49;
    kb[3] = hi_113;
    kb[12] = lo_114;
    float _fmax_50 = fmaxf(kb[4], kb[6]);
    float hi_115 = _fmax_50;
    float _min_50 = fminf(kb[4], kb[6]);
    float lo_116 = _min_50;
    kb[4] = hi_115;
    kb[6] = lo_116;
    float _fmax_51 = fmaxf(kb[5], kb[7]);
    float hi_117 = _fmax_51;
    float _min_51 = fminf(kb[5], kb[7]);
    float lo_118 = _min_51;
    kb[5] = hi_117;
    kb[7] = lo_118;
    float _fmax_52 = fmaxf(kb[8], kb[10]);
    float hi_119 = _fmax_52;
    float _min_52 = fminf(kb[8], kb[10]);
    float lo_120 = _min_52;
    kb[8] = hi_119;
    kb[10] = lo_120;
    float _fmax_53 = fmaxf(kb[9], kb[11]);
    float hi_121 = _fmax_53;
    float _min_53 = fminf(kb[9], kb[11]);
    float lo_122 = _min_53;
    kb[9] = hi_121;
    kb[11] = lo_122;
    float _fmax_54 = fmaxf(kb[13], kb[14]);
    float hi_123 = _fmax_54;
    float _min_54 = fminf(kb[13], kb[14]);
    float lo_124 = _min_54;
    kb[13] = hi_123;
    kb[14] = lo_124;
    float _fmax_55 = fmaxf(kb[1], kb[4]);
    float hi_125 = _fmax_55;
    float _min_55 = fminf(kb[1], kb[4]);
    float lo_126 = _min_55;
    kb[1] = hi_125;
    kb[4] = lo_126;
    float _fmax_56 = fmaxf(kb[2], kb[6]);
    float hi_127 = _fmax_56;
    float _min_56 = fminf(kb[2], kb[6]);
    float lo_128 = _min_56;
    kb[2] = hi_127;
    kb[6] = lo_128;
    float _fmax_57 = fmaxf(kb[5], kb[8]);
    float hi_129 = _fmax_57;
    float _min_57 = fminf(kb[5], kb[8]);
    float lo_130 = _min_57;
    kb[5] = hi_129;
    kb[8] = lo_130;
    float _fmax_58 = fmaxf(kb[7], kb[10]);
    float hi_131 = _fmax_58;
    float _min_58 = fminf(kb[7], kb[10]);
    float lo_132 = _min_58;
    kb[7] = hi_131;
    kb[10] = lo_132;
    float _fmax_59 = fmaxf(kb[9], kb[13]);
    float hi_133 = _fmax_59;
    float _min_59 = fminf(kb[9], kb[13]);
    float lo_134 = _min_59;
    kb[9] = hi_133;
    kb[13] = lo_134;
    float _fmax_60 = fmaxf(kb[11], kb[14]);
    float hi_135 = _fmax_60;
    float _min_60 = fminf(kb[11], kb[14]);
    float lo_136 = _min_60;
    kb[11] = hi_135;
    kb[14] = lo_136;
    float _fmax_61 = fmaxf(kb[2], kb[4]);
    float hi_137 = _fmax_61;
    float _min_61 = fminf(kb[2], kb[4]);
    float lo_138 = _min_61;
    kb[2] = hi_137;
    kb[4] = lo_138;
    float _fmax_62 = fmaxf(kb[3], kb[6]);
    float hi_139 = _fmax_62;
    float _min_62 = fminf(kb[3], kb[6]);
    float lo_140 = _min_62;
    kb[3] = hi_139;
    kb[6] = lo_140;
    float _fmax_63 = fmaxf(kb[9], kb[12]);
    float hi_141 = _fmax_63;
    float _min_63 = fminf(kb[9], kb[12]);
    float lo_142 = _min_63;
    kb[9] = hi_141;
    kb[12] = lo_142;
    float _fmax_64 = fmaxf(kb[11], kb[13]);
    float hi_143 = _fmax_64;
    float _min_64 = fminf(kb[11], kb[13]);
    float lo_144 = _min_64;
    kb[11] = hi_143;
    kb[13] = lo_144;
    float _fmax_65 = fmaxf(kb[3], kb[5]);
    float hi_145 = _fmax_65;
    float _min_65 = fminf(kb[3], kb[5]);
    float lo_146 = _min_65;
    kb[3] = hi_145;
    kb[5] = lo_146;
    float _fmax_66 = fmaxf(kb[6], kb[8]);
    float hi_147 = _fmax_66;
    float _min_66 = fminf(kb[6], kb[8]);
    float lo_148 = _min_66;
    kb[6] = hi_147;
    kb[8] = lo_148;
    float _fmax_67 = fmaxf(kb[7], kb[9]);
    float hi_149 = _fmax_67;
    float _min_67 = fminf(kb[7], kb[9]);
    float lo_150 = _min_67;
    kb[7] = hi_149;
    kb[9] = lo_150;
    float _fmax_68 = fmaxf(kb[10], kb[12]);
    float hi_151 = _fmax_68;
    float _min_68 = fminf(kb[10], kb[12]);
    float lo_152 = _min_68;
    kb[10] = hi_151;
    kb[12] = lo_152;
    float _fmax_69 = fmaxf(kb[3], kb[4]);
    float hi_153 = _fmax_69;
    float _min_69 = fminf(kb[3], kb[4]);
    float lo_154 = _min_69;
    kb[3] = hi_153;
    kb[4] = lo_154;
    float _fmax_70 = fmaxf(kb[5], kb[6]);
    float hi_155 = _fmax_70;
    float _min_70 = fminf(kb[5], kb[6]);
    float lo_156 = _min_70;
    kb[5] = hi_155;
    kb[6] = lo_156;
    float _fmax_71 = fmaxf(kb[7], kb[8]);
    float hi_157 = _fmax_71;
    float _min_71 = fminf(kb[7], kb[8]);
    float lo_158 = _min_71;
    kb[7] = hi_157;
    kb[8] = lo_158;
    float _fmax_72 = fmaxf(kb[9], kb[10]);
    float hi_159 = _fmax_72;
    float _min_72 = fminf(kb[9], kb[10]);
    float lo_160 = _min_72;
    kb[9] = hi_159;
    kb[10] = lo_160;
    float _fmax_73 = fmaxf(kb[11], kb[12]);
    float hi_161 = _fmax_73;
    float _min_73 = fminf(kb[11], kb[12]);
    float lo_162 = _min_73;
    kb[11] = hi_161;
    kb[12] = lo_162;
    float _fmax_74 = fmaxf(kb[6], kb[7]);
    float hi_163 = _fmax_74;
    float _min_74 = fminf(kb[6], kb[7]);
    float lo_164 = _min_74;
    kb[6] = hi_163;
    kb[7] = lo_164;
    float _fmax_75 = fmaxf(kb[8], kb[9]);
    float hi_165 = _fmax_75;
    float _min_75 = fminf(kb[8], kb[9]);
    float lo_166 = _min_75;
    kb[8] = hi_165;
    kb[9] = lo_166;
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
    #pragma unroll 1
    for (int k = 0; k < 2; k++) {
        int m = 8 << k;
        float ob[16];
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, a[0], m);
        ob[0] = _shfl_xor_0;
        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, a[1], m);
        ob[1] = _shfl_xor_1;
        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, a[2], m);
        ob[2] = _shfl_xor_2;
        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, a[3], m);
        ob[3] = _shfl_xor_3;
        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, a[4], m);
        ob[4] = _shfl_xor_4;
        float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, a[5], m);
        ob[5] = _shfl_xor_5;
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, a[6], m);
        ob[6] = _shfl_xor_6;
        float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, a[7], m);
        ob[7] = _shfl_xor_7;
        float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, a[8], m);
        ob[8] = _shfl_xor_8;
        float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, a[9], m);
        ob[9] = _shfl_xor_9;
        float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, a[10], m);
        ob[10] = _shfl_xor_10;
        float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, a[11], m);
        ob[11] = _shfl_xor_11;
        float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, a[12], m);
        ob[12] = _shfl_xor_12;
        float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, a[13], m);
        ob[13] = _shfl_xor_13;
        float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, a[14], m);
        ob[14] = _shfl_xor_14;
        float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, a[15], m);
        ob[15] = _shfl_xor_15;
        float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, rej, m);
        float orej = _shfl_xor_16;
        float _fmax_76 = fmaxf(rej, orej);
        rej = _fmax_76;
        float r = rej;
        float _fmax_77 = fmaxf(a[0], ob[15]);
        float hi_0 = _fmax_77;
        float _min_76 = fminf(a[0], ob[15]);
        float lo_1 = _min_76;
        a[0] = hi_0;
        float _fmax_78 = fmaxf(r, lo_1);
        r = _fmax_78;
        float _fmax_79 = fmaxf(a[1], ob[14]);
        float hi_2 = _fmax_79;
        float _min_77 = fminf(a[1], ob[14]);
        float lo_3 = _min_77;
        a[1] = hi_2;
        float _fmax_80 = fmaxf(r, lo_3);
        r = _fmax_80;
        float _fmax_81 = fmaxf(a[2], ob[13]);
        float hi_4 = _fmax_81;
        float _min_78 = fminf(a[2], ob[13]);
        float lo_5 = _min_78;
        a[2] = hi_4;
        float _fmax_82 = fmaxf(r, lo_5);
        r = _fmax_82;
        float _fmax_83 = fmaxf(a[3], ob[12]);
        float hi_6 = _fmax_83;
        float _min_79 = fminf(a[3], ob[12]);
        float lo_7 = _min_79;
        a[3] = hi_6;
        float _fmax_84 = fmaxf(r, lo_7);
        r = _fmax_84;
        float _fmax_85 = fmaxf(a[4], ob[11]);
        float hi_8 = _fmax_85;
        float _min_80 = fminf(a[4], ob[11]);
        float lo_9 = _min_80;
        a[4] = hi_8;
        float _fmax_86 = fmaxf(r, lo_9);
        r = _fmax_86;
        float _fmax_87 = fmaxf(a[5], ob[10]);
        float hi_10 = _fmax_87;
        float _min_81 = fminf(a[5], ob[10]);
        float lo_11 = _min_81;
        a[5] = hi_10;
        float _fmax_88 = fmaxf(r, lo_11);
        r = _fmax_88;
        float _fmax_89 = fmaxf(a[6], ob[9]);
        float hi_12 = _fmax_89;
        float _min_82 = fminf(a[6], ob[9]);
        float lo_13 = _min_82;
        a[6] = hi_12;
        float _fmax_90 = fmaxf(r, lo_13);
        r = _fmax_90;
        float _fmax_91 = fmaxf(a[7], ob[8]);
        float hi_14 = _fmax_91;
        float _min_83 = fminf(a[7], ob[8]);
        float lo_15 = _min_83;
        a[7] = hi_14;
        float _fmax_92 = fmaxf(r, lo_15);
        r = _fmax_92;
        float _fmax_93 = fmaxf(a[8], ob[7]);
        float hi_16 = _fmax_93;
        float _min_84 = fminf(a[8], ob[7]);
        float lo_17 = _min_84;
        a[8] = hi_16;
        float _fmax_94 = fmaxf(r, lo_17);
        r = _fmax_94;
        float _fmax_95 = fmaxf(a[9], ob[6]);
        float hi_18 = _fmax_95;
        float _min_85 = fminf(a[9], ob[6]);
        float lo_19 = _min_85;
        a[9] = hi_18;
        float _fmax_96 = fmaxf(r, lo_19);
        r = _fmax_96;
        float _fmax_97 = fmaxf(a[10], ob[5]);
        float hi_20 = _fmax_97;
        float _min_86 = fminf(a[10], ob[5]);
        float lo_21 = _min_86;
        a[10] = hi_20;
        float _fmax_98 = fmaxf(r, lo_21);
        r = _fmax_98;
        float _fmax_99 = fmaxf(a[11], ob[4]);
        float hi_22 = _fmax_99;
        float _min_87 = fminf(a[11], ob[4]);
        float lo_23 = _min_87;
        a[11] = hi_22;
        float _fmax_100 = fmaxf(r, lo_23);
        r = _fmax_100;
        float _fmax_101 = fmaxf(a[12], ob[3]);
        float hi_24 = _fmax_101;
        float _min_88 = fminf(a[12], ob[3]);
        float lo_25 = _min_88;
        a[12] = hi_24;
        float _fmax_102 = fmaxf(r, lo_25);
        r = _fmax_102;
        float _fmax_103 = fmaxf(a[13], ob[2]);
        float hi_26 = _fmax_103;
        float _min_89 = fminf(a[13], ob[2]);
        float lo_27 = _min_89;
        a[13] = hi_26;
        float _fmax_104 = fmaxf(r, lo_27);
        r = _fmax_104;
        float _fmax_105 = fmaxf(a[14], ob[1]);
        float hi_28 = _fmax_105;
        float _min_90 = fminf(a[14], ob[1]);
        float lo_29 = _min_90;
        a[14] = hi_28;
        float _fmax_106 = fmaxf(r, lo_29);
        r = _fmax_106;
        float _fmax_107 = fmaxf(a[15], ob[0]);
        float hi_30 = _fmax_107;
        float _min_91 = fminf(a[15], ob[0]);
        float lo_31 = _min_91;
        a[15] = hi_30;
        float _fmax_108 = fmaxf(r, lo_31);
        r = _fmax_108;
        float _fmax_109 = fmaxf(a[0], a[8]);
        float hi_32 = _fmax_109;
        float _min_92 = fminf(a[0], a[8]);
        float lo_33 = _min_92;
        a[0] = hi_32;
        a[8] = lo_33;
        float _fmax_110 = fmaxf(a[1], a[9]);
        float hi_34 = _fmax_110;
        float _min_93 = fminf(a[1], a[9]);
        float lo_35 = _min_93;
        a[1] = hi_34;
        a[9] = lo_35;
        float _fmax_111 = fmaxf(a[2], a[10]);
        float hi_36 = _fmax_111;
        float _min_94 = fminf(a[2], a[10]);
        float lo_37 = _min_94;
        a[2] = hi_36;
        a[10] = lo_37;
        float _fmax_112 = fmaxf(a[3], a[11]);
        float hi_38 = _fmax_112;
        float _min_95 = fminf(a[3], a[11]);
        float lo_39 = _min_95;
        a[3] = hi_38;
        a[11] = lo_39;
        float _fmax_113 = fmaxf(a[4], a[12]);
        float hi_40 = _fmax_113;
        float _min_96 = fminf(a[4], a[12]);
        float lo_41 = _min_96;
        a[4] = hi_40;
        a[12] = lo_41;
        float _fmax_114 = fmaxf(a[5], a[13]);
        float hi_42 = _fmax_114;
        float _min_97 = fminf(a[5], a[13]);
        float lo_43 = _min_97;
        a[5] = hi_42;
        a[13] = lo_43;
        float _fmax_115 = fmaxf(a[6], a[14]);
        float hi_44 = _fmax_115;
        float _min_98 = fminf(a[6], a[14]);
        float lo_45 = _min_98;
        a[6] = hi_44;
        a[14] = lo_45;
        float _fmax_116 = fmaxf(a[7], a[15]);
        float hi_46 = _fmax_116;
        float _min_99 = fminf(a[7], a[15]);
        float lo_47 = _min_99;
        a[7] = hi_46;
        a[15] = lo_47;
        float _fmax_117 = fmaxf(a[0], a[4]);
        float hi_48 = _fmax_117;
        float _min_100 = fminf(a[0], a[4]);
        float lo_49 = _min_100;
        a[0] = hi_48;
        a[4] = lo_49;
        float _fmax_118 = fmaxf(a[1], a[5]);
        float hi_50 = _fmax_118;
        float _min_101 = fminf(a[1], a[5]);
        float lo_51 = _min_101;
        a[1] = hi_50;
        a[5] = lo_51;
        float _fmax_119 = fmaxf(a[2], a[6]);
        float hi_52 = _fmax_119;
        float _min_102 = fminf(a[2], a[6]);
        float lo_53 = _min_102;
        a[2] = hi_52;
        a[6] = lo_53;
        float _fmax_120 = fmaxf(a[3], a[7]);
        float hi_54 = _fmax_120;
        float _min_103 = fminf(a[3], a[7]);
        float lo_55 = _min_103;
        a[3] = hi_54;
        a[7] = lo_55;
        float _fmax_121 = fmaxf(a[8], a[12]);
        float hi_56 = _fmax_121;
        float _min_104 = fminf(a[8], a[12]);
        float lo_57 = _min_104;
        a[8] = hi_56;
        a[12] = lo_57;
        float _fmax_122 = fmaxf(a[9], a[13]);
        float hi_58 = _fmax_122;
        float _min_105 = fminf(a[9], a[13]);
        float lo_59 = _min_105;
        a[9] = hi_58;
        a[13] = lo_59;
        float _fmax_123 = fmaxf(a[10], a[14]);
        float hi_60 = _fmax_123;
        float _min_106 = fminf(a[10], a[14]);
        float lo_61 = _min_106;
        a[10] = hi_60;
        a[14] = lo_61;
        float _fmax_124 = fmaxf(a[11], a[15]);
        float hi_62 = _fmax_124;
        float _min_107 = fminf(a[11], a[15]);
        float lo_63 = _min_107;
        a[11] = hi_62;
        a[15] = lo_63;
        float _fmax_125 = fmaxf(a[0], a[2]);
        float hi_64 = _fmax_125;
        float _min_108 = fminf(a[0], a[2]);
        float lo_65 = _min_108;
        a[0] = hi_64;
        a[2] = lo_65;
        float _fmax_126 = fmaxf(a[1], a[3]);
        float hi_66 = _fmax_126;
        float _min_109 = fminf(a[1], a[3]);
        float lo_67 = _min_109;
        a[1] = hi_66;
        a[3] = lo_67;
        float _fmax_127 = fmaxf(a[4], a[6]);
        float hi_68 = _fmax_127;
        float _min_110 = fminf(a[4], a[6]);
        float lo_69 = _min_110;
        a[4] = hi_68;
        a[6] = lo_69;
        float _fmax_128 = fmaxf(a[5], a[7]);
        float hi_70 = _fmax_128;
        float _min_111 = fminf(a[5], a[7]);
        float lo_71 = _min_111;
        a[5] = hi_70;
        a[7] = lo_71;
        float _fmax_129 = fmaxf(a[8], a[10]);
        float hi_72 = _fmax_129;
        float _min_112 = fminf(a[8], a[10]);
        float lo_73 = _min_112;
        a[8] = hi_72;
        a[10] = lo_73;
        float _fmax_130 = fmaxf(a[9], a[11]);
        float hi_74 = _fmax_130;
        float _min_113 = fminf(a[9], a[11]);
        float lo_75 = _min_113;
        a[9] = hi_74;
        a[11] = lo_75;
        float _fmax_131 = fmaxf(a[12], a[14]);
        float hi_76 = _fmax_131;
        float _min_114 = fminf(a[12], a[14]);
        float lo_77 = _min_114;
        a[12] = hi_76;
        a[14] = lo_77;
        float _fmax_132 = fmaxf(a[13], a[15]);
        float hi_78 = _fmax_132;
        float _min_115 = fminf(a[13], a[15]);
        float lo_79 = _min_115;
        a[13] = hi_78;
        a[15] = lo_79;
        float _fmax_133 = fmaxf(a[0], a[1]);
        float hi_80 = _fmax_133;
        float _min_116 = fminf(a[0], a[1]);
        float lo_81 = _min_116;
        a[0] = hi_80;
        a[1] = lo_81;
        float _fmax_134 = fmaxf(a[2], a[3]);
        float hi_82 = _fmax_134;
        float _min_117 = fminf(a[2], a[3]);
        float lo_83 = _min_117;
        a[2] = hi_82;
        a[3] = lo_83;
        float _fmax_135 = fmaxf(a[4], a[5]);
        float hi_84 = _fmax_135;
        float _min_118 = fminf(a[4], a[5]);
        float lo_85 = _min_118;
        a[4] = hi_84;
        a[5] = lo_85;
        float _fmax_136 = fmaxf(a[6], a[7]);
        float hi_86 = _fmax_136;
        float _min_119 = fminf(a[6], a[7]);
        float lo_87 = _min_119;
        a[6] = hi_86;
        a[7] = lo_87;
        float _fmax_137 = fmaxf(a[8], a[9]);
        float hi_88 = _fmax_137;
        float _min_120 = fminf(a[8], a[9]);
        float lo_89 = _min_120;
        a[8] = hi_88;
        a[9] = lo_89;
        float _fmax_138 = fmaxf(a[10], a[11]);
        float hi_90 = _fmax_138;
        float _min_121 = fminf(a[10], a[11]);
        float lo_91 = _min_121;
        a[10] = hi_90;
        a[11] = lo_91;
        float _fmax_139 = fmaxf(a[12], a[13]);
        float hi_92 = _fmax_139;
        float _min_122 = fminf(a[12], a[13]);
        float lo_93 = _min_122;
        a[12] = hi_92;
        a[13] = lo_93;
        float _fmax_140 = fmaxf(a[14], a[15]);
        float hi_94 = _fmax_140;
        float _min_123 = fminf(a[14], a[15]);
        float lo_95 = _min_123;
        a[14] = hi_94;
        a[15] = lo_95;
        rej = r;
    }
    if (w == 0) {
        unsigned int u16 = __as_u32(a[15]);
        unsigned int c16 = u16 & 4294967232u;
        unsigned int cr = __as_u32(rej) & 4294967232u;
        unsigned int q = 4294967295;
        if (c16 == cr && u16 < 4278190080u && c16 != 2139094976 && col < total_q) {
            q = c16;
            flagw[0] = 1;
        }
        qcol[c] = q;
    }
    asm volatile("barrier.sync 8, 32;" ::: "memory");
    unsigned int need2 = flagw[0];
    unsigned int qc = qcol[c];
    int flagged = 0;
    if (qc != 4294967295u) {
        flagged = 1;
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
    if (need2 != 0) {
        int lim2 = 0;
        if (flagged != 0) {
            lim2 = lim;
        }
        if (flagged != 0) {
            float kb2o[16];
            int t0_3 = w * 16;
            int t0_4 = t0_3;
            float qf = __uint_as_float(qc);
            float sc_6 = __uint_as_float(cb[0]);
            float _fmax_141 = fmaxf(sc_6, -1.7014118346046923e+38f);
            sc_6 = _fmax_141;
            float _min_124 = fminf(sc_6, 1.7014118346046923e+38f);
            sc_6 = _min_124;
            sc_6 = sc_6;
            float sc_9 = sc_6;
            unsigned int u = __as_u32(sc_9);
            unsigned int cls = u & 4294967232u;
            unsigned int key_10 = 0;
            if (qf < __uint_as_float(cls) && cls < 4278190080u) {
                key_10 = 1073741824 | (unsigned int)t0_4;
            }
            if (cls == qc) {
                unsigned int lowb = (u ^ (unsigned int)((int)u >> 31) & 63) & 63;
                key_10 = 536870912 | lowb << 6 | (unsigned int)t0_4;
            }
            kb2o[0] = __uint_as_float(key_10);
            float sc_12 = __uint_as_float(cb[1]);
            float _fmax_142 = fmaxf(sc_12, -1.7014118346046923e+38f);
            sc_12 = _fmax_142;
            float _min_125 = fminf(sc_12, 1.7014118346046923e+38f);
            sc_12 = _min_125;
            sc_12 = sc_12;
            float sc_15 = sc_12;
            unsigned int u_16 = __as_u32(sc_15);
            unsigned int cls_17 = u_16 & 4294967232u;
            unsigned int key_19 = 0;
            if (qf < __uint_as_float(cls_17) && cls_17 < 4278190080u) {
                key_19 = 1073741824 | (unsigned int)(t0_4 + 1);
            }
            if (cls_17 == qc) {
                unsigned int lowb_1 = (u_16 ^ (unsigned int)((int)u_16 >> 31) & 63) & 63;
                key_19 = 536870912 | lowb_1 << 6 | (unsigned int)(t0_4 + 1);
            }
            kb2o[1] = __uint_as_float(key_19);
            float sc_21 = __uint_as_float(cb[2]);
            float _fmax_143 = fmaxf(sc_21, -1.7014118346046923e+38f);
            sc_21 = _fmax_143;
            float _min_126 = fminf(sc_21, 1.7014118346046923e+38f);
            sc_21 = _min_126;
            sc_21 = sc_21;
            float sc_24 = sc_21;
            unsigned int u_25 = __as_u32(sc_24);
            unsigned int cls_26 = u_25 & 4294967232u;
            unsigned int key_28 = 0;
            if (qf < __uint_as_float(cls_26) && cls_26 < 4278190080u) {
                key_28 = 1073741824 | (unsigned int)(t0_4 + 2);
            }
            if (cls_26 == qc) {
                unsigned int lowb_2 = (u_25 ^ (unsigned int)((int)u_25 >> 31) & 63) & 63;
                key_28 = 536870912 | lowb_2 << 6 | (unsigned int)(t0_4 + 2);
            }
            kb2o[2] = __uint_as_float(key_28);
            float sc_30 = __uint_as_float(cb[3]);
            float _fmax_144 = fmaxf(sc_30, -1.7014118346046923e+38f);
            sc_30 = _fmax_144;
            float _min_127 = fminf(sc_30, 1.7014118346046923e+38f);
            sc_30 = _min_127;
            sc_30 = sc_30;
            float sc_33 = sc_30;
            unsigned int u_34 = __as_u32(sc_33);
            unsigned int cls_35 = u_34 & 4294967232u;
            unsigned int key_37 = 0;
            if (qf < __uint_as_float(cls_35) && cls_35 < 4278190080u) {
                key_37 = 1073741824 | (unsigned int)(t0_4 + 3);
            }
            if (cls_35 == qc) {
                unsigned int lowb_3 = (u_34 ^ (unsigned int)((int)u_34 >> 31) & 63) & 63;
                key_37 = 536870912 | lowb_3 << 6 | (unsigned int)(t0_4 + 3);
            }
            kb2o[3] = __uint_as_float(key_37);
            float sc_39 = __uint_as_float(cb[4]);
            float _fmax_145 = fmaxf(sc_39, -1.7014118346046923e+38f);
            sc_39 = _fmax_145;
            float _min_128 = fminf(sc_39, 1.7014118346046923e+38f);
            sc_39 = _min_128;
            sc_39 = sc_39;
            float sc_42 = sc_39;
            unsigned int u_43 = __as_u32(sc_42);
            unsigned int cls_44 = u_43 & 4294967232u;
            unsigned int key_46 = 0;
            if (qf < __uint_as_float(cls_44) && cls_44 < 4278190080u) {
                key_46 = 1073741824 | (unsigned int)(t0_4 + 4);
            }
            if (cls_44 == qc) {
                unsigned int lowb_4 = (u_43 ^ (unsigned int)((int)u_43 >> 31) & 63) & 63;
                key_46 = 536870912 | lowb_4 << 6 | (unsigned int)(t0_4 + 4);
            }
            kb2o[4] = __uint_as_float(key_46);
            float sc_48 = __uint_as_float(cb[5]);
            float _fmax_146 = fmaxf(sc_48, -1.7014118346046923e+38f);
            sc_48 = _fmax_146;
            float _min_129 = fminf(sc_48, 1.7014118346046923e+38f);
            sc_48 = _min_129;
            sc_48 = sc_48;
            float sc_49 = sc_48;
            unsigned int u_50 = __as_u32(sc_49);
            unsigned int cls_51 = u_50 & 4294967232u;
            unsigned int key_52 = 0;
            if (qf < __uint_as_float(cls_51) && cls_51 < 4278190080u) {
                key_52 = 1073741824 | (unsigned int)(t0_4 + 5);
            }
            if (cls_51 == qc) {
                unsigned int lowb_5 = (u_50 ^ (unsigned int)((int)u_50 >> 31) & 63) & 63;
                key_52 = 536870912 | lowb_5 << 6 | (unsigned int)(t0_4 + 5);
            }
            kb2o[5] = __uint_as_float(key_52);
            float sc_53 = __uint_as_float(cb[6]);
            float _fmax_147 = fmaxf(sc_53, -1.7014118346046923e+38f);
            sc_53 = _fmax_147;
            float _min_130 = fminf(sc_53, 1.7014118346046923e+38f);
            sc_53 = _min_130;
            sc_53 = sc_53;
            float sc_54 = sc_53;
            unsigned int u_55 = __as_u32(sc_54);
            unsigned int cls_56 = u_55 & 4294967232u;
            unsigned int key_57 = 0;
            if (qf < __uint_as_float(cls_56) && cls_56 < 4278190080u) {
                key_57 = 1073741824 | (unsigned int)(t0_4 + 6);
            }
            if (cls_56 == qc) {
                unsigned int lowb_6 = (u_55 ^ (unsigned int)((int)u_55 >> 31) & 63) & 63;
                key_57 = 536870912 | lowb_6 << 6 | (unsigned int)(t0_4 + 6);
            }
            kb2o[6] = __uint_as_float(key_57);
            float sc_58 = __uint_as_float(cb[7]);
            float _fmax_148 = fmaxf(sc_58, -1.7014118346046923e+38f);
            sc_58 = _fmax_148;
            float _min_131 = fminf(sc_58, 1.7014118346046923e+38f);
            sc_58 = _min_131;
            sc_58 = sc_58;
            float sc_59 = sc_58;
            unsigned int u_60 = __as_u32(sc_59);
            unsigned int cls_61 = u_60 & 4294967232u;
            unsigned int key_62 = 0;
            if (qf < __uint_as_float(cls_61) && cls_61 < 4278190080u) {
                key_62 = 1073741824 | (unsigned int)(t0_4 + 7);
            }
            if (cls_61 == qc) {
                unsigned int lowb_7 = (u_60 ^ (unsigned int)((int)u_60 >> 31) & 63) & 63;
                key_62 = 536870912 | lowb_7 << 6 | (unsigned int)(t0_4 + 7);
            }
            kb2o[7] = __uint_as_float(key_62);
            float sc_63 = __uint_as_float(cb[8]);
            float _fmax_149 = fmaxf(sc_63, -1.7014118346046923e+38f);
            sc_63 = _fmax_149;
            float _min_132 = fminf(sc_63, 1.7014118346046923e+38f);
            sc_63 = _min_132;
            sc_63 = sc_63;
            float sc_64 = sc_63;
            unsigned int u_65 = __as_u32(sc_64);
            unsigned int cls_66 = u_65 & 4294967232u;
            unsigned int key_67 = 0;
            if (qf < __uint_as_float(cls_66) && cls_66 < 4278190080u) {
                key_67 = 1073741824 | (unsigned int)(t0_4 + 8);
            }
            if (cls_66 == qc) {
                unsigned int lowb_8 = (u_65 ^ (unsigned int)((int)u_65 >> 31) & 63) & 63;
                key_67 = 536870912 | lowb_8 << 6 | (unsigned int)(t0_4 + 8);
            }
            kb2o[8] = __uint_as_float(key_67);
            float sc_68 = __uint_as_float(cb[9]);
            float _fmax_150 = fmaxf(sc_68, -1.7014118346046923e+38f);
            sc_68 = _fmax_150;
            float _min_133 = fminf(sc_68, 1.7014118346046923e+38f);
            sc_68 = _min_133;
            sc_68 = sc_68;
            float sc_69 = sc_68;
            unsigned int u_70 = __as_u32(sc_69);
            unsigned int cls_71 = u_70 & 4294967232u;
            unsigned int key_72 = 0;
            if (qf < __uint_as_float(cls_71) && cls_71 < 4278190080u) {
                key_72 = 1073741824 | (unsigned int)(t0_4 + 9);
            }
            if (cls_71 == qc) {
                unsigned int lowb_9 = (u_70 ^ (unsigned int)((int)u_70 >> 31) & 63) & 63;
                key_72 = 536870912 | lowb_9 << 6 | (unsigned int)(t0_4 + 9);
            }
            kb2o[9] = __uint_as_float(key_72);
            float sc_73 = __uint_as_float(cb[10]);
            float _fmax_151 = fmaxf(sc_73, -1.7014118346046923e+38f);
            sc_73 = _fmax_151;
            float _min_134 = fminf(sc_73, 1.7014118346046923e+38f);
            sc_73 = _min_134;
            sc_73 = sc_73;
            float sc_74 = sc_73;
            unsigned int u_75 = __as_u32(sc_74);
            unsigned int cls_76 = u_75 & 4294967232u;
            unsigned int key_77 = 0;
            if (qf < __uint_as_float(cls_76) && cls_76 < 4278190080u) {
                key_77 = 1073741824 | (unsigned int)(t0_4 + 10);
            }
            if (cls_76 == qc) {
                unsigned int lowb_10 = (u_75 ^ (unsigned int)((int)u_75 >> 31) & 63) & 63;
                key_77 = 536870912 | lowb_10 << 6 | (unsigned int)(t0_4 + 10);
            }
            kb2o[10] = __uint_as_float(key_77);
            float sc_78 = __uint_as_float(cb[11]);
            float _fmax_152 = fmaxf(sc_78, -1.7014118346046923e+38f);
            sc_78 = _fmax_152;
            float _min_135 = fminf(sc_78, 1.7014118346046923e+38f);
            sc_78 = _min_135;
            sc_78 = sc_78;
            float sc_79 = sc_78;
            unsigned int u_80 = __as_u32(sc_79);
            unsigned int cls_81 = u_80 & 4294967232u;
            unsigned int key_82 = 0;
            if (qf < __uint_as_float(cls_81) && cls_81 < 4278190080u) {
                key_82 = 1073741824 | (unsigned int)(t0_4 + 11);
            }
            if (cls_81 == qc) {
                unsigned int lowb_11 = (u_80 ^ (unsigned int)((int)u_80 >> 31) & 63) & 63;
                key_82 = 536870912 | lowb_11 << 6 | (unsigned int)(t0_4 + 11);
            }
            kb2o[11] = __uint_as_float(key_82);
            float sc_83 = __uint_as_float(cb[12]);
            float _fmax_153 = fmaxf(sc_83, -1.7014118346046923e+38f);
            sc_83 = _fmax_153;
            float _min_136 = fminf(sc_83, 1.7014118346046923e+38f);
            sc_83 = _min_136;
            sc_83 = sc_83;
            float sc_84 = sc_83;
            unsigned int u_85 = __as_u32(sc_84);
            unsigned int cls_86 = u_85 & 4294967232u;
            unsigned int key_87 = 0;
            if (qf < __uint_as_float(cls_86) && cls_86 < 4278190080u) {
                key_87 = 1073741824 | (unsigned int)(t0_4 + 12);
            }
            if (cls_86 == qc) {
                unsigned int lowb_12 = (u_85 ^ (unsigned int)((int)u_85 >> 31) & 63) & 63;
                key_87 = 536870912 | lowb_12 << 6 | (unsigned int)(t0_4 + 12);
            }
            kb2o[12] = __uint_as_float(key_87);
            float sc_88 = __uint_as_float(cb[13]);
            float _fmax_154 = fmaxf(sc_88, -1.7014118346046923e+38f);
            sc_88 = _fmax_154;
            float _min_137 = fminf(sc_88, 1.7014118346046923e+38f);
            sc_88 = _min_137;
            sc_88 = sc_88;
            float sc_89 = sc_88;
            unsigned int u_90 = __as_u32(sc_89);
            unsigned int cls_91 = u_90 & 4294967232u;
            unsigned int key_92 = 0;
            if (qf < __uint_as_float(cls_91) && cls_91 < 4278190080u) {
                key_92 = 1073741824 | (unsigned int)(t0_4 + 13);
            }
            if (cls_91 == qc) {
                unsigned int lowb_13 = (u_90 ^ (unsigned int)((int)u_90 >> 31) & 63) & 63;
                key_92 = 536870912 | lowb_13 << 6 | (unsigned int)(t0_4 + 13);
            }
            kb2o[13] = __uint_as_float(key_92);
            float sc_93 = __uint_as_float(cb[14]);
            float _fmax_155 = fmaxf(sc_93, -1.7014118346046923e+38f);
            sc_93 = _fmax_155;
            float _min_138 = fminf(sc_93, 1.7014118346046923e+38f);
            sc_93 = _min_138;
            sc_93 = sc_93;
            float sc_94 = sc_93;
            unsigned int u_95 = __as_u32(sc_94);
            unsigned int cls_96 = u_95 & 4294967232u;
            unsigned int key_97 = 0;
            if (qf < __uint_as_float(cls_96) && cls_96 < 4278190080u) {
                key_97 = 1073741824 | (unsigned int)(t0_4 + 14);
            }
            if (cls_96 == qc) {
                unsigned int lowb_14 = (u_95 ^ (unsigned int)((int)u_95 >> 31) & 63) & 63;
                key_97 = 536870912 | lowb_14 << 6 | (unsigned int)(t0_4 + 14);
            }
            kb2o[14] = __uint_as_float(key_97);
            float sc_98 = __uint_as_float(cb[15]);
            float _fmax_156 = fmaxf(sc_98, -1.7014118346046923e+38f);
            sc_98 = _fmax_156;
            float _min_139 = fminf(sc_98, 1.7014118346046923e+38f);
            sc_98 = _min_139;
            sc_98 = sc_98;
            float sc_99 = sc_98;
            unsigned int u_100 = __as_u32(sc_99);
            unsigned int cls_101 = u_100 & 4294967232u;
            unsigned int key_102 = 0;
            if (qf < __uint_as_float(cls_101) && cls_101 < 4278190080u) {
                key_102 = 1073741824 | (unsigned int)(t0_4 + 15);
            }
            if (cls_101 == qc) {
                unsigned int lowb_15 = (u_100 ^ (unsigned int)((int)u_100 >> 31) & 63) & 63;
                key_102 = 536870912 | lowb_15 << 6 | (unsigned int)(t0_4 + 15);
            }
            kb2o[15] = __uint_as_float(key_102);
            float _fmax_157 = fmaxf(kb2o[0], kb2o[13]);
            float hi_104 = _fmax_157;
            float _min_140 = fminf(kb2o[0], kb2o[13]);
            float lo_105 = _min_140;
            kb2o[0] = hi_104;
            kb2o[13] = lo_105;
            float _fmax_158 = fmaxf(kb2o[1], kb2o[12]);
            float hi_106 = _fmax_158;
            float _min_141 = fminf(kb2o[1], kb2o[12]);
            float lo_107 = _min_141;
            kb2o[1] = hi_106;
            kb2o[12] = lo_107;
            float _fmax_159 = fmaxf(kb2o[2], kb2o[15]);
            float hi_108 = _fmax_159;
            float _min_142 = fminf(kb2o[2], kb2o[15]);
            float lo_109 = _min_142;
            kb2o[2] = hi_108;
            kb2o[15] = lo_109;
            float _fmax_160 = fmaxf(kb2o[3], kb2o[14]);
            float hi_110 = _fmax_160;
            float _min_143 = fminf(kb2o[3], kb2o[14]);
            float lo_111 = _min_143;
            kb2o[3] = hi_110;
            kb2o[14] = lo_111;
            float _fmax_161 = fmaxf(kb2o[4], kb2o[8]);
            float hi_112 = _fmax_161;
            float _min_144 = fminf(kb2o[4], kb2o[8]);
            float lo_113 = _min_144;
            kb2o[4] = hi_112;
            kb2o[8] = lo_113;
            float _fmax_162 = fmaxf(kb2o[5], kb2o[6]);
            float hi_114 = _fmax_162;
            float _min_145 = fminf(kb2o[5], kb2o[6]);
            float lo_115 = _min_145;
            kb2o[5] = hi_114;
            kb2o[6] = lo_115;
            float _fmax_163 = fmaxf(kb2o[7], kb2o[11]);
            float hi_116 = _fmax_163;
            float _min_146 = fminf(kb2o[7], kb2o[11]);
            float lo_117 = _min_146;
            kb2o[7] = hi_116;
            kb2o[11] = lo_117;
            float _fmax_164 = fmaxf(kb2o[9], kb2o[10]);
            float hi_118 = _fmax_164;
            float _min_147 = fminf(kb2o[9], kb2o[10]);
            float lo_119 = _min_147;
            kb2o[9] = hi_118;
            kb2o[10] = lo_119;
            float _fmax_165 = fmaxf(kb2o[0], kb2o[5]);
            float hi_120 = _fmax_165;
            float _min_148 = fminf(kb2o[0], kb2o[5]);
            float lo_121 = _min_148;
            kb2o[0] = hi_120;
            kb2o[5] = lo_121;
            float _fmax_166 = fmaxf(kb2o[1], kb2o[7]);
            float hi_122 = _fmax_166;
            float _min_149 = fminf(kb2o[1], kb2o[7]);
            float lo_123 = _min_149;
            kb2o[1] = hi_122;
            kb2o[7] = lo_123;
            float _fmax_167 = fmaxf(kb2o[2], kb2o[9]);
            float hi_124 = _fmax_167;
            float _min_150 = fminf(kb2o[2], kb2o[9]);
            float lo_125 = _min_150;
            kb2o[2] = hi_124;
            kb2o[9] = lo_125;
            float _fmax_168 = fmaxf(kb2o[3], kb2o[4]);
            float hi_126 = _fmax_168;
            float _min_151 = fminf(kb2o[3], kb2o[4]);
            float lo_127 = _min_151;
            kb2o[3] = hi_126;
            kb2o[4] = lo_127;
            float _fmax_169 = fmaxf(kb2o[6], kb2o[13]);
            float hi_128 = _fmax_169;
            float _min_152 = fminf(kb2o[6], kb2o[13]);
            float lo_129 = _min_152;
            kb2o[6] = hi_128;
            kb2o[13] = lo_129;
            float _fmax_170 = fmaxf(kb2o[8], kb2o[14]);
            float hi_130 = _fmax_170;
            float _min_153 = fminf(kb2o[8], kb2o[14]);
            float lo_131 = _min_153;
            kb2o[8] = hi_130;
            kb2o[14] = lo_131;
            float _fmax_171 = fmaxf(kb2o[10], kb2o[15]);
            float hi_132 = _fmax_171;
            float _min_154 = fminf(kb2o[10], kb2o[15]);
            float lo_133 = _min_154;
            kb2o[10] = hi_132;
            kb2o[15] = lo_133;
            float _fmax_172 = fmaxf(kb2o[11], kb2o[12]);
            float hi_134 = _fmax_172;
            float _min_155 = fminf(kb2o[11], kb2o[12]);
            float lo_135 = _min_155;
            kb2o[11] = hi_134;
            kb2o[12] = lo_135;
            float _fmax_173 = fmaxf(kb2o[0], kb2o[1]);
            float hi_136 = _fmax_173;
            float _min_156 = fminf(kb2o[0], kb2o[1]);
            float lo_137 = _min_156;
            kb2o[0] = hi_136;
            kb2o[1] = lo_137;
            float _fmax_174 = fmaxf(kb2o[2], kb2o[3]);
            float hi_138 = _fmax_174;
            float _min_157 = fminf(kb2o[2], kb2o[3]);
            float lo_139 = _min_157;
            kb2o[2] = hi_138;
            kb2o[3] = lo_139;
            float _fmax_175 = fmaxf(kb2o[4], kb2o[5]);
            float hi_140 = _fmax_175;
            float _min_158 = fminf(kb2o[4], kb2o[5]);
            float lo_141 = _min_158;
            kb2o[4] = hi_140;
            kb2o[5] = lo_141;
            float _fmax_176 = fmaxf(kb2o[6], kb2o[8]);
            float hi_142 = _fmax_176;
            float _min_159 = fminf(kb2o[6], kb2o[8]);
            float lo_143 = _min_159;
            kb2o[6] = hi_142;
            kb2o[8] = lo_143;
            float _fmax_177 = fmaxf(kb2o[7], kb2o[9]);
            float hi_144 = _fmax_177;
            float _min_160 = fminf(kb2o[7], kb2o[9]);
            float lo_145 = _min_160;
            kb2o[7] = hi_144;
            kb2o[9] = lo_145;
            float _fmax_178 = fmaxf(kb2o[10], kb2o[11]);
            float hi_146 = _fmax_178;
            float _min_161 = fminf(kb2o[10], kb2o[11]);
            float lo_147 = _min_161;
            kb2o[10] = hi_146;
            kb2o[11] = lo_147;
            float _fmax_179 = fmaxf(kb2o[12], kb2o[13]);
            float hi_148 = _fmax_179;
            float _min_162 = fminf(kb2o[12], kb2o[13]);
            float lo_149 = _min_162;
            kb2o[12] = hi_148;
            kb2o[13] = lo_149;
            float _fmax_180 = fmaxf(kb2o[14], kb2o[15]);
            float hi_150 = _fmax_180;
            float _min_163 = fminf(kb2o[14], kb2o[15]);
            float lo_151 = _min_163;
            kb2o[14] = hi_150;
            kb2o[15] = lo_151;
            float _fmax_181 = fmaxf(kb2o[0], kb2o[2]);
            float hi_152 = _fmax_181;
            float _min_164 = fminf(kb2o[0], kb2o[2]);
            float lo_153 = _min_164;
            kb2o[0] = hi_152;
            kb2o[2] = lo_153;
            float _fmax_182 = fmaxf(kb2o[1], kb2o[3]);
            float hi_154 = _fmax_182;
            float _min_165 = fminf(kb2o[1], kb2o[3]);
            float lo_155 = _min_165;
            kb2o[1] = hi_154;
            kb2o[3] = lo_155;
            float _fmax_183 = fmaxf(kb2o[4], kb2o[10]);
            float hi_156 = _fmax_183;
            float _min_166 = fminf(kb2o[4], kb2o[10]);
            float lo_157 = _min_166;
            kb2o[4] = hi_156;
            kb2o[10] = lo_157;
            float _fmax_184 = fmaxf(kb2o[5], kb2o[11]);
            float hi_158 = _fmax_184;
            float _min_167 = fminf(kb2o[5], kb2o[11]);
            float lo_159 = _min_167;
            kb2o[5] = hi_158;
            kb2o[11] = lo_159;
            float _fmax_185 = fmaxf(kb2o[6], kb2o[7]);
            float hi_160 = _fmax_185;
            float _min_168 = fminf(kb2o[6], kb2o[7]);
            float lo_161 = _min_168;
            kb2o[6] = hi_160;
            kb2o[7] = lo_161;
            float _fmax_186 = fmaxf(kb2o[8], kb2o[9]);
            float hi_162 = _fmax_186;
            float _min_169 = fminf(kb2o[8], kb2o[9]);
            float lo_163 = _min_169;
            kb2o[8] = hi_162;
            kb2o[9] = lo_163;
            float _fmax_187 = fmaxf(kb2o[12], kb2o[14]);
            float hi_164 = _fmax_187;
            float _min_170 = fminf(kb2o[12], kb2o[14]);
            float lo_165 = _min_170;
            kb2o[12] = hi_164;
            kb2o[14] = lo_165;
            float _fmax_188 = fmaxf(kb2o[13], kb2o[15]);
            float hi_166 = _fmax_188;
            float _min_171 = fminf(kb2o[13], kb2o[15]);
            float lo_167 = _min_171;
            kb2o[13] = hi_166;
            kb2o[15] = lo_167;
            float _fmax_189 = fmaxf(kb2o[1], kb2o[2]);
            float hi_168 = _fmax_189;
            float _min_172 = fminf(kb2o[1], kb2o[2]);
            float lo_169 = _min_172;
            kb2o[1] = hi_168;
            kb2o[2] = lo_169;
            float _fmax_190 = fmaxf(kb2o[3], kb2o[12]);
            float hi_170 = _fmax_190;
            float _min_173 = fminf(kb2o[3], kb2o[12]);
            float lo_171 = _min_173;
            kb2o[3] = hi_170;
            kb2o[12] = lo_171;
            float _fmax_191 = fmaxf(kb2o[4], kb2o[6]);
            float hi_172 = _fmax_191;
            float _min_174 = fminf(kb2o[4], kb2o[6]);
            float lo_173 = _min_174;
            kb2o[4] = hi_172;
            kb2o[6] = lo_173;
            float _fmax_192 = fmaxf(kb2o[5], kb2o[7]);
            float hi_174 = _fmax_192;
            float _min_175 = fminf(kb2o[5], kb2o[7]);
            float lo_175 = _min_175;
            kb2o[5] = hi_174;
            kb2o[7] = lo_175;
            float _fmax_193 = fmaxf(kb2o[8], kb2o[10]);
            float hi_176 = _fmax_193;
            float _min_176 = fminf(kb2o[8], kb2o[10]);
            float lo_177 = _min_176;
            kb2o[8] = hi_176;
            kb2o[10] = lo_177;
            float _fmax_194 = fmaxf(kb2o[9], kb2o[11]);
            float hi_178 = _fmax_194;
            float _min_177 = fminf(kb2o[9], kb2o[11]);
            float lo_179 = _min_177;
            kb2o[9] = hi_178;
            kb2o[11] = lo_179;
            float _fmax_195 = fmaxf(kb2o[13], kb2o[14]);
            float hi_180 = _fmax_195;
            float _min_178 = fminf(kb2o[13], kb2o[14]);
            float lo_181 = _min_178;
            kb2o[13] = hi_180;
            kb2o[14] = lo_181;
            float _fmax_196 = fmaxf(kb2o[1], kb2o[4]);
            float hi_182 = _fmax_196;
            float _min_179 = fminf(kb2o[1], kb2o[4]);
            float lo_183 = _min_179;
            kb2o[1] = hi_182;
            kb2o[4] = lo_183;
            float _fmax_197 = fmaxf(kb2o[2], kb2o[6]);
            float hi_184 = _fmax_197;
            float _min_180 = fminf(kb2o[2], kb2o[6]);
            float lo_185 = _min_180;
            kb2o[2] = hi_184;
            kb2o[6] = lo_185;
            float _fmax_198 = fmaxf(kb2o[5], kb2o[8]);
            float hi_186 = _fmax_198;
            float _min_181 = fminf(kb2o[5], kb2o[8]);
            float lo_187 = _min_181;
            kb2o[5] = hi_186;
            kb2o[8] = lo_187;
            float _fmax_199 = fmaxf(kb2o[7], kb2o[10]);
            float hi_188 = _fmax_199;
            float _min_182 = fminf(kb2o[7], kb2o[10]);
            float lo_189 = _min_182;
            kb2o[7] = hi_188;
            kb2o[10] = lo_189;
            float _fmax_200 = fmaxf(kb2o[9], kb2o[13]);
            float hi_190 = _fmax_200;
            float _min_183 = fminf(kb2o[9], kb2o[13]);
            float lo_191 = _min_183;
            kb2o[9] = hi_190;
            kb2o[13] = lo_191;
            float _fmax_201 = fmaxf(kb2o[11], kb2o[14]);
            float hi_192 = _fmax_201;
            float _min_184 = fminf(kb2o[11], kb2o[14]);
            float lo_193 = _min_184;
            kb2o[11] = hi_192;
            kb2o[14] = lo_193;
            float _fmax_202 = fmaxf(kb2o[2], kb2o[4]);
            float hi_194 = _fmax_202;
            float _min_185 = fminf(kb2o[2], kb2o[4]);
            float lo_195 = _min_185;
            kb2o[2] = hi_194;
            kb2o[4] = lo_195;
            float _fmax_203 = fmaxf(kb2o[3], kb2o[6]);
            float hi_196 = _fmax_203;
            float _min_186 = fminf(kb2o[3], kb2o[6]);
            float lo_197 = _min_186;
            kb2o[3] = hi_196;
            kb2o[6] = lo_197;
            float _fmax_204 = fmaxf(kb2o[9], kb2o[12]);
            float hi_198 = _fmax_204;
            float _min_187 = fminf(kb2o[9], kb2o[12]);
            float lo_199 = _min_187;
            kb2o[9] = hi_198;
            kb2o[12] = lo_199;
            float _fmax_205 = fmaxf(kb2o[11], kb2o[13]);
            float hi_200 = _fmax_205;
            float _min_188 = fminf(kb2o[11], kb2o[13]);
            float lo_201 = _min_188;
            kb2o[11] = hi_200;
            kb2o[13] = lo_201;
            float _fmax_206 = fmaxf(kb2o[3], kb2o[5]);
            float hi_202 = _fmax_206;
            float _min_189 = fminf(kb2o[3], kb2o[5]);
            float lo_203 = _min_189;
            kb2o[3] = hi_202;
            kb2o[5] = lo_203;
            float _fmax_207 = fmaxf(kb2o[6], kb2o[8]);
            float hi_204 = _fmax_207;
            float _min_190 = fminf(kb2o[6], kb2o[8]);
            float lo_205 = _min_190;
            kb2o[6] = hi_204;
            kb2o[8] = lo_205;
            float _fmax_208 = fmaxf(kb2o[7], kb2o[9]);
            float hi_206 = _fmax_208;
            float _min_191 = fminf(kb2o[7], kb2o[9]);
            float lo_207 = _min_191;
            kb2o[7] = hi_206;
            kb2o[9] = lo_207;
            float _fmax_209 = fmaxf(kb2o[10], kb2o[12]);
            float hi_208 = _fmax_209;
            float _min_192 = fminf(kb2o[10], kb2o[12]);
            float lo_209 = _min_192;
            kb2o[10] = hi_208;
            kb2o[12] = lo_209;
            float _fmax_210 = fmaxf(kb2o[3], kb2o[4]);
            float hi_210 = _fmax_210;
            float _min_193 = fminf(kb2o[3], kb2o[4]);
            float lo_211 = _min_193;
            kb2o[3] = hi_210;
            kb2o[4] = lo_211;
            float _fmax_211 = fmaxf(kb2o[5], kb2o[6]);
            float hi_212 = _fmax_211;
            float _min_194 = fminf(kb2o[5], kb2o[6]);
            float lo_213 = _min_194;
            kb2o[5] = hi_212;
            kb2o[6] = lo_213;
            float _fmax_212 = fmaxf(kb2o[7], kb2o[8]);
            float hi_214 = _fmax_212;
            float _min_195 = fminf(kb2o[7], kb2o[8]);
            float lo_215 = _min_195;
            kb2o[7] = hi_214;
            kb2o[8] = lo_215;
            float _fmax_213 = fmaxf(kb2o[9], kb2o[10]);
            float hi_216 = _fmax_213;
            float _min_196 = fminf(kb2o[9], kb2o[10]);
            float lo_217 = _min_196;
            kb2o[9] = hi_216;
            kb2o[10] = lo_217;
            float _fmax_214 = fmaxf(kb2o[11], kb2o[12]);
            float hi_218 = _fmax_214;
            float _min_197 = fminf(kb2o[11], kb2o[12]);
            float lo_219 = _min_197;
            kb2o[11] = hi_218;
            kb2o[12] = lo_219;
            float _fmax_215 = fmaxf(kb2o[6], kb2o[7]);
            float hi_220 = _fmax_215;
            float _min_198 = fminf(kb2o[6], kb2o[7]);
            float lo_221 = _min_198;
            kb2o[6] = hi_220;
            kb2o[7] = lo_221;
            float _fmax_216 = fmaxf(kb2o[8], kb2o[9]);
            float hi_222 = _fmax_216;
            float _min_199 = fminf(kb2o[8], kb2o[9]);
            float lo_223 = _min_199;
            kb2o[8] = hi_222;
            kb2o[9] = lo_223;
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
        #pragma unroll 1
        for (int k_1 = 0; k_1 < 2; k_1++) {
            int m2 = 8 << k_1;
            float ob2[16];
            float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, a2[0], m2);
            ob2[0] = _shfl_xor_17;
            float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, a2[1], m2);
            ob2[1] = _shfl_xor_18;
            float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, a2[2], m2);
            ob2[2] = _shfl_xor_19;
            float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, a2[3], m2);
            ob2[3] = _shfl_xor_20;
            float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, a2[4], m2);
            ob2[4] = _shfl_xor_21;
            float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, a2[5], m2);
            ob2[5] = _shfl_xor_22;
            float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, a2[6], m2);
            ob2[6] = _shfl_xor_23;
            float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, a2[7], m2);
            ob2[7] = _shfl_xor_24;
            float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, a2[8], m2);
            ob2[8] = _shfl_xor_25;
            float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, a2[9], m2);
            ob2[9] = _shfl_xor_26;
            float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, a2[10], m2);
            ob2[10] = _shfl_xor_27;
            float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, a2[11], m2);
            ob2[11] = _shfl_xor_28;
            float _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, a2[12], m2);
            ob2[12] = _shfl_xor_29;
            float _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, a2[13], m2);
            ob2[13] = _shfl_xor_30;
            float _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, a2[14], m2);
            ob2[14] = _shfl_xor_31;
            float _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, a2[15], m2);
            ob2[15] = _shfl_xor_32;
            float _fmax_217 = fmaxf(a2[0], ob2[15]);
            float hi_0_1 = _fmax_217;
            a2[0] = hi_0_1;
            float _fmax_218 = fmaxf(a2[1], ob2[14]);
            float hi_1 = _fmax_218;
            a2[1] = hi_1;
            float _fmax_219 = fmaxf(a2[2], ob2[13]);
            float hi_2_1 = _fmax_219;
            a2[2] = hi_2_1;
            float _fmax_220 = fmaxf(a2[3], ob2[12]);
            float hi_3 = _fmax_220;
            a2[3] = hi_3;
            float _fmax_221 = fmaxf(a2[4], ob2[11]);
            float hi_4_1 = _fmax_221;
            a2[4] = hi_4_1;
            float _fmax_222 = fmaxf(a2[5], ob2[10]);
            float hi_5 = _fmax_222;
            a2[5] = hi_5;
            float _fmax_223 = fmaxf(a2[6], ob2[9]);
            float hi_6_1 = _fmax_223;
            a2[6] = hi_6_1;
            float _fmax_224 = fmaxf(a2[7], ob2[8]);
            float hi_7 = _fmax_224;
            a2[7] = hi_7;
            float _fmax_225 = fmaxf(a2[8], ob2[7]);
            float hi_8_1 = _fmax_225;
            a2[8] = hi_8_1;
            float _fmax_226 = fmaxf(a2[9], ob2[6]);
            float hi_9 = _fmax_226;
            a2[9] = hi_9;
            float _fmax_227 = fmaxf(a2[10], ob2[5]);
            float hi_10_1 = _fmax_227;
            a2[10] = hi_10_1;
            float _fmax_228 = fmaxf(a2[11], ob2[4]);
            float hi_11 = _fmax_228;
            a2[11] = hi_11;
            float _fmax_229 = fmaxf(a2[12], ob2[3]);
            float hi_12_1 = _fmax_229;
            a2[12] = hi_12_1;
            float _fmax_230 = fmaxf(a2[13], ob2[2]);
            float hi_13 = _fmax_230;
            a2[13] = hi_13;
            float _fmax_231 = fmaxf(a2[14], ob2[1]);
            float hi_14_1 = _fmax_231;
            a2[14] = hi_14_1;
            float _fmax_232 = fmaxf(a2[15], ob2[0]);
            float hi_15 = _fmax_232;
            a2[15] = hi_15;
            float _fmax_233 = fmaxf(a2[0], a2[8]);
            float hi_16_1 = _fmax_233;
            float _min_200 = fminf(a2[0], a2[8]);
            float lo_17_1 = _min_200;
            a2[0] = hi_16_1;
            a2[8] = lo_17_1;
            float _fmax_234 = fmaxf(a2[1], a2[9]);
            float hi_18_1 = _fmax_234;
            float _min_201 = fminf(a2[1], a2[9]);
            float lo_19_1 = _min_201;
            a2[1] = hi_18_1;
            a2[9] = lo_19_1;
            float _fmax_235 = fmaxf(a2[2], a2[10]);
            float hi_20_1 = _fmax_235;
            float _min_202 = fminf(a2[2], a2[10]);
            float lo_21_1 = _min_202;
            a2[2] = hi_20_1;
            a2[10] = lo_21_1;
            float _fmax_236 = fmaxf(a2[3], a2[11]);
            float hi_22_1 = _fmax_236;
            float _min_203 = fminf(a2[3], a2[11]);
            float lo_23_1 = _min_203;
            a2[3] = hi_22_1;
            a2[11] = lo_23_1;
            float _fmax_237 = fmaxf(a2[4], a2[12]);
            float hi_24_1 = _fmax_237;
            float _min_204 = fminf(a2[4], a2[12]);
            float lo_25_1 = _min_204;
            a2[4] = hi_24_1;
            a2[12] = lo_25_1;
            float _fmax_238 = fmaxf(a2[5], a2[13]);
            float hi_26_1 = _fmax_238;
            float _min_205 = fminf(a2[5], a2[13]);
            float lo_27_1 = _min_205;
            a2[5] = hi_26_1;
            a2[13] = lo_27_1;
            float _fmax_239 = fmaxf(a2[6], a2[14]);
            float hi_28_1 = _fmax_239;
            float _min_206 = fminf(a2[6], a2[14]);
            float lo_29_1 = _min_206;
            a2[6] = hi_28_1;
            a2[14] = lo_29_1;
            float _fmax_240 = fmaxf(a2[7], a2[15]);
            float hi_30_1 = _fmax_240;
            float _min_207 = fminf(a2[7], a2[15]);
            float lo_31_1 = _min_207;
            a2[7] = hi_30_1;
            a2[15] = lo_31_1;
            float _fmax_241 = fmaxf(a2[0], a2[4]);
            float hi_32_1 = _fmax_241;
            float _min_208 = fminf(a2[0], a2[4]);
            float lo_33_1 = _min_208;
            a2[0] = hi_32_1;
            a2[4] = lo_33_1;
            float _fmax_242 = fmaxf(a2[1], a2[5]);
            float hi_34_1 = _fmax_242;
            float _min_209 = fminf(a2[1], a2[5]);
            float lo_35_1 = _min_209;
            a2[1] = hi_34_1;
            a2[5] = lo_35_1;
            float _fmax_243 = fmaxf(a2[2], a2[6]);
            float hi_36_1 = _fmax_243;
            float _min_210 = fminf(a2[2], a2[6]);
            float lo_37_1 = _min_210;
            a2[2] = hi_36_1;
            a2[6] = lo_37_1;
            float _fmax_244 = fmaxf(a2[3], a2[7]);
            float hi_38_1 = _fmax_244;
            float _min_211 = fminf(a2[3], a2[7]);
            float lo_39_1 = _min_211;
            a2[3] = hi_38_1;
            a2[7] = lo_39_1;
            float _fmax_245 = fmaxf(a2[8], a2[12]);
            float hi_40_1 = _fmax_245;
            float _min_212 = fminf(a2[8], a2[12]);
            float lo_41_1 = _min_212;
            a2[8] = hi_40_1;
            a2[12] = lo_41_1;
            float _fmax_246 = fmaxf(a2[9], a2[13]);
            float hi_42_1 = _fmax_246;
            float _min_213 = fminf(a2[9], a2[13]);
            float lo_43_1 = _min_213;
            a2[9] = hi_42_1;
            a2[13] = lo_43_1;
            float _fmax_247 = fmaxf(a2[10], a2[14]);
            float hi_44_1 = _fmax_247;
            float _min_214 = fminf(a2[10], a2[14]);
            float lo_45_1 = _min_214;
            a2[10] = hi_44_1;
            a2[14] = lo_45_1;
            float _fmax_248 = fmaxf(a2[11], a2[15]);
            float hi_46_1 = _fmax_248;
            float _min_215 = fminf(a2[11], a2[15]);
            float lo_47_1 = _min_215;
            a2[11] = hi_46_1;
            a2[15] = lo_47_1;
            float _fmax_249 = fmaxf(a2[0], a2[2]);
            float hi_48_1 = _fmax_249;
            float _min_216 = fminf(a2[0], a2[2]);
            float lo_49_1 = _min_216;
            a2[0] = hi_48_1;
            a2[2] = lo_49_1;
            float _fmax_250 = fmaxf(a2[1], a2[3]);
            float hi_50_1 = _fmax_250;
            float _min_217 = fminf(a2[1], a2[3]);
            float lo_51_1 = _min_217;
            a2[1] = hi_50_1;
            a2[3] = lo_51_1;
            float _fmax_251 = fmaxf(a2[4], a2[6]);
            float hi_52_1 = _fmax_251;
            float _min_218 = fminf(a2[4], a2[6]);
            float lo_53_1 = _min_218;
            a2[4] = hi_52_1;
            a2[6] = lo_53_1;
            float _fmax_252 = fmaxf(a2[5], a2[7]);
            float hi_54_1 = _fmax_252;
            float _min_219 = fminf(a2[5], a2[7]);
            float lo_55_1 = _min_219;
            a2[5] = hi_54_1;
            a2[7] = lo_55_1;
            float _fmax_253 = fmaxf(a2[8], a2[10]);
            float hi_56_1 = _fmax_253;
            float _min_220 = fminf(a2[8], a2[10]);
            float lo_57_1 = _min_220;
            a2[8] = hi_56_1;
            a2[10] = lo_57_1;
            float _fmax_254 = fmaxf(a2[9], a2[11]);
            float hi_58_1 = _fmax_254;
            float _min_221 = fminf(a2[9], a2[11]);
            float lo_59_1 = _min_221;
            a2[9] = hi_58_1;
            a2[11] = lo_59_1;
            float _fmax_255 = fmaxf(a2[12], a2[14]);
            float hi_60_1 = _fmax_255;
            float _min_222 = fminf(a2[12], a2[14]);
            float lo_61_1 = _min_222;
            a2[12] = hi_60_1;
            a2[14] = lo_61_1;
            float _fmax_256 = fmaxf(a2[13], a2[15]);
            float hi_62_1 = _fmax_256;
            float _min_223 = fminf(a2[13], a2[15]);
            float lo_63_1 = _min_223;
            a2[13] = hi_62_1;
            a2[15] = lo_63_1;
            float _fmax_257 = fmaxf(a2[0], a2[1]);
            float hi_64_1 = _fmax_257;
            float _min_224 = fminf(a2[0], a2[1]);
            float lo_65_1 = _min_224;
            a2[0] = hi_64_1;
            a2[1] = lo_65_1;
            float _fmax_258 = fmaxf(a2[2], a2[3]);
            float hi_66_1 = _fmax_258;
            float _min_225 = fminf(a2[2], a2[3]);
            float lo_67_1 = _min_225;
            a2[2] = hi_66_1;
            a2[3] = lo_67_1;
            float _fmax_259 = fmaxf(a2[4], a2[5]);
            float hi_68_1 = _fmax_259;
            float _min_226 = fminf(a2[4], a2[5]);
            float lo_69_1 = _min_226;
            a2[4] = hi_68_1;
            a2[5] = lo_69_1;
            float _fmax_260 = fmaxf(a2[6], a2[7]);
            float hi_70_1 = _fmax_260;
            float _min_227 = fminf(a2[6], a2[7]);
            float lo_71_1 = _min_227;
            a2[6] = hi_70_1;
            a2[7] = lo_71_1;
            float _fmax_261 = fmaxf(a2[8], a2[9]);
            float hi_72_1 = _fmax_261;
            float _min_228 = fminf(a2[8], a2[9]);
            float lo_73_1 = _min_228;
            a2[8] = hi_72_1;
            a2[9] = lo_73_1;
            float _fmax_262 = fmaxf(a2[10], a2[11]);
            float hi_74_1 = _fmax_262;
            float _min_229 = fminf(a2[10], a2[11]);
            float lo_75_1 = _min_229;
            a2[10] = hi_74_1;
            a2[11] = lo_75_1;
            float _fmax_263 = fmaxf(a2[12], a2[13]);
            float hi_76_1 = _fmax_263;
            float _min_230 = fminf(a2[12], a2[13]);
            float lo_77_1 = _min_230;
            a2[12] = hi_76_1;
            a2[13] = lo_77_1;
            float _fmax_264 = fmaxf(a2[14], a2[15]);
            float hi_78_1 = _fmax_264;
            float _min_231 = fminf(a2[14], a2[15]);
            float lo_79_1 = _min_231;
            a2[14] = hi_78_1;
            a2[15] = lo_79_1;
        }
    }
    if (w == 0 && col < total_q) {
        unsigned int o[16];
        unsigned int k1 = __as_u32(a[0]);
        unsigned int idx = 4294967295;
        if (k1 < 4278190080u) {
            idx = k1 & 63;
        }
        if (flagged != 0) {
            unsigned int k2 = __as_u32(a2[0]);
            idx = 4294967295;
            if (k2 != 0) {
                idx = k2 & 63;
            }
        }
        o[0] = idx;
        unsigned int k1_0 = __as_u32(a[1]);
        unsigned int idx_1 = 4294967295;
        if (k1_0 < 4278190080u) {
            idx_1 = k1_0 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_1 = __as_u32(a2[1]);
            idx_1 = 4294967295;
            if (k2_1 != 0) {
                idx_1 = k2_1 & 63;
            }
        }
        o[1] = idx_1;
        unsigned int k1_2 = __as_u32(a[2]);
        unsigned int idx_3 = 4294967295;
        if (k1_2 < 4278190080u) {
            idx_3 = k1_2 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_2 = __as_u32(a2[2]);
            idx_3 = 4294967295;
            if (k2_2 != 0) {
                idx_3 = k2_2 & 63;
            }
        }
        o[2] = idx_3;
        unsigned int k1_4 = __as_u32(a[3]);
        unsigned int idx_5 = 4294967295;
        if (k1_4 < 4278190080u) {
            idx_5 = k1_4 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_3 = __as_u32(a2[3]);
            idx_5 = 4294967295;
            if (k2_3 != 0) {
                idx_5 = k2_3 & 63;
            }
        }
        o[3] = idx_5;
        unsigned int k1_6 = __as_u32(a[4]);
        unsigned int idx_7 = 4294967295;
        if (k1_6 < 4278190080u) {
            idx_7 = k1_6 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_4 = __as_u32(a2[4]);
            idx_7 = 4294967295;
            if (k2_4 != 0) {
                idx_7 = k2_4 & 63;
            }
        }
        o[4] = idx_7;
        unsigned int k1_8 = __as_u32(a[5]);
        unsigned int idx_9 = 4294967295;
        if (k1_8 < 4278190080u) {
            idx_9 = k1_8 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_5 = __as_u32(a2[5]);
            idx_9 = 4294967295;
            if (k2_5 != 0) {
                idx_9 = k2_5 & 63;
            }
        }
        o[5] = idx_9;
        unsigned int k1_10 = __as_u32(a[6]);
        unsigned int idx_11 = 4294967295;
        if (k1_10 < 4278190080u) {
            idx_11 = k1_10 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_6 = __as_u32(a2[6]);
            idx_11 = 4294967295;
            if (k2_6 != 0) {
                idx_11 = k2_6 & 63;
            }
        }
        o[6] = idx_11;
        unsigned int k1_12 = __as_u32(a[7]);
        unsigned int idx_13 = 4294967295;
        if (k1_12 < 4278190080u) {
            idx_13 = k1_12 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_7 = __as_u32(a2[7]);
            idx_13 = 4294967295;
            if (k2_7 != 0) {
                idx_13 = k2_7 & 63;
            }
        }
        o[7] = idx_13;
        unsigned int k1_14 = __as_u32(a[8]);
        unsigned int idx_15 = 4294967295;
        if (k1_14 < 4278190080u) {
            idx_15 = k1_14 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_8 = __as_u32(a2[8]);
            idx_15 = 4294967295;
            if (k2_8 != 0) {
                idx_15 = k2_8 & 63;
            }
        }
        o[8] = idx_15;
        unsigned int k1_16 = __as_u32(a[9]);
        unsigned int idx_17 = 4294967295;
        if (k1_16 < 4278190080u) {
            idx_17 = k1_16 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_9 = __as_u32(a2[9]);
            idx_17 = 4294967295;
            if (k2_9 != 0) {
                idx_17 = k2_9 & 63;
            }
        }
        o[9] = idx_17;
        unsigned int k1_18 = __as_u32(a[10]);
        unsigned int idx_19 = 4294967295;
        if (k1_18 < 4278190080u) {
            idx_19 = k1_18 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_10 = __as_u32(a2[10]);
            idx_19 = 4294967295;
            if (k2_10 != 0) {
                idx_19 = k2_10 & 63;
            }
        }
        o[10] = idx_19;
        unsigned int k1_20 = __as_u32(a[11]);
        unsigned int idx_21 = 4294967295;
        if (k1_20 < 4278190080u) {
            idx_21 = k1_20 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_11 = __as_u32(a2[11]);
            idx_21 = 4294967295;
            if (k2_11 != 0) {
                idx_21 = k2_11 & 63;
            }
        }
        o[11] = idx_21;
        unsigned int k1_22 = __as_u32(a[12]);
        unsigned int idx_23 = 4294967295;
        if (k1_22 < 4278190080u) {
            idx_23 = k1_22 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_12 = __as_u32(a2[12]);
            idx_23 = 4294967295;
            if (k2_12 != 0) {
                idx_23 = k2_12 & 63;
            }
        }
        o[12] = idx_23;
        unsigned int k1_24 = __as_u32(a[13]);
        unsigned int idx_25 = 4294967295;
        if (k1_24 < 4278190080u) {
            idx_25 = k1_24 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_13 = __as_u32(a2[13]);
            idx_25 = 4294967295;
            if (k2_13 != 0) {
                idx_25 = k2_13 & 63;
            }
        }
        o[13] = idx_25;
        unsigned int k1_26 = __as_u32(a[14]);
        unsigned int idx_27 = 4294967295;
        if (k1_26 < 4278190080u) {
            idx_27 = k1_26 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_14 = __as_u32(a2[14]);
            idx_27 = 4294967295;
            if (k2_14 != 0) {
                idx_27 = k2_14 & 63;
            }
        }
        o[14] = idx_27;
        unsigned int k1_28 = __as_u32(a[15]);
        unsigned int idx_29 = 4294967295;
        if (k1_28 < 4278190080u) {
            idx_29 = k1_28 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_15 = __as_u32(a2[15]);
            idx_29 = 4294967295;
            if (k2_15 != 0) {
                idx_29 = k2_15 & 63;
            }
        }
        o[15] = idx_29;
        unsigned int _max_0 = ((o[0]) > (o[13]) ? (o[0]) : (o[13]));
        unsigned int hi_30_2 = _max_0;
        unsigned int _min_232 = ((o[0]) < (o[13]) ? (o[0]) : (o[13]));
        unsigned int lo_31_2 = _min_232;
        o[0] = lo_31_2;
        o[13] = hi_30_2;
        unsigned int _max_1 = ((o[1]) > (o[12]) ? (o[1]) : (o[12]));
        unsigned int hi_32_2 = _max_1;
        unsigned int _min_233 = ((o[1]) < (o[12]) ? (o[1]) : (o[12]));
        unsigned int lo_33_2 = _min_233;
        o[1] = lo_33_2;
        o[12] = hi_32_2;
        unsigned int _max_2 = ((o[2]) > (o[15]) ? (o[2]) : (o[15]));
        unsigned int hi_34_2 = _max_2;
        unsigned int _min_234 = ((o[2]) < (o[15]) ? (o[2]) : (o[15]));
        unsigned int lo_35_2 = _min_234;
        o[2] = lo_35_2;
        o[15] = hi_34_2;
        unsigned int _max_3 = ((o[3]) > (o[14]) ? (o[3]) : (o[14]));
        unsigned int hi_36_2 = _max_3;
        unsigned int _min_235 = ((o[3]) < (o[14]) ? (o[3]) : (o[14]));
        unsigned int lo_37_2 = _min_235;
        o[3] = lo_37_2;
        o[14] = hi_36_2;
        unsigned int _max_4 = ((o[4]) > (o[8]) ? (o[4]) : (o[8]));
        unsigned int hi_38_2 = _max_4;
        unsigned int _min_236 = ((o[4]) < (o[8]) ? (o[4]) : (o[8]));
        unsigned int lo_39_2 = _min_236;
        o[4] = lo_39_2;
        o[8] = hi_38_2;
        unsigned int _max_5 = ((o[5]) > (o[6]) ? (o[5]) : (o[6]));
        unsigned int hi_40_2 = _max_5;
        unsigned int _min_237 = ((o[5]) < (o[6]) ? (o[5]) : (o[6]));
        unsigned int lo_41_2 = _min_237;
        o[5] = lo_41_2;
        o[6] = hi_40_2;
        unsigned int _max_6 = ((o[7]) > (o[11]) ? (o[7]) : (o[11]));
        unsigned int hi_42_2 = _max_6;
        unsigned int _min_238 = ((o[7]) < (o[11]) ? (o[7]) : (o[11]));
        unsigned int lo_43_2 = _min_238;
        o[7] = lo_43_2;
        o[11] = hi_42_2;
        unsigned int _max_7 = ((o[9]) > (o[10]) ? (o[9]) : (o[10]));
        unsigned int hi_44_2 = _max_7;
        unsigned int _min_239 = ((o[9]) < (o[10]) ? (o[9]) : (o[10]));
        unsigned int lo_45_2 = _min_239;
        o[9] = lo_45_2;
        o[10] = hi_44_2;
        unsigned int _max_8 = ((o[0]) > (o[5]) ? (o[0]) : (o[5]));
        unsigned int hi_46_2 = _max_8;
        unsigned int _min_240 = ((o[0]) < (o[5]) ? (o[0]) : (o[5]));
        unsigned int lo_47_2 = _min_240;
        o[0] = lo_47_2;
        o[5] = hi_46_2;
        unsigned int _max_9 = ((o[1]) > (o[7]) ? (o[1]) : (o[7]));
        unsigned int hi_48_2 = _max_9;
        unsigned int _min_241 = ((o[1]) < (o[7]) ? (o[1]) : (o[7]));
        unsigned int lo_49_2 = _min_241;
        o[1] = lo_49_2;
        o[7] = hi_48_2;
        unsigned int _max_10 = ((o[2]) > (o[9]) ? (o[2]) : (o[9]));
        unsigned int hi_50_2 = _max_10;
        unsigned int _min_242 = ((o[2]) < (o[9]) ? (o[2]) : (o[9]));
        unsigned int lo_51_2 = _min_242;
        o[2] = lo_51_2;
        o[9] = hi_50_2;
        unsigned int _max_11 = ((o[3]) > (o[4]) ? (o[3]) : (o[4]));
        unsigned int hi_52_2 = _max_11;
        unsigned int _min_243 = ((o[3]) < (o[4]) ? (o[3]) : (o[4]));
        unsigned int lo_53_2 = _min_243;
        o[3] = lo_53_2;
        o[4] = hi_52_2;
        unsigned int _max_12 = ((o[6]) > (o[13]) ? (o[6]) : (o[13]));
        unsigned int hi_54_2 = _max_12;
        unsigned int _min_244 = ((o[6]) < (o[13]) ? (o[6]) : (o[13]));
        unsigned int lo_55_2 = _min_244;
        o[6] = lo_55_2;
        o[13] = hi_54_2;
        unsigned int _max_13 = ((o[8]) > (o[14]) ? (o[8]) : (o[14]));
        unsigned int hi_56_2 = _max_13;
        unsigned int _min_245 = ((o[8]) < (o[14]) ? (o[8]) : (o[14]));
        unsigned int lo_57_2 = _min_245;
        o[8] = lo_57_2;
        o[14] = hi_56_2;
        unsigned int _max_14 = ((o[10]) > (o[15]) ? (o[10]) : (o[15]));
        unsigned int hi_58_2 = _max_14;
        unsigned int _min_246 = ((o[10]) < (o[15]) ? (o[10]) : (o[15]));
        unsigned int lo_59_2 = _min_246;
        o[10] = lo_59_2;
        o[15] = hi_58_2;
        unsigned int _max_15 = ((o[11]) > (o[12]) ? (o[11]) : (o[12]));
        unsigned int hi_60_2 = _max_15;
        unsigned int _min_247 = ((o[11]) < (o[12]) ? (o[11]) : (o[12]));
        unsigned int lo_61_2 = _min_247;
        o[11] = lo_61_2;
        o[12] = hi_60_2;
        unsigned int _max_16 = ((o[0]) > (o[1]) ? (o[0]) : (o[1]));
        unsigned int hi_62_2 = _max_16;
        unsigned int _min_248 = ((o[0]) < (o[1]) ? (o[0]) : (o[1]));
        unsigned int lo_63_2 = _min_248;
        o[0] = lo_63_2;
        o[1] = hi_62_2;
        unsigned int _max_17 = ((o[2]) > (o[3]) ? (o[2]) : (o[3]));
        unsigned int hi_64_2 = _max_17;
        unsigned int _min_249 = ((o[2]) < (o[3]) ? (o[2]) : (o[3]));
        unsigned int lo_65_2 = _min_249;
        o[2] = lo_65_2;
        o[3] = hi_64_2;
        unsigned int _max_18 = ((o[4]) > (o[5]) ? (o[4]) : (o[5]));
        unsigned int hi_66_2 = _max_18;
        unsigned int _min_250 = ((o[4]) < (o[5]) ? (o[4]) : (o[5]));
        unsigned int lo_67_2 = _min_250;
        o[4] = lo_67_2;
        o[5] = hi_66_2;
        unsigned int _max_19 = ((o[6]) > (o[8]) ? (o[6]) : (o[8]));
        unsigned int hi_68_2 = _max_19;
        unsigned int _min_251 = ((o[6]) < (o[8]) ? (o[6]) : (o[8]));
        unsigned int lo_69_2 = _min_251;
        o[6] = lo_69_2;
        o[8] = hi_68_2;
        unsigned int _max_20 = ((o[7]) > (o[9]) ? (o[7]) : (o[9]));
        unsigned int hi_70_2 = _max_20;
        unsigned int _min_252 = ((o[7]) < (o[9]) ? (o[7]) : (o[9]));
        unsigned int lo_71_2 = _min_252;
        o[7] = lo_71_2;
        o[9] = hi_70_2;
        unsigned int _max_21 = ((o[10]) > (o[11]) ? (o[10]) : (o[11]));
        unsigned int hi_72_2 = _max_21;
        unsigned int _min_253 = ((o[10]) < (o[11]) ? (o[10]) : (o[11]));
        unsigned int lo_73_2 = _min_253;
        o[10] = lo_73_2;
        o[11] = hi_72_2;
        unsigned int _max_22 = ((o[12]) > (o[13]) ? (o[12]) : (o[13]));
        unsigned int hi_74_2 = _max_22;
        unsigned int _min_254 = ((o[12]) < (o[13]) ? (o[12]) : (o[13]));
        unsigned int lo_75_2 = _min_254;
        o[12] = lo_75_2;
        o[13] = hi_74_2;
        unsigned int _max_23 = ((o[14]) > (o[15]) ? (o[14]) : (o[15]));
        unsigned int hi_76_2 = _max_23;
        unsigned int _min_255 = ((o[14]) < (o[15]) ? (o[14]) : (o[15]));
        unsigned int lo_77_2 = _min_255;
        o[14] = lo_77_2;
        o[15] = hi_76_2;
        unsigned int _max_24 = ((o[0]) > (o[2]) ? (o[0]) : (o[2]));
        unsigned int hi_78_2 = _max_24;
        unsigned int _min_256 = ((o[0]) < (o[2]) ? (o[0]) : (o[2]));
        unsigned int lo_79_2 = _min_256;
        o[0] = lo_79_2;
        o[2] = hi_78_2;
        unsigned int _max_25 = ((o[1]) > (o[3]) ? (o[1]) : (o[3]));
        unsigned int hi_80_1 = _max_25;
        unsigned int _min_257 = ((o[1]) < (o[3]) ? (o[1]) : (o[3]));
        unsigned int lo_81_1 = _min_257;
        o[1] = lo_81_1;
        o[3] = hi_80_1;
        unsigned int _max_26 = ((o[4]) > (o[10]) ? (o[4]) : (o[10]));
        unsigned int hi_82_1 = _max_26;
        unsigned int _min_258 = ((o[4]) < (o[10]) ? (o[4]) : (o[10]));
        unsigned int lo_83_1 = _min_258;
        o[4] = lo_83_1;
        o[10] = hi_82_1;
        unsigned int _max_27 = ((o[5]) > (o[11]) ? (o[5]) : (o[11]));
        unsigned int hi_84_1 = _max_27;
        unsigned int _min_259 = ((o[5]) < (o[11]) ? (o[5]) : (o[11]));
        unsigned int lo_85_1 = _min_259;
        o[5] = lo_85_1;
        o[11] = hi_84_1;
        unsigned int _max_28 = ((o[6]) > (o[7]) ? (o[6]) : (o[7]));
        unsigned int hi_86_1 = _max_28;
        unsigned int _min_260 = ((o[6]) < (o[7]) ? (o[6]) : (o[7]));
        unsigned int lo_87_1 = _min_260;
        o[6] = lo_87_1;
        o[7] = hi_86_1;
        unsigned int _max_29 = ((o[8]) > (o[9]) ? (o[8]) : (o[9]));
        unsigned int hi_88_1 = _max_29;
        unsigned int _min_261 = ((o[8]) < (o[9]) ? (o[8]) : (o[9]));
        unsigned int lo_89_1 = _min_261;
        o[8] = lo_89_1;
        o[9] = hi_88_1;
        unsigned int _max_30 = ((o[12]) > (o[14]) ? (o[12]) : (o[14]));
        unsigned int hi_90_1 = _max_30;
        unsigned int _min_262 = ((o[12]) < (o[14]) ? (o[12]) : (o[14]));
        unsigned int lo_91_1 = _min_262;
        o[12] = lo_91_1;
        o[14] = hi_90_1;
        unsigned int _max_31 = ((o[13]) > (o[15]) ? (o[13]) : (o[15]));
        unsigned int hi_92_1 = _max_31;
        unsigned int _min_263 = ((o[13]) < (o[15]) ? (o[13]) : (o[15]));
        unsigned int lo_93_1 = _min_263;
        o[13] = lo_93_1;
        o[15] = hi_92_1;
        unsigned int _max_32 = ((o[1]) > (o[2]) ? (o[1]) : (o[2]));
        unsigned int hi_94_1 = _max_32;
        unsigned int _min_264 = ((o[1]) < (o[2]) ? (o[1]) : (o[2]));
        unsigned int lo_95_1 = _min_264;
        o[1] = lo_95_1;
        o[2] = hi_94_1;
        unsigned int _max_33 = ((o[3]) > (o[12]) ? (o[3]) : (o[12]));
        unsigned int hi_96 = _max_33;
        unsigned int _min_265 = ((o[3]) < (o[12]) ? (o[3]) : (o[12]));
        unsigned int lo_97 = _min_265;
        o[3] = lo_97;
        o[12] = hi_96;
        unsigned int _max_34 = ((o[4]) > (o[6]) ? (o[4]) : (o[6]));
        unsigned int hi_98 = _max_34;
        unsigned int _min_266 = ((o[4]) < (o[6]) ? (o[4]) : (o[6]));
        unsigned int lo_99 = _min_266;
        o[4] = lo_99;
        o[6] = hi_98;
        unsigned int _max_35 = ((o[5]) > (o[7]) ? (o[5]) : (o[7]));
        unsigned int hi_100 = _max_35;
        unsigned int _min_267 = ((o[5]) < (o[7]) ? (o[5]) : (o[7]));
        unsigned int lo_101 = _min_267;
        o[5] = lo_101;
        o[7] = hi_100;
        unsigned int _max_36 = ((o[8]) > (o[10]) ? (o[8]) : (o[10]));
        unsigned int hi_102 = _max_36;
        unsigned int _min_268 = ((o[8]) < (o[10]) ? (o[8]) : (o[10]));
        unsigned int lo_103 = _min_268;
        o[8] = lo_103;
        o[10] = hi_102;
        unsigned int _max_37 = ((o[9]) > (o[11]) ? (o[9]) : (o[11]));
        unsigned int hi_104_1 = _max_37;
        unsigned int _min_269 = ((o[9]) < (o[11]) ? (o[9]) : (o[11]));
        unsigned int lo_105_1 = _min_269;
        o[9] = lo_105_1;
        o[11] = hi_104_1;
        unsigned int _max_38 = ((o[13]) > (o[14]) ? (o[13]) : (o[14]));
        unsigned int hi_106_1 = _max_38;
        unsigned int _min_270 = ((o[13]) < (o[14]) ? (o[13]) : (o[14]));
        unsigned int lo_107_1 = _min_270;
        o[13] = lo_107_1;
        o[14] = hi_106_1;
        unsigned int _max_39 = ((o[1]) > (o[4]) ? (o[1]) : (o[4]));
        unsigned int hi_108_1 = _max_39;
        unsigned int _min_271 = ((o[1]) < (o[4]) ? (o[1]) : (o[4]));
        unsigned int lo_109_1 = _min_271;
        o[1] = lo_109_1;
        o[4] = hi_108_1;
        unsigned int _max_40 = ((o[2]) > (o[6]) ? (o[2]) : (o[6]));
        unsigned int hi_110_1 = _max_40;
        unsigned int _min_272 = ((o[2]) < (o[6]) ? (o[2]) : (o[6]));
        unsigned int lo_111_1 = _min_272;
        o[2] = lo_111_1;
        o[6] = hi_110_1;
        unsigned int _max_41 = ((o[5]) > (o[8]) ? (o[5]) : (o[8]));
        unsigned int hi_112_1 = _max_41;
        unsigned int _min_273 = ((o[5]) < (o[8]) ? (o[5]) : (o[8]));
        unsigned int lo_113_1 = _min_273;
        o[5] = lo_113_1;
        o[8] = hi_112_1;
        unsigned int _max_42 = ((o[7]) > (o[10]) ? (o[7]) : (o[10]));
        unsigned int hi_114_1 = _max_42;
        unsigned int _min_274 = ((o[7]) < (o[10]) ? (o[7]) : (o[10]));
        unsigned int lo_115_1 = _min_274;
        o[7] = lo_115_1;
        o[10] = hi_114_1;
        unsigned int _max_43 = ((o[9]) > (o[13]) ? (o[9]) : (o[13]));
        unsigned int hi_116_1 = _max_43;
        unsigned int _min_275 = ((o[9]) < (o[13]) ? (o[9]) : (o[13]));
        unsigned int lo_117_1 = _min_275;
        o[9] = lo_117_1;
        o[13] = hi_116_1;
        unsigned int _max_44 = ((o[11]) > (o[14]) ? (o[11]) : (o[14]));
        unsigned int hi_118_1 = _max_44;
        unsigned int _min_276 = ((o[11]) < (o[14]) ? (o[11]) : (o[14]));
        unsigned int lo_119_1 = _min_276;
        o[11] = lo_119_1;
        o[14] = hi_118_1;
        unsigned int _max_45 = ((o[2]) > (o[4]) ? (o[2]) : (o[4]));
        unsigned int hi_120_1 = _max_45;
        unsigned int _min_277 = ((o[2]) < (o[4]) ? (o[2]) : (o[4]));
        unsigned int lo_121_1 = _min_277;
        o[2] = lo_121_1;
        o[4] = hi_120_1;
        unsigned int _max_46 = ((o[3]) > (o[6]) ? (o[3]) : (o[6]));
        unsigned int hi_122_1 = _max_46;
        unsigned int _min_278 = ((o[3]) < (o[6]) ? (o[3]) : (o[6]));
        unsigned int lo_123_1 = _min_278;
        o[3] = lo_123_1;
        o[6] = hi_122_1;
        unsigned int _max_47 = ((o[9]) > (o[12]) ? (o[9]) : (o[12]));
        unsigned int hi_124_1 = _max_47;
        unsigned int _min_279 = ((o[9]) < (o[12]) ? (o[9]) : (o[12]));
        unsigned int lo_125_1 = _min_279;
        o[9] = lo_125_1;
        o[12] = hi_124_1;
        unsigned int _max_48 = ((o[11]) > (o[13]) ? (o[11]) : (o[13]));
        unsigned int hi_126_1 = _max_48;
        unsigned int _min_280 = ((o[11]) < (o[13]) ? (o[11]) : (o[13]));
        unsigned int lo_127_1 = _min_280;
        o[11] = lo_127_1;
        o[13] = hi_126_1;
        unsigned int _max_49 = ((o[3]) > (o[5]) ? (o[3]) : (o[5]));
        unsigned int hi_128_1 = _max_49;
        unsigned int _min_281 = ((o[3]) < (o[5]) ? (o[3]) : (o[5]));
        unsigned int lo_129_1 = _min_281;
        o[3] = lo_129_1;
        o[5] = hi_128_1;
        unsigned int _max_50 = ((o[6]) > (o[8]) ? (o[6]) : (o[8]));
        unsigned int hi_130_1 = _max_50;
        unsigned int _min_282 = ((o[6]) < (o[8]) ? (o[6]) : (o[8]));
        unsigned int lo_131_1 = _min_282;
        o[6] = lo_131_1;
        o[8] = hi_130_1;
        unsigned int _max_51 = ((o[7]) > (o[9]) ? (o[7]) : (o[9]));
        unsigned int hi_132_1 = _max_51;
        unsigned int _min_283 = ((o[7]) < (o[9]) ? (o[7]) : (o[9]));
        unsigned int lo_133_1 = _min_283;
        o[7] = lo_133_1;
        o[9] = hi_132_1;
        unsigned int _max_52 = ((o[10]) > (o[12]) ? (o[10]) : (o[12]));
        unsigned int hi_134_1 = _max_52;
        unsigned int _min_284 = ((o[10]) < (o[12]) ? (o[10]) : (o[12]));
        unsigned int lo_135_1 = _min_284;
        o[10] = lo_135_1;
        o[12] = hi_134_1;
        unsigned int _max_53 = ((o[3]) > (o[4]) ? (o[3]) : (o[4]));
        unsigned int hi_136_1 = _max_53;
        unsigned int _min_285 = ((o[3]) < (o[4]) ? (o[3]) : (o[4]));
        unsigned int lo_137_1 = _min_285;
        o[3] = lo_137_1;
        o[4] = hi_136_1;
        unsigned int _max_54 = ((o[5]) > (o[6]) ? (o[5]) : (o[6]));
        unsigned int hi_138_1 = _max_54;
        unsigned int _min_286 = ((o[5]) < (o[6]) ? (o[5]) : (o[6]));
        unsigned int lo_139_1 = _min_286;
        o[5] = lo_139_1;
        o[6] = hi_138_1;
        unsigned int _max_55 = ((o[7]) > (o[8]) ? (o[7]) : (o[8]));
        unsigned int hi_140_1 = _max_55;
        unsigned int _min_287 = ((o[7]) < (o[8]) ? (o[7]) : (o[8]));
        unsigned int lo_141_1 = _min_287;
        o[7] = lo_141_1;
        o[8] = hi_140_1;
        unsigned int _max_56 = ((o[9]) > (o[10]) ? (o[9]) : (o[10]));
        unsigned int hi_142_1 = _max_56;
        unsigned int _min_288 = ((o[9]) < (o[10]) ? (o[9]) : (o[10]));
        unsigned int lo_143_1 = _min_288;
        o[9] = lo_143_1;
        o[10] = hi_142_1;
        unsigned int _max_57 = ((o[11]) > (o[12]) ? (o[11]) : (o[12]));
        unsigned int hi_144_1 = _max_57;
        unsigned int _min_289 = ((o[11]) < (o[12]) ? (o[11]) : (o[12]));
        unsigned int lo_145_1 = _min_289;
        o[11] = lo_145_1;
        o[12] = hi_144_1;
        unsigned int _max_58 = ((o[6]) > (o[7]) ? (o[6]) : (o[7]));
        unsigned int hi_146_1 = _max_58;
        unsigned int _min_290 = ((o[6]) < (o[7]) ? (o[6]) : (o[7]));
        unsigned int lo_147_1 = _min_290;
        o[6] = lo_147_1;
        o[7] = hi_146_1;
        unsigned int _max_59 = ((o[8]) > (o[9]) ? (o[8]) : (o[9]));
        unsigned int hi_148_1 = _max_59;
        unsigned int _min_291 = ((o[8]) < (o[9]) ? (o[8]) : (o[9]));
        unsigned int lo_149_1 = _min_291;
        o[8] = lo_149_1;
        o[9] = hi_148_1;
        long long obase = ((long long)col * (long long)num_heads + (long long)head) * 16;
        {
            int4 _iv4 = make_int4(o[0 + 0], o[0 + 1], o[0 + 2], o[0 + 3]);
            *reinterpret_cast<int4*>(out + obase) = _iv4;
        }
        {
            int4 _iv4 = make_int4(o[4 + 0], o[4 + 1], o[4 + 2], o[4 + 3]);
            *reinterpret_cast<int4*>(out + obase + 4) = _iv4;
        }
        {
            int4 _iv4 = make_int4(o[8 + 0], o[8 + 1], o[8 + 2], o[8 + 3]);
            *reinterpret_cast<int4*>(out + obase + 8) = _iv4;
        }
        {
            int4 _iv4 = make_int4(o[12 + 0], o[12 + 1], o[12 + 2], o[12 + 3]);
            *reinterpret_cast<int4*>(out + obase + 12) = _iv4;
        }
    }
}

} // extern "C"
