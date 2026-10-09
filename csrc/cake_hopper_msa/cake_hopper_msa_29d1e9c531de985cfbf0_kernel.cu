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
#define SMEM_QCOL_STAGE_BYTES 8
#define SMEM_QCOL_STRIDE 8
#define SMEM_FLAGW_OFF 8
#define SMEM_FLAGW_STAGE_BYTES 16
#define SMEM_FLAGW_STRIDE 16
#define SMEM_CCNT_OFF 19480
#define SMEM_CCNT_STAGE_BYTES 8
#define SMEM_CCNT_STRIDE 8
#define SMEM_LIMC_OFF 19488
#define SMEM_LIMC_STAGE_BYTES 8
#define SMEM_LIMC_STRIDE 8
#define SMEM_CBUF_OFF 19496
#define SMEM_CBUF_STAGE_BYTES 128
#define SMEM_CBUF_STRIDE 128
#define SMEM_PUB_OFF 24
#define SMEM_PUB_STAGE_BYTES 17408
#define SMEM_PUB_STRIDE 17408
#define SMEM_Q2_OFF 17432
#define SMEM_Q2_STAGE_BYTES 2048
#define SMEM_Q2_STRIDE 2048
#define SMEM_TOTAL 19712
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
kernel_cake_hopper_msa_29d1e9c531de985cfbf0(unsigned int* __restrict__ S, int* __restrict__ nvp, int* __restrict__ out, int tiles, int total_q, int num_heads, int num_chunks, int fb, int fe)
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
    unsigned int* flagw = reinterpret_cast<unsigned int*>(smem_raw + 8);
    const int flagw_addr = smem + 8;
    unsigned int* ccnt = reinterpret_cast<unsigned int*>(smem_raw + 19480);
    const int ccnt_addr = smem + 19480;
    int* limc = reinterpret_cast<int*>(smem_raw + 19488);
    const int limc_addr = smem + 19488;
    unsigned int* cbuf = reinterpret_cast<unsigned int*>(smem_raw + 19496);
    const int cbuf_addr = smem + 19496;
    float* pub = reinterpret_cast<float*>(smem_raw + 24);
    const int pub_addr = smem + 24;
    float* q2 = reinterpret_cast<float*>(smem_raw + 17432);
    const int q2_addr = smem + 17432;

    // === Task calls (dependency order) ===
    int tid_1 = threadIdx.x;
    int w = tid_1 / 2;
    int c = tid_1 - w * 2;
    int whi = w / 16;
    int col = blockIdx.x * 2 + c;
    int head = blockIdx.y;
    if (tid_1 == 0) {
        flagw[0] = 0;
    }
    if (tid_1 == 0) {
        flagw[1] = 0;
    }
    if (tid_1 < 2) {
        ccnt[tid_1] = 0;
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
    unsigned int key = __as_u32(sc_3) & 4294965248u | (unsigned int)t0_2;
    kb[0] = __uint_as_float(key);
    float sc_4 = __uint_as_float(cb[1]);
    float _fmax_1 = fmaxf(sc_4, -1.7014118346046923e+38f);
    sc_4 = _fmax_1;
    float _min_1 = fminf(sc_4, 1.7014118346046923e+38f);
    sc_4 = _min_1;
    sc_4 = sc_4;
    float sc_5 = sc_4;
    unsigned int key_6 = __as_u32(sc_5) & 4294965248u | (unsigned int)(t0_2 + 1);
    kb[1] = __uint_as_float(key_6);
    float sc_7 = __uint_as_float(cb[2]);
    float _fmax_2 = fmaxf(sc_7, -1.7014118346046923e+38f);
    sc_7 = _fmax_2;
    float _min_2 = fminf(sc_7, 1.7014118346046923e+38f);
    sc_7 = _min_2;
    sc_7 = sc_7;
    float sc_8 = sc_7;
    unsigned int key_9 = __as_u32(sc_8) & 4294965248u | (unsigned int)(t0_2 + 2);
    kb[2] = __uint_as_float(key_9);
    float sc_10 = __uint_as_float(cb[3]);
    float _fmax_3 = fmaxf(sc_10, -1.7014118346046923e+38f);
    sc_10 = _fmax_3;
    float _min_3 = fminf(sc_10, 1.7014118346046923e+38f);
    sc_10 = _min_3;
    sc_10 = sc_10;
    float sc_11 = sc_10;
    unsigned int key_12 = __as_u32(sc_11) & 4294965248u | (unsigned int)(t0_2 + 3);
    kb[3] = __uint_as_float(key_12);
    float sc_13 = __uint_as_float(cb[4]);
    float _fmax_4 = fmaxf(sc_13, -1.7014118346046923e+38f);
    sc_13 = _fmax_4;
    float _min_4 = fminf(sc_13, 1.7014118346046923e+38f);
    sc_13 = _min_4;
    sc_13 = sc_13;
    float sc_14 = sc_13;
    unsigned int key_15 = __as_u32(sc_14) & 4294965248u | (unsigned int)(t0_2 + 4);
    kb[4] = __uint_as_float(key_15);
    float sc_16 = __uint_as_float(cb[5]);
    float _fmax_5 = fmaxf(sc_16, -1.7014118346046923e+38f);
    sc_16 = _fmax_5;
    float _min_5 = fminf(sc_16, 1.7014118346046923e+38f);
    sc_16 = _min_5;
    sc_16 = sc_16;
    float sc_17 = sc_16;
    unsigned int key_18 = __as_u32(sc_17) & 4294965248u | (unsigned int)(t0_2 + 5);
    kb[5] = __uint_as_float(key_18);
    float sc_19 = __uint_as_float(cb[6]);
    float _fmax_6 = fmaxf(sc_19, -1.7014118346046923e+38f);
    sc_19 = _fmax_6;
    float _min_6 = fminf(sc_19, 1.7014118346046923e+38f);
    sc_19 = _min_6;
    sc_19 = sc_19;
    float sc_20 = sc_19;
    unsigned int key_21 = __as_u32(sc_20) & 4294965248u | (unsigned int)(t0_2 + 6);
    kb[6] = __uint_as_float(key_21);
    float sc_22 = __uint_as_float(cb[7]);
    float _fmax_7 = fmaxf(sc_22, -1.7014118346046923e+38f);
    sc_22 = _fmax_7;
    float _min_7 = fminf(sc_22, 1.7014118346046923e+38f);
    sc_22 = _min_7;
    sc_22 = sc_22;
    float sc_23 = sc_22;
    unsigned int key_24 = __as_u32(sc_23) & 4294965248u | (unsigned int)(t0_2 + 7);
    kb[7] = __uint_as_float(key_24);
    float sc_25 = __uint_as_float(cb[8]);
    float _fmax_8 = fmaxf(sc_25, -1.7014118346046923e+38f);
    sc_25 = _fmax_8;
    float _min_8 = fminf(sc_25, 1.7014118346046923e+38f);
    sc_25 = _min_8;
    sc_25 = sc_25;
    float sc_26 = sc_25;
    unsigned int key_27 = __as_u32(sc_26) & 4294965248u | (unsigned int)(t0_2 + 8);
    kb[8] = __uint_as_float(key_27);
    float sc_28 = __uint_as_float(cb[9]);
    float _fmax_9 = fmaxf(sc_28, -1.7014118346046923e+38f);
    sc_28 = _fmax_9;
    float _min_9 = fminf(sc_28, 1.7014118346046923e+38f);
    sc_28 = _min_9;
    sc_28 = sc_28;
    float sc_29 = sc_28;
    unsigned int key_30 = __as_u32(sc_29) & 4294965248u | (unsigned int)(t0_2 + 9);
    kb[9] = __uint_as_float(key_30);
    float sc_31 = __uint_as_float(cb[10]);
    float _fmax_10 = fmaxf(sc_31, -1.7014118346046923e+38f);
    sc_31 = _fmax_10;
    float _min_10 = fminf(sc_31, 1.7014118346046923e+38f);
    sc_31 = _min_10;
    sc_31 = sc_31;
    float sc_32 = sc_31;
    unsigned int key_33 = __as_u32(sc_32) & 4294965248u | (unsigned int)(t0_2 + 10);
    kb[10] = __uint_as_float(key_33);
    float sc_34 = __uint_as_float(cb[11]);
    float _fmax_11 = fmaxf(sc_34, -1.7014118346046923e+38f);
    sc_34 = _fmax_11;
    float _min_11 = fminf(sc_34, 1.7014118346046923e+38f);
    sc_34 = _min_11;
    sc_34 = sc_34;
    float sc_35 = sc_34;
    unsigned int key_36 = __as_u32(sc_35) & 4294965248u | (unsigned int)(t0_2 + 11);
    kb[11] = __uint_as_float(key_36);
    float sc_37 = __uint_as_float(cb[12]);
    float _fmax_12 = fmaxf(sc_37, -1.7014118346046923e+38f);
    sc_37 = _fmax_12;
    float _min_12 = fminf(sc_37, 1.7014118346046923e+38f);
    sc_37 = _min_12;
    sc_37 = sc_37;
    float sc_38 = sc_37;
    unsigned int key_39 = __as_u32(sc_38) & 4294965248u | (unsigned int)(t0_2 + 12);
    kb[12] = __uint_as_float(key_39);
    float sc_40 = __uint_as_float(cb[13]);
    float _fmax_13 = fmaxf(sc_40, -1.7014118346046923e+38f);
    sc_40 = _fmax_13;
    float _min_13 = fminf(sc_40, 1.7014118346046923e+38f);
    sc_40 = _min_13;
    sc_40 = sc_40;
    float sc_41 = sc_40;
    unsigned int key_42 = __as_u32(sc_41) & 4294965248u | (unsigned int)(t0_2 + 13);
    kb[13] = __uint_as_float(key_42);
    float sc_43 = __uint_as_float(cb[14]);
    float _fmax_14 = fmaxf(sc_43, -1.7014118346046923e+38f);
    sc_43 = _fmax_14;
    float _min_14 = fminf(sc_43, 1.7014118346046923e+38f);
    sc_43 = _min_14;
    sc_43 = sc_43;
    float sc_44 = sc_43;
    unsigned int key_45 = __as_u32(sc_44) & 4294965248u | (unsigned int)(t0_2 + 14);
    kb[14] = __uint_as_float(key_45);
    float sc_46 = __uint_as_float(cb[15]);
    float _fmax_15 = fmaxf(sc_46, -1.7014118346046923e+38f);
    sc_46 = _fmax_15;
    float _min_15 = fminf(sc_46, 1.7014118346046923e+38f);
    sc_46 = _min_15;
    sc_46 = sc_46;
    float sc_47 = sc_46;
    unsigned int key_48 = __as_u32(sc_47) & 4294965248u | (unsigned int)(t0_2 + 15);
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
    int ln = tid_1 & 15;
    int g = tid_1 >> 4;
    int cg = g & 1;
    int sg = g >> 1;
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
    int s0 = (sg * 16 * 2 + cg) * 17;
    int s1 = ((sg * 16 + 8) * 2 + cg) * 17;
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
    float hi_167 = _fmax_78;
    float _min_77 = fminf(cur, pv);
    float lo_168 = _min_77;
    cur = ((up[0] != 0) ? hi_167 : lo_168);
    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, cur, 4);
    float pv_169 = _shfl_xor_1;
    float _fmax_79 = fmaxf(cur, pv_169);
    float hi_170 = _fmax_79;
    float _min_78 = fminf(cur, pv_169);
    float lo_171 = _min_78;
    cur = ((up[1] != 0) ? hi_170 : lo_171);
    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, cur, 2);
    float pv_172 = _shfl_xor_2;
    float _fmax_80 = fmaxf(cur, pv_172);
    float hi_173 = _fmax_80;
    float _min_79 = fminf(cur, pv_172);
    float lo_174 = _min_79;
    cur = ((up[2] != 0) ? hi_173 : lo_174);
    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, cur, 1);
    float pv_175 = _shfl_xor_3;
    float _fmax_81 = fmaxf(cur, pv_175);
    float hi_176 = _fmax_81;
    float _min_80 = fminf(cur, pv_175);
    float lo_177 = _min_80;
    cur = ((up[3] != 0) ? hi_176 : lo_177);
    V[0] = cur;
    int s0_178 = ((sg * 16 + 1) * 2 + cg) * 17;
    int s1_179 = ((sg * 16 + 1 + 8) * 2 + cg) * 17;
    float x0_180 = pub[s0_178 + ln];
    float y0_181 = pub[s1_179 + lnr];
    float _min_81 = fminf(x0_180, y0_181);
    float lo0_182 = _min_81;
    float _fmax_82 = fmaxf(r, lo0_182);
    r = _fmax_82;
    float _fmax_83 = fmaxf(x0_180, y0_181);
    float hi0_183 = _fmax_83;
    float cur_184 = hi0_183;
    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, cur_184, 8);
    float pv_185 = _shfl_xor_4;
    float _fmax_84 = fmaxf(cur_184, pv_185);
    float hi_186 = _fmax_84;
    float _min_82 = fminf(cur_184, pv_185);
    float lo_187 = _min_82;
    cur_184 = ((up[0] != 0) ? hi_186 : lo_187);
    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, cur_184, 4);
    float pv_188 = _shfl_xor_5;
    float _fmax_85 = fmaxf(cur_184, pv_188);
    float hi_189 = _fmax_85;
    float _min_83 = fminf(cur_184, pv_188);
    float lo_190 = _min_83;
    cur_184 = ((up[1] != 0) ? hi_189 : lo_190);
    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, cur_184, 2);
    float pv_191 = _shfl_xor_6;
    float _fmax_86 = fmaxf(cur_184, pv_191);
    float hi_192 = _fmax_86;
    float _min_84 = fminf(cur_184, pv_191);
    float lo_193 = _min_84;
    cur_184 = ((up[2] != 0) ? hi_192 : lo_193);
    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, cur_184, 1);
    float pv_194 = _shfl_xor_7;
    float _fmax_87 = fmaxf(cur_184, pv_194);
    float hi_195 = _fmax_87;
    float _min_85 = fminf(cur_184, pv_194);
    float lo_196 = _min_85;
    cur_184 = ((up[3] != 0) ? hi_195 : lo_196);
    V[1] = cur_184;
    int s0_197 = ((sg * 16 + 2) * 2 + cg) * 17;
    int s1_198 = ((sg * 16 + 2 + 8) * 2 + cg) * 17;
    float x0_199 = pub[s0_197 + ln];
    float y0_200 = pub[s1_198 + lnr];
    float _min_86 = fminf(x0_199, y0_200);
    float lo0_201 = _min_86;
    float _fmax_88 = fmaxf(r, lo0_201);
    r = _fmax_88;
    float _fmax_89 = fmaxf(x0_199, y0_200);
    float hi0_202 = _fmax_89;
    float cur_203 = hi0_202;
    float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, cur_203, 8);
    float pv_204 = _shfl_xor_8;
    float _fmax_90 = fmaxf(cur_203, pv_204);
    float hi_205 = _fmax_90;
    float _min_87 = fminf(cur_203, pv_204);
    float lo_206 = _min_87;
    cur_203 = ((up[0] != 0) ? hi_205 : lo_206);
    float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, cur_203, 4);
    float pv_207 = _shfl_xor_9;
    float _fmax_91 = fmaxf(cur_203, pv_207);
    float hi_208 = _fmax_91;
    float _min_88 = fminf(cur_203, pv_207);
    float lo_209 = _min_88;
    cur_203 = ((up[1] != 0) ? hi_208 : lo_209);
    float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, cur_203, 2);
    float pv_210 = _shfl_xor_10;
    float _fmax_92 = fmaxf(cur_203, pv_210);
    float hi_211 = _fmax_92;
    float _min_89 = fminf(cur_203, pv_210);
    float lo_212 = _min_89;
    cur_203 = ((up[2] != 0) ? hi_211 : lo_212);
    float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, cur_203, 1);
    float pv_213 = _shfl_xor_11;
    float _fmax_93 = fmaxf(cur_203, pv_213);
    float hi_214 = _fmax_93;
    float _min_90 = fminf(cur_203, pv_213);
    float lo_215 = _min_90;
    cur_203 = ((up[3] != 0) ? hi_214 : lo_215);
    V[2] = cur_203;
    int s0_216 = ((sg * 16 + 3) * 2 + cg) * 17;
    int s1_217 = ((sg * 16 + 3 + 8) * 2 + cg) * 17;
    float x0_218 = pub[s0_216 + ln];
    float y0_219 = pub[s1_217 + lnr];
    float _min_91 = fminf(x0_218, y0_219);
    float lo0_220 = _min_91;
    float _fmax_94 = fmaxf(r, lo0_220);
    r = _fmax_94;
    float _fmax_95 = fmaxf(x0_218, y0_219);
    float hi0_221 = _fmax_95;
    float cur_222 = hi0_221;
    float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, cur_222, 8);
    float pv_223 = _shfl_xor_12;
    float _fmax_96 = fmaxf(cur_222, pv_223);
    float hi_224 = _fmax_96;
    float _min_92 = fminf(cur_222, pv_223);
    float lo_225 = _min_92;
    cur_222 = ((up[0] != 0) ? hi_224 : lo_225);
    float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, cur_222, 4);
    float pv_226 = _shfl_xor_13;
    float _fmax_97 = fmaxf(cur_222, pv_226);
    float hi_227 = _fmax_97;
    float _min_93 = fminf(cur_222, pv_226);
    float lo_228 = _min_93;
    cur_222 = ((up[1] != 0) ? hi_227 : lo_228);
    float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, cur_222, 2);
    float pv_229 = _shfl_xor_14;
    float _fmax_98 = fmaxf(cur_222, pv_229);
    float hi_230 = _fmax_98;
    float _min_94 = fminf(cur_222, pv_229);
    float lo_231 = _min_94;
    cur_222 = ((up[2] != 0) ? hi_230 : lo_231);
    float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, cur_222, 1);
    float pv_232 = _shfl_xor_15;
    float _fmax_99 = fmaxf(cur_222, pv_232);
    float hi_233 = _fmax_99;
    float _min_95 = fminf(cur_222, pv_232);
    float lo_234 = _min_95;
    cur_222 = ((up[3] != 0) ? hi_233 : lo_234);
    V[3] = cur_222;
    int s0_235 = ((sg * 16 + 4) * 2 + cg) * 17;
    int s1_236 = ((sg * 16 + 4 + 8) * 2 + cg) * 17;
    float x0_237 = pub[s0_235 + ln];
    float y0_238 = pub[s1_236 + lnr];
    float _min_96 = fminf(x0_237, y0_238);
    float lo0_239 = _min_96;
    float _fmax_100 = fmaxf(r, lo0_239);
    r = _fmax_100;
    float _fmax_101 = fmaxf(x0_237, y0_238);
    float hi0_240 = _fmax_101;
    float cur_241 = hi0_240;
    float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, cur_241, 8);
    float pv_242 = _shfl_xor_16;
    float _fmax_102 = fmaxf(cur_241, pv_242);
    float hi_243 = _fmax_102;
    float _min_97 = fminf(cur_241, pv_242);
    float lo_244 = _min_97;
    cur_241 = ((up[0] != 0) ? hi_243 : lo_244);
    float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, cur_241, 4);
    float pv_245 = _shfl_xor_17;
    float _fmax_103 = fmaxf(cur_241, pv_245);
    float hi_246 = _fmax_103;
    float _min_98 = fminf(cur_241, pv_245);
    float lo_247 = _min_98;
    cur_241 = ((up[1] != 0) ? hi_246 : lo_247);
    float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, cur_241, 2);
    float pv_248 = _shfl_xor_18;
    float _fmax_104 = fmaxf(cur_241, pv_248);
    float hi_249 = _fmax_104;
    float _min_99 = fminf(cur_241, pv_248);
    float lo_250 = _min_99;
    cur_241 = ((up[2] != 0) ? hi_249 : lo_250);
    float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, cur_241, 1);
    float pv_251 = _shfl_xor_19;
    float _fmax_105 = fmaxf(cur_241, pv_251);
    float hi_252 = _fmax_105;
    float _min_100 = fminf(cur_241, pv_251);
    float lo_253 = _min_100;
    cur_241 = ((up[3] != 0) ? hi_252 : lo_253);
    V[4] = cur_241;
    int s0_254 = ((sg * 16 + 5) * 2 + cg) * 17;
    int s1_255 = ((sg * 16 + 5 + 8) * 2 + cg) * 17;
    float x0_256 = pub[s0_254 + ln];
    float y0_257 = pub[s1_255 + lnr];
    float _min_101 = fminf(x0_256, y0_257);
    float lo0_258 = _min_101;
    float _fmax_106 = fmaxf(r, lo0_258);
    r = _fmax_106;
    float _fmax_107 = fmaxf(x0_256, y0_257);
    float hi0_259 = _fmax_107;
    float cur_260 = hi0_259;
    float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, cur_260, 8);
    float pv_261 = _shfl_xor_20;
    float _fmax_108 = fmaxf(cur_260, pv_261);
    float hi_262 = _fmax_108;
    float _min_102 = fminf(cur_260, pv_261);
    float lo_263 = _min_102;
    cur_260 = ((up[0] != 0) ? hi_262 : lo_263);
    float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, cur_260, 4);
    float pv_264 = _shfl_xor_21;
    float _fmax_109 = fmaxf(cur_260, pv_264);
    float hi_265 = _fmax_109;
    float _min_103 = fminf(cur_260, pv_264);
    float lo_266 = _min_103;
    cur_260 = ((up[1] != 0) ? hi_265 : lo_266);
    float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, cur_260, 2);
    float pv_267 = _shfl_xor_22;
    float _fmax_110 = fmaxf(cur_260, pv_267);
    float hi_268 = _fmax_110;
    float _min_104 = fminf(cur_260, pv_267);
    float lo_269 = _min_104;
    cur_260 = ((up[2] != 0) ? hi_268 : lo_269);
    float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, cur_260, 1);
    float pv_270 = _shfl_xor_23;
    float _fmax_111 = fmaxf(cur_260, pv_270);
    float hi_271 = _fmax_111;
    float _min_105 = fminf(cur_260, pv_270);
    float lo_272 = _min_105;
    cur_260 = ((up[3] != 0) ? hi_271 : lo_272);
    V[5] = cur_260;
    int s0_273 = ((sg * 16 + 6) * 2 + cg) * 17;
    int s1_274 = ((sg * 16 + 6 + 8) * 2 + cg) * 17;
    float x0_275 = pub[s0_273 + ln];
    float y0_276 = pub[s1_274 + lnr];
    float _min_106 = fminf(x0_275, y0_276);
    float lo0_277 = _min_106;
    float _fmax_112 = fmaxf(r, lo0_277);
    r = _fmax_112;
    float _fmax_113 = fmaxf(x0_275, y0_276);
    float hi0_278 = _fmax_113;
    float cur_279 = hi0_278;
    float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, cur_279, 8);
    float pv_280 = _shfl_xor_24;
    float _fmax_114 = fmaxf(cur_279, pv_280);
    float hi_281 = _fmax_114;
    float _min_107 = fminf(cur_279, pv_280);
    float lo_282 = _min_107;
    cur_279 = ((up[0] != 0) ? hi_281 : lo_282);
    float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, cur_279, 4);
    float pv_283 = _shfl_xor_25;
    float _fmax_115 = fmaxf(cur_279, pv_283);
    float hi_284 = _fmax_115;
    float _min_108 = fminf(cur_279, pv_283);
    float lo_285 = _min_108;
    cur_279 = ((up[1] != 0) ? hi_284 : lo_285);
    float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, cur_279, 2);
    float pv_286 = _shfl_xor_26;
    float _fmax_116 = fmaxf(cur_279, pv_286);
    float hi_287 = _fmax_116;
    float _min_109 = fminf(cur_279, pv_286);
    float lo_288 = _min_109;
    cur_279 = ((up[2] != 0) ? hi_287 : lo_288);
    float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, cur_279, 1);
    float pv_289 = _shfl_xor_27;
    float _fmax_117 = fmaxf(cur_279, pv_289);
    float hi_290 = _fmax_117;
    float _min_110 = fminf(cur_279, pv_289);
    float lo_291 = _min_110;
    cur_279 = ((up[3] != 0) ? hi_290 : lo_291);
    V[6] = cur_279;
    int s0_292 = ((sg * 16 + 7) * 2 + cg) * 17;
    int s1_293 = ((sg * 16 + 7 + 8) * 2 + cg) * 17;
    float x0_294 = pub[s0_292 + ln];
    float y0_295 = pub[s1_293 + lnr];
    float _min_111 = fminf(x0_294, y0_295);
    float lo0_296 = _min_111;
    float _fmax_118 = fmaxf(r, lo0_296);
    r = _fmax_118;
    float _fmax_119 = fmaxf(x0_294, y0_295);
    float hi0_297 = _fmax_119;
    float cur_298 = hi0_297;
    float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, cur_298, 8);
    float pv_299 = _shfl_xor_28;
    float _fmax_120 = fmaxf(cur_298, pv_299);
    float hi_300 = _fmax_120;
    float _min_112 = fminf(cur_298, pv_299);
    float lo_301 = _min_112;
    cur_298 = ((up[0] != 0) ? hi_300 : lo_301);
    float _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, cur_298, 4);
    float pv_302 = _shfl_xor_29;
    float _fmax_121 = fmaxf(cur_298, pv_302);
    float hi_303 = _fmax_121;
    float _min_113 = fminf(cur_298, pv_302);
    float lo_304 = _min_113;
    cur_298 = ((up[1] != 0) ? hi_303 : lo_304);
    float _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, cur_298, 2);
    float pv_305 = _shfl_xor_30;
    float _fmax_122 = fmaxf(cur_298, pv_305);
    float hi_306 = _fmax_122;
    float _min_114 = fminf(cur_298, pv_305);
    float lo_307 = _min_114;
    cur_298 = ((up[2] != 0) ? hi_306 : lo_307);
    float _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, cur_298, 1);
    float pv_308 = _shfl_xor_31;
    float _fmax_123 = fmaxf(cur_298, pv_308);
    float hi_309 = _fmax_123;
    float _min_115 = fminf(cur_298, pv_308);
    float lo_310 = _min_115;
    cur_298 = ((up[3] != 0) ? hi_309 : lo_310);
    V[7] = cur_298;
    float rs = pub[((sg * 16 + ln) * 2 + cg) * 17 + 16];
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
    float cur_311 = hi1;
    float _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, cur_311, 8);
    float pv_312 = _shfl_xor_33;
    float _fmax_127 = fmaxf(cur_311, pv_312);
    float hi_313 = _fmax_127;
    float _min_117 = fminf(cur_311, pv_312);
    float lo_314 = _min_117;
    cur_311 = ((up[0] != 0) ? hi_313 : lo_314);
    float _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, cur_311, 4);
    float pv_315 = _shfl_xor_34;
    float _fmax_128 = fmaxf(cur_311, pv_315);
    float hi_316 = _fmax_128;
    float _min_118 = fminf(cur_311, pv_315);
    float lo_317 = _min_118;
    cur_311 = ((up[1] != 0) ? hi_316 : lo_317);
    float _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, cur_311, 2);
    float pv_318 = _shfl_xor_35;
    float _fmax_129 = fmaxf(cur_311, pv_318);
    float hi_319 = _fmax_129;
    float _min_119 = fminf(cur_311, pv_318);
    float lo_320 = _min_119;
    cur_311 = ((up[2] != 0) ? hi_319 : lo_320);
    float _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, cur_311, 1);
    float pv_321 = _shfl_xor_36;
    float _fmax_130 = fmaxf(cur_311, pv_321);
    float hi_322 = _fmax_130;
    float _min_120 = fminf(cur_311, pv_321);
    float lo_323 = _min_120;
    cur_311 = ((up[3] != 0) ? hi_322 : lo_323);
    V[0] = cur_311;
    float _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, V[5], 15);
    float y1_324 = _shfl_xor_37;
    float _min_121 = fminf(V[1], y1_324);
    float lo1_325 = _min_121;
    float _fmax_131 = fmaxf(r, lo1_325);
    r = _fmax_131;
    float _fmax_132 = fmaxf(V[1], y1_324);
    float hi1_326 = _fmax_132;
    float cur_327 = hi1_326;
    float _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, cur_327, 8);
    float pv_328 = _shfl_xor_38;
    float _fmax_133 = fmaxf(cur_327, pv_328);
    float hi_329 = _fmax_133;
    float _min_122 = fminf(cur_327, pv_328);
    float lo_330 = _min_122;
    cur_327 = ((up[0] != 0) ? hi_329 : lo_330);
    float _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, cur_327, 4);
    float pv_331 = _shfl_xor_39;
    float _fmax_134 = fmaxf(cur_327, pv_331);
    float hi_332 = _fmax_134;
    float _min_123 = fminf(cur_327, pv_331);
    float lo_333 = _min_123;
    cur_327 = ((up[1] != 0) ? hi_332 : lo_333);
    float _shfl_xor_40 = __shfl_xor_sync(0xFFFFFFFF, cur_327, 2);
    float pv_334 = _shfl_xor_40;
    float _fmax_135 = fmaxf(cur_327, pv_334);
    float hi_335 = _fmax_135;
    float _min_124 = fminf(cur_327, pv_334);
    float lo_336 = _min_124;
    cur_327 = ((up[2] != 0) ? hi_335 : lo_336);
    float _shfl_xor_41 = __shfl_xor_sync(0xFFFFFFFF, cur_327, 1);
    float pv_337 = _shfl_xor_41;
    float _fmax_136 = fmaxf(cur_327, pv_337);
    float hi_338 = _fmax_136;
    float _min_125 = fminf(cur_327, pv_337);
    float lo_339 = _min_125;
    cur_327 = ((up[3] != 0) ? hi_338 : lo_339);
    V[1] = cur_327;
    float _shfl_xor_42 = __shfl_xor_sync(0xFFFFFFFF, V[6], 15);
    float y1_340 = _shfl_xor_42;
    float _min_126 = fminf(V[2], y1_340);
    float lo1_341 = _min_126;
    float _fmax_137 = fmaxf(r, lo1_341);
    r = _fmax_137;
    float _fmax_138 = fmaxf(V[2], y1_340);
    float hi1_342 = _fmax_138;
    float cur_343 = hi1_342;
    float _shfl_xor_43 = __shfl_xor_sync(0xFFFFFFFF, cur_343, 8);
    float pv_344 = _shfl_xor_43;
    float _fmax_139 = fmaxf(cur_343, pv_344);
    float hi_345 = _fmax_139;
    float _min_127 = fminf(cur_343, pv_344);
    float lo_346 = _min_127;
    cur_343 = ((up[0] != 0) ? hi_345 : lo_346);
    float _shfl_xor_44 = __shfl_xor_sync(0xFFFFFFFF, cur_343, 4);
    float pv_347 = _shfl_xor_44;
    float _fmax_140 = fmaxf(cur_343, pv_347);
    float hi_348 = _fmax_140;
    float _min_128 = fminf(cur_343, pv_347);
    float lo_349 = _min_128;
    cur_343 = ((up[1] != 0) ? hi_348 : lo_349);
    float _shfl_xor_45 = __shfl_xor_sync(0xFFFFFFFF, cur_343, 2);
    float pv_350 = _shfl_xor_45;
    float _fmax_141 = fmaxf(cur_343, pv_350);
    float hi_351 = _fmax_141;
    float _min_129 = fminf(cur_343, pv_350);
    float lo_352 = _min_129;
    cur_343 = ((up[2] != 0) ? hi_351 : lo_352);
    float _shfl_xor_46 = __shfl_xor_sync(0xFFFFFFFF, cur_343, 1);
    float pv_353 = _shfl_xor_46;
    float _fmax_142 = fmaxf(cur_343, pv_353);
    float hi_354 = _fmax_142;
    float _min_130 = fminf(cur_343, pv_353);
    float lo_355 = _min_130;
    cur_343 = ((up[3] != 0) ? hi_354 : lo_355);
    V[2] = cur_343;
    float _shfl_xor_47 = __shfl_xor_sync(0xFFFFFFFF, V[7], 15);
    float y1_356 = _shfl_xor_47;
    float _min_131 = fminf(V[3], y1_356);
    float lo1_357 = _min_131;
    float _fmax_143 = fmaxf(r, lo1_357);
    r = _fmax_143;
    float _fmax_144 = fmaxf(V[3], y1_356);
    float hi1_358 = _fmax_144;
    float cur_359 = hi1_358;
    float _shfl_xor_48 = __shfl_xor_sync(0xFFFFFFFF, cur_359, 8);
    float pv_360 = _shfl_xor_48;
    float _fmax_145 = fmaxf(cur_359, pv_360);
    float hi_361 = _fmax_145;
    float _min_132 = fminf(cur_359, pv_360);
    float lo_362 = _min_132;
    cur_359 = ((up[0] != 0) ? hi_361 : lo_362);
    float _shfl_xor_49 = __shfl_xor_sync(0xFFFFFFFF, cur_359, 4);
    float pv_363 = _shfl_xor_49;
    float _fmax_146 = fmaxf(cur_359, pv_363);
    float hi_364 = _fmax_146;
    float _min_133 = fminf(cur_359, pv_363);
    float lo_365 = _min_133;
    cur_359 = ((up[1] != 0) ? hi_364 : lo_365);
    float _shfl_xor_50 = __shfl_xor_sync(0xFFFFFFFF, cur_359, 2);
    float pv_366 = _shfl_xor_50;
    float _fmax_147 = fmaxf(cur_359, pv_366);
    float hi_367 = _fmax_147;
    float _min_134 = fminf(cur_359, pv_366);
    float lo_368 = _min_134;
    cur_359 = ((up[2] != 0) ? hi_367 : lo_368);
    float _shfl_xor_51 = __shfl_xor_sync(0xFFFFFFFF, cur_359, 1);
    float pv_369 = _shfl_xor_51;
    float _fmax_148 = fmaxf(cur_359, pv_369);
    float hi_370 = _fmax_148;
    float _min_135 = fminf(cur_359, pv_369);
    float lo_371 = _min_135;
    cur_359 = ((up[3] != 0) ? hi_370 : lo_371);
    V[3] = cur_359;
    float _shfl_xor_52 = __shfl_xor_sync(0xFFFFFFFF, V[2], 15);
    float y1_372 = _shfl_xor_52;
    float _min_136 = fminf(V[0], y1_372);
    float lo1_373 = _min_136;
    float _fmax_149 = fmaxf(r, lo1_373);
    r = _fmax_149;
    float _fmax_150 = fmaxf(V[0], y1_372);
    float hi1_374 = _fmax_150;
    float cur_375 = hi1_374;
    float _shfl_xor_53 = __shfl_xor_sync(0xFFFFFFFF, cur_375, 8);
    float pv_376 = _shfl_xor_53;
    float _fmax_151 = fmaxf(cur_375, pv_376);
    float hi_377 = _fmax_151;
    float _min_137 = fminf(cur_375, pv_376);
    float lo_378 = _min_137;
    cur_375 = ((up[0] != 0) ? hi_377 : lo_378);
    float _shfl_xor_54 = __shfl_xor_sync(0xFFFFFFFF, cur_375, 4);
    float pv_379 = _shfl_xor_54;
    float _fmax_152 = fmaxf(cur_375, pv_379);
    float hi_380 = _fmax_152;
    float _min_138 = fminf(cur_375, pv_379);
    float lo_381 = _min_138;
    cur_375 = ((up[1] != 0) ? hi_380 : lo_381);
    float _shfl_xor_55 = __shfl_xor_sync(0xFFFFFFFF, cur_375, 2);
    float pv_382 = _shfl_xor_55;
    float _fmax_153 = fmaxf(cur_375, pv_382);
    float hi_383 = _fmax_153;
    float _min_139 = fminf(cur_375, pv_382);
    float lo_384 = _min_139;
    cur_375 = ((up[2] != 0) ? hi_383 : lo_384);
    float _shfl_xor_56 = __shfl_xor_sync(0xFFFFFFFF, cur_375, 1);
    float pv_385 = _shfl_xor_56;
    float _fmax_154 = fmaxf(cur_375, pv_385);
    float hi_386 = _fmax_154;
    float _min_140 = fminf(cur_375, pv_385);
    float lo_387 = _min_140;
    cur_375 = ((up[3] != 0) ? hi_386 : lo_387);
    V[0] = cur_375;
    float _shfl_xor_57 = __shfl_xor_sync(0xFFFFFFFF, V[3], 15);
    float y1_388 = _shfl_xor_57;
    float _min_141 = fminf(V[1], y1_388);
    float lo1_389 = _min_141;
    float _fmax_155 = fmaxf(r, lo1_389);
    r = _fmax_155;
    float _fmax_156 = fmaxf(V[1], y1_388);
    float hi1_390 = _fmax_156;
    float cur_391 = hi1_390;
    float _shfl_xor_58 = __shfl_xor_sync(0xFFFFFFFF, cur_391, 8);
    float pv_392 = _shfl_xor_58;
    float _fmax_157 = fmaxf(cur_391, pv_392);
    float hi_393 = _fmax_157;
    float _min_142 = fminf(cur_391, pv_392);
    float lo_394 = _min_142;
    cur_391 = ((up[0] != 0) ? hi_393 : lo_394);
    float _shfl_xor_59 = __shfl_xor_sync(0xFFFFFFFF, cur_391, 4);
    float pv_395 = _shfl_xor_59;
    float _fmax_158 = fmaxf(cur_391, pv_395);
    float hi_396 = _fmax_158;
    float _min_143 = fminf(cur_391, pv_395);
    float lo_397 = _min_143;
    cur_391 = ((up[1] != 0) ? hi_396 : lo_397);
    float _shfl_xor_60 = __shfl_xor_sync(0xFFFFFFFF, cur_391, 2);
    float pv_398 = _shfl_xor_60;
    float _fmax_159 = fmaxf(cur_391, pv_398);
    float hi_399 = _fmax_159;
    float _min_144 = fminf(cur_391, pv_398);
    float lo_400 = _min_144;
    cur_391 = ((up[2] != 0) ? hi_399 : lo_400);
    float _shfl_xor_61 = __shfl_xor_sync(0xFFFFFFFF, cur_391, 1);
    float pv_401 = _shfl_xor_61;
    float _fmax_160 = fmaxf(cur_391, pv_401);
    float hi_402 = _fmax_160;
    float _min_145 = fminf(cur_391, pv_401);
    float lo_403 = _min_145;
    cur_391 = ((up[3] != 0) ? hi_402 : lo_403);
    V[1] = cur_391;
    float _shfl_xor_62 = __shfl_xor_sync(0xFFFFFFFF, V[1], 15);
    float yl = _shfl_xor_62;
    float _min_146 = fminf(V[0], yl);
    float lol = _min_146;
    float _fmax_161 = fmaxf(r, lol);
    r = _fmax_161;
    float _fmax_162 = fmaxf(V[0], yl);
    float hil = _fmax_162;
    float cur_404 = hil;
    float _shfl_xor_63 = __shfl_xor_sync(0xFFFFFFFF, cur_404, 8);
    float pv_405 = _shfl_xor_63;
    float _fmax_163 = fmaxf(cur_404, pv_405);
    float hi_406 = _fmax_163;
    float _min_147 = fminf(cur_404, pv_405);
    float lo_407 = _min_147;
    cur_404 = ((up[0] != 0) ? hi_406 : lo_407);
    float _shfl_xor_64 = __shfl_xor_sync(0xFFFFFFFF, cur_404, 4);
    float pv_408 = _shfl_xor_64;
    float _fmax_164 = fmaxf(cur_404, pv_408);
    float hi_409 = _fmax_164;
    float _min_148 = fminf(cur_404, pv_408);
    float lo_410 = _min_148;
    cur_404 = ((up[1] != 0) ? hi_409 : lo_410);
    float _shfl_xor_65 = __shfl_xor_sync(0xFFFFFFFF, cur_404, 2);
    float pv_411 = _shfl_xor_65;
    float _fmax_165 = fmaxf(cur_404, pv_411);
    float hi_412 = _fmax_165;
    float _min_149 = fminf(cur_404, pv_411);
    float lo_413 = _min_149;
    cur_404 = ((up[2] != 0) ? hi_412 : lo_413);
    float _shfl_xor_66 = __shfl_xor_sync(0xFFFFFFFF, cur_404, 1);
    float pv_414 = _shfl_xor_66;
    float _fmax_166 = fmaxf(cur_404, pv_414);
    float hi_415 = _fmax_166;
    float _min_150 = fminf(cur_404, pv_414);
    float lo_416 = _min_150;
    cur_404 = ((up[3] != 0) ? hi_415 : lo_416);
    V[0] = cur_404;
    float K = V[0];
    int qb = (sg * 2 + cg) * 32;
    q2[qb + ln] = K;
    q2[qb + 16 + ln] = r;
    asm volatile("barrier.sync 8, 256;" ::: "memory");
    if (tid_1 < 32) {
        float r2 = neg_inf;
        float _fmax_167 = fmaxf(r2, q2[cg * 32 + 16 + ln]);
        r2 = _fmax_167;
        float _fmax_168 = fmaxf(r2, q2[(2 + cg) * 32 + 16 + ln]);
        r2 = _fmax_168;
        float _fmax_169 = fmaxf(r2, q2[(4 + cg) * 32 + 16 + ln]);
        r2 = _fmax_169;
        float _fmax_170 = fmaxf(r2, q2[(6 + cg) * 32 + 16 + ln]);
        r2 = _fmax_170;
        float _fmax_171 = fmaxf(r2, q2[(8 + cg) * 32 + 16 + ln]);
        r2 = _fmax_171;
        float _fmax_172 = fmaxf(r2, q2[(10 + cg) * 32 + 16 + ln]);
        r2 = _fmax_172;
        float _fmax_173 = fmaxf(r2, q2[(12 + cg) * 32 + 16 + ln]);
        r2 = _fmax_173;
        float _fmax_174 = fmaxf(r2, q2[(14 + cg) * 32 + 16 + ln]);
        r2 = _fmax_174;
        float V2[4];
        float x2 = q2[cg * 32 + ln];
        float y2 = q2[(8 + cg) * 32 + lnr];
        float _min_151 = fminf(x2, y2);
        float lo2 = _min_151;
        float _fmax_175 = fmaxf(r2, lo2);
        r2 = _fmax_175;
        float _fmax_176 = fmaxf(x2, y2);
        float hi2 = _fmax_176;
        float cur_0 = hi2;
        float _shfl_xor_67 = __shfl_xor_sync(0xFFFFFFFF, cur_0, 8);
        float pv_1 = _shfl_xor_67;
        float _fmax_177 = fmaxf(cur_0, pv_1);
        float hi_2 = _fmax_177;
        float _min_152 = fminf(cur_0, pv_1);
        float lo_3 = _min_152;
        cur_0 = ((up[0] != 0) ? hi_2 : lo_3);
        float _shfl_xor_68 = __shfl_xor_sync(0xFFFFFFFF, cur_0, 4);
        float pv_4 = _shfl_xor_68;
        float _fmax_178 = fmaxf(cur_0, pv_4);
        float hi_5 = _fmax_178;
        float _min_153 = fminf(cur_0, pv_4);
        float lo_6 = _min_153;
        cur_0 = ((up[1] != 0) ? hi_5 : lo_6);
        float _shfl_xor_69 = __shfl_xor_sync(0xFFFFFFFF, cur_0, 2);
        float pv_7 = _shfl_xor_69;
        float _fmax_179 = fmaxf(cur_0, pv_7);
        float hi_8 = _fmax_179;
        float _min_154 = fminf(cur_0, pv_7);
        float lo_9 = _min_154;
        cur_0 = ((up[2] != 0) ? hi_8 : lo_9);
        float _shfl_xor_70 = __shfl_xor_sync(0xFFFFFFFF, cur_0, 1);
        float pv_10 = _shfl_xor_70;
        float _fmax_180 = fmaxf(cur_0, pv_10);
        float hi_11 = _fmax_180;
        float _min_155 = fminf(cur_0, pv_10);
        float lo_12 = _min_155;
        cur_0 = ((up[3] != 0) ? hi_11 : lo_12);
        V2[0] = cur_0;
        float x2_13 = q2[(2 + cg) * 32 + ln];
        float y2_14 = q2[(10 + cg) * 32 + lnr];
        float _min_156 = fminf(x2_13, y2_14);
        float lo2_15 = _min_156;
        float _fmax_181 = fmaxf(r2, lo2_15);
        r2 = _fmax_181;
        float _fmax_182 = fmaxf(x2_13, y2_14);
        float hi2_16 = _fmax_182;
        float cur_17 = hi2_16;
        float _shfl_xor_71 = __shfl_xor_sync(0xFFFFFFFF, cur_17, 8);
        float pv_18 = _shfl_xor_71;
        float _fmax_183 = fmaxf(cur_17, pv_18);
        float hi_19 = _fmax_183;
        float _min_157 = fminf(cur_17, pv_18);
        float lo_20 = _min_157;
        cur_17 = ((up[0] != 0) ? hi_19 : lo_20);
        float _shfl_xor_72 = __shfl_xor_sync(0xFFFFFFFF, cur_17, 4);
        float pv_21 = _shfl_xor_72;
        float _fmax_184 = fmaxf(cur_17, pv_21);
        float hi_22 = _fmax_184;
        float _min_158 = fminf(cur_17, pv_21);
        float lo_23 = _min_158;
        cur_17 = ((up[1] != 0) ? hi_22 : lo_23);
        float _shfl_xor_73 = __shfl_xor_sync(0xFFFFFFFF, cur_17, 2);
        float pv_24 = _shfl_xor_73;
        float _fmax_185 = fmaxf(cur_17, pv_24);
        float hi_25 = _fmax_185;
        float _min_159 = fminf(cur_17, pv_24);
        float lo_26 = _min_159;
        cur_17 = ((up[2] != 0) ? hi_25 : lo_26);
        float _shfl_xor_74 = __shfl_xor_sync(0xFFFFFFFF, cur_17, 1);
        float pv_27 = _shfl_xor_74;
        float _fmax_186 = fmaxf(cur_17, pv_27);
        float hi_28 = _fmax_186;
        float _min_160 = fminf(cur_17, pv_27);
        float lo_29 = _min_160;
        cur_17 = ((up[3] != 0) ? hi_28 : lo_29);
        V2[1] = cur_17;
        float x2_30 = q2[(4 + cg) * 32 + ln];
        float y2_31 = q2[(12 + cg) * 32 + lnr];
        float _min_161 = fminf(x2_30, y2_31);
        float lo2_32 = _min_161;
        float _fmax_187 = fmaxf(r2, lo2_32);
        r2 = _fmax_187;
        float _fmax_188 = fmaxf(x2_30, y2_31);
        float hi2_33 = _fmax_188;
        float cur_34 = hi2_33;
        float _shfl_xor_75 = __shfl_xor_sync(0xFFFFFFFF, cur_34, 8);
        float pv_35 = _shfl_xor_75;
        float _fmax_189 = fmaxf(cur_34, pv_35);
        float hi_36 = _fmax_189;
        float _min_162 = fminf(cur_34, pv_35);
        float lo_37 = _min_162;
        cur_34 = ((up[0] != 0) ? hi_36 : lo_37);
        float _shfl_xor_76 = __shfl_xor_sync(0xFFFFFFFF, cur_34, 4);
        float pv_38 = _shfl_xor_76;
        float _fmax_190 = fmaxf(cur_34, pv_38);
        float hi_39 = _fmax_190;
        float _min_163 = fminf(cur_34, pv_38);
        float lo_40 = _min_163;
        cur_34 = ((up[1] != 0) ? hi_39 : lo_40);
        float _shfl_xor_77 = __shfl_xor_sync(0xFFFFFFFF, cur_34, 2);
        float pv_41 = _shfl_xor_77;
        float _fmax_191 = fmaxf(cur_34, pv_41);
        float hi_42 = _fmax_191;
        float _min_164 = fminf(cur_34, pv_41);
        float lo_43 = _min_164;
        cur_34 = ((up[2] != 0) ? hi_42 : lo_43);
        float _shfl_xor_78 = __shfl_xor_sync(0xFFFFFFFF, cur_34, 1);
        float pv_44 = _shfl_xor_78;
        float _fmax_192 = fmaxf(cur_34, pv_44);
        float hi_45 = _fmax_192;
        float _min_165 = fminf(cur_34, pv_44);
        float lo_46 = _min_165;
        cur_34 = ((up[3] != 0) ? hi_45 : lo_46);
        V2[2] = cur_34;
        float x2_47 = q2[(6 + cg) * 32 + ln];
        float y2_48 = q2[(14 + cg) * 32 + lnr];
        float _min_166 = fminf(x2_47, y2_48);
        float lo2_49 = _min_166;
        float _fmax_193 = fmaxf(r2, lo2_49);
        r2 = _fmax_193;
        float _fmax_194 = fmaxf(x2_47, y2_48);
        float hi2_50 = _fmax_194;
        float cur_51 = hi2_50;
        float _shfl_xor_79 = __shfl_xor_sync(0xFFFFFFFF, cur_51, 8);
        float pv_52 = _shfl_xor_79;
        float _fmax_195 = fmaxf(cur_51, pv_52);
        float hi_54 = _fmax_195;
        float _min_167 = fminf(cur_51, pv_52);
        float lo_55 = _min_167;
        cur_51 = ((up[0] != 0) ? hi_54 : lo_55);
        float _shfl_xor_80 = __shfl_xor_sync(0xFFFFFFFF, cur_51, 4);
        float pv_56 = _shfl_xor_80;
        float _fmax_196 = fmaxf(cur_51, pv_56);
        float hi_58 = _fmax_196;
        float _min_168 = fminf(cur_51, pv_56);
        float lo_59 = _min_168;
        cur_51 = ((up[1] != 0) ? hi_58 : lo_59);
        float _shfl_xor_81 = __shfl_xor_sync(0xFFFFFFFF, cur_51, 2);
        float pv_60 = _shfl_xor_81;
        float _fmax_197 = fmaxf(cur_51, pv_60);
        float hi_62 = _fmax_197;
        float _min_169 = fminf(cur_51, pv_60);
        float lo_63 = _min_169;
        cur_51 = ((up[2] != 0) ? hi_62 : lo_63);
        float _shfl_xor_82 = __shfl_xor_sync(0xFFFFFFFF, cur_51, 1);
        float pv_64 = _shfl_xor_82;
        float _fmax_198 = fmaxf(cur_51, pv_64);
        float hi_66 = _fmax_198;
        float _min_170 = fminf(cur_51, pv_64);
        float lo_67 = _min_170;
        cur_51 = ((up[3] != 0) ? hi_66 : lo_67);
        V2[3] = cur_51;
        float _shfl_xor_83 = __shfl_xor_sync(0xFFFFFFFF, V2[2], 15);
        float y3 = _shfl_xor_83;
        float _min_171 = fminf(V2[0], y3);
        float lo3 = _min_171;
        float _fmax_199 = fmaxf(r2, lo3);
        r2 = _fmax_199;
        float _fmax_200 = fmaxf(V2[0], y3);
        float hi3 = _fmax_200;
        float cur_68 = hi3;
        float _shfl_xor_84 = __shfl_xor_sync(0xFFFFFFFF, cur_68, 8);
        float pv_69 = _shfl_xor_84;
        float _fmax_201 = fmaxf(cur_68, pv_69);
        float hi_70 = _fmax_201;
        float _min_172 = fminf(cur_68, pv_69);
        float lo_71 = _min_172;
        cur_68 = ((up[0] != 0) ? hi_70 : lo_71);
        float _shfl_xor_85 = __shfl_xor_sync(0xFFFFFFFF, cur_68, 4);
        float pv_72 = _shfl_xor_85;
        float _fmax_202 = fmaxf(cur_68, pv_72);
        float hi_74 = _fmax_202;
        float _min_173 = fminf(cur_68, pv_72);
        float lo_75 = _min_173;
        cur_68 = ((up[1] != 0) ? hi_74 : lo_75);
        float _shfl_xor_86 = __shfl_xor_sync(0xFFFFFFFF, cur_68, 2);
        float pv_76 = _shfl_xor_86;
        float _fmax_203 = fmaxf(cur_68, pv_76);
        float hi_78 = _fmax_203;
        float _min_174 = fminf(cur_68, pv_76);
        float lo_79 = _min_174;
        cur_68 = ((up[2] != 0) ? hi_78 : lo_79);
        float _shfl_xor_87 = __shfl_xor_sync(0xFFFFFFFF, cur_68, 1);
        float pv_80 = _shfl_xor_87;
        float _fmax_204 = fmaxf(cur_68, pv_80);
        float hi_82 = _fmax_204;
        float _min_175 = fminf(cur_68, pv_80);
        float lo_83 = _min_175;
        cur_68 = ((up[3] != 0) ? hi_82 : lo_83);
        V2[0] = cur_68;
        float _shfl_xor_88 = __shfl_xor_sync(0xFFFFFFFF, V2[3], 15);
        float y3_84 = _shfl_xor_88;
        float _min_176 = fminf(V2[1], y3_84);
        float lo3_85 = _min_176;
        float _fmax_205 = fmaxf(r2, lo3_85);
        r2 = _fmax_205;
        float _fmax_206 = fmaxf(V2[1], y3_84);
        float hi3_86 = _fmax_206;
        float cur_87 = hi3_86;
        float _shfl_xor_89 = __shfl_xor_sync(0xFFFFFFFF, cur_87, 8);
        float pv_88 = _shfl_xor_89;
        float _fmax_207 = fmaxf(cur_87, pv_88);
        float hi_90 = _fmax_207;
        float _min_177 = fminf(cur_87, pv_88);
        float lo_91 = _min_177;
        cur_87 = ((up[0] != 0) ? hi_90 : lo_91);
        float _shfl_xor_90 = __shfl_xor_sync(0xFFFFFFFF, cur_87, 4);
        float pv_92 = _shfl_xor_90;
        float _fmax_208 = fmaxf(cur_87, pv_92);
        float hi_94 = _fmax_208;
        float _min_178 = fminf(cur_87, pv_92);
        float lo_95 = _min_178;
        cur_87 = ((up[1] != 0) ? hi_94 : lo_95);
        float _shfl_xor_91 = __shfl_xor_sync(0xFFFFFFFF, cur_87, 2);
        float pv_96 = _shfl_xor_91;
        float _fmax_209 = fmaxf(cur_87, pv_96);
        float hi_98 = _fmax_209;
        float _min_179 = fminf(cur_87, pv_96);
        float lo_99 = _min_179;
        cur_87 = ((up[2] != 0) ? hi_98 : lo_99);
        float _shfl_xor_92 = __shfl_xor_sync(0xFFFFFFFF, cur_87, 1);
        float pv_100 = _shfl_xor_92;
        float _fmax_210 = fmaxf(cur_87, pv_100);
        float hi_102 = _fmax_210;
        float _min_180 = fminf(cur_87, pv_100);
        float lo_103 = _min_180;
        cur_87 = ((up[3] != 0) ? hi_102 : lo_103);
        V2[1] = cur_87;
        float _shfl_xor_93 = __shfl_xor_sync(0xFFFFFFFF, V2[1], 15);
        float yl2 = _shfl_xor_93;
        float _min_181 = fminf(V2[0], yl2);
        float lol2 = _min_181;
        float _fmax_211 = fmaxf(r2, lol2);
        r2 = _fmax_211;
        float _fmax_212 = fmaxf(V2[0], yl2);
        V2[0] = _fmax_212;
        K = V2[0];
        r = r2;
    }
    rr1[0] = r;
    int ucol = blockIdx.x * 2 + cg;
    int commit = 0;
    if (g < 2 && ucol < total_q) {
        commit = 1;
    }
    if (tid_1 < 32) {
        float u16 = K;
        float cr = rr1[0];
        float _shfl_xor_94 = __shfl_xor_sync(0xFFFFFFFF, u16, 8);
        float _min_182 = fminf(u16, _shfl_xor_94);
        u16 = _min_182;
        float _shfl_xor_95 = __shfl_xor_sync(0xFFFFFFFF, cr, 8);
        float _fmax_213 = fmaxf(cr, _shfl_xor_95);
        cr = _fmax_213;
        float _shfl_xor_96 = __shfl_xor_sync(0xFFFFFFFF, u16, 4);
        float _min_183 = fminf(u16, _shfl_xor_96);
        u16 = _min_183;
        float _shfl_xor_97 = __shfl_xor_sync(0xFFFFFFFF, cr, 4);
        float _fmax_214 = fmaxf(cr, _shfl_xor_97);
        cr = _fmax_214;
        float _shfl_xor_98 = __shfl_xor_sync(0xFFFFFFFF, u16, 2);
        float _min_184 = fminf(u16, _shfl_xor_98);
        u16 = _min_184;
        float _shfl_xor_99 = __shfl_xor_sync(0xFFFFFFFF, cr, 2);
        float _fmax_215 = fmaxf(cr, _shfl_xor_99);
        cr = _fmax_215;
        float _shfl_xor_100 = __shfl_xor_sync(0xFFFFFFFF, u16, 1);
        float _min_185 = fminf(u16, _shfl_xor_100);
        u16 = _min_185;
        float _shfl_xor_101 = __shfl_xor_sync(0xFFFFFFFF, cr, 1);
        float _fmax_216 = fmaxf(cr, _shfl_xor_101);
        cr = _fmax_216;
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
        if (ln == 0 && g < 2) {
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
            int t0_3 = w * 16;
            float qf6 = __uint_as_float(qc);
            float sc_6 = __uint_as_float(cb[0]);
            float _fmax_217 = fmaxf(sc_6, -1.7014118346046923e+38f);
            sc_6 = _fmax_217;
            float _min_186 = fminf(sc_6, 1.7014118346046923e+38f);
            sc_6 = _min_186;
            sc_6 = sc_6;
            float sc8 = sc_6;
            unsigned int u8 = __as_u32(sc8);
            unsigned int key8 = 0;
            if ((u8 & 4294965248u) == qc) {
                unsigned int lowb8 = (u8 ^ (unsigned int)((int)u8 >> 31) & 2047) & 2047;
                key8 = 536870912 | lowb8 << 11 | (unsigned int)t0_3;
            }
            unsigned int k6 = key8;
            if (k6 != 0) {
                unsigned int _atomic_old_0 = atomicAdd(&ccnt[c], 1);
                unsigned int p6 = _atomic_old_0;
                if (p6 < 16) {
                    cbuf[c * 16 + (int)p6] = k6;
                } else {
                    flagw[1] = 1;
                }
            }
            float sc_9 = __uint_as_float(cb[1]);
            float _fmax_218 = fmaxf(sc_9, -1.7014118346046923e+38f);
            sc_9 = _fmax_218;
            float _min_187 = fminf(sc_9, 1.7014118346046923e+38f);
            sc_9 = _min_187;
            sc_9 = sc_9;
            float sc8_10 = sc_9;
            unsigned int u8_11 = __as_u32(sc8_10);
            unsigned int key8_12 = 0;
            if ((u8_11 & 4294965248u) == qc) {
                unsigned int lowb8_1 = (u8_11 ^ (unsigned int)((int)u8_11 >> 31) & 2047) & 2047;
                key8_12 = 536870912 | lowb8_1 << 11 | (unsigned int)(t0_3 + 1);
            }
            unsigned int k6_13 = key8_12;
            if (k6_13 != 0) {
                unsigned int _atomic_old_1 = atomicAdd(&ccnt[c], 1);
                unsigned int p6_1 = _atomic_old_1;
                if (p6_1 < 16) {
                    cbuf[c * 16 + (int)p6_1] = k6_13;
                } else {
                    flagw[1] = 1;
                }
            }
            float sc_15 = __uint_as_float(cb[2]);
            float _fmax_219 = fmaxf(sc_15, -1.7014118346046923e+38f);
            sc_15 = _fmax_219;
            float _min_188 = fminf(sc_15, 1.7014118346046923e+38f);
            sc_15 = _min_188;
            sc_15 = sc_15;
            float sc8_16 = sc_15;
            unsigned int u8_17 = __as_u32(sc8_16);
            unsigned int key8_18 = 0;
            if ((u8_17 & 4294965248u) == qc) {
                unsigned int lowb8_2 = (u8_17 ^ (unsigned int)((int)u8_17 >> 31) & 2047) & 2047;
                key8_18 = 536870912 | lowb8_2 << 11 | (unsigned int)(t0_3 + 2);
            }
            unsigned int k6_19 = key8_18;
            if (k6_19 != 0) {
                unsigned int _atomic_old_2 = atomicAdd(&ccnt[c], 1);
                unsigned int p6_2 = _atomic_old_2;
                if (p6_2 < 16) {
                    cbuf[c * 16 + (int)p6_2] = k6_19;
                } else {
                    flagw[1] = 1;
                }
            }
            float sc_21 = __uint_as_float(cb[3]);
            float _fmax_220 = fmaxf(sc_21, -1.7014118346046923e+38f);
            sc_21 = _fmax_220;
            float _min_189 = fminf(sc_21, 1.7014118346046923e+38f);
            sc_21 = _min_189;
            sc_21 = sc_21;
            float sc8_22 = sc_21;
            unsigned int u8_23 = __as_u32(sc8_22);
            unsigned int key8_24 = 0;
            if ((u8_23 & 4294965248u) == qc) {
                unsigned int lowb8_3 = (u8_23 ^ (unsigned int)((int)u8_23 >> 31) & 2047) & 2047;
                key8_24 = 536870912 | lowb8_3 << 11 | (unsigned int)(t0_3 + 3);
            }
            unsigned int k6_25 = key8_24;
            if (k6_25 != 0) {
                unsigned int _atomic_old_3 = atomicAdd(&ccnt[c], 1);
                unsigned int p6_3 = _atomic_old_3;
                if (p6_3 < 16) {
                    cbuf[c * 16 + (int)p6_3] = k6_25;
                } else {
                    flagw[1] = 1;
                }
            }
            float sc_27 = __uint_as_float(cb[4]);
            float _fmax_221 = fmaxf(sc_27, -1.7014118346046923e+38f);
            sc_27 = _fmax_221;
            float _min_190 = fminf(sc_27, 1.7014118346046923e+38f);
            sc_27 = _min_190;
            sc_27 = sc_27;
            float sc8_28 = sc_27;
            unsigned int u8_29 = __as_u32(sc8_28);
            unsigned int key8_30 = 0;
            if ((u8_29 & 4294965248u) == qc) {
                unsigned int lowb8_4 = (u8_29 ^ (unsigned int)((int)u8_29 >> 31) & 2047) & 2047;
                key8_30 = 536870912 | lowb8_4 << 11 | (unsigned int)(t0_3 + 4);
            }
            unsigned int k6_31 = key8_30;
            if (k6_31 != 0) {
                unsigned int _atomic_old_4 = atomicAdd(&ccnt[c], 1);
                unsigned int p6_4 = _atomic_old_4;
                if (p6_4 < 16) {
                    cbuf[c * 16 + (int)p6_4] = k6_31;
                } else {
                    flagw[1] = 1;
                }
            }
            float sc_33 = __uint_as_float(cb[5]);
            float _fmax_222 = fmaxf(sc_33, -1.7014118346046923e+38f);
            sc_33 = _fmax_222;
            float _min_191 = fminf(sc_33, 1.7014118346046923e+38f);
            sc_33 = _min_191;
            sc_33 = sc_33;
            float sc8_34 = sc_33;
            unsigned int u8_35 = __as_u32(sc8_34);
            unsigned int key8_36 = 0;
            if ((u8_35 & 4294965248u) == qc) {
                unsigned int lowb8_5 = (u8_35 ^ (unsigned int)((int)u8_35 >> 31) & 2047) & 2047;
                key8_36 = 536870912 | lowb8_5 << 11 | (unsigned int)(t0_3 + 5);
            }
            unsigned int k6_37 = key8_36;
            if (k6_37 != 0) {
                unsigned int _atomic_old_5 = atomicAdd(&ccnt[c], 1);
                unsigned int p6_5 = _atomic_old_5;
                if (p6_5 < 16) {
                    cbuf[c * 16 + (int)p6_5] = k6_37;
                } else {
                    flagw[1] = 1;
                }
            }
            float sc_39 = __uint_as_float(cb[6]);
            float _fmax_223 = fmaxf(sc_39, -1.7014118346046923e+38f);
            sc_39 = _fmax_223;
            float _min_192 = fminf(sc_39, 1.7014118346046923e+38f);
            sc_39 = _min_192;
            sc_39 = sc_39;
            float sc8_40 = sc_39;
            unsigned int u8_41 = __as_u32(sc8_40);
            unsigned int key8_42 = 0;
            if ((u8_41 & 4294965248u) == qc) {
                unsigned int lowb8_6 = (u8_41 ^ (unsigned int)((int)u8_41 >> 31) & 2047) & 2047;
                key8_42 = 536870912 | lowb8_6 << 11 | (unsigned int)(t0_3 + 6);
            }
            unsigned int k6_43 = key8_42;
            if (k6_43 != 0) {
                unsigned int _atomic_old_6 = atomicAdd(&ccnt[c], 1);
                unsigned int p6_6 = _atomic_old_6;
                if (p6_6 < 16) {
                    cbuf[c * 16 + (int)p6_6] = k6_43;
                } else {
                    flagw[1] = 1;
                }
            }
            float sc_45 = __uint_as_float(cb[7]);
            float _fmax_224 = fmaxf(sc_45, -1.7014118346046923e+38f);
            sc_45 = _fmax_224;
            float _min_193 = fminf(sc_45, 1.7014118346046923e+38f);
            sc_45 = _min_193;
            sc_45 = sc_45;
            float sc8_46 = sc_45;
            unsigned int u8_47 = __as_u32(sc8_46);
            unsigned int key8_48 = 0;
            if ((u8_47 & 4294965248u) == qc) {
                unsigned int lowb8_7 = (u8_47 ^ (unsigned int)((int)u8_47 >> 31) & 2047) & 2047;
                key8_48 = 536870912 | lowb8_7 << 11 | (unsigned int)(t0_3 + 7);
            }
            unsigned int k6_49 = key8_48;
            if (k6_49 != 0) {
                unsigned int _atomic_old_7 = atomicAdd(&ccnt[c], 1);
                unsigned int p6_7 = _atomic_old_7;
                if (p6_7 < 16) {
                    cbuf[c * 16 + (int)p6_7] = k6_49;
                } else {
                    flagw[1] = 1;
                }
            }
            float sc_50 = __uint_as_float(cb[8]);
            float _fmax_225 = fmaxf(sc_50, -1.7014118346046923e+38f);
            sc_50 = _fmax_225;
            float _min_194 = fminf(sc_50, 1.7014118346046923e+38f);
            sc_50 = _min_194;
            sc_50 = sc_50;
            float sc8_51 = sc_50;
            unsigned int u8_52 = __as_u32(sc8_51);
            unsigned int key8_53 = 0;
            if ((u8_52 & 4294965248u) == qc) {
                unsigned int lowb8_8 = (u8_52 ^ (unsigned int)((int)u8_52 >> 31) & 2047) & 2047;
                key8_53 = 536870912 | lowb8_8 << 11 | (unsigned int)(t0_3 + 8);
            }
            unsigned int k6_54 = key8_53;
            if (k6_54 != 0) {
                unsigned int _atomic_old_8 = atomicAdd(&ccnt[c], 1);
                unsigned int p6_8 = _atomic_old_8;
                if (p6_8 < 16) {
                    cbuf[c * 16 + (int)p6_8] = k6_54;
                } else {
                    flagw[1] = 1;
                }
            }
            float sc_55 = __uint_as_float(cb[9]);
            float _fmax_226 = fmaxf(sc_55, -1.7014118346046923e+38f);
            sc_55 = _fmax_226;
            float _min_195 = fminf(sc_55, 1.7014118346046923e+38f);
            sc_55 = _min_195;
            sc_55 = sc_55;
            float sc8_56 = sc_55;
            unsigned int u8_57 = __as_u32(sc8_56);
            unsigned int key8_58 = 0;
            if ((u8_57 & 4294965248u) == qc) {
                unsigned int lowb8_9 = (u8_57 ^ (unsigned int)((int)u8_57 >> 31) & 2047) & 2047;
                key8_58 = 536870912 | lowb8_9 << 11 | (unsigned int)(t0_3 + 9);
            }
            unsigned int k6_59 = key8_58;
            if (k6_59 != 0) {
                unsigned int _atomic_old_9 = atomicAdd(&ccnt[c], 1);
                unsigned int p6_9 = _atomic_old_9;
                if (p6_9 < 16) {
                    cbuf[c * 16 + (int)p6_9] = k6_59;
                } else {
                    flagw[1] = 1;
                }
            }
            float sc_60 = __uint_as_float(cb[10]);
            float _fmax_227 = fmaxf(sc_60, -1.7014118346046923e+38f);
            sc_60 = _fmax_227;
            float _min_196 = fminf(sc_60, 1.7014118346046923e+38f);
            sc_60 = _min_196;
            sc_60 = sc_60;
            float sc8_61 = sc_60;
            unsigned int u8_62 = __as_u32(sc8_61);
            unsigned int key8_63 = 0;
            if ((u8_62 & 4294965248u) == qc) {
                unsigned int lowb8_10 = (u8_62 ^ (unsigned int)((int)u8_62 >> 31) & 2047) & 2047;
                key8_63 = 536870912 | lowb8_10 << 11 | (unsigned int)(t0_3 + 10);
            }
            unsigned int k6_64 = key8_63;
            if (k6_64 != 0) {
                unsigned int _atomic_old_10 = atomicAdd(&ccnt[c], 1);
                unsigned int p6_10 = _atomic_old_10;
                if (p6_10 < 16) {
                    cbuf[c * 16 + (int)p6_10] = k6_64;
                } else {
                    flagw[1] = 1;
                }
            }
            float sc_65 = __uint_as_float(cb[11]);
            float _fmax_228 = fmaxf(sc_65, -1.7014118346046923e+38f);
            sc_65 = _fmax_228;
            float _min_197 = fminf(sc_65, 1.7014118346046923e+38f);
            sc_65 = _min_197;
            sc_65 = sc_65;
            float sc8_66 = sc_65;
            unsigned int u8_67 = __as_u32(sc8_66);
            unsigned int key8_68 = 0;
            if ((u8_67 & 4294965248u) == qc) {
                unsigned int lowb8_11 = (u8_67 ^ (unsigned int)((int)u8_67 >> 31) & 2047) & 2047;
                key8_68 = 536870912 | lowb8_11 << 11 | (unsigned int)(t0_3 + 11);
            }
            unsigned int k6_69 = key8_68;
            if (k6_69 != 0) {
                unsigned int _atomic_old_11 = atomicAdd(&ccnt[c], 1);
                unsigned int p6_11 = _atomic_old_11;
                if (p6_11 < 16) {
                    cbuf[c * 16 + (int)p6_11] = k6_69;
                } else {
                    flagw[1] = 1;
                }
            }
            float sc_70 = __uint_as_float(cb[12]);
            float _fmax_229 = fmaxf(sc_70, -1.7014118346046923e+38f);
            sc_70 = _fmax_229;
            float _min_198 = fminf(sc_70, 1.7014118346046923e+38f);
            sc_70 = _min_198;
            sc_70 = sc_70;
            float sc8_71 = sc_70;
            unsigned int u8_72 = __as_u32(sc8_71);
            unsigned int key8_73 = 0;
            if ((u8_72 & 4294965248u) == qc) {
                unsigned int lowb8_12 = (u8_72 ^ (unsigned int)((int)u8_72 >> 31) & 2047) & 2047;
                key8_73 = 536870912 | lowb8_12 << 11 | (unsigned int)(t0_3 + 12);
            }
            unsigned int k6_74 = key8_73;
            if (k6_74 != 0) {
                unsigned int _atomic_old_12 = atomicAdd(&ccnt[c], 1);
                unsigned int p6_12 = _atomic_old_12;
                if (p6_12 < 16) {
                    cbuf[c * 16 + (int)p6_12] = k6_74;
                } else {
                    flagw[1] = 1;
                }
            }
            float sc_75 = __uint_as_float(cb[13]);
            float _fmax_230 = fmaxf(sc_75, -1.7014118346046923e+38f);
            sc_75 = _fmax_230;
            float _min_199 = fminf(sc_75, 1.7014118346046923e+38f);
            sc_75 = _min_199;
            sc_75 = sc_75;
            float sc8_76 = sc_75;
            unsigned int u8_77 = __as_u32(sc8_76);
            unsigned int key8_78 = 0;
            if ((u8_77 & 4294965248u) == qc) {
                unsigned int lowb8_13 = (u8_77 ^ (unsigned int)((int)u8_77 >> 31) & 2047) & 2047;
                key8_78 = 536870912 | lowb8_13 << 11 | (unsigned int)(t0_3 + 13);
            }
            unsigned int k6_79 = key8_78;
            if (k6_79 != 0) {
                unsigned int _atomic_old_13 = atomicAdd(&ccnt[c], 1);
                unsigned int p6_13 = _atomic_old_13;
                if (p6_13 < 16) {
                    cbuf[c * 16 + (int)p6_13] = k6_79;
                } else {
                    flagw[1] = 1;
                }
            }
            float sc_80 = __uint_as_float(cb[14]);
            float _fmax_231 = fmaxf(sc_80, -1.7014118346046923e+38f);
            sc_80 = _fmax_231;
            float _min_200 = fminf(sc_80, 1.7014118346046923e+38f);
            sc_80 = _min_200;
            sc_80 = sc_80;
            float sc8_81 = sc_80;
            unsigned int u8_82 = __as_u32(sc8_81);
            unsigned int key8_83 = 0;
            if ((u8_82 & 4294965248u) == qc) {
                unsigned int lowb8_14 = (u8_82 ^ (unsigned int)((int)u8_82 >> 31) & 2047) & 2047;
                key8_83 = 536870912 | lowb8_14 << 11 | (unsigned int)(t0_3 + 14);
            }
            unsigned int k6_84 = key8_83;
            if (k6_84 != 0) {
                unsigned int _atomic_old_14 = atomicAdd(&ccnt[c], 1);
                unsigned int p6_14 = _atomic_old_14;
                if (p6_14 < 16) {
                    cbuf[c * 16 + (int)p6_14] = k6_84;
                } else {
                    flagw[1] = 1;
                }
            }
            float sc_85 = __uint_as_float(cb[15]);
            float _fmax_232 = fmaxf(sc_85, -1.7014118346046923e+38f);
            sc_85 = _fmax_232;
            float _min_201 = fminf(sc_85, 1.7014118346046923e+38f);
            sc_85 = _min_201;
            sc_85 = sc_85;
            float sc8_86 = sc_85;
            unsigned int u8_87 = __as_u32(sc8_86);
            unsigned int key8_88 = 0;
            if ((u8_87 & 4294965248u) == qc) {
                unsigned int lowb8_15 = (u8_87 ^ (unsigned int)((int)u8_87 >> 31) & 2047) & 2047;
                key8_88 = 536870912 | lowb8_15 << 11 | (unsigned int)(t0_3 + 15);
            }
            unsigned int k6_89 = key8_88;
            if (k6_89 != 0) {
                unsigned int _atomic_old_15 = atomicAdd(&ccnt[c], 1);
                unsigned int p6_15 = _atomic_old_15;
                if (p6_15 < 16) {
                    cbuf[c * 16 + (int)p6_15] = k6_89;
                } else {
                    flagw[1] = 1;
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
                float kb2o[16];
                int t0_3_1 = w * 16;
                int t0_4 = t0_3_1;
                float qf = __uint_as_float(qc);
                float sc_6_1 = __uint_as_float(cb[0]);
                float _fmax_233 = fmaxf(sc_6_1, -1.7014118346046923e+38f);
                sc_6_1 = _fmax_233;
                float _min_202 = fminf(sc_6_1, 1.7014118346046923e+38f);
                sc_6_1 = _min_202;
                sc_6_1 = sc_6_1;
                float sc_9_1 = sc_6_1;
                unsigned int u = __as_u32(sc_9_1);
                unsigned int cls = u & 4294965248u;
                unsigned int key_10 = 0;
                if (qf < __uint_as_float(cls) && cls < 4278190080u) {
                    key_10 = 1073741824 | (unsigned int)t0_4;
                }
                if (cls == qc) {
                    unsigned int lowb = (u ^ (unsigned int)((int)u >> 31) & 2047) & 2047;
                    key_10 = 536870912 | lowb << 11 | (unsigned int)t0_4;
                }
                kb2o[0] = __uint_as_float(key_10);
                float sc_12 = __uint_as_float(cb[1]);
                float _fmax_234 = fmaxf(sc_12, -1.7014118346046923e+38f);
                sc_12 = _fmax_234;
                float _min_203 = fminf(sc_12, 1.7014118346046923e+38f);
                sc_12 = _min_203;
                sc_12 = sc_12;
                float sc_15_1 = sc_12;
                unsigned int u_16 = __as_u32(sc_15_1);
                unsigned int cls_17 = u_16 & 4294965248u;
                unsigned int key_19 = 0;
                if (qf < __uint_as_float(cls_17) && cls_17 < 4278190080u) {
                    key_19 = 1073741824 | (unsigned int)(t0_4 + 1);
                }
                if (cls_17 == qc) {
                    unsigned int lowb_1 = (u_16 ^ (unsigned int)((int)u_16 >> 31) & 2047) & 2047;
                    key_19 = 536870912 | lowb_1 << 11 | (unsigned int)(t0_4 + 1);
                }
                kb2o[1] = __uint_as_float(key_19);
                float sc_21_1 = __uint_as_float(cb[2]);
                float _fmax_235 = fmaxf(sc_21_1, -1.7014118346046923e+38f);
                sc_21_1 = _fmax_235;
                float _min_204 = fminf(sc_21_1, 1.7014118346046923e+38f);
                sc_21_1 = _min_204;
                sc_21_1 = sc_21_1;
                float sc_24 = sc_21_1;
                unsigned int u_25 = __as_u32(sc_24);
                unsigned int cls_26 = u_25 & 4294965248u;
                unsigned int key_28 = 0;
                if (qf < __uint_as_float(cls_26) && cls_26 < 4278190080u) {
                    key_28 = 1073741824 | (unsigned int)(t0_4 + 2);
                }
                if (cls_26 == qc) {
                    unsigned int lowb_2 = (u_25 ^ (unsigned int)((int)u_25 >> 31) & 2047) & 2047;
                    key_28 = 536870912 | lowb_2 << 11 | (unsigned int)(t0_4 + 2);
                }
                kb2o[2] = __uint_as_float(key_28);
                float sc_30 = __uint_as_float(cb[3]);
                float _fmax_236 = fmaxf(sc_30, -1.7014118346046923e+38f);
                sc_30 = _fmax_236;
                float _min_205 = fminf(sc_30, 1.7014118346046923e+38f);
                sc_30 = _min_205;
                sc_30 = sc_30;
                float sc_33_1 = sc_30;
                unsigned int u_34 = __as_u32(sc_33_1);
                unsigned int cls_35 = u_34 & 4294965248u;
                unsigned int key_37 = 0;
                if (qf < __uint_as_float(cls_35) && cls_35 < 4278190080u) {
                    key_37 = 1073741824 | (unsigned int)(t0_4 + 3);
                }
                if (cls_35 == qc) {
                    unsigned int lowb_3 = (u_34 ^ (unsigned int)((int)u_34 >> 31) & 2047) & 2047;
                    key_37 = 536870912 | lowb_3 << 11 | (unsigned int)(t0_4 + 3);
                }
                kb2o[3] = __uint_as_float(key_37);
                float sc_39_1 = __uint_as_float(cb[4]);
                float _fmax_237 = fmaxf(sc_39_1, -1.7014118346046923e+38f);
                sc_39_1 = _fmax_237;
                float _min_206 = fminf(sc_39_1, 1.7014118346046923e+38f);
                sc_39_1 = _min_206;
                sc_39_1 = sc_39_1;
                float sc_42 = sc_39_1;
                unsigned int u_43 = __as_u32(sc_42);
                unsigned int cls_44 = u_43 & 4294965248u;
                unsigned int key_46 = 0;
                if (qf < __uint_as_float(cls_44) && cls_44 < 4278190080u) {
                    key_46 = 1073741824 | (unsigned int)(t0_4 + 4);
                }
                if (cls_44 == qc) {
                    unsigned int lowb_4 = (u_43 ^ (unsigned int)((int)u_43 >> 31) & 2047) & 2047;
                    key_46 = 536870912 | lowb_4 << 11 | (unsigned int)(t0_4 + 4);
                }
                kb2o[4] = __uint_as_float(key_46);
                float sc_48 = __uint_as_float(cb[5]);
                float _fmax_238 = fmaxf(sc_48, -1.7014118346046923e+38f);
                sc_48 = _fmax_238;
                float _min_207 = fminf(sc_48, 1.7014118346046923e+38f);
                sc_48 = _min_207;
                sc_48 = sc_48;
                float sc_49 = sc_48;
                unsigned int u_50 = __as_u32(sc_49);
                unsigned int cls_51 = u_50 & 4294965248u;
                unsigned int key_52 = 0;
                if (qf < __uint_as_float(cls_51) && cls_51 < 4278190080u) {
                    key_52 = 1073741824 | (unsigned int)(t0_4 + 5);
                }
                if (cls_51 == qc) {
                    unsigned int lowb_5 = (u_50 ^ (unsigned int)((int)u_50 >> 31) & 2047) & 2047;
                    key_52 = 536870912 | lowb_5 << 11 | (unsigned int)(t0_4 + 5);
                }
                kb2o[5] = __uint_as_float(key_52);
                float sc_53 = __uint_as_float(cb[6]);
                float _fmax_239 = fmaxf(sc_53, -1.7014118346046923e+38f);
                sc_53 = _fmax_239;
                float _min_208 = fminf(sc_53, 1.7014118346046923e+38f);
                sc_53 = _min_208;
                sc_53 = sc_53;
                float sc_54 = sc_53;
                unsigned int u_55 = __as_u32(sc_54);
                unsigned int cls_56 = u_55 & 4294965248u;
                unsigned int key_57 = 0;
                if (qf < __uint_as_float(cls_56) && cls_56 < 4278190080u) {
                    key_57 = 1073741824 | (unsigned int)(t0_4 + 6);
                }
                if (cls_56 == qc) {
                    unsigned int lowb_6 = (u_55 ^ (unsigned int)((int)u_55 >> 31) & 2047) & 2047;
                    key_57 = 536870912 | lowb_6 << 11 | (unsigned int)(t0_4 + 6);
                }
                kb2o[6] = __uint_as_float(key_57);
                float sc_58 = __uint_as_float(cb[7]);
                float _fmax_240 = fmaxf(sc_58, -1.7014118346046923e+38f);
                sc_58 = _fmax_240;
                float _min_209 = fminf(sc_58, 1.7014118346046923e+38f);
                sc_58 = _min_209;
                sc_58 = sc_58;
                float sc_59 = sc_58;
                unsigned int u_60 = __as_u32(sc_59);
                unsigned int cls_61 = u_60 & 4294965248u;
                unsigned int key_62 = 0;
                if (qf < __uint_as_float(cls_61) && cls_61 < 4278190080u) {
                    key_62 = 1073741824 | (unsigned int)(t0_4 + 7);
                }
                if (cls_61 == qc) {
                    unsigned int lowb_7 = (u_60 ^ (unsigned int)((int)u_60 >> 31) & 2047) & 2047;
                    key_62 = 536870912 | lowb_7 << 11 | (unsigned int)(t0_4 + 7);
                }
                kb2o[7] = __uint_as_float(key_62);
                float sc_63 = __uint_as_float(cb[8]);
                float _fmax_241 = fmaxf(sc_63, -1.7014118346046923e+38f);
                sc_63 = _fmax_241;
                float _min_210 = fminf(sc_63, 1.7014118346046923e+38f);
                sc_63 = _min_210;
                sc_63 = sc_63;
                float sc_64 = sc_63;
                unsigned int u_65 = __as_u32(sc_64);
                unsigned int cls_66 = u_65 & 4294965248u;
                unsigned int key_67 = 0;
                if (qf < __uint_as_float(cls_66) && cls_66 < 4278190080u) {
                    key_67 = 1073741824 | (unsigned int)(t0_4 + 8);
                }
                if (cls_66 == qc) {
                    unsigned int lowb_8 = (u_65 ^ (unsigned int)((int)u_65 >> 31) & 2047) & 2047;
                    key_67 = 536870912 | lowb_8 << 11 | (unsigned int)(t0_4 + 8);
                }
                kb2o[8] = __uint_as_float(key_67);
                float sc_68 = __uint_as_float(cb[9]);
                float _fmax_242 = fmaxf(sc_68, -1.7014118346046923e+38f);
                sc_68 = _fmax_242;
                float _min_211 = fminf(sc_68, 1.7014118346046923e+38f);
                sc_68 = _min_211;
                sc_68 = sc_68;
                float sc_69 = sc_68;
                unsigned int u_70 = __as_u32(sc_69);
                unsigned int cls_71 = u_70 & 4294965248u;
                unsigned int key_72 = 0;
                if (qf < __uint_as_float(cls_71) && cls_71 < 4278190080u) {
                    key_72 = 1073741824 | (unsigned int)(t0_4 + 9);
                }
                if (cls_71 == qc) {
                    unsigned int lowb_9 = (u_70 ^ (unsigned int)((int)u_70 >> 31) & 2047) & 2047;
                    key_72 = 536870912 | lowb_9 << 11 | (unsigned int)(t0_4 + 9);
                }
                kb2o[9] = __uint_as_float(key_72);
                float sc_73 = __uint_as_float(cb[10]);
                float _fmax_243 = fmaxf(sc_73, -1.7014118346046923e+38f);
                sc_73 = _fmax_243;
                float _min_212 = fminf(sc_73, 1.7014118346046923e+38f);
                sc_73 = _min_212;
                sc_73 = sc_73;
                float sc_74 = sc_73;
                unsigned int u_75 = __as_u32(sc_74);
                unsigned int cls_76 = u_75 & 4294965248u;
                unsigned int key_77 = 0;
                if (qf < __uint_as_float(cls_76) && cls_76 < 4278190080u) {
                    key_77 = 1073741824 | (unsigned int)(t0_4 + 10);
                }
                if (cls_76 == qc) {
                    unsigned int lowb_10 = (u_75 ^ (unsigned int)((int)u_75 >> 31) & 2047) & 2047;
                    key_77 = 536870912 | lowb_10 << 11 | (unsigned int)(t0_4 + 10);
                }
                kb2o[10] = __uint_as_float(key_77);
                float sc_78 = __uint_as_float(cb[11]);
                float _fmax_244 = fmaxf(sc_78, -1.7014118346046923e+38f);
                sc_78 = _fmax_244;
                float _min_213 = fminf(sc_78, 1.7014118346046923e+38f);
                sc_78 = _min_213;
                sc_78 = sc_78;
                float sc_79 = sc_78;
                unsigned int u_80 = __as_u32(sc_79);
                unsigned int cls_81 = u_80 & 4294965248u;
                unsigned int key_82 = 0;
                if (qf < __uint_as_float(cls_81) && cls_81 < 4278190080u) {
                    key_82 = 1073741824 | (unsigned int)(t0_4 + 11);
                }
                if (cls_81 == qc) {
                    unsigned int lowb_11 = (u_80 ^ (unsigned int)((int)u_80 >> 31) & 2047) & 2047;
                    key_82 = 536870912 | lowb_11 << 11 | (unsigned int)(t0_4 + 11);
                }
                kb2o[11] = __uint_as_float(key_82);
                float sc_83 = __uint_as_float(cb[12]);
                float _fmax_245 = fmaxf(sc_83, -1.7014118346046923e+38f);
                sc_83 = _fmax_245;
                float _min_214 = fminf(sc_83, 1.7014118346046923e+38f);
                sc_83 = _min_214;
                sc_83 = sc_83;
                float sc_84 = sc_83;
                unsigned int u_85 = __as_u32(sc_84);
                unsigned int cls_86 = u_85 & 4294965248u;
                unsigned int key_87 = 0;
                if (qf < __uint_as_float(cls_86) && cls_86 < 4278190080u) {
                    key_87 = 1073741824 | (unsigned int)(t0_4 + 12);
                }
                if (cls_86 == qc) {
                    unsigned int lowb_12 = (u_85 ^ (unsigned int)((int)u_85 >> 31) & 2047) & 2047;
                    key_87 = 536870912 | lowb_12 << 11 | (unsigned int)(t0_4 + 12);
                }
                kb2o[12] = __uint_as_float(key_87);
                float sc_88 = __uint_as_float(cb[13]);
                float _fmax_246 = fmaxf(sc_88, -1.7014118346046923e+38f);
                sc_88 = _fmax_246;
                float _min_215 = fminf(sc_88, 1.7014118346046923e+38f);
                sc_88 = _min_215;
                sc_88 = sc_88;
                float sc_89 = sc_88;
                unsigned int u_90 = __as_u32(sc_89);
                unsigned int cls_91 = u_90 & 4294965248u;
                unsigned int key_92 = 0;
                if (qf < __uint_as_float(cls_91) && cls_91 < 4278190080u) {
                    key_92 = 1073741824 | (unsigned int)(t0_4 + 13);
                }
                if (cls_91 == qc) {
                    unsigned int lowb_13 = (u_90 ^ (unsigned int)((int)u_90 >> 31) & 2047) & 2047;
                    key_92 = 536870912 | lowb_13 << 11 | (unsigned int)(t0_4 + 13);
                }
                kb2o[13] = __uint_as_float(key_92);
                float sc_93 = __uint_as_float(cb[14]);
                float _fmax_247 = fmaxf(sc_93, -1.7014118346046923e+38f);
                sc_93 = _fmax_247;
                float _min_216 = fminf(sc_93, 1.7014118346046923e+38f);
                sc_93 = _min_216;
                sc_93 = sc_93;
                float sc_94 = sc_93;
                unsigned int u_95 = __as_u32(sc_94);
                unsigned int cls_96 = u_95 & 4294965248u;
                unsigned int key_97 = 0;
                if (qf < __uint_as_float(cls_96) && cls_96 < 4278190080u) {
                    key_97 = 1073741824 | (unsigned int)(t0_4 + 14);
                }
                if (cls_96 == qc) {
                    unsigned int lowb_14 = (u_95 ^ (unsigned int)((int)u_95 >> 31) & 2047) & 2047;
                    key_97 = 536870912 | lowb_14 << 11 | (unsigned int)(t0_4 + 14);
                }
                kb2o[14] = __uint_as_float(key_97);
                float sc_98 = __uint_as_float(cb[15]);
                float _fmax_248 = fmaxf(sc_98, -1.7014118346046923e+38f);
                sc_98 = _fmax_248;
                float _min_217 = fminf(sc_98, 1.7014118346046923e+38f);
                sc_98 = _min_217;
                sc_98 = sc_98;
                float sc_99 = sc_98;
                unsigned int u_100 = __as_u32(sc_99);
                unsigned int cls_101 = u_100 & 4294965248u;
                unsigned int key_102 = 0;
                if (qf < __uint_as_float(cls_101) && cls_101 < 4278190080u) {
                    key_102 = 1073741824 | (unsigned int)(t0_4 + 15);
                }
                if (cls_101 == qc) {
                    unsigned int lowb_15 = (u_100 ^ (unsigned int)((int)u_100 >> 31) & 2047) & 2047;
                    key_102 = 536870912 | lowb_15 << 11 | (unsigned int)(t0_4 + 15);
                }
                kb2o[15] = __uint_as_float(key_102);
                float _fmax_249 = fmaxf(kb2o[0], kb2o[13]);
                float hi_104 = _fmax_249;
                float _min_218 = fminf(kb2o[0], kb2o[13]);
                float lo_105 = _min_218;
                kb2o[0] = hi_104;
                kb2o[13] = lo_105;
                float _fmax_250 = fmaxf(kb2o[1], kb2o[12]);
                float hi_106 = _fmax_250;
                float _min_219 = fminf(kb2o[1], kb2o[12]);
                float lo_107 = _min_219;
                kb2o[1] = hi_106;
                kb2o[12] = lo_107;
                float _fmax_251 = fmaxf(kb2o[2], kb2o[15]);
                float hi_108 = _fmax_251;
                float _min_220 = fminf(kb2o[2], kb2o[15]);
                float lo_109 = _min_220;
                kb2o[2] = hi_108;
                kb2o[15] = lo_109;
                float _fmax_252 = fmaxf(kb2o[3], kb2o[14]);
                float hi_110 = _fmax_252;
                float _min_221 = fminf(kb2o[3], kb2o[14]);
                float lo_111 = _min_221;
                kb2o[3] = hi_110;
                kb2o[14] = lo_111;
                float _fmax_253 = fmaxf(kb2o[4], kb2o[8]);
                float hi_112 = _fmax_253;
                float _min_222 = fminf(kb2o[4], kb2o[8]);
                float lo_113 = _min_222;
                kb2o[4] = hi_112;
                kb2o[8] = lo_113;
                float _fmax_254 = fmaxf(kb2o[5], kb2o[6]);
                float hi_114 = _fmax_254;
                float _min_223 = fminf(kb2o[5], kb2o[6]);
                float lo_115 = _min_223;
                kb2o[5] = hi_114;
                kb2o[6] = lo_115;
                float _fmax_255 = fmaxf(kb2o[7], kb2o[11]);
                float hi_116 = _fmax_255;
                float _min_224 = fminf(kb2o[7], kb2o[11]);
                float lo_117 = _min_224;
                kb2o[7] = hi_116;
                kb2o[11] = lo_117;
                float _fmax_256 = fmaxf(kb2o[9], kb2o[10]);
                float hi_118 = _fmax_256;
                float _min_225 = fminf(kb2o[9], kb2o[10]);
                float lo_119 = _min_225;
                kb2o[9] = hi_118;
                kb2o[10] = lo_119;
                float _fmax_257 = fmaxf(kb2o[0], kb2o[5]);
                float hi_120 = _fmax_257;
                float _min_226 = fminf(kb2o[0], kb2o[5]);
                float lo_121 = _min_226;
                kb2o[0] = hi_120;
                kb2o[5] = lo_121;
                float _fmax_258 = fmaxf(kb2o[1], kb2o[7]);
                float hi_122 = _fmax_258;
                float _min_227 = fminf(kb2o[1], kb2o[7]);
                float lo_123 = _min_227;
                kb2o[1] = hi_122;
                kb2o[7] = lo_123;
                float _fmax_259 = fmaxf(kb2o[2], kb2o[9]);
                float hi_124 = _fmax_259;
                float _min_228 = fminf(kb2o[2], kb2o[9]);
                float lo_125 = _min_228;
                kb2o[2] = hi_124;
                kb2o[9] = lo_125;
                float _fmax_260 = fmaxf(kb2o[3], kb2o[4]);
                float hi_126 = _fmax_260;
                float _min_229 = fminf(kb2o[3], kb2o[4]);
                float lo_127 = _min_229;
                kb2o[3] = hi_126;
                kb2o[4] = lo_127;
                float _fmax_261 = fmaxf(kb2o[6], kb2o[13]);
                float hi_128 = _fmax_261;
                float _min_230 = fminf(kb2o[6], kb2o[13]);
                float lo_129 = _min_230;
                kb2o[6] = hi_128;
                kb2o[13] = lo_129;
                float _fmax_262 = fmaxf(kb2o[8], kb2o[14]);
                float hi_130 = _fmax_262;
                float _min_231 = fminf(kb2o[8], kb2o[14]);
                float lo_131 = _min_231;
                kb2o[8] = hi_130;
                kb2o[14] = lo_131;
                float _fmax_263 = fmaxf(kb2o[10], kb2o[15]);
                float hi_132 = _fmax_263;
                float _min_232 = fminf(kb2o[10], kb2o[15]);
                float lo_133 = _min_232;
                kb2o[10] = hi_132;
                kb2o[15] = lo_133;
                float _fmax_264 = fmaxf(kb2o[11], kb2o[12]);
                float hi_134 = _fmax_264;
                float _min_233 = fminf(kb2o[11], kb2o[12]);
                float lo_135 = _min_233;
                kb2o[11] = hi_134;
                kb2o[12] = lo_135;
                float _fmax_265 = fmaxf(kb2o[0], kb2o[1]);
                float hi_136 = _fmax_265;
                float _min_234 = fminf(kb2o[0], kb2o[1]);
                float lo_137 = _min_234;
                kb2o[0] = hi_136;
                kb2o[1] = lo_137;
                float _fmax_266 = fmaxf(kb2o[2], kb2o[3]);
                float hi_138 = _fmax_266;
                float _min_235 = fminf(kb2o[2], kb2o[3]);
                float lo_139 = _min_235;
                kb2o[2] = hi_138;
                kb2o[3] = lo_139;
                float _fmax_267 = fmaxf(kb2o[4], kb2o[5]);
                float hi_140 = _fmax_267;
                float _min_236 = fminf(kb2o[4], kb2o[5]);
                float lo_141 = _min_236;
                kb2o[4] = hi_140;
                kb2o[5] = lo_141;
                float _fmax_268 = fmaxf(kb2o[6], kb2o[8]);
                float hi_142 = _fmax_268;
                float _min_237 = fminf(kb2o[6], kb2o[8]);
                float lo_143 = _min_237;
                kb2o[6] = hi_142;
                kb2o[8] = lo_143;
                float _fmax_269 = fmaxf(kb2o[7], kb2o[9]);
                float hi_144 = _fmax_269;
                float _min_238 = fminf(kb2o[7], kb2o[9]);
                float lo_145 = _min_238;
                kb2o[7] = hi_144;
                kb2o[9] = lo_145;
                float _fmax_270 = fmaxf(kb2o[10], kb2o[11]);
                float hi_146 = _fmax_270;
                float _min_239 = fminf(kb2o[10], kb2o[11]);
                float lo_147 = _min_239;
                kb2o[10] = hi_146;
                kb2o[11] = lo_147;
                float _fmax_271 = fmaxf(kb2o[12], kb2o[13]);
                float hi_148 = _fmax_271;
                float _min_240 = fminf(kb2o[12], kb2o[13]);
                float lo_149 = _min_240;
                kb2o[12] = hi_148;
                kb2o[13] = lo_149;
                float _fmax_272 = fmaxf(kb2o[14], kb2o[15]);
                float hi_150 = _fmax_272;
                float _min_241 = fminf(kb2o[14], kb2o[15]);
                float lo_151 = _min_241;
                kb2o[14] = hi_150;
                kb2o[15] = lo_151;
                float _fmax_273 = fmaxf(kb2o[0], kb2o[2]);
                float hi_152 = _fmax_273;
                float _min_242 = fminf(kb2o[0], kb2o[2]);
                float lo_153 = _min_242;
                kb2o[0] = hi_152;
                kb2o[2] = lo_153;
                float _fmax_274 = fmaxf(kb2o[1], kb2o[3]);
                float hi_154 = _fmax_274;
                float _min_243 = fminf(kb2o[1], kb2o[3]);
                float lo_155 = _min_243;
                kb2o[1] = hi_154;
                kb2o[3] = lo_155;
                float _fmax_275 = fmaxf(kb2o[4], kb2o[10]);
                float hi_156 = _fmax_275;
                float _min_244 = fminf(kb2o[4], kb2o[10]);
                float lo_157 = _min_244;
                kb2o[4] = hi_156;
                kb2o[10] = lo_157;
                float _fmax_276 = fmaxf(kb2o[5], kb2o[11]);
                float hi_158 = _fmax_276;
                float _min_245 = fminf(kb2o[5], kb2o[11]);
                float lo_159 = _min_245;
                kb2o[5] = hi_158;
                kb2o[11] = lo_159;
                float _fmax_277 = fmaxf(kb2o[6], kb2o[7]);
                float hi_160 = _fmax_277;
                float _min_246 = fminf(kb2o[6], kb2o[7]);
                float lo_161 = _min_246;
                kb2o[6] = hi_160;
                kb2o[7] = lo_161;
                float _fmax_278 = fmaxf(kb2o[8], kb2o[9]);
                float hi_162 = _fmax_278;
                float _min_247 = fminf(kb2o[8], kb2o[9]);
                float lo_163 = _min_247;
                kb2o[8] = hi_162;
                kb2o[9] = lo_163;
                float _fmax_279 = fmaxf(kb2o[12], kb2o[14]);
                float hi_164 = _fmax_279;
                float _min_248 = fminf(kb2o[12], kb2o[14]);
                float lo_165 = _min_248;
                kb2o[12] = hi_164;
                kb2o[14] = lo_165;
                float _fmax_280 = fmaxf(kb2o[13], kb2o[15]);
                float hi_166 = _fmax_280;
                float _min_249 = fminf(kb2o[13], kb2o[15]);
                float lo_167 = _min_249;
                kb2o[13] = hi_166;
                kb2o[15] = lo_167;
                float _fmax_281 = fmaxf(kb2o[1], kb2o[2]);
                float hi_168 = _fmax_281;
                float _min_250 = fminf(kb2o[1], kb2o[2]);
                float lo_169 = _min_250;
                kb2o[1] = hi_168;
                kb2o[2] = lo_169;
                float _fmax_282 = fmaxf(kb2o[3], kb2o[12]);
                float hi_171 = _fmax_282;
                float _min_251 = fminf(kb2o[3], kb2o[12]);
                float lo_172 = _min_251;
                kb2o[3] = hi_171;
                kb2o[12] = lo_172;
                float _fmax_283 = fmaxf(kb2o[4], kb2o[6]);
                float hi_174 = _fmax_283;
                float _min_252 = fminf(kb2o[4], kb2o[6]);
                float lo_175 = _min_252;
                kb2o[4] = hi_174;
                kb2o[6] = lo_175;
                float _fmax_284 = fmaxf(kb2o[5], kb2o[7]);
                float hi_177 = _fmax_284;
                float _min_253 = fminf(kb2o[5], kb2o[7]);
                float lo_178 = _min_253;
                kb2o[5] = hi_177;
                kb2o[7] = lo_178;
                float _fmax_285 = fmaxf(kb2o[8], kb2o[10]);
                float hi_179 = _fmax_285;
                float _min_254 = fminf(kb2o[8], kb2o[10]);
                float lo_180 = _min_254;
                kb2o[8] = hi_179;
                kb2o[10] = lo_180;
                float _fmax_286 = fmaxf(kb2o[9], kb2o[11]);
                float hi_181 = _fmax_286;
                float _min_255 = fminf(kb2o[9], kb2o[11]);
                float lo_182 = _min_255;
                kb2o[9] = hi_181;
                kb2o[11] = lo_182;
                float _fmax_287 = fmaxf(kb2o[13], kb2o[14]);
                float hi_183 = _fmax_287;
                float _min_256 = fminf(kb2o[13], kb2o[14]);
                float lo_184 = _min_256;
                kb2o[13] = hi_183;
                kb2o[14] = lo_184;
                float _fmax_288 = fmaxf(kb2o[1], kb2o[4]);
                float hi_185 = _fmax_288;
                float _min_257 = fminf(kb2o[1], kb2o[4]);
                float lo_186 = _min_257;
                kb2o[1] = hi_185;
                kb2o[4] = lo_186;
                float _fmax_289 = fmaxf(kb2o[2], kb2o[6]);
                float hi_187 = _fmax_289;
                float _min_258 = fminf(kb2o[2], kb2o[6]);
                float lo_188 = _min_258;
                kb2o[2] = hi_187;
                kb2o[6] = lo_188;
                float _fmax_290 = fmaxf(kb2o[5], kb2o[8]);
                float hi_190 = _fmax_290;
                float _min_259 = fminf(kb2o[5], kb2o[8]);
                float lo_191 = _min_259;
                kb2o[5] = hi_190;
                kb2o[8] = lo_191;
                float _fmax_291 = fmaxf(kb2o[7], kb2o[10]);
                float hi_193 = _fmax_291;
                float _min_260 = fminf(kb2o[7], kb2o[10]);
                float lo_194 = _min_260;
                kb2o[7] = hi_193;
                kb2o[10] = lo_194;
                float _fmax_292 = fmaxf(kb2o[9], kb2o[13]);
                float hi_196 = _fmax_292;
                float _min_261 = fminf(kb2o[9], kb2o[13]);
                float lo_197 = _min_261;
                kb2o[9] = hi_196;
                kb2o[13] = lo_197;
                float _fmax_293 = fmaxf(kb2o[11], kb2o[14]);
                float hi_198 = _fmax_293;
                float _min_262 = fminf(kb2o[11], kb2o[14]);
                float lo_199 = _min_262;
                kb2o[11] = hi_198;
                kb2o[14] = lo_199;
                float _fmax_294 = fmaxf(kb2o[2], kb2o[4]);
                float hi_200 = _fmax_294;
                float _min_263 = fminf(kb2o[2], kb2o[4]);
                float lo_201 = _min_263;
                kb2o[2] = hi_200;
                kb2o[4] = lo_201;
                float _fmax_295 = fmaxf(kb2o[3], kb2o[6]);
                float hi_202 = _fmax_295;
                float _min_264 = fminf(kb2o[3], kb2o[6]);
                float lo_203 = _min_264;
                kb2o[3] = hi_202;
                kb2o[6] = lo_203;
                float _fmax_296 = fmaxf(kb2o[9], kb2o[12]);
                float hi_204 = _fmax_296;
                float _min_265 = fminf(kb2o[9], kb2o[12]);
                float lo_205 = _min_265;
                kb2o[9] = hi_204;
                kb2o[12] = lo_205;
                float _fmax_297 = fmaxf(kb2o[11], kb2o[13]);
                float hi_206 = _fmax_297;
                float _min_266 = fminf(kb2o[11], kb2o[13]);
                float lo_207 = _min_266;
                kb2o[11] = hi_206;
                kb2o[13] = lo_207;
                float _fmax_298 = fmaxf(kb2o[3], kb2o[5]);
                float hi_209 = _fmax_298;
                float _min_267 = fminf(kb2o[3], kb2o[5]);
                float lo_210 = _min_267;
                kb2o[3] = hi_209;
                kb2o[5] = lo_210;
                float _fmax_299 = fmaxf(kb2o[6], kb2o[8]);
                float hi_212 = _fmax_299;
                float _min_268 = fminf(kb2o[6], kb2o[8]);
                float lo_213 = _min_268;
                kb2o[6] = hi_212;
                kb2o[8] = lo_213;
                float _fmax_300 = fmaxf(kb2o[7], kb2o[9]);
                float hi_215 = _fmax_300;
                float _min_269 = fminf(kb2o[7], kb2o[9]);
                float lo_216 = _min_269;
                kb2o[7] = hi_215;
                kb2o[9] = lo_216;
                float _fmax_301 = fmaxf(kb2o[10], kb2o[12]);
                float hi_217 = _fmax_301;
                float _min_270 = fminf(kb2o[10], kb2o[12]);
                float lo_218 = _min_270;
                kb2o[10] = hi_217;
                kb2o[12] = lo_218;
                float _fmax_302 = fmaxf(kb2o[3], kb2o[4]);
                float hi_219 = _fmax_302;
                float _min_271 = fminf(kb2o[3], kb2o[4]);
                float lo_220 = _min_271;
                kb2o[3] = hi_219;
                kb2o[4] = lo_220;
                float _fmax_303 = fmaxf(kb2o[5], kb2o[6]);
                float hi_221 = _fmax_303;
                float _min_272 = fminf(kb2o[5], kb2o[6]);
                float lo_222 = _min_272;
                kb2o[5] = hi_221;
                kb2o[6] = lo_222;
                float _fmax_304 = fmaxf(kb2o[7], kb2o[8]);
                float hi_223 = _fmax_304;
                float _min_273 = fminf(kb2o[7], kb2o[8]);
                float lo_224 = _min_273;
                kb2o[7] = hi_223;
                kb2o[8] = lo_224;
                float _fmax_305 = fmaxf(kb2o[9], kb2o[10]);
                float hi_225 = _fmax_305;
                float _min_274 = fminf(kb2o[9], kb2o[10]);
                float lo_226 = _min_274;
                kb2o[9] = hi_225;
                kb2o[10] = lo_226;
                float _fmax_306 = fmaxf(kb2o[11], kb2o[12]);
                float hi_228 = _fmax_306;
                float _min_275 = fminf(kb2o[11], kb2o[12]);
                float lo_229 = _min_275;
                kb2o[11] = hi_228;
                kb2o[12] = lo_229;
                float _fmax_307 = fmaxf(kb2o[6], kb2o[7]);
                float hi_231 = _fmax_307;
                float _min_276 = fminf(kb2o[6], kb2o[7]);
                float lo_232 = _min_276;
                kb2o[6] = hi_231;
                kb2o[7] = lo_232;
                float _fmax_308 = fmaxf(kb2o[8], kb2o[9]);
                float hi_234 = _fmax_308;
                float _min_277 = fminf(kb2o[8], kb2o[9]);
                float lo_235 = _min_277;
                kb2o[8] = hi_234;
                kb2o[9] = lo_235;
                a2x[0] = kb2o[0];
                a2x[1] = kb2o[1];
                a2x[2] = kb2o[2];
                a2x[3] = kb2o[3];
                a2x[4] = kb2o[4];
                a2x[5] = kb2o[5];
                a2x[6] = kb2o[6];
                a2x[7] = kb2o[7];
                a2x[8] = kb2o[8];
                a2x[9] = kb2o[9];
                a2x[10] = kb2o[10];
                a2x[11] = kb2o[11];
                a2x[12] = kb2o[12];
                a2x[13] = kb2o[13];
                a2x[14] = kb2o[14];
                a2x[15] = kb2o[15];
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
            int s0_3 = (sg * 16 * 2 + cg) * 17;
            int s1_4 = ((sg * 16 + 8) * 2 + cg) * 17;
            float x0_5 = pub[s0_3 + ln];
            float y0_6 = pub[s1_4 + lnr];
            float _min_278 = fminf(x0_5, y0_6);
            float lo0_7 = _min_278;
            float _fmax_309 = fmaxf(r_1, lo0_7);
            r_1 = _fmax_309;
            float _fmax_310 = fmaxf(x0_5, y0_6);
            float hi0_8 = _fmax_310;
            float cur_9 = hi0_8;
            float _shfl_xor_102 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 8);
            float pv_10_1 = _shfl_xor_102;
            float _fmax_311 = fmaxf(cur_9, pv_10_1);
            float hi_11_1 = _fmax_311;
            float _min_279 = fminf(cur_9, pv_10_1);
            float lo_12_1 = _min_279;
            cur_9 = ((up[0] != 0) ? hi_11_1 : lo_12_1);
            float _shfl_xor_103 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 4);
            float pv_13 = _shfl_xor_103;
            float _fmax_312 = fmaxf(cur_9, pv_13);
            float hi_14 = _fmax_312;
            float _min_280 = fminf(cur_9, pv_13);
            float lo_15 = _min_280;
            cur_9 = ((up[1] != 0) ? hi_14 : lo_15);
            float _shfl_xor_104 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 2);
            float pv_16 = _shfl_xor_104;
            float _fmax_313 = fmaxf(cur_9, pv_16);
            float hi_17 = _fmax_313;
            float _min_281 = fminf(cur_9, pv_16);
            float lo_18 = _min_281;
            cur_9 = ((up[2] != 0) ? hi_17 : lo_18);
            float _shfl_xor_105 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 1);
            float pv_19 = _shfl_xor_105;
            float _fmax_314 = fmaxf(cur_9, pv_19);
            float hi_20 = _fmax_314;
            float _min_282 = fminf(cur_9, pv_19);
            float lo_21 = _min_282;
            cur_9 = ((up[3] != 0) ? hi_20 : lo_21);
            V_2[0] = cur_9;
            int s0_22 = ((sg * 16 + 1) * 2 + cg) * 17;
            int s1_23 = ((sg * 16 + 1 + 8) * 2 + cg) * 17;
            float x0_24 = pub[s0_22 + ln];
            float y0_25 = pub[s1_23 + lnr];
            float _min_283 = fminf(x0_24, y0_25);
            float lo0_26 = _min_283;
            float _fmax_315 = fmaxf(r_1, lo0_26);
            r_1 = _fmax_315;
            float _fmax_316 = fmaxf(x0_24, y0_25);
            float hi0_27 = _fmax_316;
            float cur_28 = hi0_27;
            float _shfl_xor_106 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 8);
            float pv_29 = _shfl_xor_106;
            float _fmax_317 = fmaxf(cur_28, pv_29);
            float hi_30 = _fmax_317;
            float _min_284 = fminf(cur_28, pv_29);
            float lo_31 = _min_284;
            cur_28 = ((up[0] != 0) ? hi_30 : lo_31);
            float _shfl_xor_107 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 4);
            float pv_32 = _shfl_xor_107;
            float _fmax_318 = fmaxf(cur_28, pv_32);
            float hi_33 = _fmax_318;
            float _min_285 = fminf(cur_28, pv_32);
            float lo_34 = _min_285;
            cur_28 = ((up[1] != 0) ? hi_33 : lo_34);
            float _shfl_xor_108 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 2);
            float pv_35_1 = _shfl_xor_108;
            float _fmax_319 = fmaxf(cur_28, pv_35_1);
            float hi_36_1 = _fmax_319;
            float _min_286 = fminf(cur_28, pv_35_1);
            float lo_37_1 = _min_286;
            cur_28 = ((up[2] != 0) ? hi_36_1 : lo_37_1);
            float _shfl_xor_109 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 1);
            float pv_38_1 = _shfl_xor_109;
            float _fmax_320 = fmaxf(cur_28, pv_38_1);
            float hi_39_1 = _fmax_320;
            float _min_287 = fminf(cur_28, pv_38_1);
            float lo_40_1 = _min_287;
            cur_28 = ((up[3] != 0) ? hi_39_1 : lo_40_1);
            V_2[1] = cur_28;
            int s0_41 = ((sg * 16 + 2) * 2 + cg) * 17;
            int s1_42 = ((sg * 16 + 2 + 8) * 2 + cg) * 17;
            float x0_43 = pub[s0_41 + ln];
            float y0_44 = pub[s1_42 + lnr];
            float _min_288 = fminf(x0_43, y0_44);
            float lo0_45 = _min_288;
            float _fmax_321 = fmaxf(r_1, lo0_45);
            r_1 = _fmax_321;
            float _fmax_322 = fmaxf(x0_43, y0_44);
            float hi0_46 = _fmax_322;
            float cur_47 = hi0_46;
            float _shfl_xor_110 = __shfl_xor_sync(0xFFFFFFFF, cur_47, 8);
            float pv_48 = _shfl_xor_110;
            float _fmax_323 = fmaxf(cur_47, pv_48);
            float hi_50 = _fmax_323;
            float _min_289 = fminf(cur_47, pv_48);
            float lo_51 = _min_289;
            cur_47 = ((up[0] != 0) ? hi_50 : lo_51);
            float _shfl_xor_111 = __shfl_xor_sync(0xFFFFFFFF, cur_47, 4);
            float pv_52_1 = _shfl_xor_111;
            float _fmax_324 = fmaxf(cur_47, pv_52_1);
            float hi_54_1 = _fmax_324;
            float _min_290 = fminf(cur_47, pv_52_1);
            float lo_55_1 = _min_290;
            cur_47 = ((up[1] != 0) ? hi_54_1 : lo_55_1);
            float _shfl_xor_112 = __shfl_xor_sync(0xFFFFFFFF, cur_47, 2);
            float pv_56_1 = _shfl_xor_112;
            float _fmax_325 = fmaxf(cur_47, pv_56_1);
            float hi_58_1 = _fmax_325;
            float _min_291 = fminf(cur_47, pv_56_1);
            float lo_59_1 = _min_291;
            cur_47 = ((up[2] != 0) ? hi_58_1 : lo_59_1);
            float _shfl_xor_113 = __shfl_xor_sync(0xFFFFFFFF, cur_47, 1);
            float pv_60_1 = _shfl_xor_113;
            float _fmax_326 = fmaxf(cur_47, pv_60_1);
            float hi_62_1 = _fmax_326;
            float _min_292 = fminf(cur_47, pv_60_1);
            float lo_63_1 = _min_292;
            cur_47 = ((up[3] != 0) ? hi_62_1 : lo_63_1);
            V_2[2] = cur_47;
            int s0_64 = ((sg * 16 + 3) * 2 + cg) * 17;
            int s1_65 = ((sg * 16 + 3 + 8) * 2 + cg) * 17;
            float x0_66 = pub[s0_64 + ln];
            float y0_67 = pub[s1_65 + lnr];
            float _min_293 = fminf(x0_66, y0_67);
            float lo0_68 = _min_293;
            float _fmax_327 = fmaxf(r_1, lo0_68);
            r_1 = _fmax_327;
            float _fmax_328 = fmaxf(x0_66, y0_67);
            float hi0_69 = _fmax_328;
            float cur_70 = hi0_69;
            float _shfl_xor_114 = __shfl_xor_sync(0xFFFFFFFF, cur_70, 8);
            float pv_71 = _shfl_xor_114;
            float _fmax_329 = fmaxf(cur_70, pv_71);
            float hi_72 = _fmax_329;
            float _min_294 = fminf(cur_70, pv_71);
            float lo_73 = _min_294;
            cur_70 = ((up[0] != 0) ? hi_72 : lo_73);
            float _shfl_xor_115 = __shfl_xor_sync(0xFFFFFFFF, cur_70, 4);
            float pv_74 = _shfl_xor_115;
            float _fmax_330 = fmaxf(cur_70, pv_74);
            float hi_76 = _fmax_330;
            float _min_295 = fminf(cur_70, pv_74);
            float lo_77 = _min_295;
            cur_70 = ((up[1] != 0) ? hi_76 : lo_77);
            float _shfl_xor_116 = __shfl_xor_sync(0xFFFFFFFF, cur_70, 2);
            float pv_78 = _shfl_xor_116;
            float _fmax_331 = fmaxf(cur_70, pv_78);
            float hi_80 = _fmax_331;
            float _min_296 = fminf(cur_70, pv_78);
            float lo_81 = _min_296;
            cur_70 = ((up[2] != 0) ? hi_80 : lo_81);
            float _shfl_xor_117 = __shfl_xor_sync(0xFFFFFFFF, cur_70, 1);
            float pv_82 = _shfl_xor_117;
            float _fmax_332 = fmaxf(cur_70, pv_82);
            float hi_84 = _fmax_332;
            float _min_297 = fminf(cur_70, pv_82);
            float lo_85 = _min_297;
            cur_70 = ((up[3] != 0) ? hi_84 : lo_85);
            V_2[3] = cur_70;
            int s0_86 = ((sg * 16 + 4) * 2 + cg) * 17;
            int s1_87 = ((sg * 16 + 4 + 8) * 2 + cg) * 17;
            float x0_88 = pub[s0_86 + ln];
            float y0_89 = pub[s1_87 + lnr];
            float _min_298 = fminf(x0_88, y0_89);
            float lo0_90 = _min_298;
            float _fmax_333 = fmaxf(r_1, lo0_90);
            r_1 = _fmax_333;
            float _fmax_334 = fmaxf(x0_88, y0_89);
            float hi0_91 = _fmax_334;
            float cur_92 = hi0_91;
            float _shfl_xor_118 = __shfl_xor_sync(0xFFFFFFFF, cur_92, 8);
            float pv_93 = _shfl_xor_118;
            float _fmax_335 = fmaxf(cur_92, pv_93);
            float hi_94_1 = _fmax_335;
            float _min_299 = fminf(cur_92, pv_93);
            float lo_95_1 = _min_299;
            cur_92 = ((up[0] != 0) ? hi_94_1 : lo_95_1);
            float _shfl_xor_119 = __shfl_xor_sync(0xFFFFFFFF, cur_92, 4);
            float pv_96_1 = _shfl_xor_119;
            float _fmax_336 = fmaxf(cur_92, pv_96_1);
            float hi_98_1 = _fmax_336;
            float _min_300 = fminf(cur_92, pv_96_1);
            float lo_99_1 = _min_300;
            cur_92 = ((up[1] != 0) ? hi_98_1 : lo_99_1);
            float _shfl_xor_120 = __shfl_xor_sync(0xFFFFFFFF, cur_92, 2);
            float pv_100_1 = _shfl_xor_120;
            float _fmax_337 = fmaxf(cur_92, pv_100_1);
            float hi_102_1 = _fmax_337;
            float _min_301 = fminf(cur_92, pv_100_1);
            float lo_103_1 = _min_301;
            cur_92 = ((up[2] != 0) ? hi_102_1 : lo_103_1);
            float _shfl_xor_121 = __shfl_xor_sync(0xFFFFFFFF, cur_92, 1);
            float pv_104 = _shfl_xor_121;
            float _fmax_338 = fmaxf(cur_92, pv_104);
            float hi_106_1 = _fmax_338;
            float _min_302 = fminf(cur_92, pv_104);
            float lo_107_1 = _min_302;
            cur_92 = ((up[3] != 0) ? hi_106_1 : lo_107_1);
            V_2[4] = cur_92;
            int s0_108 = ((sg * 16 + 5) * 2 + cg) * 17;
            int s1_109 = ((sg * 16 + 5 + 8) * 2 + cg) * 17;
            float x0_110 = pub[s0_108 + ln];
            float y0_111 = pub[s1_109 + lnr];
            float _min_303 = fminf(x0_110, y0_111);
            float lo0_112 = _min_303;
            float _fmax_339 = fmaxf(r_1, lo0_112);
            r_1 = _fmax_339;
            float _fmax_340 = fmaxf(x0_110, y0_111);
            float hi0_113 = _fmax_340;
            float cur_114 = hi0_113;
            float _shfl_xor_122 = __shfl_xor_sync(0xFFFFFFFF, cur_114, 8);
            float pv_115 = _shfl_xor_122;
            float _fmax_341 = fmaxf(cur_114, pv_115);
            float hi_116_1 = _fmax_341;
            float _min_304 = fminf(cur_114, pv_115);
            float lo_117_1 = _min_304;
            cur_114 = ((up[0] != 0) ? hi_116_1 : lo_117_1);
            float _shfl_xor_123 = __shfl_xor_sync(0xFFFFFFFF, cur_114, 4);
            float pv_118 = _shfl_xor_123;
            float _fmax_342 = fmaxf(cur_114, pv_118);
            float hi_120_1 = _fmax_342;
            float _min_305 = fminf(cur_114, pv_118);
            float lo_121_1 = _min_305;
            cur_114 = ((up[1] != 0) ? hi_120_1 : lo_121_1);
            float _shfl_xor_124 = __shfl_xor_sync(0xFFFFFFFF, cur_114, 2);
            float pv_122 = _shfl_xor_124;
            float _fmax_343 = fmaxf(cur_114, pv_122);
            float hi_124_1 = _fmax_343;
            float _min_306 = fminf(cur_114, pv_122);
            float lo_125_1 = _min_306;
            cur_114 = ((up[2] != 0) ? hi_124_1 : lo_125_1);
            float _shfl_xor_125 = __shfl_xor_sync(0xFFFFFFFF, cur_114, 1);
            float pv_126 = _shfl_xor_125;
            float _fmax_344 = fmaxf(cur_114, pv_126);
            float hi_128_1 = _fmax_344;
            float _min_307 = fminf(cur_114, pv_126);
            float lo_129_1 = _min_307;
            cur_114 = ((up[3] != 0) ? hi_128_1 : lo_129_1);
            V_2[5] = cur_114;
            int s0_130 = ((sg * 16 + 6) * 2 + cg) * 17;
            int s1_131 = ((sg * 16 + 6 + 8) * 2 + cg) * 17;
            float x0_132 = pub[s0_130 + ln];
            float y0_133 = pub[s1_131 + lnr];
            float _min_308 = fminf(x0_132, y0_133);
            float lo0_134 = _min_308;
            float _fmax_345 = fmaxf(r_1, lo0_134);
            r_1 = _fmax_345;
            float _fmax_346 = fmaxf(x0_132, y0_133);
            float hi0_135 = _fmax_346;
            float cur_136 = hi0_135;
            float _shfl_xor_126 = __shfl_xor_sync(0xFFFFFFFF, cur_136, 8);
            float pv_137 = _shfl_xor_126;
            float _fmax_347 = fmaxf(cur_136, pv_137);
            float hi_138_1 = _fmax_347;
            float _min_309 = fminf(cur_136, pv_137);
            float lo_139_1 = _min_309;
            cur_136 = ((up[0] != 0) ? hi_138_1 : lo_139_1);
            float _shfl_xor_127 = __shfl_xor_sync(0xFFFFFFFF, cur_136, 4);
            float pv_140 = _shfl_xor_127;
            float _fmax_348 = fmaxf(cur_136, pv_140);
            float hi_142_1 = _fmax_348;
            float _min_310 = fminf(cur_136, pv_140);
            float lo_143_1 = _min_310;
            cur_136 = ((up[1] != 0) ? hi_142_1 : lo_143_1);
            float _shfl_xor_128 = __shfl_xor_sync(0xFFFFFFFF, cur_136, 2);
            float pv_144 = _shfl_xor_128;
            float _fmax_349 = fmaxf(cur_136, pv_144);
            float hi_146_1 = _fmax_349;
            float _min_311 = fminf(cur_136, pv_144);
            float lo_147_1 = _min_311;
            cur_136 = ((up[2] != 0) ? hi_146_1 : lo_147_1);
            float _shfl_xor_129 = __shfl_xor_sync(0xFFFFFFFF, cur_136, 1);
            float pv_148 = _shfl_xor_129;
            float _fmax_350 = fmaxf(cur_136, pv_148);
            float hi_150_1 = _fmax_350;
            float _min_312 = fminf(cur_136, pv_148);
            float lo_151_1 = _min_312;
            cur_136 = ((up[3] != 0) ? hi_150_1 : lo_151_1);
            V_2[6] = cur_136;
            int s0_152 = ((sg * 16 + 7) * 2 + cg) * 17;
            int s1_153 = ((sg * 16 + 7 + 8) * 2 + cg) * 17;
            float x0_154 = pub[s0_152 + ln];
            float y0_155 = pub[s1_153 + lnr];
            float _min_313 = fminf(x0_154, y0_155);
            float lo0_156 = _min_313;
            float _fmax_351 = fmaxf(r_1, lo0_156);
            r_1 = _fmax_351;
            float _fmax_352 = fmaxf(x0_154, y0_155);
            float hi0_157 = _fmax_352;
            float cur_158 = hi0_157;
            float _shfl_xor_130 = __shfl_xor_sync(0xFFFFFFFF, cur_158, 8);
            float pv_159 = _shfl_xor_130;
            float _fmax_353 = fmaxf(cur_158, pv_159);
            float hi_160_1 = _fmax_353;
            float _min_314 = fminf(cur_158, pv_159);
            float lo_161_1 = _min_314;
            cur_158 = ((up[0] != 0) ? hi_160_1 : lo_161_1);
            float _shfl_xor_131 = __shfl_xor_sync(0xFFFFFFFF, cur_158, 4);
            float pv_162 = _shfl_xor_131;
            float _fmax_354 = fmaxf(cur_158, pv_162);
            float hi_164_1 = _fmax_354;
            float _min_315 = fminf(cur_158, pv_162);
            float lo_165_1 = _min_315;
            cur_158 = ((up[1] != 0) ? hi_164_1 : lo_165_1);
            float _shfl_xor_132 = __shfl_xor_sync(0xFFFFFFFF, cur_158, 2);
            float pv_166 = _shfl_xor_132;
            float _fmax_355 = fmaxf(cur_158, pv_166);
            float hi_168_1 = _fmax_355;
            float _min_316 = fminf(cur_158, pv_166);
            float lo_169_1 = _min_316;
            cur_158 = ((up[2] != 0) ? hi_168_1 : lo_169_1);
            float _shfl_xor_133 = __shfl_xor_sync(0xFFFFFFFF, cur_158, 1);
            float pv_170 = _shfl_xor_133;
            float _fmax_356 = fmaxf(cur_158, pv_170);
            float hi_171_1 = _fmax_356;
            float _min_317 = fminf(cur_158, pv_170);
            float lo_172_1 = _min_317;
            cur_158 = ((up[3] != 0) ? hi_171_1 : lo_172_1);
            V_2[7] = cur_158;
            float rs_173 = pub[((sg * 16 + ln) * 2 + cg) * 17 + 16];
            float _fmax_357 = fmaxf(r_1, rs_173);
            r_1 = _fmax_357;
            float _shfl_xor_134 = __shfl_xor_sync(0xFFFFFFFF, V_2[4], 15);
            float y1_174 = _shfl_xor_134;
            float _min_318 = fminf(V_2[0], y1_174);
            float lo1_175 = _min_318;
            float _fmax_358 = fmaxf(r_1, lo1_175);
            r_1 = _fmax_358;
            float _fmax_359 = fmaxf(V_2[0], y1_174);
            float hi1_176 = _fmax_359;
            float cur_177 = hi1_176;
            float _shfl_xor_135 = __shfl_xor_sync(0xFFFFFFFF, cur_177, 8);
            float pv_178 = _shfl_xor_135;
            float _fmax_360 = fmaxf(cur_177, pv_178);
            float hi_179_1 = _fmax_360;
            float _min_319 = fminf(cur_177, pv_178);
            float lo_180_1 = _min_319;
            cur_177 = ((up[0] != 0) ? hi_179_1 : lo_180_1);
            float _shfl_xor_136 = __shfl_xor_sync(0xFFFFFFFF, cur_177, 4);
            float pv_181 = _shfl_xor_136;
            float _fmax_361 = fmaxf(cur_177, pv_181);
            float hi_182 = _fmax_361;
            float _min_320 = fminf(cur_177, pv_181);
            float lo_183 = _min_320;
            cur_177 = ((up[1] != 0) ? hi_182 : lo_183);
            float _shfl_xor_137 = __shfl_xor_sync(0xFFFFFFFF, cur_177, 2);
            float pv_184 = _shfl_xor_137;
            float _fmax_362 = fmaxf(cur_177, pv_184);
            float hi_185_1 = _fmax_362;
            float _min_321 = fminf(cur_177, pv_184);
            float lo_186_1 = _min_321;
            cur_177 = ((up[2] != 0) ? hi_185_1 : lo_186_1);
            float _shfl_xor_138 = __shfl_xor_sync(0xFFFFFFFF, cur_177, 1);
            float pv_187 = _shfl_xor_138;
            float _fmax_363 = fmaxf(cur_177, pv_187);
            float hi_188 = _fmax_363;
            float _min_322 = fminf(cur_177, pv_187);
            float lo_189 = _min_322;
            cur_177 = ((up[3] != 0) ? hi_188 : lo_189);
            V_2[0] = cur_177;
            float _shfl_xor_139 = __shfl_xor_sync(0xFFFFFFFF, V_2[5], 15);
            float y1_190 = _shfl_xor_139;
            float _min_323 = fminf(V_2[1], y1_190);
            float lo1_191 = _min_323;
            float _fmax_364 = fmaxf(r_1, lo1_191);
            r_1 = _fmax_364;
            float _fmax_365 = fmaxf(V_2[1], y1_190);
            float hi1_192 = _fmax_365;
            float cur_193 = hi1_192;
            float _shfl_xor_140 = __shfl_xor_sync(0xFFFFFFFF, cur_193, 8);
            float pv_195 = _shfl_xor_140;
            float _fmax_366 = fmaxf(cur_193, pv_195);
            float hi_196_1 = _fmax_366;
            float _min_324 = fminf(cur_193, pv_195);
            float lo_197_1 = _min_324;
            cur_193 = ((up[0] != 0) ? hi_196_1 : lo_197_1);
            float _shfl_xor_141 = __shfl_xor_sync(0xFFFFFFFF, cur_193, 4);
            float pv_198 = _shfl_xor_141;
            float _fmax_367 = fmaxf(cur_193, pv_198);
            float hi_199 = _fmax_367;
            float _min_325 = fminf(cur_193, pv_198);
            float lo_200 = _min_325;
            cur_193 = ((up[1] != 0) ? hi_199 : lo_200);
            float _shfl_xor_142 = __shfl_xor_sync(0xFFFFFFFF, cur_193, 2);
            float pv_201 = _shfl_xor_142;
            float _fmax_368 = fmaxf(cur_193, pv_201);
            float hi_202_1 = _fmax_368;
            float _min_326 = fminf(cur_193, pv_201);
            float lo_203_1 = _min_326;
            cur_193 = ((up[2] != 0) ? hi_202_1 : lo_203_1);
            float _shfl_xor_143 = __shfl_xor_sync(0xFFFFFFFF, cur_193, 1);
            float pv_205 = _shfl_xor_143;
            float _fmax_369 = fmaxf(cur_193, pv_205);
            float hi_206_1 = _fmax_369;
            float _min_327 = fminf(cur_193, pv_205);
            float lo_207_1 = _min_327;
            cur_193 = ((up[3] != 0) ? hi_206_1 : lo_207_1);
            V_2[1] = cur_193;
            float _shfl_xor_144 = __shfl_xor_sync(0xFFFFFFFF, V_2[6], 15);
            float y1_208 = _shfl_xor_144;
            float _min_328 = fminf(V_2[2], y1_208);
            float lo1_209 = _min_328;
            float _fmax_370 = fmaxf(r_1, lo1_209);
            r_1 = _fmax_370;
            float _fmax_371 = fmaxf(V_2[2], y1_208);
            float hi1_210 = _fmax_371;
            float cur_211 = hi1_210;
            float _shfl_xor_145 = __shfl_xor_sync(0xFFFFFFFF, cur_211, 8);
            float pv_212 = _shfl_xor_145;
            float _fmax_372 = fmaxf(cur_211, pv_212);
            float hi_213 = _fmax_372;
            float _min_329 = fminf(cur_211, pv_212);
            float lo_214 = _min_329;
            cur_211 = ((up[0] != 0) ? hi_213 : lo_214);
            float _shfl_xor_146 = __shfl_xor_sync(0xFFFFFFFF, cur_211, 4);
            float pv_215 = _shfl_xor_146;
            float _fmax_373 = fmaxf(cur_211, pv_215);
            float hi_216 = _fmax_373;
            float _min_330 = fminf(cur_211, pv_215);
            float lo_217 = _min_330;
            cur_211 = ((up[1] != 0) ? hi_216 : lo_217);
            float _shfl_xor_147 = __shfl_xor_sync(0xFFFFFFFF, cur_211, 2);
            float pv_218 = _shfl_xor_147;
            float _fmax_374 = fmaxf(cur_211, pv_218);
            float hi_219_1 = _fmax_374;
            float _min_331 = fminf(cur_211, pv_218);
            float lo_220_1 = _min_331;
            cur_211 = ((up[2] != 0) ? hi_219_1 : lo_220_1);
            float _shfl_xor_148 = __shfl_xor_sync(0xFFFFFFFF, cur_211, 1);
            float pv_221 = _shfl_xor_148;
            float _fmax_375 = fmaxf(cur_211, pv_221);
            float hi_222 = _fmax_375;
            float _min_332 = fminf(cur_211, pv_221);
            float lo_223 = _min_332;
            cur_211 = ((up[3] != 0) ? hi_222 : lo_223);
            V_2[2] = cur_211;
            float _shfl_xor_149 = __shfl_xor_sync(0xFFFFFFFF, V_2[7], 15);
            float y1_224 = _shfl_xor_149;
            float _min_333 = fminf(V_2[3], y1_224);
            float lo1_225 = _min_333;
            float _fmax_376 = fmaxf(r_1, lo1_225);
            r_1 = _fmax_376;
            float _fmax_377 = fmaxf(V_2[3], y1_224);
            float hi1_226 = _fmax_377;
            float cur_227 = hi1_226;
            float _shfl_xor_150 = __shfl_xor_sync(0xFFFFFFFF, cur_227, 8);
            float pv_228 = _shfl_xor_150;
            float _fmax_378 = fmaxf(cur_227, pv_228);
            float hi_229 = _fmax_378;
            float _min_334 = fminf(cur_227, pv_228);
            float lo_230 = _min_334;
            cur_227 = ((up[0] != 0) ? hi_229 : lo_230);
            float _shfl_xor_151 = __shfl_xor_sync(0xFFFFFFFF, cur_227, 4);
            float pv_231 = _shfl_xor_151;
            float _fmax_379 = fmaxf(cur_227, pv_231);
            float hi_232 = _fmax_379;
            float _min_335 = fminf(cur_227, pv_231);
            float lo_233 = _min_335;
            cur_227 = ((up[1] != 0) ? hi_232 : lo_233);
            float _shfl_xor_152 = __shfl_xor_sync(0xFFFFFFFF, cur_227, 2);
            float pv_234 = _shfl_xor_152;
            float _fmax_380 = fmaxf(cur_227, pv_234);
            float hi_235 = _fmax_380;
            float _min_336 = fminf(cur_227, pv_234);
            float lo_236 = _min_336;
            cur_227 = ((up[2] != 0) ? hi_235 : lo_236);
            float _shfl_xor_153 = __shfl_xor_sync(0xFFFFFFFF, cur_227, 1);
            float pv_237 = _shfl_xor_153;
            float _fmax_381 = fmaxf(cur_227, pv_237);
            float hi_238 = _fmax_381;
            float _min_337 = fminf(cur_227, pv_237);
            float lo_239 = _min_337;
            cur_227 = ((up[3] != 0) ? hi_238 : lo_239);
            V_2[3] = cur_227;
            float _shfl_xor_154 = __shfl_xor_sync(0xFFFFFFFF, V_2[2], 15);
            float y1_240 = _shfl_xor_154;
            float _min_338 = fminf(V_2[0], y1_240);
            float lo1_241 = _min_338;
            float _fmax_382 = fmaxf(r_1, lo1_241);
            r_1 = _fmax_382;
            float _fmax_383 = fmaxf(V_2[0], y1_240);
            float hi1_242 = _fmax_383;
            float cur_243 = hi1_242;
            float _shfl_xor_155 = __shfl_xor_sync(0xFFFFFFFF, cur_243, 8);
            float pv_244 = _shfl_xor_155;
            float _fmax_384 = fmaxf(cur_243, pv_244);
            float hi_245 = _fmax_384;
            float _min_339 = fminf(cur_243, pv_244);
            float lo_246 = _min_339;
            cur_243 = ((up[0] != 0) ? hi_245 : lo_246);
            float _shfl_xor_156 = __shfl_xor_sync(0xFFFFFFFF, cur_243, 4);
            float pv_247 = _shfl_xor_156;
            float _fmax_385 = fmaxf(cur_243, pv_247);
            float hi_248 = _fmax_385;
            float _min_340 = fminf(cur_243, pv_247);
            float lo_249 = _min_340;
            cur_243 = ((up[1] != 0) ? hi_248 : lo_249);
            float _shfl_xor_157 = __shfl_xor_sync(0xFFFFFFFF, cur_243, 2);
            float pv_250 = _shfl_xor_157;
            float _fmax_386 = fmaxf(cur_243, pv_250);
            float hi_251 = _fmax_386;
            float _min_341 = fminf(cur_243, pv_250);
            float lo_252 = _min_341;
            cur_243 = ((up[2] != 0) ? hi_251 : lo_252);
            float _shfl_xor_158 = __shfl_xor_sync(0xFFFFFFFF, cur_243, 1);
            float pv_253 = _shfl_xor_158;
            float _fmax_387 = fmaxf(cur_243, pv_253);
            float hi_254 = _fmax_387;
            float _min_342 = fminf(cur_243, pv_253);
            float lo_255 = _min_342;
            cur_243 = ((up[3] != 0) ? hi_254 : lo_255);
            V_2[0] = cur_243;
            float _shfl_xor_159 = __shfl_xor_sync(0xFFFFFFFF, V_2[3], 15);
            float y1_256 = _shfl_xor_159;
            float _min_343 = fminf(V_2[1], y1_256);
            float lo1_257 = _min_343;
            float _fmax_388 = fmaxf(r_1, lo1_257);
            r_1 = _fmax_388;
            float _fmax_389 = fmaxf(V_2[1], y1_256);
            float hi1_258 = _fmax_389;
            float cur_259 = hi1_258;
            float _shfl_xor_160 = __shfl_xor_sync(0xFFFFFFFF, cur_259, 8);
            float pv_260 = _shfl_xor_160;
            float _fmax_390 = fmaxf(cur_259, pv_260);
            float hi_261 = _fmax_390;
            float _min_344 = fminf(cur_259, pv_260);
            float lo_262 = _min_344;
            cur_259 = ((up[0] != 0) ? hi_261 : lo_262);
            float _shfl_xor_161 = __shfl_xor_sync(0xFFFFFFFF, cur_259, 4);
            float pv_263 = _shfl_xor_161;
            float _fmax_391 = fmaxf(cur_259, pv_263);
            float hi_264 = _fmax_391;
            float _min_345 = fminf(cur_259, pv_263);
            float lo_265 = _min_345;
            cur_259 = ((up[1] != 0) ? hi_264 : lo_265);
            float _shfl_xor_162 = __shfl_xor_sync(0xFFFFFFFF, cur_259, 2);
            float pv_266 = _shfl_xor_162;
            float _fmax_392 = fmaxf(cur_259, pv_266);
            float hi_267 = _fmax_392;
            float _min_346 = fminf(cur_259, pv_266);
            float lo_268 = _min_346;
            cur_259 = ((up[2] != 0) ? hi_267 : lo_268);
            float _shfl_xor_163 = __shfl_xor_sync(0xFFFFFFFF, cur_259, 1);
            float pv_269 = _shfl_xor_163;
            float _fmax_393 = fmaxf(cur_259, pv_269);
            float hi_270 = _fmax_393;
            float _min_347 = fminf(cur_259, pv_269);
            float lo_271 = _min_347;
            cur_259 = ((up[3] != 0) ? hi_270 : lo_271);
            V_2[1] = cur_259;
            float _shfl_xor_164 = __shfl_xor_sync(0xFFFFFFFF, V_2[1], 15);
            float yl_272 = _shfl_xor_164;
            float _min_348 = fminf(V_2[0], yl_272);
            float lol_273 = _min_348;
            float _fmax_394 = fmaxf(r_1, lol_273);
            r_1 = _fmax_394;
            float _fmax_395 = fmaxf(V_2[0], yl_272);
            float hil_274 = _fmax_395;
            float cur_275 = hil_274;
            float _shfl_xor_165 = __shfl_xor_sync(0xFFFFFFFF, cur_275, 8);
            float pv_276 = _shfl_xor_165;
            float _fmax_396 = fmaxf(cur_275, pv_276);
            float hi_277 = _fmax_396;
            float _min_349 = fminf(cur_275, pv_276);
            float lo_278 = _min_349;
            cur_275 = ((up[0] != 0) ? hi_277 : lo_278);
            float _shfl_xor_166 = __shfl_xor_sync(0xFFFFFFFF, cur_275, 4);
            float pv_279 = _shfl_xor_166;
            float _fmax_397 = fmaxf(cur_275, pv_279);
            float hi_280 = _fmax_397;
            float _min_350 = fminf(cur_275, pv_279);
            float lo_281 = _min_350;
            cur_275 = ((up[1] != 0) ? hi_280 : lo_281);
            float _shfl_xor_167 = __shfl_xor_sync(0xFFFFFFFF, cur_275, 2);
            float pv_282 = _shfl_xor_167;
            float _fmax_398 = fmaxf(cur_275, pv_282);
            float hi_283 = _fmax_398;
            float _min_351 = fminf(cur_275, pv_282);
            float lo_284 = _min_351;
            cur_275 = ((up[2] != 0) ? hi_283 : lo_284);
            float _shfl_xor_168 = __shfl_xor_sync(0xFFFFFFFF, cur_275, 1);
            float pv_285 = _shfl_xor_168;
            float _fmax_399 = fmaxf(cur_275, pv_285);
            float hi_286 = _fmax_399;
            float _min_352 = fminf(cur_275, pv_285);
            float lo_287 = _min_352;
            cur_275 = ((up[3] != 0) ? hi_286 : lo_287);
            V_2[0] = cur_275;
            float K_288 = V_2[0];
            int qb_289 = (sg * 2 + cg) * 32;
            q2[qb_289 + ln] = K_288;
            q2[qb_289 + 16 + ln] = r_1;
            asm volatile("barrier.sync 8, 256;" ::: "memory");
            if (tid_1 < 32) {
                float r2_1 = neg_inf;
                float _fmax_400 = fmaxf(r2_1, q2[cg * 32 + 16 + ln]);
                r2_1 = _fmax_400;
                float _fmax_401 = fmaxf(r2_1, q2[(2 + cg) * 32 + 16 + ln]);
                r2_1 = _fmax_401;
                float _fmax_402 = fmaxf(r2_1, q2[(4 + cg) * 32 + 16 + ln]);
                r2_1 = _fmax_402;
                float _fmax_403 = fmaxf(r2_1, q2[(6 + cg) * 32 + 16 + ln]);
                r2_1 = _fmax_403;
                float _fmax_404 = fmaxf(r2_1, q2[(8 + cg) * 32 + 16 + ln]);
                r2_1 = _fmax_404;
                float _fmax_405 = fmaxf(r2_1, q2[(10 + cg) * 32 + 16 + ln]);
                r2_1 = _fmax_405;
                float _fmax_406 = fmaxf(r2_1, q2[(12 + cg) * 32 + 16 + ln]);
                r2_1 = _fmax_406;
                float _fmax_407 = fmaxf(r2_1, q2[(14 + cg) * 32 + 16 + ln]);
                r2_1 = _fmax_407;
                float V2_1[4];
                float x2_1 = q2[cg * 32 + ln];
                float y2_1 = q2[(8 + cg) * 32 + lnr];
                float _min_353 = fminf(x2_1, y2_1);
                float lo2_1 = _min_353;
                float _fmax_408 = fmaxf(r2_1, lo2_1);
                r2_1 = _fmax_408;
                float _fmax_409 = fmaxf(x2_1, y2_1);
                float hi2_1 = _fmax_409;
                float cur_0_1 = hi2_1;
                float _shfl_xor_169 = __shfl_xor_sync(0xFFFFFFFF, cur_0_1, 8);
                float pv_1_1 = _shfl_xor_169;
                float _fmax_410 = fmaxf(cur_0_1, pv_1_1);
                float hi_2_1 = _fmax_410;
                float _min_354 = fminf(cur_0_1, pv_1_1);
                float lo_3_1 = _min_354;
                cur_0_1 = ((up[0] != 0) ? hi_2_1 : lo_3_1);
                float _shfl_xor_170 = __shfl_xor_sync(0xFFFFFFFF, cur_0_1, 4);
                float pv_4_1 = _shfl_xor_170;
                float _fmax_411 = fmaxf(cur_0_1, pv_4_1);
                float hi_5_1 = _fmax_411;
                float _min_355 = fminf(cur_0_1, pv_4_1);
                float lo_6_1 = _min_355;
                cur_0_1 = ((up[1] != 0) ? hi_5_1 : lo_6_1);
                float _shfl_xor_171 = __shfl_xor_sync(0xFFFFFFFF, cur_0_1, 2);
                float pv_7_1 = _shfl_xor_171;
                float _fmax_412 = fmaxf(cur_0_1, pv_7_1);
                float hi_8_1 = _fmax_412;
                float _min_356 = fminf(cur_0_1, pv_7_1);
                float lo_9_1 = _min_356;
                cur_0_1 = ((up[2] != 0) ? hi_8_1 : lo_9_1);
                float _shfl_xor_172 = __shfl_xor_sync(0xFFFFFFFF, cur_0_1, 1);
                float pv_11 = _shfl_xor_172;
                float _fmax_413 = fmaxf(cur_0_1, pv_11);
                float hi_12 = _fmax_413;
                float _min_357 = fminf(cur_0_1, pv_11);
                float lo_13 = _min_357;
                cur_0_1 = ((up[3] != 0) ? hi_12 : lo_13);
                V2_1[0] = cur_0_1;
                float x2_14 = q2[(2 + cg) * 32 + ln];
                float y2_15 = q2[(10 + cg) * 32 + lnr];
                float _min_358 = fminf(x2_14, y2_15);
                float lo2_16 = _min_358;
                float _fmax_414 = fmaxf(r2_1, lo2_16);
                r2_1 = _fmax_414;
                float _fmax_415 = fmaxf(x2_14, y2_15);
                float hi2_17 = _fmax_415;
                float cur_18 = hi2_17;
                float _shfl_xor_173 = __shfl_xor_sync(0xFFFFFFFF, cur_18, 8);
                float pv_20 = _shfl_xor_173;
                float _fmax_416 = fmaxf(cur_18, pv_20);
                float hi_21 = _fmax_416;
                float _min_359 = fminf(cur_18, pv_20);
                float lo_22 = _min_359;
                cur_18 = ((up[0] != 0) ? hi_21 : lo_22);
                float _shfl_xor_174 = __shfl_xor_sync(0xFFFFFFFF, cur_18, 4);
                float pv_23 = _shfl_xor_174;
                float _fmax_417 = fmaxf(cur_18, pv_23);
                float hi_24 = _fmax_417;
                float _min_360 = fminf(cur_18, pv_23);
                float lo_25 = _min_360;
                cur_18 = ((up[1] != 0) ? hi_24 : lo_25);
                float _shfl_xor_175 = __shfl_xor_sync(0xFFFFFFFF, cur_18, 2);
                float pv_26 = _shfl_xor_175;
                float _fmax_418 = fmaxf(cur_18, pv_26);
                float hi_27 = _fmax_418;
                float _min_361 = fminf(cur_18, pv_26);
                float lo_28 = _min_361;
                cur_18 = ((up[2] != 0) ? hi_27 : lo_28);
                float _shfl_xor_176 = __shfl_xor_sync(0xFFFFFFFF, cur_18, 1);
                float pv_30 = _shfl_xor_176;
                float _fmax_419 = fmaxf(cur_18, pv_30);
                float hi_31 = _fmax_419;
                float _min_362 = fminf(cur_18, pv_30);
                float lo_32 = _min_362;
                cur_18 = ((up[3] != 0) ? hi_31 : lo_32);
                V2_1[1] = cur_18;
                float x2_33 = q2[(4 + cg) * 32 + ln];
                float y2_34 = q2[(12 + cg) * 32 + lnr];
                float _min_363 = fminf(x2_33, y2_34);
                float lo2_35 = _min_363;
                float _fmax_420 = fmaxf(r2_1, lo2_35);
                r2_1 = _fmax_420;
                float _fmax_421 = fmaxf(x2_33, y2_34);
                float hi2_36 = _fmax_421;
                float cur_37 = hi2_36;
                float _shfl_xor_177 = __shfl_xor_sync(0xFFFFFFFF, cur_37, 8);
                float pv_39 = _shfl_xor_177;
                float _fmax_422 = fmaxf(cur_37, pv_39);
                float hi_40 = _fmax_422;
                float _min_364 = fminf(cur_37, pv_39);
                float lo_41 = _min_364;
                cur_37 = ((up[0] != 0) ? hi_40 : lo_41);
                float _shfl_xor_178 = __shfl_xor_sync(0xFFFFFFFF, cur_37, 4);
                float pv_42 = _shfl_xor_178;
                float _fmax_423 = fmaxf(cur_37, pv_42);
                float hi_43 = _fmax_423;
                float _min_365 = fminf(cur_37, pv_42);
                float lo_44 = _min_365;
                cur_37 = ((up[1] != 0) ? hi_43 : lo_44);
                float _shfl_xor_179 = __shfl_xor_sync(0xFFFFFFFF, cur_37, 2);
                float pv_45 = _shfl_xor_179;
                float _fmax_424 = fmaxf(cur_37, pv_45);
                float hi_46 = _fmax_424;
                float _min_366 = fminf(cur_37, pv_45);
                float lo_47 = _min_366;
                cur_37 = ((up[2] != 0) ? hi_46 : lo_47);
                float _shfl_xor_180 = __shfl_xor_sync(0xFFFFFFFF, cur_37, 1);
                float pv_49 = _shfl_xor_180;
                float _fmax_425 = fmaxf(cur_37, pv_49);
                float hi_52 = _fmax_425;
                float _min_367 = fminf(cur_37, pv_49);
                float lo_53 = _min_367;
                cur_37 = ((up[3] != 0) ? hi_52 : lo_53);
                V2_1[2] = cur_37;
                float x2_54 = q2[(6 + cg) * 32 + ln];
                float y2_55 = q2[(14 + cg) * 32 + lnr];
                float _min_368 = fminf(x2_54, y2_55);
                float lo2_56 = _min_368;
                float _fmax_426 = fmaxf(r2_1, lo2_56);
                r2_1 = _fmax_426;
                float _fmax_427 = fmaxf(x2_54, y2_55);
                float hi2_57 = _fmax_427;
                float cur_58 = hi2_57;
                float _shfl_xor_181 = __shfl_xor_sync(0xFFFFFFFF, cur_58, 8);
                float pv_59 = _shfl_xor_181;
                float _fmax_428 = fmaxf(cur_58, pv_59);
                float hi_60 = _fmax_428;
                float _min_369 = fminf(cur_58, pv_59);
                float lo_61 = _min_369;
                cur_58 = ((up[0] != 0) ? hi_60 : lo_61);
                float _shfl_xor_182 = __shfl_xor_sync(0xFFFFFFFF, cur_58, 4);
                float pv_62 = _shfl_xor_182;
                float _fmax_429 = fmaxf(cur_58, pv_62);
                float hi_64 = _fmax_429;
                float _min_370 = fminf(cur_58, pv_62);
                float lo_65 = _min_370;
                cur_58 = ((up[1] != 0) ? hi_64 : lo_65);
                float _shfl_xor_183 = __shfl_xor_sync(0xFFFFFFFF, cur_58, 2);
                float pv_66 = _shfl_xor_183;
                float _fmax_430 = fmaxf(cur_58, pv_66);
                float hi_68 = _fmax_430;
                float _min_371 = fminf(cur_58, pv_66);
                float lo_69 = _min_371;
                cur_58 = ((up[2] != 0) ? hi_68 : lo_69);
                float _shfl_xor_184 = __shfl_xor_sync(0xFFFFFFFF, cur_58, 1);
                float pv_70 = _shfl_xor_184;
                float _fmax_431 = fmaxf(cur_58, pv_70);
                float hi_74_1 = _fmax_431;
                float _min_372 = fminf(cur_58, pv_70);
                float lo_75_1 = _min_372;
                cur_58 = ((up[3] != 0) ? hi_74_1 : lo_75_1);
                V2_1[3] = cur_58;
                float _shfl_xor_185 = __shfl_xor_sync(0xFFFFFFFF, V2_1[2], 15);
                float y3_1 = _shfl_xor_185;
                float _min_373 = fminf(V2_1[0], y3_1);
                float lo3_1 = _min_373;
                float _fmax_432 = fmaxf(r2_1, lo3_1);
                r2_1 = _fmax_432;
                float _fmax_433 = fmaxf(V2_1[0], y3_1);
                float hi3_1 = _fmax_433;
                float cur_76 = hi3_1;
                float _shfl_xor_186 = __shfl_xor_sync(0xFFFFFFFF, cur_76, 8);
                float pv_77 = _shfl_xor_186;
                float _fmax_434 = fmaxf(cur_76, pv_77);
                float hi_78_1 = _fmax_434;
                float _min_374 = fminf(cur_76, pv_77);
                float lo_79_1 = _min_374;
                cur_76 = ((up[0] != 0) ? hi_78_1 : lo_79_1);
                float _shfl_xor_187 = __shfl_xor_sync(0xFFFFFFFF, cur_76, 4);
                float pv_80_1 = _shfl_xor_187;
                float _fmax_435 = fmaxf(cur_76, pv_80_1);
                float hi_82_1 = _fmax_435;
                float _min_375 = fminf(cur_76, pv_80_1);
                float lo_83_1 = _min_375;
                cur_76 = ((up[1] != 0) ? hi_82_1 : lo_83_1);
                float _shfl_xor_188 = __shfl_xor_sync(0xFFFFFFFF, cur_76, 2);
                float pv_84 = _shfl_xor_188;
                float _fmax_436 = fmaxf(cur_76, pv_84);
                float hi_86 = _fmax_436;
                float _min_376 = fminf(cur_76, pv_84);
                float lo_87 = _min_376;
                cur_76 = ((up[2] != 0) ? hi_86 : lo_87);
                float _shfl_xor_189 = __shfl_xor_sync(0xFFFFFFFF, cur_76, 1);
                float pv_88_1 = _shfl_xor_189;
                float _fmax_437 = fmaxf(cur_76, pv_88_1);
                float hi_90_1 = _fmax_437;
                float _min_377 = fminf(cur_76, pv_88_1);
                float lo_91_1 = _min_377;
                cur_76 = ((up[3] != 0) ? hi_90_1 : lo_91_1);
                V2_1[0] = cur_76;
                float _shfl_xor_190 = __shfl_xor_sync(0xFFFFFFFF, V2_1[3], 15);
                float y3_92 = _shfl_xor_190;
                float _min_378 = fminf(V2_1[1], y3_92);
                float lo3_93 = _min_378;
                float _fmax_438 = fmaxf(r2_1, lo3_93);
                r2_1 = _fmax_438;
                float _fmax_439 = fmaxf(V2_1[1], y3_92);
                float hi3_94 = _fmax_439;
                float cur_95 = hi3_94;
                float _shfl_xor_191 = __shfl_xor_sync(0xFFFFFFFF, cur_95, 8);
                float pv_97 = _shfl_xor_191;
                float _fmax_440 = fmaxf(cur_95, pv_97);
                float hi_100 = _fmax_440;
                float _min_379 = fminf(cur_95, pv_97);
                float lo_101 = _min_379;
                cur_95 = ((up[0] != 0) ? hi_100 : lo_101);
                float _shfl_xor_192 = __shfl_xor_sync(0xFFFFFFFF, cur_95, 4);
                float pv_102 = _shfl_xor_192;
                float _fmax_441 = fmaxf(cur_95, pv_102);
                float hi_104_1 = _fmax_441;
                float _min_380 = fminf(cur_95, pv_102);
                float lo_105_1 = _min_380;
                cur_95 = ((up[1] != 0) ? hi_104_1 : lo_105_1);
                float _shfl_xor_193 = __shfl_xor_sync(0xFFFFFFFF, cur_95, 2);
                float pv_106 = _shfl_xor_193;
                float _fmax_442 = fmaxf(cur_95, pv_106);
                float hi_108_1 = _fmax_442;
                float _min_381 = fminf(cur_95, pv_106);
                float lo_109_1 = _min_381;
                cur_95 = ((up[2] != 0) ? hi_108_1 : lo_109_1);
                float _shfl_xor_194 = __shfl_xor_sync(0xFFFFFFFF, cur_95, 1);
                float pv_110 = _shfl_xor_194;
                float _fmax_443 = fmaxf(cur_95, pv_110);
                float hi_112_1 = _fmax_443;
                float _min_382 = fminf(cur_95, pv_110);
                float lo_113_1 = _min_382;
                cur_95 = ((up[3] != 0) ? hi_112_1 : lo_113_1);
                V2_1[1] = cur_95;
                float _shfl_xor_195 = __shfl_xor_sync(0xFFFFFFFF, V2_1[1], 15);
                float yl2_1 = _shfl_xor_195;
                float _min_383 = fminf(V2_1[0], yl2_1);
                float lol2_1 = _min_383;
                float _fmax_444 = fmaxf(r2_1, lol2_1);
                r2_1 = _fmax_444;
                float _fmax_445 = fmaxf(V2_1[0], yl2_1);
                V2_1[0] = _fmax_445;
                K_288 = V2_1[0];
                r_1 = r2_1;
            }
            rr2x[0] = r_1;
            K2 = K_288;
        }
        if (tid_1 < 32) {
            unsigned int cgx = ccnt[cg];
            unsigned int k1x = __as_u32(K);
            int isq = 0;
            if (gflag != 0 && (k1x & 4294965248u) == qg) {
                isq = 1;
            }
            unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, isq != 0);
            unsigned int mqx = _vote_0;
            unsigned int gmx = mqx & 65535;
            if ((tid_1 & 16) != 0) {
                gmx = mqx >> 16 & 65535;
            }
            unsigned int lowx = (unsigned int)((1 << ln) - 1);
            int _popc_0 = __popc(gmx & lowx);
            int rx = _popc_0;
            if (gflag != 0 && cgx <= 16) {
                K2 = __uint_as_float(1073741824 | k1x & 2047);
                if (isq != 0) {
                    float xs[16];
                    xs[0] = 0.0f;
                    if (cgx > 0) {
                        xs[0] = __uint_as_float(cbuf[cg * 16]);
                    }
                    xs[1] = 0.0f;
                    if (cgx > 1) {
                        xs[1] = __uint_as_float(cbuf[cg * 16 + 1]);
                    }
                    xs[2] = 0.0f;
                    if (cgx > 2) {
                        xs[2] = __uint_as_float(cbuf[cg * 16 + 2]);
                    }
                    xs[3] = 0.0f;
                    if (cgx > 3) {
                        xs[3] = __uint_as_float(cbuf[cg * 16 + 3]);
                    }
                    xs[4] = 0.0f;
                    if (cgx > 4) {
                        xs[4] = __uint_as_float(cbuf[cg * 16 + 4]);
                    }
                    xs[5] = 0.0f;
                    if (cgx > 5) {
                        xs[5] = __uint_as_float(cbuf[cg * 16 + 5]);
                    }
                    xs[6] = 0.0f;
                    if (cgx > 6) {
                        xs[6] = __uint_as_float(cbuf[cg * 16 + 6]);
                    }
                    xs[7] = 0.0f;
                    if (cgx > 7) {
                        xs[7] = __uint_as_float(cbuf[cg * 16 + 7]);
                    }
                    xs[8] = 0.0f;
                    if (cgx > 8) {
                        xs[8] = __uint_as_float(cbuf[cg * 16 + 8]);
                    }
                    xs[9] = 0.0f;
                    if (cgx > 9) {
                        xs[9] = __uint_as_float(cbuf[cg * 16 + 9]);
                    }
                    xs[10] = 0.0f;
                    if (cgx > 10) {
                        xs[10] = __uint_as_float(cbuf[cg * 16 + 10]);
                    }
                    xs[11] = 0.0f;
                    if (cgx > 11) {
                        xs[11] = __uint_as_float(cbuf[cg * 16 + 11]);
                    }
                    xs[12] = 0.0f;
                    if (cgx > 12) {
                        xs[12] = __uint_as_float(cbuf[cg * 16 + 12]);
                    }
                    xs[13] = 0.0f;
                    if (cgx > 13) {
                        xs[13] = __uint_as_float(cbuf[cg * 16 + 13]);
                    }
                    xs[14] = 0.0f;
                    if (cgx > 14) {
                        xs[14] = __uint_as_float(cbuf[cg * 16 + 14]);
                    }
                    xs[15] = 0.0f;
                    if (cgx > 15) {
                        xs[15] = __uint_as_float(cbuf[cg * 16 + 15]);
                    }
                    float _fmax_446 = fmaxf(xs[0], xs[13]);
                    float hi_0 = _fmax_446;
                    float _min_384 = fminf(xs[0], xs[13]);
                    float lo_1 = _min_384;
                    xs[0] = hi_0;
                    xs[13] = lo_1;
                    float _fmax_447 = fmaxf(xs[1], xs[12]);
                    float hi_2_2 = _fmax_447;
                    float _min_385 = fminf(xs[1], xs[12]);
                    float lo_3_2 = _min_385;
                    xs[1] = hi_2_2;
                    xs[12] = lo_3_2;
                    float _fmax_448 = fmaxf(xs[2], xs[15]);
                    float hi_4 = _fmax_448;
                    float _min_386 = fminf(xs[2], xs[15]);
                    float lo_5 = _min_386;
                    xs[2] = hi_4;
                    xs[15] = lo_5;
                    float _fmax_449 = fmaxf(xs[3], xs[14]);
                    float hi_6 = _fmax_449;
                    float _min_387 = fminf(xs[3], xs[14]);
                    float lo_7 = _min_387;
                    xs[3] = hi_6;
                    xs[14] = lo_7;
                    float _fmax_450 = fmaxf(xs[4], xs[8]);
                    float hi_8_2 = _fmax_450;
                    float _min_388 = fminf(xs[4], xs[8]);
                    float lo_9_2 = _min_388;
                    xs[4] = hi_8_2;
                    xs[8] = lo_9_2;
                    float _fmax_451 = fmaxf(xs[5], xs[6]);
                    float hi_10 = _fmax_451;
                    float _min_389 = fminf(xs[5], xs[6]);
                    float lo_11 = _min_389;
                    xs[5] = hi_10;
                    xs[6] = lo_11;
                    float _fmax_452 = fmaxf(xs[7], xs[11]);
                    float hi_12_1 = _fmax_452;
                    float _min_390 = fminf(xs[7], xs[11]);
                    float lo_13_1 = _min_390;
                    xs[7] = hi_12_1;
                    xs[11] = lo_13_1;
                    float _fmax_453 = fmaxf(xs[9], xs[10]);
                    float hi_14_1 = _fmax_453;
                    float _min_391 = fminf(xs[9], xs[10]);
                    float lo_15_1 = _min_391;
                    xs[9] = hi_14_1;
                    xs[10] = lo_15_1;
                    float _fmax_454 = fmaxf(xs[0], xs[5]);
                    float hi_16 = _fmax_454;
                    float _min_392 = fminf(xs[0], xs[5]);
                    float lo_17 = _min_392;
                    xs[0] = hi_16;
                    xs[5] = lo_17;
                    float _fmax_455 = fmaxf(xs[1], xs[7]);
                    float hi_18 = _fmax_455;
                    float _min_393 = fminf(xs[1], xs[7]);
                    float lo_19 = _min_393;
                    xs[1] = hi_18;
                    xs[7] = lo_19;
                    float _fmax_456 = fmaxf(xs[2], xs[9]);
                    float hi_20_1 = _fmax_456;
                    float _min_394 = fminf(xs[2], xs[9]);
                    float lo_21_1 = _min_394;
                    xs[2] = hi_20_1;
                    xs[9] = lo_21_1;
                    float _fmax_457 = fmaxf(xs[3], xs[4]);
                    float hi_22_1 = _fmax_457;
                    float _min_395 = fminf(xs[3], xs[4]);
                    float lo_23_1 = _min_395;
                    xs[3] = hi_22_1;
                    xs[4] = lo_23_1;
                    float _fmax_458 = fmaxf(xs[6], xs[13]);
                    float hi_24_1 = _fmax_458;
                    float _min_396 = fminf(xs[6], xs[13]);
                    float lo_25_1 = _min_396;
                    xs[6] = hi_24_1;
                    xs[13] = lo_25_1;
                    float _fmax_459 = fmaxf(xs[8], xs[14]);
                    float hi_26 = _fmax_459;
                    float _min_397 = fminf(xs[8], xs[14]);
                    float lo_27 = _min_397;
                    xs[8] = hi_26;
                    xs[14] = lo_27;
                    float _fmax_460 = fmaxf(xs[10], xs[15]);
                    float hi_28_1 = _fmax_460;
                    float _min_398 = fminf(xs[10], xs[15]);
                    float lo_29_1 = _min_398;
                    xs[10] = hi_28_1;
                    xs[15] = lo_29_1;
                    float _fmax_461 = fmaxf(xs[11], xs[12]);
                    float hi_30_1 = _fmax_461;
                    float _min_399 = fminf(xs[11], xs[12]);
                    float lo_31_1 = _min_399;
                    xs[11] = hi_30_1;
                    xs[12] = lo_31_1;
                    float _fmax_462 = fmaxf(xs[0], xs[1]);
                    float hi_32 = _fmax_462;
                    float _min_400 = fminf(xs[0], xs[1]);
                    float lo_33 = _min_400;
                    xs[0] = hi_32;
                    xs[1] = lo_33;
                    float _fmax_463 = fmaxf(xs[2], xs[3]);
                    float hi_34 = _fmax_463;
                    float _min_401 = fminf(xs[2], xs[3]);
                    float lo_35 = _min_401;
                    xs[2] = hi_34;
                    xs[3] = lo_35;
                    float _fmax_464 = fmaxf(xs[4], xs[5]);
                    float hi_36_2 = _fmax_464;
                    float _min_402 = fminf(xs[4], xs[5]);
                    float lo_37_2 = _min_402;
                    xs[4] = hi_36_2;
                    xs[5] = lo_37_2;
                    float _fmax_465 = fmaxf(xs[6], xs[8]);
                    float hi_38 = _fmax_465;
                    float _min_403 = fminf(xs[6], xs[8]);
                    float lo_39 = _min_403;
                    xs[6] = hi_38;
                    xs[8] = lo_39;
                    float _fmax_466 = fmaxf(xs[7], xs[9]);
                    float hi_40_1 = _fmax_466;
                    float _min_404 = fminf(xs[7], xs[9]);
                    float lo_41_1 = _min_404;
                    xs[7] = hi_40_1;
                    xs[9] = lo_41_1;
                    float _fmax_467 = fmaxf(xs[10], xs[11]);
                    float hi_42_1 = _fmax_467;
                    float _min_405 = fminf(xs[10], xs[11]);
                    float lo_43_1 = _min_405;
                    xs[10] = hi_42_1;
                    xs[11] = lo_43_1;
                    float _fmax_468 = fmaxf(xs[12], xs[13]);
                    float hi_44 = _fmax_468;
                    float _min_406 = fminf(xs[12], xs[13]);
                    float lo_45 = _min_406;
                    xs[12] = hi_44;
                    xs[13] = lo_45;
                    float _fmax_469 = fmaxf(xs[14], xs[15]);
                    float hi_46_1 = _fmax_469;
                    float _min_407 = fminf(xs[14], xs[15]);
                    float lo_47_1 = _min_407;
                    xs[14] = hi_46_1;
                    xs[15] = lo_47_1;
                    float _fmax_470 = fmaxf(xs[0], xs[2]);
                    float hi_48 = _fmax_470;
                    float _min_408 = fminf(xs[0], xs[2]);
                    float lo_49 = _min_408;
                    xs[0] = hi_48;
                    xs[2] = lo_49;
                    float _fmax_471 = fmaxf(xs[1], xs[3]);
                    float hi_50_1 = _fmax_471;
                    float _min_409 = fminf(xs[1], xs[3]);
                    float lo_51_1 = _min_409;
                    xs[1] = hi_50_1;
                    xs[3] = lo_51_1;
                    float _fmax_472 = fmaxf(xs[4], xs[10]);
                    float hi_52_1 = _fmax_472;
                    float _min_410 = fminf(xs[4], xs[10]);
                    float lo_53_1 = _min_410;
                    xs[4] = hi_52_1;
                    xs[10] = lo_53_1;
                    float _fmax_473 = fmaxf(xs[5], xs[11]);
                    float hi_54_2 = _fmax_473;
                    float _min_411 = fminf(xs[5], xs[11]);
                    float lo_55_2 = _min_411;
                    xs[5] = hi_54_2;
                    xs[11] = lo_55_2;
                    float _fmax_474 = fmaxf(xs[6], xs[7]);
                    float hi_56 = _fmax_474;
                    float _min_412 = fminf(xs[6], xs[7]);
                    float lo_57 = _min_412;
                    xs[6] = hi_56;
                    xs[7] = lo_57;
                    float _fmax_475 = fmaxf(xs[8], xs[9]);
                    float hi_58_2 = _fmax_475;
                    float _min_413 = fminf(xs[8], xs[9]);
                    float lo_59_2 = _min_413;
                    xs[8] = hi_58_2;
                    xs[9] = lo_59_2;
                    float _fmax_476 = fmaxf(xs[12], xs[14]);
                    float hi_60_1 = _fmax_476;
                    float _min_414 = fminf(xs[12], xs[14]);
                    float lo_61_1 = _min_414;
                    xs[12] = hi_60_1;
                    xs[14] = lo_61_1;
                    float _fmax_477 = fmaxf(xs[13], xs[15]);
                    float hi_62_2 = _fmax_477;
                    float _min_415 = fminf(xs[13], xs[15]);
                    float lo_63_2 = _min_415;
                    xs[13] = hi_62_2;
                    xs[15] = lo_63_2;
                    float _fmax_478 = fmaxf(xs[1], xs[2]);
                    float hi_64_1 = _fmax_478;
                    float _min_416 = fminf(xs[1], xs[2]);
                    float lo_65_1 = _min_416;
                    xs[1] = hi_64_1;
                    xs[2] = lo_65_1;
                    float _fmax_479 = fmaxf(xs[3], xs[12]);
                    float hi_66_1 = _fmax_479;
                    float _min_417 = fminf(xs[3], xs[12]);
                    float lo_67_1 = _min_417;
                    xs[3] = hi_66_1;
                    xs[12] = lo_67_1;
                    float _fmax_480 = fmaxf(xs[4], xs[6]);
                    float hi_68_1 = _fmax_480;
                    float _min_418 = fminf(xs[4], xs[6]);
                    float lo_69_1 = _min_418;
                    xs[4] = hi_68_1;
                    xs[6] = lo_69_1;
                    float _fmax_481 = fmaxf(xs[5], xs[7]);
                    float hi_70_1 = _fmax_481;
                    float _min_419 = fminf(xs[5], xs[7]);
                    float lo_71_1 = _min_419;
                    xs[5] = hi_70_1;
                    xs[7] = lo_71_1;
                    float _fmax_482 = fmaxf(xs[8], xs[10]);
                    float hi_72_1 = _fmax_482;
                    float _min_420 = fminf(xs[8], xs[10]);
                    float lo_73_1 = _min_420;
                    xs[8] = hi_72_1;
                    xs[10] = lo_73_1;
                    float _fmax_483 = fmaxf(xs[9], xs[11]);
                    float hi_74_2 = _fmax_483;
                    float _min_421 = fminf(xs[9], xs[11]);
                    float lo_75_2 = _min_421;
                    xs[9] = hi_74_2;
                    xs[11] = lo_75_2;
                    float _fmax_484 = fmaxf(xs[13], xs[14]);
                    float hi_76_1 = _fmax_484;
                    float _min_422 = fminf(xs[13], xs[14]);
                    float lo_77_1 = _min_422;
                    xs[13] = hi_76_1;
                    xs[14] = lo_77_1;
                    float _fmax_485 = fmaxf(xs[1], xs[4]);
                    float hi_78_2 = _fmax_485;
                    float _min_423 = fminf(xs[1], xs[4]);
                    float lo_79_2 = _min_423;
                    xs[1] = hi_78_2;
                    xs[4] = lo_79_2;
                    float _fmax_486 = fmaxf(xs[2], xs[6]);
                    float hi_80_1 = _fmax_486;
                    float _min_424 = fminf(xs[2], xs[6]);
                    float lo_81_1 = _min_424;
                    xs[2] = hi_80_1;
                    xs[6] = lo_81_1;
                    float _fmax_487 = fmaxf(xs[5], xs[8]);
                    float hi_82_2 = _fmax_487;
                    float _min_425 = fminf(xs[5], xs[8]);
                    float lo_83_2 = _min_425;
                    xs[5] = hi_82_2;
                    xs[8] = lo_83_2;
                    float _fmax_488 = fmaxf(xs[7], xs[10]);
                    float hi_84_1 = _fmax_488;
                    float _min_426 = fminf(xs[7], xs[10]);
                    float lo_85_1 = _min_426;
                    xs[7] = hi_84_1;
                    xs[10] = lo_85_1;
                    float _fmax_489 = fmaxf(xs[9], xs[13]);
                    float hi_86_1 = _fmax_489;
                    float _min_427 = fminf(xs[9], xs[13]);
                    float lo_87_1 = _min_427;
                    xs[9] = hi_86_1;
                    xs[13] = lo_87_1;
                    float _fmax_490 = fmaxf(xs[11], xs[14]);
                    float hi_88 = _fmax_490;
                    float _min_428 = fminf(xs[11], xs[14]);
                    float lo_89 = _min_428;
                    xs[11] = hi_88;
                    xs[14] = lo_89;
                    float _fmax_491 = fmaxf(xs[2], xs[4]);
                    float hi_90_2 = _fmax_491;
                    float _min_429 = fminf(xs[2], xs[4]);
                    float lo_91_2 = _min_429;
                    xs[2] = hi_90_2;
                    xs[4] = lo_91_2;
                    float _fmax_492 = fmaxf(xs[3], xs[6]);
                    float hi_92 = _fmax_492;
                    float _min_430 = fminf(xs[3], xs[6]);
                    float lo_93 = _min_430;
                    xs[3] = hi_92;
                    xs[6] = lo_93;
                    float _fmax_493 = fmaxf(xs[9], xs[12]);
                    float hi_94_2 = _fmax_493;
                    float _min_431 = fminf(xs[9], xs[12]);
                    float lo_95_2 = _min_431;
                    xs[9] = hi_94_2;
                    xs[12] = lo_95_2;
                    float _fmax_494 = fmaxf(xs[11], xs[13]);
                    float hi_96 = _fmax_494;
                    float _min_432 = fminf(xs[11], xs[13]);
                    float lo_97 = _min_432;
                    xs[11] = hi_96;
                    xs[13] = lo_97;
                    float _fmax_495 = fmaxf(xs[3], xs[5]);
                    float hi_98_2 = _fmax_495;
                    float _min_433 = fminf(xs[3], xs[5]);
                    float lo_99_2 = _min_433;
                    xs[3] = hi_98_2;
                    xs[5] = lo_99_2;
                    float _fmax_496 = fmaxf(xs[6], xs[8]);
                    float hi_100_1 = _fmax_496;
                    float _min_434 = fminf(xs[6], xs[8]);
                    float lo_101_1 = _min_434;
                    xs[6] = hi_100_1;
                    xs[8] = lo_101_1;
                    float _fmax_497 = fmaxf(xs[7], xs[9]);
                    float hi_102_2 = _fmax_497;
                    float _min_435 = fminf(xs[7], xs[9]);
                    float lo_103_2 = _min_435;
                    xs[7] = hi_102_2;
                    xs[9] = lo_103_2;
                    float _fmax_498 = fmaxf(xs[10], xs[12]);
                    float hi_104_2 = _fmax_498;
                    float _min_436 = fminf(xs[10], xs[12]);
                    float lo_105_2 = _min_436;
                    xs[10] = hi_104_2;
                    xs[12] = lo_105_2;
                    float _fmax_499 = fmaxf(xs[3], xs[4]);
                    float hi_106_2 = _fmax_499;
                    float _min_437 = fminf(xs[3], xs[4]);
                    float lo_107_2 = _min_437;
                    xs[3] = hi_106_2;
                    xs[4] = lo_107_2;
                    float _fmax_500 = fmaxf(xs[5], xs[6]);
                    float hi_108_2 = _fmax_500;
                    float _min_438 = fminf(xs[5], xs[6]);
                    float lo_109_2 = _min_438;
                    xs[5] = hi_108_2;
                    xs[6] = lo_109_2;
                    float _fmax_501 = fmaxf(xs[7], xs[8]);
                    float hi_110_1 = _fmax_501;
                    float _min_439 = fminf(xs[7], xs[8]);
                    float lo_111_1 = _min_439;
                    xs[7] = hi_110_1;
                    xs[8] = lo_111_1;
                    float _fmax_502 = fmaxf(xs[9], xs[10]);
                    float hi_112_2 = _fmax_502;
                    float _min_440 = fminf(xs[9], xs[10]);
                    float lo_113_2 = _min_440;
                    xs[9] = hi_112_2;
                    xs[10] = lo_113_2;
                    float _fmax_503 = fmaxf(xs[11], xs[12]);
                    float hi_114_1 = _fmax_503;
                    float _min_441 = fminf(xs[11], xs[12]);
                    float lo_115_1 = _min_441;
                    xs[11] = hi_114_1;
                    xs[12] = lo_115_1;
                    float _fmax_504 = fmaxf(xs[6], xs[7]);
                    float hi_116_2 = _fmax_504;
                    float _min_442 = fminf(xs[6], xs[7]);
                    float lo_117_2 = _min_442;
                    xs[6] = hi_116_2;
                    xs[7] = lo_117_2;
                    float _fmax_505 = fmaxf(xs[8], xs[9]);
                    float hi_118_1 = _fmax_505;
                    float _min_443 = fminf(xs[8], xs[9]);
                    float lo_119_1 = _min_443;
                    xs[8] = hi_118_1;
                    xs[9] = lo_119_1;
                    K2 = xs[0];
                    if (rx == 1) {
                        K2 = xs[1];
                    }
                    if (rx == 2) {
                        K2 = xs[2];
                    }
                    if (rx == 3) {
                        K2 = xs[3];
                    }
                    if (rx == 4) {
                        K2 = xs[4];
                    }
                    if (rx == 5) {
                        K2 = xs[5];
                    }
                    if (rx == 6) {
                        K2 = xs[6];
                    }
                    if (rx == 7) {
                        K2 = xs[7];
                    }
                    if (rx == 8) {
                        K2 = xs[8];
                    }
                    if (rx == 9) {
                        K2 = xs[9];
                    }
                    if (rx == 10) {
                        K2 = xs[10];
                    }
                    if (rx == 11) {
                        K2 = xs[11];
                    }
                    if (rx == 12) {
                        K2 = xs[12];
                    }
                    if (rx == 13) {
                        K2 = xs[13];
                    }
                    if (rx == 14) {
                        K2 = xs[14];
                    }
                    if (rx == 15) {
                        K2 = xs[15];
                    }
                }
            }
        }
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
