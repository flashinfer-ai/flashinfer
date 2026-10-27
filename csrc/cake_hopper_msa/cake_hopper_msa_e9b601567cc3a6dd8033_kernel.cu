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

#define CAKE_WARP_UNIFORM_INT 1
#include "cake_hopper_msa_device.h"

#define NUM_MAIN_1_STAGES 1
#define SMEM_QCOL_OFF 0
#define SMEM_QCOL_STAGE_BYTES 8
#define SMEM_QCOL_STRIDE 8
#define SMEM_FLAGW_OFF 8
#define SMEM_FLAGW_STAGE_BYTES 16
#define SMEM_FLAGW_STRIDE 16
#define SMEM_CCNT_OFF 4888
#define SMEM_CCNT_STAGE_BYTES 8
#define SMEM_CCNT_STRIDE 8
#define SMEM_LIMC_OFF 4896
#define SMEM_LIMC_STAGE_BYTES 8
#define SMEM_LIMC_STRIDE 8
#define SMEM_CBUF_OFF 4904
#define SMEM_CBUF_STAGE_BYTES 128
#define SMEM_CBUF_STRIDE 128
#define SMEM_PUB_OFF 24
#define SMEM_PUB_STAGE_BYTES 4352
#define SMEM_PUB_STRIDE 4352
#define SMEM_Q2_OFF 4376
#define SMEM_Q2_STAGE_BYTES 512
#define SMEM_Q2_STRIDE 512
#define SMEM_TOTAL 5120
#define THREADS 64

extern "C" {

__global__ __launch_bounds__(64) void
kernel_cake_hopper_msa_e9b601567cc3a6dd8033(unsigned int* __restrict__ S, int* __restrict__ nvp, int* __restrict__ out, int tiles, int total_q, int num_heads, int num_chunks, int fb, int fe)
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
    unsigned int* ccnt = reinterpret_cast<unsigned int*>(smem_raw + 4888);
    const int ccnt_addr = smem + 4888;
    int* limc = reinterpret_cast<int*>(smem_raw + 4896);
    const int limc_addr = smem + 4896;
    unsigned int* cbuf = reinterpret_cast<unsigned int*>(smem_raw + 4904);
    const int cbuf_addr = smem + 4904;
    float* pub = reinterpret_cast<float*>(smem_raw + 24);
    const int pub_addr = smem + 24;
    float* q2 = reinterpret_cast<float*>(smem_raw + 4376);
    const int q2_addr = smem + 4376;

    // === Task calls (declaration order) ===
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
    float neg_inf = -CUDART_INF_F;
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
    long long p = cbase + (long long)t0 * nq64;
    cb[0] = 4286578688u;
    if (lim_full > t0) {
        cb[0] = S[p];
    }
    cb[1] = 4286578688u;
    if (lim_full > t0 + 1) {
        cb[1] = S[p + nq64];
    }
    cb[2] = 4286578688u;
    if (lim_full > t0 + 2) {
        cb[2] = S[p + 2 * nq64];
    }
    cb[3] = 4286578688u;
    if (lim_full > t0 + 3) {
        cb[3] = S[p + 3 * nq64];
    }
    cb[4] = 4286578688u;
    if (lim_full > t0 + 4) {
        cb[4] = S[p + 4 * nq64];
    }
    cb[5] = 4286578688u;
    if (lim_full > t0 + 5) {
        cb[5] = S[p + 5 * nq64];
    }
    cb[6] = 4286578688u;
    if (lim_full > t0 + 6) {
        cb[6] = S[p + 6 * nq64];
    }
    cb[7] = 4286578688u;
    if (lim_full > t0 + 7) {
        cb[7] = S[p + 7 * nq64];
    }
    cb[8] = 4286578688u;
    if (lim_full > t0 + 8) {
        cb[8] = S[p + 8 * nq64];
    }
    cb[9] = 4286578688u;
    if (lim_full > t0 + 9) {
        cb[9] = S[p + 9 * nq64];
    }
    cb[10] = 4286578688u;
    if (lim_full > t0 + 10) {
        cb[10] = S[p + 10 * nq64];
    }
    cb[11] = 4286578688u;
    if (lim_full > t0 + 11) {
        cb[11] = S[p + 11 * nq64];
    }
    cb[12] = 4286578688u;
    if (lim_full > t0 + 12) {
        cb[12] = S[p + 12 * nq64];
    }
    cb[13] = 4286578688u;
    if (lim_full > t0 + 13) {
        cb[13] = S[p + 13 * nq64];
    }
    cb[14] = 4286578688u;
    if (lim_full > t0 + 14) {
        cb[14] = S[p + 14 * nq64];
    }
    cb[15] = 4286578688u;
    if (lim_full > t0 + 15) {
        cb[15] = S[p + 15 * nq64];
    }
    asm volatile("" ::: "memory");
    int t0_0 = w * 16;
    if (lim <= t0_0) {
        cb[0] = 4286578688u;
    }
    if (lim <= t0_0 + 1) {
        cb[1] = 4286578688u;
    }
    if (lim <= t0_0 + 2) {
        cb[2] = 4286578688u;
    }
    if (lim <= t0_0 + 3) {
        cb[3] = 4286578688u;
    }
    if (lim <= t0_0 + 4) {
        cb[4] = 4286578688u;
    }
    if (lim <= t0_0 + 5) {
        cb[5] = 4286578688u;
    }
    if (lim <= t0_0 + 6) {
        cb[6] = 4286578688u;
    }
    if (lim <= t0_0 + 7) {
        cb[7] = 4286578688u;
    }
    if (lim <= t0_0 + 8) {
        cb[8] = 4286578688u;
    }
    if (lim <= t0_0 + 9) {
        cb[9] = 4286578688u;
    }
    if (lim <= t0_0 + 10) {
        cb[10] = 4286578688u;
    }
    if (lim <= t0_0 + 11) {
        cb[11] = 4286578688u;
    }
    if (lim <= t0_0 + 12) {
        cb[12] = 4286578688u;
    }
    if (lim <= t0_0 + 13) {
        cb[13] = 4286578688u;
    }
    if (lim <= t0_0 + 14) {
        cb[14] = 4286578688u;
    }
    if (lim <= t0_0 + 15) {
        cb[15] = 4286578688u;
    }
    int t0_1 = w * 16;
    float sc = __uint_as_float(cb[0]);
    sc = fmaxf(sc, -1.7014118e+38f);
    sc = fminf(sc, 1.7014118e+38f);
    unsigned int key = __as_u32(sc) & 4294966784u | (unsigned int)t0_1;
    kb[0] = __uint_as_float(key);
    float sc_2 = __uint_as_float(cb[1]);
    sc_2 = fmaxf(sc_2, -1.7014118e+38f);
    sc_2 = fminf(sc_2, 1.7014118e+38f);
    unsigned int key_3 = __as_u32(sc_2) & 4294966784u | (unsigned int)(t0_1 + 1);
    kb[1] = __uint_as_float(key_3);
    float sc_4 = __uint_as_float(cb[2]);
    sc_4 = fmaxf(sc_4, -1.7014118e+38f);
    sc_4 = fminf(sc_4, 1.7014118e+38f);
    unsigned int key_5 = __as_u32(sc_4) & 4294966784u | (unsigned int)(t0_1 + 2);
    kb[2] = __uint_as_float(key_5);
    float sc_6 = __uint_as_float(cb[3]);
    sc_6 = fmaxf(sc_6, -1.7014118e+38f);
    sc_6 = fminf(sc_6, 1.7014118e+38f);
    unsigned int key_7 = __as_u32(sc_6) & 4294966784u | (unsigned int)(t0_1 + 3);
    kb[3] = __uint_as_float(key_7);
    float sc_8 = __uint_as_float(cb[4]);
    sc_8 = fmaxf(sc_8, -1.7014118e+38f);
    sc_8 = fminf(sc_8, 1.7014118e+38f);
    unsigned int key_9 = __as_u32(sc_8) & 4294966784u | (unsigned int)(t0_1 + 4);
    kb[4] = __uint_as_float(key_9);
    float sc_10 = __uint_as_float(cb[5]);
    sc_10 = fmaxf(sc_10, -1.7014118e+38f);
    sc_10 = fminf(sc_10, 1.7014118e+38f);
    unsigned int key_11 = __as_u32(sc_10) & 4294966784u | (unsigned int)(t0_1 + 5);
    kb[5] = __uint_as_float(key_11);
    float sc_12 = __uint_as_float(cb[6]);
    sc_12 = fmaxf(sc_12, -1.7014118e+38f);
    sc_12 = fminf(sc_12, 1.7014118e+38f);
    unsigned int key_13 = __as_u32(sc_12) & 4294966784u | (unsigned int)(t0_1 + 6);
    kb[6] = __uint_as_float(key_13);
    float sc_14 = __uint_as_float(cb[7]);
    sc_14 = fmaxf(sc_14, -1.7014118e+38f);
    sc_14 = fminf(sc_14, 1.7014118e+38f);
    unsigned int key_15 = __as_u32(sc_14) & 4294966784u | (unsigned int)(t0_1 + 7);
    kb[7] = __uint_as_float(key_15);
    float sc_16 = __uint_as_float(cb[8]);
    sc_16 = fmaxf(sc_16, -1.7014118e+38f);
    sc_16 = fminf(sc_16, 1.7014118e+38f);
    unsigned int key_17 = __as_u32(sc_16) & 4294966784u | (unsigned int)(t0_1 + 8);
    kb[8] = __uint_as_float(key_17);
    float sc_18 = __uint_as_float(cb[9]);
    sc_18 = fmaxf(sc_18, -1.7014118e+38f);
    sc_18 = fminf(sc_18, 1.7014118e+38f);
    unsigned int key_19 = __as_u32(sc_18) & 4294966784u | (unsigned int)(t0_1 + 9);
    kb[9] = __uint_as_float(key_19);
    float sc_20 = __uint_as_float(cb[10]);
    sc_20 = fmaxf(sc_20, -1.7014118e+38f);
    sc_20 = fminf(sc_20, 1.7014118e+38f);
    unsigned int key_21 = __as_u32(sc_20) & 4294966784u | (unsigned int)(t0_1 + 10);
    kb[10] = __uint_as_float(key_21);
    float sc_22 = __uint_as_float(cb[11]);
    sc_22 = fmaxf(sc_22, -1.7014118e+38f);
    sc_22 = fminf(sc_22, 1.7014118e+38f);
    unsigned int key_23 = __as_u32(sc_22) & 4294966784u | (unsigned int)(t0_1 + 11);
    kb[11] = __uint_as_float(key_23);
    float sc_24 = __uint_as_float(cb[12]);
    sc_24 = fmaxf(sc_24, -1.7014118e+38f);
    sc_24 = fminf(sc_24, 1.7014118e+38f);
    unsigned int key_25 = __as_u32(sc_24) & 4294966784u | (unsigned int)(t0_1 + 12);
    kb[12] = __uint_as_float(key_25);
    float sc_26 = __uint_as_float(cb[13]);
    sc_26 = fmaxf(sc_26, -1.7014118e+38f);
    sc_26 = fminf(sc_26, 1.7014118e+38f);
    unsigned int key_27 = __as_u32(sc_26) & 4294966784u | (unsigned int)(t0_1 + 13);
    kb[13] = __uint_as_float(key_27);
    float sc_28 = __uint_as_float(cb[14]);
    sc_28 = fmaxf(sc_28, -1.7014118e+38f);
    sc_28 = fminf(sc_28, 1.7014118e+38f);
    unsigned int key_29 = __as_u32(sc_28) & 4294966784u | (unsigned int)(t0_1 + 14);
    kb[14] = __uint_as_float(key_29);
    float sc_30 = __uint_as_float(cb[15]);
    sc_30 = fmaxf(sc_30, -1.7014118e+38f);
    sc_30 = fminf(sc_30, 1.7014118e+38f);
    unsigned int key_31 = __as_u32(sc_30) & 4294966784u | (unsigned int)(t0_1 + 15);
    kb[15] = __uint_as_float(key_31);
    int f = 0;
    if ((t0_1 < fb || t0_1 + 16 > lim - fe) && t0_1 < lim) {
        f = 1;
    }
    if (f != 0) {
        int f_0 = 0;
        if ((t0_1 < fb || t0_1 >= lim - fe) && lim > t0_1) {
            f_0 = 1;
        }
        if (f_0 != 0) {
            kb[0] = __uint_as_float(2139094528 | (unsigned int)t0_1);
        }
        int f_1 = 0;
        if ((t0_1 + 1 < fb || t0_1 + 1 >= lim - fe) && lim > t0_1 + 1) {
            f_1 = 1;
        }
        if (f_1 != 0) {
            kb[1] = __uint_as_float(2139094528 | (unsigned int)(t0_1 + 1));
        }
        int f_2 = 0;
        if ((t0_1 + 2 < fb || t0_1 + 2 >= lim - fe) && lim > t0_1 + 2) {
            f_2 = 1;
        }
        if (f_2 != 0) {
            kb[2] = __uint_as_float(2139094528 | (unsigned int)(t0_1 + 2));
        }
        int f_3 = 0;
        if ((t0_1 + 3 < fb || t0_1 + 3 >= lim - fe) && lim > t0_1 + 3) {
            f_3 = 1;
        }
        if (f_3 != 0) {
            kb[3] = __uint_as_float(2139094528 | (unsigned int)(t0_1 + 3));
        }
        int f_4 = 0;
        if ((t0_1 + 4 < fb || t0_1 + 4 >= lim - fe) && lim > t0_1 + 4) {
            f_4 = 1;
        }
        if (f_4 != 0) {
            kb[4] = __uint_as_float(2139094528 | (unsigned int)(t0_1 + 4));
        }
        int f_5 = 0;
        if ((t0_1 + 5 < fb || t0_1 + 5 >= lim - fe) && lim > t0_1 + 5) {
            f_5 = 1;
        }
        if (f_5 != 0) {
            kb[5] = __uint_as_float(2139094528 | (unsigned int)(t0_1 + 5));
        }
        int f_6 = 0;
        if ((t0_1 + 6 < fb || t0_1 + 6 >= lim - fe) && lim > t0_1 + 6) {
            f_6 = 1;
        }
        if (f_6 != 0) {
            kb[6] = __uint_as_float(2139094528 | (unsigned int)(t0_1 + 6));
        }
        int f_7 = 0;
        if ((t0_1 + 7 < fb || t0_1 + 7 >= lim - fe) && lim > t0_1 + 7) {
            f_7 = 1;
        }
        if (f_7 != 0) {
            kb[7] = __uint_as_float(2139094528 | (unsigned int)(t0_1 + 7));
        }
        int f_8 = 0;
        if ((t0_1 + 8 < fb || t0_1 + 8 >= lim - fe) && lim > t0_1 + 8) {
            f_8 = 1;
        }
        if (f_8 != 0) {
            kb[8] = __uint_as_float(2139094528 | (unsigned int)(t0_1 + 8));
        }
        int f_9 = 0;
        if ((t0_1 + 9 < fb || t0_1 + 9 >= lim - fe) && lim > t0_1 + 9) {
            f_9 = 1;
        }
        if (f_9 != 0) {
            kb[9] = __uint_as_float(2139094528 | (unsigned int)(t0_1 + 9));
        }
        int f_10 = 0;
        if ((t0_1 + 10 < fb || t0_1 + 10 >= lim - fe) && lim > t0_1 + 10) {
            f_10 = 1;
        }
        if (f_10 != 0) {
            kb[10] = __uint_as_float(2139094528 | (unsigned int)(t0_1 + 10));
        }
        int f_11 = 0;
        if ((t0_1 + 11 < fb || t0_1 + 11 >= lim - fe) && lim > t0_1 + 11) {
            f_11 = 1;
        }
        if (f_11 != 0) {
            kb[11] = __uint_as_float(2139094528 | (unsigned int)(t0_1 + 11));
        }
        int f_12 = 0;
        if ((t0_1 + 12 < fb || t0_1 + 12 >= lim - fe) && lim > t0_1 + 12) {
            f_12 = 1;
        }
        if (f_12 != 0) {
            kb[12] = __uint_as_float(2139094528 | (unsigned int)(t0_1 + 12));
        }
        int f_13 = 0;
        if ((t0_1 + 13 < fb || t0_1 + 13 >= lim - fe) && lim > t0_1 + 13) {
            f_13 = 1;
        }
        if (f_13 != 0) {
            kb[13] = __uint_as_float(2139094528 | (unsigned int)(t0_1 + 13));
        }
        int f_14 = 0;
        if ((t0_1 + 14 < fb || t0_1 + 14 >= lim - fe) && lim > t0_1 + 14) {
            f_14 = 1;
        }
        if (f_14 != 0) {
            kb[14] = __uint_as_float(2139094528 | (unsigned int)(t0_1 + 14));
        }
        int f_15 = 0;
        if ((t0_1 + 15 < fb || t0_1 + 15 >= lim - fe) && lim > t0_1 + 15) {
            f_15 = 1;
        }
        if (f_15 != 0) {
            kb[15] = __uint_as_float(2139094528 | (unsigned int)(t0_1 + 15));
        }
    }
    float hi = fmaxf(kb[0], kb[13]);
    float lo = fminf(kb[0], kb[13]);
    kb[0] = hi;
    kb[13] = lo;
    float hi_32 = fmaxf(kb[1], kb[12]);
    float lo_33 = fminf(kb[1], kb[12]);
    kb[1] = hi_32;
    kb[12] = lo_33;
    float hi_34 = fmaxf(kb[2], kb[15]);
    float lo_35 = fminf(kb[2], kb[15]);
    kb[2] = hi_34;
    kb[15] = lo_35;
    float hi_36 = fmaxf(kb[3], kb[14]);
    float lo_37 = fminf(kb[3], kb[14]);
    kb[3] = hi_36;
    kb[14] = lo_37;
    float hi_38 = fmaxf(kb[4], kb[8]);
    float lo_39 = fminf(kb[4], kb[8]);
    kb[4] = hi_38;
    kb[8] = lo_39;
    float hi_40 = fmaxf(kb[5], kb[6]);
    float lo_41 = fminf(kb[5], kb[6]);
    kb[5] = hi_40;
    kb[6] = lo_41;
    float hi_42 = fmaxf(kb[7], kb[11]);
    float lo_43 = fminf(kb[7], kb[11]);
    kb[7] = hi_42;
    kb[11] = lo_43;
    float hi_44 = fmaxf(kb[9], kb[10]);
    float lo_45 = fminf(kb[9], kb[10]);
    kb[9] = hi_44;
    kb[10] = lo_45;
    float hi_46 = fmaxf(kb[0], kb[5]);
    float lo_47 = fminf(kb[0], kb[5]);
    kb[0] = hi_46;
    kb[5] = lo_47;
    float hi_48 = fmaxf(kb[1], kb[7]);
    float lo_49 = fminf(kb[1], kb[7]);
    kb[1] = hi_48;
    kb[7] = lo_49;
    float hi_50 = fmaxf(kb[2], kb[9]);
    float lo_51 = fminf(kb[2], kb[9]);
    kb[2] = hi_50;
    kb[9] = lo_51;
    float hi_52 = fmaxf(kb[3], kb[4]);
    float lo_53 = fminf(kb[3], kb[4]);
    kb[3] = hi_52;
    kb[4] = lo_53;
    float hi_54 = fmaxf(kb[6], kb[13]);
    float lo_55 = fminf(kb[6], kb[13]);
    kb[6] = hi_54;
    kb[13] = lo_55;
    float hi_56 = fmaxf(kb[8], kb[14]);
    float lo_57 = fminf(kb[8], kb[14]);
    kb[8] = hi_56;
    kb[14] = lo_57;
    float hi_58 = fmaxf(kb[10], kb[15]);
    float lo_59 = fminf(kb[10], kb[15]);
    kb[10] = hi_58;
    kb[15] = lo_59;
    float hi_60 = fmaxf(kb[11], kb[12]);
    float lo_61 = fminf(kb[11], kb[12]);
    kb[11] = hi_60;
    kb[12] = lo_61;
    float hi_62 = fmaxf(kb[0], kb[1]);
    float lo_63 = fminf(kb[0], kb[1]);
    kb[0] = hi_62;
    kb[1] = lo_63;
    float hi_64 = fmaxf(kb[2], kb[3]);
    float lo_65 = fminf(kb[2], kb[3]);
    kb[2] = hi_64;
    kb[3] = lo_65;
    float hi_66 = fmaxf(kb[4], kb[5]);
    float lo_67 = fminf(kb[4], kb[5]);
    kb[4] = hi_66;
    kb[5] = lo_67;
    float hi_68 = fmaxf(kb[6], kb[8]);
    float lo_69 = fminf(kb[6], kb[8]);
    kb[6] = hi_68;
    kb[8] = lo_69;
    float hi_70 = fmaxf(kb[7], kb[9]);
    float lo_71 = fminf(kb[7], kb[9]);
    kb[7] = hi_70;
    kb[9] = lo_71;
    float hi_72 = fmaxf(kb[10], kb[11]);
    float lo_73 = fminf(kb[10], kb[11]);
    kb[10] = hi_72;
    kb[11] = lo_73;
    float hi_74 = fmaxf(kb[12], kb[13]);
    float lo_75 = fminf(kb[12], kb[13]);
    kb[12] = hi_74;
    kb[13] = lo_75;
    float hi_76 = fmaxf(kb[14], kb[15]);
    float lo_77 = fminf(kb[14], kb[15]);
    kb[14] = hi_76;
    kb[15] = lo_77;
    float hi_78 = fmaxf(kb[0], kb[2]);
    float lo_79 = fminf(kb[0], kb[2]);
    kb[0] = hi_78;
    kb[2] = lo_79;
    float hi_80 = fmaxf(kb[1], kb[3]);
    float lo_81 = fminf(kb[1], kb[3]);
    kb[1] = hi_80;
    kb[3] = lo_81;
    float hi_82 = fmaxf(kb[4], kb[10]);
    float lo_83 = fminf(kb[4], kb[10]);
    kb[4] = hi_82;
    kb[10] = lo_83;
    float hi_84 = fmaxf(kb[5], kb[11]);
    float lo_85 = fminf(kb[5], kb[11]);
    kb[5] = hi_84;
    kb[11] = lo_85;
    float hi_86 = fmaxf(kb[6], kb[7]);
    float lo_87 = fminf(kb[6], kb[7]);
    kb[6] = hi_86;
    kb[7] = lo_87;
    float hi_88 = fmaxf(kb[8], kb[9]);
    float lo_89 = fminf(kb[8], kb[9]);
    kb[8] = hi_88;
    kb[9] = lo_89;
    float hi_90 = fmaxf(kb[12], kb[14]);
    float lo_91 = fminf(kb[12], kb[14]);
    kb[12] = hi_90;
    kb[14] = lo_91;
    float hi_92 = fmaxf(kb[13], kb[15]);
    float lo_93 = fminf(kb[13], kb[15]);
    kb[13] = hi_92;
    kb[15] = lo_93;
    float hi_94 = fmaxf(kb[1], kb[2]);
    float lo_95 = fminf(kb[1], kb[2]);
    kb[1] = hi_94;
    kb[2] = lo_95;
    float hi_96 = fmaxf(kb[3], kb[12]);
    float lo_97 = fminf(kb[3], kb[12]);
    kb[3] = hi_96;
    kb[12] = lo_97;
    float hi_98 = fmaxf(kb[4], kb[6]);
    float lo_99 = fminf(kb[4], kb[6]);
    kb[4] = hi_98;
    kb[6] = lo_99;
    float hi_100 = fmaxf(kb[5], kb[7]);
    float lo_101 = fminf(kb[5], kb[7]);
    kb[5] = hi_100;
    kb[7] = lo_101;
    float hi_102 = fmaxf(kb[8], kb[10]);
    float lo_103 = fminf(kb[8], kb[10]);
    kb[8] = hi_102;
    kb[10] = lo_103;
    float hi_104 = fmaxf(kb[9], kb[11]);
    float lo_105 = fminf(kb[9], kb[11]);
    kb[9] = hi_104;
    kb[11] = lo_105;
    float hi_106 = fmaxf(kb[13], kb[14]);
    float lo_107 = fminf(kb[13], kb[14]);
    kb[13] = hi_106;
    kb[14] = lo_107;
    float hi_108 = fmaxf(kb[1], kb[4]);
    float lo_109 = fminf(kb[1], kb[4]);
    kb[1] = hi_108;
    kb[4] = lo_109;
    float hi_110 = fmaxf(kb[2], kb[6]);
    float lo_111 = fminf(kb[2], kb[6]);
    kb[2] = hi_110;
    kb[6] = lo_111;
    float hi_112 = fmaxf(kb[5], kb[8]);
    float lo_113 = fminf(kb[5], kb[8]);
    kb[5] = hi_112;
    kb[8] = lo_113;
    float hi_114 = fmaxf(kb[7], kb[10]);
    float lo_115 = fminf(kb[7], kb[10]);
    kb[7] = hi_114;
    kb[10] = lo_115;
    float hi_116 = fmaxf(kb[9], kb[13]);
    float lo_117 = fminf(kb[9], kb[13]);
    kb[9] = hi_116;
    kb[13] = lo_117;
    float hi_118 = fmaxf(kb[11], kb[14]);
    float lo_119 = fminf(kb[11], kb[14]);
    kb[11] = hi_118;
    kb[14] = lo_119;
    float hi_120 = fmaxf(kb[2], kb[4]);
    float lo_121 = fminf(kb[2], kb[4]);
    kb[2] = hi_120;
    kb[4] = lo_121;
    float hi_122 = fmaxf(kb[3], kb[6]);
    float lo_123 = fminf(kb[3], kb[6]);
    kb[3] = hi_122;
    kb[6] = lo_123;
    float hi_124 = fmaxf(kb[9], kb[12]);
    float lo_125 = fminf(kb[9], kb[12]);
    kb[9] = hi_124;
    kb[12] = lo_125;
    float hi_126 = fmaxf(kb[11], kb[13]);
    float lo_127 = fminf(kb[11], kb[13]);
    kb[11] = hi_126;
    kb[13] = lo_127;
    float hi_128 = fmaxf(kb[3], kb[5]);
    float lo_129 = fminf(kb[3], kb[5]);
    kb[3] = hi_128;
    kb[5] = lo_129;
    float hi_130 = fmaxf(kb[6], kb[8]);
    float lo_131 = fminf(kb[6], kb[8]);
    kb[6] = hi_130;
    kb[8] = lo_131;
    float hi_132 = fmaxf(kb[7], kb[9]);
    float lo_133 = fminf(kb[7], kb[9]);
    kb[7] = hi_132;
    kb[9] = lo_133;
    float hi_134 = fmaxf(kb[10], kb[12]);
    float lo_135 = fminf(kb[10], kb[12]);
    kb[10] = hi_134;
    kb[12] = lo_135;
    float hi_136 = fmaxf(kb[3], kb[4]);
    float lo_137 = fminf(kb[3], kb[4]);
    kb[3] = hi_136;
    kb[4] = lo_137;
    float hi_138 = fmaxf(kb[5], kb[6]);
    float lo_139 = fminf(kb[5], kb[6]);
    kb[5] = hi_138;
    kb[6] = lo_139;
    float hi_140 = fmaxf(kb[7], kb[8]);
    float lo_141 = fminf(kb[7], kb[8]);
    kb[7] = hi_140;
    kb[8] = lo_141;
    float hi_142 = fmaxf(kb[9], kb[10]);
    float lo_143 = fminf(kb[9], kb[10]);
    kb[9] = hi_142;
    kb[10] = lo_143;
    float hi_144 = fmaxf(kb[11], kb[12]);
    float lo_145 = fminf(kb[11], kb[12]);
    kb[11] = hi_144;
    kb[12] = lo_145;
    float hi_146 = fmaxf(kb[6], kb[7]);
    float lo_147 = fminf(kb[6], kb[7]);
    kb[6] = hi_146;
    kb[7] = lo_147;
    float hi_148 = fmaxf(kb[8], kb[9]);
    float lo_149 = fminf(kb[8], kb[9]);
    kb[8] = hi_148;
    kb[9] = lo_149;
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
    asm volatile("barrier.sync 1, 64;" ::: "memory");
    float r = neg_inf;
    float V[8];
    int s0 = (sg * 16 * 2 + cg) * 17;
    int s1 = ((sg * 16 + 8) * 2 + cg) * 17;
    float x0 = pub[s0 + ln];
    float y0 = pub[s1 + lnr];
    float lo0 = fminf(x0, y0);
    r = fmaxf(r, lo0);
    float hi0 = fmaxf(x0, y0);
    float cur = hi0;
    float pv = __shfl_xor_sync(0xFFFFFFFF, cur, 8);
    float hi_150 = fmaxf(cur, pv);
    float lo_151 = fminf(cur, pv);
    cur = ((up[0] != 0) ? hi_150 : lo_151);
    float pv_152 = __shfl_xor_sync(0xFFFFFFFF, cur, 4);
    float hi_153 = fmaxf(cur, pv_152);
    float lo_154 = fminf(cur, pv_152);
    cur = ((up[1] != 0) ? hi_153 : lo_154);
    float pv_155 = __shfl_xor_sync(0xFFFFFFFF, cur, 2);
    float hi_156 = fmaxf(cur, pv_155);
    float lo_157 = fminf(cur, pv_155);
    cur = ((up[2] != 0) ? hi_156 : lo_157);
    float pv_158 = __shfl_xor_sync(0xFFFFFFFF, cur, 1);
    float hi_159 = fmaxf(cur, pv_158);
    float lo_160 = fminf(cur, pv_158);
    cur = ((up[3] != 0) ? hi_159 : lo_160);
    V[0] = cur;
    int s0_161 = ((sg * 16 + 1) * 2 + cg) * 17;
    int s1_162 = ((sg * 16 + 1 + 8) * 2 + cg) * 17;
    float x0_163 = pub[s0_161 + ln];
    float y0_164 = pub[s1_162 + lnr];
    float lo0_165 = fminf(x0_163, y0_164);
    r = fmaxf(r, lo0_165);
    float hi0_166 = fmaxf(x0_163, y0_164);
    float cur_167 = hi0_166;
    float pv_168 = __shfl_xor_sync(0xFFFFFFFF, cur_167, 8);
    float hi_169 = fmaxf(cur_167, pv_168);
    float lo_170 = fminf(cur_167, pv_168);
    cur_167 = ((up[0] != 0) ? hi_169 : lo_170);
    float pv_171 = __shfl_xor_sync(0xFFFFFFFF, cur_167, 4);
    float hi_172 = fmaxf(cur_167, pv_171);
    float lo_173 = fminf(cur_167, pv_171);
    cur_167 = ((up[1] != 0) ? hi_172 : lo_173);
    float pv_174 = __shfl_xor_sync(0xFFFFFFFF, cur_167, 2);
    float hi_175 = fmaxf(cur_167, pv_174);
    float lo_176 = fminf(cur_167, pv_174);
    cur_167 = ((up[2] != 0) ? hi_175 : lo_176);
    float pv_177 = __shfl_xor_sync(0xFFFFFFFF, cur_167, 1);
    float hi_178 = fmaxf(cur_167, pv_177);
    float lo_179 = fminf(cur_167, pv_177);
    cur_167 = ((up[3] != 0) ? hi_178 : lo_179);
    V[1] = cur_167;
    int s0_180 = ((sg * 16 + 2) * 2 + cg) * 17;
    int s1_181 = ((sg * 16 + 2 + 8) * 2 + cg) * 17;
    float x0_182 = pub[s0_180 + ln];
    float y0_183 = pub[s1_181 + lnr];
    float lo0_184 = fminf(x0_182, y0_183);
    r = fmaxf(r, lo0_184);
    float hi0_185 = fmaxf(x0_182, y0_183);
    float cur_186 = hi0_185;
    float pv_187 = __shfl_xor_sync(0xFFFFFFFF, cur_186, 8);
    float hi_188 = fmaxf(cur_186, pv_187);
    float lo_189 = fminf(cur_186, pv_187);
    cur_186 = ((up[0] != 0) ? hi_188 : lo_189);
    float pv_190 = __shfl_xor_sync(0xFFFFFFFF, cur_186, 4);
    float hi_191 = fmaxf(cur_186, pv_190);
    float lo_192 = fminf(cur_186, pv_190);
    cur_186 = ((up[1] != 0) ? hi_191 : lo_192);
    float pv_193 = __shfl_xor_sync(0xFFFFFFFF, cur_186, 2);
    float hi_194 = fmaxf(cur_186, pv_193);
    float lo_195 = fminf(cur_186, pv_193);
    cur_186 = ((up[2] != 0) ? hi_194 : lo_195);
    float pv_196 = __shfl_xor_sync(0xFFFFFFFF, cur_186, 1);
    float hi_197 = fmaxf(cur_186, pv_196);
    float lo_198 = fminf(cur_186, pv_196);
    cur_186 = ((up[3] != 0) ? hi_197 : lo_198);
    V[2] = cur_186;
    int s0_199 = ((sg * 16 + 3) * 2 + cg) * 17;
    int s1_200 = ((sg * 16 + 3 + 8) * 2 + cg) * 17;
    float x0_201 = pub[s0_199 + ln];
    float y0_202 = pub[s1_200 + lnr];
    float lo0_203 = fminf(x0_201, y0_202);
    r = fmaxf(r, lo0_203);
    float hi0_204 = fmaxf(x0_201, y0_202);
    float cur_205 = hi0_204;
    float pv_206 = __shfl_xor_sync(0xFFFFFFFF, cur_205, 8);
    float hi_207 = fmaxf(cur_205, pv_206);
    float lo_208 = fminf(cur_205, pv_206);
    cur_205 = ((up[0] != 0) ? hi_207 : lo_208);
    float pv_209 = __shfl_xor_sync(0xFFFFFFFF, cur_205, 4);
    float hi_210 = fmaxf(cur_205, pv_209);
    float lo_211 = fminf(cur_205, pv_209);
    cur_205 = ((up[1] != 0) ? hi_210 : lo_211);
    float pv_212 = __shfl_xor_sync(0xFFFFFFFF, cur_205, 2);
    float hi_213 = fmaxf(cur_205, pv_212);
    float lo_214 = fminf(cur_205, pv_212);
    cur_205 = ((up[2] != 0) ? hi_213 : lo_214);
    float pv_215 = __shfl_xor_sync(0xFFFFFFFF, cur_205, 1);
    float hi_216 = fmaxf(cur_205, pv_215);
    float lo_217 = fminf(cur_205, pv_215);
    cur_205 = ((up[3] != 0) ? hi_216 : lo_217);
    V[3] = cur_205;
    int s0_218 = ((sg * 16 + 4) * 2 + cg) * 17;
    int s1_219 = ((sg * 16 + 4 + 8) * 2 + cg) * 17;
    float x0_220 = pub[s0_218 + ln];
    float y0_221 = pub[s1_219 + lnr];
    float lo0_222 = fminf(x0_220, y0_221);
    r = fmaxf(r, lo0_222);
    float hi0_223 = fmaxf(x0_220, y0_221);
    float cur_224 = hi0_223;
    float pv_225 = __shfl_xor_sync(0xFFFFFFFF, cur_224, 8);
    float hi_226 = fmaxf(cur_224, pv_225);
    float lo_227 = fminf(cur_224, pv_225);
    cur_224 = ((up[0] != 0) ? hi_226 : lo_227);
    float pv_228 = __shfl_xor_sync(0xFFFFFFFF, cur_224, 4);
    float hi_229 = fmaxf(cur_224, pv_228);
    float lo_230 = fminf(cur_224, pv_228);
    cur_224 = ((up[1] != 0) ? hi_229 : lo_230);
    float pv_231 = __shfl_xor_sync(0xFFFFFFFF, cur_224, 2);
    float hi_232 = fmaxf(cur_224, pv_231);
    float lo_233 = fminf(cur_224, pv_231);
    cur_224 = ((up[2] != 0) ? hi_232 : lo_233);
    float pv_234 = __shfl_xor_sync(0xFFFFFFFF, cur_224, 1);
    float hi_235 = fmaxf(cur_224, pv_234);
    float lo_236 = fminf(cur_224, pv_234);
    cur_224 = ((up[3] != 0) ? hi_235 : lo_236);
    V[4] = cur_224;
    int s0_237 = ((sg * 16 + 5) * 2 + cg) * 17;
    int s1_238 = ((sg * 16 + 5 + 8) * 2 + cg) * 17;
    float x0_239 = pub[s0_237 + ln];
    float y0_240 = pub[s1_238 + lnr];
    float lo0_241 = fminf(x0_239, y0_240);
    r = fmaxf(r, lo0_241);
    float hi0_242 = fmaxf(x0_239, y0_240);
    float cur_243 = hi0_242;
    float pv_244 = __shfl_xor_sync(0xFFFFFFFF, cur_243, 8);
    float hi_245 = fmaxf(cur_243, pv_244);
    float lo_246 = fminf(cur_243, pv_244);
    cur_243 = ((up[0] != 0) ? hi_245 : lo_246);
    float pv_247 = __shfl_xor_sync(0xFFFFFFFF, cur_243, 4);
    float hi_248 = fmaxf(cur_243, pv_247);
    float lo_249 = fminf(cur_243, pv_247);
    cur_243 = ((up[1] != 0) ? hi_248 : lo_249);
    float pv_250 = __shfl_xor_sync(0xFFFFFFFF, cur_243, 2);
    float hi_251 = fmaxf(cur_243, pv_250);
    float lo_252 = fminf(cur_243, pv_250);
    cur_243 = ((up[2] != 0) ? hi_251 : lo_252);
    float pv_253 = __shfl_xor_sync(0xFFFFFFFF, cur_243, 1);
    float hi_254 = fmaxf(cur_243, pv_253);
    float lo_255 = fminf(cur_243, pv_253);
    cur_243 = ((up[3] != 0) ? hi_254 : lo_255);
    V[5] = cur_243;
    int s0_256 = ((sg * 16 + 6) * 2 + cg) * 17;
    int s1_257 = ((sg * 16 + 6 + 8) * 2 + cg) * 17;
    float x0_258 = pub[s0_256 + ln];
    float y0_259 = pub[s1_257 + lnr];
    float lo0_260 = fminf(x0_258, y0_259);
    r = fmaxf(r, lo0_260);
    float hi0_261 = fmaxf(x0_258, y0_259);
    float cur_262 = hi0_261;
    float pv_263 = __shfl_xor_sync(0xFFFFFFFF, cur_262, 8);
    float hi_264 = fmaxf(cur_262, pv_263);
    float lo_265 = fminf(cur_262, pv_263);
    cur_262 = ((up[0] != 0) ? hi_264 : lo_265);
    float pv_266 = __shfl_xor_sync(0xFFFFFFFF, cur_262, 4);
    float hi_267 = fmaxf(cur_262, pv_266);
    float lo_268 = fminf(cur_262, pv_266);
    cur_262 = ((up[1] != 0) ? hi_267 : lo_268);
    float pv_269 = __shfl_xor_sync(0xFFFFFFFF, cur_262, 2);
    float hi_270 = fmaxf(cur_262, pv_269);
    float lo_271 = fminf(cur_262, pv_269);
    cur_262 = ((up[2] != 0) ? hi_270 : lo_271);
    float pv_272 = __shfl_xor_sync(0xFFFFFFFF, cur_262, 1);
    float hi_273 = fmaxf(cur_262, pv_272);
    float lo_274 = fminf(cur_262, pv_272);
    cur_262 = ((up[3] != 0) ? hi_273 : lo_274);
    V[6] = cur_262;
    int s0_275 = ((sg * 16 + 7) * 2 + cg) * 17;
    int s1_276 = ((sg * 16 + 7 + 8) * 2 + cg) * 17;
    float x0_277 = pub[s0_275 + ln];
    float y0_278 = pub[s1_276 + lnr];
    float lo0_279 = fminf(x0_277, y0_278);
    r = fmaxf(r, lo0_279);
    float hi0_280 = fmaxf(x0_277, y0_278);
    float cur_281 = hi0_280;
    float pv_282 = __shfl_xor_sync(0xFFFFFFFF, cur_281, 8);
    float hi_283 = fmaxf(cur_281, pv_282);
    float lo_284 = fminf(cur_281, pv_282);
    cur_281 = ((up[0] != 0) ? hi_283 : lo_284);
    float pv_285 = __shfl_xor_sync(0xFFFFFFFF, cur_281, 4);
    float hi_286 = fmaxf(cur_281, pv_285);
    float lo_287 = fminf(cur_281, pv_285);
    cur_281 = ((up[1] != 0) ? hi_286 : lo_287);
    float pv_288 = __shfl_xor_sync(0xFFFFFFFF, cur_281, 2);
    float hi_289 = fmaxf(cur_281, pv_288);
    float lo_290 = fminf(cur_281, pv_288);
    cur_281 = ((up[2] != 0) ? hi_289 : lo_290);
    float pv_291 = __shfl_xor_sync(0xFFFFFFFF, cur_281, 1);
    float hi_292 = fmaxf(cur_281, pv_291);
    float lo_293 = fminf(cur_281, pv_291);
    cur_281 = ((up[3] != 0) ? hi_292 : lo_293);
    V[7] = cur_281;
    float rs = pub[((sg * 16 + ln) * 2 + cg) * 17 + 16];
    r = fmaxf(r, rs);
    float y1 = __shfl_xor_sync(0xFFFFFFFF, V[4], 15);
    float lo1 = fminf(V[0], y1);
    r = fmaxf(r, lo1);
    float hi1 = fmaxf(V[0], y1);
    float cur_294 = hi1;
    float pv_295 = __shfl_xor_sync(0xFFFFFFFF, cur_294, 8);
    float hi_296 = fmaxf(cur_294, pv_295);
    float lo_297 = fminf(cur_294, pv_295);
    cur_294 = ((up[0] != 0) ? hi_296 : lo_297);
    float pv_298 = __shfl_xor_sync(0xFFFFFFFF, cur_294, 4);
    float hi_299 = fmaxf(cur_294, pv_298);
    float lo_300 = fminf(cur_294, pv_298);
    cur_294 = ((up[1] != 0) ? hi_299 : lo_300);
    float pv_301 = __shfl_xor_sync(0xFFFFFFFF, cur_294, 2);
    float hi_302 = fmaxf(cur_294, pv_301);
    float lo_303 = fminf(cur_294, pv_301);
    cur_294 = ((up[2] != 0) ? hi_302 : lo_303);
    float pv_304 = __shfl_xor_sync(0xFFFFFFFF, cur_294, 1);
    float hi_305 = fmaxf(cur_294, pv_304);
    float lo_306 = fminf(cur_294, pv_304);
    cur_294 = ((up[3] != 0) ? hi_305 : lo_306);
    V[0] = cur_294;
    float y1_307 = __shfl_xor_sync(0xFFFFFFFF, V[5], 15);
    float lo1_308 = fminf(V[1], y1_307);
    r = fmaxf(r, lo1_308);
    float hi1_309 = fmaxf(V[1], y1_307);
    float cur_310 = hi1_309;
    float pv_311 = __shfl_xor_sync(0xFFFFFFFF, cur_310, 8);
    float hi_312 = fmaxf(cur_310, pv_311);
    float lo_313 = fminf(cur_310, pv_311);
    cur_310 = ((up[0] != 0) ? hi_312 : lo_313);
    float pv_314 = __shfl_xor_sync(0xFFFFFFFF, cur_310, 4);
    float hi_315 = fmaxf(cur_310, pv_314);
    float lo_316 = fminf(cur_310, pv_314);
    cur_310 = ((up[1] != 0) ? hi_315 : lo_316);
    float pv_317 = __shfl_xor_sync(0xFFFFFFFF, cur_310, 2);
    float hi_318 = fmaxf(cur_310, pv_317);
    float lo_319 = fminf(cur_310, pv_317);
    cur_310 = ((up[2] != 0) ? hi_318 : lo_319);
    float pv_320 = __shfl_xor_sync(0xFFFFFFFF, cur_310, 1);
    float hi_321 = fmaxf(cur_310, pv_320);
    float lo_322 = fminf(cur_310, pv_320);
    cur_310 = ((up[3] != 0) ? hi_321 : lo_322);
    V[1] = cur_310;
    float y1_323 = __shfl_xor_sync(0xFFFFFFFF, V[6], 15);
    float lo1_324 = fminf(V[2], y1_323);
    r = fmaxf(r, lo1_324);
    float hi1_325 = fmaxf(V[2], y1_323);
    float cur_326 = hi1_325;
    float pv_327 = __shfl_xor_sync(0xFFFFFFFF, cur_326, 8);
    float hi_328 = fmaxf(cur_326, pv_327);
    float lo_329 = fminf(cur_326, pv_327);
    cur_326 = ((up[0] != 0) ? hi_328 : lo_329);
    float pv_330 = __shfl_xor_sync(0xFFFFFFFF, cur_326, 4);
    float hi_331 = fmaxf(cur_326, pv_330);
    float lo_332 = fminf(cur_326, pv_330);
    cur_326 = ((up[1] != 0) ? hi_331 : lo_332);
    float pv_333 = __shfl_xor_sync(0xFFFFFFFF, cur_326, 2);
    float hi_334 = fmaxf(cur_326, pv_333);
    float lo_335 = fminf(cur_326, pv_333);
    cur_326 = ((up[2] != 0) ? hi_334 : lo_335);
    float pv_336 = __shfl_xor_sync(0xFFFFFFFF, cur_326, 1);
    float hi_337 = fmaxf(cur_326, pv_336);
    float lo_338 = fminf(cur_326, pv_336);
    cur_326 = ((up[3] != 0) ? hi_337 : lo_338);
    V[2] = cur_326;
    float y1_339 = __shfl_xor_sync(0xFFFFFFFF, V[7], 15);
    float lo1_340 = fminf(V[3], y1_339);
    r = fmaxf(r, lo1_340);
    float hi1_341 = fmaxf(V[3], y1_339);
    float cur_342 = hi1_341;
    float pv_343 = __shfl_xor_sync(0xFFFFFFFF, cur_342, 8);
    float hi_344 = fmaxf(cur_342, pv_343);
    float lo_345 = fminf(cur_342, pv_343);
    cur_342 = ((up[0] != 0) ? hi_344 : lo_345);
    float pv_346 = __shfl_xor_sync(0xFFFFFFFF, cur_342, 4);
    float hi_347 = fmaxf(cur_342, pv_346);
    float lo_348 = fminf(cur_342, pv_346);
    cur_342 = ((up[1] != 0) ? hi_347 : lo_348);
    float pv_349 = __shfl_xor_sync(0xFFFFFFFF, cur_342, 2);
    float hi_350 = fmaxf(cur_342, pv_349);
    float lo_351 = fminf(cur_342, pv_349);
    cur_342 = ((up[2] != 0) ? hi_350 : lo_351);
    float pv_352 = __shfl_xor_sync(0xFFFFFFFF, cur_342, 1);
    float hi_353 = fmaxf(cur_342, pv_352);
    float lo_354 = fminf(cur_342, pv_352);
    cur_342 = ((up[3] != 0) ? hi_353 : lo_354);
    V[3] = cur_342;
    float y1_355 = __shfl_xor_sync(0xFFFFFFFF, V[2], 15);
    float lo1_356 = fminf(V[0], y1_355);
    r = fmaxf(r, lo1_356);
    float hi1_357 = fmaxf(V[0], y1_355);
    float cur_358 = hi1_357;
    float pv_359 = __shfl_xor_sync(0xFFFFFFFF, cur_358, 8);
    float hi_360 = fmaxf(cur_358, pv_359);
    float lo_361 = fminf(cur_358, pv_359);
    cur_358 = ((up[0] != 0) ? hi_360 : lo_361);
    float pv_362 = __shfl_xor_sync(0xFFFFFFFF, cur_358, 4);
    float hi_363 = fmaxf(cur_358, pv_362);
    float lo_364 = fminf(cur_358, pv_362);
    cur_358 = ((up[1] != 0) ? hi_363 : lo_364);
    float pv_365 = __shfl_xor_sync(0xFFFFFFFF, cur_358, 2);
    float hi_366 = fmaxf(cur_358, pv_365);
    float lo_367 = fminf(cur_358, pv_365);
    cur_358 = ((up[2] != 0) ? hi_366 : lo_367);
    float pv_368 = __shfl_xor_sync(0xFFFFFFFF, cur_358, 1);
    float hi_369 = fmaxf(cur_358, pv_368);
    float lo_370 = fminf(cur_358, pv_368);
    cur_358 = ((up[3] != 0) ? hi_369 : lo_370);
    V[0] = cur_358;
    float y1_371 = __shfl_xor_sync(0xFFFFFFFF, V[3], 15);
    float lo1_372 = fminf(V[1], y1_371);
    r = fmaxf(r, lo1_372);
    float hi1_373 = fmaxf(V[1], y1_371);
    float cur_374 = hi1_373;
    float pv_375 = __shfl_xor_sync(0xFFFFFFFF, cur_374, 8);
    float hi_376 = fmaxf(cur_374, pv_375);
    float lo_377 = fminf(cur_374, pv_375);
    cur_374 = ((up[0] != 0) ? hi_376 : lo_377);
    float pv_378 = __shfl_xor_sync(0xFFFFFFFF, cur_374, 4);
    float hi_379 = fmaxf(cur_374, pv_378);
    float lo_380 = fminf(cur_374, pv_378);
    cur_374 = ((up[1] != 0) ? hi_379 : lo_380);
    float pv_381 = __shfl_xor_sync(0xFFFFFFFF, cur_374, 2);
    float hi_382 = fmaxf(cur_374, pv_381);
    float lo_383 = fminf(cur_374, pv_381);
    cur_374 = ((up[2] != 0) ? hi_382 : lo_383);
    float pv_384 = __shfl_xor_sync(0xFFFFFFFF, cur_374, 1);
    float hi_385 = fmaxf(cur_374, pv_384);
    float lo_386 = fminf(cur_374, pv_384);
    cur_374 = ((up[3] != 0) ? hi_385 : lo_386);
    V[1] = cur_374;
    float yl = __shfl_xor_sync(0xFFFFFFFF, V[1], 15);
    float lol = fminf(V[0], yl);
    r = fmaxf(r, lol);
    float hil = fmaxf(V[0], yl);
    float cur_387 = hil;
    float pv_388 = __shfl_xor_sync(0xFFFFFFFF, cur_387, 8);
    float hi_389 = fmaxf(cur_387, pv_388);
    float lo_390 = fminf(cur_387, pv_388);
    cur_387 = ((up[0] != 0) ? hi_389 : lo_390);
    float pv_391 = __shfl_xor_sync(0xFFFFFFFF, cur_387, 4);
    float hi_392 = fmaxf(cur_387, pv_391);
    float lo_393 = fminf(cur_387, pv_391);
    cur_387 = ((up[1] != 0) ? hi_392 : lo_393);
    float pv_394 = __shfl_xor_sync(0xFFFFFFFF, cur_387, 2);
    float hi_395 = fmaxf(cur_387, pv_394);
    float lo_396 = fminf(cur_387, pv_394);
    cur_387 = ((up[2] != 0) ? hi_395 : lo_396);
    float pv_397 = __shfl_xor_sync(0xFFFFFFFF, cur_387, 1);
    float hi_398 = fmaxf(cur_387, pv_397);
    float lo_399 = fminf(cur_387, pv_397);
    cur_387 = ((up[3] != 0) ? hi_398 : lo_399);
    V[0] = cur_387;
    float K = V[0];
    int qb = (sg * 2 + cg) * 32;
    q2[qb + ln] = K;
    q2[qb + 16 + ln] = r;
    asm volatile("barrier.sync 1, 64;" ::: "memory");
    float k2o[1];
    float r2v = neg_inf;
    r2v = fmaxf(r2v, q2[cg * 32 + 16 + ln]);
    r2v = fmaxf(r2v, q2[(2 + cg) * 32 + 16 + ln]);
    float V2[1];
    float x2 = q2[cg * 32 + ln];
    float y2 = q2[(2 + cg) * 32 + lnr];
    float lo2 = fminf(x2, y2);
    r2v = fmaxf(r2v, lo2);
    float hi2 = fmaxf(x2, y2);
    V2[0] = hi2;
    k2o[0] = V2[0];
    K = k2o[0];
    r = r2v;
    rr1[0] = r;
    int ucol = blockIdx.x * 2 + cg;
    int commit = 0;
    if (g < 2 && ucol < total_q) {
        commit = 1;
    }
    float u16 = K;
    float cr = rr1[0];
    float _shfl_xor_sync_0 = __shfl_xor_sync(0xFFFFFFFF, u16, 8);
    u16 = fminf(u16, _shfl_xor_sync_0);
    float _shfl_xor_sync_1 = __shfl_xor_sync(0xFFFFFFFF, cr, 8);
    cr = fmaxf(cr, _shfl_xor_sync_1);
    float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, u16, 4);
    u16 = fminf(u16, _shfl_xor_sync_2);
    float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, cr, 4);
    cr = fmaxf(cr, _shfl_xor_sync_3);
    float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, u16, 2);
    u16 = fminf(u16, _shfl_xor_sync_4);
    float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, cr, 2);
    cr = fmaxf(cr, _shfl_xor_sync_5);
    float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, u16, 1);
    u16 = fminf(u16, _shfl_xor_sync_6);
    float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, cr, 1);
    cr = fmaxf(cr, _shfl_xor_sync_7);
    unsigned int u16b = __as_u32(u16);
    unsigned int c16 = u16b & 4294966784u;
    unsigned int crb = __as_u32(cr) & 4294966784u;
    unsigned int q = 4294967295u;
    if (c16 == crb && u16b < 4278190080u && c16 != 2139094528 && ucol < total_q) {
        q = c16;
    }
    unsigned int qo5 = __shfl_xor_sync(0xFFFFFFFF, q, 16);
    unsigned int need2 = 0;
    if (q != 4294967295u || qo5 != 4294967295u) {
        need2 = 1;
    }
    unsigned int qc = qo5;
    if (cg == c) {
        qc = q;
    }
    int flagged = 0;
    if (qc != 4294967295u) {
        flagged = 1;
    }
    unsigned int qg = q;
    int gflag = 0;
    if (qg != 4294967295u) {
        gflag = 1;
    }
    float K2 = 0.0f;
    if (need2 != 0) {
        if (flagged != 0) {
            int anyq = 0;
            if ((__as_u32(a[0]) & 4294966784u) == qc) {
                anyq = 1;
            }
            if ((__as_u32(a[1]) & 4294966784u) == qc) {
                anyq = 1;
            }
            if ((__as_u32(a[2]) & 4294966784u) == qc) {
                anyq = 1;
            }
            if ((__as_u32(a[3]) & 4294966784u) == qc) {
                anyq = 1;
            }
            if ((__as_u32(a[4]) & 4294966784u) == qc) {
                anyq = 1;
            }
            if ((__as_u32(a[5]) & 4294966784u) == qc) {
                anyq = 1;
            }
            if ((__as_u32(a[6]) & 4294966784u) == qc) {
                anyq = 1;
            }
            if ((__as_u32(a[7]) & 4294966784u) == qc) {
                anyq = 1;
            }
            if ((__as_u32(a[8]) & 4294966784u) == qc) {
                anyq = 1;
            }
            if ((__as_u32(a[9]) & 4294966784u) == qc) {
                anyq = 1;
            }
            if ((__as_u32(a[10]) & 4294966784u) == qc) {
                anyq = 1;
            }
            if ((__as_u32(a[11]) & 4294966784u) == qc) {
                anyq = 1;
            }
            if ((__as_u32(a[12]) & 4294966784u) == qc) {
                anyq = 1;
            }
            if ((__as_u32(a[13]) & 4294966784u) == qc) {
                anyq = 1;
            }
            if ((__as_u32(a[14]) & 4294966784u) == qc) {
                anyq = 1;
            }
            if ((__as_u32(a[15]) & 4294966784u) == qc) {
                anyq = 1;
            }
            if ((__as_u32(rej) & 4294966784u) == qc) {
                anyq = 1;
            }
            if (anyq != 0) {
                int t0_2 = w * 16;
                unsigned int mm = 0;
                float sc_3 = __uint_as_float(cb[0]);
                sc_3 = fmaxf(sc_3, -1.7014118e+38f);
                sc_3 = fminf(sc_3, 1.7014118e+38f);
                unsigned int cls9 = __as_u32(sc_3) & 4294966784u;
                if (cls9 == qc) {
                    mm = mm | 1;
                }
                float sc_5 = __uint_as_float(cb[1]);
                sc_5 = fmaxf(sc_5, -1.7014118e+38f);
                sc_5 = fminf(sc_5, 1.7014118e+38f);
                unsigned int cls9_6 = __as_u32(sc_5) & 4294966784u;
                if (cls9_6 == qc) {
                    mm = mm | 2;
                }
                float sc_7 = __uint_as_float(cb[2]);
                sc_7 = fmaxf(sc_7, -1.7014118e+38f);
                sc_7 = fminf(sc_7, 1.7014118e+38f);
                unsigned int cls9_8 = __as_u32(sc_7) & 4294966784u;
                if (cls9_8 == qc) {
                    mm = mm | 4;
                }
                float sc_9 = __uint_as_float(cb[3]);
                sc_9 = fmaxf(sc_9, -1.7014118e+38f);
                sc_9 = fminf(sc_9, 1.7014118e+38f);
                unsigned int cls9_10 = __as_u32(sc_9) & 4294966784u;
                if (cls9_10 == qc) {
                    mm = mm | 8;
                }
                float sc_11 = __uint_as_float(cb[4]);
                sc_11 = fmaxf(sc_11, -1.7014118e+38f);
                sc_11 = fminf(sc_11, 1.7014118e+38f);
                unsigned int cls9_12 = __as_u32(sc_11) & 4294966784u;
                if (cls9_12 == qc) {
                    mm = mm | 16;
                }
                float sc_13 = __uint_as_float(cb[5]);
                sc_13 = fmaxf(sc_13, -1.7014118e+38f);
                sc_13 = fminf(sc_13, 1.7014118e+38f);
                unsigned int cls9_14 = __as_u32(sc_13) & 4294966784u;
                if (cls9_14 == qc) {
                    mm = mm | 32;
                }
                float sc_15 = __uint_as_float(cb[6]);
                sc_15 = fmaxf(sc_15, -1.7014118e+38f);
                sc_15 = fminf(sc_15, 1.7014118e+38f);
                unsigned int cls9_16 = __as_u32(sc_15) & 4294966784u;
                if (cls9_16 == qc) {
                    mm = mm | 64;
                }
                float sc_17 = __uint_as_float(cb[7]);
                sc_17 = fmaxf(sc_17, -1.7014118e+38f);
                sc_17 = fminf(sc_17, 1.7014118e+38f);
                unsigned int cls9_18 = __as_u32(sc_17) & 4294966784u;
                if (cls9_18 == qc) {
                    mm = mm | 128;
                }
                float sc_19 = __uint_as_float(cb[8]);
                sc_19 = fmaxf(sc_19, -1.7014118e+38f);
                sc_19 = fminf(sc_19, 1.7014118e+38f);
                unsigned int cls9_20 = __as_u32(sc_19) & 4294966784u;
                if (cls9_20 == qc) {
                    mm = mm | 256;
                }
                float sc_21 = __uint_as_float(cb[9]);
                sc_21 = fmaxf(sc_21, -1.7014118e+38f);
                sc_21 = fminf(sc_21, 1.7014118e+38f);
                unsigned int cls9_22 = __as_u32(sc_21) & 4294966784u;
                if (cls9_22 == qc) {
                    mm = mm | 512;
                }
                float sc_23 = __uint_as_float(cb[10]);
                sc_23 = fmaxf(sc_23, -1.7014118e+38f);
                sc_23 = fminf(sc_23, 1.7014118e+38f);
                unsigned int cls9_24 = __as_u32(sc_23) & 4294966784u;
                if (cls9_24 == qc) {
                    mm = mm | 1024;
                }
                float sc_25 = __uint_as_float(cb[11]);
                sc_25 = fmaxf(sc_25, -1.7014118e+38f);
                sc_25 = fminf(sc_25, 1.7014118e+38f);
                unsigned int cls9_26 = __as_u32(sc_25) & 4294966784u;
                if (cls9_26 == qc) {
                    mm = mm | 2048;
                }
                float sc_27 = __uint_as_float(cb[12]);
                sc_27 = fmaxf(sc_27, -1.7014118e+38f);
                sc_27 = fminf(sc_27, 1.7014118e+38f);
                unsigned int cls9_28 = __as_u32(sc_27) & 4294966784u;
                if (cls9_28 == qc) {
                    mm = mm | 4096;
                }
                float sc_29 = __uint_as_float(cb[13]);
                sc_29 = fmaxf(sc_29, -1.7014118e+38f);
                sc_29 = fminf(sc_29, 1.7014118e+38f);
                unsigned int cls9_30 = __as_u32(sc_29) & 4294966784u;
                if (cls9_30 == qc) {
                    mm = mm | 8192;
                }
                float sc_31 = __uint_as_float(cb[14]);
                sc_31 = fmaxf(sc_31, -1.7014118e+38f);
                sc_31 = fminf(sc_31, 1.7014118e+38f);
                unsigned int cls9_32 = __as_u32(sc_31) & 4294966784u;
                if (cls9_32 == qc) {
                    mm = mm | 16384;
                }
                float sc_33 = __uint_as_float(cb[15]);
                sc_33 = fmaxf(sc_33, -1.7014118e+38f);
                sc_33 = fminf(sc_33, 1.7014118e+38f);
                unsigned int cls9_34 = __as_u32(sc_33) & 4294966784u;
                if (cls9_34 == qc) {
                    mm = mm | 32768;
                }
                int f_35 = 0;
                if ((t0_2 < fb || t0_2 + 16 > lim - fe) && t0_2 < lim) {
                    f_35 = 1;
                }
                if (f_35 != 0) {
                    int f_0_1 = 0;
                    if ((t0_2 < fb || t0_2 >= lim - fe) && lim > t0_2) {
                        f_0_1 = 1;
                    }
                    if (f_0_1 != 0) {
                        mm = mm & 4294967294u;
                    }
                    int f_1_1 = 0;
                    if ((t0_2 + 1 < fb || t0_2 + 1 >= lim - fe) && lim > t0_2 + 1) {
                        f_1_1 = 1;
                    }
                    if (f_1_1 != 0) {
                        mm = mm & 4294967293u;
                    }
                    int f_2_1 = 0;
                    if ((t0_2 + 2 < fb || t0_2 + 2 >= lim - fe) && lim > t0_2 + 2) {
                        f_2_1 = 1;
                    }
                    if (f_2_1 != 0) {
                        mm = mm & 4294967291u;
                    }
                    int f_3_1 = 0;
                    if ((t0_2 + 3 < fb || t0_2 + 3 >= lim - fe) && lim > t0_2 + 3) {
                        f_3_1 = 1;
                    }
                    if (f_3_1 != 0) {
                        mm = mm & 4294967287u;
                    }
                    int f_4_1 = 0;
                    if ((t0_2 + 4 < fb || t0_2 + 4 >= lim - fe) && lim > t0_2 + 4) {
                        f_4_1 = 1;
                    }
                    if (f_4_1 != 0) {
                        mm = mm & 4294967279u;
                    }
                    int f_5_1 = 0;
                    if ((t0_2 + 5 < fb || t0_2 + 5 >= lim - fe) && lim > t0_2 + 5) {
                        f_5_1 = 1;
                    }
                    if (f_5_1 != 0) {
                        mm = mm & 4294967263u;
                    }
                    int f_6_1 = 0;
                    if ((t0_2 + 6 < fb || t0_2 + 6 >= lim - fe) && lim > t0_2 + 6) {
                        f_6_1 = 1;
                    }
                    if (f_6_1 != 0) {
                        mm = mm & 4294967231u;
                    }
                    int f_7_1 = 0;
                    if ((t0_2 + 7 < fb || t0_2 + 7 >= lim - fe) && lim > t0_2 + 7) {
                        f_7_1 = 1;
                    }
                    if (f_7_1 != 0) {
                        mm = mm & 4294967167u;
                    }
                    int f_8_1 = 0;
                    if ((t0_2 + 8 < fb || t0_2 + 8 >= lim - fe) && lim > t0_2 + 8) {
                        f_8_1 = 1;
                    }
                    if (f_8_1 != 0) {
                        mm = mm & 4294967039u;
                    }
                    int f_9_1 = 0;
                    if ((t0_2 + 9 < fb || t0_2 + 9 >= lim - fe) && lim > t0_2 + 9) {
                        f_9_1 = 1;
                    }
                    if (f_9_1 != 0) {
                        mm = mm & 4294966783u;
                    }
                    int f_10_1 = 0;
                    if ((t0_2 + 10 < fb || t0_2 + 10 >= lim - fe) && lim > t0_2 + 10) {
                        f_10_1 = 1;
                    }
                    if (f_10_1 != 0) {
                        mm = mm & 4294966271u;
                    }
                    int f_11_1 = 0;
                    if ((t0_2 + 11 < fb || t0_2 + 11 >= lim - fe) && lim > t0_2 + 11) {
                        f_11_1 = 1;
                    }
                    if (f_11_1 != 0) {
                        mm = mm & 4294965247u;
                    }
                    int f_12_1 = 0;
                    if ((t0_2 + 12 < fb || t0_2 + 12 >= lim - fe) && lim > t0_2 + 12) {
                        f_12_1 = 1;
                    }
                    if (f_12_1 != 0) {
                        mm = mm & 4294963199u;
                    }
                    int f_13_1 = 0;
                    if ((t0_2 + 13 < fb || t0_2 + 13 >= lim - fe) && lim > t0_2 + 13) {
                        f_13_1 = 1;
                    }
                    if (f_13_1 != 0) {
                        mm = mm & 4294959103u;
                    }
                    int f_14_1 = 0;
                    if ((t0_2 + 14 < fb || t0_2 + 14 >= lim - fe) && lim > t0_2 + 14) {
                        f_14_1 = 1;
                    }
                    if (f_14_1 != 0) {
                        mm = mm & 4294950911u;
                    }
                    int f_15_1 = 0;
                    if ((t0_2 + 15 < fb || t0_2 + 15 >= lim - fe) && lim > t0_2 + 15) {
                        f_15_1 = 1;
                    }
                    if (f_15_1 != 0) {
                        mm = mm & 4294934527u;
                    }
                }
                if (mm != 0) {
                    int n9 = (int)(unsigned int)__popc(mm);
                    unsigned int base9 = atomicAdd(&ccnt[c], (unsigned int)n9);
                    if (base9 + (unsigned int)n9 > 16) {
                        flagw[1] = 1;
                    } else {
                        int pos9 = c * 16 + (int)base9;
                        if ((mm & 1) != 0) {
                            float sc_0 = __uint_as_float(cb[0]);
                            sc_0 = fmaxf(sc_0, -1.7014118e+38f);
                            sc_0 = fminf(sc_0, 1.7014118e+38f);
                            unsigned int u9 = __as_u32(sc_0);
                            unsigned int lowb9 = (u9 ^ (unsigned int)((int)u9 >> 31) & 511) & 511;
                            unsigned int km9 = 536870912 | lowb9 << 9 | (unsigned int)t0_2;
                            cbuf[pos9] = km9;
                            pos9 = pos9 + 1;
                        }
                        if ((mm & 2) != 0) {
                            float sc_0_1 = __uint_as_float(cb[1]);
                            sc_0_1 = fmaxf(sc_0_1, -1.7014118e+38f);
                            sc_0_1 = fminf(sc_0_1, 1.7014118e+38f);
                            unsigned int u9_1 = __as_u32(sc_0_1);
                            unsigned int lowb9_1 = (u9_1 ^ (unsigned int)((int)u9_1 >> 31) & 511) & 511;
                            unsigned int km9_1 = 536870912 | lowb9_1 << 9 | (unsigned int)(t0_2 + 1);
                            cbuf[pos9] = km9_1;
                            pos9 = pos9 + 1;
                        }
                        if ((mm & 4) != 0) {
                            float sc_0_2 = __uint_as_float(cb[2]);
                            sc_0_2 = fmaxf(sc_0_2, -1.7014118e+38f);
                            sc_0_2 = fminf(sc_0_2, 1.7014118e+38f);
                            unsigned int u9_2 = __as_u32(sc_0_2);
                            unsigned int lowb9_2 = (u9_2 ^ (unsigned int)((int)u9_2 >> 31) & 511) & 511;
                            unsigned int km9_2 = 536870912 | lowb9_2 << 9 | (unsigned int)(t0_2 + 2);
                            cbuf[pos9] = km9_2;
                            pos9 = pos9 + 1;
                        }
                        if ((mm & 8) != 0) {
                            float sc_0_3 = __uint_as_float(cb[3]);
                            sc_0_3 = fmaxf(sc_0_3, -1.7014118e+38f);
                            sc_0_3 = fminf(sc_0_3, 1.7014118e+38f);
                            unsigned int u9_3 = __as_u32(sc_0_3);
                            unsigned int lowb9_3 = (u9_3 ^ (unsigned int)((int)u9_3 >> 31) & 511) & 511;
                            unsigned int km9_3 = 536870912 | lowb9_3 << 9 | (unsigned int)(t0_2 + 3);
                            cbuf[pos9] = km9_3;
                            pos9 = pos9 + 1;
                        }
                        if ((mm & 16) != 0) {
                            float sc_0_4 = __uint_as_float(cb[4]);
                            sc_0_4 = fmaxf(sc_0_4, -1.7014118e+38f);
                            sc_0_4 = fminf(sc_0_4, 1.7014118e+38f);
                            unsigned int u9_4 = __as_u32(sc_0_4);
                            unsigned int lowb9_4 = (u9_4 ^ (unsigned int)((int)u9_4 >> 31) & 511) & 511;
                            unsigned int km9_4 = 536870912 | lowb9_4 << 9 | (unsigned int)(t0_2 + 4);
                            cbuf[pos9] = km9_4;
                            pos9 = pos9 + 1;
                        }
                        if ((mm & 32) != 0) {
                            float sc_0_5 = __uint_as_float(cb[5]);
                            sc_0_5 = fmaxf(sc_0_5, -1.7014118e+38f);
                            sc_0_5 = fminf(sc_0_5, 1.7014118e+38f);
                            unsigned int u9_5 = __as_u32(sc_0_5);
                            unsigned int lowb9_5 = (u9_5 ^ (unsigned int)((int)u9_5 >> 31) & 511) & 511;
                            unsigned int km9_5 = 536870912 | lowb9_5 << 9 | (unsigned int)(t0_2 + 5);
                            cbuf[pos9] = km9_5;
                            pos9 = pos9 + 1;
                        }
                        if ((mm & 64) != 0) {
                            float sc_0_6 = __uint_as_float(cb[6]);
                            sc_0_6 = fmaxf(sc_0_6, -1.7014118e+38f);
                            sc_0_6 = fminf(sc_0_6, 1.7014118e+38f);
                            unsigned int u9_6 = __as_u32(sc_0_6);
                            unsigned int lowb9_6 = (u9_6 ^ (unsigned int)((int)u9_6 >> 31) & 511) & 511;
                            unsigned int km9_6 = 536870912 | lowb9_6 << 9 | (unsigned int)(t0_2 + 6);
                            cbuf[pos9] = km9_6;
                            pos9 = pos9 + 1;
                        }
                        if ((mm & 128) != 0) {
                            float sc_0_7 = __uint_as_float(cb[7]);
                            sc_0_7 = fmaxf(sc_0_7, -1.7014118e+38f);
                            sc_0_7 = fminf(sc_0_7, 1.7014118e+38f);
                            unsigned int u9_7 = __as_u32(sc_0_7);
                            unsigned int lowb9_7 = (u9_7 ^ (unsigned int)((int)u9_7 >> 31) & 511) & 511;
                            unsigned int km9_7 = 536870912 | lowb9_7 << 9 | (unsigned int)(t0_2 + 7);
                            cbuf[pos9] = km9_7;
                            pos9 = pos9 + 1;
                        }
                        if ((mm & 256) != 0) {
                            float sc_0_8 = __uint_as_float(cb[8]);
                            sc_0_8 = fmaxf(sc_0_8, -1.7014118e+38f);
                            sc_0_8 = fminf(sc_0_8, 1.7014118e+38f);
                            unsigned int u9_8 = __as_u32(sc_0_8);
                            unsigned int lowb9_8 = (u9_8 ^ (unsigned int)((int)u9_8 >> 31) & 511) & 511;
                            unsigned int km9_8 = 536870912 | lowb9_8 << 9 | (unsigned int)(t0_2 + 8);
                            cbuf[pos9] = km9_8;
                            pos9 = pos9 + 1;
                        }
                        if ((mm & 512) != 0) {
                            float sc_0_9 = __uint_as_float(cb[9]);
                            sc_0_9 = fmaxf(sc_0_9, -1.7014118e+38f);
                            sc_0_9 = fminf(sc_0_9, 1.7014118e+38f);
                            unsigned int u9_9 = __as_u32(sc_0_9);
                            unsigned int lowb9_9 = (u9_9 ^ (unsigned int)((int)u9_9 >> 31) & 511) & 511;
                            unsigned int km9_9 = 536870912 | lowb9_9 << 9 | (unsigned int)(t0_2 + 9);
                            cbuf[pos9] = km9_9;
                            pos9 = pos9 + 1;
                        }
                        if ((mm & 1024) != 0) {
                            float sc_0_10 = __uint_as_float(cb[10]);
                            sc_0_10 = fmaxf(sc_0_10, -1.7014118e+38f);
                            sc_0_10 = fminf(sc_0_10, 1.7014118e+38f);
                            unsigned int u9_10 = __as_u32(sc_0_10);
                            unsigned int lowb9_10 = (u9_10 ^ (unsigned int)((int)u9_10 >> 31) & 511) & 511;
                            unsigned int km9_10 = 536870912 | lowb9_10 << 9 | (unsigned int)(t0_2 + 10);
                            cbuf[pos9] = km9_10;
                            pos9 = pos9 + 1;
                        }
                        if ((mm & 2048) != 0) {
                            float sc_0_11 = __uint_as_float(cb[11]);
                            sc_0_11 = fmaxf(sc_0_11, -1.7014118e+38f);
                            sc_0_11 = fminf(sc_0_11, 1.7014118e+38f);
                            unsigned int u9_11 = __as_u32(sc_0_11);
                            unsigned int lowb9_11 = (u9_11 ^ (unsigned int)((int)u9_11 >> 31) & 511) & 511;
                            unsigned int km9_11 = 536870912 | lowb9_11 << 9 | (unsigned int)(t0_2 + 11);
                            cbuf[pos9] = km9_11;
                            pos9 = pos9 + 1;
                        }
                        if ((mm & 4096) != 0) {
                            float sc_0_12 = __uint_as_float(cb[12]);
                            sc_0_12 = fmaxf(sc_0_12, -1.7014118e+38f);
                            sc_0_12 = fminf(sc_0_12, 1.7014118e+38f);
                            unsigned int u9_12 = __as_u32(sc_0_12);
                            unsigned int lowb9_12 = (u9_12 ^ (unsigned int)((int)u9_12 >> 31) & 511) & 511;
                            unsigned int km9_12 = 536870912 | lowb9_12 << 9 | (unsigned int)(t0_2 + 12);
                            cbuf[pos9] = km9_12;
                            pos9 = pos9 + 1;
                        }
                        if ((mm & 8192) != 0) {
                            float sc_0_13 = __uint_as_float(cb[13]);
                            sc_0_13 = fmaxf(sc_0_13, -1.7014118e+38f);
                            sc_0_13 = fminf(sc_0_13, 1.7014118e+38f);
                            unsigned int u9_13 = __as_u32(sc_0_13);
                            unsigned int lowb9_13 = (u9_13 ^ (unsigned int)((int)u9_13 >> 31) & 511) & 511;
                            unsigned int km9_13 = 536870912 | lowb9_13 << 9 | (unsigned int)(t0_2 + 13);
                            cbuf[pos9] = km9_13;
                            pos9 = pos9 + 1;
                        }
                        if ((mm & 16384) != 0) {
                            float sc_0_14 = __uint_as_float(cb[14]);
                            sc_0_14 = fmaxf(sc_0_14, -1.7014118e+38f);
                            sc_0_14 = fminf(sc_0_14, 1.7014118e+38f);
                            unsigned int u9_14 = __as_u32(sc_0_14);
                            unsigned int lowb9_14 = (u9_14 ^ (unsigned int)((int)u9_14 >> 31) & 511) & 511;
                            unsigned int km9_14 = 536870912 | lowb9_14 << 9 | (unsigned int)(t0_2 + 14);
                            cbuf[pos9] = km9_14;
                            pos9 = pos9 + 1;
                        }
                        if ((mm & 32768) != 0) {
                            float sc_0_15 = __uint_as_float(cb[15]);
                            sc_0_15 = fmaxf(sc_0_15, -1.7014118e+38f);
                            sc_0_15 = fminf(sc_0_15, 1.7014118e+38f);
                            unsigned int u9_15 = __as_u32(sc_0_15);
                            unsigned int lowb9_15 = (u9_15 ^ (unsigned int)((int)u9_15 >> 31) & 511) & 511;
                            unsigned int km9_15 = 536870912 | lowb9_15 << 9 | (unsigned int)(t0_2 + 15);
                            cbuf[pos9] = km9_15;
                            pos9 = pos9 + 1;
                        }
                    }
                }
            }
        }
        asm volatile("barrier.sync 1, 64;" ::: "memory");
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
                int t0_2_1 = w * 16;
                float qf = __uint_as_float(qc);
                float sc_3_1 = __uint_as_float(cb[0]);
                sc_3_1 = fmaxf(sc_3_1, -1.7014118e+38f);
                sc_3_1 = fminf(sc_3_1, 1.7014118e+38f);
                unsigned int u = __as_u32(sc_3_1);
                unsigned int cls = u & 4294966784u;
                int f_4_2 = 0;
                if ((t0_2_1 < fb || t0_2_1 >= lim - fe) && lim > t0_2_1) {
                    f_4_2 = 1;
                }
                if (f_4_2 != 0) {
                    cls = 2139094528;
                }
                unsigned int key_6 = 0;
                if (qf < __uint_as_float(cls) && cls < 4278190080u) {
                    key_6 = 1073741824 | (unsigned int)t0_2_1;
                }
                if (cls == qc) {
                    unsigned int lowb = (u ^ (unsigned int)((int)u >> 31) & 511) & 511;
                    key_6 = 536870912 | lowb << 9 | (unsigned int)t0_2_1;
                }
                kb2o[0] = __uint_as_float(key_6);
                float sc_7_1 = __uint_as_float(cb[1]);
                sc_7_1 = fmaxf(sc_7_1, -1.7014118e+38f);
                sc_7_1 = fminf(sc_7_1, 1.7014118e+38f);
                unsigned int u_8 = __as_u32(sc_7_1);
                unsigned int cls_9 = u_8 & 4294966784u;
                int f_10_2 = 0;
                if ((t0_2_1 + 1 < fb || t0_2_1 + 1 >= lim - fe) && lim > t0_2_1 + 1) {
                    f_10_2 = 1;
                }
                if (f_10_2 != 0) {
                    cls_9 = 2139094528;
                }
                unsigned int key_12 = 0;
                if (qf < __uint_as_float(cls_9) && cls_9 < 4278190080u) {
                    key_12 = 1073741824 | (unsigned int)(t0_2_1 + 1);
                }
                if (cls_9 == qc) {
                    unsigned int lowb_1 = (u_8 ^ (unsigned int)((int)u_8 >> 31) & 511) & 511;
                    key_12 = 536870912 | lowb_1 << 9 | (unsigned int)(t0_2_1 + 1);
                }
                kb2o[1] = __uint_as_float(key_12);
                float sc_13_1 = __uint_as_float(cb[2]);
                sc_13_1 = fmaxf(sc_13_1, -1.7014118e+38f);
                sc_13_1 = fminf(sc_13_1, 1.7014118e+38f);
                unsigned int u_14 = __as_u32(sc_13_1);
                unsigned int cls_15 = u_14 & 4294966784u;
                int f_16 = 0;
                if ((t0_2_1 + 2 < fb || t0_2_1 + 2 >= lim - fe) && lim > t0_2_1 + 2) {
                    f_16 = 1;
                }
                if (f_16 != 0) {
                    cls_15 = 2139094528;
                }
                unsigned int key_18 = 0;
                if (qf < __uint_as_float(cls_15) && cls_15 < 4278190080u) {
                    key_18 = 1073741824 | (unsigned int)(t0_2_1 + 2);
                }
                if (cls_15 == qc) {
                    unsigned int lowb_2 = (u_14 ^ (unsigned int)((int)u_14 >> 31) & 511) & 511;
                    key_18 = 536870912 | lowb_2 << 9 | (unsigned int)(t0_2_1 + 2);
                }
                kb2o[2] = __uint_as_float(key_18);
                float sc_19_1 = __uint_as_float(cb[3]);
                sc_19_1 = fmaxf(sc_19_1, -1.7014118e+38f);
                sc_19_1 = fminf(sc_19_1, 1.7014118e+38f);
                unsigned int u_20 = __as_u32(sc_19_1);
                unsigned int cls_21 = u_20 & 4294966784u;
                int f_22 = 0;
                if ((t0_2_1 + 3 < fb || t0_2_1 + 3 >= lim - fe) && lim > t0_2_1 + 3) {
                    f_22 = 1;
                }
                if (f_22 != 0) {
                    cls_21 = 2139094528;
                }
                unsigned int key_24 = 0;
                if (qf < __uint_as_float(cls_21) && cls_21 < 4278190080u) {
                    key_24 = 1073741824 | (unsigned int)(t0_2_1 + 3);
                }
                if (cls_21 == qc) {
                    unsigned int lowb_3 = (u_20 ^ (unsigned int)((int)u_20 >> 31) & 511) & 511;
                    key_24 = 536870912 | lowb_3 << 9 | (unsigned int)(t0_2_1 + 3);
                }
                kb2o[3] = __uint_as_float(key_24);
                float sc_25_1 = __uint_as_float(cb[4]);
                sc_25_1 = fmaxf(sc_25_1, -1.7014118e+38f);
                sc_25_1 = fminf(sc_25_1, 1.7014118e+38f);
                unsigned int u_26 = __as_u32(sc_25_1);
                unsigned int cls_27 = u_26 & 4294966784u;
                int f_28 = 0;
                if ((t0_2_1 + 4 < fb || t0_2_1 + 4 >= lim - fe) && lim > t0_2_1 + 4) {
                    f_28 = 1;
                }
                if (f_28 != 0) {
                    cls_27 = 2139094528;
                }
                unsigned int key_30 = 0;
                if (qf < __uint_as_float(cls_27) && cls_27 < 4278190080u) {
                    key_30 = 1073741824 | (unsigned int)(t0_2_1 + 4);
                }
                if (cls_27 == qc) {
                    unsigned int lowb_4 = (u_26 ^ (unsigned int)((int)u_26 >> 31) & 511) & 511;
                    key_30 = 536870912 | lowb_4 << 9 | (unsigned int)(t0_2_1 + 4);
                }
                kb2o[4] = __uint_as_float(key_30);
                float sc_31_1 = __uint_as_float(cb[5]);
                sc_31_1 = fmaxf(sc_31_1, -1.7014118e+38f);
                sc_31_1 = fminf(sc_31_1, 1.7014118e+38f);
                unsigned int u_32 = __as_u32(sc_31_1);
                unsigned int cls_33 = u_32 & 4294966784u;
                int f_34 = 0;
                if ((t0_2_1 + 5 < fb || t0_2_1 + 5 >= lim - fe) && lim > t0_2_1 + 5) {
                    f_34 = 1;
                }
                if (f_34 != 0) {
                    cls_33 = 2139094528;
                }
                unsigned int key_35 = 0;
                if (qf < __uint_as_float(cls_33) && cls_33 < 4278190080u) {
                    key_35 = 1073741824 | (unsigned int)(t0_2_1 + 5);
                }
                if (cls_33 == qc) {
                    unsigned int lowb_5 = (u_32 ^ (unsigned int)((int)u_32 >> 31) & 511) & 511;
                    key_35 = 536870912 | lowb_5 << 9 | (unsigned int)(t0_2_1 + 5);
                }
                kb2o[5] = __uint_as_float(key_35);
                float sc_36 = __uint_as_float(cb[6]);
                sc_36 = fmaxf(sc_36, -1.7014118e+38f);
                sc_36 = fminf(sc_36, 1.7014118e+38f);
                unsigned int u_37 = __as_u32(sc_36);
                unsigned int cls_38 = u_37 & 4294966784u;
                int f_39 = 0;
                if ((t0_2_1 + 6 < fb || t0_2_1 + 6 >= lim - fe) && lim > t0_2_1 + 6) {
                    f_39 = 1;
                }
                if (f_39 != 0) {
                    cls_38 = 2139094528;
                }
                unsigned int key_40 = 0;
                if (qf < __uint_as_float(cls_38) && cls_38 < 4278190080u) {
                    key_40 = 1073741824 | (unsigned int)(t0_2_1 + 6);
                }
                if (cls_38 == qc) {
                    unsigned int lowb_6 = (u_37 ^ (unsigned int)((int)u_37 >> 31) & 511) & 511;
                    key_40 = 536870912 | lowb_6 << 9 | (unsigned int)(t0_2_1 + 6);
                }
                kb2o[6] = __uint_as_float(key_40);
                float sc_41 = __uint_as_float(cb[7]);
                sc_41 = fmaxf(sc_41, -1.7014118e+38f);
                sc_41 = fminf(sc_41, 1.7014118e+38f);
                unsigned int u_42 = __as_u32(sc_41);
                unsigned int cls_43 = u_42 & 4294966784u;
                int f_44 = 0;
                if ((t0_2_1 + 7 < fb || t0_2_1 + 7 >= lim - fe) && lim > t0_2_1 + 7) {
                    f_44 = 1;
                }
                if (f_44 != 0) {
                    cls_43 = 2139094528;
                }
                unsigned int key_45 = 0;
                if (qf < __uint_as_float(cls_43) && cls_43 < 4278190080u) {
                    key_45 = 1073741824 | (unsigned int)(t0_2_1 + 7);
                }
                if (cls_43 == qc) {
                    unsigned int lowb_7 = (u_42 ^ (unsigned int)((int)u_42 >> 31) & 511) & 511;
                    key_45 = 536870912 | lowb_7 << 9 | (unsigned int)(t0_2_1 + 7);
                }
                kb2o[7] = __uint_as_float(key_45);
                float sc_46 = __uint_as_float(cb[8]);
                sc_46 = fmaxf(sc_46, -1.7014118e+38f);
                sc_46 = fminf(sc_46, 1.7014118e+38f);
                unsigned int u_47 = __as_u32(sc_46);
                unsigned int cls_48 = u_47 & 4294966784u;
                int f_49 = 0;
                if ((t0_2_1 + 8 < fb || t0_2_1 + 8 >= lim - fe) && lim > t0_2_1 + 8) {
                    f_49 = 1;
                }
                if (f_49 != 0) {
                    cls_48 = 2139094528;
                }
                unsigned int key_50 = 0;
                if (qf < __uint_as_float(cls_48) && cls_48 < 4278190080u) {
                    key_50 = 1073741824 | (unsigned int)(t0_2_1 + 8);
                }
                if (cls_48 == qc) {
                    unsigned int lowb_8 = (u_47 ^ (unsigned int)((int)u_47 >> 31) & 511) & 511;
                    key_50 = 536870912 | lowb_8 << 9 | (unsigned int)(t0_2_1 + 8);
                }
                kb2o[8] = __uint_as_float(key_50);
                float sc_51 = __uint_as_float(cb[9]);
                sc_51 = fmaxf(sc_51, -1.7014118e+38f);
                sc_51 = fminf(sc_51, 1.7014118e+38f);
                unsigned int u_52 = __as_u32(sc_51);
                unsigned int cls_53 = u_52 & 4294966784u;
                int f_54 = 0;
                if ((t0_2_1 + 9 < fb || t0_2_1 + 9 >= lim - fe) && lim > t0_2_1 + 9) {
                    f_54 = 1;
                }
                if (f_54 != 0) {
                    cls_53 = 2139094528;
                }
                unsigned int key_55 = 0;
                if (qf < __uint_as_float(cls_53) && cls_53 < 4278190080u) {
                    key_55 = 1073741824 | (unsigned int)(t0_2_1 + 9);
                }
                if (cls_53 == qc) {
                    unsigned int lowb_9 = (u_52 ^ (unsigned int)((int)u_52 >> 31) & 511) & 511;
                    key_55 = 536870912 | lowb_9 << 9 | (unsigned int)(t0_2_1 + 9);
                }
                kb2o[9] = __uint_as_float(key_55);
                float sc_56 = __uint_as_float(cb[10]);
                sc_56 = fmaxf(sc_56, -1.7014118e+38f);
                sc_56 = fminf(sc_56, 1.7014118e+38f);
                unsigned int u_57 = __as_u32(sc_56);
                unsigned int cls_58 = u_57 & 4294966784u;
                int f_59 = 0;
                if ((t0_2_1 + 10 < fb || t0_2_1 + 10 >= lim - fe) && lim > t0_2_1 + 10) {
                    f_59 = 1;
                }
                if (f_59 != 0) {
                    cls_58 = 2139094528;
                }
                unsigned int key_60 = 0;
                if (qf < __uint_as_float(cls_58) && cls_58 < 4278190080u) {
                    key_60 = 1073741824 | (unsigned int)(t0_2_1 + 10);
                }
                if (cls_58 == qc) {
                    unsigned int lowb_10 = (u_57 ^ (unsigned int)((int)u_57 >> 31) & 511) & 511;
                    key_60 = 536870912 | lowb_10 << 9 | (unsigned int)(t0_2_1 + 10);
                }
                kb2o[10] = __uint_as_float(key_60);
                float sc_61 = __uint_as_float(cb[11]);
                sc_61 = fmaxf(sc_61, -1.7014118e+38f);
                sc_61 = fminf(sc_61, 1.7014118e+38f);
                unsigned int u_62 = __as_u32(sc_61);
                unsigned int cls_63 = u_62 & 4294966784u;
                int f_64 = 0;
                if ((t0_2_1 + 11 < fb || t0_2_1 + 11 >= lim - fe) && lim > t0_2_1 + 11) {
                    f_64 = 1;
                }
                if (f_64 != 0) {
                    cls_63 = 2139094528;
                }
                unsigned int key_65 = 0;
                if (qf < __uint_as_float(cls_63) && cls_63 < 4278190080u) {
                    key_65 = 1073741824 | (unsigned int)(t0_2_1 + 11);
                }
                if (cls_63 == qc) {
                    unsigned int lowb_11 = (u_62 ^ (unsigned int)((int)u_62 >> 31) & 511) & 511;
                    key_65 = 536870912 | lowb_11 << 9 | (unsigned int)(t0_2_1 + 11);
                }
                kb2o[11] = __uint_as_float(key_65);
                float sc_66 = __uint_as_float(cb[12]);
                sc_66 = fmaxf(sc_66, -1.7014118e+38f);
                sc_66 = fminf(sc_66, 1.7014118e+38f);
                unsigned int u_67 = __as_u32(sc_66);
                unsigned int cls_68 = u_67 & 4294966784u;
                int f_69 = 0;
                if ((t0_2_1 + 12 < fb || t0_2_1 + 12 >= lim - fe) && lim > t0_2_1 + 12) {
                    f_69 = 1;
                }
                if (f_69 != 0) {
                    cls_68 = 2139094528;
                }
                unsigned int key_70 = 0;
                if (qf < __uint_as_float(cls_68) && cls_68 < 4278190080u) {
                    key_70 = 1073741824 | (unsigned int)(t0_2_1 + 12);
                }
                if (cls_68 == qc) {
                    unsigned int lowb_12 = (u_67 ^ (unsigned int)((int)u_67 >> 31) & 511) & 511;
                    key_70 = 536870912 | lowb_12 << 9 | (unsigned int)(t0_2_1 + 12);
                }
                kb2o[12] = __uint_as_float(key_70);
                float sc_71 = __uint_as_float(cb[13]);
                sc_71 = fmaxf(sc_71, -1.7014118e+38f);
                sc_71 = fminf(sc_71, 1.7014118e+38f);
                unsigned int u_72 = __as_u32(sc_71);
                unsigned int cls_73 = u_72 & 4294966784u;
                int f_74 = 0;
                if ((t0_2_1 + 13 < fb || t0_2_1 + 13 >= lim - fe) && lim > t0_2_1 + 13) {
                    f_74 = 1;
                }
                if (f_74 != 0) {
                    cls_73 = 2139094528;
                }
                unsigned int key_75 = 0;
                if (qf < __uint_as_float(cls_73) && cls_73 < 4278190080u) {
                    key_75 = 1073741824 | (unsigned int)(t0_2_1 + 13);
                }
                if (cls_73 == qc) {
                    unsigned int lowb_13 = (u_72 ^ (unsigned int)((int)u_72 >> 31) & 511) & 511;
                    key_75 = 536870912 | lowb_13 << 9 | (unsigned int)(t0_2_1 + 13);
                }
                kb2o[13] = __uint_as_float(key_75);
                float sc_76 = __uint_as_float(cb[14]);
                sc_76 = fmaxf(sc_76, -1.7014118e+38f);
                sc_76 = fminf(sc_76, 1.7014118e+38f);
                unsigned int u_77 = __as_u32(sc_76);
                unsigned int cls_78 = u_77 & 4294966784u;
                int f_79 = 0;
                if ((t0_2_1 + 14 < fb || t0_2_1 + 14 >= lim - fe) && lim > t0_2_1 + 14) {
                    f_79 = 1;
                }
                if (f_79 != 0) {
                    cls_78 = 2139094528;
                }
                unsigned int key_80 = 0;
                if (qf < __uint_as_float(cls_78) && cls_78 < 4278190080u) {
                    key_80 = 1073741824 | (unsigned int)(t0_2_1 + 14);
                }
                if (cls_78 == qc) {
                    unsigned int lowb_14 = (u_77 ^ (unsigned int)((int)u_77 >> 31) & 511) & 511;
                    key_80 = 536870912 | lowb_14 << 9 | (unsigned int)(t0_2_1 + 14);
                }
                kb2o[14] = __uint_as_float(key_80);
                float sc_81 = __uint_as_float(cb[15]);
                sc_81 = fmaxf(sc_81, -1.7014118e+38f);
                sc_81 = fminf(sc_81, 1.7014118e+38f);
                unsigned int u_82 = __as_u32(sc_81);
                unsigned int cls_83 = u_82 & 4294966784u;
                int f_84 = 0;
                if ((t0_2_1 + 15 < fb || t0_2_1 + 15 >= lim - fe) && lim > t0_2_1 + 15) {
                    f_84 = 1;
                }
                if (f_84 != 0) {
                    cls_83 = 2139094528;
                }
                unsigned int key_85 = 0;
                if (qf < __uint_as_float(cls_83) && cls_83 < 4278190080u) {
                    key_85 = 1073741824 | (unsigned int)(t0_2_1 + 15);
                }
                if (cls_83 == qc) {
                    unsigned int lowb_15 = (u_82 ^ (unsigned int)((int)u_82 >> 31) & 511) & 511;
                    key_85 = 536870912 | lowb_15 << 9 | (unsigned int)(t0_2_1 + 15);
                }
                kb2o[15] = __uint_as_float(key_85);
                float hi_87 = fmaxf(kb2o[0], kb2o[13]);
                float lo_88 = fminf(kb2o[0], kb2o[13]);
                kb2o[0] = hi_87;
                kb2o[13] = lo_88;
                float hi_89 = fmaxf(kb2o[1], kb2o[12]);
                float lo_90 = fminf(kb2o[1], kb2o[12]);
                kb2o[1] = hi_89;
                kb2o[12] = lo_90;
                float hi_91 = fmaxf(kb2o[2], kb2o[15]);
                float lo_92 = fminf(kb2o[2], kb2o[15]);
                kb2o[2] = hi_91;
                kb2o[15] = lo_92;
                float hi_93 = fmaxf(kb2o[3], kb2o[14]);
                float lo_94 = fminf(kb2o[3], kb2o[14]);
                kb2o[3] = hi_93;
                kb2o[14] = lo_94;
                float hi_95 = fmaxf(kb2o[4], kb2o[8]);
                float lo_96 = fminf(kb2o[4], kb2o[8]);
                kb2o[4] = hi_95;
                kb2o[8] = lo_96;
                float hi_97 = fmaxf(kb2o[5], kb2o[6]);
                float lo_98 = fminf(kb2o[5], kb2o[6]);
                kb2o[5] = hi_97;
                kb2o[6] = lo_98;
                float hi_99 = fmaxf(kb2o[7], kb2o[11]);
                float lo_100 = fminf(kb2o[7], kb2o[11]);
                kb2o[7] = hi_99;
                kb2o[11] = lo_100;
                float hi_101 = fmaxf(kb2o[9], kb2o[10]);
                float lo_102 = fminf(kb2o[9], kb2o[10]);
                kb2o[9] = hi_101;
                kb2o[10] = lo_102;
                float hi_103 = fmaxf(kb2o[0], kb2o[5]);
                float lo_104 = fminf(kb2o[0], kb2o[5]);
                kb2o[0] = hi_103;
                kb2o[5] = lo_104;
                float hi_105 = fmaxf(kb2o[1], kb2o[7]);
                float lo_106 = fminf(kb2o[1], kb2o[7]);
                kb2o[1] = hi_105;
                kb2o[7] = lo_106;
                float hi_107 = fmaxf(kb2o[2], kb2o[9]);
                float lo_108 = fminf(kb2o[2], kb2o[9]);
                kb2o[2] = hi_107;
                kb2o[9] = lo_108;
                float hi_109 = fmaxf(kb2o[3], kb2o[4]);
                float lo_110 = fminf(kb2o[3], kb2o[4]);
                kb2o[3] = hi_109;
                kb2o[4] = lo_110;
                float hi_111 = fmaxf(kb2o[6], kb2o[13]);
                float lo_112 = fminf(kb2o[6], kb2o[13]);
                kb2o[6] = hi_111;
                kb2o[13] = lo_112;
                float hi_113 = fmaxf(kb2o[8], kb2o[14]);
                float lo_114 = fminf(kb2o[8], kb2o[14]);
                kb2o[8] = hi_113;
                kb2o[14] = lo_114;
                float hi_115 = fmaxf(kb2o[10], kb2o[15]);
                float lo_116 = fminf(kb2o[10], kb2o[15]);
                kb2o[10] = hi_115;
                kb2o[15] = lo_116;
                float hi_117 = fmaxf(kb2o[11], kb2o[12]);
                float lo_118 = fminf(kb2o[11], kb2o[12]);
                kb2o[11] = hi_117;
                kb2o[12] = lo_118;
                float hi_119 = fmaxf(kb2o[0], kb2o[1]);
                float lo_120 = fminf(kb2o[0], kb2o[1]);
                kb2o[0] = hi_119;
                kb2o[1] = lo_120;
                float hi_121 = fmaxf(kb2o[2], kb2o[3]);
                float lo_122 = fminf(kb2o[2], kb2o[3]);
                kb2o[2] = hi_121;
                kb2o[3] = lo_122;
                float hi_123 = fmaxf(kb2o[4], kb2o[5]);
                float lo_124 = fminf(kb2o[4], kb2o[5]);
                kb2o[4] = hi_123;
                kb2o[5] = lo_124;
                float hi_125 = fmaxf(kb2o[6], kb2o[8]);
                float lo_126 = fminf(kb2o[6], kb2o[8]);
                kb2o[6] = hi_125;
                kb2o[8] = lo_126;
                float hi_127 = fmaxf(kb2o[7], kb2o[9]);
                float lo_128 = fminf(kb2o[7], kb2o[9]);
                kb2o[7] = hi_127;
                kb2o[9] = lo_128;
                float hi_129 = fmaxf(kb2o[10], kb2o[11]);
                float lo_130 = fminf(kb2o[10], kb2o[11]);
                kb2o[10] = hi_129;
                kb2o[11] = lo_130;
                float hi_131 = fmaxf(kb2o[12], kb2o[13]);
                float lo_132 = fminf(kb2o[12], kb2o[13]);
                kb2o[12] = hi_131;
                kb2o[13] = lo_132;
                float hi_133 = fmaxf(kb2o[14], kb2o[15]);
                float lo_134 = fminf(kb2o[14], kb2o[15]);
                kb2o[14] = hi_133;
                kb2o[15] = lo_134;
                float hi_135 = fmaxf(kb2o[0], kb2o[2]);
                float lo_136 = fminf(kb2o[0], kb2o[2]);
                kb2o[0] = hi_135;
                kb2o[2] = lo_136;
                float hi_137 = fmaxf(kb2o[1], kb2o[3]);
                float lo_138 = fminf(kb2o[1], kb2o[3]);
                kb2o[1] = hi_137;
                kb2o[3] = lo_138;
                float hi_139 = fmaxf(kb2o[4], kb2o[10]);
                float lo_140 = fminf(kb2o[4], kb2o[10]);
                kb2o[4] = hi_139;
                kb2o[10] = lo_140;
                float hi_141 = fmaxf(kb2o[5], kb2o[11]);
                float lo_142 = fminf(kb2o[5], kb2o[11]);
                kb2o[5] = hi_141;
                kb2o[11] = lo_142;
                float hi_143 = fmaxf(kb2o[6], kb2o[7]);
                float lo_144 = fminf(kb2o[6], kb2o[7]);
                kb2o[6] = hi_143;
                kb2o[7] = lo_144;
                float hi_145 = fmaxf(kb2o[8], kb2o[9]);
                float lo_146 = fminf(kb2o[8], kb2o[9]);
                kb2o[8] = hi_145;
                kb2o[9] = lo_146;
                float hi_147 = fmaxf(kb2o[12], kb2o[14]);
                float lo_148 = fminf(kb2o[12], kb2o[14]);
                kb2o[12] = hi_147;
                kb2o[14] = lo_148;
                float hi_149 = fmaxf(kb2o[13], kb2o[15]);
                float lo_150 = fminf(kb2o[13], kb2o[15]);
                kb2o[13] = hi_149;
                kb2o[15] = lo_150;
                float hi_151 = fmaxf(kb2o[1], kb2o[2]);
                float lo_152 = fminf(kb2o[1], kb2o[2]);
                kb2o[1] = hi_151;
                kb2o[2] = lo_152;
                float hi_154 = fmaxf(kb2o[3], kb2o[12]);
                float lo_155 = fminf(kb2o[3], kb2o[12]);
                kb2o[3] = hi_154;
                kb2o[12] = lo_155;
                float hi_157 = fmaxf(kb2o[4], kb2o[6]);
                float lo_158 = fminf(kb2o[4], kb2o[6]);
                kb2o[4] = hi_157;
                kb2o[6] = lo_158;
                float hi_160 = fmaxf(kb2o[5], kb2o[7]);
                float lo_161 = fminf(kb2o[5], kb2o[7]);
                kb2o[5] = hi_160;
                kb2o[7] = lo_161;
                float hi_162 = fmaxf(kb2o[8], kb2o[10]);
                float lo_163 = fminf(kb2o[8], kb2o[10]);
                kb2o[8] = hi_162;
                kb2o[10] = lo_163;
                float hi_164 = fmaxf(kb2o[9], kb2o[11]);
                float lo_165 = fminf(kb2o[9], kb2o[11]);
                kb2o[9] = hi_164;
                kb2o[11] = lo_165;
                float hi_166 = fmaxf(kb2o[13], kb2o[14]);
                float lo_167 = fminf(kb2o[13], kb2o[14]);
                kb2o[13] = hi_166;
                kb2o[14] = lo_167;
                float hi_168 = fmaxf(kb2o[1], kb2o[4]);
                float lo_169 = fminf(kb2o[1], kb2o[4]);
                kb2o[1] = hi_168;
                kb2o[4] = lo_169;
                float hi_170 = fmaxf(kb2o[2], kb2o[6]);
                float lo_171 = fminf(kb2o[2], kb2o[6]);
                kb2o[2] = hi_170;
                kb2o[6] = lo_171;
                float hi_173 = fmaxf(kb2o[5], kb2o[8]);
                float lo_174 = fminf(kb2o[5], kb2o[8]);
                kb2o[5] = hi_173;
                kb2o[8] = lo_174;
                float hi_176 = fmaxf(kb2o[7], kb2o[10]);
                float lo_177 = fminf(kb2o[7], kb2o[10]);
                kb2o[7] = hi_176;
                kb2o[10] = lo_177;
                float hi_179 = fmaxf(kb2o[9], kb2o[13]);
                float lo_180 = fminf(kb2o[9], kb2o[13]);
                kb2o[9] = hi_179;
                kb2o[13] = lo_180;
                float hi_181 = fmaxf(kb2o[11], kb2o[14]);
                float lo_182 = fminf(kb2o[11], kb2o[14]);
                kb2o[11] = hi_181;
                kb2o[14] = lo_182;
                float hi_183 = fmaxf(kb2o[2], kb2o[4]);
                float lo_184 = fminf(kb2o[2], kb2o[4]);
                kb2o[2] = hi_183;
                kb2o[4] = lo_184;
                float hi_185 = fmaxf(kb2o[3], kb2o[6]);
                float lo_186 = fminf(kb2o[3], kb2o[6]);
                kb2o[3] = hi_185;
                kb2o[6] = lo_186;
                float hi_187 = fmaxf(kb2o[9], kb2o[12]);
                float lo_188 = fminf(kb2o[9], kb2o[12]);
                kb2o[9] = hi_187;
                kb2o[12] = lo_188;
                float hi_189 = fmaxf(kb2o[11], kb2o[13]);
                float lo_190 = fminf(kb2o[11], kb2o[13]);
                kb2o[11] = hi_189;
                kb2o[13] = lo_190;
                float hi_192 = fmaxf(kb2o[3], kb2o[5]);
                float lo_193 = fminf(kb2o[3], kb2o[5]);
                kb2o[3] = hi_192;
                kb2o[5] = lo_193;
                float hi_195 = fmaxf(kb2o[6], kb2o[8]);
                float lo_196 = fminf(kb2o[6], kb2o[8]);
                kb2o[6] = hi_195;
                kb2o[8] = lo_196;
                float hi_198 = fmaxf(kb2o[7], kb2o[9]);
                float lo_199 = fminf(kb2o[7], kb2o[9]);
                kb2o[7] = hi_198;
                kb2o[9] = lo_199;
                float hi_200 = fmaxf(kb2o[10], kb2o[12]);
                float lo_201 = fminf(kb2o[10], kb2o[12]);
                kb2o[10] = hi_200;
                kb2o[12] = lo_201;
                float hi_202 = fmaxf(kb2o[3], kb2o[4]);
                float lo_203 = fminf(kb2o[3], kb2o[4]);
                kb2o[3] = hi_202;
                kb2o[4] = lo_203;
                float hi_204 = fmaxf(kb2o[5], kb2o[6]);
                float lo_205 = fminf(kb2o[5], kb2o[6]);
                kb2o[5] = hi_204;
                kb2o[6] = lo_205;
                float hi_206 = fmaxf(kb2o[7], kb2o[8]);
                float lo_207 = fminf(kb2o[7], kb2o[8]);
                kb2o[7] = hi_206;
                kb2o[8] = lo_207;
                float hi_208 = fmaxf(kb2o[9], kb2o[10]);
                float lo_209 = fminf(kb2o[9], kb2o[10]);
                kb2o[9] = hi_208;
                kb2o[10] = lo_209;
                float hi_211 = fmaxf(kb2o[11], kb2o[12]);
                float lo_212 = fminf(kb2o[11], kb2o[12]);
                kb2o[11] = hi_211;
                kb2o[12] = lo_212;
                float hi_214 = fmaxf(kb2o[6], kb2o[7]);
                float lo_215 = fminf(kb2o[6], kb2o[7]);
                kb2o[6] = hi_214;
                kb2o[7] = lo_215;
                float hi_217 = fmaxf(kb2o[8], kb2o[9]);
                float lo_218 = fminf(kb2o[8], kb2o[9]);
                kb2o[8] = hi_217;
                kb2o[9] = lo_218;
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
            asm volatile("barrier.sync 1, 64;" ::: "memory");
            float r_1 = neg_inf;
            float V_2[8];
            int s0_3 = (sg * 16 * 2 + cg) * 17;
            int s1_4 = ((sg * 16 + 8) * 2 + cg) * 17;
            float x0_5 = pub[s0_3 + ln];
            float y0_6 = pub[s1_4 + lnr];
            float lo0_7 = fminf(x0_5, y0_6);
            r_1 = fmaxf(r_1, lo0_7);
            float hi0_8 = fmaxf(x0_5, y0_6);
            float cur_9 = hi0_8;
            float pv_10 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 8);
            float hi_11 = fmaxf(cur_9, pv_10);
            float lo_12 = fminf(cur_9, pv_10);
            cur_9 = ((up[0] != 0) ? hi_11 : lo_12);
            float pv_13 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 4);
            float hi_14 = fmaxf(cur_9, pv_13);
            float lo_15 = fminf(cur_9, pv_13);
            cur_9 = ((up[1] != 0) ? hi_14 : lo_15);
            float pv_16 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 2);
            float hi_17 = fmaxf(cur_9, pv_16);
            float lo_18 = fminf(cur_9, pv_16);
            cur_9 = ((up[2] != 0) ? hi_17 : lo_18);
            float pv_19 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 1);
            float hi_20 = fmaxf(cur_9, pv_19);
            float lo_21 = fminf(cur_9, pv_19);
            cur_9 = ((up[3] != 0) ? hi_20 : lo_21);
            V_2[0] = cur_9;
            int s0_22 = ((sg * 16 + 1) * 2 + cg) * 17;
            int s1_23 = ((sg * 16 + 1 + 8) * 2 + cg) * 17;
            float x0_24 = pub[s0_22 + ln];
            float y0_25 = pub[s1_23 + lnr];
            float lo0_26 = fminf(x0_24, y0_25);
            r_1 = fmaxf(r_1, lo0_26);
            float hi0_27 = fmaxf(x0_24, y0_25);
            float cur_28 = hi0_27;
            float pv_29 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 8);
            float hi_30 = fmaxf(cur_28, pv_29);
            float lo_31 = fminf(cur_28, pv_29);
            cur_28 = ((up[0] != 0) ? hi_30 : lo_31);
            float pv_32 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 4);
            float hi_33 = fmaxf(cur_28, pv_32);
            float lo_34 = fminf(cur_28, pv_32);
            cur_28 = ((up[1] != 0) ? hi_33 : lo_34);
            float pv_35 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 2);
            float hi_37 = fmaxf(cur_28, pv_35);
            float lo_38 = fminf(cur_28, pv_35);
            cur_28 = ((up[2] != 0) ? hi_37 : lo_38);
            float pv_39 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 1);
            float hi_41 = fmaxf(cur_28, pv_39);
            float lo_42 = fminf(cur_28, pv_39);
            cur_28 = ((up[3] != 0) ? hi_41 : lo_42);
            V_2[1] = cur_28;
            int s0_43 = ((sg * 16 + 2) * 2 + cg) * 17;
            int s1_44 = ((sg * 16 + 2 + 8) * 2 + cg) * 17;
            float x0_45 = pub[s0_43 + ln];
            float y0_46 = pub[s1_44 + lnr];
            float lo0_47 = fminf(x0_45, y0_46);
            r_1 = fmaxf(r_1, lo0_47);
            float hi0_48 = fmaxf(x0_45, y0_46);
            float cur_49 = hi0_48;
            float pv_50 = __shfl_xor_sync(0xFFFFFFFF, cur_49, 8);
            float hi_51 = fmaxf(cur_49, pv_50);
            float lo_52 = fminf(cur_49, pv_50);
            cur_49 = ((up[0] != 0) ? hi_51 : lo_52);
            float pv_53 = __shfl_xor_sync(0xFFFFFFFF, cur_49, 4);
            float hi_55 = fmaxf(cur_49, pv_53);
            float lo_56 = fminf(cur_49, pv_53);
            cur_49 = ((up[1] != 0) ? hi_55 : lo_56);
            float pv_57 = __shfl_xor_sync(0xFFFFFFFF, cur_49, 2);
            float hi_59 = fmaxf(cur_49, pv_57);
            float lo_60 = fminf(cur_49, pv_57);
            cur_49 = ((up[2] != 0) ? hi_59 : lo_60);
            float pv_61 = __shfl_xor_sync(0xFFFFFFFF, cur_49, 1);
            float hi_63 = fmaxf(cur_49, pv_61);
            float lo_64 = fminf(cur_49, pv_61);
            cur_49 = ((up[3] != 0) ? hi_63 : lo_64);
            V_2[2] = cur_49;
            int s0_65 = ((sg * 16 + 3) * 2 + cg) * 17;
            int s1_66 = ((sg * 16 + 3 + 8) * 2 + cg) * 17;
            float x0_67 = pub[s0_65 + ln];
            float y0_68 = pub[s1_66 + lnr];
            float lo0_69 = fminf(x0_67, y0_68);
            r_1 = fmaxf(r_1, lo0_69);
            float hi0_70 = fmaxf(x0_67, y0_68);
            float cur_71 = hi0_70;
            float pv_72 = __shfl_xor_sync(0xFFFFFFFF, cur_71, 8);
            float hi_73 = fmaxf(cur_71, pv_72);
            float lo_74 = fminf(cur_71, pv_72);
            cur_71 = ((up[0] != 0) ? hi_73 : lo_74);
            float pv_75 = __shfl_xor_sync(0xFFFFFFFF, cur_71, 4);
            float hi_77 = fmaxf(cur_71, pv_75);
            float lo_78 = fminf(cur_71, pv_75);
            cur_71 = ((up[1] != 0) ? hi_77 : lo_78);
            float pv_79 = __shfl_xor_sync(0xFFFFFFFF, cur_71, 2);
            float hi_81 = fmaxf(cur_71, pv_79);
            float lo_82 = fminf(cur_71, pv_79);
            cur_71 = ((up[2] != 0) ? hi_81 : lo_82);
            float pv_83 = __shfl_xor_sync(0xFFFFFFFF, cur_71, 1);
            float hi_85 = fmaxf(cur_71, pv_83);
            float lo_86 = fminf(cur_71, pv_83);
            cur_71 = ((up[3] != 0) ? hi_85 : lo_86);
            V_2[3] = cur_71;
            int s0_87 = ((sg * 16 + 4) * 2 + cg) * 17;
            int s1_88 = ((sg * 16 + 4 + 8) * 2 + cg) * 17;
            float x0_89 = pub[s0_87 + ln];
            float y0_90 = pub[s1_88 + lnr];
            float lo0_91 = fminf(x0_89, y0_90);
            r_1 = fmaxf(r_1, lo0_91);
            float hi0_92 = fmaxf(x0_89, y0_90);
            float cur_93 = hi0_92;
            float pv_94 = __shfl_xor_sync(0xFFFFFFFF, cur_93, 8);
            float hi_95_1 = fmaxf(cur_93, pv_94);
            float lo_96_1 = fminf(cur_93, pv_94);
            cur_93 = ((up[0] != 0) ? hi_95_1 : lo_96_1);
            float pv_97 = __shfl_xor_sync(0xFFFFFFFF, cur_93, 4);
            float hi_99_1 = fmaxf(cur_93, pv_97);
            float lo_100_1 = fminf(cur_93, pv_97);
            cur_93 = ((up[1] != 0) ? hi_99_1 : lo_100_1);
            float pv_101 = __shfl_xor_sync(0xFFFFFFFF, cur_93, 2);
            float hi_103_1 = fmaxf(cur_93, pv_101);
            float lo_104_1 = fminf(cur_93, pv_101);
            cur_93 = ((up[2] != 0) ? hi_103_1 : lo_104_1);
            float pv_105 = __shfl_xor_sync(0xFFFFFFFF, cur_93, 1);
            float hi_107_1 = fmaxf(cur_93, pv_105);
            float lo_108_1 = fminf(cur_93, pv_105);
            cur_93 = ((up[3] != 0) ? hi_107_1 : lo_108_1);
            V_2[4] = cur_93;
            int s0_109 = ((sg * 16 + 5) * 2 + cg) * 17;
            int s1_110 = ((sg * 16 + 5 + 8) * 2 + cg) * 17;
            float x0_111 = pub[s0_109 + ln];
            float y0_112 = pub[s1_110 + lnr];
            float lo0_113 = fminf(x0_111, y0_112);
            r_1 = fmaxf(r_1, lo0_113);
            float hi0_114 = fmaxf(x0_111, y0_112);
            float cur_115 = hi0_114;
            float pv_116 = __shfl_xor_sync(0xFFFFFFFF, cur_115, 8);
            float hi_117_1 = fmaxf(cur_115, pv_116);
            float lo_118_1 = fminf(cur_115, pv_116);
            cur_115 = ((up[0] != 0) ? hi_117_1 : lo_118_1);
            float pv_119 = __shfl_xor_sync(0xFFFFFFFF, cur_115, 4);
            float hi_121_1 = fmaxf(cur_115, pv_119);
            float lo_122_1 = fminf(cur_115, pv_119);
            cur_115 = ((up[1] != 0) ? hi_121_1 : lo_122_1);
            float pv_123 = __shfl_xor_sync(0xFFFFFFFF, cur_115, 2);
            float hi_125_1 = fmaxf(cur_115, pv_123);
            float lo_126_1 = fminf(cur_115, pv_123);
            cur_115 = ((up[2] != 0) ? hi_125_1 : lo_126_1);
            float pv_127 = __shfl_xor_sync(0xFFFFFFFF, cur_115, 1);
            float hi_129_1 = fmaxf(cur_115, pv_127);
            float lo_130_1 = fminf(cur_115, pv_127);
            cur_115 = ((up[3] != 0) ? hi_129_1 : lo_130_1);
            V_2[5] = cur_115;
            int s0_131 = ((sg * 16 + 6) * 2 + cg) * 17;
            int s1_132 = ((sg * 16 + 6 + 8) * 2 + cg) * 17;
            float x0_133 = pub[s0_131 + ln];
            float y0_134 = pub[s1_132 + lnr];
            float lo0_135 = fminf(x0_133, y0_134);
            r_1 = fmaxf(r_1, lo0_135);
            float hi0_136 = fmaxf(x0_133, y0_134);
            float cur_137 = hi0_136;
            float pv_138 = __shfl_xor_sync(0xFFFFFFFF, cur_137, 8);
            float hi_139_1 = fmaxf(cur_137, pv_138);
            float lo_140_1 = fminf(cur_137, pv_138);
            cur_137 = ((up[0] != 0) ? hi_139_1 : lo_140_1);
            float pv_141 = __shfl_xor_sync(0xFFFFFFFF, cur_137, 4);
            float hi_143_1 = fmaxf(cur_137, pv_141);
            float lo_144_1 = fminf(cur_137, pv_141);
            cur_137 = ((up[1] != 0) ? hi_143_1 : lo_144_1);
            float pv_145 = __shfl_xor_sync(0xFFFFFFFF, cur_137, 2);
            float hi_147_1 = fmaxf(cur_137, pv_145);
            float lo_148_1 = fminf(cur_137, pv_145);
            cur_137 = ((up[2] != 0) ? hi_147_1 : lo_148_1);
            float pv_149 = __shfl_xor_sync(0xFFFFFFFF, cur_137, 1);
            float hi_151_1 = fmaxf(cur_137, pv_149);
            float lo_152_1 = fminf(cur_137, pv_149);
            cur_137 = ((up[3] != 0) ? hi_151_1 : lo_152_1);
            V_2[6] = cur_137;
            int s0_153 = ((sg * 16 + 7) * 2 + cg) * 17;
            int s1_154 = ((sg * 16 + 7 + 8) * 2 + cg) * 17;
            float x0_155 = pub[s0_153 + ln];
            float y0_156 = pub[s1_154 + lnr];
            float lo0_157 = fminf(x0_155, y0_156);
            r_1 = fmaxf(r_1, lo0_157);
            float hi0_158 = fmaxf(x0_155, y0_156);
            float cur_159 = hi0_158;
            float pv_160 = __shfl_xor_sync(0xFFFFFFFF, cur_159, 8);
            float hi_161 = fmaxf(cur_159, pv_160);
            float lo_162 = fminf(cur_159, pv_160);
            cur_159 = ((up[0] != 0) ? hi_161 : lo_162);
            float pv_163 = __shfl_xor_sync(0xFFFFFFFF, cur_159, 4);
            float hi_164_1 = fmaxf(cur_159, pv_163);
            float lo_165_1 = fminf(cur_159, pv_163);
            cur_159 = ((up[1] != 0) ? hi_164_1 : lo_165_1);
            float pv_166 = __shfl_xor_sync(0xFFFFFFFF, cur_159, 2);
            float hi_167 = fmaxf(cur_159, pv_166);
            float lo_168 = fminf(cur_159, pv_166);
            cur_159 = ((up[2] != 0) ? hi_167 : lo_168);
            float pv_169 = __shfl_xor_sync(0xFFFFFFFF, cur_159, 1);
            float hi_170_1 = fmaxf(cur_159, pv_169);
            float lo_171_1 = fminf(cur_159, pv_169);
            cur_159 = ((up[3] != 0) ? hi_170_1 : lo_171_1);
            V_2[7] = cur_159;
            float rs_172 = pub[((sg * 16 + ln) * 2 + cg) * 17 + 16];
            r_1 = fmaxf(r_1, rs_172);
            float y1_173 = __shfl_xor_sync(0xFFFFFFFF, V_2[4], 15);
            float lo1_174 = fminf(V_2[0], y1_173);
            r_1 = fmaxf(r_1, lo1_174);
            float hi1_175 = fmaxf(V_2[0], y1_173);
            float cur_176 = hi1_175;
            float pv_178 = __shfl_xor_sync(0xFFFFFFFF, cur_176, 8);
            float hi_179_1 = fmaxf(cur_176, pv_178);
            float lo_180_1 = fminf(cur_176, pv_178);
            cur_176 = ((up[0] != 0) ? hi_179_1 : lo_180_1);
            float pv_181 = __shfl_xor_sync(0xFFFFFFFF, cur_176, 4);
            float hi_182 = fmaxf(cur_176, pv_181);
            float lo_183 = fminf(cur_176, pv_181);
            cur_176 = ((up[1] != 0) ? hi_182 : lo_183);
            float pv_184 = __shfl_xor_sync(0xFFFFFFFF, cur_176, 2);
            float hi_185_1 = fmaxf(cur_176, pv_184);
            float lo_186_1 = fminf(cur_176, pv_184);
            cur_176 = ((up[2] != 0) ? hi_185_1 : lo_186_1);
            float pv_188 = __shfl_xor_sync(0xFFFFFFFF, cur_176, 1);
            float hi_189_1 = fmaxf(cur_176, pv_188);
            float lo_190_1 = fminf(cur_176, pv_188);
            cur_176 = ((up[3] != 0) ? hi_189_1 : lo_190_1);
            V_2[0] = cur_176;
            float y1_191 = __shfl_xor_sync(0xFFFFFFFF, V_2[5], 15);
            float lo1_192 = fminf(V_2[1], y1_191);
            r_1 = fmaxf(r_1, lo1_192);
            float hi1_193 = fmaxf(V_2[1], y1_191);
            float cur_194 = hi1_193;
            float pv_195 = __shfl_xor_sync(0xFFFFFFFF, cur_194, 8);
            float hi_196 = fmaxf(cur_194, pv_195);
            float lo_197 = fminf(cur_194, pv_195);
            cur_194 = ((up[0] != 0) ? hi_196 : lo_197);
            float pv_198 = __shfl_xor_sync(0xFFFFFFFF, cur_194, 4);
            float hi_199 = fmaxf(cur_194, pv_198);
            float lo_200 = fminf(cur_194, pv_198);
            cur_194 = ((up[1] != 0) ? hi_199 : lo_200);
            float pv_201 = __shfl_xor_sync(0xFFFFFFFF, cur_194, 2);
            float hi_202_1 = fmaxf(cur_194, pv_201);
            float lo_203_1 = fminf(cur_194, pv_201);
            cur_194 = ((up[2] != 0) ? hi_202_1 : lo_203_1);
            float pv_204 = __shfl_xor_sync(0xFFFFFFFF, cur_194, 1);
            float hi_205 = fmaxf(cur_194, pv_204);
            float lo_206 = fminf(cur_194, pv_204);
            cur_194 = ((up[3] != 0) ? hi_205 : lo_206);
            V_2[1] = cur_194;
            float y1_207 = __shfl_xor_sync(0xFFFFFFFF, V_2[6], 15);
            float lo1_208 = fminf(V_2[2], y1_207);
            r_1 = fmaxf(r_1, lo1_208);
            float hi1_209 = fmaxf(V_2[2], y1_207);
            float cur_210 = hi1_209;
            float pv_211 = __shfl_xor_sync(0xFFFFFFFF, cur_210, 8);
            float hi_212 = fmaxf(cur_210, pv_211);
            float lo_213 = fminf(cur_210, pv_211);
            cur_210 = ((up[0] != 0) ? hi_212 : lo_213);
            float pv_214 = __shfl_xor_sync(0xFFFFFFFF, cur_210, 4);
            float hi_215 = fmaxf(cur_210, pv_214);
            float lo_216 = fminf(cur_210, pv_214);
            cur_210 = ((up[1] != 0) ? hi_215 : lo_216);
            float pv_217 = __shfl_xor_sync(0xFFFFFFFF, cur_210, 2);
            float hi_218 = fmaxf(cur_210, pv_217);
            float lo_219 = fminf(cur_210, pv_217);
            cur_210 = ((up[2] != 0) ? hi_218 : lo_219);
            float pv_220 = __shfl_xor_sync(0xFFFFFFFF, cur_210, 1);
            float hi_221 = fmaxf(cur_210, pv_220);
            float lo_222 = fminf(cur_210, pv_220);
            cur_210 = ((up[3] != 0) ? hi_221 : lo_222);
            V_2[2] = cur_210;
            float y1_223 = __shfl_xor_sync(0xFFFFFFFF, V_2[7], 15);
            float lo1_224 = fminf(V_2[3], y1_223);
            r_1 = fmaxf(r_1, lo1_224);
            float hi1_225 = fmaxf(V_2[3], y1_223);
            float cur_226 = hi1_225;
            float pv_227 = __shfl_xor_sync(0xFFFFFFFF, cur_226, 8);
            float hi_228 = fmaxf(cur_226, pv_227);
            float lo_229 = fminf(cur_226, pv_227);
            cur_226 = ((up[0] != 0) ? hi_228 : lo_229);
            float pv_230 = __shfl_xor_sync(0xFFFFFFFF, cur_226, 4);
            float hi_231 = fmaxf(cur_226, pv_230);
            float lo_232 = fminf(cur_226, pv_230);
            cur_226 = ((up[1] != 0) ? hi_231 : lo_232);
            float pv_233 = __shfl_xor_sync(0xFFFFFFFF, cur_226, 2);
            float hi_234 = fmaxf(cur_226, pv_233);
            float lo_235 = fminf(cur_226, pv_233);
            cur_226 = ((up[2] != 0) ? hi_234 : lo_235);
            float pv_236 = __shfl_xor_sync(0xFFFFFFFF, cur_226, 1);
            float hi_237 = fmaxf(cur_226, pv_236);
            float lo_238 = fminf(cur_226, pv_236);
            cur_226 = ((up[3] != 0) ? hi_237 : lo_238);
            V_2[3] = cur_226;
            float y1_239 = __shfl_xor_sync(0xFFFFFFFF, V_2[2], 15);
            float lo1_240 = fminf(V_2[0], y1_239);
            r_1 = fmaxf(r_1, lo1_240);
            float hi1_241 = fmaxf(V_2[0], y1_239);
            float cur_242 = hi1_241;
            float pv_243 = __shfl_xor_sync(0xFFFFFFFF, cur_242, 8);
            float hi_244 = fmaxf(cur_242, pv_243);
            float lo_245 = fminf(cur_242, pv_243);
            cur_242 = ((up[0] != 0) ? hi_244 : lo_245);
            float pv_246 = __shfl_xor_sync(0xFFFFFFFF, cur_242, 4);
            float hi_247 = fmaxf(cur_242, pv_246);
            float lo_248 = fminf(cur_242, pv_246);
            cur_242 = ((up[1] != 0) ? hi_247 : lo_248);
            float pv_249 = __shfl_xor_sync(0xFFFFFFFF, cur_242, 2);
            float hi_250 = fmaxf(cur_242, pv_249);
            float lo_251 = fminf(cur_242, pv_249);
            cur_242 = ((up[2] != 0) ? hi_250 : lo_251);
            float pv_252 = __shfl_xor_sync(0xFFFFFFFF, cur_242, 1);
            float hi_253 = fmaxf(cur_242, pv_252);
            float lo_254 = fminf(cur_242, pv_252);
            cur_242 = ((up[3] != 0) ? hi_253 : lo_254);
            V_2[0] = cur_242;
            float y1_255 = __shfl_xor_sync(0xFFFFFFFF, V_2[3], 15);
            float lo1_256 = fminf(V_2[1], y1_255);
            r_1 = fmaxf(r_1, lo1_256);
            float hi1_257 = fmaxf(V_2[1], y1_255);
            float cur_258 = hi1_257;
            float pv_259 = __shfl_xor_sync(0xFFFFFFFF, cur_258, 8);
            float hi_260 = fmaxf(cur_258, pv_259);
            float lo_261 = fminf(cur_258, pv_259);
            cur_258 = ((up[0] != 0) ? hi_260 : lo_261);
            float pv_262 = __shfl_xor_sync(0xFFFFFFFF, cur_258, 4);
            float hi_263 = fmaxf(cur_258, pv_262);
            float lo_264 = fminf(cur_258, pv_262);
            cur_258 = ((up[1] != 0) ? hi_263 : lo_264);
            float pv_265 = __shfl_xor_sync(0xFFFFFFFF, cur_258, 2);
            float hi_266 = fmaxf(cur_258, pv_265);
            float lo_267 = fminf(cur_258, pv_265);
            cur_258 = ((up[2] != 0) ? hi_266 : lo_267);
            float pv_268 = __shfl_xor_sync(0xFFFFFFFF, cur_258, 1);
            float hi_269 = fmaxf(cur_258, pv_268);
            float lo_270 = fminf(cur_258, pv_268);
            cur_258 = ((up[3] != 0) ? hi_269 : lo_270);
            V_2[1] = cur_258;
            float yl_271 = __shfl_xor_sync(0xFFFFFFFF, V_2[1], 15);
            float lol_272 = fminf(V_2[0], yl_271);
            r_1 = fmaxf(r_1, lol_272);
            float hil_273 = fmaxf(V_2[0], yl_271);
            float cur_274 = hil_273;
            float pv_275 = __shfl_xor_sync(0xFFFFFFFF, cur_274, 8);
            float hi_276 = fmaxf(cur_274, pv_275);
            float lo_277 = fminf(cur_274, pv_275);
            cur_274 = ((up[0] != 0) ? hi_276 : lo_277);
            float pv_278 = __shfl_xor_sync(0xFFFFFFFF, cur_274, 4);
            float hi_279 = fmaxf(cur_274, pv_278);
            float lo_280 = fminf(cur_274, pv_278);
            cur_274 = ((up[1] != 0) ? hi_279 : lo_280);
            float pv_281 = __shfl_xor_sync(0xFFFFFFFF, cur_274, 2);
            float hi_282 = fmaxf(cur_274, pv_281);
            float lo_283 = fminf(cur_274, pv_281);
            cur_274 = ((up[2] != 0) ? hi_282 : lo_283);
            float pv_284 = __shfl_xor_sync(0xFFFFFFFF, cur_274, 1);
            float hi_285 = fmaxf(cur_274, pv_284);
            float lo_286 = fminf(cur_274, pv_284);
            cur_274 = ((up[3] != 0) ? hi_285 : lo_286);
            V_2[0] = cur_274;
            float K_287 = V_2[0];
            int qb_288 = (sg * 2 + cg) * 32;
            q2[qb_288 + ln] = K_287;
            q2[qb_288 + 16 + ln] = r_1;
            asm volatile("barrier.sync 1, 64;" ::: "memory");
            float k2o_289[1];
            float r2v_290 = neg_inf;
            r2v_290 = fmaxf(r2v_290, q2[cg * 32 + 16 + ln]);
            r2v_290 = fmaxf(r2v_290, q2[(2 + cg) * 32 + 16 + ln]);
            float V2_291[1];
            float x2_292 = q2[cg * 32 + ln];
            float y2_293 = q2[(2 + cg) * 32 + lnr];
            float lo2_294 = fminf(x2_292, y2_293);
            r2v_290 = fmaxf(r2v_290, lo2_294);
            float hi2_295 = fmaxf(x2_292, y2_293);
            V2_291[0] = hi2_295;
            k2o_289[0] = V2_291[0];
            K_287 = k2o_289[0];
            r_1 = r2v_290;
            rr2x[0] = r_1;
            K2 = K_287;
        }
        if (tid_1 < 32) {
            unsigned int cgx = ccnt[cg];
            unsigned int k1x = __as_u32(K);
            int isq = 0;
            if (gflag != 0 && (k1x & 4294966784u) == qg) {
                isq = 1;
            }
            unsigned int mqx = __ballot_sync(0xFFFFFFFF, isq != 0);
            unsigned int gmx = mqx & 65535;
            if ((tid_1 & 16) != 0) {
                gmx = mqx >> 16 & 65535;
            }
            unsigned int lowx = (1 << (unsigned int)ln) - 1;
            int rx = (int)(unsigned int)__popc(gmx & lowx);
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
                        float hi_0 = fmaxf(xsl[0], xsl[13]);
                        float lo_1 = fminf(xsl[0], xsl[13]);
                        xsl[0] = hi_0;
                        xsl[13] = lo_1;
                        float hi_2 = fmaxf(xsl[1], xsl[12]);
                        float lo_3 = fminf(xsl[1], xsl[12]);
                        xsl[1] = hi_2;
                        xsl[12] = lo_3;
                        float hi_4 = fmaxf(xsl[2], xsl[15]);
                        float lo_5 = fminf(xsl[2], xsl[15]);
                        xsl[2] = hi_4;
                        xsl[15] = lo_5;
                        float hi_6 = fmaxf(xsl[3], xsl[14]);
                        float lo_7 = fminf(xsl[3], xsl[14]);
                        xsl[3] = hi_6;
                        xsl[14] = lo_7;
                        float hi_8 = fmaxf(xsl[4], xsl[8]);
                        float lo_9 = fminf(xsl[4], xsl[8]);
                        xsl[4] = hi_8;
                        xsl[8] = lo_9;
                        float hi_10 = fmaxf(xsl[5], xsl[6]);
                        float lo_11 = fminf(xsl[5], xsl[6]);
                        xsl[5] = hi_10;
                        xsl[6] = lo_11;
                        float hi_12 = fmaxf(xsl[7], xsl[11]);
                        float lo_13 = fminf(xsl[7], xsl[11]);
                        xsl[7] = hi_12;
                        xsl[11] = lo_13;
                        float hi_14_1 = fmaxf(xsl[9], xsl[10]);
                        float lo_15_1 = fminf(xsl[9], xsl[10]);
                        xsl[9] = hi_14_1;
                        xsl[10] = lo_15_1;
                        float hi_16 = fmaxf(xsl[0], xsl[5]);
                        float lo_17 = fminf(xsl[0], xsl[5]);
                        xsl[0] = hi_16;
                        xsl[5] = lo_17;
                        float hi_18 = fmaxf(xsl[1], xsl[7]);
                        float lo_19 = fminf(xsl[1], xsl[7]);
                        xsl[1] = hi_18;
                        xsl[7] = lo_19;
                        float hi_20_1 = fmaxf(xsl[2], xsl[9]);
                        float lo_21_1 = fminf(xsl[2], xsl[9]);
                        xsl[2] = hi_20_1;
                        xsl[9] = lo_21_1;
                        float hi_22 = fmaxf(xsl[3], xsl[4]);
                        float lo_23 = fminf(xsl[3], xsl[4]);
                        xsl[3] = hi_22;
                        xsl[4] = lo_23;
                        float hi_24 = fmaxf(xsl[6], xsl[13]);
                        float lo_25 = fminf(xsl[6], xsl[13]);
                        xsl[6] = hi_24;
                        xsl[13] = lo_25;
                        float hi_26 = fmaxf(xsl[8], xsl[14]);
                        float lo_27 = fminf(xsl[8], xsl[14]);
                        xsl[8] = hi_26;
                        xsl[14] = lo_27;
                        float hi_28 = fmaxf(xsl[10], xsl[15]);
                        float lo_29 = fminf(xsl[10], xsl[15]);
                        xsl[10] = hi_28;
                        xsl[15] = lo_29;
                        float hi_30_1 = fmaxf(xsl[11], xsl[12]);
                        float lo_31_1 = fminf(xsl[11], xsl[12]);
                        xsl[11] = hi_30_1;
                        xsl[12] = lo_31_1;
                        float hi_33_1 = fmaxf(xsl[0], xsl[1]);
                        float lo_34_1 = fminf(xsl[0], xsl[1]);
                        xsl[0] = hi_33_1;
                        xsl[1] = lo_34_1;
                        float hi_35 = fmaxf(xsl[2], xsl[3]);
                        float lo_36 = fminf(xsl[2], xsl[3]);
                        xsl[2] = hi_35;
                        xsl[3] = lo_36;
                        float hi_37_1 = fmaxf(xsl[4], xsl[5]);
                        float lo_38_1 = fminf(xsl[4], xsl[5]);
                        xsl[4] = hi_37_1;
                        xsl[5] = lo_38_1;
                        float hi_39 = fmaxf(xsl[6], xsl[8]);
                        float lo_40 = fminf(xsl[6], xsl[8]);
                        xsl[6] = hi_39;
                        xsl[8] = lo_40;
                        float hi_41_1 = fmaxf(xsl[7], xsl[9]);
                        float lo_42_1 = fminf(xsl[7], xsl[9]);
                        xsl[7] = hi_41_1;
                        xsl[9] = lo_42_1;
                        float hi_43 = fmaxf(xsl[10], xsl[11]);
                        float lo_44 = fminf(xsl[10], xsl[11]);
                        xsl[10] = hi_43;
                        xsl[11] = lo_44;
                        float hi_45 = fmaxf(xsl[12], xsl[13]);
                        float lo_46 = fminf(xsl[12], xsl[13]);
                        xsl[12] = hi_45;
                        xsl[13] = lo_46;
                        float hi_47 = fmaxf(xsl[14], xsl[15]);
                        float lo_48 = fminf(xsl[14], xsl[15]);
                        xsl[14] = hi_47;
                        xsl[15] = lo_48;
                        float hi_49 = fmaxf(xsl[0], xsl[2]);
                        float lo_50 = fminf(xsl[0], xsl[2]);
                        xsl[0] = hi_49;
                        xsl[2] = lo_50;
                        float hi_51_1 = fmaxf(xsl[1], xsl[3]);
                        float lo_52_1 = fminf(xsl[1], xsl[3]);
                        xsl[1] = hi_51_1;
                        xsl[3] = lo_52_1;
                        float hi_53 = fmaxf(xsl[4], xsl[10]);
                        float lo_54 = fminf(xsl[4], xsl[10]);
                        xsl[4] = hi_53;
                        xsl[10] = lo_54;
                        float hi_55_1 = fmaxf(xsl[5], xsl[11]);
                        float lo_56_1 = fminf(xsl[5], xsl[11]);
                        xsl[5] = hi_55_1;
                        xsl[11] = lo_56_1;
                        float hi_57 = fmaxf(xsl[6], xsl[7]);
                        float lo_58 = fminf(xsl[6], xsl[7]);
                        xsl[6] = hi_57;
                        xsl[7] = lo_58;
                        float hi_59_1 = fmaxf(xsl[8], xsl[9]);
                        float lo_60_1 = fminf(xsl[8], xsl[9]);
                        xsl[8] = hi_59_1;
                        xsl[9] = lo_60_1;
                        float hi_61 = fmaxf(xsl[12], xsl[14]);
                        float lo_62 = fminf(xsl[12], xsl[14]);
                        xsl[12] = hi_61;
                        xsl[14] = lo_62;
                        float hi_63_1 = fmaxf(xsl[13], xsl[15]);
                        float lo_64_1 = fminf(xsl[13], xsl[15]);
                        xsl[13] = hi_63_1;
                        xsl[15] = lo_64_1;
                        float hi_65 = fmaxf(xsl[1], xsl[2]);
                        float lo_66 = fminf(xsl[1], xsl[2]);
                        xsl[1] = hi_65;
                        xsl[2] = lo_66;
                        float hi_67 = fmaxf(xsl[3], xsl[12]);
                        float lo_68 = fminf(xsl[3], xsl[12]);
                        xsl[3] = hi_67;
                        xsl[12] = lo_68;
                        float hi_69 = fmaxf(xsl[4], xsl[6]);
                        float lo_70 = fminf(xsl[4], xsl[6]);
                        xsl[4] = hi_69;
                        xsl[6] = lo_70;
                        float hi_71 = fmaxf(xsl[5], xsl[7]);
                        float lo_72 = fminf(xsl[5], xsl[7]);
                        xsl[5] = hi_71;
                        xsl[7] = lo_72;
                        float hi_73_1 = fmaxf(xsl[8], xsl[10]);
                        float lo_74_1 = fminf(xsl[8], xsl[10]);
                        xsl[8] = hi_73_1;
                        xsl[10] = lo_74_1;
                        float hi_75 = fmaxf(xsl[9], xsl[11]);
                        float lo_76 = fminf(xsl[9], xsl[11]);
                        xsl[9] = hi_75;
                        xsl[11] = lo_76;
                        float hi_77_1 = fmaxf(xsl[13], xsl[14]);
                        float lo_78_1 = fminf(xsl[13], xsl[14]);
                        xsl[13] = hi_77_1;
                        xsl[14] = lo_78_1;
                        float hi_79 = fmaxf(xsl[1], xsl[4]);
                        float lo_80 = fminf(xsl[1], xsl[4]);
                        xsl[1] = hi_79;
                        xsl[4] = lo_80;
                        float hi_81_1 = fmaxf(xsl[2], xsl[6]);
                        float lo_82_1 = fminf(xsl[2], xsl[6]);
                        xsl[2] = hi_81_1;
                        xsl[6] = lo_82_1;
                        float hi_83 = fmaxf(xsl[5], xsl[8]);
                        float lo_84 = fminf(xsl[5], xsl[8]);
                        xsl[5] = hi_83;
                        xsl[8] = lo_84;
                        float hi_85_1 = fmaxf(xsl[7], xsl[10]);
                        float lo_86_1 = fminf(xsl[7], xsl[10]);
                        xsl[7] = hi_85_1;
                        xsl[10] = lo_86_1;
                        float hi_87_1 = fmaxf(xsl[9], xsl[13]);
                        float lo_88_1 = fminf(xsl[9], xsl[13]);
                        xsl[9] = hi_87_1;
                        xsl[13] = lo_88_1;
                        float hi_89_1 = fmaxf(xsl[11], xsl[14]);
                        float lo_90_1 = fminf(xsl[11], xsl[14]);
                        xsl[11] = hi_89_1;
                        xsl[14] = lo_90_1;
                        float hi_91_1 = fmaxf(xsl[2], xsl[4]);
                        float lo_92_1 = fminf(xsl[2], xsl[4]);
                        xsl[2] = hi_91_1;
                        xsl[4] = lo_92_1;
                        float hi_93_1 = fmaxf(xsl[3], xsl[6]);
                        float lo_94_1 = fminf(xsl[3], xsl[6]);
                        xsl[3] = hi_93_1;
                        xsl[6] = lo_94_1;
                        float hi_95_2 = fmaxf(xsl[9], xsl[12]);
                        float lo_96_2 = fminf(xsl[9], xsl[12]);
                        xsl[9] = hi_95_2;
                        xsl[12] = lo_96_2;
                        float hi_97_1 = fmaxf(xsl[11], xsl[13]);
                        float lo_98_1 = fminf(xsl[11], xsl[13]);
                        xsl[11] = hi_97_1;
                        xsl[13] = lo_98_1;
                        float hi_99_2 = fmaxf(xsl[3], xsl[5]);
                        float lo_100_2 = fminf(xsl[3], xsl[5]);
                        xsl[3] = hi_99_2;
                        xsl[5] = lo_100_2;
                        float hi_101_1 = fmaxf(xsl[6], xsl[8]);
                        float lo_102_1 = fminf(xsl[6], xsl[8]);
                        xsl[6] = hi_101_1;
                        xsl[8] = lo_102_1;
                        float hi_103_2 = fmaxf(xsl[7], xsl[9]);
                        float lo_104_2 = fminf(xsl[7], xsl[9]);
                        xsl[7] = hi_103_2;
                        xsl[9] = lo_104_2;
                        float hi_105_1 = fmaxf(xsl[10], xsl[12]);
                        float lo_106_1 = fminf(xsl[10], xsl[12]);
                        xsl[10] = hi_105_1;
                        xsl[12] = lo_106_1;
                        float hi_107_2 = fmaxf(xsl[3], xsl[4]);
                        float lo_108_2 = fminf(xsl[3], xsl[4]);
                        xsl[3] = hi_107_2;
                        xsl[4] = lo_108_2;
                        float hi_109_1 = fmaxf(xsl[5], xsl[6]);
                        float lo_110_1 = fminf(xsl[5], xsl[6]);
                        xsl[5] = hi_109_1;
                        xsl[6] = lo_110_1;
                        float hi_111_1 = fmaxf(xsl[7], xsl[8]);
                        float lo_112_1 = fminf(xsl[7], xsl[8]);
                        xsl[7] = hi_111_1;
                        xsl[8] = lo_112_1;
                        float hi_113_1 = fmaxf(xsl[9], xsl[10]);
                        float lo_114_1 = fminf(xsl[9], xsl[10]);
                        xsl[9] = hi_113_1;
                        xsl[10] = lo_114_1;
                        float hi_115_1 = fmaxf(xsl[11], xsl[12]);
                        float lo_116_1 = fminf(xsl[11], xsl[12]);
                        xsl[11] = hi_115_1;
                        xsl[12] = lo_116_1;
                        float hi_117_2 = fmaxf(xsl[6], xsl[7]);
                        float lo_118_2 = fminf(xsl[6], xsl[7]);
                        xsl[6] = hi_117_2;
                        xsl[7] = lo_118_2;
                        float hi_119_1 = fmaxf(xsl[8], xsl[9]);
                        float lo_120_1 = fminf(xsl[8], xsl[9]);
                        xsl[8] = hi_119_1;
                        xsl[9] = lo_120_1;
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
    if (tid_1 < 32) {
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
        unsigned int ox = __shfl_xor_sync(0xFFFFFFFF, rkey, 1);
        if (ox < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_0 = __shfl_xor_sync(0xFFFFFFFF, rkey, 2);
        if (ox_0 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_1 = __shfl_xor_sync(0xFFFFFFFF, rkey, 3);
        if (ox_1 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_2 = __shfl_xor_sync(0xFFFFFFFF, rkey, 4);
        if (ox_2 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_3 = __shfl_xor_sync(0xFFFFFFFF, rkey, 5);
        if (ox_3 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_4 = __shfl_xor_sync(0xFFFFFFFF, rkey, 6);
        if (ox_4 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_5 = __shfl_xor_sync(0xFFFFFFFF, rkey, 7);
        if (ox_5 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_6 = __shfl_xor_sync(0xFFFFFFFF, rkey, 8);
        if (ox_6 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_7 = __shfl_xor_sync(0xFFFFFFFF, rkey, 9);
        if (ox_7 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_8 = __shfl_xor_sync(0xFFFFFFFF, rkey, 10);
        if (ox_8 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_9 = __shfl_xor_sync(0xFFFFFFFF, rkey, 11);
        if (ox_9 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_10 = __shfl_xor_sync(0xFFFFFFFF, rkey, 12);
        if (ox_10 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_11 = __shfl_xor_sync(0xFFFFFFFF, rkey, 13);
        if (ox_11 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_12 = __shfl_xor_sync(0xFFFFFFFF, rkey, 14);
        if (ox_12 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_13 = __shfl_xor_sync(0xFFFFFFFF, rkey, 15);
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
