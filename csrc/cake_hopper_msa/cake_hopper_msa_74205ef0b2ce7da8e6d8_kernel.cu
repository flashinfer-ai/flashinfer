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

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_hopper_msa_74205ef0b2ce7da8e6d8(unsigned int* __restrict__ S, int* __restrict__ nvp, int* __restrict__ out, int tiles, int total_q, int num_heads, int num_chunks, int fb, int fe)
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

    // Prologue
    float* tree = reinterpret_cast<float*>(smem_raw + 0);
    const int tree_addr = smem + 0;
    unsigned int* qcol = reinterpret_cast<unsigned int*>(smem_raw + 8704);
    const int qcol_addr = smem + 8704;
    unsigned int* flagw = reinterpret_cast<unsigned int*>(smem_raw + 8832);
    const int flagw_addr = smem + 8832;
    unsigned int* ccnt = reinterpret_cast<unsigned int*>(smem_raw + 8848);
    const int ccnt_addr = smem + 8848;
    int* limc = reinterpret_cast<int*>(smem_raw + 8976);
    const int limc_addr = smem + 8976;
    unsigned int* cbuf = reinterpret_cast<unsigned int*>(smem_raw + 9104);
    const int cbuf_addr = smem + 9104;

    // === Task calls (declaration order) ===
    int tid_1 = threadIdx.x;
    int w = tid_1 / 32;
    int c = tid_1 - w * 32;
    int whi = w;
    int col = blockIdx.x * 32 + c;
    int head = blockIdx.y;
    if (tid_1 == 0) {
        flagw[0] = 0;
    }
    if (tid_1 == 0) {
        flagw[1] = 0;
    }
    if (tid_1 < 32) {
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
        cb[2] = S[p + 2ll * nq64];
    }
    cb[3] = 4286578688u;
    if (lim_full > t0 + 3) {
        cb[3] = S[p + 3ll * nq64];
    }
    cb[4] = 4286578688u;
    if (lim_full > t0 + 4) {
        cb[4] = S[p + 4ll * nq64];
    }
    cb[5] = 4286578688u;
    if (lim_full > t0 + 5) {
        cb[5] = S[p + 5ll * nq64];
    }
    cb[6] = 4286578688u;
    if (lim_full > t0 + 6) {
        cb[6] = S[p + 6ll * nq64];
    }
    cb[7] = 4286578688u;
    if (lim_full > t0 + 7) {
        cb[7] = S[p + 7ll * nq64];
    }
    cb[8] = 4286578688u;
    if (lim_full > t0 + 8) {
        cb[8] = S[p + 8ll * nq64];
    }
    cb[9] = 4286578688u;
    if (lim_full > t0 + 9) {
        cb[9] = S[p + 9ll * nq64];
    }
    cb[10] = 4286578688u;
    if (lim_full > t0 + 10) {
        cb[10] = S[p + 10ll * nq64];
    }
    cb[11] = 4286578688u;
    if (lim_full > t0 + 11) {
        cb[11] = S[p + 11ll * nq64];
    }
    cb[12] = 4286578688u;
    if (lim_full > t0 + 12) {
        cb[12] = S[p + 12ll * nq64];
    }
    cb[13] = 4286578688u;
    if (lim_full > t0 + 13) {
        cb[13] = S[p + 13ll * nq64];
    }
    cb[14] = 4286578688u;
    if (lim_full > t0 + 14) {
        cb[14] = S[p + 14ll * nq64];
    }
    cb[15] = 4286578688u;
    if (lim_full > t0 + 15) {
        cb[15] = S[p + 15ll * nq64];
    }
    asm volatile("" ::: "memory");
    int t0_1 = w * 16;
    if (lim <= t0_1) {
        cb[0] = 4286578688u;
    }
    if (lim <= t0_1 + 1) {
        cb[1] = 4286578688u;
    }
    if (lim <= t0_1 + 2) {
        cb[2] = 4286578688u;
    }
    if (lim <= t0_1 + 3) {
        cb[3] = 4286578688u;
    }
    if (lim <= t0_1 + 4) {
        cb[4] = 4286578688u;
    }
    if (lim <= t0_1 + 5) {
        cb[5] = 4286578688u;
    }
    if (lim <= t0_1 + 6) {
        cb[6] = 4286578688u;
    }
    if (lim <= t0_1 + 7) {
        cb[7] = 4286578688u;
    }
    if (lim <= t0_1 + 8) {
        cb[8] = 4286578688u;
    }
    if (lim <= t0_1 + 9) {
        cb[9] = 4286578688u;
    }
    if (lim <= t0_1 + 10) {
        cb[10] = 4286578688u;
    }
    if (lim <= t0_1 + 11) {
        cb[11] = 4286578688u;
    }
    if (lim <= t0_1 + 12) {
        cb[12] = 4286578688u;
    }
    if (lim <= t0_1 + 13) {
        cb[13] = 4286578688u;
    }
    if (lim <= t0_1 + 14) {
        cb[14] = 4286578688u;
    }
    if (lim <= t0_1 + 15) {
        cb[15] = 4286578688u;
    }
    int t0_2 = w * 16;
    float sc = __uint_as_float(cb[0]);
    sc = fmaxf(sc, -1.7014118e+38f);
    sc = fminf(sc, 1.7014118e+38f);
    unsigned int kbits = __as_u32(sc);
    if (cb[0] == 4286578688u) {
        kbits = 4282384384u;
    }
    unsigned int key_1_0 = kbits & 4294967232u | (unsigned int)t0_2;
    kb[0] = __uint_as_float(key_1_0);
    float sc_0 = __uint_as_float(cb[1]);
    sc_0 = fmaxf(sc_0, -1.7014118e+38f);
    sc_0 = fminf(sc_0, 1.7014118e+38f);
    unsigned int kbits_0 = __as_u32(sc_0);
    if (cb[1] == 4286578688u) {
        kbits_0 = 4282384384u;
    }
    unsigned int key_1_1 = kbits_0 & 4294967232u | (unsigned int)(t0_2 + 1);
    kb[1] = __uint_as_float(key_1_1);
    float sc_1 = __uint_as_float(cb[2]);
    sc_1 = fmaxf(sc_1, -1.7014118e+38f);
    sc_1 = fminf(sc_1, 1.7014118e+38f);
    unsigned int kbits_1 = __as_u32(sc_1);
    if (cb[2] == 4286578688u) {
        kbits_1 = 4282384384u;
    }
    unsigned int key_1_2 = kbits_1 & 4294967232u | (unsigned int)(t0_2 + 2);
    kb[2] = __uint_as_float(key_1_2);
    float sc_2 = __uint_as_float(cb[3]);
    sc_2 = fmaxf(sc_2, -1.7014118e+38f);
    sc_2 = fminf(sc_2, 1.7014118e+38f);
    unsigned int kbits_2 = __as_u32(sc_2);
    if (cb[3] == 4286578688u) {
        kbits_2 = 4282384384u;
    }
    unsigned int key_1_3 = kbits_2 & 4294967232u | (unsigned int)(t0_2 + 3);
    kb[3] = __uint_as_float(key_1_3);
    float sc_3 = __uint_as_float(cb[4]);
    sc_3 = fmaxf(sc_3, -1.7014118e+38f);
    sc_3 = fminf(sc_3, 1.7014118e+38f);
    unsigned int kbits_3 = __as_u32(sc_3);
    if (cb[4] == 4286578688u) {
        kbits_3 = 4282384384u;
    }
    unsigned int key_1_4 = kbits_3 & 4294967232u | (unsigned int)(t0_2 + 4);
    kb[4] = __uint_as_float(key_1_4);
    float sc_4 = __uint_as_float(cb[5]);
    sc_4 = fmaxf(sc_4, -1.7014118e+38f);
    sc_4 = fminf(sc_4, 1.7014118e+38f);
    unsigned int kbits_4 = __as_u32(sc_4);
    if (cb[5] == 4286578688u) {
        kbits_4 = 4282384384u;
    }
    unsigned int key_1_5 = kbits_4 & 4294967232u | (unsigned int)(t0_2 + 5);
    kb[5] = __uint_as_float(key_1_5);
    float sc_5 = __uint_as_float(cb[6]);
    sc_5 = fmaxf(sc_5, -1.7014118e+38f);
    sc_5 = fminf(sc_5, 1.7014118e+38f);
    unsigned int kbits_5 = __as_u32(sc_5);
    if (cb[6] == 4286578688u) {
        kbits_5 = 4282384384u;
    }
    unsigned int key_1_6 = kbits_5 & 4294967232u | (unsigned int)(t0_2 + 6);
    kb[6] = __uint_as_float(key_1_6);
    float sc_6 = __uint_as_float(cb[7]);
    sc_6 = fmaxf(sc_6, -1.7014118e+38f);
    sc_6 = fminf(sc_6, 1.7014118e+38f);
    unsigned int kbits_6 = __as_u32(sc_6);
    if (cb[7] == 4286578688u) {
        kbits_6 = 4282384384u;
    }
    unsigned int key_1_7 = kbits_6 & 4294967232u | (unsigned int)(t0_2 + 7);
    kb[7] = __uint_as_float(key_1_7);
    float sc_7 = __uint_as_float(cb[8]);
    sc_7 = fmaxf(sc_7, -1.7014118e+38f);
    sc_7 = fminf(sc_7, 1.7014118e+38f);
    unsigned int kbits_7 = __as_u32(sc_7);
    if (cb[8] == 4286578688u) {
        kbits_7 = 4282384384u;
    }
    unsigned int key_1_8 = kbits_7 & 4294967232u | (unsigned int)(t0_2 + 8);
    kb[8] = __uint_as_float(key_1_8);
    float sc_8 = __uint_as_float(cb[9]);
    sc_8 = fmaxf(sc_8, -1.7014118e+38f);
    sc_8 = fminf(sc_8, 1.7014118e+38f);
    unsigned int kbits_8 = __as_u32(sc_8);
    if (cb[9] == 4286578688u) {
        kbits_8 = 4282384384u;
    }
    unsigned int key_1_9 = kbits_8 & 4294967232u | (unsigned int)(t0_2 + 9);
    kb[9] = __uint_as_float(key_1_9);
    float sc_9 = __uint_as_float(cb[10]);
    sc_9 = fmaxf(sc_9, -1.7014118e+38f);
    sc_9 = fminf(sc_9, 1.7014118e+38f);
    unsigned int kbits_9 = __as_u32(sc_9);
    if (cb[10] == 4286578688u) {
        kbits_9 = 4282384384u;
    }
    unsigned int key_1_10 = kbits_9 & 4294967232u | (unsigned int)(t0_2 + 10);
    kb[10] = __uint_as_float(key_1_10);
    float sc_10 = __uint_as_float(cb[11]);
    sc_10 = fmaxf(sc_10, -1.7014118e+38f);
    sc_10 = fminf(sc_10, 1.7014118e+38f);
    unsigned int kbits_10 = __as_u32(sc_10);
    if (cb[11] == 4286578688u) {
        kbits_10 = 4282384384u;
    }
    unsigned int key_1_11 = kbits_10 & 4294967232u | (unsigned int)(t0_2 + 11);
    kb[11] = __uint_as_float(key_1_11);
    float sc_11 = __uint_as_float(cb[12]);
    sc_11 = fmaxf(sc_11, -1.7014118e+38f);
    sc_11 = fminf(sc_11, 1.7014118e+38f);
    unsigned int kbits_11 = __as_u32(sc_11);
    if (cb[12] == 4286578688u) {
        kbits_11 = 4282384384u;
    }
    unsigned int key_1_12 = kbits_11 & 4294967232u | (unsigned int)(t0_2 + 12);
    kb[12] = __uint_as_float(key_1_12);
    float sc_12 = __uint_as_float(cb[13]);
    sc_12 = fmaxf(sc_12, -1.7014118e+38f);
    sc_12 = fminf(sc_12, 1.7014118e+38f);
    unsigned int kbits_12 = __as_u32(sc_12);
    if (cb[13] == 4286578688u) {
        kbits_12 = 4282384384u;
    }
    unsigned int key_1_13 = kbits_12 & 4294967232u | (unsigned int)(t0_2 + 13);
    kb[13] = __uint_as_float(key_1_13);
    float sc_13 = __uint_as_float(cb[14]);
    sc_13 = fmaxf(sc_13, -1.7014118e+38f);
    sc_13 = fminf(sc_13, 1.7014118e+38f);
    unsigned int kbits_13 = __as_u32(sc_13);
    if (cb[14] == 4286578688u) {
        kbits_13 = 4282384384u;
    }
    unsigned int key_1_14 = kbits_13 & 4294967232u | (unsigned int)(t0_2 + 14);
    kb[14] = __uint_as_float(key_1_14);
    float sc_14 = __uint_as_float(cb[15]);
    sc_14 = fmaxf(sc_14, -1.7014118e+38f);
    sc_14 = fminf(sc_14, 1.7014118e+38f);
    unsigned int kbits_14 = __as_u32(sc_14);
    if (cb[15] == 4286578688u) {
        kbits_14 = 4282384384u;
    }
    unsigned int key_1_15 = kbits_14 & 4294967232u | (unsigned int)(t0_2 + 15);
    kb[15] = __uint_as_float(key_1_15);
    int f = 0;
    if ((t0_2 < fb || t0_2 + 16 > lim - fe) && t0_2 < lim) {
        f = 1;
    }
    if (f != 0) {
        int f_0 = 0;
        if ((t0_2 < fb || t0_2 >= lim - fe) && lim > t0_2) {
            f_0 = 1;
        }
        if (f_0 != 0) {
            kb[0] = __uint_as_float(2139094976 | (unsigned int)t0_2);
        }
        int f_1 = 0;
        if ((t0_2 + 1 < fb || t0_2 + 1 >= lim - fe) && lim > t0_2 + 1) {
            f_1 = 1;
        }
        if (f_1 != 0) {
            kb[1] = __uint_as_float(2139094976 | (unsigned int)(t0_2 + 1));
        }
        int f_2 = 0;
        if ((t0_2 + 2 < fb || t0_2 + 2 >= lim - fe) && lim > t0_2 + 2) {
            f_2 = 1;
        }
        if (f_2 != 0) {
            kb[2] = __uint_as_float(2139094976 | (unsigned int)(t0_2 + 2));
        }
        int f_3 = 0;
        if ((t0_2 + 3 < fb || t0_2 + 3 >= lim - fe) && lim > t0_2 + 3) {
            f_3 = 1;
        }
        if (f_3 != 0) {
            kb[3] = __uint_as_float(2139094976 | (unsigned int)(t0_2 + 3));
        }
        int f_4 = 0;
        if ((t0_2 + 4 < fb || t0_2 + 4 >= lim - fe) && lim > t0_2 + 4) {
            f_4 = 1;
        }
        if (f_4 != 0) {
            kb[4] = __uint_as_float(2139094976 | (unsigned int)(t0_2 + 4));
        }
        int f_5 = 0;
        if ((t0_2 + 5 < fb || t0_2 + 5 >= lim - fe) && lim > t0_2 + 5) {
            f_5 = 1;
        }
        if (f_5 != 0) {
            kb[5] = __uint_as_float(2139094976 | (unsigned int)(t0_2 + 5));
        }
        int f_6 = 0;
        if ((t0_2 + 6 < fb || t0_2 + 6 >= lim - fe) && lim > t0_2 + 6) {
            f_6 = 1;
        }
        if (f_6 != 0) {
            kb[6] = __uint_as_float(2139094976 | (unsigned int)(t0_2 + 6));
        }
        int f_7 = 0;
        if ((t0_2 + 7 < fb || t0_2 + 7 >= lim - fe) && lim > t0_2 + 7) {
            f_7 = 1;
        }
        if (f_7 != 0) {
            kb[7] = __uint_as_float(2139094976 | (unsigned int)(t0_2 + 7));
        }
        int f_8 = 0;
        if ((t0_2 + 8 < fb || t0_2 + 8 >= lim - fe) && lim > t0_2 + 8) {
            f_8 = 1;
        }
        if (f_8 != 0) {
            kb[8] = __uint_as_float(2139094976 | (unsigned int)(t0_2 + 8));
        }
        int f_9 = 0;
        if ((t0_2 + 9 < fb || t0_2 + 9 >= lim - fe) && lim > t0_2 + 9) {
            f_9 = 1;
        }
        if (f_9 != 0) {
            kb[9] = __uint_as_float(2139094976 | (unsigned int)(t0_2 + 9));
        }
        int f_10 = 0;
        if ((t0_2 + 10 < fb || t0_2 + 10 >= lim - fe) && lim > t0_2 + 10) {
            f_10 = 1;
        }
        if (f_10 != 0) {
            kb[10] = __uint_as_float(2139094976 | (unsigned int)(t0_2 + 10));
        }
        int f_11 = 0;
        if ((t0_2 + 11 < fb || t0_2 + 11 >= lim - fe) && lim > t0_2 + 11) {
            f_11 = 1;
        }
        if (f_11 != 0) {
            kb[11] = __uint_as_float(2139094976 | (unsigned int)(t0_2 + 11));
        }
        int f_12 = 0;
        if ((t0_2 + 12 < fb || t0_2 + 12 >= lim - fe) && lim > t0_2 + 12) {
            f_12 = 1;
        }
        if (f_12 != 0) {
            kb[12] = __uint_as_float(2139094976 | (unsigned int)(t0_2 + 12));
        }
        int f_13 = 0;
        if ((t0_2 + 13 < fb || t0_2 + 13 >= lim - fe) && lim > t0_2 + 13) {
            f_13 = 1;
        }
        if (f_13 != 0) {
            kb[13] = __uint_as_float(2139094976 | (unsigned int)(t0_2 + 13));
        }
        int f_14 = 0;
        if ((t0_2 + 14 < fb || t0_2 + 14 >= lim - fe) && lim > t0_2 + 14) {
            f_14 = 1;
        }
        if (f_14 != 0) {
            kb[14] = __uint_as_float(2139094976 | (unsigned int)(t0_2 + 14));
        }
        int f_15 = 0;
        if ((t0_2 + 15 < fb || t0_2 + 15 >= lim - fe) && lim > t0_2 + 15) {
            f_15 = 1;
        }
        if (f_15 != 0) {
            kb[15] = __uint_as_float(2139094976 | (unsigned int)(t0_2 + 15));
        }
    }
    float hi_1_0 = fmaxf(kb[0], kb[13]);
    float lo_1_0 = fminf(kb[0], kb[13]);
    kb[0] = hi_1_0;
    kb[13] = lo_1_0;
    float hi_1_1 = fmaxf(kb[1], kb[12]);
    float lo_1_1 = fminf(kb[1], kb[12]);
    kb[1] = hi_1_1;
    kb[12] = lo_1_1;
    float hi_1_2 = fmaxf(kb[2], kb[15]);
    float lo_1_2 = fminf(kb[2], kb[15]);
    kb[2] = hi_1_2;
    kb[15] = lo_1_2;
    float hi_1_3 = fmaxf(kb[3], kb[14]);
    float lo_1_3 = fminf(kb[3], kb[14]);
    kb[3] = hi_1_3;
    kb[14] = lo_1_3;
    float hi_1_4 = fmaxf(kb[4], kb[8]);
    float lo_1_4 = fminf(kb[4], kb[8]);
    kb[4] = hi_1_4;
    kb[8] = lo_1_4;
    float hi_1_5 = fmaxf(kb[5], kb[6]);
    float lo_1_5 = fminf(kb[5], kb[6]);
    kb[5] = hi_1_5;
    kb[6] = lo_1_5;
    float hi_1_6 = fmaxf(kb[7], kb[11]);
    float lo_1_6 = fminf(kb[7], kb[11]);
    kb[7] = hi_1_6;
    kb[11] = lo_1_6;
    float hi_1_7 = fmaxf(kb[9], kb[10]);
    float lo_1_7 = fminf(kb[9], kb[10]);
    kb[9] = hi_1_7;
    kb[10] = lo_1_7;
    float hi_1_8 = fmaxf(kb[0], kb[5]);
    float lo_1_8 = fminf(kb[0], kb[5]);
    kb[0] = hi_1_8;
    kb[5] = lo_1_8;
    float hi_1_9 = fmaxf(kb[1], kb[7]);
    float lo_1_9 = fminf(kb[1], kb[7]);
    kb[1] = hi_1_9;
    kb[7] = lo_1_9;
    float hi_1_10 = fmaxf(kb[2], kb[9]);
    float lo_1_10 = fminf(kb[2], kb[9]);
    kb[2] = hi_1_10;
    kb[9] = lo_1_10;
    float hi_1_11 = fmaxf(kb[3], kb[4]);
    float lo_1_11 = fminf(kb[3], kb[4]);
    kb[3] = hi_1_11;
    kb[4] = lo_1_11;
    float hi_1_12 = fmaxf(kb[6], kb[13]);
    float lo_1_12 = fminf(kb[6], kb[13]);
    kb[6] = hi_1_12;
    kb[13] = lo_1_12;
    float hi_1_13 = fmaxf(kb[8], kb[14]);
    float lo_1_13 = fminf(kb[8], kb[14]);
    kb[8] = hi_1_13;
    kb[14] = lo_1_13;
    float hi_1_14 = fmaxf(kb[10], kb[15]);
    float lo_1_14 = fminf(kb[10], kb[15]);
    kb[10] = hi_1_14;
    kb[15] = lo_1_14;
    float hi_1_15 = fmaxf(kb[11], kb[12]);
    float lo_1_15 = fminf(kb[11], kb[12]);
    kb[11] = hi_1_15;
    kb[12] = lo_1_15;
    float hi_1_16 = fmaxf(kb[0], kb[1]);
    float lo_1_16 = fminf(kb[0], kb[1]);
    kb[0] = hi_1_16;
    kb[1] = lo_1_16;
    float hi_1_17 = fmaxf(kb[2], kb[3]);
    float lo_1_17 = fminf(kb[2], kb[3]);
    kb[2] = hi_1_17;
    kb[3] = lo_1_17;
    float hi_1_18 = fmaxf(kb[4], kb[5]);
    float lo_1_18 = fminf(kb[4], kb[5]);
    kb[4] = hi_1_18;
    kb[5] = lo_1_18;
    float hi_1_19 = fmaxf(kb[6], kb[8]);
    float lo_1_19 = fminf(kb[6], kb[8]);
    kb[6] = hi_1_19;
    kb[8] = lo_1_19;
    float hi_1_20 = fmaxf(kb[7], kb[9]);
    float lo_1_20 = fminf(kb[7], kb[9]);
    kb[7] = hi_1_20;
    kb[9] = lo_1_20;
    float hi_1_21 = fmaxf(kb[10], kb[11]);
    float lo_1_21 = fminf(kb[10], kb[11]);
    kb[10] = hi_1_21;
    kb[11] = lo_1_21;
    float hi_1_22 = fmaxf(kb[12], kb[13]);
    float lo_1_22 = fminf(kb[12], kb[13]);
    kb[12] = hi_1_22;
    kb[13] = lo_1_22;
    float hi_1_23 = fmaxf(kb[14], kb[15]);
    float lo_1_23 = fminf(kb[14], kb[15]);
    kb[14] = hi_1_23;
    kb[15] = lo_1_23;
    float hi_1_24 = fmaxf(kb[0], kb[2]);
    float lo_1_24 = fminf(kb[0], kb[2]);
    kb[0] = hi_1_24;
    kb[2] = lo_1_24;
    float hi_1_25 = fmaxf(kb[1], kb[3]);
    float lo_1_25 = fminf(kb[1], kb[3]);
    kb[1] = hi_1_25;
    kb[3] = lo_1_25;
    float hi_1_26 = fmaxf(kb[4], kb[10]);
    float lo_1_26 = fminf(kb[4], kb[10]);
    kb[4] = hi_1_26;
    kb[10] = lo_1_26;
    float hi_1_27 = fmaxf(kb[5], kb[11]);
    float lo_1_27 = fminf(kb[5], kb[11]);
    kb[5] = hi_1_27;
    kb[11] = lo_1_27;
    float hi_1_28 = fmaxf(kb[6], kb[7]);
    float lo_1_28 = fminf(kb[6], kb[7]);
    kb[6] = hi_1_28;
    kb[7] = lo_1_28;
    float hi_1_29 = fmaxf(kb[8], kb[9]);
    float lo_1_29 = fminf(kb[8], kb[9]);
    kb[8] = hi_1_29;
    kb[9] = lo_1_29;
    float hi_1_30 = fmaxf(kb[12], kb[14]);
    float lo_1_30 = fminf(kb[12], kb[14]);
    kb[12] = hi_1_30;
    kb[14] = lo_1_30;
    float hi_1_31 = fmaxf(kb[13], kb[15]);
    float lo_1_31 = fminf(kb[13], kb[15]);
    kb[13] = hi_1_31;
    kb[15] = lo_1_31;
    float hi_1_32 = fmaxf(kb[1], kb[2]);
    float lo_1_32 = fminf(kb[1], kb[2]);
    kb[1] = hi_1_32;
    kb[2] = lo_1_32;
    float hi_1_33 = fmaxf(kb[3], kb[12]);
    float lo_1_33 = fminf(kb[3], kb[12]);
    kb[3] = hi_1_33;
    kb[12] = lo_1_33;
    float hi_1_34 = fmaxf(kb[4], kb[6]);
    float lo_1_34 = fminf(kb[4], kb[6]);
    kb[4] = hi_1_34;
    kb[6] = lo_1_34;
    float hi_1_35 = fmaxf(kb[5], kb[7]);
    float lo_1_35 = fminf(kb[5], kb[7]);
    kb[5] = hi_1_35;
    kb[7] = lo_1_35;
    float hi_1_36 = fmaxf(kb[8], kb[10]);
    float lo_1_36 = fminf(kb[8], kb[10]);
    kb[8] = hi_1_36;
    kb[10] = lo_1_36;
    float hi_1_37 = fmaxf(kb[9], kb[11]);
    float lo_1_37 = fminf(kb[9], kb[11]);
    kb[9] = hi_1_37;
    kb[11] = lo_1_37;
    float hi_1_38 = fmaxf(kb[13], kb[14]);
    float lo_1_38 = fminf(kb[13], kb[14]);
    kb[13] = hi_1_38;
    kb[14] = lo_1_38;
    float hi_1_39 = fmaxf(kb[1], kb[4]);
    float lo_1_39 = fminf(kb[1], kb[4]);
    kb[1] = hi_1_39;
    kb[4] = lo_1_39;
    float hi_1_40 = fmaxf(kb[2], kb[6]);
    float lo_1_40 = fminf(kb[2], kb[6]);
    kb[2] = hi_1_40;
    kb[6] = lo_1_40;
    float hi_1_41 = fmaxf(kb[5], kb[8]);
    float lo_1_41 = fminf(kb[5], kb[8]);
    kb[5] = hi_1_41;
    kb[8] = lo_1_41;
    float hi_1_42 = fmaxf(kb[7], kb[10]);
    float lo_1_42 = fminf(kb[7], kb[10]);
    kb[7] = hi_1_42;
    kb[10] = lo_1_42;
    float hi_1_43 = fmaxf(kb[9], kb[13]);
    float lo_1_43 = fminf(kb[9], kb[13]);
    kb[9] = hi_1_43;
    kb[13] = lo_1_43;
    float hi_1_44 = fmaxf(kb[11], kb[14]);
    float lo_1_44 = fminf(kb[11], kb[14]);
    kb[11] = hi_1_44;
    kb[14] = lo_1_44;
    float hi_1_45 = fmaxf(kb[2], kb[4]);
    float lo_1_45 = fminf(kb[2], kb[4]);
    kb[2] = hi_1_45;
    kb[4] = lo_1_45;
    float hi_1_46 = fmaxf(kb[3], kb[6]);
    float lo_1_46 = fminf(kb[3], kb[6]);
    kb[3] = hi_1_46;
    kb[6] = lo_1_46;
    float hi_1_47 = fmaxf(kb[9], kb[12]);
    float lo_1_47 = fminf(kb[9], kb[12]);
    kb[9] = hi_1_47;
    kb[12] = lo_1_47;
    float hi_1_48 = fmaxf(kb[11], kb[13]);
    float lo_1_48 = fminf(kb[11], kb[13]);
    kb[11] = hi_1_48;
    kb[13] = lo_1_48;
    float hi_1_49 = fmaxf(kb[3], kb[5]);
    float lo_1_49 = fminf(kb[3], kb[5]);
    kb[3] = hi_1_49;
    kb[5] = lo_1_49;
    float hi_1_50 = fmaxf(kb[6], kb[8]);
    float lo_1_50 = fminf(kb[6], kb[8]);
    kb[6] = hi_1_50;
    kb[8] = lo_1_50;
    float hi_1_51 = fmaxf(kb[7], kb[9]);
    float lo_1_51 = fminf(kb[7], kb[9]);
    kb[7] = hi_1_51;
    kb[9] = lo_1_51;
    float hi_1_52 = fmaxf(kb[10], kb[12]);
    float lo_1_52 = fminf(kb[10], kb[12]);
    kb[10] = hi_1_52;
    kb[12] = lo_1_52;
    float hi_1_53 = fmaxf(kb[3], kb[4]);
    float lo_1_53 = fminf(kb[3], kb[4]);
    kb[3] = hi_1_53;
    kb[4] = lo_1_53;
    float hi_1_54 = fmaxf(kb[5], kb[6]);
    float lo_1_54 = fminf(kb[5], kb[6]);
    kb[5] = hi_1_54;
    kb[6] = lo_1_54;
    float hi_1_55 = fmaxf(kb[7], kb[8]);
    float lo_1_55 = fminf(kb[7], kb[8]);
    kb[7] = hi_1_55;
    kb[8] = lo_1_55;
    float hi_1_56 = fmaxf(kb[9], kb[10]);
    float lo_1_56 = fminf(kb[9], kb[10]);
    kb[9] = hi_1_56;
    kb[10] = lo_1_56;
    float hi_1_57 = fmaxf(kb[11], kb[12]);
    float lo_1_57 = fminf(kb[11], kb[12]);
    kb[11] = hi_1_57;
    kb[12] = lo_1_57;
    float hi_1_58 = fmaxf(kb[6], kb[7]);
    float lo_1_58 = fminf(kb[6], kb[7]);
    kb[6] = hi_1_58;
    kb[7] = lo_1_58;
    float hi_1_59 = fmaxf(kb[8], kb[9]);
    float lo_1_59 = fminf(kb[8], kb[9]);
    kb[8] = hi_1_59;
    kb[9] = lo_1_59;
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
        int bit = 1 << k;
        int low = whi & (bit << 1) - 1;
        if (low == bit) {
            int base = whi * 544 + c;
            tree[base] = a[0];
            tree[base + 32] = a[1];
            tree[base + 64] = a[2];
            tree[base + 96] = a[3];
            tree[base + 128] = a[4];
            tree[base + 160] = a[5];
            tree[base + 192] = a[6];
            tree[base + 224] = a[7];
            tree[base + 256] = a[8];
            tree[base + 288] = a[9];
            tree[base + 320] = a[10];
            tree[base + 352] = a[11];
            tree[base + 384] = a[12];
            tree[base + 416] = a[13];
            tree[base + 448] = a[14];
            tree[base + 480] = a[15];
            tree[base + 512] = rej;
        }
        asm volatile("barrier.sync 1, 128;" ::: "memory");
        if (low == 0) {
            float sb[16];
            int rbase = (whi + bit) * 544 + c;
            sb[0] = tree[rbase];
            sb[1] = tree[rbase + 32];
            sb[2] = tree[rbase + 64];
            sb[3] = tree[rbase + 96];
            sb[4] = tree[rbase + 128];
            sb[5] = tree[rbase + 160];
            sb[6] = tree[rbase + 192];
            sb[7] = tree[rbase + 224];
            sb[8] = tree[rbase + 256];
            sb[9] = tree[rbase + 288];
            sb[10] = tree[rbase + 320];
            sb[11] = tree[rbase + 352];
            sb[12] = tree[rbase + 384];
            sb[13] = tree[rbase + 416];
            sb[14] = tree[rbase + 448];
            sb[15] = tree[rbase + 480];
            float srej = tree[rbase + 512];
            rej = fmaxf(rej, srej);
            float r = rej;
            float hi_0 = fmaxf(a[0], sb[15]);
            float lo_0 = fminf(a[0], sb[15]);
            a[0] = hi_0;
            r = fmaxf(r, lo_0);
            float hi_1 = fmaxf(a[1], sb[14]);
            float lo_1 = fminf(a[1], sb[14]);
            a[1] = hi_1;
            r = fmaxf(r, lo_1);
            float hi_2 = fmaxf(a[2], sb[13]);
            float lo_2 = fminf(a[2], sb[13]);
            a[2] = hi_2;
            r = fmaxf(r, lo_2);
            float hi_3 = fmaxf(a[3], sb[12]);
            float lo_3 = fminf(a[3], sb[12]);
            a[3] = hi_3;
            r = fmaxf(r, lo_3);
            float hi_4 = fmaxf(a[4], sb[11]);
            float lo_4 = fminf(a[4], sb[11]);
            a[4] = hi_4;
            r = fmaxf(r, lo_4);
            float hi_5 = fmaxf(a[5], sb[10]);
            float lo_5 = fminf(a[5], sb[10]);
            a[5] = hi_5;
            r = fmaxf(r, lo_5);
            float hi_6 = fmaxf(a[6], sb[9]);
            float lo_6 = fminf(a[6], sb[9]);
            a[6] = hi_6;
            r = fmaxf(r, lo_6);
            float hi_7 = fmaxf(a[7], sb[8]);
            float lo_7 = fminf(a[7], sb[8]);
            a[7] = hi_7;
            r = fmaxf(r, lo_7);
            float hi_8 = fmaxf(a[8], sb[7]);
            float lo_8 = fminf(a[8], sb[7]);
            a[8] = hi_8;
            r = fmaxf(r, lo_8);
            float hi_9 = fmaxf(a[9], sb[6]);
            float lo_9 = fminf(a[9], sb[6]);
            a[9] = hi_9;
            r = fmaxf(r, lo_9);
            float hi_10 = fmaxf(a[10], sb[5]);
            float lo_10 = fminf(a[10], sb[5]);
            a[10] = hi_10;
            r = fmaxf(r, lo_10);
            float hi_11 = fmaxf(a[11], sb[4]);
            float lo_11 = fminf(a[11], sb[4]);
            a[11] = hi_11;
            r = fmaxf(r, lo_11);
            float hi_12 = fmaxf(a[12], sb[3]);
            float lo_12 = fminf(a[12], sb[3]);
            a[12] = hi_12;
            r = fmaxf(r, lo_12);
            float hi_13 = fmaxf(a[13], sb[2]);
            float lo_13 = fminf(a[13], sb[2]);
            a[13] = hi_13;
            r = fmaxf(r, lo_13);
            float hi_14 = fmaxf(a[14], sb[1]);
            float lo_14 = fminf(a[14], sb[1]);
            a[14] = hi_14;
            r = fmaxf(r, lo_14);
            float hi_15 = fmaxf(a[15], sb[0]);
            float lo_15 = fminf(a[15], sb[0]);
            a[15] = hi_15;
            r = fmaxf(r, lo_15);
            float hi_16 = fmaxf(a[0], a[8]);
            float lo_16 = fminf(a[0], a[8]);
            a[0] = hi_16;
            a[8] = lo_16;
            float hi_17 = fmaxf(a[1], a[9]);
            float lo_17 = fminf(a[1], a[9]);
            a[1] = hi_17;
            a[9] = lo_17;
            float hi_18 = fmaxf(a[2], a[10]);
            float lo_18 = fminf(a[2], a[10]);
            a[2] = hi_18;
            a[10] = lo_18;
            float hi_19 = fmaxf(a[3], a[11]);
            float lo_19 = fminf(a[3], a[11]);
            a[3] = hi_19;
            a[11] = lo_19;
            float hi_20 = fmaxf(a[4], a[12]);
            float lo_20 = fminf(a[4], a[12]);
            a[4] = hi_20;
            a[12] = lo_20;
            float hi_21 = fmaxf(a[5], a[13]);
            float lo_21 = fminf(a[5], a[13]);
            a[5] = hi_21;
            a[13] = lo_21;
            float hi_22 = fmaxf(a[6], a[14]);
            float lo_22 = fminf(a[6], a[14]);
            a[6] = hi_22;
            a[14] = lo_22;
            float hi_23 = fmaxf(a[7], a[15]);
            float lo_23 = fminf(a[7], a[15]);
            a[7] = hi_23;
            a[15] = lo_23;
            float hi_24 = fmaxf(a[0], a[4]);
            float lo_24 = fminf(a[0], a[4]);
            a[0] = hi_24;
            a[4] = lo_24;
            float hi_25 = fmaxf(a[1], a[5]);
            float lo_25 = fminf(a[1], a[5]);
            a[1] = hi_25;
            a[5] = lo_25;
            float hi_26 = fmaxf(a[2], a[6]);
            float lo_26 = fminf(a[2], a[6]);
            a[2] = hi_26;
            a[6] = lo_26;
            float hi_27 = fmaxf(a[3], a[7]);
            float lo_27 = fminf(a[3], a[7]);
            a[3] = hi_27;
            a[7] = lo_27;
            float hi_28 = fmaxf(a[8], a[12]);
            float lo_28 = fminf(a[8], a[12]);
            a[8] = hi_28;
            a[12] = lo_28;
            float hi_29 = fmaxf(a[9], a[13]);
            float lo_29 = fminf(a[9], a[13]);
            a[9] = hi_29;
            a[13] = lo_29;
            float hi_30 = fmaxf(a[10], a[14]);
            float lo_30 = fminf(a[10], a[14]);
            a[10] = hi_30;
            a[14] = lo_30;
            float hi_31 = fmaxf(a[11], a[15]);
            float lo_31 = fminf(a[11], a[15]);
            a[11] = hi_31;
            a[15] = lo_31;
            float hi_32 = fmaxf(a[0], a[2]);
            float lo_32 = fminf(a[0], a[2]);
            a[0] = hi_32;
            a[2] = lo_32;
            float hi_33 = fmaxf(a[1], a[3]);
            float lo_33 = fminf(a[1], a[3]);
            a[1] = hi_33;
            a[3] = lo_33;
            float hi_34 = fmaxf(a[4], a[6]);
            float lo_34 = fminf(a[4], a[6]);
            a[4] = hi_34;
            a[6] = lo_34;
            float hi_35 = fmaxf(a[5], a[7]);
            float lo_35 = fminf(a[5], a[7]);
            a[5] = hi_35;
            a[7] = lo_35;
            float hi_36 = fmaxf(a[8], a[10]);
            float lo_36 = fminf(a[8], a[10]);
            a[8] = hi_36;
            a[10] = lo_36;
            float hi_37 = fmaxf(a[9], a[11]);
            float lo_37 = fminf(a[9], a[11]);
            a[9] = hi_37;
            a[11] = lo_37;
            float hi_38 = fmaxf(a[12], a[14]);
            float lo_38 = fminf(a[12], a[14]);
            a[12] = hi_38;
            a[14] = lo_38;
            float hi_39 = fmaxf(a[13], a[15]);
            float lo_39 = fminf(a[13], a[15]);
            a[13] = hi_39;
            a[15] = lo_39;
            float hi_40 = fmaxf(a[0], a[1]);
            float lo_40 = fminf(a[0], a[1]);
            a[0] = hi_40;
            a[1] = lo_40;
            float hi_41 = fmaxf(a[2], a[3]);
            float lo_41 = fminf(a[2], a[3]);
            a[2] = hi_41;
            a[3] = lo_41;
            float hi_42 = fmaxf(a[4], a[5]);
            float lo_42 = fminf(a[4], a[5]);
            a[4] = hi_42;
            a[5] = lo_42;
            float hi_43 = fmaxf(a[6], a[7]);
            float lo_43 = fminf(a[6], a[7]);
            a[6] = hi_43;
            a[7] = lo_43;
            float hi_44 = fmaxf(a[8], a[9]);
            float lo_44 = fminf(a[8], a[9]);
            a[8] = hi_44;
            a[9] = lo_44;
            float hi_45 = fmaxf(a[10], a[11]);
            float lo_45 = fminf(a[10], a[11]);
            a[10] = hi_45;
            a[11] = lo_45;
            float hi_46 = fmaxf(a[12], a[13]);
            float lo_46 = fminf(a[12], a[13]);
            a[12] = hi_46;
            a[13] = lo_46;
            float hi_47 = fmaxf(a[14], a[15]);
            float lo_47 = fminf(a[14], a[15]);
            a[14] = hi_47;
            a[15] = lo_47;
            rej = r;
        }
    }
    if (w == 0) {
        unsigned int u16 = __as_u32(a[15]);
        unsigned int c16 = u16 & 4294967232u;
        unsigned int cr = __as_u32(rej) & 4294967232u;
        unsigned int q = 4294967295u;
        if (c16 == cr && u16 < 4278190080u && c16 != 2139094976 && col < total_q) {
            q = c16;
            flagw[0] = 1;
        }
        qcol[c] = q;
    }
    asm volatile("barrier.sync 1, 128;" ::: "memory");
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
        if (flagged != 0) {
            int t0_3 = w * 16;
            float sc_15 = __uint_as_float(cb[0]);
            sc_15 = fmaxf(sc_15, -1.7014118e+38f);
            sc_15 = fminf(sc_15, 1.7014118e+38f);
            unsigned int u8 = __as_u32(sc_15);
            unsigned int key8 = 0;
            if ((u8 & 4294967232u) == qc) {
                unsigned int lowb8 = (u8 ^ (unsigned int)((int)u8 >> 31) & 63) & 63;
                key8 = 536870912 | lowb8 << 6 | (unsigned int)t0_3;
            }
            int f2 = 0;
            if ((t0_3 < fb || t0_3 >= lim - fe) && lim > t0_3) {
                f2 = 1;
            }
            if (f2 != 0) {
                key8 = 0;
            }
            if (key8 != 0) {
                unsigned int p6 = atomicAdd(&ccnt[c], 1);
                if (p6 < 16) {
                    cbuf[c * 16 + (int)p6] = key8;
                } else {
                    flagw[1] = 1;
                }
            }
            float sc_16 = __uint_as_float(cb[1]);
            sc_16 = fmaxf(sc_16, -1.7014118e+38f);
            sc_16 = fminf(sc_16, 1.7014118e+38f);
            unsigned int u8_0 = __as_u32(sc_16);
            unsigned int key8_0 = 0;
            if ((u8_0 & 4294967232u) == qc) {
                unsigned int lowb8_1 = (u8_0 ^ (unsigned int)((int)u8_0 >> 31) & 63) & 63;
                key8_0 = 536870912 | lowb8_1 << 6 | (unsigned int)(t0_3 + 1);
            }
            int f2_0 = 0;
            if ((t0_3 + 1 < fb || t0_3 + 1 >= lim - fe) && lim > t0_3 + 1) {
                f2_0 = 1;
            }
            if (f2_0 != 0) {
                key8_0 = 0;
            }
            if (key8_0 != 0) {
                unsigned int p6_1 = atomicAdd(&ccnt[c], 1);
                if (p6_1 < 16) {
                    cbuf[c * 16 + (int)p6_1] = key8_0;
                } else {
                    flagw[1] = 1;
                }
            }
            float sc_17 = __uint_as_float(cb[2]);
            sc_17 = fmaxf(sc_17, -1.7014118e+38f);
            sc_17 = fminf(sc_17, 1.7014118e+38f);
            unsigned int u8_1 = __as_u32(sc_17);
            unsigned int key8_1 = 0;
            if ((u8_1 & 4294967232u) == qc) {
                unsigned int lowb8_2 = (u8_1 ^ (unsigned int)((int)u8_1 >> 31) & 63) & 63;
                key8_1 = 536870912 | lowb8_2 << 6 | (unsigned int)(t0_3 + 2);
            }
            int f2_1 = 0;
            if ((t0_3 + 2 < fb || t0_3 + 2 >= lim - fe) && lim > t0_3 + 2) {
                f2_1 = 1;
            }
            if (f2_1 != 0) {
                key8_1 = 0;
            }
            if (key8_1 != 0) {
                unsigned int p6_2 = atomicAdd(&ccnt[c], 1);
                if (p6_2 < 16) {
                    cbuf[c * 16 + (int)p6_2] = key8_1;
                } else {
                    flagw[1] = 1;
                }
            }
            float sc_18 = __uint_as_float(cb[3]);
            sc_18 = fmaxf(sc_18, -1.7014118e+38f);
            sc_18 = fminf(sc_18, 1.7014118e+38f);
            unsigned int u8_2 = __as_u32(sc_18);
            unsigned int key8_2 = 0;
            if ((u8_2 & 4294967232u) == qc) {
                unsigned int lowb8_3 = (u8_2 ^ (unsigned int)((int)u8_2 >> 31) & 63) & 63;
                key8_2 = 536870912 | lowb8_3 << 6 | (unsigned int)(t0_3 + 3);
            }
            int f2_2 = 0;
            if ((t0_3 + 3 < fb || t0_3 + 3 >= lim - fe) && lim > t0_3 + 3) {
                f2_2 = 1;
            }
            if (f2_2 != 0) {
                key8_2 = 0;
            }
            if (key8_2 != 0) {
                unsigned int p6_3 = atomicAdd(&ccnt[c], 1);
                if (p6_3 < 16) {
                    cbuf[c * 16 + (int)p6_3] = key8_2;
                } else {
                    flagw[1] = 1;
                }
            }
            float sc_19 = __uint_as_float(cb[4]);
            sc_19 = fmaxf(sc_19, -1.7014118e+38f);
            sc_19 = fminf(sc_19, 1.7014118e+38f);
            unsigned int u8_3 = __as_u32(sc_19);
            unsigned int key8_3 = 0;
            if ((u8_3 & 4294967232u) == qc) {
                unsigned int lowb8_4 = (u8_3 ^ (unsigned int)((int)u8_3 >> 31) & 63) & 63;
                key8_3 = 536870912 | lowb8_4 << 6 | (unsigned int)(t0_3 + 4);
            }
            int f2_3 = 0;
            if ((t0_3 + 4 < fb || t0_3 + 4 >= lim - fe) && lim > t0_3 + 4) {
                f2_3 = 1;
            }
            if (f2_3 != 0) {
                key8_3 = 0;
            }
            if (key8_3 != 0) {
                unsigned int p6_4 = atomicAdd(&ccnt[c], 1);
                if (p6_4 < 16) {
                    cbuf[c * 16 + (int)p6_4] = key8_3;
                } else {
                    flagw[1] = 1;
                }
            }
            float sc_20 = __uint_as_float(cb[5]);
            sc_20 = fmaxf(sc_20, -1.7014118e+38f);
            sc_20 = fminf(sc_20, 1.7014118e+38f);
            unsigned int u8_4 = __as_u32(sc_20);
            unsigned int key8_4 = 0;
            if ((u8_4 & 4294967232u) == qc) {
                unsigned int lowb8_5 = (u8_4 ^ (unsigned int)((int)u8_4 >> 31) & 63) & 63;
                key8_4 = 536870912 | lowb8_5 << 6 | (unsigned int)(t0_3 + 5);
            }
            int f2_4 = 0;
            if ((t0_3 + 5 < fb || t0_3 + 5 >= lim - fe) && lim > t0_3 + 5) {
                f2_4 = 1;
            }
            if (f2_4 != 0) {
                key8_4 = 0;
            }
            if (key8_4 != 0) {
                unsigned int p6_5 = atomicAdd(&ccnt[c], 1);
                if (p6_5 < 16) {
                    cbuf[c * 16 + (int)p6_5] = key8_4;
                } else {
                    flagw[1] = 1;
                }
            }
            float sc_21 = __uint_as_float(cb[6]);
            sc_21 = fmaxf(sc_21, -1.7014118e+38f);
            sc_21 = fminf(sc_21, 1.7014118e+38f);
            unsigned int u8_5 = __as_u32(sc_21);
            unsigned int key8_5 = 0;
            if ((u8_5 & 4294967232u) == qc) {
                unsigned int lowb8_6 = (u8_5 ^ (unsigned int)((int)u8_5 >> 31) & 63) & 63;
                key8_5 = 536870912 | lowb8_6 << 6 | (unsigned int)(t0_3 + 6);
            }
            int f2_5 = 0;
            if ((t0_3 + 6 < fb || t0_3 + 6 >= lim - fe) && lim > t0_3 + 6) {
                f2_5 = 1;
            }
            if (f2_5 != 0) {
                key8_5 = 0;
            }
            if (key8_5 != 0) {
                unsigned int p6_6 = atomicAdd(&ccnt[c], 1);
                if (p6_6 < 16) {
                    cbuf[c * 16 + (int)p6_6] = key8_5;
                } else {
                    flagw[1] = 1;
                }
            }
            float sc_22 = __uint_as_float(cb[7]);
            sc_22 = fmaxf(sc_22, -1.7014118e+38f);
            sc_22 = fminf(sc_22, 1.7014118e+38f);
            unsigned int u8_6 = __as_u32(sc_22);
            unsigned int key8_6 = 0;
            if ((u8_6 & 4294967232u) == qc) {
                unsigned int lowb8_7 = (u8_6 ^ (unsigned int)((int)u8_6 >> 31) & 63) & 63;
                key8_6 = 536870912 | lowb8_7 << 6 | (unsigned int)(t0_3 + 7);
            }
            int f2_6 = 0;
            if ((t0_3 + 7 < fb || t0_3 + 7 >= lim - fe) && lim > t0_3 + 7) {
                f2_6 = 1;
            }
            if (f2_6 != 0) {
                key8_6 = 0;
            }
            if (key8_6 != 0) {
                unsigned int p6_7 = atomicAdd(&ccnt[c], 1);
                if (p6_7 < 16) {
                    cbuf[c * 16 + (int)p6_7] = key8_6;
                } else {
                    flagw[1] = 1;
                }
            }
            float sc_23 = __uint_as_float(cb[8]);
            sc_23 = fmaxf(sc_23, -1.7014118e+38f);
            sc_23 = fminf(sc_23, 1.7014118e+38f);
            unsigned int u8_7 = __as_u32(sc_23);
            unsigned int key8_7 = 0;
            if ((u8_7 & 4294967232u) == qc) {
                unsigned int lowb8_8 = (u8_7 ^ (unsigned int)((int)u8_7 >> 31) & 63) & 63;
                key8_7 = 536870912 | lowb8_8 << 6 | (unsigned int)(t0_3 + 8);
            }
            int f2_7 = 0;
            if ((t0_3 + 8 < fb || t0_3 + 8 >= lim - fe) && lim > t0_3 + 8) {
                f2_7 = 1;
            }
            if (f2_7 != 0) {
                key8_7 = 0;
            }
            if (key8_7 != 0) {
                unsigned int p6_8 = atomicAdd(&ccnt[c], 1);
                if (p6_8 < 16) {
                    cbuf[c * 16 + (int)p6_8] = key8_7;
                } else {
                    flagw[1] = 1;
                }
            }
            float sc_24 = __uint_as_float(cb[9]);
            sc_24 = fmaxf(sc_24, -1.7014118e+38f);
            sc_24 = fminf(sc_24, 1.7014118e+38f);
            unsigned int u8_8 = __as_u32(sc_24);
            unsigned int key8_8 = 0;
            if ((u8_8 & 4294967232u) == qc) {
                unsigned int lowb8_9 = (u8_8 ^ (unsigned int)((int)u8_8 >> 31) & 63) & 63;
                key8_8 = 536870912 | lowb8_9 << 6 | (unsigned int)(t0_3 + 9);
            }
            int f2_8 = 0;
            if ((t0_3 + 9 < fb || t0_3 + 9 >= lim - fe) && lim > t0_3 + 9) {
                f2_8 = 1;
            }
            if (f2_8 != 0) {
                key8_8 = 0;
            }
            if (key8_8 != 0) {
                unsigned int p6_9 = atomicAdd(&ccnt[c], 1);
                if (p6_9 < 16) {
                    cbuf[c * 16 + (int)p6_9] = key8_8;
                } else {
                    flagw[1] = 1;
                }
            }
            float sc_25 = __uint_as_float(cb[10]);
            sc_25 = fmaxf(sc_25, -1.7014118e+38f);
            sc_25 = fminf(sc_25, 1.7014118e+38f);
            unsigned int u8_9 = __as_u32(sc_25);
            unsigned int key8_9 = 0;
            if ((u8_9 & 4294967232u) == qc) {
                unsigned int lowb8_10 = (u8_9 ^ (unsigned int)((int)u8_9 >> 31) & 63) & 63;
                key8_9 = 536870912 | lowb8_10 << 6 | (unsigned int)(t0_3 + 10);
            }
            int f2_9 = 0;
            if ((t0_3 + 10 < fb || t0_3 + 10 >= lim - fe) && lim > t0_3 + 10) {
                f2_9 = 1;
            }
            if (f2_9 != 0) {
                key8_9 = 0;
            }
            if (key8_9 != 0) {
                unsigned int p6_10 = atomicAdd(&ccnt[c], 1);
                if (p6_10 < 16) {
                    cbuf[c * 16 + (int)p6_10] = key8_9;
                } else {
                    flagw[1] = 1;
                }
            }
            float sc_26 = __uint_as_float(cb[11]);
            sc_26 = fmaxf(sc_26, -1.7014118e+38f);
            sc_26 = fminf(sc_26, 1.7014118e+38f);
            unsigned int u8_10 = __as_u32(sc_26);
            unsigned int key8_10 = 0;
            if ((u8_10 & 4294967232u) == qc) {
                unsigned int lowb8_11 = (u8_10 ^ (unsigned int)((int)u8_10 >> 31) & 63) & 63;
                key8_10 = 536870912 | lowb8_11 << 6 | (unsigned int)(t0_3 + 11);
            }
            int f2_10 = 0;
            if ((t0_3 + 11 < fb || t0_3 + 11 >= lim - fe) && lim > t0_3 + 11) {
                f2_10 = 1;
            }
            if (f2_10 != 0) {
                key8_10 = 0;
            }
            if (key8_10 != 0) {
                unsigned int p6_11 = atomicAdd(&ccnt[c], 1);
                if (p6_11 < 16) {
                    cbuf[c * 16 + (int)p6_11] = key8_10;
                } else {
                    flagw[1] = 1;
                }
            }
            float sc_27 = __uint_as_float(cb[12]);
            sc_27 = fmaxf(sc_27, -1.7014118e+38f);
            sc_27 = fminf(sc_27, 1.7014118e+38f);
            unsigned int u8_11 = __as_u32(sc_27);
            unsigned int key8_11 = 0;
            if ((u8_11 & 4294967232u) == qc) {
                unsigned int lowb8_12 = (u8_11 ^ (unsigned int)((int)u8_11 >> 31) & 63) & 63;
                key8_11 = 536870912 | lowb8_12 << 6 | (unsigned int)(t0_3 + 12);
            }
            int f2_11 = 0;
            if ((t0_3 + 12 < fb || t0_3 + 12 >= lim - fe) && lim > t0_3 + 12) {
                f2_11 = 1;
            }
            if (f2_11 != 0) {
                key8_11 = 0;
            }
            if (key8_11 != 0) {
                unsigned int p6_12 = atomicAdd(&ccnt[c], 1);
                if (p6_12 < 16) {
                    cbuf[c * 16 + (int)p6_12] = key8_11;
                } else {
                    flagw[1] = 1;
                }
            }
            float sc_28 = __uint_as_float(cb[13]);
            sc_28 = fmaxf(sc_28, -1.7014118e+38f);
            sc_28 = fminf(sc_28, 1.7014118e+38f);
            unsigned int u8_12 = __as_u32(sc_28);
            unsigned int key8_12 = 0;
            if ((u8_12 & 4294967232u) == qc) {
                unsigned int lowb8_13 = (u8_12 ^ (unsigned int)((int)u8_12 >> 31) & 63) & 63;
                key8_12 = 536870912 | lowb8_13 << 6 | (unsigned int)(t0_3 + 13);
            }
            int f2_12 = 0;
            if ((t0_3 + 13 < fb || t0_3 + 13 >= lim - fe) && lim > t0_3 + 13) {
                f2_12 = 1;
            }
            if (f2_12 != 0) {
                key8_12 = 0;
            }
            if (key8_12 != 0) {
                unsigned int p6_13 = atomicAdd(&ccnt[c], 1);
                if (p6_13 < 16) {
                    cbuf[c * 16 + (int)p6_13] = key8_12;
                } else {
                    flagw[1] = 1;
                }
            }
            float sc_29 = __uint_as_float(cb[14]);
            sc_29 = fmaxf(sc_29, -1.7014118e+38f);
            sc_29 = fminf(sc_29, 1.7014118e+38f);
            unsigned int u8_13 = __as_u32(sc_29);
            unsigned int key8_13 = 0;
            if ((u8_13 & 4294967232u) == qc) {
                unsigned int lowb8_14 = (u8_13 ^ (unsigned int)((int)u8_13 >> 31) & 63) & 63;
                key8_13 = 536870912 | lowb8_14 << 6 | (unsigned int)(t0_3 + 14);
            }
            int f2_13 = 0;
            if ((t0_3 + 14 < fb || t0_3 + 14 >= lim - fe) && lim > t0_3 + 14) {
                f2_13 = 1;
            }
            if (f2_13 != 0) {
                key8_13 = 0;
            }
            if (key8_13 != 0) {
                unsigned int p6_14 = atomicAdd(&ccnt[c], 1);
                if (p6_14 < 16) {
                    cbuf[c * 16 + (int)p6_14] = key8_13;
                } else {
                    flagw[1] = 1;
                }
            }
            float sc_30 = __uint_as_float(cb[15]);
            sc_30 = fmaxf(sc_30, -1.7014118e+38f);
            sc_30 = fminf(sc_30, 1.7014118e+38f);
            unsigned int u8_14 = __as_u32(sc_30);
            unsigned int key8_14 = 0;
            if ((u8_14 & 4294967232u) == qc) {
                unsigned int lowb8_15 = (u8_14 ^ (unsigned int)((int)u8_14 >> 31) & 63) & 63;
                key8_14 = 536870912 | lowb8_15 << 6 | (unsigned int)(t0_3 + 15);
            }
            int f2_14 = 0;
            if ((t0_3 + 15 < fb || t0_3 + 15 >= lim - fe) && lim > t0_3 + 15) {
                f2_14 = 1;
            }
            if (f2_14 != 0) {
                key8_14 = 0;
            }
            if (key8_14 != 0) {
                unsigned int p6_15 = atomicAdd(&ccnt[c], 1);
                if (p6_15 < 16) {
                    cbuf[c * 16 + (int)p6_15] = key8_14;
                } else {
                    flagw[1] = 1;
                }
            }
        }
        asm volatile("barrier.sync 1, 128;" ::: "memory");
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
            if (ovfx != 0) {
                float kb2o[16];
                int t0_4 = w * 16;
                float qf = __uint_as_float(qc);
                float sc_31 = __uint_as_float(cb[0]);
                sc_31 = fmaxf(sc_31, -1.7014118e+38f);
                sc_31 = fminf(sc_31, 1.7014118e+38f);
                unsigned int u_1_0 = __as_u32(sc_31);
                unsigned int cls_1_0 = u_1_0 & 4294967232u;
                int f_16 = 0;
                if ((t0_4 < fb || t0_4 >= lim - fe) && lim > t0_4) {
                    f_16 = 1;
                }
                if (f_16 != 0) {
                    cls_1_0 = 2139094976;
                }
                unsigned int key = 0;
                if ((qf < __uint_as_float(cls_1_0) || __uint_as_float(cls_1_0) == qf && cls_1_0 < qc) && cls_1_0 < 4278190080u) {
                    key = 1073741824 | (unsigned int)t0_4;
                }
                if (cls_1_0 == qc) {
                    unsigned int lowb = (u_1_0 ^ (unsigned int)((int)u_1_0 >> 31) & 63) & 63;
                    key = 536870912 | lowb << 6 | (unsigned int)t0_4;
                }
                kb2o[0] = __uint_as_float(key);
                float sc_32 = __uint_as_float(cb[1]);
                sc_32 = fmaxf(sc_32, -1.7014118e+38f);
                sc_32 = fminf(sc_32, 1.7014118e+38f);
                unsigned int u_1_1 = __as_u32(sc_32);
                unsigned int cls_1_1 = u_1_1 & 4294967232u;
                int f_17 = 0;
                if ((t0_4 + 1 < fb || t0_4 + 1 >= lim - fe) && lim > t0_4 + 1) {
                    f_17 = 1;
                }
                if (f_17 != 0) {
                    cls_1_1 = 2139094976;
                }
                unsigned int key_0 = 0;
                if ((qf < __uint_as_float(cls_1_1) || __uint_as_float(cls_1_1) == qf && cls_1_1 < qc) && cls_1_1 < 4278190080u) {
                    key_0 = 1073741824 | (unsigned int)(t0_4 + 1);
                }
                if (cls_1_1 == qc) {
                    unsigned int lowb_1 = (u_1_1 ^ (unsigned int)((int)u_1_1 >> 31) & 63) & 63;
                    key_0 = 536870912 | lowb_1 << 6 | (unsigned int)(t0_4 + 1);
                }
                kb2o[1] = __uint_as_float(key_0);
                float sc_33 = __uint_as_float(cb[2]);
                sc_33 = fmaxf(sc_33, -1.7014118e+38f);
                sc_33 = fminf(sc_33, 1.7014118e+38f);
                unsigned int u_1_2 = __as_u32(sc_33);
                unsigned int cls_1_2 = u_1_2 & 4294967232u;
                int f_18 = 0;
                if ((t0_4 + 2 < fb || t0_4 + 2 >= lim - fe) && lim > t0_4 + 2) {
                    f_18 = 1;
                }
                if (f_18 != 0) {
                    cls_1_2 = 2139094976;
                }
                unsigned int key_1 = 0;
                if ((qf < __uint_as_float(cls_1_2) || __uint_as_float(cls_1_2) == qf && cls_1_2 < qc) && cls_1_2 < 4278190080u) {
                    key_1 = 1073741824 | (unsigned int)(t0_4 + 2);
                }
                if (cls_1_2 == qc) {
                    unsigned int lowb_2 = (u_1_2 ^ (unsigned int)((int)u_1_2 >> 31) & 63) & 63;
                    key_1 = 536870912 | lowb_2 << 6 | (unsigned int)(t0_4 + 2);
                }
                kb2o[2] = __uint_as_float(key_1);
                float sc_34 = __uint_as_float(cb[3]);
                sc_34 = fmaxf(sc_34, -1.7014118e+38f);
                sc_34 = fminf(sc_34, 1.7014118e+38f);
                unsigned int u_1_3 = __as_u32(sc_34);
                unsigned int cls_1_3 = u_1_3 & 4294967232u;
                int f_19 = 0;
                if ((t0_4 + 3 < fb || t0_4 + 3 >= lim - fe) && lim > t0_4 + 3) {
                    f_19 = 1;
                }
                if (f_19 != 0) {
                    cls_1_3 = 2139094976;
                }
                unsigned int key_2 = 0;
                if ((qf < __uint_as_float(cls_1_3) || __uint_as_float(cls_1_3) == qf && cls_1_3 < qc) && cls_1_3 < 4278190080u) {
                    key_2 = 1073741824 | (unsigned int)(t0_4 + 3);
                }
                if (cls_1_3 == qc) {
                    unsigned int lowb_3 = (u_1_3 ^ (unsigned int)((int)u_1_3 >> 31) & 63) & 63;
                    key_2 = 536870912 | lowb_3 << 6 | (unsigned int)(t0_4 + 3);
                }
                kb2o[3] = __uint_as_float(key_2);
                float sc_35 = __uint_as_float(cb[4]);
                sc_35 = fmaxf(sc_35, -1.7014118e+38f);
                sc_35 = fminf(sc_35, 1.7014118e+38f);
                unsigned int u_1_4 = __as_u32(sc_35);
                unsigned int cls_1_4 = u_1_4 & 4294967232u;
                int f_20 = 0;
                if ((t0_4 + 4 < fb || t0_4 + 4 >= lim - fe) && lim > t0_4 + 4) {
                    f_20 = 1;
                }
                if (f_20 != 0) {
                    cls_1_4 = 2139094976;
                }
                unsigned int key_3 = 0;
                if ((qf < __uint_as_float(cls_1_4) || __uint_as_float(cls_1_4) == qf && cls_1_4 < qc) && cls_1_4 < 4278190080u) {
                    key_3 = 1073741824 | (unsigned int)(t0_4 + 4);
                }
                if (cls_1_4 == qc) {
                    unsigned int lowb_4 = (u_1_4 ^ (unsigned int)((int)u_1_4 >> 31) & 63) & 63;
                    key_3 = 536870912 | lowb_4 << 6 | (unsigned int)(t0_4 + 4);
                }
                kb2o[4] = __uint_as_float(key_3);
                float sc_36 = __uint_as_float(cb[5]);
                sc_36 = fmaxf(sc_36, -1.7014118e+38f);
                sc_36 = fminf(sc_36, 1.7014118e+38f);
                unsigned int u_1_5 = __as_u32(sc_36);
                unsigned int cls_1_5 = u_1_5 & 4294967232u;
                int f_21 = 0;
                if ((t0_4 + 5 < fb || t0_4 + 5 >= lim - fe) && lim > t0_4 + 5) {
                    f_21 = 1;
                }
                if (f_21 != 0) {
                    cls_1_5 = 2139094976;
                }
                unsigned int key_4 = 0;
                if ((qf < __uint_as_float(cls_1_5) || __uint_as_float(cls_1_5) == qf && cls_1_5 < qc) && cls_1_5 < 4278190080u) {
                    key_4 = 1073741824 | (unsigned int)(t0_4 + 5);
                }
                if (cls_1_5 == qc) {
                    unsigned int lowb_5 = (u_1_5 ^ (unsigned int)((int)u_1_5 >> 31) & 63) & 63;
                    key_4 = 536870912 | lowb_5 << 6 | (unsigned int)(t0_4 + 5);
                }
                kb2o[5] = __uint_as_float(key_4);
                float sc_37 = __uint_as_float(cb[6]);
                sc_37 = fmaxf(sc_37, -1.7014118e+38f);
                sc_37 = fminf(sc_37, 1.7014118e+38f);
                unsigned int u_1_6 = __as_u32(sc_37);
                unsigned int cls_1_6 = u_1_6 & 4294967232u;
                int f_22 = 0;
                if ((t0_4 + 6 < fb || t0_4 + 6 >= lim - fe) && lim > t0_4 + 6) {
                    f_22 = 1;
                }
                if (f_22 != 0) {
                    cls_1_6 = 2139094976;
                }
                unsigned int key_5 = 0;
                if ((qf < __uint_as_float(cls_1_6) || __uint_as_float(cls_1_6) == qf && cls_1_6 < qc) && cls_1_6 < 4278190080u) {
                    key_5 = 1073741824 | (unsigned int)(t0_4 + 6);
                }
                if (cls_1_6 == qc) {
                    unsigned int lowb_6 = (u_1_6 ^ (unsigned int)((int)u_1_6 >> 31) & 63) & 63;
                    key_5 = 536870912 | lowb_6 << 6 | (unsigned int)(t0_4 + 6);
                }
                kb2o[6] = __uint_as_float(key_5);
                float sc_38 = __uint_as_float(cb[7]);
                sc_38 = fmaxf(sc_38, -1.7014118e+38f);
                sc_38 = fminf(sc_38, 1.7014118e+38f);
                unsigned int u_1_7 = __as_u32(sc_38);
                unsigned int cls_1_7 = u_1_7 & 4294967232u;
                int f_23 = 0;
                if ((t0_4 + 7 < fb || t0_4 + 7 >= lim - fe) && lim > t0_4 + 7) {
                    f_23 = 1;
                }
                if (f_23 != 0) {
                    cls_1_7 = 2139094976;
                }
                unsigned int key_6 = 0;
                if ((qf < __uint_as_float(cls_1_7) || __uint_as_float(cls_1_7) == qf && cls_1_7 < qc) && cls_1_7 < 4278190080u) {
                    key_6 = 1073741824 | (unsigned int)(t0_4 + 7);
                }
                if (cls_1_7 == qc) {
                    unsigned int lowb_7 = (u_1_7 ^ (unsigned int)((int)u_1_7 >> 31) & 63) & 63;
                    key_6 = 536870912 | lowb_7 << 6 | (unsigned int)(t0_4 + 7);
                }
                kb2o[7] = __uint_as_float(key_6);
                float sc_39 = __uint_as_float(cb[8]);
                sc_39 = fmaxf(sc_39, -1.7014118e+38f);
                sc_39 = fminf(sc_39, 1.7014118e+38f);
                unsigned int u_1_8 = __as_u32(sc_39);
                unsigned int cls_1_8 = u_1_8 & 4294967232u;
                int f_24 = 0;
                if ((t0_4 + 8 < fb || t0_4 + 8 >= lim - fe) && lim > t0_4 + 8) {
                    f_24 = 1;
                }
                if (f_24 != 0) {
                    cls_1_8 = 2139094976;
                }
                unsigned int key_7 = 0;
                if ((qf < __uint_as_float(cls_1_8) || __uint_as_float(cls_1_8) == qf && cls_1_8 < qc) && cls_1_8 < 4278190080u) {
                    key_7 = 1073741824 | (unsigned int)(t0_4 + 8);
                }
                if (cls_1_8 == qc) {
                    unsigned int lowb_8 = (u_1_8 ^ (unsigned int)((int)u_1_8 >> 31) & 63) & 63;
                    key_7 = 536870912 | lowb_8 << 6 | (unsigned int)(t0_4 + 8);
                }
                kb2o[8] = __uint_as_float(key_7);
                float sc_40 = __uint_as_float(cb[9]);
                sc_40 = fmaxf(sc_40, -1.7014118e+38f);
                sc_40 = fminf(sc_40, 1.7014118e+38f);
                unsigned int u_1_9 = __as_u32(sc_40);
                unsigned int cls_1_9 = u_1_9 & 4294967232u;
                int f_25 = 0;
                if ((t0_4 + 9 < fb || t0_4 + 9 >= lim - fe) && lim > t0_4 + 9) {
                    f_25 = 1;
                }
                if (f_25 != 0) {
                    cls_1_9 = 2139094976;
                }
                unsigned int key_8 = 0;
                if ((qf < __uint_as_float(cls_1_9) || __uint_as_float(cls_1_9) == qf && cls_1_9 < qc) && cls_1_9 < 4278190080u) {
                    key_8 = 1073741824 | (unsigned int)(t0_4 + 9);
                }
                if (cls_1_9 == qc) {
                    unsigned int lowb_9 = (u_1_9 ^ (unsigned int)((int)u_1_9 >> 31) & 63) & 63;
                    key_8 = 536870912 | lowb_9 << 6 | (unsigned int)(t0_4 + 9);
                }
                kb2o[9] = __uint_as_float(key_8);
                float sc_41 = __uint_as_float(cb[10]);
                sc_41 = fmaxf(sc_41, -1.7014118e+38f);
                sc_41 = fminf(sc_41, 1.7014118e+38f);
                unsigned int u_1_10 = __as_u32(sc_41);
                unsigned int cls_1_10 = u_1_10 & 4294967232u;
                int f_26 = 0;
                if ((t0_4 + 10 < fb || t0_4 + 10 >= lim - fe) && lim > t0_4 + 10) {
                    f_26 = 1;
                }
                if (f_26 != 0) {
                    cls_1_10 = 2139094976;
                }
                unsigned int key_9 = 0;
                if ((qf < __uint_as_float(cls_1_10) || __uint_as_float(cls_1_10) == qf && cls_1_10 < qc) && cls_1_10 < 4278190080u) {
                    key_9 = 1073741824 | (unsigned int)(t0_4 + 10);
                }
                if (cls_1_10 == qc) {
                    unsigned int lowb_10 = (u_1_10 ^ (unsigned int)((int)u_1_10 >> 31) & 63) & 63;
                    key_9 = 536870912 | lowb_10 << 6 | (unsigned int)(t0_4 + 10);
                }
                kb2o[10] = __uint_as_float(key_9);
                float sc_42 = __uint_as_float(cb[11]);
                sc_42 = fmaxf(sc_42, -1.7014118e+38f);
                sc_42 = fminf(sc_42, 1.7014118e+38f);
                unsigned int u_1_11 = __as_u32(sc_42);
                unsigned int cls_1_11 = u_1_11 & 4294967232u;
                int f_27 = 0;
                if ((t0_4 + 11 < fb || t0_4 + 11 >= lim - fe) && lim > t0_4 + 11) {
                    f_27 = 1;
                }
                if (f_27 != 0) {
                    cls_1_11 = 2139094976;
                }
                unsigned int key_10 = 0;
                if ((qf < __uint_as_float(cls_1_11) || __uint_as_float(cls_1_11) == qf && cls_1_11 < qc) && cls_1_11 < 4278190080u) {
                    key_10 = 1073741824 | (unsigned int)(t0_4 + 11);
                }
                if (cls_1_11 == qc) {
                    unsigned int lowb_11 = (u_1_11 ^ (unsigned int)((int)u_1_11 >> 31) & 63) & 63;
                    key_10 = 536870912 | lowb_11 << 6 | (unsigned int)(t0_4 + 11);
                }
                kb2o[11] = __uint_as_float(key_10);
                float sc_43 = __uint_as_float(cb[12]);
                sc_43 = fmaxf(sc_43, -1.7014118e+38f);
                sc_43 = fminf(sc_43, 1.7014118e+38f);
                unsigned int u_1_12 = __as_u32(sc_43);
                unsigned int cls_1_12 = u_1_12 & 4294967232u;
                int f_28 = 0;
                if ((t0_4 + 12 < fb || t0_4 + 12 >= lim - fe) && lim > t0_4 + 12) {
                    f_28 = 1;
                }
                if (f_28 != 0) {
                    cls_1_12 = 2139094976;
                }
                unsigned int key_11 = 0;
                if ((qf < __uint_as_float(cls_1_12) || __uint_as_float(cls_1_12) == qf && cls_1_12 < qc) && cls_1_12 < 4278190080u) {
                    key_11 = 1073741824 | (unsigned int)(t0_4 + 12);
                }
                if (cls_1_12 == qc) {
                    unsigned int lowb_12 = (u_1_12 ^ (unsigned int)((int)u_1_12 >> 31) & 63) & 63;
                    key_11 = 536870912 | lowb_12 << 6 | (unsigned int)(t0_4 + 12);
                }
                kb2o[12] = __uint_as_float(key_11);
                float sc_44 = __uint_as_float(cb[13]);
                sc_44 = fmaxf(sc_44, -1.7014118e+38f);
                sc_44 = fminf(sc_44, 1.7014118e+38f);
                unsigned int u_1_13 = __as_u32(sc_44);
                unsigned int cls_1_13 = u_1_13 & 4294967232u;
                int f_29 = 0;
                if ((t0_4 + 13 < fb || t0_4 + 13 >= lim - fe) && lim > t0_4 + 13) {
                    f_29 = 1;
                }
                if (f_29 != 0) {
                    cls_1_13 = 2139094976;
                }
                unsigned int key_12 = 0;
                if ((qf < __uint_as_float(cls_1_13) || __uint_as_float(cls_1_13) == qf && cls_1_13 < qc) && cls_1_13 < 4278190080u) {
                    key_12 = 1073741824 | (unsigned int)(t0_4 + 13);
                }
                if (cls_1_13 == qc) {
                    unsigned int lowb_13 = (u_1_13 ^ (unsigned int)((int)u_1_13 >> 31) & 63) & 63;
                    key_12 = 536870912 | lowb_13 << 6 | (unsigned int)(t0_4 + 13);
                }
                kb2o[13] = __uint_as_float(key_12);
                float sc_45 = __uint_as_float(cb[14]);
                sc_45 = fmaxf(sc_45, -1.7014118e+38f);
                sc_45 = fminf(sc_45, 1.7014118e+38f);
                unsigned int u_1_14 = __as_u32(sc_45);
                unsigned int cls_1_14 = u_1_14 & 4294967232u;
                int f_30 = 0;
                if ((t0_4 + 14 < fb || t0_4 + 14 >= lim - fe) && lim > t0_4 + 14) {
                    f_30 = 1;
                }
                if (f_30 != 0) {
                    cls_1_14 = 2139094976;
                }
                unsigned int key_13 = 0;
                if ((qf < __uint_as_float(cls_1_14) || __uint_as_float(cls_1_14) == qf && cls_1_14 < qc) && cls_1_14 < 4278190080u) {
                    key_13 = 1073741824 | (unsigned int)(t0_4 + 14);
                }
                if (cls_1_14 == qc) {
                    unsigned int lowb_14 = (u_1_14 ^ (unsigned int)((int)u_1_14 >> 31) & 63) & 63;
                    key_13 = 536870912 | lowb_14 << 6 | (unsigned int)(t0_4 + 14);
                }
                kb2o[14] = __uint_as_float(key_13);
                float sc_46 = __uint_as_float(cb[15]);
                sc_46 = fmaxf(sc_46, -1.7014118e+38f);
                sc_46 = fminf(sc_46, 1.7014118e+38f);
                unsigned int u_1_15 = __as_u32(sc_46);
                unsigned int cls_1_15 = u_1_15 & 4294967232u;
                int f_31 = 0;
                if ((t0_4 + 15 < fb || t0_4 + 15 >= lim - fe) && lim > t0_4 + 15) {
                    f_31 = 1;
                }
                if (f_31 != 0) {
                    cls_1_15 = 2139094976;
                }
                unsigned int key_14 = 0;
                if ((qf < __uint_as_float(cls_1_15) || __uint_as_float(cls_1_15) == qf && cls_1_15 < qc) && cls_1_15 < 4278190080u) {
                    key_14 = 1073741824 | (unsigned int)(t0_4 + 15);
                }
                if (cls_1_15 == qc) {
                    unsigned int lowb_15 = (u_1_15 ^ (unsigned int)((int)u_1_15 >> 31) & 63) & 63;
                    key_14 = 536870912 | lowb_15 << 6 | (unsigned int)(t0_4 + 15);
                }
                kb2o[15] = __uint_as_float(key_14);
                float hi_1_60 = fmaxf(kb2o[0], kb2o[13]);
                float lo_1_60 = fminf(kb2o[0], kb2o[13]);
                kb2o[0] = hi_1_60;
                kb2o[13] = lo_1_60;
                float hi_1_61 = fmaxf(kb2o[1], kb2o[12]);
                float lo_1_61 = fminf(kb2o[1], kb2o[12]);
                kb2o[1] = hi_1_61;
                kb2o[12] = lo_1_61;
                float hi_1_62 = fmaxf(kb2o[2], kb2o[15]);
                float lo_1_62 = fminf(kb2o[2], kb2o[15]);
                kb2o[2] = hi_1_62;
                kb2o[15] = lo_1_62;
                float hi_1_63 = fmaxf(kb2o[3], kb2o[14]);
                float lo_1_63 = fminf(kb2o[3], kb2o[14]);
                kb2o[3] = hi_1_63;
                kb2o[14] = lo_1_63;
                float hi_1_64 = fmaxf(kb2o[4], kb2o[8]);
                float lo_1_64 = fminf(kb2o[4], kb2o[8]);
                kb2o[4] = hi_1_64;
                kb2o[8] = lo_1_64;
                float hi_1_65 = fmaxf(kb2o[5], kb2o[6]);
                float lo_1_65 = fminf(kb2o[5], kb2o[6]);
                kb2o[5] = hi_1_65;
                kb2o[6] = lo_1_65;
                float hi_1_66 = fmaxf(kb2o[7], kb2o[11]);
                float lo_1_66 = fminf(kb2o[7], kb2o[11]);
                kb2o[7] = hi_1_66;
                kb2o[11] = lo_1_66;
                float hi_1_67 = fmaxf(kb2o[9], kb2o[10]);
                float lo_1_67 = fminf(kb2o[9], kb2o[10]);
                kb2o[9] = hi_1_67;
                kb2o[10] = lo_1_67;
                float hi_1_68 = fmaxf(kb2o[0], kb2o[5]);
                float lo_1_68 = fminf(kb2o[0], kb2o[5]);
                kb2o[0] = hi_1_68;
                kb2o[5] = lo_1_68;
                float hi_1_69 = fmaxf(kb2o[1], kb2o[7]);
                float lo_1_69 = fminf(kb2o[1], kb2o[7]);
                kb2o[1] = hi_1_69;
                kb2o[7] = lo_1_69;
                float hi_1_70 = fmaxf(kb2o[2], kb2o[9]);
                float lo_1_70 = fminf(kb2o[2], kb2o[9]);
                kb2o[2] = hi_1_70;
                kb2o[9] = lo_1_70;
                float hi_1_71 = fmaxf(kb2o[3], kb2o[4]);
                float lo_1_71 = fminf(kb2o[3], kb2o[4]);
                kb2o[3] = hi_1_71;
                kb2o[4] = lo_1_71;
                float hi_1_72 = fmaxf(kb2o[6], kb2o[13]);
                float lo_1_72 = fminf(kb2o[6], kb2o[13]);
                kb2o[6] = hi_1_72;
                kb2o[13] = lo_1_72;
                float hi_1_73 = fmaxf(kb2o[8], kb2o[14]);
                float lo_1_73 = fminf(kb2o[8], kb2o[14]);
                kb2o[8] = hi_1_73;
                kb2o[14] = lo_1_73;
                float hi_1_74 = fmaxf(kb2o[10], kb2o[15]);
                float lo_1_74 = fminf(kb2o[10], kb2o[15]);
                kb2o[10] = hi_1_74;
                kb2o[15] = lo_1_74;
                float hi_1_75 = fmaxf(kb2o[11], kb2o[12]);
                float lo_1_75 = fminf(kb2o[11], kb2o[12]);
                kb2o[11] = hi_1_75;
                kb2o[12] = lo_1_75;
                float hi_1_76 = fmaxf(kb2o[0], kb2o[1]);
                float lo_1_76 = fminf(kb2o[0], kb2o[1]);
                kb2o[0] = hi_1_76;
                kb2o[1] = lo_1_76;
                float hi_1_77 = fmaxf(kb2o[2], kb2o[3]);
                float lo_1_77 = fminf(kb2o[2], kb2o[3]);
                kb2o[2] = hi_1_77;
                kb2o[3] = lo_1_77;
                float hi_1_78 = fmaxf(kb2o[4], kb2o[5]);
                float lo_1_78 = fminf(kb2o[4], kb2o[5]);
                kb2o[4] = hi_1_78;
                kb2o[5] = lo_1_78;
                float hi_1_79 = fmaxf(kb2o[6], kb2o[8]);
                float lo_1_79 = fminf(kb2o[6], kb2o[8]);
                kb2o[6] = hi_1_79;
                kb2o[8] = lo_1_79;
                float hi_1_80 = fmaxf(kb2o[7], kb2o[9]);
                float lo_1_80 = fminf(kb2o[7], kb2o[9]);
                kb2o[7] = hi_1_80;
                kb2o[9] = lo_1_80;
                float hi_1_81 = fmaxf(kb2o[10], kb2o[11]);
                float lo_1_81 = fminf(kb2o[10], kb2o[11]);
                kb2o[10] = hi_1_81;
                kb2o[11] = lo_1_81;
                float hi_1_82 = fmaxf(kb2o[12], kb2o[13]);
                float lo_1_82 = fminf(kb2o[12], kb2o[13]);
                kb2o[12] = hi_1_82;
                kb2o[13] = lo_1_82;
                float hi_1_83 = fmaxf(kb2o[14], kb2o[15]);
                float lo_1_83 = fminf(kb2o[14], kb2o[15]);
                kb2o[14] = hi_1_83;
                kb2o[15] = lo_1_83;
                float hi_1_84 = fmaxf(kb2o[0], kb2o[2]);
                float lo_1_84 = fminf(kb2o[0], kb2o[2]);
                kb2o[0] = hi_1_84;
                kb2o[2] = lo_1_84;
                float hi_1_85 = fmaxf(kb2o[1], kb2o[3]);
                float lo_1_85 = fminf(kb2o[1], kb2o[3]);
                kb2o[1] = hi_1_85;
                kb2o[3] = lo_1_85;
                float hi_1_86 = fmaxf(kb2o[4], kb2o[10]);
                float lo_1_86 = fminf(kb2o[4], kb2o[10]);
                kb2o[4] = hi_1_86;
                kb2o[10] = lo_1_86;
                float hi_1_87 = fmaxf(kb2o[5], kb2o[11]);
                float lo_1_87 = fminf(kb2o[5], kb2o[11]);
                kb2o[5] = hi_1_87;
                kb2o[11] = lo_1_87;
                float hi_1_88 = fmaxf(kb2o[6], kb2o[7]);
                float lo_1_88 = fminf(kb2o[6], kb2o[7]);
                kb2o[6] = hi_1_88;
                kb2o[7] = lo_1_88;
                float hi_1_89 = fmaxf(kb2o[8], kb2o[9]);
                float lo_1_89 = fminf(kb2o[8], kb2o[9]);
                kb2o[8] = hi_1_89;
                kb2o[9] = lo_1_89;
                float hi_1_90 = fmaxf(kb2o[12], kb2o[14]);
                float lo_1_90 = fminf(kb2o[12], kb2o[14]);
                kb2o[12] = hi_1_90;
                kb2o[14] = lo_1_90;
                float hi_1_91 = fmaxf(kb2o[13], kb2o[15]);
                float lo_1_91 = fminf(kb2o[13], kb2o[15]);
                kb2o[13] = hi_1_91;
                kb2o[15] = lo_1_91;
                float hi_1_92 = fmaxf(kb2o[1], kb2o[2]);
                float lo_1_92 = fminf(kb2o[1], kb2o[2]);
                kb2o[1] = hi_1_92;
                kb2o[2] = lo_1_92;
                float hi_1_93 = fmaxf(kb2o[3], kb2o[12]);
                float lo_1_93 = fminf(kb2o[3], kb2o[12]);
                kb2o[3] = hi_1_93;
                kb2o[12] = lo_1_93;
                float hi_1_94 = fmaxf(kb2o[4], kb2o[6]);
                float lo_1_94 = fminf(kb2o[4], kb2o[6]);
                kb2o[4] = hi_1_94;
                kb2o[6] = lo_1_94;
                float hi_1_95 = fmaxf(kb2o[5], kb2o[7]);
                float lo_1_95 = fminf(kb2o[5], kb2o[7]);
                kb2o[5] = hi_1_95;
                kb2o[7] = lo_1_95;
                float hi_1_96 = fmaxf(kb2o[8], kb2o[10]);
                float lo_1_96 = fminf(kb2o[8], kb2o[10]);
                kb2o[8] = hi_1_96;
                kb2o[10] = lo_1_96;
                float hi_1_97 = fmaxf(kb2o[9], kb2o[11]);
                float lo_1_97 = fminf(kb2o[9], kb2o[11]);
                kb2o[9] = hi_1_97;
                kb2o[11] = lo_1_97;
                float hi_1_98 = fmaxf(kb2o[13], kb2o[14]);
                float lo_1_98 = fminf(kb2o[13], kb2o[14]);
                kb2o[13] = hi_1_98;
                kb2o[14] = lo_1_98;
                float hi_1_99 = fmaxf(kb2o[1], kb2o[4]);
                float lo_1_99 = fminf(kb2o[1], kb2o[4]);
                kb2o[1] = hi_1_99;
                kb2o[4] = lo_1_99;
                float hi_1_100 = fmaxf(kb2o[2], kb2o[6]);
                float lo_1_100 = fminf(kb2o[2], kb2o[6]);
                kb2o[2] = hi_1_100;
                kb2o[6] = lo_1_100;
                float hi_1_101 = fmaxf(kb2o[5], kb2o[8]);
                float lo_1_101 = fminf(kb2o[5], kb2o[8]);
                kb2o[5] = hi_1_101;
                kb2o[8] = lo_1_101;
                float hi_1_102 = fmaxf(kb2o[7], kb2o[10]);
                float lo_1_102 = fminf(kb2o[7], kb2o[10]);
                kb2o[7] = hi_1_102;
                kb2o[10] = lo_1_102;
                float hi_1_103 = fmaxf(kb2o[9], kb2o[13]);
                float lo_1_103 = fminf(kb2o[9], kb2o[13]);
                kb2o[9] = hi_1_103;
                kb2o[13] = lo_1_103;
                float hi_1_104 = fmaxf(kb2o[11], kb2o[14]);
                float lo_1_104 = fminf(kb2o[11], kb2o[14]);
                kb2o[11] = hi_1_104;
                kb2o[14] = lo_1_104;
                float hi_1_105 = fmaxf(kb2o[2], kb2o[4]);
                float lo_1_105 = fminf(kb2o[2], kb2o[4]);
                kb2o[2] = hi_1_105;
                kb2o[4] = lo_1_105;
                float hi_1_106 = fmaxf(kb2o[3], kb2o[6]);
                float lo_1_106 = fminf(kb2o[3], kb2o[6]);
                kb2o[3] = hi_1_106;
                kb2o[6] = lo_1_106;
                float hi_1_107 = fmaxf(kb2o[9], kb2o[12]);
                float lo_1_107 = fminf(kb2o[9], kb2o[12]);
                kb2o[9] = hi_1_107;
                kb2o[12] = lo_1_107;
                float hi_1_108 = fmaxf(kb2o[11], kb2o[13]);
                float lo_1_108 = fminf(kb2o[11], kb2o[13]);
                kb2o[11] = hi_1_108;
                kb2o[13] = lo_1_108;
                float hi_1_109 = fmaxf(kb2o[3], kb2o[5]);
                float lo_1_109 = fminf(kb2o[3], kb2o[5]);
                kb2o[3] = hi_1_109;
                kb2o[5] = lo_1_109;
                float hi_1_110 = fmaxf(kb2o[6], kb2o[8]);
                float lo_1_110 = fminf(kb2o[6], kb2o[8]);
                kb2o[6] = hi_1_110;
                kb2o[8] = lo_1_110;
                float hi_1_111 = fmaxf(kb2o[7], kb2o[9]);
                float lo_1_111 = fminf(kb2o[7], kb2o[9]);
                kb2o[7] = hi_1_111;
                kb2o[9] = lo_1_111;
                float hi_1_112 = fmaxf(kb2o[10], kb2o[12]);
                float lo_1_112 = fminf(kb2o[10], kb2o[12]);
                kb2o[10] = hi_1_112;
                kb2o[12] = lo_1_112;
                float hi_1_113 = fmaxf(kb2o[3], kb2o[4]);
                float lo_1_113 = fminf(kb2o[3], kb2o[4]);
                kb2o[3] = hi_1_113;
                kb2o[4] = lo_1_113;
                float hi_1_114 = fmaxf(kb2o[5], kb2o[6]);
                float lo_1_114 = fminf(kb2o[5], kb2o[6]);
                kb2o[5] = hi_1_114;
                kb2o[6] = lo_1_114;
                float hi_1_115 = fmaxf(kb2o[7], kb2o[8]);
                float lo_1_115 = fminf(kb2o[7], kb2o[8]);
                kb2o[7] = hi_1_115;
                kb2o[8] = lo_1_115;
                float hi_1_116 = fmaxf(kb2o[9], kb2o[10]);
                float lo_1_116 = fminf(kb2o[9], kb2o[10]);
                kb2o[9] = hi_1_116;
                kb2o[10] = lo_1_116;
                float hi_1_117 = fmaxf(kb2o[11], kb2o[12]);
                float lo_1_117 = fminf(kb2o[11], kb2o[12]);
                kb2o[11] = hi_1_117;
                kb2o[12] = lo_1_117;
                float hi_1_118 = fmaxf(kb2o[6], kb2o[7]);
                float lo_1_118 = fminf(kb2o[6], kb2o[7]);
                kb2o[6] = hi_1_118;
                kb2o[7] = lo_1_118;
                float hi_1_119 = fmaxf(kb2o[8], kb2o[9]);
                float lo_1_119 = fminf(kb2o[8], kb2o[9]);
                kb2o[8] = hi_1_119;
                kb2o[9] = lo_1_119;
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
                int bit2 = 1 << k_1;
                int low2 = whi & (bit2 << 1) - 1;
                if (low2 == bit2) {
                    int base_1 = whi * 544 + c;
                    tree[base_1] = a2[0];
                    tree[base_1 + 32] = a2[1];
                    tree[base_1 + 64] = a2[2];
                    tree[base_1 + 96] = a2[3];
                    tree[base_1 + 128] = a2[4];
                    tree[base_1 + 160] = a2[5];
                    tree[base_1 + 192] = a2[6];
                    tree[base_1 + 224] = a2[7];
                    tree[base_1 + 256] = a2[8];
                    tree[base_1 + 288] = a2[9];
                    tree[base_1 + 320] = a2[10];
                    tree[base_1 + 352] = a2[11];
                    tree[base_1 + 384] = a2[12];
                    tree[base_1 + 416] = a2[13];
                    tree[base_1 + 448] = a2[14];
                    tree[base_1 + 480] = a2[15];
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (low2 == 0) {
                    float sb2[16];
                    int rbase2 = (whi + bit2) * 544 + c;
                    sb2[0] = tree[rbase2];
                    sb2[1] = tree[rbase2 + 32];
                    sb2[2] = tree[rbase2 + 64];
                    sb2[3] = tree[rbase2 + 96];
                    sb2[4] = tree[rbase2 + 128];
                    sb2[5] = tree[rbase2 + 160];
                    sb2[6] = tree[rbase2 + 192];
                    sb2[7] = tree[rbase2 + 224];
                    sb2[8] = tree[rbase2 + 256];
                    sb2[9] = tree[rbase2 + 288];
                    sb2[10] = tree[rbase2 + 320];
                    sb2[11] = tree[rbase2 + 352];
                    sb2[12] = tree[rbase2 + 384];
                    sb2[13] = tree[rbase2 + 416];
                    sb2[14] = tree[rbase2 + 448];
                    sb2[15] = tree[rbase2 + 480];
                    float hi_48 = fmaxf(a2[0], sb2[15]);
                    a2[0] = hi_48;
                    float hi_49 = fmaxf(a2[1], sb2[14]);
                    a2[1] = hi_49;
                    float hi_50 = fmaxf(a2[2], sb2[13]);
                    a2[2] = hi_50;
                    float hi_51 = fmaxf(a2[3], sb2[12]);
                    a2[3] = hi_51;
                    float hi_52 = fmaxf(a2[4], sb2[11]);
                    a2[4] = hi_52;
                    float hi_53 = fmaxf(a2[5], sb2[10]);
                    a2[5] = hi_53;
                    float hi_54 = fmaxf(a2[6], sb2[9]);
                    a2[6] = hi_54;
                    float hi_55 = fmaxf(a2[7], sb2[8]);
                    a2[7] = hi_55;
                    float hi_56 = fmaxf(a2[8], sb2[7]);
                    a2[8] = hi_56;
                    float hi_57 = fmaxf(a2[9], sb2[6]);
                    a2[9] = hi_57;
                    float hi_58 = fmaxf(a2[10], sb2[5]);
                    a2[10] = hi_58;
                    float hi_59 = fmaxf(a2[11], sb2[4]);
                    a2[11] = hi_59;
                    float hi_60 = fmaxf(a2[12], sb2[3]);
                    a2[12] = hi_60;
                    float hi_61 = fmaxf(a2[13], sb2[2]);
                    a2[13] = hi_61;
                    float hi_62 = fmaxf(a2[14], sb2[1]);
                    a2[14] = hi_62;
                    float hi_63 = fmaxf(a2[15], sb2[0]);
                    a2[15] = hi_63;
                    float hi_64 = fmaxf(a2[0], a2[8]);
                    float lo_48 = fminf(a2[0], a2[8]);
                    a2[0] = hi_64;
                    a2[8] = lo_48;
                    float hi_65 = fmaxf(a2[1], a2[9]);
                    float lo_49 = fminf(a2[1], a2[9]);
                    a2[1] = hi_65;
                    a2[9] = lo_49;
                    float hi_66 = fmaxf(a2[2], a2[10]);
                    float lo_50 = fminf(a2[2], a2[10]);
                    a2[2] = hi_66;
                    a2[10] = lo_50;
                    float hi_67 = fmaxf(a2[3], a2[11]);
                    float lo_51 = fminf(a2[3], a2[11]);
                    a2[3] = hi_67;
                    a2[11] = lo_51;
                    float hi_68 = fmaxf(a2[4], a2[12]);
                    float lo_52 = fminf(a2[4], a2[12]);
                    a2[4] = hi_68;
                    a2[12] = lo_52;
                    float hi_69 = fmaxf(a2[5], a2[13]);
                    float lo_53 = fminf(a2[5], a2[13]);
                    a2[5] = hi_69;
                    a2[13] = lo_53;
                    float hi_70 = fmaxf(a2[6], a2[14]);
                    float lo_54 = fminf(a2[6], a2[14]);
                    a2[6] = hi_70;
                    a2[14] = lo_54;
                    float hi_71 = fmaxf(a2[7], a2[15]);
                    float lo_55 = fminf(a2[7], a2[15]);
                    a2[7] = hi_71;
                    a2[15] = lo_55;
                    float hi_72 = fmaxf(a2[0], a2[4]);
                    float lo_56 = fminf(a2[0], a2[4]);
                    a2[0] = hi_72;
                    a2[4] = lo_56;
                    float hi_73 = fmaxf(a2[1], a2[5]);
                    float lo_57 = fminf(a2[1], a2[5]);
                    a2[1] = hi_73;
                    a2[5] = lo_57;
                    float hi_74 = fmaxf(a2[2], a2[6]);
                    float lo_58 = fminf(a2[2], a2[6]);
                    a2[2] = hi_74;
                    a2[6] = lo_58;
                    float hi_75 = fmaxf(a2[3], a2[7]);
                    float lo_59 = fminf(a2[3], a2[7]);
                    a2[3] = hi_75;
                    a2[7] = lo_59;
                    float hi_76 = fmaxf(a2[8], a2[12]);
                    float lo_60 = fminf(a2[8], a2[12]);
                    a2[8] = hi_76;
                    a2[12] = lo_60;
                    float hi_77 = fmaxf(a2[9], a2[13]);
                    float lo_61 = fminf(a2[9], a2[13]);
                    a2[9] = hi_77;
                    a2[13] = lo_61;
                    float hi_78 = fmaxf(a2[10], a2[14]);
                    float lo_62 = fminf(a2[10], a2[14]);
                    a2[10] = hi_78;
                    a2[14] = lo_62;
                    float hi_79 = fmaxf(a2[11], a2[15]);
                    float lo_63 = fminf(a2[11], a2[15]);
                    a2[11] = hi_79;
                    a2[15] = lo_63;
                    float hi_80 = fmaxf(a2[0], a2[2]);
                    float lo_64 = fminf(a2[0], a2[2]);
                    a2[0] = hi_80;
                    a2[2] = lo_64;
                    float hi_81 = fmaxf(a2[1], a2[3]);
                    float lo_65 = fminf(a2[1], a2[3]);
                    a2[1] = hi_81;
                    a2[3] = lo_65;
                    float hi_82 = fmaxf(a2[4], a2[6]);
                    float lo_66 = fminf(a2[4], a2[6]);
                    a2[4] = hi_82;
                    a2[6] = lo_66;
                    float hi_83 = fmaxf(a2[5], a2[7]);
                    float lo_67 = fminf(a2[5], a2[7]);
                    a2[5] = hi_83;
                    a2[7] = lo_67;
                    float hi_84 = fmaxf(a2[8], a2[10]);
                    float lo_68 = fminf(a2[8], a2[10]);
                    a2[8] = hi_84;
                    a2[10] = lo_68;
                    float hi_85 = fmaxf(a2[9], a2[11]);
                    float lo_69 = fminf(a2[9], a2[11]);
                    a2[9] = hi_85;
                    a2[11] = lo_69;
                    float hi_86 = fmaxf(a2[12], a2[14]);
                    float lo_70 = fminf(a2[12], a2[14]);
                    a2[12] = hi_86;
                    a2[14] = lo_70;
                    float hi_87 = fmaxf(a2[13], a2[15]);
                    float lo_71 = fminf(a2[13], a2[15]);
                    a2[13] = hi_87;
                    a2[15] = lo_71;
                    float hi_88 = fmaxf(a2[0], a2[1]);
                    float lo_72 = fminf(a2[0], a2[1]);
                    a2[0] = hi_88;
                    a2[1] = lo_72;
                    float hi_89 = fmaxf(a2[2], a2[3]);
                    float lo_73 = fminf(a2[2], a2[3]);
                    a2[2] = hi_89;
                    a2[3] = lo_73;
                    float hi_90 = fmaxf(a2[4], a2[5]);
                    float lo_74 = fminf(a2[4], a2[5]);
                    a2[4] = hi_90;
                    a2[5] = lo_74;
                    float hi_91 = fmaxf(a2[6], a2[7]);
                    float lo_75 = fminf(a2[6], a2[7]);
                    a2[6] = hi_91;
                    a2[7] = lo_75;
                    float hi_92 = fmaxf(a2[8], a2[9]);
                    float lo_76 = fminf(a2[8], a2[9]);
                    a2[8] = hi_92;
                    a2[9] = lo_76;
                    float hi_93 = fmaxf(a2[10], a2[11]);
                    float lo_77 = fminf(a2[10], a2[11]);
                    a2[10] = hi_93;
                    a2[11] = lo_77;
                    float hi_94 = fmaxf(a2[12], a2[13]);
                    float lo_78 = fminf(a2[12], a2[13]);
                    a2[12] = hi_94;
                    a2[13] = lo_78;
                    float hi_95 = fmaxf(a2[14], a2[15]);
                    float lo_79 = fminf(a2[14], a2[15]);
                    a2[14] = hi_95;
                    a2[15] = lo_79;
                }
            }
        }
        if (w == 0 && flagged != 0 && cntx <= 16) {
            unsigned int k1s_15_0 = __as_u32(a[0]);
            a2[0] = __uint_as_float(1073741824 | k1s_15_0 & 63);
            if ((k1s_15_0 & 4294967232u) == qc) {
                a2[0] = 0.0f;
            }
            unsigned int k1s_15_1 = __as_u32(a[1]);
            a2[1] = __uint_as_float(1073741824 | k1s_15_1 & 63);
            if ((k1s_15_1 & 4294967232u) == qc) {
                a2[1] = 0.0f;
            }
            unsigned int k1s_15_2 = __as_u32(a[2]);
            a2[2] = __uint_as_float(1073741824 | k1s_15_2 & 63);
            if ((k1s_15_2 & 4294967232u) == qc) {
                a2[2] = 0.0f;
            }
            unsigned int k1s_15_3 = __as_u32(a[3]);
            a2[3] = __uint_as_float(1073741824 | k1s_15_3 & 63);
            if ((k1s_15_3 & 4294967232u) == qc) {
                a2[3] = 0.0f;
            }
            unsigned int k1s_15_4 = __as_u32(a[4]);
            a2[4] = __uint_as_float(1073741824 | k1s_15_4 & 63);
            if ((k1s_15_4 & 4294967232u) == qc) {
                a2[4] = 0.0f;
            }
            unsigned int k1s_15_5 = __as_u32(a[5]);
            a2[5] = __uint_as_float(1073741824 | k1s_15_5 & 63);
            if ((k1s_15_5 & 4294967232u) == qc) {
                a2[5] = 0.0f;
            }
            unsigned int k1s_15_6 = __as_u32(a[6]);
            a2[6] = __uint_as_float(1073741824 | k1s_15_6 & 63);
            if ((k1s_15_6 & 4294967232u) == qc) {
                a2[6] = 0.0f;
            }
            unsigned int k1s_15_7 = __as_u32(a[7]);
            a2[7] = __uint_as_float(1073741824 | k1s_15_7 & 63);
            if ((k1s_15_7 & 4294967232u) == qc) {
                a2[7] = 0.0f;
            }
            unsigned int k1s_15_8 = __as_u32(a[8]);
            a2[8] = __uint_as_float(1073741824 | k1s_15_8 & 63);
            if ((k1s_15_8 & 4294967232u) == qc) {
                a2[8] = 0.0f;
            }
            unsigned int k1s_15_9 = __as_u32(a[9]);
            a2[9] = __uint_as_float(1073741824 | k1s_15_9 & 63);
            if ((k1s_15_9 & 4294967232u) == qc) {
                a2[9] = 0.0f;
            }
            unsigned int k1s_15_10 = __as_u32(a[10]);
            a2[10] = __uint_as_float(1073741824 | k1s_15_10 & 63);
            if ((k1s_15_10 & 4294967232u) == qc) {
                a2[10] = 0.0f;
            }
            unsigned int k1s_15_11 = __as_u32(a[11]);
            a2[11] = __uint_as_float(1073741824 | k1s_15_11 & 63);
            if ((k1s_15_11 & 4294967232u) == qc) {
                a2[11] = 0.0f;
            }
            unsigned int k1s_15_12 = __as_u32(a[12]);
            a2[12] = __uint_as_float(1073741824 | k1s_15_12 & 63);
            if ((k1s_15_12 & 4294967232u) == qc) {
                a2[12] = 0.0f;
            }
            unsigned int k1s_15_13 = __as_u32(a[13]);
            a2[13] = __uint_as_float(1073741824 | k1s_15_13 & 63);
            if ((k1s_15_13 & 4294967232u) == qc) {
                a2[13] = 0.0f;
            }
            unsigned int k1s_15_14 = __as_u32(a[14]);
            a2[14] = __uint_as_float(1073741824 | k1s_15_14 & 63);
            if ((k1s_15_14 & 4294967232u) == qc) {
                a2[14] = 0.0f;
            }
            unsigned int k1s_15_15 = __as_u32(a[15]);
            a2[15] = __uint_as_float(1073741824 | k1s_15_15 & 63);
            if ((k1s_15_15 & 4294967232u) == qc) {
                a2[15] = 0.0f;
            }
            float xs2[16];
            xs2[0] = 0.0f;
            if (cntx > 0) {
                xs2[0] = __uint_as_float(cbuf[c * 16]);
            }
            xs2[1] = 0.0f;
            if (cntx > 1) {
                xs2[1] = __uint_as_float(cbuf[c * 16 + 1]);
            }
            xs2[2] = 0.0f;
            if (cntx > 2) {
                xs2[2] = __uint_as_float(cbuf[c * 16 + 2]);
            }
            xs2[3] = 0.0f;
            if (cntx > 3) {
                xs2[3] = __uint_as_float(cbuf[c * 16 + 3]);
            }
            xs2[4] = 0.0f;
            if (cntx > 4) {
                xs2[4] = __uint_as_float(cbuf[c * 16 + 4]);
            }
            xs2[5] = 0.0f;
            if (cntx > 5) {
                xs2[5] = __uint_as_float(cbuf[c * 16 + 5]);
            }
            xs2[6] = 0.0f;
            if (cntx > 6) {
                xs2[6] = __uint_as_float(cbuf[c * 16 + 6]);
            }
            xs2[7] = 0.0f;
            if (cntx > 7) {
                xs2[7] = __uint_as_float(cbuf[c * 16 + 7]);
            }
            xs2[8] = 0.0f;
            if (cntx > 8) {
                xs2[8] = __uint_as_float(cbuf[c * 16 + 8]);
            }
            xs2[9] = 0.0f;
            if (cntx > 9) {
                xs2[9] = __uint_as_float(cbuf[c * 16 + 9]);
            }
            xs2[10] = 0.0f;
            if (cntx > 10) {
                xs2[10] = __uint_as_float(cbuf[c * 16 + 10]);
            }
            xs2[11] = 0.0f;
            if (cntx > 11) {
                xs2[11] = __uint_as_float(cbuf[c * 16 + 11]);
            }
            xs2[12] = 0.0f;
            if (cntx > 12) {
                xs2[12] = __uint_as_float(cbuf[c * 16 + 12]);
            }
            xs2[13] = 0.0f;
            if (cntx > 13) {
                xs2[13] = __uint_as_float(cbuf[c * 16 + 13]);
            }
            xs2[14] = 0.0f;
            if (cntx > 14) {
                xs2[14] = __uint_as_float(cbuf[c * 16 + 14]);
            }
            xs2[15] = 0.0f;
            if (cntx > 15) {
                xs2[15] = __uint_as_float(cbuf[c * 16 + 15]);
            }
            float hi_1_120 = fmaxf(xs2[0], xs2[13]);
            float lo_1_120 = fminf(xs2[0], xs2[13]);
            xs2[0] = hi_1_120;
            xs2[13] = lo_1_120;
            float hi_1_121 = fmaxf(xs2[1], xs2[12]);
            float lo_1_121 = fminf(xs2[1], xs2[12]);
            xs2[1] = hi_1_121;
            xs2[12] = lo_1_121;
            float hi_1_122 = fmaxf(xs2[2], xs2[15]);
            float lo_1_122 = fminf(xs2[2], xs2[15]);
            xs2[2] = hi_1_122;
            xs2[15] = lo_1_122;
            float hi_1_123 = fmaxf(xs2[3], xs2[14]);
            float lo_1_123 = fminf(xs2[3], xs2[14]);
            xs2[3] = hi_1_123;
            xs2[14] = lo_1_123;
            float hi_1_124 = fmaxf(xs2[4], xs2[8]);
            float lo_1_124 = fminf(xs2[4], xs2[8]);
            xs2[4] = hi_1_124;
            xs2[8] = lo_1_124;
            float hi_1_125 = fmaxf(xs2[5], xs2[6]);
            float lo_1_125 = fminf(xs2[5], xs2[6]);
            xs2[5] = hi_1_125;
            xs2[6] = lo_1_125;
            float hi_1_126 = fmaxf(xs2[7], xs2[11]);
            float lo_1_126 = fminf(xs2[7], xs2[11]);
            xs2[7] = hi_1_126;
            xs2[11] = lo_1_126;
            float hi_1_127 = fmaxf(xs2[9], xs2[10]);
            float lo_1_127 = fminf(xs2[9], xs2[10]);
            xs2[9] = hi_1_127;
            xs2[10] = lo_1_127;
            float hi_1_128 = fmaxf(xs2[0], xs2[5]);
            float lo_1_128 = fminf(xs2[0], xs2[5]);
            xs2[0] = hi_1_128;
            xs2[5] = lo_1_128;
            float hi_1_129 = fmaxf(xs2[1], xs2[7]);
            float lo_1_129 = fminf(xs2[1], xs2[7]);
            xs2[1] = hi_1_129;
            xs2[7] = lo_1_129;
            float hi_1_130 = fmaxf(xs2[2], xs2[9]);
            float lo_1_130 = fminf(xs2[2], xs2[9]);
            xs2[2] = hi_1_130;
            xs2[9] = lo_1_130;
            float hi_1_131 = fmaxf(xs2[3], xs2[4]);
            float lo_1_131 = fminf(xs2[3], xs2[4]);
            xs2[3] = hi_1_131;
            xs2[4] = lo_1_131;
            float hi_1_132 = fmaxf(xs2[6], xs2[13]);
            float lo_1_132 = fminf(xs2[6], xs2[13]);
            xs2[6] = hi_1_132;
            xs2[13] = lo_1_132;
            float hi_1_133 = fmaxf(xs2[8], xs2[14]);
            float lo_1_133 = fminf(xs2[8], xs2[14]);
            xs2[8] = hi_1_133;
            xs2[14] = lo_1_133;
            float hi_1_134 = fmaxf(xs2[10], xs2[15]);
            float lo_1_134 = fminf(xs2[10], xs2[15]);
            xs2[10] = hi_1_134;
            xs2[15] = lo_1_134;
            float hi_1_135 = fmaxf(xs2[11], xs2[12]);
            float lo_1_135 = fminf(xs2[11], xs2[12]);
            xs2[11] = hi_1_135;
            xs2[12] = lo_1_135;
            float hi_1_136 = fmaxf(xs2[0], xs2[1]);
            float lo_1_136 = fminf(xs2[0], xs2[1]);
            xs2[0] = hi_1_136;
            xs2[1] = lo_1_136;
            float hi_1_137 = fmaxf(xs2[2], xs2[3]);
            float lo_1_137 = fminf(xs2[2], xs2[3]);
            xs2[2] = hi_1_137;
            xs2[3] = lo_1_137;
            float hi_1_138 = fmaxf(xs2[4], xs2[5]);
            float lo_1_138 = fminf(xs2[4], xs2[5]);
            xs2[4] = hi_1_138;
            xs2[5] = lo_1_138;
            float hi_1_139 = fmaxf(xs2[6], xs2[8]);
            float lo_1_139 = fminf(xs2[6], xs2[8]);
            xs2[6] = hi_1_139;
            xs2[8] = lo_1_139;
            float hi_1_140 = fmaxf(xs2[7], xs2[9]);
            float lo_1_140 = fminf(xs2[7], xs2[9]);
            xs2[7] = hi_1_140;
            xs2[9] = lo_1_140;
            float hi_1_141 = fmaxf(xs2[10], xs2[11]);
            float lo_1_141 = fminf(xs2[10], xs2[11]);
            xs2[10] = hi_1_141;
            xs2[11] = lo_1_141;
            float hi_1_142 = fmaxf(xs2[12], xs2[13]);
            float lo_1_142 = fminf(xs2[12], xs2[13]);
            xs2[12] = hi_1_142;
            xs2[13] = lo_1_142;
            float hi_1_143 = fmaxf(xs2[14], xs2[15]);
            float lo_1_143 = fminf(xs2[14], xs2[15]);
            xs2[14] = hi_1_143;
            xs2[15] = lo_1_143;
            float hi_1_144 = fmaxf(xs2[0], xs2[2]);
            float lo_1_144 = fminf(xs2[0], xs2[2]);
            xs2[0] = hi_1_144;
            xs2[2] = lo_1_144;
            float hi_1_145 = fmaxf(xs2[1], xs2[3]);
            float lo_1_145 = fminf(xs2[1], xs2[3]);
            xs2[1] = hi_1_145;
            xs2[3] = lo_1_145;
            float hi_1_146 = fmaxf(xs2[4], xs2[10]);
            float lo_1_146 = fminf(xs2[4], xs2[10]);
            xs2[4] = hi_1_146;
            xs2[10] = lo_1_146;
            float hi_1_147 = fmaxf(xs2[5], xs2[11]);
            float lo_1_147 = fminf(xs2[5], xs2[11]);
            xs2[5] = hi_1_147;
            xs2[11] = lo_1_147;
            float hi_1_148 = fmaxf(xs2[6], xs2[7]);
            float lo_1_148 = fminf(xs2[6], xs2[7]);
            xs2[6] = hi_1_148;
            xs2[7] = lo_1_148;
            float hi_1_149 = fmaxf(xs2[8], xs2[9]);
            float lo_1_149 = fminf(xs2[8], xs2[9]);
            xs2[8] = hi_1_149;
            xs2[9] = lo_1_149;
            float hi_1_150 = fmaxf(xs2[12], xs2[14]);
            float lo_1_150 = fminf(xs2[12], xs2[14]);
            xs2[12] = hi_1_150;
            xs2[14] = lo_1_150;
            float hi_1_151 = fmaxf(xs2[13], xs2[15]);
            float lo_1_151 = fminf(xs2[13], xs2[15]);
            xs2[13] = hi_1_151;
            xs2[15] = lo_1_151;
            float hi_1_152 = fmaxf(xs2[1], xs2[2]);
            float lo_1_152 = fminf(xs2[1], xs2[2]);
            xs2[1] = hi_1_152;
            xs2[2] = lo_1_152;
            float hi_1_153 = fmaxf(xs2[3], xs2[12]);
            float lo_1_153 = fminf(xs2[3], xs2[12]);
            xs2[3] = hi_1_153;
            xs2[12] = lo_1_153;
            float hi_1_154 = fmaxf(xs2[4], xs2[6]);
            float lo_1_154 = fminf(xs2[4], xs2[6]);
            xs2[4] = hi_1_154;
            xs2[6] = lo_1_154;
            float hi_1_155 = fmaxf(xs2[5], xs2[7]);
            float lo_1_155 = fminf(xs2[5], xs2[7]);
            xs2[5] = hi_1_155;
            xs2[7] = lo_1_155;
            float hi_1_156 = fmaxf(xs2[8], xs2[10]);
            float lo_1_156 = fminf(xs2[8], xs2[10]);
            xs2[8] = hi_1_156;
            xs2[10] = lo_1_156;
            float hi_1_157 = fmaxf(xs2[9], xs2[11]);
            float lo_1_157 = fminf(xs2[9], xs2[11]);
            xs2[9] = hi_1_157;
            xs2[11] = lo_1_157;
            float hi_1_158 = fmaxf(xs2[13], xs2[14]);
            float lo_1_158 = fminf(xs2[13], xs2[14]);
            xs2[13] = hi_1_158;
            xs2[14] = lo_1_158;
            float hi_1_159 = fmaxf(xs2[1], xs2[4]);
            float lo_1_159 = fminf(xs2[1], xs2[4]);
            xs2[1] = hi_1_159;
            xs2[4] = lo_1_159;
            float hi_1_160 = fmaxf(xs2[2], xs2[6]);
            float lo_1_160 = fminf(xs2[2], xs2[6]);
            xs2[2] = hi_1_160;
            xs2[6] = lo_1_160;
            float hi_1_161 = fmaxf(xs2[5], xs2[8]);
            float lo_1_161 = fminf(xs2[5], xs2[8]);
            xs2[5] = hi_1_161;
            xs2[8] = lo_1_161;
            float hi_1_162 = fmaxf(xs2[7], xs2[10]);
            float lo_1_162 = fminf(xs2[7], xs2[10]);
            xs2[7] = hi_1_162;
            xs2[10] = lo_1_162;
            float hi_1_163 = fmaxf(xs2[9], xs2[13]);
            float lo_1_163 = fminf(xs2[9], xs2[13]);
            xs2[9] = hi_1_163;
            xs2[13] = lo_1_163;
            float hi_1_164 = fmaxf(xs2[11], xs2[14]);
            float lo_1_164 = fminf(xs2[11], xs2[14]);
            xs2[11] = hi_1_164;
            xs2[14] = lo_1_164;
            float hi_1_165 = fmaxf(xs2[2], xs2[4]);
            float lo_1_165 = fminf(xs2[2], xs2[4]);
            xs2[2] = hi_1_165;
            xs2[4] = lo_1_165;
            float hi_1_166 = fmaxf(xs2[3], xs2[6]);
            float lo_1_166 = fminf(xs2[3], xs2[6]);
            xs2[3] = hi_1_166;
            xs2[6] = lo_1_166;
            float hi_1_167 = fmaxf(xs2[9], xs2[12]);
            float lo_1_167 = fminf(xs2[9], xs2[12]);
            xs2[9] = hi_1_167;
            xs2[12] = lo_1_167;
            float hi_1_168 = fmaxf(xs2[11], xs2[13]);
            float lo_1_168 = fminf(xs2[11], xs2[13]);
            xs2[11] = hi_1_168;
            xs2[13] = lo_1_168;
            float hi_1_169 = fmaxf(xs2[3], xs2[5]);
            float lo_1_169 = fminf(xs2[3], xs2[5]);
            xs2[3] = hi_1_169;
            xs2[5] = lo_1_169;
            float hi_1_170 = fmaxf(xs2[6], xs2[8]);
            float lo_1_170 = fminf(xs2[6], xs2[8]);
            xs2[6] = hi_1_170;
            xs2[8] = lo_1_170;
            float hi_1_171 = fmaxf(xs2[7], xs2[9]);
            float lo_1_171 = fminf(xs2[7], xs2[9]);
            xs2[7] = hi_1_171;
            xs2[9] = lo_1_171;
            float hi_1_172 = fmaxf(xs2[10], xs2[12]);
            float lo_1_172 = fminf(xs2[10], xs2[12]);
            xs2[10] = hi_1_172;
            xs2[12] = lo_1_172;
            float hi_1_173 = fmaxf(xs2[3], xs2[4]);
            float lo_1_173 = fminf(xs2[3], xs2[4]);
            xs2[3] = hi_1_173;
            xs2[4] = lo_1_173;
            float hi_1_174 = fmaxf(xs2[5], xs2[6]);
            float lo_1_174 = fminf(xs2[5], xs2[6]);
            xs2[5] = hi_1_174;
            xs2[6] = lo_1_174;
            float hi_1_175 = fmaxf(xs2[7], xs2[8]);
            float lo_1_175 = fminf(xs2[7], xs2[8]);
            xs2[7] = hi_1_175;
            xs2[8] = lo_1_175;
            float hi_1_176 = fmaxf(xs2[9], xs2[10]);
            float lo_1_176 = fminf(xs2[9], xs2[10]);
            xs2[9] = hi_1_176;
            xs2[10] = lo_1_176;
            float hi_1_177 = fmaxf(xs2[11], xs2[12]);
            float lo_1_177 = fminf(xs2[11], xs2[12]);
            xs2[11] = hi_1_177;
            xs2[12] = lo_1_177;
            float hi_1_178 = fmaxf(xs2[6], xs2[7]);
            float lo_1_178 = fminf(xs2[6], xs2[7]);
            xs2[6] = hi_1_178;
            xs2[7] = lo_1_178;
            float hi_1_179 = fmaxf(xs2[8], xs2[9]);
            float lo_1_179 = fminf(xs2[8], xs2[9]);
            xs2[8] = hi_1_179;
            xs2[9] = lo_1_179;
            a2[0] = fmaxf(a2[0], xs2[15]);
            a2[1] = fmaxf(a2[1], xs2[14]);
            a2[2] = fmaxf(a2[2], xs2[13]);
            a2[3] = fmaxf(a2[3], xs2[12]);
            a2[4] = fmaxf(a2[4], xs2[11]);
            a2[5] = fmaxf(a2[5], xs2[10]);
            a2[6] = fmaxf(a2[6], xs2[9]);
            a2[7] = fmaxf(a2[7], xs2[8]);
            a2[8] = fmaxf(a2[8], xs2[7]);
            a2[9] = fmaxf(a2[9], xs2[6]);
            a2[10] = fmaxf(a2[10], xs2[5]);
            a2[11] = fmaxf(a2[11], xs2[4]);
            a2[12] = fmaxf(a2[12], xs2[3]);
            a2[13] = fmaxf(a2[13], xs2[2]);
            a2[14] = fmaxf(a2[14], xs2[1]);
            a2[15] = fmaxf(a2[15], xs2[0]);
        }
    }
    if (w == 0 && col < total_q) {
        unsigned int o[16];
        unsigned int k1_1_0 = __as_u32(a[0]);
        unsigned int idx_2_0 = 4294967295u;
        if (k1_1_0 < 4282384384u) {
            idx_2_0 = k1_1_0 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_1_0 = __as_u32(a2[0]);
            idx_2_0 = 4294967295u;
            if (k2_1_0 != 0) {
                idx_2_0 = k2_1_0 & 63;
            }
        }
        o[0] = idx_2_0;
        unsigned int k1_1_1 = __as_u32(a[1]);
        unsigned int idx_2_1 = 4294967295u;
        if (k1_1_1 < 4282384384u) {
            idx_2_1 = k1_1_1 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_1_1 = __as_u32(a2[1]);
            idx_2_1 = 4294967295u;
            if (k2_1_1 != 0) {
                idx_2_1 = k2_1_1 & 63;
            }
        }
        o[1] = idx_2_1;
        unsigned int k1_1_2 = __as_u32(a[2]);
        unsigned int idx_2_2 = 4294967295u;
        if (k1_1_2 < 4282384384u) {
            idx_2_2 = k1_1_2 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_1_2 = __as_u32(a2[2]);
            idx_2_2 = 4294967295u;
            if (k2_1_2 != 0) {
                idx_2_2 = k2_1_2 & 63;
            }
        }
        o[2] = idx_2_2;
        unsigned int k1_1_3 = __as_u32(a[3]);
        unsigned int idx_2_3 = 4294967295u;
        if (k1_1_3 < 4282384384u) {
            idx_2_3 = k1_1_3 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_1_3 = __as_u32(a2[3]);
            idx_2_3 = 4294967295u;
            if (k2_1_3 != 0) {
                idx_2_3 = k2_1_3 & 63;
            }
        }
        o[3] = idx_2_3;
        unsigned int k1_1_4 = __as_u32(a[4]);
        unsigned int idx_2_4 = 4294967295u;
        if (k1_1_4 < 4282384384u) {
            idx_2_4 = k1_1_4 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_1_4 = __as_u32(a2[4]);
            idx_2_4 = 4294967295u;
            if (k2_1_4 != 0) {
                idx_2_4 = k2_1_4 & 63;
            }
        }
        o[4] = idx_2_4;
        unsigned int k1_1_5 = __as_u32(a[5]);
        unsigned int idx_2_5 = 4294967295u;
        if (k1_1_5 < 4282384384u) {
            idx_2_5 = k1_1_5 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_1_5 = __as_u32(a2[5]);
            idx_2_5 = 4294967295u;
            if (k2_1_5 != 0) {
                idx_2_5 = k2_1_5 & 63;
            }
        }
        o[5] = idx_2_5;
        unsigned int k1_1_6 = __as_u32(a[6]);
        unsigned int idx_2_6 = 4294967295u;
        if (k1_1_6 < 4282384384u) {
            idx_2_6 = k1_1_6 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_1_6 = __as_u32(a2[6]);
            idx_2_6 = 4294967295u;
            if (k2_1_6 != 0) {
                idx_2_6 = k2_1_6 & 63;
            }
        }
        o[6] = idx_2_6;
        unsigned int k1_1_7 = __as_u32(a[7]);
        unsigned int idx_2_7 = 4294967295u;
        if (k1_1_7 < 4282384384u) {
            idx_2_7 = k1_1_7 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_1_7 = __as_u32(a2[7]);
            idx_2_7 = 4294967295u;
            if (k2_1_7 != 0) {
                idx_2_7 = k2_1_7 & 63;
            }
        }
        o[7] = idx_2_7;
        unsigned int k1_1_8 = __as_u32(a[8]);
        unsigned int idx_2_8 = 4294967295u;
        if (k1_1_8 < 4282384384u) {
            idx_2_8 = k1_1_8 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_1_8 = __as_u32(a2[8]);
            idx_2_8 = 4294967295u;
            if (k2_1_8 != 0) {
                idx_2_8 = k2_1_8 & 63;
            }
        }
        o[8] = idx_2_8;
        unsigned int k1_1_9 = __as_u32(a[9]);
        unsigned int idx_2_9 = 4294967295u;
        if (k1_1_9 < 4282384384u) {
            idx_2_9 = k1_1_9 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_1_9 = __as_u32(a2[9]);
            idx_2_9 = 4294967295u;
            if (k2_1_9 != 0) {
                idx_2_9 = k2_1_9 & 63;
            }
        }
        o[9] = idx_2_9;
        unsigned int k1_1_10 = __as_u32(a[10]);
        unsigned int idx_2_10 = 4294967295u;
        if (k1_1_10 < 4282384384u) {
            idx_2_10 = k1_1_10 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_1_10 = __as_u32(a2[10]);
            idx_2_10 = 4294967295u;
            if (k2_1_10 != 0) {
                idx_2_10 = k2_1_10 & 63;
            }
        }
        o[10] = idx_2_10;
        unsigned int k1_1_11 = __as_u32(a[11]);
        unsigned int idx_2_11 = 4294967295u;
        if (k1_1_11 < 4282384384u) {
            idx_2_11 = k1_1_11 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_1_11 = __as_u32(a2[11]);
            idx_2_11 = 4294967295u;
            if (k2_1_11 != 0) {
                idx_2_11 = k2_1_11 & 63;
            }
        }
        o[11] = idx_2_11;
        unsigned int k1_1_12 = __as_u32(a[12]);
        unsigned int idx_2_12 = 4294967295u;
        if (k1_1_12 < 4282384384u) {
            idx_2_12 = k1_1_12 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_1_12 = __as_u32(a2[12]);
            idx_2_12 = 4294967295u;
            if (k2_1_12 != 0) {
                idx_2_12 = k2_1_12 & 63;
            }
        }
        o[12] = idx_2_12;
        unsigned int k1_1_13 = __as_u32(a[13]);
        unsigned int idx_2_13 = 4294967295u;
        if (k1_1_13 < 4282384384u) {
            idx_2_13 = k1_1_13 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_1_13 = __as_u32(a2[13]);
            idx_2_13 = 4294967295u;
            if (k2_1_13 != 0) {
                idx_2_13 = k2_1_13 & 63;
            }
        }
        o[13] = idx_2_13;
        unsigned int k1_1_14 = __as_u32(a[14]);
        unsigned int idx_2_14 = 4294967295u;
        if (k1_1_14 < 4282384384u) {
            idx_2_14 = k1_1_14 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_1_14 = __as_u32(a2[14]);
            idx_2_14 = 4294967295u;
            if (k2_1_14 != 0) {
                idx_2_14 = k2_1_14 & 63;
            }
        }
        o[14] = idx_2_14;
        unsigned int k1_1_15 = __as_u32(a[15]);
        unsigned int idx_2_15 = 4294967295u;
        if (k1_1_15 < 4282384384u) {
            idx_2_15 = k1_1_15 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_1_15 = __as_u32(a2[15]);
            idx_2_15 = 4294967295u;
            if (k2_1_15 != 0) {
                idx_2_15 = k2_1_15 & 63;
            }
        }
        o[15] = idx_2_15;
        unsigned int hi_96 = ((o[0]) > (o[13]) ? (o[0]) : (o[13]));
        unsigned int lo_80 = ((o[0]) < (o[13]) ? (o[0]) : (o[13]));
        o[0] = lo_80;
        o[13] = hi_96;
        unsigned int hi_97 = ((o[1]) > (o[12]) ? (o[1]) : (o[12]));
        unsigned int lo_81 = ((o[1]) < (o[12]) ? (o[1]) : (o[12]));
        o[1] = lo_81;
        o[12] = hi_97;
        unsigned int hi_98 = ((o[2]) > (o[15]) ? (o[2]) : (o[15]));
        unsigned int lo_82 = ((o[2]) < (o[15]) ? (o[2]) : (o[15]));
        o[2] = lo_82;
        o[15] = hi_98;
        unsigned int hi_99 = ((o[3]) > (o[14]) ? (o[3]) : (o[14]));
        unsigned int lo_83 = ((o[3]) < (o[14]) ? (o[3]) : (o[14]));
        o[3] = lo_83;
        o[14] = hi_99;
        unsigned int hi_100 = ((o[4]) > (o[8]) ? (o[4]) : (o[8]));
        unsigned int lo_84 = ((o[4]) < (o[8]) ? (o[4]) : (o[8]));
        o[4] = lo_84;
        o[8] = hi_100;
        unsigned int hi_101 = ((o[5]) > (o[6]) ? (o[5]) : (o[6]));
        unsigned int lo_85 = ((o[5]) < (o[6]) ? (o[5]) : (o[6]));
        o[5] = lo_85;
        o[6] = hi_101;
        unsigned int hi_102 = ((o[7]) > (o[11]) ? (o[7]) : (o[11]));
        unsigned int lo_86 = ((o[7]) < (o[11]) ? (o[7]) : (o[11]));
        o[7] = lo_86;
        o[11] = hi_102;
        unsigned int hi_103 = ((o[9]) > (o[10]) ? (o[9]) : (o[10]));
        unsigned int lo_87 = ((o[9]) < (o[10]) ? (o[9]) : (o[10]));
        o[9] = lo_87;
        o[10] = hi_103;
        unsigned int hi_104 = ((o[0]) > (o[5]) ? (o[0]) : (o[5]));
        unsigned int lo_88 = ((o[0]) < (o[5]) ? (o[0]) : (o[5]));
        o[0] = lo_88;
        o[5] = hi_104;
        unsigned int hi_105 = ((o[1]) > (o[7]) ? (o[1]) : (o[7]));
        unsigned int lo_89 = ((o[1]) < (o[7]) ? (o[1]) : (o[7]));
        o[1] = lo_89;
        o[7] = hi_105;
        unsigned int hi_106 = ((o[2]) > (o[9]) ? (o[2]) : (o[9]));
        unsigned int lo_90 = ((o[2]) < (o[9]) ? (o[2]) : (o[9]));
        o[2] = lo_90;
        o[9] = hi_106;
        unsigned int hi_107 = ((o[3]) > (o[4]) ? (o[3]) : (o[4]));
        unsigned int lo_91 = ((o[3]) < (o[4]) ? (o[3]) : (o[4]));
        o[3] = lo_91;
        o[4] = hi_107;
        unsigned int hi_108 = ((o[6]) > (o[13]) ? (o[6]) : (o[13]));
        unsigned int lo_92 = ((o[6]) < (o[13]) ? (o[6]) : (o[13]));
        o[6] = lo_92;
        o[13] = hi_108;
        unsigned int hi_109 = ((o[8]) > (o[14]) ? (o[8]) : (o[14]));
        unsigned int lo_93 = ((o[8]) < (o[14]) ? (o[8]) : (o[14]));
        o[8] = lo_93;
        o[14] = hi_109;
        unsigned int hi_110 = ((o[10]) > (o[15]) ? (o[10]) : (o[15]));
        unsigned int lo_94 = ((o[10]) < (o[15]) ? (o[10]) : (o[15]));
        o[10] = lo_94;
        o[15] = hi_110;
        unsigned int hi_111 = ((o[11]) > (o[12]) ? (o[11]) : (o[12]));
        unsigned int lo_95 = ((o[11]) < (o[12]) ? (o[11]) : (o[12]));
        o[11] = lo_95;
        o[12] = hi_111;
        unsigned int hi_112 = ((o[0]) > (o[1]) ? (o[0]) : (o[1]));
        unsigned int lo_96 = ((o[0]) < (o[1]) ? (o[0]) : (o[1]));
        o[0] = lo_96;
        o[1] = hi_112;
        unsigned int hi_113 = ((o[2]) > (o[3]) ? (o[2]) : (o[3]));
        unsigned int lo_97 = ((o[2]) < (o[3]) ? (o[2]) : (o[3]));
        o[2] = lo_97;
        o[3] = hi_113;
        unsigned int hi_114 = ((o[4]) > (o[5]) ? (o[4]) : (o[5]));
        unsigned int lo_98 = ((o[4]) < (o[5]) ? (o[4]) : (o[5]));
        o[4] = lo_98;
        o[5] = hi_114;
        unsigned int hi_115 = ((o[6]) > (o[8]) ? (o[6]) : (o[8]));
        unsigned int lo_99 = ((o[6]) < (o[8]) ? (o[6]) : (o[8]));
        o[6] = lo_99;
        o[8] = hi_115;
        unsigned int hi_116 = ((o[7]) > (o[9]) ? (o[7]) : (o[9]));
        unsigned int lo_100 = ((o[7]) < (o[9]) ? (o[7]) : (o[9]));
        o[7] = lo_100;
        o[9] = hi_116;
        unsigned int hi_117 = ((o[10]) > (o[11]) ? (o[10]) : (o[11]));
        unsigned int lo_101 = ((o[10]) < (o[11]) ? (o[10]) : (o[11]));
        o[10] = lo_101;
        o[11] = hi_117;
        unsigned int hi_118 = ((o[12]) > (o[13]) ? (o[12]) : (o[13]));
        unsigned int lo_102 = ((o[12]) < (o[13]) ? (o[12]) : (o[13]));
        o[12] = lo_102;
        o[13] = hi_118;
        unsigned int hi_119 = ((o[14]) > (o[15]) ? (o[14]) : (o[15]));
        unsigned int lo_103 = ((o[14]) < (o[15]) ? (o[14]) : (o[15]));
        o[14] = lo_103;
        o[15] = hi_119;
        unsigned int hi_120 = ((o[0]) > (o[2]) ? (o[0]) : (o[2]));
        unsigned int lo_104 = ((o[0]) < (o[2]) ? (o[0]) : (o[2]));
        o[0] = lo_104;
        o[2] = hi_120;
        unsigned int hi_121 = ((o[1]) > (o[3]) ? (o[1]) : (o[3]));
        unsigned int lo_105 = ((o[1]) < (o[3]) ? (o[1]) : (o[3]));
        o[1] = lo_105;
        o[3] = hi_121;
        unsigned int hi_122 = ((o[4]) > (o[10]) ? (o[4]) : (o[10]));
        unsigned int lo_106 = ((o[4]) < (o[10]) ? (o[4]) : (o[10]));
        o[4] = lo_106;
        o[10] = hi_122;
        unsigned int hi_123 = ((o[5]) > (o[11]) ? (o[5]) : (o[11]));
        unsigned int lo_107 = ((o[5]) < (o[11]) ? (o[5]) : (o[11]));
        o[5] = lo_107;
        o[11] = hi_123;
        unsigned int hi_124 = ((o[6]) > (o[7]) ? (o[6]) : (o[7]));
        unsigned int lo_108 = ((o[6]) < (o[7]) ? (o[6]) : (o[7]));
        o[6] = lo_108;
        o[7] = hi_124;
        unsigned int hi_125 = ((o[8]) > (o[9]) ? (o[8]) : (o[9]));
        unsigned int lo_109 = ((o[8]) < (o[9]) ? (o[8]) : (o[9]));
        o[8] = lo_109;
        o[9] = hi_125;
        unsigned int hi_126 = ((o[12]) > (o[14]) ? (o[12]) : (o[14]));
        unsigned int lo_110 = ((o[12]) < (o[14]) ? (o[12]) : (o[14]));
        o[12] = lo_110;
        o[14] = hi_126;
        unsigned int hi_127 = ((o[13]) > (o[15]) ? (o[13]) : (o[15]));
        unsigned int lo_111 = ((o[13]) < (o[15]) ? (o[13]) : (o[15]));
        o[13] = lo_111;
        o[15] = hi_127;
        unsigned int hi_128 = ((o[1]) > (o[2]) ? (o[1]) : (o[2]));
        unsigned int lo_112 = ((o[1]) < (o[2]) ? (o[1]) : (o[2]));
        o[1] = lo_112;
        o[2] = hi_128;
        unsigned int hi_129 = ((o[3]) > (o[12]) ? (o[3]) : (o[12]));
        unsigned int lo_113 = ((o[3]) < (o[12]) ? (o[3]) : (o[12]));
        o[3] = lo_113;
        o[12] = hi_129;
        unsigned int hi_130 = ((o[4]) > (o[6]) ? (o[4]) : (o[6]));
        unsigned int lo_114 = ((o[4]) < (o[6]) ? (o[4]) : (o[6]));
        o[4] = lo_114;
        o[6] = hi_130;
        unsigned int hi_131 = ((o[5]) > (o[7]) ? (o[5]) : (o[7]));
        unsigned int lo_115 = ((o[5]) < (o[7]) ? (o[5]) : (o[7]));
        o[5] = lo_115;
        o[7] = hi_131;
        unsigned int hi_132 = ((o[8]) > (o[10]) ? (o[8]) : (o[10]));
        unsigned int lo_116 = ((o[8]) < (o[10]) ? (o[8]) : (o[10]));
        o[8] = lo_116;
        o[10] = hi_132;
        unsigned int hi_133 = ((o[9]) > (o[11]) ? (o[9]) : (o[11]));
        unsigned int lo_117 = ((o[9]) < (o[11]) ? (o[9]) : (o[11]));
        o[9] = lo_117;
        o[11] = hi_133;
        unsigned int hi_134 = ((o[13]) > (o[14]) ? (o[13]) : (o[14]));
        unsigned int lo_118 = ((o[13]) < (o[14]) ? (o[13]) : (o[14]));
        o[13] = lo_118;
        o[14] = hi_134;
        unsigned int hi_135 = ((o[1]) > (o[4]) ? (o[1]) : (o[4]));
        unsigned int lo_119 = ((o[1]) < (o[4]) ? (o[1]) : (o[4]));
        o[1] = lo_119;
        o[4] = hi_135;
        unsigned int hi_136 = ((o[2]) > (o[6]) ? (o[2]) : (o[6]));
        unsigned int lo_120 = ((o[2]) < (o[6]) ? (o[2]) : (o[6]));
        o[2] = lo_120;
        o[6] = hi_136;
        unsigned int hi_137 = ((o[5]) > (o[8]) ? (o[5]) : (o[8]));
        unsigned int lo_121 = ((o[5]) < (o[8]) ? (o[5]) : (o[8]));
        o[5] = lo_121;
        o[8] = hi_137;
        unsigned int hi_138 = ((o[7]) > (o[10]) ? (o[7]) : (o[10]));
        unsigned int lo_122 = ((o[7]) < (o[10]) ? (o[7]) : (o[10]));
        o[7] = lo_122;
        o[10] = hi_138;
        unsigned int hi_139 = ((o[9]) > (o[13]) ? (o[9]) : (o[13]));
        unsigned int lo_123 = ((o[9]) < (o[13]) ? (o[9]) : (o[13]));
        o[9] = lo_123;
        o[13] = hi_139;
        unsigned int hi_140 = ((o[11]) > (o[14]) ? (o[11]) : (o[14]));
        unsigned int lo_124 = ((o[11]) < (o[14]) ? (o[11]) : (o[14]));
        o[11] = lo_124;
        o[14] = hi_140;
        unsigned int hi_141 = ((o[2]) > (o[4]) ? (o[2]) : (o[4]));
        unsigned int lo_125 = ((o[2]) < (o[4]) ? (o[2]) : (o[4]));
        o[2] = lo_125;
        o[4] = hi_141;
        unsigned int hi_142 = ((o[3]) > (o[6]) ? (o[3]) : (o[6]));
        unsigned int lo_126 = ((o[3]) < (o[6]) ? (o[3]) : (o[6]));
        o[3] = lo_126;
        o[6] = hi_142;
        unsigned int hi_143 = ((o[9]) > (o[12]) ? (o[9]) : (o[12]));
        unsigned int lo_127 = ((o[9]) < (o[12]) ? (o[9]) : (o[12]));
        o[9] = lo_127;
        o[12] = hi_143;
        unsigned int hi_144 = ((o[11]) > (o[13]) ? (o[11]) : (o[13]));
        unsigned int lo_128 = ((o[11]) < (o[13]) ? (o[11]) : (o[13]));
        o[11] = lo_128;
        o[13] = hi_144;
        unsigned int hi_145 = ((o[3]) > (o[5]) ? (o[3]) : (o[5]));
        unsigned int lo_129 = ((o[3]) < (o[5]) ? (o[3]) : (o[5]));
        o[3] = lo_129;
        o[5] = hi_145;
        unsigned int hi_146 = ((o[6]) > (o[8]) ? (o[6]) : (o[8]));
        unsigned int lo_130 = ((o[6]) < (o[8]) ? (o[6]) : (o[8]));
        o[6] = lo_130;
        o[8] = hi_146;
        unsigned int hi_147 = ((o[7]) > (o[9]) ? (o[7]) : (o[9]));
        unsigned int lo_131 = ((o[7]) < (o[9]) ? (o[7]) : (o[9]));
        o[7] = lo_131;
        o[9] = hi_147;
        unsigned int hi_148 = ((o[10]) > (o[12]) ? (o[10]) : (o[12]));
        unsigned int lo_132 = ((o[10]) < (o[12]) ? (o[10]) : (o[12]));
        o[10] = lo_132;
        o[12] = hi_148;
        unsigned int hi_149 = ((o[3]) > (o[4]) ? (o[3]) : (o[4]));
        unsigned int lo_133 = ((o[3]) < (o[4]) ? (o[3]) : (o[4]));
        o[3] = lo_133;
        o[4] = hi_149;
        unsigned int hi_150 = ((o[5]) > (o[6]) ? (o[5]) : (o[6]));
        unsigned int lo_134 = ((o[5]) < (o[6]) ? (o[5]) : (o[6]));
        o[5] = lo_134;
        o[6] = hi_150;
        unsigned int hi_151 = ((o[7]) > (o[8]) ? (o[7]) : (o[8]));
        unsigned int lo_135 = ((o[7]) < (o[8]) ? (o[7]) : (o[8]));
        o[7] = lo_135;
        o[8] = hi_151;
        unsigned int hi_152 = ((o[9]) > (o[10]) ? (o[9]) : (o[10]));
        unsigned int lo_136 = ((o[9]) < (o[10]) ? (o[9]) : (o[10]));
        o[9] = lo_136;
        o[10] = hi_152;
        unsigned int hi_153 = ((o[11]) > (o[12]) ? (o[11]) : (o[12]));
        unsigned int lo_137 = ((o[11]) < (o[12]) ? (o[11]) : (o[12]));
        o[11] = lo_137;
        o[12] = hi_153;
        unsigned int hi_154 = ((o[6]) > (o[7]) ? (o[6]) : (o[7]));
        unsigned int lo_138 = ((o[6]) < (o[7]) ? (o[6]) : (o[7]));
        o[6] = lo_138;
        o[7] = hi_154;
        unsigned int hi_155 = ((o[8]) > (o[9]) ? (o[8]) : (o[9]));
        unsigned int lo_139 = ((o[8]) < (o[9]) ? (o[8]) : (o[9]));
        o[8] = lo_139;
        o[9] = hi_155;
        int oi[16];
        oi[0] = (int)o[0];
        oi[1] = (int)o[1];
        oi[2] = (int)o[2];
        oi[3] = (int)o[3];
        oi[4] = (int)o[4];
        oi[5] = (int)o[5];
        oi[6] = (int)o[6];
        oi[7] = (int)o[7];
        oi[8] = (int)o[8];
        oi[9] = (int)o[9];
        oi[10] = (int)o[10];
        oi[11] = (int)o[11];
        oi[12] = (int)o[12];
        oi[13] = (int)o[13];
        oi[14] = (int)o[14];
        oi[15] = (int)o[15];
        long long obase = ((long long)col * (long long)num_heads + (long long)head) * 16ll;
        {
            int4 _iv4 = make_int4(oi[0 + 0], oi[0 + 1], oi[0 + 2], oi[0 + 3]);
            *reinterpret_cast<int4*>(out + obase) = _iv4;
        }
        {
            int4 _iv4 = make_int4(oi[4 + 0], oi[4 + 1], oi[4 + 2], oi[4 + 3]);
            *reinterpret_cast<int4*>(out + obase + 4ll) = _iv4;
        }
        {
            int4 _iv4 = make_int4(oi[8 + 0], oi[8 + 1], oi[8 + 2], oi[8 + 3]);
            *reinterpret_cast<int4*>(out + obase + 8ll) = _iv4;
        }
        {
            int4 _iv4 = make_int4(oi[12 + 0], oi[12 + 1], oi[12 + 2], oi[12 + 3]);
            *reinterpret_cast<int4*>(out + obase + 12ll) = _iv4;
        }
    }
}

} // extern "C"
