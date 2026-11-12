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

__global__ __launch_bounds__(32) void
kernel_cake_hopper_msa_db50816bd273fd52da82(unsigned int* __restrict__ S, int* __restrict__ nvp, int* __restrict__ out, int tiles, int total_q, int num_heads, int num_chunks, int fb, int fe)
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
    unsigned int* qcol = reinterpret_cast<unsigned int*>(smem_raw + 0);
    const int qcol_addr = smem + 0;
    unsigned int* flagw = reinterpret_cast<unsigned int*>(smem_raw + 4);
    const int flagw_addr = smem + 4;
    unsigned int* ccnt = reinterpret_cast<unsigned int*>(smem_raw + 2452);
    const int ccnt_addr = smem + 2452;
    int* limc = reinterpret_cast<int*>(smem_raw + 2456);
    const int limc_addr = smem + 2456;
    unsigned int* cbuf = reinterpret_cast<unsigned int*>(smem_raw + 2460);
    const int cbuf_addr = smem + 2460;
    float* pub = reinterpret_cast<float*>(smem_raw + 20);
    const int pub_addr = smem + 20;
    float* q2 = reinterpret_cast<float*>(smem_raw + 2196);
    const int q2_addr = smem + 2196;

    // === Task calls (declaration order) ===
    int tid_1 = threadIdx.x;
    int w = tid_1;
    int c = tid_1 - w;
    int whi = w / 32;
    int col = blockIdx.x + c;
    int head = blockIdx.y;
    if (tid_1 == 0) {
        flagw[0] = 0;
    }
    if (tid_1 == 0) {
        flagw[1] = 0;
    }
    if (tid_1 < 1) {
        ccnt[tid_1] = 0;
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
    int lim_fe = lim - fe;
    if (w == 0) {
        limc[c] = lim;
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
    if (lim > t0) {
        cb[0] = S[p];
    }
    cb[1] = 4286578688u;
    if (lim > t0 + 1) {
        cb[1] = S[p + nq64];
    }
    cb[2] = 4286578688u;
    if (lim > t0 + 2) {
        cb[2] = S[p + 2ll * nq64];
    }
    cb[3] = 4286578688u;
    if (lim > t0 + 3) {
        cb[3] = S[p + 3ll * nq64];
    }
    cb[4] = 4286578688u;
    if (lim > t0 + 4) {
        cb[4] = S[p + 4ll * nq64];
    }
    cb[5] = 4286578688u;
    if (lim > t0 + 5) {
        cb[5] = S[p + 5ll * nq64];
    }
    cb[6] = 4286578688u;
    if (lim > t0 + 6) {
        cb[6] = S[p + 6ll * nq64];
    }
    cb[7] = 4286578688u;
    if (lim > t0 + 7) {
        cb[7] = S[p + 7ll * nq64];
    }
    cb[8] = 4286578688u;
    if (lim > t0 + 8) {
        cb[8] = S[p + 8ll * nq64];
    }
    cb[9] = 4286578688u;
    if (lim > t0 + 9) {
        cb[9] = S[p + 9ll * nq64];
    }
    cb[10] = 4286578688u;
    if (lim > t0 + 10) {
        cb[10] = S[p + 10ll * nq64];
    }
    cb[11] = 4286578688u;
    if (lim > t0 + 11) {
        cb[11] = S[p + 11ll * nq64];
    }
    cb[12] = 4286578688u;
    if (lim > t0 + 12) {
        cb[12] = S[p + 12ll * nq64];
    }
    cb[13] = 4286578688u;
    if (lim > t0 + 13) {
        cb[13] = S[p + 13ll * nq64];
    }
    cb[14] = 4286578688u;
    if (lim > t0 + 14) {
        cb[14] = S[p + 14ll * nq64];
    }
    cb[15] = 4286578688u;
    if (lim > t0 + 15) {
        cb[15] = S[p + 15ll * nq64];
    }
    asm volatile("" ::: "memory");
    unsigned int nb[16];
    #pragma unroll 1
    for (int j = 0; j < num_chunks; j++) {
        int t0_1 = ((j + 1) * 32 + w) * 16;
        long long p_0 = cbase + (long long)t0_1 * nq64;
        nb[0] = 4286578688u;
        if (lim > t0_1) {
            nb[0] = S[p_0];
        }
        nb[1] = 4286578688u;
        if (lim > t0_1 + 1) {
            nb[1] = S[p_0 + nq64];
        }
        nb[2] = 4286578688u;
        if (lim > t0_1 + 2) {
            nb[2] = S[p_0 + 2ll * nq64];
        }
        nb[3] = 4286578688u;
        if (lim > t0_1 + 3) {
            nb[3] = S[p_0 + 3ll * nq64];
        }
        nb[4] = 4286578688u;
        if (lim > t0_1 + 4) {
            nb[4] = S[p_0 + 4ll * nq64];
        }
        nb[5] = 4286578688u;
        if (lim > t0_1 + 5) {
            nb[5] = S[p_0 + 5ll * nq64];
        }
        nb[6] = 4286578688u;
        if (lim > t0_1 + 6) {
            nb[6] = S[p_0 + 6ll * nq64];
        }
        nb[7] = 4286578688u;
        if (lim > t0_1 + 7) {
            nb[7] = S[p_0 + 7ll * nq64];
        }
        nb[8] = 4286578688u;
        if (lim > t0_1 + 8) {
            nb[8] = S[p_0 + 8ll * nq64];
        }
        nb[9] = 4286578688u;
        if (lim > t0_1 + 9) {
            nb[9] = S[p_0 + 9ll * nq64];
        }
        nb[10] = 4286578688u;
        if (lim > t0_1 + 10) {
            nb[10] = S[p_0 + 10ll * nq64];
        }
        nb[11] = 4286578688u;
        if (lim > t0_1 + 11) {
            nb[11] = S[p_0 + 11ll * nq64];
        }
        nb[12] = 4286578688u;
        if (lim > t0_1 + 12) {
            nb[12] = S[p_0 + 12ll * nq64];
        }
        nb[13] = 4286578688u;
        if (lim > t0_1 + 13) {
            nb[13] = S[p_0 + 13ll * nq64];
        }
        nb[14] = 4286578688u;
        if (lim > t0_1 + 14) {
            nb[14] = S[p_0 + 14ll * nq64];
        }
        nb[15] = 4286578688u;
        if (lim > t0_1 + 15) {
            nb[15] = S[p_0 + 15ll * nq64];
        }
        asm volatile("" ::: "memory");
        int t0_2 = (j * 32 + w) * 16;
        float sc = __uint_as_float(cb[0]);
        sc = fmaxf(sc, -1.7014118e+38f);
        sc = fminf(sc, 1.7014118e+38f);
        unsigned int kbits = __as_u32(sc);
        if (cb[0] == 4286578688u) {
            kbits = 4282384384u;
        }
        unsigned int key_1_0 = kbits & 4294966784u | (unsigned int)t0_2;
        int f = 0;
        if ((t0_2 < fb || lim_fe <= t0_2) && lim > t0_2) {
            f = 1;
        }
        if (f != 0) {
            key_1_0 = 2139094528 | (unsigned int)t0_2;
        }
        kb[0] = __uint_as_float(key_1_0);
        float sc_0 = __uint_as_float(cb[1]);
        sc_0 = fmaxf(sc_0, -1.7014118e+38f);
        sc_0 = fminf(sc_0, 1.7014118e+38f);
        unsigned int kbits_0 = __as_u32(sc_0);
        if (cb[1] == 4286578688u) {
            kbits_0 = 4282384384u;
        }
        unsigned int key_1_1 = kbits_0 & 4294966784u | (unsigned int)(t0_2 + 1);
        int f_0 = 0;
        if ((t0_2 + 1 < fb || lim_fe <= t0_2 + 1) && lim > t0_2 + 1) {
            f_0 = 1;
        }
        if (f_0 != 0) {
            key_1_1 = 2139094528 | (unsigned int)(t0_2 + 1);
        }
        kb[1] = __uint_as_float(key_1_1);
        float sc_1 = __uint_as_float(cb[2]);
        sc_1 = fmaxf(sc_1, -1.7014118e+38f);
        sc_1 = fminf(sc_1, 1.7014118e+38f);
        unsigned int kbits_1 = __as_u32(sc_1);
        if (cb[2] == 4286578688u) {
            kbits_1 = 4282384384u;
        }
        unsigned int key_1_2 = kbits_1 & 4294966784u | (unsigned int)(t0_2 + 2);
        int f_1 = 0;
        if ((t0_2 + 2 < fb || lim_fe <= t0_2 + 2) && lim > t0_2 + 2) {
            f_1 = 1;
        }
        if (f_1 != 0) {
            key_1_2 = 2139094528 | (unsigned int)(t0_2 + 2);
        }
        kb[2] = __uint_as_float(key_1_2);
        float sc_2 = __uint_as_float(cb[3]);
        sc_2 = fmaxf(sc_2, -1.7014118e+38f);
        sc_2 = fminf(sc_2, 1.7014118e+38f);
        unsigned int kbits_2 = __as_u32(sc_2);
        if (cb[3] == 4286578688u) {
            kbits_2 = 4282384384u;
        }
        unsigned int key_1_3 = kbits_2 & 4294966784u | (unsigned int)(t0_2 + 3);
        int f_2 = 0;
        if ((t0_2 + 3 < fb || lim_fe <= t0_2 + 3) && lim > t0_2 + 3) {
            f_2 = 1;
        }
        if (f_2 != 0) {
            key_1_3 = 2139094528 | (unsigned int)(t0_2 + 3);
        }
        kb[3] = __uint_as_float(key_1_3);
        float sc_3 = __uint_as_float(cb[4]);
        sc_3 = fmaxf(sc_3, -1.7014118e+38f);
        sc_3 = fminf(sc_3, 1.7014118e+38f);
        unsigned int kbits_3 = __as_u32(sc_3);
        if (cb[4] == 4286578688u) {
            kbits_3 = 4282384384u;
        }
        unsigned int key_1_4 = kbits_3 & 4294966784u | (unsigned int)(t0_2 + 4);
        int f_3 = 0;
        if ((t0_2 + 4 < fb || lim_fe <= t0_2 + 4) && lim > t0_2 + 4) {
            f_3 = 1;
        }
        if (f_3 != 0) {
            key_1_4 = 2139094528 | (unsigned int)(t0_2 + 4);
        }
        kb[4] = __uint_as_float(key_1_4);
        float sc_4 = __uint_as_float(cb[5]);
        sc_4 = fmaxf(sc_4, -1.7014118e+38f);
        sc_4 = fminf(sc_4, 1.7014118e+38f);
        unsigned int kbits_4 = __as_u32(sc_4);
        if (cb[5] == 4286578688u) {
            kbits_4 = 4282384384u;
        }
        unsigned int key_1_5 = kbits_4 & 4294966784u | (unsigned int)(t0_2 + 5);
        int f_4 = 0;
        if ((t0_2 + 5 < fb || lim_fe <= t0_2 + 5) && lim > t0_2 + 5) {
            f_4 = 1;
        }
        if (f_4 != 0) {
            key_1_5 = 2139094528 | (unsigned int)(t0_2 + 5);
        }
        kb[5] = __uint_as_float(key_1_5);
        float sc_5 = __uint_as_float(cb[6]);
        sc_5 = fmaxf(sc_5, -1.7014118e+38f);
        sc_5 = fminf(sc_5, 1.7014118e+38f);
        unsigned int kbits_5 = __as_u32(sc_5);
        if (cb[6] == 4286578688u) {
            kbits_5 = 4282384384u;
        }
        unsigned int key_1_6 = kbits_5 & 4294966784u | (unsigned int)(t0_2 + 6);
        int f_5 = 0;
        if ((t0_2 + 6 < fb || lim_fe <= t0_2 + 6) && lim > t0_2 + 6) {
            f_5 = 1;
        }
        if (f_5 != 0) {
            key_1_6 = 2139094528 | (unsigned int)(t0_2 + 6);
        }
        kb[6] = __uint_as_float(key_1_6);
        float sc_6 = __uint_as_float(cb[7]);
        sc_6 = fmaxf(sc_6, -1.7014118e+38f);
        sc_6 = fminf(sc_6, 1.7014118e+38f);
        unsigned int kbits_6 = __as_u32(sc_6);
        if (cb[7] == 4286578688u) {
            kbits_6 = 4282384384u;
        }
        unsigned int key_1_7 = kbits_6 & 4294966784u | (unsigned int)(t0_2 + 7);
        int f_6 = 0;
        if ((t0_2 + 7 < fb || lim_fe <= t0_2 + 7) && lim > t0_2 + 7) {
            f_6 = 1;
        }
        if (f_6 != 0) {
            key_1_7 = 2139094528 | (unsigned int)(t0_2 + 7);
        }
        kb[7] = __uint_as_float(key_1_7);
        float sc_7 = __uint_as_float(cb[8]);
        sc_7 = fmaxf(sc_7, -1.7014118e+38f);
        sc_7 = fminf(sc_7, 1.7014118e+38f);
        unsigned int kbits_7 = __as_u32(sc_7);
        if (cb[8] == 4286578688u) {
            kbits_7 = 4282384384u;
        }
        unsigned int key_1_8 = kbits_7 & 4294966784u | (unsigned int)(t0_2 + 8);
        int f_7 = 0;
        if ((t0_2 + 8 < fb || lim_fe <= t0_2 + 8) && lim > t0_2 + 8) {
            f_7 = 1;
        }
        if (f_7 != 0) {
            key_1_8 = 2139094528 | (unsigned int)(t0_2 + 8);
        }
        kb[8] = __uint_as_float(key_1_8);
        float sc_8 = __uint_as_float(cb[9]);
        sc_8 = fmaxf(sc_8, -1.7014118e+38f);
        sc_8 = fminf(sc_8, 1.7014118e+38f);
        unsigned int kbits_8 = __as_u32(sc_8);
        if (cb[9] == 4286578688u) {
            kbits_8 = 4282384384u;
        }
        unsigned int key_1_9 = kbits_8 & 4294966784u | (unsigned int)(t0_2 + 9);
        int f_8 = 0;
        if ((t0_2 + 9 < fb || lim_fe <= t0_2 + 9) && lim > t0_2 + 9) {
            f_8 = 1;
        }
        if (f_8 != 0) {
            key_1_9 = 2139094528 | (unsigned int)(t0_2 + 9);
        }
        kb[9] = __uint_as_float(key_1_9);
        float sc_9 = __uint_as_float(cb[10]);
        sc_9 = fmaxf(sc_9, -1.7014118e+38f);
        sc_9 = fminf(sc_9, 1.7014118e+38f);
        unsigned int kbits_9 = __as_u32(sc_9);
        if (cb[10] == 4286578688u) {
            kbits_9 = 4282384384u;
        }
        unsigned int key_1_10 = kbits_9 & 4294966784u | (unsigned int)(t0_2 + 10);
        int f_9 = 0;
        if ((t0_2 + 10 < fb || lim_fe <= t0_2 + 10) && lim > t0_2 + 10) {
            f_9 = 1;
        }
        if (f_9 != 0) {
            key_1_10 = 2139094528 | (unsigned int)(t0_2 + 10);
        }
        kb[10] = __uint_as_float(key_1_10);
        float sc_10 = __uint_as_float(cb[11]);
        sc_10 = fmaxf(sc_10, -1.7014118e+38f);
        sc_10 = fminf(sc_10, 1.7014118e+38f);
        unsigned int kbits_10 = __as_u32(sc_10);
        if (cb[11] == 4286578688u) {
            kbits_10 = 4282384384u;
        }
        unsigned int key_1_11 = kbits_10 & 4294966784u | (unsigned int)(t0_2 + 11);
        int f_10 = 0;
        if ((t0_2 + 11 < fb || lim_fe <= t0_2 + 11) && lim > t0_2 + 11) {
            f_10 = 1;
        }
        if (f_10 != 0) {
            key_1_11 = 2139094528 | (unsigned int)(t0_2 + 11);
        }
        kb[11] = __uint_as_float(key_1_11);
        float sc_11 = __uint_as_float(cb[12]);
        sc_11 = fmaxf(sc_11, -1.7014118e+38f);
        sc_11 = fminf(sc_11, 1.7014118e+38f);
        unsigned int kbits_11 = __as_u32(sc_11);
        if (cb[12] == 4286578688u) {
            kbits_11 = 4282384384u;
        }
        unsigned int key_1_12 = kbits_11 & 4294966784u | (unsigned int)(t0_2 + 12);
        int f_11 = 0;
        if ((t0_2 + 12 < fb || lim_fe <= t0_2 + 12) && lim > t0_2 + 12) {
            f_11 = 1;
        }
        if (f_11 != 0) {
            key_1_12 = 2139094528 | (unsigned int)(t0_2 + 12);
        }
        kb[12] = __uint_as_float(key_1_12);
        float sc_12 = __uint_as_float(cb[13]);
        sc_12 = fmaxf(sc_12, -1.7014118e+38f);
        sc_12 = fminf(sc_12, 1.7014118e+38f);
        unsigned int kbits_12 = __as_u32(sc_12);
        if (cb[13] == 4286578688u) {
            kbits_12 = 4282384384u;
        }
        unsigned int key_1_13 = kbits_12 & 4294966784u | (unsigned int)(t0_2 + 13);
        int f_12 = 0;
        if ((t0_2 + 13 < fb || lim_fe <= t0_2 + 13) && lim > t0_2 + 13) {
            f_12 = 1;
        }
        if (f_12 != 0) {
            key_1_13 = 2139094528 | (unsigned int)(t0_2 + 13);
        }
        kb[13] = __uint_as_float(key_1_13);
        float sc_13 = __uint_as_float(cb[14]);
        sc_13 = fmaxf(sc_13, -1.7014118e+38f);
        sc_13 = fminf(sc_13, 1.7014118e+38f);
        unsigned int kbits_13 = __as_u32(sc_13);
        if (cb[14] == 4286578688u) {
            kbits_13 = 4282384384u;
        }
        unsigned int key_1_14 = kbits_13 & 4294966784u | (unsigned int)(t0_2 + 14);
        int f_13 = 0;
        if ((t0_2 + 14 < fb || lim_fe <= t0_2 + 14) && lim > t0_2 + 14) {
            f_13 = 1;
        }
        if (f_13 != 0) {
            key_1_14 = 2139094528 | (unsigned int)(t0_2 + 14);
        }
        kb[14] = __uint_as_float(key_1_14);
        float sc_14 = __uint_as_float(cb[15]);
        sc_14 = fmaxf(sc_14, -1.7014118e+38f);
        sc_14 = fminf(sc_14, 1.7014118e+38f);
        unsigned int kbits_14 = __as_u32(sc_14);
        if (cb[15] == 4286578688u) {
            kbits_14 = 4282384384u;
        }
        unsigned int key_1_15 = kbits_14 & 4294966784u | (unsigned int)(t0_2 + 15);
        int f_14 = 0;
        if ((t0_2 + 15 < fb || lim_fe <= t0_2 + 15) && lim > t0_2 + 15) {
            f_14 = 1;
        }
        if (f_14 != 0) {
            key_1_15 = 2139094528 | (unsigned int)(t0_2 + 15);
        }
        kb[15] = __uint_as_float(key_1_15);
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
        float r = rej;
        float hi_0 = fmaxf(a[0], kb[15]);
        float lo_0 = fminf(a[0], kb[15]);
        a[0] = hi_0;
        r = fmaxf(r, lo_0);
        float hi_1 = fmaxf(a[1], kb[14]);
        float lo_1 = fminf(a[1], kb[14]);
        a[1] = hi_1;
        r = fmaxf(r, lo_1);
        float hi_2 = fmaxf(a[2], kb[13]);
        float lo_2 = fminf(a[2], kb[13]);
        a[2] = hi_2;
        r = fmaxf(r, lo_2);
        float hi_3 = fmaxf(a[3], kb[12]);
        float lo_3 = fminf(a[3], kb[12]);
        a[3] = hi_3;
        r = fmaxf(r, lo_3);
        float hi_4 = fmaxf(a[4], kb[11]);
        float lo_4 = fminf(a[4], kb[11]);
        a[4] = hi_4;
        r = fmaxf(r, lo_4);
        float hi_5 = fmaxf(a[5], kb[10]);
        float lo_5 = fminf(a[5], kb[10]);
        a[5] = hi_5;
        r = fmaxf(r, lo_5);
        float hi_6 = fmaxf(a[6], kb[9]);
        float lo_6 = fminf(a[6], kb[9]);
        a[6] = hi_6;
        r = fmaxf(r, lo_6);
        float hi_7 = fmaxf(a[7], kb[8]);
        float lo_7 = fminf(a[7], kb[8]);
        a[7] = hi_7;
        r = fmaxf(r, lo_7);
        float hi_8 = fmaxf(a[8], kb[7]);
        float lo_8 = fminf(a[8], kb[7]);
        a[8] = hi_8;
        r = fmaxf(r, lo_8);
        float hi_9 = fmaxf(a[9], kb[6]);
        float lo_9 = fminf(a[9], kb[6]);
        a[9] = hi_9;
        r = fmaxf(r, lo_9);
        float hi_10 = fmaxf(a[10], kb[5]);
        float lo_10 = fminf(a[10], kb[5]);
        a[10] = hi_10;
        r = fmaxf(r, lo_10);
        float hi_11 = fmaxf(a[11], kb[4]);
        float lo_11 = fminf(a[11], kb[4]);
        a[11] = hi_11;
        r = fmaxf(r, lo_11);
        float hi_12 = fmaxf(a[12], kb[3]);
        float lo_12 = fminf(a[12], kb[3]);
        a[12] = hi_12;
        r = fmaxf(r, lo_12);
        float hi_13 = fmaxf(a[13], kb[2]);
        float lo_13 = fminf(a[13], kb[2]);
        a[13] = hi_13;
        r = fmaxf(r, lo_13);
        float hi_14 = fmaxf(a[14], kb[1]);
        float lo_14 = fminf(a[14], kb[1]);
        a[14] = hi_14;
        r = fmaxf(r, lo_14);
        float hi_15 = fmaxf(a[15], kb[0]);
        float lo_15 = fminf(a[15], kb[0]);
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
    asm volatile("barrier.sync 1, 32;" ::: "memory");
    float r_1 = neg_inf;
    float V[8];
    int s0_0 = (sg * 16 + cg) * 17;
    int s1_0 = (sg * 16 + 8 + cg) * 17;
    float x0_0 = pub[s0_0 + ln];
    float y0_0 = pub[s1_0 + lnr];
    float lo0_0 = fminf(x0_0, y0_0);
    r_1 = fmaxf(r_1, lo0_0);
    float hi0_0 = fmaxf(x0_0, y0_0);
    float cur = hi0_0;
    float pv_0 = __shfl_xor_sync(0xFFFFFFFF, cur, 8);
    float hi_48 = fmaxf(cur, pv_0);
    float lo_48 = fminf(cur, pv_0);
    cur = ((up[0] != 0) ? hi_48 : lo_48);
    float pv_1 = __shfl_xor_sync(0xFFFFFFFF, cur, 4);
    float hi_49 = fmaxf(cur, pv_1);
    float lo_49 = fminf(cur, pv_1);
    cur = ((up[1] != 0) ? hi_49 : lo_49);
    float pv_2 = __shfl_xor_sync(0xFFFFFFFF, cur, 2);
    float hi_50 = fmaxf(cur, pv_2);
    float lo_50 = fminf(cur, pv_2);
    cur = ((up[2] != 0) ? hi_50 : lo_50);
    float pv_3 = __shfl_xor_sync(0xFFFFFFFF, cur, 1);
    float hi_51 = fmaxf(cur, pv_3);
    float lo_51 = fminf(cur, pv_3);
    cur = ((up[3] != 0) ? hi_51 : lo_51);
    V[0] = cur;
    int s0_1 = (sg * 16 + 1 + cg) * 17;
    int s1_1 = (sg * 16 + 1 + 8 + cg) * 17;
    float x0_1 = pub[s0_1 + ln];
    float y0_1 = pub[s1_1 + lnr];
    float lo0_1 = fminf(x0_1, y0_1);
    r_1 = fmaxf(r_1, lo0_1);
    float hi0_1 = fmaxf(x0_1, y0_1);
    float cur_0 = hi0_1;
    float pv_4 = __shfl_xor_sync(0xFFFFFFFF, cur_0, 8);
    float hi_52 = fmaxf(cur_0, pv_4);
    float lo_52 = fminf(cur_0, pv_4);
    cur_0 = ((up[0] != 0) ? hi_52 : lo_52);
    float pv_5 = __shfl_xor_sync(0xFFFFFFFF, cur_0, 4);
    float hi_53 = fmaxf(cur_0, pv_5);
    float lo_53 = fminf(cur_0, pv_5);
    cur_0 = ((up[1] != 0) ? hi_53 : lo_53);
    float pv_6 = __shfl_xor_sync(0xFFFFFFFF, cur_0, 2);
    float hi_54 = fmaxf(cur_0, pv_6);
    float lo_54 = fminf(cur_0, pv_6);
    cur_0 = ((up[2] != 0) ? hi_54 : lo_54);
    float pv_7 = __shfl_xor_sync(0xFFFFFFFF, cur_0, 1);
    float hi_55 = fmaxf(cur_0, pv_7);
    float lo_55 = fminf(cur_0, pv_7);
    cur_0 = ((up[3] != 0) ? hi_55 : lo_55);
    V[1] = cur_0;
    int s0_2 = (sg * 16 + 2 + cg) * 17;
    int s1_2 = (sg * 16 + 2 + 8 + cg) * 17;
    float x0_2 = pub[s0_2 + ln];
    float y0_2 = pub[s1_2 + lnr];
    float lo0_2 = fminf(x0_2, y0_2);
    r_1 = fmaxf(r_1, lo0_2);
    float hi0_2 = fmaxf(x0_2, y0_2);
    float cur_1 = hi0_2;
    float pv_8 = __shfl_xor_sync(0xFFFFFFFF, cur_1, 8);
    float hi_56 = fmaxf(cur_1, pv_8);
    float lo_56 = fminf(cur_1, pv_8);
    cur_1 = ((up[0] != 0) ? hi_56 : lo_56);
    float pv_9 = __shfl_xor_sync(0xFFFFFFFF, cur_1, 4);
    float hi_57 = fmaxf(cur_1, pv_9);
    float lo_57 = fminf(cur_1, pv_9);
    cur_1 = ((up[1] != 0) ? hi_57 : lo_57);
    float pv_10 = __shfl_xor_sync(0xFFFFFFFF, cur_1, 2);
    float hi_58 = fmaxf(cur_1, pv_10);
    float lo_58 = fminf(cur_1, pv_10);
    cur_1 = ((up[2] != 0) ? hi_58 : lo_58);
    float pv_11 = __shfl_xor_sync(0xFFFFFFFF, cur_1, 1);
    float hi_59 = fmaxf(cur_1, pv_11);
    float lo_59 = fminf(cur_1, pv_11);
    cur_1 = ((up[3] != 0) ? hi_59 : lo_59);
    V[2] = cur_1;
    int s0_3 = (sg * 16 + 3 + cg) * 17;
    int s1_3 = (sg * 16 + 3 + 8 + cg) * 17;
    float x0_3 = pub[s0_3 + ln];
    float y0_3 = pub[s1_3 + lnr];
    float lo0_3 = fminf(x0_3, y0_3);
    r_1 = fmaxf(r_1, lo0_3);
    float hi0_3 = fmaxf(x0_3, y0_3);
    float cur_2 = hi0_3;
    float pv_12 = __shfl_xor_sync(0xFFFFFFFF, cur_2, 8);
    float hi_60 = fmaxf(cur_2, pv_12);
    float lo_60 = fminf(cur_2, pv_12);
    cur_2 = ((up[0] != 0) ? hi_60 : lo_60);
    float pv_13 = __shfl_xor_sync(0xFFFFFFFF, cur_2, 4);
    float hi_61 = fmaxf(cur_2, pv_13);
    float lo_61 = fminf(cur_2, pv_13);
    cur_2 = ((up[1] != 0) ? hi_61 : lo_61);
    float pv_14 = __shfl_xor_sync(0xFFFFFFFF, cur_2, 2);
    float hi_62 = fmaxf(cur_2, pv_14);
    float lo_62 = fminf(cur_2, pv_14);
    cur_2 = ((up[2] != 0) ? hi_62 : lo_62);
    float pv_15 = __shfl_xor_sync(0xFFFFFFFF, cur_2, 1);
    float hi_63 = fmaxf(cur_2, pv_15);
    float lo_63 = fminf(cur_2, pv_15);
    cur_2 = ((up[3] != 0) ? hi_63 : lo_63);
    V[3] = cur_2;
    int s0_4 = (sg * 16 + 4 + cg) * 17;
    int s1_4 = (sg * 16 + 4 + 8 + cg) * 17;
    float x0_4 = pub[s0_4 + ln];
    float y0_4 = pub[s1_4 + lnr];
    float lo0_4 = fminf(x0_4, y0_4);
    r_1 = fmaxf(r_1, lo0_4);
    float hi0_4 = fmaxf(x0_4, y0_4);
    float cur_3 = hi0_4;
    float pv_16 = __shfl_xor_sync(0xFFFFFFFF, cur_3, 8);
    float hi_64 = fmaxf(cur_3, pv_16);
    float lo_64 = fminf(cur_3, pv_16);
    cur_3 = ((up[0] != 0) ? hi_64 : lo_64);
    float pv_17 = __shfl_xor_sync(0xFFFFFFFF, cur_3, 4);
    float hi_65 = fmaxf(cur_3, pv_17);
    float lo_65 = fminf(cur_3, pv_17);
    cur_3 = ((up[1] != 0) ? hi_65 : lo_65);
    float pv_18 = __shfl_xor_sync(0xFFFFFFFF, cur_3, 2);
    float hi_66 = fmaxf(cur_3, pv_18);
    float lo_66 = fminf(cur_3, pv_18);
    cur_3 = ((up[2] != 0) ? hi_66 : lo_66);
    float pv_19 = __shfl_xor_sync(0xFFFFFFFF, cur_3, 1);
    float hi_67 = fmaxf(cur_3, pv_19);
    float lo_67 = fminf(cur_3, pv_19);
    cur_3 = ((up[3] != 0) ? hi_67 : lo_67);
    V[4] = cur_3;
    int s0_5 = (sg * 16 + 5 + cg) * 17;
    int s1_5 = (sg * 16 + 5 + 8 + cg) * 17;
    float x0_5 = pub[s0_5 + ln];
    float y0_5 = pub[s1_5 + lnr];
    float lo0_5 = fminf(x0_5, y0_5);
    r_1 = fmaxf(r_1, lo0_5);
    float hi0_5 = fmaxf(x0_5, y0_5);
    float cur_4 = hi0_5;
    float pv_20 = __shfl_xor_sync(0xFFFFFFFF, cur_4, 8);
    float hi_68 = fmaxf(cur_4, pv_20);
    float lo_68 = fminf(cur_4, pv_20);
    cur_4 = ((up[0] != 0) ? hi_68 : lo_68);
    float pv_21 = __shfl_xor_sync(0xFFFFFFFF, cur_4, 4);
    float hi_69 = fmaxf(cur_4, pv_21);
    float lo_69 = fminf(cur_4, pv_21);
    cur_4 = ((up[1] != 0) ? hi_69 : lo_69);
    float pv_22 = __shfl_xor_sync(0xFFFFFFFF, cur_4, 2);
    float hi_70 = fmaxf(cur_4, pv_22);
    float lo_70 = fminf(cur_4, pv_22);
    cur_4 = ((up[2] != 0) ? hi_70 : lo_70);
    float pv_23 = __shfl_xor_sync(0xFFFFFFFF, cur_4, 1);
    float hi_71 = fmaxf(cur_4, pv_23);
    float lo_71 = fminf(cur_4, pv_23);
    cur_4 = ((up[3] != 0) ? hi_71 : lo_71);
    V[5] = cur_4;
    int s0_6 = (sg * 16 + 6 + cg) * 17;
    int s1_6 = (sg * 16 + 6 + 8 + cg) * 17;
    float x0_6 = pub[s0_6 + ln];
    float y0_6 = pub[s1_6 + lnr];
    float lo0_6 = fminf(x0_6, y0_6);
    r_1 = fmaxf(r_1, lo0_6);
    float hi0_6 = fmaxf(x0_6, y0_6);
    float cur_5 = hi0_6;
    float pv_24 = __shfl_xor_sync(0xFFFFFFFF, cur_5, 8);
    float hi_72 = fmaxf(cur_5, pv_24);
    float lo_72 = fminf(cur_5, pv_24);
    cur_5 = ((up[0] != 0) ? hi_72 : lo_72);
    float pv_25 = __shfl_xor_sync(0xFFFFFFFF, cur_5, 4);
    float hi_73 = fmaxf(cur_5, pv_25);
    float lo_73 = fminf(cur_5, pv_25);
    cur_5 = ((up[1] != 0) ? hi_73 : lo_73);
    float pv_26 = __shfl_xor_sync(0xFFFFFFFF, cur_5, 2);
    float hi_74 = fmaxf(cur_5, pv_26);
    float lo_74 = fminf(cur_5, pv_26);
    cur_5 = ((up[2] != 0) ? hi_74 : lo_74);
    float pv_27 = __shfl_xor_sync(0xFFFFFFFF, cur_5, 1);
    float hi_75 = fmaxf(cur_5, pv_27);
    float lo_75 = fminf(cur_5, pv_27);
    cur_5 = ((up[3] != 0) ? hi_75 : lo_75);
    V[6] = cur_5;
    int s0_7 = (sg * 16 + 7 + cg) * 17;
    int s1_7 = (sg * 16 + 7 + 8 + cg) * 17;
    float x0_7 = pub[s0_7 + ln];
    float y0_7 = pub[s1_7 + lnr];
    float lo0_7 = fminf(x0_7, y0_7);
    r_1 = fmaxf(r_1, lo0_7);
    float hi0_7 = fmaxf(x0_7, y0_7);
    float cur_6 = hi0_7;
    float pv_28 = __shfl_xor_sync(0xFFFFFFFF, cur_6, 8);
    float hi_76 = fmaxf(cur_6, pv_28);
    float lo_76 = fminf(cur_6, pv_28);
    cur_6 = ((up[0] != 0) ? hi_76 : lo_76);
    float pv_29 = __shfl_xor_sync(0xFFFFFFFF, cur_6, 4);
    float hi_77 = fmaxf(cur_6, pv_29);
    float lo_77 = fminf(cur_6, pv_29);
    cur_6 = ((up[1] != 0) ? hi_77 : lo_77);
    float pv_30 = __shfl_xor_sync(0xFFFFFFFF, cur_6, 2);
    float hi_78 = fmaxf(cur_6, pv_30);
    float lo_78 = fminf(cur_6, pv_30);
    cur_6 = ((up[2] != 0) ? hi_78 : lo_78);
    float pv_31 = __shfl_xor_sync(0xFFFFFFFF, cur_6, 1);
    float hi_79 = fmaxf(cur_6, pv_31);
    float lo_79 = fminf(cur_6, pv_31);
    cur_6 = ((up[3] != 0) ? hi_79 : lo_79);
    V[7] = cur_6;
    float rs = pub[(sg * 16 + ln + cg) * 17 + 16];
    r_1 = fmaxf(r_1, rs);
    float y1_0 = __shfl_xor_sync(0xFFFFFFFF, V[4], 15);
    float lo1_0 = fminf(V[0], y1_0);
    r_1 = fmaxf(r_1, lo1_0);
    float hi1_0 = fmaxf(V[0], y1_0);
    float cur_7 = hi1_0;
    float pv_32 = __shfl_xor_sync(0xFFFFFFFF, cur_7, 8);
    float hi_80 = fmaxf(cur_7, pv_32);
    float lo_80 = fminf(cur_7, pv_32);
    cur_7 = ((up[0] != 0) ? hi_80 : lo_80);
    float pv_33 = __shfl_xor_sync(0xFFFFFFFF, cur_7, 4);
    float hi_81 = fmaxf(cur_7, pv_33);
    float lo_81 = fminf(cur_7, pv_33);
    cur_7 = ((up[1] != 0) ? hi_81 : lo_81);
    float pv_34 = __shfl_xor_sync(0xFFFFFFFF, cur_7, 2);
    float hi_82 = fmaxf(cur_7, pv_34);
    float lo_82 = fminf(cur_7, pv_34);
    cur_7 = ((up[2] != 0) ? hi_82 : lo_82);
    float pv_35 = __shfl_xor_sync(0xFFFFFFFF, cur_7, 1);
    float hi_83 = fmaxf(cur_7, pv_35);
    float lo_83 = fminf(cur_7, pv_35);
    cur_7 = ((up[3] != 0) ? hi_83 : lo_83);
    V[0] = cur_7;
    float y1_1 = __shfl_xor_sync(0xFFFFFFFF, V[5], 15);
    float lo1_1 = fminf(V[1], y1_1);
    r_1 = fmaxf(r_1, lo1_1);
    float hi1_1 = fmaxf(V[1], y1_1);
    float cur_8 = hi1_1;
    float pv_36 = __shfl_xor_sync(0xFFFFFFFF, cur_8, 8);
    float hi_84 = fmaxf(cur_8, pv_36);
    float lo_84 = fminf(cur_8, pv_36);
    cur_8 = ((up[0] != 0) ? hi_84 : lo_84);
    float pv_37 = __shfl_xor_sync(0xFFFFFFFF, cur_8, 4);
    float hi_85 = fmaxf(cur_8, pv_37);
    float lo_85 = fminf(cur_8, pv_37);
    cur_8 = ((up[1] != 0) ? hi_85 : lo_85);
    float pv_38 = __shfl_xor_sync(0xFFFFFFFF, cur_8, 2);
    float hi_86 = fmaxf(cur_8, pv_38);
    float lo_86 = fminf(cur_8, pv_38);
    cur_8 = ((up[2] != 0) ? hi_86 : lo_86);
    float pv_39 = __shfl_xor_sync(0xFFFFFFFF, cur_8, 1);
    float hi_87 = fmaxf(cur_8, pv_39);
    float lo_87 = fminf(cur_8, pv_39);
    cur_8 = ((up[3] != 0) ? hi_87 : lo_87);
    V[1] = cur_8;
    float y1_2 = __shfl_xor_sync(0xFFFFFFFF, V[6], 15);
    float lo1_2 = fminf(V[2], y1_2);
    r_1 = fmaxf(r_1, lo1_2);
    float hi1_2 = fmaxf(V[2], y1_2);
    float cur_9 = hi1_2;
    float pv_40 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 8);
    float hi_88 = fmaxf(cur_9, pv_40);
    float lo_88 = fminf(cur_9, pv_40);
    cur_9 = ((up[0] != 0) ? hi_88 : lo_88);
    float pv_41 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 4);
    float hi_89 = fmaxf(cur_9, pv_41);
    float lo_89 = fminf(cur_9, pv_41);
    cur_9 = ((up[1] != 0) ? hi_89 : lo_89);
    float pv_42 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 2);
    float hi_90 = fmaxf(cur_9, pv_42);
    float lo_90 = fminf(cur_9, pv_42);
    cur_9 = ((up[2] != 0) ? hi_90 : lo_90);
    float pv_43 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 1);
    float hi_91 = fmaxf(cur_9, pv_43);
    float lo_91 = fminf(cur_9, pv_43);
    cur_9 = ((up[3] != 0) ? hi_91 : lo_91);
    V[2] = cur_9;
    float y1_3 = __shfl_xor_sync(0xFFFFFFFF, V[7], 15);
    float lo1_3 = fminf(V[3], y1_3);
    r_1 = fmaxf(r_1, lo1_3);
    float hi1_3 = fmaxf(V[3], y1_3);
    float cur_10 = hi1_3;
    float pv_44 = __shfl_xor_sync(0xFFFFFFFF, cur_10, 8);
    float hi_92 = fmaxf(cur_10, pv_44);
    float lo_92 = fminf(cur_10, pv_44);
    cur_10 = ((up[0] != 0) ? hi_92 : lo_92);
    float pv_45 = __shfl_xor_sync(0xFFFFFFFF, cur_10, 4);
    float hi_93 = fmaxf(cur_10, pv_45);
    float lo_93 = fminf(cur_10, pv_45);
    cur_10 = ((up[1] != 0) ? hi_93 : lo_93);
    float pv_46 = __shfl_xor_sync(0xFFFFFFFF, cur_10, 2);
    float hi_94 = fmaxf(cur_10, pv_46);
    float lo_94 = fminf(cur_10, pv_46);
    cur_10 = ((up[2] != 0) ? hi_94 : lo_94);
    float pv_47 = __shfl_xor_sync(0xFFFFFFFF, cur_10, 1);
    float hi_95 = fmaxf(cur_10, pv_47);
    float lo_95 = fminf(cur_10, pv_47);
    cur_10 = ((up[3] != 0) ? hi_95 : lo_95);
    V[3] = cur_10;
    float y1_4 = __shfl_xor_sync(0xFFFFFFFF, V[2], 15);
    float lo1_4 = fminf(V[0], y1_4);
    r_1 = fmaxf(r_1, lo1_4);
    float hi1_4 = fmaxf(V[0], y1_4);
    float cur_11 = hi1_4;
    float pv_48 = __shfl_xor_sync(0xFFFFFFFF, cur_11, 8);
    float hi_96 = fmaxf(cur_11, pv_48);
    float lo_96 = fminf(cur_11, pv_48);
    cur_11 = ((up[0] != 0) ? hi_96 : lo_96);
    float pv_49 = __shfl_xor_sync(0xFFFFFFFF, cur_11, 4);
    float hi_97 = fmaxf(cur_11, pv_49);
    float lo_97 = fminf(cur_11, pv_49);
    cur_11 = ((up[1] != 0) ? hi_97 : lo_97);
    float pv_50 = __shfl_xor_sync(0xFFFFFFFF, cur_11, 2);
    float hi_98 = fmaxf(cur_11, pv_50);
    float lo_98 = fminf(cur_11, pv_50);
    cur_11 = ((up[2] != 0) ? hi_98 : lo_98);
    float pv_51 = __shfl_xor_sync(0xFFFFFFFF, cur_11, 1);
    float hi_99 = fmaxf(cur_11, pv_51);
    float lo_99 = fminf(cur_11, pv_51);
    cur_11 = ((up[3] != 0) ? hi_99 : lo_99);
    V[0] = cur_11;
    float y1_5 = __shfl_xor_sync(0xFFFFFFFF, V[3], 15);
    float lo1_5 = fminf(V[1], y1_5);
    r_1 = fmaxf(r_1, lo1_5);
    float hi1_5 = fmaxf(V[1], y1_5);
    float cur_12 = hi1_5;
    float pv_52 = __shfl_xor_sync(0xFFFFFFFF, cur_12, 8);
    float hi_100 = fmaxf(cur_12, pv_52);
    float lo_100 = fminf(cur_12, pv_52);
    cur_12 = ((up[0] != 0) ? hi_100 : lo_100);
    float pv_53 = __shfl_xor_sync(0xFFFFFFFF, cur_12, 4);
    float hi_101 = fmaxf(cur_12, pv_53);
    float lo_101 = fminf(cur_12, pv_53);
    cur_12 = ((up[1] != 0) ? hi_101 : lo_101);
    float pv_54 = __shfl_xor_sync(0xFFFFFFFF, cur_12, 2);
    float hi_102 = fmaxf(cur_12, pv_54);
    float lo_102 = fminf(cur_12, pv_54);
    cur_12 = ((up[2] != 0) ? hi_102 : lo_102);
    float pv_55 = __shfl_xor_sync(0xFFFFFFFF, cur_12, 1);
    float hi_103 = fmaxf(cur_12, pv_55);
    float lo_103 = fminf(cur_12, pv_55);
    cur_12 = ((up[3] != 0) ? hi_103 : lo_103);
    V[1] = cur_12;
    float y1_6 = __shfl_xor_sync(0xFFFFFFFF, V[1], 15);
    float lo1_6 = fminf(V[0], y1_6);
    r_1 = fmaxf(r_1, lo1_6);
    float hi1_6 = fmaxf(V[0], y1_6);
    float cur_13 = hi1_6;
    float pv_56 = __shfl_xor_sync(0xFFFFFFFF, cur_13, 8);
    float hi_104 = fmaxf(cur_13, pv_56);
    float lo_104 = fminf(cur_13, pv_56);
    cur_13 = ((up[0] != 0) ? hi_104 : lo_104);
    float pv_57 = __shfl_xor_sync(0xFFFFFFFF, cur_13, 4);
    float hi_105 = fmaxf(cur_13, pv_57);
    float lo_105 = fminf(cur_13, pv_57);
    cur_13 = ((up[1] != 0) ? hi_105 : lo_105);
    float pv_58 = __shfl_xor_sync(0xFFFFFFFF, cur_13, 2);
    float hi_106 = fmaxf(cur_13, pv_58);
    float lo_106 = fminf(cur_13, pv_58);
    cur_13 = ((up[2] != 0) ? hi_106 : lo_106);
    float pv_59 = __shfl_xor_sync(0xFFFFFFFF, cur_13, 1);
    float hi_107 = fmaxf(cur_13, pv_59);
    float lo_107 = fminf(cur_13, pv_59);
    cur_13 = ((up[3] != 0) ? hi_107 : lo_107);
    V[0] = cur_13;
    float K = V[0];
    int qb = (sg + cg) * 32;
    q2[qb + ln] = K;
    q2[qb + 16 + ln] = r_1;
    asm volatile("barrier.sync 1, 32;" ::: "memory");
    float k2o[1];
    float r2v = neg_inf;
    r2v = fmaxf(r2v, q2[cg * 32 + 16 + ln]);
    r2v = fmaxf(r2v, q2[(1 + cg) * 32 + 16 + ln]);
    float V2[1];
    float x2 = q2[cg * 32 + ln];
    float y2 = q2[(1 + cg) * 32 + lnr];
    float lo2 = fminf(x2, y2);
    r2v = fmaxf(r2v, lo2);
    float hi2 = fmaxf(x2, y2);
    V2[0] = hi2;
    k2o[0] = V2[0];
    K = k2o[0];
    r_1 = r2v;
    rr1[0] = r_1;
    int ucol = blockIdx.x + cg;
    int commit = 0;
    if (g < 1 && ucol < total_q) {
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
    unsigned int qo5 = q;
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
        unsigned int qv7 = 4294967295u;
        if ((tid_1 & 31) < 1) {
            qv7 = qc;
        }
        unsigned int fm7 = __ballot_sync(0xFFFFFFFF, qv7 != 4294967295u);
        int nf7 = (int)(unsigned int)__popc(fm7);
        #pragma unroll 1
        for (int it7 = 0; it7 < nf7; it7++) {
            int _find_first_set_8 = __ffs(fm7);
            int fc7 = _find_first_set_8 - 1;
            fm7 = fm7 & fm7 - 1;
            unsigned int qf7 = qc;
            int limv = limc[fc7];
            int limv_fe = limv - fe;
            long long cbase2 = (long long)head * (long long)tiles * nq64 + (long long)(blockIdx.x + fc7);
            int lane_1 = tid_1 & 31;
            unsigned int bb[16];
            bb[0] = 4286578688u;
            int s5 = lane_1;
            int j5 = s5 >> 4 & 0;
            int v5 = 0;
            int s4 = lane_1;
            int i4 = s4 & 15;
            int j4 = s4 >> 4 & 0;
            int g4 = whi * 32 + (s4 >> 4);
            int t4 = (j4 * 32 + g4) * 16 + i4;
            if (j5 < num_chunks && t4 < limv) {
                v5 = 1;
            }
            if (v5 != 0) {
                int s4_0 = lane_1;
                int i4_0 = s4_0 & 15;
                int j4_0 = s4_0 >> 4 & 0;
                int g4_0 = whi * 32 + (s4_0 >> 4);
                int t4_0 = (j4_0 * 32 + g4_0) * 16 + i4_0;
                bb[0] = S[cbase2 + (long long)t4_0 * nq64];
            }
            bb[1] = 4286578688u;
            int s5_0 = lane_1 + 32;
            int j5_0 = s5_0 >> 4 & 0;
            int v5_0 = 0;
            int s4_1 = lane_1 + 32;
            int i4_1 = s4_1 & 15;
            int j4_1 = s4_1 >> 4 & 0;
            int g4_1 = whi * 32 + (s4_1 >> 4);
            int t4_1 = (j4_1 * 32 + g4_1) * 16 + i4_1;
            if (j5_0 < num_chunks && t4_1 < limv) {
                v5_0 = 1;
            }
            if (v5_0 != 0) {
                int s4_2 = lane_1 + 32;
                int i4_2 = s4_2 & 15;
                int j4_2 = s4_2 >> 4 & 0;
                int g4_2 = whi * 32 + (s4_2 >> 4);
                int t4_2 = (j4_2 * 32 + g4_2) * 16 + i4_2;
                bb[1] = S[cbase2 + (long long)t4_2 * nq64];
            }
            bb[2] = 4286578688u;
            int s5_1 = lane_1 + 64;
            int j5_1 = s5_1 >> 4 & 0;
            int v5_1 = 0;
            int s4_3 = lane_1 + 64;
            int i4_3 = s4_3 & 15;
            int j4_3 = s4_3 >> 4 & 0;
            int g4_3 = whi * 32 + (s4_3 >> 4);
            int t4_3 = (j4_3 * 32 + g4_3) * 16 + i4_3;
            if (j5_1 < num_chunks && t4_3 < limv) {
                v5_1 = 1;
            }
            if (v5_1 != 0) {
                int s4_4 = lane_1 + 64;
                int i4_4 = s4_4 & 15;
                int j4_4 = s4_4 >> 4 & 0;
                int g4_4 = whi * 32 + (s4_4 >> 4);
                int t4_4 = (j4_4 * 32 + g4_4) * 16 + i4_4;
                bb[2] = S[cbase2 + (long long)t4_4 * nq64];
            }
            bb[3] = 4286578688u;
            int s5_2 = lane_1 + 96;
            int j5_2 = s5_2 >> 4 & 0;
            int v5_2 = 0;
            int s4_5 = lane_1 + 96;
            int i4_5 = s4_5 & 15;
            int j4_5 = s4_5 >> 4 & 0;
            int g4_5 = whi * 32 + (s4_5 >> 4);
            int t4_5 = (j4_5 * 32 + g4_5) * 16 + i4_5;
            if (j5_2 < num_chunks && t4_5 < limv) {
                v5_2 = 1;
            }
            if (v5_2 != 0) {
                int s4_6 = lane_1 + 96;
                int i4_6 = s4_6 & 15;
                int j4_6 = s4_6 >> 4 & 0;
                int g4_6 = whi * 32 + (s4_6 >> 4);
                int t4_6 = (j4_6 * 32 + g4_6) * 16 + i4_6;
                bb[3] = S[cbase2 + (long long)t4_6 * nq64];
            }
            bb[4] = 4286578688u;
            int s5_3 = lane_1 + 128;
            int j5_3 = s5_3 >> 4 & 0;
            int v5_3 = 0;
            int s4_7 = lane_1 + 128;
            int i4_7 = s4_7 & 15;
            int j4_7 = s4_7 >> 4 & 0;
            int g4_7 = whi * 32 + (s4_7 >> 4);
            int t4_7 = (j4_7 * 32 + g4_7) * 16 + i4_7;
            if (j5_3 < num_chunks && t4_7 < limv) {
                v5_3 = 1;
            }
            if (v5_3 != 0) {
                int s4_8 = lane_1 + 128;
                int i4_8 = s4_8 & 15;
                int j4_8 = s4_8 >> 4 & 0;
                int g4_8 = whi * 32 + (s4_8 >> 4);
                int t4_8 = (j4_8 * 32 + g4_8) * 16 + i4_8;
                bb[4] = S[cbase2 + (long long)t4_8 * nq64];
            }
            bb[5] = 4286578688u;
            int s5_4 = lane_1 + 160;
            int j5_4 = s5_4 >> 4 & 0;
            int v5_4 = 0;
            int s4_9 = lane_1 + 160;
            int i4_9 = s4_9 & 15;
            int j4_9 = s4_9 >> 4 & 0;
            int g4_9 = whi * 32 + (s4_9 >> 4);
            int t4_9 = (j4_9 * 32 + g4_9) * 16 + i4_9;
            if (j5_4 < num_chunks && t4_9 < limv) {
                v5_4 = 1;
            }
            if (v5_4 != 0) {
                int s4_10 = lane_1 + 160;
                int i4_10 = s4_10 & 15;
                int j4_10 = s4_10 >> 4 & 0;
                int g4_10 = whi * 32 + (s4_10 >> 4);
                int t4_10 = (j4_10 * 32 + g4_10) * 16 + i4_10;
                bb[5] = S[cbase2 + (long long)t4_10 * nq64];
            }
            bb[6] = 4286578688u;
            int s5_5 = lane_1 + 192;
            int j5_5 = s5_5 >> 4 & 0;
            int v5_5 = 0;
            int s4_11 = lane_1 + 192;
            int i4_11 = s4_11 & 15;
            int j4_11 = s4_11 >> 4 & 0;
            int g4_11 = whi * 32 + (s4_11 >> 4);
            int t4_11 = (j4_11 * 32 + g4_11) * 16 + i4_11;
            if (j5_5 < num_chunks && t4_11 < limv) {
                v5_5 = 1;
            }
            if (v5_5 != 0) {
                int s4_12 = lane_1 + 192;
                int i4_12 = s4_12 & 15;
                int j4_12 = s4_12 >> 4 & 0;
                int g4_12 = whi * 32 + (s4_12 >> 4);
                int t4_12 = (j4_12 * 32 + g4_12) * 16 + i4_12;
                bb[6] = S[cbase2 + (long long)t4_12 * nq64];
            }
            bb[7] = 4286578688u;
            int s5_6 = lane_1 + 224;
            int j5_6 = s5_6 >> 4 & 0;
            int v5_6 = 0;
            int s4_13 = lane_1 + 224;
            int i4_13 = s4_13 & 15;
            int j4_13 = s4_13 >> 4 & 0;
            int g4_13 = whi * 32 + (s4_13 >> 4);
            int t4_13 = (j4_13 * 32 + g4_13) * 16 + i4_13;
            if (j5_6 < num_chunks && t4_13 < limv) {
                v5_6 = 1;
            }
            if (v5_6 != 0) {
                int s4_14 = lane_1 + 224;
                int i4_14 = s4_14 & 15;
                int j4_14 = s4_14 >> 4 & 0;
                int g4_14 = whi * 32 + (s4_14 >> 4);
                int t4_14 = (j4_14 * 32 + g4_14) * 16 + i4_14;
                bb[7] = S[cbase2 + (long long)t4_14 * nq64];
            }
            bb[8] = 4286578688u;
            int s5_7 = lane_1 + 256;
            int j5_7 = s5_7 >> 4 & 0;
            int v5_7 = 0;
            int s4_15 = lane_1 + 256;
            int i4_15 = s4_15 & 15;
            int j4_15 = s4_15 >> 4 & 0;
            int g4_15 = whi * 32 + (s4_15 >> 4);
            int t4_15 = (j4_15 * 32 + g4_15) * 16 + i4_15;
            if (j5_7 < num_chunks && t4_15 < limv) {
                v5_7 = 1;
            }
            if (v5_7 != 0) {
                int s4_16 = lane_1 + 256;
                int i4_16 = s4_16 & 15;
                int j4_16 = s4_16 >> 4 & 0;
                int g4_16 = whi * 32 + (s4_16 >> 4);
                int t4_16 = (j4_16 * 32 + g4_16) * 16 + i4_16;
                bb[8] = S[cbase2 + (long long)t4_16 * nq64];
            }
            bb[9] = 4286578688u;
            int s5_8 = lane_1 + 288;
            int j5_8 = s5_8 >> 4 & 0;
            int v5_8 = 0;
            int s4_17 = lane_1 + 288;
            int i4_17 = s4_17 & 15;
            int j4_17 = s4_17 >> 4 & 0;
            int g4_17 = whi * 32 + (s4_17 >> 4);
            int t4_17 = (j4_17 * 32 + g4_17) * 16 + i4_17;
            if (j5_8 < num_chunks && t4_17 < limv) {
                v5_8 = 1;
            }
            if (v5_8 != 0) {
                int s4_18 = lane_1 + 288;
                int i4_18 = s4_18 & 15;
                int j4_18 = s4_18 >> 4 & 0;
                int g4_18 = whi * 32 + (s4_18 >> 4);
                int t4_18 = (j4_18 * 32 + g4_18) * 16 + i4_18;
                bb[9] = S[cbase2 + (long long)t4_18 * nq64];
            }
            bb[10] = 4286578688u;
            int s5_9 = lane_1 + 320;
            int j5_9 = s5_9 >> 4 & 0;
            int v5_9 = 0;
            int s4_19 = lane_1 + 320;
            int i4_19 = s4_19 & 15;
            int j4_19 = s4_19 >> 4 & 0;
            int g4_19 = whi * 32 + (s4_19 >> 4);
            int t4_19 = (j4_19 * 32 + g4_19) * 16 + i4_19;
            if (j5_9 < num_chunks && t4_19 < limv) {
                v5_9 = 1;
            }
            if (v5_9 != 0) {
                int s4_20 = lane_1 + 320;
                int i4_20 = s4_20 & 15;
                int j4_20 = s4_20 >> 4 & 0;
                int g4_20 = whi * 32 + (s4_20 >> 4);
                int t4_20 = (j4_20 * 32 + g4_20) * 16 + i4_20;
                bb[10] = S[cbase2 + (long long)t4_20 * nq64];
            }
            bb[11] = 4286578688u;
            int s5_10 = lane_1 + 352;
            int j5_10 = s5_10 >> 4 & 0;
            int v5_10 = 0;
            int s4_21 = lane_1 + 352;
            int i4_21 = s4_21 & 15;
            int j4_21 = s4_21 >> 4 & 0;
            int g4_21 = whi * 32 + (s4_21 >> 4);
            int t4_21 = (j4_21 * 32 + g4_21) * 16 + i4_21;
            if (j5_10 < num_chunks && t4_21 < limv) {
                v5_10 = 1;
            }
            if (v5_10 != 0) {
                int s4_22 = lane_1 + 352;
                int i4_22 = s4_22 & 15;
                int j4_22 = s4_22 >> 4 & 0;
                int g4_22 = whi * 32 + (s4_22 >> 4);
                int t4_22 = (j4_22 * 32 + g4_22) * 16 + i4_22;
                bb[11] = S[cbase2 + (long long)t4_22 * nq64];
            }
            bb[12] = 4286578688u;
            int s5_11 = lane_1 + 384;
            int j5_11 = s5_11 >> 4 & 0;
            int v5_11 = 0;
            int s4_23 = lane_1 + 384;
            int i4_23 = s4_23 & 15;
            int j4_23 = s4_23 >> 4 & 0;
            int g4_23 = whi * 32 + (s4_23 >> 4);
            int t4_23 = (j4_23 * 32 + g4_23) * 16 + i4_23;
            if (j5_11 < num_chunks && t4_23 < limv) {
                v5_11 = 1;
            }
            if (v5_11 != 0) {
                int s4_24 = lane_1 + 384;
                int i4_24 = s4_24 & 15;
                int j4_24 = s4_24 >> 4 & 0;
                int g4_24 = whi * 32 + (s4_24 >> 4);
                int t4_24 = (j4_24 * 32 + g4_24) * 16 + i4_24;
                bb[12] = S[cbase2 + (long long)t4_24 * nq64];
            }
            bb[13] = 4286578688u;
            int s5_12 = lane_1 + 416;
            int j5_12 = s5_12 >> 4 & 0;
            int v5_12 = 0;
            int s4_25 = lane_1 + 416;
            int i4_25 = s4_25 & 15;
            int j4_25 = s4_25 >> 4 & 0;
            int g4_25 = whi * 32 + (s4_25 >> 4);
            int t4_25 = (j4_25 * 32 + g4_25) * 16 + i4_25;
            if (j5_12 < num_chunks && t4_25 < limv) {
                v5_12 = 1;
            }
            if (v5_12 != 0) {
                int s4_26 = lane_1 + 416;
                int i4_26 = s4_26 & 15;
                int j4_26 = s4_26 >> 4 & 0;
                int g4_26 = whi * 32 + (s4_26 >> 4);
                int t4_26 = (j4_26 * 32 + g4_26) * 16 + i4_26;
                bb[13] = S[cbase2 + (long long)t4_26 * nq64];
            }
            bb[14] = 4286578688u;
            int s5_13 = lane_1 + 448;
            int j5_13 = s5_13 >> 4 & 0;
            int v5_13 = 0;
            int s4_27 = lane_1 + 448;
            int i4_27 = s4_27 & 15;
            int j4_27 = s4_27 >> 4 & 0;
            int g4_27 = whi * 32 + (s4_27 >> 4);
            int t4_27 = (j4_27 * 32 + g4_27) * 16 + i4_27;
            if (j5_13 < num_chunks && t4_27 < limv) {
                v5_13 = 1;
            }
            if (v5_13 != 0) {
                int s4_28 = lane_1 + 448;
                int i4_28 = s4_28 & 15;
                int j4_28 = s4_28 >> 4 & 0;
                int g4_28 = whi * 32 + (s4_28 >> 4);
                int t4_28 = (j4_28 * 32 + g4_28) * 16 + i4_28;
                bb[14] = S[cbase2 + (long long)t4_28 * nq64];
            }
            bb[15] = 4286578688u;
            int s5_14 = lane_1 + 480;
            int j5_14 = s5_14 >> 4 & 0;
            int v5_14 = 0;
            int s4_29 = lane_1 + 480;
            int i4_29 = s4_29 & 15;
            int j4_29 = s4_29 >> 4 & 0;
            int g4_29 = whi * 32 + (s4_29 >> 4);
            int t4_29 = (j4_29 * 32 + g4_29) * 16 + i4_29;
            if (j5_14 < num_chunks && t4_29 < limv) {
                v5_14 = 1;
            }
            if (v5_14 != 0) {
                int s4_30 = lane_1 + 480;
                int i4_30 = s4_30 & 15;
                int j4_30 = s4_30 >> 4 & 0;
                int g4_30 = whi * 32 + (s4_30 >> 4);
                int t4_30 = (j4_30 * 32 + g4_30) * 16 + i4_30;
                bb[15] = S[cbase2 + (long long)t4_30 * nq64];
            }
            asm volatile("" ::: "memory");
            unsigned int kk2[16];
            kk2[0] = 0;
            int s5_15 = lane_1;
            int j5_15 = s5_15 >> 4 & 0;
            int v5_15 = 0;
            int s4_31 = lane_1;
            int i4_31 = s4_31 & 15;
            int j4_31 = s4_31 >> 4 & 0;
            int g4_31 = whi * 32 + (s4_31 >> 4);
            int t4_31 = (j4_31 * 32 + g4_31) * 16 + i4_31;
            if (j5_15 < num_chunks && t4_31 < limv) {
                v5_15 = 1;
            }
            if (v5_15 != 0) {
                int s4_0_1 = lane_1;
                int i4_1_0 = s4_0_1 & 15;
                int j4_2_1 = s4_0_1 >> 4 & 0;
                int g4_3_0 = whi * 32 + (s4_0_1 >> 4);
                int t4_4_1 = (j4_2_1 * 32 + g4_3_0) * 16 + i4_1_0;
                float sc_15 = __uint_as_float(bb[0]);
                sc_15 = fmaxf(sc_15, -1.7014118e+38f);
                sc_15 = fminf(sc_15, 1.7014118e+38f);
                unsigned int u8 = __as_u32(sc_15);
                unsigned int key8 = 0;
                if ((u8 & 4294966784u) == qf7) {
                    unsigned int lowb8 = (u8 ^ (unsigned int)((int)u8 >> 31) & 511) & 511;
                    key8 = 536870912 | lowb8 << 9 | (unsigned int)t4_4_1;
                }
                int f2 = 0;
                if ((t4_4_1 < fb || t4_4_1 >= limv_fe) && t4_4_1 < limv) {
                    f2 = 1;
                }
                if (f2 != 0) {
                    key8 = 0;
                }
                kk2[0] = key8;
            }
            kk2[1] = 0;
            int s5_16 = lane_1 + 32;
            int j5_16 = s5_16 >> 4 & 0;
            int v5_16 = 0;
            int s4_32 = lane_1 + 32;
            int i4_32 = s4_32 & 15;
            int j4_32 = s4_32 >> 4 & 0;
            int g4_32 = whi * 32 + (s4_32 >> 4);
            int t4_32 = (j4_32 * 32 + g4_32) * 16 + i4_32;
            if (j5_16 < num_chunks && t4_32 < limv) {
                v5_16 = 1;
            }
            if (v5_16 != 0) {
                int s4_0_2 = lane_1 + 32;
                int i4_1_1 = s4_0_2 & 15;
                int j4_2_2 = s4_0_2 >> 4 & 0;
                int g4_3_1 = whi * 32 + (s4_0_2 >> 4);
                int t4_4_2 = (j4_2_2 * 32 + g4_3_1) * 16 + i4_1_1;
                float sc_16 = __uint_as_float(bb[1]);
                sc_16 = fmaxf(sc_16, -1.7014118e+38f);
                sc_16 = fminf(sc_16, 1.7014118e+38f);
                unsigned int u8_1 = __as_u32(sc_16);
                unsigned int key8_1 = 0;
                if ((u8_1 & 4294966784u) == qf7) {
                    unsigned int lowb8_1 = (u8_1 ^ (unsigned int)((int)u8_1 >> 31) & 511) & 511;
                    key8_1 = 536870912 | lowb8_1 << 9 | (unsigned int)t4_4_2;
                }
                int f2_1 = 0;
                if ((t4_4_2 < fb || t4_4_2 >= limv_fe) && t4_4_2 < limv) {
                    f2_1 = 1;
                }
                if (f2_1 != 0) {
                    key8_1 = 0;
                }
                kk2[1] = key8_1;
            }
            kk2[2] = 0;
            int s5_17 = lane_1 + 64;
            int j5_17 = s5_17 >> 4 & 0;
            int v5_17 = 0;
            int s4_33 = lane_1 + 64;
            int i4_33 = s4_33 & 15;
            int j4_33 = s4_33 >> 4 & 0;
            int g4_33 = whi * 32 + (s4_33 >> 4);
            int t4_33 = (j4_33 * 32 + g4_33) * 16 + i4_33;
            if (j5_17 < num_chunks && t4_33 < limv) {
                v5_17 = 1;
            }
            if (v5_17 != 0) {
                int s4_0_3 = lane_1 + 64;
                int i4_1_2 = s4_0_3 & 15;
                int j4_2_3 = s4_0_3 >> 4 & 0;
                int g4_3_2 = whi * 32 + (s4_0_3 >> 4);
                int t4_4_3 = (j4_2_3 * 32 + g4_3_2) * 16 + i4_1_2;
                float sc_17 = __uint_as_float(bb[2]);
                sc_17 = fmaxf(sc_17, -1.7014118e+38f);
                sc_17 = fminf(sc_17, 1.7014118e+38f);
                unsigned int u8_2 = __as_u32(sc_17);
                unsigned int key8_2 = 0;
                if ((u8_2 & 4294966784u) == qf7) {
                    unsigned int lowb8_2 = (u8_2 ^ (unsigned int)((int)u8_2 >> 31) & 511) & 511;
                    key8_2 = 536870912 | lowb8_2 << 9 | (unsigned int)t4_4_3;
                }
                int f2_2 = 0;
                if ((t4_4_3 < fb || t4_4_3 >= limv_fe) && t4_4_3 < limv) {
                    f2_2 = 1;
                }
                if (f2_2 != 0) {
                    key8_2 = 0;
                }
                kk2[2] = key8_2;
            }
            kk2[3] = 0;
            int s5_18 = lane_1 + 96;
            int j5_18 = s5_18 >> 4 & 0;
            int v5_18 = 0;
            int s4_34 = lane_1 + 96;
            int i4_34 = s4_34 & 15;
            int j4_34 = s4_34 >> 4 & 0;
            int g4_34 = whi * 32 + (s4_34 >> 4);
            int t4_34 = (j4_34 * 32 + g4_34) * 16 + i4_34;
            if (j5_18 < num_chunks && t4_34 < limv) {
                v5_18 = 1;
            }
            if (v5_18 != 0) {
                int s4_0_4 = lane_1 + 96;
                int i4_1_3 = s4_0_4 & 15;
                int j4_2_4 = s4_0_4 >> 4 & 0;
                int g4_3_3 = whi * 32 + (s4_0_4 >> 4);
                int t4_4_4 = (j4_2_4 * 32 + g4_3_3) * 16 + i4_1_3;
                float sc_18 = __uint_as_float(bb[3]);
                sc_18 = fmaxf(sc_18, -1.7014118e+38f);
                sc_18 = fminf(sc_18, 1.7014118e+38f);
                unsigned int u8_3 = __as_u32(sc_18);
                unsigned int key8_3 = 0;
                if ((u8_3 & 4294966784u) == qf7) {
                    unsigned int lowb8_3 = (u8_3 ^ (unsigned int)((int)u8_3 >> 31) & 511) & 511;
                    key8_3 = 536870912 | lowb8_3 << 9 | (unsigned int)t4_4_4;
                }
                int f2_3 = 0;
                if ((t4_4_4 < fb || t4_4_4 >= limv_fe) && t4_4_4 < limv) {
                    f2_3 = 1;
                }
                if (f2_3 != 0) {
                    key8_3 = 0;
                }
                kk2[3] = key8_3;
            }
            kk2[4] = 0;
            int s5_19 = lane_1 + 128;
            int j5_19 = s5_19 >> 4 & 0;
            int v5_19 = 0;
            int s4_35 = lane_1 + 128;
            int i4_35 = s4_35 & 15;
            int j4_35 = s4_35 >> 4 & 0;
            int g4_35 = whi * 32 + (s4_35 >> 4);
            int t4_35 = (j4_35 * 32 + g4_35) * 16 + i4_35;
            if (j5_19 < num_chunks && t4_35 < limv) {
                v5_19 = 1;
            }
            if (v5_19 != 0) {
                int s4_0_5 = lane_1 + 128;
                int i4_1_4 = s4_0_5 & 15;
                int j4_2_5 = s4_0_5 >> 4 & 0;
                int g4_3_4 = whi * 32 + (s4_0_5 >> 4);
                int t4_4_5 = (j4_2_5 * 32 + g4_3_4) * 16 + i4_1_4;
                float sc_19 = __uint_as_float(bb[4]);
                sc_19 = fmaxf(sc_19, -1.7014118e+38f);
                sc_19 = fminf(sc_19, 1.7014118e+38f);
                unsigned int u8_4 = __as_u32(sc_19);
                unsigned int key8_4 = 0;
                if ((u8_4 & 4294966784u) == qf7) {
                    unsigned int lowb8_4 = (u8_4 ^ (unsigned int)((int)u8_4 >> 31) & 511) & 511;
                    key8_4 = 536870912 | lowb8_4 << 9 | (unsigned int)t4_4_5;
                }
                int f2_4 = 0;
                if ((t4_4_5 < fb || t4_4_5 >= limv_fe) && t4_4_5 < limv) {
                    f2_4 = 1;
                }
                if (f2_4 != 0) {
                    key8_4 = 0;
                }
                kk2[4] = key8_4;
            }
            kk2[5] = 0;
            int s5_20 = lane_1 + 160;
            int j5_20 = s5_20 >> 4 & 0;
            int v5_20 = 0;
            int s4_36 = lane_1 + 160;
            int i4_36 = s4_36 & 15;
            int j4_36 = s4_36 >> 4 & 0;
            int g4_36 = whi * 32 + (s4_36 >> 4);
            int t4_36 = (j4_36 * 32 + g4_36) * 16 + i4_36;
            if (j5_20 < num_chunks && t4_36 < limv) {
                v5_20 = 1;
            }
            if (v5_20 != 0) {
                int s4_0_6 = lane_1 + 160;
                int i4_1_5 = s4_0_6 & 15;
                int j4_2_6 = s4_0_6 >> 4 & 0;
                int g4_3_5 = whi * 32 + (s4_0_6 >> 4);
                int t4_4_6 = (j4_2_6 * 32 + g4_3_5) * 16 + i4_1_5;
                float sc_20 = __uint_as_float(bb[5]);
                sc_20 = fmaxf(sc_20, -1.7014118e+38f);
                sc_20 = fminf(sc_20, 1.7014118e+38f);
                unsigned int u8_5 = __as_u32(sc_20);
                unsigned int key8_5 = 0;
                if ((u8_5 & 4294966784u) == qf7) {
                    unsigned int lowb8_5 = (u8_5 ^ (unsigned int)((int)u8_5 >> 31) & 511) & 511;
                    key8_5 = 536870912 | lowb8_5 << 9 | (unsigned int)t4_4_6;
                }
                int f2_5 = 0;
                if ((t4_4_6 < fb || t4_4_6 >= limv_fe) && t4_4_6 < limv) {
                    f2_5 = 1;
                }
                if (f2_5 != 0) {
                    key8_5 = 0;
                }
                kk2[5] = key8_5;
            }
            kk2[6] = 0;
            int s5_21 = lane_1 + 192;
            int j5_21 = s5_21 >> 4 & 0;
            int v5_21 = 0;
            int s4_37 = lane_1 + 192;
            int i4_37 = s4_37 & 15;
            int j4_37 = s4_37 >> 4 & 0;
            int g4_37 = whi * 32 + (s4_37 >> 4);
            int t4_37 = (j4_37 * 32 + g4_37) * 16 + i4_37;
            if (j5_21 < num_chunks && t4_37 < limv) {
                v5_21 = 1;
            }
            if (v5_21 != 0) {
                int s4_0_7 = lane_1 + 192;
                int i4_1_6 = s4_0_7 & 15;
                int j4_2_7 = s4_0_7 >> 4 & 0;
                int g4_3_6 = whi * 32 + (s4_0_7 >> 4);
                int t4_4_7 = (j4_2_7 * 32 + g4_3_6) * 16 + i4_1_6;
                float sc_21 = __uint_as_float(bb[6]);
                sc_21 = fmaxf(sc_21, -1.7014118e+38f);
                sc_21 = fminf(sc_21, 1.7014118e+38f);
                unsigned int u8_6 = __as_u32(sc_21);
                unsigned int key8_6 = 0;
                if ((u8_6 & 4294966784u) == qf7) {
                    unsigned int lowb8_6 = (u8_6 ^ (unsigned int)((int)u8_6 >> 31) & 511) & 511;
                    key8_6 = 536870912 | lowb8_6 << 9 | (unsigned int)t4_4_7;
                }
                int f2_6 = 0;
                if ((t4_4_7 < fb || t4_4_7 >= limv_fe) && t4_4_7 < limv) {
                    f2_6 = 1;
                }
                if (f2_6 != 0) {
                    key8_6 = 0;
                }
                kk2[6] = key8_6;
            }
            kk2[7] = 0;
            int s5_22 = lane_1 + 224;
            int j5_22 = s5_22 >> 4 & 0;
            int v5_22 = 0;
            int s4_38 = lane_1 + 224;
            int i4_38 = s4_38 & 15;
            int j4_38 = s4_38 >> 4 & 0;
            int g4_38 = whi * 32 + (s4_38 >> 4);
            int t4_38 = (j4_38 * 32 + g4_38) * 16 + i4_38;
            if (j5_22 < num_chunks && t4_38 < limv) {
                v5_22 = 1;
            }
            if (v5_22 != 0) {
                int s4_0_8 = lane_1 + 224;
                int i4_1_7 = s4_0_8 & 15;
                int j4_2_8 = s4_0_8 >> 4 & 0;
                int g4_3_7 = whi * 32 + (s4_0_8 >> 4);
                int t4_4_8 = (j4_2_8 * 32 + g4_3_7) * 16 + i4_1_7;
                float sc_22 = __uint_as_float(bb[7]);
                sc_22 = fmaxf(sc_22, -1.7014118e+38f);
                sc_22 = fminf(sc_22, 1.7014118e+38f);
                unsigned int u8_7 = __as_u32(sc_22);
                unsigned int key8_7 = 0;
                if ((u8_7 & 4294966784u) == qf7) {
                    unsigned int lowb8_7 = (u8_7 ^ (unsigned int)((int)u8_7 >> 31) & 511) & 511;
                    key8_7 = 536870912 | lowb8_7 << 9 | (unsigned int)t4_4_8;
                }
                int f2_7 = 0;
                if ((t4_4_8 < fb || t4_4_8 >= limv_fe) && t4_4_8 < limv) {
                    f2_7 = 1;
                }
                if (f2_7 != 0) {
                    key8_7 = 0;
                }
                kk2[7] = key8_7;
            }
            kk2[8] = 0;
            int s5_23 = lane_1 + 256;
            int j5_23 = s5_23 >> 4 & 0;
            int v5_23 = 0;
            int s4_39 = lane_1 + 256;
            int i4_39 = s4_39 & 15;
            int j4_39 = s4_39 >> 4 & 0;
            int g4_39 = whi * 32 + (s4_39 >> 4);
            int t4_39 = (j4_39 * 32 + g4_39) * 16 + i4_39;
            if (j5_23 < num_chunks && t4_39 < limv) {
                v5_23 = 1;
            }
            if (v5_23 != 0) {
                int s4_0_9 = lane_1 + 256;
                int i4_1_8 = s4_0_9 & 15;
                int j4_2_9 = s4_0_9 >> 4 & 0;
                int g4_3_8 = whi * 32 + (s4_0_9 >> 4);
                int t4_4_9 = (j4_2_9 * 32 + g4_3_8) * 16 + i4_1_8;
                float sc_23 = __uint_as_float(bb[8]);
                sc_23 = fmaxf(sc_23, -1.7014118e+38f);
                sc_23 = fminf(sc_23, 1.7014118e+38f);
                unsigned int u8_8 = __as_u32(sc_23);
                unsigned int key8_8 = 0;
                if ((u8_8 & 4294966784u) == qf7) {
                    unsigned int lowb8_8 = (u8_8 ^ (unsigned int)((int)u8_8 >> 31) & 511) & 511;
                    key8_8 = 536870912 | lowb8_8 << 9 | (unsigned int)t4_4_9;
                }
                int f2_8 = 0;
                if ((t4_4_9 < fb || t4_4_9 >= limv_fe) && t4_4_9 < limv) {
                    f2_8 = 1;
                }
                if (f2_8 != 0) {
                    key8_8 = 0;
                }
                kk2[8] = key8_8;
            }
            kk2[9] = 0;
            int s5_24 = lane_1 + 288;
            int j5_24 = s5_24 >> 4 & 0;
            int v5_24 = 0;
            int s4_40 = lane_1 + 288;
            int i4_40 = s4_40 & 15;
            int j4_40 = s4_40 >> 4 & 0;
            int g4_40 = whi * 32 + (s4_40 >> 4);
            int t4_40 = (j4_40 * 32 + g4_40) * 16 + i4_40;
            if (j5_24 < num_chunks && t4_40 < limv) {
                v5_24 = 1;
            }
            if (v5_24 != 0) {
                int s4_0_10 = lane_1 + 288;
                int i4_1_9 = s4_0_10 & 15;
                int j4_2_10 = s4_0_10 >> 4 & 0;
                int g4_3_9 = whi * 32 + (s4_0_10 >> 4);
                int t4_4_10 = (j4_2_10 * 32 + g4_3_9) * 16 + i4_1_9;
                float sc_24 = __uint_as_float(bb[9]);
                sc_24 = fmaxf(sc_24, -1.7014118e+38f);
                sc_24 = fminf(sc_24, 1.7014118e+38f);
                unsigned int u8_9 = __as_u32(sc_24);
                unsigned int key8_9 = 0;
                if ((u8_9 & 4294966784u) == qf7) {
                    unsigned int lowb8_9 = (u8_9 ^ (unsigned int)((int)u8_9 >> 31) & 511) & 511;
                    key8_9 = 536870912 | lowb8_9 << 9 | (unsigned int)t4_4_10;
                }
                int f2_9 = 0;
                if ((t4_4_10 < fb || t4_4_10 >= limv_fe) && t4_4_10 < limv) {
                    f2_9 = 1;
                }
                if (f2_9 != 0) {
                    key8_9 = 0;
                }
                kk2[9] = key8_9;
            }
            kk2[10] = 0;
            int s5_25 = lane_1 + 320;
            int j5_25 = s5_25 >> 4 & 0;
            int v5_25 = 0;
            int s4_41 = lane_1 + 320;
            int i4_41 = s4_41 & 15;
            int j4_41 = s4_41 >> 4 & 0;
            int g4_41 = whi * 32 + (s4_41 >> 4);
            int t4_41 = (j4_41 * 32 + g4_41) * 16 + i4_41;
            if (j5_25 < num_chunks && t4_41 < limv) {
                v5_25 = 1;
            }
            if (v5_25 != 0) {
                int s4_0_11 = lane_1 + 320;
                int i4_1_10 = s4_0_11 & 15;
                int j4_2_11 = s4_0_11 >> 4 & 0;
                int g4_3_10 = whi * 32 + (s4_0_11 >> 4);
                int t4_4_11 = (j4_2_11 * 32 + g4_3_10) * 16 + i4_1_10;
                float sc_25 = __uint_as_float(bb[10]);
                sc_25 = fmaxf(sc_25, -1.7014118e+38f);
                sc_25 = fminf(sc_25, 1.7014118e+38f);
                unsigned int u8_10 = __as_u32(sc_25);
                unsigned int key8_10 = 0;
                if ((u8_10 & 4294966784u) == qf7) {
                    unsigned int lowb8_10 = (u8_10 ^ (unsigned int)((int)u8_10 >> 31) & 511) & 511;
                    key8_10 = 536870912 | lowb8_10 << 9 | (unsigned int)t4_4_11;
                }
                int f2_10 = 0;
                if ((t4_4_11 < fb || t4_4_11 >= limv_fe) && t4_4_11 < limv) {
                    f2_10 = 1;
                }
                if (f2_10 != 0) {
                    key8_10 = 0;
                }
                kk2[10] = key8_10;
            }
            kk2[11] = 0;
            int s5_26 = lane_1 + 352;
            int j5_26 = s5_26 >> 4 & 0;
            int v5_26 = 0;
            int s4_42 = lane_1 + 352;
            int i4_42 = s4_42 & 15;
            int j4_42 = s4_42 >> 4 & 0;
            int g4_42 = whi * 32 + (s4_42 >> 4);
            int t4_42 = (j4_42 * 32 + g4_42) * 16 + i4_42;
            if (j5_26 < num_chunks && t4_42 < limv) {
                v5_26 = 1;
            }
            if (v5_26 != 0) {
                int s4_0_12 = lane_1 + 352;
                int i4_1_11 = s4_0_12 & 15;
                int j4_2_12 = s4_0_12 >> 4 & 0;
                int g4_3_11 = whi * 32 + (s4_0_12 >> 4);
                int t4_4_12 = (j4_2_12 * 32 + g4_3_11) * 16 + i4_1_11;
                float sc_26 = __uint_as_float(bb[11]);
                sc_26 = fmaxf(sc_26, -1.7014118e+38f);
                sc_26 = fminf(sc_26, 1.7014118e+38f);
                unsigned int u8_11 = __as_u32(sc_26);
                unsigned int key8_11 = 0;
                if ((u8_11 & 4294966784u) == qf7) {
                    unsigned int lowb8_11 = (u8_11 ^ (unsigned int)((int)u8_11 >> 31) & 511) & 511;
                    key8_11 = 536870912 | lowb8_11 << 9 | (unsigned int)t4_4_12;
                }
                int f2_11 = 0;
                if ((t4_4_12 < fb || t4_4_12 >= limv_fe) && t4_4_12 < limv) {
                    f2_11 = 1;
                }
                if (f2_11 != 0) {
                    key8_11 = 0;
                }
                kk2[11] = key8_11;
            }
            kk2[12] = 0;
            int s5_27 = lane_1 + 384;
            int j5_27 = s5_27 >> 4 & 0;
            int v5_27 = 0;
            int s4_43 = lane_1 + 384;
            int i4_43 = s4_43 & 15;
            int j4_43 = s4_43 >> 4 & 0;
            int g4_43 = whi * 32 + (s4_43 >> 4);
            int t4_43 = (j4_43 * 32 + g4_43) * 16 + i4_43;
            if (j5_27 < num_chunks && t4_43 < limv) {
                v5_27 = 1;
            }
            if (v5_27 != 0) {
                int s4_0_13 = lane_1 + 384;
                int i4_1_12 = s4_0_13 & 15;
                int j4_2_13 = s4_0_13 >> 4 & 0;
                int g4_3_12 = whi * 32 + (s4_0_13 >> 4);
                int t4_4_13 = (j4_2_13 * 32 + g4_3_12) * 16 + i4_1_12;
                float sc_27 = __uint_as_float(bb[12]);
                sc_27 = fmaxf(sc_27, -1.7014118e+38f);
                sc_27 = fminf(sc_27, 1.7014118e+38f);
                unsigned int u8_12 = __as_u32(sc_27);
                unsigned int key8_12 = 0;
                if ((u8_12 & 4294966784u) == qf7) {
                    unsigned int lowb8_12 = (u8_12 ^ (unsigned int)((int)u8_12 >> 31) & 511) & 511;
                    key8_12 = 536870912 | lowb8_12 << 9 | (unsigned int)t4_4_13;
                }
                int f2_12 = 0;
                if ((t4_4_13 < fb || t4_4_13 >= limv_fe) && t4_4_13 < limv) {
                    f2_12 = 1;
                }
                if (f2_12 != 0) {
                    key8_12 = 0;
                }
                kk2[12] = key8_12;
            }
            kk2[13] = 0;
            int s5_28 = lane_1 + 416;
            int j5_28 = s5_28 >> 4 & 0;
            int v5_28 = 0;
            int s4_44 = lane_1 + 416;
            int i4_44 = s4_44 & 15;
            int j4_44 = s4_44 >> 4 & 0;
            int g4_44 = whi * 32 + (s4_44 >> 4);
            int t4_44 = (j4_44 * 32 + g4_44) * 16 + i4_44;
            if (j5_28 < num_chunks && t4_44 < limv) {
                v5_28 = 1;
            }
            if (v5_28 != 0) {
                int s4_0_14 = lane_1 + 416;
                int i4_1_13 = s4_0_14 & 15;
                int j4_2_14 = s4_0_14 >> 4 & 0;
                int g4_3_13 = whi * 32 + (s4_0_14 >> 4);
                int t4_4_14 = (j4_2_14 * 32 + g4_3_13) * 16 + i4_1_13;
                float sc_28 = __uint_as_float(bb[13]);
                sc_28 = fmaxf(sc_28, -1.7014118e+38f);
                sc_28 = fminf(sc_28, 1.7014118e+38f);
                unsigned int u8_13 = __as_u32(sc_28);
                unsigned int key8_13 = 0;
                if ((u8_13 & 4294966784u) == qf7) {
                    unsigned int lowb8_13 = (u8_13 ^ (unsigned int)((int)u8_13 >> 31) & 511) & 511;
                    key8_13 = 536870912 | lowb8_13 << 9 | (unsigned int)t4_4_14;
                }
                int f2_13 = 0;
                if ((t4_4_14 < fb || t4_4_14 >= limv_fe) && t4_4_14 < limv) {
                    f2_13 = 1;
                }
                if (f2_13 != 0) {
                    key8_13 = 0;
                }
                kk2[13] = key8_13;
            }
            kk2[14] = 0;
            int s5_29 = lane_1 + 448;
            int j5_29 = s5_29 >> 4 & 0;
            int v5_29 = 0;
            int s4_45 = lane_1 + 448;
            int i4_45 = s4_45 & 15;
            int j4_45 = s4_45 >> 4 & 0;
            int g4_45 = whi * 32 + (s4_45 >> 4);
            int t4_45 = (j4_45 * 32 + g4_45) * 16 + i4_45;
            if (j5_29 < num_chunks && t4_45 < limv) {
                v5_29 = 1;
            }
            if (v5_29 != 0) {
                int s4_0_15 = lane_1 + 448;
                int i4_1_14 = s4_0_15 & 15;
                int j4_2_15 = s4_0_15 >> 4 & 0;
                int g4_3_14 = whi * 32 + (s4_0_15 >> 4);
                int t4_4_15 = (j4_2_15 * 32 + g4_3_14) * 16 + i4_1_14;
                float sc_29 = __uint_as_float(bb[14]);
                sc_29 = fmaxf(sc_29, -1.7014118e+38f);
                sc_29 = fminf(sc_29, 1.7014118e+38f);
                unsigned int u8_14 = __as_u32(sc_29);
                unsigned int key8_14 = 0;
                if ((u8_14 & 4294966784u) == qf7) {
                    unsigned int lowb8_14 = (u8_14 ^ (unsigned int)((int)u8_14 >> 31) & 511) & 511;
                    key8_14 = 536870912 | lowb8_14 << 9 | (unsigned int)t4_4_15;
                }
                int f2_14 = 0;
                if ((t4_4_15 < fb || t4_4_15 >= limv_fe) && t4_4_15 < limv) {
                    f2_14 = 1;
                }
                if (f2_14 != 0) {
                    key8_14 = 0;
                }
                kk2[14] = key8_14;
            }
            kk2[15] = 0;
            int s5_30 = lane_1 + 480;
            int j5_30 = s5_30 >> 4 & 0;
            int v5_30 = 0;
            int s4_46 = lane_1 + 480;
            int i4_46 = s4_46 & 15;
            int j4_46 = s4_46 >> 4 & 0;
            int g4_46 = whi * 32 + (s4_46 >> 4);
            int t4_46 = (j4_46 * 32 + g4_46) * 16 + i4_46;
            if (j5_30 < num_chunks && t4_46 < limv) {
                v5_30 = 1;
            }
            if (v5_30 != 0) {
                int s4_0_16 = lane_1 + 480;
                int i4_1_15 = s4_0_16 & 15;
                int j4_2_16 = s4_0_16 >> 4 & 0;
                int g4_3_15 = whi * 32 + (s4_0_16 >> 4);
                int t4_4_16 = (j4_2_16 * 32 + g4_3_15) * 16 + i4_1_15;
                float sc_30 = __uint_as_float(bb[15]);
                sc_30 = fmaxf(sc_30, -1.7014118e+38f);
                sc_30 = fminf(sc_30, 1.7014118e+38f);
                unsigned int u8_15 = __as_u32(sc_30);
                unsigned int key8_15 = 0;
                if ((u8_15 & 4294966784u) == qf7) {
                    unsigned int lowb8_15 = (u8_15 ^ (unsigned int)((int)u8_15 >> 31) & 511) & 511;
                    key8_15 = 536870912 | lowb8_15 << 9 | (unsigned int)t4_4_16;
                }
                int f2_15 = 0;
                if ((t4_4_16 < fb || t4_4_16 >= limv_fe) && t4_4_16 < limv) {
                    f2_15 = 1;
                }
                if (f2_15 != 0) {
                    key8_15 = 0;
                }
                kk2[15] = key8_15;
            }
            int n3 = 0;
            if (kk2[0] != 0) {
                n3 = n3 + 1;
            }
            if (kk2[1] != 0) {
                n3 = n3 + 1;
            }
            if (kk2[2] != 0) {
                n3 = n3 + 1;
            }
            if (kk2[3] != 0) {
                n3 = n3 + 1;
            }
            if (kk2[4] != 0) {
                n3 = n3 + 1;
            }
            if (kk2[5] != 0) {
                n3 = n3 + 1;
            }
            if (kk2[6] != 0) {
                n3 = n3 + 1;
            }
            if (kk2[7] != 0) {
                n3 = n3 + 1;
            }
            if (kk2[8] != 0) {
                n3 = n3 + 1;
            }
            if (kk2[9] != 0) {
                n3 = n3 + 1;
            }
            if (kk2[10] != 0) {
                n3 = n3 + 1;
            }
            if (kk2[11] != 0) {
                n3 = n3 + 1;
            }
            if (kk2[12] != 0) {
                n3 = n3 + 1;
            }
            if (kk2[13] != 0) {
                n3 = n3 + 1;
            }
            if (kk2[14] != 0) {
                n3 = n3 + 1;
            }
            if (kk2[15] != 0) {
                n3 = n3 + 1;
            }
            if (n3 > 0) {
                unsigned int base3 = atomicAdd(&ccnt[fc7], (unsigned int)n3);
                if (base3 + (unsigned int)n3 > 16) {
                    flagw[1] = 1;
                } else {
                    int pos3 = fc7 * 16 + (int)base3;
                    if (kk2[0] != 0) {
                        cbuf[pos3] = kk2[0];
                        pos3 = pos3 + 1;
                    }
                    if (kk2[1] != 0) {
                        cbuf[pos3] = kk2[1];
                        pos3 = pos3 + 1;
                    }
                    if (kk2[2] != 0) {
                        cbuf[pos3] = kk2[2];
                        pos3 = pos3 + 1;
                    }
                    if (kk2[3] != 0) {
                        cbuf[pos3] = kk2[3];
                        pos3 = pos3 + 1;
                    }
                    if (kk2[4] != 0) {
                        cbuf[pos3] = kk2[4];
                        pos3 = pos3 + 1;
                    }
                    if (kk2[5] != 0) {
                        cbuf[pos3] = kk2[5];
                        pos3 = pos3 + 1;
                    }
                    if (kk2[6] != 0) {
                        cbuf[pos3] = kk2[6];
                        pos3 = pos3 + 1;
                    }
                    if (kk2[7] != 0) {
                        cbuf[pos3] = kk2[7];
                        pos3 = pos3 + 1;
                    }
                    if (kk2[8] != 0) {
                        cbuf[pos3] = kk2[8];
                        pos3 = pos3 + 1;
                    }
                    if (kk2[9] != 0) {
                        cbuf[pos3] = kk2[9];
                        pos3 = pos3 + 1;
                    }
                    if (kk2[10] != 0) {
                        cbuf[pos3] = kk2[10];
                        pos3 = pos3 + 1;
                    }
                    if (kk2[11] != 0) {
                        cbuf[pos3] = kk2[11];
                        pos3 = pos3 + 1;
                    }
                    if (kk2[12] != 0) {
                        cbuf[pos3] = kk2[12];
                        pos3 = pos3 + 1;
                    }
                    if (kk2[13] != 0) {
                        cbuf[pos3] = kk2[13];
                        pos3 = pos3 + 1;
                    }
                    if (kk2[14] != 0) {
                        cbuf[pos3] = kk2[14];
                        pos3 = pos3 + 1;
                    }
                    if (kk2[15] != 0) {
                        cbuf[pos3] = kk2[15];
                        pos3 = pos3 + 1;
                    }
                }
            }
        }
        asm volatile("barrier.sync 1, 32;" ::: "memory");
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
            int _warp_vote_9 = __any_sync(0xFFFFFFFF, ovfx != 0);
            if (_warp_vote_9 != 0) {
                unsigned int cb2[16];
                unsigned int nb2[16];
                float kb2[16];
                int t0_3 = w * 16;
                long long p_1 = cbase + (long long)t0_3 * nq64;
                cb2[0] = 4286578688u;
                if (lim2x > t0_3) {
                    cb2[0] = S[p_1];
                }
                cb2[1] = 4286578688u;
                if (lim2x > t0_3 + 1) {
                    cb2[1] = S[p_1 + nq64];
                }
                cb2[2] = 4286578688u;
                if (lim2x > t0_3 + 2) {
                    cb2[2] = S[p_1 + 2ll * nq64];
                }
                cb2[3] = 4286578688u;
                if (lim2x > t0_3 + 3) {
                    cb2[3] = S[p_1 + 3ll * nq64];
                }
                cb2[4] = 4286578688u;
                if (lim2x > t0_3 + 4) {
                    cb2[4] = S[p_1 + 4ll * nq64];
                }
                cb2[5] = 4286578688u;
                if (lim2x > t0_3 + 5) {
                    cb2[5] = S[p_1 + 5ll * nq64];
                }
                cb2[6] = 4286578688u;
                if (lim2x > t0_3 + 6) {
                    cb2[6] = S[p_1 + 6ll * nq64];
                }
                cb2[7] = 4286578688u;
                if (lim2x > t0_3 + 7) {
                    cb2[7] = S[p_1 + 7ll * nq64];
                }
                cb2[8] = 4286578688u;
                if (lim2x > t0_3 + 8) {
                    cb2[8] = S[p_1 + 8ll * nq64];
                }
                cb2[9] = 4286578688u;
                if (lim2x > t0_3 + 9) {
                    cb2[9] = S[p_1 + 9ll * nq64];
                }
                cb2[10] = 4286578688u;
                if (lim2x > t0_3 + 10) {
                    cb2[10] = S[p_1 + 10ll * nq64];
                }
                cb2[11] = 4286578688u;
                if (lim2x > t0_3 + 11) {
                    cb2[11] = S[p_1 + 11ll * nq64];
                }
                cb2[12] = 4286578688u;
                if (lim2x > t0_3 + 12) {
                    cb2[12] = S[p_1 + 12ll * nq64];
                }
                cb2[13] = 4286578688u;
                if (lim2x > t0_3 + 13) {
                    cb2[13] = S[p_1 + 13ll * nq64];
                }
                cb2[14] = 4286578688u;
                if (lim2x > t0_3 + 14) {
                    cb2[14] = S[p_1 + 14ll * nq64];
                }
                cb2[15] = 4286578688u;
                if (lim2x > t0_3 + 15) {
                    cb2[15] = S[p_1 + 15ll * nq64];
                }
                asm volatile("" ::: "memory");
                #pragma unroll 1
                for (int j_1 = 0; j_1 < num_chunks; j_1++) {
                    int t0_4 = ((j_1 + 1) * 32 + w) * 16;
                    long long p_2 = cbase + (long long)t0_4 * nq64;
                    nb2[0] = 4286578688u;
                    if (lim2x > t0_4) {
                        nb2[0] = S[p_2];
                    }
                    nb2[1] = 4286578688u;
                    if (lim2x > t0_4 + 1) {
                        nb2[1] = S[p_2 + nq64];
                    }
                    nb2[2] = 4286578688u;
                    if (lim2x > t0_4 + 2) {
                        nb2[2] = S[p_2 + 2ll * nq64];
                    }
                    nb2[3] = 4286578688u;
                    if (lim2x > t0_4 + 3) {
                        nb2[3] = S[p_2 + 3ll * nq64];
                    }
                    nb2[4] = 4286578688u;
                    if (lim2x > t0_4 + 4) {
                        nb2[4] = S[p_2 + 4ll * nq64];
                    }
                    nb2[5] = 4286578688u;
                    if (lim2x > t0_4 + 5) {
                        nb2[5] = S[p_2 + 5ll * nq64];
                    }
                    nb2[6] = 4286578688u;
                    if (lim2x > t0_4 + 6) {
                        nb2[6] = S[p_2 + 6ll * nq64];
                    }
                    nb2[7] = 4286578688u;
                    if (lim2x > t0_4 + 7) {
                        nb2[7] = S[p_2 + 7ll * nq64];
                    }
                    nb2[8] = 4286578688u;
                    if (lim2x > t0_4 + 8) {
                        nb2[8] = S[p_2 + 8ll * nq64];
                    }
                    nb2[9] = 4286578688u;
                    if (lim2x > t0_4 + 9) {
                        nb2[9] = S[p_2 + 9ll * nq64];
                    }
                    nb2[10] = 4286578688u;
                    if (lim2x > t0_4 + 10) {
                        nb2[10] = S[p_2 + 10ll * nq64];
                    }
                    nb2[11] = 4286578688u;
                    if (lim2x > t0_4 + 11) {
                        nb2[11] = S[p_2 + 11ll * nq64];
                    }
                    nb2[12] = 4286578688u;
                    if (lim2x > t0_4 + 12) {
                        nb2[12] = S[p_2 + 12ll * nq64];
                    }
                    nb2[13] = 4286578688u;
                    if (lim2x > t0_4 + 13) {
                        nb2[13] = S[p_2 + 13ll * nq64];
                    }
                    nb2[14] = 4286578688u;
                    if (lim2x > t0_4 + 14) {
                        nb2[14] = S[p_2 + 14ll * nq64];
                    }
                    nb2[15] = 4286578688u;
                    if (lim2x > t0_4 + 15) {
                        nb2[15] = S[p_2 + 15ll * nq64];
                    }
                    asm volatile("" ::: "memory");
                    int t0_5 = (j_1 * 32 + w) * 16;
                    float qf = __uint_as_float(qc);
                    float sc_31 = __uint_as_float(cb2[0]);
                    sc_31 = fmaxf(sc_31, -1.7014118e+38f);
                    sc_31 = fminf(sc_31, 1.7014118e+38f);
                    unsigned int u_1_0 = __as_u32(sc_31);
                    unsigned int cls_1_0 = u_1_0 & 4294966784u;
                    int f_15 = 0;
                    if ((t0_5 < fb || lim_fe <= t0_5) && lim > t0_5) {
                        f_15 = 1;
                    }
                    if (f_15 != 0) {
                        cls_1_0 = 2139094528;
                    }
                    unsigned int key = 0;
                    if ((qf < __uint_as_float(cls_1_0) || __uint_as_float(cls_1_0) == qf && cls_1_0 < qc) && cls_1_0 < 4278190080u) {
                        key = 1073741824 | (unsigned int)t0_5;
                    }
                    if (cls_1_0 == qc) {
                        unsigned int lowb = (u_1_0 ^ (unsigned int)((int)u_1_0 >> 31) & 511) & 511;
                        key = 536870912 | lowb << 9 | (unsigned int)t0_5;
                    }
                    kb2[0] = __uint_as_float(key);
                    float sc_15_1 = __uint_as_float(cb2[1]);
                    sc_15_1 = fmaxf(sc_15_1, -1.7014118e+38f);
                    sc_15_1 = fminf(sc_15_1, 1.7014118e+38f);
                    unsigned int u_1_1 = __as_u32(sc_15_1);
                    unsigned int cls_1_1 = u_1_1 & 4294966784u;
                    int f_15_1 = 0;
                    if ((t0_5 + 1 < fb || lim_fe <= t0_5 + 1) && lim > t0_5 + 1) {
                        f_15_1 = 1;
                    }
                    if (f_15_1 != 0) {
                        cls_1_1 = 2139094528;
                    }
                    unsigned int key_0 = 0;
                    if ((qf < __uint_as_float(cls_1_1) || __uint_as_float(cls_1_1) == qf && cls_1_1 < qc) && cls_1_1 < 4278190080u) {
                        key_0 = 1073741824 | (unsigned int)(t0_5 + 1);
                    }
                    if (cls_1_1 == qc) {
                        unsigned int lowb_1 = (u_1_1 ^ (unsigned int)((int)u_1_1 >> 31) & 511) & 511;
                        key_0 = 536870912 | lowb_1 << 9 | (unsigned int)(t0_5 + 1);
                    }
                    kb2[1] = __uint_as_float(key_0);
                    float sc_16_1 = __uint_as_float(cb2[2]);
                    sc_16_1 = fmaxf(sc_16_1, -1.7014118e+38f);
                    sc_16_1 = fminf(sc_16_1, 1.7014118e+38f);
                    unsigned int u_1_2 = __as_u32(sc_16_1);
                    unsigned int cls_1_2 = u_1_2 & 4294966784u;
                    int f_16 = 0;
                    if ((t0_5 + 2 < fb || lim_fe <= t0_5 + 2) && lim > t0_5 + 2) {
                        f_16 = 1;
                    }
                    if (f_16 != 0) {
                        cls_1_2 = 2139094528;
                    }
                    unsigned int key_1 = 0;
                    if ((qf < __uint_as_float(cls_1_2) || __uint_as_float(cls_1_2) == qf && cls_1_2 < qc) && cls_1_2 < 4278190080u) {
                        key_1 = 1073741824 | (unsigned int)(t0_5 + 2);
                    }
                    if (cls_1_2 == qc) {
                        unsigned int lowb_2 = (u_1_2 ^ (unsigned int)((int)u_1_2 >> 31) & 511) & 511;
                        key_1 = 536870912 | lowb_2 << 9 | (unsigned int)(t0_5 + 2);
                    }
                    kb2[2] = __uint_as_float(key_1);
                    float sc_17_1 = __uint_as_float(cb2[3]);
                    sc_17_1 = fmaxf(sc_17_1, -1.7014118e+38f);
                    sc_17_1 = fminf(sc_17_1, 1.7014118e+38f);
                    unsigned int u_1_3 = __as_u32(sc_17_1);
                    unsigned int cls_1_3 = u_1_3 & 4294966784u;
                    int f_17 = 0;
                    if ((t0_5 + 3 < fb || lim_fe <= t0_5 + 3) && lim > t0_5 + 3) {
                        f_17 = 1;
                    }
                    if (f_17 != 0) {
                        cls_1_3 = 2139094528;
                    }
                    unsigned int key_2 = 0;
                    if ((qf < __uint_as_float(cls_1_3) || __uint_as_float(cls_1_3) == qf && cls_1_3 < qc) && cls_1_3 < 4278190080u) {
                        key_2 = 1073741824 | (unsigned int)(t0_5 + 3);
                    }
                    if (cls_1_3 == qc) {
                        unsigned int lowb_3 = (u_1_3 ^ (unsigned int)((int)u_1_3 >> 31) & 511) & 511;
                        key_2 = 536870912 | lowb_3 << 9 | (unsigned int)(t0_5 + 3);
                    }
                    kb2[3] = __uint_as_float(key_2);
                    float sc_18_1 = __uint_as_float(cb2[4]);
                    sc_18_1 = fmaxf(sc_18_1, -1.7014118e+38f);
                    sc_18_1 = fminf(sc_18_1, 1.7014118e+38f);
                    unsigned int u_1_4 = __as_u32(sc_18_1);
                    unsigned int cls_1_4 = u_1_4 & 4294966784u;
                    int f_18 = 0;
                    if ((t0_5 + 4 < fb || lim_fe <= t0_5 + 4) && lim > t0_5 + 4) {
                        f_18 = 1;
                    }
                    if (f_18 != 0) {
                        cls_1_4 = 2139094528;
                    }
                    unsigned int key_3 = 0;
                    if ((qf < __uint_as_float(cls_1_4) || __uint_as_float(cls_1_4) == qf && cls_1_4 < qc) && cls_1_4 < 4278190080u) {
                        key_3 = 1073741824 | (unsigned int)(t0_5 + 4);
                    }
                    if (cls_1_4 == qc) {
                        unsigned int lowb_4 = (u_1_4 ^ (unsigned int)((int)u_1_4 >> 31) & 511) & 511;
                        key_3 = 536870912 | lowb_4 << 9 | (unsigned int)(t0_5 + 4);
                    }
                    kb2[4] = __uint_as_float(key_3);
                    float sc_19_1 = __uint_as_float(cb2[5]);
                    sc_19_1 = fmaxf(sc_19_1, -1.7014118e+38f);
                    sc_19_1 = fminf(sc_19_1, 1.7014118e+38f);
                    unsigned int u_1_5 = __as_u32(sc_19_1);
                    unsigned int cls_1_5 = u_1_5 & 4294966784u;
                    int f_19 = 0;
                    if ((t0_5 + 5 < fb || lim_fe <= t0_5 + 5) && lim > t0_5 + 5) {
                        f_19 = 1;
                    }
                    if (f_19 != 0) {
                        cls_1_5 = 2139094528;
                    }
                    unsigned int key_4 = 0;
                    if ((qf < __uint_as_float(cls_1_5) || __uint_as_float(cls_1_5) == qf && cls_1_5 < qc) && cls_1_5 < 4278190080u) {
                        key_4 = 1073741824 | (unsigned int)(t0_5 + 5);
                    }
                    if (cls_1_5 == qc) {
                        unsigned int lowb_5 = (u_1_5 ^ (unsigned int)((int)u_1_5 >> 31) & 511) & 511;
                        key_4 = 536870912 | lowb_5 << 9 | (unsigned int)(t0_5 + 5);
                    }
                    kb2[5] = __uint_as_float(key_4);
                    float sc_20_1 = __uint_as_float(cb2[6]);
                    sc_20_1 = fmaxf(sc_20_1, -1.7014118e+38f);
                    sc_20_1 = fminf(sc_20_1, 1.7014118e+38f);
                    unsigned int u_1_6 = __as_u32(sc_20_1);
                    unsigned int cls_1_6 = u_1_6 & 4294966784u;
                    int f_20 = 0;
                    if ((t0_5 + 6 < fb || lim_fe <= t0_5 + 6) && lim > t0_5 + 6) {
                        f_20 = 1;
                    }
                    if (f_20 != 0) {
                        cls_1_6 = 2139094528;
                    }
                    unsigned int key_5 = 0;
                    if ((qf < __uint_as_float(cls_1_6) || __uint_as_float(cls_1_6) == qf && cls_1_6 < qc) && cls_1_6 < 4278190080u) {
                        key_5 = 1073741824 | (unsigned int)(t0_5 + 6);
                    }
                    if (cls_1_6 == qc) {
                        unsigned int lowb_6 = (u_1_6 ^ (unsigned int)((int)u_1_6 >> 31) & 511) & 511;
                        key_5 = 536870912 | lowb_6 << 9 | (unsigned int)(t0_5 + 6);
                    }
                    kb2[6] = __uint_as_float(key_5);
                    float sc_21_1 = __uint_as_float(cb2[7]);
                    sc_21_1 = fmaxf(sc_21_1, -1.7014118e+38f);
                    sc_21_1 = fminf(sc_21_1, 1.7014118e+38f);
                    unsigned int u_1_7 = __as_u32(sc_21_1);
                    unsigned int cls_1_7 = u_1_7 & 4294966784u;
                    int f_21 = 0;
                    if ((t0_5 + 7 < fb || lim_fe <= t0_5 + 7) && lim > t0_5 + 7) {
                        f_21 = 1;
                    }
                    if (f_21 != 0) {
                        cls_1_7 = 2139094528;
                    }
                    unsigned int key_6 = 0;
                    if ((qf < __uint_as_float(cls_1_7) || __uint_as_float(cls_1_7) == qf && cls_1_7 < qc) && cls_1_7 < 4278190080u) {
                        key_6 = 1073741824 | (unsigned int)(t0_5 + 7);
                    }
                    if (cls_1_7 == qc) {
                        unsigned int lowb_7 = (u_1_7 ^ (unsigned int)((int)u_1_7 >> 31) & 511) & 511;
                        key_6 = 536870912 | lowb_7 << 9 | (unsigned int)(t0_5 + 7);
                    }
                    kb2[7] = __uint_as_float(key_6);
                    float sc_22_1 = __uint_as_float(cb2[8]);
                    sc_22_1 = fmaxf(sc_22_1, -1.7014118e+38f);
                    sc_22_1 = fminf(sc_22_1, 1.7014118e+38f);
                    unsigned int u_1_8 = __as_u32(sc_22_1);
                    unsigned int cls_1_8 = u_1_8 & 4294966784u;
                    int f_22 = 0;
                    if ((t0_5 + 8 < fb || lim_fe <= t0_5 + 8) && lim > t0_5 + 8) {
                        f_22 = 1;
                    }
                    if (f_22 != 0) {
                        cls_1_8 = 2139094528;
                    }
                    unsigned int key_7 = 0;
                    if ((qf < __uint_as_float(cls_1_8) || __uint_as_float(cls_1_8) == qf && cls_1_8 < qc) && cls_1_8 < 4278190080u) {
                        key_7 = 1073741824 | (unsigned int)(t0_5 + 8);
                    }
                    if (cls_1_8 == qc) {
                        unsigned int lowb_8 = (u_1_8 ^ (unsigned int)((int)u_1_8 >> 31) & 511) & 511;
                        key_7 = 536870912 | lowb_8 << 9 | (unsigned int)(t0_5 + 8);
                    }
                    kb2[8] = __uint_as_float(key_7);
                    float sc_23_1 = __uint_as_float(cb2[9]);
                    sc_23_1 = fmaxf(sc_23_1, -1.7014118e+38f);
                    sc_23_1 = fminf(sc_23_1, 1.7014118e+38f);
                    unsigned int u_1_9 = __as_u32(sc_23_1);
                    unsigned int cls_1_9 = u_1_9 & 4294966784u;
                    int f_23 = 0;
                    if ((t0_5 + 9 < fb || lim_fe <= t0_5 + 9) && lim > t0_5 + 9) {
                        f_23 = 1;
                    }
                    if (f_23 != 0) {
                        cls_1_9 = 2139094528;
                    }
                    unsigned int key_8 = 0;
                    if ((qf < __uint_as_float(cls_1_9) || __uint_as_float(cls_1_9) == qf && cls_1_9 < qc) && cls_1_9 < 4278190080u) {
                        key_8 = 1073741824 | (unsigned int)(t0_5 + 9);
                    }
                    if (cls_1_9 == qc) {
                        unsigned int lowb_9 = (u_1_9 ^ (unsigned int)((int)u_1_9 >> 31) & 511) & 511;
                        key_8 = 536870912 | lowb_9 << 9 | (unsigned int)(t0_5 + 9);
                    }
                    kb2[9] = __uint_as_float(key_8);
                    float sc_24_1 = __uint_as_float(cb2[10]);
                    sc_24_1 = fmaxf(sc_24_1, -1.7014118e+38f);
                    sc_24_1 = fminf(sc_24_1, 1.7014118e+38f);
                    unsigned int u_1_10 = __as_u32(sc_24_1);
                    unsigned int cls_1_10 = u_1_10 & 4294966784u;
                    int f_24 = 0;
                    if ((t0_5 + 10 < fb || lim_fe <= t0_5 + 10) && lim > t0_5 + 10) {
                        f_24 = 1;
                    }
                    if (f_24 != 0) {
                        cls_1_10 = 2139094528;
                    }
                    unsigned int key_9 = 0;
                    if ((qf < __uint_as_float(cls_1_10) || __uint_as_float(cls_1_10) == qf && cls_1_10 < qc) && cls_1_10 < 4278190080u) {
                        key_9 = 1073741824 | (unsigned int)(t0_5 + 10);
                    }
                    if (cls_1_10 == qc) {
                        unsigned int lowb_10 = (u_1_10 ^ (unsigned int)((int)u_1_10 >> 31) & 511) & 511;
                        key_9 = 536870912 | lowb_10 << 9 | (unsigned int)(t0_5 + 10);
                    }
                    kb2[10] = __uint_as_float(key_9);
                    float sc_25_1 = __uint_as_float(cb2[11]);
                    sc_25_1 = fmaxf(sc_25_1, -1.7014118e+38f);
                    sc_25_1 = fminf(sc_25_1, 1.7014118e+38f);
                    unsigned int u_1_11 = __as_u32(sc_25_1);
                    unsigned int cls_1_11 = u_1_11 & 4294966784u;
                    int f_25 = 0;
                    if ((t0_5 + 11 < fb || lim_fe <= t0_5 + 11) && lim > t0_5 + 11) {
                        f_25 = 1;
                    }
                    if (f_25 != 0) {
                        cls_1_11 = 2139094528;
                    }
                    unsigned int key_10 = 0;
                    if ((qf < __uint_as_float(cls_1_11) || __uint_as_float(cls_1_11) == qf && cls_1_11 < qc) && cls_1_11 < 4278190080u) {
                        key_10 = 1073741824 | (unsigned int)(t0_5 + 11);
                    }
                    if (cls_1_11 == qc) {
                        unsigned int lowb_11 = (u_1_11 ^ (unsigned int)((int)u_1_11 >> 31) & 511) & 511;
                        key_10 = 536870912 | lowb_11 << 9 | (unsigned int)(t0_5 + 11);
                    }
                    kb2[11] = __uint_as_float(key_10);
                    float sc_26_1 = __uint_as_float(cb2[12]);
                    sc_26_1 = fmaxf(sc_26_1, -1.7014118e+38f);
                    sc_26_1 = fminf(sc_26_1, 1.7014118e+38f);
                    unsigned int u_1_12 = __as_u32(sc_26_1);
                    unsigned int cls_1_12 = u_1_12 & 4294966784u;
                    int f_26 = 0;
                    if ((t0_5 + 12 < fb || lim_fe <= t0_5 + 12) && lim > t0_5 + 12) {
                        f_26 = 1;
                    }
                    if (f_26 != 0) {
                        cls_1_12 = 2139094528;
                    }
                    unsigned int key_11 = 0;
                    if ((qf < __uint_as_float(cls_1_12) || __uint_as_float(cls_1_12) == qf && cls_1_12 < qc) && cls_1_12 < 4278190080u) {
                        key_11 = 1073741824 | (unsigned int)(t0_5 + 12);
                    }
                    if (cls_1_12 == qc) {
                        unsigned int lowb_12 = (u_1_12 ^ (unsigned int)((int)u_1_12 >> 31) & 511) & 511;
                        key_11 = 536870912 | lowb_12 << 9 | (unsigned int)(t0_5 + 12);
                    }
                    kb2[12] = __uint_as_float(key_11);
                    float sc_27_1 = __uint_as_float(cb2[13]);
                    sc_27_1 = fmaxf(sc_27_1, -1.7014118e+38f);
                    sc_27_1 = fminf(sc_27_1, 1.7014118e+38f);
                    unsigned int u_1_13 = __as_u32(sc_27_1);
                    unsigned int cls_1_13 = u_1_13 & 4294966784u;
                    int f_27 = 0;
                    if ((t0_5 + 13 < fb || lim_fe <= t0_5 + 13) && lim > t0_5 + 13) {
                        f_27 = 1;
                    }
                    if (f_27 != 0) {
                        cls_1_13 = 2139094528;
                    }
                    unsigned int key_12 = 0;
                    if ((qf < __uint_as_float(cls_1_13) || __uint_as_float(cls_1_13) == qf && cls_1_13 < qc) && cls_1_13 < 4278190080u) {
                        key_12 = 1073741824 | (unsigned int)(t0_5 + 13);
                    }
                    if (cls_1_13 == qc) {
                        unsigned int lowb_13 = (u_1_13 ^ (unsigned int)((int)u_1_13 >> 31) & 511) & 511;
                        key_12 = 536870912 | lowb_13 << 9 | (unsigned int)(t0_5 + 13);
                    }
                    kb2[13] = __uint_as_float(key_12);
                    float sc_28_1 = __uint_as_float(cb2[14]);
                    sc_28_1 = fmaxf(sc_28_1, -1.7014118e+38f);
                    sc_28_1 = fminf(sc_28_1, 1.7014118e+38f);
                    unsigned int u_1_14 = __as_u32(sc_28_1);
                    unsigned int cls_1_14 = u_1_14 & 4294966784u;
                    int f_28 = 0;
                    if ((t0_5 + 14 < fb || lim_fe <= t0_5 + 14) && lim > t0_5 + 14) {
                        f_28 = 1;
                    }
                    if (f_28 != 0) {
                        cls_1_14 = 2139094528;
                    }
                    unsigned int key_13 = 0;
                    if ((qf < __uint_as_float(cls_1_14) || __uint_as_float(cls_1_14) == qf && cls_1_14 < qc) && cls_1_14 < 4278190080u) {
                        key_13 = 1073741824 | (unsigned int)(t0_5 + 14);
                    }
                    if (cls_1_14 == qc) {
                        unsigned int lowb_14 = (u_1_14 ^ (unsigned int)((int)u_1_14 >> 31) & 511) & 511;
                        key_13 = 536870912 | lowb_14 << 9 | (unsigned int)(t0_5 + 14);
                    }
                    kb2[14] = __uint_as_float(key_13);
                    float sc_29_1 = __uint_as_float(cb2[15]);
                    sc_29_1 = fmaxf(sc_29_1, -1.7014118e+38f);
                    sc_29_1 = fminf(sc_29_1, 1.7014118e+38f);
                    unsigned int u_1_15 = __as_u32(sc_29_1);
                    unsigned int cls_1_15 = u_1_15 & 4294966784u;
                    int f_29 = 0;
                    if ((t0_5 + 15 < fb || lim_fe <= t0_5 + 15) && lim > t0_5 + 15) {
                        f_29 = 1;
                    }
                    if (f_29 != 0) {
                        cls_1_15 = 2139094528;
                    }
                    unsigned int key_14 = 0;
                    if ((qf < __uint_as_float(cls_1_15) || __uint_as_float(cls_1_15) == qf && cls_1_15 < qc) && cls_1_15 < 4278190080u) {
                        key_14 = 1073741824 | (unsigned int)(t0_5 + 15);
                    }
                    if (cls_1_15 == qc) {
                        unsigned int lowb_15 = (u_1_15 ^ (unsigned int)((int)u_1_15 >> 31) & 511) & 511;
                        key_14 = 536870912 | lowb_15 << 9 | (unsigned int)(t0_5 + 15);
                    }
                    kb2[15] = __uint_as_float(key_14);
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
                    int _warp_vote_10 = __any_sync(0xFFFFFFFF, nzc != 0);
                    if (_warp_vote_10 != 0) {
                        float hi_1_60 = fmaxf(kb2[0], kb2[13]);
                        float lo_1_60 = fminf(kb2[0], kb2[13]);
                        kb2[0] = hi_1_60;
                        kb2[13] = lo_1_60;
                        float hi_1_61 = fmaxf(kb2[1], kb2[12]);
                        float lo_1_61 = fminf(kb2[1], kb2[12]);
                        kb2[1] = hi_1_61;
                        kb2[12] = lo_1_61;
                        float hi_1_62 = fmaxf(kb2[2], kb2[15]);
                        float lo_1_62 = fminf(kb2[2], kb2[15]);
                        kb2[2] = hi_1_62;
                        kb2[15] = lo_1_62;
                        float hi_1_63 = fmaxf(kb2[3], kb2[14]);
                        float lo_1_63 = fminf(kb2[3], kb2[14]);
                        kb2[3] = hi_1_63;
                        kb2[14] = lo_1_63;
                        float hi_1_64 = fmaxf(kb2[4], kb2[8]);
                        float lo_1_64 = fminf(kb2[4], kb2[8]);
                        kb2[4] = hi_1_64;
                        kb2[8] = lo_1_64;
                        float hi_1_65 = fmaxf(kb2[5], kb2[6]);
                        float lo_1_65 = fminf(kb2[5], kb2[6]);
                        kb2[5] = hi_1_65;
                        kb2[6] = lo_1_65;
                        float hi_1_66 = fmaxf(kb2[7], kb2[11]);
                        float lo_1_66 = fminf(kb2[7], kb2[11]);
                        kb2[7] = hi_1_66;
                        kb2[11] = lo_1_66;
                        float hi_1_67 = fmaxf(kb2[9], kb2[10]);
                        float lo_1_67 = fminf(kb2[9], kb2[10]);
                        kb2[9] = hi_1_67;
                        kb2[10] = lo_1_67;
                        float hi_1_68 = fmaxf(kb2[0], kb2[5]);
                        float lo_1_68 = fminf(kb2[0], kb2[5]);
                        kb2[0] = hi_1_68;
                        kb2[5] = lo_1_68;
                        float hi_1_69 = fmaxf(kb2[1], kb2[7]);
                        float lo_1_69 = fminf(kb2[1], kb2[7]);
                        kb2[1] = hi_1_69;
                        kb2[7] = lo_1_69;
                        float hi_1_70 = fmaxf(kb2[2], kb2[9]);
                        float lo_1_70 = fminf(kb2[2], kb2[9]);
                        kb2[2] = hi_1_70;
                        kb2[9] = lo_1_70;
                        float hi_1_71 = fmaxf(kb2[3], kb2[4]);
                        float lo_1_71 = fminf(kb2[3], kb2[4]);
                        kb2[3] = hi_1_71;
                        kb2[4] = lo_1_71;
                        float hi_1_72 = fmaxf(kb2[6], kb2[13]);
                        float lo_1_72 = fminf(kb2[6], kb2[13]);
                        kb2[6] = hi_1_72;
                        kb2[13] = lo_1_72;
                        float hi_1_73 = fmaxf(kb2[8], kb2[14]);
                        float lo_1_73 = fminf(kb2[8], kb2[14]);
                        kb2[8] = hi_1_73;
                        kb2[14] = lo_1_73;
                        float hi_1_74 = fmaxf(kb2[10], kb2[15]);
                        float lo_1_74 = fminf(kb2[10], kb2[15]);
                        kb2[10] = hi_1_74;
                        kb2[15] = lo_1_74;
                        float hi_1_75 = fmaxf(kb2[11], kb2[12]);
                        float lo_1_75 = fminf(kb2[11], kb2[12]);
                        kb2[11] = hi_1_75;
                        kb2[12] = lo_1_75;
                        float hi_1_76 = fmaxf(kb2[0], kb2[1]);
                        float lo_1_76 = fminf(kb2[0], kb2[1]);
                        kb2[0] = hi_1_76;
                        kb2[1] = lo_1_76;
                        float hi_1_77 = fmaxf(kb2[2], kb2[3]);
                        float lo_1_77 = fminf(kb2[2], kb2[3]);
                        kb2[2] = hi_1_77;
                        kb2[3] = lo_1_77;
                        float hi_1_78 = fmaxf(kb2[4], kb2[5]);
                        float lo_1_78 = fminf(kb2[4], kb2[5]);
                        kb2[4] = hi_1_78;
                        kb2[5] = lo_1_78;
                        float hi_1_79 = fmaxf(kb2[6], kb2[8]);
                        float lo_1_79 = fminf(kb2[6], kb2[8]);
                        kb2[6] = hi_1_79;
                        kb2[8] = lo_1_79;
                        float hi_1_80 = fmaxf(kb2[7], kb2[9]);
                        float lo_1_80 = fminf(kb2[7], kb2[9]);
                        kb2[7] = hi_1_80;
                        kb2[9] = lo_1_80;
                        float hi_1_81 = fmaxf(kb2[10], kb2[11]);
                        float lo_1_81 = fminf(kb2[10], kb2[11]);
                        kb2[10] = hi_1_81;
                        kb2[11] = lo_1_81;
                        float hi_1_82 = fmaxf(kb2[12], kb2[13]);
                        float lo_1_82 = fminf(kb2[12], kb2[13]);
                        kb2[12] = hi_1_82;
                        kb2[13] = lo_1_82;
                        float hi_1_83 = fmaxf(kb2[14], kb2[15]);
                        float lo_1_83 = fminf(kb2[14], kb2[15]);
                        kb2[14] = hi_1_83;
                        kb2[15] = lo_1_83;
                        float hi_1_84 = fmaxf(kb2[0], kb2[2]);
                        float lo_1_84 = fminf(kb2[0], kb2[2]);
                        kb2[0] = hi_1_84;
                        kb2[2] = lo_1_84;
                        float hi_1_85 = fmaxf(kb2[1], kb2[3]);
                        float lo_1_85 = fminf(kb2[1], kb2[3]);
                        kb2[1] = hi_1_85;
                        kb2[3] = lo_1_85;
                        float hi_1_86 = fmaxf(kb2[4], kb2[10]);
                        float lo_1_86 = fminf(kb2[4], kb2[10]);
                        kb2[4] = hi_1_86;
                        kb2[10] = lo_1_86;
                        float hi_1_87 = fmaxf(kb2[5], kb2[11]);
                        float lo_1_87 = fminf(kb2[5], kb2[11]);
                        kb2[5] = hi_1_87;
                        kb2[11] = lo_1_87;
                        float hi_1_88 = fmaxf(kb2[6], kb2[7]);
                        float lo_1_88 = fminf(kb2[6], kb2[7]);
                        kb2[6] = hi_1_88;
                        kb2[7] = lo_1_88;
                        float hi_1_89 = fmaxf(kb2[8], kb2[9]);
                        float lo_1_89 = fminf(kb2[8], kb2[9]);
                        kb2[8] = hi_1_89;
                        kb2[9] = lo_1_89;
                        float hi_1_90 = fmaxf(kb2[12], kb2[14]);
                        float lo_1_90 = fminf(kb2[12], kb2[14]);
                        kb2[12] = hi_1_90;
                        kb2[14] = lo_1_90;
                        float hi_1_91 = fmaxf(kb2[13], kb2[15]);
                        float lo_1_91 = fminf(kb2[13], kb2[15]);
                        kb2[13] = hi_1_91;
                        kb2[15] = lo_1_91;
                        float hi_1_92 = fmaxf(kb2[1], kb2[2]);
                        float lo_1_92 = fminf(kb2[1], kb2[2]);
                        kb2[1] = hi_1_92;
                        kb2[2] = lo_1_92;
                        float hi_1_93 = fmaxf(kb2[3], kb2[12]);
                        float lo_1_93 = fminf(kb2[3], kb2[12]);
                        kb2[3] = hi_1_93;
                        kb2[12] = lo_1_93;
                        float hi_1_94 = fmaxf(kb2[4], kb2[6]);
                        float lo_1_94 = fminf(kb2[4], kb2[6]);
                        kb2[4] = hi_1_94;
                        kb2[6] = lo_1_94;
                        float hi_1_95 = fmaxf(kb2[5], kb2[7]);
                        float lo_1_95 = fminf(kb2[5], kb2[7]);
                        kb2[5] = hi_1_95;
                        kb2[7] = lo_1_95;
                        float hi_1_96 = fmaxf(kb2[8], kb2[10]);
                        float lo_1_96 = fminf(kb2[8], kb2[10]);
                        kb2[8] = hi_1_96;
                        kb2[10] = lo_1_96;
                        float hi_1_97 = fmaxf(kb2[9], kb2[11]);
                        float lo_1_97 = fminf(kb2[9], kb2[11]);
                        kb2[9] = hi_1_97;
                        kb2[11] = lo_1_97;
                        float hi_1_98 = fmaxf(kb2[13], kb2[14]);
                        float lo_1_98 = fminf(kb2[13], kb2[14]);
                        kb2[13] = hi_1_98;
                        kb2[14] = lo_1_98;
                        float hi_1_99 = fmaxf(kb2[1], kb2[4]);
                        float lo_1_99 = fminf(kb2[1], kb2[4]);
                        kb2[1] = hi_1_99;
                        kb2[4] = lo_1_99;
                        float hi_1_100 = fmaxf(kb2[2], kb2[6]);
                        float lo_1_100 = fminf(kb2[2], kb2[6]);
                        kb2[2] = hi_1_100;
                        kb2[6] = lo_1_100;
                        float hi_1_101 = fmaxf(kb2[5], kb2[8]);
                        float lo_1_101 = fminf(kb2[5], kb2[8]);
                        kb2[5] = hi_1_101;
                        kb2[8] = lo_1_101;
                        float hi_1_102 = fmaxf(kb2[7], kb2[10]);
                        float lo_1_102 = fminf(kb2[7], kb2[10]);
                        kb2[7] = hi_1_102;
                        kb2[10] = lo_1_102;
                        float hi_1_103 = fmaxf(kb2[9], kb2[13]);
                        float lo_1_103 = fminf(kb2[9], kb2[13]);
                        kb2[9] = hi_1_103;
                        kb2[13] = lo_1_103;
                        float hi_1_104 = fmaxf(kb2[11], kb2[14]);
                        float lo_1_104 = fminf(kb2[11], kb2[14]);
                        kb2[11] = hi_1_104;
                        kb2[14] = lo_1_104;
                        float hi_1_105 = fmaxf(kb2[2], kb2[4]);
                        float lo_1_105 = fminf(kb2[2], kb2[4]);
                        kb2[2] = hi_1_105;
                        kb2[4] = lo_1_105;
                        float hi_1_106 = fmaxf(kb2[3], kb2[6]);
                        float lo_1_106 = fminf(kb2[3], kb2[6]);
                        kb2[3] = hi_1_106;
                        kb2[6] = lo_1_106;
                        float hi_1_107 = fmaxf(kb2[9], kb2[12]);
                        float lo_1_107 = fminf(kb2[9], kb2[12]);
                        kb2[9] = hi_1_107;
                        kb2[12] = lo_1_107;
                        float hi_1_108 = fmaxf(kb2[11], kb2[13]);
                        float lo_1_108 = fminf(kb2[11], kb2[13]);
                        kb2[11] = hi_1_108;
                        kb2[13] = lo_1_108;
                        float hi_1_109 = fmaxf(kb2[3], kb2[5]);
                        float lo_1_109 = fminf(kb2[3], kb2[5]);
                        kb2[3] = hi_1_109;
                        kb2[5] = lo_1_109;
                        float hi_1_110 = fmaxf(kb2[6], kb2[8]);
                        float lo_1_110 = fminf(kb2[6], kb2[8]);
                        kb2[6] = hi_1_110;
                        kb2[8] = lo_1_110;
                        float hi_1_111 = fmaxf(kb2[7], kb2[9]);
                        float lo_1_111 = fminf(kb2[7], kb2[9]);
                        kb2[7] = hi_1_111;
                        kb2[9] = lo_1_111;
                        float hi_1_112 = fmaxf(kb2[10], kb2[12]);
                        float lo_1_112 = fminf(kb2[10], kb2[12]);
                        kb2[10] = hi_1_112;
                        kb2[12] = lo_1_112;
                        float hi_1_113 = fmaxf(kb2[3], kb2[4]);
                        float lo_1_113 = fminf(kb2[3], kb2[4]);
                        kb2[3] = hi_1_113;
                        kb2[4] = lo_1_113;
                        float hi_1_114 = fmaxf(kb2[5], kb2[6]);
                        float lo_1_114 = fminf(kb2[5], kb2[6]);
                        kb2[5] = hi_1_114;
                        kb2[6] = lo_1_114;
                        float hi_1_115 = fmaxf(kb2[7], kb2[8]);
                        float lo_1_115 = fminf(kb2[7], kb2[8]);
                        kb2[7] = hi_1_115;
                        kb2[8] = lo_1_115;
                        float hi_1_116 = fmaxf(kb2[9], kb2[10]);
                        float lo_1_116 = fminf(kb2[9], kb2[10]);
                        kb2[9] = hi_1_116;
                        kb2[10] = lo_1_116;
                        float hi_1_117 = fmaxf(kb2[11], kb2[12]);
                        float lo_1_117 = fminf(kb2[11], kb2[12]);
                        kb2[11] = hi_1_117;
                        kb2[12] = lo_1_117;
                        float hi_1_118 = fmaxf(kb2[6], kb2[7]);
                        float lo_1_118 = fminf(kb2[6], kb2[7]);
                        kb2[6] = hi_1_118;
                        kb2[7] = lo_1_118;
                        float hi_1_119 = fmaxf(kb2[8], kb2[9]);
                        float lo_1_119 = fminf(kb2[8], kb2[9]);
                        kb2[8] = hi_1_119;
                        kb2[9] = lo_1_119;
                        float hi_108 = fmaxf(a2x[0], kb2[15]);
                        a2x[0] = hi_108;
                        float hi_109 = fmaxf(a2x[1], kb2[14]);
                        a2x[1] = hi_109;
                        float hi_110 = fmaxf(a2x[2], kb2[13]);
                        a2x[2] = hi_110;
                        float hi_111 = fmaxf(a2x[3], kb2[12]);
                        a2x[3] = hi_111;
                        float hi_112 = fmaxf(a2x[4], kb2[11]);
                        a2x[4] = hi_112;
                        float hi_113 = fmaxf(a2x[5], kb2[10]);
                        a2x[5] = hi_113;
                        float hi_114 = fmaxf(a2x[6], kb2[9]);
                        a2x[6] = hi_114;
                        float hi_115 = fmaxf(a2x[7], kb2[8]);
                        a2x[7] = hi_115;
                        float hi_116 = fmaxf(a2x[8], kb2[7]);
                        a2x[8] = hi_116;
                        float hi_117 = fmaxf(a2x[9], kb2[6]);
                        a2x[9] = hi_117;
                        float hi_118 = fmaxf(a2x[10], kb2[5]);
                        a2x[10] = hi_118;
                        float hi_119 = fmaxf(a2x[11], kb2[4]);
                        a2x[11] = hi_119;
                        float hi_120 = fmaxf(a2x[12], kb2[3]);
                        a2x[12] = hi_120;
                        float hi_121 = fmaxf(a2x[13], kb2[2]);
                        a2x[13] = hi_121;
                        float hi_122 = fmaxf(a2x[14], kb2[1]);
                        a2x[14] = hi_122;
                        float hi_123 = fmaxf(a2x[15], kb2[0]);
                        a2x[15] = hi_123;
                        float hi_124 = fmaxf(a2x[0], a2x[8]);
                        float lo_108 = fminf(a2x[0], a2x[8]);
                        a2x[0] = hi_124;
                        a2x[8] = lo_108;
                        float hi_125 = fmaxf(a2x[1], a2x[9]);
                        float lo_109 = fminf(a2x[1], a2x[9]);
                        a2x[1] = hi_125;
                        a2x[9] = lo_109;
                        float hi_126 = fmaxf(a2x[2], a2x[10]);
                        float lo_110 = fminf(a2x[2], a2x[10]);
                        a2x[2] = hi_126;
                        a2x[10] = lo_110;
                        float hi_127 = fmaxf(a2x[3], a2x[11]);
                        float lo_111 = fminf(a2x[3], a2x[11]);
                        a2x[3] = hi_127;
                        a2x[11] = lo_111;
                        float hi_128 = fmaxf(a2x[4], a2x[12]);
                        float lo_112 = fminf(a2x[4], a2x[12]);
                        a2x[4] = hi_128;
                        a2x[12] = lo_112;
                        float hi_129 = fmaxf(a2x[5], a2x[13]);
                        float lo_113 = fminf(a2x[5], a2x[13]);
                        a2x[5] = hi_129;
                        a2x[13] = lo_113;
                        float hi_130 = fmaxf(a2x[6], a2x[14]);
                        float lo_114 = fminf(a2x[6], a2x[14]);
                        a2x[6] = hi_130;
                        a2x[14] = lo_114;
                        float hi_131 = fmaxf(a2x[7], a2x[15]);
                        float lo_115 = fminf(a2x[7], a2x[15]);
                        a2x[7] = hi_131;
                        a2x[15] = lo_115;
                        float hi_132 = fmaxf(a2x[0], a2x[4]);
                        float lo_116 = fminf(a2x[0], a2x[4]);
                        a2x[0] = hi_132;
                        a2x[4] = lo_116;
                        float hi_133 = fmaxf(a2x[1], a2x[5]);
                        float lo_117 = fminf(a2x[1], a2x[5]);
                        a2x[1] = hi_133;
                        a2x[5] = lo_117;
                        float hi_134 = fmaxf(a2x[2], a2x[6]);
                        float lo_118 = fminf(a2x[2], a2x[6]);
                        a2x[2] = hi_134;
                        a2x[6] = lo_118;
                        float hi_135 = fmaxf(a2x[3], a2x[7]);
                        float lo_119 = fminf(a2x[3], a2x[7]);
                        a2x[3] = hi_135;
                        a2x[7] = lo_119;
                        float hi_136 = fmaxf(a2x[8], a2x[12]);
                        float lo_120 = fminf(a2x[8], a2x[12]);
                        a2x[8] = hi_136;
                        a2x[12] = lo_120;
                        float hi_137 = fmaxf(a2x[9], a2x[13]);
                        float lo_121 = fminf(a2x[9], a2x[13]);
                        a2x[9] = hi_137;
                        a2x[13] = lo_121;
                        float hi_138 = fmaxf(a2x[10], a2x[14]);
                        float lo_122 = fminf(a2x[10], a2x[14]);
                        a2x[10] = hi_138;
                        a2x[14] = lo_122;
                        float hi_139 = fmaxf(a2x[11], a2x[15]);
                        float lo_123 = fminf(a2x[11], a2x[15]);
                        a2x[11] = hi_139;
                        a2x[15] = lo_123;
                        float hi_140 = fmaxf(a2x[0], a2x[2]);
                        float lo_124 = fminf(a2x[0], a2x[2]);
                        a2x[0] = hi_140;
                        a2x[2] = lo_124;
                        float hi_141 = fmaxf(a2x[1], a2x[3]);
                        float lo_125 = fminf(a2x[1], a2x[3]);
                        a2x[1] = hi_141;
                        a2x[3] = lo_125;
                        float hi_142 = fmaxf(a2x[4], a2x[6]);
                        float lo_126 = fminf(a2x[4], a2x[6]);
                        a2x[4] = hi_142;
                        a2x[6] = lo_126;
                        float hi_143 = fmaxf(a2x[5], a2x[7]);
                        float lo_127 = fminf(a2x[5], a2x[7]);
                        a2x[5] = hi_143;
                        a2x[7] = lo_127;
                        float hi_144 = fmaxf(a2x[8], a2x[10]);
                        float lo_128 = fminf(a2x[8], a2x[10]);
                        a2x[8] = hi_144;
                        a2x[10] = lo_128;
                        float hi_145 = fmaxf(a2x[9], a2x[11]);
                        float lo_129 = fminf(a2x[9], a2x[11]);
                        a2x[9] = hi_145;
                        a2x[11] = lo_129;
                        float hi_146 = fmaxf(a2x[12], a2x[14]);
                        float lo_130 = fminf(a2x[12], a2x[14]);
                        a2x[12] = hi_146;
                        a2x[14] = lo_130;
                        float hi_147 = fmaxf(a2x[13], a2x[15]);
                        float lo_131 = fminf(a2x[13], a2x[15]);
                        a2x[13] = hi_147;
                        a2x[15] = lo_131;
                        float hi_148 = fmaxf(a2x[0], a2x[1]);
                        float lo_132 = fminf(a2x[0], a2x[1]);
                        a2x[0] = hi_148;
                        a2x[1] = lo_132;
                        float hi_149 = fmaxf(a2x[2], a2x[3]);
                        float lo_133 = fminf(a2x[2], a2x[3]);
                        a2x[2] = hi_149;
                        a2x[3] = lo_133;
                        float hi_150 = fmaxf(a2x[4], a2x[5]);
                        float lo_134 = fminf(a2x[4], a2x[5]);
                        a2x[4] = hi_150;
                        a2x[5] = lo_134;
                        float hi_151 = fmaxf(a2x[6], a2x[7]);
                        float lo_135 = fminf(a2x[6], a2x[7]);
                        a2x[6] = hi_151;
                        a2x[7] = lo_135;
                        float hi_152 = fmaxf(a2x[8], a2x[9]);
                        float lo_136 = fminf(a2x[8], a2x[9]);
                        a2x[8] = hi_152;
                        a2x[9] = lo_136;
                        float hi_153 = fmaxf(a2x[10], a2x[11]);
                        float lo_137 = fminf(a2x[10], a2x[11]);
                        a2x[10] = hi_153;
                        a2x[11] = lo_137;
                        float hi_154 = fmaxf(a2x[12], a2x[13]);
                        float lo_138 = fminf(a2x[12], a2x[13]);
                        a2x[12] = hi_154;
                        a2x[13] = lo_138;
                        float hi_155 = fmaxf(a2x[14], a2x[15]);
                        float lo_139 = fminf(a2x[14], a2x[15]);
                        a2x[14] = hi_155;
                        a2x[15] = lo_139;
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
            asm volatile("barrier.sync 1, 32;" ::: "memory");
            float r_0 = neg_inf;
            float V_0[8];
            int s0_8 = (sg * 16 + cg) * 17;
            int s1_8 = (sg * 16 + 8 + cg) * 17;
            float x0_8 = pub[s0_8 + ln];
            float y0_8 = pub[s1_8 + lnr];
            float lo0_8 = fminf(x0_8, y0_8);
            r_0 = fmaxf(r_0, lo0_8);
            float hi0_8 = fmaxf(x0_8, y0_8);
            float cur_14 = hi0_8;
            float pv_60 = __shfl_xor_sync(0xFFFFFFFF, cur_14, 8);
            float hi_156 = fmaxf(cur_14, pv_60);
            float lo_140 = fminf(cur_14, pv_60);
            cur_14 = ((up[0] != 0) ? hi_156 : lo_140);
            float pv_61 = __shfl_xor_sync(0xFFFFFFFF, cur_14, 4);
            float hi_157 = fmaxf(cur_14, pv_61);
            float lo_141 = fminf(cur_14, pv_61);
            cur_14 = ((up[1] != 0) ? hi_157 : lo_141);
            float pv_62 = __shfl_xor_sync(0xFFFFFFFF, cur_14, 2);
            float hi_158 = fmaxf(cur_14, pv_62);
            float lo_142 = fminf(cur_14, pv_62);
            cur_14 = ((up[2] != 0) ? hi_158 : lo_142);
            float pv_63 = __shfl_xor_sync(0xFFFFFFFF, cur_14, 1);
            float hi_159 = fmaxf(cur_14, pv_63);
            float lo_143 = fminf(cur_14, pv_63);
            cur_14 = ((up[3] != 0) ? hi_159 : lo_143);
            V_0[0] = cur_14;
            int s0_9 = (sg * 16 + 1 + cg) * 17;
            int s1_9 = (sg * 16 + 1 + 8 + cg) * 17;
            float x0_9 = pub[s0_9 + ln];
            float y0_9 = pub[s1_9 + lnr];
            float lo0_9 = fminf(x0_9, y0_9);
            r_0 = fmaxf(r_0, lo0_9);
            float hi0_9 = fmaxf(x0_9, y0_9);
            float cur_15 = hi0_9;
            float pv_64 = __shfl_xor_sync(0xFFFFFFFF, cur_15, 8);
            float hi_160 = fmaxf(cur_15, pv_64);
            float lo_144 = fminf(cur_15, pv_64);
            cur_15 = ((up[0] != 0) ? hi_160 : lo_144);
            float pv_65 = __shfl_xor_sync(0xFFFFFFFF, cur_15, 4);
            float hi_161 = fmaxf(cur_15, pv_65);
            float lo_145 = fminf(cur_15, pv_65);
            cur_15 = ((up[1] != 0) ? hi_161 : lo_145);
            float pv_66 = __shfl_xor_sync(0xFFFFFFFF, cur_15, 2);
            float hi_162 = fmaxf(cur_15, pv_66);
            float lo_146 = fminf(cur_15, pv_66);
            cur_15 = ((up[2] != 0) ? hi_162 : lo_146);
            float pv_67 = __shfl_xor_sync(0xFFFFFFFF, cur_15, 1);
            float hi_163 = fmaxf(cur_15, pv_67);
            float lo_147 = fminf(cur_15, pv_67);
            cur_15 = ((up[3] != 0) ? hi_163 : lo_147);
            V_0[1] = cur_15;
            int s0_10 = (sg * 16 + 2 + cg) * 17;
            int s1_10 = (sg * 16 + 2 + 8 + cg) * 17;
            float x0_10 = pub[s0_10 + ln];
            float y0_10 = pub[s1_10 + lnr];
            float lo0_10 = fminf(x0_10, y0_10);
            r_0 = fmaxf(r_0, lo0_10);
            float hi0_10 = fmaxf(x0_10, y0_10);
            float cur_16 = hi0_10;
            float pv_68 = __shfl_xor_sync(0xFFFFFFFF, cur_16, 8);
            float hi_164 = fmaxf(cur_16, pv_68);
            float lo_148 = fminf(cur_16, pv_68);
            cur_16 = ((up[0] != 0) ? hi_164 : lo_148);
            float pv_69 = __shfl_xor_sync(0xFFFFFFFF, cur_16, 4);
            float hi_165 = fmaxf(cur_16, pv_69);
            float lo_149 = fminf(cur_16, pv_69);
            cur_16 = ((up[1] != 0) ? hi_165 : lo_149);
            float pv_70 = __shfl_xor_sync(0xFFFFFFFF, cur_16, 2);
            float hi_166 = fmaxf(cur_16, pv_70);
            float lo_150 = fminf(cur_16, pv_70);
            cur_16 = ((up[2] != 0) ? hi_166 : lo_150);
            float pv_71 = __shfl_xor_sync(0xFFFFFFFF, cur_16, 1);
            float hi_167 = fmaxf(cur_16, pv_71);
            float lo_151 = fminf(cur_16, pv_71);
            cur_16 = ((up[3] != 0) ? hi_167 : lo_151);
            V_0[2] = cur_16;
            int s0_11 = (sg * 16 + 3 + cg) * 17;
            int s1_11 = (sg * 16 + 3 + 8 + cg) * 17;
            float x0_11 = pub[s0_11 + ln];
            float y0_11 = pub[s1_11 + lnr];
            float lo0_11 = fminf(x0_11, y0_11);
            r_0 = fmaxf(r_0, lo0_11);
            float hi0_11 = fmaxf(x0_11, y0_11);
            float cur_17 = hi0_11;
            float pv_72 = __shfl_xor_sync(0xFFFFFFFF, cur_17, 8);
            float hi_168 = fmaxf(cur_17, pv_72);
            float lo_152 = fminf(cur_17, pv_72);
            cur_17 = ((up[0] != 0) ? hi_168 : lo_152);
            float pv_73 = __shfl_xor_sync(0xFFFFFFFF, cur_17, 4);
            float hi_169 = fmaxf(cur_17, pv_73);
            float lo_153 = fminf(cur_17, pv_73);
            cur_17 = ((up[1] != 0) ? hi_169 : lo_153);
            float pv_74 = __shfl_xor_sync(0xFFFFFFFF, cur_17, 2);
            float hi_170 = fmaxf(cur_17, pv_74);
            float lo_154 = fminf(cur_17, pv_74);
            cur_17 = ((up[2] != 0) ? hi_170 : lo_154);
            float pv_75 = __shfl_xor_sync(0xFFFFFFFF, cur_17, 1);
            float hi_171 = fmaxf(cur_17, pv_75);
            float lo_155 = fminf(cur_17, pv_75);
            cur_17 = ((up[3] != 0) ? hi_171 : lo_155);
            V_0[3] = cur_17;
            int s0_12 = (sg * 16 + 4 + cg) * 17;
            int s1_12 = (sg * 16 + 4 + 8 + cg) * 17;
            float x0_12 = pub[s0_12 + ln];
            float y0_12 = pub[s1_12 + lnr];
            float lo0_12 = fminf(x0_12, y0_12);
            r_0 = fmaxf(r_0, lo0_12);
            float hi0_12 = fmaxf(x0_12, y0_12);
            float cur_18 = hi0_12;
            float pv_76 = __shfl_xor_sync(0xFFFFFFFF, cur_18, 8);
            float hi_172 = fmaxf(cur_18, pv_76);
            float lo_156 = fminf(cur_18, pv_76);
            cur_18 = ((up[0] != 0) ? hi_172 : lo_156);
            float pv_77 = __shfl_xor_sync(0xFFFFFFFF, cur_18, 4);
            float hi_173 = fmaxf(cur_18, pv_77);
            float lo_157 = fminf(cur_18, pv_77);
            cur_18 = ((up[1] != 0) ? hi_173 : lo_157);
            float pv_78 = __shfl_xor_sync(0xFFFFFFFF, cur_18, 2);
            float hi_174 = fmaxf(cur_18, pv_78);
            float lo_158 = fminf(cur_18, pv_78);
            cur_18 = ((up[2] != 0) ? hi_174 : lo_158);
            float pv_79 = __shfl_xor_sync(0xFFFFFFFF, cur_18, 1);
            float hi_175 = fmaxf(cur_18, pv_79);
            float lo_159 = fminf(cur_18, pv_79);
            cur_18 = ((up[3] != 0) ? hi_175 : lo_159);
            V_0[4] = cur_18;
            int s0_13 = (sg * 16 + 5 + cg) * 17;
            int s1_13 = (sg * 16 + 5 + 8 + cg) * 17;
            float x0_13 = pub[s0_13 + ln];
            float y0_13 = pub[s1_13 + lnr];
            float lo0_13 = fminf(x0_13, y0_13);
            r_0 = fmaxf(r_0, lo0_13);
            float hi0_13 = fmaxf(x0_13, y0_13);
            float cur_19 = hi0_13;
            float pv_80 = __shfl_xor_sync(0xFFFFFFFF, cur_19, 8);
            float hi_176 = fmaxf(cur_19, pv_80);
            float lo_160 = fminf(cur_19, pv_80);
            cur_19 = ((up[0] != 0) ? hi_176 : lo_160);
            float pv_81 = __shfl_xor_sync(0xFFFFFFFF, cur_19, 4);
            float hi_177 = fmaxf(cur_19, pv_81);
            float lo_161 = fminf(cur_19, pv_81);
            cur_19 = ((up[1] != 0) ? hi_177 : lo_161);
            float pv_82 = __shfl_xor_sync(0xFFFFFFFF, cur_19, 2);
            float hi_178 = fmaxf(cur_19, pv_82);
            float lo_162 = fminf(cur_19, pv_82);
            cur_19 = ((up[2] != 0) ? hi_178 : lo_162);
            float pv_83 = __shfl_xor_sync(0xFFFFFFFF, cur_19, 1);
            float hi_179 = fmaxf(cur_19, pv_83);
            float lo_163 = fminf(cur_19, pv_83);
            cur_19 = ((up[3] != 0) ? hi_179 : lo_163);
            V_0[5] = cur_19;
            int s0_14 = (sg * 16 + 6 + cg) * 17;
            int s1_14 = (sg * 16 + 6 + 8 + cg) * 17;
            float x0_14 = pub[s0_14 + ln];
            float y0_14 = pub[s1_14 + lnr];
            float lo0_14 = fminf(x0_14, y0_14);
            r_0 = fmaxf(r_0, lo0_14);
            float hi0_14 = fmaxf(x0_14, y0_14);
            float cur_20 = hi0_14;
            float pv_84 = __shfl_xor_sync(0xFFFFFFFF, cur_20, 8);
            float hi_180 = fmaxf(cur_20, pv_84);
            float lo_164 = fminf(cur_20, pv_84);
            cur_20 = ((up[0] != 0) ? hi_180 : lo_164);
            float pv_85 = __shfl_xor_sync(0xFFFFFFFF, cur_20, 4);
            float hi_181 = fmaxf(cur_20, pv_85);
            float lo_165 = fminf(cur_20, pv_85);
            cur_20 = ((up[1] != 0) ? hi_181 : lo_165);
            float pv_86 = __shfl_xor_sync(0xFFFFFFFF, cur_20, 2);
            float hi_182 = fmaxf(cur_20, pv_86);
            float lo_166 = fminf(cur_20, pv_86);
            cur_20 = ((up[2] != 0) ? hi_182 : lo_166);
            float pv_87 = __shfl_xor_sync(0xFFFFFFFF, cur_20, 1);
            float hi_183 = fmaxf(cur_20, pv_87);
            float lo_167 = fminf(cur_20, pv_87);
            cur_20 = ((up[3] != 0) ? hi_183 : lo_167);
            V_0[6] = cur_20;
            int s0_15 = (sg * 16 + 7 + cg) * 17;
            int s1_15 = (sg * 16 + 7 + 8 + cg) * 17;
            float x0_15 = pub[s0_15 + ln];
            float y0_15 = pub[s1_15 + lnr];
            float lo0_15 = fminf(x0_15, y0_15);
            r_0 = fmaxf(r_0, lo0_15);
            float hi0_15 = fmaxf(x0_15, y0_15);
            float cur_21 = hi0_15;
            float pv_88 = __shfl_xor_sync(0xFFFFFFFF, cur_21, 8);
            float hi_184 = fmaxf(cur_21, pv_88);
            float lo_168 = fminf(cur_21, pv_88);
            cur_21 = ((up[0] != 0) ? hi_184 : lo_168);
            float pv_89 = __shfl_xor_sync(0xFFFFFFFF, cur_21, 4);
            float hi_185 = fmaxf(cur_21, pv_89);
            float lo_169 = fminf(cur_21, pv_89);
            cur_21 = ((up[1] != 0) ? hi_185 : lo_169);
            float pv_90 = __shfl_xor_sync(0xFFFFFFFF, cur_21, 2);
            float hi_186 = fmaxf(cur_21, pv_90);
            float lo_170 = fminf(cur_21, pv_90);
            cur_21 = ((up[2] != 0) ? hi_186 : lo_170);
            float pv_91 = __shfl_xor_sync(0xFFFFFFFF, cur_21, 1);
            float hi_187 = fmaxf(cur_21, pv_91);
            float lo_171 = fminf(cur_21, pv_91);
            cur_21 = ((up[3] != 0) ? hi_187 : lo_171);
            V_0[7] = cur_21;
            float rs_0 = pub[(sg * 16 + ln + cg) * 17 + 16];
            r_0 = fmaxf(r_0, rs_0);
            float y1_7 = __shfl_xor_sync(0xFFFFFFFF, V_0[4], 15);
            float lo1_7 = fminf(V_0[0], y1_7);
            r_0 = fmaxf(r_0, lo1_7);
            float hi1_7 = fmaxf(V_0[0], y1_7);
            float cur_22 = hi1_7;
            float pv_92 = __shfl_xor_sync(0xFFFFFFFF, cur_22, 8);
            float hi_188 = fmaxf(cur_22, pv_92);
            float lo_172 = fminf(cur_22, pv_92);
            cur_22 = ((up[0] != 0) ? hi_188 : lo_172);
            float pv_93 = __shfl_xor_sync(0xFFFFFFFF, cur_22, 4);
            float hi_189 = fmaxf(cur_22, pv_93);
            float lo_173 = fminf(cur_22, pv_93);
            cur_22 = ((up[1] != 0) ? hi_189 : lo_173);
            float pv_94 = __shfl_xor_sync(0xFFFFFFFF, cur_22, 2);
            float hi_190 = fmaxf(cur_22, pv_94);
            float lo_174 = fminf(cur_22, pv_94);
            cur_22 = ((up[2] != 0) ? hi_190 : lo_174);
            float pv_95 = __shfl_xor_sync(0xFFFFFFFF, cur_22, 1);
            float hi_191 = fmaxf(cur_22, pv_95);
            float lo_175 = fminf(cur_22, pv_95);
            cur_22 = ((up[3] != 0) ? hi_191 : lo_175);
            V_0[0] = cur_22;
            float y1_8 = __shfl_xor_sync(0xFFFFFFFF, V_0[5], 15);
            float lo1_8 = fminf(V_0[1], y1_8);
            r_0 = fmaxf(r_0, lo1_8);
            float hi1_8 = fmaxf(V_0[1], y1_8);
            float cur_23 = hi1_8;
            float pv_96 = __shfl_xor_sync(0xFFFFFFFF, cur_23, 8);
            float hi_192 = fmaxf(cur_23, pv_96);
            float lo_176 = fminf(cur_23, pv_96);
            cur_23 = ((up[0] != 0) ? hi_192 : lo_176);
            float pv_97 = __shfl_xor_sync(0xFFFFFFFF, cur_23, 4);
            float hi_193 = fmaxf(cur_23, pv_97);
            float lo_177 = fminf(cur_23, pv_97);
            cur_23 = ((up[1] != 0) ? hi_193 : lo_177);
            float pv_98 = __shfl_xor_sync(0xFFFFFFFF, cur_23, 2);
            float hi_194 = fmaxf(cur_23, pv_98);
            float lo_178 = fminf(cur_23, pv_98);
            cur_23 = ((up[2] != 0) ? hi_194 : lo_178);
            float pv_99 = __shfl_xor_sync(0xFFFFFFFF, cur_23, 1);
            float hi_195 = fmaxf(cur_23, pv_99);
            float lo_179 = fminf(cur_23, pv_99);
            cur_23 = ((up[3] != 0) ? hi_195 : lo_179);
            V_0[1] = cur_23;
            float y1_9 = __shfl_xor_sync(0xFFFFFFFF, V_0[6], 15);
            float lo1_9 = fminf(V_0[2], y1_9);
            r_0 = fmaxf(r_0, lo1_9);
            float hi1_9 = fmaxf(V_0[2], y1_9);
            float cur_24 = hi1_9;
            float pv_100 = __shfl_xor_sync(0xFFFFFFFF, cur_24, 8);
            float hi_196 = fmaxf(cur_24, pv_100);
            float lo_180 = fminf(cur_24, pv_100);
            cur_24 = ((up[0] != 0) ? hi_196 : lo_180);
            float pv_101 = __shfl_xor_sync(0xFFFFFFFF, cur_24, 4);
            float hi_197 = fmaxf(cur_24, pv_101);
            float lo_181 = fminf(cur_24, pv_101);
            cur_24 = ((up[1] != 0) ? hi_197 : lo_181);
            float pv_102 = __shfl_xor_sync(0xFFFFFFFF, cur_24, 2);
            float hi_198 = fmaxf(cur_24, pv_102);
            float lo_182 = fminf(cur_24, pv_102);
            cur_24 = ((up[2] != 0) ? hi_198 : lo_182);
            float pv_103 = __shfl_xor_sync(0xFFFFFFFF, cur_24, 1);
            float hi_199 = fmaxf(cur_24, pv_103);
            float lo_183 = fminf(cur_24, pv_103);
            cur_24 = ((up[3] != 0) ? hi_199 : lo_183);
            V_0[2] = cur_24;
            float y1_10 = __shfl_xor_sync(0xFFFFFFFF, V_0[7], 15);
            float lo1_10 = fminf(V_0[3], y1_10);
            r_0 = fmaxf(r_0, lo1_10);
            float hi1_10 = fmaxf(V_0[3], y1_10);
            float cur_25 = hi1_10;
            float pv_104 = __shfl_xor_sync(0xFFFFFFFF, cur_25, 8);
            float hi_200 = fmaxf(cur_25, pv_104);
            float lo_184 = fminf(cur_25, pv_104);
            cur_25 = ((up[0] != 0) ? hi_200 : lo_184);
            float pv_105 = __shfl_xor_sync(0xFFFFFFFF, cur_25, 4);
            float hi_201 = fmaxf(cur_25, pv_105);
            float lo_185 = fminf(cur_25, pv_105);
            cur_25 = ((up[1] != 0) ? hi_201 : lo_185);
            float pv_106 = __shfl_xor_sync(0xFFFFFFFF, cur_25, 2);
            float hi_202 = fmaxf(cur_25, pv_106);
            float lo_186 = fminf(cur_25, pv_106);
            cur_25 = ((up[2] != 0) ? hi_202 : lo_186);
            float pv_107 = __shfl_xor_sync(0xFFFFFFFF, cur_25, 1);
            float hi_203 = fmaxf(cur_25, pv_107);
            float lo_187 = fminf(cur_25, pv_107);
            cur_25 = ((up[3] != 0) ? hi_203 : lo_187);
            V_0[3] = cur_25;
            float y1_11 = __shfl_xor_sync(0xFFFFFFFF, V_0[2], 15);
            float lo1_11 = fminf(V_0[0], y1_11);
            r_0 = fmaxf(r_0, lo1_11);
            float hi1_11 = fmaxf(V_0[0], y1_11);
            float cur_26 = hi1_11;
            float pv_108 = __shfl_xor_sync(0xFFFFFFFF, cur_26, 8);
            float hi_204 = fmaxf(cur_26, pv_108);
            float lo_188 = fminf(cur_26, pv_108);
            cur_26 = ((up[0] != 0) ? hi_204 : lo_188);
            float pv_109 = __shfl_xor_sync(0xFFFFFFFF, cur_26, 4);
            float hi_205 = fmaxf(cur_26, pv_109);
            float lo_189 = fminf(cur_26, pv_109);
            cur_26 = ((up[1] != 0) ? hi_205 : lo_189);
            float pv_110 = __shfl_xor_sync(0xFFFFFFFF, cur_26, 2);
            float hi_206 = fmaxf(cur_26, pv_110);
            float lo_190 = fminf(cur_26, pv_110);
            cur_26 = ((up[2] != 0) ? hi_206 : lo_190);
            float pv_111 = __shfl_xor_sync(0xFFFFFFFF, cur_26, 1);
            float hi_207 = fmaxf(cur_26, pv_111);
            float lo_191 = fminf(cur_26, pv_111);
            cur_26 = ((up[3] != 0) ? hi_207 : lo_191);
            V_0[0] = cur_26;
            float y1_12 = __shfl_xor_sync(0xFFFFFFFF, V_0[3], 15);
            float lo1_12 = fminf(V_0[1], y1_12);
            r_0 = fmaxf(r_0, lo1_12);
            float hi1_12 = fmaxf(V_0[1], y1_12);
            float cur_27 = hi1_12;
            float pv_112 = __shfl_xor_sync(0xFFFFFFFF, cur_27, 8);
            float hi_208 = fmaxf(cur_27, pv_112);
            float lo_192 = fminf(cur_27, pv_112);
            cur_27 = ((up[0] != 0) ? hi_208 : lo_192);
            float pv_113 = __shfl_xor_sync(0xFFFFFFFF, cur_27, 4);
            float hi_209 = fmaxf(cur_27, pv_113);
            float lo_193 = fminf(cur_27, pv_113);
            cur_27 = ((up[1] != 0) ? hi_209 : lo_193);
            float pv_114 = __shfl_xor_sync(0xFFFFFFFF, cur_27, 2);
            float hi_210 = fmaxf(cur_27, pv_114);
            float lo_194 = fminf(cur_27, pv_114);
            cur_27 = ((up[2] != 0) ? hi_210 : lo_194);
            float pv_115 = __shfl_xor_sync(0xFFFFFFFF, cur_27, 1);
            float hi_211 = fmaxf(cur_27, pv_115);
            float lo_195 = fminf(cur_27, pv_115);
            cur_27 = ((up[3] != 0) ? hi_211 : lo_195);
            V_0[1] = cur_27;
            float y1_13 = __shfl_xor_sync(0xFFFFFFFF, V_0[1], 15);
            float lo1_13 = fminf(V_0[0], y1_13);
            r_0 = fmaxf(r_0, lo1_13);
            float hi1_13 = fmaxf(V_0[0], y1_13);
            float cur_28 = hi1_13;
            float pv_116 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 8);
            float hi_212 = fmaxf(cur_28, pv_116);
            float lo_196 = fminf(cur_28, pv_116);
            cur_28 = ((up[0] != 0) ? hi_212 : lo_196);
            float pv_117 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 4);
            float hi_213 = fmaxf(cur_28, pv_117);
            float lo_197 = fminf(cur_28, pv_117);
            cur_28 = ((up[1] != 0) ? hi_213 : lo_197);
            float pv_118 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 2);
            float hi_214 = fmaxf(cur_28, pv_118);
            float lo_198 = fminf(cur_28, pv_118);
            cur_28 = ((up[2] != 0) ? hi_214 : lo_198);
            float pv_119 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 1);
            float hi_215 = fmaxf(cur_28, pv_119);
            float lo_199 = fminf(cur_28, pv_119);
            cur_28 = ((up[3] != 0) ? hi_215 : lo_199);
            V_0[0] = cur_28;
            float K_0 = V_0[0];
            int qb_0 = (sg + cg) * 32;
            q2[qb_0 + ln] = K_0;
            q2[qb_0 + 16 + ln] = r_0;
            asm volatile("barrier.sync 1, 32;" ::: "memory");
            float k2o_0[1];
            float r2v_0 = neg_inf;
            r2v_0 = fmaxf(r2v_0, q2[cg * 32 + 16 + ln]);
            r2v_0 = fmaxf(r2v_0, q2[(1 + cg) * 32 + 16 + ln]);
            float V2_0[1];
            float x2_0 = q2[cg * 32 + ln];
            float y2_0 = q2[(1 + cg) * 32 + lnr];
            float lo2_0 = fminf(x2_0, y2_0);
            r2v_0 = fmaxf(r2v_0, lo2_0);
            float hi2_0 = fmaxf(x2_0, y2_0);
            V2_0[0] = hi2_0;
            k2o_0[0] = V2_0[0];
            K_0 = k2o_0[0];
            r_0 = r2v_0;
            rr2x[0] = r_0;
            K2 = K_0;
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
                    float hi_1_120 = fmaxf(xs[0], xs[13]);
                    float lo_1_120 = fminf(xs[0], xs[13]);
                    xs[0] = hi_1_120;
                    xs[13] = lo_1_120;
                    float hi_1_121 = fmaxf(xs[1], xs[12]);
                    float lo_1_121 = fminf(xs[1], xs[12]);
                    xs[1] = hi_1_121;
                    xs[12] = lo_1_121;
                    float hi_1_122 = fmaxf(xs[2], xs[15]);
                    float lo_1_122 = fminf(xs[2], xs[15]);
                    xs[2] = hi_1_122;
                    xs[15] = lo_1_122;
                    float hi_1_123 = fmaxf(xs[3], xs[14]);
                    float lo_1_123 = fminf(xs[3], xs[14]);
                    xs[3] = hi_1_123;
                    xs[14] = lo_1_123;
                    float hi_1_124 = fmaxf(xs[4], xs[8]);
                    float lo_1_124 = fminf(xs[4], xs[8]);
                    xs[4] = hi_1_124;
                    xs[8] = lo_1_124;
                    float hi_1_125 = fmaxf(xs[5], xs[6]);
                    float lo_1_125 = fminf(xs[5], xs[6]);
                    xs[5] = hi_1_125;
                    xs[6] = lo_1_125;
                    float hi_1_126 = fmaxf(xs[7], xs[11]);
                    float lo_1_126 = fminf(xs[7], xs[11]);
                    xs[7] = hi_1_126;
                    xs[11] = lo_1_126;
                    float hi_1_127 = fmaxf(xs[9], xs[10]);
                    float lo_1_127 = fminf(xs[9], xs[10]);
                    xs[9] = hi_1_127;
                    xs[10] = lo_1_127;
                    float hi_1_128 = fmaxf(xs[0], xs[5]);
                    float lo_1_128 = fminf(xs[0], xs[5]);
                    xs[0] = hi_1_128;
                    xs[5] = lo_1_128;
                    float hi_1_129 = fmaxf(xs[1], xs[7]);
                    float lo_1_129 = fminf(xs[1], xs[7]);
                    xs[1] = hi_1_129;
                    xs[7] = lo_1_129;
                    float hi_1_130 = fmaxf(xs[2], xs[9]);
                    float lo_1_130 = fminf(xs[2], xs[9]);
                    xs[2] = hi_1_130;
                    xs[9] = lo_1_130;
                    float hi_1_131 = fmaxf(xs[3], xs[4]);
                    float lo_1_131 = fminf(xs[3], xs[4]);
                    xs[3] = hi_1_131;
                    xs[4] = lo_1_131;
                    float hi_1_132 = fmaxf(xs[6], xs[13]);
                    float lo_1_132 = fminf(xs[6], xs[13]);
                    xs[6] = hi_1_132;
                    xs[13] = lo_1_132;
                    float hi_1_133 = fmaxf(xs[8], xs[14]);
                    float lo_1_133 = fminf(xs[8], xs[14]);
                    xs[8] = hi_1_133;
                    xs[14] = lo_1_133;
                    float hi_1_134 = fmaxf(xs[10], xs[15]);
                    float lo_1_134 = fminf(xs[10], xs[15]);
                    xs[10] = hi_1_134;
                    xs[15] = lo_1_134;
                    float hi_1_135 = fmaxf(xs[11], xs[12]);
                    float lo_1_135 = fminf(xs[11], xs[12]);
                    xs[11] = hi_1_135;
                    xs[12] = lo_1_135;
                    float hi_1_136 = fmaxf(xs[0], xs[1]);
                    float lo_1_136 = fminf(xs[0], xs[1]);
                    xs[0] = hi_1_136;
                    xs[1] = lo_1_136;
                    float hi_1_137 = fmaxf(xs[2], xs[3]);
                    float lo_1_137 = fminf(xs[2], xs[3]);
                    xs[2] = hi_1_137;
                    xs[3] = lo_1_137;
                    float hi_1_138 = fmaxf(xs[4], xs[5]);
                    float lo_1_138 = fminf(xs[4], xs[5]);
                    xs[4] = hi_1_138;
                    xs[5] = lo_1_138;
                    float hi_1_139 = fmaxf(xs[6], xs[8]);
                    float lo_1_139 = fminf(xs[6], xs[8]);
                    xs[6] = hi_1_139;
                    xs[8] = lo_1_139;
                    float hi_1_140 = fmaxf(xs[7], xs[9]);
                    float lo_1_140 = fminf(xs[7], xs[9]);
                    xs[7] = hi_1_140;
                    xs[9] = lo_1_140;
                    float hi_1_141 = fmaxf(xs[10], xs[11]);
                    float lo_1_141 = fminf(xs[10], xs[11]);
                    xs[10] = hi_1_141;
                    xs[11] = lo_1_141;
                    float hi_1_142 = fmaxf(xs[12], xs[13]);
                    float lo_1_142 = fminf(xs[12], xs[13]);
                    xs[12] = hi_1_142;
                    xs[13] = lo_1_142;
                    float hi_1_143 = fmaxf(xs[14], xs[15]);
                    float lo_1_143 = fminf(xs[14], xs[15]);
                    xs[14] = hi_1_143;
                    xs[15] = lo_1_143;
                    float hi_1_144 = fmaxf(xs[0], xs[2]);
                    float lo_1_144 = fminf(xs[0], xs[2]);
                    xs[0] = hi_1_144;
                    xs[2] = lo_1_144;
                    float hi_1_145 = fmaxf(xs[1], xs[3]);
                    float lo_1_145 = fminf(xs[1], xs[3]);
                    xs[1] = hi_1_145;
                    xs[3] = lo_1_145;
                    float hi_1_146 = fmaxf(xs[4], xs[10]);
                    float lo_1_146 = fminf(xs[4], xs[10]);
                    xs[4] = hi_1_146;
                    xs[10] = lo_1_146;
                    float hi_1_147 = fmaxf(xs[5], xs[11]);
                    float lo_1_147 = fminf(xs[5], xs[11]);
                    xs[5] = hi_1_147;
                    xs[11] = lo_1_147;
                    float hi_1_148 = fmaxf(xs[6], xs[7]);
                    float lo_1_148 = fminf(xs[6], xs[7]);
                    xs[6] = hi_1_148;
                    xs[7] = lo_1_148;
                    float hi_1_149 = fmaxf(xs[8], xs[9]);
                    float lo_1_149 = fminf(xs[8], xs[9]);
                    xs[8] = hi_1_149;
                    xs[9] = lo_1_149;
                    float hi_1_150 = fmaxf(xs[12], xs[14]);
                    float lo_1_150 = fminf(xs[12], xs[14]);
                    xs[12] = hi_1_150;
                    xs[14] = lo_1_150;
                    float hi_1_151 = fmaxf(xs[13], xs[15]);
                    float lo_1_151 = fminf(xs[13], xs[15]);
                    xs[13] = hi_1_151;
                    xs[15] = lo_1_151;
                    float hi_1_152 = fmaxf(xs[1], xs[2]);
                    float lo_1_152 = fminf(xs[1], xs[2]);
                    xs[1] = hi_1_152;
                    xs[2] = lo_1_152;
                    float hi_1_153 = fmaxf(xs[3], xs[12]);
                    float lo_1_153 = fminf(xs[3], xs[12]);
                    xs[3] = hi_1_153;
                    xs[12] = lo_1_153;
                    float hi_1_154 = fmaxf(xs[4], xs[6]);
                    float lo_1_154 = fminf(xs[4], xs[6]);
                    xs[4] = hi_1_154;
                    xs[6] = lo_1_154;
                    float hi_1_155 = fmaxf(xs[5], xs[7]);
                    float lo_1_155 = fminf(xs[5], xs[7]);
                    xs[5] = hi_1_155;
                    xs[7] = lo_1_155;
                    float hi_1_156 = fmaxf(xs[8], xs[10]);
                    float lo_1_156 = fminf(xs[8], xs[10]);
                    xs[8] = hi_1_156;
                    xs[10] = lo_1_156;
                    float hi_1_157 = fmaxf(xs[9], xs[11]);
                    float lo_1_157 = fminf(xs[9], xs[11]);
                    xs[9] = hi_1_157;
                    xs[11] = lo_1_157;
                    float hi_1_158 = fmaxf(xs[13], xs[14]);
                    float lo_1_158 = fminf(xs[13], xs[14]);
                    xs[13] = hi_1_158;
                    xs[14] = lo_1_158;
                    float hi_1_159 = fmaxf(xs[1], xs[4]);
                    float lo_1_159 = fminf(xs[1], xs[4]);
                    xs[1] = hi_1_159;
                    xs[4] = lo_1_159;
                    float hi_1_160 = fmaxf(xs[2], xs[6]);
                    float lo_1_160 = fminf(xs[2], xs[6]);
                    xs[2] = hi_1_160;
                    xs[6] = lo_1_160;
                    float hi_1_161 = fmaxf(xs[5], xs[8]);
                    float lo_1_161 = fminf(xs[5], xs[8]);
                    xs[5] = hi_1_161;
                    xs[8] = lo_1_161;
                    float hi_1_162 = fmaxf(xs[7], xs[10]);
                    float lo_1_162 = fminf(xs[7], xs[10]);
                    xs[7] = hi_1_162;
                    xs[10] = lo_1_162;
                    float hi_1_163 = fmaxf(xs[9], xs[13]);
                    float lo_1_163 = fminf(xs[9], xs[13]);
                    xs[9] = hi_1_163;
                    xs[13] = lo_1_163;
                    float hi_1_164 = fmaxf(xs[11], xs[14]);
                    float lo_1_164 = fminf(xs[11], xs[14]);
                    xs[11] = hi_1_164;
                    xs[14] = lo_1_164;
                    float hi_1_165 = fmaxf(xs[2], xs[4]);
                    float lo_1_165 = fminf(xs[2], xs[4]);
                    xs[2] = hi_1_165;
                    xs[4] = lo_1_165;
                    float hi_1_166 = fmaxf(xs[3], xs[6]);
                    float lo_1_166 = fminf(xs[3], xs[6]);
                    xs[3] = hi_1_166;
                    xs[6] = lo_1_166;
                    float hi_1_167 = fmaxf(xs[9], xs[12]);
                    float lo_1_167 = fminf(xs[9], xs[12]);
                    xs[9] = hi_1_167;
                    xs[12] = lo_1_167;
                    float hi_1_168 = fmaxf(xs[11], xs[13]);
                    float lo_1_168 = fminf(xs[11], xs[13]);
                    xs[11] = hi_1_168;
                    xs[13] = lo_1_168;
                    float hi_1_169 = fmaxf(xs[3], xs[5]);
                    float lo_1_169 = fminf(xs[3], xs[5]);
                    xs[3] = hi_1_169;
                    xs[5] = lo_1_169;
                    float hi_1_170 = fmaxf(xs[6], xs[8]);
                    float lo_1_170 = fminf(xs[6], xs[8]);
                    xs[6] = hi_1_170;
                    xs[8] = lo_1_170;
                    float hi_1_171 = fmaxf(xs[7], xs[9]);
                    float lo_1_171 = fminf(xs[7], xs[9]);
                    xs[7] = hi_1_171;
                    xs[9] = lo_1_171;
                    float hi_1_172 = fmaxf(xs[10], xs[12]);
                    float lo_1_172 = fminf(xs[10], xs[12]);
                    xs[10] = hi_1_172;
                    xs[12] = lo_1_172;
                    float hi_1_173 = fmaxf(xs[3], xs[4]);
                    float lo_1_173 = fminf(xs[3], xs[4]);
                    xs[3] = hi_1_173;
                    xs[4] = lo_1_173;
                    float hi_1_174 = fmaxf(xs[5], xs[6]);
                    float lo_1_174 = fminf(xs[5], xs[6]);
                    xs[5] = hi_1_174;
                    xs[6] = lo_1_174;
                    float hi_1_175 = fmaxf(xs[7], xs[8]);
                    float lo_1_175 = fminf(xs[7], xs[8]);
                    xs[7] = hi_1_175;
                    xs[8] = lo_1_175;
                    float hi_1_176 = fmaxf(xs[9], xs[10]);
                    float lo_1_176 = fminf(xs[9], xs[10]);
                    xs[9] = hi_1_176;
                    xs[10] = lo_1_176;
                    float hi_1_177 = fmaxf(xs[11], xs[12]);
                    float lo_1_177 = fminf(xs[11], xs[12]);
                    xs[11] = hi_1_177;
                    xs[12] = lo_1_177;
                    float hi_1_178 = fmaxf(xs[6], xs[7]);
                    float lo_1_178 = fminf(xs[6], xs[7]);
                    xs[6] = hi_1_178;
                    xs[7] = lo_1_178;
                    float hi_1_179 = fmaxf(xs[8], xs[9]);
                    float lo_1_179 = fminf(xs[8], xs[9]);
                    xs[8] = hi_1_179;
                    xs[9] = lo_1_179;
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
        unsigned int idx = 512;
        if (kk < 4282384384u) {
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
        unsigned int ox_14_0 = __shfl_xor_sync(0xFFFFFFFF, rkey, 1);
        if (ox_14_0 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_14_1 = __shfl_xor_sync(0xFFFFFFFF, rkey, 2);
        if (ox_14_1 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_14_2 = __shfl_xor_sync(0xFFFFFFFF, rkey, 3);
        if (ox_14_2 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_14_3 = __shfl_xor_sync(0xFFFFFFFF, rkey, 4);
        if (ox_14_3 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_14_4 = __shfl_xor_sync(0xFFFFFFFF, rkey, 5);
        if (ox_14_4 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_14_5 = __shfl_xor_sync(0xFFFFFFFF, rkey, 6);
        if (ox_14_5 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_14_6 = __shfl_xor_sync(0xFFFFFFFF, rkey, 7);
        if (ox_14_6 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_14_7 = __shfl_xor_sync(0xFFFFFFFF, rkey, 8);
        if (ox_14_7 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_14_8 = __shfl_xor_sync(0xFFFFFFFF, rkey, 9);
        if (ox_14_8 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_14_9 = __shfl_xor_sync(0xFFFFFFFF, rkey, 10);
        if (ox_14_9 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_14_10 = __shfl_xor_sync(0xFFFFFFFF, rkey, 11);
        if (ox_14_10 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_14_11 = __shfl_xor_sync(0xFFFFFFFF, rkey, 12);
        if (ox_14_11 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_14_12 = __shfl_xor_sync(0xFFFFFFFF, rkey, 13);
        if (ox_14_12 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_14_13 = __shfl_xor_sync(0xFFFFFFFF, rkey, 14);
        if (ox_14_13 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_14_14 = __shfl_xor_sync(0xFFFFFFFF, rkey, 15);
        if (ox_14_14 < rkey) {
            rank = rank + 1;
        }
        int val = -1;
        if (idx <= 511) {
            val = (int)idx;
        }
        if (commit != 0) {
            long long obase = ((long long)ucol * (long long)num_heads + (long long)head) * 16ll;
            out[obase + (long long)rank] = val;
        }
    }
}

} // extern "C"
