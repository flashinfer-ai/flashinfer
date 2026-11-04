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

__global__ __launch_bounds__(256) void
kernel_cake_hopper_msa_43e957550739d30313fb(unsigned int* __restrict__ S, int* __restrict__ nvp, int* __restrict__ out, int tiles, int total_q, int num_heads, int num_chunks, int fb, int fe)
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
    unsigned int* flagw = reinterpret_cast<unsigned int*>(smem_raw + 32);
    const int flagw_addr = smem + 32;
    unsigned int* ccnt = reinterpret_cast<unsigned int*>(smem_raw + 19504);
    const int ccnt_addr = smem + 19504;
    int* limc = reinterpret_cast<int*>(smem_raw + 19536);
    const int limc_addr = smem + 19536;
    unsigned int* cbuf = reinterpret_cast<unsigned int*>(smem_raw + 19568);
    const int cbuf_addr = smem + 19568;
    float* pub = reinterpret_cast<float*>(smem_raw + 48);
    const int pub_addr = smem + 48;
    float* q2 = reinterpret_cast<float*>(smem_raw + 17456);
    const int q2_addr = smem + 17456;

    // === Task calls (declaration order) ===
    int tid_1 = threadIdx.x;
    int w = tid_1 / 8;
    int c = tid_1 - w * 8;
    int whi = w / 4;
    int col = blockIdx.x * 8 + c;
    int head = blockIdx.y;
    if (tid_1 == 0) {
        flagw[0] = 0;
    }
    if (tid_1 == 0) {
        flagw[1] = 0;
    }
    if (tid_1 < 8) {
        ccnt[tid_1] = 0;
    }
    int lim = tiles;
    if (col >= total_q) {
        lim = 0;
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
        cb[2] = S[p + 2 * nq64];
    }
    cb[3] = 4286578688u;
    if (lim > t0 + 3) {
        cb[3] = S[p + 3 * nq64];
    }
    cb[4] = 4286578688u;
    if (lim > t0 + 4) {
        cb[4] = S[p + 4 * nq64];
    }
    cb[5] = 4286578688u;
    if (lim > t0 + 5) {
        cb[5] = S[p + 5 * nq64];
    }
    cb[6] = 4286578688u;
    if (lim > t0 + 6) {
        cb[6] = S[p + 6 * nq64];
    }
    cb[7] = 4286578688u;
    if (lim > t0 + 7) {
        cb[7] = S[p + 7 * nq64];
    }
    cb[8] = 4286578688u;
    if (lim > t0 + 8) {
        cb[8] = S[p + 8 * nq64];
    }
    cb[9] = 4286578688u;
    if (lim > t0 + 9) {
        cb[9] = S[p + 9 * nq64];
    }
    cb[10] = 4286578688u;
    if (lim > t0 + 10) {
        cb[10] = S[p + 10 * nq64];
    }
    cb[11] = 4286578688u;
    if (lim > t0 + 11) {
        cb[11] = S[p + 11 * nq64];
    }
    cb[12] = 4286578688u;
    if (lim > t0 + 12) {
        cb[12] = S[p + 12 * nq64];
    }
    cb[13] = 4286578688u;
    if (lim > t0 + 13) {
        cb[13] = S[p + 13 * nq64];
    }
    cb[14] = 4286578688u;
    if (lim > t0 + 14) {
        cb[14] = S[p + 14 * nq64];
    }
    cb[15] = 4286578688u;
    if (lim > t0 + 15) {
        cb[15] = S[p + 15 * nq64];
    }
    asm volatile("" ::: "memory");
    int t0_1 = w * 16;
    float sc = __uint_as_float(cb[0]);
    sc = fmaxf(sc, -1.7014118e+38f);
    sc = fminf(sc, 1.7014118e+38f);
    unsigned int key_1_0 = __as_u32(sc) & 4294966784u | (unsigned int)t0_1;
    kb[0] = __uint_as_float(key_1_0);
    float sc_0 = __uint_as_float(cb[1]);
    sc_0 = fmaxf(sc_0, -1.7014118e+38f);
    sc_0 = fminf(sc_0, 1.7014118e+38f);
    unsigned int key_1_1 = __as_u32(sc_0) & 4294966784u | (unsigned int)(t0_1 + 1);
    kb[1] = __uint_as_float(key_1_1);
    float sc_1 = __uint_as_float(cb[2]);
    sc_1 = fmaxf(sc_1, -1.7014118e+38f);
    sc_1 = fminf(sc_1, 1.7014118e+38f);
    unsigned int key_1_2 = __as_u32(sc_1) & 4294966784u | (unsigned int)(t0_1 + 2);
    kb[2] = __uint_as_float(key_1_2);
    float sc_2 = __uint_as_float(cb[3]);
    sc_2 = fmaxf(sc_2, -1.7014118e+38f);
    sc_2 = fminf(sc_2, 1.7014118e+38f);
    unsigned int key_1_3 = __as_u32(sc_2) & 4294966784u | (unsigned int)(t0_1 + 3);
    kb[3] = __uint_as_float(key_1_3);
    float sc_3 = __uint_as_float(cb[4]);
    sc_3 = fmaxf(sc_3, -1.7014118e+38f);
    sc_3 = fminf(sc_3, 1.7014118e+38f);
    unsigned int key_1_4 = __as_u32(sc_3) & 4294966784u | (unsigned int)(t0_1 + 4);
    kb[4] = __uint_as_float(key_1_4);
    float sc_4 = __uint_as_float(cb[5]);
    sc_4 = fmaxf(sc_4, -1.7014118e+38f);
    sc_4 = fminf(sc_4, 1.7014118e+38f);
    unsigned int key_1_5 = __as_u32(sc_4) & 4294966784u | (unsigned int)(t0_1 + 5);
    kb[5] = __uint_as_float(key_1_5);
    float sc_5 = __uint_as_float(cb[6]);
    sc_5 = fmaxf(sc_5, -1.7014118e+38f);
    sc_5 = fminf(sc_5, 1.7014118e+38f);
    unsigned int key_1_6 = __as_u32(sc_5) & 4294966784u | (unsigned int)(t0_1 + 6);
    kb[6] = __uint_as_float(key_1_6);
    float sc_6 = __uint_as_float(cb[7]);
    sc_6 = fmaxf(sc_6, -1.7014118e+38f);
    sc_6 = fminf(sc_6, 1.7014118e+38f);
    unsigned int key_1_7 = __as_u32(sc_6) & 4294966784u | (unsigned int)(t0_1 + 7);
    kb[7] = __uint_as_float(key_1_7);
    float sc_7 = __uint_as_float(cb[8]);
    sc_7 = fmaxf(sc_7, -1.7014118e+38f);
    sc_7 = fminf(sc_7, 1.7014118e+38f);
    unsigned int key_1_8 = __as_u32(sc_7) & 4294966784u | (unsigned int)(t0_1 + 8);
    kb[8] = __uint_as_float(key_1_8);
    float sc_8 = __uint_as_float(cb[9]);
    sc_8 = fmaxf(sc_8, -1.7014118e+38f);
    sc_8 = fminf(sc_8, 1.7014118e+38f);
    unsigned int key_1_9 = __as_u32(sc_8) & 4294966784u | (unsigned int)(t0_1 + 9);
    kb[9] = __uint_as_float(key_1_9);
    float sc_9 = __uint_as_float(cb[10]);
    sc_9 = fmaxf(sc_9, -1.7014118e+38f);
    sc_9 = fminf(sc_9, 1.7014118e+38f);
    unsigned int key_1_10 = __as_u32(sc_9) & 4294966784u | (unsigned int)(t0_1 + 10);
    kb[10] = __uint_as_float(key_1_10);
    float sc_10 = __uint_as_float(cb[11]);
    sc_10 = fmaxf(sc_10, -1.7014118e+38f);
    sc_10 = fminf(sc_10, 1.7014118e+38f);
    unsigned int key_1_11 = __as_u32(sc_10) & 4294966784u | (unsigned int)(t0_1 + 11);
    kb[11] = __uint_as_float(key_1_11);
    float sc_11 = __uint_as_float(cb[12]);
    sc_11 = fmaxf(sc_11, -1.7014118e+38f);
    sc_11 = fminf(sc_11, 1.7014118e+38f);
    unsigned int key_1_12 = __as_u32(sc_11) & 4294966784u | (unsigned int)(t0_1 + 12);
    kb[12] = __uint_as_float(key_1_12);
    float sc_12 = __uint_as_float(cb[13]);
    sc_12 = fmaxf(sc_12, -1.7014118e+38f);
    sc_12 = fminf(sc_12, 1.7014118e+38f);
    unsigned int key_1_13 = __as_u32(sc_12) & 4294966784u | (unsigned int)(t0_1 + 13);
    kb[13] = __uint_as_float(key_1_13);
    float sc_13 = __uint_as_float(cb[14]);
    sc_13 = fmaxf(sc_13, -1.7014118e+38f);
    sc_13 = fminf(sc_13, 1.7014118e+38f);
    unsigned int key_1_14 = __as_u32(sc_13) & 4294966784u | (unsigned int)(t0_1 + 14);
    kb[14] = __uint_as_float(key_1_14);
    float sc_14 = __uint_as_float(cb[15]);
    sc_14 = fmaxf(sc_14, -1.7014118e+38f);
    sc_14 = fminf(sc_14, 1.7014118e+38f);
    unsigned int key_1_15 = __as_u32(sc_14) & 4294966784u | (unsigned int)(t0_1 + 15);
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
    asm volatile("barrier.sync 1, 256;" ::: "memory");
    float r = neg_inf;
    float V[8];
    int s0_0 = (sg * 16 * 8 + cg) * 17;
    int s1_0 = ((sg * 16 + 8) * 8 + cg) * 17;
    float x0_0 = pub[s0_0 + ln];
    float y0_0 = pub[s1_0 + lnr];
    float lo0_0 = fminf(x0_0, y0_0);
    r = fmaxf(r, lo0_0);
    float hi0_0 = fmaxf(x0_0, y0_0);
    float cur = hi0_0;
    float pv_0 = __shfl_xor_sync(0xFFFFFFFF, cur, 8);
    float hi_0 = fmaxf(cur, pv_0);
    float lo_0 = fminf(cur, pv_0);
    cur = ((up[0] != 0) ? hi_0 : lo_0);
    float pv_1 = __shfl_xor_sync(0xFFFFFFFF, cur, 4);
    float hi_1 = fmaxf(cur, pv_1);
    float lo_1 = fminf(cur, pv_1);
    cur = ((up[1] != 0) ? hi_1 : lo_1);
    float pv_2 = __shfl_xor_sync(0xFFFFFFFF, cur, 2);
    float hi_2 = fmaxf(cur, pv_2);
    float lo_2 = fminf(cur, pv_2);
    cur = ((up[2] != 0) ? hi_2 : lo_2);
    float pv_3 = __shfl_xor_sync(0xFFFFFFFF, cur, 1);
    float hi_3 = fmaxf(cur, pv_3);
    float lo_3 = fminf(cur, pv_3);
    cur = ((up[3] != 0) ? hi_3 : lo_3);
    V[0] = cur;
    int s0_1 = ((sg * 16 + 1) * 8 + cg) * 17;
    int s1_1 = ((sg * 16 + 1 + 8) * 8 + cg) * 17;
    float x0_1 = pub[s0_1 + ln];
    float y0_1 = pub[s1_1 + lnr];
    float lo0_1 = fminf(x0_1, y0_1);
    r = fmaxf(r, lo0_1);
    float hi0_1 = fmaxf(x0_1, y0_1);
    float cur_0 = hi0_1;
    float pv_4 = __shfl_xor_sync(0xFFFFFFFF, cur_0, 8);
    float hi_4 = fmaxf(cur_0, pv_4);
    float lo_4 = fminf(cur_0, pv_4);
    cur_0 = ((up[0] != 0) ? hi_4 : lo_4);
    float pv_5 = __shfl_xor_sync(0xFFFFFFFF, cur_0, 4);
    float hi_5 = fmaxf(cur_0, pv_5);
    float lo_5 = fminf(cur_0, pv_5);
    cur_0 = ((up[1] != 0) ? hi_5 : lo_5);
    float pv_6 = __shfl_xor_sync(0xFFFFFFFF, cur_0, 2);
    float hi_6 = fmaxf(cur_0, pv_6);
    float lo_6 = fminf(cur_0, pv_6);
    cur_0 = ((up[2] != 0) ? hi_6 : lo_6);
    float pv_7 = __shfl_xor_sync(0xFFFFFFFF, cur_0, 1);
    float hi_7 = fmaxf(cur_0, pv_7);
    float lo_7 = fminf(cur_0, pv_7);
    cur_0 = ((up[3] != 0) ? hi_7 : lo_7);
    V[1] = cur_0;
    int s0_2 = ((sg * 16 + 2) * 8 + cg) * 17;
    int s1_2 = ((sg * 16 + 2 + 8) * 8 + cg) * 17;
    float x0_2 = pub[s0_2 + ln];
    float y0_2 = pub[s1_2 + lnr];
    float lo0_2 = fminf(x0_2, y0_2);
    r = fmaxf(r, lo0_2);
    float hi0_2 = fmaxf(x0_2, y0_2);
    float cur_1 = hi0_2;
    float pv_8 = __shfl_xor_sync(0xFFFFFFFF, cur_1, 8);
    float hi_8 = fmaxf(cur_1, pv_8);
    float lo_8 = fminf(cur_1, pv_8);
    cur_1 = ((up[0] != 0) ? hi_8 : lo_8);
    float pv_9 = __shfl_xor_sync(0xFFFFFFFF, cur_1, 4);
    float hi_9 = fmaxf(cur_1, pv_9);
    float lo_9 = fminf(cur_1, pv_9);
    cur_1 = ((up[1] != 0) ? hi_9 : lo_9);
    float pv_10 = __shfl_xor_sync(0xFFFFFFFF, cur_1, 2);
    float hi_10 = fmaxf(cur_1, pv_10);
    float lo_10 = fminf(cur_1, pv_10);
    cur_1 = ((up[2] != 0) ? hi_10 : lo_10);
    float pv_11 = __shfl_xor_sync(0xFFFFFFFF, cur_1, 1);
    float hi_11 = fmaxf(cur_1, pv_11);
    float lo_11 = fminf(cur_1, pv_11);
    cur_1 = ((up[3] != 0) ? hi_11 : lo_11);
    V[2] = cur_1;
    int s0_3 = ((sg * 16 + 3) * 8 + cg) * 17;
    int s1_3 = ((sg * 16 + 3 + 8) * 8 + cg) * 17;
    float x0_3 = pub[s0_3 + ln];
    float y0_3 = pub[s1_3 + lnr];
    float lo0_3 = fminf(x0_3, y0_3);
    r = fmaxf(r, lo0_3);
    float hi0_3 = fmaxf(x0_3, y0_3);
    float cur_2 = hi0_3;
    float pv_12 = __shfl_xor_sync(0xFFFFFFFF, cur_2, 8);
    float hi_12 = fmaxf(cur_2, pv_12);
    float lo_12 = fminf(cur_2, pv_12);
    cur_2 = ((up[0] != 0) ? hi_12 : lo_12);
    float pv_13 = __shfl_xor_sync(0xFFFFFFFF, cur_2, 4);
    float hi_13 = fmaxf(cur_2, pv_13);
    float lo_13 = fminf(cur_2, pv_13);
    cur_2 = ((up[1] != 0) ? hi_13 : lo_13);
    float pv_14 = __shfl_xor_sync(0xFFFFFFFF, cur_2, 2);
    float hi_14 = fmaxf(cur_2, pv_14);
    float lo_14 = fminf(cur_2, pv_14);
    cur_2 = ((up[2] != 0) ? hi_14 : lo_14);
    float pv_15 = __shfl_xor_sync(0xFFFFFFFF, cur_2, 1);
    float hi_15 = fmaxf(cur_2, pv_15);
    float lo_15 = fminf(cur_2, pv_15);
    cur_2 = ((up[3] != 0) ? hi_15 : lo_15);
    V[3] = cur_2;
    int s0_4 = ((sg * 16 + 4) * 8 + cg) * 17;
    int s1_4 = ((sg * 16 + 4 + 8) * 8 + cg) * 17;
    float x0_4 = pub[s0_4 + ln];
    float y0_4 = pub[s1_4 + lnr];
    float lo0_4 = fminf(x0_4, y0_4);
    r = fmaxf(r, lo0_4);
    float hi0_4 = fmaxf(x0_4, y0_4);
    float cur_3 = hi0_4;
    float pv_16 = __shfl_xor_sync(0xFFFFFFFF, cur_3, 8);
    float hi_16 = fmaxf(cur_3, pv_16);
    float lo_16 = fminf(cur_3, pv_16);
    cur_3 = ((up[0] != 0) ? hi_16 : lo_16);
    float pv_17 = __shfl_xor_sync(0xFFFFFFFF, cur_3, 4);
    float hi_17 = fmaxf(cur_3, pv_17);
    float lo_17 = fminf(cur_3, pv_17);
    cur_3 = ((up[1] != 0) ? hi_17 : lo_17);
    float pv_18 = __shfl_xor_sync(0xFFFFFFFF, cur_3, 2);
    float hi_18 = fmaxf(cur_3, pv_18);
    float lo_18 = fminf(cur_3, pv_18);
    cur_3 = ((up[2] != 0) ? hi_18 : lo_18);
    float pv_19 = __shfl_xor_sync(0xFFFFFFFF, cur_3, 1);
    float hi_19 = fmaxf(cur_3, pv_19);
    float lo_19 = fminf(cur_3, pv_19);
    cur_3 = ((up[3] != 0) ? hi_19 : lo_19);
    V[4] = cur_3;
    int s0_5 = ((sg * 16 + 5) * 8 + cg) * 17;
    int s1_5 = ((sg * 16 + 5 + 8) * 8 + cg) * 17;
    float x0_5 = pub[s0_5 + ln];
    float y0_5 = pub[s1_5 + lnr];
    float lo0_5 = fminf(x0_5, y0_5);
    r = fmaxf(r, lo0_5);
    float hi0_5 = fmaxf(x0_5, y0_5);
    float cur_4 = hi0_5;
    float pv_20 = __shfl_xor_sync(0xFFFFFFFF, cur_4, 8);
    float hi_20 = fmaxf(cur_4, pv_20);
    float lo_20 = fminf(cur_4, pv_20);
    cur_4 = ((up[0] != 0) ? hi_20 : lo_20);
    float pv_21 = __shfl_xor_sync(0xFFFFFFFF, cur_4, 4);
    float hi_21 = fmaxf(cur_4, pv_21);
    float lo_21 = fminf(cur_4, pv_21);
    cur_4 = ((up[1] != 0) ? hi_21 : lo_21);
    float pv_22 = __shfl_xor_sync(0xFFFFFFFF, cur_4, 2);
    float hi_22 = fmaxf(cur_4, pv_22);
    float lo_22 = fminf(cur_4, pv_22);
    cur_4 = ((up[2] != 0) ? hi_22 : lo_22);
    float pv_23 = __shfl_xor_sync(0xFFFFFFFF, cur_4, 1);
    float hi_23 = fmaxf(cur_4, pv_23);
    float lo_23 = fminf(cur_4, pv_23);
    cur_4 = ((up[3] != 0) ? hi_23 : lo_23);
    V[5] = cur_4;
    int s0_6 = ((sg * 16 + 6) * 8 + cg) * 17;
    int s1_6 = ((sg * 16 + 6 + 8) * 8 + cg) * 17;
    float x0_6 = pub[s0_6 + ln];
    float y0_6 = pub[s1_6 + lnr];
    float lo0_6 = fminf(x0_6, y0_6);
    r = fmaxf(r, lo0_6);
    float hi0_6 = fmaxf(x0_6, y0_6);
    float cur_5 = hi0_6;
    float pv_24 = __shfl_xor_sync(0xFFFFFFFF, cur_5, 8);
    float hi_24 = fmaxf(cur_5, pv_24);
    float lo_24 = fminf(cur_5, pv_24);
    cur_5 = ((up[0] != 0) ? hi_24 : lo_24);
    float pv_25 = __shfl_xor_sync(0xFFFFFFFF, cur_5, 4);
    float hi_25 = fmaxf(cur_5, pv_25);
    float lo_25 = fminf(cur_5, pv_25);
    cur_5 = ((up[1] != 0) ? hi_25 : lo_25);
    float pv_26 = __shfl_xor_sync(0xFFFFFFFF, cur_5, 2);
    float hi_26 = fmaxf(cur_5, pv_26);
    float lo_26 = fminf(cur_5, pv_26);
    cur_5 = ((up[2] != 0) ? hi_26 : lo_26);
    float pv_27 = __shfl_xor_sync(0xFFFFFFFF, cur_5, 1);
    float hi_27 = fmaxf(cur_5, pv_27);
    float lo_27 = fminf(cur_5, pv_27);
    cur_5 = ((up[3] != 0) ? hi_27 : lo_27);
    V[6] = cur_5;
    int s0_7 = ((sg * 16 + 7) * 8 + cg) * 17;
    int s1_7 = ((sg * 16 + 7 + 8) * 8 + cg) * 17;
    float x0_7 = pub[s0_7 + ln];
    float y0_7 = pub[s1_7 + lnr];
    float lo0_7 = fminf(x0_7, y0_7);
    r = fmaxf(r, lo0_7);
    float hi0_7 = fmaxf(x0_7, y0_7);
    float cur_6 = hi0_7;
    float pv_28 = __shfl_xor_sync(0xFFFFFFFF, cur_6, 8);
    float hi_28 = fmaxf(cur_6, pv_28);
    float lo_28 = fminf(cur_6, pv_28);
    cur_6 = ((up[0] != 0) ? hi_28 : lo_28);
    float pv_29 = __shfl_xor_sync(0xFFFFFFFF, cur_6, 4);
    float hi_29 = fmaxf(cur_6, pv_29);
    float lo_29 = fminf(cur_6, pv_29);
    cur_6 = ((up[1] != 0) ? hi_29 : lo_29);
    float pv_30 = __shfl_xor_sync(0xFFFFFFFF, cur_6, 2);
    float hi_30 = fmaxf(cur_6, pv_30);
    float lo_30 = fminf(cur_6, pv_30);
    cur_6 = ((up[2] != 0) ? hi_30 : lo_30);
    float pv_31 = __shfl_xor_sync(0xFFFFFFFF, cur_6, 1);
    float hi_31 = fmaxf(cur_6, pv_31);
    float lo_31 = fminf(cur_6, pv_31);
    cur_6 = ((up[3] != 0) ? hi_31 : lo_31);
    V[7] = cur_6;
    float rs = pub[((sg * 16 + ln) * 8 + cg) * 17 + 16];
    r = fmaxf(r, rs);
    float y1_0 = __shfl_xor_sync(0xFFFFFFFF, V[4], 15);
    float lo1_0 = fminf(V[0], y1_0);
    r = fmaxf(r, lo1_0);
    float hi1_0 = fmaxf(V[0], y1_0);
    float cur_7 = hi1_0;
    float pv_32 = __shfl_xor_sync(0xFFFFFFFF, cur_7, 8);
    float hi_32 = fmaxf(cur_7, pv_32);
    float lo_32 = fminf(cur_7, pv_32);
    cur_7 = ((up[0] != 0) ? hi_32 : lo_32);
    float pv_33 = __shfl_xor_sync(0xFFFFFFFF, cur_7, 4);
    float hi_33 = fmaxf(cur_7, pv_33);
    float lo_33 = fminf(cur_7, pv_33);
    cur_7 = ((up[1] != 0) ? hi_33 : lo_33);
    float pv_34 = __shfl_xor_sync(0xFFFFFFFF, cur_7, 2);
    float hi_34 = fmaxf(cur_7, pv_34);
    float lo_34 = fminf(cur_7, pv_34);
    cur_7 = ((up[2] != 0) ? hi_34 : lo_34);
    float pv_35 = __shfl_xor_sync(0xFFFFFFFF, cur_7, 1);
    float hi_35 = fmaxf(cur_7, pv_35);
    float lo_35 = fminf(cur_7, pv_35);
    cur_7 = ((up[3] != 0) ? hi_35 : lo_35);
    V[0] = cur_7;
    float y1_1 = __shfl_xor_sync(0xFFFFFFFF, V[5], 15);
    float lo1_1 = fminf(V[1], y1_1);
    r = fmaxf(r, lo1_1);
    float hi1_1 = fmaxf(V[1], y1_1);
    float cur_8 = hi1_1;
    float pv_36 = __shfl_xor_sync(0xFFFFFFFF, cur_8, 8);
    float hi_36 = fmaxf(cur_8, pv_36);
    float lo_36 = fminf(cur_8, pv_36);
    cur_8 = ((up[0] != 0) ? hi_36 : lo_36);
    float pv_37 = __shfl_xor_sync(0xFFFFFFFF, cur_8, 4);
    float hi_37 = fmaxf(cur_8, pv_37);
    float lo_37 = fminf(cur_8, pv_37);
    cur_8 = ((up[1] != 0) ? hi_37 : lo_37);
    float pv_38 = __shfl_xor_sync(0xFFFFFFFF, cur_8, 2);
    float hi_38 = fmaxf(cur_8, pv_38);
    float lo_38 = fminf(cur_8, pv_38);
    cur_8 = ((up[2] != 0) ? hi_38 : lo_38);
    float pv_39 = __shfl_xor_sync(0xFFFFFFFF, cur_8, 1);
    float hi_39 = fmaxf(cur_8, pv_39);
    float lo_39 = fminf(cur_8, pv_39);
    cur_8 = ((up[3] != 0) ? hi_39 : lo_39);
    V[1] = cur_8;
    float y1_2 = __shfl_xor_sync(0xFFFFFFFF, V[6], 15);
    float lo1_2 = fminf(V[2], y1_2);
    r = fmaxf(r, lo1_2);
    float hi1_2 = fmaxf(V[2], y1_2);
    float cur_9 = hi1_2;
    float pv_40 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 8);
    float hi_40 = fmaxf(cur_9, pv_40);
    float lo_40 = fminf(cur_9, pv_40);
    cur_9 = ((up[0] != 0) ? hi_40 : lo_40);
    float pv_41 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 4);
    float hi_41 = fmaxf(cur_9, pv_41);
    float lo_41 = fminf(cur_9, pv_41);
    cur_9 = ((up[1] != 0) ? hi_41 : lo_41);
    float pv_42 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 2);
    float hi_42 = fmaxf(cur_9, pv_42);
    float lo_42 = fminf(cur_9, pv_42);
    cur_9 = ((up[2] != 0) ? hi_42 : lo_42);
    float pv_43 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 1);
    float hi_43 = fmaxf(cur_9, pv_43);
    float lo_43 = fminf(cur_9, pv_43);
    cur_9 = ((up[3] != 0) ? hi_43 : lo_43);
    V[2] = cur_9;
    float y1_3 = __shfl_xor_sync(0xFFFFFFFF, V[7], 15);
    float lo1_3 = fminf(V[3], y1_3);
    r = fmaxf(r, lo1_3);
    float hi1_3 = fmaxf(V[3], y1_3);
    float cur_10 = hi1_3;
    float pv_44 = __shfl_xor_sync(0xFFFFFFFF, cur_10, 8);
    float hi_44 = fmaxf(cur_10, pv_44);
    float lo_44 = fminf(cur_10, pv_44);
    cur_10 = ((up[0] != 0) ? hi_44 : lo_44);
    float pv_45 = __shfl_xor_sync(0xFFFFFFFF, cur_10, 4);
    float hi_45 = fmaxf(cur_10, pv_45);
    float lo_45 = fminf(cur_10, pv_45);
    cur_10 = ((up[1] != 0) ? hi_45 : lo_45);
    float pv_46 = __shfl_xor_sync(0xFFFFFFFF, cur_10, 2);
    float hi_46 = fmaxf(cur_10, pv_46);
    float lo_46 = fminf(cur_10, pv_46);
    cur_10 = ((up[2] != 0) ? hi_46 : lo_46);
    float pv_47 = __shfl_xor_sync(0xFFFFFFFF, cur_10, 1);
    float hi_47 = fmaxf(cur_10, pv_47);
    float lo_47 = fminf(cur_10, pv_47);
    cur_10 = ((up[3] != 0) ? hi_47 : lo_47);
    V[3] = cur_10;
    float y1_4 = __shfl_xor_sync(0xFFFFFFFF, V[2], 15);
    float lo1_4 = fminf(V[0], y1_4);
    r = fmaxf(r, lo1_4);
    float hi1_4 = fmaxf(V[0], y1_4);
    float cur_11 = hi1_4;
    float pv_48 = __shfl_xor_sync(0xFFFFFFFF, cur_11, 8);
    float hi_48 = fmaxf(cur_11, pv_48);
    float lo_48 = fminf(cur_11, pv_48);
    cur_11 = ((up[0] != 0) ? hi_48 : lo_48);
    float pv_49 = __shfl_xor_sync(0xFFFFFFFF, cur_11, 4);
    float hi_49 = fmaxf(cur_11, pv_49);
    float lo_49 = fminf(cur_11, pv_49);
    cur_11 = ((up[1] != 0) ? hi_49 : lo_49);
    float pv_50 = __shfl_xor_sync(0xFFFFFFFF, cur_11, 2);
    float hi_50 = fmaxf(cur_11, pv_50);
    float lo_50 = fminf(cur_11, pv_50);
    cur_11 = ((up[2] != 0) ? hi_50 : lo_50);
    float pv_51 = __shfl_xor_sync(0xFFFFFFFF, cur_11, 1);
    float hi_51 = fmaxf(cur_11, pv_51);
    float lo_51 = fminf(cur_11, pv_51);
    cur_11 = ((up[3] != 0) ? hi_51 : lo_51);
    V[0] = cur_11;
    float y1_5 = __shfl_xor_sync(0xFFFFFFFF, V[3], 15);
    float lo1_5 = fminf(V[1], y1_5);
    r = fmaxf(r, lo1_5);
    float hi1_5 = fmaxf(V[1], y1_5);
    float cur_12 = hi1_5;
    float pv_52 = __shfl_xor_sync(0xFFFFFFFF, cur_12, 8);
    float hi_52 = fmaxf(cur_12, pv_52);
    float lo_52 = fminf(cur_12, pv_52);
    cur_12 = ((up[0] != 0) ? hi_52 : lo_52);
    float pv_53 = __shfl_xor_sync(0xFFFFFFFF, cur_12, 4);
    float hi_53 = fmaxf(cur_12, pv_53);
    float lo_53 = fminf(cur_12, pv_53);
    cur_12 = ((up[1] != 0) ? hi_53 : lo_53);
    float pv_54 = __shfl_xor_sync(0xFFFFFFFF, cur_12, 2);
    float hi_54 = fmaxf(cur_12, pv_54);
    float lo_54 = fminf(cur_12, pv_54);
    cur_12 = ((up[2] != 0) ? hi_54 : lo_54);
    float pv_55 = __shfl_xor_sync(0xFFFFFFFF, cur_12, 1);
    float hi_55 = fmaxf(cur_12, pv_55);
    float lo_55 = fminf(cur_12, pv_55);
    cur_12 = ((up[3] != 0) ? hi_55 : lo_55);
    V[1] = cur_12;
    float y1_6 = __shfl_xor_sync(0xFFFFFFFF, V[1], 15);
    float lo1_6 = fminf(V[0], y1_6);
    r = fmaxf(r, lo1_6);
    float hi1_6 = fmaxf(V[0], y1_6);
    float cur_13 = hi1_6;
    float pv_56 = __shfl_xor_sync(0xFFFFFFFF, cur_13, 8);
    float hi_56 = fmaxf(cur_13, pv_56);
    float lo_56 = fminf(cur_13, pv_56);
    cur_13 = ((up[0] != 0) ? hi_56 : lo_56);
    float pv_57 = __shfl_xor_sync(0xFFFFFFFF, cur_13, 4);
    float hi_57 = fmaxf(cur_13, pv_57);
    float lo_57 = fminf(cur_13, pv_57);
    cur_13 = ((up[1] != 0) ? hi_57 : lo_57);
    float pv_58 = __shfl_xor_sync(0xFFFFFFFF, cur_13, 2);
    float hi_58 = fmaxf(cur_13, pv_58);
    float lo_58 = fminf(cur_13, pv_58);
    cur_13 = ((up[2] != 0) ? hi_58 : lo_58);
    float pv_59 = __shfl_xor_sync(0xFFFFFFFF, cur_13, 1);
    float hi_59 = fmaxf(cur_13, pv_59);
    float lo_59 = fminf(cur_13, pv_59);
    cur_13 = ((up[3] != 0) ? hi_59 : lo_59);
    V[0] = cur_13;
    float K = V[0];
    int qb = (sg * 8 + cg) * 32;
    q2[qb + ln] = K;
    q2[qb + 16 + ln] = r;
    asm volatile("barrier.sync 1, 256;" ::: "memory");
    float k2o[1];
    if (tid_1 < 128) {
        float r2v = neg_inf;
        r2v = fmaxf(r2v, q2[cg * 32 + 16 + ln]);
        r2v = fmaxf(r2v, q2[(8 + cg) * 32 + 16 + ln]);
        float V2[1];
        float x2 = q2[cg * 32 + ln];
        float y2 = q2[(8 + cg) * 32 + lnr];
        float lo2 = fminf(x2, y2);
        r2v = fmaxf(r2v, lo2);
        float hi2 = fmaxf(x2, y2);
        V2[0] = hi2;
        k2o[0] = V2[0];
        K = k2o[0];
        r = r2v;
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
    asm volatile("barrier.sync 1, 256;" ::: "memory");
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
            int t0_2 = w * 16;
            float sc_15 = __uint_as_float(cb[0]);
            sc_15 = fmaxf(sc_15, -1.7014118e+38f);
            sc_15 = fminf(sc_15, 1.7014118e+38f);
            unsigned int u8 = __as_u32(sc_15);
            unsigned int key8 = 0;
            if ((u8 & 4294966784u) == qc) {
                unsigned int lowb8 = (u8 ^ (unsigned int)((int)u8 >> 31) & 511) & 511;
                key8 = 536870912 | lowb8 << 9 | (unsigned int)t0_2;
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
            if ((u8_0 & 4294966784u) == qc) {
                unsigned int lowb8_1 = (u8_0 ^ (unsigned int)((int)u8_0 >> 31) & 511) & 511;
                key8_0 = 536870912 | lowb8_1 << 9 | (unsigned int)(t0_2 + 1);
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
            if ((u8_1 & 4294966784u) == qc) {
                unsigned int lowb8_2 = (u8_1 ^ (unsigned int)((int)u8_1 >> 31) & 511) & 511;
                key8_1 = 536870912 | lowb8_2 << 9 | (unsigned int)(t0_2 + 2);
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
            if ((u8_2 & 4294966784u) == qc) {
                unsigned int lowb8_3 = (u8_2 ^ (unsigned int)((int)u8_2 >> 31) & 511) & 511;
                key8_2 = 536870912 | lowb8_3 << 9 | (unsigned int)(t0_2 + 3);
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
            if ((u8_3 & 4294966784u) == qc) {
                unsigned int lowb8_4 = (u8_3 ^ (unsigned int)((int)u8_3 >> 31) & 511) & 511;
                key8_3 = 536870912 | lowb8_4 << 9 | (unsigned int)(t0_2 + 4);
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
            if ((u8_4 & 4294966784u) == qc) {
                unsigned int lowb8_5 = (u8_4 ^ (unsigned int)((int)u8_4 >> 31) & 511) & 511;
                key8_4 = 536870912 | lowb8_5 << 9 | (unsigned int)(t0_2 + 5);
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
            if ((u8_5 & 4294966784u) == qc) {
                unsigned int lowb8_6 = (u8_5 ^ (unsigned int)((int)u8_5 >> 31) & 511) & 511;
                key8_5 = 536870912 | lowb8_6 << 9 | (unsigned int)(t0_2 + 6);
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
            if ((u8_6 & 4294966784u) == qc) {
                unsigned int lowb8_7 = (u8_6 ^ (unsigned int)((int)u8_6 >> 31) & 511) & 511;
                key8_6 = 536870912 | lowb8_7 << 9 | (unsigned int)(t0_2 + 7);
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
            if ((u8_7 & 4294966784u) == qc) {
                unsigned int lowb8_8 = (u8_7 ^ (unsigned int)((int)u8_7 >> 31) & 511) & 511;
                key8_7 = 536870912 | lowb8_8 << 9 | (unsigned int)(t0_2 + 8);
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
            if ((u8_8 & 4294966784u) == qc) {
                unsigned int lowb8_9 = (u8_8 ^ (unsigned int)((int)u8_8 >> 31) & 511) & 511;
                key8_8 = 536870912 | lowb8_9 << 9 | (unsigned int)(t0_2 + 9);
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
            if ((u8_9 & 4294966784u) == qc) {
                unsigned int lowb8_10 = (u8_9 ^ (unsigned int)((int)u8_9 >> 31) & 511) & 511;
                key8_9 = 536870912 | lowb8_10 << 9 | (unsigned int)(t0_2 + 10);
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
            if ((u8_10 & 4294966784u) == qc) {
                unsigned int lowb8_11 = (u8_10 ^ (unsigned int)((int)u8_10 >> 31) & 511) & 511;
                key8_10 = 536870912 | lowb8_11 << 9 | (unsigned int)(t0_2 + 11);
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
            if ((u8_11 & 4294966784u) == qc) {
                unsigned int lowb8_12 = (u8_11 ^ (unsigned int)((int)u8_11 >> 31) & 511) & 511;
                key8_11 = 536870912 | lowb8_12 << 9 | (unsigned int)(t0_2 + 12);
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
            if ((u8_12 & 4294966784u) == qc) {
                unsigned int lowb8_13 = (u8_12 ^ (unsigned int)((int)u8_12 >> 31) & 511) & 511;
                key8_12 = 536870912 | lowb8_13 << 9 | (unsigned int)(t0_2 + 13);
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
            if ((u8_13 & 4294966784u) == qc) {
                unsigned int lowb8_14 = (u8_13 ^ (unsigned int)((int)u8_13 >> 31) & 511) & 511;
                key8_13 = 536870912 | lowb8_14 << 9 | (unsigned int)(t0_2 + 14);
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
            if ((u8_14 & 4294966784u) == qc) {
                unsigned int lowb8_15 = (u8_14 ^ (unsigned int)((int)u8_14 >> 31) & 511) & 511;
                key8_14 = 536870912 | lowb8_15 << 9 | (unsigned int)(t0_2 + 15);
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
        asm volatile("barrier.sync 1, 256;" ::: "memory");
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
                int t0_3 = w * 16;
                float qf = __uint_as_float(qc);
                float sc_31 = __uint_as_float(cb[0]);
                sc_31 = fmaxf(sc_31, -1.7014118e+38f);
                sc_31 = fminf(sc_31, 1.7014118e+38f);
                unsigned int u_1_0 = __as_u32(sc_31);
                unsigned int cls_1_0 = u_1_0 & 4294966784u;
                unsigned int key = 0;
                if ((qf < __uint_as_float(cls_1_0) || __uint_as_float(cls_1_0) == qf && cls_1_0 < qc) && cls_1_0 < 4278190080u) {
                    key = 1073741824 | (unsigned int)t0_3;
                }
                if (cls_1_0 == qc) {
                    unsigned int lowb = (u_1_0 ^ (unsigned int)((int)u_1_0 >> 31) & 511) & 511;
                    key = 536870912 | lowb << 9 | (unsigned int)t0_3;
                }
                kb2o[0] = __uint_as_float(key);
                float sc_32 = __uint_as_float(cb[1]);
                sc_32 = fmaxf(sc_32, -1.7014118e+38f);
                sc_32 = fminf(sc_32, 1.7014118e+38f);
                unsigned int u_1_1 = __as_u32(sc_32);
                unsigned int cls_1_1 = u_1_1 & 4294966784u;
                unsigned int key_0 = 0;
                if ((qf < __uint_as_float(cls_1_1) || __uint_as_float(cls_1_1) == qf && cls_1_1 < qc) && cls_1_1 < 4278190080u) {
                    key_0 = 1073741824 | (unsigned int)(t0_3 + 1);
                }
                if (cls_1_1 == qc) {
                    unsigned int lowb_1 = (u_1_1 ^ (unsigned int)((int)u_1_1 >> 31) & 511) & 511;
                    key_0 = 536870912 | lowb_1 << 9 | (unsigned int)(t0_3 + 1);
                }
                kb2o[1] = __uint_as_float(key_0);
                float sc_33 = __uint_as_float(cb[2]);
                sc_33 = fmaxf(sc_33, -1.7014118e+38f);
                sc_33 = fminf(sc_33, 1.7014118e+38f);
                unsigned int u_1_2 = __as_u32(sc_33);
                unsigned int cls_1_2 = u_1_2 & 4294966784u;
                unsigned int key_1 = 0;
                if ((qf < __uint_as_float(cls_1_2) || __uint_as_float(cls_1_2) == qf && cls_1_2 < qc) && cls_1_2 < 4278190080u) {
                    key_1 = 1073741824 | (unsigned int)(t0_3 + 2);
                }
                if (cls_1_2 == qc) {
                    unsigned int lowb_2 = (u_1_2 ^ (unsigned int)((int)u_1_2 >> 31) & 511) & 511;
                    key_1 = 536870912 | lowb_2 << 9 | (unsigned int)(t0_3 + 2);
                }
                kb2o[2] = __uint_as_float(key_1);
                float sc_34 = __uint_as_float(cb[3]);
                sc_34 = fmaxf(sc_34, -1.7014118e+38f);
                sc_34 = fminf(sc_34, 1.7014118e+38f);
                unsigned int u_1_3 = __as_u32(sc_34);
                unsigned int cls_1_3 = u_1_3 & 4294966784u;
                unsigned int key_2 = 0;
                if ((qf < __uint_as_float(cls_1_3) || __uint_as_float(cls_1_3) == qf && cls_1_3 < qc) && cls_1_3 < 4278190080u) {
                    key_2 = 1073741824 | (unsigned int)(t0_3 + 3);
                }
                if (cls_1_3 == qc) {
                    unsigned int lowb_3 = (u_1_3 ^ (unsigned int)((int)u_1_3 >> 31) & 511) & 511;
                    key_2 = 536870912 | lowb_3 << 9 | (unsigned int)(t0_3 + 3);
                }
                kb2o[3] = __uint_as_float(key_2);
                float sc_35 = __uint_as_float(cb[4]);
                sc_35 = fmaxf(sc_35, -1.7014118e+38f);
                sc_35 = fminf(sc_35, 1.7014118e+38f);
                unsigned int u_1_4 = __as_u32(sc_35);
                unsigned int cls_1_4 = u_1_4 & 4294966784u;
                unsigned int key_3 = 0;
                if ((qf < __uint_as_float(cls_1_4) || __uint_as_float(cls_1_4) == qf && cls_1_4 < qc) && cls_1_4 < 4278190080u) {
                    key_3 = 1073741824 | (unsigned int)(t0_3 + 4);
                }
                if (cls_1_4 == qc) {
                    unsigned int lowb_4 = (u_1_4 ^ (unsigned int)((int)u_1_4 >> 31) & 511) & 511;
                    key_3 = 536870912 | lowb_4 << 9 | (unsigned int)(t0_3 + 4);
                }
                kb2o[4] = __uint_as_float(key_3);
                float sc_36 = __uint_as_float(cb[5]);
                sc_36 = fmaxf(sc_36, -1.7014118e+38f);
                sc_36 = fminf(sc_36, 1.7014118e+38f);
                unsigned int u_1_5 = __as_u32(sc_36);
                unsigned int cls_1_5 = u_1_5 & 4294966784u;
                unsigned int key_4 = 0;
                if ((qf < __uint_as_float(cls_1_5) || __uint_as_float(cls_1_5) == qf && cls_1_5 < qc) && cls_1_5 < 4278190080u) {
                    key_4 = 1073741824 | (unsigned int)(t0_3 + 5);
                }
                if (cls_1_5 == qc) {
                    unsigned int lowb_5 = (u_1_5 ^ (unsigned int)((int)u_1_5 >> 31) & 511) & 511;
                    key_4 = 536870912 | lowb_5 << 9 | (unsigned int)(t0_3 + 5);
                }
                kb2o[5] = __uint_as_float(key_4);
                float sc_37 = __uint_as_float(cb[6]);
                sc_37 = fmaxf(sc_37, -1.7014118e+38f);
                sc_37 = fminf(sc_37, 1.7014118e+38f);
                unsigned int u_1_6 = __as_u32(sc_37);
                unsigned int cls_1_6 = u_1_6 & 4294966784u;
                unsigned int key_5 = 0;
                if ((qf < __uint_as_float(cls_1_6) || __uint_as_float(cls_1_6) == qf && cls_1_6 < qc) && cls_1_6 < 4278190080u) {
                    key_5 = 1073741824 | (unsigned int)(t0_3 + 6);
                }
                if (cls_1_6 == qc) {
                    unsigned int lowb_6 = (u_1_6 ^ (unsigned int)((int)u_1_6 >> 31) & 511) & 511;
                    key_5 = 536870912 | lowb_6 << 9 | (unsigned int)(t0_3 + 6);
                }
                kb2o[6] = __uint_as_float(key_5);
                float sc_38 = __uint_as_float(cb[7]);
                sc_38 = fmaxf(sc_38, -1.7014118e+38f);
                sc_38 = fminf(sc_38, 1.7014118e+38f);
                unsigned int u_1_7 = __as_u32(sc_38);
                unsigned int cls_1_7 = u_1_7 & 4294966784u;
                unsigned int key_6 = 0;
                if ((qf < __uint_as_float(cls_1_7) || __uint_as_float(cls_1_7) == qf && cls_1_7 < qc) && cls_1_7 < 4278190080u) {
                    key_6 = 1073741824 | (unsigned int)(t0_3 + 7);
                }
                if (cls_1_7 == qc) {
                    unsigned int lowb_7 = (u_1_7 ^ (unsigned int)((int)u_1_7 >> 31) & 511) & 511;
                    key_6 = 536870912 | lowb_7 << 9 | (unsigned int)(t0_3 + 7);
                }
                kb2o[7] = __uint_as_float(key_6);
                float sc_39 = __uint_as_float(cb[8]);
                sc_39 = fmaxf(sc_39, -1.7014118e+38f);
                sc_39 = fminf(sc_39, 1.7014118e+38f);
                unsigned int u_1_8 = __as_u32(sc_39);
                unsigned int cls_1_8 = u_1_8 & 4294966784u;
                unsigned int key_7 = 0;
                if ((qf < __uint_as_float(cls_1_8) || __uint_as_float(cls_1_8) == qf && cls_1_8 < qc) && cls_1_8 < 4278190080u) {
                    key_7 = 1073741824 | (unsigned int)(t0_3 + 8);
                }
                if (cls_1_8 == qc) {
                    unsigned int lowb_8 = (u_1_8 ^ (unsigned int)((int)u_1_8 >> 31) & 511) & 511;
                    key_7 = 536870912 | lowb_8 << 9 | (unsigned int)(t0_3 + 8);
                }
                kb2o[8] = __uint_as_float(key_7);
                float sc_40 = __uint_as_float(cb[9]);
                sc_40 = fmaxf(sc_40, -1.7014118e+38f);
                sc_40 = fminf(sc_40, 1.7014118e+38f);
                unsigned int u_1_9 = __as_u32(sc_40);
                unsigned int cls_1_9 = u_1_9 & 4294966784u;
                unsigned int key_8 = 0;
                if ((qf < __uint_as_float(cls_1_9) || __uint_as_float(cls_1_9) == qf && cls_1_9 < qc) && cls_1_9 < 4278190080u) {
                    key_8 = 1073741824 | (unsigned int)(t0_3 + 9);
                }
                if (cls_1_9 == qc) {
                    unsigned int lowb_9 = (u_1_9 ^ (unsigned int)((int)u_1_9 >> 31) & 511) & 511;
                    key_8 = 536870912 | lowb_9 << 9 | (unsigned int)(t0_3 + 9);
                }
                kb2o[9] = __uint_as_float(key_8);
                float sc_41 = __uint_as_float(cb[10]);
                sc_41 = fmaxf(sc_41, -1.7014118e+38f);
                sc_41 = fminf(sc_41, 1.7014118e+38f);
                unsigned int u_1_10 = __as_u32(sc_41);
                unsigned int cls_1_10 = u_1_10 & 4294966784u;
                unsigned int key_9 = 0;
                if ((qf < __uint_as_float(cls_1_10) || __uint_as_float(cls_1_10) == qf && cls_1_10 < qc) && cls_1_10 < 4278190080u) {
                    key_9 = 1073741824 | (unsigned int)(t0_3 + 10);
                }
                if (cls_1_10 == qc) {
                    unsigned int lowb_10 = (u_1_10 ^ (unsigned int)((int)u_1_10 >> 31) & 511) & 511;
                    key_9 = 536870912 | lowb_10 << 9 | (unsigned int)(t0_3 + 10);
                }
                kb2o[10] = __uint_as_float(key_9);
                float sc_42 = __uint_as_float(cb[11]);
                sc_42 = fmaxf(sc_42, -1.7014118e+38f);
                sc_42 = fminf(sc_42, 1.7014118e+38f);
                unsigned int u_1_11 = __as_u32(sc_42);
                unsigned int cls_1_11 = u_1_11 & 4294966784u;
                unsigned int key_10 = 0;
                if ((qf < __uint_as_float(cls_1_11) || __uint_as_float(cls_1_11) == qf && cls_1_11 < qc) && cls_1_11 < 4278190080u) {
                    key_10 = 1073741824 | (unsigned int)(t0_3 + 11);
                }
                if (cls_1_11 == qc) {
                    unsigned int lowb_11 = (u_1_11 ^ (unsigned int)((int)u_1_11 >> 31) & 511) & 511;
                    key_10 = 536870912 | lowb_11 << 9 | (unsigned int)(t0_3 + 11);
                }
                kb2o[11] = __uint_as_float(key_10);
                float sc_43 = __uint_as_float(cb[12]);
                sc_43 = fmaxf(sc_43, -1.7014118e+38f);
                sc_43 = fminf(sc_43, 1.7014118e+38f);
                unsigned int u_1_12 = __as_u32(sc_43);
                unsigned int cls_1_12 = u_1_12 & 4294966784u;
                unsigned int key_11 = 0;
                if ((qf < __uint_as_float(cls_1_12) || __uint_as_float(cls_1_12) == qf && cls_1_12 < qc) && cls_1_12 < 4278190080u) {
                    key_11 = 1073741824 | (unsigned int)(t0_3 + 12);
                }
                if (cls_1_12 == qc) {
                    unsigned int lowb_12 = (u_1_12 ^ (unsigned int)((int)u_1_12 >> 31) & 511) & 511;
                    key_11 = 536870912 | lowb_12 << 9 | (unsigned int)(t0_3 + 12);
                }
                kb2o[12] = __uint_as_float(key_11);
                float sc_44 = __uint_as_float(cb[13]);
                sc_44 = fmaxf(sc_44, -1.7014118e+38f);
                sc_44 = fminf(sc_44, 1.7014118e+38f);
                unsigned int u_1_13 = __as_u32(sc_44);
                unsigned int cls_1_13 = u_1_13 & 4294966784u;
                unsigned int key_12 = 0;
                if ((qf < __uint_as_float(cls_1_13) || __uint_as_float(cls_1_13) == qf && cls_1_13 < qc) && cls_1_13 < 4278190080u) {
                    key_12 = 1073741824 | (unsigned int)(t0_3 + 13);
                }
                if (cls_1_13 == qc) {
                    unsigned int lowb_13 = (u_1_13 ^ (unsigned int)((int)u_1_13 >> 31) & 511) & 511;
                    key_12 = 536870912 | lowb_13 << 9 | (unsigned int)(t0_3 + 13);
                }
                kb2o[13] = __uint_as_float(key_12);
                float sc_45 = __uint_as_float(cb[14]);
                sc_45 = fmaxf(sc_45, -1.7014118e+38f);
                sc_45 = fminf(sc_45, 1.7014118e+38f);
                unsigned int u_1_14 = __as_u32(sc_45);
                unsigned int cls_1_14 = u_1_14 & 4294966784u;
                unsigned int key_13 = 0;
                if ((qf < __uint_as_float(cls_1_14) || __uint_as_float(cls_1_14) == qf && cls_1_14 < qc) && cls_1_14 < 4278190080u) {
                    key_13 = 1073741824 | (unsigned int)(t0_3 + 14);
                }
                if (cls_1_14 == qc) {
                    unsigned int lowb_14 = (u_1_14 ^ (unsigned int)((int)u_1_14 >> 31) & 511) & 511;
                    key_13 = 536870912 | lowb_14 << 9 | (unsigned int)(t0_3 + 14);
                }
                kb2o[14] = __uint_as_float(key_13);
                float sc_46 = __uint_as_float(cb[15]);
                sc_46 = fmaxf(sc_46, -1.7014118e+38f);
                sc_46 = fminf(sc_46, 1.7014118e+38f);
                unsigned int u_1_15 = __as_u32(sc_46);
                unsigned int cls_1_15 = u_1_15 & 4294966784u;
                unsigned int key_14 = 0;
                if ((qf < __uint_as_float(cls_1_15) || __uint_as_float(cls_1_15) == qf && cls_1_15 < qc) && cls_1_15 < 4278190080u) {
                    key_14 = 1073741824 | (unsigned int)(t0_3 + 15);
                }
                if (cls_1_15 == qc) {
                    unsigned int lowb_15 = (u_1_15 ^ (unsigned int)((int)u_1_15 >> 31) & 511) & 511;
                    key_14 = 536870912 | lowb_15 << 9 | (unsigned int)(t0_3 + 15);
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
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            float r_0 = neg_inf;
            float V_0[8];
            int s0_8 = (sg * 16 * 8 + cg) * 17;
            int s1_8 = ((sg * 16 + 8) * 8 + cg) * 17;
            float x0_8 = pub[s0_8 + ln];
            float y0_8 = pub[s1_8 + lnr];
            float lo0_8 = fminf(x0_8, y0_8);
            r_0 = fmaxf(r_0, lo0_8);
            float hi0_8 = fmaxf(x0_8, y0_8);
            float cur_14 = hi0_8;
            float pv_60 = __shfl_xor_sync(0xFFFFFFFF, cur_14, 8);
            float hi_60 = fmaxf(cur_14, pv_60);
            float lo_60 = fminf(cur_14, pv_60);
            cur_14 = ((up[0] != 0) ? hi_60 : lo_60);
            float pv_61 = __shfl_xor_sync(0xFFFFFFFF, cur_14, 4);
            float hi_61 = fmaxf(cur_14, pv_61);
            float lo_61 = fminf(cur_14, pv_61);
            cur_14 = ((up[1] != 0) ? hi_61 : lo_61);
            float pv_62 = __shfl_xor_sync(0xFFFFFFFF, cur_14, 2);
            float hi_62 = fmaxf(cur_14, pv_62);
            float lo_62 = fminf(cur_14, pv_62);
            cur_14 = ((up[2] != 0) ? hi_62 : lo_62);
            float pv_63 = __shfl_xor_sync(0xFFFFFFFF, cur_14, 1);
            float hi_63 = fmaxf(cur_14, pv_63);
            float lo_63 = fminf(cur_14, pv_63);
            cur_14 = ((up[3] != 0) ? hi_63 : lo_63);
            V_0[0] = cur_14;
            int s0_9 = ((sg * 16 + 1) * 8 + cg) * 17;
            int s1_9 = ((sg * 16 + 1 + 8) * 8 + cg) * 17;
            float x0_9 = pub[s0_9 + ln];
            float y0_9 = pub[s1_9 + lnr];
            float lo0_9 = fminf(x0_9, y0_9);
            r_0 = fmaxf(r_0, lo0_9);
            float hi0_9 = fmaxf(x0_9, y0_9);
            float cur_15 = hi0_9;
            float pv_64 = __shfl_xor_sync(0xFFFFFFFF, cur_15, 8);
            float hi_64 = fmaxf(cur_15, pv_64);
            float lo_64 = fminf(cur_15, pv_64);
            cur_15 = ((up[0] != 0) ? hi_64 : lo_64);
            float pv_65 = __shfl_xor_sync(0xFFFFFFFF, cur_15, 4);
            float hi_65 = fmaxf(cur_15, pv_65);
            float lo_65 = fminf(cur_15, pv_65);
            cur_15 = ((up[1] != 0) ? hi_65 : lo_65);
            float pv_66 = __shfl_xor_sync(0xFFFFFFFF, cur_15, 2);
            float hi_66 = fmaxf(cur_15, pv_66);
            float lo_66 = fminf(cur_15, pv_66);
            cur_15 = ((up[2] != 0) ? hi_66 : lo_66);
            float pv_67 = __shfl_xor_sync(0xFFFFFFFF, cur_15, 1);
            float hi_67 = fmaxf(cur_15, pv_67);
            float lo_67 = fminf(cur_15, pv_67);
            cur_15 = ((up[3] != 0) ? hi_67 : lo_67);
            V_0[1] = cur_15;
            int s0_10 = ((sg * 16 + 2) * 8 + cg) * 17;
            int s1_10 = ((sg * 16 + 2 + 8) * 8 + cg) * 17;
            float x0_10 = pub[s0_10 + ln];
            float y0_10 = pub[s1_10 + lnr];
            float lo0_10 = fminf(x0_10, y0_10);
            r_0 = fmaxf(r_0, lo0_10);
            float hi0_10 = fmaxf(x0_10, y0_10);
            float cur_16 = hi0_10;
            float pv_68 = __shfl_xor_sync(0xFFFFFFFF, cur_16, 8);
            float hi_68 = fmaxf(cur_16, pv_68);
            float lo_68 = fminf(cur_16, pv_68);
            cur_16 = ((up[0] != 0) ? hi_68 : lo_68);
            float pv_69 = __shfl_xor_sync(0xFFFFFFFF, cur_16, 4);
            float hi_69 = fmaxf(cur_16, pv_69);
            float lo_69 = fminf(cur_16, pv_69);
            cur_16 = ((up[1] != 0) ? hi_69 : lo_69);
            float pv_70 = __shfl_xor_sync(0xFFFFFFFF, cur_16, 2);
            float hi_70 = fmaxf(cur_16, pv_70);
            float lo_70 = fminf(cur_16, pv_70);
            cur_16 = ((up[2] != 0) ? hi_70 : lo_70);
            float pv_71 = __shfl_xor_sync(0xFFFFFFFF, cur_16, 1);
            float hi_71 = fmaxf(cur_16, pv_71);
            float lo_71 = fminf(cur_16, pv_71);
            cur_16 = ((up[3] != 0) ? hi_71 : lo_71);
            V_0[2] = cur_16;
            int s0_11 = ((sg * 16 + 3) * 8 + cg) * 17;
            int s1_11 = ((sg * 16 + 3 + 8) * 8 + cg) * 17;
            float x0_11 = pub[s0_11 + ln];
            float y0_11 = pub[s1_11 + lnr];
            float lo0_11 = fminf(x0_11, y0_11);
            r_0 = fmaxf(r_0, lo0_11);
            float hi0_11 = fmaxf(x0_11, y0_11);
            float cur_17 = hi0_11;
            float pv_72 = __shfl_xor_sync(0xFFFFFFFF, cur_17, 8);
            float hi_72 = fmaxf(cur_17, pv_72);
            float lo_72 = fminf(cur_17, pv_72);
            cur_17 = ((up[0] != 0) ? hi_72 : lo_72);
            float pv_73 = __shfl_xor_sync(0xFFFFFFFF, cur_17, 4);
            float hi_73 = fmaxf(cur_17, pv_73);
            float lo_73 = fminf(cur_17, pv_73);
            cur_17 = ((up[1] != 0) ? hi_73 : lo_73);
            float pv_74 = __shfl_xor_sync(0xFFFFFFFF, cur_17, 2);
            float hi_74 = fmaxf(cur_17, pv_74);
            float lo_74 = fminf(cur_17, pv_74);
            cur_17 = ((up[2] != 0) ? hi_74 : lo_74);
            float pv_75 = __shfl_xor_sync(0xFFFFFFFF, cur_17, 1);
            float hi_75 = fmaxf(cur_17, pv_75);
            float lo_75 = fminf(cur_17, pv_75);
            cur_17 = ((up[3] != 0) ? hi_75 : lo_75);
            V_0[3] = cur_17;
            int s0_12 = ((sg * 16 + 4) * 8 + cg) * 17;
            int s1_12 = ((sg * 16 + 4 + 8) * 8 + cg) * 17;
            float x0_12 = pub[s0_12 + ln];
            float y0_12 = pub[s1_12 + lnr];
            float lo0_12 = fminf(x0_12, y0_12);
            r_0 = fmaxf(r_0, lo0_12);
            float hi0_12 = fmaxf(x0_12, y0_12);
            float cur_18 = hi0_12;
            float pv_76 = __shfl_xor_sync(0xFFFFFFFF, cur_18, 8);
            float hi_76 = fmaxf(cur_18, pv_76);
            float lo_76 = fminf(cur_18, pv_76);
            cur_18 = ((up[0] != 0) ? hi_76 : lo_76);
            float pv_77 = __shfl_xor_sync(0xFFFFFFFF, cur_18, 4);
            float hi_77 = fmaxf(cur_18, pv_77);
            float lo_77 = fminf(cur_18, pv_77);
            cur_18 = ((up[1] != 0) ? hi_77 : lo_77);
            float pv_78 = __shfl_xor_sync(0xFFFFFFFF, cur_18, 2);
            float hi_78 = fmaxf(cur_18, pv_78);
            float lo_78 = fminf(cur_18, pv_78);
            cur_18 = ((up[2] != 0) ? hi_78 : lo_78);
            float pv_79 = __shfl_xor_sync(0xFFFFFFFF, cur_18, 1);
            float hi_79 = fmaxf(cur_18, pv_79);
            float lo_79 = fminf(cur_18, pv_79);
            cur_18 = ((up[3] != 0) ? hi_79 : lo_79);
            V_0[4] = cur_18;
            int s0_13 = ((sg * 16 + 5) * 8 + cg) * 17;
            int s1_13 = ((sg * 16 + 5 + 8) * 8 + cg) * 17;
            float x0_13 = pub[s0_13 + ln];
            float y0_13 = pub[s1_13 + lnr];
            float lo0_13 = fminf(x0_13, y0_13);
            r_0 = fmaxf(r_0, lo0_13);
            float hi0_13 = fmaxf(x0_13, y0_13);
            float cur_19 = hi0_13;
            float pv_80 = __shfl_xor_sync(0xFFFFFFFF, cur_19, 8);
            float hi_80 = fmaxf(cur_19, pv_80);
            float lo_80 = fminf(cur_19, pv_80);
            cur_19 = ((up[0] != 0) ? hi_80 : lo_80);
            float pv_81 = __shfl_xor_sync(0xFFFFFFFF, cur_19, 4);
            float hi_81 = fmaxf(cur_19, pv_81);
            float lo_81 = fminf(cur_19, pv_81);
            cur_19 = ((up[1] != 0) ? hi_81 : lo_81);
            float pv_82 = __shfl_xor_sync(0xFFFFFFFF, cur_19, 2);
            float hi_82 = fmaxf(cur_19, pv_82);
            float lo_82 = fminf(cur_19, pv_82);
            cur_19 = ((up[2] != 0) ? hi_82 : lo_82);
            float pv_83 = __shfl_xor_sync(0xFFFFFFFF, cur_19, 1);
            float hi_83 = fmaxf(cur_19, pv_83);
            float lo_83 = fminf(cur_19, pv_83);
            cur_19 = ((up[3] != 0) ? hi_83 : lo_83);
            V_0[5] = cur_19;
            int s0_14 = ((sg * 16 + 6) * 8 + cg) * 17;
            int s1_14 = ((sg * 16 + 6 + 8) * 8 + cg) * 17;
            float x0_14 = pub[s0_14 + ln];
            float y0_14 = pub[s1_14 + lnr];
            float lo0_14 = fminf(x0_14, y0_14);
            r_0 = fmaxf(r_0, lo0_14);
            float hi0_14 = fmaxf(x0_14, y0_14);
            float cur_20 = hi0_14;
            float pv_84 = __shfl_xor_sync(0xFFFFFFFF, cur_20, 8);
            float hi_84 = fmaxf(cur_20, pv_84);
            float lo_84 = fminf(cur_20, pv_84);
            cur_20 = ((up[0] != 0) ? hi_84 : lo_84);
            float pv_85 = __shfl_xor_sync(0xFFFFFFFF, cur_20, 4);
            float hi_85 = fmaxf(cur_20, pv_85);
            float lo_85 = fminf(cur_20, pv_85);
            cur_20 = ((up[1] != 0) ? hi_85 : lo_85);
            float pv_86 = __shfl_xor_sync(0xFFFFFFFF, cur_20, 2);
            float hi_86 = fmaxf(cur_20, pv_86);
            float lo_86 = fminf(cur_20, pv_86);
            cur_20 = ((up[2] != 0) ? hi_86 : lo_86);
            float pv_87 = __shfl_xor_sync(0xFFFFFFFF, cur_20, 1);
            float hi_87 = fmaxf(cur_20, pv_87);
            float lo_87 = fminf(cur_20, pv_87);
            cur_20 = ((up[3] != 0) ? hi_87 : lo_87);
            V_0[6] = cur_20;
            int s0_15 = ((sg * 16 + 7) * 8 + cg) * 17;
            int s1_15 = ((sg * 16 + 7 + 8) * 8 + cg) * 17;
            float x0_15 = pub[s0_15 + ln];
            float y0_15 = pub[s1_15 + lnr];
            float lo0_15 = fminf(x0_15, y0_15);
            r_0 = fmaxf(r_0, lo0_15);
            float hi0_15 = fmaxf(x0_15, y0_15);
            float cur_21 = hi0_15;
            float pv_88 = __shfl_xor_sync(0xFFFFFFFF, cur_21, 8);
            float hi_88 = fmaxf(cur_21, pv_88);
            float lo_88 = fminf(cur_21, pv_88);
            cur_21 = ((up[0] != 0) ? hi_88 : lo_88);
            float pv_89 = __shfl_xor_sync(0xFFFFFFFF, cur_21, 4);
            float hi_89 = fmaxf(cur_21, pv_89);
            float lo_89 = fminf(cur_21, pv_89);
            cur_21 = ((up[1] != 0) ? hi_89 : lo_89);
            float pv_90 = __shfl_xor_sync(0xFFFFFFFF, cur_21, 2);
            float hi_90 = fmaxf(cur_21, pv_90);
            float lo_90 = fminf(cur_21, pv_90);
            cur_21 = ((up[2] != 0) ? hi_90 : lo_90);
            float pv_91 = __shfl_xor_sync(0xFFFFFFFF, cur_21, 1);
            float hi_91 = fmaxf(cur_21, pv_91);
            float lo_91 = fminf(cur_21, pv_91);
            cur_21 = ((up[3] != 0) ? hi_91 : lo_91);
            V_0[7] = cur_21;
            float rs_0 = pub[((sg * 16 + ln) * 8 + cg) * 17 + 16];
            r_0 = fmaxf(r_0, rs_0);
            float y1_7 = __shfl_xor_sync(0xFFFFFFFF, V_0[4], 15);
            float lo1_7 = fminf(V_0[0], y1_7);
            r_0 = fmaxf(r_0, lo1_7);
            float hi1_7 = fmaxf(V_0[0], y1_7);
            float cur_22 = hi1_7;
            float pv_92 = __shfl_xor_sync(0xFFFFFFFF, cur_22, 8);
            float hi_92 = fmaxf(cur_22, pv_92);
            float lo_92 = fminf(cur_22, pv_92);
            cur_22 = ((up[0] != 0) ? hi_92 : lo_92);
            float pv_93 = __shfl_xor_sync(0xFFFFFFFF, cur_22, 4);
            float hi_93 = fmaxf(cur_22, pv_93);
            float lo_93 = fminf(cur_22, pv_93);
            cur_22 = ((up[1] != 0) ? hi_93 : lo_93);
            float pv_94 = __shfl_xor_sync(0xFFFFFFFF, cur_22, 2);
            float hi_94 = fmaxf(cur_22, pv_94);
            float lo_94 = fminf(cur_22, pv_94);
            cur_22 = ((up[2] != 0) ? hi_94 : lo_94);
            float pv_95 = __shfl_xor_sync(0xFFFFFFFF, cur_22, 1);
            float hi_95 = fmaxf(cur_22, pv_95);
            float lo_95 = fminf(cur_22, pv_95);
            cur_22 = ((up[3] != 0) ? hi_95 : lo_95);
            V_0[0] = cur_22;
            float y1_8 = __shfl_xor_sync(0xFFFFFFFF, V_0[5], 15);
            float lo1_8 = fminf(V_0[1], y1_8);
            r_0 = fmaxf(r_0, lo1_8);
            float hi1_8 = fmaxf(V_0[1], y1_8);
            float cur_23 = hi1_8;
            float pv_96 = __shfl_xor_sync(0xFFFFFFFF, cur_23, 8);
            float hi_96 = fmaxf(cur_23, pv_96);
            float lo_96 = fminf(cur_23, pv_96);
            cur_23 = ((up[0] != 0) ? hi_96 : lo_96);
            float pv_97 = __shfl_xor_sync(0xFFFFFFFF, cur_23, 4);
            float hi_97 = fmaxf(cur_23, pv_97);
            float lo_97 = fminf(cur_23, pv_97);
            cur_23 = ((up[1] != 0) ? hi_97 : lo_97);
            float pv_98 = __shfl_xor_sync(0xFFFFFFFF, cur_23, 2);
            float hi_98 = fmaxf(cur_23, pv_98);
            float lo_98 = fminf(cur_23, pv_98);
            cur_23 = ((up[2] != 0) ? hi_98 : lo_98);
            float pv_99 = __shfl_xor_sync(0xFFFFFFFF, cur_23, 1);
            float hi_99 = fmaxf(cur_23, pv_99);
            float lo_99 = fminf(cur_23, pv_99);
            cur_23 = ((up[3] != 0) ? hi_99 : lo_99);
            V_0[1] = cur_23;
            float y1_9 = __shfl_xor_sync(0xFFFFFFFF, V_0[6], 15);
            float lo1_9 = fminf(V_0[2], y1_9);
            r_0 = fmaxf(r_0, lo1_9);
            float hi1_9 = fmaxf(V_0[2], y1_9);
            float cur_24 = hi1_9;
            float pv_100 = __shfl_xor_sync(0xFFFFFFFF, cur_24, 8);
            float hi_100 = fmaxf(cur_24, pv_100);
            float lo_100 = fminf(cur_24, pv_100);
            cur_24 = ((up[0] != 0) ? hi_100 : lo_100);
            float pv_101 = __shfl_xor_sync(0xFFFFFFFF, cur_24, 4);
            float hi_101 = fmaxf(cur_24, pv_101);
            float lo_101 = fminf(cur_24, pv_101);
            cur_24 = ((up[1] != 0) ? hi_101 : lo_101);
            float pv_102 = __shfl_xor_sync(0xFFFFFFFF, cur_24, 2);
            float hi_102 = fmaxf(cur_24, pv_102);
            float lo_102 = fminf(cur_24, pv_102);
            cur_24 = ((up[2] != 0) ? hi_102 : lo_102);
            float pv_103 = __shfl_xor_sync(0xFFFFFFFF, cur_24, 1);
            float hi_103 = fmaxf(cur_24, pv_103);
            float lo_103 = fminf(cur_24, pv_103);
            cur_24 = ((up[3] != 0) ? hi_103 : lo_103);
            V_0[2] = cur_24;
            float y1_10 = __shfl_xor_sync(0xFFFFFFFF, V_0[7], 15);
            float lo1_10 = fminf(V_0[3], y1_10);
            r_0 = fmaxf(r_0, lo1_10);
            float hi1_10 = fmaxf(V_0[3], y1_10);
            float cur_25 = hi1_10;
            float pv_104 = __shfl_xor_sync(0xFFFFFFFF, cur_25, 8);
            float hi_104 = fmaxf(cur_25, pv_104);
            float lo_104 = fminf(cur_25, pv_104);
            cur_25 = ((up[0] != 0) ? hi_104 : lo_104);
            float pv_105 = __shfl_xor_sync(0xFFFFFFFF, cur_25, 4);
            float hi_105 = fmaxf(cur_25, pv_105);
            float lo_105 = fminf(cur_25, pv_105);
            cur_25 = ((up[1] != 0) ? hi_105 : lo_105);
            float pv_106 = __shfl_xor_sync(0xFFFFFFFF, cur_25, 2);
            float hi_106 = fmaxf(cur_25, pv_106);
            float lo_106 = fminf(cur_25, pv_106);
            cur_25 = ((up[2] != 0) ? hi_106 : lo_106);
            float pv_107 = __shfl_xor_sync(0xFFFFFFFF, cur_25, 1);
            float hi_107 = fmaxf(cur_25, pv_107);
            float lo_107 = fminf(cur_25, pv_107);
            cur_25 = ((up[3] != 0) ? hi_107 : lo_107);
            V_0[3] = cur_25;
            float y1_11 = __shfl_xor_sync(0xFFFFFFFF, V_0[2], 15);
            float lo1_11 = fminf(V_0[0], y1_11);
            r_0 = fmaxf(r_0, lo1_11);
            float hi1_11 = fmaxf(V_0[0], y1_11);
            float cur_26 = hi1_11;
            float pv_108 = __shfl_xor_sync(0xFFFFFFFF, cur_26, 8);
            float hi_108 = fmaxf(cur_26, pv_108);
            float lo_108 = fminf(cur_26, pv_108);
            cur_26 = ((up[0] != 0) ? hi_108 : lo_108);
            float pv_109 = __shfl_xor_sync(0xFFFFFFFF, cur_26, 4);
            float hi_109 = fmaxf(cur_26, pv_109);
            float lo_109 = fminf(cur_26, pv_109);
            cur_26 = ((up[1] != 0) ? hi_109 : lo_109);
            float pv_110 = __shfl_xor_sync(0xFFFFFFFF, cur_26, 2);
            float hi_110 = fmaxf(cur_26, pv_110);
            float lo_110 = fminf(cur_26, pv_110);
            cur_26 = ((up[2] != 0) ? hi_110 : lo_110);
            float pv_111 = __shfl_xor_sync(0xFFFFFFFF, cur_26, 1);
            float hi_111 = fmaxf(cur_26, pv_111);
            float lo_111 = fminf(cur_26, pv_111);
            cur_26 = ((up[3] != 0) ? hi_111 : lo_111);
            V_0[0] = cur_26;
            float y1_12 = __shfl_xor_sync(0xFFFFFFFF, V_0[3], 15);
            float lo1_12 = fminf(V_0[1], y1_12);
            r_0 = fmaxf(r_0, lo1_12);
            float hi1_12 = fmaxf(V_0[1], y1_12);
            float cur_27 = hi1_12;
            float pv_112 = __shfl_xor_sync(0xFFFFFFFF, cur_27, 8);
            float hi_112 = fmaxf(cur_27, pv_112);
            float lo_112 = fminf(cur_27, pv_112);
            cur_27 = ((up[0] != 0) ? hi_112 : lo_112);
            float pv_113 = __shfl_xor_sync(0xFFFFFFFF, cur_27, 4);
            float hi_113 = fmaxf(cur_27, pv_113);
            float lo_113 = fminf(cur_27, pv_113);
            cur_27 = ((up[1] != 0) ? hi_113 : lo_113);
            float pv_114 = __shfl_xor_sync(0xFFFFFFFF, cur_27, 2);
            float hi_114 = fmaxf(cur_27, pv_114);
            float lo_114 = fminf(cur_27, pv_114);
            cur_27 = ((up[2] != 0) ? hi_114 : lo_114);
            float pv_115 = __shfl_xor_sync(0xFFFFFFFF, cur_27, 1);
            float hi_115 = fmaxf(cur_27, pv_115);
            float lo_115 = fminf(cur_27, pv_115);
            cur_27 = ((up[3] != 0) ? hi_115 : lo_115);
            V_0[1] = cur_27;
            float y1_13 = __shfl_xor_sync(0xFFFFFFFF, V_0[1], 15);
            float lo1_13 = fminf(V_0[0], y1_13);
            r_0 = fmaxf(r_0, lo1_13);
            float hi1_13 = fmaxf(V_0[0], y1_13);
            float cur_28 = hi1_13;
            float pv_116 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 8);
            float hi_116 = fmaxf(cur_28, pv_116);
            float lo_116 = fminf(cur_28, pv_116);
            cur_28 = ((up[0] != 0) ? hi_116 : lo_116);
            float pv_117 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 4);
            float hi_117 = fmaxf(cur_28, pv_117);
            float lo_117 = fminf(cur_28, pv_117);
            cur_28 = ((up[1] != 0) ? hi_117 : lo_117);
            float pv_118 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 2);
            float hi_118 = fmaxf(cur_28, pv_118);
            float lo_118 = fminf(cur_28, pv_118);
            cur_28 = ((up[2] != 0) ? hi_118 : lo_118);
            float pv_119 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 1);
            float hi_119 = fmaxf(cur_28, pv_119);
            float lo_119 = fminf(cur_28, pv_119);
            cur_28 = ((up[3] != 0) ? hi_119 : lo_119);
            V_0[0] = cur_28;
            float K_0 = V_0[0];
            int qb_0 = (sg * 8 + cg) * 32;
            q2[qb_0 + ln] = K_0;
            q2[qb_0 + 16 + ln] = r_0;
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            float k2o_0[1];
            if (tid_1 < 128) {
                float r2v_1 = neg_inf;
                r2v_1 = fmaxf(r2v_1, q2[cg * 32 + 16 + ln]);
                r2v_1 = fmaxf(r2v_1, q2[(8 + cg) * 32 + 16 + ln]);
                float V2_1[1];
                float x2_1 = q2[cg * 32 + ln];
                float y2_1 = q2[(8 + cg) * 32 + lnr];
                float lo2_1 = fminf(x2_1, y2_1);
                r2v_1 = fmaxf(r2v_1, lo2_1);
                float hi2_1 = fmaxf(x2_1, y2_1);
                V2_1[0] = hi2_1;
                k2o_0[0] = V2_1[0];
                K_0 = k2o_0[0];
                r_0 = r2v_1;
            }
            rr2x[0] = r_0;
            K2 = K_0;
        }
        if (tid_1 < 128) {
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
            long long obase = ((long long)ucol * (long long)num_heads + (long long)head) * 16;
            out[obase + (long long)rank] = val;
        }
    }
}

} // extern "C"
