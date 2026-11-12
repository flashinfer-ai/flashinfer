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
kernel_cake_hopper_msa_e3f7ae829be0044f3725(unsigned int* __restrict__ S, int* __restrict__ nvp, int* __restrict__ out, int tiles, int total_q, int num_heads, int num_chunks, int fb, int fe)
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

    // === Task calls (declaration order) ===
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
    int tb = w;
    unsigned int cbt[32];
    float kbt[32];
    long long p = cbase + (long long)tb * nq64;
    cbt[0] = 4286578688u;
    if (lim > tb) {
        cbt[0] = S[p];
    }
    cbt[1] = 4286578688u;
    if (lim > tb + 16) {
        cbt[1] = S[p + 16ll * nq64];
    }
    cbt[2] = 4286578688u;
    if (lim > tb + 32) {
        cbt[2] = S[p + 32ll * nq64];
    }
    cbt[3] = 4286578688u;
    if (lim > tb + 48) {
        cbt[3] = S[p + 48ll * nq64];
    }
    cbt[4] = 4286578688u;
    if (lim > tb + 64) {
        cbt[4] = S[p + 64ll * nq64];
    }
    cbt[5] = 4286578688u;
    if (lim > tb + 80) {
        cbt[5] = S[p + 80ll * nq64];
    }
    cbt[6] = 4286578688u;
    if (lim > tb + 96) {
        cbt[6] = S[p + 96ll * nq64];
    }
    cbt[7] = 4286578688u;
    if (lim > tb + 112) {
        cbt[7] = S[p + 112ll * nq64];
    }
    cbt[8] = 4286578688u;
    if (lim > tb + 128) {
        cbt[8] = S[p + 128ll * nq64];
    }
    cbt[9] = 4286578688u;
    if (lim > tb + 144) {
        cbt[9] = S[p + 144ll * nq64];
    }
    cbt[10] = 4286578688u;
    if (lim > tb + 160) {
        cbt[10] = S[p + 160ll * nq64];
    }
    cbt[11] = 4286578688u;
    if (lim > tb + 176) {
        cbt[11] = S[p + 176ll * nq64];
    }
    cbt[12] = 4286578688u;
    if (lim > tb + 192) {
        cbt[12] = S[p + 192ll * nq64];
    }
    cbt[13] = 4286578688u;
    if (lim > tb + 208) {
        cbt[13] = S[p + 208ll * nq64];
    }
    cbt[14] = 4286578688u;
    if (lim > tb + 224) {
        cbt[14] = S[p + 224ll * nq64];
    }
    cbt[15] = 4286578688u;
    if (lim > tb + 240) {
        cbt[15] = S[p + 240ll * nq64];
    }
    cbt[16] = 4286578688u;
    if (lim > tb + 256) {
        cbt[16] = S[p + 256ll * nq64];
    }
    cbt[17] = 4286578688u;
    if (lim > tb + 272) {
        cbt[17] = S[p + 272ll * nq64];
    }
    cbt[18] = 4286578688u;
    if (lim > tb + 288) {
        cbt[18] = S[p + 288ll * nq64];
    }
    cbt[19] = 4286578688u;
    if (lim > tb + 304) {
        cbt[19] = S[p + 304ll * nq64];
    }
    cbt[20] = 4286578688u;
    if (lim > tb + 320) {
        cbt[20] = S[p + 320ll * nq64];
    }
    cbt[21] = 4286578688u;
    if (lim > tb + 336) {
        cbt[21] = S[p + 336ll * nq64];
    }
    cbt[22] = 4286578688u;
    if (lim > tb + 352) {
        cbt[22] = S[p + 352ll * nq64];
    }
    cbt[23] = 4286578688u;
    if (lim > tb + 368) {
        cbt[23] = S[p + 368ll * nq64];
    }
    cbt[24] = 4286578688u;
    if (lim > tb + 384) {
        cbt[24] = S[p + 384ll * nq64];
    }
    cbt[25] = 4286578688u;
    if (lim > tb + 400) {
        cbt[25] = S[p + 400ll * nq64];
    }
    cbt[26] = 4286578688u;
    if (lim > tb + 416) {
        cbt[26] = S[p + 416ll * nq64];
    }
    cbt[27] = 4286578688u;
    if (lim > tb + 432) {
        cbt[27] = S[p + 432ll * nq64];
    }
    cbt[28] = 4286578688u;
    if (lim > tb + 448) {
        cbt[28] = S[p + 448ll * nq64];
    }
    cbt[29] = 4286578688u;
    if (lim > tb + 464) {
        cbt[29] = S[p + 464ll * nq64];
    }
    cbt[30] = 4286578688u;
    if (lim > tb + 480) {
        cbt[30] = S[p + 480ll * nq64];
    }
    cbt[31] = 4286578688u;
    if (lim > tb + 496) {
        cbt[31] = S[p + 496ll * nq64];
    }
    float sc = __uint_as_float(cbt[0]);
    sc = fmaxf(sc, -1.7014118e+38f);
    sc = fminf(sc, 1.7014118e+38f);
    unsigned int kbits = __as_u32(sc);
    if (cbt[0] == 4286578688u) {
        kbits = 4282384384u;
    }
    unsigned int key_2_0 = kbits & 4294966784u | (unsigned int)tb;
    kbt[0] = __uint_as_float(key_2_0);
    float sc_0 = __uint_as_float(cbt[1]);
    sc_0 = fmaxf(sc_0, -1.7014118e+38f);
    sc_0 = fminf(sc_0, 1.7014118e+38f);
    unsigned int kbits_0 = __as_u32(sc_0);
    if (cbt[1] == 4286578688u) {
        kbits_0 = 4282384384u;
    }
    unsigned int key_2_1 = kbits_0 & 4294966784u | (unsigned int)(tb + 16);
    kbt[1] = __uint_as_float(key_2_1);
    float sc_1 = __uint_as_float(cbt[2]);
    sc_1 = fmaxf(sc_1, -1.7014118e+38f);
    sc_1 = fminf(sc_1, 1.7014118e+38f);
    unsigned int kbits_1 = __as_u32(sc_1);
    if (cbt[2] == 4286578688u) {
        kbits_1 = 4282384384u;
    }
    unsigned int key_2_2 = kbits_1 & 4294966784u | (unsigned int)(tb + 32);
    kbt[2] = __uint_as_float(key_2_2);
    float sc_2 = __uint_as_float(cbt[3]);
    sc_2 = fmaxf(sc_2, -1.7014118e+38f);
    sc_2 = fminf(sc_2, 1.7014118e+38f);
    unsigned int kbits_2 = __as_u32(sc_2);
    if (cbt[3] == 4286578688u) {
        kbits_2 = 4282384384u;
    }
    unsigned int key_2_3 = kbits_2 & 4294966784u | (unsigned int)(tb + 48);
    kbt[3] = __uint_as_float(key_2_3);
    float sc_3 = __uint_as_float(cbt[4]);
    sc_3 = fmaxf(sc_3, -1.7014118e+38f);
    sc_3 = fminf(sc_3, 1.7014118e+38f);
    unsigned int kbits_3 = __as_u32(sc_3);
    if (cbt[4] == 4286578688u) {
        kbits_3 = 4282384384u;
    }
    unsigned int key_2_4 = kbits_3 & 4294966784u | (unsigned int)(tb + 64);
    kbt[4] = __uint_as_float(key_2_4);
    float sc_4 = __uint_as_float(cbt[5]);
    sc_4 = fmaxf(sc_4, -1.7014118e+38f);
    sc_4 = fminf(sc_4, 1.7014118e+38f);
    unsigned int kbits_4 = __as_u32(sc_4);
    if (cbt[5] == 4286578688u) {
        kbits_4 = 4282384384u;
    }
    unsigned int key_2_5 = kbits_4 & 4294966784u | (unsigned int)(tb + 80);
    kbt[5] = __uint_as_float(key_2_5);
    float sc_5 = __uint_as_float(cbt[6]);
    sc_5 = fmaxf(sc_5, -1.7014118e+38f);
    sc_5 = fminf(sc_5, 1.7014118e+38f);
    unsigned int kbits_5 = __as_u32(sc_5);
    if (cbt[6] == 4286578688u) {
        kbits_5 = 4282384384u;
    }
    unsigned int key_2_6 = kbits_5 & 4294966784u | (unsigned int)(tb + 96);
    kbt[6] = __uint_as_float(key_2_6);
    float sc_6 = __uint_as_float(cbt[7]);
    sc_6 = fmaxf(sc_6, -1.7014118e+38f);
    sc_6 = fminf(sc_6, 1.7014118e+38f);
    unsigned int kbits_6 = __as_u32(sc_6);
    if (cbt[7] == 4286578688u) {
        kbits_6 = 4282384384u;
    }
    unsigned int key_2_7 = kbits_6 & 4294966784u | (unsigned int)(tb + 112);
    kbt[7] = __uint_as_float(key_2_7);
    float sc_7 = __uint_as_float(cbt[8]);
    sc_7 = fmaxf(sc_7, -1.7014118e+38f);
    sc_7 = fminf(sc_7, 1.7014118e+38f);
    unsigned int kbits_7 = __as_u32(sc_7);
    if (cbt[8] == 4286578688u) {
        kbits_7 = 4282384384u;
    }
    unsigned int key_2_8 = kbits_7 & 4294966784u | (unsigned int)(tb + 128);
    kbt[8] = __uint_as_float(key_2_8);
    float sc_8 = __uint_as_float(cbt[9]);
    sc_8 = fmaxf(sc_8, -1.7014118e+38f);
    sc_8 = fminf(sc_8, 1.7014118e+38f);
    unsigned int kbits_8 = __as_u32(sc_8);
    if (cbt[9] == 4286578688u) {
        kbits_8 = 4282384384u;
    }
    unsigned int key_2_9 = kbits_8 & 4294966784u | (unsigned int)(tb + 144);
    kbt[9] = __uint_as_float(key_2_9);
    float sc_9 = __uint_as_float(cbt[10]);
    sc_9 = fmaxf(sc_9, -1.7014118e+38f);
    sc_9 = fminf(sc_9, 1.7014118e+38f);
    unsigned int kbits_9 = __as_u32(sc_9);
    if (cbt[10] == 4286578688u) {
        kbits_9 = 4282384384u;
    }
    unsigned int key_2_10 = kbits_9 & 4294966784u | (unsigned int)(tb + 160);
    kbt[10] = __uint_as_float(key_2_10);
    float sc_10 = __uint_as_float(cbt[11]);
    sc_10 = fmaxf(sc_10, -1.7014118e+38f);
    sc_10 = fminf(sc_10, 1.7014118e+38f);
    unsigned int kbits_10 = __as_u32(sc_10);
    if (cbt[11] == 4286578688u) {
        kbits_10 = 4282384384u;
    }
    unsigned int key_2_11 = kbits_10 & 4294966784u | (unsigned int)(tb + 176);
    kbt[11] = __uint_as_float(key_2_11);
    float sc_11 = __uint_as_float(cbt[12]);
    sc_11 = fmaxf(sc_11, -1.7014118e+38f);
    sc_11 = fminf(sc_11, 1.7014118e+38f);
    unsigned int kbits_11 = __as_u32(sc_11);
    if (cbt[12] == 4286578688u) {
        kbits_11 = 4282384384u;
    }
    unsigned int key_2_12 = kbits_11 & 4294966784u | (unsigned int)(tb + 192);
    kbt[12] = __uint_as_float(key_2_12);
    float sc_12 = __uint_as_float(cbt[13]);
    sc_12 = fmaxf(sc_12, -1.7014118e+38f);
    sc_12 = fminf(sc_12, 1.7014118e+38f);
    unsigned int kbits_12 = __as_u32(sc_12);
    if (cbt[13] == 4286578688u) {
        kbits_12 = 4282384384u;
    }
    unsigned int key_2_13 = kbits_12 & 4294966784u | (unsigned int)(tb + 208);
    kbt[13] = __uint_as_float(key_2_13);
    float sc_13 = __uint_as_float(cbt[14]);
    sc_13 = fmaxf(sc_13, -1.7014118e+38f);
    sc_13 = fminf(sc_13, 1.7014118e+38f);
    unsigned int kbits_13 = __as_u32(sc_13);
    if (cbt[14] == 4286578688u) {
        kbits_13 = 4282384384u;
    }
    unsigned int key_2_14 = kbits_13 & 4294966784u | (unsigned int)(tb + 224);
    kbt[14] = __uint_as_float(key_2_14);
    float sc_14 = __uint_as_float(cbt[15]);
    sc_14 = fmaxf(sc_14, -1.7014118e+38f);
    sc_14 = fminf(sc_14, 1.7014118e+38f);
    unsigned int kbits_14 = __as_u32(sc_14);
    if (cbt[15] == 4286578688u) {
        kbits_14 = 4282384384u;
    }
    unsigned int key_2_15 = kbits_14 & 4294966784u | (unsigned int)(tb + 240);
    kbt[15] = __uint_as_float(key_2_15);
    float sc_15 = __uint_as_float(cbt[16]);
    sc_15 = fmaxf(sc_15, -1.7014118e+38f);
    sc_15 = fminf(sc_15, 1.7014118e+38f);
    unsigned int kbits_15 = __as_u32(sc_15);
    if (cbt[16] == 4286578688u) {
        kbits_15 = 4282384384u;
    }
    unsigned int key_2_16 = kbits_15 & 4294966784u | (unsigned int)(tb + 256);
    kbt[16] = __uint_as_float(key_2_16);
    float sc_16 = __uint_as_float(cbt[17]);
    sc_16 = fmaxf(sc_16, -1.7014118e+38f);
    sc_16 = fminf(sc_16, 1.7014118e+38f);
    unsigned int kbits_16 = __as_u32(sc_16);
    if (cbt[17] == 4286578688u) {
        kbits_16 = 4282384384u;
    }
    unsigned int key_2_17 = kbits_16 & 4294966784u | (unsigned int)(tb + 272);
    kbt[17] = __uint_as_float(key_2_17);
    float sc_17 = __uint_as_float(cbt[18]);
    sc_17 = fmaxf(sc_17, -1.7014118e+38f);
    sc_17 = fminf(sc_17, 1.7014118e+38f);
    unsigned int kbits_17 = __as_u32(sc_17);
    if (cbt[18] == 4286578688u) {
        kbits_17 = 4282384384u;
    }
    unsigned int key_2_18 = kbits_17 & 4294966784u | (unsigned int)(tb + 288);
    kbt[18] = __uint_as_float(key_2_18);
    float sc_18 = __uint_as_float(cbt[19]);
    sc_18 = fmaxf(sc_18, -1.7014118e+38f);
    sc_18 = fminf(sc_18, 1.7014118e+38f);
    unsigned int kbits_18 = __as_u32(sc_18);
    if (cbt[19] == 4286578688u) {
        kbits_18 = 4282384384u;
    }
    unsigned int key_2_19 = kbits_18 & 4294966784u | (unsigned int)(tb + 304);
    kbt[19] = __uint_as_float(key_2_19);
    float sc_19 = __uint_as_float(cbt[20]);
    sc_19 = fmaxf(sc_19, -1.7014118e+38f);
    sc_19 = fminf(sc_19, 1.7014118e+38f);
    unsigned int kbits_19 = __as_u32(sc_19);
    if (cbt[20] == 4286578688u) {
        kbits_19 = 4282384384u;
    }
    unsigned int key_2_20 = kbits_19 & 4294966784u | (unsigned int)(tb + 320);
    kbt[20] = __uint_as_float(key_2_20);
    float sc_20 = __uint_as_float(cbt[21]);
    sc_20 = fmaxf(sc_20, -1.7014118e+38f);
    sc_20 = fminf(sc_20, 1.7014118e+38f);
    unsigned int kbits_20 = __as_u32(sc_20);
    if (cbt[21] == 4286578688u) {
        kbits_20 = 4282384384u;
    }
    unsigned int key_2_21 = kbits_20 & 4294966784u | (unsigned int)(tb + 336);
    kbt[21] = __uint_as_float(key_2_21);
    float sc_21 = __uint_as_float(cbt[22]);
    sc_21 = fmaxf(sc_21, -1.7014118e+38f);
    sc_21 = fminf(sc_21, 1.7014118e+38f);
    unsigned int kbits_21 = __as_u32(sc_21);
    if (cbt[22] == 4286578688u) {
        kbits_21 = 4282384384u;
    }
    unsigned int key_2_22 = kbits_21 & 4294966784u | (unsigned int)(tb + 352);
    kbt[22] = __uint_as_float(key_2_22);
    float sc_22 = __uint_as_float(cbt[23]);
    sc_22 = fmaxf(sc_22, -1.7014118e+38f);
    sc_22 = fminf(sc_22, 1.7014118e+38f);
    unsigned int kbits_22 = __as_u32(sc_22);
    if (cbt[23] == 4286578688u) {
        kbits_22 = 4282384384u;
    }
    unsigned int key_2_23 = kbits_22 & 4294966784u | (unsigned int)(tb + 368);
    kbt[23] = __uint_as_float(key_2_23);
    float sc_23 = __uint_as_float(cbt[24]);
    sc_23 = fmaxf(sc_23, -1.7014118e+38f);
    sc_23 = fminf(sc_23, 1.7014118e+38f);
    unsigned int kbits_23 = __as_u32(sc_23);
    if (cbt[24] == 4286578688u) {
        kbits_23 = 4282384384u;
    }
    unsigned int key_2_24 = kbits_23 & 4294966784u | (unsigned int)(tb + 384);
    kbt[24] = __uint_as_float(key_2_24);
    float sc_24 = __uint_as_float(cbt[25]);
    sc_24 = fmaxf(sc_24, -1.7014118e+38f);
    sc_24 = fminf(sc_24, 1.7014118e+38f);
    unsigned int kbits_24 = __as_u32(sc_24);
    if (cbt[25] == 4286578688u) {
        kbits_24 = 4282384384u;
    }
    unsigned int key_2_25 = kbits_24 & 4294966784u | (unsigned int)(tb + 400);
    kbt[25] = __uint_as_float(key_2_25);
    float sc_25 = __uint_as_float(cbt[26]);
    sc_25 = fmaxf(sc_25, -1.7014118e+38f);
    sc_25 = fminf(sc_25, 1.7014118e+38f);
    unsigned int kbits_25 = __as_u32(sc_25);
    if (cbt[26] == 4286578688u) {
        kbits_25 = 4282384384u;
    }
    unsigned int key_2_26 = kbits_25 & 4294966784u | (unsigned int)(tb + 416);
    kbt[26] = __uint_as_float(key_2_26);
    float sc_26 = __uint_as_float(cbt[27]);
    sc_26 = fmaxf(sc_26, -1.7014118e+38f);
    sc_26 = fminf(sc_26, 1.7014118e+38f);
    unsigned int kbits_26 = __as_u32(sc_26);
    if (cbt[27] == 4286578688u) {
        kbits_26 = 4282384384u;
    }
    unsigned int key_2_27 = kbits_26 & 4294966784u | (unsigned int)(tb + 432);
    kbt[27] = __uint_as_float(key_2_27);
    float sc_27 = __uint_as_float(cbt[28]);
    sc_27 = fmaxf(sc_27, -1.7014118e+38f);
    sc_27 = fminf(sc_27, 1.7014118e+38f);
    unsigned int kbits_27 = __as_u32(sc_27);
    if (cbt[28] == 4286578688u) {
        kbits_27 = 4282384384u;
    }
    unsigned int key_2_28 = kbits_27 & 4294966784u | (unsigned int)(tb + 448);
    kbt[28] = __uint_as_float(key_2_28);
    float sc_28 = __uint_as_float(cbt[29]);
    sc_28 = fmaxf(sc_28, -1.7014118e+38f);
    sc_28 = fminf(sc_28, 1.7014118e+38f);
    unsigned int kbits_28 = __as_u32(sc_28);
    if (cbt[29] == 4286578688u) {
        kbits_28 = 4282384384u;
    }
    unsigned int key_2_29 = kbits_28 & 4294966784u | (unsigned int)(tb + 464);
    kbt[29] = __uint_as_float(key_2_29);
    float sc_29 = __uint_as_float(cbt[30]);
    sc_29 = fmaxf(sc_29, -1.7014118e+38f);
    sc_29 = fminf(sc_29, 1.7014118e+38f);
    unsigned int kbits_29 = __as_u32(sc_29);
    if (cbt[30] == 4286578688u) {
        kbits_29 = 4282384384u;
    }
    unsigned int key_2_30 = kbits_29 & 4294966784u | (unsigned int)(tb + 480);
    kbt[30] = __uint_as_float(key_2_30);
    float sc_30 = __uint_as_float(cbt[31]);
    sc_30 = fmaxf(sc_30, -1.7014118e+38f);
    sc_30 = fminf(sc_30, 1.7014118e+38f);
    unsigned int kbits_30 = __as_u32(sc_30);
    if (cbt[31] == 4286578688u) {
        kbits_30 = 4282384384u;
    }
    unsigned int key_2_31 = kbits_30 & 4294966784u | (unsigned int)(tb + 496);
    kbt[31] = __uint_as_float(key_2_31);
    float rr = rej;
    float hi_1_0 = fmaxf(kbt[0], kbt[13]);
    float lo_1_0 = fminf(kbt[0], kbt[13]);
    kbt[0] = hi_1_0;
    kbt[13] = lo_1_0;
    float hi_1_1 = fmaxf(kbt[1], kbt[12]);
    float lo_1_1 = fminf(kbt[1], kbt[12]);
    kbt[1] = hi_1_1;
    kbt[12] = lo_1_1;
    float hi_1_2 = fmaxf(kbt[2], kbt[15]);
    float lo_1_2 = fminf(kbt[2], kbt[15]);
    kbt[2] = hi_1_2;
    kbt[15] = lo_1_2;
    float hi_1_3 = fmaxf(kbt[3], kbt[14]);
    float lo_1_3 = fminf(kbt[3], kbt[14]);
    kbt[3] = hi_1_3;
    kbt[14] = lo_1_3;
    float hi_1_4 = fmaxf(kbt[4], kbt[8]);
    float lo_1_4 = fminf(kbt[4], kbt[8]);
    kbt[4] = hi_1_4;
    kbt[8] = lo_1_4;
    float hi_1_5 = fmaxf(kbt[5], kbt[6]);
    float lo_1_5 = fminf(kbt[5], kbt[6]);
    kbt[5] = hi_1_5;
    kbt[6] = lo_1_5;
    float hi_1_6 = fmaxf(kbt[7], kbt[11]);
    float lo_1_6 = fminf(kbt[7], kbt[11]);
    kbt[7] = hi_1_6;
    kbt[11] = lo_1_6;
    float hi_1_7 = fmaxf(kbt[9], kbt[10]);
    float lo_1_7 = fminf(kbt[9], kbt[10]);
    kbt[9] = hi_1_7;
    kbt[10] = lo_1_7;
    float hi_1_8 = fmaxf(kbt[0], kbt[5]);
    float lo_1_8 = fminf(kbt[0], kbt[5]);
    kbt[0] = hi_1_8;
    kbt[5] = lo_1_8;
    float hi_1_9 = fmaxf(kbt[1], kbt[7]);
    float lo_1_9 = fminf(kbt[1], kbt[7]);
    kbt[1] = hi_1_9;
    kbt[7] = lo_1_9;
    float hi_1_10 = fmaxf(kbt[2], kbt[9]);
    float lo_1_10 = fminf(kbt[2], kbt[9]);
    kbt[2] = hi_1_10;
    kbt[9] = lo_1_10;
    float hi_1_11 = fmaxf(kbt[3], kbt[4]);
    float lo_1_11 = fminf(kbt[3], kbt[4]);
    kbt[3] = hi_1_11;
    kbt[4] = lo_1_11;
    float hi_1_12 = fmaxf(kbt[6], kbt[13]);
    float lo_1_12 = fminf(kbt[6], kbt[13]);
    kbt[6] = hi_1_12;
    kbt[13] = lo_1_12;
    float hi_1_13 = fmaxf(kbt[8], kbt[14]);
    float lo_1_13 = fminf(kbt[8], kbt[14]);
    kbt[8] = hi_1_13;
    kbt[14] = lo_1_13;
    float hi_1_14 = fmaxf(kbt[10], kbt[15]);
    float lo_1_14 = fminf(kbt[10], kbt[15]);
    kbt[10] = hi_1_14;
    kbt[15] = lo_1_14;
    float hi_1_15 = fmaxf(kbt[11], kbt[12]);
    float lo_1_15 = fminf(kbt[11], kbt[12]);
    kbt[11] = hi_1_15;
    kbt[12] = lo_1_15;
    float hi_1_16 = fmaxf(kbt[0], kbt[1]);
    float lo_1_16 = fminf(kbt[0], kbt[1]);
    kbt[0] = hi_1_16;
    kbt[1] = lo_1_16;
    float hi_1_17 = fmaxf(kbt[2], kbt[3]);
    float lo_1_17 = fminf(kbt[2], kbt[3]);
    kbt[2] = hi_1_17;
    kbt[3] = lo_1_17;
    float hi_1_18 = fmaxf(kbt[4], kbt[5]);
    float lo_1_18 = fminf(kbt[4], kbt[5]);
    kbt[4] = hi_1_18;
    kbt[5] = lo_1_18;
    float hi_1_19 = fmaxf(kbt[6], kbt[8]);
    float lo_1_19 = fminf(kbt[6], kbt[8]);
    kbt[6] = hi_1_19;
    kbt[8] = lo_1_19;
    float hi_1_20 = fmaxf(kbt[7], kbt[9]);
    float lo_1_20 = fminf(kbt[7], kbt[9]);
    kbt[7] = hi_1_20;
    kbt[9] = lo_1_20;
    float hi_1_21 = fmaxf(kbt[10], kbt[11]);
    float lo_1_21 = fminf(kbt[10], kbt[11]);
    kbt[10] = hi_1_21;
    kbt[11] = lo_1_21;
    float hi_1_22 = fmaxf(kbt[12], kbt[13]);
    float lo_1_22 = fminf(kbt[12], kbt[13]);
    kbt[12] = hi_1_22;
    kbt[13] = lo_1_22;
    float hi_1_23 = fmaxf(kbt[14], kbt[15]);
    float lo_1_23 = fminf(kbt[14], kbt[15]);
    kbt[14] = hi_1_23;
    kbt[15] = lo_1_23;
    float hi_1_24 = fmaxf(kbt[0], kbt[2]);
    float lo_1_24 = fminf(kbt[0], kbt[2]);
    kbt[0] = hi_1_24;
    kbt[2] = lo_1_24;
    float hi_1_25 = fmaxf(kbt[1], kbt[3]);
    float lo_1_25 = fminf(kbt[1], kbt[3]);
    kbt[1] = hi_1_25;
    kbt[3] = lo_1_25;
    float hi_1_26 = fmaxf(kbt[4], kbt[10]);
    float lo_1_26 = fminf(kbt[4], kbt[10]);
    kbt[4] = hi_1_26;
    kbt[10] = lo_1_26;
    float hi_1_27 = fmaxf(kbt[5], kbt[11]);
    float lo_1_27 = fminf(kbt[5], kbt[11]);
    kbt[5] = hi_1_27;
    kbt[11] = lo_1_27;
    float hi_1_28 = fmaxf(kbt[6], kbt[7]);
    float lo_1_28 = fminf(kbt[6], kbt[7]);
    kbt[6] = hi_1_28;
    kbt[7] = lo_1_28;
    float hi_1_29 = fmaxf(kbt[8], kbt[9]);
    float lo_1_29 = fminf(kbt[8], kbt[9]);
    kbt[8] = hi_1_29;
    kbt[9] = lo_1_29;
    float hi_1_30 = fmaxf(kbt[12], kbt[14]);
    float lo_1_30 = fminf(kbt[12], kbt[14]);
    kbt[12] = hi_1_30;
    kbt[14] = lo_1_30;
    float hi_1_31 = fmaxf(kbt[13], kbt[15]);
    float lo_1_31 = fminf(kbt[13], kbt[15]);
    kbt[13] = hi_1_31;
    kbt[15] = lo_1_31;
    float hi_1_32 = fmaxf(kbt[1], kbt[2]);
    float lo_1_32 = fminf(kbt[1], kbt[2]);
    kbt[1] = hi_1_32;
    kbt[2] = lo_1_32;
    float hi_1_33 = fmaxf(kbt[3], kbt[12]);
    float lo_1_33 = fminf(kbt[3], kbt[12]);
    kbt[3] = hi_1_33;
    kbt[12] = lo_1_33;
    float hi_1_34 = fmaxf(kbt[4], kbt[6]);
    float lo_1_34 = fminf(kbt[4], kbt[6]);
    kbt[4] = hi_1_34;
    kbt[6] = lo_1_34;
    float hi_1_35 = fmaxf(kbt[5], kbt[7]);
    float lo_1_35 = fminf(kbt[5], kbt[7]);
    kbt[5] = hi_1_35;
    kbt[7] = lo_1_35;
    float hi_1_36 = fmaxf(kbt[8], kbt[10]);
    float lo_1_36 = fminf(kbt[8], kbt[10]);
    kbt[8] = hi_1_36;
    kbt[10] = lo_1_36;
    float hi_1_37 = fmaxf(kbt[9], kbt[11]);
    float lo_1_37 = fminf(kbt[9], kbt[11]);
    kbt[9] = hi_1_37;
    kbt[11] = lo_1_37;
    float hi_1_38 = fmaxf(kbt[13], kbt[14]);
    float lo_1_38 = fminf(kbt[13], kbt[14]);
    kbt[13] = hi_1_38;
    kbt[14] = lo_1_38;
    float hi_1_39 = fmaxf(kbt[1], kbt[4]);
    float lo_1_39 = fminf(kbt[1], kbt[4]);
    kbt[1] = hi_1_39;
    kbt[4] = lo_1_39;
    float hi_1_40 = fmaxf(kbt[2], kbt[6]);
    float lo_1_40 = fminf(kbt[2], kbt[6]);
    kbt[2] = hi_1_40;
    kbt[6] = lo_1_40;
    float hi_1_41 = fmaxf(kbt[5], kbt[8]);
    float lo_1_41 = fminf(kbt[5], kbt[8]);
    kbt[5] = hi_1_41;
    kbt[8] = lo_1_41;
    float hi_1_42 = fmaxf(kbt[7], kbt[10]);
    float lo_1_42 = fminf(kbt[7], kbt[10]);
    kbt[7] = hi_1_42;
    kbt[10] = lo_1_42;
    float hi_1_43 = fmaxf(kbt[9], kbt[13]);
    float lo_1_43 = fminf(kbt[9], kbt[13]);
    kbt[9] = hi_1_43;
    kbt[13] = lo_1_43;
    float hi_1_44 = fmaxf(kbt[11], kbt[14]);
    float lo_1_44 = fminf(kbt[11], kbt[14]);
    kbt[11] = hi_1_44;
    kbt[14] = lo_1_44;
    float hi_1_45 = fmaxf(kbt[2], kbt[4]);
    float lo_1_45 = fminf(kbt[2], kbt[4]);
    kbt[2] = hi_1_45;
    kbt[4] = lo_1_45;
    float hi_1_46 = fmaxf(kbt[3], kbt[6]);
    float lo_1_46 = fminf(kbt[3], kbt[6]);
    kbt[3] = hi_1_46;
    kbt[6] = lo_1_46;
    float hi_1_47 = fmaxf(kbt[9], kbt[12]);
    float lo_1_47 = fminf(kbt[9], kbt[12]);
    kbt[9] = hi_1_47;
    kbt[12] = lo_1_47;
    float hi_1_48 = fmaxf(kbt[11], kbt[13]);
    float lo_1_48 = fminf(kbt[11], kbt[13]);
    kbt[11] = hi_1_48;
    kbt[13] = lo_1_48;
    float hi_1_49 = fmaxf(kbt[3], kbt[5]);
    float lo_1_49 = fminf(kbt[3], kbt[5]);
    kbt[3] = hi_1_49;
    kbt[5] = lo_1_49;
    float hi_1_50 = fmaxf(kbt[6], kbt[8]);
    float lo_1_50 = fminf(kbt[6], kbt[8]);
    kbt[6] = hi_1_50;
    kbt[8] = lo_1_50;
    float hi_1_51 = fmaxf(kbt[7], kbt[9]);
    float lo_1_51 = fminf(kbt[7], kbt[9]);
    kbt[7] = hi_1_51;
    kbt[9] = lo_1_51;
    float hi_1_52 = fmaxf(kbt[10], kbt[12]);
    float lo_1_52 = fminf(kbt[10], kbt[12]);
    kbt[10] = hi_1_52;
    kbt[12] = lo_1_52;
    float hi_1_53 = fmaxf(kbt[3], kbt[4]);
    float lo_1_53 = fminf(kbt[3], kbt[4]);
    kbt[3] = hi_1_53;
    kbt[4] = lo_1_53;
    float hi_1_54 = fmaxf(kbt[5], kbt[6]);
    float lo_1_54 = fminf(kbt[5], kbt[6]);
    kbt[5] = hi_1_54;
    kbt[6] = lo_1_54;
    float hi_1_55 = fmaxf(kbt[7], kbt[8]);
    float lo_1_55 = fminf(kbt[7], kbt[8]);
    kbt[7] = hi_1_55;
    kbt[8] = lo_1_55;
    float hi_1_56 = fmaxf(kbt[9], kbt[10]);
    float lo_1_56 = fminf(kbt[9], kbt[10]);
    kbt[9] = hi_1_56;
    kbt[10] = lo_1_56;
    float hi_1_57 = fmaxf(kbt[11], kbt[12]);
    float lo_1_57 = fminf(kbt[11], kbt[12]);
    kbt[11] = hi_1_57;
    kbt[12] = lo_1_57;
    float hi_1_58 = fmaxf(kbt[6], kbt[7]);
    float lo_1_58 = fminf(kbt[6], kbt[7]);
    kbt[6] = hi_1_58;
    kbt[7] = lo_1_58;
    float hi_1_59 = fmaxf(kbt[8], kbt[9]);
    float lo_1_59 = fminf(kbt[8], kbt[9]);
    kbt[8] = hi_1_59;
    kbt[9] = lo_1_59;
    float hi_1_60 = fmaxf(kbt[16], kbt[29]);
    float lo_1_60 = fminf(kbt[16], kbt[29]);
    kbt[16] = hi_1_60;
    kbt[29] = lo_1_60;
    float hi_1_61 = fmaxf(kbt[17], kbt[28]);
    float lo_1_61 = fminf(kbt[17], kbt[28]);
    kbt[17] = hi_1_61;
    kbt[28] = lo_1_61;
    float hi_1_62 = fmaxf(kbt[18], kbt[31]);
    float lo_1_62 = fminf(kbt[18], kbt[31]);
    kbt[18] = hi_1_62;
    kbt[31] = lo_1_62;
    float hi_1_63 = fmaxf(kbt[19], kbt[30]);
    float lo_1_63 = fminf(kbt[19], kbt[30]);
    kbt[19] = hi_1_63;
    kbt[30] = lo_1_63;
    float hi_1_64 = fmaxf(kbt[20], kbt[24]);
    float lo_1_64 = fminf(kbt[20], kbt[24]);
    kbt[20] = hi_1_64;
    kbt[24] = lo_1_64;
    float hi_1_65 = fmaxf(kbt[21], kbt[22]);
    float lo_1_65 = fminf(kbt[21], kbt[22]);
    kbt[21] = hi_1_65;
    kbt[22] = lo_1_65;
    float hi_1_66 = fmaxf(kbt[23], kbt[27]);
    float lo_1_66 = fminf(kbt[23], kbt[27]);
    kbt[23] = hi_1_66;
    kbt[27] = lo_1_66;
    float hi_1_67 = fmaxf(kbt[25], kbt[26]);
    float lo_1_67 = fminf(kbt[25], kbt[26]);
    kbt[25] = hi_1_67;
    kbt[26] = lo_1_67;
    float hi_1_68 = fmaxf(kbt[16], kbt[21]);
    float lo_1_68 = fminf(kbt[16], kbt[21]);
    kbt[16] = hi_1_68;
    kbt[21] = lo_1_68;
    float hi_1_69 = fmaxf(kbt[17], kbt[23]);
    float lo_1_69 = fminf(kbt[17], kbt[23]);
    kbt[17] = hi_1_69;
    kbt[23] = lo_1_69;
    float hi_1_70 = fmaxf(kbt[18], kbt[25]);
    float lo_1_70 = fminf(kbt[18], kbt[25]);
    kbt[18] = hi_1_70;
    kbt[25] = lo_1_70;
    float hi_1_71 = fmaxf(kbt[19], kbt[20]);
    float lo_1_71 = fminf(kbt[19], kbt[20]);
    kbt[19] = hi_1_71;
    kbt[20] = lo_1_71;
    float hi_1_72 = fmaxf(kbt[22], kbt[29]);
    float lo_1_72 = fminf(kbt[22], kbt[29]);
    kbt[22] = hi_1_72;
    kbt[29] = lo_1_72;
    float hi_1_73 = fmaxf(kbt[24], kbt[30]);
    float lo_1_73 = fminf(kbt[24], kbt[30]);
    kbt[24] = hi_1_73;
    kbt[30] = lo_1_73;
    float hi_1_74 = fmaxf(kbt[26], kbt[31]);
    float lo_1_74 = fminf(kbt[26], kbt[31]);
    kbt[26] = hi_1_74;
    kbt[31] = lo_1_74;
    float hi_1_75 = fmaxf(kbt[27], kbt[28]);
    float lo_1_75 = fminf(kbt[27], kbt[28]);
    kbt[27] = hi_1_75;
    kbt[28] = lo_1_75;
    float hi_1_76 = fmaxf(kbt[16], kbt[17]);
    float lo_1_76 = fminf(kbt[16], kbt[17]);
    kbt[16] = hi_1_76;
    kbt[17] = lo_1_76;
    float hi_1_77 = fmaxf(kbt[18], kbt[19]);
    float lo_1_77 = fminf(kbt[18], kbt[19]);
    kbt[18] = hi_1_77;
    kbt[19] = lo_1_77;
    float hi_1_78 = fmaxf(kbt[20], kbt[21]);
    float lo_1_78 = fminf(kbt[20], kbt[21]);
    kbt[20] = hi_1_78;
    kbt[21] = lo_1_78;
    float hi_1_79 = fmaxf(kbt[22], kbt[24]);
    float lo_1_79 = fminf(kbt[22], kbt[24]);
    kbt[22] = hi_1_79;
    kbt[24] = lo_1_79;
    float hi_1_80 = fmaxf(kbt[23], kbt[25]);
    float lo_1_80 = fminf(kbt[23], kbt[25]);
    kbt[23] = hi_1_80;
    kbt[25] = lo_1_80;
    float hi_1_81 = fmaxf(kbt[26], kbt[27]);
    float lo_1_81 = fminf(kbt[26], kbt[27]);
    kbt[26] = hi_1_81;
    kbt[27] = lo_1_81;
    float hi_1_82 = fmaxf(kbt[28], kbt[29]);
    float lo_1_82 = fminf(kbt[28], kbt[29]);
    kbt[28] = hi_1_82;
    kbt[29] = lo_1_82;
    float hi_1_83 = fmaxf(kbt[30], kbt[31]);
    float lo_1_83 = fminf(kbt[30], kbt[31]);
    kbt[30] = hi_1_83;
    kbt[31] = lo_1_83;
    float hi_1_84 = fmaxf(kbt[16], kbt[18]);
    float lo_1_84 = fminf(kbt[16], kbt[18]);
    kbt[16] = hi_1_84;
    kbt[18] = lo_1_84;
    float hi_1_85 = fmaxf(kbt[17], kbt[19]);
    float lo_1_85 = fminf(kbt[17], kbt[19]);
    kbt[17] = hi_1_85;
    kbt[19] = lo_1_85;
    float hi_1_86 = fmaxf(kbt[20], kbt[26]);
    float lo_1_86 = fminf(kbt[20], kbt[26]);
    kbt[20] = hi_1_86;
    kbt[26] = lo_1_86;
    float hi_1_87 = fmaxf(kbt[21], kbt[27]);
    float lo_1_87 = fminf(kbt[21], kbt[27]);
    kbt[21] = hi_1_87;
    kbt[27] = lo_1_87;
    float hi_1_88 = fmaxf(kbt[22], kbt[23]);
    float lo_1_88 = fminf(kbt[22], kbt[23]);
    kbt[22] = hi_1_88;
    kbt[23] = lo_1_88;
    float hi_1_89 = fmaxf(kbt[24], kbt[25]);
    float lo_1_89 = fminf(kbt[24], kbt[25]);
    kbt[24] = hi_1_89;
    kbt[25] = lo_1_89;
    float hi_1_90 = fmaxf(kbt[28], kbt[30]);
    float lo_1_90 = fminf(kbt[28], kbt[30]);
    kbt[28] = hi_1_90;
    kbt[30] = lo_1_90;
    float hi_1_91 = fmaxf(kbt[29], kbt[31]);
    float lo_1_91 = fminf(kbt[29], kbt[31]);
    kbt[29] = hi_1_91;
    kbt[31] = lo_1_91;
    float hi_1_92 = fmaxf(kbt[17], kbt[18]);
    float lo_1_92 = fminf(kbt[17], kbt[18]);
    kbt[17] = hi_1_92;
    kbt[18] = lo_1_92;
    float hi_1_93 = fmaxf(kbt[19], kbt[28]);
    float lo_1_93 = fminf(kbt[19], kbt[28]);
    kbt[19] = hi_1_93;
    kbt[28] = lo_1_93;
    float hi_1_94 = fmaxf(kbt[20], kbt[22]);
    float lo_1_94 = fminf(kbt[20], kbt[22]);
    kbt[20] = hi_1_94;
    kbt[22] = lo_1_94;
    float hi_1_95 = fmaxf(kbt[21], kbt[23]);
    float lo_1_95 = fminf(kbt[21], kbt[23]);
    kbt[21] = hi_1_95;
    kbt[23] = lo_1_95;
    float hi_1_96 = fmaxf(kbt[24], kbt[26]);
    float lo_1_96 = fminf(kbt[24], kbt[26]);
    kbt[24] = hi_1_96;
    kbt[26] = lo_1_96;
    float hi_1_97 = fmaxf(kbt[25], kbt[27]);
    float lo_1_97 = fminf(kbt[25], kbt[27]);
    kbt[25] = hi_1_97;
    kbt[27] = lo_1_97;
    float hi_1_98 = fmaxf(kbt[29], kbt[30]);
    float lo_1_98 = fminf(kbt[29], kbt[30]);
    kbt[29] = hi_1_98;
    kbt[30] = lo_1_98;
    float hi_1_99 = fmaxf(kbt[17], kbt[20]);
    float lo_1_99 = fminf(kbt[17], kbt[20]);
    kbt[17] = hi_1_99;
    kbt[20] = lo_1_99;
    float hi_1_100 = fmaxf(kbt[18], kbt[22]);
    float lo_1_100 = fminf(kbt[18], kbt[22]);
    kbt[18] = hi_1_100;
    kbt[22] = lo_1_100;
    float hi_1_101 = fmaxf(kbt[21], kbt[24]);
    float lo_1_101 = fminf(kbt[21], kbt[24]);
    kbt[21] = hi_1_101;
    kbt[24] = lo_1_101;
    float hi_1_102 = fmaxf(kbt[23], kbt[26]);
    float lo_1_102 = fminf(kbt[23], kbt[26]);
    kbt[23] = hi_1_102;
    kbt[26] = lo_1_102;
    float hi_1_103 = fmaxf(kbt[25], kbt[29]);
    float lo_1_103 = fminf(kbt[25], kbt[29]);
    kbt[25] = hi_1_103;
    kbt[29] = lo_1_103;
    float hi_1_104 = fmaxf(kbt[27], kbt[30]);
    float lo_1_104 = fminf(kbt[27], kbt[30]);
    kbt[27] = hi_1_104;
    kbt[30] = lo_1_104;
    float hi_1_105 = fmaxf(kbt[18], kbt[20]);
    float lo_1_105 = fminf(kbt[18], kbt[20]);
    kbt[18] = hi_1_105;
    kbt[20] = lo_1_105;
    float hi_1_106 = fmaxf(kbt[19], kbt[22]);
    float lo_1_106 = fminf(kbt[19], kbt[22]);
    kbt[19] = hi_1_106;
    kbt[22] = lo_1_106;
    float hi_1_107 = fmaxf(kbt[25], kbt[28]);
    float lo_1_107 = fminf(kbt[25], kbt[28]);
    kbt[25] = hi_1_107;
    kbt[28] = lo_1_107;
    float hi_1_108 = fmaxf(kbt[27], kbt[29]);
    float lo_1_108 = fminf(kbt[27], kbt[29]);
    kbt[27] = hi_1_108;
    kbt[29] = lo_1_108;
    float hi_1_109 = fmaxf(kbt[19], kbt[21]);
    float lo_1_109 = fminf(kbt[19], kbt[21]);
    kbt[19] = hi_1_109;
    kbt[21] = lo_1_109;
    float hi_1_110 = fmaxf(kbt[22], kbt[24]);
    float lo_1_110 = fminf(kbt[22], kbt[24]);
    kbt[22] = hi_1_110;
    kbt[24] = lo_1_110;
    float hi_1_111 = fmaxf(kbt[23], kbt[25]);
    float lo_1_111 = fminf(kbt[23], kbt[25]);
    kbt[23] = hi_1_111;
    kbt[25] = lo_1_111;
    float hi_1_112 = fmaxf(kbt[26], kbt[28]);
    float lo_1_112 = fminf(kbt[26], kbt[28]);
    kbt[26] = hi_1_112;
    kbt[28] = lo_1_112;
    float hi_1_113 = fmaxf(kbt[19], kbt[20]);
    float lo_1_113 = fminf(kbt[19], kbt[20]);
    kbt[19] = hi_1_113;
    kbt[20] = lo_1_113;
    float hi_1_114 = fmaxf(kbt[21], kbt[22]);
    float lo_1_114 = fminf(kbt[21], kbt[22]);
    kbt[21] = hi_1_114;
    kbt[22] = lo_1_114;
    float hi_1_115 = fmaxf(kbt[23], kbt[24]);
    float lo_1_115 = fminf(kbt[23], kbt[24]);
    kbt[23] = hi_1_115;
    kbt[24] = lo_1_115;
    float hi_1_116 = fmaxf(kbt[25], kbt[26]);
    float lo_1_116 = fminf(kbt[25], kbt[26]);
    kbt[25] = hi_1_116;
    kbt[26] = lo_1_116;
    float hi_1_117 = fmaxf(kbt[27], kbt[28]);
    float lo_1_117 = fminf(kbt[27], kbt[28]);
    kbt[27] = hi_1_117;
    kbt[28] = lo_1_117;
    float hi_1_118 = fmaxf(kbt[22], kbt[23]);
    float lo_1_118 = fminf(kbt[22], kbt[23]);
    kbt[22] = hi_1_118;
    kbt[23] = lo_1_118;
    float hi_1_119 = fmaxf(kbt[24], kbt[25]);
    float lo_1_119 = fminf(kbt[24], kbt[25]);
    kbt[24] = hi_1_119;
    kbt[25] = lo_1_119;
    float hi_0 = fmaxf(kbt[0], kbt[31]);
    float lo_0 = fminf(kbt[0], kbt[31]);
    kbt[0] = hi_0;
    rr = fmaxf(rr, lo_0);
    float hi_1 = fmaxf(kbt[1], kbt[30]);
    float lo_1 = fminf(kbt[1], kbt[30]);
    kbt[1] = hi_1;
    rr = fmaxf(rr, lo_1);
    float hi_2 = fmaxf(kbt[2], kbt[29]);
    float lo_2 = fminf(kbt[2], kbt[29]);
    kbt[2] = hi_2;
    rr = fmaxf(rr, lo_2);
    float hi_3 = fmaxf(kbt[3], kbt[28]);
    float lo_3 = fminf(kbt[3], kbt[28]);
    kbt[3] = hi_3;
    rr = fmaxf(rr, lo_3);
    float hi_4 = fmaxf(kbt[4], kbt[27]);
    float lo_4 = fminf(kbt[4], kbt[27]);
    kbt[4] = hi_4;
    rr = fmaxf(rr, lo_4);
    float hi_5 = fmaxf(kbt[5], kbt[26]);
    float lo_5 = fminf(kbt[5], kbt[26]);
    kbt[5] = hi_5;
    rr = fmaxf(rr, lo_5);
    float hi_6 = fmaxf(kbt[6], kbt[25]);
    float lo_6 = fminf(kbt[6], kbt[25]);
    kbt[6] = hi_6;
    rr = fmaxf(rr, lo_6);
    float hi_7 = fmaxf(kbt[7], kbt[24]);
    float lo_7 = fminf(kbt[7], kbt[24]);
    kbt[7] = hi_7;
    rr = fmaxf(rr, lo_7);
    float hi_8 = fmaxf(kbt[8], kbt[23]);
    float lo_8 = fminf(kbt[8], kbt[23]);
    kbt[8] = hi_8;
    rr = fmaxf(rr, lo_8);
    float hi_9 = fmaxf(kbt[9], kbt[22]);
    float lo_9 = fminf(kbt[9], kbt[22]);
    kbt[9] = hi_9;
    rr = fmaxf(rr, lo_9);
    float hi_10 = fmaxf(kbt[10], kbt[21]);
    float lo_10 = fminf(kbt[10], kbt[21]);
    kbt[10] = hi_10;
    rr = fmaxf(rr, lo_10);
    float hi_11 = fmaxf(kbt[11], kbt[20]);
    float lo_11 = fminf(kbt[11], kbt[20]);
    kbt[11] = hi_11;
    rr = fmaxf(rr, lo_11);
    float hi_12 = fmaxf(kbt[12], kbt[19]);
    float lo_12 = fminf(kbt[12], kbt[19]);
    kbt[12] = hi_12;
    rr = fmaxf(rr, lo_12);
    float hi_13 = fmaxf(kbt[13], kbt[18]);
    float lo_13 = fminf(kbt[13], kbt[18]);
    kbt[13] = hi_13;
    rr = fmaxf(rr, lo_13);
    float hi_14 = fmaxf(kbt[14], kbt[17]);
    float lo_14 = fminf(kbt[14], kbt[17]);
    kbt[14] = hi_14;
    rr = fmaxf(rr, lo_14);
    float hi_15 = fmaxf(kbt[15], kbt[16]);
    float lo_15 = fminf(kbt[15], kbt[16]);
    kbt[15] = hi_15;
    rr = fmaxf(rr, lo_15);
    float hi_16 = fmaxf(kbt[0], kbt[8]);
    float lo_16 = fminf(kbt[0], kbt[8]);
    kbt[0] = hi_16;
    kbt[8] = lo_16;
    float hi_17 = fmaxf(kbt[1], kbt[9]);
    float lo_17 = fminf(kbt[1], kbt[9]);
    kbt[1] = hi_17;
    kbt[9] = lo_17;
    float hi_18 = fmaxf(kbt[2], kbt[10]);
    float lo_18 = fminf(kbt[2], kbt[10]);
    kbt[2] = hi_18;
    kbt[10] = lo_18;
    float hi_19 = fmaxf(kbt[3], kbt[11]);
    float lo_19 = fminf(kbt[3], kbt[11]);
    kbt[3] = hi_19;
    kbt[11] = lo_19;
    float hi_20 = fmaxf(kbt[4], kbt[12]);
    float lo_20 = fminf(kbt[4], kbt[12]);
    kbt[4] = hi_20;
    kbt[12] = lo_20;
    float hi_21 = fmaxf(kbt[5], kbt[13]);
    float lo_21 = fminf(kbt[5], kbt[13]);
    kbt[5] = hi_21;
    kbt[13] = lo_21;
    float hi_22 = fmaxf(kbt[6], kbt[14]);
    float lo_22 = fminf(kbt[6], kbt[14]);
    kbt[6] = hi_22;
    kbt[14] = lo_22;
    float hi_23 = fmaxf(kbt[7], kbt[15]);
    float lo_23 = fminf(kbt[7], kbt[15]);
    kbt[7] = hi_23;
    kbt[15] = lo_23;
    float hi_24 = fmaxf(kbt[0], kbt[4]);
    float lo_24 = fminf(kbt[0], kbt[4]);
    kbt[0] = hi_24;
    kbt[4] = lo_24;
    float hi_25 = fmaxf(kbt[1], kbt[5]);
    float lo_25 = fminf(kbt[1], kbt[5]);
    kbt[1] = hi_25;
    kbt[5] = lo_25;
    float hi_26 = fmaxf(kbt[2], kbt[6]);
    float lo_26 = fminf(kbt[2], kbt[6]);
    kbt[2] = hi_26;
    kbt[6] = lo_26;
    float hi_27 = fmaxf(kbt[3], kbt[7]);
    float lo_27 = fminf(kbt[3], kbt[7]);
    kbt[3] = hi_27;
    kbt[7] = lo_27;
    float hi_28 = fmaxf(kbt[8], kbt[12]);
    float lo_28 = fminf(kbt[8], kbt[12]);
    kbt[8] = hi_28;
    kbt[12] = lo_28;
    float hi_29 = fmaxf(kbt[9], kbt[13]);
    float lo_29 = fminf(kbt[9], kbt[13]);
    kbt[9] = hi_29;
    kbt[13] = lo_29;
    float hi_30 = fmaxf(kbt[10], kbt[14]);
    float lo_30 = fminf(kbt[10], kbt[14]);
    kbt[10] = hi_30;
    kbt[14] = lo_30;
    float hi_31 = fmaxf(kbt[11], kbt[15]);
    float lo_31 = fminf(kbt[11], kbt[15]);
    kbt[11] = hi_31;
    kbt[15] = lo_31;
    float hi_32 = fmaxf(kbt[0], kbt[2]);
    float lo_32 = fminf(kbt[0], kbt[2]);
    kbt[0] = hi_32;
    kbt[2] = lo_32;
    float hi_33 = fmaxf(kbt[1], kbt[3]);
    float lo_33 = fminf(kbt[1], kbt[3]);
    kbt[1] = hi_33;
    kbt[3] = lo_33;
    float hi_34 = fmaxf(kbt[4], kbt[6]);
    float lo_34 = fminf(kbt[4], kbt[6]);
    kbt[4] = hi_34;
    kbt[6] = lo_34;
    float hi_35 = fmaxf(kbt[5], kbt[7]);
    float lo_35 = fminf(kbt[5], kbt[7]);
    kbt[5] = hi_35;
    kbt[7] = lo_35;
    float hi_36 = fmaxf(kbt[8], kbt[10]);
    float lo_36 = fminf(kbt[8], kbt[10]);
    kbt[8] = hi_36;
    kbt[10] = lo_36;
    float hi_37 = fmaxf(kbt[9], kbt[11]);
    float lo_37 = fminf(kbt[9], kbt[11]);
    kbt[9] = hi_37;
    kbt[11] = lo_37;
    float hi_38 = fmaxf(kbt[12], kbt[14]);
    float lo_38 = fminf(kbt[12], kbt[14]);
    kbt[12] = hi_38;
    kbt[14] = lo_38;
    float hi_39 = fmaxf(kbt[13], kbt[15]);
    float lo_39 = fminf(kbt[13], kbt[15]);
    kbt[13] = hi_39;
    kbt[15] = lo_39;
    float hi_40 = fmaxf(kbt[0], kbt[1]);
    float lo_40 = fminf(kbt[0], kbt[1]);
    kbt[0] = hi_40;
    kbt[1] = lo_40;
    float hi_41 = fmaxf(kbt[2], kbt[3]);
    float lo_41 = fminf(kbt[2], kbt[3]);
    kbt[2] = hi_41;
    kbt[3] = lo_41;
    float hi_42 = fmaxf(kbt[4], kbt[5]);
    float lo_42 = fminf(kbt[4], kbt[5]);
    kbt[4] = hi_42;
    kbt[5] = lo_42;
    float hi_43 = fmaxf(kbt[6], kbt[7]);
    float lo_43 = fminf(kbt[6], kbt[7]);
    kbt[6] = hi_43;
    kbt[7] = lo_43;
    float hi_44 = fmaxf(kbt[8], kbt[9]);
    float lo_44 = fminf(kbt[8], kbt[9]);
    kbt[8] = hi_44;
    kbt[9] = lo_44;
    float hi_45 = fmaxf(kbt[10], kbt[11]);
    float lo_45 = fminf(kbt[10], kbt[11]);
    kbt[10] = hi_45;
    kbt[11] = lo_45;
    float hi_46 = fmaxf(kbt[12], kbt[13]);
    float lo_46 = fminf(kbt[12], kbt[13]);
    kbt[12] = hi_46;
    kbt[13] = lo_46;
    float hi_47 = fmaxf(kbt[14], kbt[15]);
    float lo_47 = fminf(kbt[14], kbt[15]);
    kbt[14] = hi_47;
    kbt[15] = lo_47;
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
    asm volatile("barrier.sync 1, 256;" ::: "memory");
    float r = neg_inf;
    float V[8];
    int s0_0 = (sg * 16 * 16 + cg) * 17;
    int s1_0 = ((sg * 16 + 8) * 16 + cg) * 17;
    float x0_0 = pub[s0_0 + ln];
    float y0_0 = pub[s1_0 + lnr];
    float lo0_0 = fminf(x0_0, y0_0);
    r = fmaxf(r, lo0_0);
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
    int s0_1 = ((sg * 16 + 1) * 16 + cg) * 17;
    int s1_1 = ((sg * 16 + 1 + 8) * 16 + cg) * 17;
    float x0_1 = pub[s0_1 + ln];
    float y0_1 = pub[s1_1 + lnr];
    float lo0_1 = fminf(x0_1, y0_1);
    r = fmaxf(r, lo0_1);
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
    int s0_2 = ((sg * 16 + 2) * 16 + cg) * 17;
    int s1_2 = ((sg * 16 + 2 + 8) * 16 + cg) * 17;
    float x0_2 = pub[s0_2 + ln];
    float y0_2 = pub[s1_2 + lnr];
    float lo0_2 = fminf(x0_2, y0_2);
    r = fmaxf(r, lo0_2);
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
    int s0_3 = ((sg * 16 + 3) * 16 + cg) * 17;
    int s1_3 = ((sg * 16 + 3 + 8) * 16 + cg) * 17;
    float x0_3 = pub[s0_3 + ln];
    float y0_3 = pub[s1_3 + lnr];
    float lo0_3 = fminf(x0_3, y0_3);
    r = fmaxf(r, lo0_3);
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
    int s0_4 = ((sg * 16 + 4) * 16 + cg) * 17;
    int s1_4 = ((sg * 16 + 4 + 8) * 16 + cg) * 17;
    float x0_4 = pub[s0_4 + ln];
    float y0_4 = pub[s1_4 + lnr];
    float lo0_4 = fminf(x0_4, y0_4);
    r = fmaxf(r, lo0_4);
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
    int s0_5 = ((sg * 16 + 5) * 16 + cg) * 17;
    int s1_5 = ((sg * 16 + 5 + 8) * 16 + cg) * 17;
    float x0_5 = pub[s0_5 + ln];
    float y0_5 = pub[s1_5 + lnr];
    float lo0_5 = fminf(x0_5, y0_5);
    r = fmaxf(r, lo0_5);
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
    int s0_6 = ((sg * 16 + 6) * 16 + cg) * 17;
    int s1_6 = ((sg * 16 + 6 + 8) * 16 + cg) * 17;
    float x0_6 = pub[s0_6 + ln];
    float y0_6 = pub[s1_6 + lnr];
    float lo0_6 = fminf(x0_6, y0_6);
    r = fmaxf(r, lo0_6);
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
    int s0_7 = ((sg * 16 + 7) * 16 + cg) * 17;
    int s1_7 = ((sg * 16 + 7 + 8) * 16 + cg) * 17;
    float x0_7 = pub[s0_7 + ln];
    float y0_7 = pub[s1_7 + lnr];
    float lo0_7 = fminf(x0_7, y0_7);
    r = fmaxf(r, lo0_7);
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
    float rs = pub[((sg * 16 + ln) * 16 + cg) * 17 + 16];
    r = fmaxf(r, rs);
    float y1_0 = __shfl_xor_sync(0xFFFFFFFF, V[4], 15);
    float lo1_0 = fminf(V[0], y1_0);
    r = fmaxf(r, lo1_0);
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
    r = fmaxf(r, lo1_1);
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
    r = fmaxf(r, lo1_2);
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
    r = fmaxf(r, lo1_3);
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
    r = fmaxf(r, lo1_4);
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
    r = fmaxf(r, lo1_5);
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
    r = fmaxf(r, lo1_6);
    float hi1_6 = fmaxf(V[0], y1_6);
    V[0] = hi1_6;
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
        if (ln == 0 && g < 16) {
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
            unsigned int mm = 0;
            float sc_31 = __uint_as_float(cbt[0]);
            sc_31 = fmaxf(sc_31, -1.7014118e+38f);
            sc_31 = fminf(sc_31, 1.7014118e+38f);
            unsigned int cls9_1_0 = __as_u32(sc_31) & 4294966784u;
            if (cls9_1_0 == qc) {
                mm = mm | 1;
            }
            float sc_32 = __uint_as_float(cbt[1]);
            sc_32 = fmaxf(sc_32, -1.7014118e+38f);
            sc_32 = fminf(sc_32, 1.7014118e+38f);
            unsigned int cls9_1_1 = __as_u32(sc_32) & 4294966784u;
            if (cls9_1_1 == qc) {
                mm = mm | 2;
            }
            float sc_33 = __uint_as_float(cbt[2]);
            sc_33 = fmaxf(sc_33, -1.7014118e+38f);
            sc_33 = fminf(sc_33, 1.7014118e+38f);
            unsigned int cls9_1_2 = __as_u32(sc_33) & 4294966784u;
            if (cls9_1_2 == qc) {
                mm = mm | 4;
            }
            float sc_34 = __uint_as_float(cbt[3]);
            sc_34 = fmaxf(sc_34, -1.7014118e+38f);
            sc_34 = fminf(sc_34, 1.7014118e+38f);
            unsigned int cls9_1_3 = __as_u32(sc_34) & 4294966784u;
            if (cls9_1_3 == qc) {
                mm = mm | 8;
            }
            float sc_35 = __uint_as_float(cbt[4]);
            sc_35 = fmaxf(sc_35, -1.7014118e+38f);
            sc_35 = fminf(sc_35, 1.7014118e+38f);
            unsigned int cls9_1_4 = __as_u32(sc_35) & 4294966784u;
            if (cls9_1_4 == qc) {
                mm = mm | 16;
            }
            float sc_36 = __uint_as_float(cbt[5]);
            sc_36 = fmaxf(sc_36, -1.7014118e+38f);
            sc_36 = fminf(sc_36, 1.7014118e+38f);
            unsigned int cls9_1_5 = __as_u32(sc_36) & 4294966784u;
            if (cls9_1_5 == qc) {
                mm = mm | 32;
            }
            float sc_37 = __uint_as_float(cbt[6]);
            sc_37 = fmaxf(sc_37, -1.7014118e+38f);
            sc_37 = fminf(sc_37, 1.7014118e+38f);
            unsigned int cls9_1_6 = __as_u32(sc_37) & 4294966784u;
            if (cls9_1_6 == qc) {
                mm = mm | 64;
            }
            float sc_38 = __uint_as_float(cbt[7]);
            sc_38 = fmaxf(sc_38, -1.7014118e+38f);
            sc_38 = fminf(sc_38, 1.7014118e+38f);
            unsigned int cls9_1_7 = __as_u32(sc_38) & 4294966784u;
            if (cls9_1_7 == qc) {
                mm = mm | 128;
            }
            float sc_39 = __uint_as_float(cbt[8]);
            sc_39 = fmaxf(sc_39, -1.7014118e+38f);
            sc_39 = fminf(sc_39, 1.7014118e+38f);
            unsigned int cls9_1_8 = __as_u32(sc_39) & 4294966784u;
            if (cls9_1_8 == qc) {
                mm = mm | 256;
            }
            float sc_40 = __uint_as_float(cbt[9]);
            sc_40 = fmaxf(sc_40, -1.7014118e+38f);
            sc_40 = fminf(sc_40, 1.7014118e+38f);
            unsigned int cls9_1_9 = __as_u32(sc_40) & 4294966784u;
            if (cls9_1_9 == qc) {
                mm = mm | 512;
            }
            float sc_41 = __uint_as_float(cbt[10]);
            sc_41 = fmaxf(sc_41, -1.7014118e+38f);
            sc_41 = fminf(sc_41, 1.7014118e+38f);
            unsigned int cls9_1_10 = __as_u32(sc_41) & 4294966784u;
            if (cls9_1_10 == qc) {
                mm = mm | 1024;
            }
            float sc_42 = __uint_as_float(cbt[11]);
            sc_42 = fmaxf(sc_42, -1.7014118e+38f);
            sc_42 = fminf(sc_42, 1.7014118e+38f);
            unsigned int cls9_1_11 = __as_u32(sc_42) & 4294966784u;
            if (cls9_1_11 == qc) {
                mm = mm | 2048;
            }
            float sc_43 = __uint_as_float(cbt[12]);
            sc_43 = fmaxf(sc_43, -1.7014118e+38f);
            sc_43 = fminf(sc_43, 1.7014118e+38f);
            unsigned int cls9_1_12 = __as_u32(sc_43) & 4294966784u;
            if (cls9_1_12 == qc) {
                mm = mm | 4096;
            }
            float sc_44 = __uint_as_float(cbt[13]);
            sc_44 = fmaxf(sc_44, -1.7014118e+38f);
            sc_44 = fminf(sc_44, 1.7014118e+38f);
            unsigned int cls9_1_13 = __as_u32(sc_44) & 4294966784u;
            if (cls9_1_13 == qc) {
                mm = mm | 8192;
            }
            float sc_45 = __uint_as_float(cbt[14]);
            sc_45 = fmaxf(sc_45, -1.7014118e+38f);
            sc_45 = fminf(sc_45, 1.7014118e+38f);
            unsigned int cls9_1_14 = __as_u32(sc_45) & 4294966784u;
            if (cls9_1_14 == qc) {
                mm = mm | 16384;
            }
            float sc_46 = __uint_as_float(cbt[15]);
            sc_46 = fmaxf(sc_46, -1.7014118e+38f);
            sc_46 = fminf(sc_46, 1.7014118e+38f);
            unsigned int cls9_1_15 = __as_u32(sc_46) & 4294966784u;
            if (cls9_1_15 == qc) {
                mm = mm | 32768;
            }
            float sc_47 = __uint_as_float(cbt[16]);
            sc_47 = fmaxf(sc_47, -1.7014118e+38f);
            sc_47 = fminf(sc_47, 1.7014118e+38f);
            unsigned int cls9_1_16 = __as_u32(sc_47) & 4294966784u;
            if (cls9_1_16 == qc) {
                mm = mm | 65536;
            }
            float sc_48 = __uint_as_float(cbt[17]);
            sc_48 = fmaxf(sc_48, -1.7014118e+38f);
            sc_48 = fminf(sc_48, 1.7014118e+38f);
            unsigned int cls9_1_17 = __as_u32(sc_48) & 4294966784u;
            if (cls9_1_17 == qc) {
                mm = mm | 131072;
            }
            float sc_49 = __uint_as_float(cbt[18]);
            sc_49 = fmaxf(sc_49, -1.7014118e+38f);
            sc_49 = fminf(sc_49, 1.7014118e+38f);
            unsigned int cls9_1_18 = __as_u32(sc_49) & 4294966784u;
            if (cls9_1_18 == qc) {
                mm = mm | 262144;
            }
            float sc_50 = __uint_as_float(cbt[19]);
            sc_50 = fmaxf(sc_50, -1.7014118e+38f);
            sc_50 = fminf(sc_50, 1.7014118e+38f);
            unsigned int cls9_1_19 = __as_u32(sc_50) & 4294966784u;
            if (cls9_1_19 == qc) {
                mm = mm | 524288;
            }
            float sc_51 = __uint_as_float(cbt[20]);
            sc_51 = fmaxf(sc_51, -1.7014118e+38f);
            sc_51 = fminf(sc_51, 1.7014118e+38f);
            unsigned int cls9_1_20 = __as_u32(sc_51) & 4294966784u;
            if (cls9_1_20 == qc) {
                mm = mm | 1048576;
            }
            float sc_52 = __uint_as_float(cbt[21]);
            sc_52 = fmaxf(sc_52, -1.7014118e+38f);
            sc_52 = fminf(sc_52, 1.7014118e+38f);
            unsigned int cls9_1_21 = __as_u32(sc_52) & 4294966784u;
            if (cls9_1_21 == qc) {
                mm = mm | 2097152;
            }
            float sc_53 = __uint_as_float(cbt[22]);
            sc_53 = fmaxf(sc_53, -1.7014118e+38f);
            sc_53 = fminf(sc_53, 1.7014118e+38f);
            unsigned int cls9_1_22 = __as_u32(sc_53) & 4294966784u;
            if (cls9_1_22 == qc) {
                mm = mm | 4194304;
            }
            float sc_54 = __uint_as_float(cbt[23]);
            sc_54 = fmaxf(sc_54, -1.7014118e+38f);
            sc_54 = fminf(sc_54, 1.7014118e+38f);
            unsigned int cls9_1_23 = __as_u32(sc_54) & 4294966784u;
            if (cls9_1_23 == qc) {
                mm = mm | 8388608;
            }
            float sc_55 = __uint_as_float(cbt[24]);
            sc_55 = fmaxf(sc_55, -1.7014118e+38f);
            sc_55 = fminf(sc_55, 1.7014118e+38f);
            unsigned int cls9_1_24 = __as_u32(sc_55) & 4294966784u;
            if (cls9_1_24 == qc) {
                mm = mm | 16777216;
            }
            float sc_56 = __uint_as_float(cbt[25]);
            sc_56 = fmaxf(sc_56, -1.7014118e+38f);
            sc_56 = fminf(sc_56, 1.7014118e+38f);
            unsigned int cls9_1_25 = __as_u32(sc_56) & 4294966784u;
            if (cls9_1_25 == qc) {
                mm = mm | 33554432;
            }
            float sc_57 = __uint_as_float(cbt[26]);
            sc_57 = fmaxf(sc_57, -1.7014118e+38f);
            sc_57 = fminf(sc_57, 1.7014118e+38f);
            unsigned int cls9_1_26 = __as_u32(sc_57) & 4294966784u;
            if (cls9_1_26 == qc) {
                mm = mm | 67108864;
            }
            float sc_58 = __uint_as_float(cbt[27]);
            sc_58 = fmaxf(sc_58, -1.7014118e+38f);
            sc_58 = fminf(sc_58, 1.7014118e+38f);
            unsigned int cls9_1_27 = __as_u32(sc_58) & 4294966784u;
            if (cls9_1_27 == qc) {
                mm = mm | 134217728;
            }
            float sc_59 = __uint_as_float(cbt[28]);
            sc_59 = fmaxf(sc_59, -1.7014118e+38f);
            sc_59 = fminf(sc_59, 1.7014118e+38f);
            unsigned int cls9_1_28 = __as_u32(sc_59) & 4294966784u;
            if (cls9_1_28 == qc) {
                mm = mm | 268435456;
            }
            float sc_60 = __uint_as_float(cbt[29]);
            sc_60 = fmaxf(sc_60, -1.7014118e+38f);
            sc_60 = fminf(sc_60, 1.7014118e+38f);
            unsigned int cls9_1_29 = __as_u32(sc_60) & 4294966784u;
            if (cls9_1_29 == qc) {
                mm = mm | 536870912;
            }
            float sc_61 = __uint_as_float(cbt[30]);
            sc_61 = fmaxf(sc_61, -1.7014118e+38f);
            sc_61 = fminf(sc_61, 1.7014118e+38f);
            unsigned int cls9_1_30 = __as_u32(sc_61) & 4294966784u;
            if (cls9_1_30 == qc) {
                mm = mm | 1073741824;
            }
            float sc_62 = __uint_as_float(cbt[31]);
            sc_62 = fmaxf(sc_62, -1.7014118e+38f);
            sc_62 = fminf(sc_62, 1.7014118e+38f);
            unsigned int cls9_1_31 = __as_u32(sc_62) & 4294966784u;
            if (cls9_1_31 == qc) {
                mm = mm | 2147483648u;
            }
            if (mm != 0) {
                int n9 = (int)(unsigned int)__popc(mm);
                unsigned int base9 = atomicAdd(&ccnt[c], (unsigned int)n9);
                if (base9 + (unsigned int)n9 > 16) {
                    flagw[1] = 1;
                } else {
                    int pos9 = c * 16 + (int)base9;
                    if ((mm & 1) != 0) {
                        float sc_63 = __uint_as_float(cbt[0]);
                        sc_63 = fmaxf(sc_63, -1.7014118e+38f);
                        sc_63 = fminf(sc_63, 1.7014118e+38f);
                        unsigned int u9_0 = __as_u32(sc_63);
                        unsigned int lowb9_0 = (u9_0 ^ (unsigned int)((int)u9_0 >> 31) & 511) & 511;
                        unsigned int km9_0 = 536870912 | lowb9_0 << 9 | (unsigned int)tb;
                        cbuf[pos9] = km9_0;
                        pos9 = pos9 + 1;
                    }
                    if ((mm & 2) != 0) {
                        float sc_64 = __uint_as_float(cbt[1]);
                        sc_64 = fmaxf(sc_64, -1.7014118e+38f);
                        sc_64 = fminf(sc_64, 1.7014118e+38f);
                        unsigned int u9_1 = __as_u32(sc_64);
                        unsigned int lowb9_1 = (u9_1 ^ (unsigned int)((int)u9_1 >> 31) & 511) & 511;
                        unsigned int km9_1 = 536870912 | lowb9_1 << 9 | (unsigned int)(tb + 16);
                        cbuf[pos9] = km9_1;
                        pos9 = pos9 + 1;
                    }
                    if ((mm & 4) != 0) {
                        float sc_65 = __uint_as_float(cbt[2]);
                        sc_65 = fmaxf(sc_65, -1.7014118e+38f);
                        sc_65 = fminf(sc_65, 1.7014118e+38f);
                        unsigned int u9_2 = __as_u32(sc_65);
                        unsigned int lowb9_2 = (u9_2 ^ (unsigned int)((int)u9_2 >> 31) & 511) & 511;
                        unsigned int km9_2 = 536870912 | lowb9_2 << 9 | (unsigned int)(tb + 32);
                        cbuf[pos9] = km9_2;
                        pos9 = pos9 + 1;
                    }
                    if ((mm & 8) != 0) {
                        float sc_66 = __uint_as_float(cbt[3]);
                        sc_66 = fmaxf(sc_66, -1.7014118e+38f);
                        sc_66 = fminf(sc_66, 1.7014118e+38f);
                        unsigned int u9_3 = __as_u32(sc_66);
                        unsigned int lowb9_3 = (u9_3 ^ (unsigned int)((int)u9_3 >> 31) & 511) & 511;
                        unsigned int km9_3 = 536870912 | lowb9_3 << 9 | (unsigned int)(tb + 48);
                        cbuf[pos9] = km9_3;
                        pos9 = pos9 + 1;
                    }
                    if ((mm & 16) != 0) {
                        float sc_67 = __uint_as_float(cbt[4]);
                        sc_67 = fmaxf(sc_67, -1.7014118e+38f);
                        sc_67 = fminf(sc_67, 1.7014118e+38f);
                        unsigned int u9_4 = __as_u32(sc_67);
                        unsigned int lowb9_4 = (u9_4 ^ (unsigned int)((int)u9_4 >> 31) & 511) & 511;
                        unsigned int km9_4 = 536870912 | lowb9_4 << 9 | (unsigned int)(tb + 64);
                        cbuf[pos9] = km9_4;
                        pos9 = pos9 + 1;
                    }
                    if ((mm & 32) != 0) {
                        float sc_68 = __uint_as_float(cbt[5]);
                        sc_68 = fmaxf(sc_68, -1.7014118e+38f);
                        sc_68 = fminf(sc_68, 1.7014118e+38f);
                        unsigned int u9_5 = __as_u32(sc_68);
                        unsigned int lowb9_5 = (u9_5 ^ (unsigned int)((int)u9_5 >> 31) & 511) & 511;
                        unsigned int km9_5 = 536870912 | lowb9_5 << 9 | (unsigned int)(tb + 80);
                        cbuf[pos9] = km9_5;
                        pos9 = pos9 + 1;
                    }
                    if ((mm & 64) != 0) {
                        float sc_69 = __uint_as_float(cbt[6]);
                        sc_69 = fmaxf(sc_69, -1.7014118e+38f);
                        sc_69 = fminf(sc_69, 1.7014118e+38f);
                        unsigned int u9_6 = __as_u32(sc_69);
                        unsigned int lowb9_6 = (u9_6 ^ (unsigned int)((int)u9_6 >> 31) & 511) & 511;
                        unsigned int km9_6 = 536870912 | lowb9_6 << 9 | (unsigned int)(tb + 96);
                        cbuf[pos9] = km9_6;
                        pos9 = pos9 + 1;
                    }
                    if ((mm & 128) != 0) {
                        float sc_70 = __uint_as_float(cbt[7]);
                        sc_70 = fmaxf(sc_70, -1.7014118e+38f);
                        sc_70 = fminf(sc_70, 1.7014118e+38f);
                        unsigned int u9_7 = __as_u32(sc_70);
                        unsigned int lowb9_7 = (u9_7 ^ (unsigned int)((int)u9_7 >> 31) & 511) & 511;
                        unsigned int km9_7 = 536870912 | lowb9_7 << 9 | (unsigned int)(tb + 112);
                        cbuf[pos9] = km9_7;
                        pos9 = pos9 + 1;
                    }
                    if ((mm & 256) != 0) {
                        float sc_71 = __uint_as_float(cbt[8]);
                        sc_71 = fmaxf(sc_71, -1.7014118e+38f);
                        sc_71 = fminf(sc_71, 1.7014118e+38f);
                        unsigned int u9_8 = __as_u32(sc_71);
                        unsigned int lowb9_8 = (u9_8 ^ (unsigned int)((int)u9_8 >> 31) & 511) & 511;
                        unsigned int km9_8 = 536870912 | lowb9_8 << 9 | (unsigned int)(tb + 128);
                        cbuf[pos9] = km9_8;
                        pos9 = pos9 + 1;
                    }
                    if ((mm & 512) != 0) {
                        float sc_72 = __uint_as_float(cbt[9]);
                        sc_72 = fmaxf(sc_72, -1.7014118e+38f);
                        sc_72 = fminf(sc_72, 1.7014118e+38f);
                        unsigned int u9_9 = __as_u32(sc_72);
                        unsigned int lowb9_9 = (u9_9 ^ (unsigned int)((int)u9_9 >> 31) & 511) & 511;
                        unsigned int km9_9 = 536870912 | lowb9_9 << 9 | (unsigned int)(tb + 144);
                        cbuf[pos9] = km9_9;
                        pos9 = pos9 + 1;
                    }
                    if ((mm & 1024) != 0) {
                        float sc_73 = __uint_as_float(cbt[10]);
                        sc_73 = fmaxf(sc_73, -1.7014118e+38f);
                        sc_73 = fminf(sc_73, 1.7014118e+38f);
                        unsigned int u9_10 = __as_u32(sc_73);
                        unsigned int lowb9_10 = (u9_10 ^ (unsigned int)((int)u9_10 >> 31) & 511) & 511;
                        unsigned int km9_10 = 536870912 | lowb9_10 << 9 | (unsigned int)(tb + 160);
                        cbuf[pos9] = km9_10;
                        pos9 = pos9 + 1;
                    }
                    if ((mm & 2048) != 0) {
                        float sc_74 = __uint_as_float(cbt[11]);
                        sc_74 = fmaxf(sc_74, -1.7014118e+38f);
                        sc_74 = fminf(sc_74, 1.7014118e+38f);
                        unsigned int u9_11 = __as_u32(sc_74);
                        unsigned int lowb9_11 = (u9_11 ^ (unsigned int)((int)u9_11 >> 31) & 511) & 511;
                        unsigned int km9_11 = 536870912 | lowb9_11 << 9 | (unsigned int)(tb + 176);
                        cbuf[pos9] = km9_11;
                        pos9 = pos9 + 1;
                    }
                    if ((mm & 4096) != 0) {
                        float sc_75 = __uint_as_float(cbt[12]);
                        sc_75 = fmaxf(sc_75, -1.7014118e+38f);
                        sc_75 = fminf(sc_75, 1.7014118e+38f);
                        unsigned int u9_12 = __as_u32(sc_75);
                        unsigned int lowb9_12 = (u9_12 ^ (unsigned int)((int)u9_12 >> 31) & 511) & 511;
                        unsigned int km9_12 = 536870912 | lowb9_12 << 9 | (unsigned int)(tb + 192);
                        cbuf[pos9] = km9_12;
                        pos9 = pos9 + 1;
                    }
                    if ((mm & 8192) != 0) {
                        float sc_76 = __uint_as_float(cbt[13]);
                        sc_76 = fmaxf(sc_76, -1.7014118e+38f);
                        sc_76 = fminf(sc_76, 1.7014118e+38f);
                        unsigned int u9_13 = __as_u32(sc_76);
                        unsigned int lowb9_13 = (u9_13 ^ (unsigned int)((int)u9_13 >> 31) & 511) & 511;
                        unsigned int km9_13 = 536870912 | lowb9_13 << 9 | (unsigned int)(tb + 208);
                        cbuf[pos9] = km9_13;
                        pos9 = pos9 + 1;
                    }
                    if ((mm & 16384) != 0) {
                        float sc_77 = __uint_as_float(cbt[14]);
                        sc_77 = fmaxf(sc_77, -1.7014118e+38f);
                        sc_77 = fminf(sc_77, 1.7014118e+38f);
                        unsigned int u9_14 = __as_u32(sc_77);
                        unsigned int lowb9_14 = (u9_14 ^ (unsigned int)((int)u9_14 >> 31) & 511) & 511;
                        unsigned int km9_14 = 536870912 | lowb9_14 << 9 | (unsigned int)(tb + 224);
                        cbuf[pos9] = km9_14;
                        pos9 = pos9 + 1;
                    }
                    if ((mm & 32768) != 0) {
                        float sc_78 = __uint_as_float(cbt[15]);
                        sc_78 = fmaxf(sc_78, -1.7014118e+38f);
                        sc_78 = fminf(sc_78, 1.7014118e+38f);
                        unsigned int u9_15 = __as_u32(sc_78);
                        unsigned int lowb9_15 = (u9_15 ^ (unsigned int)((int)u9_15 >> 31) & 511) & 511;
                        unsigned int km9_15 = 536870912 | lowb9_15 << 9 | (unsigned int)(tb + 240);
                        cbuf[pos9] = km9_15;
                        pos9 = pos9 + 1;
                    }
                    if ((mm & 65536) != 0) {
                        float sc_79 = __uint_as_float(cbt[16]);
                        sc_79 = fmaxf(sc_79, -1.7014118e+38f);
                        sc_79 = fminf(sc_79, 1.7014118e+38f);
                        unsigned int u9_16 = __as_u32(sc_79);
                        unsigned int lowb9_16 = (u9_16 ^ (unsigned int)((int)u9_16 >> 31) & 511) & 511;
                        unsigned int km9_16 = 536870912 | lowb9_16 << 9 | (unsigned int)(tb + 256);
                        cbuf[pos9] = km9_16;
                        pos9 = pos9 + 1;
                    }
                    if ((mm & 131072) != 0) {
                        float sc_80 = __uint_as_float(cbt[17]);
                        sc_80 = fmaxf(sc_80, -1.7014118e+38f);
                        sc_80 = fminf(sc_80, 1.7014118e+38f);
                        unsigned int u9_17 = __as_u32(sc_80);
                        unsigned int lowb9_17 = (u9_17 ^ (unsigned int)((int)u9_17 >> 31) & 511) & 511;
                        unsigned int km9_17 = 536870912 | lowb9_17 << 9 | (unsigned int)(tb + 272);
                        cbuf[pos9] = km9_17;
                        pos9 = pos9 + 1;
                    }
                    if ((mm & 262144) != 0) {
                        float sc_81 = __uint_as_float(cbt[18]);
                        sc_81 = fmaxf(sc_81, -1.7014118e+38f);
                        sc_81 = fminf(sc_81, 1.7014118e+38f);
                        unsigned int u9_18 = __as_u32(sc_81);
                        unsigned int lowb9_18 = (u9_18 ^ (unsigned int)((int)u9_18 >> 31) & 511) & 511;
                        unsigned int km9_18 = 536870912 | lowb9_18 << 9 | (unsigned int)(tb + 288);
                        cbuf[pos9] = km9_18;
                        pos9 = pos9 + 1;
                    }
                    if ((mm & 524288) != 0) {
                        float sc_82 = __uint_as_float(cbt[19]);
                        sc_82 = fmaxf(sc_82, -1.7014118e+38f);
                        sc_82 = fminf(sc_82, 1.7014118e+38f);
                        unsigned int u9_19 = __as_u32(sc_82);
                        unsigned int lowb9_19 = (u9_19 ^ (unsigned int)((int)u9_19 >> 31) & 511) & 511;
                        unsigned int km9_19 = 536870912 | lowb9_19 << 9 | (unsigned int)(tb + 304);
                        cbuf[pos9] = km9_19;
                        pos9 = pos9 + 1;
                    }
                    if ((mm & 1048576) != 0) {
                        float sc_83 = __uint_as_float(cbt[20]);
                        sc_83 = fmaxf(sc_83, -1.7014118e+38f);
                        sc_83 = fminf(sc_83, 1.7014118e+38f);
                        unsigned int u9_20 = __as_u32(sc_83);
                        unsigned int lowb9_20 = (u9_20 ^ (unsigned int)((int)u9_20 >> 31) & 511) & 511;
                        unsigned int km9_20 = 536870912 | lowb9_20 << 9 | (unsigned int)(tb + 320);
                        cbuf[pos9] = km9_20;
                        pos9 = pos9 + 1;
                    }
                    if ((mm & 2097152) != 0) {
                        float sc_84 = __uint_as_float(cbt[21]);
                        sc_84 = fmaxf(sc_84, -1.7014118e+38f);
                        sc_84 = fminf(sc_84, 1.7014118e+38f);
                        unsigned int u9_21 = __as_u32(sc_84);
                        unsigned int lowb9_21 = (u9_21 ^ (unsigned int)((int)u9_21 >> 31) & 511) & 511;
                        unsigned int km9_21 = 536870912 | lowb9_21 << 9 | (unsigned int)(tb + 336);
                        cbuf[pos9] = km9_21;
                        pos9 = pos9 + 1;
                    }
                    if ((mm & 4194304) != 0) {
                        float sc_85 = __uint_as_float(cbt[22]);
                        sc_85 = fmaxf(sc_85, -1.7014118e+38f);
                        sc_85 = fminf(sc_85, 1.7014118e+38f);
                        unsigned int u9_22 = __as_u32(sc_85);
                        unsigned int lowb9_22 = (u9_22 ^ (unsigned int)((int)u9_22 >> 31) & 511) & 511;
                        unsigned int km9_22 = 536870912 | lowb9_22 << 9 | (unsigned int)(tb + 352);
                        cbuf[pos9] = km9_22;
                        pos9 = pos9 + 1;
                    }
                    if ((mm & 8388608) != 0) {
                        float sc_86 = __uint_as_float(cbt[23]);
                        sc_86 = fmaxf(sc_86, -1.7014118e+38f);
                        sc_86 = fminf(sc_86, 1.7014118e+38f);
                        unsigned int u9_23 = __as_u32(sc_86);
                        unsigned int lowb9_23 = (u9_23 ^ (unsigned int)((int)u9_23 >> 31) & 511) & 511;
                        unsigned int km9_23 = 536870912 | lowb9_23 << 9 | (unsigned int)(tb + 368);
                        cbuf[pos9] = km9_23;
                        pos9 = pos9 + 1;
                    }
                    if ((mm & 16777216) != 0) {
                        float sc_87 = __uint_as_float(cbt[24]);
                        sc_87 = fmaxf(sc_87, -1.7014118e+38f);
                        sc_87 = fminf(sc_87, 1.7014118e+38f);
                        unsigned int u9_24 = __as_u32(sc_87);
                        unsigned int lowb9_24 = (u9_24 ^ (unsigned int)((int)u9_24 >> 31) & 511) & 511;
                        unsigned int km9_24 = 536870912 | lowb9_24 << 9 | (unsigned int)(tb + 384);
                        cbuf[pos9] = km9_24;
                        pos9 = pos9 + 1;
                    }
                    if ((mm & 33554432) != 0) {
                        float sc_88 = __uint_as_float(cbt[25]);
                        sc_88 = fmaxf(sc_88, -1.7014118e+38f);
                        sc_88 = fminf(sc_88, 1.7014118e+38f);
                        unsigned int u9_25 = __as_u32(sc_88);
                        unsigned int lowb9_25 = (u9_25 ^ (unsigned int)((int)u9_25 >> 31) & 511) & 511;
                        unsigned int km9_25 = 536870912 | lowb9_25 << 9 | (unsigned int)(tb + 400);
                        cbuf[pos9] = km9_25;
                        pos9 = pos9 + 1;
                    }
                    if ((mm & 67108864) != 0) {
                        float sc_89 = __uint_as_float(cbt[26]);
                        sc_89 = fmaxf(sc_89, -1.7014118e+38f);
                        sc_89 = fminf(sc_89, 1.7014118e+38f);
                        unsigned int u9_26 = __as_u32(sc_89);
                        unsigned int lowb9_26 = (u9_26 ^ (unsigned int)((int)u9_26 >> 31) & 511) & 511;
                        unsigned int km9_26 = 536870912 | lowb9_26 << 9 | (unsigned int)(tb + 416);
                        cbuf[pos9] = km9_26;
                        pos9 = pos9 + 1;
                    }
                    if ((mm & 134217728) != 0) {
                        float sc_90 = __uint_as_float(cbt[27]);
                        sc_90 = fmaxf(sc_90, -1.7014118e+38f);
                        sc_90 = fminf(sc_90, 1.7014118e+38f);
                        unsigned int u9_27 = __as_u32(sc_90);
                        unsigned int lowb9_27 = (u9_27 ^ (unsigned int)((int)u9_27 >> 31) & 511) & 511;
                        unsigned int km9_27 = 536870912 | lowb9_27 << 9 | (unsigned int)(tb + 432);
                        cbuf[pos9] = km9_27;
                        pos9 = pos9 + 1;
                    }
                    if ((mm & 268435456) != 0) {
                        float sc_91 = __uint_as_float(cbt[28]);
                        sc_91 = fmaxf(sc_91, -1.7014118e+38f);
                        sc_91 = fminf(sc_91, 1.7014118e+38f);
                        unsigned int u9_28 = __as_u32(sc_91);
                        unsigned int lowb9_28 = (u9_28 ^ (unsigned int)((int)u9_28 >> 31) & 511) & 511;
                        unsigned int km9_28 = 536870912 | lowb9_28 << 9 | (unsigned int)(tb + 448);
                        cbuf[pos9] = km9_28;
                        pos9 = pos9 + 1;
                    }
                    if ((mm & 536870912) != 0) {
                        float sc_92 = __uint_as_float(cbt[29]);
                        sc_92 = fmaxf(sc_92, -1.7014118e+38f);
                        sc_92 = fminf(sc_92, 1.7014118e+38f);
                        unsigned int u9_29 = __as_u32(sc_92);
                        unsigned int lowb9_29 = (u9_29 ^ (unsigned int)((int)u9_29 >> 31) & 511) & 511;
                        unsigned int km9_29 = 536870912 | lowb9_29 << 9 | (unsigned int)(tb + 464);
                        cbuf[pos9] = km9_29;
                        pos9 = pos9 + 1;
                    }
                    if ((mm & 1073741824) != 0) {
                        float sc_93 = __uint_as_float(cbt[30]);
                        sc_93 = fmaxf(sc_93, -1.7014118e+38f);
                        sc_93 = fminf(sc_93, 1.7014118e+38f);
                        unsigned int u9_30 = __as_u32(sc_93);
                        unsigned int lowb9_30 = (u9_30 ^ (unsigned int)((int)u9_30 >> 31) & 511) & 511;
                        unsigned int km9_30 = 536870912 | lowb9_30 << 9 | (unsigned int)(tb + 480);
                        cbuf[pos9] = km9_30;
                        pos9 = pos9 + 1;
                    }
                    if ((mm & 2147483648u) != 0) {
                        float sc_94 = __uint_as_float(cbt[31]);
                        sc_94 = fmaxf(sc_94, -1.7014118e+38f);
                        sc_94 = fminf(sc_94, 1.7014118e+38f);
                        unsigned int u9_31 = __as_u32(sc_94);
                        unsigned int lowb9_31 = (u9_31 ^ (unsigned int)((int)u9_31 >> 31) & 511) & 511;
                        unsigned int km9_31 = 536870912 | lowb9_31 << 9 | (unsigned int)(tb + 496);
                        cbuf[pos9] = km9_31;
                        pos9 = pos9 + 1;
                    }
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
                float kbt2[32];
                float r2o = neg_inf;
                int thr = 2147483647;
                if (qc != 4294967295u) {
                    unsigned int img = qc ^ (unsigned int)((int)qc >> 31) & 2147483647;
                    thr = (int)img;
                }
                float sc_95 = __uint_as_float(cbt[0]);
                sc_95 = fmaxf(sc_95, -1.7014118e+38f);
                sc_95 = fminf(sc_95, 1.7014118e+38f);
                unsigned int u_1_0 = __as_u32(sc_95);
                unsigned int cls_1_0 = u_1_0 & 4294966784u;
                unsigned int key = 0;
                unsigned int img_1 = cls_1_0 ^ (unsigned int)((int)cls_1_0 >> 31) & 2147483647;
                if (thr < (int)img_1) {
                    key = 1073741824 | (unsigned int)tb;
                }
                if (cls_1_0 == qc) {
                    unsigned int lowb = (u_1_0 ^ (unsigned int)((int)u_1_0 >> 31) & 511) & 511;
                    key = 536870912 | lowb << 9 | (unsigned int)tb;
                }
                kbt2[0] = __uint_as_float(key);
                float sc_96 = __uint_as_float(cbt[1]);
                sc_96 = fmaxf(sc_96, -1.7014118e+38f);
                sc_96 = fminf(sc_96, 1.7014118e+38f);
                unsigned int u_1_1 = __as_u32(sc_96);
                unsigned int cls_1_1 = u_1_1 & 4294966784u;
                unsigned int key_0 = 0;
                unsigned int img_0 = cls_1_1 ^ (unsigned int)((int)cls_1_1 >> 31) & 2147483647;
                if (thr < (int)img_0) {
                    key_0 = 1073741824 | (unsigned int)(tb + 16);
                }
                if (cls_1_1 == qc) {
                    unsigned int lowb_1 = (u_1_1 ^ (unsigned int)((int)u_1_1 >> 31) & 511) & 511;
                    key_0 = 536870912 | lowb_1 << 9 | (unsigned int)(tb + 16);
                }
                kbt2[1] = __uint_as_float(key_0);
                float sc_97 = __uint_as_float(cbt[2]);
                sc_97 = fmaxf(sc_97, -1.7014118e+38f);
                sc_97 = fminf(sc_97, 1.7014118e+38f);
                unsigned int u_1_2 = __as_u32(sc_97);
                unsigned int cls_1_2 = u_1_2 & 4294966784u;
                unsigned int key_1 = 0;
                unsigned int img_1_1 = cls_1_2 ^ (unsigned int)((int)cls_1_2 >> 31) & 2147483647;
                if (thr < (int)img_1_1) {
                    key_1 = 1073741824 | (unsigned int)(tb + 32);
                }
                if (cls_1_2 == qc) {
                    unsigned int lowb_2 = (u_1_2 ^ (unsigned int)((int)u_1_2 >> 31) & 511) & 511;
                    key_1 = 536870912 | lowb_2 << 9 | (unsigned int)(tb + 32);
                }
                kbt2[2] = __uint_as_float(key_1);
                float sc_98 = __uint_as_float(cbt[3]);
                sc_98 = fmaxf(sc_98, -1.7014118e+38f);
                sc_98 = fminf(sc_98, 1.7014118e+38f);
                unsigned int u_1_3 = __as_u32(sc_98);
                unsigned int cls_1_3 = u_1_3 & 4294966784u;
                unsigned int key_2 = 0;
                unsigned int img_2 = cls_1_3 ^ (unsigned int)((int)cls_1_3 >> 31) & 2147483647;
                if (thr < (int)img_2) {
                    key_2 = 1073741824 | (unsigned int)(tb + 48);
                }
                if (cls_1_3 == qc) {
                    unsigned int lowb_3 = (u_1_3 ^ (unsigned int)((int)u_1_3 >> 31) & 511) & 511;
                    key_2 = 536870912 | lowb_3 << 9 | (unsigned int)(tb + 48);
                }
                kbt2[3] = __uint_as_float(key_2);
                float sc_99 = __uint_as_float(cbt[4]);
                sc_99 = fmaxf(sc_99, -1.7014118e+38f);
                sc_99 = fminf(sc_99, 1.7014118e+38f);
                unsigned int u_1_4 = __as_u32(sc_99);
                unsigned int cls_1_4 = u_1_4 & 4294966784u;
                unsigned int key_3 = 0;
                unsigned int img_3 = cls_1_4 ^ (unsigned int)((int)cls_1_4 >> 31) & 2147483647;
                if (thr < (int)img_3) {
                    key_3 = 1073741824 | (unsigned int)(tb + 64);
                }
                if (cls_1_4 == qc) {
                    unsigned int lowb_4 = (u_1_4 ^ (unsigned int)((int)u_1_4 >> 31) & 511) & 511;
                    key_3 = 536870912 | lowb_4 << 9 | (unsigned int)(tb + 64);
                }
                kbt2[4] = __uint_as_float(key_3);
                float sc_100 = __uint_as_float(cbt[5]);
                sc_100 = fmaxf(sc_100, -1.7014118e+38f);
                sc_100 = fminf(sc_100, 1.7014118e+38f);
                unsigned int u_1_5 = __as_u32(sc_100);
                unsigned int cls_1_5 = u_1_5 & 4294966784u;
                unsigned int key_4 = 0;
                unsigned int img_4 = cls_1_5 ^ (unsigned int)((int)cls_1_5 >> 31) & 2147483647;
                if (thr < (int)img_4) {
                    key_4 = 1073741824 | (unsigned int)(tb + 80);
                }
                if (cls_1_5 == qc) {
                    unsigned int lowb_5 = (u_1_5 ^ (unsigned int)((int)u_1_5 >> 31) & 511) & 511;
                    key_4 = 536870912 | lowb_5 << 9 | (unsigned int)(tb + 80);
                }
                kbt2[5] = __uint_as_float(key_4);
                float sc_101 = __uint_as_float(cbt[6]);
                sc_101 = fmaxf(sc_101, -1.7014118e+38f);
                sc_101 = fminf(sc_101, 1.7014118e+38f);
                unsigned int u_1_6 = __as_u32(sc_101);
                unsigned int cls_1_6 = u_1_6 & 4294966784u;
                unsigned int key_5 = 0;
                unsigned int img_5 = cls_1_6 ^ (unsigned int)((int)cls_1_6 >> 31) & 2147483647;
                if (thr < (int)img_5) {
                    key_5 = 1073741824 | (unsigned int)(tb + 96);
                }
                if (cls_1_6 == qc) {
                    unsigned int lowb_6 = (u_1_6 ^ (unsigned int)((int)u_1_6 >> 31) & 511) & 511;
                    key_5 = 536870912 | lowb_6 << 9 | (unsigned int)(tb + 96);
                }
                kbt2[6] = __uint_as_float(key_5);
                float sc_102 = __uint_as_float(cbt[7]);
                sc_102 = fmaxf(sc_102, -1.7014118e+38f);
                sc_102 = fminf(sc_102, 1.7014118e+38f);
                unsigned int u_1_7 = __as_u32(sc_102);
                unsigned int cls_1_7 = u_1_7 & 4294966784u;
                unsigned int key_6 = 0;
                unsigned int img_6 = cls_1_7 ^ (unsigned int)((int)cls_1_7 >> 31) & 2147483647;
                if (thr < (int)img_6) {
                    key_6 = 1073741824 | (unsigned int)(tb + 112);
                }
                if (cls_1_7 == qc) {
                    unsigned int lowb_7 = (u_1_7 ^ (unsigned int)((int)u_1_7 >> 31) & 511) & 511;
                    key_6 = 536870912 | lowb_7 << 9 | (unsigned int)(tb + 112);
                }
                kbt2[7] = __uint_as_float(key_6);
                float sc_103 = __uint_as_float(cbt[8]);
                sc_103 = fmaxf(sc_103, -1.7014118e+38f);
                sc_103 = fminf(sc_103, 1.7014118e+38f);
                unsigned int u_1_8 = __as_u32(sc_103);
                unsigned int cls_1_8 = u_1_8 & 4294966784u;
                unsigned int key_7 = 0;
                unsigned int img_7 = cls_1_8 ^ (unsigned int)((int)cls_1_8 >> 31) & 2147483647;
                if (thr < (int)img_7) {
                    key_7 = 1073741824 | (unsigned int)(tb + 128);
                }
                if (cls_1_8 == qc) {
                    unsigned int lowb_8 = (u_1_8 ^ (unsigned int)((int)u_1_8 >> 31) & 511) & 511;
                    key_7 = 536870912 | lowb_8 << 9 | (unsigned int)(tb + 128);
                }
                kbt2[8] = __uint_as_float(key_7);
                float sc_104 = __uint_as_float(cbt[9]);
                sc_104 = fmaxf(sc_104, -1.7014118e+38f);
                sc_104 = fminf(sc_104, 1.7014118e+38f);
                unsigned int u_1_9 = __as_u32(sc_104);
                unsigned int cls_1_9 = u_1_9 & 4294966784u;
                unsigned int key_8 = 0;
                unsigned int img_8 = cls_1_9 ^ (unsigned int)((int)cls_1_9 >> 31) & 2147483647;
                if (thr < (int)img_8) {
                    key_8 = 1073741824 | (unsigned int)(tb + 144);
                }
                if (cls_1_9 == qc) {
                    unsigned int lowb_9 = (u_1_9 ^ (unsigned int)((int)u_1_9 >> 31) & 511) & 511;
                    key_8 = 536870912 | lowb_9 << 9 | (unsigned int)(tb + 144);
                }
                kbt2[9] = __uint_as_float(key_8);
                float sc_105 = __uint_as_float(cbt[10]);
                sc_105 = fmaxf(sc_105, -1.7014118e+38f);
                sc_105 = fminf(sc_105, 1.7014118e+38f);
                unsigned int u_1_10 = __as_u32(sc_105);
                unsigned int cls_1_10 = u_1_10 & 4294966784u;
                unsigned int key_9 = 0;
                unsigned int img_9 = cls_1_10 ^ (unsigned int)((int)cls_1_10 >> 31) & 2147483647;
                if (thr < (int)img_9) {
                    key_9 = 1073741824 | (unsigned int)(tb + 160);
                }
                if (cls_1_10 == qc) {
                    unsigned int lowb_10 = (u_1_10 ^ (unsigned int)((int)u_1_10 >> 31) & 511) & 511;
                    key_9 = 536870912 | lowb_10 << 9 | (unsigned int)(tb + 160);
                }
                kbt2[10] = __uint_as_float(key_9);
                float sc_106 = __uint_as_float(cbt[11]);
                sc_106 = fmaxf(sc_106, -1.7014118e+38f);
                sc_106 = fminf(sc_106, 1.7014118e+38f);
                unsigned int u_1_11 = __as_u32(sc_106);
                unsigned int cls_1_11 = u_1_11 & 4294966784u;
                unsigned int key_10 = 0;
                unsigned int img_10 = cls_1_11 ^ (unsigned int)((int)cls_1_11 >> 31) & 2147483647;
                if (thr < (int)img_10) {
                    key_10 = 1073741824 | (unsigned int)(tb + 176);
                }
                if (cls_1_11 == qc) {
                    unsigned int lowb_11 = (u_1_11 ^ (unsigned int)((int)u_1_11 >> 31) & 511) & 511;
                    key_10 = 536870912 | lowb_11 << 9 | (unsigned int)(tb + 176);
                }
                kbt2[11] = __uint_as_float(key_10);
                float sc_107 = __uint_as_float(cbt[12]);
                sc_107 = fmaxf(sc_107, -1.7014118e+38f);
                sc_107 = fminf(sc_107, 1.7014118e+38f);
                unsigned int u_1_12 = __as_u32(sc_107);
                unsigned int cls_1_12 = u_1_12 & 4294966784u;
                unsigned int key_11 = 0;
                unsigned int img_11 = cls_1_12 ^ (unsigned int)((int)cls_1_12 >> 31) & 2147483647;
                if (thr < (int)img_11) {
                    key_11 = 1073741824 | (unsigned int)(tb + 192);
                }
                if (cls_1_12 == qc) {
                    unsigned int lowb_12 = (u_1_12 ^ (unsigned int)((int)u_1_12 >> 31) & 511) & 511;
                    key_11 = 536870912 | lowb_12 << 9 | (unsigned int)(tb + 192);
                }
                kbt2[12] = __uint_as_float(key_11);
                float sc_108 = __uint_as_float(cbt[13]);
                sc_108 = fmaxf(sc_108, -1.7014118e+38f);
                sc_108 = fminf(sc_108, 1.7014118e+38f);
                unsigned int u_1_13 = __as_u32(sc_108);
                unsigned int cls_1_13 = u_1_13 & 4294966784u;
                unsigned int key_12 = 0;
                unsigned int img_12 = cls_1_13 ^ (unsigned int)((int)cls_1_13 >> 31) & 2147483647;
                if (thr < (int)img_12) {
                    key_12 = 1073741824 | (unsigned int)(tb + 208);
                }
                if (cls_1_13 == qc) {
                    unsigned int lowb_13 = (u_1_13 ^ (unsigned int)((int)u_1_13 >> 31) & 511) & 511;
                    key_12 = 536870912 | lowb_13 << 9 | (unsigned int)(tb + 208);
                }
                kbt2[13] = __uint_as_float(key_12);
                float sc_109 = __uint_as_float(cbt[14]);
                sc_109 = fmaxf(sc_109, -1.7014118e+38f);
                sc_109 = fminf(sc_109, 1.7014118e+38f);
                unsigned int u_1_14 = __as_u32(sc_109);
                unsigned int cls_1_14 = u_1_14 & 4294966784u;
                unsigned int key_13 = 0;
                unsigned int img_13 = cls_1_14 ^ (unsigned int)((int)cls_1_14 >> 31) & 2147483647;
                if (thr < (int)img_13) {
                    key_13 = 1073741824 | (unsigned int)(tb + 224);
                }
                if (cls_1_14 == qc) {
                    unsigned int lowb_14 = (u_1_14 ^ (unsigned int)((int)u_1_14 >> 31) & 511) & 511;
                    key_13 = 536870912 | lowb_14 << 9 | (unsigned int)(tb + 224);
                }
                kbt2[14] = __uint_as_float(key_13);
                float sc_110 = __uint_as_float(cbt[15]);
                sc_110 = fmaxf(sc_110, -1.7014118e+38f);
                sc_110 = fminf(sc_110, 1.7014118e+38f);
                unsigned int u_1_15 = __as_u32(sc_110);
                unsigned int cls_1_15 = u_1_15 & 4294966784u;
                unsigned int key_14 = 0;
                unsigned int img_14 = cls_1_15 ^ (unsigned int)((int)cls_1_15 >> 31) & 2147483647;
                if (thr < (int)img_14) {
                    key_14 = 1073741824 | (unsigned int)(tb + 240);
                }
                if (cls_1_15 == qc) {
                    unsigned int lowb_15 = (u_1_15 ^ (unsigned int)((int)u_1_15 >> 31) & 511) & 511;
                    key_14 = 536870912 | lowb_15 << 9 | (unsigned int)(tb + 240);
                }
                kbt2[15] = __uint_as_float(key_14);
                float sc_111 = __uint_as_float(cbt[16]);
                sc_111 = fmaxf(sc_111, -1.7014118e+38f);
                sc_111 = fminf(sc_111, 1.7014118e+38f);
                unsigned int u_1_16 = __as_u32(sc_111);
                unsigned int cls_1_16 = u_1_16 & 4294966784u;
                unsigned int key_15 = 0;
                unsigned int img_15 = cls_1_16 ^ (unsigned int)((int)cls_1_16 >> 31) & 2147483647;
                if (thr < (int)img_15) {
                    key_15 = 1073741824 | (unsigned int)(tb + 256);
                }
                if (cls_1_16 == qc) {
                    unsigned int lowb_16 = (u_1_16 ^ (unsigned int)((int)u_1_16 >> 31) & 511) & 511;
                    key_15 = 536870912 | lowb_16 << 9 | (unsigned int)(tb + 256);
                }
                kbt2[16] = __uint_as_float(key_15);
                float sc_112 = __uint_as_float(cbt[17]);
                sc_112 = fmaxf(sc_112, -1.7014118e+38f);
                sc_112 = fminf(sc_112, 1.7014118e+38f);
                unsigned int u_1_17 = __as_u32(sc_112);
                unsigned int cls_1_17 = u_1_17 & 4294966784u;
                unsigned int key_16 = 0;
                unsigned int img_16 = cls_1_17 ^ (unsigned int)((int)cls_1_17 >> 31) & 2147483647;
                if (thr < (int)img_16) {
                    key_16 = 1073741824 | (unsigned int)(tb + 272);
                }
                if (cls_1_17 == qc) {
                    unsigned int lowb_17 = (u_1_17 ^ (unsigned int)((int)u_1_17 >> 31) & 511) & 511;
                    key_16 = 536870912 | lowb_17 << 9 | (unsigned int)(tb + 272);
                }
                kbt2[17] = __uint_as_float(key_16);
                float sc_113 = __uint_as_float(cbt[18]);
                sc_113 = fmaxf(sc_113, -1.7014118e+38f);
                sc_113 = fminf(sc_113, 1.7014118e+38f);
                unsigned int u_1_18 = __as_u32(sc_113);
                unsigned int cls_1_18 = u_1_18 & 4294966784u;
                unsigned int key_17 = 0;
                unsigned int img_17 = cls_1_18 ^ (unsigned int)((int)cls_1_18 >> 31) & 2147483647;
                if (thr < (int)img_17) {
                    key_17 = 1073741824 | (unsigned int)(tb + 288);
                }
                if (cls_1_18 == qc) {
                    unsigned int lowb_18 = (u_1_18 ^ (unsigned int)((int)u_1_18 >> 31) & 511) & 511;
                    key_17 = 536870912 | lowb_18 << 9 | (unsigned int)(tb + 288);
                }
                kbt2[18] = __uint_as_float(key_17);
                float sc_114 = __uint_as_float(cbt[19]);
                sc_114 = fmaxf(sc_114, -1.7014118e+38f);
                sc_114 = fminf(sc_114, 1.7014118e+38f);
                unsigned int u_1_19 = __as_u32(sc_114);
                unsigned int cls_1_19 = u_1_19 & 4294966784u;
                unsigned int key_18 = 0;
                unsigned int img_18 = cls_1_19 ^ (unsigned int)((int)cls_1_19 >> 31) & 2147483647;
                if (thr < (int)img_18) {
                    key_18 = 1073741824 | (unsigned int)(tb + 304);
                }
                if (cls_1_19 == qc) {
                    unsigned int lowb_19 = (u_1_19 ^ (unsigned int)((int)u_1_19 >> 31) & 511) & 511;
                    key_18 = 536870912 | lowb_19 << 9 | (unsigned int)(tb + 304);
                }
                kbt2[19] = __uint_as_float(key_18);
                float sc_115 = __uint_as_float(cbt[20]);
                sc_115 = fmaxf(sc_115, -1.7014118e+38f);
                sc_115 = fminf(sc_115, 1.7014118e+38f);
                unsigned int u_1_20 = __as_u32(sc_115);
                unsigned int cls_1_20 = u_1_20 & 4294966784u;
                unsigned int key_19 = 0;
                unsigned int img_19 = cls_1_20 ^ (unsigned int)((int)cls_1_20 >> 31) & 2147483647;
                if (thr < (int)img_19) {
                    key_19 = 1073741824 | (unsigned int)(tb + 320);
                }
                if (cls_1_20 == qc) {
                    unsigned int lowb_20 = (u_1_20 ^ (unsigned int)((int)u_1_20 >> 31) & 511) & 511;
                    key_19 = 536870912 | lowb_20 << 9 | (unsigned int)(tb + 320);
                }
                kbt2[20] = __uint_as_float(key_19);
                float sc_116 = __uint_as_float(cbt[21]);
                sc_116 = fmaxf(sc_116, -1.7014118e+38f);
                sc_116 = fminf(sc_116, 1.7014118e+38f);
                unsigned int u_1_21 = __as_u32(sc_116);
                unsigned int cls_1_21 = u_1_21 & 4294966784u;
                unsigned int key_20 = 0;
                unsigned int img_20 = cls_1_21 ^ (unsigned int)((int)cls_1_21 >> 31) & 2147483647;
                if (thr < (int)img_20) {
                    key_20 = 1073741824 | (unsigned int)(tb + 336);
                }
                if (cls_1_21 == qc) {
                    unsigned int lowb_21 = (u_1_21 ^ (unsigned int)((int)u_1_21 >> 31) & 511) & 511;
                    key_20 = 536870912 | lowb_21 << 9 | (unsigned int)(tb + 336);
                }
                kbt2[21] = __uint_as_float(key_20);
                float sc_117 = __uint_as_float(cbt[22]);
                sc_117 = fmaxf(sc_117, -1.7014118e+38f);
                sc_117 = fminf(sc_117, 1.7014118e+38f);
                unsigned int u_1_22 = __as_u32(sc_117);
                unsigned int cls_1_22 = u_1_22 & 4294966784u;
                unsigned int key_21 = 0;
                unsigned int img_21 = cls_1_22 ^ (unsigned int)((int)cls_1_22 >> 31) & 2147483647;
                if (thr < (int)img_21) {
                    key_21 = 1073741824 | (unsigned int)(tb + 352);
                }
                if (cls_1_22 == qc) {
                    unsigned int lowb_22 = (u_1_22 ^ (unsigned int)((int)u_1_22 >> 31) & 511) & 511;
                    key_21 = 536870912 | lowb_22 << 9 | (unsigned int)(tb + 352);
                }
                kbt2[22] = __uint_as_float(key_21);
                float sc_118 = __uint_as_float(cbt[23]);
                sc_118 = fmaxf(sc_118, -1.7014118e+38f);
                sc_118 = fminf(sc_118, 1.7014118e+38f);
                unsigned int u_1_23 = __as_u32(sc_118);
                unsigned int cls_1_23 = u_1_23 & 4294966784u;
                unsigned int key_22 = 0;
                unsigned int img_22 = cls_1_23 ^ (unsigned int)((int)cls_1_23 >> 31) & 2147483647;
                if (thr < (int)img_22) {
                    key_22 = 1073741824 | (unsigned int)(tb + 368);
                }
                if (cls_1_23 == qc) {
                    unsigned int lowb_23 = (u_1_23 ^ (unsigned int)((int)u_1_23 >> 31) & 511) & 511;
                    key_22 = 536870912 | lowb_23 << 9 | (unsigned int)(tb + 368);
                }
                kbt2[23] = __uint_as_float(key_22);
                float sc_119 = __uint_as_float(cbt[24]);
                sc_119 = fmaxf(sc_119, -1.7014118e+38f);
                sc_119 = fminf(sc_119, 1.7014118e+38f);
                unsigned int u_1_24 = __as_u32(sc_119);
                unsigned int cls_1_24 = u_1_24 & 4294966784u;
                unsigned int key_23 = 0;
                unsigned int img_23 = cls_1_24 ^ (unsigned int)((int)cls_1_24 >> 31) & 2147483647;
                if (thr < (int)img_23) {
                    key_23 = 1073741824 | (unsigned int)(tb + 384);
                }
                if (cls_1_24 == qc) {
                    unsigned int lowb_24 = (u_1_24 ^ (unsigned int)((int)u_1_24 >> 31) & 511) & 511;
                    key_23 = 536870912 | lowb_24 << 9 | (unsigned int)(tb + 384);
                }
                kbt2[24] = __uint_as_float(key_23);
                float sc_120 = __uint_as_float(cbt[25]);
                sc_120 = fmaxf(sc_120, -1.7014118e+38f);
                sc_120 = fminf(sc_120, 1.7014118e+38f);
                unsigned int u_1_25 = __as_u32(sc_120);
                unsigned int cls_1_25 = u_1_25 & 4294966784u;
                unsigned int key_24 = 0;
                unsigned int img_24 = cls_1_25 ^ (unsigned int)((int)cls_1_25 >> 31) & 2147483647;
                if (thr < (int)img_24) {
                    key_24 = 1073741824 | (unsigned int)(tb + 400);
                }
                if (cls_1_25 == qc) {
                    unsigned int lowb_25 = (u_1_25 ^ (unsigned int)((int)u_1_25 >> 31) & 511) & 511;
                    key_24 = 536870912 | lowb_25 << 9 | (unsigned int)(tb + 400);
                }
                kbt2[25] = __uint_as_float(key_24);
                float sc_121 = __uint_as_float(cbt[26]);
                sc_121 = fmaxf(sc_121, -1.7014118e+38f);
                sc_121 = fminf(sc_121, 1.7014118e+38f);
                unsigned int u_1_26 = __as_u32(sc_121);
                unsigned int cls_1_26 = u_1_26 & 4294966784u;
                unsigned int key_25 = 0;
                unsigned int img_25 = cls_1_26 ^ (unsigned int)((int)cls_1_26 >> 31) & 2147483647;
                if (thr < (int)img_25) {
                    key_25 = 1073741824 | (unsigned int)(tb + 416);
                }
                if (cls_1_26 == qc) {
                    unsigned int lowb_26 = (u_1_26 ^ (unsigned int)((int)u_1_26 >> 31) & 511) & 511;
                    key_25 = 536870912 | lowb_26 << 9 | (unsigned int)(tb + 416);
                }
                kbt2[26] = __uint_as_float(key_25);
                float sc_122 = __uint_as_float(cbt[27]);
                sc_122 = fmaxf(sc_122, -1.7014118e+38f);
                sc_122 = fminf(sc_122, 1.7014118e+38f);
                unsigned int u_1_27 = __as_u32(sc_122);
                unsigned int cls_1_27 = u_1_27 & 4294966784u;
                unsigned int key_26 = 0;
                unsigned int img_26 = cls_1_27 ^ (unsigned int)((int)cls_1_27 >> 31) & 2147483647;
                if (thr < (int)img_26) {
                    key_26 = 1073741824 | (unsigned int)(tb + 432);
                }
                if (cls_1_27 == qc) {
                    unsigned int lowb_27 = (u_1_27 ^ (unsigned int)((int)u_1_27 >> 31) & 511) & 511;
                    key_26 = 536870912 | lowb_27 << 9 | (unsigned int)(tb + 432);
                }
                kbt2[27] = __uint_as_float(key_26);
                float sc_123 = __uint_as_float(cbt[28]);
                sc_123 = fmaxf(sc_123, -1.7014118e+38f);
                sc_123 = fminf(sc_123, 1.7014118e+38f);
                unsigned int u_1_28 = __as_u32(sc_123);
                unsigned int cls_1_28 = u_1_28 & 4294966784u;
                unsigned int key_27 = 0;
                unsigned int img_27 = cls_1_28 ^ (unsigned int)((int)cls_1_28 >> 31) & 2147483647;
                if (thr < (int)img_27) {
                    key_27 = 1073741824 | (unsigned int)(tb + 448);
                }
                if (cls_1_28 == qc) {
                    unsigned int lowb_28 = (u_1_28 ^ (unsigned int)((int)u_1_28 >> 31) & 511) & 511;
                    key_27 = 536870912 | lowb_28 << 9 | (unsigned int)(tb + 448);
                }
                kbt2[28] = __uint_as_float(key_27);
                float sc_124 = __uint_as_float(cbt[29]);
                sc_124 = fmaxf(sc_124, -1.7014118e+38f);
                sc_124 = fminf(sc_124, 1.7014118e+38f);
                unsigned int u_1_29 = __as_u32(sc_124);
                unsigned int cls_1_29 = u_1_29 & 4294966784u;
                unsigned int key_28 = 0;
                unsigned int img_28 = cls_1_29 ^ (unsigned int)((int)cls_1_29 >> 31) & 2147483647;
                if (thr < (int)img_28) {
                    key_28 = 1073741824 | (unsigned int)(tb + 464);
                }
                if (cls_1_29 == qc) {
                    unsigned int lowb_29 = (u_1_29 ^ (unsigned int)((int)u_1_29 >> 31) & 511) & 511;
                    key_28 = 536870912 | lowb_29 << 9 | (unsigned int)(tb + 464);
                }
                kbt2[29] = __uint_as_float(key_28);
                float sc_125 = __uint_as_float(cbt[30]);
                sc_125 = fmaxf(sc_125, -1.7014118e+38f);
                sc_125 = fminf(sc_125, 1.7014118e+38f);
                unsigned int u_1_30 = __as_u32(sc_125);
                unsigned int cls_1_30 = u_1_30 & 4294966784u;
                unsigned int key_29 = 0;
                unsigned int img_29 = cls_1_30 ^ (unsigned int)((int)cls_1_30 >> 31) & 2147483647;
                if (thr < (int)img_29) {
                    key_29 = 1073741824 | (unsigned int)(tb + 480);
                }
                if (cls_1_30 == qc) {
                    unsigned int lowb_30 = (u_1_30 ^ (unsigned int)((int)u_1_30 >> 31) & 511) & 511;
                    key_29 = 536870912 | lowb_30 << 9 | (unsigned int)(tb + 480);
                }
                kbt2[30] = __uint_as_float(key_29);
                float sc_126 = __uint_as_float(cbt[31]);
                sc_126 = fmaxf(sc_126, -1.7014118e+38f);
                sc_126 = fminf(sc_126, 1.7014118e+38f);
                unsigned int u_1_31 = __as_u32(sc_126);
                unsigned int cls_1_31 = u_1_31 & 4294966784u;
                unsigned int key_30 = 0;
                unsigned int img_30 = cls_1_31 ^ (unsigned int)((int)cls_1_31 >> 31) & 2147483647;
                if (thr < (int)img_30) {
                    key_30 = 1073741824 | (unsigned int)(tb + 496);
                }
                if (cls_1_31 == qc) {
                    unsigned int lowb_31 = (u_1_31 ^ (unsigned int)((int)u_1_31 >> 31) & 511) & 511;
                    key_30 = 536870912 | lowb_31 << 9 | (unsigned int)(tb + 496);
                }
                kbt2[31] = __uint_as_float(key_30);
                float rr_0 = r2o;
                float hi_1_120 = fmaxf(kbt2[0], kbt2[13]);
                float lo_1_120 = fminf(kbt2[0], kbt2[13]);
                kbt2[0] = hi_1_120;
                kbt2[13] = lo_1_120;
                float hi_1_121 = fmaxf(kbt2[1], kbt2[12]);
                float lo_1_121 = fminf(kbt2[1], kbt2[12]);
                kbt2[1] = hi_1_121;
                kbt2[12] = lo_1_121;
                float hi_1_122 = fmaxf(kbt2[2], kbt2[15]);
                float lo_1_122 = fminf(kbt2[2], kbt2[15]);
                kbt2[2] = hi_1_122;
                kbt2[15] = lo_1_122;
                float hi_1_123 = fmaxf(kbt2[3], kbt2[14]);
                float lo_1_123 = fminf(kbt2[3], kbt2[14]);
                kbt2[3] = hi_1_123;
                kbt2[14] = lo_1_123;
                float hi_1_124 = fmaxf(kbt2[4], kbt2[8]);
                float lo_1_124 = fminf(kbt2[4], kbt2[8]);
                kbt2[4] = hi_1_124;
                kbt2[8] = lo_1_124;
                float hi_1_125 = fmaxf(kbt2[5], kbt2[6]);
                float lo_1_125 = fminf(kbt2[5], kbt2[6]);
                kbt2[5] = hi_1_125;
                kbt2[6] = lo_1_125;
                float hi_1_126 = fmaxf(kbt2[7], kbt2[11]);
                float lo_1_126 = fminf(kbt2[7], kbt2[11]);
                kbt2[7] = hi_1_126;
                kbt2[11] = lo_1_126;
                float hi_1_127 = fmaxf(kbt2[9], kbt2[10]);
                float lo_1_127 = fminf(kbt2[9], kbt2[10]);
                kbt2[9] = hi_1_127;
                kbt2[10] = lo_1_127;
                float hi_1_128 = fmaxf(kbt2[0], kbt2[5]);
                float lo_1_128 = fminf(kbt2[0], kbt2[5]);
                kbt2[0] = hi_1_128;
                kbt2[5] = lo_1_128;
                float hi_1_129 = fmaxf(kbt2[1], kbt2[7]);
                float lo_1_129 = fminf(kbt2[1], kbt2[7]);
                kbt2[1] = hi_1_129;
                kbt2[7] = lo_1_129;
                float hi_1_130 = fmaxf(kbt2[2], kbt2[9]);
                float lo_1_130 = fminf(kbt2[2], kbt2[9]);
                kbt2[2] = hi_1_130;
                kbt2[9] = lo_1_130;
                float hi_1_131 = fmaxf(kbt2[3], kbt2[4]);
                float lo_1_131 = fminf(kbt2[3], kbt2[4]);
                kbt2[3] = hi_1_131;
                kbt2[4] = lo_1_131;
                float hi_1_132 = fmaxf(kbt2[6], kbt2[13]);
                float lo_1_132 = fminf(kbt2[6], kbt2[13]);
                kbt2[6] = hi_1_132;
                kbt2[13] = lo_1_132;
                float hi_1_133 = fmaxf(kbt2[8], kbt2[14]);
                float lo_1_133 = fminf(kbt2[8], kbt2[14]);
                kbt2[8] = hi_1_133;
                kbt2[14] = lo_1_133;
                float hi_1_134 = fmaxf(kbt2[10], kbt2[15]);
                float lo_1_134 = fminf(kbt2[10], kbt2[15]);
                kbt2[10] = hi_1_134;
                kbt2[15] = lo_1_134;
                float hi_1_135 = fmaxf(kbt2[11], kbt2[12]);
                float lo_1_135 = fminf(kbt2[11], kbt2[12]);
                kbt2[11] = hi_1_135;
                kbt2[12] = lo_1_135;
                float hi_1_136 = fmaxf(kbt2[0], kbt2[1]);
                float lo_1_136 = fminf(kbt2[0], kbt2[1]);
                kbt2[0] = hi_1_136;
                kbt2[1] = lo_1_136;
                float hi_1_137 = fmaxf(kbt2[2], kbt2[3]);
                float lo_1_137 = fminf(kbt2[2], kbt2[3]);
                kbt2[2] = hi_1_137;
                kbt2[3] = lo_1_137;
                float hi_1_138 = fmaxf(kbt2[4], kbt2[5]);
                float lo_1_138 = fminf(kbt2[4], kbt2[5]);
                kbt2[4] = hi_1_138;
                kbt2[5] = lo_1_138;
                float hi_1_139 = fmaxf(kbt2[6], kbt2[8]);
                float lo_1_139 = fminf(kbt2[6], kbt2[8]);
                kbt2[6] = hi_1_139;
                kbt2[8] = lo_1_139;
                float hi_1_140 = fmaxf(kbt2[7], kbt2[9]);
                float lo_1_140 = fminf(kbt2[7], kbt2[9]);
                kbt2[7] = hi_1_140;
                kbt2[9] = lo_1_140;
                float hi_1_141 = fmaxf(kbt2[10], kbt2[11]);
                float lo_1_141 = fminf(kbt2[10], kbt2[11]);
                kbt2[10] = hi_1_141;
                kbt2[11] = lo_1_141;
                float hi_1_142 = fmaxf(kbt2[12], kbt2[13]);
                float lo_1_142 = fminf(kbt2[12], kbt2[13]);
                kbt2[12] = hi_1_142;
                kbt2[13] = lo_1_142;
                float hi_1_143 = fmaxf(kbt2[14], kbt2[15]);
                float lo_1_143 = fminf(kbt2[14], kbt2[15]);
                kbt2[14] = hi_1_143;
                kbt2[15] = lo_1_143;
                float hi_1_144 = fmaxf(kbt2[0], kbt2[2]);
                float lo_1_144 = fminf(kbt2[0], kbt2[2]);
                kbt2[0] = hi_1_144;
                kbt2[2] = lo_1_144;
                float hi_1_145 = fmaxf(kbt2[1], kbt2[3]);
                float lo_1_145 = fminf(kbt2[1], kbt2[3]);
                kbt2[1] = hi_1_145;
                kbt2[3] = lo_1_145;
                float hi_1_146 = fmaxf(kbt2[4], kbt2[10]);
                float lo_1_146 = fminf(kbt2[4], kbt2[10]);
                kbt2[4] = hi_1_146;
                kbt2[10] = lo_1_146;
                float hi_1_147 = fmaxf(kbt2[5], kbt2[11]);
                float lo_1_147 = fminf(kbt2[5], kbt2[11]);
                kbt2[5] = hi_1_147;
                kbt2[11] = lo_1_147;
                float hi_1_148 = fmaxf(kbt2[6], kbt2[7]);
                float lo_1_148 = fminf(kbt2[6], kbt2[7]);
                kbt2[6] = hi_1_148;
                kbt2[7] = lo_1_148;
                float hi_1_149 = fmaxf(kbt2[8], kbt2[9]);
                float lo_1_149 = fminf(kbt2[8], kbt2[9]);
                kbt2[8] = hi_1_149;
                kbt2[9] = lo_1_149;
                float hi_1_150 = fmaxf(kbt2[12], kbt2[14]);
                float lo_1_150 = fminf(kbt2[12], kbt2[14]);
                kbt2[12] = hi_1_150;
                kbt2[14] = lo_1_150;
                float hi_1_151 = fmaxf(kbt2[13], kbt2[15]);
                float lo_1_151 = fminf(kbt2[13], kbt2[15]);
                kbt2[13] = hi_1_151;
                kbt2[15] = lo_1_151;
                float hi_1_152 = fmaxf(kbt2[1], kbt2[2]);
                float lo_1_152 = fminf(kbt2[1], kbt2[2]);
                kbt2[1] = hi_1_152;
                kbt2[2] = lo_1_152;
                float hi_1_153 = fmaxf(kbt2[3], kbt2[12]);
                float lo_1_153 = fminf(kbt2[3], kbt2[12]);
                kbt2[3] = hi_1_153;
                kbt2[12] = lo_1_153;
                float hi_1_154 = fmaxf(kbt2[4], kbt2[6]);
                float lo_1_154 = fminf(kbt2[4], kbt2[6]);
                kbt2[4] = hi_1_154;
                kbt2[6] = lo_1_154;
                float hi_1_155 = fmaxf(kbt2[5], kbt2[7]);
                float lo_1_155 = fminf(kbt2[5], kbt2[7]);
                kbt2[5] = hi_1_155;
                kbt2[7] = lo_1_155;
                float hi_1_156 = fmaxf(kbt2[8], kbt2[10]);
                float lo_1_156 = fminf(kbt2[8], kbt2[10]);
                kbt2[8] = hi_1_156;
                kbt2[10] = lo_1_156;
                float hi_1_157 = fmaxf(kbt2[9], kbt2[11]);
                float lo_1_157 = fminf(kbt2[9], kbt2[11]);
                kbt2[9] = hi_1_157;
                kbt2[11] = lo_1_157;
                float hi_1_158 = fmaxf(kbt2[13], kbt2[14]);
                float lo_1_158 = fminf(kbt2[13], kbt2[14]);
                kbt2[13] = hi_1_158;
                kbt2[14] = lo_1_158;
                float hi_1_159 = fmaxf(kbt2[1], kbt2[4]);
                float lo_1_159 = fminf(kbt2[1], kbt2[4]);
                kbt2[1] = hi_1_159;
                kbt2[4] = lo_1_159;
                float hi_1_160 = fmaxf(kbt2[2], kbt2[6]);
                float lo_1_160 = fminf(kbt2[2], kbt2[6]);
                kbt2[2] = hi_1_160;
                kbt2[6] = lo_1_160;
                float hi_1_161 = fmaxf(kbt2[5], kbt2[8]);
                float lo_1_161 = fminf(kbt2[5], kbt2[8]);
                kbt2[5] = hi_1_161;
                kbt2[8] = lo_1_161;
                float hi_1_162 = fmaxf(kbt2[7], kbt2[10]);
                float lo_1_162 = fminf(kbt2[7], kbt2[10]);
                kbt2[7] = hi_1_162;
                kbt2[10] = lo_1_162;
                float hi_1_163 = fmaxf(kbt2[9], kbt2[13]);
                float lo_1_163 = fminf(kbt2[9], kbt2[13]);
                kbt2[9] = hi_1_163;
                kbt2[13] = lo_1_163;
                float hi_1_164 = fmaxf(kbt2[11], kbt2[14]);
                float lo_1_164 = fminf(kbt2[11], kbt2[14]);
                kbt2[11] = hi_1_164;
                kbt2[14] = lo_1_164;
                float hi_1_165 = fmaxf(kbt2[2], kbt2[4]);
                float lo_1_165 = fminf(kbt2[2], kbt2[4]);
                kbt2[2] = hi_1_165;
                kbt2[4] = lo_1_165;
                float hi_1_166 = fmaxf(kbt2[3], kbt2[6]);
                float lo_1_166 = fminf(kbt2[3], kbt2[6]);
                kbt2[3] = hi_1_166;
                kbt2[6] = lo_1_166;
                float hi_1_167 = fmaxf(kbt2[9], kbt2[12]);
                float lo_1_167 = fminf(kbt2[9], kbt2[12]);
                kbt2[9] = hi_1_167;
                kbt2[12] = lo_1_167;
                float hi_1_168 = fmaxf(kbt2[11], kbt2[13]);
                float lo_1_168 = fminf(kbt2[11], kbt2[13]);
                kbt2[11] = hi_1_168;
                kbt2[13] = lo_1_168;
                float hi_1_169 = fmaxf(kbt2[3], kbt2[5]);
                float lo_1_169 = fminf(kbt2[3], kbt2[5]);
                kbt2[3] = hi_1_169;
                kbt2[5] = lo_1_169;
                float hi_1_170 = fmaxf(kbt2[6], kbt2[8]);
                float lo_1_170 = fminf(kbt2[6], kbt2[8]);
                kbt2[6] = hi_1_170;
                kbt2[8] = lo_1_170;
                float hi_1_171 = fmaxf(kbt2[7], kbt2[9]);
                float lo_1_171 = fminf(kbt2[7], kbt2[9]);
                kbt2[7] = hi_1_171;
                kbt2[9] = lo_1_171;
                float hi_1_172 = fmaxf(kbt2[10], kbt2[12]);
                float lo_1_172 = fminf(kbt2[10], kbt2[12]);
                kbt2[10] = hi_1_172;
                kbt2[12] = lo_1_172;
                float hi_1_173 = fmaxf(kbt2[3], kbt2[4]);
                float lo_1_173 = fminf(kbt2[3], kbt2[4]);
                kbt2[3] = hi_1_173;
                kbt2[4] = lo_1_173;
                float hi_1_174 = fmaxf(kbt2[5], kbt2[6]);
                float lo_1_174 = fminf(kbt2[5], kbt2[6]);
                kbt2[5] = hi_1_174;
                kbt2[6] = lo_1_174;
                float hi_1_175 = fmaxf(kbt2[7], kbt2[8]);
                float lo_1_175 = fminf(kbt2[7], kbt2[8]);
                kbt2[7] = hi_1_175;
                kbt2[8] = lo_1_175;
                float hi_1_176 = fmaxf(kbt2[9], kbt2[10]);
                float lo_1_176 = fminf(kbt2[9], kbt2[10]);
                kbt2[9] = hi_1_176;
                kbt2[10] = lo_1_176;
                float hi_1_177 = fmaxf(kbt2[11], kbt2[12]);
                float lo_1_177 = fminf(kbt2[11], kbt2[12]);
                kbt2[11] = hi_1_177;
                kbt2[12] = lo_1_177;
                float hi_1_178 = fmaxf(kbt2[6], kbt2[7]);
                float lo_1_178 = fminf(kbt2[6], kbt2[7]);
                kbt2[6] = hi_1_178;
                kbt2[7] = lo_1_178;
                float hi_1_179 = fmaxf(kbt2[8], kbt2[9]);
                float lo_1_179 = fminf(kbt2[8], kbt2[9]);
                kbt2[8] = hi_1_179;
                kbt2[9] = lo_1_179;
                float hi_1_180 = fmaxf(kbt2[16], kbt2[29]);
                float lo_1_180 = fminf(kbt2[16], kbt2[29]);
                kbt2[16] = hi_1_180;
                kbt2[29] = lo_1_180;
                float hi_1_181 = fmaxf(kbt2[17], kbt2[28]);
                float lo_1_181 = fminf(kbt2[17], kbt2[28]);
                kbt2[17] = hi_1_181;
                kbt2[28] = lo_1_181;
                float hi_1_182 = fmaxf(kbt2[18], kbt2[31]);
                float lo_1_182 = fminf(kbt2[18], kbt2[31]);
                kbt2[18] = hi_1_182;
                kbt2[31] = lo_1_182;
                float hi_1_183 = fmaxf(kbt2[19], kbt2[30]);
                float lo_1_183 = fminf(kbt2[19], kbt2[30]);
                kbt2[19] = hi_1_183;
                kbt2[30] = lo_1_183;
                float hi_1_184 = fmaxf(kbt2[20], kbt2[24]);
                float lo_1_184 = fminf(kbt2[20], kbt2[24]);
                kbt2[20] = hi_1_184;
                kbt2[24] = lo_1_184;
                float hi_1_185 = fmaxf(kbt2[21], kbt2[22]);
                float lo_1_185 = fminf(kbt2[21], kbt2[22]);
                kbt2[21] = hi_1_185;
                kbt2[22] = lo_1_185;
                float hi_1_186 = fmaxf(kbt2[23], kbt2[27]);
                float lo_1_186 = fminf(kbt2[23], kbt2[27]);
                kbt2[23] = hi_1_186;
                kbt2[27] = lo_1_186;
                float hi_1_187 = fmaxf(kbt2[25], kbt2[26]);
                float lo_1_187 = fminf(kbt2[25], kbt2[26]);
                kbt2[25] = hi_1_187;
                kbt2[26] = lo_1_187;
                float hi_1_188 = fmaxf(kbt2[16], kbt2[21]);
                float lo_1_188 = fminf(kbt2[16], kbt2[21]);
                kbt2[16] = hi_1_188;
                kbt2[21] = lo_1_188;
                float hi_1_189 = fmaxf(kbt2[17], kbt2[23]);
                float lo_1_189 = fminf(kbt2[17], kbt2[23]);
                kbt2[17] = hi_1_189;
                kbt2[23] = lo_1_189;
                float hi_1_190 = fmaxf(kbt2[18], kbt2[25]);
                float lo_1_190 = fminf(kbt2[18], kbt2[25]);
                kbt2[18] = hi_1_190;
                kbt2[25] = lo_1_190;
                float hi_1_191 = fmaxf(kbt2[19], kbt2[20]);
                float lo_1_191 = fminf(kbt2[19], kbt2[20]);
                kbt2[19] = hi_1_191;
                kbt2[20] = lo_1_191;
                float hi_1_192 = fmaxf(kbt2[22], kbt2[29]);
                float lo_1_192 = fminf(kbt2[22], kbt2[29]);
                kbt2[22] = hi_1_192;
                kbt2[29] = lo_1_192;
                float hi_1_193 = fmaxf(kbt2[24], kbt2[30]);
                float lo_1_193 = fminf(kbt2[24], kbt2[30]);
                kbt2[24] = hi_1_193;
                kbt2[30] = lo_1_193;
                float hi_1_194 = fmaxf(kbt2[26], kbt2[31]);
                float lo_1_194 = fminf(kbt2[26], kbt2[31]);
                kbt2[26] = hi_1_194;
                kbt2[31] = lo_1_194;
                float hi_1_195 = fmaxf(kbt2[27], kbt2[28]);
                float lo_1_195 = fminf(kbt2[27], kbt2[28]);
                kbt2[27] = hi_1_195;
                kbt2[28] = lo_1_195;
                float hi_1_196 = fmaxf(kbt2[16], kbt2[17]);
                float lo_1_196 = fminf(kbt2[16], kbt2[17]);
                kbt2[16] = hi_1_196;
                kbt2[17] = lo_1_196;
                float hi_1_197 = fmaxf(kbt2[18], kbt2[19]);
                float lo_1_197 = fminf(kbt2[18], kbt2[19]);
                kbt2[18] = hi_1_197;
                kbt2[19] = lo_1_197;
                float hi_1_198 = fmaxf(kbt2[20], kbt2[21]);
                float lo_1_198 = fminf(kbt2[20], kbt2[21]);
                kbt2[20] = hi_1_198;
                kbt2[21] = lo_1_198;
                float hi_1_199 = fmaxf(kbt2[22], kbt2[24]);
                float lo_1_199 = fminf(kbt2[22], kbt2[24]);
                kbt2[22] = hi_1_199;
                kbt2[24] = lo_1_199;
                float hi_1_200 = fmaxf(kbt2[23], kbt2[25]);
                float lo_1_200 = fminf(kbt2[23], kbt2[25]);
                kbt2[23] = hi_1_200;
                kbt2[25] = lo_1_200;
                float hi_1_201 = fmaxf(kbt2[26], kbt2[27]);
                float lo_1_201 = fminf(kbt2[26], kbt2[27]);
                kbt2[26] = hi_1_201;
                kbt2[27] = lo_1_201;
                float hi_1_202 = fmaxf(kbt2[28], kbt2[29]);
                float lo_1_202 = fminf(kbt2[28], kbt2[29]);
                kbt2[28] = hi_1_202;
                kbt2[29] = lo_1_202;
                float hi_1_203 = fmaxf(kbt2[30], kbt2[31]);
                float lo_1_203 = fminf(kbt2[30], kbt2[31]);
                kbt2[30] = hi_1_203;
                kbt2[31] = lo_1_203;
                float hi_1_204 = fmaxf(kbt2[16], kbt2[18]);
                float lo_1_204 = fminf(kbt2[16], kbt2[18]);
                kbt2[16] = hi_1_204;
                kbt2[18] = lo_1_204;
                float hi_1_205 = fmaxf(kbt2[17], kbt2[19]);
                float lo_1_205 = fminf(kbt2[17], kbt2[19]);
                kbt2[17] = hi_1_205;
                kbt2[19] = lo_1_205;
                float hi_1_206 = fmaxf(kbt2[20], kbt2[26]);
                float lo_1_206 = fminf(kbt2[20], kbt2[26]);
                kbt2[20] = hi_1_206;
                kbt2[26] = lo_1_206;
                float hi_1_207 = fmaxf(kbt2[21], kbt2[27]);
                float lo_1_207 = fminf(kbt2[21], kbt2[27]);
                kbt2[21] = hi_1_207;
                kbt2[27] = lo_1_207;
                float hi_1_208 = fmaxf(kbt2[22], kbt2[23]);
                float lo_1_208 = fminf(kbt2[22], kbt2[23]);
                kbt2[22] = hi_1_208;
                kbt2[23] = lo_1_208;
                float hi_1_209 = fmaxf(kbt2[24], kbt2[25]);
                float lo_1_209 = fminf(kbt2[24], kbt2[25]);
                kbt2[24] = hi_1_209;
                kbt2[25] = lo_1_209;
                float hi_1_210 = fmaxf(kbt2[28], kbt2[30]);
                float lo_1_210 = fminf(kbt2[28], kbt2[30]);
                kbt2[28] = hi_1_210;
                kbt2[30] = lo_1_210;
                float hi_1_211 = fmaxf(kbt2[29], kbt2[31]);
                float lo_1_211 = fminf(kbt2[29], kbt2[31]);
                kbt2[29] = hi_1_211;
                kbt2[31] = lo_1_211;
                float hi_1_212 = fmaxf(kbt2[17], kbt2[18]);
                float lo_1_212 = fminf(kbt2[17], kbt2[18]);
                kbt2[17] = hi_1_212;
                kbt2[18] = lo_1_212;
                float hi_1_213 = fmaxf(kbt2[19], kbt2[28]);
                float lo_1_213 = fminf(kbt2[19], kbt2[28]);
                kbt2[19] = hi_1_213;
                kbt2[28] = lo_1_213;
                float hi_1_214 = fmaxf(kbt2[20], kbt2[22]);
                float lo_1_214 = fminf(kbt2[20], kbt2[22]);
                kbt2[20] = hi_1_214;
                kbt2[22] = lo_1_214;
                float hi_1_215 = fmaxf(kbt2[21], kbt2[23]);
                float lo_1_215 = fminf(kbt2[21], kbt2[23]);
                kbt2[21] = hi_1_215;
                kbt2[23] = lo_1_215;
                float hi_1_216 = fmaxf(kbt2[24], kbt2[26]);
                float lo_1_216 = fminf(kbt2[24], kbt2[26]);
                kbt2[24] = hi_1_216;
                kbt2[26] = lo_1_216;
                float hi_1_217 = fmaxf(kbt2[25], kbt2[27]);
                float lo_1_217 = fminf(kbt2[25], kbt2[27]);
                kbt2[25] = hi_1_217;
                kbt2[27] = lo_1_217;
                float hi_1_218 = fmaxf(kbt2[29], kbt2[30]);
                float lo_1_218 = fminf(kbt2[29], kbt2[30]);
                kbt2[29] = hi_1_218;
                kbt2[30] = lo_1_218;
                float hi_1_219 = fmaxf(kbt2[17], kbt2[20]);
                float lo_1_219 = fminf(kbt2[17], kbt2[20]);
                kbt2[17] = hi_1_219;
                kbt2[20] = lo_1_219;
                float hi_1_220 = fmaxf(kbt2[18], kbt2[22]);
                float lo_1_220 = fminf(kbt2[18], kbt2[22]);
                kbt2[18] = hi_1_220;
                kbt2[22] = lo_1_220;
                float hi_1_221 = fmaxf(kbt2[21], kbt2[24]);
                float lo_1_221 = fminf(kbt2[21], kbt2[24]);
                kbt2[21] = hi_1_221;
                kbt2[24] = lo_1_221;
                float hi_1_222 = fmaxf(kbt2[23], kbt2[26]);
                float lo_1_222 = fminf(kbt2[23], kbt2[26]);
                kbt2[23] = hi_1_222;
                kbt2[26] = lo_1_222;
                float hi_1_223 = fmaxf(kbt2[25], kbt2[29]);
                float lo_1_223 = fminf(kbt2[25], kbt2[29]);
                kbt2[25] = hi_1_223;
                kbt2[29] = lo_1_223;
                float hi_1_224 = fmaxf(kbt2[27], kbt2[30]);
                float lo_1_224 = fminf(kbt2[27], kbt2[30]);
                kbt2[27] = hi_1_224;
                kbt2[30] = lo_1_224;
                float hi_1_225 = fmaxf(kbt2[18], kbt2[20]);
                float lo_1_225 = fminf(kbt2[18], kbt2[20]);
                kbt2[18] = hi_1_225;
                kbt2[20] = lo_1_225;
                float hi_1_226 = fmaxf(kbt2[19], kbt2[22]);
                float lo_1_226 = fminf(kbt2[19], kbt2[22]);
                kbt2[19] = hi_1_226;
                kbt2[22] = lo_1_226;
                float hi_1_227 = fmaxf(kbt2[25], kbt2[28]);
                float lo_1_227 = fminf(kbt2[25], kbt2[28]);
                kbt2[25] = hi_1_227;
                kbt2[28] = lo_1_227;
                float hi_1_228 = fmaxf(kbt2[27], kbt2[29]);
                float lo_1_228 = fminf(kbt2[27], kbt2[29]);
                kbt2[27] = hi_1_228;
                kbt2[29] = lo_1_228;
                float hi_1_229 = fmaxf(kbt2[19], kbt2[21]);
                float lo_1_229 = fminf(kbt2[19], kbt2[21]);
                kbt2[19] = hi_1_229;
                kbt2[21] = lo_1_229;
                float hi_1_230 = fmaxf(kbt2[22], kbt2[24]);
                float lo_1_230 = fminf(kbt2[22], kbt2[24]);
                kbt2[22] = hi_1_230;
                kbt2[24] = lo_1_230;
                float hi_1_231 = fmaxf(kbt2[23], kbt2[25]);
                float lo_1_231 = fminf(kbt2[23], kbt2[25]);
                kbt2[23] = hi_1_231;
                kbt2[25] = lo_1_231;
                float hi_1_232 = fmaxf(kbt2[26], kbt2[28]);
                float lo_1_232 = fminf(kbt2[26], kbt2[28]);
                kbt2[26] = hi_1_232;
                kbt2[28] = lo_1_232;
                float hi_1_233 = fmaxf(kbt2[19], kbt2[20]);
                float lo_1_233 = fminf(kbt2[19], kbt2[20]);
                kbt2[19] = hi_1_233;
                kbt2[20] = lo_1_233;
                float hi_1_234 = fmaxf(kbt2[21], kbt2[22]);
                float lo_1_234 = fminf(kbt2[21], kbt2[22]);
                kbt2[21] = hi_1_234;
                kbt2[22] = lo_1_234;
                float hi_1_235 = fmaxf(kbt2[23], kbt2[24]);
                float lo_1_235 = fminf(kbt2[23], kbt2[24]);
                kbt2[23] = hi_1_235;
                kbt2[24] = lo_1_235;
                float hi_1_236 = fmaxf(kbt2[25], kbt2[26]);
                float lo_1_236 = fminf(kbt2[25], kbt2[26]);
                kbt2[25] = hi_1_236;
                kbt2[26] = lo_1_236;
                float hi_1_237 = fmaxf(kbt2[27], kbt2[28]);
                float lo_1_237 = fminf(kbt2[27], kbt2[28]);
                kbt2[27] = hi_1_237;
                kbt2[28] = lo_1_237;
                float hi_1_238 = fmaxf(kbt2[22], kbt2[23]);
                float lo_1_238 = fminf(kbt2[22], kbt2[23]);
                kbt2[22] = hi_1_238;
                kbt2[23] = lo_1_238;
                float hi_1_239 = fmaxf(kbt2[24], kbt2[25]);
                float lo_1_239 = fminf(kbt2[24], kbt2[25]);
                kbt2[24] = hi_1_239;
                kbt2[25] = lo_1_239;
                float hi_104 = fmaxf(kbt2[0], kbt2[31]);
                float lo_104 = fminf(kbt2[0], kbt2[31]);
                kbt2[0] = hi_104;
                rr_0 = fmaxf(rr_0, lo_104);
                float hi_105 = fmaxf(kbt2[1], kbt2[30]);
                float lo_105 = fminf(kbt2[1], kbt2[30]);
                kbt2[1] = hi_105;
                rr_0 = fmaxf(rr_0, lo_105);
                float hi_106 = fmaxf(kbt2[2], kbt2[29]);
                float lo_106 = fminf(kbt2[2], kbt2[29]);
                kbt2[2] = hi_106;
                rr_0 = fmaxf(rr_0, lo_106);
                float hi_107 = fmaxf(kbt2[3], kbt2[28]);
                float lo_107 = fminf(kbt2[3], kbt2[28]);
                kbt2[3] = hi_107;
                rr_0 = fmaxf(rr_0, lo_107);
                float hi_108 = fmaxf(kbt2[4], kbt2[27]);
                float lo_108 = fminf(kbt2[4], kbt2[27]);
                kbt2[4] = hi_108;
                rr_0 = fmaxf(rr_0, lo_108);
                float hi_109 = fmaxf(kbt2[5], kbt2[26]);
                float lo_109 = fminf(kbt2[5], kbt2[26]);
                kbt2[5] = hi_109;
                rr_0 = fmaxf(rr_0, lo_109);
                float hi_110 = fmaxf(kbt2[6], kbt2[25]);
                float lo_110 = fminf(kbt2[6], kbt2[25]);
                kbt2[6] = hi_110;
                rr_0 = fmaxf(rr_0, lo_110);
                float hi_111 = fmaxf(kbt2[7], kbt2[24]);
                float lo_111 = fminf(kbt2[7], kbt2[24]);
                kbt2[7] = hi_111;
                rr_0 = fmaxf(rr_0, lo_111);
                float hi_112 = fmaxf(kbt2[8], kbt2[23]);
                float lo_112 = fminf(kbt2[8], kbt2[23]);
                kbt2[8] = hi_112;
                rr_0 = fmaxf(rr_0, lo_112);
                float hi_113 = fmaxf(kbt2[9], kbt2[22]);
                float lo_113 = fminf(kbt2[9], kbt2[22]);
                kbt2[9] = hi_113;
                rr_0 = fmaxf(rr_0, lo_113);
                float hi_114 = fmaxf(kbt2[10], kbt2[21]);
                float lo_114 = fminf(kbt2[10], kbt2[21]);
                kbt2[10] = hi_114;
                rr_0 = fmaxf(rr_0, lo_114);
                float hi_115 = fmaxf(kbt2[11], kbt2[20]);
                float lo_115 = fminf(kbt2[11], kbt2[20]);
                kbt2[11] = hi_115;
                rr_0 = fmaxf(rr_0, lo_115);
                float hi_116 = fmaxf(kbt2[12], kbt2[19]);
                float lo_116 = fminf(kbt2[12], kbt2[19]);
                kbt2[12] = hi_116;
                rr_0 = fmaxf(rr_0, lo_116);
                float hi_117 = fmaxf(kbt2[13], kbt2[18]);
                float lo_117 = fminf(kbt2[13], kbt2[18]);
                kbt2[13] = hi_117;
                rr_0 = fmaxf(rr_0, lo_117);
                float hi_118 = fmaxf(kbt2[14], kbt2[17]);
                float lo_118 = fminf(kbt2[14], kbt2[17]);
                kbt2[14] = hi_118;
                rr_0 = fmaxf(rr_0, lo_118);
                float hi_119 = fmaxf(kbt2[15], kbt2[16]);
                float lo_119 = fminf(kbt2[15], kbt2[16]);
                kbt2[15] = hi_119;
                rr_0 = fmaxf(rr_0, lo_119);
                float hi_120 = fmaxf(kbt2[0], kbt2[8]);
                float lo_120 = fminf(kbt2[0], kbt2[8]);
                kbt2[0] = hi_120;
                kbt2[8] = lo_120;
                float hi_121 = fmaxf(kbt2[1], kbt2[9]);
                float lo_121 = fminf(kbt2[1], kbt2[9]);
                kbt2[1] = hi_121;
                kbt2[9] = lo_121;
                float hi_122 = fmaxf(kbt2[2], kbt2[10]);
                float lo_122 = fminf(kbt2[2], kbt2[10]);
                kbt2[2] = hi_122;
                kbt2[10] = lo_122;
                float hi_123 = fmaxf(kbt2[3], kbt2[11]);
                float lo_123 = fminf(kbt2[3], kbt2[11]);
                kbt2[3] = hi_123;
                kbt2[11] = lo_123;
                float hi_124 = fmaxf(kbt2[4], kbt2[12]);
                float lo_124 = fminf(kbt2[4], kbt2[12]);
                kbt2[4] = hi_124;
                kbt2[12] = lo_124;
                float hi_125 = fmaxf(kbt2[5], kbt2[13]);
                float lo_125 = fminf(kbt2[5], kbt2[13]);
                kbt2[5] = hi_125;
                kbt2[13] = lo_125;
                float hi_126 = fmaxf(kbt2[6], kbt2[14]);
                float lo_126 = fminf(kbt2[6], kbt2[14]);
                kbt2[6] = hi_126;
                kbt2[14] = lo_126;
                float hi_127 = fmaxf(kbt2[7], kbt2[15]);
                float lo_127 = fminf(kbt2[7], kbt2[15]);
                kbt2[7] = hi_127;
                kbt2[15] = lo_127;
                float hi_128 = fmaxf(kbt2[0], kbt2[4]);
                float lo_128 = fminf(kbt2[0], kbt2[4]);
                kbt2[0] = hi_128;
                kbt2[4] = lo_128;
                float hi_129 = fmaxf(kbt2[1], kbt2[5]);
                float lo_129 = fminf(kbt2[1], kbt2[5]);
                kbt2[1] = hi_129;
                kbt2[5] = lo_129;
                float hi_130 = fmaxf(kbt2[2], kbt2[6]);
                float lo_130 = fminf(kbt2[2], kbt2[6]);
                kbt2[2] = hi_130;
                kbt2[6] = lo_130;
                float hi_131 = fmaxf(kbt2[3], kbt2[7]);
                float lo_131 = fminf(kbt2[3], kbt2[7]);
                kbt2[3] = hi_131;
                kbt2[7] = lo_131;
                float hi_132 = fmaxf(kbt2[8], kbt2[12]);
                float lo_132 = fminf(kbt2[8], kbt2[12]);
                kbt2[8] = hi_132;
                kbt2[12] = lo_132;
                float hi_133 = fmaxf(kbt2[9], kbt2[13]);
                float lo_133 = fminf(kbt2[9], kbt2[13]);
                kbt2[9] = hi_133;
                kbt2[13] = lo_133;
                float hi_134 = fmaxf(kbt2[10], kbt2[14]);
                float lo_134 = fminf(kbt2[10], kbt2[14]);
                kbt2[10] = hi_134;
                kbt2[14] = lo_134;
                float hi_135 = fmaxf(kbt2[11], kbt2[15]);
                float lo_135 = fminf(kbt2[11], kbt2[15]);
                kbt2[11] = hi_135;
                kbt2[15] = lo_135;
                float hi_136 = fmaxf(kbt2[0], kbt2[2]);
                float lo_136 = fminf(kbt2[0], kbt2[2]);
                kbt2[0] = hi_136;
                kbt2[2] = lo_136;
                float hi_137 = fmaxf(kbt2[1], kbt2[3]);
                float lo_137 = fminf(kbt2[1], kbt2[3]);
                kbt2[1] = hi_137;
                kbt2[3] = lo_137;
                float hi_138 = fmaxf(kbt2[4], kbt2[6]);
                float lo_138 = fminf(kbt2[4], kbt2[6]);
                kbt2[4] = hi_138;
                kbt2[6] = lo_138;
                float hi_139 = fmaxf(kbt2[5], kbt2[7]);
                float lo_139 = fminf(kbt2[5], kbt2[7]);
                kbt2[5] = hi_139;
                kbt2[7] = lo_139;
                float hi_140 = fmaxf(kbt2[8], kbt2[10]);
                float lo_140 = fminf(kbt2[8], kbt2[10]);
                kbt2[8] = hi_140;
                kbt2[10] = lo_140;
                float hi_141 = fmaxf(kbt2[9], kbt2[11]);
                float lo_141 = fminf(kbt2[9], kbt2[11]);
                kbt2[9] = hi_141;
                kbt2[11] = lo_141;
                float hi_142 = fmaxf(kbt2[12], kbt2[14]);
                float lo_142 = fminf(kbt2[12], kbt2[14]);
                kbt2[12] = hi_142;
                kbt2[14] = lo_142;
                float hi_143 = fmaxf(kbt2[13], kbt2[15]);
                float lo_143 = fminf(kbt2[13], kbt2[15]);
                kbt2[13] = hi_143;
                kbt2[15] = lo_143;
                float hi_144 = fmaxf(kbt2[0], kbt2[1]);
                float lo_144 = fminf(kbt2[0], kbt2[1]);
                kbt2[0] = hi_144;
                kbt2[1] = lo_144;
                float hi_145 = fmaxf(kbt2[2], kbt2[3]);
                float lo_145 = fminf(kbt2[2], kbt2[3]);
                kbt2[2] = hi_145;
                kbt2[3] = lo_145;
                float hi_146 = fmaxf(kbt2[4], kbt2[5]);
                float lo_146 = fminf(kbt2[4], kbt2[5]);
                kbt2[4] = hi_146;
                kbt2[5] = lo_146;
                float hi_147 = fmaxf(kbt2[6], kbt2[7]);
                float lo_147 = fminf(kbt2[6], kbt2[7]);
                kbt2[6] = hi_147;
                kbt2[7] = lo_147;
                float hi_148 = fmaxf(kbt2[8], kbt2[9]);
                float lo_148 = fminf(kbt2[8], kbt2[9]);
                kbt2[8] = hi_148;
                kbt2[9] = lo_148;
                float hi_149 = fmaxf(kbt2[10], kbt2[11]);
                float lo_149 = fminf(kbt2[10], kbt2[11]);
                kbt2[10] = hi_149;
                kbt2[11] = lo_149;
                float hi_150 = fmaxf(kbt2[12], kbt2[13]);
                float lo_150 = fminf(kbt2[12], kbt2[13]);
                kbt2[12] = hi_150;
                kbt2[13] = lo_150;
                float hi_151 = fmaxf(kbt2[14], kbt2[15]);
                float lo_151 = fminf(kbt2[14], kbt2[15]);
                kbt2[14] = hi_151;
                kbt2[15] = lo_151;
                r2o = rr_0;
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
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            float r_0 = neg_inf;
            float V_0[8];
            int s0_8 = (sg * 16 * 16 + cg) * 17;
            int s1_8 = ((sg * 16 + 8) * 16 + cg) * 17;
            float x0_8 = pub[s0_8 + ln];
            float y0_8 = pub[s1_8 + lnr];
            float lo0_8 = fminf(x0_8, y0_8);
            r_0 = fmaxf(r_0, lo0_8);
            float hi0_8 = fmaxf(x0_8, y0_8);
            float cur_13 = hi0_8;
            float pv_56 = __shfl_xor_sync(0xFFFFFFFF, cur_13, 8);
            float hi_152 = fmaxf(cur_13, pv_56);
            float lo_152 = fminf(cur_13, pv_56);
            cur_13 = ((up[0] != 0) ? hi_152 : lo_152);
            float pv_57 = __shfl_xor_sync(0xFFFFFFFF, cur_13, 4);
            float hi_153 = fmaxf(cur_13, pv_57);
            float lo_153 = fminf(cur_13, pv_57);
            cur_13 = ((up[1] != 0) ? hi_153 : lo_153);
            float pv_58 = __shfl_xor_sync(0xFFFFFFFF, cur_13, 2);
            float hi_154 = fmaxf(cur_13, pv_58);
            float lo_154 = fminf(cur_13, pv_58);
            cur_13 = ((up[2] != 0) ? hi_154 : lo_154);
            float pv_59 = __shfl_xor_sync(0xFFFFFFFF, cur_13, 1);
            float hi_155 = fmaxf(cur_13, pv_59);
            float lo_155 = fminf(cur_13, pv_59);
            cur_13 = ((up[3] != 0) ? hi_155 : lo_155);
            V_0[0] = cur_13;
            int s0_9 = ((sg * 16 + 1) * 16 + cg) * 17;
            int s1_9 = ((sg * 16 + 1 + 8) * 16 + cg) * 17;
            float x0_9 = pub[s0_9 + ln];
            float y0_9 = pub[s1_9 + lnr];
            float lo0_9 = fminf(x0_9, y0_9);
            r_0 = fmaxf(r_0, lo0_9);
            float hi0_9 = fmaxf(x0_9, y0_9);
            float cur_14 = hi0_9;
            float pv_60 = __shfl_xor_sync(0xFFFFFFFF, cur_14, 8);
            float hi_156 = fmaxf(cur_14, pv_60);
            float lo_156 = fminf(cur_14, pv_60);
            cur_14 = ((up[0] != 0) ? hi_156 : lo_156);
            float pv_61 = __shfl_xor_sync(0xFFFFFFFF, cur_14, 4);
            float hi_157 = fmaxf(cur_14, pv_61);
            float lo_157 = fminf(cur_14, pv_61);
            cur_14 = ((up[1] != 0) ? hi_157 : lo_157);
            float pv_62 = __shfl_xor_sync(0xFFFFFFFF, cur_14, 2);
            float hi_158 = fmaxf(cur_14, pv_62);
            float lo_158 = fminf(cur_14, pv_62);
            cur_14 = ((up[2] != 0) ? hi_158 : lo_158);
            float pv_63 = __shfl_xor_sync(0xFFFFFFFF, cur_14, 1);
            float hi_159 = fmaxf(cur_14, pv_63);
            float lo_159 = fminf(cur_14, pv_63);
            cur_14 = ((up[3] != 0) ? hi_159 : lo_159);
            V_0[1] = cur_14;
            int s0_10 = ((sg * 16 + 2) * 16 + cg) * 17;
            int s1_10 = ((sg * 16 + 2 + 8) * 16 + cg) * 17;
            float x0_10 = pub[s0_10 + ln];
            float y0_10 = pub[s1_10 + lnr];
            float lo0_10 = fminf(x0_10, y0_10);
            r_0 = fmaxf(r_0, lo0_10);
            float hi0_10 = fmaxf(x0_10, y0_10);
            float cur_15 = hi0_10;
            float pv_64 = __shfl_xor_sync(0xFFFFFFFF, cur_15, 8);
            float hi_160 = fmaxf(cur_15, pv_64);
            float lo_160 = fminf(cur_15, pv_64);
            cur_15 = ((up[0] != 0) ? hi_160 : lo_160);
            float pv_65 = __shfl_xor_sync(0xFFFFFFFF, cur_15, 4);
            float hi_161 = fmaxf(cur_15, pv_65);
            float lo_161 = fminf(cur_15, pv_65);
            cur_15 = ((up[1] != 0) ? hi_161 : lo_161);
            float pv_66 = __shfl_xor_sync(0xFFFFFFFF, cur_15, 2);
            float hi_162 = fmaxf(cur_15, pv_66);
            float lo_162 = fminf(cur_15, pv_66);
            cur_15 = ((up[2] != 0) ? hi_162 : lo_162);
            float pv_67 = __shfl_xor_sync(0xFFFFFFFF, cur_15, 1);
            float hi_163 = fmaxf(cur_15, pv_67);
            float lo_163 = fminf(cur_15, pv_67);
            cur_15 = ((up[3] != 0) ? hi_163 : lo_163);
            V_0[2] = cur_15;
            int s0_11 = ((sg * 16 + 3) * 16 + cg) * 17;
            int s1_11 = ((sg * 16 + 3 + 8) * 16 + cg) * 17;
            float x0_11 = pub[s0_11 + ln];
            float y0_11 = pub[s1_11 + lnr];
            float lo0_11 = fminf(x0_11, y0_11);
            r_0 = fmaxf(r_0, lo0_11);
            float hi0_11 = fmaxf(x0_11, y0_11);
            float cur_16 = hi0_11;
            float pv_68 = __shfl_xor_sync(0xFFFFFFFF, cur_16, 8);
            float hi_164 = fmaxf(cur_16, pv_68);
            float lo_164 = fminf(cur_16, pv_68);
            cur_16 = ((up[0] != 0) ? hi_164 : lo_164);
            float pv_69 = __shfl_xor_sync(0xFFFFFFFF, cur_16, 4);
            float hi_165 = fmaxf(cur_16, pv_69);
            float lo_165 = fminf(cur_16, pv_69);
            cur_16 = ((up[1] != 0) ? hi_165 : lo_165);
            float pv_70 = __shfl_xor_sync(0xFFFFFFFF, cur_16, 2);
            float hi_166 = fmaxf(cur_16, pv_70);
            float lo_166 = fminf(cur_16, pv_70);
            cur_16 = ((up[2] != 0) ? hi_166 : lo_166);
            float pv_71 = __shfl_xor_sync(0xFFFFFFFF, cur_16, 1);
            float hi_167 = fmaxf(cur_16, pv_71);
            float lo_167 = fminf(cur_16, pv_71);
            cur_16 = ((up[3] != 0) ? hi_167 : lo_167);
            V_0[3] = cur_16;
            int s0_12 = ((sg * 16 + 4) * 16 + cg) * 17;
            int s1_12 = ((sg * 16 + 4 + 8) * 16 + cg) * 17;
            float x0_12 = pub[s0_12 + ln];
            float y0_12 = pub[s1_12 + lnr];
            float lo0_12 = fminf(x0_12, y0_12);
            r_0 = fmaxf(r_0, lo0_12);
            float hi0_12 = fmaxf(x0_12, y0_12);
            float cur_17 = hi0_12;
            float pv_72 = __shfl_xor_sync(0xFFFFFFFF, cur_17, 8);
            float hi_168 = fmaxf(cur_17, pv_72);
            float lo_168 = fminf(cur_17, pv_72);
            cur_17 = ((up[0] != 0) ? hi_168 : lo_168);
            float pv_73 = __shfl_xor_sync(0xFFFFFFFF, cur_17, 4);
            float hi_169 = fmaxf(cur_17, pv_73);
            float lo_169 = fminf(cur_17, pv_73);
            cur_17 = ((up[1] != 0) ? hi_169 : lo_169);
            float pv_74 = __shfl_xor_sync(0xFFFFFFFF, cur_17, 2);
            float hi_170 = fmaxf(cur_17, pv_74);
            float lo_170 = fminf(cur_17, pv_74);
            cur_17 = ((up[2] != 0) ? hi_170 : lo_170);
            float pv_75 = __shfl_xor_sync(0xFFFFFFFF, cur_17, 1);
            float hi_171 = fmaxf(cur_17, pv_75);
            float lo_171 = fminf(cur_17, pv_75);
            cur_17 = ((up[3] != 0) ? hi_171 : lo_171);
            V_0[4] = cur_17;
            int s0_13 = ((sg * 16 + 5) * 16 + cg) * 17;
            int s1_13 = ((sg * 16 + 5 + 8) * 16 + cg) * 17;
            float x0_13 = pub[s0_13 + ln];
            float y0_13 = pub[s1_13 + lnr];
            float lo0_13 = fminf(x0_13, y0_13);
            r_0 = fmaxf(r_0, lo0_13);
            float hi0_13 = fmaxf(x0_13, y0_13);
            float cur_18 = hi0_13;
            float pv_76 = __shfl_xor_sync(0xFFFFFFFF, cur_18, 8);
            float hi_172 = fmaxf(cur_18, pv_76);
            float lo_172 = fminf(cur_18, pv_76);
            cur_18 = ((up[0] != 0) ? hi_172 : lo_172);
            float pv_77 = __shfl_xor_sync(0xFFFFFFFF, cur_18, 4);
            float hi_173 = fmaxf(cur_18, pv_77);
            float lo_173 = fminf(cur_18, pv_77);
            cur_18 = ((up[1] != 0) ? hi_173 : lo_173);
            float pv_78 = __shfl_xor_sync(0xFFFFFFFF, cur_18, 2);
            float hi_174 = fmaxf(cur_18, pv_78);
            float lo_174 = fminf(cur_18, pv_78);
            cur_18 = ((up[2] != 0) ? hi_174 : lo_174);
            float pv_79 = __shfl_xor_sync(0xFFFFFFFF, cur_18, 1);
            float hi_175 = fmaxf(cur_18, pv_79);
            float lo_175 = fminf(cur_18, pv_79);
            cur_18 = ((up[3] != 0) ? hi_175 : lo_175);
            V_0[5] = cur_18;
            int s0_14 = ((sg * 16 + 6) * 16 + cg) * 17;
            int s1_14 = ((sg * 16 + 6 + 8) * 16 + cg) * 17;
            float x0_14 = pub[s0_14 + ln];
            float y0_14 = pub[s1_14 + lnr];
            float lo0_14 = fminf(x0_14, y0_14);
            r_0 = fmaxf(r_0, lo0_14);
            float hi0_14 = fmaxf(x0_14, y0_14);
            float cur_19 = hi0_14;
            float pv_80 = __shfl_xor_sync(0xFFFFFFFF, cur_19, 8);
            float hi_176 = fmaxf(cur_19, pv_80);
            float lo_176 = fminf(cur_19, pv_80);
            cur_19 = ((up[0] != 0) ? hi_176 : lo_176);
            float pv_81 = __shfl_xor_sync(0xFFFFFFFF, cur_19, 4);
            float hi_177 = fmaxf(cur_19, pv_81);
            float lo_177 = fminf(cur_19, pv_81);
            cur_19 = ((up[1] != 0) ? hi_177 : lo_177);
            float pv_82 = __shfl_xor_sync(0xFFFFFFFF, cur_19, 2);
            float hi_178 = fmaxf(cur_19, pv_82);
            float lo_178 = fminf(cur_19, pv_82);
            cur_19 = ((up[2] != 0) ? hi_178 : lo_178);
            float pv_83 = __shfl_xor_sync(0xFFFFFFFF, cur_19, 1);
            float hi_179 = fmaxf(cur_19, pv_83);
            float lo_179 = fminf(cur_19, pv_83);
            cur_19 = ((up[3] != 0) ? hi_179 : lo_179);
            V_0[6] = cur_19;
            int s0_15 = ((sg * 16 + 7) * 16 + cg) * 17;
            int s1_15 = ((sg * 16 + 7 + 8) * 16 + cg) * 17;
            float x0_15 = pub[s0_15 + ln];
            float y0_15 = pub[s1_15 + lnr];
            float lo0_15 = fminf(x0_15, y0_15);
            r_0 = fmaxf(r_0, lo0_15);
            float hi0_15 = fmaxf(x0_15, y0_15);
            float cur_20 = hi0_15;
            float pv_84 = __shfl_xor_sync(0xFFFFFFFF, cur_20, 8);
            float hi_180 = fmaxf(cur_20, pv_84);
            float lo_180 = fminf(cur_20, pv_84);
            cur_20 = ((up[0] != 0) ? hi_180 : lo_180);
            float pv_85 = __shfl_xor_sync(0xFFFFFFFF, cur_20, 4);
            float hi_181 = fmaxf(cur_20, pv_85);
            float lo_181 = fminf(cur_20, pv_85);
            cur_20 = ((up[1] != 0) ? hi_181 : lo_181);
            float pv_86 = __shfl_xor_sync(0xFFFFFFFF, cur_20, 2);
            float hi_182 = fmaxf(cur_20, pv_86);
            float lo_182 = fminf(cur_20, pv_86);
            cur_20 = ((up[2] != 0) ? hi_182 : lo_182);
            float pv_87 = __shfl_xor_sync(0xFFFFFFFF, cur_20, 1);
            float hi_183 = fmaxf(cur_20, pv_87);
            float lo_183 = fminf(cur_20, pv_87);
            cur_20 = ((up[3] != 0) ? hi_183 : lo_183);
            V_0[7] = cur_20;
            float rs_0 = pub[((sg * 16 + ln) * 16 + cg) * 17 + 16];
            r_0 = fmaxf(r_0, rs_0);
            float y1_7 = __shfl_xor_sync(0xFFFFFFFF, V_0[4], 15);
            float lo1_7 = fminf(V_0[0], y1_7);
            r_0 = fmaxf(r_0, lo1_7);
            float hi1_7 = fmaxf(V_0[0], y1_7);
            float cur_21 = hi1_7;
            float pv_88 = __shfl_xor_sync(0xFFFFFFFF, cur_21, 8);
            float hi_184 = fmaxf(cur_21, pv_88);
            float lo_184 = fminf(cur_21, pv_88);
            cur_21 = ((up[0] != 0) ? hi_184 : lo_184);
            float pv_89 = __shfl_xor_sync(0xFFFFFFFF, cur_21, 4);
            float hi_185 = fmaxf(cur_21, pv_89);
            float lo_185 = fminf(cur_21, pv_89);
            cur_21 = ((up[1] != 0) ? hi_185 : lo_185);
            float pv_90 = __shfl_xor_sync(0xFFFFFFFF, cur_21, 2);
            float hi_186 = fmaxf(cur_21, pv_90);
            float lo_186 = fminf(cur_21, pv_90);
            cur_21 = ((up[2] != 0) ? hi_186 : lo_186);
            float pv_91 = __shfl_xor_sync(0xFFFFFFFF, cur_21, 1);
            float hi_187 = fmaxf(cur_21, pv_91);
            float lo_187 = fminf(cur_21, pv_91);
            cur_21 = ((up[3] != 0) ? hi_187 : lo_187);
            V_0[0] = cur_21;
            float y1_8 = __shfl_xor_sync(0xFFFFFFFF, V_0[5], 15);
            float lo1_8 = fminf(V_0[1], y1_8);
            r_0 = fmaxf(r_0, lo1_8);
            float hi1_8 = fmaxf(V_0[1], y1_8);
            float cur_22 = hi1_8;
            float pv_92 = __shfl_xor_sync(0xFFFFFFFF, cur_22, 8);
            float hi_188 = fmaxf(cur_22, pv_92);
            float lo_188 = fminf(cur_22, pv_92);
            cur_22 = ((up[0] != 0) ? hi_188 : lo_188);
            float pv_93 = __shfl_xor_sync(0xFFFFFFFF, cur_22, 4);
            float hi_189 = fmaxf(cur_22, pv_93);
            float lo_189 = fminf(cur_22, pv_93);
            cur_22 = ((up[1] != 0) ? hi_189 : lo_189);
            float pv_94 = __shfl_xor_sync(0xFFFFFFFF, cur_22, 2);
            float hi_190 = fmaxf(cur_22, pv_94);
            float lo_190 = fminf(cur_22, pv_94);
            cur_22 = ((up[2] != 0) ? hi_190 : lo_190);
            float pv_95 = __shfl_xor_sync(0xFFFFFFFF, cur_22, 1);
            float hi_191 = fmaxf(cur_22, pv_95);
            float lo_191 = fminf(cur_22, pv_95);
            cur_22 = ((up[3] != 0) ? hi_191 : lo_191);
            V_0[1] = cur_22;
            float y1_9 = __shfl_xor_sync(0xFFFFFFFF, V_0[6], 15);
            float lo1_9 = fminf(V_0[2], y1_9);
            r_0 = fmaxf(r_0, lo1_9);
            float hi1_9 = fmaxf(V_0[2], y1_9);
            float cur_23 = hi1_9;
            float pv_96 = __shfl_xor_sync(0xFFFFFFFF, cur_23, 8);
            float hi_192 = fmaxf(cur_23, pv_96);
            float lo_192 = fminf(cur_23, pv_96);
            cur_23 = ((up[0] != 0) ? hi_192 : lo_192);
            float pv_97 = __shfl_xor_sync(0xFFFFFFFF, cur_23, 4);
            float hi_193 = fmaxf(cur_23, pv_97);
            float lo_193 = fminf(cur_23, pv_97);
            cur_23 = ((up[1] != 0) ? hi_193 : lo_193);
            float pv_98 = __shfl_xor_sync(0xFFFFFFFF, cur_23, 2);
            float hi_194 = fmaxf(cur_23, pv_98);
            float lo_194 = fminf(cur_23, pv_98);
            cur_23 = ((up[2] != 0) ? hi_194 : lo_194);
            float pv_99 = __shfl_xor_sync(0xFFFFFFFF, cur_23, 1);
            float hi_195 = fmaxf(cur_23, pv_99);
            float lo_195 = fminf(cur_23, pv_99);
            cur_23 = ((up[3] != 0) ? hi_195 : lo_195);
            V_0[2] = cur_23;
            float y1_10 = __shfl_xor_sync(0xFFFFFFFF, V_0[7], 15);
            float lo1_10 = fminf(V_0[3], y1_10);
            r_0 = fmaxf(r_0, lo1_10);
            float hi1_10 = fmaxf(V_0[3], y1_10);
            float cur_24 = hi1_10;
            float pv_100 = __shfl_xor_sync(0xFFFFFFFF, cur_24, 8);
            float hi_196 = fmaxf(cur_24, pv_100);
            float lo_196 = fminf(cur_24, pv_100);
            cur_24 = ((up[0] != 0) ? hi_196 : lo_196);
            float pv_101 = __shfl_xor_sync(0xFFFFFFFF, cur_24, 4);
            float hi_197 = fmaxf(cur_24, pv_101);
            float lo_197 = fminf(cur_24, pv_101);
            cur_24 = ((up[1] != 0) ? hi_197 : lo_197);
            float pv_102 = __shfl_xor_sync(0xFFFFFFFF, cur_24, 2);
            float hi_198 = fmaxf(cur_24, pv_102);
            float lo_198 = fminf(cur_24, pv_102);
            cur_24 = ((up[2] != 0) ? hi_198 : lo_198);
            float pv_103 = __shfl_xor_sync(0xFFFFFFFF, cur_24, 1);
            float hi_199 = fmaxf(cur_24, pv_103);
            float lo_199 = fminf(cur_24, pv_103);
            cur_24 = ((up[3] != 0) ? hi_199 : lo_199);
            V_0[3] = cur_24;
            float y1_11 = __shfl_xor_sync(0xFFFFFFFF, V_0[2], 15);
            float lo1_11 = fminf(V_0[0], y1_11);
            r_0 = fmaxf(r_0, lo1_11);
            float hi1_11 = fmaxf(V_0[0], y1_11);
            float cur_25 = hi1_11;
            float pv_104 = __shfl_xor_sync(0xFFFFFFFF, cur_25, 8);
            float hi_200 = fmaxf(cur_25, pv_104);
            float lo_200 = fminf(cur_25, pv_104);
            cur_25 = ((up[0] != 0) ? hi_200 : lo_200);
            float pv_105 = __shfl_xor_sync(0xFFFFFFFF, cur_25, 4);
            float hi_201 = fmaxf(cur_25, pv_105);
            float lo_201 = fminf(cur_25, pv_105);
            cur_25 = ((up[1] != 0) ? hi_201 : lo_201);
            float pv_106 = __shfl_xor_sync(0xFFFFFFFF, cur_25, 2);
            float hi_202 = fmaxf(cur_25, pv_106);
            float lo_202 = fminf(cur_25, pv_106);
            cur_25 = ((up[2] != 0) ? hi_202 : lo_202);
            float pv_107 = __shfl_xor_sync(0xFFFFFFFF, cur_25, 1);
            float hi_203 = fmaxf(cur_25, pv_107);
            float lo_203 = fminf(cur_25, pv_107);
            cur_25 = ((up[3] != 0) ? hi_203 : lo_203);
            V_0[0] = cur_25;
            float y1_12 = __shfl_xor_sync(0xFFFFFFFF, V_0[3], 15);
            float lo1_12 = fminf(V_0[1], y1_12);
            r_0 = fmaxf(r_0, lo1_12);
            float hi1_12 = fmaxf(V_0[1], y1_12);
            float cur_26 = hi1_12;
            float pv_108 = __shfl_xor_sync(0xFFFFFFFF, cur_26, 8);
            float hi_204 = fmaxf(cur_26, pv_108);
            float lo_204 = fminf(cur_26, pv_108);
            cur_26 = ((up[0] != 0) ? hi_204 : lo_204);
            float pv_109 = __shfl_xor_sync(0xFFFFFFFF, cur_26, 4);
            float hi_205 = fmaxf(cur_26, pv_109);
            float lo_205 = fminf(cur_26, pv_109);
            cur_26 = ((up[1] != 0) ? hi_205 : lo_205);
            float pv_110 = __shfl_xor_sync(0xFFFFFFFF, cur_26, 2);
            float hi_206 = fmaxf(cur_26, pv_110);
            float lo_206 = fminf(cur_26, pv_110);
            cur_26 = ((up[2] != 0) ? hi_206 : lo_206);
            float pv_111 = __shfl_xor_sync(0xFFFFFFFF, cur_26, 1);
            float hi_207 = fmaxf(cur_26, pv_111);
            float lo_207 = fminf(cur_26, pv_111);
            cur_26 = ((up[3] != 0) ? hi_207 : lo_207);
            V_0[1] = cur_26;
            float y1_13 = __shfl_xor_sync(0xFFFFFFFF, V_0[1], 15);
            float lo1_13 = fminf(V_0[0], y1_13);
            r_0 = fmaxf(r_0, lo1_13);
            float hi1_13 = fmaxf(V_0[0], y1_13);
            V_0[0] = hi1_13;
            float K_0 = V_0[0];
            rr2x[0] = r_0;
            K2 = K_0;
        }
        if (tid_1 < 256) {
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
                        int rk4_3_0 = 0;
                        if (x4[1] > x4[0]) {
                            rk4_3_0 = rk4_3_0 + 1;
                        }
                        if (x4[2] > x4[0]) {
                            rk4_3_0 = rk4_3_0 + 1;
                        }
                        if (x4[3] > x4[0]) {
                            rk4_3_0 = rk4_3_0 + 1;
                        }
                        if (rk4_3_0 == rx) {
                            K2 = x4[0];
                        }
                        int rk4_3_1 = 0;
                        if (x4[0] > x4[1]) {
                            rk4_3_1 = rk4_3_1 + 1;
                        }
                        if (x4[2] > x4[1]) {
                            rk4_3_1 = rk4_3_1 + 1;
                        }
                        if (x4[3] > x4[1]) {
                            rk4_3_1 = rk4_3_1 + 1;
                        }
                        if (rk4_3_1 == rx) {
                            K2 = x4[1];
                        }
                        int rk4_3_2 = 0;
                        if (x4[0] > x4[2]) {
                            rk4_3_2 = rk4_3_2 + 1;
                        }
                        if (x4[1] > x4[2]) {
                            rk4_3_2 = rk4_3_2 + 1;
                        }
                        if (x4[3] > x4[2]) {
                            rk4_3_2 = rk4_3_2 + 1;
                        }
                        if (rk4_3_2 == rx) {
                            K2 = x4[2];
                        }
                        int rk4_3_3 = 0;
                        if (x4[0] > x4[3]) {
                            rk4_3_3 = rk4_3_3 + 1;
                        }
                        if (x4[1] > x4[3]) {
                            rk4_3_3 = rk4_3_3 + 1;
                        }
                        if (x4[2] > x4[3]) {
                            rk4_3_3 = rk4_3_3 + 1;
                        }
                        if (rk4_3_3 == rx) {
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
                        float hi_1_240 = fmaxf(xsl[0], xsl[13]);
                        float lo_1_240 = fminf(xsl[0], xsl[13]);
                        xsl[0] = hi_1_240;
                        xsl[13] = lo_1_240;
                        float hi_1_241 = fmaxf(xsl[1], xsl[12]);
                        float lo_1_241 = fminf(xsl[1], xsl[12]);
                        xsl[1] = hi_1_241;
                        xsl[12] = lo_1_241;
                        float hi_1_242 = fmaxf(xsl[2], xsl[15]);
                        float lo_1_242 = fminf(xsl[2], xsl[15]);
                        xsl[2] = hi_1_242;
                        xsl[15] = lo_1_242;
                        float hi_1_243 = fmaxf(xsl[3], xsl[14]);
                        float lo_1_243 = fminf(xsl[3], xsl[14]);
                        xsl[3] = hi_1_243;
                        xsl[14] = lo_1_243;
                        float hi_1_244 = fmaxf(xsl[4], xsl[8]);
                        float lo_1_244 = fminf(xsl[4], xsl[8]);
                        xsl[4] = hi_1_244;
                        xsl[8] = lo_1_244;
                        float hi_1_245 = fmaxf(xsl[5], xsl[6]);
                        float lo_1_245 = fminf(xsl[5], xsl[6]);
                        xsl[5] = hi_1_245;
                        xsl[6] = lo_1_245;
                        float hi_1_246 = fmaxf(xsl[7], xsl[11]);
                        float lo_1_246 = fminf(xsl[7], xsl[11]);
                        xsl[7] = hi_1_246;
                        xsl[11] = lo_1_246;
                        float hi_1_247 = fmaxf(xsl[9], xsl[10]);
                        float lo_1_247 = fminf(xsl[9], xsl[10]);
                        xsl[9] = hi_1_247;
                        xsl[10] = lo_1_247;
                        float hi_1_248 = fmaxf(xsl[0], xsl[5]);
                        float lo_1_248 = fminf(xsl[0], xsl[5]);
                        xsl[0] = hi_1_248;
                        xsl[5] = lo_1_248;
                        float hi_1_249 = fmaxf(xsl[1], xsl[7]);
                        float lo_1_249 = fminf(xsl[1], xsl[7]);
                        xsl[1] = hi_1_249;
                        xsl[7] = lo_1_249;
                        float hi_1_250 = fmaxf(xsl[2], xsl[9]);
                        float lo_1_250 = fminf(xsl[2], xsl[9]);
                        xsl[2] = hi_1_250;
                        xsl[9] = lo_1_250;
                        float hi_1_251 = fmaxf(xsl[3], xsl[4]);
                        float lo_1_251 = fminf(xsl[3], xsl[4]);
                        xsl[3] = hi_1_251;
                        xsl[4] = lo_1_251;
                        float hi_1_252 = fmaxf(xsl[6], xsl[13]);
                        float lo_1_252 = fminf(xsl[6], xsl[13]);
                        xsl[6] = hi_1_252;
                        xsl[13] = lo_1_252;
                        float hi_1_253 = fmaxf(xsl[8], xsl[14]);
                        float lo_1_253 = fminf(xsl[8], xsl[14]);
                        xsl[8] = hi_1_253;
                        xsl[14] = lo_1_253;
                        float hi_1_254 = fmaxf(xsl[10], xsl[15]);
                        float lo_1_254 = fminf(xsl[10], xsl[15]);
                        xsl[10] = hi_1_254;
                        xsl[15] = lo_1_254;
                        float hi_1_255 = fmaxf(xsl[11], xsl[12]);
                        float lo_1_255 = fminf(xsl[11], xsl[12]);
                        xsl[11] = hi_1_255;
                        xsl[12] = lo_1_255;
                        float hi_1_256 = fmaxf(xsl[0], xsl[1]);
                        float lo_1_256 = fminf(xsl[0], xsl[1]);
                        xsl[0] = hi_1_256;
                        xsl[1] = lo_1_256;
                        float hi_1_257 = fmaxf(xsl[2], xsl[3]);
                        float lo_1_257 = fminf(xsl[2], xsl[3]);
                        xsl[2] = hi_1_257;
                        xsl[3] = lo_1_257;
                        float hi_1_258 = fmaxf(xsl[4], xsl[5]);
                        float lo_1_258 = fminf(xsl[4], xsl[5]);
                        xsl[4] = hi_1_258;
                        xsl[5] = lo_1_258;
                        float hi_1_259 = fmaxf(xsl[6], xsl[8]);
                        float lo_1_259 = fminf(xsl[6], xsl[8]);
                        xsl[6] = hi_1_259;
                        xsl[8] = lo_1_259;
                        float hi_1_260 = fmaxf(xsl[7], xsl[9]);
                        float lo_1_260 = fminf(xsl[7], xsl[9]);
                        xsl[7] = hi_1_260;
                        xsl[9] = lo_1_260;
                        float hi_1_261 = fmaxf(xsl[10], xsl[11]);
                        float lo_1_261 = fminf(xsl[10], xsl[11]);
                        xsl[10] = hi_1_261;
                        xsl[11] = lo_1_261;
                        float hi_1_262 = fmaxf(xsl[12], xsl[13]);
                        float lo_1_262 = fminf(xsl[12], xsl[13]);
                        xsl[12] = hi_1_262;
                        xsl[13] = lo_1_262;
                        float hi_1_263 = fmaxf(xsl[14], xsl[15]);
                        float lo_1_263 = fminf(xsl[14], xsl[15]);
                        xsl[14] = hi_1_263;
                        xsl[15] = lo_1_263;
                        float hi_1_264 = fmaxf(xsl[0], xsl[2]);
                        float lo_1_264 = fminf(xsl[0], xsl[2]);
                        xsl[0] = hi_1_264;
                        xsl[2] = lo_1_264;
                        float hi_1_265 = fmaxf(xsl[1], xsl[3]);
                        float lo_1_265 = fminf(xsl[1], xsl[3]);
                        xsl[1] = hi_1_265;
                        xsl[3] = lo_1_265;
                        float hi_1_266 = fmaxf(xsl[4], xsl[10]);
                        float lo_1_266 = fminf(xsl[4], xsl[10]);
                        xsl[4] = hi_1_266;
                        xsl[10] = lo_1_266;
                        float hi_1_267 = fmaxf(xsl[5], xsl[11]);
                        float lo_1_267 = fminf(xsl[5], xsl[11]);
                        xsl[5] = hi_1_267;
                        xsl[11] = lo_1_267;
                        float hi_1_268 = fmaxf(xsl[6], xsl[7]);
                        float lo_1_268 = fminf(xsl[6], xsl[7]);
                        xsl[6] = hi_1_268;
                        xsl[7] = lo_1_268;
                        float hi_1_269 = fmaxf(xsl[8], xsl[9]);
                        float lo_1_269 = fminf(xsl[8], xsl[9]);
                        xsl[8] = hi_1_269;
                        xsl[9] = lo_1_269;
                        float hi_1_270 = fmaxf(xsl[12], xsl[14]);
                        float lo_1_270 = fminf(xsl[12], xsl[14]);
                        xsl[12] = hi_1_270;
                        xsl[14] = lo_1_270;
                        float hi_1_271 = fmaxf(xsl[13], xsl[15]);
                        float lo_1_271 = fminf(xsl[13], xsl[15]);
                        xsl[13] = hi_1_271;
                        xsl[15] = lo_1_271;
                        float hi_1_272 = fmaxf(xsl[1], xsl[2]);
                        float lo_1_272 = fminf(xsl[1], xsl[2]);
                        xsl[1] = hi_1_272;
                        xsl[2] = lo_1_272;
                        float hi_1_273 = fmaxf(xsl[3], xsl[12]);
                        float lo_1_273 = fminf(xsl[3], xsl[12]);
                        xsl[3] = hi_1_273;
                        xsl[12] = lo_1_273;
                        float hi_1_274 = fmaxf(xsl[4], xsl[6]);
                        float lo_1_274 = fminf(xsl[4], xsl[6]);
                        xsl[4] = hi_1_274;
                        xsl[6] = lo_1_274;
                        float hi_1_275 = fmaxf(xsl[5], xsl[7]);
                        float lo_1_275 = fminf(xsl[5], xsl[7]);
                        xsl[5] = hi_1_275;
                        xsl[7] = lo_1_275;
                        float hi_1_276 = fmaxf(xsl[8], xsl[10]);
                        float lo_1_276 = fminf(xsl[8], xsl[10]);
                        xsl[8] = hi_1_276;
                        xsl[10] = lo_1_276;
                        float hi_1_277 = fmaxf(xsl[9], xsl[11]);
                        float lo_1_277 = fminf(xsl[9], xsl[11]);
                        xsl[9] = hi_1_277;
                        xsl[11] = lo_1_277;
                        float hi_1_278 = fmaxf(xsl[13], xsl[14]);
                        float lo_1_278 = fminf(xsl[13], xsl[14]);
                        xsl[13] = hi_1_278;
                        xsl[14] = lo_1_278;
                        float hi_1_279 = fmaxf(xsl[1], xsl[4]);
                        float lo_1_279 = fminf(xsl[1], xsl[4]);
                        xsl[1] = hi_1_279;
                        xsl[4] = lo_1_279;
                        float hi_1_280 = fmaxf(xsl[2], xsl[6]);
                        float lo_1_280 = fminf(xsl[2], xsl[6]);
                        xsl[2] = hi_1_280;
                        xsl[6] = lo_1_280;
                        float hi_1_281 = fmaxf(xsl[5], xsl[8]);
                        float lo_1_281 = fminf(xsl[5], xsl[8]);
                        xsl[5] = hi_1_281;
                        xsl[8] = lo_1_281;
                        float hi_1_282 = fmaxf(xsl[7], xsl[10]);
                        float lo_1_282 = fminf(xsl[7], xsl[10]);
                        xsl[7] = hi_1_282;
                        xsl[10] = lo_1_282;
                        float hi_1_283 = fmaxf(xsl[9], xsl[13]);
                        float lo_1_283 = fminf(xsl[9], xsl[13]);
                        xsl[9] = hi_1_283;
                        xsl[13] = lo_1_283;
                        float hi_1_284 = fmaxf(xsl[11], xsl[14]);
                        float lo_1_284 = fminf(xsl[11], xsl[14]);
                        xsl[11] = hi_1_284;
                        xsl[14] = lo_1_284;
                        float hi_1_285 = fmaxf(xsl[2], xsl[4]);
                        float lo_1_285 = fminf(xsl[2], xsl[4]);
                        xsl[2] = hi_1_285;
                        xsl[4] = lo_1_285;
                        float hi_1_286 = fmaxf(xsl[3], xsl[6]);
                        float lo_1_286 = fminf(xsl[3], xsl[6]);
                        xsl[3] = hi_1_286;
                        xsl[6] = lo_1_286;
                        float hi_1_287 = fmaxf(xsl[9], xsl[12]);
                        float lo_1_287 = fminf(xsl[9], xsl[12]);
                        xsl[9] = hi_1_287;
                        xsl[12] = lo_1_287;
                        float hi_1_288 = fmaxf(xsl[11], xsl[13]);
                        float lo_1_288 = fminf(xsl[11], xsl[13]);
                        xsl[11] = hi_1_288;
                        xsl[13] = lo_1_288;
                        float hi_1_289 = fmaxf(xsl[3], xsl[5]);
                        float lo_1_289 = fminf(xsl[3], xsl[5]);
                        xsl[3] = hi_1_289;
                        xsl[5] = lo_1_289;
                        float hi_1_290 = fmaxf(xsl[6], xsl[8]);
                        float lo_1_290 = fminf(xsl[6], xsl[8]);
                        xsl[6] = hi_1_290;
                        xsl[8] = lo_1_290;
                        float hi_1_291 = fmaxf(xsl[7], xsl[9]);
                        float lo_1_291 = fminf(xsl[7], xsl[9]);
                        xsl[7] = hi_1_291;
                        xsl[9] = lo_1_291;
                        float hi_1_292 = fmaxf(xsl[10], xsl[12]);
                        float lo_1_292 = fminf(xsl[10], xsl[12]);
                        xsl[10] = hi_1_292;
                        xsl[12] = lo_1_292;
                        float hi_1_293 = fmaxf(xsl[3], xsl[4]);
                        float lo_1_293 = fminf(xsl[3], xsl[4]);
                        xsl[3] = hi_1_293;
                        xsl[4] = lo_1_293;
                        float hi_1_294 = fmaxf(xsl[5], xsl[6]);
                        float lo_1_294 = fminf(xsl[5], xsl[6]);
                        xsl[5] = hi_1_294;
                        xsl[6] = lo_1_294;
                        float hi_1_295 = fmaxf(xsl[7], xsl[8]);
                        float lo_1_295 = fminf(xsl[7], xsl[8]);
                        xsl[7] = hi_1_295;
                        xsl[8] = lo_1_295;
                        float hi_1_296 = fmaxf(xsl[9], xsl[10]);
                        float lo_1_296 = fminf(xsl[9], xsl[10]);
                        xsl[9] = hi_1_296;
                        xsl[10] = lo_1_296;
                        float hi_1_297 = fmaxf(xsl[11], xsl[12]);
                        float lo_1_297 = fminf(xsl[11], xsl[12]);
                        xsl[11] = hi_1_297;
                        xsl[12] = lo_1_297;
                        float hi_1_298 = fmaxf(xsl[6], xsl[7]);
                        float lo_1_298 = fminf(xsl[6], xsl[7]);
                        xsl[6] = hi_1_298;
                        xsl[7] = lo_1_298;
                        float hi_1_299 = fmaxf(xsl[8], xsl[9]);
                        float lo_1_299 = fminf(xsl[8], xsl[9]);
                        xsl[8] = hi_1_299;
                        xsl[9] = lo_1_299;
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
