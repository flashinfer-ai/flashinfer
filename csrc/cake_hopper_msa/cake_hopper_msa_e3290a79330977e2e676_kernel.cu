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

extern "C" {

__global__ __launch_bounds__(256) void kernel_cake_hopper_msa_e3290a79330977e2e676(
    unsigned int* __restrict__ S, int* __restrict__ nvp, int* __restrict__ out, int tiles,
    int total_q, int num_heads, int num_chunks, int fb, int fe) {
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
  int R = num_chunks * 32;
  int tb = w * R;
  unsigned int cbt[32];
  float kbt[32];
  long long p = cbase + (long long)tb * nq64;
  cbt[0] = 4286578688u;
  if (lim_full > tb) {
    cbt[0] = S[p];
  }
  cbt[1] = 4286578688u;
  if (lim_full > tb + 1) {
    cbt[1] = S[p + nq64];
  }
  cbt[2] = 4286578688u;
  if (lim_full > tb + 2) {
    cbt[2] = S[p + 2 * nq64];
  }
  cbt[3] = 4286578688u;
  if (lim_full > tb + 3) {
    cbt[3] = S[p + 3 * nq64];
  }
  cbt[4] = 4286578688u;
  if (lim_full > tb + 4) {
    cbt[4] = S[p + 4 * nq64];
  }
  cbt[5] = 4286578688u;
  if (lim_full > tb + 5) {
    cbt[5] = S[p + 5 * nq64];
  }
  cbt[6] = 4286578688u;
  if (lim_full > tb + 6) {
    cbt[6] = S[p + 6 * nq64];
  }
  cbt[7] = 4286578688u;
  if (lim_full > tb + 7) {
    cbt[7] = S[p + 7 * nq64];
  }
  cbt[8] = 4286578688u;
  if (lim_full > tb + 8) {
    cbt[8] = S[p + 8 * nq64];
  }
  cbt[9] = 4286578688u;
  if (lim_full > tb + 9) {
    cbt[9] = S[p + 9 * nq64];
  }
  cbt[10] = 4286578688u;
  if (lim_full > tb + 10) {
    cbt[10] = S[p + 10 * nq64];
  }
  cbt[11] = 4286578688u;
  if (lim_full > tb + 11) {
    cbt[11] = S[p + 11 * nq64];
  }
  cbt[12] = 4286578688u;
  if (lim_full > tb + 12) {
    cbt[12] = S[p + 12 * nq64];
  }
  cbt[13] = 4286578688u;
  if (lim_full > tb + 13) {
    cbt[13] = S[p + 13 * nq64];
  }
  cbt[14] = 4286578688u;
  if (lim_full > tb + 14) {
    cbt[14] = S[p + 14 * nq64];
  }
  cbt[15] = 4286578688u;
  if (lim_full > tb + 15) {
    cbt[15] = S[p + 15 * nq64];
  }
  cbt[16] = 4286578688u;
  if (lim_full > tb + 16) {
    cbt[16] = S[p + 16 * nq64];
  }
  cbt[17] = 4286578688u;
  if (lim_full > tb + 17) {
    cbt[17] = S[p + 17 * nq64];
  }
  cbt[18] = 4286578688u;
  if (lim_full > tb + 18) {
    cbt[18] = S[p + 18 * nq64];
  }
  cbt[19] = 4286578688u;
  if (lim_full > tb + 19) {
    cbt[19] = S[p + 19 * nq64];
  }
  cbt[20] = 4286578688u;
  if (lim_full > tb + 20) {
    cbt[20] = S[p + 20 * nq64];
  }
  cbt[21] = 4286578688u;
  if (lim_full > tb + 21) {
    cbt[21] = S[p + 21 * nq64];
  }
  cbt[22] = 4286578688u;
  if (lim_full > tb + 22) {
    cbt[22] = S[p + 22 * nq64];
  }
  cbt[23] = 4286578688u;
  if (lim_full > tb + 23) {
    cbt[23] = S[p + 23 * nq64];
  }
  cbt[24] = 4286578688u;
  if (lim_full > tb + 24) {
    cbt[24] = S[p + 24 * nq64];
  }
  cbt[25] = 4286578688u;
  if (lim_full > tb + 25) {
    cbt[25] = S[p + 25 * nq64];
  }
  cbt[26] = 4286578688u;
  if (lim_full > tb + 26) {
    cbt[26] = S[p + 26 * nq64];
  }
  cbt[27] = 4286578688u;
  if (lim_full > tb + 27) {
    cbt[27] = S[p + 27 * nq64];
  }
  cbt[28] = 4286578688u;
  if (lim_full > tb + 28) {
    cbt[28] = S[p + 28 * nq64];
  }
  cbt[29] = 4286578688u;
  if (lim_full > tb + 29) {
    cbt[29] = S[p + 29 * nq64];
  }
  cbt[30] = 4286578688u;
  if (lim_full > tb + 30) {
    cbt[30] = S[p + 30 * nq64];
  }
  cbt[31] = 4286578688u;
  if (lim_full > tb + 31) {
    cbt[31] = S[p + 31 * nq64];
  }
  if (lim <= tb) {
    cbt[0] = 4286578688u;
  }
  if (lim <= tb + 1) {
    cbt[1] = 4286578688u;
  }
  if (lim <= tb + 2) {
    cbt[2] = 4286578688u;
  }
  if (lim <= tb + 3) {
    cbt[3] = 4286578688u;
  }
  if (lim <= tb + 4) {
    cbt[4] = 4286578688u;
  }
  if (lim <= tb + 5) {
    cbt[5] = 4286578688u;
  }
  if (lim <= tb + 6) {
    cbt[6] = 4286578688u;
  }
  if (lim <= tb + 7) {
    cbt[7] = 4286578688u;
  }
  if (lim <= tb + 8) {
    cbt[8] = 4286578688u;
  }
  if (lim <= tb + 9) {
    cbt[9] = 4286578688u;
  }
  if (lim <= tb + 10) {
    cbt[10] = 4286578688u;
  }
  if (lim <= tb + 11) {
    cbt[11] = 4286578688u;
  }
  if (lim <= tb + 12) {
    cbt[12] = 4286578688u;
  }
  if (lim <= tb + 13) {
    cbt[13] = 4286578688u;
  }
  if (lim <= tb + 14) {
    cbt[14] = 4286578688u;
  }
  if (lim <= tb + 15) {
    cbt[15] = 4286578688u;
  }
  if (lim <= tb + 16) {
    cbt[16] = 4286578688u;
  }
  if (lim <= tb + 17) {
    cbt[17] = 4286578688u;
  }
  if (lim <= tb + 18) {
    cbt[18] = 4286578688u;
  }
  if (lim <= tb + 19) {
    cbt[19] = 4286578688u;
  }
  if (lim <= tb + 20) {
    cbt[20] = 4286578688u;
  }
  if (lim <= tb + 21) {
    cbt[21] = 4286578688u;
  }
  if (lim <= tb + 22) {
    cbt[22] = 4286578688u;
  }
  if (lim <= tb + 23) {
    cbt[23] = 4286578688u;
  }
  if (lim <= tb + 24) {
    cbt[24] = 4286578688u;
  }
  if (lim <= tb + 25) {
    cbt[25] = 4286578688u;
  }
  if (lim <= tb + 26) {
    cbt[26] = 4286578688u;
  }
  if (lim <= tb + 27) {
    cbt[27] = 4286578688u;
  }
  if (lim <= tb + 28) {
    cbt[28] = 4286578688u;
  }
  if (lim <= tb + 29) {
    cbt[29] = 4286578688u;
  }
  if (lim <= tb + 30) {
    cbt[30] = 4286578688u;
  }
  if (lim <= tb + 31) {
    cbt[31] = 4286578688u;
  }
  float sc = __uint_as_float(cbt[0]);
  sc = fmaxf(sc, -1.7014118346046923e+38f);
  sc = fminf(sc, 1.7014118346046923e+38f);
  unsigned int key = __as_u32(sc) & 4294966784u | (unsigned int)tb;
  kbt[0] = __uint_as_float(key);
  float sc_0 = __uint_as_float(cbt[1]);
  sc_0 = fmaxf(sc_0, -1.7014118346046923e+38f);
  sc_0 = fminf(sc_0, 1.7014118346046923e+38f);
  unsigned int key_1 = __as_u32(sc_0) & 4294966784u | (unsigned int)(tb + 1);
  kbt[1] = __uint_as_float(key_1);
  float sc_2 = __uint_as_float(cbt[2]);
  sc_2 = fmaxf(sc_2, -1.7014118346046923e+38f);
  sc_2 = fminf(sc_2, 1.7014118346046923e+38f);
  unsigned int key_3 = __as_u32(sc_2) & 4294966784u | (unsigned int)(tb + 2);
  kbt[2] = __uint_as_float(key_3);
  float sc_4 = __uint_as_float(cbt[3]);
  sc_4 = fmaxf(sc_4, -1.7014118346046923e+38f);
  sc_4 = fminf(sc_4, 1.7014118346046923e+38f);
  unsigned int key_5 = __as_u32(sc_4) & 4294966784u | (unsigned int)(tb + 3);
  kbt[3] = __uint_as_float(key_5);
  float sc_6 = __uint_as_float(cbt[4]);
  sc_6 = fmaxf(sc_6, -1.7014118346046923e+38f);
  sc_6 = fminf(sc_6, 1.7014118346046923e+38f);
  unsigned int key_7 = __as_u32(sc_6) & 4294966784u | (unsigned int)(tb + 4);
  kbt[4] = __uint_as_float(key_7);
  float sc_8 = __uint_as_float(cbt[5]);
  sc_8 = fmaxf(sc_8, -1.7014118346046923e+38f);
  sc_8 = fminf(sc_8, 1.7014118346046923e+38f);
  unsigned int key_9 = __as_u32(sc_8) & 4294966784u | (unsigned int)(tb + 5);
  kbt[5] = __uint_as_float(key_9);
  float sc_10 = __uint_as_float(cbt[6]);
  sc_10 = fmaxf(sc_10, -1.7014118346046923e+38f);
  sc_10 = fminf(sc_10, 1.7014118346046923e+38f);
  unsigned int key_11 = __as_u32(sc_10) & 4294966784u | (unsigned int)(tb + 6);
  kbt[6] = __uint_as_float(key_11);
  float sc_12 = __uint_as_float(cbt[7]);
  sc_12 = fmaxf(sc_12, -1.7014118346046923e+38f);
  sc_12 = fminf(sc_12, 1.7014118346046923e+38f);
  unsigned int key_13 = __as_u32(sc_12) & 4294966784u | (unsigned int)(tb + 7);
  kbt[7] = __uint_as_float(key_13);
  float sc_14 = __uint_as_float(cbt[8]);
  sc_14 = fmaxf(sc_14, -1.7014118346046923e+38f);
  sc_14 = fminf(sc_14, 1.7014118346046923e+38f);
  unsigned int key_15 = __as_u32(sc_14) & 4294966784u | (unsigned int)(tb + 8);
  kbt[8] = __uint_as_float(key_15);
  float sc_16 = __uint_as_float(cbt[9]);
  sc_16 = fmaxf(sc_16, -1.7014118346046923e+38f);
  sc_16 = fminf(sc_16, 1.7014118346046923e+38f);
  unsigned int key_17 = __as_u32(sc_16) & 4294966784u | (unsigned int)(tb + 9);
  kbt[9] = __uint_as_float(key_17);
  float sc_18 = __uint_as_float(cbt[10]);
  sc_18 = fmaxf(sc_18, -1.7014118346046923e+38f);
  sc_18 = fminf(sc_18, 1.7014118346046923e+38f);
  unsigned int key_19 = __as_u32(sc_18) & 4294966784u | (unsigned int)(tb + 10);
  kbt[10] = __uint_as_float(key_19);
  float sc_20 = __uint_as_float(cbt[11]);
  sc_20 = fmaxf(sc_20, -1.7014118346046923e+38f);
  sc_20 = fminf(sc_20, 1.7014118346046923e+38f);
  unsigned int key_21 = __as_u32(sc_20) & 4294966784u | (unsigned int)(tb + 11);
  kbt[11] = __uint_as_float(key_21);
  float sc_22 = __uint_as_float(cbt[12]);
  sc_22 = fmaxf(sc_22, -1.7014118346046923e+38f);
  sc_22 = fminf(sc_22, 1.7014118346046923e+38f);
  unsigned int key_23 = __as_u32(sc_22) & 4294966784u | (unsigned int)(tb + 12);
  kbt[12] = __uint_as_float(key_23);
  float sc_24 = __uint_as_float(cbt[13]);
  sc_24 = fmaxf(sc_24, -1.7014118346046923e+38f);
  sc_24 = fminf(sc_24, 1.7014118346046923e+38f);
  unsigned int key_25 = __as_u32(sc_24) & 4294966784u | (unsigned int)(tb + 13);
  kbt[13] = __uint_as_float(key_25);
  float sc_26 = __uint_as_float(cbt[14]);
  sc_26 = fmaxf(sc_26, -1.7014118346046923e+38f);
  sc_26 = fminf(sc_26, 1.7014118346046923e+38f);
  unsigned int key_27 = __as_u32(sc_26) & 4294966784u | (unsigned int)(tb + 14);
  kbt[14] = __uint_as_float(key_27);
  float sc_28 = __uint_as_float(cbt[15]);
  sc_28 = fmaxf(sc_28, -1.7014118346046923e+38f);
  sc_28 = fminf(sc_28, 1.7014118346046923e+38f);
  unsigned int key_29 = __as_u32(sc_28) & 4294966784u | (unsigned int)(tb + 15);
  kbt[15] = __uint_as_float(key_29);
  float sc_30 = __uint_as_float(cbt[16]);
  sc_30 = fmaxf(sc_30, -1.7014118346046923e+38f);
  sc_30 = fminf(sc_30, 1.7014118346046923e+38f);
  unsigned int key_31 = __as_u32(sc_30) & 4294966784u | (unsigned int)(tb + 16);
  kbt[16] = __uint_as_float(key_31);
  float sc_32 = __uint_as_float(cbt[17]);
  sc_32 = fmaxf(sc_32, -1.7014118346046923e+38f);
  sc_32 = fminf(sc_32, 1.7014118346046923e+38f);
  unsigned int key_33 = __as_u32(sc_32) & 4294966784u | (unsigned int)(tb + 17);
  kbt[17] = __uint_as_float(key_33);
  float sc_34 = __uint_as_float(cbt[18]);
  sc_34 = fmaxf(sc_34, -1.7014118346046923e+38f);
  sc_34 = fminf(sc_34, 1.7014118346046923e+38f);
  unsigned int key_35 = __as_u32(sc_34) & 4294966784u | (unsigned int)(tb + 18);
  kbt[18] = __uint_as_float(key_35);
  float sc_36 = __uint_as_float(cbt[19]);
  sc_36 = fmaxf(sc_36, -1.7014118346046923e+38f);
  sc_36 = fminf(sc_36, 1.7014118346046923e+38f);
  unsigned int key_37 = __as_u32(sc_36) & 4294966784u | (unsigned int)(tb + 19);
  kbt[19] = __uint_as_float(key_37);
  float sc_38 = __uint_as_float(cbt[20]);
  sc_38 = fmaxf(sc_38, -1.7014118346046923e+38f);
  sc_38 = fminf(sc_38, 1.7014118346046923e+38f);
  unsigned int key_39 = __as_u32(sc_38) & 4294966784u | (unsigned int)(tb + 20);
  kbt[20] = __uint_as_float(key_39);
  float sc_40 = __uint_as_float(cbt[21]);
  sc_40 = fmaxf(sc_40, -1.7014118346046923e+38f);
  sc_40 = fminf(sc_40, 1.7014118346046923e+38f);
  unsigned int key_41 = __as_u32(sc_40) & 4294966784u | (unsigned int)(tb + 21);
  kbt[21] = __uint_as_float(key_41);
  float sc_42 = __uint_as_float(cbt[22]);
  sc_42 = fmaxf(sc_42, -1.7014118346046923e+38f);
  sc_42 = fminf(sc_42, 1.7014118346046923e+38f);
  unsigned int key_43 = __as_u32(sc_42) & 4294966784u | (unsigned int)(tb + 22);
  kbt[22] = __uint_as_float(key_43);
  float sc_44 = __uint_as_float(cbt[23]);
  sc_44 = fmaxf(sc_44, -1.7014118346046923e+38f);
  sc_44 = fminf(sc_44, 1.7014118346046923e+38f);
  unsigned int key_45 = __as_u32(sc_44) & 4294966784u | (unsigned int)(tb + 23);
  kbt[23] = __uint_as_float(key_45);
  float sc_46 = __uint_as_float(cbt[24]);
  sc_46 = fmaxf(sc_46, -1.7014118346046923e+38f);
  sc_46 = fminf(sc_46, 1.7014118346046923e+38f);
  unsigned int key_47 = __as_u32(sc_46) & 4294966784u | (unsigned int)(tb + 24);
  kbt[24] = __uint_as_float(key_47);
  float sc_48 = __uint_as_float(cbt[25]);
  sc_48 = fmaxf(sc_48, -1.7014118346046923e+38f);
  sc_48 = fminf(sc_48, 1.7014118346046923e+38f);
  unsigned int key_49 = __as_u32(sc_48) & 4294966784u | (unsigned int)(tb + 25);
  kbt[25] = __uint_as_float(key_49);
  float sc_50 = __uint_as_float(cbt[26]);
  sc_50 = fmaxf(sc_50, -1.7014118346046923e+38f);
  sc_50 = fminf(sc_50, 1.7014118346046923e+38f);
  unsigned int key_51 = __as_u32(sc_50) & 4294966784u | (unsigned int)(tb + 26);
  kbt[26] = __uint_as_float(key_51);
  float sc_52 = __uint_as_float(cbt[27]);
  sc_52 = fmaxf(sc_52, -1.7014118346046923e+38f);
  sc_52 = fminf(sc_52, 1.7014118346046923e+38f);
  unsigned int key_53 = __as_u32(sc_52) & 4294966784u | (unsigned int)(tb + 27);
  kbt[27] = __uint_as_float(key_53);
  float sc_54 = __uint_as_float(cbt[28]);
  sc_54 = fmaxf(sc_54, -1.7014118346046923e+38f);
  sc_54 = fminf(sc_54, 1.7014118346046923e+38f);
  unsigned int key_55 = __as_u32(sc_54) & 4294966784u | (unsigned int)(tb + 28);
  kbt[28] = __uint_as_float(key_55);
  float sc_56 = __uint_as_float(cbt[29]);
  sc_56 = fmaxf(sc_56, -1.7014118346046923e+38f);
  sc_56 = fminf(sc_56, 1.7014118346046923e+38f);
  unsigned int key_57 = __as_u32(sc_56) & 4294966784u | (unsigned int)(tb + 29);
  kbt[29] = __uint_as_float(key_57);
  float sc_58 = __uint_as_float(cbt[30]);
  sc_58 = fmaxf(sc_58, -1.7014118346046923e+38f);
  sc_58 = fminf(sc_58, 1.7014118346046923e+38f);
  unsigned int key_59 = __as_u32(sc_58) & 4294966784u | (unsigned int)(tb + 30);
  kbt[30] = __uint_as_float(key_59);
  float sc_60 = __uint_as_float(cbt[31]);
  sc_60 = fmaxf(sc_60, -1.7014118346046923e+38f);
  sc_60 = fminf(sc_60, 1.7014118346046923e+38f);
  unsigned int key_61 = __as_u32(sc_60) & 4294966784u | (unsigned int)(tb + 31);
  kbt[31] = __uint_as_float(key_61);
  int f = 0;
  if ((tb < fb || tb + 32 > lim - fe) && tb < lim) {
    f = 1;
  }
  if (f != 0) {
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
  float hi = fmaxf(kbt[0], kbt[13]);
  float lo = fminf(kbt[0], kbt[13]);
  kbt[0] = hi;
  kbt[13] = lo;
  float hi_62 = fmaxf(kbt[1], kbt[12]);
  float lo_63 = fminf(kbt[1], kbt[12]);
  kbt[1] = hi_62;
  kbt[12] = lo_63;
  float hi_64 = fmaxf(kbt[2], kbt[15]);
  float lo_65 = fminf(kbt[2], kbt[15]);
  kbt[2] = hi_64;
  kbt[15] = lo_65;
  float hi_66 = fmaxf(kbt[3], kbt[14]);
  float lo_67 = fminf(kbt[3], kbt[14]);
  kbt[3] = hi_66;
  kbt[14] = lo_67;
  float hi_68 = fmaxf(kbt[4], kbt[8]);
  float lo_69 = fminf(kbt[4], kbt[8]);
  kbt[4] = hi_68;
  kbt[8] = lo_69;
  float hi_70 = fmaxf(kbt[5], kbt[6]);
  float lo_71 = fminf(kbt[5], kbt[6]);
  kbt[5] = hi_70;
  kbt[6] = lo_71;
  float hi_72 = fmaxf(kbt[7], kbt[11]);
  float lo_73 = fminf(kbt[7], kbt[11]);
  kbt[7] = hi_72;
  kbt[11] = lo_73;
  float hi_74 = fmaxf(kbt[9], kbt[10]);
  float lo_75 = fminf(kbt[9], kbt[10]);
  kbt[9] = hi_74;
  kbt[10] = lo_75;
  float hi_76 = fmaxf(kbt[0], kbt[5]);
  float lo_77 = fminf(kbt[0], kbt[5]);
  kbt[0] = hi_76;
  kbt[5] = lo_77;
  float hi_78 = fmaxf(kbt[1], kbt[7]);
  float lo_79 = fminf(kbt[1], kbt[7]);
  kbt[1] = hi_78;
  kbt[7] = lo_79;
  float hi_80 = fmaxf(kbt[2], kbt[9]);
  float lo_81 = fminf(kbt[2], kbt[9]);
  kbt[2] = hi_80;
  kbt[9] = lo_81;
  float hi_82 = fmaxf(kbt[3], kbt[4]);
  float lo_83 = fminf(kbt[3], kbt[4]);
  kbt[3] = hi_82;
  kbt[4] = lo_83;
  float hi_84 = fmaxf(kbt[6], kbt[13]);
  float lo_85 = fminf(kbt[6], kbt[13]);
  kbt[6] = hi_84;
  kbt[13] = lo_85;
  float hi_86 = fmaxf(kbt[8], kbt[14]);
  float lo_87 = fminf(kbt[8], kbt[14]);
  kbt[8] = hi_86;
  kbt[14] = lo_87;
  float hi_88 = fmaxf(kbt[10], kbt[15]);
  float lo_89 = fminf(kbt[10], kbt[15]);
  kbt[10] = hi_88;
  kbt[15] = lo_89;
  float hi_90 = fmaxf(kbt[11], kbt[12]);
  float lo_91 = fminf(kbt[11], kbt[12]);
  kbt[11] = hi_90;
  kbt[12] = lo_91;
  float hi_92 = fmaxf(kbt[0], kbt[1]);
  float lo_93 = fminf(kbt[0], kbt[1]);
  kbt[0] = hi_92;
  kbt[1] = lo_93;
  float hi_94 = fmaxf(kbt[2], kbt[3]);
  float lo_95 = fminf(kbt[2], kbt[3]);
  kbt[2] = hi_94;
  kbt[3] = lo_95;
  float hi_96 = fmaxf(kbt[4], kbt[5]);
  float lo_97 = fminf(kbt[4], kbt[5]);
  kbt[4] = hi_96;
  kbt[5] = lo_97;
  float hi_98 = fmaxf(kbt[6], kbt[8]);
  float lo_99 = fminf(kbt[6], kbt[8]);
  kbt[6] = hi_98;
  kbt[8] = lo_99;
  float hi_100 = fmaxf(kbt[7], kbt[9]);
  float lo_101 = fminf(kbt[7], kbt[9]);
  kbt[7] = hi_100;
  kbt[9] = lo_101;
  float hi_102 = fmaxf(kbt[10], kbt[11]);
  float lo_103 = fminf(kbt[10], kbt[11]);
  kbt[10] = hi_102;
  kbt[11] = lo_103;
  float hi_104 = fmaxf(kbt[12], kbt[13]);
  float lo_105 = fminf(kbt[12], kbt[13]);
  kbt[12] = hi_104;
  kbt[13] = lo_105;
  float hi_106 = fmaxf(kbt[14], kbt[15]);
  float lo_107 = fminf(kbt[14], kbt[15]);
  kbt[14] = hi_106;
  kbt[15] = lo_107;
  float hi_108 = fmaxf(kbt[0], kbt[2]);
  float lo_109 = fminf(kbt[0], kbt[2]);
  kbt[0] = hi_108;
  kbt[2] = lo_109;
  float hi_110 = fmaxf(kbt[1], kbt[3]);
  float lo_111 = fminf(kbt[1], kbt[3]);
  kbt[1] = hi_110;
  kbt[3] = lo_111;
  float hi_112 = fmaxf(kbt[4], kbt[10]);
  float lo_113 = fminf(kbt[4], kbt[10]);
  kbt[4] = hi_112;
  kbt[10] = lo_113;
  float hi_114 = fmaxf(kbt[5], kbt[11]);
  float lo_115 = fminf(kbt[5], kbt[11]);
  kbt[5] = hi_114;
  kbt[11] = lo_115;
  float hi_116 = fmaxf(kbt[6], kbt[7]);
  float lo_117 = fminf(kbt[6], kbt[7]);
  kbt[6] = hi_116;
  kbt[7] = lo_117;
  float hi_118 = fmaxf(kbt[8], kbt[9]);
  float lo_119 = fminf(kbt[8], kbt[9]);
  kbt[8] = hi_118;
  kbt[9] = lo_119;
  float hi_120 = fmaxf(kbt[12], kbt[14]);
  float lo_121 = fminf(kbt[12], kbt[14]);
  kbt[12] = hi_120;
  kbt[14] = lo_121;
  float hi_122 = fmaxf(kbt[13], kbt[15]);
  float lo_123 = fminf(kbt[13], kbt[15]);
  kbt[13] = hi_122;
  kbt[15] = lo_123;
  float hi_124 = fmaxf(kbt[1], kbt[2]);
  float lo_125 = fminf(kbt[1], kbt[2]);
  kbt[1] = hi_124;
  kbt[2] = lo_125;
  float hi_126 = fmaxf(kbt[3], kbt[12]);
  float lo_127 = fminf(kbt[3], kbt[12]);
  kbt[3] = hi_126;
  kbt[12] = lo_127;
  float hi_128 = fmaxf(kbt[4], kbt[6]);
  float lo_129 = fminf(kbt[4], kbt[6]);
  kbt[4] = hi_128;
  kbt[6] = lo_129;
  float hi_130 = fmaxf(kbt[5], kbt[7]);
  float lo_131 = fminf(kbt[5], kbt[7]);
  kbt[5] = hi_130;
  kbt[7] = lo_131;
  float hi_132 = fmaxf(kbt[8], kbt[10]);
  float lo_133 = fminf(kbt[8], kbt[10]);
  kbt[8] = hi_132;
  kbt[10] = lo_133;
  float hi_134 = fmaxf(kbt[9], kbt[11]);
  float lo_135 = fminf(kbt[9], kbt[11]);
  kbt[9] = hi_134;
  kbt[11] = lo_135;
  float hi_136 = fmaxf(kbt[13], kbt[14]);
  float lo_137 = fminf(kbt[13], kbt[14]);
  kbt[13] = hi_136;
  kbt[14] = lo_137;
  float hi_138 = fmaxf(kbt[1], kbt[4]);
  float lo_139 = fminf(kbt[1], kbt[4]);
  kbt[1] = hi_138;
  kbt[4] = lo_139;
  float hi_140 = fmaxf(kbt[2], kbt[6]);
  float lo_141 = fminf(kbt[2], kbt[6]);
  kbt[2] = hi_140;
  kbt[6] = lo_141;
  float hi_142 = fmaxf(kbt[5], kbt[8]);
  float lo_143 = fminf(kbt[5], kbt[8]);
  kbt[5] = hi_142;
  kbt[8] = lo_143;
  float hi_144 = fmaxf(kbt[7], kbt[10]);
  float lo_145 = fminf(kbt[7], kbt[10]);
  kbt[7] = hi_144;
  kbt[10] = lo_145;
  float hi_146 = fmaxf(kbt[9], kbt[13]);
  float lo_147 = fminf(kbt[9], kbt[13]);
  kbt[9] = hi_146;
  kbt[13] = lo_147;
  float hi_148 = fmaxf(kbt[11], kbt[14]);
  float lo_149 = fminf(kbt[11], kbt[14]);
  kbt[11] = hi_148;
  kbt[14] = lo_149;
  float hi_150 = fmaxf(kbt[2], kbt[4]);
  float lo_151 = fminf(kbt[2], kbt[4]);
  kbt[2] = hi_150;
  kbt[4] = lo_151;
  float hi_152 = fmaxf(kbt[3], kbt[6]);
  float lo_153 = fminf(kbt[3], kbt[6]);
  kbt[3] = hi_152;
  kbt[6] = lo_153;
  float hi_154 = fmaxf(kbt[9], kbt[12]);
  float lo_155 = fminf(kbt[9], kbt[12]);
  kbt[9] = hi_154;
  kbt[12] = lo_155;
  float hi_156 = fmaxf(kbt[11], kbt[13]);
  float lo_157 = fminf(kbt[11], kbt[13]);
  kbt[11] = hi_156;
  kbt[13] = lo_157;
  float hi_158 = fmaxf(kbt[3], kbt[5]);
  float lo_159 = fminf(kbt[3], kbt[5]);
  kbt[3] = hi_158;
  kbt[5] = lo_159;
  float hi_160 = fmaxf(kbt[6], kbt[8]);
  float lo_161 = fminf(kbt[6], kbt[8]);
  kbt[6] = hi_160;
  kbt[8] = lo_161;
  float hi_162 = fmaxf(kbt[7], kbt[9]);
  float lo_163 = fminf(kbt[7], kbt[9]);
  kbt[7] = hi_162;
  kbt[9] = lo_163;
  float hi_164 = fmaxf(kbt[10], kbt[12]);
  float lo_165 = fminf(kbt[10], kbt[12]);
  kbt[10] = hi_164;
  kbt[12] = lo_165;
  float hi_166 = fmaxf(kbt[3], kbt[4]);
  float lo_167 = fminf(kbt[3], kbt[4]);
  kbt[3] = hi_166;
  kbt[4] = lo_167;
  float hi_168 = fmaxf(kbt[5], kbt[6]);
  float lo_169 = fminf(kbt[5], kbt[6]);
  kbt[5] = hi_168;
  kbt[6] = lo_169;
  float hi_170 = fmaxf(kbt[7], kbt[8]);
  float lo_171 = fminf(kbt[7], kbt[8]);
  kbt[7] = hi_170;
  kbt[8] = lo_171;
  float hi_172 = fmaxf(kbt[9], kbt[10]);
  float lo_173 = fminf(kbt[9], kbt[10]);
  kbt[9] = hi_172;
  kbt[10] = lo_173;
  float hi_174 = fmaxf(kbt[11], kbt[12]);
  float lo_175 = fminf(kbt[11], kbt[12]);
  kbt[11] = hi_174;
  kbt[12] = lo_175;
  float hi_176 = fmaxf(kbt[6], kbt[7]);
  float lo_177 = fminf(kbt[6], kbt[7]);
  kbt[6] = hi_176;
  kbt[7] = lo_177;
  float hi_178 = fmaxf(kbt[8], kbt[9]);
  float lo_179 = fminf(kbt[8], kbt[9]);
  kbt[8] = hi_178;
  kbt[9] = lo_179;
  float hi_180 = fmaxf(kbt[16], kbt[29]);
  float lo_181 = fminf(kbt[16], kbt[29]);
  kbt[16] = hi_180;
  kbt[29] = lo_181;
  float hi_182 = fmaxf(kbt[17], kbt[28]);
  float lo_183 = fminf(kbt[17], kbt[28]);
  kbt[17] = hi_182;
  kbt[28] = lo_183;
  float hi_184 = fmaxf(kbt[18], kbt[31]);
  float lo_185 = fminf(kbt[18], kbt[31]);
  kbt[18] = hi_184;
  kbt[31] = lo_185;
  float hi_186 = fmaxf(kbt[19], kbt[30]);
  float lo_187 = fminf(kbt[19], kbt[30]);
  kbt[19] = hi_186;
  kbt[30] = lo_187;
  float hi_188 = fmaxf(kbt[20], kbt[24]);
  float lo_189 = fminf(kbt[20], kbt[24]);
  kbt[20] = hi_188;
  kbt[24] = lo_189;
  float hi_190 = fmaxf(kbt[21], kbt[22]);
  float lo_191 = fminf(kbt[21], kbt[22]);
  kbt[21] = hi_190;
  kbt[22] = lo_191;
  float hi_192 = fmaxf(kbt[23], kbt[27]);
  float lo_193 = fminf(kbt[23], kbt[27]);
  kbt[23] = hi_192;
  kbt[27] = lo_193;
  float hi_194 = fmaxf(kbt[25], kbt[26]);
  float lo_195 = fminf(kbt[25], kbt[26]);
  kbt[25] = hi_194;
  kbt[26] = lo_195;
  float hi_196 = fmaxf(kbt[16], kbt[21]);
  float lo_197 = fminf(kbt[16], kbt[21]);
  kbt[16] = hi_196;
  kbt[21] = lo_197;
  float hi_198 = fmaxf(kbt[17], kbt[23]);
  float lo_199 = fminf(kbt[17], kbt[23]);
  kbt[17] = hi_198;
  kbt[23] = lo_199;
  float hi_200 = fmaxf(kbt[18], kbt[25]);
  float lo_201 = fminf(kbt[18], kbt[25]);
  kbt[18] = hi_200;
  kbt[25] = lo_201;
  float hi_202 = fmaxf(kbt[19], kbt[20]);
  float lo_203 = fminf(kbt[19], kbt[20]);
  kbt[19] = hi_202;
  kbt[20] = lo_203;
  float hi_204 = fmaxf(kbt[22], kbt[29]);
  float lo_205 = fminf(kbt[22], kbt[29]);
  kbt[22] = hi_204;
  kbt[29] = lo_205;
  float hi_206 = fmaxf(kbt[24], kbt[30]);
  float lo_207 = fminf(kbt[24], kbt[30]);
  kbt[24] = hi_206;
  kbt[30] = lo_207;
  float hi_208 = fmaxf(kbt[26], kbt[31]);
  float lo_209 = fminf(kbt[26], kbt[31]);
  kbt[26] = hi_208;
  kbt[31] = lo_209;
  float hi_210 = fmaxf(kbt[27], kbt[28]);
  float lo_211 = fminf(kbt[27], kbt[28]);
  kbt[27] = hi_210;
  kbt[28] = lo_211;
  float hi_212 = fmaxf(kbt[16], kbt[17]);
  float lo_213 = fminf(kbt[16], kbt[17]);
  kbt[16] = hi_212;
  kbt[17] = lo_213;
  float hi_214 = fmaxf(kbt[18], kbt[19]);
  float lo_215 = fminf(kbt[18], kbt[19]);
  kbt[18] = hi_214;
  kbt[19] = lo_215;
  float hi_216 = fmaxf(kbt[20], kbt[21]);
  float lo_217 = fminf(kbt[20], kbt[21]);
  kbt[20] = hi_216;
  kbt[21] = lo_217;
  float hi_218 = fmaxf(kbt[22], kbt[24]);
  float lo_219 = fminf(kbt[22], kbt[24]);
  kbt[22] = hi_218;
  kbt[24] = lo_219;
  float hi_220 = fmaxf(kbt[23], kbt[25]);
  float lo_221 = fminf(kbt[23], kbt[25]);
  kbt[23] = hi_220;
  kbt[25] = lo_221;
  float hi_222 = fmaxf(kbt[26], kbt[27]);
  float lo_223 = fminf(kbt[26], kbt[27]);
  kbt[26] = hi_222;
  kbt[27] = lo_223;
  float hi_224 = fmaxf(kbt[28], kbt[29]);
  float lo_225 = fminf(kbt[28], kbt[29]);
  kbt[28] = hi_224;
  kbt[29] = lo_225;
  float hi_226 = fmaxf(kbt[30], kbt[31]);
  float lo_227 = fminf(kbt[30], kbt[31]);
  kbt[30] = hi_226;
  kbt[31] = lo_227;
  float hi_228 = fmaxf(kbt[16], kbt[18]);
  float lo_229 = fminf(kbt[16], kbt[18]);
  kbt[16] = hi_228;
  kbt[18] = lo_229;
  float hi_230 = fmaxf(kbt[17], kbt[19]);
  float lo_231 = fminf(kbt[17], kbt[19]);
  kbt[17] = hi_230;
  kbt[19] = lo_231;
  float hi_232 = fmaxf(kbt[20], kbt[26]);
  float lo_233 = fminf(kbt[20], kbt[26]);
  kbt[20] = hi_232;
  kbt[26] = lo_233;
  float hi_234 = fmaxf(kbt[21], kbt[27]);
  float lo_235 = fminf(kbt[21], kbt[27]);
  kbt[21] = hi_234;
  kbt[27] = lo_235;
  float hi_236 = fmaxf(kbt[22], kbt[23]);
  float lo_237 = fminf(kbt[22], kbt[23]);
  kbt[22] = hi_236;
  kbt[23] = lo_237;
  float hi_238 = fmaxf(kbt[24], kbt[25]);
  float lo_239 = fminf(kbt[24], kbt[25]);
  kbt[24] = hi_238;
  kbt[25] = lo_239;
  float hi_240 = fmaxf(kbt[28], kbt[30]);
  float lo_241 = fminf(kbt[28], kbt[30]);
  kbt[28] = hi_240;
  kbt[30] = lo_241;
  float hi_242 = fmaxf(kbt[29], kbt[31]);
  float lo_243 = fminf(kbt[29], kbt[31]);
  kbt[29] = hi_242;
  kbt[31] = lo_243;
  float hi_244 = fmaxf(kbt[17], kbt[18]);
  float lo_245 = fminf(kbt[17], kbt[18]);
  kbt[17] = hi_244;
  kbt[18] = lo_245;
  float hi_246 = fmaxf(kbt[19], kbt[28]);
  float lo_247 = fminf(kbt[19], kbt[28]);
  kbt[19] = hi_246;
  kbt[28] = lo_247;
  float hi_248 = fmaxf(kbt[20], kbt[22]);
  float lo_249 = fminf(kbt[20], kbt[22]);
  kbt[20] = hi_248;
  kbt[22] = lo_249;
  float hi_250 = fmaxf(kbt[21], kbt[23]);
  float lo_251 = fminf(kbt[21], kbt[23]);
  kbt[21] = hi_250;
  kbt[23] = lo_251;
  float hi_252 = fmaxf(kbt[24], kbt[26]);
  float lo_253 = fminf(kbt[24], kbt[26]);
  kbt[24] = hi_252;
  kbt[26] = lo_253;
  float hi_254 = fmaxf(kbt[25], kbt[27]);
  float lo_255 = fminf(kbt[25], kbt[27]);
  kbt[25] = hi_254;
  kbt[27] = lo_255;
  float hi_256 = fmaxf(kbt[29], kbt[30]);
  float lo_257 = fminf(kbt[29], kbt[30]);
  kbt[29] = hi_256;
  kbt[30] = lo_257;
  float hi_258 = fmaxf(kbt[17], kbt[20]);
  float lo_259 = fminf(kbt[17], kbt[20]);
  kbt[17] = hi_258;
  kbt[20] = lo_259;
  float hi_260 = fmaxf(kbt[18], kbt[22]);
  float lo_261 = fminf(kbt[18], kbt[22]);
  kbt[18] = hi_260;
  kbt[22] = lo_261;
  float hi_262 = fmaxf(kbt[21], kbt[24]);
  float lo_263 = fminf(kbt[21], kbt[24]);
  kbt[21] = hi_262;
  kbt[24] = lo_263;
  float hi_264 = fmaxf(kbt[23], kbt[26]);
  float lo_265 = fminf(kbt[23], kbt[26]);
  kbt[23] = hi_264;
  kbt[26] = lo_265;
  float hi_266 = fmaxf(kbt[25], kbt[29]);
  float lo_267 = fminf(kbt[25], kbt[29]);
  kbt[25] = hi_266;
  kbt[29] = lo_267;
  float hi_268 = fmaxf(kbt[27], kbt[30]);
  float lo_269 = fminf(kbt[27], kbt[30]);
  kbt[27] = hi_268;
  kbt[30] = lo_269;
  float hi_270 = fmaxf(kbt[18], kbt[20]);
  float lo_271 = fminf(kbt[18], kbt[20]);
  kbt[18] = hi_270;
  kbt[20] = lo_271;
  float hi_272 = fmaxf(kbt[19], kbt[22]);
  float lo_273 = fminf(kbt[19], kbt[22]);
  kbt[19] = hi_272;
  kbt[22] = lo_273;
  float hi_274 = fmaxf(kbt[25], kbt[28]);
  float lo_275 = fminf(kbt[25], kbt[28]);
  kbt[25] = hi_274;
  kbt[28] = lo_275;
  float hi_276 = fmaxf(kbt[27], kbt[29]);
  float lo_277 = fminf(kbt[27], kbt[29]);
  kbt[27] = hi_276;
  kbt[29] = lo_277;
  float hi_278 = fmaxf(kbt[19], kbt[21]);
  float lo_279 = fminf(kbt[19], kbt[21]);
  kbt[19] = hi_278;
  kbt[21] = lo_279;
  float hi_280 = fmaxf(kbt[22], kbt[24]);
  float lo_281 = fminf(kbt[22], kbt[24]);
  kbt[22] = hi_280;
  kbt[24] = lo_281;
  float hi_282 = fmaxf(kbt[23], kbt[25]);
  float lo_283 = fminf(kbt[23], kbt[25]);
  kbt[23] = hi_282;
  kbt[25] = lo_283;
  float hi_284 = fmaxf(kbt[26], kbt[28]);
  float lo_285 = fminf(kbt[26], kbt[28]);
  kbt[26] = hi_284;
  kbt[28] = lo_285;
  float hi_286 = fmaxf(kbt[19], kbt[20]);
  float lo_287 = fminf(kbt[19], kbt[20]);
  kbt[19] = hi_286;
  kbt[20] = lo_287;
  float hi_288 = fmaxf(kbt[21], kbt[22]);
  float lo_289 = fminf(kbt[21], kbt[22]);
  kbt[21] = hi_288;
  kbt[22] = lo_289;
  float hi_290 = fmaxf(kbt[23], kbt[24]);
  float lo_291 = fminf(kbt[23], kbt[24]);
  kbt[23] = hi_290;
  kbt[24] = lo_291;
  float hi_292 = fmaxf(kbt[25], kbt[26]);
  float lo_293 = fminf(kbt[25], kbt[26]);
  kbt[25] = hi_292;
  kbt[26] = lo_293;
  float hi_294 = fmaxf(kbt[27], kbt[28]);
  float lo_295 = fminf(kbt[27], kbt[28]);
  kbt[27] = hi_294;
  kbt[28] = lo_295;
  float hi_296 = fmaxf(kbt[22], kbt[23]);
  float lo_297 = fminf(kbt[22], kbt[23]);
  kbt[22] = hi_296;
  kbt[23] = lo_297;
  float hi_298 = fmaxf(kbt[24], kbt[25]);
  float lo_299 = fminf(kbt[24], kbt[25]);
  kbt[24] = hi_298;
  kbt[25] = lo_299;
  float hi_300 = fmaxf(kbt[0], kbt[31]);
  float lo_301 = fminf(kbt[0], kbt[31]);
  kbt[0] = hi_300;
  rr = fmaxf(rr, lo_301);
  float hi_302 = fmaxf(kbt[1], kbt[30]);
  float lo_303 = fminf(kbt[1], kbt[30]);
  kbt[1] = hi_302;
  rr = fmaxf(rr, lo_303);
  float hi_304 = fmaxf(kbt[2], kbt[29]);
  float lo_305 = fminf(kbt[2], kbt[29]);
  kbt[2] = hi_304;
  rr = fmaxf(rr, lo_305);
  float hi_306 = fmaxf(kbt[3], kbt[28]);
  float lo_307 = fminf(kbt[3], kbt[28]);
  kbt[3] = hi_306;
  rr = fmaxf(rr, lo_307);
  float hi_308 = fmaxf(kbt[4], kbt[27]);
  float lo_309 = fminf(kbt[4], kbt[27]);
  kbt[4] = hi_308;
  rr = fmaxf(rr, lo_309);
  float hi_310 = fmaxf(kbt[5], kbt[26]);
  float lo_311 = fminf(kbt[5], kbt[26]);
  kbt[5] = hi_310;
  rr = fmaxf(rr, lo_311);
  float hi_312 = fmaxf(kbt[6], kbt[25]);
  float lo_313 = fminf(kbt[6], kbt[25]);
  kbt[6] = hi_312;
  rr = fmaxf(rr, lo_313);
  float hi_314 = fmaxf(kbt[7], kbt[24]);
  float lo_315 = fminf(kbt[7], kbt[24]);
  kbt[7] = hi_314;
  rr = fmaxf(rr, lo_315);
  float hi_316 = fmaxf(kbt[8], kbt[23]);
  float lo_317 = fminf(kbt[8], kbt[23]);
  kbt[8] = hi_316;
  rr = fmaxf(rr, lo_317);
  float hi_318 = fmaxf(kbt[9], kbt[22]);
  float lo_319 = fminf(kbt[9], kbt[22]);
  kbt[9] = hi_318;
  rr = fmaxf(rr, lo_319);
  float hi_320 = fmaxf(kbt[10], kbt[21]);
  float lo_321 = fminf(kbt[10], kbt[21]);
  kbt[10] = hi_320;
  rr = fmaxf(rr, lo_321);
  float hi_322 = fmaxf(kbt[11], kbt[20]);
  float lo_323 = fminf(kbt[11], kbt[20]);
  kbt[11] = hi_322;
  rr = fmaxf(rr, lo_323);
  float hi_324 = fmaxf(kbt[12], kbt[19]);
  float lo_325 = fminf(kbt[12], kbt[19]);
  kbt[12] = hi_324;
  rr = fmaxf(rr, lo_325);
  float hi_326 = fmaxf(kbt[13], kbt[18]);
  float lo_327 = fminf(kbt[13], kbt[18]);
  kbt[13] = hi_326;
  rr = fmaxf(rr, lo_327);
  float hi_328 = fmaxf(kbt[14], kbt[17]);
  float lo_329 = fminf(kbt[14], kbt[17]);
  kbt[14] = hi_328;
  rr = fmaxf(rr, lo_329);
  float hi_330 = fmaxf(kbt[15], kbt[16]);
  float lo_331 = fminf(kbt[15], kbt[16]);
  kbt[15] = hi_330;
  rr = fmaxf(rr, lo_331);
  float hi_332 = fmaxf(kbt[0], kbt[8]);
  float lo_333 = fminf(kbt[0], kbt[8]);
  kbt[0] = hi_332;
  kbt[8] = lo_333;
  float hi_334 = fmaxf(kbt[1], kbt[9]);
  float lo_335 = fminf(kbt[1], kbt[9]);
  kbt[1] = hi_334;
  kbt[9] = lo_335;
  float hi_336 = fmaxf(kbt[2], kbt[10]);
  float lo_337 = fminf(kbt[2], kbt[10]);
  kbt[2] = hi_336;
  kbt[10] = lo_337;
  float hi_338 = fmaxf(kbt[3], kbt[11]);
  float lo_339 = fminf(kbt[3], kbt[11]);
  kbt[3] = hi_338;
  kbt[11] = lo_339;
  float hi_340 = fmaxf(kbt[4], kbt[12]);
  float lo_341 = fminf(kbt[4], kbt[12]);
  kbt[4] = hi_340;
  kbt[12] = lo_341;
  float hi_342 = fmaxf(kbt[5], kbt[13]);
  float lo_343 = fminf(kbt[5], kbt[13]);
  kbt[5] = hi_342;
  kbt[13] = lo_343;
  float hi_344 = fmaxf(kbt[6], kbt[14]);
  float lo_345 = fminf(kbt[6], kbt[14]);
  kbt[6] = hi_344;
  kbt[14] = lo_345;
  float hi_346 = fmaxf(kbt[7], kbt[15]);
  float lo_347 = fminf(kbt[7], kbt[15]);
  kbt[7] = hi_346;
  kbt[15] = lo_347;
  float hi_348 = fmaxf(kbt[0], kbt[4]);
  float lo_349 = fminf(kbt[0], kbt[4]);
  kbt[0] = hi_348;
  kbt[4] = lo_349;
  float hi_350 = fmaxf(kbt[1], kbt[5]);
  float lo_351 = fminf(kbt[1], kbt[5]);
  kbt[1] = hi_350;
  kbt[5] = lo_351;
  float hi_352 = fmaxf(kbt[2], kbt[6]);
  float lo_353 = fminf(kbt[2], kbt[6]);
  kbt[2] = hi_352;
  kbt[6] = lo_353;
  float hi_354 = fmaxf(kbt[3], kbt[7]);
  float lo_355 = fminf(kbt[3], kbt[7]);
  kbt[3] = hi_354;
  kbt[7] = lo_355;
  float hi_356 = fmaxf(kbt[8], kbt[12]);
  float lo_357 = fminf(kbt[8], kbt[12]);
  kbt[8] = hi_356;
  kbt[12] = lo_357;
  float hi_358 = fmaxf(kbt[9], kbt[13]);
  float lo_359 = fminf(kbt[9], kbt[13]);
  kbt[9] = hi_358;
  kbt[13] = lo_359;
  float hi_360 = fmaxf(kbt[10], kbt[14]);
  float lo_361 = fminf(kbt[10], kbt[14]);
  kbt[10] = hi_360;
  kbt[14] = lo_361;
  float hi_362 = fmaxf(kbt[11], kbt[15]);
  float lo_363 = fminf(kbt[11], kbt[15]);
  kbt[11] = hi_362;
  kbt[15] = lo_363;
  float hi_364 = fmaxf(kbt[0], kbt[2]);
  float lo_365 = fminf(kbt[0], kbt[2]);
  kbt[0] = hi_364;
  kbt[2] = lo_365;
  float hi_366 = fmaxf(kbt[1], kbt[3]);
  float lo_367 = fminf(kbt[1], kbt[3]);
  kbt[1] = hi_366;
  kbt[3] = lo_367;
  float hi_368 = fmaxf(kbt[4], kbt[6]);
  float lo_369 = fminf(kbt[4], kbt[6]);
  kbt[4] = hi_368;
  kbt[6] = lo_369;
  float hi_370 = fmaxf(kbt[5], kbt[7]);
  float lo_371 = fminf(kbt[5], kbt[7]);
  kbt[5] = hi_370;
  kbt[7] = lo_371;
  float hi_372 = fmaxf(kbt[8], kbt[10]);
  float lo_373 = fminf(kbt[8], kbt[10]);
  kbt[8] = hi_372;
  kbt[10] = lo_373;
  float hi_374 = fmaxf(kbt[9], kbt[11]);
  float lo_375 = fminf(kbt[9], kbt[11]);
  kbt[9] = hi_374;
  kbt[11] = lo_375;
  float hi_376 = fmaxf(kbt[12], kbt[14]);
  float lo_377 = fminf(kbt[12], kbt[14]);
  kbt[12] = hi_376;
  kbt[14] = lo_377;
  float hi_378 = fmaxf(kbt[13], kbt[15]);
  float lo_379 = fminf(kbt[13], kbt[15]);
  kbt[13] = hi_378;
  kbt[15] = lo_379;
  float hi_380 = fmaxf(kbt[0], kbt[1]);
  float lo_381 = fminf(kbt[0], kbt[1]);
  kbt[0] = hi_380;
  kbt[1] = lo_381;
  float hi_382 = fmaxf(kbt[2], kbt[3]);
  float lo_383 = fminf(kbt[2], kbt[3]);
  kbt[2] = hi_382;
  kbt[3] = lo_383;
  float hi_384 = fmaxf(kbt[4], kbt[5]);
  float lo_385 = fminf(kbt[4], kbt[5]);
  kbt[4] = hi_384;
  kbt[5] = lo_385;
  float hi_386 = fmaxf(kbt[6], kbt[7]);
  float lo_387 = fminf(kbt[6], kbt[7]);
  kbt[6] = hi_386;
  kbt[7] = lo_387;
  float hi_388 = fmaxf(kbt[8], kbt[9]);
  float lo_389 = fminf(kbt[8], kbt[9]);
  kbt[8] = hi_388;
  kbt[9] = lo_389;
  float hi_390 = fmaxf(kbt[10], kbt[11]);
  float lo_391 = fminf(kbt[10], kbt[11]);
  kbt[10] = hi_390;
  kbt[11] = lo_391;
  float hi_392 = fmaxf(kbt[12], kbt[13]);
  float lo_393 = fminf(kbt[12], kbt[13]);
  kbt[12] = hi_392;
  kbt[13] = lo_393;
  float hi_394 = fmaxf(kbt[14], kbt[15]);
  float lo_395 = fminf(kbt[14], kbt[15]);
  kbt[14] = hi_394;
  kbt[15] = lo_395;
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
  int s0 = (sg * 16 * 16 + cg) * 17;
  int s1 = ((sg * 16 + 8) * 16 + cg) * 17;
  float x0 = pub[s0 + ln];
  float y0 = pub[s1 + lnr];
  float lo0 = fminf(x0, y0);
  r = fmaxf(r, lo0);
  float hi0 = fmaxf(x0, y0);
  float cur = hi0;
  float pv = __shfl_xor_sync(0xFFFFFFFF, cur, 8);
  float hi_396 = fmaxf(cur, pv);
  float lo_397 = fminf(cur, pv);
  cur = ((up[0] != 0) ? hi_396 : lo_397);
  float pv_398 = __shfl_xor_sync(0xFFFFFFFF, cur, 4);
  float hi_399 = fmaxf(cur, pv_398);
  float lo_400 = fminf(cur, pv_398);
  cur = ((up[1] != 0) ? hi_399 : lo_400);
  float pv_401 = __shfl_xor_sync(0xFFFFFFFF, cur, 2);
  float hi_402 = fmaxf(cur, pv_401);
  float lo_403 = fminf(cur, pv_401);
  cur = ((up[2] != 0) ? hi_402 : lo_403);
  float pv_404 = __shfl_xor_sync(0xFFFFFFFF, cur, 1);
  float hi_405 = fmaxf(cur, pv_404);
  float lo_406 = fminf(cur, pv_404);
  cur = ((up[3] != 0) ? hi_405 : lo_406);
  V[0] = cur;
  int s0_407 = ((sg * 16 + 1) * 16 + cg) * 17;
  int s1_408 = ((sg * 16 + 1 + 8) * 16 + cg) * 17;
  float x0_409 = pub[s0_407 + ln];
  float y0_410 = pub[s1_408 + lnr];
  float lo0_411 = fminf(x0_409, y0_410);
  r = fmaxf(r, lo0_411);
  float hi0_412 = fmaxf(x0_409, y0_410);
  float cur_413 = hi0_412;
  float pv_414 = __shfl_xor_sync(0xFFFFFFFF, cur_413, 8);
  float hi_415 = fmaxf(cur_413, pv_414);
  float lo_416 = fminf(cur_413, pv_414);
  cur_413 = ((up[0] != 0) ? hi_415 : lo_416);
  float pv_417 = __shfl_xor_sync(0xFFFFFFFF, cur_413, 4);
  float hi_418 = fmaxf(cur_413, pv_417);
  float lo_419 = fminf(cur_413, pv_417);
  cur_413 = ((up[1] != 0) ? hi_418 : lo_419);
  float pv_420 = __shfl_xor_sync(0xFFFFFFFF, cur_413, 2);
  float hi_421 = fmaxf(cur_413, pv_420);
  float lo_422 = fminf(cur_413, pv_420);
  cur_413 = ((up[2] != 0) ? hi_421 : lo_422);
  float pv_423 = __shfl_xor_sync(0xFFFFFFFF, cur_413, 1);
  float hi_424 = fmaxf(cur_413, pv_423);
  float lo_425 = fminf(cur_413, pv_423);
  cur_413 = ((up[3] != 0) ? hi_424 : lo_425);
  V[1] = cur_413;
  int s0_426 = ((sg * 16 + 2) * 16 + cg) * 17;
  int s1_427 = ((sg * 16 + 2 + 8) * 16 + cg) * 17;
  float x0_428 = pub[s0_426 + ln];
  float y0_429 = pub[s1_427 + lnr];
  float lo0_430 = fminf(x0_428, y0_429);
  r = fmaxf(r, lo0_430);
  float hi0_431 = fmaxf(x0_428, y0_429);
  float cur_432 = hi0_431;
  float pv_433 = __shfl_xor_sync(0xFFFFFFFF, cur_432, 8);
  float hi_434 = fmaxf(cur_432, pv_433);
  float lo_435 = fminf(cur_432, pv_433);
  cur_432 = ((up[0] != 0) ? hi_434 : lo_435);
  float pv_436 = __shfl_xor_sync(0xFFFFFFFF, cur_432, 4);
  float hi_437 = fmaxf(cur_432, pv_436);
  float lo_438 = fminf(cur_432, pv_436);
  cur_432 = ((up[1] != 0) ? hi_437 : lo_438);
  float pv_439 = __shfl_xor_sync(0xFFFFFFFF, cur_432, 2);
  float hi_440 = fmaxf(cur_432, pv_439);
  float lo_441 = fminf(cur_432, pv_439);
  cur_432 = ((up[2] != 0) ? hi_440 : lo_441);
  float pv_442 = __shfl_xor_sync(0xFFFFFFFF, cur_432, 1);
  float hi_443 = fmaxf(cur_432, pv_442);
  float lo_444 = fminf(cur_432, pv_442);
  cur_432 = ((up[3] != 0) ? hi_443 : lo_444);
  V[2] = cur_432;
  int s0_445 = ((sg * 16 + 3) * 16 + cg) * 17;
  int s1_446 = ((sg * 16 + 3 + 8) * 16 + cg) * 17;
  float x0_447 = pub[s0_445 + ln];
  float y0_448 = pub[s1_446 + lnr];
  float lo0_449 = fminf(x0_447, y0_448);
  r = fmaxf(r, lo0_449);
  float hi0_450 = fmaxf(x0_447, y0_448);
  float cur_451 = hi0_450;
  float pv_452 = __shfl_xor_sync(0xFFFFFFFF, cur_451, 8);
  float hi_453 = fmaxf(cur_451, pv_452);
  float lo_454 = fminf(cur_451, pv_452);
  cur_451 = ((up[0] != 0) ? hi_453 : lo_454);
  float pv_455 = __shfl_xor_sync(0xFFFFFFFF, cur_451, 4);
  float hi_456 = fmaxf(cur_451, pv_455);
  float lo_457 = fminf(cur_451, pv_455);
  cur_451 = ((up[1] != 0) ? hi_456 : lo_457);
  float pv_458 = __shfl_xor_sync(0xFFFFFFFF, cur_451, 2);
  float hi_459 = fmaxf(cur_451, pv_458);
  float lo_460 = fminf(cur_451, pv_458);
  cur_451 = ((up[2] != 0) ? hi_459 : lo_460);
  float pv_461 = __shfl_xor_sync(0xFFFFFFFF, cur_451, 1);
  float hi_462 = fmaxf(cur_451, pv_461);
  float lo_463 = fminf(cur_451, pv_461);
  cur_451 = ((up[3] != 0) ? hi_462 : lo_463);
  V[3] = cur_451;
  int s0_464 = ((sg * 16 + 4) * 16 + cg) * 17;
  int s1_465 = ((sg * 16 + 4 + 8) * 16 + cg) * 17;
  float x0_466 = pub[s0_464 + ln];
  float y0_467 = pub[s1_465 + lnr];
  float lo0_468 = fminf(x0_466, y0_467);
  r = fmaxf(r, lo0_468);
  float hi0_469 = fmaxf(x0_466, y0_467);
  float cur_470 = hi0_469;
  float pv_471 = __shfl_xor_sync(0xFFFFFFFF, cur_470, 8);
  float hi_472 = fmaxf(cur_470, pv_471);
  float lo_473 = fminf(cur_470, pv_471);
  cur_470 = ((up[0] != 0) ? hi_472 : lo_473);
  float pv_474 = __shfl_xor_sync(0xFFFFFFFF, cur_470, 4);
  float hi_475 = fmaxf(cur_470, pv_474);
  float lo_476 = fminf(cur_470, pv_474);
  cur_470 = ((up[1] != 0) ? hi_475 : lo_476);
  float pv_477 = __shfl_xor_sync(0xFFFFFFFF, cur_470, 2);
  float hi_478 = fmaxf(cur_470, pv_477);
  float lo_479 = fminf(cur_470, pv_477);
  cur_470 = ((up[2] != 0) ? hi_478 : lo_479);
  float pv_480 = __shfl_xor_sync(0xFFFFFFFF, cur_470, 1);
  float hi_481 = fmaxf(cur_470, pv_480);
  float lo_482 = fminf(cur_470, pv_480);
  cur_470 = ((up[3] != 0) ? hi_481 : lo_482);
  V[4] = cur_470;
  int s0_483 = ((sg * 16 + 5) * 16 + cg) * 17;
  int s1_484 = ((sg * 16 + 5 + 8) * 16 + cg) * 17;
  float x0_485 = pub[s0_483 + ln];
  float y0_486 = pub[s1_484 + lnr];
  float lo0_487 = fminf(x0_485, y0_486);
  r = fmaxf(r, lo0_487);
  float hi0_488 = fmaxf(x0_485, y0_486);
  float cur_489 = hi0_488;
  float pv_490 = __shfl_xor_sync(0xFFFFFFFF, cur_489, 8);
  float hi_491 = fmaxf(cur_489, pv_490);
  float lo_492 = fminf(cur_489, pv_490);
  cur_489 = ((up[0] != 0) ? hi_491 : lo_492);
  float pv_493 = __shfl_xor_sync(0xFFFFFFFF, cur_489, 4);
  float hi_494 = fmaxf(cur_489, pv_493);
  float lo_495 = fminf(cur_489, pv_493);
  cur_489 = ((up[1] != 0) ? hi_494 : lo_495);
  float pv_496 = __shfl_xor_sync(0xFFFFFFFF, cur_489, 2);
  float hi_497 = fmaxf(cur_489, pv_496);
  float lo_498 = fminf(cur_489, pv_496);
  cur_489 = ((up[2] != 0) ? hi_497 : lo_498);
  float pv_499 = __shfl_xor_sync(0xFFFFFFFF, cur_489, 1);
  float hi_500 = fmaxf(cur_489, pv_499);
  float lo_501 = fminf(cur_489, pv_499);
  cur_489 = ((up[3] != 0) ? hi_500 : lo_501);
  V[5] = cur_489;
  int s0_502 = ((sg * 16 + 6) * 16 + cg) * 17;
  int s1_503 = ((sg * 16 + 6 + 8) * 16 + cg) * 17;
  float x0_504 = pub[s0_502 + ln];
  float y0_505 = pub[s1_503 + lnr];
  float lo0_506 = fminf(x0_504, y0_505);
  r = fmaxf(r, lo0_506);
  float hi0_507 = fmaxf(x0_504, y0_505);
  float cur_508 = hi0_507;
  float pv_509 = __shfl_xor_sync(0xFFFFFFFF, cur_508, 8);
  float hi_510 = fmaxf(cur_508, pv_509);
  float lo_511 = fminf(cur_508, pv_509);
  cur_508 = ((up[0] != 0) ? hi_510 : lo_511);
  float pv_512 = __shfl_xor_sync(0xFFFFFFFF, cur_508, 4);
  float hi_513 = fmaxf(cur_508, pv_512);
  float lo_514 = fminf(cur_508, pv_512);
  cur_508 = ((up[1] != 0) ? hi_513 : lo_514);
  float pv_515 = __shfl_xor_sync(0xFFFFFFFF, cur_508, 2);
  float hi_516 = fmaxf(cur_508, pv_515);
  float lo_517 = fminf(cur_508, pv_515);
  cur_508 = ((up[2] != 0) ? hi_516 : lo_517);
  float pv_518 = __shfl_xor_sync(0xFFFFFFFF, cur_508, 1);
  float hi_519 = fmaxf(cur_508, pv_518);
  float lo_520 = fminf(cur_508, pv_518);
  cur_508 = ((up[3] != 0) ? hi_519 : lo_520);
  V[6] = cur_508;
  int s0_521 = ((sg * 16 + 7) * 16 + cg) * 17;
  int s1_522 = ((sg * 16 + 7 + 8) * 16 + cg) * 17;
  float x0_523 = pub[s0_521 + ln];
  float y0_524 = pub[s1_522 + lnr];
  float lo0_525 = fminf(x0_523, y0_524);
  r = fmaxf(r, lo0_525);
  float hi0_526 = fmaxf(x0_523, y0_524);
  float cur_527 = hi0_526;
  float pv_528 = __shfl_xor_sync(0xFFFFFFFF, cur_527, 8);
  float hi_529 = fmaxf(cur_527, pv_528);
  float lo_530 = fminf(cur_527, pv_528);
  cur_527 = ((up[0] != 0) ? hi_529 : lo_530);
  float pv_531 = __shfl_xor_sync(0xFFFFFFFF, cur_527, 4);
  float hi_532 = fmaxf(cur_527, pv_531);
  float lo_533 = fminf(cur_527, pv_531);
  cur_527 = ((up[1] != 0) ? hi_532 : lo_533);
  float pv_534 = __shfl_xor_sync(0xFFFFFFFF, cur_527, 2);
  float hi_535 = fmaxf(cur_527, pv_534);
  float lo_536 = fminf(cur_527, pv_534);
  cur_527 = ((up[2] != 0) ? hi_535 : lo_536);
  float pv_537 = __shfl_xor_sync(0xFFFFFFFF, cur_527, 1);
  float hi_538 = fmaxf(cur_527, pv_537);
  float lo_539 = fminf(cur_527, pv_537);
  cur_527 = ((up[3] != 0) ? hi_538 : lo_539);
  V[7] = cur_527;
  float rs = pub[((sg * 16 + ln) * 16 + cg) * 17 + 16];
  r = fmaxf(r, rs);
  float y1 = __shfl_xor_sync(0xFFFFFFFF, V[4], 15);
  float lo1 = fminf(V[0], y1);
  r = fmaxf(r, lo1);
  float hi1 = fmaxf(V[0], y1);
  float cur_540 = hi1;
  float pv_541 = __shfl_xor_sync(0xFFFFFFFF, cur_540, 8);
  float hi_542 = fmaxf(cur_540, pv_541);
  float lo_543 = fminf(cur_540, pv_541);
  cur_540 = ((up[0] != 0) ? hi_542 : lo_543);
  float pv_544 = __shfl_xor_sync(0xFFFFFFFF, cur_540, 4);
  float hi_545 = fmaxf(cur_540, pv_544);
  float lo_546 = fminf(cur_540, pv_544);
  cur_540 = ((up[1] != 0) ? hi_545 : lo_546);
  float pv_547 = __shfl_xor_sync(0xFFFFFFFF, cur_540, 2);
  float hi_548 = fmaxf(cur_540, pv_547);
  float lo_549 = fminf(cur_540, pv_547);
  cur_540 = ((up[2] != 0) ? hi_548 : lo_549);
  float pv_550 = __shfl_xor_sync(0xFFFFFFFF, cur_540, 1);
  float hi_551 = fmaxf(cur_540, pv_550);
  float lo_552 = fminf(cur_540, pv_550);
  cur_540 = ((up[3] != 0) ? hi_551 : lo_552);
  V[0] = cur_540;
  float y1_553 = __shfl_xor_sync(0xFFFFFFFF, V[5], 15);
  float lo1_554 = fminf(V[1], y1_553);
  r = fmaxf(r, lo1_554);
  float hi1_555 = fmaxf(V[1], y1_553);
  float cur_556 = hi1_555;
  float pv_557 = __shfl_xor_sync(0xFFFFFFFF, cur_556, 8);
  float hi_558 = fmaxf(cur_556, pv_557);
  float lo_559 = fminf(cur_556, pv_557);
  cur_556 = ((up[0] != 0) ? hi_558 : lo_559);
  float pv_560 = __shfl_xor_sync(0xFFFFFFFF, cur_556, 4);
  float hi_561 = fmaxf(cur_556, pv_560);
  float lo_562 = fminf(cur_556, pv_560);
  cur_556 = ((up[1] != 0) ? hi_561 : lo_562);
  float pv_563 = __shfl_xor_sync(0xFFFFFFFF, cur_556, 2);
  float hi_564 = fmaxf(cur_556, pv_563);
  float lo_565 = fminf(cur_556, pv_563);
  cur_556 = ((up[2] != 0) ? hi_564 : lo_565);
  float pv_566 = __shfl_xor_sync(0xFFFFFFFF, cur_556, 1);
  float hi_567 = fmaxf(cur_556, pv_566);
  float lo_568 = fminf(cur_556, pv_566);
  cur_556 = ((up[3] != 0) ? hi_567 : lo_568);
  V[1] = cur_556;
  float y1_569 = __shfl_xor_sync(0xFFFFFFFF, V[6], 15);
  float lo1_570 = fminf(V[2], y1_569);
  r = fmaxf(r, lo1_570);
  float hi1_571 = fmaxf(V[2], y1_569);
  float cur_572 = hi1_571;
  float pv_573 = __shfl_xor_sync(0xFFFFFFFF, cur_572, 8);
  float hi_574 = fmaxf(cur_572, pv_573);
  float lo_575 = fminf(cur_572, pv_573);
  cur_572 = ((up[0] != 0) ? hi_574 : lo_575);
  float pv_576 = __shfl_xor_sync(0xFFFFFFFF, cur_572, 4);
  float hi_577 = fmaxf(cur_572, pv_576);
  float lo_578 = fminf(cur_572, pv_576);
  cur_572 = ((up[1] != 0) ? hi_577 : lo_578);
  float pv_579 = __shfl_xor_sync(0xFFFFFFFF, cur_572, 2);
  float hi_580 = fmaxf(cur_572, pv_579);
  float lo_581 = fminf(cur_572, pv_579);
  cur_572 = ((up[2] != 0) ? hi_580 : lo_581);
  float pv_582 = __shfl_xor_sync(0xFFFFFFFF, cur_572, 1);
  float hi_583 = fmaxf(cur_572, pv_582);
  float lo_584 = fminf(cur_572, pv_582);
  cur_572 = ((up[3] != 0) ? hi_583 : lo_584);
  V[2] = cur_572;
  float y1_585 = __shfl_xor_sync(0xFFFFFFFF, V[7], 15);
  float lo1_586 = fminf(V[3], y1_585);
  r = fmaxf(r, lo1_586);
  float hi1_587 = fmaxf(V[3], y1_585);
  float cur_588 = hi1_587;
  float pv_589 = __shfl_xor_sync(0xFFFFFFFF, cur_588, 8);
  float hi_590 = fmaxf(cur_588, pv_589);
  float lo_591 = fminf(cur_588, pv_589);
  cur_588 = ((up[0] != 0) ? hi_590 : lo_591);
  float pv_592 = __shfl_xor_sync(0xFFFFFFFF, cur_588, 4);
  float hi_593 = fmaxf(cur_588, pv_592);
  float lo_594 = fminf(cur_588, pv_592);
  cur_588 = ((up[1] != 0) ? hi_593 : lo_594);
  float pv_595 = __shfl_xor_sync(0xFFFFFFFF, cur_588, 2);
  float hi_596 = fmaxf(cur_588, pv_595);
  float lo_597 = fminf(cur_588, pv_595);
  cur_588 = ((up[2] != 0) ? hi_596 : lo_597);
  float pv_598 = __shfl_xor_sync(0xFFFFFFFF, cur_588, 1);
  float hi_599 = fmaxf(cur_588, pv_598);
  float lo_600 = fminf(cur_588, pv_598);
  cur_588 = ((up[3] != 0) ? hi_599 : lo_600);
  V[3] = cur_588;
  float y1_601 = __shfl_xor_sync(0xFFFFFFFF, V[2], 15);
  float lo1_602 = fminf(V[0], y1_601);
  r = fmaxf(r, lo1_602);
  float hi1_603 = fmaxf(V[0], y1_601);
  float cur_604 = hi1_603;
  float pv_605 = __shfl_xor_sync(0xFFFFFFFF, cur_604, 8);
  float hi_606 = fmaxf(cur_604, pv_605);
  float lo_607 = fminf(cur_604, pv_605);
  cur_604 = ((up[0] != 0) ? hi_606 : lo_607);
  float pv_608 = __shfl_xor_sync(0xFFFFFFFF, cur_604, 4);
  float hi_609 = fmaxf(cur_604, pv_608);
  float lo_610 = fminf(cur_604, pv_608);
  cur_604 = ((up[1] != 0) ? hi_609 : lo_610);
  float pv_611 = __shfl_xor_sync(0xFFFFFFFF, cur_604, 2);
  float hi_612 = fmaxf(cur_604, pv_611);
  float lo_613 = fminf(cur_604, pv_611);
  cur_604 = ((up[2] != 0) ? hi_612 : lo_613);
  float pv_614 = __shfl_xor_sync(0xFFFFFFFF, cur_604, 1);
  float hi_615 = fmaxf(cur_604, pv_614);
  float lo_616 = fminf(cur_604, pv_614);
  cur_604 = ((up[3] != 0) ? hi_615 : lo_616);
  V[0] = cur_604;
  float y1_617 = __shfl_xor_sync(0xFFFFFFFF, V[3], 15);
  float lo1_618 = fminf(V[1], y1_617);
  r = fmaxf(r, lo1_618);
  float hi1_619 = fmaxf(V[1], y1_617);
  float cur_620 = hi1_619;
  float pv_621 = __shfl_xor_sync(0xFFFFFFFF, cur_620, 8);
  float hi_622 = fmaxf(cur_620, pv_621);
  float lo_623 = fminf(cur_620, pv_621);
  cur_620 = ((up[0] != 0) ? hi_622 : lo_623);
  float pv_624 = __shfl_xor_sync(0xFFFFFFFF, cur_620, 4);
  float hi_625 = fmaxf(cur_620, pv_624);
  float lo_626 = fminf(cur_620, pv_624);
  cur_620 = ((up[1] != 0) ? hi_625 : lo_626);
  float pv_627 = __shfl_xor_sync(0xFFFFFFFF, cur_620, 2);
  float hi_628 = fmaxf(cur_620, pv_627);
  float lo_629 = fminf(cur_620, pv_627);
  cur_620 = ((up[2] != 0) ? hi_628 : lo_629);
  float pv_630 = __shfl_xor_sync(0xFFFFFFFF, cur_620, 1);
  float hi_631 = fmaxf(cur_620, pv_630);
  float lo_632 = fminf(cur_620, pv_630);
  cur_620 = ((up[3] != 0) ? hi_631 : lo_632);
  V[1] = cur_620;
  float yl = __shfl_xor_sync(0xFFFFFFFF, V[1], 15);
  float lol = fminf(V[0], yl);
  r = fmaxf(r, lol);
  float hil = fmaxf(V[0], yl);
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
      float sc_1 = __uint_as_float(cbt[0]);
      sc_1 = fmaxf(sc_1, -1.7014118346046923e+38f);
      sc_1 = fminf(sc_1, 1.7014118346046923e+38f);
      unsigned int cls9 = __as_u32(sc_1) & 4294966784u;
      if (cls9 == qc) {
        mm = mm | 1;
      }
      float sc_3 = __uint_as_float(cbt[1]);
      sc_3 = fmaxf(sc_3, -1.7014118346046923e+38f);
      sc_3 = fminf(sc_3, 1.7014118346046923e+38f);
      unsigned int cls9_4 = __as_u32(sc_3) & 4294966784u;
      if (cls9_4 == qc) {
        mm = mm | 2;
      }
      float sc_5 = __uint_as_float(cbt[2]);
      sc_5 = fmaxf(sc_5, -1.7014118346046923e+38f);
      sc_5 = fminf(sc_5, 1.7014118346046923e+38f);
      unsigned int cls9_6 = __as_u32(sc_5) & 4294966784u;
      if (cls9_6 == qc) {
        mm = mm | 4;
      }
      float sc_7 = __uint_as_float(cbt[3]);
      sc_7 = fmaxf(sc_7, -1.7014118346046923e+38f);
      sc_7 = fminf(sc_7, 1.7014118346046923e+38f);
      unsigned int cls9_8 = __as_u32(sc_7) & 4294966784u;
      if (cls9_8 == qc) {
        mm = mm | 8;
      }
      float sc_9 = __uint_as_float(cbt[4]);
      sc_9 = fmaxf(sc_9, -1.7014118346046923e+38f);
      sc_9 = fminf(sc_9, 1.7014118346046923e+38f);
      unsigned int cls9_10 = __as_u32(sc_9) & 4294966784u;
      if (cls9_10 == qc) {
        mm = mm | 16;
      }
      float sc_11 = __uint_as_float(cbt[5]);
      sc_11 = fmaxf(sc_11, -1.7014118346046923e+38f);
      sc_11 = fminf(sc_11, 1.7014118346046923e+38f);
      unsigned int cls9_12 = __as_u32(sc_11) & 4294966784u;
      if (cls9_12 == qc) {
        mm = mm | 32;
      }
      float sc_13 = __uint_as_float(cbt[6]);
      sc_13 = fmaxf(sc_13, -1.7014118346046923e+38f);
      sc_13 = fminf(sc_13, 1.7014118346046923e+38f);
      unsigned int cls9_14 = __as_u32(sc_13) & 4294966784u;
      if (cls9_14 == qc) {
        mm = mm | 64;
      }
      float sc_15 = __uint_as_float(cbt[7]);
      sc_15 = fmaxf(sc_15, -1.7014118346046923e+38f);
      sc_15 = fminf(sc_15, 1.7014118346046923e+38f);
      unsigned int cls9_16 = __as_u32(sc_15) & 4294966784u;
      if (cls9_16 == qc) {
        mm = mm | 128;
      }
      float sc_17 = __uint_as_float(cbt[8]);
      sc_17 = fmaxf(sc_17, -1.7014118346046923e+38f);
      sc_17 = fminf(sc_17, 1.7014118346046923e+38f);
      unsigned int cls9_18 = __as_u32(sc_17) & 4294966784u;
      if (cls9_18 == qc) {
        mm = mm | 256;
      }
      float sc_19 = __uint_as_float(cbt[9]);
      sc_19 = fmaxf(sc_19, -1.7014118346046923e+38f);
      sc_19 = fminf(sc_19, 1.7014118346046923e+38f);
      unsigned int cls9_20 = __as_u32(sc_19) & 4294966784u;
      if (cls9_20 == qc) {
        mm = mm | 512;
      }
      float sc_21 = __uint_as_float(cbt[10]);
      sc_21 = fmaxf(sc_21, -1.7014118346046923e+38f);
      sc_21 = fminf(sc_21, 1.7014118346046923e+38f);
      unsigned int cls9_22 = __as_u32(sc_21) & 4294966784u;
      if (cls9_22 == qc) {
        mm = mm | 1024;
      }
      float sc_23 = __uint_as_float(cbt[11]);
      sc_23 = fmaxf(sc_23, -1.7014118346046923e+38f);
      sc_23 = fminf(sc_23, 1.7014118346046923e+38f);
      unsigned int cls9_24 = __as_u32(sc_23) & 4294966784u;
      if (cls9_24 == qc) {
        mm = mm | 2048;
      }
      float sc_25 = __uint_as_float(cbt[12]);
      sc_25 = fmaxf(sc_25, -1.7014118346046923e+38f);
      sc_25 = fminf(sc_25, 1.7014118346046923e+38f);
      unsigned int cls9_26 = __as_u32(sc_25) & 4294966784u;
      if (cls9_26 == qc) {
        mm = mm | 4096;
      }
      float sc_27 = __uint_as_float(cbt[13]);
      sc_27 = fmaxf(sc_27, -1.7014118346046923e+38f);
      sc_27 = fminf(sc_27, 1.7014118346046923e+38f);
      unsigned int cls9_28 = __as_u32(sc_27) & 4294966784u;
      if (cls9_28 == qc) {
        mm = mm | 8192;
      }
      float sc_29 = __uint_as_float(cbt[14]);
      sc_29 = fmaxf(sc_29, -1.7014118346046923e+38f);
      sc_29 = fminf(sc_29, 1.7014118346046923e+38f);
      unsigned int cls9_30 = __as_u32(sc_29) & 4294966784u;
      if (cls9_30 == qc) {
        mm = mm | 16384;
      }
      float sc_31 = __uint_as_float(cbt[15]);
      sc_31 = fmaxf(sc_31, -1.7014118346046923e+38f);
      sc_31 = fminf(sc_31, 1.7014118346046923e+38f);
      unsigned int cls9_32 = __as_u32(sc_31) & 4294966784u;
      if (cls9_32 == qc) {
        mm = mm | 32768;
      }
      float sc_33 = __uint_as_float(cbt[16]);
      sc_33 = fmaxf(sc_33, -1.7014118346046923e+38f);
      sc_33 = fminf(sc_33, 1.7014118346046923e+38f);
      unsigned int cls9_34 = __as_u32(sc_33) & 4294966784u;
      if (cls9_34 == qc) {
        mm = mm | 65536;
      }
      float sc_35 = __uint_as_float(cbt[17]);
      sc_35 = fmaxf(sc_35, -1.7014118346046923e+38f);
      sc_35 = fminf(sc_35, 1.7014118346046923e+38f);
      unsigned int cls9_36 = __as_u32(sc_35) & 4294966784u;
      if (cls9_36 == qc) {
        mm = mm | 131072;
      }
      float sc_37 = __uint_as_float(cbt[18]);
      sc_37 = fmaxf(sc_37, -1.7014118346046923e+38f);
      sc_37 = fminf(sc_37, 1.7014118346046923e+38f);
      unsigned int cls9_38 = __as_u32(sc_37) & 4294966784u;
      if (cls9_38 == qc) {
        mm = mm | 262144;
      }
      float sc_39 = __uint_as_float(cbt[19]);
      sc_39 = fmaxf(sc_39, -1.7014118346046923e+38f);
      sc_39 = fminf(sc_39, 1.7014118346046923e+38f);
      unsigned int cls9_40 = __as_u32(sc_39) & 4294966784u;
      if (cls9_40 == qc) {
        mm = mm | 524288;
      }
      float sc_41 = __uint_as_float(cbt[20]);
      sc_41 = fmaxf(sc_41, -1.7014118346046923e+38f);
      sc_41 = fminf(sc_41, 1.7014118346046923e+38f);
      unsigned int cls9_42 = __as_u32(sc_41) & 4294966784u;
      if (cls9_42 == qc) {
        mm = mm | 1048576;
      }
      float sc_43 = __uint_as_float(cbt[21]);
      sc_43 = fmaxf(sc_43, -1.7014118346046923e+38f);
      sc_43 = fminf(sc_43, 1.7014118346046923e+38f);
      unsigned int cls9_44 = __as_u32(sc_43) & 4294966784u;
      if (cls9_44 == qc) {
        mm = mm | 2097152;
      }
      float sc_45 = __uint_as_float(cbt[22]);
      sc_45 = fmaxf(sc_45, -1.7014118346046923e+38f);
      sc_45 = fminf(sc_45, 1.7014118346046923e+38f);
      unsigned int cls9_46 = __as_u32(sc_45) & 4294966784u;
      if (cls9_46 == qc) {
        mm = mm | 4194304;
      }
      float sc_47 = __uint_as_float(cbt[23]);
      sc_47 = fmaxf(sc_47, -1.7014118346046923e+38f);
      sc_47 = fminf(sc_47, 1.7014118346046923e+38f);
      unsigned int cls9_48 = __as_u32(sc_47) & 4294966784u;
      if (cls9_48 == qc) {
        mm = mm | 8388608;
      }
      float sc_49 = __uint_as_float(cbt[24]);
      sc_49 = fmaxf(sc_49, -1.7014118346046923e+38f);
      sc_49 = fminf(sc_49, 1.7014118346046923e+38f);
      unsigned int cls9_50 = __as_u32(sc_49) & 4294966784u;
      if (cls9_50 == qc) {
        mm = mm | 16777216;
      }
      float sc_51 = __uint_as_float(cbt[25]);
      sc_51 = fmaxf(sc_51, -1.7014118346046923e+38f);
      sc_51 = fminf(sc_51, 1.7014118346046923e+38f);
      unsigned int cls9_52 = __as_u32(sc_51) & 4294966784u;
      if (cls9_52 == qc) {
        mm = mm | 33554432;
      }
      float sc_53 = __uint_as_float(cbt[26]);
      sc_53 = fmaxf(sc_53, -1.7014118346046923e+38f);
      sc_53 = fminf(sc_53, 1.7014118346046923e+38f);
      unsigned int cls9_54 = __as_u32(sc_53) & 4294966784u;
      if (cls9_54 == qc) {
        mm = mm | 67108864;
      }
      float sc_55 = __uint_as_float(cbt[27]);
      sc_55 = fmaxf(sc_55, -1.7014118346046923e+38f);
      sc_55 = fminf(sc_55, 1.7014118346046923e+38f);
      unsigned int cls9_56 = __as_u32(sc_55) & 4294966784u;
      if (cls9_56 == qc) {
        mm = mm | 134217728;
      }
      float sc_57 = __uint_as_float(cbt[28]);
      sc_57 = fmaxf(sc_57, -1.7014118346046923e+38f);
      sc_57 = fminf(sc_57, 1.7014118346046923e+38f);
      unsigned int cls9_58 = __as_u32(sc_57) & 4294966784u;
      if (cls9_58 == qc) {
        mm = mm | 268435456;
      }
      float sc_59 = __uint_as_float(cbt[29]);
      sc_59 = fmaxf(sc_59, -1.7014118346046923e+38f);
      sc_59 = fminf(sc_59, 1.7014118346046923e+38f);
      unsigned int cls9_60 = __as_u32(sc_59) & 4294966784u;
      if (cls9_60 == qc) {
        mm = mm | 536870912;
      }
      float sc_61 = __uint_as_float(cbt[30]);
      sc_61 = fmaxf(sc_61, -1.7014118346046923e+38f);
      sc_61 = fminf(sc_61, 1.7014118346046923e+38f);
      unsigned int cls9_62 = __as_u32(sc_61) & 4294966784u;
      if (cls9_62 == qc) {
        mm = mm | 1073741824;
      }
      float sc_63 = __uint_as_float(cbt[31]);
      sc_63 = fmaxf(sc_63, -1.7014118346046923e+38f);
      sc_63 = fminf(sc_63, 1.7014118346046923e+38f);
      unsigned int cls9_64 = __as_u32(sc_63) & 4294966784u;
      if (cls9_64 == qc) {
        mm = mm | 2147483648u;
      }
      int f_65 = 0;
      if ((tb < fb || tb + 32 > lim - fe) && tb < lim) {
        f_65 = 1;
      }
      if (f_65 != 0) {
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
      if (mm != 0) {
        int n9 = (int)(unsigned int)__popc(mm);
        unsigned int base9 = atomicAdd(&ccnt[c], (unsigned int)n9);
        if (base9 + (unsigned int)n9 > 16) {
          flagw[1] = 1;
        } else {
          int pos9 = c * 16 + (int)base9;
          if ((mm & 1) != 0) {
            float sc_62 = __uint_as_float(cbt[0]);
            sc_62 = fmaxf(sc_62, -1.7014118346046923e+38f);
            sc_62 = fminf(sc_62, 1.7014118346046923e+38f);
            unsigned int u9 = __as_u32(sc_62);
            unsigned int lowb9 = (u9 ^ (unsigned int)((int)u9 >> 31) & 511) & 511;
            unsigned int km9 = 536870912 | lowb9 << 9 | (unsigned int)tb;
            cbuf[pos9] = km9;
            pos9 = pos9 + 1;
          }
          if ((mm & 2) != 0) {
            float sc_62_1 = __uint_as_float(cbt[1]);
            sc_62_1 = fmaxf(sc_62_1, -1.7014118346046923e+38f);
            sc_62_1 = fminf(sc_62_1, 1.7014118346046923e+38f);
            unsigned int u9_1 = __as_u32(sc_62_1);
            unsigned int lowb9_1 = (u9_1 ^ (unsigned int)((int)u9_1 >> 31) & 511) & 511;
            unsigned int km9_1 = 536870912 | lowb9_1 << 9 | (unsigned int)(tb + 1);
            cbuf[pos9] = km9_1;
            pos9 = pos9 + 1;
          }
          if ((mm & 4) != 0) {
            float sc_62_2 = __uint_as_float(cbt[2]);
            sc_62_2 = fmaxf(sc_62_2, -1.7014118346046923e+38f);
            sc_62_2 = fminf(sc_62_2, 1.7014118346046923e+38f);
            unsigned int u9_2 = __as_u32(sc_62_2);
            unsigned int lowb9_2 = (u9_2 ^ (unsigned int)((int)u9_2 >> 31) & 511) & 511;
            unsigned int km9_2 = 536870912 | lowb9_2 << 9 | (unsigned int)(tb + 2);
            cbuf[pos9] = km9_2;
            pos9 = pos9 + 1;
          }
          if ((mm & 8) != 0) {
            float sc_62_3 = __uint_as_float(cbt[3]);
            sc_62_3 = fmaxf(sc_62_3, -1.7014118346046923e+38f);
            sc_62_3 = fminf(sc_62_3, 1.7014118346046923e+38f);
            unsigned int u9_3 = __as_u32(sc_62_3);
            unsigned int lowb9_3 = (u9_3 ^ (unsigned int)((int)u9_3 >> 31) & 511) & 511;
            unsigned int km9_3 = 536870912 | lowb9_3 << 9 | (unsigned int)(tb + 3);
            cbuf[pos9] = km9_3;
            pos9 = pos9 + 1;
          }
          if ((mm & 16) != 0) {
            float sc_62_4 = __uint_as_float(cbt[4]);
            sc_62_4 = fmaxf(sc_62_4, -1.7014118346046923e+38f);
            sc_62_4 = fminf(sc_62_4, 1.7014118346046923e+38f);
            unsigned int u9_4 = __as_u32(sc_62_4);
            unsigned int lowb9_4 = (u9_4 ^ (unsigned int)((int)u9_4 >> 31) & 511) & 511;
            unsigned int km9_4 = 536870912 | lowb9_4 << 9 | (unsigned int)(tb + 4);
            cbuf[pos9] = km9_4;
            pos9 = pos9 + 1;
          }
          if ((mm & 32) != 0) {
            float sc_62_5 = __uint_as_float(cbt[5]);
            sc_62_5 = fmaxf(sc_62_5, -1.7014118346046923e+38f);
            sc_62_5 = fminf(sc_62_5, 1.7014118346046923e+38f);
            unsigned int u9_5 = __as_u32(sc_62_5);
            unsigned int lowb9_5 = (u9_5 ^ (unsigned int)((int)u9_5 >> 31) & 511) & 511;
            unsigned int km9_5 = 536870912 | lowb9_5 << 9 | (unsigned int)(tb + 5);
            cbuf[pos9] = km9_5;
            pos9 = pos9 + 1;
          }
          if ((mm & 64) != 0) {
            float sc_62_6 = __uint_as_float(cbt[6]);
            sc_62_6 = fmaxf(sc_62_6, -1.7014118346046923e+38f);
            sc_62_6 = fminf(sc_62_6, 1.7014118346046923e+38f);
            unsigned int u9_6 = __as_u32(sc_62_6);
            unsigned int lowb9_6 = (u9_6 ^ (unsigned int)((int)u9_6 >> 31) & 511) & 511;
            unsigned int km9_6 = 536870912 | lowb9_6 << 9 | (unsigned int)(tb + 6);
            cbuf[pos9] = km9_6;
            pos9 = pos9 + 1;
          }
          if ((mm & 128) != 0) {
            float sc_62_7 = __uint_as_float(cbt[7]);
            sc_62_7 = fmaxf(sc_62_7, -1.7014118346046923e+38f);
            sc_62_7 = fminf(sc_62_7, 1.7014118346046923e+38f);
            unsigned int u9_7 = __as_u32(sc_62_7);
            unsigned int lowb9_7 = (u9_7 ^ (unsigned int)((int)u9_7 >> 31) & 511) & 511;
            unsigned int km9_7 = 536870912 | lowb9_7 << 9 | (unsigned int)(tb + 7);
            cbuf[pos9] = km9_7;
            pos9 = pos9 + 1;
          }
          if ((mm & 256) != 0) {
            float sc_62_8 = __uint_as_float(cbt[8]);
            sc_62_8 = fmaxf(sc_62_8, -1.7014118346046923e+38f);
            sc_62_8 = fminf(sc_62_8, 1.7014118346046923e+38f);
            unsigned int u9_8 = __as_u32(sc_62_8);
            unsigned int lowb9_8 = (u9_8 ^ (unsigned int)((int)u9_8 >> 31) & 511) & 511;
            unsigned int km9_8 = 536870912 | lowb9_8 << 9 | (unsigned int)(tb + 8);
            cbuf[pos9] = km9_8;
            pos9 = pos9 + 1;
          }
          if ((mm & 512) != 0) {
            float sc_62_9 = __uint_as_float(cbt[9]);
            sc_62_9 = fmaxf(sc_62_9, -1.7014118346046923e+38f);
            sc_62_9 = fminf(sc_62_9, 1.7014118346046923e+38f);
            unsigned int u9_9 = __as_u32(sc_62_9);
            unsigned int lowb9_9 = (u9_9 ^ (unsigned int)((int)u9_9 >> 31) & 511) & 511;
            unsigned int km9_9 = 536870912 | lowb9_9 << 9 | (unsigned int)(tb + 9);
            cbuf[pos9] = km9_9;
            pos9 = pos9 + 1;
          }
          if ((mm & 1024) != 0) {
            float sc_62_10 = __uint_as_float(cbt[10]);
            sc_62_10 = fmaxf(sc_62_10, -1.7014118346046923e+38f);
            sc_62_10 = fminf(sc_62_10, 1.7014118346046923e+38f);
            unsigned int u9_10 = __as_u32(sc_62_10);
            unsigned int lowb9_10 = (u9_10 ^ (unsigned int)((int)u9_10 >> 31) & 511) & 511;
            unsigned int km9_10 = 536870912 | lowb9_10 << 9 | (unsigned int)(tb + 10);
            cbuf[pos9] = km9_10;
            pos9 = pos9 + 1;
          }
          if ((mm & 2048) != 0) {
            float sc_62_11 = __uint_as_float(cbt[11]);
            sc_62_11 = fmaxf(sc_62_11, -1.7014118346046923e+38f);
            sc_62_11 = fminf(sc_62_11, 1.7014118346046923e+38f);
            unsigned int u9_11 = __as_u32(sc_62_11);
            unsigned int lowb9_11 = (u9_11 ^ (unsigned int)((int)u9_11 >> 31) & 511) & 511;
            unsigned int km9_11 = 536870912 | lowb9_11 << 9 | (unsigned int)(tb + 11);
            cbuf[pos9] = km9_11;
            pos9 = pos9 + 1;
          }
          if ((mm & 4096) != 0) {
            float sc_62_12 = __uint_as_float(cbt[12]);
            sc_62_12 = fmaxf(sc_62_12, -1.7014118346046923e+38f);
            sc_62_12 = fminf(sc_62_12, 1.7014118346046923e+38f);
            unsigned int u9_12 = __as_u32(sc_62_12);
            unsigned int lowb9_12 = (u9_12 ^ (unsigned int)((int)u9_12 >> 31) & 511) & 511;
            unsigned int km9_12 = 536870912 | lowb9_12 << 9 | (unsigned int)(tb + 12);
            cbuf[pos9] = km9_12;
            pos9 = pos9 + 1;
          }
          if ((mm & 8192) != 0) {
            float sc_62_13 = __uint_as_float(cbt[13]);
            sc_62_13 = fmaxf(sc_62_13, -1.7014118346046923e+38f);
            sc_62_13 = fminf(sc_62_13, 1.7014118346046923e+38f);
            unsigned int u9_13 = __as_u32(sc_62_13);
            unsigned int lowb9_13 = (u9_13 ^ (unsigned int)((int)u9_13 >> 31) & 511) & 511;
            unsigned int km9_13 = 536870912 | lowb9_13 << 9 | (unsigned int)(tb + 13);
            cbuf[pos9] = km9_13;
            pos9 = pos9 + 1;
          }
          if ((mm & 16384) != 0) {
            float sc_62_14 = __uint_as_float(cbt[14]);
            sc_62_14 = fmaxf(sc_62_14, -1.7014118346046923e+38f);
            sc_62_14 = fminf(sc_62_14, 1.7014118346046923e+38f);
            unsigned int u9_14 = __as_u32(sc_62_14);
            unsigned int lowb9_14 = (u9_14 ^ (unsigned int)((int)u9_14 >> 31) & 511) & 511;
            unsigned int km9_14 = 536870912 | lowb9_14 << 9 | (unsigned int)(tb + 14);
            cbuf[pos9] = km9_14;
            pos9 = pos9 + 1;
          }
          if ((mm & 32768) != 0) {
            float sc_62_15 = __uint_as_float(cbt[15]);
            sc_62_15 = fmaxf(sc_62_15, -1.7014118346046923e+38f);
            sc_62_15 = fminf(sc_62_15, 1.7014118346046923e+38f);
            unsigned int u9_15 = __as_u32(sc_62_15);
            unsigned int lowb9_15 = (u9_15 ^ (unsigned int)((int)u9_15 >> 31) & 511) & 511;
            unsigned int km9_15 = 536870912 | lowb9_15 << 9 | (unsigned int)(tb + 15);
            cbuf[pos9] = km9_15;
            pos9 = pos9 + 1;
          }
          if ((mm & 65536) != 0) {
            float sc_62_16 = __uint_as_float(cbt[16]);
            sc_62_16 = fmaxf(sc_62_16, -1.7014118346046923e+38f);
            sc_62_16 = fminf(sc_62_16, 1.7014118346046923e+38f);
            unsigned int u9_16 = __as_u32(sc_62_16);
            unsigned int lowb9_16 = (u9_16 ^ (unsigned int)((int)u9_16 >> 31) & 511) & 511;
            unsigned int km9_16 = 536870912 | lowb9_16 << 9 | (unsigned int)(tb + 16);
            cbuf[pos9] = km9_16;
            pos9 = pos9 + 1;
          }
          if ((mm & 131072) != 0) {
            float sc_62_17 = __uint_as_float(cbt[17]);
            sc_62_17 = fmaxf(sc_62_17, -1.7014118346046923e+38f);
            sc_62_17 = fminf(sc_62_17, 1.7014118346046923e+38f);
            unsigned int u9_17 = __as_u32(sc_62_17);
            unsigned int lowb9_17 = (u9_17 ^ (unsigned int)((int)u9_17 >> 31) & 511) & 511;
            unsigned int km9_17 = 536870912 | lowb9_17 << 9 | (unsigned int)(tb + 17);
            cbuf[pos9] = km9_17;
            pos9 = pos9 + 1;
          }
          if ((mm & 262144) != 0) {
            float sc_62_18 = __uint_as_float(cbt[18]);
            sc_62_18 = fmaxf(sc_62_18, -1.7014118346046923e+38f);
            sc_62_18 = fminf(sc_62_18, 1.7014118346046923e+38f);
            unsigned int u9_18 = __as_u32(sc_62_18);
            unsigned int lowb9_18 = (u9_18 ^ (unsigned int)((int)u9_18 >> 31) & 511) & 511;
            unsigned int km9_18 = 536870912 | lowb9_18 << 9 | (unsigned int)(tb + 18);
            cbuf[pos9] = km9_18;
            pos9 = pos9 + 1;
          }
          if ((mm & 524288) != 0) {
            float sc_62_19 = __uint_as_float(cbt[19]);
            sc_62_19 = fmaxf(sc_62_19, -1.7014118346046923e+38f);
            sc_62_19 = fminf(sc_62_19, 1.7014118346046923e+38f);
            unsigned int u9_19 = __as_u32(sc_62_19);
            unsigned int lowb9_19 = (u9_19 ^ (unsigned int)((int)u9_19 >> 31) & 511) & 511;
            unsigned int km9_19 = 536870912 | lowb9_19 << 9 | (unsigned int)(tb + 19);
            cbuf[pos9] = km9_19;
            pos9 = pos9 + 1;
          }
          if ((mm & 1048576) != 0) {
            float sc_62_20 = __uint_as_float(cbt[20]);
            sc_62_20 = fmaxf(sc_62_20, -1.7014118346046923e+38f);
            sc_62_20 = fminf(sc_62_20, 1.7014118346046923e+38f);
            unsigned int u9_20 = __as_u32(sc_62_20);
            unsigned int lowb9_20 = (u9_20 ^ (unsigned int)((int)u9_20 >> 31) & 511) & 511;
            unsigned int km9_20 = 536870912 | lowb9_20 << 9 | (unsigned int)(tb + 20);
            cbuf[pos9] = km9_20;
            pos9 = pos9 + 1;
          }
          if ((mm & 2097152) != 0) {
            float sc_62_21 = __uint_as_float(cbt[21]);
            sc_62_21 = fmaxf(sc_62_21, -1.7014118346046923e+38f);
            sc_62_21 = fminf(sc_62_21, 1.7014118346046923e+38f);
            unsigned int u9_21 = __as_u32(sc_62_21);
            unsigned int lowb9_21 = (u9_21 ^ (unsigned int)((int)u9_21 >> 31) & 511) & 511;
            unsigned int km9_21 = 536870912 | lowb9_21 << 9 | (unsigned int)(tb + 21);
            cbuf[pos9] = km9_21;
            pos9 = pos9 + 1;
          }
          if ((mm & 4194304) != 0) {
            float sc_62_22 = __uint_as_float(cbt[22]);
            sc_62_22 = fmaxf(sc_62_22, -1.7014118346046923e+38f);
            sc_62_22 = fminf(sc_62_22, 1.7014118346046923e+38f);
            unsigned int u9_22 = __as_u32(sc_62_22);
            unsigned int lowb9_22 = (u9_22 ^ (unsigned int)((int)u9_22 >> 31) & 511) & 511;
            unsigned int km9_22 = 536870912 | lowb9_22 << 9 | (unsigned int)(tb + 22);
            cbuf[pos9] = km9_22;
            pos9 = pos9 + 1;
          }
          if ((mm & 8388608) != 0) {
            float sc_62_23 = __uint_as_float(cbt[23]);
            sc_62_23 = fmaxf(sc_62_23, -1.7014118346046923e+38f);
            sc_62_23 = fminf(sc_62_23, 1.7014118346046923e+38f);
            unsigned int u9_23 = __as_u32(sc_62_23);
            unsigned int lowb9_23 = (u9_23 ^ (unsigned int)((int)u9_23 >> 31) & 511) & 511;
            unsigned int km9_23 = 536870912 | lowb9_23 << 9 | (unsigned int)(tb + 23);
            cbuf[pos9] = km9_23;
            pos9 = pos9 + 1;
          }
          if ((mm & 16777216) != 0) {
            float sc_62_24 = __uint_as_float(cbt[24]);
            sc_62_24 = fmaxf(sc_62_24, -1.7014118346046923e+38f);
            sc_62_24 = fminf(sc_62_24, 1.7014118346046923e+38f);
            unsigned int u9_24 = __as_u32(sc_62_24);
            unsigned int lowb9_24 = (u9_24 ^ (unsigned int)((int)u9_24 >> 31) & 511) & 511;
            unsigned int km9_24 = 536870912 | lowb9_24 << 9 | (unsigned int)(tb + 24);
            cbuf[pos9] = km9_24;
            pos9 = pos9 + 1;
          }
          if ((mm & 33554432) != 0) {
            float sc_62_25 = __uint_as_float(cbt[25]);
            sc_62_25 = fmaxf(sc_62_25, -1.7014118346046923e+38f);
            sc_62_25 = fminf(sc_62_25, 1.7014118346046923e+38f);
            unsigned int u9_25 = __as_u32(sc_62_25);
            unsigned int lowb9_25 = (u9_25 ^ (unsigned int)((int)u9_25 >> 31) & 511) & 511;
            unsigned int km9_25 = 536870912 | lowb9_25 << 9 | (unsigned int)(tb + 25);
            cbuf[pos9] = km9_25;
            pos9 = pos9 + 1;
          }
          if ((mm & 67108864) != 0) {
            float sc_62_26 = __uint_as_float(cbt[26]);
            sc_62_26 = fmaxf(sc_62_26, -1.7014118346046923e+38f);
            sc_62_26 = fminf(sc_62_26, 1.7014118346046923e+38f);
            unsigned int u9_26 = __as_u32(sc_62_26);
            unsigned int lowb9_26 = (u9_26 ^ (unsigned int)((int)u9_26 >> 31) & 511) & 511;
            unsigned int km9_26 = 536870912 | lowb9_26 << 9 | (unsigned int)(tb + 26);
            cbuf[pos9] = km9_26;
            pos9 = pos9 + 1;
          }
          if ((mm & 134217728) != 0) {
            float sc_62_27 = __uint_as_float(cbt[27]);
            sc_62_27 = fmaxf(sc_62_27, -1.7014118346046923e+38f);
            sc_62_27 = fminf(sc_62_27, 1.7014118346046923e+38f);
            unsigned int u9_27 = __as_u32(sc_62_27);
            unsigned int lowb9_27 = (u9_27 ^ (unsigned int)((int)u9_27 >> 31) & 511) & 511;
            unsigned int km9_27 = 536870912 | lowb9_27 << 9 | (unsigned int)(tb + 27);
            cbuf[pos9] = km9_27;
            pos9 = pos9 + 1;
          }
          if ((mm & 268435456) != 0) {
            float sc_62_28 = __uint_as_float(cbt[28]);
            sc_62_28 = fmaxf(sc_62_28, -1.7014118346046923e+38f);
            sc_62_28 = fminf(sc_62_28, 1.7014118346046923e+38f);
            unsigned int u9_28 = __as_u32(sc_62_28);
            unsigned int lowb9_28 = (u9_28 ^ (unsigned int)((int)u9_28 >> 31) & 511) & 511;
            unsigned int km9_28 = 536870912 | lowb9_28 << 9 | (unsigned int)(tb + 28);
            cbuf[pos9] = km9_28;
            pos9 = pos9 + 1;
          }
          if ((mm & 536870912) != 0) {
            float sc_62_29 = __uint_as_float(cbt[29]);
            sc_62_29 = fmaxf(sc_62_29, -1.7014118346046923e+38f);
            sc_62_29 = fminf(sc_62_29, 1.7014118346046923e+38f);
            unsigned int u9_29 = __as_u32(sc_62_29);
            unsigned int lowb9_29 = (u9_29 ^ (unsigned int)((int)u9_29 >> 31) & 511) & 511;
            unsigned int km9_29 = 536870912 | lowb9_29 << 9 | (unsigned int)(tb + 29);
            cbuf[pos9] = km9_29;
            pos9 = pos9 + 1;
          }
          if ((mm & 1073741824) != 0) {
            float sc_62_30 = __uint_as_float(cbt[30]);
            sc_62_30 = fmaxf(sc_62_30, -1.7014118346046923e+38f);
            sc_62_30 = fminf(sc_62_30, 1.7014118346046923e+38f);
            unsigned int u9_30 = __as_u32(sc_62_30);
            unsigned int lowb9_30 = (u9_30 ^ (unsigned int)((int)u9_30 >> 31) & 511) & 511;
            unsigned int km9_30 = 536870912 | lowb9_30 << 9 | (unsigned int)(tb + 30);
            cbuf[pos9] = km9_30;
            pos9 = pos9 + 1;
          }
          if ((mm & 2147483648u) != 0) {
            float sc_62_31 = __uint_as_float(cbt[31]);
            sc_62_31 = fmaxf(sc_62_31, -1.7014118346046923e+38f);
            sc_62_31 = fminf(sc_62_31, 1.7014118346046923e+38f);
            unsigned int u9_31 = __as_u32(sc_62_31);
            unsigned int lowb9_31 = (u9_31 ^ (unsigned int)((int)u9_31 >> 31) & 511) & 511;
            unsigned int km9_31 = 536870912 | lowb9_31 << 9 | (unsigned int)(tb + 31);
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
        float qf = __uint_as_float(qc);
        float sc_1_1 = __uint_as_float(cbt[0]);
        sc_1_1 = fmaxf(sc_1_1, -1.7014118346046923e+38f);
        sc_1_1 = fminf(sc_1_1, 1.7014118346046923e+38f);
        unsigned int u = __as_u32(sc_1_1);
        unsigned int cls = u & 4294966784u;
        int f_2_2 = 0;
        if ((tb < fb || tb >= lim - fe) && lim > tb) {
          f_2_2 = 1;
        }
        if (f_2_2 != 0) {
          cls = 2139094528;
        }
        unsigned int key_4 = 0;
        if (qf < __uint_as_float(cls) && cls < 4278190080u) {
          key_4 = 1073741824 | (unsigned int)tb;
        }
        if (cls == qc) {
          unsigned int lowb = (u ^ (unsigned int)((int)u >> 31) & 511) & 511;
          key_4 = 536870912 | lowb << 9 | (unsigned int)tb;
        }
        kbt2[0] = __uint_as_float(key_4);
        float sc_5_1 = __uint_as_float(cbt[1]);
        sc_5_1 = fmaxf(sc_5_1, -1.7014118346046923e+38f);
        sc_5_1 = fminf(sc_5_1, 1.7014118346046923e+38f);
        unsigned int u_6 = __as_u32(sc_5_1);
        unsigned int cls_7 = u_6 & 4294966784u;
        int f_8_2 = 0;
        if ((tb + 1 < fb || tb + 1 >= lim - fe) && lim > tb + 1) {
          f_8_2 = 1;
        }
        if (f_8_2 != 0) {
          cls_7 = 2139094528;
        }
        unsigned int key_10 = 0;
        if (qf < __uint_as_float(cls_7) && cls_7 < 4278190080u) {
          key_10 = 1073741824 | (unsigned int)(tb + 1);
        }
        if (cls_7 == qc) {
          unsigned int lowb_1 = (u_6 ^ (unsigned int)((int)u_6 >> 31) & 511) & 511;
          key_10 = 536870912 | lowb_1 << 9 | (unsigned int)(tb + 1);
        }
        kbt2[1] = __uint_as_float(key_10);
        float sc_11_1 = __uint_as_float(cbt[2]);
        sc_11_1 = fmaxf(sc_11_1, -1.7014118346046923e+38f);
        sc_11_1 = fminf(sc_11_1, 1.7014118346046923e+38f);
        unsigned int u_12 = __as_u32(sc_11_1);
        unsigned int cls_13 = u_12 & 4294966784u;
        int f_14_2 = 0;
        if ((tb + 2 < fb || tb + 2 >= lim - fe) && lim > tb + 2) {
          f_14_2 = 1;
        }
        if (f_14_2 != 0) {
          cls_13 = 2139094528;
        }
        unsigned int key_16 = 0;
        if (qf < __uint_as_float(cls_13) && cls_13 < 4278190080u) {
          key_16 = 1073741824 | (unsigned int)(tb + 2);
        }
        if (cls_13 == qc) {
          unsigned int lowb_2 = (u_12 ^ (unsigned int)((int)u_12 >> 31) & 511) & 511;
          key_16 = 536870912 | lowb_2 << 9 | (unsigned int)(tb + 2);
        }
        kbt2[2] = __uint_as_float(key_16);
        float sc_17_1 = __uint_as_float(cbt[3]);
        sc_17_1 = fmaxf(sc_17_1, -1.7014118346046923e+38f);
        sc_17_1 = fminf(sc_17_1, 1.7014118346046923e+38f);
        unsigned int u_18 = __as_u32(sc_17_1);
        unsigned int cls_19 = u_18 & 4294966784u;
        int f_20_2 = 0;
        if ((tb + 3 < fb || tb + 3 >= lim - fe) && lim > tb + 3) {
          f_20_2 = 1;
        }
        if (f_20_2 != 0) {
          cls_19 = 2139094528;
        }
        unsigned int key_22 = 0;
        if (qf < __uint_as_float(cls_19) && cls_19 < 4278190080u) {
          key_22 = 1073741824 | (unsigned int)(tb + 3);
        }
        if (cls_19 == qc) {
          unsigned int lowb_3 = (u_18 ^ (unsigned int)((int)u_18 >> 31) & 511) & 511;
          key_22 = 536870912 | lowb_3 << 9 | (unsigned int)(tb + 3);
        }
        kbt2[3] = __uint_as_float(key_22);
        float sc_23_1 = __uint_as_float(cbt[4]);
        sc_23_1 = fmaxf(sc_23_1, -1.7014118346046923e+38f);
        sc_23_1 = fminf(sc_23_1, 1.7014118346046923e+38f);
        unsigned int u_24 = __as_u32(sc_23_1);
        unsigned int cls_25 = u_24 & 4294966784u;
        int f_26_2 = 0;
        if ((tb + 4 < fb || tb + 4 >= lim - fe) && lim > tb + 4) {
          f_26_2 = 1;
        }
        if (f_26_2 != 0) {
          cls_25 = 2139094528;
        }
        unsigned int key_28 = 0;
        if (qf < __uint_as_float(cls_25) && cls_25 < 4278190080u) {
          key_28 = 1073741824 | (unsigned int)(tb + 4);
        }
        if (cls_25 == qc) {
          unsigned int lowb_4 = (u_24 ^ (unsigned int)((int)u_24 >> 31) & 511) & 511;
          key_28 = 536870912 | lowb_4 << 9 | (unsigned int)(tb + 4);
        }
        kbt2[4] = __uint_as_float(key_28);
        float sc_29_1 = __uint_as_float(cbt[5]);
        sc_29_1 = fmaxf(sc_29_1, -1.7014118346046923e+38f);
        sc_29_1 = fminf(sc_29_1, 1.7014118346046923e+38f);
        unsigned int u_30 = __as_u32(sc_29_1);
        unsigned int cls_31 = u_30 & 4294966784u;
        int f_32 = 0;
        if ((tb + 5 < fb || tb + 5 >= lim - fe) && lim > tb + 5) {
          f_32 = 1;
        }
        if (f_32 != 0) {
          cls_31 = 2139094528;
        }
        unsigned int key_34 = 0;
        if (qf < __uint_as_float(cls_31) && cls_31 < 4278190080u) {
          key_34 = 1073741824 | (unsigned int)(tb + 5);
        }
        if (cls_31 == qc) {
          unsigned int lowb_5 = (u_30 ^ (unsigned int)((int)u_30 >> 31) & 511) & 511;
          key_34 = 536870912 | lowb_5 << 9 | (unsigned int)(tb + 5);
        }
        kbt2[5] = __uint_as_float(key_34);
        float sc_35_1 = __uint_as_float(cbt[6]);
        sc_35_1 = fmaxf(sc_35_1, -1.7014118346046923e+38f);
        sc_35_1 = fminf(sc_35_1, 1.7014118346046923e+38f);
        unsigned int u_36 = __as_u32(sc_35_1);
        unsigned int cls_37 = u_36 & 4294966784u;
        int f_38 = 0;
        if ((tb + 6 < fb || tb + 6 >= lim - fe) && lim > tb + 6) {
          f_38 = 1;
        }
        if (f_38 != 0) {
          cls_37 = 2139094528;
        }
        unsigned int key_40 = 0;
        if (qf < __uint_as_float(cls_37) && cls_37 < 4278190080u) {
          key_40 = 1073741824 | (unsigned int)(tb + 6);
        }
        if (cls_37 == qc) {
          unsigned int lowb_6 = (u_36 ^ (unsigned int)((int)u_36 >> 31) & 511) & 511;
          key_40 = 536870912 | lowb_6 << 9 | (unsigned int)(tb + 6);
        }
        kbt2[6] = __uint_as_float(key_40);
        float sc_41_1 = __uint_as_float(cbt[7]);
        sc_41_1 = fmaxf(sc_41_1, -1.7014118346046923e+38f);
        sc_41_1 = fminf(sc_41_1, 1.7014118346046923e+38f);
        unsigned int u_42 = __as_u32(sc_41_1);
        unsigned int cls_43 = u_42 & 4294966784u;
        int f_44 = 0;
        if ((tb + 7 < fb || tb + 7 >= lim - fe) && lim > tb + 7) {
          f_44 = 1;
        }
        if (f_44 != 0) {
          cls_43 = 2139094528;
        }
        unsigned int key_46 = 0;
        if (qf < __uint_as_float(cls_43) && cls_43 < 4278190080u) {
          key_46 = 1073741824 | (unsigned int)(tb + 7);
        }
        if (cls_43 == qc) {
          unsigned int lowb_7 = (u_42 ^ (unsigned int)((int)u_42 >> 31) & 511) & 511;
          key_46 = 536870912 | lowb_7 << 9 | (unsigned int)(tb + 7);
        }
        kbt2[7] = __uint_as_float(key_46);
        float sc_47_1 = __uint_as_float(cbt[8]);
        sc_47_1 = fmaxf(sc_47_1, -1.7014118346046923e+38f);
        sc_47_1 = fminf(sc_47_1, 1.7014118346046923e+38f);
        unsigned int u_48 = __as_u32(sc_47_1);
        unsigned int cls_49 = u_48 & 4294966784u;
        int f_50 = 0;
        if ((tb + 8 < fb || tb + 8 >= lim - fe) && lim > tb + 8) {
          f_50 = 1;
        }
        if (f_50 != 0) {
          cls_49 = 2139094528;
        }
        unsigned int key_52 = 0;
        if (qf < __uint_as_float(cls_49) && cls_49 < 4278190080u) {
          key_52 = 1073741824 | (unsigned int)(tb + 8);
        }
        if (cls_49 == qc) {
          unsigned int lowb_8 = (u_48 ^ (unsigned int)((int)u_48 >> 31) & 511) & 511;
          key_52 = 536870912 | lowb_8 << 9 | (unsigned int)(tb + 8);
        }
        kbt2[8] = __uint_as_float(key_52);
        float sc_53_1 = __uint_as_float(cbt[9]);
        sc_53_1 = fmaxf(sc_53_1, -1.7014118346046923e+38f);
        sc_53_1 = fminf(sc_53_1, 1.7014118346046923e+38f);
        unsigned int u_54 = __as_u32(sc_53_1);
        unsigned int cls_55 = u_54 & 4294966784u;
        int f_56 = 0;
        if ((tb + 9 < fb || tb + 9 >= lim - fe) && lim > tb + 9) {
          f_56 = 1;
        }
        if (f_56 != 0) {
          cls_55 = 2139094528;
        }
        unsigned int key_58 = 0;
        if (qf < __uint_as_float(cls_55) && cls_55 < 4278190080u) {
          key_58 = 1073741824 | (unsigned int)(tb + 9);
        }
        if (cls_55 == qc) {
          unsigned int lowb_9 = (u_54 ^ (unsigned int)((int)u_54 >> 31) & 511) & 511;
          key_58 = 536870912 | lowb_9 << 9 | (unsigned int)(tb + 9);
        }
        kbt2[9] = __uint_as_float(key_58);
        float sc_59_1 = __uint_as_float(cbt[10]);
        sc_59_1 = fmaxf(sc_59_1, -1.7014118346046923e+38f);
        sc_59_1 = fminf(sc_59_1, 1.7014118346046923e+38f);
        unsigned int u_60 = __as_u32(sc_59_1);
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
        float sc_64 = __uint_as_float(cbt[11]);
        sc_64 = fmaxf(sc_64, -1.7014118346046923e+38f);
        sc_64 = fminf(sc_64, 1.7014118346046923e+38f);
        unsigned int u_65 = __as_u32(sc_64);
        unsigned int cls_66 = u_65 & 4294966784u;
        int f_67 = 0;
        if ((tb + 11 < fb || tb + 11 >= lim - fe) && lim > tb + 11) {
          f_67 = 1;
        }
        if (f_67 != 0) {
          cls_66 = 2139094528;
        }
        unsigned int key_68 = 0;
        if (qf < __uint_as_float(cls_66) && cls_66 < 4278190080u) {
          key_68 = 1073741824 | (unsigned int)(tb + 11);
        }
        if (cls_66 == qc) {
          unsigned int lowb_11 = (u_65 ^ (unsigned int)((int)u_65 >> 31) & 511) & 511;
          key_68 = 536870912 | lowb_11 << 9 | (unsigned int)(tb + 11);
        }
        kbt2[11] = __uint_as_float(key_68);
        float sc_69 = __uint_as_float(cbt[12]);
        sc_69 = fmaxf(sc_69, -1.7014118346046923e+38f);
        sc_69 = fminf(sc_69, 1.7014118346046923e+38f);
        unsigned int u_70 = __as_u32(sc_69);
        unsigned int cls_71 = u_70 & 4294966784u;
        int f_72 = 0;
        if ((tb + 12 < fb || tb + 12 >= lim - fe) && lim > tb + 12) {
          f_72 = 1;
        }
        if (f_72 != 0) {
          cls_71 = 2139094528;
        }
        unsigned int key_73 = 0;
        if (qf < __uint_as_float(cls_71) && cls_71 < 4278190080u) {
          key_73 = 1073741824 | (unsigned int)(tb + 12);
        }
        if (cls_71 == qc) {
          unsigned int lowb_12 = (u_70 ^ (unsigned int)((int)u_70 >> 31) & 511) & 511;
          key_73 = 536870912 | lowb_12 << 9 | (unsigned int)(tb + 12);
        }
        kbt2[12] = __uint_as_float(key_73);
        float sc_74 = __uint_as_float(cbt[13]);
        sc_74 = fmaxf(sc_74, -1.7014118346046923e+38f);
        sc_74 = fminf(sc_74, 1.7014118346046923e+38f);
        unsigned int u_75 = __as_u32(sc_74);
        unsigned int cls_76 = u_75 & 4294966784u;
        int f_77 = 0;
        if ((tb + 13 < fb || tb + 13 >= lim - fe) && lim > tb + 13) {
          f_77 = 1;
        }
        if (f_77 != 0) {
          cls_76 = 2139094528;
        }
        unsigned int key_78 = 0;
        if (qf < __uint_as_float(cls_76) && cls_76 < 4278190080u) {
          key_78 = 1073741824 | (unsigned int)(tb + 13);
        }
        if (cls_76 == qc) {
          unsigned int lowb_13 = (u_75 ^ (unsigned int)((int)u_75 >> 31) & 511) & 511;
          key_78 = 536870912 | lowb_13 << 9 | (unsigned int)(tb + 13);
        }
        kbt2[13] = __uint_as_float(key_78);
        float sc_79 = __uint_as_float(cbt[14]);
        sc_79 = fmaxf(sc_79, -1.7014118346046923e+38f);
        sc_79 = fminf(sc_79, 1.7014118346046923e+38f);
        unsigned int u_80 = __as_u32(sc_79);
        unsigned int cls_81 = u_80 & 4294966784u;
        int f_82 = 0;
        if ((tb + 14 < fb || tb + 14 >= lim - fe) && lim > tb + 14) {
          f_82 = 1;
        }
        if (f_82 != 0) {
          cls_81 = 2139094528;
        }
        unsigned int key_83 = 0;
        if (qf < __uint_as_float(cls_81) && cls_81 < 4278190080u) {
          key_83 = 1073741824 | (unsigned int)(tb + 14);
        }
        if (cls_81 == qc) {
          unsigned int lowb_14 = (u_80 ^ (unsigned int)((int)u_80 >> 31) & 511) & 511;
          key_83 = 536870912 | lowb_14 << 9 | (unsigned int)(tb + 14);
        }
        kbt2[14] = __uint_as_float(key_83);
        float sc_84 = __uint_as_float(cbt[15]);
        sc_84 = fmaxf(sc_84, -1.7014118346046923e+38f);
        sc_84 = fminf(sc_84, 1.7014118346046923e+38f);
        unsigned int u_85 = __as_u32(sc_84);
        unsigned int cls_86 = u_85 & 4294966784u;
        int f_87 = 0;
        if ((tb + 15 < fb || tb + 15 >= lim - fe) && lim > tb + 15) {
          f_87 = 1;
        }
        if (f_87 != 0) {
          cls_86 = 2139094528;
        }
        unsigned int key_88 = 0;
        if (qf < __uint_as_float(cls_86) && cls_86 < 4278190080u) {
          key_88 = 1073741824 | (unsigned int)(tb + 15);
        }
        if (cls_86 == qc) {
          unsigned int lowb_15 = (u_85 ^ (unsigned int)((int)u_85 >> 31) & 511) & 511;
          key_88 = 536870912 | lowb_15 << 9 | (unsigned int)(tb + 15);
        }
        kbt2[15] = __uint_as_float(key_88);
        float sc_89 = __uint_as_float(cbt[16]);
        sc_89 = fmaxf(sc_89, -1.7014118346046923e+38f);
        sc_89 = fminf(sc_89, 1.7014118346046923e+38f);
        unsigned int u_90 = __as_u32(sc_89);
        unsigned int cls_91 = u_90 & 4294966784u;
        int f_92 = 0;
        if ((tb + 16 < fb || tb + 16 >= lim - fe) && lim > tb + 16) {
          f_92 = 1;
        }
        if (f_92 != 0) {
          cls_91 = 2139094528;
        }
        unsigned int key_93 = 0;
        if (qf < __uint_as_float(cls_91) && cls_91 < 4278190080u) {
          key_93 = 1073741824 | (unsigned int)(tb + 16);
        }
        if (cls_91 == qc) {
          unsigned int lowb_16 = (u_90 ^ (unsigned int)((int)u_90 >> 31) & 511) & 511;
          key_93 = 536870912 | lowb_16 << 9 | (unsigned int)(tb + 16);
        }
        kbt2[16] = __uint_as_float(key_93);
        float sc_94 = __uint_as_float(cbt[17]);
        sc_94 = fmaxf(sc_94, -1.7014118346046923e+38f);
        sc_94 = fminf(sc_94, 1.7014118346046923e+38f);
        unsigned int u_95 = __as_u32(sc_94);
        unsigned int cls_96 = u_95 & 4294966784u;
        int f_97 = 0;
        if ((tb + 17 < fb || tb + 17 >= lim - fe) && lim > tb + 17) {
          f_97 = 1;
        }
        if (f_97 != 0) {
          cls_96 = 2139094528;
        }
        unsigned int key_98 = 0;
        if (qf < __uint_as_float(cls_96) && cls_96 < 4278190080u) {
          key_98 = 1073741824 | (unsigned int)(tb + 17);
        }
        if (cls_96 == qc) {
          unsigned int lowb_17 = (u_95 ^ (unsigned int)((int)u_95 >> 31) & 511) & 511;
          key_98 = 536870912 | lowb_17 << 9 | (unsigned int)(tb + 17);
        }
        kbt2[17] = __uint_as_float(key_98);
        float sc_99 = __uint_as_float(cbt[18]);
        sc_99 = fmaxf(sc_99, -1.7014118346046923e+38f);
        sc_99 = fminf(sc_99, 1.7014118346046923e+38f);
        unsigned int u_100 = __as_u32(sc_99);
        unsigned int cls_101 = u_100 & 4294966784u;
        int f_102 = 0;
        if ((tb + 18 < fb || tb + 18 >= lim - fe) && lim > tb + 18) {
          f_102 = 1;
        }
        if (f_102 != 0) {
          cls_101 = 2139094528;
        }
        unsigned int key_103 = 0;
        if (qf < __uint_as_float(cls_101) && cls_101 < 4278190080u) {
          key_103 = 1073741824 | (unsigned int)(tb + 18);
        }
        if (cls_101 == qc) {
          unsigned int lowb_18 = (u_100 ^ (unsigned int)((int)u_100 >> 31) & 511) & 511;
          key_103 = 536870912 | lowb_18 << 9 | (unsigned int)(tb + 18);
        }
        kbt2[18] = __uint_as_float(key_103);
        float sc_104 = __uint_as_float(cbt[19]);
        sc_104 = fmaxf(sc_104, -1.7014118346046923e+38f);
        sc_104 = fminf(sc_104, 1.7014118346046923e+38f);
        unsigned int u_105 = __as_u32(sc_104);
        unsigned int cls_106 = u_105 & 4294966784u;
        int f_107 = 0;
        if ((tb + 19 < fb || tb + 19 >= lim - fe) && lim > tb + 19) {
          f_107 = 1;
        }
        if (f_107 != 0) {
          cls_106 = 2139094528;
        }
        unsigned int key_108 = 0;
        if (qf < __uint_as_float(cls_106) && cls_106 < 4278190080u) {
          key_108 = 1073741824 | (unsigned int)(tb + 19);
        }
        if (cls_106 == qc) {
          unsigned int lowb_19 = (u_105 ^ (unsigned int)((int)u_105 >> 31) & 511) & 511;
          key_108 = 536870912 | lowb_19 << 9 | (unsigned int)(tb + 19);
        }
        kbt2[19] = __uint_as_float(key_108);
        float sc_109 = __uint_as_float(cbt[20]);
        sc_109 = fmaxf(sc_109, -1.7014118346046923e+38f);
        sc_109 = fminf(sc_109, 1.7014118346046923e+38f);
        unsigned int u_110 = __as_u32(sc_109);
        unsigned int cls_111 = u_110 & 4294966784u;
        int f_112 = 0;
        if ((tb + 20 < fb || tb + 20 >= lim - fe) && lim > tb + 20) {
          f_112 = 1;
        }
        if (f_112 != 0) {
          cls_111 = 2139094528;
        }
        unsigned int key_113 = 0;
        if (qf < __uint_as_float(cls_111) && cls_111 < 4278190080u) {
          key_113 = 1073741824 | (unsigned int)(tb + 20);
        }
        if (cls_111 == qc) {
          unsigned int lowb_20 = (u_110 ^ (unsigned int)((int)u_110 >> 31) & 511) & 511;
          key_113 = 536870912 | lowb_20 << 9 | (unsigned int)(tb + 20);
        }
        kbt2[20] = __uint_as_float(key_113);
        float sc_114 = __uint_as_float(cbt[21]);
        sc_114 = fmaxf(sc_114, -1.7014118346046923e+38f);
        sc_114 = fminf(sc_114, 1.7014118346046923e+38f);
        unsigned int u_115 = __as_u32(sc_114);
        unsigned int cls_116 = u_115 & 4294966784u;
        int f_117 = 0;
        if ((tb + 21 < fb || tb + 21 >= lim - fe) && lim > tb + 21) {
          f_117 = 1;
        }
        if (f_117 != 0) {
          cls_116 = 2139094528;
        }
        unsigned int key_118 = 0;
        if (qf < __uint_as_float(cls_116) && cls_116 < 4278190080u) {
          key_118 = 1073741824 | (unsigned int)(tb + 21);
        }
        if (cls_116 == qc) {
          unsigned int lowb_21 = (u_115 ^ (unsigned int)((int)u_115 >> 31) & 511) & 511;
          key_118 = 536870912 | lowb_21 << 9 | (unsigned int)(tb + 21);
        }
        kbt2[21] = __uint_as_float(key_118);
        float sc_119 = __uint_as_float(cbt[22]);
        sc_119 = fmaxf(sc_119, -1.7014118346046923e+38f);
        sc_119 = fminf(sc_119, 1.7014118346046923e+38f);
        unsigned int u_120 = __as_u32(sc_119);
        unsigned int cls_121 = u_120 & 4294966784u;
        int f_122 = 0;
        if ((tb + 22 < fb || tb + 22 >= lim - fe) && lim > tb + 22) {
          f_122 = 1;
        }
        if (f_122 != 0) {
          cls_121 = 2139094528;
        }
        unsigned int key_123 = 0;
        if (qf < __uint_as_float(cls_121) && cls_121 < 4278190080u) {
          key_123 = 1073741824 | (unsigned int)(tb + 22);
        }
        if (cls_121 == qc) {
          unsigned int lowb_22 = (u_120 ^ (unsigned int)((int)u_120 >> 31) & 511) & 511;
          key_123 = 536870912 | lowb_22 << 9 | (unsigned int)(tb + 22);
        }
        kbt2[22] = __uint_as_float(key_123);
        float sc_124 = __uint_as_float(cbt[23]);
        sc_124 = fmaxf(sc_124, -1.7014118346046923e+38f);
        sc_124 = fminf(sc_124, 1.7014118346046923e+38f);
        unsigned int u_125 = __as_u32(sc_124);
        unsigned int cls_126 = u_125 & 4294966784u;
        int f_127 = 0;
        if ((tb + 23 < fb || tb + 23 >= lim - fe) && lim > tb + 23) {
          f_127 = 1;
        }
        if (f_127 != 0) {
          cls_126 = 2139094528;
        }
        unsigned int key_128 = 0;
        if (qf < __uint_as_float(cls_126) && cls_126 < 4278190080u) {
          key_128 = 1073741824 | (unsigned int)(tb + 23);
        }
        if (cls_126 == qc) {
          unsigned int lowb_23 = (u_125 ^ (unsigned int)((int)u_125 >> 31) & 511) & 511;
          key_128 = 536870912 | lowb_23 << 9 | (unsigned int)(tb + 23);
        }
        kbt2[23] = __uint_as_float(key_128);
        float sc_129 = __uint_as_float(cbt[24]);
        sc_129 = fmaxf(sc_129, -1.7014118346046923e+38f);
        sc_129 = fminf(sc_129, 1.7014118346046923e+38f);
        unsigned int u_130 = __as_u32(sc_129);
        unsigned int cls_131 = u_130 & 4294966784u;
        int f_132 = 0;
        if ((tb + 24 < fb || tb + 24 >= lim - fe) && lim > tb + 24) {
          f_132 = 1;
        }
        if (f_132 != 0) {
          cls_131 = 2139094528;
        }
        unsigned int key_133 = 0;
        if (qf < __uint_as_float(cls_131) && cls_131 < 4278190080u) {
          key_133 = 1073741824 | (unsigned int)(tb + 24);
        }
        if (cls_131 == qc) {
          unsigned int lowb_24 = (u_130 ^ (unsigned int)((int)u_130 >> 31) & 511) & 511;
          key_133 = 536870912 | lowb_24 << 9 | (unsigned int)(tb + 24);
        }
        kbt2[24] = __uint_as_float(key_133);
        float sc_134 = __uint_as_float(cbt[25]);
        sc_134 = fmaxf(sc_134, -1.7014118346046923e+38f);
        sc_134 = fminf(sc_134, 1.7014118346046923e+38f);
        unsigned int u_135 = __as_u32(sc_134);
        unsigned int cls_136 = u_135 & 4294966784u;
        int f_137 = 0;
        if ((tb + 25 < fb || tb + 25 >= lim - fe) && lim > tb + 25) {
          f_137 = 1;
        }
        if (f_137 != 0) {
          cls_136 = 2139094528;
        }
        unsigned int key_138 = 0;
        if (qf < __uint_as_float(cls_136) && cls_136 < 4278190080u) {
          key_138 = 1073741824 | (unsigned int)(tb + 25);
        }
        if (cls_136 == qc) {
          unsigned int lowb_25 = (u_135 ^ (unsigned int)((int)u_135 >> 31) & 511) & 511;
          key_138 = 536870912 | lowb_25 << 9 | (unsigned int)(tb + 25);
        }
        kbt2[25] = __uint_as_float(key_138);
        float sc_139 = __uint_as_float(cbt[26]);
        sc_139 = fmaxf(sc_139, -1.7014118346046923e+38f);
        sc_139 = fminf(sc_139, 1.7014118346046923e+38f);
        unsigned int u_140 = __as_u32(sc_139);
        unsigned int cls_141 = u_140 & 4294966784u;
        int f_142 = 0;
        if ((tb + 26 < fb || tb + 26 >= lim - fe) && lim > tb + 26) {
          f_142 = 1;
        }
        if (f_142 != 0) {
          cls_141 = 2139094528;
        }
        unsigned int key_143 = 0;
        if (qf < __uint_as_float(cls_141) && cls_141 < 4278190080u) {
          key_143 = 1073741824 | (unsigned int)(tb + 26);
        }
        if (cls_141 == qc) {
          unsigned int lowb_26 = (u_140 ^ (unsigned int)((int)u_140 >> 31) & 511) & 511;
          key_143 = 536870912 | lowb_26 << 9 | (unsigned int)(tb + 26);
        }
        kbt2[26] = __uint_as_float(key_143);
        float sc_144 = __uint_as_float(cbt[27]);
        sc_144 = fmaxf(sc_144, -1.7014118346046923e+38f);
        sc_144 = fminf(sc_144, 1.7014118346046923e+38f);
        unsigned int u_145 = __as_u32(sc_144);
        unsigned int cls_146 = u_145 & 4294966784u;
        int f_147 = 0;
        if ((tb + 27 < fb || tb + 27 >= lim - fe) && lim > tb + 27) {
          f_147 = 1;
        }
        if (f_147 != 0) {
          cls_146 = 2139094528;
        }
        unsigned int key_148 = 0;
        if (qf < __uint_as_float(cls_146) && cls_146 < 4278190080u) {
          key_148 = 1073741824 | (unsigned int)(tb + 27);
        }
        if (cls_146 == qc) {
          unsigned int lowb_27 = (u_145 ^ (unsigned int)((int)u_145 >> 31) & 511) & 511;
          key_148 = 536870912 | lowb_27 << 9 | (unsigned int)(tb + 27);
        }
        kbt2[27] = __uint_as_float(key_148);
        float sc_149 = __uint_as_float(cbt[28]);
        sc_149 = fmaxf(sc_149, -1.7014118346046923e+38f);
        sc_149 = fminf(sc_149, 1.7014118346046923e+38f);
        unsigned int u_150 = __as_u32(sc_149);
        unsigned int cls_151 = u_150 & 4294966784u;
        int f_152 = 0;
        if ((tb + 28 < fb || tb + 28 >= lim - fe) && lim > tb + 28) {
          f_152 = 1;
        }
        if (f_152 != 0) {
          cls_151 = 2139094528;
        }
        unsigned int key_153 = 0;
        if (qf < __uint_as_float(cls_151) && cls_151 < 4278190080u) {
          key_153 = 1073741824 | (unsigned int)(tb + 28);
        }
        if (cls_151 == qc) {
          unsigned int lowb_28 = (u_150 ^ (unsigned int)((int)u_150 >> 31) & 511) & 511;
          key_153 = 536870912 | lowb_28 << 9 | (unsigned int)(tb + 28);
        }
        kbt2[28] = __uint_as_float(key_153);
        float sc_154 = __uint_as_float(cbt[29]);
        sc_154 = fmaxf(sc_154, -1.7014118346046923e+38f);
        sc_154 = fminf(sc_154, 1.7014118346046923e+38f);
        unsigned int u_155 = __as_u32(sc_154);
        unsigned int cls_156 = u_155 & 4294966784u;
        int f_157 = 0;
        if ((tb + 29 < fb || tb + 29 >= lim - fe) && lim > tb + 29) {
          f_157 = 1;
        }
        if (f_157 != 0) {
          cls_156 = 2139094528;
        }
        unsigned int key_158 = 0;
        if (qf < __uint_as_float(cls_156) && cls_156 < 4278190080u) {
          key_158 = 1073741824 | (unsigned int)(tb + 29);
        }
        if (cls_156 == qc) {
          unsigned int lowb_29 = (u_155 ^ (unsigned int)((int)u_155 >> 31) & 511) & 511;
          key_158 = 536870912 | lowb_29 << 9 | (unsigned int)(tb + 29);
        }
        kbt2[29] = __uint_as_float(key_158);
        float sc_159 = __uint_as_float(cbt[30]);
        sc_159 = fmaxf(sc_159, -1.7014118346046923e+38f);
        sc_159 = fminf(sc_159, 1.7014118346046923e+38f);
        unsigned int u_160 = __as_u32(sc_159);
        unsigned int cls_161 = u_160 & 4294966784u;
        int f_162 = 0;
        if ((tb + 30 < fb || tb + 30 >= lim - fe) && lim > tb + 30) {
          f_162 = 1;
        }
        if (f_162 != 0) {
          cls_161 = 2139094528;
        }
        unsigned int key_163 = 0;
        if (qf < __uint_as_float(cls_161) && cls_161 < 4278190080u) {
          key_163 = 1073741824 | (unsigned int)(tb + 30);
        }
        if (cls_161 == qc) {
          unsigned int lowb_30 = (u_160 ^ (unsigned int)((int)u_160 >> 31) & 511) & 511;
          key_163 = 536870912 | lowb_30 << 9 | (unsigned int)(tb + 30);
        }
        kbt2[30] = __uint_as_float(key_163);
        float sc_164 = __uint_as_float(cbt[31]);
        sc_164 = fmaxf(sc_164, -1.7014118346046923e+38f);
        sc_164 = fminf(sc_164, 1.7014118346046923e+38f);
        unsigned int u_165 = __as_u32(sc_164);
        unsigned int cls_166 = u_165 & 4294966784u;
        int f_167 = 0;
        if ((tb + 31 < fb || tb + 31 >= lim - fe) && lim > tb + 31) {
          f_167 = 1;
        }
        if (f_167 != 0) {
          cls_166 = 2139094528;
        }
        unsigned int key_168 = 0;
        if (qf < __uint_as_float(cls_166) && cls_166 < 4278190080u) {
          key_168 = 1073741824 | (unsigned int)(tb + 31);
        }
        if (cls_166 == qc) {
          unsigned int lowb_31 = (u_165 ^ (unsigned int)((int)u_165 >> 31) & 511) & 511;
          key_168 = 536870912 | lowb_31 << 9 | (unsigned int)(tb + 31);
        }
        kbt2[31] = __uint_as_float(key_168);
        float rr_169 = r2o;
        float hi_171 = fmaxf(kbt2[0], kbt2[13]);
        float lo_172 = fminf(kbt2[0], kbt2[13]);
        kbt2[0] = hi_171;
        kbt2[13] = lo_172;
        float hi_173 = fmaxf(kbt2[1], kbt2[12]);
        float lo_174 = fminf(kbt2[1], kbt2[12]);
        kbt2[1] = hi_173;
        kbt2[12] = lo_174;
        float hi_175 = fmaxf(kbt2[2], kbt2[15]);
        float lo_176 = fminf(kbt2[2], kbt2[15]);
        kbt2[2] = hi_175;
        kbt2[15] = lo_176;
        float hi_177 = fmaxf(kbt2[3], kbt2[14]);
        float lo_178 = fminf(kbt2[3], kbt2[14]);
        kbt2[3] = hi_177;
        kbt2[14] = lo_178;
        float hi_179 = fmaxf(kbt2[4], kbt2[8]);
        float lo_180 = fminf(kbt2[4], kbt2[8]);
        kbt2[4] = hi_179;
        kbt2[8] = lo_180;
        float hi_181 = fmaxf(kbt2[5], kbt2[6]);
        float lo_182 = fminf(kbt2[5], kbt2[6]);
        kbt2[5] = hi_181;
        kbt2[6] = lo_182;
        float hi_183 = fmaxf(kbt2[7], kbt2[11]);
        float lo_184 = fminf(kbt2[7], kbt2[11]);
        kbt2[7] = hi_183;
        kbt2[11] = lo_184;
        float hi_185 = fmaxf(kbt2[9], kbt2[10]);
        float lo_186 = fminf(kbt2[9], kbt2[10]);
        kbt2[9] = hi_185;
        kbt2[10] = lo_186;
        float hi_187 = fmaxf(kbt2[0], kbt2[5]);
        float lo_188 = fminf(kbt2[0], kbt2[5]);
        kbt2[0] = hi_187;
        kbt2[5] = lo_188;
        float hi_189 = fmaxf(kbt2[1], kbt2[7]);
        float lo_190 = fminf(kbt2[1], kbt2[7]);
        kbt2[1] = hi_189;
        kbt2[7] = lo_190;
        float hi_191 = fmaxf(kbt2[2], kbt2[9]);
        float lo_192 = fminf(kbt2[2], kbt2[9]);
        kbt2[2] = hi_191;
        kbt2[9] = lo_192;
        float hi_193 = fmaxf(kbt2[3], kbt2[4]);
        float lo_194 = fminf(kbt2[3], kbt2[4]);
        kbt2[3] = hi_193;
        kbt2[4] = lo_194;
        float hi_195 = fmaxf(kbt2[6], kbt2[13]);
        float lo_196 = fminf(kbt2[6], kbt2[13]);
        kbt2[6] = hi_195;
        kbt2[13] = lo_196;
        float hi_197 = fmaxf(kbt2[8], kbt2[14]);
        float lo_198 = fminf(kbt2[8], kbt2[14]);
        kbt2[8] = hi_197;
        kbt2[14] = lo_198;
        float hi_199 = fmaxf(kbt2[10], kbt2[15]);
        float lo_200 = fminf(kbt2[10], kbt2[15]);
        kbt2[10] = hi_199;
        kbt2[15] = lo_200;
        float hi_201 = fmaxf(kbt2[11], kbt2[12]);
        float lo_202 = fminf(kbt2[11], kbt2[12]);
        kbt2[11] = hi_201;
        kbt2[12] = lo_202;
        float hi_203 = fmaxf(kbt2[0], kbt2[1]);
        float lo_204 = fminf(kbt2[0], kbt2[1]);
        kbt2[0] = hi_203;
        kbt2[1] = lo_204;
        float hi_205 = fmaxf(kbt2[2], kbt2[3]);
        float lo_206 = fminf(kbt2[2], kbt2[3]);
        kbt2[2] = hi_205;
        kbt2[3] = lo_206;
        float hi_207 = fmaxf(kbt2[4], kbt2[5]);
        float lo_208 = fminf(kbt2[4], kbt2[5]);
        kbt2[4] = hi_207;
        kbt2[5] = lo_208;
        float hi_209 = fmaxf(kbt2[6], kbt2[8]);
        float lo_210 = fminf(kbt2[6], kbt2[8]);
        kbt2[6] = hi_209;
        kbt2[8] = lo_210;
        float hi_211 = fmaxf(kbt2[7], kbt2[9]);
        float lo_212 = fminf(kbt2[7], kbt2[9]);
        kbt2[7] = hi_211;
        kbt2[9] = lo_212;
        float hi_213 = fmaxf(kbt2[10], kbt2[11]);
        float lo_214 = fminf(kbt2[10], kbt2[11]);
        kbt2[10] = hi_213;
        kbt2[11] = lo_214;
        float hi_215 = fmaxf(kbt2[12], kbt2[13]);
        float lo_216 = fminf(kbt2[12], kbt2[13]);
        kbt2[12] = hi_215;
        kbt2[13] = lo_216;
        float hi_217 = fmaxf(kbt2[14], kbt2[15]);
        float lo_218 = fminf(kbt2[14], kbt2[15]);
        kbt2[14] = hi_217;
        kbt2[15] = lo_218;
        float hi_219 = fmaxf(kbt2[0], kbt2[2]);
        float lo_220 = fminf(kbt2[0], kbt2[2]);
        kbt2[0] = hi_219;
        kbt2[2] = lo_220;
        float hi_221 = fmaxf(kbt2[1], kbt2[3]);
        float lo_222 = fminf(kbt2[1], kbt2[3]);
        kbt2[1] = hi_221;
        kbt2[3] = lo_222;
        float hi_223 = fmaxf(kbt2[4], kbt2[10]);
        float lo_224 = fminf(kbt2[4], kbt2[10]);
        kbt2[4] = hi_223;
        kbt2[10] = lo_224;
        float hi_225 = fmaxf(kbt2[5], kbt2[11]);
        float lo_226 = fminf(kbt2[5], kbt2[11]);
        kbt2[5] = hi_225;
        kbt2[11] = lo_226;
        float hi_227 = fmaxf(kbt2[6], kbt2[7]);
        float lo_228 = fminf(kbt2[6], kbt2[7]);
        kbt2[6] = hi_227;
        kbt2[7] = lo_228;
        float hi_229 = fmaxf(kbt2[8], kbt2[9]);
        float lo_230 = fminf(kbt2[8], kbt2[9]);
        kbt2[8] = hi_229;
        kbt2[9] = lo_230;
        float hi_231 = fmaxf(kbt2[12], kbt2[14]);
        float lo_232 = fminf(kbt2[12], kbt2[14]);
        kbt2[12] = hi_231;
        kbt2[14] = lo_232;
        float hi_233 = fmaxf(kbt2[13], kbt2[15]);
        float lo_234 = fminf(kbt2[13], kbt2[15]);
        kbt2[13] = hi_233;
        kbt2[15] = lo_234;
        float hi_235 = fmaxf(kbt2[1], kbt2[2]);
        float lo_236 = fminf(kbt2[1], kbt2[2]);
        kbt2[1] = hi_235;
        kbt2[2] = lo_236;
        float hi_237 = fmaxf(kbt2[3], kbt2[12]);
        float lo_238 = fminf(kbt2[3], kbt2[12]);
        kbt2[3] = hi_237;
        kbt2[12] = lo_238;
        float hi_239 = fmaxf(kbt2[4], kbt2[6]);
        float lo_240 = fminf(kbt2[4], kbt2[6]);
        kbt2[4] = hi_239;
        kbt2[6] = lo_240;
        float hi_241 = fmaxf(kbt2[5], kbt2[7]);
        float lo_242 = fminf(kbt2[5], kbt2[7]);
        kbt2[5] = hi_241;
        kbt2[7] = lo_242;
        float hi_243 = fmaxf(kbt2[8], kbt2[10]);
        float lo_244 = fminf(kbt2[8], kbt2[10]);
        kbt2[8] = hi_243;
        kbt2[10] = lo_244;
        float hi_245 = fmaxf(kbt2[9], kbt2[11]);
        float lo_246 = fminf(kbt2[9], kbt2[11]);
        kbt2[9] = hi_245;
        kbt2[11] = lo_246;
        float hi_247 = fmaxf(kbt2[13], kbt2[14]);
        float lo_248 = fminf(kbt2[13], kbt2[14]);
        kbt2[13] = hi_247;
        kbt2[14] = lo_248;
        float hi_249 = fmaxf(kbt2[1], kbt2[4]);
        float lo_250 = fminf(kbt2[1], kbt2[4]);
        kbt2[1] = hi_249;
        kbt2[4] = lo_250;
        float hi_251 = fmaxf(kbt2[2], kbt2[6]);
        float lo_252 = fminf(kbt2[2], kbt2[6]);
        kbt2[2] = hi_251;
        kbt2[6] = lo_252;
        float hi_253 = fmaxf(kbt2[5], kbt2[8]);
        float lo_254 = fminf(kbt2[5], kbt2[8]);
        kbt2[5] = hi_253;
        kbt2[8] = lo_254;
        float hi_255 = fmaxf(kbt2[7], kbt2[10]);
        float lo_256 = fminf(kbt2[7], kbt2[10]);
        kbt2[7] = hi_255;
        kbt2[10] = lo_256;
        float hi_257 = fmaxf(kbt2[9], kbt2[13]);
        float lo_258 = fminf(kbt2[9], kbt2[13]);
        kbt2[9] = hi_257;
        kbt2[13] = lo_258;
        float hi_259 = fmaxf(kbt2[11], kbt2[14]);
        float lo_260 = fminf(kbt2[11], kbt2[14]);
        kbt2[11] = hi_259;
        kbt2[14] = lo_260;
        float hi_261 = fmaxf(kbt2[2], kbt2[4]);
        float lo_262 = fminf(kbt2[2], kbt2[4]);
        kbt2[2] = hi_261;
        kbt2[4] = lo_262;
        float hi_263 = fmaxf(kbt2[3], kbt2[6]);
        float lo_264 = fminf(kbt2[3], kbt2[6]);
        kbt2[3] = hi_263;
        kbt2[6] = lo_264;
        float hi_265 = fmaxf(kbt2[9], kbt2[12]);
        float lo_266 = fminf(kbt2[9], kbt2[12]);
        kbt2[9] = hi_265;
        kbt2[12] = lo_266;
        float hi_267 = fmaxf(kbt2[11], kbt2[13]);
        float lo_268 = fminf(kbt2[11], kbt2[13]);
        kbt2[11] = hi_267;
        kbt2[13] = lo_268;
        float hi_269 = fmaxf(kbt2[3], kbt2[5]);
        float lo_270 = fminf(kbt2[3], kbt2[5]);
        kbt2[3] = hi_269;
        kbt2[5] = lo_270;
        float hi_271 = fmaxf(kbt2[6], kbt2[8]);
        float lo_272 = fminf(kbt2[6], kbt2[8]);
        kbt2[6] = hi_271;
        kbt2[8] = lo_272;
        float hi_273 = fmaxf(kbt2[7], kbt2[9]);
        float lo_274 = fminf(kbt2[7], kbt2[9]);
        kbt2[7] = hi_273;
        kbt2[9] = lo_274;
        float hi_275 = fmaxf(kbt2[10], kbt2[12]);
        float lo_276 = fminf(kbt2[10], kbt2[12]);
        kbt2[10] = hi_275;
        kbt2[12] = lo_276;
        float hi_277 = fmaxf(kbt2[3], kbt2[4]);
        float lo_278 = fminf(kbt2[3], kbt2[4]);
        kbt2[3] = hi_277;
        kbt2[4] = lo_278;
        float hi_279 = fmaxf(kbt2[5], kbt2[6]);
        float lo_280 = fminf(kbt2[5], kbt2[6]);
        kbt2[5] = hi_279;
        kbt2[6] = lo_280;
        float hi_281 = fmaxf(kbt2[7], kbt2[8]);
        float lo_282 = fminf(kbt2[7], kbt2[8]);
        kbt2[7] = hi_281;
        kbt2[8] = lo_282;
        float hi_283 = fmaxf(kbt2[9], kbt2[10]);
        float lo_284 = fminf(kbt2[9], kbt2[10]);
        kbt2[9] = hi_283;
        kbt2[10] = lo_284;
        float hi_285 = fmaxf(kbt2[11], kbt2[12]);
        float lo_286 = fminf(kbt2[11], kbt2[12]);
        kbt2[11] = hi_285;
        kbt2[12] = lo_286;
        float hi_287 = fmaxf(kbt2[6], kbt2[7]);
        float lo_288 = fminf(kbt2[6], kbt2[7]);
        kbt2[6] = hi_287;
        kbt2[7] = lo_288;
        float hi_289 = fmaxf(kbt2[8], kbt2[9]);
        float lo_290 = fminf(kbt2[8], kbt2[9]);
        kbt2[8] = hi_289;
        kbt2[9] = lo_290;
        float hi_291 = fmaxf(kbt2[16], kbt2[29]);
        float lo_292 = fminf(kbt2[16], kbt2[29]);
        kbt2[16] = hi_291;
        kbt2[29] = lo_292;
        float hi_293 = fmaxf(kbt2[17], kbt2[28]);
        float lo_294 = fminf(kbt2[17], kbt2[28]);
        kbt2[17] = hi_293;
        kbt2[28] = lo_294;
        float hi_295 = fmaxf(kbt2[18], kbt2[31]);
        float lo_296 = fminf(kbt2[18], kbt2[31]);
        kbt2[18] = hi_295;
        kbt2[31] = lo_296;
        float hi_297 = fmaxf(kbt2[19], kbt2[30]);
        float lo_298 = fminf(kbt2[19], kbt2[30]);
        kbt2[19] = hi_297;
        kbt2[30] = lo_298;
        float hi_299 = fmaxf(kbt2[20], kbt2[24]);
        float lo_300 = fminf(kbt2[20], kbt2[24]);
        kbt2[20] = hi_299;
        kbt2[24] = lo_300;
        float hi_301 = fmaxf(kbt2[21], kbt2[22]);
        float lo_302 = fminf(kbt2[21], kbt2[22]);
        kbt2[21] = hi_301;
        kbt2[22] = lo_302;
        float hi_303 = fmaxf(kbt2[23], kbt2[27]);
        float lo_304 = fminf(kbt2[23], kbt2[27]);
        kbt2[23] = hi_303;
        kbt2[27] = lo_304;
        float hi_305 = fmaxf(kbt2[25], kbt2[26]);
        float lo_306 = fminf(kbt2[25], kbt2[26]);
        kbt2[25] = hi_305;
        kbt2[26] = lo_306;
        float hi_307 = fmaxf(kbt2[16], kbt2[21]);
        float lo_308 = fminf(kbt2[16], kbt2[21]);
        kbt2[16] = hi_307;
        kbt2[21] = lo_308;
        float hi_309 = fmaxf(kbt2[17], kbt2[23]);
        float lo_310 = fminf(kbt2[17], kbt2[23]);
        kbt2[17] = hi_309;
        kbt2[23] = lo_310;
        float hi_311 = fmaxf(kbt2[18], kbt2[25]);
        float lo_312 = fminf(kbt2[18], kbt2[25]);
        kbt2[18] = hi_311;
        kbt2[25] = lo_312;
        float hi_313 = fmaxf(kbt2[19], kbt2[20]);
        float lo_314 = fminf(kbt2[19], kbt2[20]);
        kbt2[19] = hi_313;
        kbt2[20] = lo_314;
        float hi_315 = fmaxf(kbt2[22], kbt2[29]);
        float lo_316 = fminf(kbt2[22], kbt2[29]);
        kbt2[22] = hi_315;
        kbt2[29] = lo_316;
        float hi_317 = fmaxf(kbt2[24], kbt2[30]);
        float lo_318 = fminf(kbt2[24], kbt2[30]);
        kbt2[24] = hi_317;
        kbt2[30] = lo_318;
        float hi_319 = fmaxf(kbt2[26], kbt2[31]);
        float lo_320 = fminf(kbt2[26], kbt2[31]);
        kbt2[26] = hi_319;
        kbt2[31] = lo_320;
        float hi_321 = fmaxf(kbt2[27], kbt2[28]);
        float lo_322 = fminf(kbt2[27], kbt2[28]);
        kbt2[27] = hi_321;
        kbt2[28] = lo_322;
        float hi_323 = fmaxf(kbt2[16], kbt2[17]);
        float lo_324 = fminf(kbt2[16], kbt2[17]);
        kbt2[16] = hi_323;
        kbt2[17] = lo_324;
        float hi_325 = fmaxf(kbt2[18], kbt2[19]);
        float lo_326 = fminf(kbt2[18], kbt2[19]);
        kbt2[18] = hi_325;
        kbt2[19] = lo_326;
        float hi_327 = fmaxf(kbt2[20], kbt2[21]);
        float lo_328 = fminf(kbt2[20], kbt2[21]);
        kbt2[20] = hi_327;
        kbt2[21] = lo_328;
        float hi_329 = fmaxf(kbt2[22], kbt2[24]);
        float lo_330 = fminf(kbt2[22], kbt2[24]);
        kbt2[22] = hi_329;
        kbt2[24] = lo_330;
        float hi_331 = fmaxf(kbt2[23], kbt2[25]);
        float lo_332 = fminf(kbt2[23], kbt2[25]);
        kbt2[23] = hi_331;
        kbt2[25] = lo_332;
        float hi_333 = fmaxf(kbt2[26], kbt2[27]);
        float lo_334 = fminf(kbt2[26], kbt2[27]);
        kbt2[26] = hi_333;
        kbt2[27] = lo_334;
        float hi_335 = fmaxf(kbt2[28], kbt2[29]);
        float lo_336 = fminf(kbt2[28], kbt2[29]);
        kbt2[28] = hi_335;
        kbt2[29] = lo_336;
        float hi_337 = fmaxf(kbt2[30], kbt2[31]);
        float lo_338 = fminf(kbt2[30], kbt2[31]);
        kbt2[30] = hi_337;
        kbt2[31] = lo_338;
        float hi_339 = fmaxf(kbt2[16], kbt2[18]);
        float lo_340 = fminf(kbt2[16], kbt2[18]);
        kbt2[16] = hi_339;
        kbt2[18] = lo_340;
        float hi_341 = fmaxf(kbt2[17], kbt2[19]);
        float lo_342 = fminf(kbt2[17], kbt2[19]);
        kbt2[17] = hi_341;
        kbt2[19] = lo_342;
        float hi_343 = fmaxf(kbt2[20], kbt2[26]);
        float lo_344 = fminf(kbt2[20], kbt2[26]);
        kbt2[20] = hi_343;
        kbt2[26] = lo_344;
        float hi_345 = fmaxf(kbt2[21], kbt2[27]);
        float lo_346 = fminf(kbt2[21], kbt2[27]);
        kbt2[21] = hi_345;
        kbt2[27] = lo_346;
        float hi_347 = fmaxf(kbt2[22], kbt2[23]);
        float lo_348 = fminf(kbt2[22], kbt2[23]);
        kbt2[22] = hi_347;
        kbt2[23] = lo_348;
        float hi_349 = fmaxf(kbt2[24], kbt2[25]);
        float lo_350 = fminf(kbt2[24], kbt2[25]);
        kbt2[24] = hi_349;
        kbt2[25] = lo_350;
        float hi_351 = fmaxf(kbt2[28], kbt2[30]);
        float lo_352 = fminf(kbt2[28], kbt2[30]);
        kbt2[28] = hi_351;
        kbt2[30] = lo_352;
        float hi_353 = fmaxf(kbt2[29], kbt2[31]);
        float lo_354 = fminf(kbt2[29], kbt2[31]);
        kbt2[29] = hi_353;
        kbt2[31] = lo_354;
        float hi_355 = fmaxf(kbt2[17], kbt2[18]);
        float lo_356 = fminf(kbt2[17], kbt2[18]);
        kbt2[17] = hi_355;
        kbt2[18] = lo_356;
        float hi_357 = fmaxf(kbt2[19], kbt2[28]);
        float lo_358 = fminf(kbt2[19], kbt2[28]);
        kbt2[19] = hi_357;
        kbt2[28] = lo_358;
        float hi_359 = fmaxf(kbt2[20], kbt2[22]);
        float lo_360 = fminf(kbt2[20], kbt2[22]);
        kbt2[20] = hi_359;
        kbt2[22] = lo_360;
        float hi_361 = fmaxf(kbt2[21], kbt2[23]);
        float lo_362 = fminf(kbt2[21], kbt2[23]);
        kbt2[21] = hi_361;
        kbt2[23] = lo_362;
        float hi_363 = fmaxf(kbt2[24], kbt2[26]);
        float lo_364 = fminf(kbt2[24], kbt2[26]);
        kbt2[24] = hi_363;
        kbt2[26] = lo_364;
        float hi_365 = fmaxf(kbt2[25], kbt2[27]);
        float lo_366 = fminf(kbt2[25], kbt2[27]);
        kbt2[25] = hi_365;
        kbt2[27] = lo_366;
        float hi_367 = fmaxf(kbt2[29], kbt2[30]);
        float lo_368 = fminf(kbt2[29], kbt2[30]);
        kbt2[29] = hi_367;
        kbt2[30] = lo_368;
        float hi_369 = fmaxf(kbt2[17], kbt2[20]);
        float lo_370 = fminf(kbt2[17], kbt2[20]);
        kbt2[17] = hi_369;
        kbt2[20] = lo_370;
        float hi_371 = fmaxf(kbt2[18], kbt2[22]);
        float lo_372 = fminf(kbt2[18], kbt2[22]);
        kbt2[18] = hi_371;
        kbt2[22] = lo_372;
        float hi_373 = fmaxf(kbt2[21], kbt2[24]);
        float lo_374 = fminf(kbt2[21], kbt2[24]);
        kbt2[21] = hi_373;
        kbt2[24] = lo_374;
        float hi_375 = fmaxf(kbt2[23], kbt2[26]);
        float lo_376 = fminf(kbt2[23], kbt2[26]);
        kbt2[23] = hi_375;
        kbt2[26] = lo_376;
        float hi_377 = fmaxf(kbt2[25], kbt2[29]);
        float lo_378 = fminf(kbt2[25], kbt2[29]);
        kbt2[25] = hi_377;
        kbt2[29] = lo_378;
        float hi_379 = fmaxf(kbt2[27], kbt2[30]);
        float lo_380 = fminf(kbt2[27], kbt2[30]);
        kbt2[27] = hi_379;
        kbt2[30] = lo_380;
        float hi_381 = fmaxf(kbt2[18], kbt2[20]);
        float lo_382 = fminf(kbt2[18], kbt2[20]);
        kbt2[18] = hi_381;
        kbt2[20] = lo_382;
        float hi_383 = fmaxf(kbt2[19], kbt2[22]);
        float lo_384 = fminf(kbt2[19], kbt2[22]);
        kbt2[19] = hi_383;
        kbt2[22] = lo_384;
        float hi_385 = fmaxf(kbt2[25], kbt2[28]);
        float lo_386 = fminf(kbt2[25], kbt2[28]);
        kbt2[25] = hi_385;
        kbt2[28] = lo_386;
        float hi_387 = fmaxf(kbt2[27], kbt2[29]);
        float lo_388 = fminf(kbt2[27], kbt2[29]);
        kbt2[27] = hi_387;
        kbt2[29] = lo_388;
        float hi_389 = fmaxf(kbt2[19], kbt2[21]);
        float lo_390 = fminf(kbt2[19], kbt2[21]);
        kbt2[19] = hi_389;
        kbt2[21] = lo_390;
        float hi_391 = fmaxf(kbt2[22], kbt2[24]);
        float lo_392 = fminf(kbt2[22], kbt2[24]);
        kbt2[22] = hi_391;
        kbt2[24] = lo_392;
        float hi_393 = fmaxf(kbt2[23], kbt2[25]);
        float lo_394 = fminf(kbt2[23], kbt2[25]);
        kbt2[23] = hi_393;
        kbt2[25] = lo_394;
        float hi_395 = fmaxf(kbt2[26], kbt2[28]);
        float lo_396 = fminf(kbt2[26], kbt2[28]);
        kbt2[26] = hi_395;
        kbt2[28] = lo_396;
        float hi_397 = fmaxf(kbt2[19], kbt2[20]);
        float lo_398 = fminf(kbt2[19], kbt2[20]);
        kbt2[19] = hi_397;
        kbt2[20] = lo_398;
        float hi_400 = fmaxf(kbt2[21], kbt2[22]);
        float lo_401 = fminf(kbt2[21], kbt2[22]);
        kbt2[21] = hi_400;
        kbt2[22] = lo_401;
        float hi_403 = fmaxf(kbt2[23], kbt2[24]);
        float lo_404 = fminf(kbt2[23], kbt2[24]);
        kbt2[23] = hi_403;
        kbt2[24] = lo_404;
        float hi_406 = fmaxf(kbt2[25], kbt2[26]);
        float lo_407 = fminf(kbt2[25], kbt2[26]);
        kbt2[25] = hi_406;
        kbt2[26] = lo_407;
        float hi_408 = fmaxf(kbt2[27], kbt2[28]);
        float lo_409 = fminf(kbt2[27], kbt2[28]);
        kbt2[27] = hi_408;
        kbt2[28] = lo_409;
        float hi_410 = fmaxf(kbt2[22], kbt2[23]);
        float lo_411 = fminf(kbt2[22], kbt2[23]);
        kbt2[22] = hi_410;
        kbt2[23] = lo_411;
        float hi_412 = fmaxf(kbt2[24], kbt2[25]);
        float lo_413 = fminf(kbt2[24], kbt2[25]);
        kbt2[24] = hi_412;
        kbt2[25] = lo_413;
        float hi_414 = fmaxf(kbt2[0], kbt2[31]);
        float lo_415 = fminf(kbt2[0], kbt2[31]);
        kbt2[0] = hi_414;
        rr_169 = fmaxf(rr_169, lo_415);
        float hi_416 = fmaxf(kbt2[1], kbt2[30]);
        float lo_417 = fminf(kbt2[1], kbt2[30]);
        kbt2[1] = hi_416;
        rr_169 = fmaxf(rr_169, lo_417);
        float hi_419 = fmaxf(kbt2[2], kbt2[29]);
        float lo_420 = fminf(kbt2[2], kbt2[29]);
        kbt2[2] = hi_419;
        rr_169 = fmaxf(rr_169, lo_420);
        float hi_422 = fmaxf(kbt2[3], kbt2[28]);
        float lo_423 = fminf(kbt2[3], kbt2[28]);
        kbt2[3] = hi_422;
        rr_169 = fmaxf(rr_169, lo_423);
        float hi_425 = fmaxf(kbt2[4], kbt2[27]);
        float lo_426 = fminf(kbt2[4], kbt2[27]);
        kbt2[4] = hi_425;
        rr_169 = fmaxf(rr_169, lo_426);
        float hi_427 = fmaxf(kbt2[5], kbt2[26]);
        float lo_428 = fminf(kbt2[5], kbt2[26]);
        kbt2[5] = hi_427;
        rr_169 = fmaxf(rr_169, lo_428);
        float hi_429 = fmaxf(kbt2[6], kbt2[25]);
        float lo_430 = fminf(kbt2[6], kbt2[25]);
        kbt2[6] = hi_429;
        rr_169 = fmaxf(rr_169, lo_430);
        float hi_431 = fmaxf(kbt2[7], kbt2[24]);
        float lo_432 = fminf(kbt2[7], kbt2[24]);
        kbt2[7] = hi_431;
        rr_169 = fmaxf(rr_169, lo_432);
        float hi_433 = fmaxf(kbt2[8], kbt2[23]);
        float lo_434 = fminf(kbt2[8], kbt2[23]);
        kbt2[8] = hi_433;
        rr_169 = fmaxf(rr_169, lo_434);
        float hi_435 = fmaxf(kbt2[9], kbt2[22]);
        float lo_436 = fminf(kbt2[9], kbt2[22]);
        kbt2[9] = hi_435;
        rr_169 = fmaxf(rr_169, lo_436);
        float hi_438 = fmaxf(kbt2[10], kbt2[21]);
        float lo_439 = fminf(kbt2[10], kbt2[21]);
        kbt2[10] = hi_438;
        rr_169 = fmaxf(rr_169, lo_439);
        float hi_441 = fmaxf(kbt2[11], kbt2[20]);
        float lo_442 = fminf(kbt2[11], kbt2[20]);
        kbt2[11] = hi_441;
        rr_169 = fmaxf(rr_169, lo_442);
        float hi_444 = fmaxf(kbt2[12], kbt2[19]);
        float lo_445 = fminf(kbt2[12], kbt2[19]);
        kbt2[12] = hi_444;
        rr_169 = fmaxf(rr_169, lo_445);
        float hi_446 = fmaxf(kbt2[13], kbt2[18]);
        float lo_447 = fminf(kbt2[13], kbt2[18]);
        kbt2[13] = hi_446;
        rr_169 = fmaxf(rr_169, lo_447);
        float hi_448 = fmaxf(kbt2[14], kbt2[17]);
        float lo_449 = fminf(kbt2[14], kbt2[17]);
        kbt2[14] = hi_448;
        rr_169 = fmaxf(rr_169, lo_449);
        float hi_450 = fmaxf(kbt2[15], kbt2[16]);
        float lo_451 = fminf(kbt2[15], kbt2[16]);
        kbt2[15] = hi_450;
        rr_169 = fmaxf(rr_169, lo_451);
        float hi_452 = fmaxf(kbt2[0], kbt2[8]);
        float lo_453 = fminf(kbt2[0], kbt2[8]);
        kbt2[0] = hi_452;
        kbt2[8] = lo_453;
        float hi_454 = fmaxf(kbt2[1], kbt2[9]);
        float lo_455 = fminf(kbt2[1], kbt2[9]);
        kbt2[1] = hi_454;
        kbt2[9] = lo_455;
        float hi_457 = fmaxf(kbt2[2], kbt2[10]);
        float lo_458 = fminf(kbt2[2], kbt2[10]);
        kbt2[2] = hi_457;
        kbt2[10] = lo_458;
        float hi_460 = fmaxf(kbt2[3], kbt2[11]);
        float lo_461 = fminf(kbt2[3], kbt2[11]);
        kbt2[3] = hi_460;
        kbt2[11] = lo_461;
        float hi_463 = fmaxf(kbt2[4], kbt2[12]);
        float lo_464 = fminf(kbt2[4], kbt2[12]);
        kbt2[4] = hi_463;
        kbt2[12] = lo_464;
        float hi_465 = fmaxf(kbt2[5], kbt2[13]);
        float lo_466 = fminf(kbt2[5], kbt2[13]);
        kbt2[5] = hi_465;
        kbt2[13] = lo_466;
        float hi_467 = fmaxf(kbt2[6], kbt2[14]);
        float lo_468 = fminf(kbt2[6], kbt2[14]);
        kbt2[6] = hi_467;
        kbt2[14] = lo_468;
        float hi_469 = fmaxf(kbt2[7], kbt2[15]);
        float lo_470 = fminf(kbt2[7], kbt2[15]);
        kbt2[7] = hi_469;
        kbt2[15] = lo_470;
        float hi_471 = fmaxf(kbt2[0], kbt2[4]);
        float lo_472 = fminf(kbt2[0], kbt2[4]);
        kbt2[0] = hi_471;
        kbt2[4] = lo_472;
        float hi_473 = fmaxf(kbt2[1], kbt2[5]);
        float lo_474 = fminf(kbt2[1], kbt2[5]);
        kbt2[1] = hi_473;
        kbt2[5] = lo_474;
        float hi_476 = fmaxf(kbt2[2], kbt2[6]);
        float lo_477 = fminf(kbt2[2], kbt2[6]);
        kbt2[2] = hi_476;
        kbt2[6] = lo_477;
        float hi_479 = fmaxf(kbt2[3], kbt2[7]);
        float lo_480 = fminf(kbt2[3], kbt2[7]);
        kbt2[3] = hi_479;
        kbt2[7] = lo_480;
        float hi_482 = fmaxf(kbt2[8], kbt2[12]);
        float lo_483 = fminf(kbt2[8], kbt2[12]);
        kbt2[8] = hi_482;
        kbt2[12] = lo_483;
        float hi_484 = fmaxf(kbt2[9], kbt2[13]);
        float lo_485 = fminf(kbt2[9], kbt2[13]);
        kbt2[9] = hi_484;
        kbt2[13] = lo_485;
        float hi_486 = fmaxf(kbt2[10], kbt2[14]);
        float lo_487 = fminf(kbt2[10], kbt2[14]);
        kbt2[10] = hi_486;
        kbt2[14] = lo_487;
        float hi_488 = fmaxf(kbt2[11], kbt2[15]);
        float lo_489 = fminf(kbt2[11], kbt2[15]);
        kbt2[11] = hi_488;
        kbt2[15] = lo_489;
        float hi_490 = fmaxf(kbt2[0], kbt2[2]);
        float lo_491 = fminf(kbt2[0], kbt2[2]);
        kbt2[0] = hi_490;
        kbt2[2] = lo_491;
        float hi_492 = fmaxf(kbt2[1], kbt2[3]);
        float lo_493 = fminf(kbt2[1], kbt2[3]);
        kbt2[1] = hi_492;
        kbt2[3] = lo_493;
        float hi_495 = fmaxf(kbt2[4], kbt2[6]);
        float lo_496 = fminf(kbt2[4], kbt2[6]);
        kbt2[4] = hi_495;
        kbt2[6] = lo_496;
        float hi_498 = fmaxf(kbt2[5], kbt2[7]);
        float lo_499 = fminf(kbt2[5], kbt2[7]);
        kbt2[5] = hi_498;
        kbt2[7] = lo_499;
        float hi_501 = fmaxf(kbt2[8], kbt2[10]);
        float lo_502 = fminf(kbt2[8], kbt2[10]);
        kbt2[8] = hi_501;
        kbt2[10] = lo_502;
        float hi_503 = fmaxf(kbt2[9], kbt2[11]);
        float lo_504 = fminf(kbt2[9], kbt2[11]);
        kbt2[9] = hi_503;
        kbt2[11] = lo_504;
        float hi_505 = fmaxf(kbt2[12], kbt2[14]);
        float lo_506 = fminf(kbt2[12], kbt2[14]);
        kbt2[12] = hi_505;
        kbt2[14] = lo_506;
        float hi_507 = fmaxf(kbt2[13], kbt2[15]);
        float lo_508 = fminf(kbt2[13], kbt2[15]);
        kbt2[13] = hi_507;
        kbt2[15] = lo_508;
        float hi_509 = fmaxf(kbt2[0], kbt2[1]);
        float lo_510 = fminf(kbt2[0], kbt2[1]);
        kbt2[0] = hi_509;
        kbt2[1] = lo_510;
        float hi_511 = fmaxf(kbt2[2], kbt2[3]);
        float lo_512 = fminf(kbt2[2], kbt2[3]);
        kbt2[2] = hi_511;
        kbt2[3] = lo_512;
        float hi_514 = fmaxf(kbt2[4], kbt2[5]);
        float lo_515 = fminf(kbt2[4], kbt2[5]);
        kbt2[4] = hi_514;
        kbt2[5] = lo_515;
        float hi_517 = fmaxf(kbt2[6], kbt2[7]);
        float lo_518 = fminf(kbt2[6], kbt2[7]);
        kbt2[6] = hi_517;
        kbt2[7] = lo_518;
        float hi_520 = fmaxf(kbt2[8], kbt2[9]);
        float lo_521 = fminf(kbt2[8], kbt2[9]);
        kbt2[8] = hi_520;
        kbt2[9] = lo_521;
        float hi_522 = fmaxf(kbt2[10], kbt2[11]);
        float lo_523 = fminf(kbt2[10], kbt2[11]);
        kbt2[10] = hi_522;
        kbt2[11] = lo_523;
        float hi_524 = fmaxf(kbt2[12], kbt2[13]);
        float lo_525 = fminf(kbt2[12], kbt2[13]);
        kbt2[12] = hi_524;
        kbt2[13] = lo_525;
        float hi_526 = fmaxf(kbt2[14], kbt2[15]);
        float lo_527 = fminf(kbt2[14], kbt2[15]);
        kbt2[14] = hi_526;
        kbt2[15] = lo_527;
        r2o = rr_169;
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
      float r_1 = neg_inf;
      float V_2[8];
      int s0_3 = (sg * 16 * 16 + cg) * 17;
      int s1_4 = ((sg * 16 + 8) * 16 + cg) * 17;
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
      int s0_22 = ((sg * 16 + 1) * 16 + cg) * 17;
      int s1_23 = ((sg * 16 + 1 + 8) * 16 + cg) * 17;
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
      float hi_36 = fmaxf(cur_28, pv_35);
      float lo_37 = fminf(cur_28, pv_35);
      cur_28 = ((up[2] != 0) ? hi_36 : lo_37);
      float pv_38 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 1);
      float hi_39 = fmaxf(cur_28, pv_38);
      float lo_40 = fminf(cur_28, pv_38);
      cur_28 = ((up[3] != 0) ? hi_39 : lo_40);
      V_2[1] = cur_28;
      int s0_41 = ((sg * 16 + 2) * 16 + cg) * 17;
      int s1_42 = ((sg * 16 + 2 + 8) * 16 + cg) * 17;
      float x0_43 = pub[s0_41 + ln];
      float y0_44 = pub[s1_42 + lnr];
      float lo0_45 = fminf(x0_43, y0_44);
      r_1 = fmaxf(r_1, lo0_45);
      float hi0_46 = fmaxf(x0_43, y0_44);
      float cur_47 = hi0_46;
      float pv_48 = __shfl_xor_sync(0xFFFFFFFF, cur_47, 8);
      float hi_49 = fmaxf(cur_47, pv_48);
      float lo_50 = fminf(cur_47, pv_48);
      cur_47 = ((up[0] != 0) ? hi_49 : lo_50);
      float pv_51 = __shfl_xor_sync(0xFFFFFFFF, cur_47, 4);
      float hi_52 = fmaxf(cur_47, pv_51);
      float lo_53 = fminf(cur_47, pv_51);
      cur_47 = ((up[1] != 0) ? hi_52 : lo_53);
      float pv_54 = __shfl_xor_sync(0xFFFFFFFF, cur_47, 2);
      float hi_55 = fmaxf(cur_47, pv_54);
      float lo_56 = fminf(cur_47, pv_54);
      cur_47 = ((up[2] != 0) ? hi_55 : lo_56);
      float pv_57 = __shfl_xor_sync(0xFFFFFFFF, cur_47, 1);
      float hi_58 = fmaxf(cur_47, pv_57);
      float lo_59 = fminf(cur_47, pv_57);
      cur_47 = ((up[3] != 0) ? hi_58 : lo_59);
      V_2[2] = cur_47;
      int s0_60 = ((sg * 16 + 3) * 16 + cg) * 17;
      int s1_61 = ((sg * 16 + 3 + 8) * 16 + cg) * 17;
      float x0_62 = pub[s0_60 + ln];
      float y0_63 = pub[s1_61 + lnr];
      float lo0_64 = fminf(x0_62, y0_63);
      r_1 = fmaxf(r_1, lo0_64);
      float hi0_65 = fmaxf(x0_62, y0_63);
      float cur_66 = hi0_65;
      float pv_67 = __shfl_xor_sync(0xFFFFFFFF, cur_66, 8);
      float hi_69 = fmaxf(cur_66, pv_67);
      float lo_70 = fminf(cur_66, pv_67);
      cur_66 = ((up[0] != 0) ? hi_69 : lo_70);
      float pv_71 = __shfl_xor_sync(0xFFFFFFFF, cur_66, 4);
      float hi_73 = fmaxf(cur_66, pv_71);
      float lo_74 = fminf(cur_66, pv_71);
      cur_66 = ((up[1] != 0) ? hi_73 : lo_74);
      float pv_75 = __shfl_xor_sync(0xFFFFFFFF, cur_66, 2);
      float hi_77 = fmaxf(cur_66, pv_75);
      float lo_78 = fminf(cur_66, pv_75);
      cur_66 = ((up[2] != 0) ? hi_77 : lo_78);
      float pv_79 = __shfl_xor_sync(0xFFFFFFFF, cur_66, 1);
      float hi_81 = fmaxf(cur_66, pv_79);
      float lo_82 = fminf(cur_66, pv_79);
      cur_66 = ((up[3] != 0) ? hi_81 : lo_82);
      V_2[3] = cur_66;
      int s0_83 = ((sg * 16 + 4) * 16 + cg) * 17;
      int s1_84 = ((sg * 16 + 4 + 8) * 16 + cg) * 17;
      float x0_85 = pub[s0_83 + ln];
      float y0_86 = pub[s1_84 + lnr];
      float lo0_87 = fminf(x0_85, y0_86);
      r_1 = fmaxf(r_1, lo0_87);
      float hi0_88 = fmaxf(x0_85, y0_86);
      float cur_89 = hi0_88;
      float pv_90 = __shfl_xor_sync(0xFFFFFFFF, cur_89, 8);
      float hi_91 = fmaxf(cur_89, pv_90);
      float lo_92 = fminf(cur_89, pv_90);
      cur_89 = ((up[0] != 0) ? hi_91 : lo_92);
      float pv_93 = __shfl_xor_sync(0xFFFFFFFF, cur_89, 4);
      float hi_95 = fmaxf(cur_89, pv_93);
      float lo_96 = fminf(cur_89, pv_93);
      cur_89 = ((up[1] != 0) ? hi_95 : lo_96);
      float pv_97 = __shfl_xor_sync(0xFFFFFFFF, cur_89, 2);
      float hi_99 = fmaxf(cur_89, pv_97);
      float lo_100 = fminf(cur_89, pv_97);
      cur_89 = ((up[2] != 0) ? hi_99 : lo_100);
      float pv_101 = __shfl_xor_sync(0xFFFFFFFF, cur_89, 1);
      float hi_103 = fmaxf(cur_89, pv_101);
      float lo_104 = fminf(cur_89, pv_101);
      cur_89 = ((up[3] != 0) ? hi_103 : lo_104);
      V_2[4] = cur_89;
      int s0_105 = ((sg * 16 + 5) * 16 + cg) * 17;
      int s1_106 = ((sg * 16 + 5 + 8) * 16 + cg) * 17;
      float x0_107 = pub[s0_105 + ln];
      float y0_108 = pub[s1_106 + lnr];
      float lo0_109 = fminf(x0_107, y0_108);
      r_1 = fmaxf(r_1, lo0_109);
      float hi0_110 = fmaxf(x0_107, y0_108);
      float cur_111 = hi0_110;
      float pv_112 = __shfl_xor_sync(0xFFFFFFFF, cur_111, 8);
      float hi_113 = fmaxf(cur_111, pv_112);
      float lo_114 = fminf(cur_111, pv_112);
      cur_111 = ((up[0] != 0) ? hi_113 : lo_114);
      float pv_115 = __shfl_xor_sync(0xFFFFFFFF, cur_111, 4);
      float hi_117 = fmaxf(cur_111, pv_115);
      float lo_118 = fminf(cur_111, pv_115);
      cur_111 = ((up[1] != 0) ? hi_117 : lo_118);
      float pv_119 = __shfl_xor_sync(0xFFFFFFFF, cur_111, 2);
      float hi_121 = fmaxf(cur_111, pv_119);
      float lo_122 = fminf(cur_111, pv_119);
      cur_111 = ((up[2] != 0) ? hi_121 : lo_122);
      float pv_123 = __shfl_xor_sync(0xFFFFFFFF, cur_111, 1);
      float hi_125 = fmaxf(cur_111, pv_123);
      float lo_126 = fminf(cur_111, pv_123);
      cur_111 = ((up[3] != 0) ? hi_125 : lo_126);
      V_2[5] = cur_111;
      int s0_127 = ((sg * 16 + 6) * 16 + cg) * 17;
      int s1_128 = ((sg * 16 + 6 + 8) * 16 + cg) * 17;
      float x0_129 = pub[s0_127 + ln];
      float y0_130 = pub[s1_128 + lnr];
      float lo0_131 = fminf(x0_129, y0_130);
      r_1 = fmaxf(r_1, lo0_131);
      float hi0_132 = fmaxf(x0_129, y0_130);
      float cur_133 = hi0_132;
      float pv_134 = __shfl_xor_sync(0xFFFFFFFF, cur_133, 8);
      float hi_135 = fmaxf(cur_133, pv_134);
      float lo_136 = fminf(cur_133, pv_134);
      cur_133 = ((up[0] != 0) ? hi_135 : lo_136);
      float pv_137 = __shfl_xor_sync(0xFFFFFFFF, cur_133, 4);
      float hi_139 = fmaxf(cur_133, pv_137);
      float lo_140 = fminf(cur_133, pv_137);
      cur_133 = ((up[1] != 0) ? hi_139 : lo_140);
      float pv_141 = __shfl_xor_sync(0xFFFFFFFF, cur_133, 2);
      float hi_143 = fmaxf(cur_133, pv_141);
      float lo_144 = fminf(cur_133, pv_141);
      cur_133 = ((up[2] != 0) ? hi_143 : lo_144);
      float pv_145 = __shfl_xor_sync(0xFFFFFFFF, cur_133, 1);
      float hi_147 = fmaxf(cur_133, pv_145);
      float lo_148 = fminf(cur_133, pv_145);
      cur_133 = ((up[3] != 0) ? hi_147 : lo_148);
      V_2[6] = cur_133;
      int s0_149 = ((sg * 16 + 7) * 16 + cg) * 17;
      int s1_150 = ((sg * 16 + 7 + 8) * 16 + cg) * 17;
      float x0_151 = pub[s0_149 + ln];
      float y0_152 = pub[s1_150 + lnr];
      float lo0_153 = fminf(x0_151, y0_152);
      r_1 = fmaxf(r_1, lo0_153);
      float hi0_154 = fmaxf(x0_151, y0_152);
      float cur_155 = hi0_154;
      float pv_156 = __shfl_xor_sync(0xFFFFFFFF, cur_155, 8);
      float hi_157 = fmaxf(cur_155, pv_156);
      float lo_158 = fminf(cur_155, pv_156);
      cur_155 = ((up[0] != 0) ? hi_157 : lo_158);
      float pv_159 = __shfl_xor_sync(0xFFFFFFFF, cur_155, 4);
      float hi_161 = fmaxf(cur_155, pv_159);
      float lo_162 = fminf(cur_155, pv_159);
      cur_155 = ((up[1] != 0) ? hi_161 : lo_162);
      float pv_163 = __shfl_xor_sync(0xFFFFFFFF, cur_155, 2);
      float hi_165 = fmaxf(cur_155, pv_163);
      float lo_166 = fminf(cur_155, pv_163);
      cur_155 = ((up[2] != 0) ? hi_165 : lo_166);
      float pv_167 = __shfl_xor_sync(0xFFFFFFFF, cur_155, 1);
      float hi_169 = fmaxf(cur_155, pv_167);
      float lo_170 = fminf(cur_155, pv_167);
      cur_155 = ((up[3] != 0) ? hi_169 : lo_170);
      V_2[7] = cur_155;
      float rs_171 = pub[((sg * 16 + ln) * 16 + cg) * 17 + 16];
      r_1 = fmaxf(r_1, rs_171);
      float y1_172 = __shfl_xor_sync(0xFFFFFFFF, V_2[4], 15);
      float lo1_173 = fminf(V_2[0], y1_172);
      r_1 = fmaxf(r_1, lo1_173);
      float hi1_174 = fmaxf(V_2[0], y1_172);
      float cur_175 = hi1_174;
      float pv_176 = __shfl_xor_sync(0xFFFFFFFF, cur_175, 8);
      float hi_177_1 = fmaxf(cur_175, pv_176);
      float lo_178_1 = fminf(cur_175, pv_176);
      cur_175 = ((up[0] != 0) ? hi_177_1 : lo_178_1);
      float pv_179 = __shfl_xor_sync(0xFFFFFFFF, cur_175, 4);
      float hi_181_1 = fmaxf(cur_175, pv_179);
      float lo_182_1 = fminf(cur_175, pv_179);
      cur_175 = ((up[1] != 0) ? hi_181_1 : lo_182_1);
      float pv_183 = __shfl_xor_sync(0xFFFFFFFF, cur_175, 2);
      float hi_185_1 = fmaxf(cur_175, pv_183);
      float lo_186_1 = fminf(cur_175, pv_183);
      cur_175 = ((up[2] != 0) ? hi_185_1 : lo_186_1);
      float pv_187 = __shfl_xor_sync(0xFFFFFFFF, cur_175, 1);
      float hi_189_1 = fmaxf(cur_175, pv_187);
      float lo_190_1 = fminf(cur_175, pv_187);
      cur_175 = ((up[3] != 0) ? hi_189_1 : lo_190_1);
      V_2[0] = cur_175;
      float y1_191 = __shfl_xor_sync(0xFFFFFFFF, V_2[5], 15);
      float lo1_192 = fminf(V_2[1], y1_191);
      r_1 = fmaxf(r_1, lo1_192);
      float hi1_193 = fmaxf(V_2[1], y1_191);
      float cur_194 = hi1_193;
      float pv_195 = __shfl_xor_sync(0xFFFFFFFF, cur_194, 8);
      float hi_197_1 = fmaxf(cur_194, pv_195);
      float lo_198_1 = fminf(cur_194, pv_195);
      cur_194 = ((up[0] != 0) ? hi_197_1 : lo_198_1);
      float pv_199 = __shfl_xor_sync(0xFFFFFFFF, cur_194, 4);
      float hi_201_1 = fmaxf(cur_194, pv_199);
      float lo_202_1 = fminf(cur_194, pv_199);
      cur_194 = ((up[1] != 0) ? hi_201_1 : lo_202_1);
      float pv_203 = __shfl_xor_sync(0xFFFFFFFF, cur_194, 2);
      float hi_205_1 = fmaxf(cur_194, pv_203);
      float lo_206_1 = fminf(cur_194, pv_203);
      cur_194 = ((up[2] != 0) ? hi_205_1 : lo_206_1);
      float pv_207 = __shfl_xor_sync(0xFFFFFFFF, cur_194, 1);
      float hi_209_1 = fmaxf(cur_194, pv_207);
      float lo_210_1 = fminf(cur_194, pv_207);
      cur_194 = ((up[3] != 0) ? hi_209_1 : lo_210_1);
      V_2[1] = cur_194;
      float y1_211 = __shfl_xor_sync(0xFFFFFFFF, V_2[6], 15);
      float lo1_212 = fminf(V_2[2], y1_211);
      r_1 = fmaxf(r_1, lo1_212);
      float hi1_213 = fmaxf(V_2[2], y1_211);
      float cur_214 = hi1_213;
      float pv_215 = __shfl_xor_sync(0xFFFFFFFF, cur_214, 8);
      float hi_217_1 = fmaxf(cur_214, pv_215);
      float lo_218_1 = fminf(cur_214, pv_215);
      cur_214 = ((up[0] != 0) ? hi_217_1 : lo_218_1);
      float pv_219 = __shfl_xor_sync(0xFFFFFFFF, cur_214, 4);
      float hi_221_1 = fmaxf(cur_214, pv_219);
      float lo_222_1 = fminf(cur_214, pv_219);
      cur_214 = ((up[1] != 0) ? hi_221_1 : lo_222_1);
      float pv_223 = __shfl_xor_sync(0xFFFFFFFF, cur_214, 2);
      float hi_225_1 = fmaxf(cur_214, pv_223);
      float lo_226_1 = fminf(cur_214, pv_223);
      cur_214 = ((up[2] != 0) ? hi_225_1 : lo_226_1);
      float pv_227 = __shfl_xor_sync(0xFFFFFFFF, cur_214, 1);
      float hi_229_1 = fmaxf(cur_214, pv_227);
      float lo_230_1 = fminf(cur_214, pv_227);
      cur_214 = ((up[3] != 0) ? hi_229_1 : lo_230_1);
      V_2[2] = cur_214;
      float y1_231 = __shfl_xor_sync(0xFFFFFFFF, V_2[7], 15);
      float lo1_232 = fminf(V_2[3], y1_231);
      r_1 = fmaxf(r_1, lo1_232);
      float hi1_233 = fmaxf(V_2[3], y1_231);
      float cur_234 = hi1_233;
      float pv_235 = __shfl_xor_sync(0xFFFFFFFF, cur_234, 8);
      float hi_237_1 = fmaxf(cur_234, pv_235);
      float lo_238_1 = fminf(cur_234, pv_235);
      cur_234 = ((up[0] != 0) ? hi_237_1 : lo_238_1);
      float pv_239 = __shfl_xor_sync(0xFFFFFFFF, cur_234, 4);
      float hi_241_1 = fmaxf(cur_234, pv_239);
      float lo_242_1 = fminf(cur_234, pv_239);
      cur_234 = ((up[1] != 0) ? hi_241_1 : lo_242_1);
      float pv_243 = __shfl_xor_sync(0xFFFFFFFF, cur_234, 2);
      float hi_245_1 = fmaxf(cur_234, pv_243);
      float lo_246_1 = fminf(cur_234, pv_243);
      cur_234 = ((up[2] != 0) ? hi_245_1 : lo_246_1);
      float pv_247 = __shfl_xor_sync(0xFFFFFFFF, cur_234, 1);
      float hi_249_1 = fmaxf(cur_234, pv_247);
      float lo_250_1 = fminf(cur_234, pv_247);
      cur_234 = ((up[3] != 0) ? hi_249_1 : lo_250_1);
      V_2[3] = cur_234;
      float y1_251 = __shfl_xor_sync(0xFFFFFFFF, V_2[2], 15);
      float lo1_252 = fminf(V_2[0], y1_251);
      r_1 = fmaxf(r_1, lo1_252);
      float hi1_253 = fmaxf(V_2[0], y1_251);
      float cur_254 = hi1_253;
      float pv_255 = __shfl_xor_sync(0xFFFFFFFF, cur_254, 8);
      float hi_257_1 = fmaxf(cur_254, pv_255);
      float lo_258_1 = fminf(cur_254, pv_255);
      cur_254 = ((up[0] != 0) ? hi_257_1 : lo_258_1);
      float pv_259 = __shfl_xor_sync(0xFFFFFFFF, cur_254, 4);
      float hi_261_1 = fmaxf(cur_254, pv_259);
      float lo_262_1 = fminf(cur_254, pv_259);
      cur_254 = ((up[1] != 0) ? hi_261_1 : lo_262_1);
      float pv_263 = __shfl_xor_sync(0xFFFFFFFF, cur_254, 2);
      float hi_265_1 = fmaxf(cur_254, pv_263);
      float lo_266_1 = fminf(cur_254, pv_263);
      cur_254 = ((up[2] != 0) ? hi_265_1 : lo_266_1);
      float pv_267 = __shfl_xor_sync(0xFFFFFFFF, cur_254, 1);
      float hi_269_1 = fmaxf(cur_254, pv_267);
      float lo_270_1 = fminf(cur_254, pv_267);
      cur_254 = ((up[3] != 0) ? hi_269_1 : lo_270_1);
      V_2[0] = cur_254;
      float y1_271 = __shfl_xor_sync(0xFFFFFFFF, V_2[3], 15);
      float lo1_272 = fminf(V_2[1], y1_271);
      r_1 = fmaxf(r_1, lo1_272);
      float hi1_273 = fmaxf(V_2[1], y1_271);
      float cur_274 = hi1_273;
      float pv_275 = __shfl_xor_sync(0xFFFFFFFF, cur_274, 8);
      float hi_277_1 = fmaxf(cur_274, pv_275);
      float lo_278_1 = fminf(cur_274, pv_275);
      cur_274 = ((up[0] != 0) ? hi_277_1 : lo_278_1);
      float pv_279 = __shfl_xor_sync(0xFFFFFFFF, cur_274, 4);
      float hi_281_1 = fmaxf(cur_274, pv_279);
      float lo_282_1 = fminf(cur_274, pv_279);
      cur_274 = ((up[1] != 0) ? hi_281_1 : lo_282_1);
      float pv_283 = __shfl_xor_sync(0xFFFFFFFF, cur_274, 2);
      float hi_285_1 = fmaxf(cur_274, pv_283);
      float lo_286_1 = fminf(cur_274, pv_283);
      cur_274 = ((up[2] != 0) ? hi_285_1 : lo_286_1);
      float pv_287 = __shfl_xor_sync(0xFFFFFFFF, cur_274, 1);
      float hi_289_1 = fmaxf(cur_274, pv_287);
      float lo_290_1 = fminf(cur_274, pv_287);
      cur_274 = ((up[3] != 0) ? hi_289_1 : lo_290_1);
      V_2[1] = cur_274;
      float yl_291 = __shfl_xor_sync(0xFFFFFFFF, V_2[1], 15);
      float lol_292 = fminf(V_2[0], yl_291);
      r_1 = fmaxf(r_1, lol_292);
      float hil_293 = fmaxf(V_2[0], yl_291);
      V_2[0] = hil_293;
      float K_294 = V_2[0];
      rr2x[0] = r_1;
      K2 = K_294;
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
            float hi_32 = fmaxf(xsl[0], xsl[1]);
            float lo_33 = fminf(xsl[0], xsl[1]);
            xsl[0] = hi_32;
            xsl[1] = lo_33;
            float hi_34 = fmaxf(xsl[2], xsl[3]);
            float lo_35 = fminf(xsl[2], xsl[3]);
            xsl[2] = hi_34;
            xsl[3] = lo_35;
            float hi_36_1 = fmaxf(xsl[4], xsl[5]);
            float lo_37_1 = fminf(xsl[4], xsl[5]);
            xsl[4] = hi_36_1;
            xsl[5] = lo_37_1;
            float hi_38 = fmaxf(xsl[6], xsl[8]);
            float lo_39 = fminf(xsl[6], xsl[8]);
            xsl[6] = hi_38;
            xsl[8] = lo_39;
            float hi_40 = fmaxf(xsl[7], xsl[9]);
            float lo_41 = fminf(xsl[7], xsl[9]);
            xsl[7] = hi_40;
            xsl[9] = lo_41;
            float hi_42 = fmaxf(xsl[10], xsl[11]);
            float lo_43 = fminf(xsl[10], xsl[11]);
            xsl[10] = hi_42;
            xsl[11] = lo_43;
            float hi_44 = fmaxf(xsl[12], xsl[13]);
            float lo_45 = fminf(xsl[12], xsl[13]);
            xsl[12] = hi_44;
            xsl[13] = lo_45;
            float hi_46 = fmaxf(xsl[14], xsl[15]);
            float lo_47 = fminf(xsl[14], xsl[15]);
            xsl[14] = hi_46;
            xsl[15] = lo_47;
            float hi_48 = fmaxf(xsl[0], xsl[2]);
            float lo_49 = fminf(xsl[0], xsl[2]);
            xsl[0] = hi_48;
            xsl[2] = lo_49;
            float hi_50 = fmaxf(xsl[1], xsl[3]);
            float lo_51 = fminf(xsl[1], xsl[3]);
            xsl[1] = hi_50;
            xsl[3] = lo_51;
            float hi_52_1 = fmaxf(xsl[4], xsl[10]);
            float lo_53_1 = fminf(xsl[4], xsl[10]);
            xsl[4] = hi_52_1;
            xsl[10] = lo_53_1;
            float hi_54 = fmaxf(xsl[5], xsl[11]);
            float lo_55 = fminf(xsl[5], xsl[11]);
            xsl[5] = hi_54;
            xsl[11] = lo_55;
            float hi_56 = fmaxf(xsl[6], xsl[7]);
            float lo_57 = fminf(xsl[6], xsl[7]);
            xsl[6] = hi_56;
            xsl[7] = lo_57;
            float hi_58_1 = fmaxf(xsl[8], xsl[9]);
            float lo_59_1 = fminf(xsl[8], xsl[9]);
            xsl[8] = hi_58_1;
            xsl[9] = lo_59_1;
            float hi_60 = fmaxf(xsl[12], xsl[14]);
            float lo_61 = fminf(xsl[12], xsl[14]);
            xsl[12] = hi_60;
            xsl[14] = lo_61;
            float hi_63 = fmaxf(xsl[13], xsl[15]);
            float lo_64 = fminf(xsl[13], xsl[15]);
            xsl[13] = hi_63;
            xsl[15] = lo_64;
            float hi_65 = fmaxf(xsl[1], xsl[2]);
            float lo_66 = fminf(xsl[1], xsl[2]);
            xsl[1] = hi_65;
            xsl[2] = lo_66;
            float hi_67 = fmaxf(xsl[3], xsl[12]);
            float lo_68 = fminf(xsl[3], xsl[12]);
            xsl[3] = hi_67;
            xsl[12] = lo_68;
            float hi_69_1 = fmaxf(xsl[4], xsl[6]);
            float lo_70_1 = fminf(xsl[4], xsl[6]);
            xsl[4] = hi_69_1;
            xsl[6] = lo_70_1;
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
            float hi_85 = fmaxf(xsl[7], xsl[10]);
            float lo_86 = fminf(xsl[7], xsl[10]);
            xsl[7] = hi_85;
            xsl[10] = lo_86;
            float hi_87 = fmaxf(xsl[9], xsl[13]);
            float lo_88 = fminf(xsl[9], xsl[13]);
            xsl[9] = hi_87;
            xsl[13] = lo_88;
            float hi_89 = fmaxf(xsl[11], xsl[14]);
            float lo_90 = fminf(xsl[11], xsl[14]);
            xsl[11] = hi_89;
            xsl[14] = lo_90;
            float hi_91_1 = fmaxf(xsl[2], xsl[4]);
            float lo_92_1 = fminf(xsl[2], xsl[4]);
            xsl[2] = hi_91_1;
            xsl[4] = lo_92_1;
            float hi_93 = fmaxf(xsl[3], xsl[6]);
            float lo_94 = fminf(xsl[3], xsl[6]);
            xsl[3] = hi_93;
            xsl[6] = lo_94;
            float hi_95_1 = fmaxf(xsl[9], xsl[12]);
            float lo_96_1 = fminf(xsl[9], xsl[12]);
            xsl[9] = hi_95_1;
            xsl[12] = lo_96_1;
            float hi_97 = fmaxf(xsl[11], xsl[13]);
            float lo_98 = fminf(xsl[11], xsl[13]);
            xsl[11] = hi_97;
            xsl[13] = lo_98;
            float hi_99_1 = fmaxf(xsl[3], xsl[5]);
            float lo_100_1 = fminf(xsl[3], xsl[5]);
            xsl[3] = hi_99_1;
            xsl[5] = lo_100_1;
            float hi_101 = fmaxf(xsl[6], xsl[8]);
            float lo_102 = fminf(xsl[6], xsl[8]);
            xsl[6] = hi_101;
            xsl[8] = lo_102;
            float hi_103_1 = fmaxf(xsl[7], xsl[9]);
            float lo_104_1 = fminf(xsl[7], xsl[9]);
            xsl[7] = hi_103_1;
            xsl[9] = lo_104_1;
            float hi_105 = fmaxf(xsl[10], xsl[12]);
            float lo_106 = fminf(xsl[10], xsl[12]);
            xsl[10] = hi_105;
            xsl[12] = lo_106;
            float hi_107 = fmaxf(xsl[3], xsl[4]);
            float lo_108 = fminf(xsl[3], xsl[4]);
            xsl[3] = hi_107;
            xsl[4] = lo_108;
            float hi_109 = fmaxf(xsl[5], xsl[6]);
            float lo_110 = fminf(xsl[5], xsl[6]);
            xsl[5] = hi_109;
            xsl[6] = lo_110;
            float hi_111 = fmaxf(xsl[7], xsl[8]);
            float lo_112 = fminf(xsl[7], xsl[8]);
            xsl[7] = hi_111;
            xsl[8] = lo_112;
            float hi_113_1 = fmaxf(xsl[9], xsl[10]);
            float lo_114_1 = fminf(xsl[9], xsl[10]);
            xsl[9] = hi_113_1;
            xsl[10] = lo_114_1;
            float hi_115 = fmaxf(xsl[11], xsl[12]);
            float lo_116 = fminf(xsl[11], xsl[12]);
            xsl[11] = hi_115;
            xsl[12] = lo_116;
            float hi_117_1 = fmaxf(xsl[6], xsl[7]);
            float lo_118_1 = fminf(xsl[6], xsl[7]);
            xsl[6] = hi_117_1;
            xsl[7] = lo_118_1;
            float hi_119 = fmaxf(xsl[8], xsl[9]);
            float lo_120 = fminf(xsl[8], xsl[9]);
            xsl[8] = hi_119;
            xsl[9] = lo_120;
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

}  // extern "C"
