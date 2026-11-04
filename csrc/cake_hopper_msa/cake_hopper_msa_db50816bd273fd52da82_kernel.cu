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
#define SMEM_QCOL_STAGE_BYTES 4
#define SMEM_QCOL_STRIDE 4
#define SMEM_FLAGW_OFF 4
#define SMEM_FLAGW_STAGE_BYTES 16
#define SMEM_FLAGW_STRIDE 16
#define SMEM_CCNT_OFF 2452
#define SMEM_CCNT_STAGE_BYTES 4
#define SMEM_CCNT_STRIDE 4
#define SMEM_LIMC_OFF 2456
#define SMEM_LIMC_STAGE_BYTES 4
#define SMEM_LIMC_STRIDE 4
#define SMEM_CBUF_OFF 2460
#define SMEM_CBUF_STAGE_BYTES 64
#define SMEM_CBUF_STRIDE 64
#define SMEM_PUB_OFF 20
#define SMEM_PUB_STAGE_BYTES 2176
#define SMEM_PUB_STRIDE 2176
#define SMEM_Q2_OFF 2196
#define SMEM_Q2_STAGE_BYTES 256
#define SMEM_Q2_STRIDE 256
#define SMEM_TOTAL 2560
#define THREADS 32

extern "C" {

__global__ __launch_bounds__(32) void kernel_cake_hopper_msa_db50816bd273fd52da82(
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

  // === Task calls (dependency order) ===
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
  unsigned int nb[16];
#pragma unroll 1
  for (int j = 0; j < num_chunks; j++) {
    int t0_0 = ((j + 1) * 32 + w) * 16;
    long long p_1 = cbase + (long long)t0_0 * nq64;
    nb[0] = 4286578688u;
    if (lim > t0_0) {
      nb[0] = S[p_1];
    }
    nb[1] = 4286578688u;
    if (lim > t0_0 + 1) {
      nb[1] = S[p_1 + nq64];
    }
    nb[2] = 4286578688u;
    if (lim > t0_0 + 2) {
      nb[2] = S[p_1 + 2 * nq64];
    }
    nb[3] = 4286578688u;
    if (lim > t0_0 + 3) {
      nb[3] = S[p_1 + 3 * nq64];
    }
    nb[4] = 4286578688u;
    if (lim > t0_0 + 4) {
      nb[4] = S[p_1 + 4 * nq64];
    }
    nb[5] = 4286578688u;
    if (lim > t0_0 + 5) {
      nb[5] = S[p_1 + 5 * nq64];
    }
    nb[6] = 4286578688u;
    if (lim > t0_0 + 6) {
      nb[6] = S[p_1 + 6 * nq64];
    }
    nb[7] = 4286578688u;
    if (lim > t0_0 + 7) {
      nb[7] = S[p_1 + 7 * nq64];
    }
    nb[8] = 4286578688u;
    if (lim > t0_0 + 8) {
      nb[8] = S[p_1 + 8 * nq64];
    }
    nb[9] = 4286578688u;
    if (lim > t0_0 + 9) {
      nb[9] = S[p_1 + 9 * nq64];
    }
    nb[10] = 4286578688u;
    if (lim > t0_0 + 10) {
      nb[10] = S[p_1 + 10 * nq64];
    }
    nb[11] = 4286578688u;
    if (lim > t0_0 + 11) {
      nb[11] = S[p_1 + 11 * nq64];
    }
    nb[12] = 4286578688u;
    if (lim > t0_0 + 12) {
      nb[12] = S[p_1 + 12 * nq64];
    }
    nb[13] = 4286578688u;
    if (lim > t0_0 + 13) {
      nb[13] = S[p_1 + 13 * nq64];
    }
    nb[14] = 4286578688u;
    if (lim > t0_0 + 14) {
      nb[14] = S[p_1 + 14 * nq64];
    }
    nb[15] = 4286578688u;
    if (lim > t0_0 + 15) {
      nb[15] = S[p_1 + 15 * nq64];
    }
    asm volatile("" ::: "memory");
    int t0_2 = (j * 32 + w) * 16;
    float sc = __uint_as_float(cb[0]);
    sc = fmaxf(sc, -1.7014118346046923e+38f);
    sc = fminf(sc, 1.7014118346046923e+38f);
    unsigned int key = __as_u32(sc) & 4294966784u | (unsigned int)t0_2;
    int f = 0;
    if ((t0_2 < fb || lim_fe <= t0_2) && lim > t0_2) {
      f = 1;
    }
    if (f != 0) {
      key = 2139094528 | (unsigned int)t0_2;
    }
    kb[0] = __uint_as_float(key);
    float sc_3 = __uint_as_float(cb[1]);
    sc_3 = fmaxf(sc_3, -1.7014118346046923e+38f);
    sc_3 = fminf(sc_3, 1.7014118346046923e+38f);
    unsigned int key_4 = __as_u32(sc_3) & 4294966784u | (unsigned int)(t0_2 + 1);
    int f_5 = 0;
    if ((t0_2 + 1 < fb || lim_fe <= t0_2 + 1) && lim > t0_2 + 1) {
      f_5 = 1;
    }
    if (f_5 != 0) {
      key_4 = 2139094528 | (unsigned int)(t0_2 + 1);
    }
    kb[1] = __uint_as_float(key_4);
    float sc_6 = __uint_as_float(cb[2]);
    sc_6 = fmaxf(sc_6, -1.7014118346046923e+38f);
    sc_6 = fminf(sc_6, 1.7014118346046923e+38f);
    unsigned int key_7 = __as_u32(sc_6) & 4294966784u | (unsigned int)(t0_2 + 2);
    int f_8 = 0;
    if ((t0_2 + 2 < fb || lim_fe <= t0_2 + 2) && lim > t0_2 + 2) {
      f_8 = 1;
    }
    if (f_8 != 0) {
      key_7 = 2139094528 | (unsigned int)(t0_2 + 2);
    }
    kb[2] = __uint_as_float(key_7);
    float sc_9 = __uint_as_float(cb[3]);
    sc_9 = fmaxf(sc_9, -1.7014118346046923e+38f);
    sc_9 = fminf(sc_9, 1.7014118346046923e+38f);
    unsigned int key_10 = __as_u32(sc_9) & 4294966784u | (unsigned int)(t0_2 + 3);
    int f_11 = 0;
    if ((t0_2 + 3 < fb || lim_fe <= t0_2 + 3) && lim > t0_2 + 3) {
      f_11 = 1;
    }
    if (f_11 != 0) {
      key_10 = 2139094528 | (unsigned int)(t0_2 + 3);
    }
    kb[3] = __uint_as_float(key_10);
    float sc_12 = __uint_as_float(cb[4]);
    sc_12 = fmaxf(sc_12, -1.7014118346046923e+38f);
    sc_12 = fminf(sc_12, 1.7014118346046923e+38f);
    unsigned int key_13 = __as_u32(sc_12) & 4294966784u | (unsigned int)(t0_2 + 4);
    int f_14 = 0;
    if ((t0_2 + 4 < fb || lim_fe <= t0_2 + 4) && lim > t0_2 + 4) {
      f_14 = 1;
    }
    if (f_14 != 0) {
      key_13 = 2139094528 | (unsigned int)(t0_2 + 4);
    }
    kb[4] = __uint_as_float(key_13);
    float sc_15 = __uint_as_float(cb[5]);
    sc_15 = fmaxf(sc_15, -1.7014118346046923e+38f);
    sc_15 = fminf(sc_15, 1.7014118346046923e+38f);
    unsigned int key_16 = __as_u32(sc_15) & 4294966784u | (unsigned int)(t0_2 + 5);
    int f_17 = 0;
    if ((t0_2 + 5 < fb || lim_fe <= t0_2 + 5) && lim > t0_2 + 5) {
      f_17 = 1;
    }
    if (f_17 != 0) {
      key_16 = 2139094528 | (unsigned int)(t0_2 + 5);
    }
    kb[5] = __uint_as_float(key_16);
    float sc_18 = __uint_as_float(cb[6]);
    sc_18 = fmaxf(sc_18, -1.7014118346046923e+38f);
    sc_18 = fminf(sc_18, 1.7014118346046923e+38f);
    unsigned int key_19 = __as_u32(sc_18) & 4294966784u | (unsigned int)(t0_2 + 6);
    int f_20 = 0;
    if ((t0_2 + 6 < fb || lim_fe <= t0_2 + 6) && lim > t0_2 + 6) {
      f_20 = 1;
    }
    if (f_20 != 0) {
      key_19 = 2139094528 | (unsigned int)(t0_2 + 6);
    }
    kb[6] = __uint_as_float(key_19);
    float sc_21 = __uint_as_float(cb[7]);
    sc_21 = fmaxf(sc_21, -1.7014118346046923e+38f);
    sc_21 = fminf(sc_21, 1.7014118346046923e+38f);
    unsigned int key_22 = __as_u32(sc_21) & 4294966784u | (unsigned int)(t0_2 + 7);
    int f_23 = 0;
    if ((t0_2 + 7 < fb || lim_fe <= t0_2 + 7) && lim > t0_2 + 7) {
      f_23 = 1;
    }
    if (f_23 != 0) {
      key_22 = 2139094528 | (unsigned int)(t0_2 + 7);
    }
    kb[7] = __uint_as_float(key_22);
    float sc_24 = __uint_as_float(cb[8]);
    sc_24 = fmaxf(sc_24, -1.7014118346046923e+38f);
    sc_24 = fminf(sc_24, 1.7014118346046923e+38f);
    unsigned int key_25 = __as_u32(sc_24) & 4294966784u | (unsigned int)(t0_2 + 8);
    int f_26 = 0;
    if ((t0_2 + 8 < fb || lim_fe <= t0_2 + 8) && lim > t0_2 + 8) {
      f_26 = 1;
    }
    if (f_26 != 0) {
      key_25 = 2139094528 | (unsigned int)(t0_2 + 8);
    }
    kb[8] = __uint_as_float(key_25);
    float sc_27 = __uint_as_float(cb[9]);
    sc_27 = fmaxf(sc_27, -1.7014118346046923e+38f);
    sc_27 = fminf(sc_27, 1.7014118346046923e+38f);
    unsigned int key_28 = __as_u32(sc_27) & 4294966784u | (unsigned int)(t0_2 + 9);
    int f_29 = 0;
    if ((t0_2 + 9 < fb || lim_fe <= t0_2 + 9) && lim > t0_2 + 9) {
      f_29 = 1;
    }
    if (f_29 != 0) {
      key_28 = 2139094528 | (unsigned int)(t0_2 + 9);
    }
    kb[9] = __uint_as_float(key_28);
    float sc_30 = __uint_as_float(cb[10]);
    sc_30 = fmaxf(sc_30, -1.7014118346046923e+38f);
    sc_30 = fminf(sc_30, 1.7014118346046923e+38f);
    unsigned int key_31 = __as_u32(sc_30) & 4294966784u | (unsigned int)(t0_2 + 10);
    int f_32 = 0;
    if ((t0_2 + 10 < fb || lim_fe <= t0_2 + 10) && lim > t0_2 + 10) {
      f_32 = 1;
    }
    if (f_32 != 0) {
      key_31 = 2139094528 | (unsigned int)(t0_2 + 10);
    }
    kb[10] = __uint_as_float(key_31);
    float sc_33 = __uint_as_float(cb[11]);
    sc_33 = fmaxf(sc_33, -1.7014118346046923e+38f);
    sc_33 = fminf(sc_33, 1.7014118346046923e+38f);
    unsigned int key_34 = __as_u32(sc_33) & 4294966784u | (unsigned int)(t0_2 + 11);
    int f_35 = 0;
    if ((t0_2 + 11 < fb || lim_fe <= t0_2 + 11) && lim > t0_2 + 11) {
      f_35 = 1;
    }
    if (f_35 != 0) {
      key_34 = 2139094528 | (unsigned int)(t0_2 + 11);
    }
    kb[11] = __uint_as_float(key_34);
    float sc_36 = __uint_as_float(cb[12]);
    sc_36 = fmaxf(sc_36, -1.7014118346046923e+38f);
    sc_36 = fminf(sc_36, 1.7014118346046923e+38f);
    unsigned int key_37 = __as_u32(sc_36) & 4294966784u | (unsigned int)(t0_2 + 12);
    int f_38 = 0;
    if ((t0_2 + 12 < fb || lim_fe <= t0_2 + 12) && lim > t0_2 + 12) {
      f_38 = 1;
    }
    if (f_38 != 0) {
      key_37 = 2139094528 | (unsigned int)(t0_2 + 12);
    }
    kb[12] = __uint_as_float(key_37);
    float sc_39 = __uint_as_float(cb[13]);
    sc_39 = fmaxf(sc_39, -1.7014118346046923e+38f);
    sc_39 = fminf(sc_39, 1.7014118346046923e+38f);
    unsigned int key_40 = __as_u32(sc_39) & 4294966784u | (unsigned int)(t0_2 + 13);
    int f_41 = 0;
    if ((t0_2 + 13 < fb || lim_fe <= t0_2 + 13) && lim > t0_2 + 13) {
      f_41 = 1;
    }
    if (f_41 != 0) {
      key_40 = 2139094528 | (unsigned int)(t0_2 + 13);
    }
    kb[13] = __uint_as_float(key_40);
    float sc_42 = __uint_as_float(cb[14]);
    sc_42 = fmaxf(sc_42, -1.7014118346046923e+38f);
    sc_42 = fminf(sc_42, 1.7014118346046923e+38f);
    unsigned int key_43 = __as_u32(sc_42) & 4294966784u | (unsigned int)(t0_2 + 14);
    int f_44 = 0;
    if ((t0_2 + 14 < fb || lim_fe <= t0_2 + 14) && lim > t0_2 + 14) {
      f_44 = 1;
    }
    if (f_44 != 0) {
      key_43 = 2139094528 | (unsigned int)(t0_2 + 14);
    }
    kb[14] = __uint_as_float(key_43);
    float sc_45 = __uint_as_float(cb[15]);
    sc_45 = fmaxf(sc_45, -1.7014118346046923e+38f);
    sc_45 = fminf(sc_45, 1.7014118346046923e+38f);
    unsigned int key_46 = __as_u32(sc_45) & 4294966784u | (unsigned int)(t0_2 + 15);
    int f_47 = 0;
    if ((t0_2 + 15 < fb || lim_fe <= t0_2 + 15) && lim > t0_2 + 15) {
      f_47 = 1;
    }
    if (f_47 != 0) {
      key_46 = 2139094528 | (unsigned int)(t0_2 + 15);
    }
    kb[15] = __uint_as_float(key_46);
    float hi = fmaxf(kb[0], kb[13]);
    float lo = fminf(kb[0], kb[13]);
    kb[0] = hi;
    kb[13] = lo;
    float hi_48 = fmaxf(kb[1], kb[12]);
    float lo_49 = fminf(kb[1], kb[12]);
    kb[1] = hi_48;
    kb[12] = lo_49;
    float hi_50 = fmaxf(kb[2], kb[15]);
    float lo_51 = fminf(kb[2], kb[15]);
    kb[2] = hi_50;
    kb[15] = lo_51;
    float hi_52 = fmaxf(kb[3], kb[14]);
    float lo_53 = fminf(kb[3], kb[14]);
    kb[3] = hi_52;
    kb[14] = lo_53;
    float hi_54 = fmaxf(kb[4], kb[8]);
    float lo_55 = fminf(kb[4], kb[8]);
    kb[4] = hi_54;
    kb[8] = lo_55;
    float hi_56 = fmaxf(kb[5], kb[6]);
    float lo_57 = fminf(kb[5], kb[6]);
    kb[5] = hi_56;
    kb[6] = lo_57;
    float hi_58 = fmaxf(kb[7], kb[11]);
    float lo_59 = fminf(kb[7], kb[11]);
    kb[7] = hi_58;
    kb[11] = lo_59;
    float hi_60 = fmaxf(kb[9], kb[10]);
    float lo_61 = fminf(kb[9], kb[10]);
    kb[9] = hi_60;
    kb[10] = lo_61;
    float hi_62 = fmaxf(kb[0], kb[5]);
    float lo_63 = fminf(kb[0], kb[5]);
    kb[0] = hi_62;
    kb[5] = lo_63;
    float hi_64 = fmaxf(kb[1], kb[7]);
    float lo_65 = fminf(kb[1], kb[7]);
    kb[1] = hi_64;
    kb[7] = lo_65;
    float hi_66 = fmaxf(kb[2], kb[9]);
    float lo_67 = fminf(kb[2], kb[9]);
    kb[2] = hi_66;
    kb[9] = lo_67;
    float hi_68 = fmaxf(kb[3], kb[4]);
    float lo_69 = fminf(kb[3], kb[4]);
    kb[3] = hi_68;
    kb[4] = lo_69;
    float hi_70 = fmaxf(kb[6], kb[13]);
    float lo_71 = fminf(kb[6], kb[13]);
    kb[6] = hi_70;
    kb[13] = lo_71;
    float hi_72 = fmaxf(kb[8], kb[14]);
    float lo_73 = fminf(kb[8], kb[14]);
    kb[8] = hi_72;
    kb[14] = lo_73;
    float hi_74 = fmaxf(kb[10], kb[15]);
    float lo_75 = fminf(kb[10], kb[15]);
    kb[10] = hi_74;
    kb[15] = lo_75;
    float hi_76 = fmaxf(kb[11], kb[12]);
    float lo_77 = fminf(kb[11], kb[12]);
    kb[11] = hi_76;
    kb[12] = lo_77;
    float hi_78 = fmaxf(kb[0], kb[1]);
    float lo_79 = fminf(kb[0], kb[1]);
    kb[0] = hi_78;
    kb[1] = lo_79;
    float hi_80 = fmaxf(kb[2], kb[3]);
    float lo_81 = fminf(kb[2], kb[3]);
    kb[2] = hi_80;
    kb[3] = lo_81;
    float hi_82 = fmaxf(kb[4], kb[5]);
    float lo_83 = fminf(kb[4], kb[5]);
    kb[4] = hi_82;
    kb[5] = lo_83;
    float hi_84 = fmaxf(kb[6], kb[8]);
    float lo_85 = fminf(kb[6], kb[8]);
    kb[6] = hi_84;
    kb[8] = lo_85;
    float hi_86 = fmaxf(kb[7], kb[9]);
    float lo_87 = fminf(kb[7], kb[9]);
    kb[7] = hi_86;
    kb[9] = lo_87;
    float hi_88 = fmaxf(kb[10], kb[11]);
    float lo_89 = fminf(kb[10], kb[11]);
    kb[10] = hi_88;
    kb[11] = lo_89;
    float hi_90 = fmaxf(kb[12], kb[13]);
    float lo_91 = fminf(kb[12], kb[13]);
    kb[12] = hi_90;
    kb[13] = lo_91;
    float hi_92 = fmaxf(kb[14], kb[15]);
    float lo_93 = fminf(kb[14], kb[15]);
    kb[14] = hi_92;
    kb[15] = lo_93;
    float hi_94 = fmaxf(kb[0], kb[2]);
    float lo_95 = fminf(kb[0], kb[2]);
    kb[0] = hi_94;
    kb[2] = lo_95;
    float hi_96 = fmaxf(kb[1], kb[3]);
    float lo_97 = fminf(kb[1], kb[3]);
    kb[1] = hi_96;
    kb[3] = lo_97;
    float hi_98 = fmaxf(kb[4], kb[10]);
    float lo_99 = fminf(kb[4], kb[10]);
    kb[4] = hi_98;
    kb[10] = lo_99;
    float hi_100 = fmaxf(kb[5], kb[11]);
    float lo_101 = fminf(kb[5], kb[11]);
    kb[5] = hi_100;
    kb[11] = lo_101;
    float hi_102 = fmaxf(kb[6], kb[7]);
    float lo_103 = fminf(kb[6], kb[7]);
    kb[6] = hi_102;
    kb[7] = lo_103;
    float hi_104 = fmaxf(kb[8], kb[9]);
    float lo_105 = fminf(kb[8], kb[9]);
    kb[8] = hi_104;
    kb[9] = lo_105;
    float hi_106 = fmaxf(kb[12], kb[14]);
    float lo_107 = fminf(kb[12], kb[14]);
    kb[12] = hi_106;
    kb[14] = lo_107;
    float hi_108 = fmaxf(kb[13], kb[15]);
    float lo_109 = fminf(kb[13], kb[15]);
    kb[13] = hi_108;
    kb[15] = lo_109;
    float hi_110 = fmaxf(kb[1], kb[2]);
    float lo_111 = fminf(kb[1], kb[2]);
    kb[1] = hi_110;
    kb[2] = lo_111;
    float hi_112 = fmaxf(kb[3], kb[12]);
    float lo_113 = fminf(kb[3], kb[12]);
    kb[3] = hi_112;
    kb[12] = lo_113;
    float hi_114 = fmaxf(kb[4], kb[6]);
    float lo_115 = fminf(kb[4], kb[6]);
    kb[4] = hi_114;
    kb[6] = lo_115;
    float hi_116 = fmaxf(kb[5], kb[7]);
    float lo_117 = fminf(kb[5], kb[7]);
    kb[5] = hi_116;
    kb[7] = lo_117;
    float hi_118 = fmaxf(kb[8], kb[10]);
    float lo_119 = fminf(kb[8], kb[10]);
    kb[8] = hi_118;
    kb[10] = lo_119;
    float hi_120 = fmaxf(kb[9], kb[11]);
    float lo_121 = fminf(kb[9], kb[11]);
    kb[9] = hi_120;
    kb[11] = lo_121;
    float hi_122 = fmaxf(kb[13], kb[14]);
    float lo_123 = fminf(kb[13], kb[14]);
    kb[13] = hi_122;
    kb[14] = lo_123;
    float hi_124 = fmaxf(kb[1], kb[4]);
    float lo_125 = fminf(kb[1], kb[4]);
    kb[1] = hi_124;
    kb[4] = lo_125;
    float hi_126 = fmaxf(kb[2], kb[6]);
    float lo_127 = fminf(kb[2], kb[6]);
    kb[2] = hi_126;
    kb[6] = lo_127;
    float hi_128 = fmaxf(kb[5], kb[8]);
    float lo_129 = fminf(kb[5], kb[8]);
    kb[5] = hi_128;
    kb[8] = lo_129;
    float hi_130 = fmaxf(kb[7], kb[10]);
    float lo_131 = fminf(kb[7], kb[10]);
    kb[7] = hi_130;
    kb[10] = lo_131;
    float hi_132 = fmaxf(kb[9], kb[13]);
    float lo_133 = fminf(kb[9], kb[13]);
    kb[9] = hi_132;
    kb[13] = lo_133;
    float hi_134 = fmaxf(kb[11], kb[14]);
    float lo_135 = fminf(kb[11], kb[14]);
    kb[11] = hi_134;
    kb[14] = lo_135;
    float hi_136 = fmaxf(kb[2], kb[4]);
    float lo_137 = fminf(kb[2], kb[4]);
    kb[2] = hi_136;
    kb[4] = lo_137;
    float hi_138 = fmaxf(kb[3], kb[6]);
    float lo_139 = fminf(kb[3], kb[6]);
    kb[3] = hi_138;
    kb[6] = lo_139;
    float hi_140 = fmaxf(kb[9], kb[12]);
    float lo_141 = fminf(kb[9], kb[12]);
    kb[9] = hi_140;
    kb[12] = lo_141;
    float hi_142 = fmaxf(kb[11], kb[13]);
    float lo_143 = fminf(kb[11], kb[13]);
    kb[11] = hi_142;
    kb[13] = lo_143;
    float hi_144 = fmaxf(kb[3], kb[5]);
    float lo_145 = fminf(kb[3], kb[5]);
    kb[3] = hi_144;
    kb[5] = lo_145;
    float hi_146 = fmaxf(kb[6], kb[8]);
    float lo_147 = fminf(kb[6], kb[8]);
    kb[6] = hi_146;
    kb[8] = lo_147;
    float hi_148 = fmaxf(kb[7], kb[9]);
    float lo_149 = fminf(kb[7], kb[9]);
    kb[7] = hi_148;
    kb[9] = lo_149;
    float hi_150 = fmaxf(kb[10], kb[12]);
    float lo_151 = fminf(kb[10], kb[12]);
    kb[10] = hi_150;
    kb[12] = lo_151;
    float hi_152 = fmaxf(kb[3], kb[4]);
    float lo_153 = fminf(kb[3], kb[4]);
    kb[3] = hi_152;
    kb[4] = lo_153;
    float hi_154 = fmaxf(kb[5], kb[6]);
    float lo_155 = fminf(kb[5], kb[6]);
    kb[5] = hi_154;
    kb[6] = lo_155;
    float hi_156 = fmaxf(kb[7], kb[8]);
    float lo_157 = fminf(kb[7], kb[8]);
    kb[7] = hi_156;
    kb[8] = lo_157;
    float hi_158 = fmaxf(kb[9], kb[10]);
    float lo_159 = fminf(kb[9], kb[10]);
    kb[9] = hi_158;
    kb[10] = lo_159;
    float hi_160 = fmaxf(kb[11], kb[12]);
    float lo_161 = fminf(kb[11], kb[12]);
    kb[11] = hi_160;
    kb[12] = lo_161;
    float hi_162 = fmaxf(kb[6], kb[7]);
    float lo_163 = fminf(kb[6], kb[7]);
    kb[6] = hi_162;
    kb[7] = lo_163;
    float hi_164 = fmaxf(kb[8], kb[9]);
    float lo_165 = fminf(kb[8], kb[9]);
    kb[8] = hi_164;
    kb[9] = lo_165;
    float r = rej;
    float hi_166 = fmaxf(a[0], kb[15]);
    float lo_167 = fminf(a[0], kb[15]);
    a[0] = hi_166;
    r = fmaxf(r, lo_167);
    float hi_168 = fmaxf(a[1], kb[14]);
    float lo_169 = fminf(a[1], kb[14]);
    a[1] = hi_168;
    r = fmaxf(r, lo_169);
    float hi_170 = fmaxf(a[2], kb[13]);
    float lo_171 = fminf(a[2], kb[13]);
    a[2] = hi_170;
    r = fmaxf(r, lo_171);
    float hi_172 = fmaxf(a[3], kb[12]);
    float lo_173 = fminf(a[3], kb[12]);
    a[3] = hi_172;
    r = fmaxf(r, lo_173);
    float hi_174 = fmaxf(a[4], kb[11]);
    float lo_175 = fminf(a[4], kb[11]);
    a[4] = hi_174;
    r = fmaxf(r, lo_175);
    float hi_176 = fmaxf(a[5], kb[10]);
    float lo_177 = fminf(a[5], kb[10]);
    a[5] = hi_176;
    r = fmaxf(r, lo_177);
    float hi_178 = fmaxf(a[6], kb[9]);
    float lo_179 = fminf(a[6], kb[9]);
    a[6] = hi_178;
    r = fmaxf(r, lo_179);
    float hi_180 = fmaxf(a[7], kb[8]);
    float lo_181 = fminf(a[7], kb[8]);
    a[7] = hi_180;
    r = fmaxf(r, lo_181);
    float hi_182 = fmaxf(a[8], kb[7]);
    float lo_183 = fminf(a[8], kb[7]);
    a[8] = hi_182;
    r = fmaxf(r, lo_183);
    float hi_184 = fmaxf(a[9], kb[6]);
    float lo_185 = fminf(a[9], kb[6]);
    a[9] = hi_184;
    r = fmaxf(r, lo_185);
    float hi_186 = fmaxf(a[10], kb[5]);
    float lo_187 = fminf(a[10], kb[5]);
    a[10] = hi_186;
    r = fmaxf(r, lo_187);
    float hi_188 = fmaxf(a[11], kb[4]);
    float lo_189 = fminf(a[11], kb[4]);
    a[11] = hi_188;
    r = fmaxf(r, lo_189);
    float hi_190 = fmaxf(a[12], kb[3]);
    float lo_191 = fminf(a[12], kb[3]);
    a[12] = hi_190;
    r = fmaxf(r, lo_191);
    float hi_192 = fmaxf(a[13], kb[2]);
    float lo_193 = fminf(a[13], kb[2]);
    a[13] = hi_192;
    r = fmaxf(r, lo_193);
    float hi_194 = fmaxf(a[14], kb[1]);
    float lo_195 = fminf(a[14], kb[1]);
    a[14] = hi_194;
    r = fmaxf(r, lo_195);
    float hi_196 = fmaxf(a[15], kb[0]);
    float lo_197 = fminf(a[15], kb[0]);
    a[15] = hi_196;
    r = fmaxf(r, lo_197);
    float hi_198 = fmaxf(a[0], a[8]);
    float lo_199 = fminf(a[0], a[8]);
    a[0] = hi_198;
    a[8] = lo_199;
    float hi_200 = fmaxf(a[1], a[9]);
    float lo_201 = fminf(a[1], a[9]);
    a[1] = hi_200;
    a[9] = lo_201;
    float hi_202 = fmaxf(a[2], a[10]);
    float lo_203 = fminf(a[2], a[10]);
    a[2] = hi_202;
    a[10] = lo_203;
    float hi_204 = fmaxf(a[3], a[11]);
    float lo_205 = fminf(a[3], a[11]);
    a[3] = hi_204;
    a[11] = lo_205;
    float hi_206 = fmaxf(a[4], a[12]);
    float lo_207 = fminf(a[4], a[12]);
    a[4] = hi_206;
    a[12] = lo_207;
    float hi_208 = fmaxf(a[5], a[13]);
    float lo_209 = fminf(a[5], a[13]);
    a[5] = hi_208;
    a[13] = lo_209;
    float hi_210 = fmaxf(a[6], a[14]);
    float lo_211 = fminf(a[6], a[14]);
    a[6] = hi_210;
    a[14] = lo_211;
    float hi_212 = fmaxf(a[7], a[15]);
    float lo_213 = fminf(a[7], a[15]);
    a[7] = hi_212;
    a[15] = lo_213;
    float hi_214 = fmaxf(a[0], a[4]);
    float lo_215 = fminf(a[0], a[4]);
    a[0] = hi_214;
    a[4] = lo_215;
    float hi_216 = fmaxf(a[1], a[5]);
    float lo_217 = fminf(a[1], a[5]);
    a[1] = hi_216;
    a[5] = lo_217;
    float hi_218 = fmaxf(a[2], a[6]);
    float lo_219 = fminf(a[2], a[6]);
    a[2] = hi_218;
    a[6] = lo_219;
    float hi_220 = fmaxf(a[3], a[7]);
    float lo_221 = fminf(a[3], a[7]);
    a[3] = hi_220;
    a[7] = lo_221;
    float hi_222 = fmaxf(a[8], a[12]);
    float lo_223 = fminf(a[8], a[12]);
    a[8] = hi_222;
    a[12] = lo_223;
    float hi_224 = fmaxf(a[9], a[13]);
    float lo_225 = fminf(a[9], a[13]);
    a[9] = hi_224;
    a[13] = lo_225;
    float hi_226 = fmaxf(a[10], a[14]);
    float lo_227 = fminf(a[10], a[14]);
    a[10] = hi_226;
    a[14] = lo_227;
    float hi_228 = fmaxf(a[11], a[15]);
    float lo_229 = fminf(a[11], a[15]);
    a[11] = hi_228;
    a[15] = lo_229;
    float hi_230 = fmaxf(a[0], a[2]);
    float lo_231 = fminf(a[0], a[2]);
    a[0] = hi_230;
    a[2] = lo_231;
    float hi_232 = fmaxf(a[1], a[3]);
    float lo_233 = fminf(a[1], a[3]);
    a[1] = hi_232;
    a[3] = lo_233;
    float hi_234 = fmaxf(a[4], a[6]);
    float lo_235 = fminf(a[4], a[6]);
    a[4] = hi_234;
    a[6] = lo_235;
    float hi_236 = fmaxf(a[5], a[7]);
    float lo_237 = fminf(a[5], a[7]);
    a[5] = hi_236;
    a[7] = lo_237;
    float hi_238 = fmaxf(a[8], a[10]);
    float lo_239 = fminf(a[8], a[10]);
    a[8] = hi_238;
    a[10] = lo_239;
    float hi_240 = fmaxf(a[9], a[11]);
    float lo_241 = fminf(a[9], a[11]);
    a[9] = hi_240;
    a[11] = lo_241;
    float hi_242 = fmaxf(a[12], a[14]);
    float lo_243 = fminf(a[12], a[14]);
    a[12] = hi_242;
    a[14] = lo_243;
    float hi_244 = fmaxf(a[13], a[15]);
    float lo_245 = fminf(a[13], a[15]);
    a[13] = hi_244;
    a[15] = lo_245;
    float hi_246 = fmaxf(a[0], a[1]);
    float lo_247 = fminf(a[0], a[1]);
    a[0] = hi_246;
    a[1] = lo_247;
    float hi_248 = fmaxf(a[2], a[3]);
    float lo_249 = fminf(a[2], a[3]);
    a[2] = hi_248;
    a[3] = lo_249;
    float hi_250 = fmaxf(a[4], a[5]);
    float lo_251 = fminf(a[4], a[5]);
    a[4] = hi_250;
    a[5] = lo_251;
    float hi_252 = fmaxf(a[6], a[7]);
    float lo_253 = fminf(a[6], a[7]);
    a[6] = hi_252;
    a[7] = lo_253;
    float hi_254 = fmaxf(a[8], a[9]);
    float lo_255 = fminf(a[8], a[9]);
    a[8] = hi_254;
    a[9] = lo_255;
    float hi_256 = fmaxf(a[10], a[11]);
    float lo_257 = fminf(a[10], a[11]);
    a[10] = hi_256;
    a[11] = lo_257;
    float hi_258 = fmaxf(a[12], a[13]);
    float lo_259 = fminf(a[12], a[13]);
    a[12] = hi_258;
    a[13] = lo_259;
    float hi_260 = fmaxf(a[14], a[15]);
    float lo_261 = fminf(a[14], a[15]);
    a[14] = hi_260;
    a[15] = lo_261;
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
  int s0 = (sg * 16 + cg) * 17;
  int s1 = (sg * 16 + 8 + cg) * 17;
  float x0 = pub[s0 + ln];
  float y0 = pub[s1 + lnr];
  float lo0 = fminf(x0, y0);
  r_1 = fmaxf(r_1, lo0);
  float hi0 = fmaxf(x0, y0);
  float cur = hi0;
  float pv = __shfl_xor_sync(0xFFFFFFFF, cur, 8);
  float hi_1 = fmaxf(cur, pv);
  float lo_1 = fminf(cur, pv);
  cur = ((up[0] != 0) ? hi_1 : lo_1);
  float pv_0 = __shfl_xor_sync(0xFFFFFFFF, cur, 4);
  float hi_1_1 = fmaxf(cur, pv_0);
  float lo_2 = fminf(cur, pv_0);
  cur = ((up[1] != 0) ? hi_1_1 : lo_2);
  float pv_3 = __shfl_xor_sync(0xFFFFFFFF, cur, 2);
  float hi_4 = fmaxf(cur, pv_3);
  float lo_5 = fminf(cur, pv_3);
  cur = ((up[2] != 0) ? hi_4 : lo_5);
  float pv_6 = __shfl_xor_sync(0xFFFFFFFF, cur, 1);
  float hi_7 = fmaxf(cur, pv_6);
  float lo_8 = fminf(cur, pv_6);
  cur = ((up[3] != 0) ? hi_7 : lo_8);
  V[0] = cur;
  int s0_9 = (sg * 16 + 1 + cg) * 17;
  int s1_10 = (sg * 16 + 1 + 8 + cg) * 17;
  float x0_11 = pub[s0_9 + ln];
  float y0_12 = pub[s1_10 + lnr];
  float lo0_13 = fminf(x0_11, y0_12);
  r_1 = fmaxf(r_1, lo0_13);
  float hi0_14 = fmaxf(x0_11, y0_12);
  float cur_15 = hi0_14;
  float pv_16 = __shfl_xor_sync(0xFFFFFFFF, cur_15, 8);
  float hi_17 = fmaxf(cur_15, pv_16);
  float lo_18 = fminf(cur_15, pv_16);
  cur_15 = ((up[0] != 0) ? hi_17 : lo_18);
  float pv_19 = __shfl_xor_sync(0xFFFFFFFF, cur_15, 4);
  float hi_20 = fmaxf(cur_15, pv_19);
  float lo_21 = fminf(cur_15, pv_19);
  cur_15 = ((up[1] != 0) ? hi_20 : lo_21);
  float pv_22 = __shfl_xor_sync(0xFFFFFFFF, cur_15, 2);
  float hi_23 = fmaxf(cur_15, pv_22);
  float lo_24 = fminf(cur_15, pv_22);
  cur_15 = ((up[2] != 0) ? hi_23 : lo_24);
  float pv_25 = __shfl_xor_sync(0xFFFFFFFF, cur_15, 1);
  float hi_26 = fmaxf(cur_15, pv_25);
  float lo_27 = fminf(cur_15, pv_25);
  cur_15 = ((up[3] != 0) ? hi_26 : lo_27);
  V[1] = cur_15;
  int s0_28 = (sg * 16 + 2 + cg) * 17;
  int s1_29 = (sg * 16 + 2 + 8 + cg) * 17;
  float x0_30 = pub[s0_28 + ln];
  float y0_31 = pub[s1_29 + lnr];
  float lo0_32 = fminf(x0_30, y0_31);
  r_1 = fmaxf(r_1, lo0_32);
  float hi0_33 = fmaxf(x0_30, y0_31);
  float cur_34 = hi0_33;
  float pv_35 = __shfl_xor_sync(0xFFFFFFFF, cur_34, 8);
  float hi_36 = fmaxf(cur_34, pv_35);
  float lo_37 = fminf(cur_34, pv_35);
  cur_34 = ((up[0] != 0) ? hi_36 : lo_37);
  float pv_38 = __shfl_xor_sync(0xFFFFFFFF, cur_34, 4);
  float hi_39 = fmaxf(cur_34, pv_38);
  float lo_40 = fminf(cur_34, pv_38);
  cur_34 = ((up[1] != 0) ? hi_39 : lo_40);
  float pv_41 = __shfl_xor_sync(0xFFFFFFFF, cur_34, 2);
  float hi_42 = fmaxf(cur_34, pv_41);
  float lo_43 = fminf(cur_34, pv_41);
  cur_34 = ((up[2] != 0) ? hi_42 : lo_43);
  float pv_44 = __shfl_xor_sync(0xFFFFFFFF, cur_34, 1);
  float hi_45 = fmaxf(cur_34, pv_44);
  float lo_46 = fminf(cur_34, pv_44);
  cur_34 = ((up[3] != 0) ? hi_45 : lo_46);
  V[2] = cur_34;
  int s0_47 = (sg * 16 + 3 + cg) * 17;
  int s1_48 = (sg * 16 + 3 + 8 + cg) * 17;
  float x0_49 = pub[s0_47 + ln];
  float y0_50 = pub[s1_48 + lnr];
  float lo0_51 = fminf(x0_49, y0_50);
  r_1 = fmaxf(r_1, lo0_51);
  float hi0_52 = fmaxf(x0_49, y0_50);
  float cur_53 = hi0_52;
  float pv_54 = __shfl_xor_sync(0xFFFFFFFF, cur_53, 8);
  float hi_55 = fmaxf(cur_53, pv_54);
  float lo_56 = fminf(cur_53, pv_54);
  cur_53 = ((up[0] != 0) ? hi_55 : lo_56);
  float pv_57 = __shfl_xor_sync(0xFFFFFFFF, cur_53, 4);
  float hi_58_1 = fmaxf(cur_53, pv_57);
  float lo_59_1 = fminf(cur_53, pv_57);
  cur_53 = ((up[1] != 0) ? hi_58_1 : lo_59_1);
  float pv_60 = __shfl_xor_sync(0xFFFFFFFF, cur_53, 2);
  float hi_61 = fmaxf(cur_53, pv_60);
  float lo_62 = fminf(cur_53, pv_60);
  cur_53 = ((up[2] != 0) ? hi_61 : lo_62);
  float pv_63 = __shfl_xor_sync(0xFFFFFFFF, cur_53, 1);
  float hi_64_1 = fmaxf(cur_53, pv_63);
  float lo_65_1 = fminf(cur_53, pv_63);
  cur_53 = ((up[3] != 0) ? hi_64_1 : lo_65_1);
  V[3] = cur_53;
  int s0_66 = (sg * 16 + 4 + cg) * 17;
  int s1_67 = (sg * 16 + 4 + 8 + cg) * 17;
  float x0_68 = pub[s0_66 + ln];
  float y0_69 = pub[s1_67 + lnr];
  float lo0_70 = fminf(x0_68, y0_69);
  r_1 = fmaxf(r_1, lo0_70);
  float hi0_71 = fmaxf(x0_68, y0_69);
  float cur_72 = hi0_71;
  float pv_73 = __shfl_xor_sync(0xFFFFFFFF, cur_72, 8);
  float hi_74_1 = fmaxf(cur_72, pv_73);
  float lo_75_1 = fminf(cur_72, pv_73);
  cur_72 = ((up[0] != 0) ? hi_74_1 : lo_75_1);
  float pv_76 = __shfl_xor_sync(0xFFFFFFFF, cur_72, 4);
  float hi_77 = fmaxf(cur_72, pv_76);
  float lo_78 = fminf(cur_72, pv_76);
  cur_72 = ((up[1] != 0) ? hi_77 : lo_78);
  float pv_79 = __shfl_xor_sync(0xFFFFFFFF, cur_72, 2);
  float hi_80_1 = fmaxf(cur_72, pv_79);
  float lo_81_1 = fminf(cur_72, pv_79);
  cur_72 = ((up[2] != 0) ? hi_80_1 : lo_81_1);
  float pv_82 = __shfl_xor_sync(0xFFFFFFFF, cur_72, 1);
  float hi_83 = fmaxf(cur_72, pv_82);
  float lo_84 = fminf(cur_72, pv_82);
  cur_72 = ((up[3] != 0) ? hi_83 : lo_84);
  V[4] = cur_72;
  int s0_85 = (sg * 16 + 5 + cg) * 17;
  int s1_86 = (sg * 16 + 5 + 8 + cg) * 17;
  float x0_87 = pub[s0_85 + ln];
  float y0_88 = pub[s1_86 + lnr];
  float lo0_89 = fminf(x0_87, y0_88);
  r_1 = fmaxf(r_1, lo0_89);
  float hi0_90 = fmaxf(x0_87, y0_88);
  float cur_91 = hi0_90;
  float pv_92 = __shfl_xor_sync(0xFFFFFFFF, cur_91, 8);
  float hi_93 = fmaxf(cur_91, pv_92);
  float lo_94 = fminf(cur_91, pv_92);
  cur_91 = ((up[0] != 0) ? hi_93 : lo_94);
  float pv_95 = __shfl_xor_sync(0xFFFFFFFF, cur_91, 4);
  float hi_96_1 = fmaxf(cur_91, pv_95);
  float lo_97_1 = fminf(cur_91, pv_95);
  cur_91 = ((up[1] != 0) ? hi_96_1 : lo_97_1);
  float pv_98 = __shfl_xor_sync(0xFFFFFFFF, cur_91, 2);
  float hi_99 = fmaxf(cur_91, pv_98);
  float lo_100 = fminf(cur_91, pv_98);
  cur_91 = ((up[2] != 0) ? hi_99 : lo_100);
  float pv_101 = __shfl_xor_sync(0xFFFFFFFF, cur_91, 1);
  float hi_102_1 = fmaxf(cur_91, pv_101);
  float lo_103_1 = fminf(cur_91, pv_101);
  cur_91 = ((up[3] != 0) ? hi_102_1 : lo_103_1);
  V[5] = cur_91;
  int s0_104 = (sg * 16 + 6 + cg) * 17;
  int s1_105 = (sg * 16 + 6 + 8 + cg) * 17;
  float x0_106 = pub[s0_104 + ln];
  float y0_107 = pub[s1_105 + lnr];
  float lo0_108 = fminf(x0_106, y0_107);
  r_1 = fmaxf(r_1, lo0_108);
  float hi0_109 = fmaxf(x0_106, y0_107);
  float cur_110 = hi0_109;
  float pv_111 = __shfl_xor_sync(0xFFFFFFFF, cur_110, 8);
  float hi_112_1 = fmaxf(cur_110, pv_111);
  float lo_113_1 = fminf(cur_110, pv_111);
  cur_110 = ((up[0] != 0) ? hi_112_1 : lo_113_1);
  float pv_114 = __shfl_xor_sync(0xFFFFFFFF, cur_110, 4);
  float hi_115 = fmaxf(cur_110, pv_114);
  float lo_116 = fminf(cur_110, pv_114);
  cur_110 = ((up[1] != 0) ? hi_115 : lo_116);
  float pv_117 = __shfl_xor_sync(0xFFFFFFFF, cur_110, 2);
  float hi_118_1 = fmaxf(cur_110, pv_117);
  float lo_119_1 = fminf(cur_110, pv_117);
  cur_110 = ((up[2] != 0) ? hi_118_1 : lo_119_1);
  float pv_120 = __shfl_xor_sync(0xFFFFFFFF, cur_110, 1);
  float hi_121 = fmaxf(cur_110, pv_120);
  float lo_122 = fminf(cur_110, pv_120);
  cur_110 = ((up[3] != 0) ? hi_121 : lo_122);
  V[6] = cur_110;
  int s0_123 = (sg * 16 + 7 + cg) * 17;
  int s1_124 = (sg * 16 + 7 + 8 + cg) * 17;
  float x0_125 = pub[s0_123 + ln];
  float y0_126 = pub[s1_124 + lnr];
  float lo0_127 = fminf(x0_125, y0_126);
  r_1 = fmaxf(r_1, lo0_127);
  float hi0_128 = fmaxf(x0_125, y0_126);
  float cur_129 = hi0_128;
  float pv_130 = __shfl_xor_sync(0xFFFFFFFF, cur_129, 8);
  float hi_131 = fmaxf(cur_129, pv_130);
  float lo_132 = fminf(cur_129, pv_130);
  cur_129 = ((up[0] != 0) ? hi_131 : lo_132);
  float pv_133 = __shfl_xor_sync(0xFFFFFFFF, cur_129, 4);
  float hi_134_1 = fmaxf(cur_129, pv_133);
  float lo_135_1 = fminf(cur_129, pv_133);
  cur_129 = ((up[1] != 0) ? hi_134_1 : lo_135_1);
  float pv_136 = __shfl_xor_sync(0xFFFFFFFF, cur_129, 2);
  float hi_137 = fmaxf(cur_129, pv_136);
  float lo_138 = fminf(cur_129, pv_136);
  cur_129 = ((up[2] != 0) ? hi_137 : lo_138);
  float pv_139 = __shfl_xor_sync(0xFFFFFFFF, cur_129, 1);
  float hi_140_1 = fmaxf(cur_129, pv_139);
  float lo_141_1 = fminf(cur_129, pv_139);
  cur_129 = ((up[3] != 0) ? hi_140_1 : lo_141_1);
  V[7] = cur_129;
  float rs = pub[(sg * 16 + ln + cg) * 17 + 16];
  r_1 = fmaxf(r_1, rs);
  float y1 = __shfl_xor_sync(0xFFFFFFFF, V[4], 15);
  float lo1 = fminf(V[0], y1);
  r_1 = fmaxf(r_1, lo1);
  float hi1 = fmaxf(V[0], y1);
  float cur_142 = hi1;
  float pv_143 = __shfl_xor_sync(0xFFFFFFFF, cur_142, 8);
  float hi_144_1 = fmaxf(cur_142, pv_143);
  float lo_145_1 = fminf(cur_142, pv_143);
  cur_142 = ((up[0] != 0) ? hi_144_1 : lo_145_1);
  float pv_146 = __shfl_xor_sync(0xFFFFFFFF, cur_142, 4);
  float hi_147 = fmaxf(cur_142, pv_146);
  float lo_148 = fminf(cur_142, pv_146);
  cur_142 = ((up[1] != 0) ? hi_147 : lo_148);
  float pv_149 = __shfl_xor_sync(0xFFFFFFFF, cur_142, 2);
  float hi_150_1 = fmaxf(cur_142, pv_149);
  float lo_151_1 = fminf(cur_142, pv_149);
  cur_142 = ((up[2] != 0) ? hi_150_1 : lo_151_1);
  float pv_152 = __shfl_xor_sync(0xFFFFFFFF, cur_142, 1);
  float hi_153 = fmaxf(cur_142, pv_152);
  float lo_154 = fminf(cur_142, pv_152);
  cur_142 = ((up[3] != 0) ? hi_153 : lo_154);
  V[0] = cur_142;
  float y1_155 = __shfl_xor_sync(0xFFFFFFFF, V[5], 15);
  float lo1_156 = fminf(V[1], y1_155);
  r_1 = fmaxf(r_1, lo1_156);
  float hi1_157 = fmaxf(V[1], y1_155);
  float cur_158 = hi1_157;
  float pv_159 = __shfl_xor_sync(0xFFFFFFFF, cur_158, 8);
  float hi_160_1 = fmaxf(cur_158, pv_159);
  float lo_161_1 = fminf(cur_158, pv_159);
  cur_158 = ((up[0] != 0) ? hi_160_1 : lo_161_1);
  float pv_162 = __shfl_xor_sync(0xFFFFFFFF, cur_158, 4);
  float hi_163 = fmaxf(cur_158, pv_162);
  float lo_164 = fminf(cur_158, pv_162);
  cur_158 = ((up[1] != 0) ? hi_163 : lo_164);
  float pv_165 = __shfl_xor_sync(0xFFFFFFFF, cur_158, 2);
  float hi_166_1 = fmaxf(cur_158, pv_165);
  float lo_167_1 = fminf(cur_158, pv_165);
  cur_158 = ((up[2] != 0) ? hi_166_1 : lo_167_1);
  float pv_168 = __shfl_xor_sync(0xFFFFFFFF, cur_158, 1);
  float hi_169 = fmaxf(cur_158, pv_168);
  float lo_170 = fminf(cur_158, pv_168);
  cur_158 = ((up[3] != 0) ? hi_169 : lo_170);
  V[1] = cur_158;
  float y1_171 = __shfl_xor_sync(0xFFFFFFFF, V[6], 15);
  float lo1_172 = fminf(V[2], y1_171);
  r_1 = fmaxf(r_1, lo1_172);
  float hi1_173 = fmaxf(V[2], y1_171);
  float cur_174 = hi1_173;
  float pv_175 = __shfl_xor_sync(0xFFFFFFFF, cur_174, 8);
  float hi_176_1 = fmaxf(cur_174, pv_175);
  float lo_177_1 = fminf(cur_174, pv_175);
  cur_174 = ((up[0] != 0) ? hi_176_1 : lo_177_1);
  float pv_178 = __shfl_xor_sync(0xFFFFFFFF, cur_174, 4);
  float hi_179 = fmaxf(cur_174, pv_178);
  float lo_180 = fminf(cur_174, pv_178);
  cur_174 = ((up[1] != 0) ? hi_179 : lo_180);
  float pv_181 = __shfl_xor_sync(0xFFFFFFFF, cur_174, 2);
  float hi_182_1 = fmaxf(cur_174, pv_181);
  float lo_183_1 = fminf(cur_174, pv_181);
  cur_174 = ((up[2] != 0) ? hi_182_1 : lo_183_1);
  float pv_184 = __shfl_xor_sync(0xFFFFFFFF, cur_174, 1);
  float hi_185 = fmaxf(cur_174, pv_184);
  float lo_186 = fminf(cur_174, pv_184);
  cur_174 = ((up[3] != 0) ? hi_185 : lo_186);
  V[2] = cur_174;
  float y1_187 = __shfl_xor_sync(0xFFFFFFFF, V[7], 15);
  float lo1_188 = fminf(V[3], y1_187);
  r_1 = fmaxf(r_1, lo1_188);
  float hi1_189 = fmaxf(V[3], y1_187);
  float cur_190 = hi1_189;
  float pv_191 = __shfl_xor_sync(0xFFFFFFFF, cur_190, 8);
  float hi_192_1 = fmaxf(cur_190, pv_191);
  float lo_193_1 = fminf(cur_190, pv_191);
  cur_190 = ((up[0] != 0) ? hi_192_1 : lo_193_1);
  float pv_194 = __shfl_xor_sync(0xFFFFFFFF, cur_190, 4);
  float hi_195 = fmaxf(cur_190, pv_194);
  float lo_196 = fminf(cur_190, pv_194);
  cur_190 = ((up[1] != 0) ? hi_195 : lo_196);
  float pv_197 = __shfl_xor_sync(0xFFFFFFFF, cur_190, 2);
  float hi_198_1 = fmaxf(cur_190, pv_197);
  float lo_199_1 = fminf(cur_190, pv_197);
  cur_190 = ((up[2] != 0) ? hi_198_1 : lo_199_1);
  float pv_200 = __shfl_xor_sync(0xFFFFFFFF, cur_190, 1);
  float hi_201 = fmaxf(cur_190, pv_200);
  float lo_202 = fminf(cur_190, pv_200);
  cur_190 = ((up[3] != 0) ? hi_201 : lo_202);
  V[3] = cur_190;
  float y1_203 = __shfl_xor_sync(0xFFFFFFFF, V[2], 15);
  float lo1_204 = fminf(V[0], y1_203);
  r_1 = fmaxf(r_1, lo1_204);
  float hi1_205 = fmaxf(V[0], y1_203);
  float cur_206 = hi1_205;
  float pv_207 = __shfl_xor_sync(0xFFFFFFFF, cur_206, 8);
  float hi_208_1 = fmaxf(cur_206, pv_207);
  float lo_209_1 = fminf(cur_206, pv_207);
  cur_206 = ((up[0] != 0) ? hi_208_1 : lo_209_1);
  float pv_210 = __shfl_xor_sync(0xFFFFFFFF, cur_206, 4);
  float hi_211 = fmaxf(cur_206, pv_210);
  float lo_212 = fminf(cur_206, pv_210);
  cur_206 = ((up[1] != 0) ? hi_211 : lo_212);
  float pv_213 = __shfl_xor_sync(0xFFFFFFFF, cur_206, 2);
  float hi_214_1 = fmaxf(cur_206, pv_213);
  float lo_215_1 = fminf(cur_206, pv_213);
  cur_206 = ((up[2] != 0) ? hi_214_1 : lo_215_1);
  float pv_216 = __shfl_xor_sync(0xFFFFFFFF, cur_206, 1);
  float hi_217 = fmaxf(cur_206, pv_216);
  float lo_218 = fminf(cur_206, pv_216);
  cur_206 = ((up[3] != 0) ? hi_217 : lo_218);
  V[0] = cur_206;
  float y1_219 = __shfl_xor_sync(0xFFFFFFFF, V[3], 15);
  float lo1_220 = fminf(V[1], y1_219);
  r_1 = fmaxf(r_1, lo1_220);
  float hi1_221 = fmaxf(V[1], y1_219);
  float cur_222 = hi1_221;
  float pv_223 = __shfl_xor_sync(0xFFFFFFFF, cur_222, 8);
  float hi_224_1 = fmaxf(cur_222, pv_223);
  float lo_225_1 = fminf(cur_222, pv_223);
  cur_222 = ((up[0] != 0) ? hi_224_1 : lo_225_1);
  float pv_226 = __shfl_xor_sync(0xFFFFFFFF, cur_222, 4);
  float hi_227 = fmaxf(cur_222, pv_226);
  float lo_228 = fminf(cur_222, pv_226);
  cur_222 = ((up[1] != 0) ? hi_227 : lo_228);
  float pv_229 = __shfl_xor_sync(0xFFFFFFFF, cur_222, 2);
  float hi_230_1 = fmaxf(cur_222, pv_229);
  float lo_231_1 = fminf(cur_222, pv_229);
  cur_222 = ((up[2] != 0) ? hi_230_1 : lo_231_1);
  float pv_232 = __shfl_xor_sync(0xFFFFFFFF, cur_222, 1);
  float hi_233 = fmaxf(cur_222, pv_232);
  float lo_234 = fminf(cur_222, pv_232);
  cur_222 = ((up[3] != 0) ? hi_233 : lo_234);
  V[1] = cur_222;
  float yl = __shfl_xor_sync(0xFFFFFFFF, V[1], 15);
  float lol = fminf(V[0], yl);
  r_1 = fmaxf(r_1, lol);
  float hil = fmaxf(V[0], yl);
  float cur_235 = hil;
  float pv_236 = __shfl_xor_sync(0xFFFFFFFF, cur_235, 8);
  float hi_237 = fmaxf(cur_235, pv_236);
  float lo_238 = fminf(cur_235, pv_236);
  cur_235 = ((up[0] != 0) ? hi_237 : lo_238);
  float pv_239 = __shfl_xor_sync(0xFFFFFFFF, cur_235, 4);
  float hi_240_1 = fmaxf(cur_235, pv_239);
  float lo_241_1 = fminf(cur_235, pv_239);
  cur_235 = ((up[1] != 0) ? hi_240_1 : lo_241_1);
  float pv_242 = __shfl_xor_sync(0xFFFFFFFF, cur_235, 2);
  float hi_243 = fmaxf(cur_235, pv_242);
  float lo_244 = fminf(cur_235, pv_242);
  cur_235 = ((up[2] != 0) ? hi_243 : lo_244);
  float pv_245 = __shfl_xor_sync(0xFFFFFFFF, cur_235, 1);
  float hi_246_1 = fmaxf(cur_235, pv_245);
  float lo_247_1 = fminf(cur_235, pv_245);
  cur_235 = ((up[3] != 0) ? hi_246_1 : lo_247_1);
  V[0] = cur_235;
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
        int i4_1 = s4_0 & 15;
        int j4_2 = s4_0 >> 4 & 0;
        int g4_3 = whi * 32 + (s4_0 >> 4);
        int t4_4 = (j4_2 * 32 + g4_3) * 16 + i4_1;
        bb[0] = S[cbase2 + (long long)t4_4 * nq64];
      }
      bb[1] = 4286578688u;
      int s5_0 = lane_1 + 32;
      int j5_1 = s5_0 >> 4 & 0;
      int v5_2 = 0;
      int s4_3 = lane_1 + 32;
      int i4_4 = s4_3 & 15;
      int j4_5 = s4_3 >> 4 & 0;
      int g4_6 = whi * 32 + (s4_3 >> 4);
      int t4_7 = (j4_5 * 32 + g4_6) * 16 + i4_4;
      if (j5_1 < num_chunks && t4_7 < limv) {
        v5_2 = 1;
      }
      if (v5_2 != 0) {
        int s4_0_1 = lane_1 + 32;
        int i4_1_1 = s4_0_1 & 15;
        int j4_2_1 = s4_0_1 >> 4 & 0;
        int g4_3_1 = whi * 32 + (s4_0_1 >> 4);
        int t4_4_1 = (j4_2_1 * 32 + g4_3_1) * 16 + i4_1_1;
        bb[1] = S[cbase2 + (long long)t4_4_1 * nq64];
      }
      bb[2] = 4286578688u;
      int s5_8 = lane_1 + 64;
      int j5_9 = s5_8 >> 4 & 0;
      int v5_10 = 0;
      int s4_11 = lane_1 + 64;
      int i4_12 = s4_11 & 15;
      int j4_13 = s4_11 >> 4 & 0;
      int g4_14 = whi * 32 + (s4_11 >> 4);
      int t4_15 = (j4_13 * 32 + g4_14) * 16 + i4_12;
      if (j5_9 < num_chunks && t4_15 < limv) {
        v5_10 = 1;
      }
      if (v5_10 != 0) {
        int s4_0_2 = lane_1 + 64;
        int i4_1_2 = s4_0_2 & 15;
        int j4_2_2 = s4_0_2 >> 4 & 0;
        int g4_3_2 = whi * 32 + (s4_0_2 >> 4);
        int t4_4_2 = (j4_2_2 * 32 + g4_3_2) * 16 + i4_1_2;
        bb[2] = S[cbase2 + (long long)t4_4_2 * nq64];
      }
      bb[3] = 4286578688u;
      int s5_16 = lane_1 + 96;
      int j5_17 = s5_16 >> 4 & 0;
      int v5_18 = 0;
      int s4_19 = lane_1 + 96;
      int i4_20 = s4_19 & 15;
      int j4_21 = s4_19 >> 4 & 0;
      int g4_22 = whi * 32 + (s4_19 >> 4);
      int t4_23 = (j4_21 * 32 + g4_22) * 16 + i4_20;
      if (j5_17 < num_chunks && t4_23 < limv) {
        v5_18 = 1;
      }
      if (v5_18 != 0) {
        int s4_0_3 = lane_1 + 96;
        int i4_1_3 = s4_0_3 & 15;
        int j4_2_3 = s4_0_3 >> 4 & 0;
        int g4_3_3 = whi * 32 + (s4_0_3 >> 4);
        int t4_4_3 = (j4_2_3 * 32 + g4_3_3) * 16 + i4_1_3;
        bb[3] = S[cbase2 + (long long)t4_4_3 * nq64];
      }
      bb[4] = 4286578688u;
      int s5_24 = lane_1 + 128;
      int j5_25 = s5_24 >> 4 & 0;
      int v5_26 = 0;
      int s4_27 = lane_1 + 128;
      int i4_28 = s4_27 & 15;
      int j4_29 = s4_27 >> 4 & 0;
      int g4_30 = whi * 32 + (s4_27 >> 4);
      int t4_31 = (j4_29 * 32 + g4_30) * 16 + i4_28;
      if (j5_25 < num_chunks && t4_31 < limv) {
        v5_26 = 1;
      }
      if (v5_26 != 0) {
        int s4_0_4 = lane_1 + 128;
        int i4_1_4 = s4_0_4 & 15;
        int j4_2_4 = s4_0_4 >> 4 & 0;
        int g4_3_4 = whi * 32 + (s4_0_4 >> 4);
        int t4_4_4 = (j4_2_4 * 32 + g4_3_4) * 16 + i4_1_4;
        bb[4] = S[cbase2 + (long long)t4_4_4 * nq64];
      }
      bb[5] = 4286578688u;
      int s5_32 = lane_1 + 160;
      int j5_33 = s5_32 >> 4 & 0;
      int v5_34 = 0;
      int s4_35 = lane_1 + 160;
      int i4_36 = s4_35 & 15;
      int j4_37 = s4_35 >> 4 & 0;
      int g4_38 = whi * 32 + (s4_35 >> 4);
      int t4_39 = (j4_37 * 32 + g4_38) * 16 + i4_36;
      if (j5_33 < num_chunks && t4_39 < limv) {
        v5_34 = 1;
      }
      if (v5_34 != 0) {
        int s4_0_5 = lane_1 + 160;
        int i4_1_5 = s4_0_5 & 15;
        int j4_2_5 = s4_0_5 >> 4 & 0;
        int g4_3_5 = whi * 32 + (s4_0_5 >> 4);
        int t4_4_5 = (j4_2_5 * 32 + g4_3_5) * 16 + i4_1_5;
        bb[5] = S[cbase2 + (long long)t4_4_5 * nq64];
      }
      bb[6] = 4286578688u;
      int s5_40 = lane_1 + 192;
      int j5_41 = s5_40 >> 4 & 0;
      int v5_42 = 0;
      int s4_43 = lane_1 + 192;
      int i4_44 = s4_43 & 15;
      int j4_45 = s4_43 >> 4 & 0;
      int g4_46 = whi * 32 + (s4_43 >> 4);
      int t4_47 = (j4_45 * 32 + g4_46) * 16 + i4_44;
      if (j5_41 < num_chunks && t4_47 < limv) {
        v5_42 = 1;
      }
      if (v5_42 != 0) {
        int s4_0_6 = lane_1 + 192;
        int i4_1_6 = s4_0_6 & 15;
        int j4_2_6 = s4_0_6 >> 4 & 0;
        int g4_3_6 = whi * 32 + (s4_0_6 >> 4);
        int t4_4_6 = (j4_2_6 * 32 + g4_3_6) * 16 + i4_1_6;
        bb[6] = S[cbase2 + (long long)t4_4_6 * nq64];
      }
      bb[7] = 4286578688u;
      int s5_48 = lane_1 + 224;
      int j5_49 = s5_48 >> 4 & 0;
      int v5_50 = 0;
      int s4_51 = lane_1 + 224;
      int i4_52 = s4_51 & 15;
      int j4_53 = s4_51 >> 4 & 0;
      int g4_54 = whi * 32 + (s4_51 >> 4);
      int t4_55 = (j4_53 * 32 + g4_54) * 16 + i4_52;
      if (j5_49 < num_chunks && t4_55 < limv) {
        v5_50 = 1;
      }
      if (v5_50 != 0) {
        int s4_0_7 = lane_1 + 224;
        int i4_1_7 = s4_0_7 & 15;
        int j4_2_7 = s4_0_7 >> 4 & 0;
        int g4_3_7 = whi * 32 + (s4_0_7 >> 4);
        int t4_4_7 = (j4_2_7 * 32 + g4_3_7) * 16 + i4_1_7;
        bb[7] = S[cbase2 + (long long)t4_4_7 * nq64];
      }
      bb[8] = 4286578688u;
      int s5_56 = lane_1 + 256;
      int j5_57 = s5_56 >> 4 & 0;
      int v5_58 = 0;
      int s4_59 = lane_1 + 256;
      int i4_60 = s4_59 & 15;
      int j4_61 = s4_59 >> 4 & 0;
      int g4_62 = whi * 32 + (s4_59 >> 4);
      int t4_63 = (j4_61 * 32 + g4_62) * 16 + i4_60;
      if (j5_57 < num_chunks && t4_63 < limv) {
        v5_58 = 1;
      }
      if (v5_58 != 0) {
        int s4_0_8 = lane_1 + 256;
        int i4_1_8 = s4_0_8 & 15;
        int j4_2_8 = s4_0_8 >> 4 & 0;
        int g4_3_8 = whi * 32 + (s4_0_8 >> 4);
        int t4_4_8 = (j4_2_8 * 32 + g4_3_8) * 16 + i4_1_8;
        bb[8] = S[cbase2 + (long long)t4_4_8 * nq64];
      }
      bb[9] = 4286578688u;
      int s5_64 = lane_1 + 288;
      int j5_65 = s5_64 >> 4 & 0;
      int v5_66 = 0;
      int s4_67 = lane_1 + 288;
      int i4_68 = s4_67 & 15;
      int j4_69 = s4_67 >> 4 & 0;
      int g4_70 = whi * 32 + (s4_67 >> 4);
      int t4_71 = (j4_69 * 32 + g4_70) * 16 + i4_68;
      if (j5_65 < num_chunks && t4_71 < limv) {
        v5_66 = 1;
      }
      if (v5_66 != 0) {
        int s4_0_9 = lane_1 + 288;
        int i4_1_9 = s4_0_9 & 15;
        int j4_2_9 = s4_0_9 >> 4 & 0;
        int g4_3_9 = whi * 32 + (s4_0_9 >> 4);
        int t4_4_9 = (j4_2_9 * 32 + g4_3_9) * 16 + i4_1_9;
        bb[9] = S[cbase2 + (long long)t4_4_9 * nq64];
      }
      bb[10] = 4286578688u;
      int s5_72 = lane_1 + 320;
      int j5_73 = s5_72 >> 4 & 0;
      int v5_74 = 0;
      int s4_75 = lane_1 + 320;
      int i4_76 = s4_75 & 15;
      int j4_77 = s4_75 >> 4 & 0;
      int g4_78 = whi * 32 + (s4_75 >> 4);
      int t4_79 = (j4_77 * 32 + g4_78) * 16 + i4_76;
      if (j5_73 < num_chunks && t4_79 < limv) {
        v5_74 = 1;
      }
      if (v5_74 != 0) {
        int s4_0_10 = lane_1 + 320;
        int i4_1_10 = s4_0_10 & 15;
        int j4_2_10 = s4_0_10 >> 4 & 0;
        int g4_3_10 = whi * 32 + (s4_0_10 >> 4);
        int t4_4_10 = (j4_2_10 * 32 + g4_3_10) * 16 + i4_1_10;
        bb[10] = S[cbase2 + (long long)t4_4_10 * nq64];
      }
      bb[11] = 4286578688u;
      int s5_80 = lane_1 + 352;
      int j5_81 = s5_80 >> 4 & 0;
      int v5_82 = 0;
      int s4_83 = lane_1 + 352;
      int i4_84 = s4_83 & 15;
      int j4_85 = s4_83 >> 4 & 0;
      int g4_86 = whi * 32 + (s4_83 >> 4);
      int t4_87 = (j4_85 * 32 + g4_86) * 16 + i4_84;
      if (j5_81 < num_chunks && t4_87 < limv) {
        v5_82 = 1;
      }
      if (v5_82 != 0) {
        int s4_0_11 = lane_1 + 352;
        int i4_1_11 = s4_0_11 & 15;
        int j4_2_11 = s4_0_11 >> 4 & 0;
        int g4_3_11 = whi * 32 + (s4_0_11 >> 4);
        int t4_4_11 = (j4_2_11 * 32 + g4_3_11) * 16 + i4_1_11;
        bb[11] = S[cbase2 + (long long)t4_4_11 * nq64];
      }
      bb[12] = 4286578688u;
      int s5_88 = lane_1 + 384;
      int j5_89 = s5_88 >> 4 & 0;
      int v5_90 = 0;
      int s4_91 = lane_1 + 384;
      int i4_92 = s4_91 & 15;
      int j4_93 = s4_91 >> 4 & 0;
      int g4_94 = whi * 32 + (s4_91 >> 4);
      int t4_95 = (j4_93 * 32 + g4_94) * 16 + i4_92;
      if (j5_89 < num_chunks && t4_95 < limv) {
        v5_90 = 1;
      }
      if (v5_90 != 0) {
        int s4_0_12 = lane_1 + 384;
        int i4_1_12 = s4_0_12 & 15;
        int j4_2_12 = s4_0_12 >> 4 & 0;
        int g4_3_12 = whi * 32 + (s4_0_12 >> 4);
        int t4_4_12 = (j4_2_12 * 32 + g4_3_12) * 16 + i4_1_12;
        bb[12] = S[cbase2 + (long long)t4_4_12 * nq64];
      }
      bb[13] = 4286578688u;
      int s5_96 = lane_1 + 416;
      int j5_97 = s5_96 >> 4 & 0;
      int v5_98 = 0;
      int s4_99 = lane_1 + 416;
      int i4_100 = s4_99 & 15;
      int j4_101 = s4_99 >> 4 & 0;
      int g4_102 = whi * 32 + (s4_99 >> 4);
      int t4_103 = (j4_101 * 32 + g4_102) * 16 + i4_100;
      if (j5_97 < num_chunks && t4_103 < limv) {
        v5_98 = 1;
      }
      if (v5_98 != 0) {
        int s4_0_13 = lane_1 + 416;
        int i4_1_13 = s4_0_13 & 15;
        int j4_2_13 = s4_0_13 >> 4 & 0;
        int g4_3_13 = whi * 32 + (s4_0_13 >> 4);
        int t4_4_13 = (j4_2_13 * 32 + g4_3_13) * 16 + i4_1_13;
        bb[13] = S[cbase2 + (long long)t4_4_13 * nq64];
      }
      bb[14] = 4286578688u;
      int s5_104 = lane_1 + 448;
      int j5_105 = s5_104 >> 4 & 0;
      int v5_106 = 0;
      int s4_107 = lane_1 + 448;
      int i4_108 = s4_107 & 15;
      int j4_109 = s4_107 >> 4 & 0;
      int g4_110 = whi * 32 + (s4_107 >> 4);
      int t4_111 = (j4_109 * 32 + g4_110) * 16 + i4_108;
      if (j5_105 < num_chunks && t4_111 < limv) {
        v5_106 = 1;
      }
      if (v5_106 != 0) {
        int s4_0_14 = lane_1 + 448;
        int i4_1_14 = s4_0_14 & 15;
        int j4_2_14 = s4_0_14 >> 4 & 0;
        int g4_3_14 = whi * 32 + (s4_0_14 >> 4);
        int t4_4_14 = (j4_2_14 * 32 + g4_3_14) * 16 + i4_1_14;
        bb[14] = S[cbase2 + (long long)t4_4_14 * nq64];
      }
      bb[15] = 4286578688u;
      int s5_112 = lane_1 + 480;
      int j5_113 = s5_112 >> 4 & 0;
      int v5_114 = 0;
      int s4_115 = lane_1 + 480;
      int i4_116 = s4_115 & 15;
      int j4_117 = s4_115 >> 4 & 0;
      int g4_118 = whi * 32 + (s4_115 >> 4);
      int t4_119 = (j4_117 * 32 + g4_118) * 16 + i4_116;
      if (j5_113 < num_chunks && t4_119 < limv) {
        v5_114 = 1;
      }
      if (v5_114 != 0) {
        int s4_0_15 = lane_1 + 480;
        int i4_1_15 = s4_0_15 & 15;
        int j4_2_15 = s4_0_15 >> 4 & 0;
        int g4_3_15 = whi * 32 + (s4_0_15 >> 4);
        int t4_4_15 = (j4_2_15 * 32 + g4_3_15) * 16 + i4_1_15;
        bb[15] = S[cbase2 + (long long)t4_4_15 * nq64];
      }
      asm volatile("" ::: "memory");
      unsigned int kk2[16];
      kk2[0] = 0;
      int s5_120 = lane_1;
      int j5_121 = s5_120 >> 4 & 0;
      int v5_122 = 0;
      int s4_123 = lane_1;
      int i4_124 = s4_123 & 15;
      int j4_125 = s4_123 >> 4 & 0;
      int g4_126 = whi * 32 + (s4_123 >> 4);
      int t4_127 = (j4_125 * 32 + g4_126) * 16 + i4_124;
      if (j5_121 < num_chunks && t4_127 < limv) {
        v5_122 = 1;
      }
      if (v5_122 != 0) {
        int s4_0_16 = lane_1;
        int i4_1_16 = s4_0_16 & 15;
        int j4_2_16 = s4_0_16 >> 4 & 0;
        int g4_3_16 = whi * 32 + (s4_0_16 >> 4);
        int t4_4_16 = (j4_2_16 * 32 + g4_3_16) * 16 + i4_1_16;
        float sc_1 = __uint_as_float(bb[0]);
        sc_1 = fmaxf(sc_1, -1.7014118346046923e+38f);
        sc_1 = fminf(sc_1, 1.7014118346046923e+38f);
        unsigned int u8 = __as_u32(sc_1);
        unsigned int key8 = 0;
        if ((u8 & 4294966784u) == qf7) {
          unsigned int lowb8 = (u8 ^ (unsigned int)((int)u8 >> 31) & 511) & 511;
          key8 = 536870912 | lowb8 << 9 | (unsigned int)t4_4_16;
        }
        int f2 = 0;
        if ((t4_4_16 < fb || t4_4_16 >= limv_fe) && t4_4_16 < limv) {
          f2 = 1;
        }
        if (f2 != 0) {
          key8 = 0;
        }
        kk2[0] = key8;
      }
      kk2[1] = 0;
      int s5_128 = lane_1 + 32;
      int j5_129 = s5_128 >> 4 & 0;
      int v5_130 = 0;
      int s4_131 = lane_1 + 32;
      int i4_132 = s4_131 & 15;
      int j4_133 = s4_131 >> 4 & 0;
      int g4_134 = whi * 32 + (s4_131 >> 4);
      int t4_135 = (j4_133 * 32 + g4_134) * 16 + i4_132;
      if (j5_129 < num_chunks && t4_135 < limv) {
        v5_130 = 1;
      }
      if (v5_130 != 0) {
        int s4_0_17 = lane_1 + 32;
        int i4_1_17 = s4_0_17 & 15;
        int j4_2_17 = s4_0_17 >> 4 & 0;
        int g4_3_17 = whi * 32 + (s4_0_17 >> 4);
        int t4_4_17 = (j4_2_17 * 32 + g4_3_17) * 16 + i4_1_17;
        float sc_2 = __uint_as_float(bb[1]);
        sc_2 = fmaxf(sc_2, -1.7014118346046923e+38f);
        sc_2 = fminf(sc_2, 1.7014118346046923e+38f);
        unsigned int u8_1 = __as_u32(sc_2);
        unsigned int key8_1 = 0;
        if ((u8_1 & 4294966784u) == qf7) {
          unsigned int lowb8_1 = (u8_1 ^ (unsigned int)((int)u8_1 >> 31) & 511) & 511;
          key8_1 = 536870912 | lowb8_1 << 9 | (unsigned int)t4_4_17;
        }
        int f2_1 = 0;
        if ((t4_4_17 < fb || t4_4_17 >= limv_fe) && t4_4_17 < limv) {
          f2_1 = 1;
        }
        if (f2_1 != 0) {
          key8_1 = 0;
        }
        kk2[1] = key8_1;
      }
      kk2[2] = 0;
      int s5_136 = lane_1 + 64;
      int j5_137 = s5_136 >> 4 & 0;
      int v5_138 = 0;
      int s4_139 = lane_1 + 64;
      int i4_140 = s4_139 & 15;
      int j4_141 = s4_139 >> 4 & 0;
      int g4_142 = whi * 32 + (s4_139 >> 4);
      int t4_143 = (j4_141 * 32 + g4_142) * 16 + i4_140;
      if (j5_137 < num_chunks && t4_143 < limv) {
        v5_138 = 1;
      }
      if (v5_138 != 0) {
        int s4_0_18 = lane_1 + 64;
        int i4_1_18 = s4_0_18 & 15;
        int j4_2_18 = s4_0_18 >> 4 & 0;
        int g4_3_18 = whi * 32 + (s4_0_18 >> 4);
        int t4_4_18 = (j4_2_18 * 32 + g4_3_18) * 16 + i4_1_18;
        float sc_4 = __uint_as_float(bb[2]);
        sc_4 = fmaxf(sc_4, -1.7014118346046923e+38f);
        sc_4 = fminf(sc_4, 1.7014118346046923e+38f);
        unsigned int u8_2 = __as_u32(sc_4);
        unsigned int key8_2 = 0;
        if ((u8_2 & 4294966784u) == qf7) {
          unsigned int lowb8_2 = (u8_2 ^ (unsigned int)((int)u8_2 >> 31) & 511) & 511;
          key8_2 = 536870912 | lowb8_2 << 9 | (unsigned int)t4_4_18;
        }
        int f2_2 = 0;
        if ((t4_4_18 < fb || t4_4_18 >= limv_fe) && t4_4_18 < limv) {
          f2_2 = 1;
        }
        if (f2_2 != 0) {
          key8_2 = 0;
        }
        kk2[2] = key8_2;
      }
      kk2[3] = 0;
      int s5_144 = lane_1 + 96;
      int j5_145 = s5_144 >> 4 & 0;
      int v5_146 = 0;
      int s4_147 = lane_1 + 96;
      int i4_148 = s4_147 & 15;
      int j4_149 = s4_147 >> 4 & 0;
      int g4_150 = whi * 32 + (s4_147 >> 4);
      int t4_151 = (j4_149 * 32 + g4_150) * 16 + i4_148;
      if (j5_145 < num_chunks && t4_151 < limv) {
        v5_146 = 1;
      }
      if (v5_146 != 0) {
        int s4_0_19 = lane_1 + 96;
        int i4_1_19 = s4_0_19 & 15;
        int j4_2_19 = s4_0_19 >> 4 & 0;
        int g4_3_19 = whi * 32 + (s4_0_19 >> 4);
        int t4_4_19 = (j4_2_19 * 32 + g4_3_19) * 16 + i4_1_19;
        float sc_5 = __uint_as_float(bb[3]);
        sc_5 = fmaxf(sc_5, -1.7014118346046923e+38f);
        sc_5 = fminf(sc_5, 1.7014118346046923e+38f);
        unsigned int u8_3 = __as_u32(sc_5);
        unsigned int key8_3 = 0;
        if ((u8_3 & 4294966784u) == qf7) {
          unsigned int lowb8_3 = (u8_3 ^ (unsigned int)((int)u8_3 >> 31) & 511) & 511;
          key8_3 = 536870912 | lowb8_3 << 9 | (unsigned int)t4_4_19;
        }
        int f2_3 = 0;
        if ((t4_4_19 < fb || t4_4_19 >= limv_fe) && t4_4_19 < limv) {
          f2_3 = 1;
        }
        if (f2_3 != 0) {
          key8_3 = 0;
        }
        kk2[3] = key8_3;
      }
      kk2[4] = 0;
      int s5_152 = lane_1 + 128;
      int j5_153 = s5_152 >> 4 & 0;
      int v5_154 = 0;
      int s4_155 = lane_1 + 128;
      int i4_156 = s4_155 & 15;
      int j4_157 = s4_155 >> 4 & 0;
      int g4_158 = whi * 32 + (s4_155 >> 4);
      int t4_159 = (j4_157 * 32 + g4_158) * 16 + i4_156;
      if (j5_153 < num_chunks && t4_159 < limv) {
        v5_154 = 1;
      }
      if (v5_154 != 0) {
        int s4_0_20 = lane_1 + 128;
        int i4_1_20 = s4_0_20 & 15;
        int j4_2_20 = s4_0_20 >> 4 & 0;
        int g4_3_20 = whi * 32 + (s4_0_20 >> 4);
        int t4_4_20 = (j4_2_20 * 32 + g4_3_20) * 16 + i4_1_20;
        float sc_7 = __uint_as_float(bb[4]);
        sc_7 = fmaxf(sc_7, -1.7014118346046923e+38f);
        sc_7 = fminf(sc_7, 1.7014118346046923e+38f);
        unsigned int u8_4 = __as_u32(sc_7);
        unsigned int key8_4 = 0;
        if ((u8_4 & 4294966784u) == qf7) {
          unsigned int lowb8_4 = (u8_4 ^ (unsigned int)((int)u8_4 >> 31) & 511) & 511;
          key8_4 = 536870912 | lowb8_4 << 9 | (unsigned int)t4_4_20;
        }
        int f2_4 = 0;
        if ((t4_4_20 < fb || t4_4_20 >= limv_fe) && t4_4_20 < limv) {
          f2_4 = 1;
        }
        if (f2_4 != 0) {
          key8_4 = 0;
        }
        kk2[4] = key8_4;
      }
      kk2[5] = 0;
      int s5_160 = lane_1 + 160;
      int j5_161 = s5_160 >> 4 & 0;
      int v5_162 = 0;
      int s4_163 = lane_1 + 160;
      int i4_164 = s4_163 & 15;
      int j4_165 = s4_163 >> 4 & 0;
      int g4_166 = whi * 32 + (s4_163 >> 4);
      int t4_167 = (j4_165 * 32 + g4_166) * 16 + i4_164;
      if (j5_161 < num_chunks && t4_167 < limv) {
        v5_162 = 1;
      }
      if (v5_162 != 0) {
        int s4_0_21 = lane_1 + 160;
        int i4_1_21 = s4_0_21 & 15;
        int j4_2_21 = s4_0_21 >> 4 & 0;
        int g4_3_21 = whi * 32 + (s4_0_21 >> 4);
        int t4_4_21 = (j4_2_21 * 32 + g4_3_21) * 16 + i4_1_21;
        float sc_8 = __uint_as_float(bb[5]);
        sc_8 = fmaxf(sc_8, -1.7014118346046923e+38f);
        sc_8 = fminf(sc_8, 1.7014118346046923e+38f);
        unsigned int u8_5 = __as_u32(sc_8);
        unsigned int key8_5 = 0;
        if ((u8_5 & 4294966784u) == qf7) {
          unsigned int lowb8_5 = (u8_5 ^ (unsigned int)((int)u8_5 >> 31) & 511) & 511;
          key8_5 = 536870912 | lowb8_5 << 9 | (unsigned int)t4_4_21;
        }
        int f2_5 = 0;
        if ((t4_4_21 < fb || t4_4_21 >= limv_fe) && t4_4_21 < limv) {
          f2_5 = 1;
        }
        if (f2_5 != 0) {
          key8_5 = 0;
        }
        kk2[5] = key8_5;
      }
      kk2[6] = 0;
      int s5_168 = lane_1 + 192;
      int j5_169 = s5_168 >> 4 & 0;
      int v5_170 = 0;
      int s4_171 = lane_1 + 192;
      int i4_172 = s4_171 & 15;
      int j4_173 = s4_171 >> 4 & 0;
      int g4_174 = whi * 32 + (s4_171 >> 4);
      int t4_175 = (j4_173 * 32 + g4_174) * 16 + i4_172;
      if (j5_169 < num_chunks && t4_175 < limv) {
        v5_170 = 1;
      }
      if (v5_170 != 0) {
        int s4_0_22 = lane_1 + 192;
        int i4_1_22 = s4_0_22 & 15;
        int j4_2_22 = s4_0_22 >> 4 & 0;
        int g4_3_22 = whi * 32 + (s4_0_22 >> 4);
        int t4_4_22 = (j4_2_22 * 32 + g4_3_22) * 16 + i4_1_22;
        float sc_10 = __uint_as_float(bb[6]);
        sc_10 = fmaxf(sc_10, -1.7014118346046923e+38f);
        sc_10 = fminf(sc_10, 1.7014118346046923e+38f);
        unsigned int u8_6 = __as_u32(sc_10);
        unsigned int key8_6 = 0;
        if ((u8_6 & 4294966784u) == qf7) {
          unsigned int lowb8_6 = (u8_6 ^ (unsigned int)((int)u8_6 >> 31) & 511) & 511;
          key8_6 = 536870912 | lowb8_6 << 9 | (unsigned int)t4_4_22;
        }
        int f2_6 = 0;
        if ((t4_4_22 < fb || t4_4_22 >= limv_fe) && t4_4_22 < limv) {
          f2_6 = 1;
        }
        if (f2_6 != 0) {
          key8_6 = 0;
        }
        kk2[6] = key8_6;
      }
      kk2[7] = 0;
      int s5_176 = lane_1 + 224;
      int j5_177 = s5_176 >> 4 & 0;
      int v5_178 = 0;
      int s4_179 = lane_1 + 224;
      int i4_180 = s4_179 & 15;
      int j4_181 = s4_179 >> 4 & 0;
      int g4_182 = whi * 32 + (s4_179 >> 4);
      int t4_183 = (j4_181 * 32 + g4_182) * 16 + i4_180;
      if (j5_177 < num_chunks && t4_183 < limv) {
        v5_178 = 1;
      }
      if (v5_178 != 0) {
        int s4_0_23 = lane_1 + 224;
        int i4_1_23 = s4_0_23 & 15;
        int j4_2_23 = s4_0_23 >> 4 & 0;
        int g4_3_23 = whi * 32 + (s4_0_23 >> 4);
        int t4_4_23 = (j4_2_23 * 32 + g4_3_23) * 16 + i4_1_23;
        float sc_11 = __uint_as_float(bb[7]);
        sc_11 = fmaxf(sc_11, -1.7014118346046923e+38f);
        sc_11 = fminf(sc_11, 1.7014118346046923e+38f);
        unsigned int u8_7 = __as_u32(sc_11);
        unsigned int key8_7 = 0;
        if ((u8_7 & 4294966784u) == qf7) {
          unsigned int lowb8_7 = (u8_7 ^ (unsigned int)((int)u8_7 >> 31) & 511) & 511;
          key8_7 = 536870912 | lowb8_7 << 9 | (unsigned int)t4_4_23;
        }
        int f2_7 = 0;
        if ((t4_4_23 < fb || t4_4_23 >= limv_fe) && t4_4_23 < limv) {
          f2_7 = 1;
        }
        if (f2_7 != 0) {
          key8_7 = 0;
        }
        kk2[7] = key8_7;
      }
      kk2[8] = 0;
      int s5_184 = lane_1 + 256;
      int j5_185 = s5_184 >> 4 & 0;
      int v5_186 = 0;
      int s4_187 = lane_1 + 256;
      int i4_188 = s4_187 & 15;
      int j4_189 = s4_187 >> 4 & 0;
      int g4_190 = whi * 32 + (s4_187 >> 4);
      int t4_191 = (j4_189 * 32 + g4_190) * 16 + i4_188;
      if (j5_185 < num_chunks && t4_191 < limv) {
        v5_186 = 1;
      }
      if (v5_186 != 0) {
        int s4_0_24 = lane_1 + 256;
        int i4_1_24 = s4_0_24 & 15;
        int j4_2_24 = s4_0_24 >> 4 & 0;
        int g4_3_24 = whi * 32 + (s4_0_24 >> 4);
        int t4_4_24 = (j4_2_24 * 32 + g4_3_24) * 16 + i4_1_24;
        float sc_13 = __uint_as_float(bb[8]);
        sc_13 = fmaxf(sc_13, -1.7014118346046923e+38f);
        sc_13 = fminf(sc_13, 1.7014118346046923e+38f);
        unsigned int u8_8 = __as_u32(sc_13);
        unsigned int key8_8 = 0;
        if ((u8_8 & 4294966784u) == qf7) {
          unsigned int lowb8_8 = (u8_8 ^ (unsigned int)((int)u8_8 >> 31) & 511) & 511;
          key8_8 = 536870912 | lowb8_8 << 9 | (unsigned int)t4_4_24;
        }
        int f2_8 = 0;
        if ((t4_4_24 < fb || t4_4_24 >= limv_fe) && t4_4_24 < limv) {
          f2_8 = 1;
        }
        if (f2_8 != 0) {
          key8_8 = 0;
        }
        kk2[8] = key8_8;
      }
      kk2[9] = 0;
      int s5_192 = lane_1 + 288;
      int j5_193 = s5_192 >> 4 & 0;
      int v5_194 = 0;
      int s4_195 = lane_1 + 288;
      int i4_196 = s4_195 & 15;
      int j4_197 = s4_195 >> 4 & 0;
      int g4_198 = whi * 32 + (s4_195 >> 4);
      int t4_199 = (j4_197 * 32 + g4_198) * 16 + i4_196;
      if (j5_193 < num_chunks && t4_199 < limv) {
        v5_194 = 1;
      }
      if (v5_194 != 0) {
        int s4_0_25 = lane_1 + 288;
        int i4_1_25 = s4_0_25 & 15;
        int j4_2_25 = s4_0_25 >> 4 & 0;
        int g4_3_25 = whi * 32 + (s4_0_25 >> 4);
        int t4_4_25 = (j4_2_25 * 32 + g4_3_25) * 16 + i4_1_25;
        float sc_14 = __uint_as_float(bb[9]);
        sc_14 = fmaxf(sc_14, -1.7014118346046923e+38f);
        sc_14 = fminf(sc_14, 1.7014118346046923e+38f);
        unsigned int u8_9 = __as_u32(sc_14);
        unsigned int key8_9 = 0;
        if ((u8_9 & 4294966784u) == qf7) {
          unsigned int lowb8_9 = (u8_9 ^ (unsigned int)((int)u8_9 >> 31) & 511) & 511;
          key8_9 = 536870912 | lowb8_9 << 9 | (unsigned int)t4_4_25;
        }
        int f2_9 = 0;
        if ((t4_4_25 < fb || t4_4_25 >= limv_fe) && t4_4_25 < limv) {
          f2_9 = 1;
        }
        if (f2_9 != 0) {
          key8_9 = 0;
        }
        kk2[9] = key8_9;
      }
      kk2[10] = 0;
      int s5_200 = lane_1 + 320;
      int j5_201 = s5_200 >> 4 & 0;
      int v5_202 = 0;
      int s4_203 = lane_1 + 320;
      int i4_204 = s4_203 & 15;
      int j4_205 = s4_203 >> 4 & 0;
      int g4_206 = whi * 32 + (s4_203 >> 4);
      int t4_207 = (j4_205 * 32 + g4_206) * 16 + i4_204;
      if (j5_201 < num_chunks && t4_207 < limv) {
        v5_202 = 1;
      }
      if (v5_202 != 0) {
        int s4_0_26 = lane_1 + 320;
        int i4_1_26 = s4_0_26 & 15;
        int j4_2_26 = s4_0_26 >> 4 & 0;
        int g4_3_26 = whi * 32 + (s4_0_26 >> 4);
        int t4_4_26 = (j4_2_26 * 32 + g4_3_26) * 16 + i4_1_26;
        float sc_16 = __uint_as_float(bb[10]);
        sc_16 = fmaxf(sc_16, -1.7014118346046923e+38f);
        sc_16 = fminf(sc_16, 1.7014118346046923e+38f);
        unsigned int u8_10 = __as_u32(sc_16);
        unsigned int key8_10 = 0;
        if ((u8_10 & 4294966784u) == qf7) {
          unsigned int lowb8_10 = (u8_10 ^ (unsigned int)((int)u8_10 >> 31) & 511) & 511;
          key8_10 = 536870912 | lowb8_10 << 9 | (unsigned int)t4_4_26;
        }
        int f2_10 = 0;
        if ((t4_4_26 < fb || t4_4_26 >= limv_fe) && t4_4_26 < limv) {
          f2_10 = 1;
        }
        if (f2_10 != 0) {
          key8_10 = 0;
        }
        kk2[10] = key8_10;
      }
      kk2[11] = 0;
      int s5_208 = lane_1 + 352;
      int j5_209 = s5_208 >> 4 & 0;
      int v5_210 = 0;
      int s4_211 = lane_1 + 352;
      int i4_212 = s4_211 & 15;
      int j4_213 = s4_211 >> 4 & 0;
      int g4_214 = whi * 32 + (s4_211 >> 4);
      int t4_215 = (j4_213 * 32 + g4_214) * 16 + i4_212;
      if (j5_209 < num_chunks && t4_215 < limv) {
        v5_210 = 1;
      }
      if (v5_210 != 0) {
        int s4_0_27 = lane_1 + 352;
        int i4_1_27 = s4_0_27 & 15;
        int j4_2_27 = s4_0_27 >> 4 & 0;
        int g4_3_27 = whi * 32 + (s4_0_27 >> 4);
        int t4_4_27 = (j4_2_27 * 32 + g4_3_27) * 16 + i4_1_27;
        float sc_17 = __uint_as_float(bb[11]);
        sc_17 = fmaxf(sc_17, -1.7014118346046923e+38f);
        sc_17 = fminf(sc_17, 1.7014118346046923e+38f);
        unsigned int u8_11 = __as_u32(sc_17);
        unsigned int key8_11 = 0;
        if ((u8_11 & 4294966784u) == qf7) {
          unsigned int lowb8_11 = (u8_11 ^ (unsigned int)((int)u8_11 >> 31) & 511) & 511;
          key8_11 = 536870912 | lowb8_11 << 9 | (unsigned int)t4_4_27;
        }
        int f2_11 = 0;
        if ((t4_4_27 < fb || t4_4_27 >= limv_fe) && t4_4_27 < limv) {
          f2_11 = 1;
        }
        if (f2_11 != 0) {
          key8_11 = 0;
        }
        kk2[11] = key8_11;
      }
      kk2[12] = 0;
      int s5_216 = lane_1 + 384;
      int j5_217 = s5_216 >> 4 & 0;
      int v5_218 = 0;
      int s4_219 = lane_1 + 384;
      int i4_220 = s4_219 & 15;
      int j4_221 = s4_219 >> 4 & 0;
      int g4_222 = whi * 32 + (s4_219 >> 4);
      int t4_223 = (j4_221 * 32 + g4_222) * 16 + i4_220;
      if (j5_217 < num_chunks && t4_223 < limv) {
        v5_218 = 1;
      }
      if (v5_218 != 0) {
        int s4_0_28 = lane_1 + 384;
        int i4_1_28 = s4_0_28 & 15;
        int j4_2_28 = s4_0_28 >> 4 & 0;
        int g4_3_28 = whi * 32 + (s4_0_28 >> 4);
        int t4_4_28 = (j4_2_28 * 32 + g4_3_28) * 16 + i4_1_28;
        float sc_19 = __uint_as_float(bb[12]);
        sc_19 = fmaxf(sc_19, -1.7014118346046923e+38f);
        sc_19 = fminf(sc_19, 1.7014118346046923e+38f);
        unsigned int u8_12 = __as_u32(sc_19);
        unsigned int key8_12 = 0;
        if ((u8_12 & 4294966784u) == qf7) {
          unsigned int lowb8_12 = (u8_12 ^ (unsigned int)((int)u8_12 >> 31) & 511) & 511;
          key8_12 = 536870912 | lowb8_12 << 9 | (unsigned int)t4_4_28;
        }
        int f2_12 = 0;
        if ((t4_4_28 < fb || t4_4_28 >= limv_fe) && t4_4_28 < limv) {
          f2_12 = 1;
        }
        if (f2_12 != 0) {
          key8_12 = 0;
        }
        kk2[12] = key8_12;
      }
      kk2[13] = 0;
      int s5_224 = lane_1 + 416;
      int j5_225 = s5_224 >> 4 & 0;
      int v5_226 = 0;
      int s4_227 = lane_1 + 416;
      int i4_228 = s4_227 & 15;
      int j4_229 = s4_227 >> 4 & 0;
      int g4_230 = whi * 32 + (s4_227 >> 4);
      int t4_231 = (j4_229 * 32 + g4_230) * 16 + i4_228;
      if (j5_225 < num_chunks && t4_231 < limv) {
        v5_226 = 1;
      }
      if (v5_226 != 0) {
        int s4_0_29 = lane_1 + 416;
        int i4_1_29 = s4_0_29 & 15;
        int j4_2_29 = s4_0_29 >> 4 & 0;
        int g4_3_29 = whi * 32 + (s4_0_29 >> 4);
        int t4_4_29 = (j4_2_29 * 32 + g4_3_29) * 16 + i4_1_29;
        float sc_20 = __uint_as_float(bb[13]);
        sc_20 = fmaxf(sc_20, -1.7014118346046923e+38f);
        sc_20 = fminf(sc_20, 1.7014118346046923e+38f);
        unsigned int u8_13 = __as_u32(sc_20);
        unsigned int key8_13 = 0;
        if ((u8_13 & 4294966784u) == qf7) {
          unsigned int lowb8_13 = (u8_13 ^ (unsigned int)((int)u8_13 >> 31) & 511) & 511;
          key8_13 = 536870912 | lowb8_13 << 9 | (unsigned int)t4_4_29;
        }
        int f2_13 = 0;
        if ((t4_4_29 < fb || t4_4_29 >= limv_fe) && t4_4_29 < limv) {
          f2_13 = 1;
        }
        if (f2_13 != 0) {
          key8_13 = 0;
        }
        kk2[13] = key8_13;
      }
      kk2[14] = 0;
      int s5_232 = lane_1 + 448;
      int j5_233 = s5_232 >> 4 & 0;
      int v5_234 = 0;
      int s4_235 = lane_1 + 448;
      int i4_236 = s4_235 & 15;
      int j4_237 = s4_235 >> 4 & 0;
      int g4_238 = whi * 32 + (s4_235 >> 4);
      int t4_239 = (j4_237 * 32 + g4_238) * 16 + i4_236;
      if (j5_233 < num_chunks && t4_239 < limv) {
        v5_234 = 1;
      }
      if (v5_234 != 0) {
        int s4_0_30 = lane_1 + 448;
        int i4_1_30 = s4_0_30 & 15;
        int j4_2_30 = s4_0_30 >> 4 & 0;
        int g4_3_30 = whi * 32 + (s4_0_30 >> 4);
        int t4_4_30 = (j4_2_30 * 32 + g4_3_30) * 16 + i4_1_30;
        float sc_22 = __uint_as_float(bb[14]);
        sc_22 = fmaxf(sc_22, -1.7014118346046923e+38f);
        sc_22 = fminf(sc_22, 1.7014118346046923e+38f);
        unsigned int u8_14 = __as_u32(sc_22);
        unsigned int key8_14 = 0;
        if ((u8_14 & 4294966784u) == qf7) {
          unsigned int lowb8_14 = (u8_14 ^ (unsigned int)((int)u8_14 >> 31) & 511) & 511;
          key8_14 = 536870912 | lowb8_14 << 9 | (unsigned int)t4_4_30;
        }
        int f2_14 = 0;
        if ((t4_4_30 < fb || t4_4_30 >= limv_fe) && t4_4_30 < limv) {
          f2_14 = 1;
        }
        if (f2_14 != 0) {
          key8_14 = 0;
        }
        kk2[14] = key8_14;
      }
      kk2[15] = 0;
      int s5_240 = lane_1 + 480;
      int j5_241 = s5_240 >> 4 & 0;
      int v5_242 = 0;
      int s4_243 = lane_1 + 480;
      int i4_244 = s4_243 & 15;
      int j4_245 = s4_243 >> 4 & 0;
      int g4_246 = whi * 32 + (s4_243 >> 4);
      int t4_247 = (j4_245 * 32 + g4_246) * 16 + i4_244;
      if (j5_241 < num_chunks && t4_247 < limv) {
        v5_242 = 1;
      }
      if (v5_242 != 0) {
        int s4_0_31 = lane_1 + 480;
        int i4_1_31 = s4_0_31 & 15;
        int j4_2_31 = s4_0_31 >> 4 & 0;
        int g4_3_31 = whi * 32 + (s4_0_31 >> 4);
        int t4_4_31 = (j4_2_31 * 32 + g4_3_31) * 16 + i4_1_31;
        float sc_23 = __uint_as_float(bb[15]);
        sc_23 = fmaxf(sc_23, -1.7014118346046923e+38f);
        sc_23 = fminf(sc_23, 1.7014118346046923e+38f);
        unsigned int u8_15 = __as_u32(sc_23);
        unsigned int key8_15 = 0;
        if ((u8_15 & 4294966784u) == qf7) {
          unsigned int lowb8_15 = (u8_15 ^ (unsigned int)((int)u8_15 >> 31) & 511) & 511;
          key8_15 = 536870912 | lowb8_15 << 9 | (unsigned int)t4_4_31;
        }
        int f2_15 = 0;
        if ((t4_4_31 < fb || t4_4_31 >= limv_fe) && t4_4_31 < limv) {
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
        int t0_0_1 = w * 16;
        long long p_1_1 = cbase + (long long)t0_0_1 * nq64;
        cb2[0] = 4286578688u;
        if (lim2x > t0_0_1) {
          cb2[0] = S[p_1_1];
        }
        cb2[1] = 4286578688u;
        if (lim2x > t0_0_1 + 1) {
          cb2[1] = S[p_1_1 + nq64];
        }
        cb2[2] = 4286578688u;
        if (lim2x > t0_0_1 + 2) {
          cb2[2] = S[p_1_1 + 2 * nq64];
        }
        cb2[3] = 4286578688u;
        if (lim2x > t0_0_1 + 3) {
          cb2[3] = S[p_1_1 + 3 * nq64];
        }
        cb2[4] = 4286578688u;
        if (lim2x > t0_0_1 + 4) {
          cb2[4] = S[p_1_1 + 4 * nq64];
        }
        cb2[5] = 4286578688u;
        if (lim2x > t0_0_1 + 5) {
          cb2[5] = S[p_1_1 + 5 * nq64];
        }
        cb2[6] = 4286578688u;
        if (lim2x > t0_0_1 + 6) {
          cb2[6] = S[p_1_1 + 6 * nq64];
        }
        cb2[7] = 4286578688u;
        if (lim2x > t0_0_1 + 7) {
          cb2[7] = S[p_1_1 + 7 * nq64];
        }
        cb2[8] = 4286578688u;
        if (lim2x > t0_0_1 + 8) {
          cb2[8] = S[p_1_1 + 8 * nq64];
        }
        cb2[9] = 4286578688u;
        if (lim2x > t0_0_1 + 9) {
          cb2[9] = S[p_1_1 + 9 * nq64];
        }
        cb2[10] = 4286578688u;
        if (lim2x > t0_0_1 + 10) {
          cb2[10] = S[p_1_1 + 10 * nq64];
        }
        cb2[11] = 4286578688u;
        if (lim2x > t0_0_1 + 11) {
          cb2[11] = S[p_1_1 + 11 * nq64];
        }
        cb2[12] = 4286578688u;
        if (lim2x > t0_0_1 + 12) {
          cb2[12] = S[p_1_1 + 12 * nq64];
        }
        cb2[13] = 4286578688u;
        if (lim2x > t0_0_1 + 13) {
          cb2[13] = S[p_1_1 + 13 * nq64];
        }
        cb2[14] = 4286578688u;
        if (lim2x > t0_0_1 + 14) {
          cb2[14] = S[p_1_1 + 14 * nq64];
        }
        cb2[15] = 4286578688u;
        if (lim2x > t0_0_1 + 15) {
          cb2[15] = S[p_1_1 + 15 * nq64];
        }
        asm volatile("" ::: "memory");
#pragma unroll 1
        for (int j_1 = 0; j_1 < num_chunks; j_1++) {
          int t0_1 = ((j_1 + 1) * 32 + w) * 16;
          long long p_2 = cbase + (long long)t0_1 * nq64;
          nb2[0] = 4286578688u;
          if (lim2x > t0_1) {
            nb2[0] = S[p_2];
          }
          nb2[1] = 4286578688u;
          if (lim2x > t0_1 + 1) {
            nb2[1] = S[p_2 + nq64];
          }
          nb2[2] = 4286578688u;
          if (lim2x > t0_1 + 2) {
            nb2[2] = S[p_2 + 2 * nq64];
          }
          nb2[3] = 4286578688u;
          if (lim2x > t0_1 + 3) {
            nb2[3] = S[p_2 + 3 * nq64];
          }
          nb2[4] = 4286578688u;
          if (lim2x > t0_1 + 4) {
            nb2[4] = S[p_2 + 4 * nq64];
          }
          nb2[5] = 4286578688u;
          if (lim2x > t0_1 + 5) {
            nb2[5] = S[p_2 + 5 * nq64];
          }
          nb2[6] = 4286578688u;
          if (lim2x > t0_1 + 6) {
            nb2[6] = S[p_2 + 6 * nq64];
          }
          nb2[7] = 4286578688u;
          if (lim2x > t0_1 + 7) {
            nb2[7] = S[p_2 + 7 * nq64];
          }
          nb2[8] = 4286578688u;
          if (lim2x > t0_1 + 8) {
            nb2[8] = S[p_2 + 8 * nq64];
          }
          nb2[9] = 4286578688u;
          if (lim2x > t0_1 + 9) {
            nb2[9] = S[p_2 + 9 * nq64];
          }
          nb2[10] = 4286578688u;
          if (lim2x > t0_1 + 10) {
            nb2[10] = S[p_2 + 10 * nq64];
          }
          nb2[11] = 4286578688u;
          if (lim2x > t0_1 + 11) {
            nb2[11] = S[p_2 + 11 * nq64];
          }
          nb2[12] = 4286578688u;
          if (lim2x > t0_1 + 12) {
            nb2[12] = S[p_2 + 12 * nq64];
          }
          nb2[13] = 4286578688u;
          if (lim2x > t0_1 + 13) {
            nb2[13] = S[p_2 + 13 * nq64];
          }
          nb2[14] = 4286578688u;
          if (lim2x > t0_1 + 14) {
            nb2[14] = S[p_2 + 14 * nq64];
          }
          nb2[15] = 4286578688u;
          if (lim2x > t0_1 + 15) {
            nb2[15] = S[p_2 + 15 * nq64];
          }
          asm volatile("" ::: "memory");
          int t0_3 = (j_1 * 32 + w) * 16;
          float qf = __uint_as_float(qc);
          float sc_25 = __uint_as_float(cb2[0]);
          sc_25 = fmaxf(sc_25, -1.7014118346046923e+38f);
          sc_25 = fminf(sc_25, 1.7014118346046923e+38f);
          unsigned int u = __as_u32(sc_25);
          unsigned int cls = u & 4294966784u;
          int f_1 = 0;
          if ((t0_3 < fb || lim_fe <= t0_3) && lim > t0_3) {
            f_1 = 1;
          }
          if (f_1 != 0) {
            cls = 2139094528;
          }
          unsigned int key_1 = 0;
          if (qf < __uint_as_float(cls) && cls < 4278190080u) {
            key_1 = 1073741824 | (unsigned int)t0_3;
          }
          if (cls == qc) {
            unsigned int lowb = (u ^ (unsigned int)((int)u >> 31) & 511) & 511;
            key_1 = 536870912 | lowb << 9 | (unsigned int)t0_3;
          }
          kb2[0] = __uint_as_float(key_1);
          float sc_4_1 = __uint_as_float(cb2[1]);
          sc_4_1 = fmaxf(sc_4_1, -1.7014118346046923e+38f);
          sc_4_1 = fminf(sc_4_1, 1.7014118346046923e+38f);
          unsigned int u_5 = __as_u32(sc_4_1);
          unsigned int cls_6 = u_5 & 4294966784u;
          int f_7 = 0;
          if ((t0_3 + 1 < fb || lim_fe <= t0_3 + 1) && lim > t0_3 + 1) {
            f_7 = 1;
          }
          if (f_7 != 0) {
            cls_6 = 2139094528;
          }
          unsigned int key_8 = 0;
          if (qf < __uint_as_float(cls_6) && cls_6 < 4278190080u) {
            key_8 = 1073741824 | (unsigned int)(t0_3 + 1);
          }
          if (cls_6 == qc) {
            unsigned int lowb_1 = (u_5 ^ (unsigned int)((int)u_5 >> 31) & 511) & 511;
            key_8 = 536870912 | lowb_1 << 9 | (unsigned int)(t0_3 + 1);
          }
          kb2[1] = __uint_as_float(key_8);
          float sc_9_1 = __uint_as_float(cb2[2]);
          sc_9_1 = fmaxf(sc_9_1, -1.7014118346046923e+38f);
          sc_9_1 = fminf(sc_9_1, 1.7014118346046923e+38f);
          unsigned int u_10 = __as_u32(sc_9_1);
          unsigned int cls_11 = u_10 & 4294966784u;
          int f_12 = 0;
          if ((t0_3 + 2 < fb || lim_fe <= t0_3 + 2) && lim > t0_3 + 2) {
            f_12 = 1;
          }
          if (f_12 != 0) {
            cls_11 = 2139094528;
          }
          unsigned int key_13_1 = 0;
          if (qf < __uint_as_float(cls_11) && cls_11 < 4278190080u) {
            key_13_1 = 1073741824 | (unsigned int)(t0_3 + 2);
          }
          if (cls_11 == qc) {
            unsigned int lowb_2 = (u_10 ^ (unsigned int)((int)u_10 >> 31) & 511) & 511;
            key_13_1 = 536870912 | lowb_2 << 9 | (unsigned int)(t0_3 + 2);
          }
          kb2[2] = __uint_as_float(key_13_1);
          float sc_14_1 = __uint_as_float(cb2[3]);
          sc_14_1 = fmaxf(sc_14_1, -1.7014118346046923e+38f);
          sc_14_1 = fminf(sc_14_1, 1.7014118346046923e+38f);
          unsigned int u_15 = __as_u32(sc_14_1);
          unsigned int cls_16 = u_15 & 4294966784u;
          int f_17_1 = 0;
          if ((t0_3 + 3 < fb || lim_fe <= t0_3 + 3) && lim > t0_3 + 3) {
            f_17_1 = 1;
          }
          if (f_17_1 != 0) {
            cls_16 = 2139094528;
          }
          unsigned int key_18 = 0;
          if (qf < __uint_as_float(cls_16) && cls_16 < 4278190080u) {
            key_18 = 1073741824 | (unsigned int)(t0_3 + 3);
          }
          if (cls_16 == qc) {
            unsigned int lowb_3 = (u_15 ^ (unsigned int)((int)u_15 >> 31) & 511) & 511;
            key_18 = 536870912 | lowb_3 << 9 | (unsigned int)(t0_3 + 3);
          }
          kb2[3] = __uint_as_float(key_18);
          float sc_19_1 = __uint_as_float(cb2[4]);
          sc_19_1 = fmaxf(sc_19_1, -1.7014118346046923e+38f);
          sc_19_1 = fminf(sc_19_1, 1.7014118346046923e+38f);
          unsigned int u_20 = __as_u32(sc_19_1);
          unsigned int cls_21 = u_20 & 4294966784u;
          int f_22 = 0;
          if ((t0_3 + 4 < fb || lim_fe <= t0_3 + 4) && lim > t0_3 + 4) {
            f_22 = 1;
          }
          if (f_22 != 0) {
            cls_21 = 2139094528;
          }
          unsigned int key_23 = 0;
          if (qf < __uint_as_float(cls_21) && cls_21 < 4278190080u) {
            key_23 = 1073741824 | (unsigned int)(t0_3 + 4);
          }
          if (cls_21 == qc) {
            unsigned int lowb_4 = (u_20 ^ (unsigned int)((int)u_20 >> 31) & 511) & 511;
            key_23 = 536870912 | lowb_4 << 9 | (unsigned int)(t0_3 + 4);
          }
          kb2[4] = __uint_as_float(key_23);
          float sc_24_1 = __uint_as_float(cb2[5]);
          sc_24_1 = fmaxf(sc_24_1, -1.7014118346046923e+38f);
          sc_24_1 = fminf(sc_24_1, 1.7014118346046923e+38f);
          unsigned int u_25 = __as_u32(sc_24_1);
          unsigned int cls_26 = u_25 & 4294966784u;
          int f_27 = 0;
          if ((t0_3 + 5 < fb || lim_fe <= t0_3 + 5) && lim > t0_3 + 5) {
            f_27 = 1;
          }
          if (f_27 != 0) {
            cls_26 = 2139094528;
          }
          unsigned int key_28_1 = 0;
          if (qf < __uint_as_float(cls_26) && cls_26 < 4278190080u) {
            key_28_1 = 1073741824 | (unsigned int)(t0_3 + 5);
          }
          if (cls_26 == qc) {
            unsigned int lowb_5 = (u_25 ^ (unsigned int)((int)u_25 >> 31) & 511) & 511;
            key_28_1 = 536870912 | lowb_5 << 9 | (unsigned int)(t0_3 + 5);
          }
          kb2[5] = __uint_as_float(key_28_1);
          float sc_29 = __uint_as_float(cb2[6]);
          sc_29 = fmaxf(sc_29, -1.7014118346046923e+38f);
          sc_29 = fminf(sc_29, 1.7014118346046923e+38f);
          unsigned int u_30 = __as_u32(sc_29);
          unsigned int cls_31 = u_30 & 4294966784u;
          int f_32_1 = 0;
          if ((t0_3 + 6 < fb || lim_fe <= t0_3 + 6) && lim > t0_3 + 6) {
            f_32_1 = 1;
          }
          if (f_32_1 != 0) {
            cls_31 = 2139094528;
          }
          unsigned int key_33 = 0;
          if (qf < __uint_as_float(cls_31) && cls_31 < 4278190080u) {
            key_33 = 1073741824 | (unsigned int)(t0_3 + 6);
          }
          if (cls_31 == qc) {
            unsigned int lowb_6 = (u_30 ^ (unsigned int)((int)u_30 >> 31) & 511) & 511;
            key_33 = 536870912 | lowb_6 << 9 | (unsigned int)(t0_3 + 6);
          }
          kb2[6] = __uint_as_float(key_33);
          float sc_34 = __uint_as_float(cb2[7]);
          sc_34 = fmaxf(sc_34, -1.7014118346046923e+38f);
          sc_34 = fminf(sc_34, 1.7014118346046923e+38f);
          unsigned int u_35 = __as_u32(sc_34);
          unsigned int cls_36 = u_35 & 4294966784u;
          int f_37 = 0;
          if ((t0_3 + 7 < fb || lim_fe <= t0_3 + 7) && lim > t0_3 + 7) {
            f_37 = 1;
          }
          if (f_37 != 0) {
            cls_36 = 2139094528;
          }
          unsigned int key_38 = 0;
          if (qf < __uint_as_float(cls_36) && cls_36 < 4278190080u) {
            key_38 = 1073741824 | (unsigned int)(t0_3 + 7);
          }
          if (cls_36 == qc) {
            unsigned int lowb_7 = (u_35 ^ (unsigned int)((int)u_35 >> 31) & 511) & 511;
            key_38 = 536870912 | lowb_7 << 9 | (unsigned int)(t0_3 + 7);
          }
          kb2[7] = __uint_as_float(key_38);
          float sc_39_1 = __uint_as_float(cb2[8]);
          sc_39_1 = fmaxf(sc_39_1, -1.7014118346046923e+38f);
          sc_39_1 = fminf(sc_39_1, 1.7014118346046923e+38f);
          unsigned int u_40 = __as_u32(sc_39_1);
          unsigned int cls_41 = u_40 & 4294966784u;
          int f_42 = 0;
          if ((t0_3 + 8 < fb || lim_fe <= t0_3 + 8) && lim > t0_3 + 8) {
            f_42 = 1;
          }
          if (f_42 != 0) {
            cls_41 = 2139094528;
          }
          unsigned int key_43_1 = 0;
          if (qf < __uint_as_float(cls_41) && cls_41 < 4278190080u) {
            key_43_1 = 1073741824 | (unsigned int)(t0_3 + 8);
          }
          if (cls_41 == qc) {
            unsigned int lowb_8 = (u_40 ^ (unsigned int)((int)u_40 >> 31) & 511) & 511;
            key_43_1 = 536870912 | lowb_8 << 9 | (unsigned int)(t0_3 + 8);
          }
          kb2[8] = __uint_as_float(key_43_1);
          float sc_44 = __uint_as_float(cb2[9]);
          sc_44 = fmaxf(sc_44, -1.7014118346046923e+38f);
          sc_44 = fminf(sc_44, 1.7014118346046923e+38f);
          unsigned int u_45 = __as_u32(sc_44);
          unsigned int cls_46 = u_45 & 4294966784u;
          int f_47_1 = 0;
          if ((t0_3 + 9 < fb || lim_fe <= t0_3 + 9) && lim > t0_3 + 9) {
            f_47_1 = 1;
          }
          if (f_47_1 != 0) {
            cls_46 = 2139094528;
          }
          unsigned int key_48 = 0;
          if (qf < __uint_as_float(cls_46) && cls_46 < 4278190080u) {
            key_48 = 1073741824 | (unsigned int)(t0_3 + 9);
          }
          if (cls_46 == qc) {
            unsigned int lowb_9 = (u_45 ^ (unsigned int)((int)u_45 >> 31) & 511) & 511;
            key_48 = 536870912 | lowb_9 << 9 | (unsigned int)(t0_3 + 9);
          }
          kb2[9] = __uint_as_float(key_48);
          float sc_49 = __uint_as_float(cb2[10]);
          sc_49 = fmaxf(sc_49, -1.7014118346046923e+38f);
          sc_49 = fminf(sc_49, 1.7014118346046923e+38f);
          unsigned int u_50 = __as_u32(sc_49);
          unsigned int cls_51 = u_50 & 4294966784u;
          int f_52 = 0;
          if ((t0_3 + 10 < fb || lim_fe <= t0_3 + 10) && lim > t0_3 + 10) {
            f_52 = 1;
          }
          if (f_52 != 0) {
            cls_51 = 2139094528;
          }
          unsigned int key_53 = 0;
          if (qf < __uint_as_float(cls_51) && cls_51 < 4278190080u) {
            key_53 = 1073741824 | (unsigned int)(t0_3 + 10);
          }
          if (cls_51 == qc) {
            unsigned int lowb_10 = (u_50 ^ (unsigned int)((int)u_50 >> 31) & 511) & 511;
            key_53 = 536870912 | lowb_10 << 9 | (unsigned int)(t0_3 + 10);
          }
          kb2[10] = __uint_as_float(key_53);
          float sc_54 = __uint_as_float(cb2[11]);
          sc_54 = fmaxf(sc_54, -1.7014118346046923e+38f);
          sc_54 = fminf(sc_54, 1.7014118346046923e+38f);
          unsigned int u_55 = __as_u32(sc_54);
          unsigned int cls_56 = u_55 & 4294966784u;
          int f_57 = 0;
          if ((t0_3 + 11 < fb || lim_fe <= t0_3 + 11) && lim > t0_3 + 11) {
            f_57 = 1;
          }
          if (f_57 != 0) {
            cls_56 = 2139094528;
          }
          unsigned int key_58 = 0;
          if (qf < __uint_as_float(cls_56) && cls_56 < 4278190080u) {
            key_58 = 1073741824 | (unsigned int)(t0_3 + 11);
          }
          if (cls_56 == qc) {
            unsigned int lowb_11 = (u_55 ^ (unsigned int)((int)u_55 >> 31) & 511) & 511;
            key_58 = 536870912 | lowb_11 << 9 | (unsigned int)(t0_3 + 11);
          }
          kb2[11] = __uint_as_float(key_58);
          float sc_59 = __uint_as_float(cb2[12]);
          sc_59 = fmaxf(sc_59, -1.7014118346046923e+38f);
          sc_59 = fminf(sc_59, 1.7014118346046923e+38f);
          unsigned int u_60 = __as_u32(sc_59);
          unsigned int cls_61 = u_60 & 4294966784u;
          int f_62 = 0;
          if ((t0_3 + 12 < fb || lim_fe <= t0_3 + 12) && lim > t0_3 + 12) {
            f_62 = 1;
          }
          if (f_62 != 0) {
            cls_61 = 2139094528;
          }
          unsigned int key_63 = 0;
          if (qf < __uint_as_float(cls_61) && cls_61 < 4278190080u) {
            key_63 = 1073741824 | (unsigned int)(t0_3 + 12);
          }
          if (cls_61 == qc) {
            unsigned int lowb_12 = (u_60 ^ (unsigned int)((int)u_60 >> 31) & 511) & 511;
            key_63 = 536870912 | lowb_12 << 9 | (unsigned int)(t0_3 + 12);
          }
          kb2[12] = __uint_as_float(key_63);
          float sc_64 = __uint_as_float(cb2[13]);
          sc_64 = fmaxf(sc_64, -1.7014118346046923e+38f);
          sc_64 = fminf(sc_64, 1.7014118346046923e+38f);
          unsigned int u_65 = __as_u32(sc_64);
          unsigned int cls_66 = u_65 & 4294966784u;
          int f_67 = 0;
          if ((t0_3 + 13 < fb || lim_fe <= t0_3 + 13) && lim > t0_3 + 13) {
            f_67 = 1;
          }
          if (f_67 != 0) {
            cls_66 = 2139094528;
          }
          unsigned int key_68 = 0;
          if (qf < __uint_as_float(cls_66) && cls_66 < 4278190080u) {
            key_68 = 1073741824 | (unsigned int)(t0_3 + 13);
          }
          if (cls_66 == qc) {
            unsigned int lowb_13 = (u_65 ^ (unsigned int)((int)u_65 >> 31) & 511) & 511;
            key_68 = 536870912 | lowb_13 << 9 | (unsigned int)(t0_3 + 13);
          }
          kb2[13] = __uint_as_float(key_68);
          float sc_69 = __uint_as_float(cb2[14]);
          sc_69 = fmaxf(sc_69, -1.7014118346046923e+38f);
          sc_69 = fminf(sc_69, 1.7014118346046923e+38f);
          unsigned int u_70 = __as_u32(sc_69);
          unsigned int cls_71 = u_70 & 4294966784u;
          int f_72 = 0;
          if ((t0_3 + 14 < fb || lim_fe <= t0_3 + 14) && lim > t0_3 + 14) {
            f_72 = 1;
          }
          if (f_72 != 0) {
            cls_71 = 2139094528;
          }
          unsigned int key_73 = 0;
          if (qf < __uint_as_float(cls_71) && cls_71 < 4278190080u) {
            key_73 = 1073741824 | (unsigned int)(t0_3 + 14);
          }
          if (cls_71 == qc) {
            unsigned int lowb_14 = (u_70 ^ (unsigned int)((int)u_70 >> 31) & 511) & 511;
            key_73 = 536870912 | lowb_14 << 9 | (unsigned int)(t0_3 + 14);
          }
          kb2[14] = __uint_as_float(key_73);
          float sc_74 = __uint_as_float(cb2[15]);
          sc_74 = fmaxf(sc_74, -1.7014118346046923e+38f);
          sc_74 = fminf(sc_74, 1.7014118346046923e+38f);
          unsigned int u_75 = __as_u32(sc_74);
          unsigned int cls_76 = u_75 & 4294966784u;
          int f_77 = 0;
          if ((t0_3 + 15 < fb || lim_fe <= t0_3 + 15) && lim > t0_3 + 15) {
            f_77 = 1;
          }
          if (f_77 != 0) {
            cls_76 = 2139094528;
          }
          unsigned int key_78 = 0;
          if (qf < __uint_as_float(cls_76) && cls_76 < 4278190080u) {
            key_78 = 1073741824 | (unsigned int)(t0_3 + 15);
          }
          if (cls_76 == qc) {
            unsigned int lowb_15 = (u_75 ^ (unsigned int)((int)u_75 >> 31) & 511) & 511;
            key_78 = 536870912 | lowb_15 << 9 | (unsigned int)(t0_3 + 15);
          }
          kb2[15] = __uint_as_float(key_78);
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
            float hi_0 = fmaxf(kb2[0], kb2[13]);
            float lo_1_1 = fminf(kb2[0], kb2[13]);
            kb2[0] = hi_0;
            kb2[13] = lo_1_1;
            float hi_2 = fmaxf(kb2[1], kb2[12]);
            float lo_3 = fminf(kb2[1], kb2[12]);
            kb2[1] = hi_2;
            kb2[12] = lo_3;
            float hi_5 = fmaxf(kb2[2], kb2[15]);
            float lo_6 = fminf(kb2[2], kb2[15]);
            kb2[2] = hi_5;
            kb2[15] = lo_6;
            float hi_8 = fmaxf(kb2[3], kb2[14]);
            float lo_9 = fminf(kb2[3], kb2[14]);
            kb2[3] = hi_8;
            kb2[14] = lo_9;
            float hi_10 = fmaxf(kb2[4], kb2[8]);
            float lo_11 = fminf(kb2[4], kb2[8]);
            kb2[4] = hi_10;
            kb2[8] = lo_11;
            float hi_12 = fmaxf(kb2[5], kb2[6]);
            float lo_13 = fminf(kb2[5], kb2[6]);
            kb2[5] = hi_12;
            kb2[6] = lo_13;
            float hi_14 = fmaxf(kb2[7], kb2[11]);
            float lo_15 = fminf(kb2[7], kb2[11]);
            kb2[7] = hi_14;
            kb2[11] = lo_15;
            float hi_16 = fmaxf(kb2[9], kb2[10]);
            float lo_17 = fminf(kb2[9], kb2[10]);
            kb2[9] = hi_16;
            kb2[10] = lo_17;
            float hi_18 = fmaxf(kb2[0], kb2[5]);
            float lo_19 = fminf(kb2[0], kb2[5]);
            kb2[0] = hi_18;
            kb2[5] = lo_19;
            float hi_21 = fmaxf(kb2[1], kb2[7]);
            float lo_22 = fminf(kb2[1], kb2[7]);
            kb2[1] = hi_21;
            kb2[7] = lo_22;
            float hi_24 = fmaxf(kb2[2], kb2[9]);
            float lo_25 = fminf(kb2[2], kb2[9]);
            kb2[2] = hi_24;
            kb2[9] = lo_25;
            float hi_27 = fmaxf(kb2[3], kb2[4]);
            float lo_28 = fminf(kb2[3], kb2[4]);
            kb2[3] = hi_27;
            kb2[4] = lo_28;
            float hi_29 = fmaxf(kb2[6], kb2[13]);
            float lo_30 = fminf(kb2[6], kb2[13]);
            kb2[6] = hi_29;
            kb2[13] = lo_30;
            float hi_31 = fmaxf(kb2[8], kb2[14]);
            float lo_32 = fminf(kb2[8], kb2[14]);
            kb2[8] = hi_31;
            kb2[14] = lo_32;
            float hi_33 = fmaxf(kb2[10], kb2[15]);
            float lo_34 = fminf(kb2[10], kb2[15]);
            kb2[10] = hi_33;
            kb2[15] = lo_34;
            float hi_35 = fmaxf(kb2[11], kb2[12]);
            float lo_36 = fminf(kb2[11], kb2[12]);
            kb2[11] = hi_35;
            kb2[12] = lo_36;
            float hi_37 = fmaxf(kb2[0], kb2[1]);
            float lo_38 = fminf(kb2[0], kb2[1]);
            kb2[0] = hi_37;
            kb2[1] = lo_38;
            float hi_40 = fmaxf(kb2[2], kb2[3]);
            float lo_41 = fminf(kb2[2], kb2[3]);
            kb2[2] = hi_40;
            kb2[3] = lo_41;
            float hi_43 = fmaxf(kb2[4], kb2[5]);
            float lo_44 = fminf(kb2[4], kb2[5]);
            kb2[4] = hi_43;
            kb2[5] = lo_44;
            float hi_46 = fmaxf(kb2[6], kb2[8]);
            float lo_47 = fminf(kb2[6], kb2[8]);
            kb2[6] = hi_46;
            kb2[8] = lo_47;
            float hi_48_1 = fmaxf(kb2[7], kb2[9]);
            float lo_49_1 = fminf(kb2[7], kb2[9]);
            kb2[7] = hi_48_1;
            kb2[9] = lo_49_1;
            float hi_50_1 = fmaxf(kb2[10], kb2[11]);
            float lo_51_1 = fminf(kb2[10], kb2[11]);
            kb2[10] = hi_50_1;
            kb2[11] = lo_51_1;
            float hi_52_1 = fmaxf(kb2[12], kb2[13]);
            float lo_53_1 = fminf(kb2[12], kb2[13]);
            kb2[12] = hi_52_1;
            kb2[13] = lo_53_1;
            float hi_54_1 = fmaxf(kb2[14], kb2[15]);
            float lo_55_1 = fminf(kb2[14], kb2[15]);
            kb2[14] = hi_54_1;
            kb2[15] = lo_55_1;
            float hi_56_1 = fmaxf(kb2[0], kb2[2]);
            float lo_57_1 = fminf(kb2[0], kb2[2]);
            kb2[0] = hi_56_1;
            kb2[2] = lo_57_1;
            float hi_59 = fmaxf(kb2[1], kb2[3]);
            float lo_60 = fminf(kb2[1], kb2[3]);
            kb2[1] = hi_59;
            kb2[3] = lo_60;
            float hi_62_1 = fmaxf(kb2[4], kb2[10]);
            float lo_63_1 = fminf(kb2[4], kb2[10]);
            kb2[4] = hi_62_1;
            kb2[10] = lo_63_1;
            float hi_65 = fmaxf(kb2[5], kb2[11]);
            float lo_66 = fminf(kb2[5], kb2[11]);
            kb2[5] = hi_65;
            kb2[11] = lo_66;
            float hi_67 = fmaxf(kb2[6], kb2[7]);
            float lo_68 = fminf(kb2[6], kb2[7]);
            kb2[6] = hi_67;
            kb2[7] = lo_68;
            float hi_69 = fmaxf(kb2[8], kb2[9]);
            float lo_70 = fminf(kb2[8], kb2[9]);
            kb2[8] = hi_69;
            kb2[9] = lo_70;
            float hi_71 = fmaxf(kb2[12], kb2[14]);
            float lo_72 = fminf(kb2[12], kb2[14]);
            kb2[12] = hi_71;
            kb2[14] = lo_72;
            float hi_73 = fmaxf(kb2[13], kb2[15]);
            float lo_74 = fminf(kb2[13], kb2[15]);
            kb2[13] = hi_73;
            kb2[15] = lo_74;
            float hi_75 = fmaxf(kb2[1], kb2[2]);
            float lo_76 = fminf(kb2[1], kb2[2]);
            kb2[1] = hi_75;
            kb2[2] = lo_76;
            float hi_78_1 = fmaxf(kb2[3], kb2[12]);
            float lo_79_1 = fminf(kb2[3], kb2[12]);
            kb2[3] = hi_78_1;
            kb2[12] = lo_79_1;
            float hi_81 = fmaxf(kb2[4], kb2[6]);
            float lo_82 = fminf(kb2[4], kb2[6]);
            kb2[4] = hi_81;
            kb2[6] = lo_82;
            float hi_84_1 = fmaxf(kb2[5], kb2[7]);
            float lo_85_1 = fminf(kb2[5], kb2[7]);
            kb2[5] = hi_84_1;
            kb2[7] = lo_85_1;
            float hi_86_1 = fmaxf(kb2[8], kb2[10]);
            float lo_87_1 = fminf(kb2[8], kb2[10]);
            kb2[8] = hi_86_1;
            kb2[10] = lo_87_1;
            float hi_88_1 = fmaxf(kb2[9], kb2[11]);
            float lo_89_1 = fminf(kb2[9], kb2[11]);
            kb2[9] = hi_88_1;
            kb2[11] = lo_89_1;
            float hi_90_1 = fmaxf(kb2[13], kb2[14]);
            float lo_91_1 = fminf(kb2[13], kb2[14]);
            kb2[13] = hi_90_1;
            kb2[14] = lo_91_1;
            float hi_92_1 = fmaxf(kb2[1], kb2[4]);
            float lo_93_1 = fminf(kb2[1], kb2[4]);
            kb2[1] = hi_92_1;
            kb2[4] = lo_93_1;
            float hi_94_1 = fmaxf(kb2[2], kb2[6]);
            float lo_95_1 = fminf(kb2[2], kb2[6]);
            kb2[2] = hi_94_1;
            kb2[6] = lo_95_1;
            float hi_97 = fmaxf(kb2[5], kb2[8]);
            float lo_98 = fminf(kb2[5], kb2[8]);
            kb2[5] = hi_97;
            kb2[8] = lo_98;
            float hi_100_1 = fmaxf(kb2[7], kb2[10]);
            float lo_101_1 = fminf(kb2[7], kb2[10]);
            kb2[7] = hi_100_1;
            kb2[10] = lo_101_1;
            float hi_103 = fmaxf(kb2[9], kb2[13]);
            float lo_104 = fminf(kb2[9], kb2[13]);
            kb2[9] = hi_103;
            kb2[13] = lo_104;
            float hi_105 = fmaxf(kb2[11], kb2[14]);
            float lo_106 = fminf(kb2[11], kb2[14]);
            kb2[11] = hi_105;
            kb2[14] = lo_106;
            float hi_107 = fmaxf(kb2[2], kb2[4]);
            float lo_108 = fminf(kb2[2], kb2[4]);
            kb2[2] = hi_107;
            kb2[4] = lo_108;
            float hi_109 = fmaxf(kb2[3], kb2[6]);
            float lo_110 = fminf(kb2[3], kb2[6]);
            kb2[3] = hi_109;
            kb2[6] = lo_110;
            float hi_111 = fmaxf(kb2[9], kb2[12]);
            float lo_112 = fminf(kb2[9], kb2[12]);
            kb2[9] = hi_111;
            kb2[12] = lo_112;
            float hi_113 = fmaxf(kb2[11], kb2[13]);
            float lo_114 = fminf(kb2[11], kb2[13]);
            kb2[11] = hi_113;
            kb2[13] = lo_114;
            float hi_116_1 = fmaxf(kb2[3], kb2[5]);
            float lo_117_1 = fminf(kb2[3], kb2[5]);
            kb2[3] = hi_116_1;
            kb2[5] = lo_117_1;
            float hi_119 = fmaxf(kb2[6], kb2[8]);
            float lo_120 = fminf(kb2[6], kb2[8]);
            kb2[6] = hi_119;
            kb2[8] = lo_120;
            float hi_122_1 = fmaxf(kb2[7], kb2[9]);
            float lo_123_1 = fminf(kb2[7], kb2[9]);
            kb2[7] = hi_122_1;
            kb2[9] = lo_123_1;
            float hi_124_1 = fmaxf(kb2[10], kb2[12]);
            float lo_125_1 = fminf(kb2[10], kb2[12]);
            kb2[10] = hi_124_1;
            kb2[12] = lo_125_1;
            float hi_126_1 = fmaxf(kb2[3], kb2[4]);
            float lo_127_1 = fminf(kb2[3], kb2[4]);
            kb2[3] = hi_126_1;
            kb2[4] = lo_127_1;
            float hi_128_1 = fmaxf(kb2[5], kb2[6]);
            float lo_129_1 = fminf(kb2[5], kb2[6]);
            kb2[5] = hi_128_1;
            kb2[6] = lo_129_1;
            float hi_130_1 = fmaxf(kb2[7], kb2[8]);
            float lo_131_1 = fminf(kb2[7], kb2[8]);
            kb2[7] = hi_130_1;
            kb2[8] = lo_131_1;
            float hi_132_1 = fmaxf(kb2[9], kb2[10]);
            float lo_133_1 = fminf(kb2[9], kb2[10]);
            kb2[9] = hi_132_1;
            kb2[10] = lo_133_1;
            float hi_135 = fmaxf(kb2[11], kb2[12]);
            float lo_136 = fminf(kb2[11], kb2[12]);
            kb2[11] = hi_135;
            kb2[12] = lo_136;
            float hi_138_1 = fmaxf(kb2[6], kb2[7]);
            float lo_139_1 = fminf(kb2[6], kb2[7]);
            kb2[6] = hi_138_1;
            kb2[7] = lo_139_1;
            float hi_141 = fmaxf(kb2[8], kb2[9]);
            float lo_142 = fminf(kb2[8], kb2[9]);
            kb2[8] = hi_141;
            kb2[9] = lo_142;
            float hi_143 = fmaxf(a2x[0], kb2[15]);
            a2x[0] = hi_143;
            float hi_145 = fmaxf(a2x[1], kb2[14]);
            a2x[1] = hi_145;
            float hi_146_1 = fmaxf(a2x[2], kb2[13]);
            a2x[2] = hi_146_1;
            float hi_148_1 = fmaxf(a2x[3], kb2[12]);
            a2x[3] = hi_148_1;
            float hi_149 = fmaxf(a2x[4], kb2[11]);
            a2x[4] = hi_149;
            float hi_151 = fmaxf(a2x[5], kb2[10]);
            a2x[5] = hi_151;
            float hi_152_1 = fmaxf(a2x[6], kb2[9]);
            a2x[6] = hi_152_1;
            float hi_154_1 = fmaxf(a2x[7], kb2[8]);
            a2x[7] = hi_154_1;
            float hi_155 = fmaxf(a2x[8], kb2[7]);
            a2x[8] = hi_155;
            float hi_156_1 = fmaxf(a2x[9], kb2[6]);
            a2x[9] = hi_156_1;
            float hi_157 = fmaxf(a2x[10], kb2[5]);
            a2x[10] = hi_157;
            float hi_158_1 = fmaxf(a2x[11], kb2[4]);
            a2x[11] = hi_158_1;
            float hi_159 = fmaxf(a2x[12], kb2[3]);
            a2x[12] = hi_159;
            float hi_161 = fmaxf(a2x[13], kb2[2]);
            a2x[13] = hi_161;
            float hi_162_1 = fmaxf(a2x[14], kb2[1]);
            a2x[14] = hi_162_1;
            float hi_164_1 = fmaxf(a2x[15], kb2[0]);
            a2x[15] = hi_164_1;
            float hi_165 = fmaxf(a2x[0], a2x[8]);
            float lo_166 = fminf(a2x[0], a2x[8]);
            a2x[0] = hi_165;
            a2x[8] = lo_166;
            float hi_167 = fmaxf(a2x[1], a2x[9]);
            float lo_168 = fminf(a2x[1], a2x[9]);
            a2x[1] = hi_167;
            a2x[9] = lo_168;
            float hi_170_1 = fmaxf(a2x[2], a2x[10]);
            float lo_171_1 = fminf(a2x[2], a2x[10]);
            a2x[2] = hi_170_1;
            a2x[10] = lo_171_1;
            float hi_172_1 = fmaxf(a2x[3], a2x[11]);
            float lo_173_1 = fminf(a2x[3], a2x[11]);
            a2x[3] = hi_172_1;
            a2x[11] = lo_173_1;
            float hi_174_1 = fmaxf(a2x[4], a2x[12]);
            float lo_175_1 = fminf(a2x[4], a2x[12]);
            a2x[4] = hi_174_1;
            a2x[12] = lo_175_1;
            float hi_177 = fmaxf(a2x[5], a2x[13]);
            float lo_178 = fminf(a2x[5], a2x[13]);
            a2x[5] = hi_177;
            a2x[13] = lo_178;
            float hi_180_1 = fmaxf(a2x[6], a2x[14]);
            float lo_181_1 = fminf(a2x[6], a2x[14]);
            a2x[6] = hi_180_1;
            a2x[14] = lo_181_1;
            float hi_183 = fmaxf(a2x[7], a2x[15]);
            float lo_184 = fminf(a2x[7], a2x[15]);
            a2x[7] = hi_183;
            a2x[15] = lo_184;
            float hi_186_1 = fmaxf(a2x[0], a2x[4]);
            float lo_187_1 = fminf(a2x[0], a2x[4]);
            a2x[0] = hi_186_1;
            a2x[4] = lo_187_1;
            float hi_188_1 = fmaxf(a2x[1], a2x[5]);
            float lo_189_1 = fminf(a2x[1], a2x[5]);
            a2x[1] = hi_188_1;
            a2x[5] = lo_189_1;
            float hi_190_1 = fmaxf(a2x[2], a2x[6]);
            float lo_191_1 = fminf(a2x[2], a2x[6]);
            a2x[2] = hi_190_1;
            a2x[6] = lo_191_1;
            float hi_193 = fmaxf(a2x[3], a2x[7]);
            float lo_194 = fminf(a2x[3], a2x[7]);
            a2x[3] = hi_193;
            a2x[7] = lo_194;
            float hi_196_1 = fmaxf(a2x[8], a2x[12]);
            float lo_197_1 = fminf(a2x[8], a2x[12]);
            a2x[8] = hi_196_1;
            a2x[12] = lo_197_1;
            float hi_199 = fmaxf(a2x[9], a2x[13]);
            float lo_200 = fminf(a2x[9], a2x[13]);
            a2x[9] = hi_199;
            a2x[13] = lo_200;
            float hi_202_1 = fmaxf(a2x[10], a2x[14]);
            float lo_203_1 = fminf(a2x[10], a2x[14]);
            a2x[10] = hi_202_1;
            a2x[14] = lo_203_1;
            float hi_204_1 = fmaxf(a2x[11], a2x[15]);
            float lo_205_1 = fminf(a2x[11], a2x[15]);
            a2x[11] = hi_204_1;
            a2x[15] = lo_205_1;
            float hi_206_1 = fmaxf(a2x[0], a2x[2]);
            float lo_207_1 = fminf(a2x[0], a2x[2]);
            a2x[0] = hi_206_1;
            a2x[2] = lo_207_1;
            float hi_209 = fmaxf(a2x[1], a2x[3]);
            float lo_210 = fminf(a2x[1], a2x[3]);
            a2x[1] = hi_209;
            a2x[3] = lo_210;
            float hi_212_1 = fmaxf(a2x[4], a2x[6]);
            float lo_213_1 = fminf(a2x[4], a2x[6]);
            a2x[4] = hi_212_1;
            a2x[6] = lo_213_1;
            float hi_215 = fmaxf(a2x[5], a2x[7]);
            float lo_216 = fminf(a2x[5], a2x[7]);
            a2x[5] = hi_215;
            a2x[7] = lo_216;
            float hi_218_1 = fmaxf(a2x[8], a2x[10]);
            float lo_219_1 = fminf(a2x[8], a2x[10]);
            a2x[8] = hi_218_1;
            a2x[10] = lo_219_1;
            float hi_220_1 = fmaxf(a2x[9], a2x[11]);
            float lo_221_1 = fminf(a2x[9], a2x[11]);
            a2x[9] = hi_220_1;
            a2x[11] = lo_221_1;
            float hi_222_1 = fmaxf(a2x[12], a2x[14]);
            float lo_223_1 = fminf(a2x[12], a2x[14]);
            a2x[12] = hi_222_1;
            a2x[14] = lo_223_1;
            float hi_225 = fmaxf(a2x[13], a2x[15]);
            float lo_226 = fminf(a2x[13], a2x[15]);
            a2x[13] = hi_225;
            a2x[15] = lo_226;
            float hi_228_1 = fmaxf(a2x[0], a2x[1]);
            float lo_229_1 = fminf(a2x[0], a2x[1]);
            a2x[0] = hi_228_1;
            a2x[1] = lo_229_1;
            float hi_231 = fmaxf(a2x[2], a2x[3]);
            float lo_232 = fminf(a2x[2], a2x[3]);
            a2x[2] = hi_231;
            a2x[3] = lo_232;
            float hi_234_1 = fmaxf(a2x[4], a2x[5]);
            float lo_235_1 = fminf(a2x[4], a2x[5]);
            a2x[4] = hi_234_1;
            a2x[5] = lo_235_1;
            float hi_236_1 = fmaxf(a2x[6], a2x[7]);
            float lo_237_1 = fminf(a2x[6], a2x[7]);
            a2x[6] = hi_236_1;
            a2x[7] = lo_237_1;
            float hi_238_1 = fmaxf(a2x[8], a2x[9]);
            float lo_239_1 = fminf(a2x[8], a2x[9]);
            a2x[8] = hi_238_1;
            a2x[9] = lo_239_1;
            float hi_241 = fmaxf(a2x[10], a2x[11]);
            float lo_242 = fminf(a2x[10], a2x[11]);
            a2x[10] = hi_241;
            a2x[11] = lo_242;
            float hi_244_1 = fmaxf(a2x[12], a2x[13]);
            float lo_245_1 = fminf(a2x[12], a2x[13]);
            a2x[12] = hi_244_1;
            a2x[13] = lo_245_1;
            float hi_247 = fmaxf(a2x[14], a2x[15]);
            float lo_248 = fminf(a2x[14], a2x[15]);
            a2x[14] = hi_247;
            a2x[15] = lo_248;
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
      float r_1_1 = neg_inf;
      float V_2[8];
      int s0_3 = (sg * 16 + cg) * 17;
      int s1_4 = (sg * 16 + 8 + cg) * 17;
      float x0_5 = pub[s0_3 + ln];
      float y0_6 = pub[s1_4 + lnr];
      float lo0_7 = fminf(x0_5, y0_6);
      r_1_1 = fmaxf(r_1_1, lo0_7);
      float hi0_8 = fmaxf(x0_5, y0_6);
      float cur_9 = hi0_8;
      float pv_10 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 8);
      float hi_11 = fmaxf(cur_9, pv_10);
      float lo_12 = fminf(cur_9, pv_10);
      cur_9 = ((up[0] != 0) ? hi_11 : lo_12);
      float pv_13 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 4);
      float hi_14_1 = fmaxf(cur_9, pv_13);
      float lo_15_1 = fminf(cur_9, pv_13);
      cur_9 = ((up[1] != 0) ? hi_14_1 : lo_15_1);
      float pv_17 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 2);
      float hi_18_1 = fmaxf(cur_9, pv_17);
      float lo_19_1 = fminf(cur_9, pv_17);
      cur_9 = ((up[2] != 0) ? hi_18_1 : lo_19_1);
      float pv_20 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 1);
      float hi_21_1 = fmaxf(cur_9, pv_20);
      float lo_22_1 = fminf(cur_9, pv_20);
      cur_9 = ((up[3] != 0) ? hi_21_1 : lo_22_1);
      V_2[0] = cur_9;
      int s0_23 = (sg * 16 + 1 + cg) * 17;
      int s1_24 = (sg * 16 + 1 + 8 + cg) * 17;
      float x0_25 = pub[s0_23 + ln];
      float y0_26 = pub[s1_24 + lnr];
      float lo0_27 = fminf(x0_25, y0_26);
      r_1_1 = fmaxf(r_1_1, lo0_27);
      float hi0_28 = fmaxf(x0_25, y0_26);
      float cur_29 = hi0_28;
      float pv_30 = __shfl_xor_sync(0xFFFFFFFF, cur_29, 8);
      float hi_31_1 = fmaxf(cur_29, pv_30);
      float lo_32_1 = fminf(cur_29, pv_30);
      cur_29 = ((up[0] != 0) ? hi_31_1 : lo_32_1);
      float pv_33 = __shfl_xor_sync(0xFFFFFFFF, cur_29, 4);
      float hi_34 = fmaxf(cur_29, pv_33);
      float lo_35 = fminf(cur_29, pv_33);
      cur_29 = ((up[1] != 0) ? hi_34 : lo_35);
      float pv_36 = __shfl_xor_sync(0xFFFFFFFF, cur_29, 2);
      float hi_37_1 = fmaxf(cur_29, pv_36);
      float lo_38_1 = fminf(cur_29, pv_36);
      cur_29 = ((up[2] != 0) ? hi_37_1 : lo_38_1);
      float pv_39 = __shfl_xor_sync(0xFFFFFFFF, cur_29, 1);
      float hi_40_1 = fmaxf(cur_29, pv_39);
      float lo_41_1 = fminf(cur_29, pv_39);
      cur_29 = ((up[3] != 0) ? hi_40_1 : lo_41_1);
      V_2[1] = cur_29;
      int s0_42 = (sg * 16 + 2 + cg) * 17;
      int s1_43 = (sg * 16 + 2 + 8 + cg) * 17;
      float x0_44 = pub[s0_42 + ln];
      float y0_45 = pub[s1_43 + lnr];
      float lo0_46 = fminf(x0_44, y0_45);
      r_1_1 = fmaxf(r_1_1, lo0_46);
      float hi0_47 = fmaxf(x0_44, y0_45);
      float cur_48 = hi0_47;
      float pv_49 = __shfl_xor_sync(0xFFFFFFFF, cur_48, 8);
      float hi_50_2 = fmaxf(cur_48, pv_49);
      float lo_51_2 = fminf(cur_48, pv_49);
      cur_48 = ((up[0] != 0) ? hi_50_2 : lo_51_2);
      float pv_52 = __shfl_xor_sync(0xFFFFFFFF, cur_48, 4);
      float hi_53 = fmaxf(cur_48, pv_52);
      float lo_54 = fminf(cur_48, pv_52);
      cur_48 = ((up[1] != 0) ? hi_53 : lo_54);
      float pv_55 = __shfl_xor_sync(0xFFFFFFFF, cur_48, 2);
      float hi_56_2 = fmaxf(cur_48, pv_55);
      float lo_57_2 = fminf(cur_48, pv_55);
      cur_48 = ((up[2] != 0) ? hi_56_2 : lo_57_2);
      float pv_58 = __shfl_xor_sync(0xFFFFFFFF, cur_48, 1);
      float hi_59_1 = fmaxf(cur_48, pv_58);
      float lo_60_1 = fminf(cur_48, pv_58);
      cur_48 = ((up[3] != 0) ? hi_59_1 : lo_60_1);
      V_2[2] = cur_48;
      int s0_61 = (sg * 16 + 3 + cg) * 17;
      int s1_62 = (sg * 16 + 3 + 8 + cg) * 17;
      float x0_63 = pub[s0_61 + ln];
      float y0_64 = pub[s1_62 + lnr];
      float lo0_65 = fminf(x0_63, y0_64);
      r_1_1 = fmaxf(r_1_1, lo0_65);
      float hi0_66 = fmaxf(x0_63, y0_64);
      float cur_67 = hi0_66;
      float pv_68 = __shfl_xor_sync(0xFFFFFFFF, cur_67, 8);
      float hi_69_1 = fmaxf(cur_67, pv_68);
      float lo_70_1 = fminf(cur_67, pv_68);
      cur_67 = ((up[0] != 0) ? hi_69_1 : lo_70_1);
      float pv_71 = __shfl_xor_sync(0xFFFFFFFF, cur_67, 4);
      float hi_72_1 = fmaxf(cur_67, pv_71);
      float lo_73_1 = fminf(cur_67, pv_71);
      cur_67 = ((up[1] != 0) ? hi_72_1 : lo_73_1);
      float pv_74 = __shfl_xor_sync(0xFFFFFFFF, cur_67, 2);
      float hi_75_1 = fmaxf(cur_67, pv_74);
      float lo_76_1 = fminf(cur_67, pv_74);
      cur_67 = ((up[2] != 0) ? hi_75_1 : lo_76_1);
      float pv_77 = __shfl_xor_sync(0xFFFFFFFF, cur_67, 1);
      float hi_78_2 = fmaxf(cur_67, pv_77);
      float lo_79_2 = fminf(cur_67, pv_77);
      cur_67 = ((up[3] != 0) ? hi_78_2 : lo_79_2);
      V_2[3] = cur_67;
      int s0_80 = (sg * 16 + 4 + cg) * 17;
      int s1_81 = (sg * 16 + 4 + 8 + cg) * 17;
      float x0_82 = pub[s0_80 + ln];
      float y0_83 = pub[s1_81 + lnr];
      float lo0_84 = fminf(x0_82, y0_83);
      r_1_1 = fmaxf(r_1_1, lo0_84);
      float hi0_85 = fmaxf(x0_82, y0_83);
      float cur_86 = hi0_85;
      float pv_87 = __shfl_xor_sync(0xFFFFFFFF, cur_86, 8);
      float hi_88_2 = fmaxf(cur_86, pv_87);
      float lo_89_2 = fminf(cur_86, pv_87);
      cur_86 = ((up[0] != 0) ? hi_88_2 : lo_89_2);
      float pv_90 = __shfl_xor_sync(0xFFFFFFFF, cur_86, 4);
      float hi_91 = fmaxf(cur_86, pv_90);
      float lo_92 = fminf(cur_86, pv_90);
      cur_86 = ((up[1] != 0) ? hi_91 : lo_92);
      float pv_93 = __shfl_xor_sync(0xFFFFFFFF, cur_86, 2);
      float hi_94_2 = fmaxf(cur_86, pv_93);
      float lo_95_2 = fminf(cur_86, pv_93);
      cur_86 = ((up[2] != 0) ? hi_94_2 : lo_95_2);
      float pv_96 = __shfl_xor_sync(0xFFFFFFFF, cur_86, 1);
      float hi_97_1 = fmaxf(cur_86, pv_96);
      float lo_98_1 = fminf(cur_86, pv_96);
      cur_86 = ((up[3] != 0) ? hi_97_1 : lo_98_1);
      V_2[4] = cur_86;
      int s0_99 = (sg * 16 + 5 + cg) * 17;
      int s1_100 = (sg * 16 + 5 + 8 + cg) * 17;
      float x0_101 = pub[s0_99 + ln];
      float y0_102 = pub[s1_100 + lnr];
      float lo0_103 = fminf(x0_101, y0_102);
      r_1_1 = fmaxf(r_1_1, lo0_103);
      float hi0_104 = fmaxf(x0_101, y0_102);
      float cur_105 = hi0_104;
      float pv_106 = __shfl_xor_sync(0xFFFFFFFF, cur_105, 8);
      float hi_107_1 = fmaxf(cur_105, pv_106);
      float lo_108_1 = fminf(cur_105, pv_106);
      cur_105 = ((up[0] != 0) ? hi_107_1 : lo_108_1);
      float pv_109 = __shfl_xor_sync(0xFFFFFFFF, cur_105, 4);
      float hi_110_1 = fmaxf(cur_105, pv_109);
      float lo_111_1 = fminf(cur_105, pv_109);
      cur_105 = ((up[1] != 0) ? hi_110_1 : lo_111_1);
      float pv_112 = __shfl_xor_sync(0xFFFFFFFF, cur_105, 2);
      float hi_113_1 = fmaxf(cur_105, pv_112);
      float lo_114_1 = fminf(cur_105, pv_112);
      cur_105 = ((up[2] != 0) ? hi_113_1 : lo_114_1);
      float pv_115 = __shfl_xor_sync(0xFFFFFFFF, cur_105, 1);
      float hi_116_2 = fmaxf(cur_105, pv_115);
      float lo_117_2 = fminf(cur_105, pv_115);
      cur_105 = ((up[3] != 0) ? hi_116_2 : lo_117_2);
      V_2[5] = cur_105;
      int s0_118 = (sg * 16 + 6 + cg) * 17;
      int s1_119 = (sg * 16 + 6 + 8 + cg) * 17;
      float x0_120 = pub[s0_118 + ln];
      float y0_121 = pub[s1_119 + lnr];
      float lo0_122 = fminf(x0_120, y0_121);
      r_1_1 = fmaxf(r_1_1, lo0_122);
      float hi0_123 = fmaxf(x0_120, y0_121);
      float cur_124 = hi0_123;
      float pv_125 = __shfl_xor_sync(0xFFFFFFFF, cur_124, 8);
      float hi_126_2 = fmaxf(cur_124, pv_125);
      float lo_127_2 = fminf(cur_124, pv_125);
      cur_124 = ((up[0] != 0) ? hi_126_2 : lo_127_2);
      float pv_128 = __shfl_xor_sync(0xFFFFFFFF, cur_124, 4);
      float hi_129 = fmaxf(cur_124, pv_128);
      float lo_130 = fminf(cur_124, pv_128);
      cur_124 = ((up[1] != 0) ? hi_129 : lo_130);
      float pv_131 = __shfl_xor_sync(0xFFFFFFFF, cur_124, 2);
      float hi_132_2 = fmaxf(cur_124, pv_131);
      float lo_133_2 = fminf(cur_124, pv_131);
      cur_124 = ((up[2] != 0) ? hi_132_2 : lo_133_2);
      float pv_134 = __shfl_xor_sync(0xFFFFFFFF, cur_124, 1);
      float hi_135_1 = fmaxf(cur_124, pv_134);
      float lo_136_1 = fminf(cur_124, pv_134);
      cur_124 = ((up[3] != 0) ? hi_135_1 : lo_136_1);
      V_2[6] = cur_124;
      int s0_137 = (sg * 16 + 7 + cg) * 17;
      int s1_138 = (sg * 16 + 7 + 8 + cg) * 17;
      float x0_139 = pub[s0_137 + ln];
      float y0_140 = pub[s1_138 + lnr];
      float lo0_141 = fminf(x0_139, y0_140);
      r_1_1 = fmaxf(r_1_1, lo0_141);
      float hi0_142 = fmaxf(x0_139, y0_140);
      float cur_143 = hi0_142;
      float pv_144 = __shfl_xor_sync(0xFFFFFFFF, cur_143, 8);
      float hi_145_1 = fmaxf(cur_143, pv_144);
      float lo_146 = fminf(cur_143, pv_144);
      cur_143 = ((up[0] != 0) ? hi_145_1 : lo_146);
      float pv_147 = __shfl_xor_sync(0xFFFFFFFF, cur_143, 4);
      float hi_148_2 = fmaxf(cur_143, pv_147);
      float lo_149_1 = fminf(cur_143, pv_147);
      cur_143 = ((up[1] != 0) ? hi_148_2 : lo_149_1);
      float pv_150 = __shfl_xor_sync(0xFFFFFFFF, cur_143, 2);
      float hi_151_1 = fmaxf(cur_143, pv_150);
      float lo_152 = fminf(cur_143, pv_150);
      cur_143 = ((up[2] != 0) ? hi_151_1 : lo_152);
      float pv_153 = __shfl_xor_sync(0xFFFFFFFF, cur_143, 1);
      float hi_154_2 = fmaxf(cur_143, pv_153);
      float lo_155_1 = fminf(cur_143, pv_153);
      cur_143 = ((up[3] != 0) ? hi_154_2 : lo_155_1);
      V_2[7] = cur_143;
      float rs_156 = pub[(sg * 16 + ln + cg) * 17 + 16];
      r_1_1 = fmaxf(r_1_1, rs_156);
      float y1_157 = __shfl_xor_sync(0xFFFFFFFF, V_2[4], 15);
      float lo1_158 = fminf(V_2[0], y1_157);
      r_1_1 = fmaxf(r_1_1, lo1_158);
      float hi1_159 = fmaxf(V_2[0], y1_157);
      float cur_160 = hi1_159;
      float pv_161 = __shfl_xor_sync(0xFFFFFFFF, cur_160, 8);
      float hi_162_2 = fmaxf(cur_160, pv_161);
      float lo_163_1 = fminf(cur_160, pv_161);
      cur_160 = ((up[0] != 0) ? hi_162_2 : lo_163_1);
      float pv_164 = __shfl_xor_sync(0xFFFFFFFF, cur_160, 4);
      float hi_165_1 = fmaxf(cur_160, pv_164);
      float lo_166_1 = fminf(cur_160, pv_164);
      cur_160 = ((up[1] != 0) ? hi_165_1 : lo_166_1);
      float pv_167 = __shfl_xor_sync(0xFFFFFFFF, cur_160, 2);
      float hi_168_1 = fmaxf(cur_160, pv_167);
      float lo_169_1 = fminf(cur_160, pv_167);
      cur_160 = ((up[2] != 0) ? hi_168_1 : lo_169_1);
      float pv_170 = __shfl_xor_sync(0xFFFFFFFF, cur_160, 1);
      float hi_171 = fmaxf(cur_160, pv_170);
      float lo_172 = fminf(cur_160, pv_170);
      cur_160 = ((up[3] != 0) ? hi_171 : lo_172);
      V_2[0] = cur_160;
      float y1_173 = __shfl_xor_sync(0xFFFFFFFF, V_2[5], 15);
      float lo1_174 = fminf(V_2[1], y1_173);
      r_1_1 = fmaxf(r_1_1, lo1_174);
      float hi1_175 = fmaxf(V_2[1], y1_173);
      float cur_176 = hi1_175;
      float pv_177 = __shfl_xor_sync(0xFFFFFFFF, cur_176, 8);
      float hi_178_1 = fmaxf(cur_176, pv_177);
      float lo_179_1 = fminf(cur_176, pv_177);
      cur_176 = ((up[0] != 0) ? hi_178_1 : lo_179_1);
      float pv_180 = __shfl_xor_sync(0xFFFFFFFF, cur_176, 4);
      float hi_181 = fmaxf(cur_176, pv_180);
      float lo_182 = fminf(cur_176, pv_180);
      cur_176 = ((up[1] != 0) ? hi_181 : lo_182);
      float pv_183 = __shfl_xor_sync(0xFFFFFFFF, cur_176, 2);
      float hi_184_1 = fmaxf(cur_176, pv_183);
      float lo_185_1 = fminf(cur_176, pv_183);
      cur_176 = ((up[2] != 0) ? hi_184_1 : lo_185_1);
      float pv_186 = __shfl_xor_sync(0xFFFFFFFF, cur_176, 1);
      float hi_187 = fmaxf(cur_176, pv_186);
      float lo_188 = fminf(cur_176, pv_186);
      cur_176 = ((up[3] != 0) ? hi_187 : lo_188);
      V_2[1] = cur_176;
      float y1_189 = __shfl_xor_sync(0xFFFFFFFF, V_2[6], 15);
      float lo1_190 = fminf(V_2[2], y1_189);
      r_1_1 = fmaxf(r_1_1, lo1_190);
      float hi1_191 = fmaxf(V_2[2], y1_189);
      float cur_192 = hi1_191;
      float pv_193 = __shfl_xor_sync(0xFFFFFFFF, cur_192, 8);
      float hi_194_1 = fmaxf(cur_192, pv_193);
      float lo_195_1 = fminf(cur_192, pv_193);
      cur_192 = ((up[0] != 0) ? hi_194_1 : lo_195_1);
      float pv_196 = __shfl_xor_sync(0xFFFFFFFF, cur_192, 4);
      float hi_197 = fmaxf(cur_192, pv_196);
      float lo_198 = fminf(cur_192, pv_196);
      cur_192 = ((up[1] != 0) ? hi_197 : lo_198);
      float pv_199 = __shfl_xor_sync(0xFFFFFFFF, cur_192, 2);
      float hi_200_1 = fmaxf(cur_192, pv_199);
      float lo_201_1 = fminf(cur_192, pv_199);
      cur_192 = ((up[2] != 0) ? hi_200_1 : lo_201_1);
      float pv_202 = __shfl_xor_sync(0xFFFFFFFF, cur_192, 1);
      float hi_203 = fmaxf(cur_192, pv_202);
      float lo_204 = fminf(cur_192, pv_202);
      cur_192 = ((up[3] != 0) ? hi_203 : lo_204);
      V_2[2] = cur_192;
      float y1_205 = __shfl_xor_sync(0xFFFFFFFF, V_2[7], 15);
      float lo1_206 = fminf(V_2[3], y1_205);
      r_1_1 = fmaxf(r_1_1, lo1_206);
      float hi1_207 = fmaxf(V_2[3], y1_205);
      float cur_208 = hi1_207;
      float pv_209 = __shfl_xor_sync(0xFFFFFFFF, cur_208, 8);
      float hi_210_1 = fmaxf(cur_208, pv_209);
      float lo_211_1 = fminf(cur_208, pv_209);
      cur_208 = ((up[0] != 0) ? hi_210_1 : lo_211_1);
      float pv_212 = __shfl_xor_sync(0xFFFFFFFF, cur_208, 4);
      float hi_213 = fmaxf(cur_208, pv_212);
      float lo_214 = fminf(cur_208, pv_212);
      cur_208 = ((up[1] != 0) ? hi_213 : lo_214);
      float pv_215 = __shfl_xor_sync(0xFFFFFFFF, cur_208, 2);
      float hi_216_1 = fmaxf(cur_208, pv_215);
      float lo_217_1 = fminf(cur_208, pv_215);
      cur_208 = ((up[2] != 0) ? hi_216_1 : lo_217_1);
      float pv_218 = __shfl_xor_sync(0xFFFFFFFF, cur_208, 1);
      float hi_219 = fmaxf(cur_208, pv_218);
      float lo_220 = fminf(cur_208, pv_218);
      cur_208 = ((up[3] != 0) ? hi_219 : lo_220);
      V_2[3] = cur_208;
      float y1_221 = __shfl_xor_sync(0xFFFFFFFF, V_2[2], 15);
      float lo1_222 = fminf(V_2[0], y1_221);
      r_1_1 = fmaxf(r_1_1, lo1_222);
      float hi1_223 = fmaxf(V_2[0], y1_221);
      float cur_224 = hi1_223;
      float pv_225 = __shfl_xor_sync(0xFFFFFFFF, cur_224, 8);
      float hi_226_1 = fmaxf(cur_224, pv_225);
      float lo_227_1 = fminf(cur_224, pv_225);
      cur_224 = ((up[0] != 0) ? hi_226_1 : lo_227_1);
      float pv_228 = __shfl_xor_sync(0xFFFFFFFF, cur_224, 4);
      float hi_229 = fmaxf(cur_224, pv_228);
      float lo_230 = fminf(cur_224, pv_228);
      cur_224 = ((up[1] != 0) ? hi_229 : lo_230);
      float pv_231 = __shfl_xor_sync(0xFFFFFFFF, cur_224, 2);
      float hi_232_1 = fmaxf(cur_224, pv_231);
      float lo_233_1 = fminf(cur_224, pv_231);
      cur_224 = ((up[2] != 0) ? hi_232_1 : lo_233_1);
      float pv_234 = __shfl_xor_sync(0xFFFFFFFF, cur_224, 1);
      float hi_235 = fmaxf(cur_224, pv_234);
      float lo_236 = fminf(cur_224, pv_234);
      cur_224 = ((up[3] != 0) ? hi_235 : lo_236);
      V_2[0] = cur_224;
      float y1_237 = __shfl_xor_sync(0xFFFFFFFF, V_2[3], 15);
      float lo1_238 = fminf(V_2[1], y1_237);
      r_1_1 = fmaxf(r_1_1, lo1_238);
      float hi1_239 = fmaxf(V_2[1], y1_237);
      float cur_240 = hi1_239;
      float pv_241 = __shfl_xor_sync(0xFFFFFFFF, cur_240, 8);
      float hi_242_1 = fmaxf(cur_240, pv_241);
      float lo_243_1 = fminf(cur_240, pv_241);
      cur_240 = ((up[0] != 0) ? hi_242_1 : lo_243_1);
      float pv_244 = __shfl_xor_sync(0xFFFFFFFF, cur_240, 4);
      float hi_245 = fmaxf(cur_240, pv_244);
      float lo_246 = fminf(cur_240, pv_244);
      cur_240 = ((up[1] != 0) ? hi_245 : lo_246);
      float pv_247 = __shfl_xor_sync(0xFFFFFFFF, cur_240, 2);
      float hi_248_1 = fmaxf(cur_240, pv_247);
      float lo_249_1 = fminf(cur_240, pv_247);
      cur_240 = ((up[2] != 0) ? hi_248_1 : lo_249_1);
      float pv_250 = __shfl_xor_sync(0xFFFFFFFF, cur_240, 1);
      float hi_251 = fmaxf(cur_240, pv_250);
      float lo_252 = fminf(cur_240, pv_250);
      cur_240 = ((up[3] != 0) ? hi_251 : lo_252);
      V_2[1] = cur_240;
      float yl_253 = __shfl_xor_sync(0xFFFFFFFF, V_2[1], 15);
      float lol_254 = fminf(V_2[0], yl_253);
      r_1_1 = fmaxf(r_1_1, lol_254);
      float hil_255 = fmaxf(V_2[0], yl_253);
      float cur_256 = hil_255;
      float pv_257 = __shfl_xor_sync(0xFFFFFFFF, cur_256, 8);
      float hi_258_1 = fmaxf(cur_256, pv_257);
      float lo_259_1 = fminf(cur_256, pv_257);
      cur_256 = ((up[0] != 0) ? hi_258_1 : lo_259_1);
      float pv_260 = __shfl_xor_sync(0xFFFFFFFF, cur_256, 4);
      float hi_261 = fmaxf(cur_256, pv_260);
      float lo_262 = fminf(cur_256, pv_260);
      cur_256 = ((up[1] != 0) ? hi_261 : lo_262);
      float pv_263 = __shfl_xor_sync(0xFFFFFFFF, cur_256, 2);
      float hi_264 = fmaxf(cur_256, pv_263);
      float lo_265 = fminf(cur_256, pv_263);
      cur_256 = ((up[2] != 0) ? hi_264 : lo_265);
      float pv_266 = __shfl_xor_sync(0xFFFFFFFF, cur_256, 1);
      float hi_267 = fmaxf(cur_256, pv_266);
      float lo_268 = fminf(cur_256, pv_266);
      cur_256 = ((up[3] != 0) ? hi_267 : lo_268);
      V_2[0] = cur_256;
      float K_269 = V_2[0];
      int qb_270 = (sg + cg) * 32;
      q2[qb_270 + ln] = K_269;
      q2[qb_270 + 16 + ln] = r_1_1;
      asm volatile("barrier.sync 1, 32;" ::: "memory");
      float k2o_271[1];
      float r2v_272 = neg_inf;
      r2v_272 = fmaxf(r2v_272, q2[cg * 32 + 16 + ln]);
      r2v_272 = fmaxf(r2v_272, q2[(1 + cg) * 32 + 16 + ln]);
      float V2_273[1];
      float x2_274 = q2[cg * 32 + ln];
      float y2_275 = q2[(1 + cg) * 32 + lnr];
      float lo2_276 = fminf(x2_274, y2_275);
      r2v_272 = fmaxf(r2v_272, lo2_276);
      float hi2_277 = fmaxf(x2_274, y2_275);
      V2_273[0] = hi2_277;
      k2o_271[0] = V2_273[0];
      K_269 = k2o_271[0];
      r_1_1 = r2v_272;
      rr2x[0] = r_1_1;
      K2 = K_269;
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
          float hi_0_1 = fmaxf(xs[0], xs[13]);
          float lo_1_2 = fminf(xs[0], xs[13]);
          xs[0] = hi_0_1;
          xs[13] = lo_1_2;
          float hi_2_1 = fmaxf(xs[1], xs[12]);
          float lo_3_1 = fminf(xs[1], xs[12]);
          xs[1] = hi_2_1;
          xs[12] = lo_3_1;
          float hi_5_1 = fmaxf(xs[2], xs[15]);
          float lo_6_1 = fminf(xs[2], xs[15]);
          xs[2] = hi_5_1;
          xs[15] = lo_6_1;
          float hi_8_1 = fmaxf(xs[3], xs[14]);
          float lo_9_1 = fminf(xs[3], xs[14]);
          xs[3] = hi_8_1;
          xs[14] = lo_9_1;
          float hi_10_1 = fmaxf(xs[4], xs[8]);
          float lo_11_1 = fminf(xs[4], xs[8]);
          xs[4] = hi_10_1;
          xs[8] = lo_11_1;
          float hi_12_1 = fmaxf(xs[5], xs[6]);
          float lo_13_1 = fminf(xs[5], xs[6]);
          xs[5] = hi_12_1;
          xs[6] = lo_13_1;
          float hi_14_2 = fmaxf(xs[7], xs[11]);
          float lo_15_2 = fminf(xs[7], xs[11]);
          xs[7] = hi_14_2;
          xs[11] = lo_15_2;
          float hi_16_1 = fmaxf(xs[9], xs[10]);
          float lo_17_1 = fminf(xs[9], xs[10]);
          xs[9] = hi_16_1;
          xs[10] = lo_17_1;
          float hi_18_2 = fmaxf(xs[0], xs[5]);
          float lo_19_2 = fminf(xs[0], xs[5]);
          xs[0] = hi_18_2;
          xs[5] = lo_19_2;
          float hi_21_2 = fmaxf(xs[1], xs[7]);
          float lo_22_2 = fminf(xs[1], xs[7]);
          xs[1] = hi_21_2;
          xs[7] = lo_22_2;
          float hi_24_1 = fmaxf(xs[2], xs[9]);
          float lo_25_1 = fminf(xs[2], xs[9]);
          xs[2] = hi_24_1;
          xs[9] = lo_25_1;
          float hi_27_1 = fmaxf(xs[3], xs[4]);
          float lo_28_1 = fminf(xs[3], xs[4]);
          xs[3] = hi_27_1;
          xs[4] = lo_28_1;
          float hi_29_1 = fmaxf(xs[6], xs[13]);
          float lo_30_1 = fminf(xs[6], xs[13]);
          xs[6] = hi_29_1;
          xs[13] = lo_30_1;
          float hi_31_2 = fmaxf(xs[8], xs[14]);
          float lo_32_2 = fminf(xs[8], xs[14]);
          xs[8] = hi_31_2;
          xs[14] = lo_32_2;
          float hi_33_1 = fmaxf(xs[10], xs[15]);
          float lo_34_1 = fminf(xs[10], xs[15]);
          xs[10] = hi_33_1;
          xs[15] = lo_34_1;
          float hi_35_1 = fmaxf(xs[11], xs[12]);
          float lo_36_1 = fminf(xs[11], xs[12]);
          xs[11] = hi_35_1;
          xs[12] = lo_36_1;
          float hi_37_2 = fmaxf(xs[0], xs[1]);
          float lo_38_2 = fminf(xs[0], xs[1]);
          xs[0] = hi_37_2;
          xs[1] = lo_38_2;
          float hi_40_2 = fmaxf(xs[2], xs[3]);
          float lo_41_2 = fminf(xs[2], xs[3]);
          xs[2] = hi_40_2;
          xs[3] = lo_41_2;
          float hi_43_1 = fmaxf(xs[4], xs[5]);
          float lo_44_1 = fminf(xs[4], xs[5]);
          xs[4] = hi_43_1;
          xs[5] = lo_44_1;
          float hi_46_1 = fmaxf(xs[6], xs[8]);
          float lo_47_1 = fminf(xs[6], xs[8]);
          xs[6] = hi_46_1;
          xs[8] = lo_47_1;
          float hi_48_2 = fmaxf(xs[7], xs[9]);
          float lo_49_2 = fminf(xs[7], xs[9]);
          xs[7] = hi_48_2;
          xs[9] = lo_49_2;
          float hi_50_3 = fmaxf(xs[10], xs[11]);
          float lo_51_3 = fminf(xs[10], xs[11]);
          xs[10] = hi_50_3;
          xs[11] = lo_51_3;
          float hi_52_2 = fmaxf(xs[12], xs[13]);
          float lo_53_2 = fminf(xs[12], xs[13]);
          xs[12] = hi_52_2;
          xs[13] = lo_53_2;
          float hi_54_2 = fmaxf(xs[14], xs[15]);
          float lo_55_2 = fminf(xs[14], xs[15]);
          xs[14] = hi_54_2;
          xs[15] = lo_55_2;
          float hi_56_3 = fmaxf(xs[0], xs[2]);
          float lo_57_3 = fminf(xs[0], xs[2]);
          xs[0] = hi_56_3;
          xs[2] = lo_57_3;
          float hi_59_2 = fmaxf(xs[1], xs[3]);
          float lo_60_2 = fminf(xs[1], xs[3]);
          xs[1] = hi_59_2;
          xs[3] = lo_60_2;
          float hi_62_2 = fmaxf(xs[4], xs[10]);
          float lo_63_2 = fminf(xs[4], xs[10]);
          xs[4] = hi_62_2;
          xs[10] = lo_63_2;
          float hi_65_1 = fmaxf(xs[5], xs[11]);
          float lo_66_1 = fminf(xs[5], xs[11]);
          xs[5] = hi_65_1;
          xs[11] = lo_66_1;
          float hi_67_1 = fmaxf(xs[6], xs[7]);
          float lo_68_1 = fminf(xs[6], xs[7]);
          xs[6] = hi_67_1;
          xs[7] = lo_68_1;
          float hi_69_2 = fmaxf(xs[8], xs[9]);
          float lo_70_2 = fminf(xs[8], xs[9]);
          xs[8] = hi_69_2;
          xs[9] = lo_70_2;
          float hi_71_1 = fmaxf(xs[12], xs[14]);
          float lo_72_1 = fminf(xs[12], xs[14]);
          xs[12] = hi_71_1;
          xs[14] = lo_72_1;
          float hi_73_1 = fmaxf(xs[13], xs[15]);
          float lo_74_1 = fminf(xs[13], xs[15]);
          xs[13] = hi_73_1;
          xs[15] = lo_74_1;
          float hi_75_2 = fmaxf(xs[1], xs[2]);
          float lo_76_2 = fminf(xs[1], xs[2]);
          xs[1] = hi_75_2;
          xs[2] = lo_76_2;
          float hi_78_3 = fmaxf(xs[3], xs[12]);
          float lo_79_3 = fminf(xs[3], xs[12]);
          xs[3] = hi_78_3;
          xs[12] = lo_79_3;
          float hi_81_1 = fmaxf(xs[4], xs[6]);
          float lo_82_1 = fminf(xs[4], xs[6]);
          xs[4] = hi_81_1;
          xs[6] = lo_82_1;
          float hi_84_2 = fmaxf(xs[5], xs[7]);
          float lo_85_2 = fminf(xs[5], xs[7]);
          xs[5] = hi_84_2;
          xs[7] = lo_85_2;
          float hi_86_2 = fmaxf(xs[8], xs[10]);
          float lo_87_2 = fminf(xs[8], xs[10]);
          xs[8] = hi_86_2;
          xs[10] = lo_87_2;
          float hi_88_3 = fmaxf(xs[9], xs[11]);
          float lo_89_3 = fminf(xs[9], xs[11]);
          xs[9] = hi_88_3;
          xs[11] = lo_89_3;
          float hi_90_2 = fmaxf(xs[13], xs[14]);
          float lo_91_2 = fminf(xs[13], xs[14]);
          xs[13] = hi_90_2;
          xs[14] = lo_91_2;
          float hi_92_2 = fmaxf(xs[1], xs[4]);
          float lo_93_2 = fminf(xs[1], xs[4]);
          xs[1] = hi_92_2;
          xs[4] = lo_93_2;
          float hi_94_3 = fmaxf(xs[2], xs[6]);
          float lo_95_3 = fminf(xs[2], xs[6]);
          xs[2] = hi_94_3;
          xs[6] = lo_95_3;
          float hi_97_2 = fmaxf(xs[5], xs[8]);
          float lo_98_2 = fminf(xs[5], xs[8]);
          xs[5] = hi_97_2;
          xs[8] = lo_98_2;
          float hi_100_2 = fmaxf(xs[7], xs[10]);
          float lo_101_2 = fminf(xs[7], xs[10]);
          xs[7] = hi_100_2;
          xs[10] = lo_101_2;
          float hi_103_1 = fmaxf(xs[9], xs[13]);
          float lo_104_1 = fminf(xs[9], xs[13]);
          xs[9] = hi_103_1;
          xs[13] = lo_104_1;
          float hi_105_1 = fmaxf(xs[11], xs[14]);
          float lo_106_1 = fminf(xs[11], xs[14]);
          xs[11] = hi_105_1;
          xs[14] = lo_106_1;
          float hi_107_2 = fmaxf(xs[2], xs[4]);
          float lo_108_2 = fminf(xs[2], xs[4]);
          xs[2] = hi_107_2;
          xs[4] = lo_108_2;
          float hi_109_1 = fmaxf(xs[3], xs[6]);
          float lo_110_1 = fminf(xs[3], xs[6]);
          xs[3] = hi_109_1;
          xs[6] = lo_110_1;
          float hi_111_1 = fmaxf(xs[9], xs[12]);
          float lo_112_1 = fminf(xs[9], xs[12]);
          xs[9] = hi_111_1;
          xs[12] = lo_112_1;
          float hi_113_2 = fmaxf(xs[11], xs[13]);
          float lo_114_2 = fminf(xs[11], xs[13]);
          xs[11] = hi_113_2;
          xs[13] = lo_114_2;
          float hi_116_3 = fmaxf(xs[3], xs[5]);
          float lo_117_3 = fminf(xs[3], xs[5]);
          xs[3] = hi_116_3;
          xs[5] = lo_117_3;
          float hi_119_1 = fmaxf(xs[6], xs[8]);
          float lo_120_1 = fminf(xs[6], xs[8]);
          xs[6] = hi_119_1;
          xs[8] = lo_120_1;
          float hi_122_2 = fmaxf(xs[7], xs[9]);
          float lo_123_2 = fminf(xs[7], xs[9]);
          xs[7] = hi_122_2;
          xs[9] = lo_123_2;
          float hi_124_2 = fmaxf(xs[10], xs[12]);
          float lo_125_2 = fminf(xs[10], xs[12]);
          xs[10] = hi_124_2;
          xs[12] = lo_125_2;
          float hi_126_3 = fmaxf(xs[3], xs[4]);
          float lo_127_3 = fminf(xs[3], xs[4]);
          xs[3] = hi_126_3;
          xs[4] = lo_127_3;
          float hi_128_2 = fmaxf(xs[5], xs[6]);
          float lo_129_2 = fminf(xs[5], xs[6]);
          xs[5] = hi_128_2;
          xs[6] = lo_129_2;
          float hi_130_2 = fmaxf(xs[7], xs[8]);
          float lo_131_2 = fminf(xs[7], xs[8]);
          xs[7] = hi_130_2;
          xs[8] = lo_131_2;
          float hi_132_3 = fmaxf(xs[9], xs[10]);
          float lo_133_3 = fminf(xs[9], xs[10]);
          xs[9] = hi_132_3;
          xs[10] = lo_133_3;
          float hi_135_2 = fmaxf(xs[11], xs[12]);
          float lo_136_2 = fminf(xs[11], xs[12]);
          xs[11] = hi_135_2;
          xs[12] = lo_136_2;
          float hi_138_2 = fmaxf(xs[6], xs[7]);
          float lo_139_2 = fminf(xs[6], xs[7]);
          xs[6] = hi_138_2;
          xs[7] = lo_139_2;
          float hi_141_1 = fmaxf(xs[8], xs[9]);
          float lo_142_1 = fminf(xs[8], xs[9]);
          xs[8] = hi_141_1;
          xs[9] = lo_142_1;
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
