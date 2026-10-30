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
#define SMEM_TREE_OFF 0
#define SMEM_TREE_STAGE_BYTES 8704
#define SMEM_TREE_STRIDE 8704
#define SMEM_QCOL_OFF 8704
#define SMEM_QCOL_STAGE_BYTES 128
#define SMEM_QCOL_STRIDE 128
#define SMEM_FLAGW_OFF 8832
#define SMEM_FLAGW_STAGE_BYTES 16
#define SMEM_FLAGW_STRIDE 16
#define SMEM_CCNT_OFF 8848
#define SMEM_CCNT_STAGE_BYTES 128
#define SMEM_CCNT_STRIDE 128
#define SMEM_LIMC_OFF 8976
#define SMEM_LIMC_STAGE_BYTES 128
#define SMEM_LIMC_STRIDE 128
#define SMEM_CBUF_OFF 9104
#define SMEM_CBUF_STAGE_BYTES 2048
#define SMEM_CBUF_STRIDE 2048
#define SMEM_TOTAL 11264
#define THREADS 128

extern "C" {

__global__ __launch_bounds__(128) void kernel_cake_hopper_msa_74205ef0b2ce7da8e6d8(
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

  // === Task calls (dependency order) ===
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
  sc = fmaxf(sc, -1.7014118346046923e+38f);
  sc = fminf(sc, 1.7014118346046923e+38f);
  unsigned int key = __as_u32(sc) & 4294967232u | (unsigned int)t0_1;
  kb[0] = __uint_as_float(key);
  float sc_2 = __uint_as_float(cb[1]);
  sc_2 = fmaxf(sc_2, -1.7014118346046923e+38f);
  sc_2 = fminf(sc_2, 1.7014118346046923e+38f);
  unsigned int key_3 = __as_u32(sc_2) & 4294967232u | (unsigned int)(t0_1 + 1);
  kb[1] = __uint_as_float(key_3);
  float sc_4 = __uint_as_float(cb[2]);
  sc_4 = fmaxf(sc_4, -1.7014118346046923e+38f);
  sc_4 = fminf(sc_4, 1.7014118346046923e+38f);
  unsigned int key_5 = __as_u32(sc_4) & 4294967232u | (unsigned int)(t0_1 + 2);
  kb[2] = __uint_as_float(key_5);
  float sc_6 = __uint_as_float(cb[3]);
  sc_6 = fmaxf(sc_6, -1.7014118346046923e+38f);
  sc_6 = fminf(sc_6, 1.7014118346046923e+38f);
  unsigned int key_7 = __as_u32(sc_6) & 4294967232u | (unsigned int)(t0_1 + 3);
  kb[3] = __uint_as_float(key_7);
  float sc_8 = __uint_as_float(cb[4]);
  sc_8 = fmaxf(sc_8, -1.7014118346046923e+38f);
  sc_8 = fminf(sc_8, 1.7014118346046923e+38f);
  unsigned int key_9 = __as_u32(sc_8) & 4294967232u | (unsigned int)(t0_1 + 4);
  kb[4] = __uint_as_float(key_9);
  float sc_10 = __uint_as_float(cb[5]);
  sc_10 = fmaxf(sc_10, -1.7014118346046923e+38f);
  sc_10 = fminf(sc_10, 1.7014118346046923e+38f);
  unsigned int key_11 = __as_u32(sc_10) & 4294967232u | (unsigned int)(t0_1 + 5);
  kb[5] = __uint_as_float(key_11);
  float sc_12 = __uint_as_float(cb[6]);
  sc_12 = fmaxf(sc_12, -1.7014118346046923e+38f);
  sc_12 = fminf(sc_12, 1.7014118346046923e+38f);
  unsigned int key_13 = __as_u32(sc_12) & 4294967232u | (unsigned int)(t0_1 + 6);
  kb[6] = __uint_as_float(key_13);
  float sc_14 = __uint_as_float(cb[7]);
  sc_14 = fmaxf(sc_14, -1.7014118346046923e+38f);
  sc_14 = fminf(sc_14, 1.7014118346046923e+38f);
  unsigned int key_15 = __as_u32(sc_14) & 4294967232u | (unsigned int)(t0_1 + 7);
  kb[7] = __uint_as_float(key_15);
  float sc_16 = __uint_as_float(cb[8]);
  sc_16 = fmaxf(sc_16, -1.7014118346046923e+38f);
  sc_16 = fminf(sc_16, 1.7014118346046923e+38f);
  unsigned int key_17 = __as_u32(sc_16) & 4294967232u | (unsigned int)(t0_1 + 8);
  kb[8] = __uint_as_float(key_17);
  float sc_18 = __uint_as_float(cb[9]);
  sc_18 = fmaxf(sc_18, -1.7014118346046923e+38f);
  sc_18 = fminf(sc_18, 1.7014118346046923e+38f);
  unsigned int key_19 = __as_u32(sc_18) & 4294967232u | (unsigned int)(t0_1 + 9);
  kb[9] = __uint_as_float(key_19);
  float sc_20 = __uint_as_float(cb[10]);
  sc_20 = fmaxf(sc_20, -1.7014118346046923e+38f);
  sc_20 = fminf(sc_20, 1.7014118346046923e+38f);
  unsigned int key_21 = __as_u32(sc_20) & 4294967232u | (unsigned int)(t0_1 + 10);
  kb[10] = __uint_as_float(key_21);
  float sc_22 = __uint_as_float(cb[11]);
  sc_22 = fmaxf(sc_22, -1.7014118346046923e+38f);
  sc_22 = fminf(sc_22, 1.7014118346046923e+38f);
  unsigned int key_23 = __as_u32(sc_22) & 4294967232u | (unsigned int)(t0_1 + 11);
  kb[11] = __uint_as_float(key_23);
  float sc_24 = __uint_as_float(cb[12]);
  sc_24 = fmaxf(sc_24, -1.7014118346046923e+38f);
  sc_24 = fminf(sc_24, 1.7014118346046923e+38f);
  unsigned int key_25 = __as_u32(sc_24) & 4294967232u | (unsigned int)(t0_1 + 12);
  kb[12] = __uint_as_float(key_25);
  float sc_26 = __uint_as_float(cb[13]);
  sc_26 = fmaxf(sc_26, -1.7014118346046923e+38f);
  sc_26 = fminf(sc_26, 1.7014118346046923e+38f);
  unsigned int key_27 = __as_u32(sc_26) & 4294967232u | (unsigned int)(t0_1 + 13);
  kb[13] = __uint_as_float(key_27);
  float sc_28 = __uint_as_float(cb[14]);
  sc_28 = fmaxf(sc_28, -1.7014118346046923e+38f);
  sc_28 = fminf(sc_28, 1.7014118346046923e+38f);
  unsigned int key_29 = __as_u32(sc_28) & 4294967232u | (unsigned int)(t0_1 + 14);
  kb[14] = __uint_as_float(key_29);
  float sc_30 = __uint_as_float(cb[15]);
  sc_30 = fmaxf(sc_30, -1.7014118346046923e+38f);
  sc_30 = fminf(sc_30, 1.7014118346046923e+38f);
  unsigned int key_31 = __as_u32(sc_30) & 4294967232u | (unsigned int)(t0_1 + 15);
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
      kb[0] = __uint_as_float(2139094976 | (unsigned int)t0_1);
    }
    int f_1 = 0;
    if ((t0_1 + 1 < fb || t0_1 + 1 >= lim - fe) && lim > t0_1 + 1) {
      f_1 = 1;
    }
    if (f_1 != 0) {
      kb[1] = __uint_as_float(2139094976 | (unsigned int)(t0_1 + 1));
    }
    int f_2 = 0;
    if ((t0_1 + 2 < fb || t0_1 + 2 >= lim - fe) && lim > t0_1 + 2) {
      f_2 = 1;
    }
    if (f_2 != 0) {
      kb[2] = __uint_as_float(2139094976 | (unsigned int)(t0_1 + 2));
    }
    int f_3 = 0;
    if ((t0_1 + 3 < fb || t0_1 + 3 >= lim - fe) && lim > t0_1 + 3) {
      f_3 = 1;
    }
    if (f_3 != 0) {
      kb[3] = __uint_as_float(2139094976 | (unsigned int)(t0_1 + 3));
    }
    int f_4 = 0;
    if ((t0_1 + 4 < fb || t0_1 + 4 >= lim - fe) && lim > t0_1 + 4) {
      f_4 = 1;
    }
    if (f_4 != 0) {
      kb[4] = __uint_as_float(2139094976 | (unsigned int)(t0_1 + 4));
    }
    int f_5 = 0;
    if ((t0_1 + 5 < fb || t0_1 + 5 >= lim - fe) && lim > t0_1 + 5) {
      f_5 = 1;
    }
    if (f_5 != 0) {
      kb[5] = __uint_as_float(2139094976 | (unsigned int)(t0_1 + 5));
    }
    int f_6 = 0;
    if ((t0_1 + 6 < fb || t0_1 + 6 >= lim - fe) && lim > t0_1 + 6) {
      f_6 = 1;
    }
    if (f_6 != 0) {
      kb[6] = __uint_as_float(2139094976 | (unsigned int)(t0_1 + 6));
    }
    int f_7 = 0;
    if ((t0_1 + 7 < fb || t0_1 + 7 >= lim - fe) && lim > t0_1 + 7) {
      f_7 = 1;
    }
    if (f_7 != 0) {
      kb[7] = __uint_as_float(2139094976 | (unsigned int)(t0_1 + 7));
    }
    int f_8 = 0;
    if ((t0_1 + 8 < fb || t0_1 + 8 >= lim - fe) && lim > t0_1 + 8) {
      f_8 = 1;
    }
    if (f_8 != 0) {
      kb[8] = __uint_as_float(2139094976 | (unsigned int)(t0_1 + 8));
    }
    int f_9 = 0;
    if ((t0_1 + 9 < fb || t0_1 + 9 >= lim - fe) && lim > t0_1 + 9) {
      f_9 = 1;
    }
    if (f_9 != 0) {
      kb[9] = __uint_as_float(2139094976 | (unsigned int)(t0_1 + 9));
    }
    int f_10 = 0;
    if ((t0_1 + 10 < fb || t0_1 + 10 >= lim - fe) && lim > t0_1 + 10) {
      f_10 = 1;
    }
    if (f_10 != 0) {
      kb[10] = __uint_as_float(2139094976 | (unsigned int)(t0_1 + 10));
    }
    int f_11 = 0;
    if ((t0_1 + 11 < fb || t0_1 + 11 >= lim - fe) && lim > t0_1 + 11) {
      f_11 = 1;
    }
    if (f_11 != 0) {
      kb[11] = __uint_as_float(2139094976 | (unsigned int)(t0_1 + 11));
    }
    int f_12 = 0;
    if ((t0_1 + 12 < fb || t0_1 + 12 >= lim - fe) && lim > t0_1 + 12) {
      f_12 = 1;
    }
    if (f_12 != 0) {
      kb[12] = __uint_as_float(2139094976 | (unsigned int)(t0_1 + 12));
    }
    int f_13 = 0;
    if ((t0_1 + 13 < fb || t0_1 + 13 >= lim - fe) && lim > t0_1 + 13) {
      f_13 = 1;
    }
    if (f_13 != 0) {
      kb[13] = __uint_as_float(2139094976 | (unsigned int)(t0_1 + 13));
    }
    int f_14 = 0;
    if ((t0_1 + 14 < fb || t0_1 + 14 >= lim - fe) && lim > t0_1 + 14) {
      f_14 = 1;
    }
    if (f_14 != 0) {
      kb[14] = __uint_as_float(2139094976 | (unsigned int)(t0_1 + 14));
    }
    int f_15 = 0;
    if ((t0_1 + 15 < fb || t0_1 + 15 >= lim - fe) && lim > t0_1 + 15) {
      f_15 = 1;
    }
    if (f_15 != 0) {
      kb[15] = __uint_as_float(2139094976 | (unsigned int)(t0_1 + 15));
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
      float lo_1 = fminf(a[0], sb[15]);
      a[0] = hi_0;
      r = fmaxf(r, lo_1);
      float hi_2 = fmaxf(a[1], sb[14]);
      float lo_3 = fminf(a[1], sb[14]);
      a[1] = hi_2;
      r = fmaxf(r, lo_3);
      float hi_4 = fmaxf(a[2], sb[13]);
      float lo_5 = fminf(a[2], sb[13]);
      a[2] = hi_4;
      r = fmaxf(r, lo_5);
      float hi_6 = fmaxf(a[3], sb[12]);
      float lo_7 = fminf(a[3], sb[12]);
      a[3] = hi_6;
      r = fmaxf(r, lo_7);
      float hi_8 = fmaxf(a[4], sb[11]);
      float lo_9 = fminf(a[4], sb[11]);
      a[4] = hi_8;
      r = fmaxf(r, lo_9);
      float hi_10 = fmaxf(a[5], sb[10]);
      float lo_11 = fminf(a[5], sb[10]);
      a[5] = hi_10;
      r = fmaxf(r, lo_11);
      float hi_12 = fmaxf(a[6], sb[9]);
      float lo_13 = fminf(a[6], sb[9]);
      a[6] = hi_12;
      r = fmaxf(r, lo_13);
      float hi_14 = fmaxf(a[7], sb[8]);
      float lo_15 = fminf(a[7], sb[8]);
      a[7] = hi_14;
      r = fmaxf(r, lo_15);
      float hi_16 = fmaxf(a[8], sb[7]);
      float lo_17 = fminf(a[8], sb[7]);
      a[8] = hi_16;
      r = fmaxf(r, lo_17);
      float hi_18 = fmaxf(a[9], sb[6]);
      float lo_19 = fminf(a[9], sb[6]);
      a[9] = hi_18;
      r = fmaxf(r, lo_19);
      float hi_20 = fmaxf(a[10], sb[5]);
      float lo_21 = fminf(a[10], sb[5]);
      a[10] = hi_20;
      r = fmaxf(r, lo_21);
      float hi_22 = fmaxf(a[11], sb[4]);
      float lo_23 = fminf(a[11], sb[4]);
      a[11] = hi_22;
      r = fmaxf(r, lo_23);
      float hi_24 = fmaxf(a[12], sb[3]);
      float lo_25 = fminf(a[12], sb[3]);
      a[12] = hi_24;
      r = fmaxf(r, lo_25);
      float hi_26 = fmaxf(a[13], sb[2]);
      float lo_27 = fminf(a[13], sb[2]);
      a[13] = hi_26;
      r = fmaxf(r, lo_27);
      float hi_28 = fmaxf(a[14], sb[1]);
      float lo_29 = fminf(a[14], sb[1]);
      a[14] = hi_28;
      r = fmaxf(r, lo_29);
      float hi_30 = fmaxf(a[15], sb[0]);
      float lo_31 = fminf(a[15], sb[0]);
      a[15] = hi_30;
      r = fmaxf(r, lo_31);
      float hi_33 = fmaxf(a[0], a[8]);
      float lo_34 = fminf(a[0], a[8]);
      a[0] = hi_33;
      a[8] = lo_34;
      float hi_35 = fmaxf(a[1], a[9]);
      float lo_36 = fminf(a[1], a[9]);
      a[1] = hi_35;
      a[9] = lo_36;
      float hi_37 = fmaxf(a[2], a[10]);
      float lo_38 = fminf(a[2], a[10]);
      a[2] = hi_37;
      a[10] = lo_38;
      float hi_39 = fmaxf(a[3], a[11]);
      float lo_40 = fminf(a[3], a[11]);
      a[3] = hi_39;
      a[11] = lo_40;
      float hi_41 = fmaxf(a[4], a[12]);
      float lo_42 = fminf(a[4], a[12]);
      a[4] = hi_41;
      a[12] = lo_42;
      float hi_43 = fmaxf(a[5], a[13]);
      float lo_44 = fminf(a[5], a[13]);
      a[5] = hi_43;
      a[13] = lo_44;
      float hi_45 = fmaxf(a[6], a[14]);
      float lo_46 = fminf(a[6], a[14]);
      a[6] = hi_45;
      a[14] = lo_46;
      float hi_47 = fmaxf(a[7], a[15]);
      float lo_48 = fminf(a[7], a[15]);
      a[7] = hi_47;
      a[15] = lo_48;
      float hi_49 = fmaxf(a[0], a[4]);
      float lo_50 = fminf(a[0], a[4]);
      a[0] = hi_49;
      a[4] = lo_50;
      float hi_51 = fmaxf(a[1], a[5]);
      float lo_52 = fminf(a[1], a[5]);
      a[1] = hi_51;
      a[5] = lo_52;
      float hi_53 = fmaxf(a[2], a[6]);
      float lo_54 = fminf(a[2], a[6]);
      a[2] = hi_53;
      a[6] = lo_54;
      float hi_55 = fmaxf(a[3], a[7]);
      float lo_56 = fminf(a[3], a[7]);
      a[3] = hi_55;
      a[7] = lo_56;
      float hi_57 = fmaxf(a[8], a[12]);
      float lo_58 = fminf(a[8], a[12]);
      a[8] = hi_57;
      a[12] = lo_58;
      float hi_59 = fmaxf(a[9], a[13]);
      float lo_60 = fminf(a[9], a[13]);
      a[9] = hi_59;
      a[13] = lo_60;
      float hi_61 = fmaxf(a[10], a[14]);
      float lo_62 = fminf(a[10], a[14]);
      a[10] = hi_61;
      a[14] = lo_62;
      float hi_63 = fmaxf(a[11], a[15]);
      float lo_64 = fminf(a[11], a[15]);
      a[11] = hi_63;
      a[15] = lo_64;
      float hi_65 = fmaxf(a[0], a[2]);
      float lo_66 = fminf(a[0], a[2]);
      a[0] = hi_65;
      a[2] = lo_66;
      float hi_67 = fmaxf(a[1], a[3]);
      float lo_68 = fminf(a[1], a[3]);
      a[1] = hi_67;
      a[3] = lo_68;
      float hi_69 = fmaxf(a[4], a[6]);
      float lo_70 = fminf(a[4], a[6]);
      a[4] = hi_69;
      a[6] = lo_70;
      float hi_71 = fmaxf(a[5], a[7]);
      float lo_72 = fminf(a[5], a[7]);
      a[5] = hi_71;
      a[7] = lo_72;
      float hi_73 = fmaxf(a[8], a[10]);
      float lo_74 = fminf(a[8], a[10]);
      a[8] = hi_73;
      a[10] = lo_74;
      float hi_75 = fmaxf(a[9], a[11]);
      float lo_76 = fminf(a[9], a[11]);
      a[9] = hi_75;
      a[11] = lo_76;
      float hi_77 = fmaxf(a[12], a[14]);
      float lo_78 = fminf(a[12], a[14]);
      a[12] = hi_77;
      a[14] = lo_78;
      float hi_79 = fmaxf(a[13], a[15]);
      float lo_80 = fminf(a[13], a[15]);
      a[13] = hi_79;
      a[15] = lo_80;
      float hi_81 = fmaxf(a[0], a[1]);
      float lo_82 = fminf(a[0], a[1]);
      a[0] = hi_81;
      a[1] = lo_82;
      float hi_83 = fmaxf(a[2], a[3]);
      float lo_84 = fminf(a[2], a[3]);
      a[2] = hi_83;
      a[3] = lo_84;
      float hi_85 = fmaxf(a[4], a[5]);
      float lo_86 = fminf(a[4], a[5]);
      a[4] = hi_85;
      a[5] = lo_86;
      float hi_87 = fmaxf(a[6], a[7]);
      float lo_88 = fminf(a[6], a[7]);
      a[6] = hi_87;
      a[7] = lo_88;
      float hi_89 = fmaxf(a[8], a[9]);
      float lo_90 = fminf(a[8], a[9]);
      a[8] = hi_89;
      a[9] = lo_90;
      float hi_91 = fmaxf(a[10], a[11]);
      float lo_92 = fminf(a[10], a[11]);
      a[10] = hi_91;
      a[11] = lo_92;
      float hi_93 = fmaxf(a[12], a[13]);
      float lo_94 = fminf(a[12], a[13]);
      a[12] = hi_93;
      a[13] = lo_94;
      float hi_95 = fmaxf(a[14], a[15]);
      float lo_96 = fminf(a[14], a[15]);
      a[14] = hi_95;
      a[15] = lo_96;
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
      int t0_2 = w * 16;
      float sc_3 = __uint_as_float(cb[0]);
      sc_3 = fmaxf(sc_3, -1.7014118346046923e+38f);
      sc_3 = fminf(sc_3, 1.7014118346046923e+38f);
      unsigned int u8 = __as_u32(sc_3);
      unsigned int key8 = 0;
      if ((u8 & 4294967232u) == qc) {
        unsigned int lowb8 = (u8 ^ (unsigned int)((int)u8 >> 31) & 63) & 63;
        key8 = 536870912 | lowb8 << 6 | (unsigned int)t0_2;
      }
      int f2 = 0;
      if ((t0_2 < fb || t0_2 >= lim - fe) && lim > t0_2) {
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
      float sc_5 = __uint_as_float(cb[1]);
      sc_5 = fmaxf(sc_5, -1.7014118346046923e+38f);
      sc_5 = fminf(sc_5, 1.7014118346046923e+38f);
      unsigned int u8_6 = __as_u32(sc_5);
      unsigned int key8_7 = 0;
      if ((u8_6 & 4294967232u) == qc) {
        unsigned int lowb8_1 = (u8_6 ^ (unsigned int)((int)u8_6 >> 31) & 63) & 63;
        key8_7 = 536870912 | lowb8_1 << 6 | (unsigned int)(t0_2 + 1);
      }
      int f2_8 = 0;
      if ((t0_2 + 1 < fb || t0_2 + 1 >= lim - fe) && lim > t0_2 + 1) {
        f2_8 = 1;
      }
      if (f2_8 != 0) {
        key8_7 = 0;
      }
      if (key8_7 != 0) {
        unsigned int p6_1 = atomicAdd(&ccnt[c], 1);
        if (p6_1 < 16) {
          cbuf[c * 16 + (int)p6_1] = key8_7;
        } else {
          flagw[1] = 1;
        }
      }
      float sc_9 = __uint_as_float(cb[2]);
      sc_9 = fmaxf(sc_9, -1.7014118346046923e+38f);
      sc_9 = fminf(sc_9, 1.7014118346046923e+38f);
      unsigned int u8_10 = __as_u32(sc_9);
      unsigned int key8_11 = 0;
      if ((u8_10 & 4294967232u) == qc) {
        unsigned int lowb8_2 = (u8_10 ^ (unsigned int)((int)u8_10 >> 31) & 63) & 63;
        key8_11 = 536870912 | lowb8_2 << 6 | (unsigned int)(t0_2 + 2);
      }
      int f2_12 = 0;
      if ((t0_2 + 2 < fb || t0_2 + 2 >= lim - fe) && lim > t0_2 + 2) {
        f2_12 = 1;
      }
      if (f2_12 != 0) {
        key8_11 = 0;
      }
      if (key8_11 != 0) {
        unsigned int p6_2 = atomicAdd(&ccnt[c], 1);
        if (p6_2 < 16) {
          cbuf[c * 16 + (int)p6_2] = key8_11;
        } else {
          flagw[1] = 1;
        }
      }
      float sc_13 = __uint_as_float(cb[3]);
      sc_13 = fmaxf(sc_13, -1.7014118346046923e+38f);
      sc_13 = fminf(sc_13, 1.7014118346046923e+38f);
      unsigned int u8_14 = __as_u32(sc_13);
      unsigned int key8_15 = 0;
      if ((u8_14 & 4294967232u) == qc) {
        unsigned int lowb8_3 = (u8_14 ^ (unsigned int)((int)u8_14 >> 31) & 63) & 63;
        key8_15 = 536870912 | lowb8_3 << 6 | (unsigned int)(t0_2 + 3);
      }
      int f2_16 = 0;
      if ((t0_2 + 3 < fb || t0_2 + 3 >= lim - fe) && lim > t0_2 + 3) {
        f2_16 = 1;
      }
      if (f2_16 != 0) {
        key8_15 = 0;
      }
      if (key8_15 != 0) {
        unsigned int p6_3 = atomicAdd(&ccnt[c], 1);
        if (p6_3 < 16) {
          cbuf[c * 16 + (int)p6_3] = key8_15;
        } else {
          flagw[1] = 1;
        }
      }
      float sc_17 = __uint_as_float(cb[4]);
      sc_17 = fmaxf(sc_17, -1.7014118346046923e+38f);
      sc_17 = fminf(sc_17, 1.7014118346046923e+38f);
      unsigned int u8_18 = __as_u32(sc_17);
      unsigned int key8_19 = 0;
      if ((u8_18 & 4294967232u) == qc) {
        unsigned int lowb8_4 = (u8_18 ^ (unsigned int)((int)u8_18 >> 31) & 63) & 63;
        key8_19 = 536870912 | lowb8_4 << 6 | (unsigned int)(t0_2 + 4);
      }
      int f2_20 = 0;
      if ((t0_2 + 4 < fb || t0_2 + 4 >= lim - fe) && lim > t0_2 + 4) {
        f2_20 = 1;
      }
      if (f2_20 != 0) {
        key8_19 = 0;
      }
      if (key8_19 != 0) {
        unsigned int p6_4 = atomicAdd(&ccnt[c], 1);
        if (p6_4 < 16) {
          cbuf[c * 16 + (int)p6_4] = key8_19;
        } else {
          flagw[1] = 1;
        }
      }
      float sc_21 = __uint_as_float(cb[5]);
      sc_21 = fmaxf(sc_21, -1.7014118346046923e+38f);
      sc_21 = fminf(sc_21, 1.7014118346046923e+38f);
      unsigned int u8_22 = __as_u32(sc_21);
      unsigned int key8_23 = 0;
      if ((u8_22 & 4294967232u) == qc) {
        unsigned int lowb8_5 = (u8_22 ^ (unsigned int)((int)u8_22 >> 31) & 63) & 63;
        key8_23 = 536870912 | lowb8_5 << 6 | (unsigned int)(t0_2 + 5);
      }
      int f2_24 = 0;
      if ((t0_2 + 5 < fb || t0_2 + 5 >= lim - fe) && lim > t0_2 + 5) {
        f2_24 = 1;
      }
      if (f2_24 != 0) {
        key8_23 = 0;
      }
      if (key8_23 != 0) {
        unsigned int p6_5 = atomicAdd(&ccnt[c], 1);
        if (p6_5 < 16) {
          cbuf[c * 16 + (int)p6_5] = key8_23;
        } else {
          flagw[1] = 1;
        }
      }
      float sc_25 = __uint_as_float(cb[6]);
      sc_25 = fmaxf(sc_25, -1.7014118346046923e+38f);
      sc_25 = fminf(sc_25, 1.7014118346046923e+38f);
      unsigned int u8_26 = __as_u32(sc_25);
      unsigned int key8_27 = 0;
      if ((u8_26 & 4294967232u) == qc) {
        unsigned int lowb8_6 = (u8_26 ^ (unsigned int)((int)u8_26 >> 31) & 63) & 63;
        key8_27 = 536870912 | lowb8_6 << 6 | (unsigned int)(t0_2 + 6);
      }
      int f2_28 = 0;
      if ((t0_2 + 6 < fb || t0_2 + 6 >= lim - fe) && lim > t0_2 + 6) {
        f2_28 = 1;
      }
      if (f2_28 != 0) {
        key8_27 = 0;
      }
      if (key8_27 != 0) {
        unsigned int p6_6 = atomicAdd(&ccnt[c], 1);
        if (p6_6 < 16) {
          cbuf[c * 16 + (int)p6_6] = key8_27;
        } else {
          flagw[1] = 1;
        }
      }
      float sc_29 = __uint_as_float(cb[7]);
      sc_29 = fmaxf(sc_29, -1.7014118346046923e+38f);
      sc_29 = fminf(sc_29, 1.7014118346046923e+38f);
      unsigned int u8_30 = __as_u32(sc_29);
      unsigned int key8_31 = 0;
      if ((u8_30 & 4294967232u) == qc) {
        unsigned int lowb8_7 = (u8_30 ^ (unsigned int)((int)u8_30 >> 31) & 63) & 63;
        key8_31 = 536870912 | lowb8_7 << 6 | (unsigned int)(t0_2 + 7);
      }
      int f2_32 = 0;
      if ((t0_2 + 7 < fb || t0_2 + 7 >= lim - fe) && lim > t0_2 + 7) {
        f2_32 = 1;
      }
      if (f2_32 != 0) {
        key8_31 = 0;
      }
      if (key8_31 != 0) {
        unsigned int p6_7 = atomicAdd(&ccnt[c], 1);
        if (p6_7 < 16) {
          cbuf[c * 16 + (int)p6_7] = key8_31;
        } else {
          flagw[1] = 1;
        }
      }
      float sc_33 = __uint_as_float(cb[8]);
      sc_33 = fmaxf(sc_33, -1.7014118346046923e+38f);
      sc_33 = fminf(sc_33, 1.7014118346046923e+38f);
      unsigned int u8_34 = __as_u32(sc_33);
      unsigned int key8_35 = 0;
      if ((u8_34 & 4294967232u) == qc) {
        unsigned int lowb8_8 = (u8_34 ^ (unsigned int)((int)u8_34 >> 31) & 63) & 63;
        key8_35 = 536870912 | lowb8_8 << 6 | (unsigned int)(t0_2 + 8);
      }
      int f2_36 = 0;
      if ((t0_2 + 8 < fb || t0_2 + 8 >= lim - fe) && lim > t0_2 + 8) {
        f2_36 = 1;
      }
      if (f2_36 != 0) {
        key8_35 = 0;
      }
      if (key8_35 != 0) {
        unsigned int p6_8 = atomicAdd(&ccnt[c], 1);
        if (p6_8 < 16) {
          cbuf[c * 16 + (int)p6_8] = key8_35;
        } else {
          flagw[1] = 1;
        }
      }
      float sc_37 = __uint_as_float(cb[9]);
      sc_37 = fmaxf(sc_37, -1.7014118346046923e+38f);
      sc_37 = fminf(sc_37, 1.7014118346046923e+38f);
      unsigned int u8_38 = __as_u32(sc_37);
      unsigned int key8_39 = 0;
      if ((u8_38 & 4294967232u) == qc) {
        unsigned int lowb8_9 = (u8_38 ^ (unsigned int)((int)u8_38 >> 31) & 63) & 63;
        key8_39 = 536870912 | lowb8_9 << 6 | (unsigned int)(t0_2 + 9);
      }
      int f2_40 = 0;
      if ((t0_2 + 9 < fb || t0_2 + 9 >= lim - fe) && lim > t0_2 + 9) {
        f2_40 = 1;
      }
      if (f2_40 != 0) {
        key8_39 = 0;
      }
      if (key8_39 != 0) {
        unsigned int p6_9 = atomicAdd(&ccnt[c], 1);
        if (p6_9 < 16) {
          cbuf[c * 16 + (int)p6_9] = key8_39;
        } else {
          flagw[1] = 1;
        }
      }
      float sc_41 = __uint_as_float(cb[10]);
      sc_41 = fmaxf(sc_41, -1.7014118346046923e+38f);
      sc_41 = fminf(sc_41, 1.7014118346046923e+38f);
      unsigned int u8_42 = __as_u32(sc_41);
      unsigned int key8_43 = 0;
      if ((u8_42 & 4294967232u) == qc) {
        unsigned int lowb8_10 = (u8_42 ^ (unsigned int)((int)u8_42 >> 31) & 63) & 63;
        key8_43 = 536870912 | lowb8_10 << 6 | (unsigned int)(t0_2 + 10);
      }
      int f2_44 = 0;
      if ((t0_2 + 10 < fb || t0_2 + 10 >= lim - fe) && lim > t0_2 + 10) {
        f2_44 = 1;
      }
      if (f2_44 != 0) {
        key8_43 = 0;
      }
      if (key8_43 != 0) {
        unsigned int p6_10 = atomicAdd(&ccnt[c], 1);
        if (p6_10 < 16) {
          cbuf[c * 16 + (int)p6_10] = key8_43;
        } else {
          flagw[1] = 1;
        }
      }
      float sc_45 = __uint_as_float(cb[11]);
      sc_45 = fmaxf(sc_45, -1.7014118346046923e+38f);
      sc_45 = fminf(sc_45, 1.7014118346046923e+38f);
      unsigned int u8_46 = __as_u32(sc_45);
      unsigned int key8_47 = 0;
      if ((u8_46 & 4294967232u) == qc) {
        unsigned int lowb8_11 = (u8_46 ^ (unsigned int)((int)u8_46 >> 31) & 63) & 63;
        key8_47 = 536870912 | lowb8_11 << 6 | (unsigned int)(t0_2 + 11);
      }
      int f2_48 = 0;
      if ((t0_2 + 11 < fb || t0_2 + 11 >= lim - fe) && lim > t0_2 + 11) {
        f2_48 = 1;
      }
      if (f2_48 != 0) {
        key8_47 = 0;
      }
      if (key8_47 != 0) {
        unsigned int p6_11 = atomicAdd(&ccnt[c], 1);
        if (p6_11 < 16) {
          cbuf[c * 16 + (int)p6_11] = key8_47;
        } else {
          flagw[1] = 1;
        }
      }
      float sc_49 = __uint_as_float(cb[12]);
      sc_49 = fmaxf(sc_49, -1.7014118346046923e+38f);
      sc_49 = fminf(sc_49, 1.7014118346046923e+38f);
      unsigned int u8_50 = __as_u32(sc_49);
      unsigned int key8_51 = 0;
      if ((u8_50 & 4294967232u) == qc) {
        unsigned int lowb8_12 = (u8_50 ^ (unsigned int)((int)u8_50 >> 31) & 63) & 63;
        key8_51 = 536870912 | lowb8_12 << 6 | (unsigned int)(t0_2 + 12);
      }
      int f2_52 = 0;
      if ((t0_2 + 12 < fb || t0_2 + 12 >= lim - fe) && lim > t0_2 + 12) {
        f2_52 = 1;
      }
      if (f2_52 != 0) {
        key8_51 = 0;
      }
      if (key8_51 != 0) {
        unsigned int p6_12 = atomicAdd(&ccnt[c], 1);
        if (p6_12 < 16) {
          cbuf[c * 16 + (int)p6_12] = key8_51;
        } else {
          flagw[1] = 1;
        }
      }
      float sc_53 = __uint_as_float(cb[13]);
      sc_53 = fmaxf(sc_53, -1.7014118346046923e+38f);
      sc_53 = fminf(sc_53, 1.7014118346046923e+38f);
      unsigned int u8_54 = __as_u32(sc_53);
      unsigned int key8_55 = 0;
      if ((u8_54 & 4294967232u) == qc) {
        unsigned int lowb8_13 = (u8_54 ^ (unsigned int)((int)u8_54 >> 31) & 63) & 63;
        key8_55 = 536870912 | lowb8_13 << 6 | (unsigned int)(t0_2 + 13);
      }
      int f2_56 = 0;
      if ((t0_2 + 13 < fb || t0_2 + 13 >= lim - fe) && lim > t0_2 + 13) {
        f2_56 = 1;
      }
      if (f2_56 != 0) {
        key8_55 = 0;
      }
      if (key8_55 != 0) {
        unsigned int p6_13 = atomicAdd(&ccnt[c], 1);
        if (p6_13 < 16) {
          cbuf[c * 16 + (int)p6_13] = key8_55;
        } else {
          flagw[1] = 1;
        }
      }
      float sc_57 = __uint_as_float(cb[14]);
      sc_57 = fmaxf(sc_57, -1.7014118346046923e+38f);
      sc_57 = fminf(sc_57, 1.7014118346046923e+38f);
      unsigned int u8_58 = __as_u32(sc_57);
      unsigned int key8_59 = 0;
      if ((u8_58 & 4294967232u) == qc) {
        unsigned int lowb8_14 = (u8_58 ^ (unsigned int)((int)u8_58 >> 31) & 63) & 63;
        key8_59 = 536870912 | lowb8_14 << 6 | (unsigned int)(t0_2 + 14);
      }
      int f2_60 = 0;
      if ((t0_2 + 14 < fb || t0_2 + 14 >= lim - fe) && lim > t0_2 + 14) {
        f2_60 = 1;
      }
      if (f2_60 != 0) {
        key8_59 = 0;
      }
      if (key8_59 != 0) {
        unsigned int p6_14 = atomicAdd(&ccnt[c], 1);
        if (p6_14 < 16) {
          cbuf[c * 16 + (int)p6_14] = key8_59;
        } else {
          flagw[1] = 1;
        }
      }
      float sc_61 = __uint_as_float(cb[15]);
      sc_61 = fmaxf(sc_61, -1.7014118346046923e+38f);
      sc_61 = fminf(sc_61, 1.7014118346046923e+38f);
      unsigned int u8_62 = __as_u32(sc_61);
      unsigned int key8_63 = 0;
      if ((u8_62 & 4294967232u) == qc) {
        unsigned int lowb8_15 = (u8_62 ^ (unsigned int)((int)u8_62 >> 31) & 63) & 63;
        key8_63 = 536870912 | lowb8_15 << 6 | (unsigned int)(t0_2 + 15);
      }
      int f2_64 = 0;
      if ((t0_2 + 15 < fb || t0_2 + 15 >= lim - fe) && lim > t0_2 + 15) {
        f2_64 = 1;
      }
      if (f2_64 != 0) {
        key8_63 = 0;
      }
      if (key8_63 != 0) {
        unsigned int p6_15 = atomicAdd(&ccnt[c], 1);
        if (p6_15 < 16) {
          cbuf[c * 16 + (int)p6_15] = key8_63;
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
        int t0_2_1 = w * 16;
        float qf = __uint_as_float(qc);
        float sc_3_1 = __uint_as_float(cb[0]);
        sc_3_1 = fmaxf(sc_3_1, -1.7014118346046923e+38f);
        sc_3_1 = fminf(sc_3_1, 1.7014118346046923e+38f);
        unsigned int u = __as_u32(sc_3_1);
        unsigned int cls = u & 4294967232u;
        int f_4_1 = 0;
        if ((t0_2_1 < fb || t0_2_1 >= lim - fe) && lim > t0_2_1) {
          f_4_1 = 1;
        }
        if (f_4_1 != 0) {
          cls = 2139094976;
        }
        unsigned int key_6 = 0;
        if (qf < __uint_as_float(cls) && cls < 4278190080u) {
          key_6 = 1073741824 | (unsigned int)t0_2_1;
        }
        if (cls == qc) {
          unsigned int lowb = (u ^ (unsigned int)((int)u >> 31) & 63) & 63;
          key_6 = 536870912 | lowb << 6 | (unsigned int)t0_2_1;
        }
        kb2o[0] = __uint_as_float(key_6);
        float sc_7 = __uint_as_float(cb[1]);
        sc_7 = fmaxf(sc_7, -1.7014118346046923e+38f);
        sc_7 = fminf(sc_7, 1.7014118346046923e+38f);
        unsigned int u_8 = __as_u32(sc_7);
        unsigned int cls_9 = u_8 & 4294967232u;
        int f_10_1 = 0;
        if ((t0_2_1 + 1 < fb || t0_2_1 + 1 >= lim - fe) && lim > t0_2_1 + 1) {
          f_10_1 = 1;
        }
        if (f_10_1 != 0) {
          cls_9 = 2139094976;
        }
        unsigned int key_12 = 0;
        if (qf < __uint_as_float(cls_9) && cls_9 < 4278190080u) {
          key_12 = 1073741824 | (unsigned int)(t0_2_1 + 1);
        }
        if (cls_9 == qc) {
          unsigned int lowb_1 = (u_8 ^ (unsigned int)((int)u_8 >> 31) & 63) & 63;
          key_12 = 536870912 | lowb_1 << 6 | (unsigned int)(t0_2_1 + 1);
        }
        kb2o[1] = __uint_as_float(key_12);
        float sc_13_1 = __uint_as_float(cb[2]);
        sc_13_1 = fmaxf(sc_13_1, -1.7014118346046923e+38f);
        sc_13_1 = fminf(sc_13_1, 1.7014118346046923e+38f);
        unsigned int u_14 = __as_u32(sc_13_1);
        unsigned int cls_15 = u_14 & 4294967232u;
        int f_16 = 0;
        if ((t0_2_1 + 2 < fb || t0_2_1 + 2 >= lim - fe) && lim > t0_2_1 + 2) {
          f_16 = 1;
        }
        if (f_16 != 0) {
          cls_15 = 2139094976;
        }
        unsigned int key_18 = 0;
        if (qf < __uint_as_float(cls_15) && cls_15 < 4278190080u) {
          key_18 = 1073741824 | (unsigned int)(t0_2_1 + 2);
        }
        if (cls_15 == qc) {
          unsigned int lowb_2 = (u_14 ^ (unsigned int)((int)u_14 >> 31) & 63) & 63;
          key_18 = 536870912 | lowb_2 << 6 | (unsigned int)(t0_2_1 + 2);
        }
        kb2o[2] = __uint_as_float(key_18);
        float sc_19 = __uint_as_float(cb[3]);
        sc_19 = fmaxf(sc_19, -1.7014118346046923e+38f);
        sc_19 = fminf(sc_19, 1.7014118346046923e+38f);
        unsigned int u_20 = __as_u32(sc_19);
        unsigned int cls_21 = u_20 & 4294967232u;
        int f_22 = 0;
        if ((t0_2_1 + 3 < fb || t0_2_1 + 3 >= lim - fe) && lim > t0_2_1 + 3) {
          f_22 = 1;
        }
        if (f_22 != 0) {
          cls_21 = 2139094976;
        }
        unsigned int key_24 = 0;
        if (qf < __uint_as_float(cls_21) && cls_21 < 4278190080u) {
          key_24 = 1073741824 | (unsigned int)(t0_2_1 + 3);
        }
        if (cls_21 == qc) {
          unsigned int lowb_3 = (u_20 ^ (unsigned int)((int)u_20 >> 31) & 63) & 63;
          key_24 = 536870912 | lowb_3 << 6 | (unsigned int)(t0_2_1 + 3);
        }
        kb2o[3] = __uint_as_float(key_24);
        float sc_25_1 = __uint_as_float(cb[4]);
        sc_25_1 = fmaxf(sc_25_1, -1.7014118346046923e+38f);
        sc_25_1 = fminf(sc_25_1, 1.7014118346046923e+38f);
        unsigned int u_26 = __as_u32(sc_25_1);
        unsigned int cls_27 = u_26 & 4294967232u;
        int f_28 = 0;
        if ((t0_2_1 + 4 < fb || t0_2_1 + 4 >= lim - fe) && lim > t0_2_1 + 4) {
          f_28 = 1;
        }
        if (f_28 != 0) {
          cls_27 = 2139094976;
        }
        unsigned int key_30 = 0;
        if (qf < __uint_as_float(cls_27) && cls_27 < 4278190080u) {
          key_30 = 1073741824 | (unsigned int)(t0_2_1 + 4);
        }
        if (cls_27 == qc) {
          unsigned int lowb_4 = (u_26 ^ (unsigned int)((int)u_26 >> 31) & 63) & 63;
          key_30 = 536870912 | lowb_4 << 6 | (unsigned int)(t0_2_1 + 4);
        }
        kb2o[4] = __uint_as_float(key_30);
        float sc_31 = __uint_as_float(cb[5]);
        sc_31 = fmaxf(sc_31, -1.7014118346046923e+38f);
        sc_31 = fminf(sc_31, 1.7014118346046923e+38f);
        unsigned int u_32 = __as_u32(sc_31);
        unsigned int cls_33 = u_32 & 4294967232u;
        int f_34 = 0;
        if ((t0_2_1 + 5 < fb || t0_2_1 + 5 >= lim - fe) && lim > t0_2_1 + 5) {
          f_34 = 1;
        }
        if (f_34 != 0) {
          cls_33 = 2139094976;
        }
        unsigned int key_35 = 0;
        if (qf < __uint_as_float(cls_33) && cls_33 < 4278190080u) {
          key_35 = 1073741824 | (unsigned int)(t0_2_1 + 5);
        }
        if (cls_33 == qc) {
          unsigned int lowb_5 = (u_32 ^ (unsigned int)((int)u_32 >> 31) & 63) & 63;
          key_35 = 536870912 | lowb_5 << 6 | (unsigned int)(t0_2_1 + 5);
        }
        kb2o[5] = __uint_as_float(key_35);
        float sc_36 = __uint_as_float(cb[6]);
        sc_36 = fmaxf(sc_36, -1.7014118346046923e+38f);
        sc_36 = fminf(sc_36, 1.7014118346046923e+38f);
        unsigned int u_37 = __as_u32(sc_36);
        unsigned int cls_38 = u_37 & 4294967232u;
        int f_39 = 0;
        if ((t0_2_1 + 6 < fb || t0_2_1 + 6 >= lim - fe) && lim > t0_2_1 + 6) {
          f_39 = 1;
        }
        if (f_39 != 0) {
          cls_38 = 2139094976;
        }
        unsigned int key_40 = 0;
        if (qf < __uint_as_float(cls_38) && cls_38 < 4278190080u) {
          key_40 = 1073741824 | (unsigned int)(t0_2_1 + 6);
        }
        if (cls_38 == qc) {
          unsigned int lowb_6 = (u_37 ^ (unsigned int)((int)u_37 >> 31) & 63) & 63;
          key_40 = 536870912 | lowb_6 << 6 | (unsigned int)(t0_2_1 + 6);
        }
        kb2o[6] = __uint_as_float(key_40);
        float sc_41_1 = __uint_as_float(cb[7]);
        sc_41_1 = fmaxf(sc_41_1, -1.7014118346046923e+38f);
        sc_41_1 = fminf(sc_41_1, 1.7014118346046923e+38f);
        unsigned int u_42 = __as_u32(sc_41_1);
        unsigned int cls_43 = u_42 & 4294967232u;
        int f_44 = 0;
        if ((t0_2_1 + 7 < fb || t0_2_1 + 7 >= lim - fe) && lim > t0_2_1 + 7) {
          f_44 = 1;
        }
        if (f_44 != 0) {
          cls_43 = 2139094976;
        }
        unsigned int key_45 = 0;
        if (qf < __uint_as_float(cls_43) && cls_43 < 4278190080u) {
          key_45 = 1073741824 | (unsigned int)(t0_2_1 + 7);
        }
        if (cls_43 == qc) {
          unsigned int lowb_7 = (u_42 ^ (unsigned int)((int)u_42 >> 31) & 63) & 63;
          key_45 = 536870912 | lowb_7 << 6 | (unsigned int)(t0_2_1 + 7);
        }
        kb2o[7] = __uint_as_float(key_45);
        float sc_46 = __uint_as_float(cb[8]);
        sc_46 = fmaxf(sc_46, -1.7014118346046923e+38f);
        sc_46 = fminf(sc_46, 1.7014118346046923e+38f);
        unsigned int u_47 = __as_u32(sc_46);
        unsigned int cls_48 = u_47 & 4294967232u;
        int f_49 = 0;
        if ((t0_2_1 + 8 < fb || t0_2_1 + 8 >= lim - fe) && lim > t0_2_1 + 8) {
          f_49 = 1;
        }
        if (f_49 != 0) {
          cls_48 = 2139094976;
        }
        unsigned int key_50 = 0;
        if (qf < __uint_as_float(cls_48) && cls_48 < 4278190080u) {
          key_50 = 1073741824 | (unsigned int)(t0_2_1 + 8);
        }
        if (cls_48 == qc) {
          unsigned int lowb_8 = (u_47 ^ (unsigned int)((int)u_47 >> 31) & 63) & 63;
          key_50 = 536870912 | lowb_8 << 6 | (unsigned int)(t0_2_1 + 8);
        }
        kb2o[8] = __uint_as_float(key_50);
        float sc_51 = __uint_as_float(cb[9]);
        sc_51 = fmaxf(sc_51, -1.7014118346046923e+38f);
        sc_51 = fminf(sc_51, 1.7014118346046923e+38f);
        unsigned int u_52 = __as_u32(sc_51);
        unsigned int cls_53 = u_52 & 4294967232u;
        int f_54 = 0;
        if ((t0_2_1 + 9 < fb || t0_2_1 + 9 >= lim - fe) && lim > t0_2_1 + 9) {
          f_54 = 1;
        }
        if (f_54 != 0) {
          cls_53 = 2139094976;
        }
        unsigned int key_55 = 0;
        if (qf < __uint_as_float(cls_53) && cls_53 < 4278190080u) {
          key_55 = 1073741824 | (unsigned int)(t0_2_1 + 9);
        }
        if (cls_53 == qc) {
          unsigned int lowb_9 = (u_52 ^ (unsigned int)((int)u_52 >> 31) & 63) & 63;
          key_55 = 536870912 | lowb_9 << 6 | (unsigned int)(t0_2_1 + 9);
        }
        kb2o[9] = __uint_as_float(key_55);
        float sc_56 = __uint_as_float(cb[10]);
        sc_56 = fmaxf(sc_56, -1.7014118346046923e+38f);
        sc_56 = fminf(sc_56, 1.7014118346046923e+38f);
        unsigned int u_57 = __as_u32(sc_56);
        unsigned int cls_58 = u_57 & 4294967232u;
        int f_59 = 0;
        if ((t0_2_1 + 10 < fb || t0_2_1 + 10 >= lim - fe) && lim > t0_2_1 + 10) {
          f_59 = 1;
        }
        if (f_59 != 0) {
          cls_58 = 2139094976;
        }
        unsigned int key_60 = 0;
        if (qf < __uint_as_float(cls_58) && cls_58 < 4278190080u) {
          key_60 = 1073741824 | (unsigned int)(t0_2_1 + 10);
        }
        if (cls_58 == qc) {
          unsigned int lowb_10 = (u_57 ^ (unsigned int)((int)u_57 >> 31) & 63) & 63;
          key_60 = 536870912 | lowb_10 << 6 | (unsigned int)(t0_2_1 + 10);
        }
        kb2o[10] = __uint_as_float(key_60);
        float sc_61_1 = __uint_as_float(cb[11]);
        sc_61_1 = fmaxf(sc_61_1, -1.7014118346046923e+38f);
        sc_61_1 = fminf(sc_61_1, 1.7014118346046923e+38f);
        unsigned int u_62 = __as_u32(sc_61_1);
        unsigned int cls_63 = u_62 & 4294967232u;
        int f_64 = 0;
        if ((t0_2_1 + 11 < fb || t0_2_1 + 11 >= lim - fe) && lim > t0_2_1 + 11) {
          f_64 = 1;
        }
        if (f_64 != 0) {
          cls_63 = 2139094976;
        }
        unsigned int key_65 = 0;
        if (qf < __uint_as_float(cls_63) && cls_63 < 4278190080u) {
          key_65 = 1073741824 | (unsigned int)(t0_2_1 + 11);
        }
        if (cls_63 == qc) {
          unsigned int lowb_11 = (u_62 ^ (unsigned int)((int)u_62 >> 31) & 63) & 63;
          key_65 = 536870912 | lowb_11 << 6 | (unsigned int)(t0_2_1 + 11);
        }
        kb2o[11] = __uint_as_float(key_65);
        float sc_66 = __uint_as_float(cb[12]);
        sc_66 = fmaxf(sc_66, -1.7014118346046923e+38f);
        sc_66 = fminf(sc_66, 1.7014118346046923e+38f);
        unsigned int u_67 = __as_u32(sc_66);
        unsigned int cls_68 = u_67 & 4294967232u;
        int f_69 = 0;
        if ((t0_2_1 + 12 < fb || t0_2_1 + 12 >= lim - fe) && lim > t0_2_1 + 12) {
          f_69 = 1;
        }
        if (f_69 != 0) {
          cls_68 = 2139094976;
        }
        unsigned int key_70 = 0;
        if (qf < __uint_as_float(cls_68) && cls_68 < 4278190080u) {
          key_70 = 1073741824 | (unsigned int)(t0_2_1 + 12);
        }
        if (cls_68 == qc) {
          unsigned int lowb_12 = (u_67 ^ (unsigned int)((int)u_67 >> 31) & 63) & 63;
          key_70 = 536870912 | lowb_12 << 6 | (unsigned int)(t0_2_1 + 12);
        }
        kb2o[12] = __uint_as_float(key_70);
        float sc_71 = __uint_as_float(cb[13]);
        sc_71 = fmaxf(sc_71, -1.7014118346046923e+38f);
        sc_71 = fminf(sc_71, 1.7014118346046923e+38f);
        unsigned int u_72 = __as_u32(sc_71);
        unsigned int cls_73 = u_72 & 4294967232u;
        int f_74 = 0;
        if ((t0_2_1 + 13 < fb || t0_2_1 + 13 >= lim - fe) && lim > t0_2_1 + 13) {
          f_74 = 1;
        }
        if (f_74 != 0) {
          cls_73 = 2139094976;
        }
        unsigned int key_75 = 0;
        if (qf < __uint_as_float(cls_73) && cls_73 < 4278190080u) {
          key_75 = 1073741824 | (unsigned int)(t0_2_1 + 13);
        }
        if (cls_73 == qc) {
          unsigned int lowb_13 = (u_72 ^ (unsigned int)((int)u_72 >> 31) & 63) & 63;
          key_75 = 536870912 | lowb_13 << 6 | (unsigned int)(t0_2_1 + 13);
        }
        kb2o[13] = __uint_as_float(key_75);
        float sc_76 = __uint_as_float(cb[14]);
        sc_76 = fmaxf(sc_76, -1.7014118346046923e+38f);
        sc_76 = fminf(sc_76, 1.7014118346046923e+38f);
        unsigned int u_77 = __as_u32(sc_76);
        unsigned int cls_78 = u_77 & 4294967232u;
        int f_79 = 0;
        if ((t0_2_1 + 14 < fb || t0_2_1 + 14 >= lim - fe) && lim > t0_2_1 + 14) {
          f_79 = 1;
        }
        if (f_79 != 0) {
          cls_78 = 2139094976;
        }
        unsigned int key_80 = 0;
        if (qf < __uint_as_float(cls_78) && cls_78 < 4278190080u) {
          key_80 = 1073741824 | (unsigned int)(t0_2_1 + 14);
        }
        if (cls_78 == qc) {
          unsigned int lowb_14 = (u_77 ^ (unsigned int)((int)u_77 >> 31) & 63) & 63;
          key_80 = 536870912 | lowb_14 << 6 | (unsigned int)(t0_2_1 + 14);
        }
        kb2o[14] = __uint_as_float(key_80);
        float sc_81 = __uint_as_float(cb[15]);
        sc_81 = fmaxf(sc_81, -1.7014118346046923e+38f);
        sc_81 = fminf(sc_81, 1.7014118346046923e+38f);
        unsigned int u_82 = __as_u32(sc_81);
        unsigned int cls_83 = u_82 & 4294967232u;
        int f_84 = 0;
        if ((t0_2_1 + 15 < fb || t0_2_1 + 15 >= lim - fe) && lim > t0_2_1 + 15) {
          f_84 = 1;
        }
        if (f_84 != 0) {
          cls_83 = 2139094976;
        }
        unsigned int key_85 = 0;
        if (qf < __uint_as_float(cls_83) && cls_83 < 4278190080u) {
          key_85 = 1073741824 | (unsigned int)(t0_2_1 + 15);
        }
        if (cls_83 == qc) {
          unsigned int lowb_15 = (u_82 ^ (unsigned int)((int)u_82 >> 31) & 63) & 63;
          key_85 = 536870912 | lowb_15 << 6 | (unsigned int)(t0_2_1 + 15);
        }
        kb2o[15] = __uint_as_float(key_85);
        float hi_87_1 = fmaxf(kb2o[0], kb2o[13]);
        float lo_88_1 = fminf(kb2o[0], kb2o[13]);
        kb2o[0] = hi_87_1;
        kb2o[13] = lo_88_1;
        float hi_89_1 = fmaxf(kb2o[1], kb2o[12]);
        float lo_90_1 = fminf(kb2o[1], kb2o[12]);
        kb2o[1] = hi_89_1;
        kb2o[12] = lo_90_1;
        float hi_91_1 = fmaxf(kb2o[2], kb2o[15]);
        float lo_92_1 = fminf(kb2o[2], kb2o[15]);
        kb2o[2] = hi_91_1;
        kb2o[15] = lo_92_1;
        float hi_93_1 = fmaxf(kb2o[3], kb2o[14]);
        float lo_94_1 = fminf(kb2o[3], kb2o[14]);
        kb2o[3] = hi_93_1;
        kb2o[14] = lo_94_1;
        float hi_95_1 = fmaxf(kb2o[4], kb2o[8]);
        float lo_96_1 = fminf(kb2o[4], kb2o[8]);
        kb2o[4] = hi_95_1;
        kb2o[8] = lo_96_1;
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
        float hi_153 = fmaxf(kb2o[3], kb2o[12]);
        float lo_154 = fminf(kb2o[3], kb2o[12]);
        kb2o[3] = hi_153;
        kb2o[12] = lo_154;
        float hi_155 = fmaxf(kb2o[4], kb2o[6]);
        float lo_156 = fminf(kb2o[4], kb2o[6]);
        kb2o[4] = hi_155;
        kb2o[6] = lo_156;
        float hi_157 = fmaxf(kb2o[5], kb2o[7]);
        float lo_158 = fminf(kb2o[5], kb2o[7]);
        kb2o[5] = hi_157;
        kb2o[7] = lo_158;
        float hi_159 = fmaxf(kb2o[8], kb2o[10]);
        float lo_160 = fminf(kb2o[8], kb2o[10]);
        kb2o[8] = hi_159;
        kb2o[10] = lo_160;
        float hi_161 = fmaxf(kb2o[9], kb2o[11]);
        float lo_162 = fminf(kb2o[9], kb2o[11]);
        kb2o[9] = hi_161;
        kb2o[11] = lo_162;
        float hi_163 = fmaxf(kb2o[13], kb2o[14]);
        float lo_164 = fminf(kb2o[13], kb2o[14]);
        kb2o[13] = hi_163;
        kb2o[14] = lo_164;
        float hi_165 = fmaxf(kb2o[1], kb2o[4]);
        float lo_166 = fminf(kb2o[1], kb2o[4]);
        kb2o[1] = hi_165;
        kb2o[4] = lo_166;
        float hi_167 = fmaxf(kb2o[2], kb2o[6]);
        float lo_168 = fminf(kb2o[2], kb2o[6]);
        kb2o[2] = hi_167;
        kb2o[6] = lo_168;
        float hi_169 = fmaxf(kb2o[5], kb2o[8]);
        float lo_170 = fminf(kb2o[5], kb2o[8]);
        kb2o[5] = hi_169;
        kb2o[8] = lo_170;
        float hi_171 = fmaxf(kb2o[7], kb2o[10]);
        float lo_172 = fminf(kb2o[7], kb2o[10]);
        kb2o[7] = hi_171;
        kb2o[10] = lo_172;
        float hi_173 = fmaxf(kb2o[9], kb2o[13]);
        float lo_174 = fminf(kb2o[9], kb2o[13]);
        kb2o[9] = hi_173;
        kb2o[13] = lo_174;
        float hi_175 = fmaxf(kb2o[11], kb2o[14]);
        float lo_176 = fminf(kb2o[11], kb2o[14]);
        kb2o[11] = hi_175;
        kb2o[14] = lo_176;
        float hi_177 = fmaxf(kb2o[2], kb2o[4]);
        float lo_178 = fminf(kb2o[2], kb2o[4]);
        kb2o[2] = hi_177;
        kb2o[4] = lo_178;
        float hi_179 = fmaxf(kb2o[3], kb2o[6]);
        float lo_180 = fminf(kb2o[3], kb2o[6]);
        kb2o[3] = hi_179;
        kb2o[6] = lo_180;
        float hi_181 = fmaxf(kb2o[9], kb2o[12]);
        float lo_182 = fminf(kb2o[9], kb2o[12]);
        kb2o[9] = hi_181;
        kb2o[12] = lo_182;
        float hi_183 = fmaxf(kb2o[11], kb2o[13]);
        float lo_184 = fminf(kb2o[11], kb2o[13]);
        kb2o[11] = hi_183;
        kb2o[13] = lo_184;
        float hi_185 = fmaxf(kb2o[3], kb2o[5]);
        float lo_186 = fminf(kb2o[3], kb2o[5]);
        kb2o[3] = hi_185;
        kb2o[5] = lo_186;
        float hi_187 = fmaxf(kb2o[6], kb2o[8]);
        float lo_188 = fminf(kb2o[6], kb2o[8]);
        kb2o[6] = hi_187;
        kb2o[8] = lo_188;
        float hi_189 = fmaxf(kb2o[7], kb2o[9]);
        float lo_190 = fminf(kb2o[7], kb2o[9]);
        kb2o[7] = hi_189;
        kb2o[9] = lo_190;
        float hi_191 = fmaxf(kb2o[10], kb2o[12]);
        float lo_192 = fminf(kb2o[10], kb2o[12]);
        kb2o[10] = hi_191;
        kb2o[12] = lo_192;
        float hi_193 = fmaxf(kb2o[3], kb2o[4]);
        float lo_194 = fminf(kb2o[3], kb2o[4]);
        kb2o[3] = hi_193;
        kb2o[4] = lo_194;
        float hi_195 = fmaxf(kb2o[5], kb2o[6]);
        float lo_196 = fminf(kb2o[5], kb2o[6]);
        kb2o[5] = hi_195;
        kb2o[6] = lo_196;
        float hi_197 = fmaxf(kb2o[7], kb2o[8]);
        float lo_198 = fminf(kb2o[7], kb2o[8]);
        kb2o[7] = hi_197;
        kb2o[8] = lo_198;
        float hi_199 = fmaxf(kb2o[9], kb2o[10]);
        float lo_200 = fminf(kb2o[9], kb2o[10]);
        kb2o[9] = hi_199;
        kb2o[10] = lo_200;
        float hi_201 = fmaxf(kb2o[11], kb2o[12]);
        float lo_202 = fminf(kb2o[11], kb2o[12]);
        kb2o[11] = hi_201;
        kb2o[12] = lo_202;
        float hi_203 = fmaxf(kb2o[6], kb2o[7]);
        float lo_204 = fminf(kb2o[6], kb2o[7]);
        kb2o[6] = hi_203;
        kb2o[7] = lo_204;
        float hi_205 = fmaxf(kb2o[8], kb2o[9]);
        float lo_206 = fminf(kb2o[8], kb2o[9]);
        kb2o[8] = hi_205;
        kb2o[9] = lo_206;
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
          float hi_0_1 = fmaxf(a2[0], sb2[15]);
          a2[0] = hi_0_1;
          float hi_1 = fmaxf(a2[1], sb2[14]);
          a2[1] = hi_1;
          float hi_2_1 = fmaxf(a2[2], sb2[13]);
          a2[2] = hi_2_1;
          float hi_3 = fmaxf(a2[3], sb2[12]);
          a2[3] = hi_3;
          float hi_4_1 = fmaxf(a2[4], sb2[11]);
          a2[4] = hi_4_1;
          float hi_5 = fmaxf(a2[5], sb2[10]);
          a2[5] = hi_5;
          float hi_6_1 = fmaxf(a2[6], sb2[9]);
          a2[6] = hi_6_1;
          float hi_7 = fmaxf(a2[7], sb2[8]);
          a2[7] = hi_7;
          float hi_8_1 = fmaxf(a2[8], sb2[7]);
          a2[8] = hi_8_1;
          float hi_9 = fmaxf(a2[9], sb2[6]);
          a2[9] = hi_9;
          float hi_10_1 = fmaxf(a2[10], sb2[5]);
          a2[10] = hi_10_1;
          float hi_11 = fmaxf(a2[11], sb2[4]);
          a2[11] = hi_11;
          float hi_12_1 = fmaxf(a2[12], sb2[3]);
          a2[12] = hi_12_1;
          float hi_13 = fmaxf(a2[13], sb2[2]);
          a2[13] = hi_13;
          float hi_14_1 = fmaxf(a2[14], sb2[1]);
          a2[14] = hi_14_1;
          float hi_15 = fmaxf(a2[15], sb2[0]);
          a2[15] = hi_15;
          float hi_16_1 = fmaxf(a2[0], a2[8]);
          float lo_17_1 = fminf(a2[0], a2[8]);
          a2[0] = hi_16_1;
          a2[8] = lo_17_1;
          float hi_18_1 = fmaxf(a2[1], a2[9]);
          float lo_19_1 = fminf(a2[1], a2[9]);
          a2[1] = hi_18_1;
          a2[9] = lo_19_1;
          float hi_20_1 = fmaxf(a2[2], a2[10]);
          float lo_21_1 = fminf(a2[2], a2[10]);
          a2[2] = hi_20_1;
          a2[10] = lo_21_1;
          float hi_22_1 = fmaxf(a2[3], a2[11]);
          float lo_23_1 = fminf(a2[3], a2[11]);
          a2[3] = hi_22_1;
          a2[11] = lo_23_1;
          float hi_24_1 = fmaxf(a2[4], a2[12]);
          float lo_25_1 = fminf(a2[4], a2[12]);
          a2[4] = hi_24_1;
          a2[12] = lo_25_1;
          float hi_26_1 = fmaxf(a2[5], a2[13]);
          float lo_27_1 = fminf(a2[5], a2[13]);
          a2[5] = hi_26_1;
          a2[13] = lo_27_1;
          float hi_28_1 = fmaxf(a2[6], a2[14]);
          float lo_29_1 = fminf(a2[6], a2[14]);
          a2[6] = hi_28_1;
          a2[14] = lo_29_1;
          float hi_30_1 = fmaxf(a2[7], a2[15]);
          float lo_31_1 = fminf(a2[7], a2[15]);
          a2[7] = hi_30_1;
          a2[15] = lo_31_1;
          float hi_33_1 = fmaxf(a2[0], a2[4]);
          float lo_34_1 = fminf(a2[0], a2[4]);
          a2[0] = hi_33_1;
          a2[4] = lo_34_1;
          float hi_35_1 = fmaxf(a2[1], a2[5]);
          float lo_36_1 = fminf(a2[1], a2[5]);
          a2[1] = hi_35_1;
          a2[5] = lo_36_1;
          float hi_37_1 = fmaxf(a2[2], a2[6]);
          float lo_38_1 = fminf(a2[2], a2[6]);
          a2[2] = hi_37_1;
          a2[6] = lo_38_1;
          float hi_39_1 = fmaxf(a2[3], a2[7]);
          float lo_40_1 = fminf(a2[3], a2[7]);
          a2[3] = hi_39_1;
          a2[7] = lo_40_1;
          float hi_41_1 = fmaxf(a2[8], a2[12]);
          float lo_42_1 = fminf(a2[8], a2[12]);
          a2[8] = hi_41_1;
          a2[12] = lo_42_1;
          float hi_43_1 = fmaxf(a2[9], a2[13]);
          float lo_44_1 = fminf(a2[9], a2[13]);
          a2[9] = hi_43_1;
          a2[13] = lo_44_1;
          float hi_45_1 = fmaxf(a2[10], a2[14]);
          float lo_46_1 = fminf(a2[10], a2[14]);
          a2[10] = hi_45_1;
          a2[14] = lo_46_1;
          float hi_47_1 = fmaxf(a2[11], a2[15]);
          float lo_48_1 = fminf(a2[11], a2[15]);
          a2[11] = hi_47_1;
          a2[15] = lo_48_1;
          float hi_49_1 = fmaxf(a2[0], a2[2]);
          float lo_50_1 = fminf(a2[0], a2[2]);
          a2[0] = hi_49_1;
          a2[2] = lo_50_1;
          float hi_51_1 = fmaxf(a2[1], a2[3]);
          float lo_52_1 = fminf(a2[1], a2[3]);
          a2[1] = hi_51_1;
          a2[3] = lo_52_1;
          float hi_53_1 = fmaxf(a2[4], a2[6]);
          float lo_54_1 = fminf(a2[4], a2[6]);
          a2[4] = hi_53_1;
          a2[6] = lo_54_1;
          float hi_55_1 = fmaxf(a2[5], a2[7]);
          float lo_56_1 = fminf(a2[5], a2[7]);
          a2[5] = hi_55_1;
          a2[7] = lo_56_1;
          float hi_57_1 = fmaxf(a2[8], a2[10]);
          float lo_58_1 = fminf(a2[8], a2[10]);
          a2[8] = hi_57_1;
          a2[10] = lo_58_1;
          float hi_59_1 = fmaxf(a2[9], a2[11]);
          float lo_60_1 = fminf(a2[9], a2[11]);
          a2[9] = hi_59_1;
          a2[11] = lo_60_1;
          float hi_61_1 = fmaxf(a2[12], a2[14]);
          float lo_62_1 = fminf(a2[12], a2[14]);
          a2[12] = hi_61_1;
          a2[14] = lo_62_1;
          float hi_63_1 = fmaxf(a2[13], a2[15]);
          float lo_64_1 = fminf(a2[13], a2[15]);
          a2[13] = hi_63_1;
          a2[15] = lo_64_1;
          float hi_65_1 = fmaxf(a2[0], a2[1]);
          float lo_66_1 = fminf(a2[0], a2[1]);
          a2[0] = hi_65_1;
          a2[1] = lo_66_1;
          float hi_67_1 = fmaxf(a2[2], a2[3]);
          float lo_68_1 = fminf(a2[2], a2[3]);
          a2[2] = hi_67_1;
          a2[3] = lo_68_1;
          float hi_69_1 = fmaxf(a2[4], a2[5]);
          float lo_70_1 = fminf(a2[4], a2[5]);
          a2[4] = hi_69_1;
          a2[5] = lo_70_1;
          float hi_71_1 = fmaxf(a2[6], a2[7]);
          float lo_72_1 = fminf(a2[6], a2[7]);
          a2[6] = hi_71_1;
          a2[7] = lo_72_1;
          float hi_73_1 = fmaxf(a2[8], a2[9]);
          float lo_74_1 = fminf(a2[8], a2[9]);
          a2[8] = hi_73_1;
          a2[9] = lo_74_1;
          float hi_75_1 = fmaxf(a2[10], a2[11]);
          float lo_76_1 = fminf(a2[10], a2[11]);
          a2[10] = hi_75_1;
          a2[11] = lo_76_1;
          float hi_77_1 = fmaxf(a2[12], a2[13]);
          float lo_78_1 = fminf(a2[12], a2[13]);
          a2[12] = hi_77_1;
          a2[13] = lo_78_1;
          float hi_79_1 = fmaxf(a2[14], a2[15]);
          float lo_80_1 = fminf(a2[14], a2[15]);
          a2[14] = hi_79_1;
          a2[15] = lo_80_1;
        }
      }
    }
    if (w == 0 && flagged != 0 && cntx <= 16) {
      unsigned int k1s = __as_u32(a[0]);
      a2[0] = __uint_as_float(1073741824 | k1s & 63);
      if ((k1s & 4294967232u) == qc) {
        a2[0] = 0.0f;
      }
      unsigned int k1s_0 = __as_u32(a[1]);
      a2[1] = __uint_as_float(1073741824 | k1s_0 & 63);
      if ((k1s_0 & 4294967232u) == qc) {
        a2[1] = 0.0f;
      }
      unsigned int k1s_1 = __as_u32(a[2]);
      a2[2] = __uint_as_float(1073741824 | k1s_1 & 63);
      if ((k1s_1 & 4294967232u) == qc) {
        a2[2] = 0.0f;
      }
      unsigned int k1s_2 = __as_u32(a[3]);
      a2[3] = __uint_as_float(1073741824 | k1s_2 & 63);
      if ((k1s_2 & 4294967232u) == qc) {
        a2[3] = 0.0f;
      }
      unsigned int k1s_3 = __as_u32(a[4]);
      a2[4] = __uint_as_float(1073741824 | k1s_3 & 63);
      if ((k1s_3 & 4294967232u) == qc) {
        a2[4] = 0.0f;
      }
      unsigned int k1s_4 = __as_u32(a[5]);
      a2[5] = __uint_as_float(1073741824 | k1s_4 & 63);
      if ((k1s_4 & 4294967232u) == qc) {
        a2[5] = 0.0f;
      }
      unsigned int k1s_5 = __as_u32(a[6]);
      a2[6] = __uint_as_float(1073741824 | k1s_5 & 63);
      if ((k1s_5 & 4294967232u) == qc) {
        a2[6] = 0.0f;
      }
      unsigned int k1s_6 = __as_u32(a[7]);
      a2[7] = __uint_as_float(1073741824 | k1s_6 & 63);
      if ((k1s_6 & 4294967232u) == qc) {
        a2[7] = 0.0f;
      }
      unsigned int k1s_7 = __as_u32(a[8]);
      a2[8] = __uint_as_float(1073741824 | k1s_7 & 63);
      if ((k1s_7 & 4294967232u) == qc) {
        a2[8] = 0.0f;
      }
      unsigned int k1s_8 = __as_u32(a[9]);
      a2[9] = __uint_as_float(1073741824 | k1s_8 & 63);
      if ((k1s_8 & 4294967232u) == qc) {
        a2[9] = 0.0f;
      }
      unsigned int k1s_9 = __as_u32(a[10]);
      a2[10] = __uint_as_float(1073741824 | k1s_9 & 63);
      if ((k1s_9 & 4294967232u) == qc) {
        a2[10] = 0.0f;
      }
      unsigned int k1s_10 = __as_u32(a[11]);
      a2[11] = __uint_as_float(1073741824 | k1s_10 & 63);
      if ((k1s_10 & 4294967232u) == qc) {
        a2[11] = 0.0f;
      }
      unsigned int k1s_11 = __as_u32(a[12]);
      a2[12] = __uint_as_float(1073741824 | k1s_11 & 63);
      if ((k1s_11 & 4294967232u) == qc) {
        a2[12] = 0.0f;
      }
      unsigned int k1s_12 = __as_u32(a[13]);
      a2[13] = __uint_as_float(1073741824 | k1s_12 & 63);
      if ((k1s_12 & 4294967232u) == qc) {
        a2[13] = 0.0f;
      }
      unsigned int k1s_13 = __as_u32(a[14]);
      a2[14] = __uint_as_float(1073741824 | k1s_13 & 63);
      if ((k1s_13 & 4294967232u) == qc) {
        a2[14] = 0.0f;
      }
      unsigned int k1s_14 = __as_u32(a[15]);
      a2[15] = __uint_as_float(1073741824 | k1s_14 & 63);
      if ((k1s_14 & 4294967232u) == qc) {
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
      float hi_15_1 = fmaxf(xs2[0], xs2[13]);
      float lo_16 = fminf(xs2[0], xs2[13]);
      xs2[0] = hi_15_1;
      xs2[13] = lo_16;
      float hi_17 = fmaxf(xs2[1], xs2[12]);
      float lo_18 = fminf(xs2[1], xs2[12]);
      xs2[1] = hi_17;
      xs2[12] = lo_18;
      float hi_19 = fmaxf(xs2[2], xs2[15]);
      float lo_20 = fminf(xs2[2], xs2[15]);
      xs2[2] = hi_19;
      xs2[15] = lo_20;
      float hi_21 = fmaxf(xs2[3], xs2[14]);
      float lo_22 = fminf(xs2[3], xs2[14]);
      xs2[3] = hi_21;
      xs2[14] = lo_22;
      float hi_23 = fmaxf(xs2[4], xs2[8]);
      float lo_24 = fminf(xs2[4], xs2[8]);
      xs2[4] = hi_23;
      xs2[8] = lo_24;
      float hi_25 = fmaxf(xs2[5], xs2[6]);
      float lo_26 = fminf(xs2[5], xs2[6]);
      xs2[5] = hi_25;
      xs2[6] = lo_26;
      float hi_27 = fmaxf(xs2[7], xs2[11]);
      float lo_28 = fminf(xs2[7], xs2[11]);
      xs2[7] = hi_27;
      xs2[11] = lo_28;
      float hi_29 = fmaxf(xs2[9], xs2[10]);
      float lo_30 = fminf(xs2[9], xs2[10]);
      xs2[9] = hi_29;
      xs2[10] = lo_30;
      float hi_31 = fmaxf(xs2[0], xs2[5]);
      float lo_32 = fminf(xs2[0], xs2[5]);
      xs2[0] = hi_31;
      xs2[5] = lo_32;
      float hi_33_2 = fmaxf(xs2[1], xs2[7]);
      float lo_34_2 = fminf(xs2[1], xs2[7]);
      xs2[1] = hi_33_2;
      xs2[7] = lo_34_2;
      float hi_35_2 = fmaxf(xs2[2], xs2[9]);
      float lo_36_2 = fminf(xs2[2], xs2[9]);
      xs2[2] = hi_35_2;
      xs2[9] = lo_36_2;
      float hi_37_2 = fmaxf(xs2[3], xs2[4]);
      float lo_38_2 = fminf(xs2[3], xs2[4]);
      xs2[3] = hi_37_2;
      xs2[4] = lo_38_2;
      float hi_39_2 = fmaxf(xs2[6], xs2[13]);
      float lo_40_2 = fminf(xs2[6], xs2[13]);
      xs2[6] = hi_39_2;
      xs2[13] = lo_40_2;
      float hi_41_2 = fmaxf(xs2[8], xs2[14]);
      float lo_42_2 = fminf(xs2[8], xs2[14]);
      xs2[8] = hi_41_2;
      xs2[14] = lo_42_2;
      float hi_43_2 = fmaxf(xs2[10], xs2[15]);
      float lo_44_2 = fminf(xs2[10], xs2[15]);
      xs2[10] = hi_43_2;
      xs2[15] = lo_44_2;
      float hi_45_2 = fmaxf(xs2[11], xs2[12]);
      float lo_46_2 = fminf(xs2[11], xs2[12]);
      xs2[11] = hi_45_2;
      xs2[12] = lo_46_2;
      float hi_47_2 = fmaxf(xs2[0], xs2[1]);
      float lo_48_2 = fminf(xs2[0], xs2[1]);
      xs2[0] = hi_47_2;
      xs2[1] = lo_48_2;
      float hi_49_2 = fmaxf(xs2[2], xs2[3]);
      float lo_50_2 = fminf(xs2[2], xs2[3]);
      xs2[2] = hi_49_2;
      xs2[3] = lo_50_2;
      float hi_51_2 = fmaxf(xs2[4], xs2[5]);
      float lo_52_2 = fminf(xs2[4], xs2[5]);
      xs2[4] = hi_51_2;
      xs2[5] = lo_52_2;
      float hi_53_2 = fmaxf(xs2[6], xs2[8]);
      float lo_54_2 = fminf(xs2[6], xs2[8]);
      xs2[6] = hi_53_2;
      xs2[8] = lo_54_2;
      float hi_55_2 = fmaxf(xs2[7], xs2[9]);
      float lo_56_2 = fminf(xs2[7], xs2[9]);
      xs2[7] = hi_55_2;
      xs2[9] = lo_56_2;
      float hi_57_2 = fmaxf(xs2[10], xs2[11]);
      float lo_58_2 = fminf(xs2[10], xs2[11]);
      xs2[10] = hi_57_2;
      xs2[11] = lo_58_2;
      float hi_59_2 = fmaxf(xs2[12], xs2[13]);
      float lo_60_2 = fminf(xs2[12], xs2[13]);
      xs2[12] = hi_59_2;
      xs2[13] = lo_60_2;
      float hi_61_2 = fmaxf(xs2[14], xs2[15]);
      float lo_62_2 = fminf(xs2[14], xs2[15]);
      xs2[14] = hi_61_2;
      xs2[15] = lo_62_2;
      float hi_63_2 = fmaxf(xs2[0], xs2[2]);
      float lo_64_2 = fminf(xs2[0], xs2[2]);
      xs2[0] = hi_63_2;
      xs2[2] = lo_64_2;
      float hi_65_2 = fmaxf(xs2[1], xs2[3]);
      float lo_66_2 = fminf(xs2[1], xs2[3]);
      xs2[1] = hi_65_2;
      xs2[3] = lo_66_2;
      float hi_67_2 = fmaxf(xs2[4], xs2[10]);
      float lo_68_2 = fminf(xs2[4], xs2[10]);
      xs2[4] = hi_67_2;
      xs2[10] = lo_68_2;
      float hi_69_2 = fmaxf(xs2[5], xs2[11]);
      float lo_70_2 = fminf(xs2[5], xs2[11]);
      xs2[5] = hi_69_2;
      xs2[11] = lo_70_2;
      float hi_71_2 = fmaxf(xs2[6], xs2[7]);
      float lo_72_2 = fminf(xs2[6], xs2[7]);
      xs2[6] = hi_71_2;
      xs2[7] = lo_72_2;
      float hi_73_2 = fmaxf(xs2[8], xs2[9]);
      float lo_74_2 = fminf(xs2[8], xs2[9]);
      xs2[8] = hi_73_2;
      xs2[9] = lo_74_2;
      float hi_75_2 = fmaxf(xs2[12], xs2[14]);
      float lo_76_2 = fminf(xs2[12], xs2[14]);
      xs2[12] = hi_75_2;
      xs2[14] = lo_76_2;
      float hi_77_2 = fmaxf(xs2[13], xs2[15]);
      float lo_78_2 = fminf(xs2[13], xs2[15]);
      xs2[13] = hi_77_2;
      xs2[15] = lo_78_2;
      float hi_79_2 = fmaxf(xs2[1], xs2[2]);
      float lo_80_2 = fminf(xs2[1], xs2[2]);
      xs2[1] = hi_79_2;
      xs2[2] = lo_80_2;
      float hi_81_1 = fmaxf(xs2[3], xs2[12]);
      float lo_82_1 = fminf(xs2[3], xs2[12]);
      xs2[3] = hi_81_1;
      xs2[12] = lo_82_1;
      float hi_83_1 = fmaxf(xs2[4], xs2[6]);
      float lo_84_1 = fminf(xs2[4], xs2[6]);
      xs2[4] = hi_83_1;
      xs2[6] = lo_84_1;
      float hi_85_1 = fmaxf(xs2[5], xs2[7]);
      float lo_86_1 = fminf(xs2[5], xs2[7]);
      xs2[5] = hi_85_1;
      xs2[7] = lo_86_1;
      float hi_87_2 = fmaxf(xs2[8], xs2[10]);
      float lo_88_2 = fminf(xs2[8], xs2[10]);
      xs2[8] = hi_87_2;
      xs2[10] = lo_88_2;
      float hi_89_2 = fmaxf(xs2[9], xs2[11]);
      float lo_90_2 = fminf(xs2[9], xs2[11]);
      xs2[9] = hi_89_2;
      xs2[11] = lo_90_2;
      float hi_91_2 = fmaxf(xs2[13], xs2[14]);
      float lo_92_2 = fminf(xs2[13], xs2[14]);
      xs2[13] = hi_91_2;
      xs2[14] = lo_92_2;
      float hi_93_2 = fmaxf(xs2[1], xs2[4]);
      float lo_94_2 = fminf(xs2[1], xs2[4]);
      xs2[1] = hi_93_2;
      xs2[4] = lo_94_2;
      float hi_95_2 = fmaxf(xs2[2], xs2[6]);
      float lo_96_2 = fminf(xs2[2], xs2[6]);
      xs2[2] = hi_95_2;
      xs2[6] = lo_96_2;
      float hi_97_1 = fmaxf(xs2[5], xs2[8]);
      float lo_98_1 = fminf(xs2[5], xs2[8]);
      xs2[5] = hi_97_1;
      xs2[8] = lo_98_1;
      float hi_99_1 = fmaxf(xs2[7], xs2[10]);
      float lo_100_1 = fminf(xs2[7], xs2[10]);
      xs2[7] = hi_99_1;
      xs2[10] = lo_100_1;
      float hi_101_1 = fmaxf(xs2[9], xs2[13]);
      float lo_102_1 = fminf(xs2[9], xs2[13]);
      xs2[9] = hi_101_1;
      xs2[13] = lo_102_1;
      float hi_103_1 = fmaxf(xs2[11], xs2[14]);
      float lo_104_1 = fminf(xs2[11], xs2[14]);
      xs2[11] = hi_103_1;
      xs2[14] = lo_104_1;
      float hi_105_1 = fmaxf(xs2[2], xs2[4]);
      float lo_106_1 = fminf(xs2[2], xs2[4]);
      xs2[2] = hi_105_1;
      xs2[4] = lo_106_1;
      float hi_107_1 = fmaxf(xs2[3], xs2[6]);
      float lo_108_1 = fminf(xs2[3], xs2[6]);
      xs2[3] = hi_107_1;
      xs2[6] = lo_108_1;
      float hi_109_1 = fmaxf(xs2[9], xs2[12]);
      float lo_110_1 = fminf(xs2[9], xs2[12]);
      xs2[9] = hi_109_1;
      xs2[12] = lo_110_1;
      float hi_111_1 = fmaxf(xs2[11], xs2[13]);
      float lo_112_1 = fminf(xs2[11], xs2[13]);
      xs2[11] = hi_111_1;
      xs2[13] = lo_112_1;
      float hi_113_1 = fmaxf(xs2[3], xs2[5]);
      float lo_114_1 = fminf(xs2[3], xs2[5]);
      xs2[3] = hi_113_1;
      xs2[5] = lo_114_1;
      float hi_115_1 = fmaxf(xs2[6], xs2[8]);
      float lo_116_1 = fminf(xs2[6], xs2[8]);
      xs2[6] = hi_115_1;
      xs2[8] = lo_116_1;
      float hi_117_1 = fmaxf(xs2[7], xs2[9]);
      float lo_118_1 = fminf(xs2[7], xs2[9]);
      xs2[7] = hi_117_1;
      xs2[9] = lo_118_1;
      float hi_119_1 = fmaxf(xs2[10], xs2[12]);
      float lo_120_1 = fminf(xs2[10], xs2[12]);
      xs2[10] = hi_119_1;
      xs2[12] = lo_120_1;
      float hi_121_1 = fmaxf(xs2[3], xs2[4]);
      float lo_122_1 = fminf(xs2[3], xs2[4]);
      xs2[3] = hi_121_1;
      xs2[4] = lo_122_1;
      float hi_123_1 = fmaxf(xs2[5], xs2[6]);
      float lo_124_1 = fminf(xs2[5], xs2[6]);
      xs2[5] = hi_123_1;
      xs2[6] = lo_124_1;
      float hi_125_1 = fmaxf(xs2[7], xs2[8]);
      float lo_126_1 = fminf(xs2[7], xs2[8]);
      xs2[7] = hi_125_1;
      xs2[8] = lo_126_1;
      float hi_127_1 = fmaxf(xs2[9], xs2[10]);
      float lo_128_1 = fminf(xs2[9], xs2[10]);
      xs2[9] = hi_127_1;
      xs2[10] = lo_128_1;
      float hi_129_1 = fmaxf(xs2[11], xs2[12]);
      float lo_130_1 = fminf(xs2[11], xs2[12]);
      xs2[11] = hi_129_1;
      xs2[12] = lo_130_1;
      float hi_131_1 = fmaxf(xs2[6], xs2[7]);
      float lo_132_1 = fminf(xs2[6], xs2[7]);
      xs2[6] = hi_131_1;
      xs2[7] = lo_132_1;
      float hi_133_1 = fmaxf(xs2[8], xs2[9]);
      float lo_134_1 = fminf(xs2[8], xs2[9]);
      xs2[8] = hi_133_1;
      xs2[9] = lo_134_1;
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
    unsigned int k1 = __as_u32(a[0]);
    unsigned int idx = 4294967295u;
    if (k1 < 4278190080u) {
      idx = k1 & 63;
    }
    if (flagged != 0) {
      unsigned int k2 = __as_u32(a2[0]);
      idx = 4294967295u;
      if (k2 != 0) {
        idx = k2 & 63;
      }
    }
    o[0] = idx;
    unsigned int k1_0 = __as_u32(a[1]);
    unsigned int idx_1 = 4294967295u;
    if (k1_0 < 4278190080u) {
      idx_1 = k1_0 & 63;
    }
    if (flagged != 0) {
      unsigned int k2_1 = __as_u32(a2[1]);
      idx_1 = 4294967295u;
      if (k2_1 != 0) {
        idx_1 = k2_1 & 63;
      }
    }
    o[1] = idx_1;
    unsigned int k1_2 = __as_u32(a[2]);
    unsigned int idx_3 = 4294967295u;
    if (k1_2 < 4278190080u) {
      idx_3 = k1_2 & 63;
    }
    if (flagged != 0) {
      unsigned int k2_2 = __as_u32(a2[2]);
      idx_3 = 4294967295u;
      if (k2_2 != 0) {
        idx_3 = k2_2 & 63;
      }
    }
    o[2] = idx_3;
    unsigned int k1_4 = __as_u32(a[3]);
    unsigned int idx_5 = 4294967295u;
    if (k1_4 < 4278190080u) {
      idx_5 = k1_4 & 63;
    }
    if (flagged != 0) {
      unsigned int k2_3 = __as_u32(a2[3]);
      idx_5 = 4294967295u;
      if (k2_3 != 0) {
        idx_5 = k2_3 & 63;
      }
    }
    o[3] = idx_5;
    unsigned int k1_6 = __as_u32(a[4]);
    unsigned int idx_7 = 4294967295u;
    if (k1_6 < 4278190080u) {
      idx_7 = k1_6 & 63;
    }
    if (flagged != 0) {
      unsigned int k2_4 = __as_u32(a2[4]);
      idx_7 = 4294967295u;
      if (k2_4 != 0) {
        idx_7 = k2_4 & 63;
      }
    }
    o[4] = idx_7;
    unsigned int k1_8 = __as_u32(a[5]);
    unsigned int idx_9 = 4294967295u;
    if (k1_8 < 4278190080u) {
      idx_9 = k1_8 & 63;
    }
    if (flagged != 0) {
      unsigned int k2_5 = __as_u32(a2[5]);
      idx_9 = 4294967295u;
      if (k2_5 != 0) {
        idx_9 = k2_5 & 63;
      }
    }
    o[5] = idx_9;
    unsigned int k1_10 = __as_u32(a[6]);
    unsigned int idx_11 = 4294967295u;
    if (k1_10 < 4278190080u) {
      idx_11 = k1_10 & 63;
    }
    if (flagged != 0) {
      unsigned int k2_6 = __as_u32(a2[6]);
      idx_11 = 4294967295u;
      if (k2_6 != 0) {
        idx_11 = k2_6 & 63;
      }
    }
    o[6] = idx_11;
    unsigned int k1_12 = __as_u32(a[7]);
    unsigned int idx_13 = 4294967295u;
    if (k1_12 < 4278190080u) {
      idx_13 = k1_12 & 63;
    }
    if (flagged != 0) {
      unsigned int k2_7 = __as_u32(a2[7]);
      idx_13 = 4294967295u;
      if (k2_7 != 0) {
        idx_13 = k2_7 & 63;
      }
    }
    o[7] = idx_13;
    unsigned int k1_14 = __as_u32(a[8]);
    unsigned int idx_15 = 4294967295u;
    if (k1_14 < 4278190080u) {
      idx_15 = k1_14 & 63;
    }
    if (flagged != 0) {
      unsigned int k2_8 = __as_u32(a2[8]);
      idx_15 = 4294967295u;
      if (k2_8 != 0) {
        idx_15 = k2_8 & 63;
      }
    }
    o[8] = idx_15;
    unsigned int k1_16 = __as_u32(a[9]);
    unsigned int idx_17 = 4294967295u;
    if (k1_16 < 4278190080u) {
      idx_17 = k1_16 & 63;
    }
    if (flagged != 0) {
      unsigned int k2_9 = __as_u32(a2[9]);
      idx_17 = 4294967295u;
      if (k2_9 != 0) {
        idx_17 = k2_9 & 63;
      }
    }
    o[9] = idx_17;
    unsigned int k1_18 = __as_u32(a[10]);
    unsigned int idx_19 = 4294967295u;
    if (k1_18 < 4278190080u) {
      idx_19 = k1_18 & 63;
    }
    if (flagged != 0) {
      unsigned int k2_10 = __as_u32(a2[10]);
      idx_19 = 4294967295u;
      if (k2_10 != 0) {
        idx_19 = k2_10 & 63;
      }
    }
    o[10] = idx_19;
    unsigned int k1_20 = __as_u32(a[11]);
    unsigned int idx_21 = 4294967295u;
    if (k1_20 < 4278190080u) {
      idx_21 = k1_20 & 63;
    }
    if (flagged != 0) {
      unsigned int k2_11 = __as_u32(a2[11]);
      idx_21 = 4294967295u;
      if (k2_11 != 0) {
        idx_21 = k2_11 & 63;
      }
    }
    o[11] = idx_21;
    unsigned int k1_22 = __as_u32(a[12]);
    unsigned int idx_23 = 4294967295u;
    if (k1_22 < 4278190080u) {
      idx_23 = k1_22 & 63;
    }
    if (flagged != 0) {
      unsigned int k2_12 = __as_u32(a2[12]);
      idx_23 = 4294967295u;
      if (k2_12 != 0) {
        idx_23 = k2_12 & 63;
      }
    }
    o[12] = idx_23;
    unsigned int k1_24 = __as_u32(a[13]);
    unsigned int idx_25 = 4294967295u;
    if (k1_24 < 4278190080u) {
      idx_25 = k1_24 & 63;
    }
    if (flagged != 0) {
      unsigned int k2_13 = __as_u32(a2[13]);
      idx_25 = 4294967295u;
      if (k2_13 != 0) {
        idx_25 = k2_13 & 63;
      }
    }
    o[13] = idx_25;
    unsigned int k1_26 = __as_u32(a[14]);
    unsigned int idx_27 = 4294967295u;
    if (k1_26 < 4278190080u) {
      idx_27 = k1_26 & 63;
    }
    if (flagged != 0) {
      unsigned int k2_14 = __as_u32(a2[14]);
      idx_27 = 4294967295u;
      if (k2_14 != 0) {
        idx_27 = k2_14 & 63;
      }
    }
    o[14] = idx_27;
    unsigned int k1_28 = __as_u32(a[15]);
    unsigned int idx_29 = 4294967295u;
    if (k1_28 < 4278190080u) {
      idx_29 = k1_28 & 63;
    }
    if (flagged != 0) {
      unsigned int k2_15 = __as_u32(a2[15]);
      idx_29 = 4294967295u;
      if (k2_15 != 0) {
        idx_29 = k2_15 & 63;
      }
    }
    o[15] = idx_29;
    unsigned int hi_30_2 = ((o[0]) > (o[13]) ? (o[0]) : (o[13]));
    unsigned int lo_31_2 = ((o[0]) < (o[13]) ? (o[0]) : (o[13]));
    o[0] = lo_31_2;
    o[13] = hi_30_2;
    unsigned int hi_33_3 = ((o[1]) > (o[12]) ? (o[1]) : (o[12]));
    unsigned int lo_34_3 = ((o[1]) < (o[12]) ? (o[1]) : (o[12]));
    o[1] = lo_34_3;
    o[12] = hi_33_3;
    unsigned int hi_35_3 = ((o[2]) > (o[15]) ? (o[2]) : (o[15]));
    unsigned int lo_36_3 = ((o[2]) < (o[15]) ? (o[2]) : (o[15]));
    o[2] = lo_36_3;
    o[15] = hi_35_3;
    unsigned int hi_37_3 = ((o[3]) > (o[14]) ? (o[3]) : (o[14]));
    unsigned int lo_38_3 = ((o[3]) < (o[14]) ? (o[3]) : (o[14]));
    o[3] = lo_38_3;
    o[14] = hi_37_3;
    unsigned int hi_39_3 = ((o[4]) > (o[8]) ? (o[4]) : (o[8]));
    unsigned int lo_40_3 = ((o[4]) < (o[8]) ? (o[4]) : (o[8]));
    o[4] = lo_40_3;
    o[8] = hi_39_3;
    unsigned int hi_41_3 = ((o[5]) > (o[6]) ? (o[5]) : (o[6]));
    unsigned int lo_42_3 = ((o[5]) < (o[6]) ? (o[5]) : (o[6]));
    o[5] = lo_42_3;
    o[6] = hi_41_3;
    unsigned int hi_43_3 = ((o[7]) > (o[11]) ? (o[7]) : (o[11]));
    unsigned int lo_44_3 = ((o[7]) < (o[11]) ? (o[7]) : (o[11]));
    o[7] = lo_44_3;
    o[11] = hi_43_3;
    unsigned int hi_45_3 = ((o[9]) > (o[10]) ? (o[9]) : (o[10]));
    unsigned int lo_46_3 = ((o[9]) < (o[10]) ? (o[9]) : (o[10]));
    o[9] = lo_46_3;
    o[10] = hi_45_3;
    unsigned int hi_47_3 = ((o[0]) > (o[5]) ? (o[0]) : (o[5]));
    unsigned int lo_48_3 = ((o[0]) < (o[5]) ? (o[0]) : (o[5]));
    o[0] = lo_48_3;
    o[5] = hi_47_3;
    unsigned int hi_49_3 = ((o[1]) > (o[7]) ? (o[1]) : (o[7]));
    unsigned int lo_50_3 = ((o[1]) < (o[7]) ? (o[1]) : (o[7]));
    o[1] = lo_50_3;
    o[7] = hi_49_3;
    unsigned int hi_51_3 = ((o[2]) > (o[9]) ? (o[2]) : (o[9]));
    unsigned int lo_52_3 = ((o[2]) < (o[9]) ? (o[2]) : (o[9]));
    o[2] = lo_52_3;
    o[9] = hi_51_3;
    unsigned int hi_53_3 = ((o[3]) > (o[4]) ? (o[3]) : (o[4]));
    unsigned int lo_54_3 = ((o[3]) < (o[4]) ? (o[3]) : (o[4]));
    o[3] = lo_54_3;
    o[4] = hi_53_3;
    unsigned int hi_55_3 = ((o[6]) > (o[13]) ? (o[6]) : (o[13]));
    unsigned int lo_56_3 = ((o[6]) < (o[13]) ? (o[6]) : (o[13]));
    o[6] = lo_56_3;
    o[13] = hi_55_3;
    unsigned int hi_57_3 = ((o[8]) > (o[14]) ? (o[8]) : (o[14]));
    unsigned int lo_58_3 = ((o[8]) < (o[14]) ? (o[8]) : (o[14]));
    o[8] = lo_58_3;
    o[14] = hi_57_3;
    unsigned int hi_59_3 = ((o[10]) > (o[15]) ? (o[10]) : (o[15]));
    unsigned int lo_60_3 = ((o[10]) < (o[15]) ? (o[10]) : (o[15]));
    o[10] = lo_60_3;
    o[15] = hi_59_3;
    unsigned int hi_61_3 = ((o[11]) > (o[12]) ? (o[11]) : (o[12]));
    unsigned int lo_62_3 = ((o[11]) < (o[12]) ? (o[11]) : (o[12]));
    o[11] = lo_62_3;
    o[12] = hi_61_3;
    unsigned int hi_63_3 = ((o[0]) > (o[1]) ? (o[0]) : (o[1]));
    unsigned int lo_64_3 = ((o[0]) < (o[1]) ? (o[0]) : (o[1]));
    o[0] = lo_64_3;
    o[1] = hi_63_3;
    unsigned int hi_65_3 = ((o[2]) > (o[3]) ? (o[2]) : (o[3]));
    unsigned int lo_66_3 = ((o[2]) < (o[3]) ? (o[2]) : (o[3]));
    o[2] = lo_66_3;
    o[3] = hi_65_3;
    unsigned int hi_67_3 = ((o[4]) > (o[5]) ? (o[4]) : (o[5]));
    unsigned int lo_68_3 = ((o[4]) < (o[5]) ? (o[4]) : (o[5]));
    o[4] = lo_68_3;
    o[5] = hi_67_3;
    unsigned int hi_69_3 = ((o[6]) > (o[8]) ? (o[6]) : (o[8]));
    unsigned int lo_70_3 = ((o[6]) < (o[8]) ? (o[6]) : (o[8]));
    o[6] = lo_70_3;
    o[8] = hi_69_3;
    unsigned int hi_71_3 = ((o[7]) > (o[9]) ? (o[7]) : (o[9]));
    unsigned int lo_72_3 = ((o[7]) < (o[9]) ? (o[7]) : (o[9]));
    o[7] = lo_72_3;
    o[9] = hi_71_3;
    unsigned int hi_73_3 = ((o[10]) > (o[11]) ? (o[10]) : (o[11]));
    unsigned int lo_74_3 = ((o[10]) < (o[11]) ? (o[10]) : (o[11]));
    o[10] = lo_74_3;
    o[11] = hi_73_3;
    unsigned int hi_75_3 = ((o[12]) > (o[13]) ? (o[12]) : (o[13]));
    unsigned int lo_76_3 = ((o[12]) < (o[13]) ? (o[12]) : (o[13]));
    o[12] = lo_76_3;
    o[13] = hi_75_3;
    unsigned int hi_77_3 = ((o[14]) > (o[15]) ? (o[14]) : (o[15]));
    unsigned int lo_78_3 = ((o[14]) < (o[15]) ? (o[14]) : (o[15]));
    o[14] = lo_78_3;
    o[15] = hi_77_3;
    unsigned int hi_79_3 = ((o[0]) > (o[2]) ? (o[0]) : (o[2]));
    unsigned int lo_80_3 = ((o[0]) < (o[2]) ? (o[0]) : (o[2]));
    o[0] = lo_80_3;
    o[2] = hi_79_3;
    unsigned int hi_81_2 = ((o[1]) > (o[3]) ? (o[1]) : (o[3]));
    unsigned int lo_82_2 = ((o[1]) < (o[3]) ? (o[1]) : (o[3]));
    o[1] = lo_82_2;
    o[3] = hi_81_2;
    unsigned int hi_83_2 = ((o[4]) > (o[10]) ? (o[4]) : (o[10]));
    unsigned int lo_84_2 = ((o[4]) < (o[10]) ? (o[4]) : (o[10]));
    o[4] = lo_84_2;
    o[10] = hi_83_2;
    unsigned int hi_85_2 = ((o[5]) > (o[11]) ? (o[5]) : (o[11]));
    unsigned int lo_86_2 = ((o[5]) < (o[11]) ? (o[5]) : (o[11]));
    o[5] = lo_86_2;
    o[11] = hi_85_2;
    unsigned int hi_87_3 = ((o[6]) > (o[7]) ? (o[6]) : (o[7]));
    unsigned int lo_88_3 = ((o[6]) < (o[7]) ? (o[6]) : (o[7]));
    o[6] = lo_88_3;
    o[7] = hi_87_3;
    unsigned int hi_89_3 = ((o[8]) > (o[9]) ? (o[8]) : (o[9]));
    unsigned int lo_90_3 = ((o[8]) < (o[9]) ? (o[8]) : (o[9]));
    o[8] = lo_90_3;
    o[9] = hi_89_3;
    unsigned int hi_91_3 = ((o[12]) > (o[14]) ? (o[12]) : (o[14]));
    unsigned int lo_92_3 = ((o[12]) < (o[14]) ? (o[12]) : (o[14]));
    o[12] = lo_92_3;
    o[14] = hi_91_3;
    unsigned int hi_93_3 = ((o[13]) > (o[15]) ? (o[13]) : (o[15]));
    unsigned int lo_94_3 = ((o[13]) < (o[15]) ? (o[13]) : (o[15]));
    o[13] = lo_94_3;
    o[15] = hi_93_3;
    unsigned int hi_95_3 = ((o[1]) > (o[2]) ? (o[1]) : (o[2]));
    unsigned int lo_96_3 = ((o[1]) < (o[2]) ? (o[1]) : (o[2]));
    o[1] = lo_96_3;
    o[2] = hi_95_3;
    unsigned int hi_97_2 = ((o[3]) > (o[12]) ? (o[3]) : (o[12]));
    unsigned int lo_98_2 = ((o[3]) < (o[12]) ? (o[3]) : (o[12]));
    o[3] = lo_98_2;
    o[12] = hi_97_2;
    unsigned int hi_99_2 = ((o[4]) > (o[6]) ? (o[4]) : (o[6]));
    unsigned int lo_100_2 = ((o[4]) < (o[6]) ? (o[4]) : (o[6]));
    o[4] = lo_100_2;
    o[6] = hi_99_2;
    unsigned int hi_101_2 = ((o[5]) > (o[7]) ? (o[5]) : (o[7]));
    unsigned int lo_102_2 = ((o[5]) < (o[7]) ? (o[5]) : (o[7]));
    o[5] = lo_102_2;
    o[7] = hi_101_2;
    unsigned int hi_103_2 = ((o[8]) > (o[10]) ? (o[8]) : (o[10]));
    unsigned int lo_104_2 = ((o[8]) < (o[10]) ? (o[8]) : (o[10]));
    o[8] = lo_104_2;
    o[10] = hi_103_2;
    unsigned int hi_105_2 = ((o[9]) > (o[11]) ? (o[9]) : (o[11]));
    unsigned int lo_106_2 = ((o[9]) < (o[11]) ? (o[9]) : (o[11]));
    o[9] = lo_106_2;
    o[11] = hi_105_2;
    unsigned int hi_107_2 = ((o[13]) > (o[14]) ? (o[13]) : (o[14]));
    unsigned int lo_108_2 = ((o[13]) < (o[14]) ? (o[13]) : (o[14]));
    o[13] = lo_108_2;
    o[14] = hi_107_2;
    unsigned int hi_109_2 = ((o[1]) > (o[4]) ? (o[1]) : (o[4]));
    unsigned int lo_110_2 = ((o[1]) < (o[4]) ? (o[1]) : (o[4]));
    o[1] = lo_110_2;
    o[4] = hi_109_2;
    unsigned int hi_111_2 = ((o[2]) > (o[6]) ? (o[2]) : (o[6]));
    unsigned int lo_112_2 = ((o[2]) < (o[6]) ? (o[2]) : (o[6]));
    o[2] = lo_112_2;
    o[6] = hi_111_2;
    unsigned int hi_113_2 = ((o[5]) > (o[8]) ? (o[5]) : (o[8]));
    unsigned int lo_114_2 = ((o[5]) < (o[8]) ? (o[5]) : (o[8]));
    o[5] = lo_114_2;
    o[8] = hi_113_2;
    unsigned int hi_115_2 = ((o[7]) > (o[10]) ? (o[7]) : (o[10]));
    unsigned int lo_116_2 = ((o[7]) < (o[10]) ? (o[7]) : (o[10]));
    o[7] = lo_116_2;
    o[10] = hi_115_2;
    unsigned int hi_117_2 = ((o[9]) > (o[13]) ? (o[9]) : (o[13]));
    unsigned int lo_118_2 = ((o[9]) < (o[13]) ? (o[9]) : (o[13]));
    o[9] = lo_118_2;
    o[13] = hi_117_2;
    unsigned int hi_119_2 = ((o[11]) > (o[14]) ? (o[11]) : (o[14]));
    unsigned int lo_120_2 = ((o[11]) < (o[14]) ? (o[11]) : (o[14]));
    o[11] = lo_120_2;
    o[14] = hi_119_2;
    unsigned int hi_121_2 = ((o[2]) > (o[4]) ? (o[2]) : (o[4]));
    unsigned int lo_122_2 = ((o[2]) < (o[4]) ? (o[2]) : (o[4]));
    o[2] = lo_122_2;
    o[4] = hi_121_2;
    unsigned int hi_123_2 = ((o[3]) > (o[6]) ? (o[3]) : (o[6]));
    unsigned int lo_124_2 = ((o[3]) < (o[6]) ? (o[3]) : (o[6]));
    o[3] = lo_124_2;
    o[6] = hi_123_2;
    unsigned int hi_125_2 = ((o[9]) > (o[12]) ? (o[9]) : (o[12]));
    unsigned int lo_126_2 = ((o[9]) < (o[12]) ? (o[9]) : (o[12]));
    o[9] = lo_126_2;
    o[12] = hi_125_2;
    unsigned int hi_127_2 = ((o[11]) > (o[13]) ? (o[11]) : (o[13]));
    unsigned int lo_128_2 = ((o[11]) < (o[13]) ? (o[11]) : (o[13]));
    o[11] = lo_128_2;
    o[13] = hi_127_2;
    unsigned int hi_129_2 = ((o[3]) > (o[5]) ? (o[3]) : (o[5]));
    unsigned int lo_130_2 = ((o[3]) < (o[5]) ? (o[3]) : (o[5]));
    o[3] = lo_130_2;
    o[5] = hi_129_2;
    unsigned int hi_131_2 = ((o[6]) > (o[8]) ? (o[6]) : (o[8]));
    unsigned int lo_132_2 = ((o[6]) < (o[8]) ? (o[6]) : (o[8]));
    o[6] = lo_132_2;
    o[8] = hi_131_2;
    unsigned int hi_133_2 = ((o[7]) > (o[9]) ? (o[7]) : (o[9]));
    unsigned int lo_134_2 = ((o[7]) < (o[9]) ? (o[7]) : (o[9]));
    o[7] = lo_134_2;
    o[9] = hi_133_2;
    unsigned int hi_135_1 = ((o[10]) > (o[12]) ? (o[10]) : (o[12]));
    unsigned int lo_136_1 = ((o[10]) < (o[12]) ? (o[10]) : (o[12]));
    o[10] = lo_136_1;
    o[12] = hi_135_1;
    unsigned int hi_137_1 = ((o[3]) > (o[4]) ? (o[3]) : (o[4]));
    unsigned int lo_138_1 = ((o[3]) < (o[4]) ? (o[3]) : (o[4]));
    o[3] = lo_138_1;
    o[4] = hi_137_1;
    unsigned int hi_139_1 = ((o[5]) > (o[6]) ? (o[5]) : (o[6]));
    unsigned int lo_140_1 = ((o[5]) < (o[6]) ? (o[5]) : (o[6]));
    o[5] = lo_140_1;
    o[6] = hi_139_1;
    unsigned int hi_141_1 = ((o[7]) > (o[8]) ? (o[7]) : (o[8]));
    unsigned int lo_142_1 = ((o[7]) < (o[8]) ? (o[7]) : (o[8]));
    o[7] = lo_142_1;
    o[8] = hi_141_1;
    unsigned int hi_143_1 = ((o[9]) > (o[10]) ? (o[9]) : (o[10]));
    unsigned int lo_144_1 = ((o[9]) < (o[10]) ? (o[9]) : (o[10]));
    o[9] = lo_144_1;
    o[10] = hi_143_1;
    unsigned int hi_145_1 = ((o[11]) > (o[12]) ? (o[11]) : (o[12]));
    unsigned int lo_146_1 = ((o[11]) < (o[12]) ? (o[11]) : (o[12]));
    o[11] = lo_146_1;
    o[12] = hi_145_1;
    unsigned int hi_147_1 = ((o[6]) > (o[7]) ? (o[6]) : (o[7]));
    unsigned int lo_148_1 = ((o[6]) < (o[7]) ? (o[6]) : (o[7]));
    o[6] = lo_148_1;
    o[7] = hi_147_1;
    unsigned int hi_149_1 = ((o[8]) > (o[9]) ? (o[8]) : (o[9]));
    unsigned int lo_150_1 = ((o[8]) < (o[9]) ? (o[8]) : (o[9]));
    o[8] = lo_150_1;
    o[9] = hi_149_1;
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
    long long obase = ((long long)col * (long long)num_heads + (long long)head) * 16;
    {
      int4 _iv4 = make_int4(oi[0 + 0], oi[0 + 1], oi[0 + 2], oi[0 + 3]);
      *reinterpret_cast<int4*>(out + obase) = _iv4;
    }
    {
      int4 _iv4 = make_int4(oi[4 + 0], oi[4 + 1], oi[4 + 2], oi[4 + 3]);
      *reinterpret_cast<int4*>(out + obase + 4) = _iv4;
    }
    {
      int4 _iv4 = make_int4(oi[8 + 0], oi[8 + 1], oi[8 + 2], oi[8 + 3]);
      *reinterpret_cast<int4*>(out + obase + 8) = _iv4;
    }
    {
      int4 _iv4 = make_int4(oi[12 + 0], oi[12 + 1], oi[12 + 2], oi[12 + 3]);
      *reinterpret_cast<int4*>(out + obase + 12) = _iv4;
    }
  }
}

}  // extern "C"
