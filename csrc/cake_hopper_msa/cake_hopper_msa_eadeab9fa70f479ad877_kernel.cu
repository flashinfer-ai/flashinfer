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
#define SMEM_QCOL_STAGE_BYTES 128
#define SMEM_QCOL_STRIDE 128
#define SMEM_FLAGW_OFF 128
#define SMEM_FLAGW_STAGE_BYTES 16
#define SMEM_FLAGW_STRIDE 16
#define SMEM_CCNT_OFF 34960
#define SMEM_CCNT_STAGE_BYTES 128
#define SMEM_CCNT_STRIDE 128
#define SMEM_LIMC_OFF 35088
#define SMEM_LIMC_STAGE_BYTES 128
#define SMEM_LIMC_STRIDE 128
#define SMEM_CBUF_OFF 35216
#define SMEM_CBUF_STAGE_BYTES 2048
#define SMEM_CBUF_STRIDE 2048
#define SMEM_PUB_OFF 144
#define SMEM_PUB_STAGE_BYTES 34816
#define SMEM_PUB_STRIDE 34816
#define SMEM_TOTAL 37376
#define THREADS 512

extern "C" {

__global__ __launch_bounds__(512) void kernel_cake_hopper_msa_eadeab9fa70f479ad877(
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
  unsigned int* flagw = reinterpret_cast<unsigned int*>(smem_raw + 128);
  const int flagw_addr = smem + 128;
  unsigned int* ccnt = reinterpret_cast<unsigned int*>(smem_raw + 34960);
  const int ccnt_addr = smem + 34960;
  int* limc = reinterpret_cast<int*>(smem_raw + 35088);
  const int limc_addr = smem + 35088;
  unsigned int* cbuf = reinterpret_cast<unsigned int*>(smem_raw + 35216);
  const int cbuf_addr = smem + 35216;
  float* pub = reinterpret_cast<float*>(smem_raw + 144);
  const int pub_addr = smem + 144;

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
  unsigned int nb[16];
#pragma unroll 1
  for (int j = 0; j < num_chunks; j++) {
    int t0_1 = ((j + 1) * 16 + w) * 16;
    long long p_2 = cbase + (long long)t0_1 * nq64;
    nb[0] = 4286578688u;
    if (lim > t0_1) {
      nb[0] = S[p_2];
    }
    nb[1] = 4286578688u;
    if (lim > t0_1 + 1) {
      nb[1] = S[p_2 + nq64];
    }
    nb[2] = 4286578688u;
    if (lim > t0_1 + 2) {
      nb[2] = S[p_2 + 2 * nq64];
    }
    nb[3] = 4286578688u;
    if (lim > t0_1 + 3) {
      nb[3] = S[p_2 + 3 * nq64];
    }
    nb[4] = 4286578688u;
    if (lim > t0_1 + 4) {
      nb[4] = S[p_2 + 4 * nq64];
    }
    nb[5] = 4286578688u;
    if (lim > t0_1 + 5) {
      nb[5] = S[p_2 + 5 * nq64];
    }
    nb[6] = 4286578688u;
    if (lim > t0_1 + 6) {
      nb[6] = S[p_2 + 6 * nq64];
    }
    nb[7] = 4286578688u;
    if (lim > t0_1 + 7) {
      nb[7] = S[p_2 + 7 * nq64];
    }
    nb[8] = 4286578688u;
    if (lim > t0_1 + 8) {
      nb[8] = S[p_2 + 8 * nq64];
    }
    nb[9] = 4286578688u;
    if (lim > t0_1 + 9) {
      nb[9] = S[p_2 + 9 * nq64];
    }
    nb[10] = 4286578688u;
    if (lim > t0_1 + 10) {
      nb[10] = S[p_2 + 10 * nq64];
    }
    nb[11] = 4286578688u;
    if (lim > t0_1 + 11) {
      nb[11] = S[p_2 + 11 * nq64];
    }
    nb[12] = 4286578688u;
    if (lim > t0_1 + 12) {
      nb[12] = S[p_2 + 12 * nq64];
    }
    nb[13] = 4286578688u;
    if (lim > t0_1 + 13) {
      nb[13] = S[p_2 + 13 * nq64];
    }
    nb[14] = 4286578688u;
    if (lim > t0_1 + 14) {
      nb[14] = S[p_2 + 14 * nq64];
    }
    nb[15] = 4286578688u;
    if (lim > t0_1 + 15) {
      nb[15] = S[p_2 + 15 * nq64];
    }
    asm volatile("" ::: "memory");
    int t0_3 = (j * 16 + w) * 16;
    float sc = __uint_as_float(cb[0]);
    sc = fmaxf(sc, -1.7014118346046923e+38f);
    sc = fminf(sc, 1.7014118346046923e+38f);
    unsigned int key = __as_u32(sc) & 4294965248u | (unsigned int)t0_3;
    kb[0] = __uint_as_float(key);
    float sc_4 = __uint_as_float(cb[1]);
    sc_4 = fmaxf(sc_4, -1.7014118346046923e+38f);
    sc_4 = fminf(sc_4, 1.7014118346046923e+38f);
    unsigned int key_5 = __as_u32(sc_4) & 4294965248u | (unsigned int)(t0_3 + 1);
    kb[1] = __uint_as_float(key_5);
    float sc_6 = __uint_as_float(cb[2]);
    sc_6 = fmaxf(sc_6, -1.7014118346046923e+38f);
    sc_6 = fminf(sc_6, 1.7014118346046923e+38f);
    unsigned int key_7 = __as_u32(sc_6) & 4294965248u | (unsigned int)(t0_3 + 2);
    kb[2] = __uint_as_float(key_7);
    float sc_8 = __uint_as_float(cb[3]);
    sc_8 = fmaxf(sc_8, -1.7014118346046923e+38f);
    sc_8 = fminf(sc_8, 1.7014118346046923e+38f);
    unsigned int key_9 = __as_u32(sc_8) & 4294965248u | (unsigned int)(t0_3 + 3);
    kb[3] = __uint_as_float(key_9);
    float sc_10 = __uint_as_float(cb[4]);
    sc_10 = fmaxf(sc_10, -1.7014118346046923e+38f);
    sc_10 = fminf(sc_10, 1.7014118346046923e+38f);
    unsigned int key_11 = __as_u32(sc_10) & 4294965248u | (unsigned int)(t0_3 + 4);
    kb[4] = __uint_as_float(key_11);
    float sc_12 = __uint_as_float(cb[5]);
    sc_12 = fmaxf(sc_12, -1.7014118346046923e+38f);
    sc_12 = fminf(sc_12, 1.7014118346046923e+38f);
    unsigned int key_13 = __as_u32(sc_12) & 4294965248u | (unsigned int)(t0_3 + 5);
    kb[5] = __uint_as_float(key_13);
    float sc_14 = __uint_as_float(cb[6]);
    sc_14 = fmaxf(sc_14, -1.7014118346046923e+38f);
    sc_14 = fminf(sc_14, 1.7014118346046923e+38f);
    unsigned int key_15 = __as_u32(sc_14) & 4294965248u | (unsigned int)(t0_3 + 6);
    kb[6] = __uint_as_float(key_15);
    float sc_16 = __uint_as_float(cb[7]);
    sc_16 = fmaxf(sc_16, -1.7014118346046923e+38f);
    sc_16 = fminf(sc_16, 1.7014118346046923e+38f);
    unsigned int key_17 = __as_u32(sc_16) & 4294965248u | (unsigned int)(t0_3 + 7);
    kb[7] = __uint_as_float(key_17);
    float sc_18 = __uint_as_float(cb[8]);
    sc_18 = fmaxf(sc_18, -1.7014118346046923e+38f);
    sc_18 = fminf(sc_18, 1.7014118346046923e+38f);
    unsigned int key_19 = __as_u32(sc_18) & 4294965248u | (unsigned int)(t0_3 + 8);
    kb[8] = __uint_as_float(key_19);
    float sc_20 = __uint_as_float(cb[9]);
    sc_20 = fmaxf(sc_20, -1.7014118346046923e+38f);
    sc_20 = fminf(sc_20, 1.7014118346046923e+38f);
    unsigned int key_21 = __as_u32(sc_20) & 4294965248u | (unsigned int)(t0_3 + 9);
    kb[9] = __uint_as_float(key_21);
    float sc_22 = __uint_as_float(cb[10]);
    sc_22 = fmaxf(sc_22, -1.7014118346046923e+38f);
    sc_22 = fminf(sc_22, 1.7014118346046923e+38f);
    unsigned int key_23 = __as_u32(sc_22) & 4294965248u | (unsigned int)(t0_3 + 10);
    kb[10] = __uint_as_float(key_23);
    float sc_24 = __uint_as_float(cb[11]);
    sc_24 = fmaxf(sc_24, -1.7014118346046923e+38f);
    sc_24 = fminf(sc_24, 1.7014118346046923e+38f);
    unsigned int key_25 = __as_u32(sc_24) & 4294965248u | (unsigned int)(t0_3 + 11);
    kb[11] = __uint_as_float(key_25);
    float sc_26 = __uint_as_float(cb[12]);
    sc_26 = fmaxf(sc_26, -1.7014118346046923e+38f);
    sc_26 = fminf(sc_26, 1.7014118346046923e+38f);
    unsigned int key_27 = __as_u32(sc_26) & 4294965248u | (unsigned int)(t0_3 + 12);
    kb[12] = __uint_as_float(key_27);
    float sc_28 = __uint_as_float(cb[13]);
    sc_28 = fmaxf(sc_28, -1.7014118346046923e+38f);
    sc_28 = fminf(sc_28, 1.7014118346046923e+38f);
    unsigned int key_29 = __as_u32(sc_28) & 4294965248u | (unsigned int)(t0_3 + 13);
    kb[13] = __uint_as_float(key_29);
    float sc_30 = __uint_as_float(cb[14]);
    sc_30 = fmaxf(sc_30, -1.7014118346046923e+38f);
    sc_30 = fminf(sc_30, 1.7014118346046923e+38f);
    unsigned int key_31 = __as_u32(sc_30) & 4294965248u | (unsigned int)(t0_3 + 14);
    kb[14] = __uint_as_float(key_31);
    float sc_32 = __uint_as_float(cb[15]);
    sc_32 = fmaxf(sc_32, -1.7014118346046923e+38f);
    sc_32 = fminf(sc_32, 1.7014118346046923e+38f);
    unsigned int key_33 = __as_u32(sc_32) & 4294965248u | (unsigned int)(t0_3 + 15);
    kb[15] = __uint_as_float(key_33);
    int f = 0;
    if ((t0_3 < fb || t0_3 + 16 > lim - fe) && t0_3 < lim) {
      f = 1;
    }
    if (f != 0) {
      int f_0 = 0;
      if ((t0_3 < fb || t0_3 >= lim - fe) && lim > t0_3) {
        f_0 = 1;
      }
      if (f_0 != 0) {
        kb[0] = __uint_as_float(2139092992 | (unsigned int)t0_3);
      }
      int f_1 = 0;
      if ((t0_3 + 1 < fb || t0_3 + 1 >= lim - fe) && lim > t0_3 + 1) {
        f_1 = 1;
      }
      if (f_1 != 0) {
        kb[1] = __uint_as_float(2139092992 | (unsigned int)(t0_3 + 1));
      }
      int f_2 = 0;
      if ((t0_3 + 2 < fb || t0_3 + 2 >= lim - fe) && lim > t0_3 + 2) {
        f_2 = 1;
      }
      if (f_2 != 0) {
        kb[2] = __uint_as_float(2139092992 | (unsigned int)(t0_3 + 2));
      }
      int f_3 = 0;
      if ((t0_3 + 3 < fb || t0_3 + 3 >= lim - fe) && lim > t0_3 + 3) {
        f_3 = 1;
      }
      if (f_3 != 0) {
        kb[3] = __uint_as_float(2139092992 | (unsigned int)(t0_3 + 3));
      }
      int f_4 = 0;
      if ((t0_3 + 4 < fb || t0_3 + 4 >= lim - fe) && lim > t0_3 + 4) {
        f_4 = 1;
      }
      if (f_4 != 0) {
        kb[4] = __uint_as_float(2139092992 | (unsigned int)(t0_3 + 4));
      }
      int f_5 = 0;
      if ((t0_3 + 5 < fb || t0_3 + 5 >= lim - fe) && lim > t0_3 + 5) {
        f_5 = 1;
      }
      if (f_5 != 0) {
        kb[5] = __uint_as_float(2139092992 | (unsigned int)(t0_3 + 5));
      }
      int f_6 = 0;
      if ((t0_3 + 6 < fb || t0_3 + 6 >= lim - fe) && lim > t0_3 + 6) {
        f_6 = 1;
      }
      if (f_6 != 0) {
        kb[6] = __uint_as_float(2139092992 | (unsigned int)(t0_3 + 6));
      }
      int f_7 = 0;
      if ((t0_3 + 7 < fb || t0_3 + 7 >= lim - fe) && lim > t0_3 + 7) {
        f_7 = 1;
      }
      if (f_7 != 0) {
        kb[7] = __uint_as_float(2139092992 | (unsigned int)(t0_3 + 7));
      }
      int f_8 = 0;
      if ((t0_3 + 8 < fb || t0_3 + 8 >= lim - fe) && lim > t0_3 + 8) {
        f_8 = 1;
      }
      if (f_8 != 0) {
        kb[8] = __uint_as_float(2139092992 | (unsigned int)(t0_3 + 8));
      }
      int f_9 = 0;
      if ((t0_3 + 9 < fb || t0_3 + 9 >= lim - fe) && lim > t0_3 + 9) {
        f_9 = 1;
      }
      if (f_9 != 0) {
        kb[9] = __uint_as_float(2139092992 | (unsigned int)(t0_3 + 9));
      }
      int f_10 = 0;
      if ((t0_3 + 10 < fb || t0_3 + 10 >= lim - fe) && lim > t0_3 + 10) {
        f_10 = 1;
      }
      if (f_10 != 0) {
        kb[10] = __uint_as_float(2139092992 | (unsigned int)(t0_3 + 10));
      }
      int f_11 = 0;
      if ((t0_3 + 11 < fb || t0_3 + 11 >= lim - fe) && lim > t0_3 + 11) {
        f_11 = 1;
      }
      if (f_11 != 0) {
        kb[11] = __uint_as_float(2139092992 | (unsigned int)(t0_3 + 11));
      }
      int f_12 = 0;
      if ((t0_3 + 12 < fb || t0_3 + 12 >= lim - fe) && lim > t0_3 + 12) {
        f_12 = 1;
      }
      if (f_12 != 0) {
        kb[12] = __uint_as_float(2139092992 | (unsigned int)(t0_3 + 12));
      }
      int f_13 = 0;
      if ((t0_3 + 13 < fb || t0_3 + 13 >= lim - fe) && lim > t0_3 + 13) {
        f_13 = 1;
      }
      if (f_13 != 0) {
        kb[13] = __uint_as_float(2139092992 | (unsigned int)(t0_3 + 13));
      }
      int f_14 = 0;
      if ((t0_3 + 14 < fb || t0_3 + 14 >= lim - fe) && lim > t0_3 + 14) {
        f_14 = 1;
      }
      if (f_14 != 0) {
        kb[14] = __uint_as_float(2139092992 | (unsigned int)(t0_3 + 14));
      }
      int f_15 = 0;
      if ((t0_3 + 15 < fb || t0_3 + 15 >= lim - fe) && lim > t0_3 + 15) {
        f_15 = 1;
      }
      if (f_15 != 0) {
        kb[15] = __uint_as_float(2139092992 | (unsigned int)(t0_3 + 15));
      }
    }
    float hi = fmaxf(kb[0], kb[13]);
    float lo = fminf(kb[0], kb[13]);
    kb[0] = hi;
    kb[13] = lo;
    float hi_34 = fmaxf(kb[1], kb[12]);
    float lo_35 = fminf(kb[1], kb[12]);
    kb[1] = hi_34;
    kb[12] = lo_35;
    float hi_36 = fmaxf(kb[2], kb[15]);
    float lo_37 = fminf(kb[2], kb[15]);
    kb[2] = hi_36;
    kb[15] = lo_37;
    float hi_38 = fmaxf(kb[3], kb[14]);
    float lo_39 = fminf(kb[3], kb[14]);
    kb[3] = hi_38;
    kb[14] = lo_39;
    float hi_40 = fmaxf(kb[4], kb[8]);
    float lo_41 = fminf(kb[4], kb[8]);
    kb[4] = hi_40;
    kb[8] = lo_41;
    float hi_42 = fmaxf(kb[5], kb[6]);
    float lo_43 = fminf(kb[5], kb[6]);
    kb[5] = hi_42;
    kb[6] = lo_43;
    float hi_44 = fmaxf(kb[7], kb[11]);
    float lo_45 = fminf(kb[7], kb[11]);
    kb[7] = hi_44;
    kb[11] = lo_45;
    float hi_46 = fmaxf(kb[9], kb[10]);
    float lo_47 = fminf(kb[9], kb[10]);
    kb[9] = hi_46;
    kb[10] = lo_47;
    float hi_48 = fmaxf(kb[0], kb[5]);
    float lo_49 = fminf(kb[0], kb[5]);
    kb[0] = hi_48;
    kb[5] = lo_49;
    float hi_50 = fmaxf(kb[1], kb[7]);
    float lo_51 = fminf(kb[1], kb[7]);
    kb[1] = hi_50;
    kb[7] = lo_51;
    float hi_52 = fmaxf(kb[2], kb[9]);
    float lo_53 = fminf(kb[2], kb[9]);
    kb[2] = hi_52;
    kb[9] = lo_53;
    float hi_54 = fmaxf(kb[3], kb[4]);
    float lo_55 = fminf(kb[3], kb[4]);
    kb[3] = hi_54;
    kb[4] = lo_55;
    float hi_56 = fmaxf(kb[6], kb[13]);
    float lo_57 = fminf(kb[6], kb[13]);
    kb[6] = hi_56;
    kb[13] = lo_57;
    float hi_58 = fmaxf(kb[8], kb[14]);
    float lo_59 = fminf(kb[8], kb[14]);
    kb[8] = hi_58;
    kb[14] = lo_59;
    float hi_60 = fmaxf(kb[10], kb[15]);
    float lo_61 = fminf(kb[10], kb[15]);
    kb[10] = hi_60;
    kb[15] = lo_61;
    float hi_62 = fmaxf(kb[11], kb[12]);
    float lo_63 = fminf(kb[11], kb[12]);
    kb[11] = hi_62;
    kb[12] = lo_63;
    float hi_64 = fmaxf(kb[0], kb[1]);
    float lo_65 = fminf(kb[0], kb[1]);
    kb[0] = hi_64;
    kb[1] = lo_65;
    float hi_66 = fmaxf(kb[2], kb[3]);
    float lo_67 = fminf(kb[2], kb[3]);
    kb[2] = hi_66;
    kb[3] = lo_67;
    float hi_68 = fmaxf(kb[4], kb[5]);
    float lo_69 = fminf(kb[4], kb[5]);
    kb[4] = hi_68;
    kb[5] = lo_69;
    float hi_70 = fmaxf(kb[6], kb[8]);
    float lo_71 = fminf(kb[6], kb[8]);
    kb[6] = hi_70;
    kb[8] = lo_71;
    float hi_72 = fmaxf(kb[7], kb[9]);
    float lo_73 = fminf(kb[7], kb[9]);
    kb[7] = hi_72;
    kb[9] = lo_73;
    float hi_74 = fmaxf(kb[10], kb[11]);
    float lo_75 = fminf(kb[10], kb[11]);
    kb[10] = hi_74;
    kb[11] = lo_75;
    float hi_76 = fmaxf(kb[12], kb[13]);
    float lo_77 = fminf(kb[12], kb[13]);
    kb[12] = hi_76;
    kb[13] = lo_77;
    float hi_78 = fmaxf(kb[14], kb[15]);
    float lo_79 = fminf(kb[14], kb[15]);
    kb[14] = hi_78;
    kb[15] = lo_79;
    float hi_80 = fmaxf(kb[0], kb[2]);
    float lo_81 = fminf(kb[0], kb[2]);
    kb[0] = hi_80;
    kb[2] = lo_81;
    float hi_82 = fmaxf(kb[1], kb[3]);
    float lo_83 = fminf(kb[1], kb[3]);
    kb[1] = hi_82;
    kb[3] = lo_83;
    float hi_84 = fmaxf(kb[4], kb[10]);
    float lo_85 = fminf(kb[4], kb[10]);
    kb[4] = hi_84;
    kb[10] = lo_85;
    float hi_86 = fmaxf(kb[5], kb[11]);
    float lo_87 = fminf(kb[5], kb[11]);
    kb[5] = hi_86;
    kb[11] = lo_87;
    float hi_88 = fmaxf(kb[6], kb[7]);
    float lo_89 = fminf(kb[6], kb[7]);
    kb[6] = hi_88;
    kb[7] = lo_89;
    float hi_90 = fmaxf(kb[8], kb[9]);
    float lo_91 = fminf(kb[8], kb[9]);
    kb[8] = hi_90;
    kb[9] = lo_91;
    float hi_92 = fmaxf(kb[12], kb[14]);
    float lo_93 = fminf(kb[12], kb[14]);
    kb[12] = hi_92;
    kb[14] = lo_93;
    float hi_94 = fmaxf(kb[13], kb[15]);
    float lo_95 = fminf(kb[13], kb[15]);
    kb[13] = hi_94;
    kb[15] = lo_95;
    float hi_96 = fmaxf(kb[1], kb[2]);
    float lo_97 = fminf(kb[1], kb[2]);
    kb[1] = hi_96;
    kb[2] = lo_97;
    float hi_98 = fmaxf(kb[3], kb[12]);
    float lo_99 = fminf(kb[3], kb[12]);
    kb[3] = hi_98;
    kb[12] = lo_99;
    float hi_100 = fmaxf(kb[4], kb[6]);
    float lo_101 = fminf(kb[4], kb[6]);
    kb[4] = hi_100;
    kb[6] = lo_101;
    float hi_102 = fmaxf(kb[5], kb[7]);
    float lo_103 = fminf(kb[5], kb[7]);
    kb[5] = hi_102;
    kb[7] = lo_103;
    float hi_104 = fmaxf(kb[8], kb[10]);
    float lo_105 = fminf(kb[8], kb[10]);
    kb[8] = hi_104;
    kb[10] = lo_105;
    float hi_106 = fmaxf(kb[9], kb[11]);
    float lo_107 = fminf(kb[9], kb[11]);
    kb[9] = hi_106;
    kb[11] = lo_107;
    float hi_108 = fmaxf(kb[13], kb[14]);
    float lo_109 = fminf(kb[13], kb[14]);
    kb[13] = hi_108;
    kb[14] = lo_109;
    float hi_110 = fmaxf(kb[1], kb[4]);
    float lo_111 = fminf(kb[1], kb[4]);
    kb[1] = hi_110;
    kb[4] = lo_111;
    float hi_112 = fmaxf(kb[2], kb[6]);
    float lo_113 = fminf(kb[2], kb[6]);
    kb[2] = hi_112;
    kb[6] = lo_113;
    float hi_114 = fmaxf(kb[5], kb[8]);
    float lo_115 = fminf(kb[5], kb[8]);
    kb[5] = hi_114;
    kb[8] = lo_115;
    float hi_116 = fmaxf(kb[7], kb[10]);
    float lo_117 = fminf(kb[7], kb[10]);
    kb[7] = hi_116;
    kb[10] = lo_117;
    float hi_118 = fmaxf(kb[9], kb[13]);
    float lo_119 = fminf(kb[9], kb[13]);
    kb[9] = hi_118;
    kb[13] = lo_119;
    float hi_120 = fmaxf(kb[11], kb[14]);
    float lo_121 = fminf(kb[11], kb[14]);
    kb[11] = hi_120;
    kb[14] = lo_121;
    float hi_122 = fmaxf(kb[2], kb[4]);
    float lo_123 = fminf(kb[2], kb[4]);
    kb[2] = hi_122;
    kb[4] = lo_123;
    float hi_124 = fmaxf(kb[3], kb[6]);
    float lo_125 = fminf(kb[3], kb[6]);
    kb[3] = hi_124;
    kb[6] = lo_125;
    float hi_126 = fmaxf(kb[9], kb[12]);
    float lo_127 = fminf(kb[9], kb[12]);
    kb[9] = hi_126;
    kb[12] = lo_127;
    float hi_128 = fmaxf(kb[11], kb[13]);
    float lo_129 = fminf(kb[11], kb[13]);
    kb[11] = hi_128;
    kb[13] = lo_129;
    float hi_130 = fmaxf(kb[3], kb[5]);
    float lo_131 = fminf(kb[3], kb[5]);
    kb[3] = hi_130;
    kb[5] = lo_131;
    float hi_132 = fmaxf(kb[6], kb[8]);
    float lo_133 = fminf(kb[6], kb[8]);
    kb[6] = hi_132;
    kb[8] = lo_133;
    float hi_134 = fmaxf(kb[7], kb[9]);
    float lo_135 = fminf(kb[7], kb[9]);
    kb[7] = hi_134;
    kb[9] = lo_135;
    float hi_136 = fmaxf(kb[10], kb[12]);
    float lo_137 = fminf(kb[10], kb[12]);
    kb[10] = hi_136;
    kb[12] = lo_137;
    float hi_138 = fmaxf(kb[3], kb[4]);
    float lo_139 = fminf(kb[3], kb[4]);
    kb[3] = hi_138;
    kb[4] = lo_139;
    float hi_140 = fmaxf(kb[5], kb[6]);
    float lo_141 = fminf(kb[5], kb[6]);
    kb[5] = hi_140;
    kb[6] = lo_141;
    float hi_142 = fmaxf(kb[7], kb[8]);
    float lo_143 = fminf(kb[7], kb[8]);
    kb[7] = hi_142;
    kb[8] = lo_143;
    float hi_144 = fmaxf(kb[9], kb[10]);
    float lo_145 = fminf(kb[9], kb[10]);
    kb[9] = hi_144;
    kb[10] = lo_145;
    float hi_146 = fmaxf(kb[11], kb[12]);
    float lo_147 = fminf(kb[11], kb[12]);
    kb[11] = hi_146;
    kb[12] = lo_147;
    float hi_148 = fmaxf(kb[6], kb[7]);
    float lo_149 = fminf(kb[6], kb[7]);
    kb[6] = hi_148;
    kb[7] = lo_149;
    float hi_150 = fmaxf(kb[8], kb[9]);
    float lo_151 = fminf(kb[8], kb[9]);
    kb[8] = hi_150;
    kb[9] = lo_151;
    float r = rej;
    float hi_152 = fmaxf(a[0], kb[15]);
    float lo_153 = fminf(a[0], kb[15]);
    a[0] = hi_152;
    r = fmaxf(r, lo_153);
    float hi_154 = fmaxf(a[1], kb[14]);
    float lo_155 = fminf(a[1], kb[14]);
    a[1] = hi_154;
    r = fmaxf(r, lo_155);
    float hi_156 = fmaxf(a[2], kb[13]);
    float lo_157 = fminf(a[2], kb[13]);
    a[2] = hi_156;
    r = fmaxf(r, lo_157);
    float hi_158 = fmaxf(a[3], kb[12]);
    float lo_159 = fminf(a[3], kb[12]);
    a[3] = hi_158;
    r = fmaxf(r, lo_159);
    float hi_160 = fmaxf(a[4], kb[11]);
    float lo_161 = fminf(a[4], kb[11]);
    a[4] = hi_160;
    r = fmaxf(r, lo_161);
    float hi_162 = fmaxf(a[5], kb[10]);
    float lo_163 = fminf(a[5], kb[10]);
    a[5] = hi_162;
    r = fmaxf(r, lo_163);
    float hi_164 = fmaxf(a[6], kb[9]);
    float lo_165 = fminf(a[6], kb[9]);
    a[6] = hi_164;
    r = fmaxf(r, lo_165);
    float hi_166 = fmaxf(a[7], kb[8]);
    float lo_167 = fminf(a[7], kb[8]);
    a[7] = hi_166;
    r = fmaxf(r, lo_167);
    float hi_168 = fmaxf(a[8], kb[7]);
    float lo_169 = fminf(a[8], kb[7]);
    a[8] = hi_168;
    r = fmaxf(r, lo_169);
    float hi_170 = fmaxf(a[9], kb[6]);
    float lo_171 = fminf(a[9], kb[6]);
    a[9] = hi_170;
    r = fmaxf(r, lo_171);
    float hi_172 = fmaxf(a[10], kb[5]);
    float lo_173 = fminf(a[10], kb[5]);
    a[10] = hi_172;
    r = fmaxf(r, lo_173);
    float hi_174 = fmaxf(a[11], kb[4]);
    float lo_175 = fminf(a[11], kb[4]);
    a[11] = hi_174;
    r = fmaxf(r, lo_175);
    float hi_176 = fmaxf(a[12], kb[3]);
    float lo_177 = fminf(a[12], kb[3]);
    a[12] = hi_176;
    r = fmaxf(r, lo_177);
    float hi_178 = fmaxf(a[13], kb[2]);
    float lo_179 = fminf(a[13], kb[2]);
    a[13] = hi_178;
    r = fmaxf(r, lo_179);
    float hi_180 = fmaxf(a[14], kb[1]);
    float lo_181 = fminf(a[14], kb[1]);
    a[14] = hi_180;
    r = fmaxf(r, lo_181);
    float hi_182 = fmaxf(a[15], kb[0]);
    float lo_183 = fminf(a[15], kb[0]);
    a[15] = hi_182;
    r = fmaxf(r, lo_183);
    float hi_184 = fmaxf(a[0], a[8]);
    float lo_185 = fminf(a[0], a[8]);
    a[0] = hi_184;
    a[8] = lo_185;
    float hi_186 = fmaxf(a[1], a[9]);
    float lo_187 = fminf(a[1], a[9]);
    a[1] = hi_186;
    a[9] = lo_187;
    float hi_188 = fmaxf(a[2], a[10]);
    float lo_189 = fminf(a[2], a[10]);
    a[2] = hi_188;
    a[10] = lo_189;
    float hi_190 = fmaxf(a[3], a[11]);
    float lo_191 = fminf(a[3], a[11]);
    a[3] = hi_190;
    a[11] = lo_191;
    float hi_192 = fmaxf(a[4], a[12]);
    float lo_193 = fminf(a[4], a[12]);
    a[4] = hi_192;
    a[12] = lo_193;
    float hi_194 = fmaxf(a[5], a[13]);
    float lo_195 = fminf(a[5], a[13]);
    a[5] = hi_194;
    a[13] = lo_195;
    float hi_196 = fmaxf(a[6], a[14]);
    float lo_197 = fminf(a[6], a[14]);
    a[6] = hi_196;
    a[14] = lo_197;
    float hi_198 = fmaxf(a[7], a[15]);
    float lo_199 = fminf(a[7], a[15]);
    a[7] = hi_198;
    a[15] = lo_199;
    float hi_200 = fmaxf(a[0], a[4]);
    float lo_201 = fminf(a[0], a[4]);
    a[0] = hi_200;
    a[4] = lo_201;
    float hi_202 = fmaxf(a[1], a[5]);
    float lo_203 = fminf(a[1], a[5]);
    a[1] = hi_202;
    a[5] = lo_203;
    float hi_204 = fmaxf(a[2], a[6]);
    float lo_205 = fminf(a[2], a[6]);
    a[2] = hi_204;
    a[6] = lo_205;
    float hi_206 = fmaxf(a[3], a[7]);
    float lo_207 = fminf(a[3], a[7]);
    a[3] = hi_206;
    a[7] = lo_207;
    float hi_208 = fmaxf(a[8], a[12]);
    float lo_209 = fminf(a[8], a[12]);
    a[8] = hi_208;
    a[12] = lo_209;
    float hi_210 = fmaxf(a[9], a[13]);
    float lo_211 = fminf(a[9], a[13]);
    a[9] = hi_210;
    a[13] = lo_211;
    float hi_212 = fmaxf(a[10], a[14]);
    float lo_213 = fminf(a[10], a[14]);
    a[10] = hi_212;
    a[14] = lo_213;
    float hi_214 = fmaxf(a[11], a[15]);
    float lo_215 = fminf(a[11], a[15]);
    a[11] = hi_214;
    a[15] = lo_215;
    float hi_216 = fmaxf(a[0], a[2]);
    float lo_217 = fminf(a[0], a[2]);
    a[0] = hi_216;
    a[2] = lo_217;
    float hi_218 = fmaxf(a[1], a[3]);
    float lo_219 = fminf(a[1], a[3]);
    a[1] = hi_218;
    a[3] = lo_219;
    float hi_220 = fmaxf(a[4], a[6]);
    float lo_221 = fminf(a[4], a[6]);
    a[4] = hi_220;
    a[6] = lo_221;
    float hi_222 = fmaxf(a[5], a[7]);
    float lo_223 = fminf(a[5], a[7]);
    a[5] = hi_222;
    a[7] = lo_223;
    float hi_224 = fmaxf(a[8], a[10]);
    float lo_225 = fminf(a[8], a[10]);
    a[8] = hi_224;
    a[10] = lo_225;
    float hi_226 = fmaxf(a[9], a[11]);
    float lo_227 = fminf(a[9], a[11]);
    a[9] = hi_226;
    a[11] = lo_227;
    float hi_228 = fmaxf(a[12], a[14]);
    float lo_229 = fminf(a[12], a[14]);
    a[12] = hi_228;
    a[14] = lo_229;
    float hi_230 = fmaxf(a[13], a[15]);
    float lo_231 = fminf(a[13], a[15]);
    a[13] = hi_230;
    a[15] = lo_231;
    float hi_232 = fmaxf(a[0], a[1]);
    float lo_233 = fminf(a[0], a[1]);
    a[0] = hi_232;
    a[1] = lo_233;
    float hi_234 = fmaxf(a[2], a[3]);
    float lo_235 = fminf(a[2], a[3]);
    a[2] = hi_234;
    a[3] = lo_235;
    float hi_236 = fmaxf(a[4], a[5]);
    float lo_237 = fminf(a[4], a[5]);
    a[4] = hi_236;
    a[5] = lo_237;
    float hi_238 = fmaxf(a[6], a[7]);
    float lo_239 = fminf(a[6], a[7]);
    a[6] = hi_238;
    a[7] = lo_239;
    float hi_240 = fmaxf(a[8], a[9]);
    float lo_241 = fminf(a[8], a[9]);
    a[8] = hi_240;
    a[9] = lo_241;
    float hi_242 = fmaxf(a[10], a[11]);
    float lo_243 = fminf(a[10], a[11]);
    a[10] = hi_242;
    a[11] = lo_243;
    float hi_244 = fmaxf(a[12], a[13]);
    float lo_245 = fminf(a[12], a[13]);
    a[12] = hi_244;
    a[13] = lo_245;
    float hi_246 = fmaxf(a[14], a[15]);
    float lo_247 = fminf(a[14], a[15]);
    a[14] = hi_246;
    a[15] = lo_247;
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
  int cg = g & 31;
  int sg = g >> 5;
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
  asm volatile("barrier.sync 1, 512;" ::: "memory");
  float r_1 = neg_inf;
  float V[8];
  int s0 = (sg * 16 * 32 + cg) * 17;
  int s1 = ((sg * 16 + 8) * 32 + cg) * 17;
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
  float pv_1 = __shfl_xor_sync(0xFFFFFFFF, cur, 4);
  float hi_2 = fmaxf(cur, pv_1);
  float lo_3 = fminf(cur, pv_1);
  cur = ((up[1] != 0) ? hi_2 : lo_3);
  float pv_4 = __shfl_xor_sync(0xFFFFFFFF, cur, 2);
  float hi_5 = fmaxf(cur, pv_4);
  float lo_6 = fminf(cur, pv_4);
  cur = ((up[2] != 0) ? hi_5 : lo_6);
  float pv_7 = __shfl_xor_sync(0xFFFFFFFF, cur, 1);
  float hi_8 = fmaxf(cur, pv_7);
  float lo_9 = fminf(cur, pv_7);
  cur = ((up[3] != 0) ? hi_8 : lo_9);
  V[0] = cur;
  int s0_10 = ((sg * 16 + 1) * 32 + cg) * 17;
  int s1_11 = ((sg * 16 + 1 + 8) * 32 + cg) * 17;
  float x0_12 = pub[s0_10 + ln];
  float y0_13 = pub[s1_11 + lnr];
  float lo0_14 = fminf(x0_12, y0_13);
  r_1 = fmaxf(r_1, lo0_14);
  float hi0_15 = fmaxf(x0_12, y0_13);
  float cur_16 = hi0_15;
  float pv_17 = __shfl_xor_sync(0xFFFFFFFF, cur_16, 8);
  float hi_18 = fmaxf(cur_16, pv_17);
  float lo_19 = fminf(cur_16, pv_17);
  cur_16 = ((up[0] != 0) ? hi_18 : lo_19);
  float pv_20 = __shfl_xor_sync(0xFFFFFFFF, cur_16, 4);
  float hi_21 = fmaxf(cur_16, pv_20);
  float lo_22 = fminf(cur_16, pv_20);
  cur_16 = ((up[1] != 0) ? hi_21 : lo_22);
  float pv_23 = __shfl_xor_sync(0xFFFFFFFF, cur_16, 2);
  float hi_24 = fmaxf(cur_16, pv_23);
  float lo_25 = fminf(cur_16, pv_23);
  cur_16 = ((up[2] != 0) ? hi_24 : lo_25);
  float pv_26 = __shfl_xor_sync(0xFFFFFFFF, cur_16, 1);
  float hi_27 = fmaxf(cur_16, pv_26);
  float lo_28 = fminf(cur_16, pv_26);
  cur_16 = ((up[3] != 0) ? hi_27 : lo_28);
  V[1] = cur_16;
  int s0_29 = ((sg * 16 + 2) * 32 + cg) * 17;
  int s1_30 = ((sg * 16 + 2 + 8) * 32 + cg) * 17;
  float x0_31 = pub[s0_29 + ln];
  float y0_32 = pub[s1_30 + lnr];
  float lo0_33 = fminf(x0_31, y0_32);
  r_1 = fmaxf(r_1, lo0_33);
  float hi0_34 = fmaxf(x0_31, y0_32);
  float cur_35 = hi0_34;
  float pv_36 = __shfl_xor_sync(0xFFFFFFFF, cur_35, 8);
  float hi_37 = fmaxf(cur_35, pv_36);
  float lo_38 = fminf(cur_35, pv_36);
  cur_35 = ((up[0] != 0) ? hi_37 : lo_38);
  float pv_39 = __shfl_xor_sync(0xFFFFFFFF, cur_35, 4);
  float hi_40_1 = fmaxf(cur_35, pv_39);
  float lo_41_1 = fminf(cur_35, pv_39);
  cur_35 = ((up[1] != 0) ? hi_40_1 : lo_41_1);
  float pv_42 = __shfl_xor_sync(0xFFFFFFFF, cur_35, 2);
  float hi_43 = fmaxf(cur_35, pv_42);
  float lo_44 = fminf(cur_35, pv_42);
  cur_35 = ((up[2] != 0) ? hi_43 : lo_44);
  float pv_45 = __shfl_xor_sync(0xFFFFFFFF, cur_35, 1);
  float hi_46_1 = fmaxf(cur_35, pv_45);
  float lo_47_1 = fminf(cur_35, pv_45);
  cur_35 = ((up[3] != 0) ? hi_46_1 : lo_47_1);
  V[2] = cur_35;
  int s0_48 = ((sg * 16 + 3) * 32 + cg) * 17;
  int s1_49 = ((sg * 16 + 3 + 8) * 32 + cg) * 17;
  float x0_50 = pub[s0_48 + ln];
  float y0_51 = pub[s1_49 + lnr];
  float lo0_52 = fminf(x0_50, y0_51);
  r_1 = fmaxf(r_1, lo0_52);
  float hi0_53 = fmaxf(x0_50, y0_51);
  float cur_54 = hi0_53;
  float pv_55 = __shfl_xor_sync(0xFFFFFFFF, cur_54, 8);
  float hi_56_1 = fmaxf(cur_54, pv_55);
  float lo_57_1 = fminf(cur_54, pv_55);
  cur_54 = ((up[0] != 0) ? hi_56_1 : lo_57_1);
  float pv_58 = __shfl_xor_sync(0xFFFFFFFF, cur_54, 4);
  float hi_59 = fmaxf(cur_54, pv_58);
  float lo_60 = fminf(cur_54, pv_58);
  cur_54 = ((up[1] != 0) ? hi_59 : lo_60);
  float pv_61 = __shfl_xor_sync(0xFFFFFFFF, cur_54, 2);
  float hi_62_1 = fmaxf(cur_54, pv_61);
  float lo_63_1 = fminf(cur_54, pv_61);
  cur_54 = ((up[2] != 0) ? hi_62_1 : lo_63_1);
  float pv_64 = __shfl_xor_sync(0xFFFFFFFF, cur_54, 1);
  float hi_65 = fmaxf(cur_54, pv_64);
  float lo_66 = fminf(cur_54, pv_64);
  cur_54 = ((up[3] != 0) ? hi_65 : lo_66);
  V[3] = cur_54;
  int s0_67 = ((sg * 16 + 4) * 32 + cg) * 17;
  int s1_68 = ((sg * 16 + 4 + 8) * 32 + cg) * 17;
  float x0_69 = pub[s0_67 + ln];
  float y0_70 = pub[s1_68 + lnr];
  float lo0_71 = fminf(x0_69, y0_70);
  r_1 = fmaxf(r_1, lo0_71);
  float hi0_72 = fmaxf(x0_69, y0_70);
  float cur_73 = hi0_72;
  float pv_74 = __shfl_xor_sync(0xFFFFFFFF, cur_73, 8);
  float hi_75 = fmaxf(cur_73, pv_74);
  float lo_76 = fminf(cur_73, pv_74);
  cur_73 = ((up[0] != 0) ? hi_75 : lo_76);
  float pv_77 = __shfl_xor_sync(0xFFFFFFFF, cur_73, 4);
  float hi_78_1 = fmaxf(cur_73, pv_77);
  float lo_79_1 = fminf(cur_73, pv_77);
  cur_73 = ((up[1] != 0) ? hi_78_1 : lo_79_1);
  float pv_80 = __shfl_xor_sync(0xFFFFFFFF, cur_73, 2);
  float hi_81 = fmaxf(cur_73, pv_80);
  float lo_82 = fminf(cur_73, pv_80);
  cur_73 = ((up[2] != 0) ? hi_81 : lo_82);
  float pv_83 = __shfl_xor_sync(0xFFFFFFFF, cur_73, 1);
  float hi_84_1 = fmaxf(cur_73, pv_83);
  float lo_85_1 = fminf(cur_73, pv_83);
  cur_73 = ((up[3] != 0) ? hi_84_1 : lo_85_1);
  V[4] = cur_73;
  int s0_86 = ((sg * 16 + 5) * 32 + cg) * 17;
  int s1_87 = ((sg * 16 + 5 + 8) * 32 + cg) * 17;
  float x0_88 = pub[s0_86 + ln];
  float y0_89 = pub[s1_87 + lnr];
  float lo0_90 = fminf(x0_88, y0_89);
  r_1 = fmaxf(r_1, lo0_90);
  float hi0_91 = fmaxf(x0_88, y0_89);
  float cur_92 = hi0_91;
  float pv_93 = __shfl_xor_sync(0xFFFFFFFF, cur_92, 8);
  float hi_94_1 = fmaxf(cur_92, pv_93);
  float lo_95_1 = fminf(cur_92, pv_93);
  cur_92 = ((up[0] != 0) ? hi_94_1 : lo_95_1);
  float pv_96 = __shfl_xor_sync(0xFFFFFFFF, cur_92, 4);
  float hi_97 = fmaxf(cur_92, pv_96);
  float lo_98 = fminf(cur_92, pv_96);
  cur_92 = ((up[1] != 0) ? hi_97 : lo_98);
  float pv_99 = __shfl_xor_sync(0xFFFFFFFF, cur_92, 2);
  float hi_100_1 = fmaxf(cur_92, pv_99);
  float lo_101_1 = fminf(cur_92, pv_99);
  cur_92 = ((up[2] != 0) ? hi_100_1 : lo_101_1);
  float pv_102 = __shfl_xor_sync(0xFFFFFFFF, cur_92, 1);
  float hi_103 = fmaxf(cur_92, pv_102);
  float lo_104 = fminf(cur_92, pv_102);
  cur_92 = ((up[3] != 0) ? hi_103 : lo_104);
  V[5] = cur_92;
  int s0_105 = ((sg * 16 + 6) * 32 + cg) * 17;
  int s1_106 = ((sg * 16 + 6 + 8) * 32 + cg) * 17;
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
  float hi_116_1 = fmaxf(cur_111, pv_115);
  float lo_117_1 = fminf(cur_111, pv_115);
  cur_111 = ((up[1] != 0) ? hi_116_1 : lo_117_1);
  float pv_118 = __shfl_xor_sync(0xFFFFFFFF, cur_111, 2);
  float hi_119 = fmaxf(cur_111, pv_118);
  float lo_120 = fminf(cur_111, pv_118);
  cur_111 = ((up[2] != 0) ? hi_119 : lo_120);
  float pv_121 = __shfl_xor_sync(0xFFFFFFFF, cur_111, 1);
  float hi_122_1 = fmaxf(cur_111, pv_121);
  float lo_123_1 = fminf(cur_111, pv_121);
  cur_111 = ((up[3] != 0) ? hi_122_1 : lo_123_1);
  V[6] = cur_111;
  int s0_124 = ((sg * 16 + 7) * 32 + cg) * 17;
  int s1_125 = ((sg * 16 + 7 + 8) * 32 + cg) * 17;
  float x0_126 = pub[s0_124 + ln];
  float y0_127 = pub[s1_125 + lnr];
  float lo0_128 = fminf(x0_126, y0_127);
  r_1 = fmaxf(r_1, lo0_128);
  float hi0_129 = fmaxf(x0_126, y0_127);
  float cur_130 = hi0_129;
  float pv_131 = __shfl_xor_sync(0xFFFFFFFF, cur_130, 8);
  float hi_132_1 = fmaxf(cur_130, pv_131);
  float lo_133_1 = fminf(cur_130, pv_131);
  cur_130 = ((up[0] != 0) ? hi_132_1 : lo_133_1);
  float pv_134 = __shfl_xor_sync(0xFFFFFFFF, cur_130, 4);
  float hi_135 = fmaxf(cur_130, pv_134);
  float lo_136 = fminf(cur_130, pv_134);
  cur_130 = ((up[1] != 0) ? hi_135 : lo_136);
  float pv_137 = __shfl_xor_sync(0xFFFFFFFF, cur_130, 2);
  float hi_138_1 = fmaxf(cur_130, pv_137);
  float lo_139_1 = fminf(cur_130, pv_137);
  cur_130 = ((up[2] != 0) ? hi_138_1 : lo_139_1);
  float pv_140 = __shfl_xor_sync(0xFFFFFFFF, cur_130, 1);
  float hi_141 = fmaxf(cur_130, pv_140);
  float lo_142 = fminf(cur_130, pv_140);
  cur_130 = ((up[3] != 0) ? hi_141 : lo_142);
  V[7] = cur_130;
  float rs = pub[((sg * 16 + ln) * 32 + cg) * 17 + 16];
  r_1 = fmaxf(r_1, rs);
  float y1 = __shfl_xor_sync(0xFFFFFFFF, V[4], 15);
  float lo1 = fminf(V[0], y1);
  r_1 = fmaxf(r_1, lo1);
  float hi1 = fmaxf(V[0], y1);
  float cur_143 = hi1;
  float pv_144 = __shfl_xor_sync(0xFFFFFFFF, cur_143, 8);
  float hi_145 = fmaxf(cur_143, pv_144);
  float lo_146 = fminf(cur_143, pv_144);
  cur_143 = ((up[0] != 0) ? hi_145 : lo_146);
  float pv_147 = __shfl_xor_sync(0xFFFFFFFF, cur_143, 4);
  float hi_148_1 = fmaxf(cur_143, pv_147);
  float lo_149_1 = fminf(cur_143, pv_147);
  cur_143 = ((up[1] != 0) ? hi_148_1 : lo_149_1);
  float pv_150 = __shfl_xor_sync(0xFFFFFFFF, cur_143, 2);
  float hi_151 = fmaxf(cur_143, pv_150);
  float lo_152 = fminf(cur_143, pv_150);
  cur_143 = ((up[2] != 0) ? hi_151 : lo_152);
  float pv_153 = __shfl_xor_sync(0xFFFFFFFF, cur_143, 1);
  float hi_154_1 = fmaxf(cur_143, pv_153);
  float lo_155_1 = fminf(cur_143, pv_153);
  cur_143 = ((up[3] != 0) ? hi_154_1 : lo_155_1);
  V[0] = cur_143;
  float y1_156 = __shfl_xor_sync(0xFFFFFFFF, V[5], 15);
  float lo1_157 = fminf(V[1], y1_156);
  r_1 = fmaxf(r_1, lo1_157);
  float hi1_158 = fmaxf(V[1], y1_156);
  float cur_159 = hi1_158;
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
  V[1] = cur_159;
  float y1_172 = __shfl_xor_sync(0xFFFFFFFF, V[6], 15);
  float lo1_173 = fminf(V[2], y1_172);
  r_1 = fmaxf(r_1, lo1_173);
  float hi1_174 = fmaxf(V[2], y1_172);
  float cur_175 = hi1_174;
  float pv_176 = __shfl_xor_sync(0xFFFFFFFF, cur_175, 8);
  float hi_177 = fmaxf(cur_175, pv_176);
  float lo_178 = fminf(cur_175, pv_176);
  cur_175 = ((up[0] != 0) ? hi_177 : lo_178);
  float pv_179 = __shfl_xor_sync(0xFFFFFFFF, cur_175, 4);
  float hi_180_1 = fmaxf(cur_175, pv_179);
  float lo_181_1 = fminf(cur_175, pv_179);
  cur_175 = ((up[1] != 0) ? hi_180_1 : lo_181_1);
  float pv_182 = __shfl_xor_sync(0xFFFFFFFF, cur_175, 2);
  float hi_183 = fmaxf(cur_175, pv_182);
  float lo_184 = fminf(cur_175, pv_182);
  cur_175 = ((up[2] != 0) ? hi_183 : lo_184);
  float pv_185 = __shfl_xor_sync(0xFFFFFFFF, cur_175, 1);
  float hi_186_1 = fmaxf(cur_175, pv_185);
  float lo_187_1 = fminf(cur_175, pv_185);
  cur_175 = ((up[3] != 0) ? hi_186_1 : lo_187_1);
  V[2] = cur_175;
  float y1_188 = __shfl_xor_sync(0xFFFFFFFF, V[7], 15);
  float lo1_189 = fminf(V[3], y1_188);
  r_1 = fmaxf(r_1, lo1_189);
  float hi1_190 = fmaxf(V[3], y1_188);
  float cur_191 = hi1_190;
  float pv_192 = __shfl_xor_sync(0xFFFFFFFF, cur_191, 8);
  float hi_193 = fmaxf(cur_191, pv_192);
  float lo_194 = fminf(cur_191, pv_192);
  cur_191 = ((up[0] != 0) ? hi_193 : lo_194);
  float pv_195 = __shfl_xor_sync(0xFFFFFFFF, cur_191, 4);
  float hi_196_1 = fmaxf(cur_191, pv_195);
  float lo_197_1 = fminf(cur_191, pv_195);
  cur_191 = ((up[1] != 0) ? hi_196_1 : lo_197_1);
  float pv_198 = __shfl_xor_sync(0xFFFFFFFF, cur_191, 2);
  float hi_199 = fmaxf(cur_191, pv_198);
  float lo_200 = fminf(cur_191, pv_198);
  cur_191 = ((up[2] != 0) ? hi_199 : lo_200);
  float pv_201 = __shfl_xor_sync(0xFFFFFFFF, cur_191, 1);
  float hi_202_1 = fmaxf(cur_191, pv_201);
  float lo_203_1 = fminf(cur_191, pv_201);
  cur_191 = ((up[3] != 0) ? hi_202_1 : lo_203_1);
  V[3] = cur_191;
  float y1_204 = __shfl_xor_sync(0xFFFFFFFF, V[2], 15);
  float lo1_205 = fminf(V[0], y1_204);
  r_1 = fmaxf(r_1, lo1_205);
  float hi1_206 = fmaxf(V[0], y1_204);
  float cur_207 = hi1_206;
  float pv_208 = __shfl_xor_sync(0xFFFFFFFF, cur_207, 8);
  float hi_209 = fmaxf(cur_207, pv_208);
  float lo_210 = fminf(cur_207, pv_208);
  cur_207 = ((up[0] != 0) ? hi_209 : lo_210);
  float pv_211 = __shfl_xor_sync(0xFFFFFFFF, cur_207, 4);
  float hi_212_1 = fmaxf(cur_207, pv_211);
  float lo_213_1 = fminf(cur_207, pv_211);
  cur_207 = ((up[1] != 0) ? hi_212_1 : lo_213_1);
  float pv_214 = __shfl_xor_sync(0xFFFFFFFF, cur_207, 2);
  float hi_215 = fmaxf(cur_207, pv_214);
  float lo_216 = fminf(cur_207, pv_214);
  cur_207 = ((up[2] != 0) ? hi_215 : lo_216);
  float pv_217 = __shfl_xor_sync(0xFFFFFFFF, cur_207, 1);
  float hi_218_1 = fmaxf(cur_207, pv_217);
  float lo_219_1 = fminf(cur_207, pv_217);
  cur_207 = ((up[3] != 0) ? hi_218_1 : lo_219_1);
  V[0] = cur_207;
  float y1_220 = __shfl_xor_sync(0xFFFFFFFF, V[3], 15);
  float lo1_221 = fminf(V[1], y1_220);
  r_1 = fmaxf(r_1, lo1_221);
  float hi1_222 = fmaxf(V[1], y1_220);
  float cur_223 = hi1_222;
  float pv_224 = __shfl_xor_sync(0xFFFFFFFF, cur_223, 8);
  float hi_225 = fmaxf(cur_223, pv_224);
  float lo_226 = fminf(cur_223, pv_224);
  cur_223 = ((up[0] != 0) ? hi_225 : lo_226);
  float pv_227 = __shfl_xor_sync(0xFFFFFFFF, cur_223, 4);
  float hi_228_1 = fmaxf(cur_223, pv_227);
  float lo_229_1 = fminf(cur_223, pv_227);
  cur_223 = ((up[1] != 0) ? hi_228_1 : lo_229_1);
  float pv_230 = __shfl_xor_sync(0xFFFFFFFF, cur_223, 2);
  float hi_231 = fmaxf(cur_223, pv_230);
  float lo_232 = fminf(cur_223, pv_230);
  cur_223 = ((up[2] != 0) ? hi_231 : lo_232);
  float pv_233 = __shfl_xor_sync(0xFFFFFFFF, cur_223, 1);
  float hi_234_1 = fmaxf(cur_223, pv_233);
  float lo_235_1 = fminf(cur_223, pv_233);
  cur_223 = ((up[3] != 0) ? hi_234_1 : lo_235_1);
  V[1] = cur_223;
  float yl = __shfl_xor_sync(0xFFFFFFFF, V[1], 15);
  float lol = fminf(V[0], yl);
  r_1 = fmaxf(r_1, lol);
  float hil = fmaxf(V[0], yl);
  V[0] = hil;
  float K = V[0];
  rr1[0] = r_1;
  int ucol = blockIdx.x * 32 + cg;
  int commit = 0;
  if (g < 32 && ucol < total_q) {
    commit = 1;
  }
  if (tid_1 < 512) {
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
    unsigned int c16 = u16b & 4294965248u;
    unsigned int crb = __as_u32(cr) & 4294965248u;
    unsigned int q = 4294967295u;
    if (c16 == crb && u16b < 4278190080u && c16 != 2139092992 && commit != 0) {
      q = c16;
      if (ln == 0) {
        flagw[0] = 1;
      }
    }
    if (ln == 0 && g < 32) {
      qcol[cg] = q;
    }
  }
  asm volatile("barrier.sync 1, 512;" ::: "memory");
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
    unsigned int qv7 = 4294967295u;
    if ((tid_1 & 31) < 32) {
      qv7 = qcol[tid_1 & 31];
    }
    unsigned int fm7 = __ballot_sync(0xFFFFFFFF, qv7 != 4294967295u);
    int nf7 = (int)(unsigned int)__popc(fm7);
#pragma unroll 1
    for (int it7 = 0; it7 < nf7; it7++) {
      int _find_first_set_8 = __ffs(fm7);
      int fc7 = _find_first_set_8 - 1;
      fm7 = fm7 & fm7 - 1;
      unsigned int qf7 = qcol[fc7];
      int limv = limc[fc7];
      long long cbase2 =
          (long long)head * (long long)tiles * nq64 + (long long)(blockIdx.x * 32 + fc7);
      int lane_1 = tid_1 & 31;
      unsigned int bb[4];
      bb[0] = 4286578688u;
      int s5 = lane_1;
      int j5 = s5 >> 4 & 7;
      int v5 = 0;
      int s4 = lane_1;
      int i4 = s4 & 15;
      int j4 = s4 >> 4 & 7;
      int g4 = whi + (s4 >> 7);
      int t4 = (j4 * 16 + g4) * 16 + i4;
      if (j5 < num_chunks && t4 < limv) {
        v5 = 1;
      }
      if (v5 != 0) {
        int s4_0 = lane_1;
        int i4_1 = s4_0 & 15;
        int j4_2 = s4_0 >> 4 & 7;
        int g4_3 = whi + (s4_0 >> 7);
        int t4_4 = (j4_2 * 16 + g4_3) * 16 + i4_1;
        bb[0] = S[cbase2 + (long long)t4_4 * nq64];
      }
      bb[1] = 4286578688u;
      int s5_0 = lane_1 + 32;
      int j5_1 = s5_0 >> 4 & 7;
      int v5_2 = 0;
      int s4_3 = lane_1 + 32;
      int i4_4 = s4_3 & 15;
      int j4_5 = s4_3 >> 4 & 7;
      int g4_6 = whi + (s4_3 >> 7);
      int t4_7 = (j4_5 * 16 + g4_6) * 16 + i4_4;
      if (j5_1 < num_chunks && t4_7 < limv) {
        v5_2 = 1;
      }
      if (v5_2 != 0) {
        int s4_0_1 = lane_1 + 32;
        int i4_1_1 = s4_0_1 & 15;
        int j4_2_1 = s4_0_1 >> 4 & 7;
        int g4_3_1 = whi + (s4_0_1 >> 7);
        int t4_4_1 = (j4_2_1 * 16 + g4_3_1) * 16 + i4_1_1;
        bb[1] = S[cbase2 + (long long)t4_4_1 * nq64];
      }
      bb[2] = 4286578688u;
      int s5_8 = lane_1 + 64;
      int j5_9 = s5_8 >> 4 & 7;
      int v5_10 = 0;
      int s4_11 = lane_1 + 64;
      int i4_12 = s4_11 & 15;
      int j4_13 = s4_11 >> 4 & 7;
      int g4_14 = whi + (s4_11 >> 7);
      int t4_15 = (j4_13 * 16 + g4_14) * 16 + i4_12;
      if (j5_9 < num_chunks && t4_15 < limv) {
        v5_10 = 1;
      }
      if (v5_10 != 0) {
        int s4_0_2 = lane_1 + 64;
        int i4_1_2 = s4_0_2 & 15;
        int j4_2_2 = s4_0_2 >> 4 & 7;
        int g4_3_2 = whi + (s4_0_2 >> 7);
        int t4_4_2 = (j4_2_2 * 16 + g4_3_2) * 16 + i4_1_2;
        bb[2] = S[cbase2 + (long long)t4_4_2 * nq64];
      }
      bb[3] = 4286578688u;
      int s5_16 = lane_1 + 96;
      int j5_17 = s5_16 >> 4 & 7;
      int v5_18 = 0;
      int s4_19 = lane_1 + 96;
      int i4_20 = s4_19 & 15;
      int j4_21 = s4_19 >> 4 & 7;
      int g4_22 = whi + (s4_19 >> 7);
      int t4_23 = (j4_21 * 16 + g4_22) * 16 + i4_20;
      if (j5_17 < num_chunks && t4_23 < limv) {
        v5_18 = 1;
      }
      if (v5_18 != 0) {
        int s4_0_3 = lane_1 + 96;
        int i4_1_3 = s4_0_3 & 15;
        int j4_2_3 = s4_0_3 >> 4 & 7;
        int g4_3_3 = whi + (s4_0_3 >> 7);
        int t4_4_3 = (j4_2_3 * 16 + g4_3_3) * 16 + i4_1_3;
        bb[3] = S[cbase2 + (long long)t4_4_3 * nq64];
      }
      asm volatile("" ::: "memory");
      unsigned int kk2[4];
      kk2[0] = 0;
      int s5_24 = lane_1;
      int j5_25 = s5_24 >> 4 & 7;
      int v5_26 = 0;
      int s4_27 = lane_1;
      int i4_28 = s4_27 & 15;
      int j4_29 = s4_27 >> 4 & 7;
      int g4_30 = whi + (s4_27 >> 7);
      int t4_31 = (j4_29 * 16 + g4_30) * 16 + i4_28;
      if (j5_25 < num_chunks && t4_31 < limv) {
        v5_26 = 1;
      }
      if (v5_26 != 0) {
        int s4_0_4 = lane_1;
        int i4_1_4 = s4_0_4 & 15;
        int j4_2_4 = s4_0_4 >> 4 & 7;
        int g4_3_4 = whi + (s4_0_4 >> 7);
        int t4_4_4 = (j4_2_4 * 16 + g4_3_4) * 16 + i4_1_4;
        float sc_1 = __uint_as_float(bb[0]);
        sc_1 = fmaxf(sc_1, -1.7014118346046923e+38f);
        sc_1 = fminf(sc_1, 1.7014118346046923e+38f);
        unsigned int u8 = __as_u32(sc_1);
        unsigned int key8 = 0;
        if ((u8 & 4294965248u) == qf7) {
          unsigned int lowb8 = (u8 ^ (unsigned int)((int)u8 >> 31) & 2047) & 2047;
          key8 = 536870912 | lowb8 << 11 | (unsigned int)t4_4_4;
        }
        int f2 = 0;
        if ((t4_4_4 < fb || t4_4_4 >= limv - fe) && t4_4_4 < limv) {
          f2 = 1;
        }
        if (f2 != 0) {
          key8 = 0;
        }
        kk2[0] = key8;
      }
      kk2[1] = 0;
      int s5_32 = lane_1 + 32;
      int j5_33 = s5_32 >> 4 & 7;
      int v5_34 = 0;
      int s4_35 = lane_1 + 32;
      int i4_36 = s4_35 & 15;
      int j4_37 = s4_35 >> 4 & 7;
      int g4_38 = whi + (s4_35 >> 7);
      int t4_39 = (j4_37 * 16 + g4_38) * 16 + i4_36;
      if (j5_33 < num_chunks && t4_39 < limv) {
        v5_34 = 1;
      }
      if (v5_34 != 0) {
        int s4_0_5 = lane_1 + 32;
        int i4_1_5 = s4_0_5 & 15;
        int j4_2_5 = s4_0_5 >> 4 & 7;
        int g4_3_5 = whi + (s4_0_5 >> 7);
        int t4_4_5 = (j4_2_5 * 16 + g4_3_5) * 16 + i4_1_5;
        float sc_2 = __uint_as_float(bb[1]);
        sc_2 = fmaxf(sc_2, -1.7014118346046923e+38f);
        sc_2 = fminf(sc_2, 1.7014118346046923e+38f);
        unsigned int u8_1 = __as_u32(sc_2);
        unsigned int key8_1 = 0;
        if ((u8_1 & 4294965248u) == qf7) {
          unsigned int lowb8_1 = (u8_1 ^ (unsigned int)((int)u8_1 >> 31) & 2047) & 2047;
          key8_1 = 536870912 | lowb8_1 << 11 | (unsigned int)t4_4_5;
        }
        int f2_1 = 0;
        if ((t4_4_5 < fb || t4_4_5 >= limv - fe) && t4_4_5 < limv) {
          f2_1 = 1;
        }
        if (f2_1 != 0) {
          key8_1 = 0;
        }
        kk2[1] = key8_1;
      }
      kk2[2] = 0;
      int s5_40 = lane_1 + 64;
      int j5_41 = s5_40 >> 4 & 7;
      int v5_42 = 0;
      int s4_43 = lane_1 + 64;
      int i4_44 = s4_43 & 15;
      int j4_45 = s4_43 >> 4 & 7;
      int g4_46 = whi + (s4_43 >> 7);
      int t4_47 = (j4_45 * 16 + g4_46) * 16 + i4_44;
      if (j5_41 < num_chunks && t4_47 < limv) {
        v5_42 = 1;
      }
      if (v5_42 != 0) {
        int s4_0_6 = lane_1 + 64;
        int i4_1_6 = s4_0_6 & 15;
        int j4_2_6 = s4_0_6 >> 4 & 7;
        int g4_3_6 = whi + (s4_0_6 >> 7);
        int t4_4_6 = (j4_2_6 * 16 + g4_3_6) * 16 + i4_1_6;
        float sc_3 = __uint_as_float(bb[2]);
        sc_3 = fmaxf(sc_3, -1.7014118346046923e+38f);
        sc_3 = fminf(sc_3, 1.7014118346046923e+38f);
        unsigned int u8_2 = __as_u32(sc_3);
        unsigned int key8_2 = 0;
        if ((u8_2 & 4294965248u) == qf7) {
          unsigned int lowb8_2 = (u8_2 ^ (unsigned int)((int)u8_2 >> 31) & 2047) & 2047;
          key8_2 = 536870912 | lowb8_2 << 11 | (unsigned int)t4_4_6;
        }
        int f2_2 = 0;
        if ((t4_4_6 < fb || t4_4_6 >= limv - fe) && t4_4_6 < limv) {
          f2_2 = 1;
        }
        if (f2_2 != 0) {
          key8_2 = 0;
        }
        kk2[2] = key8_2;
      }
      kk2[3] = 0;
      int s5_48 = lane_1 + 96;
      int j5_49 = s5_48 >> 4 & 7;
      int v5_50 = 0;
      int s4_51 = lane_1 + 96;
      int i4_52 = s4_51 & 15;
      int j4_53 = s4_51 >> 4 & 7;
      int g4_54 = whi + (s4_51 >> 7);
      int t4_55 = (j4_53 * 16 + g4_54) * 16 + i4_52;
      if (j5_49 < num_chunks && t4_55 < limv) {
        v5_50 = 1;
      }
      if (v5_50 != 0) {
        int s4_0_7 = lane_1 + 96;
        int i4_1_7 = s4_0_7 & 15;
        int j4_2_7 = s4_0_7 >> 4 & 7;
        int g4_3_7 = whi + (s4_0_7 >> 7);
        int t4_4_7 = (j4_2_7 * 16 + g4_3_7) * 16 + i4_1_7;
        float sc_5 = __uint_as_float(bb[3]);
        sc_5 = fmaxf(sc_5, -1.7014118346046923e+38f);
        sc_5 = fminf(sc_5, 1.7014118346046923e+38f);
        unsigned int u8_3 = __as_u32(sc_5);
        unsigned int key8_3 = 0;
        if ((u8_3 & 4294965248u) == qf7) {
          unsigned int lowb8_3 = (u8_3 ^ (unsigned int)((int)u8_3 >> 31) & 2047) & 2047;
          key8_3 = 536870912 | lowb8_3 << 11 | (unsigned int)t4_4_7;
        }
        int f2_3 = 0;
        if ((t4_4_7 < fb || t4_4_7 >= limv - fe) && t4_4_7 < limv) {
          f2_3 = 1;
        }
        if (f2_3 != 0) {
          key8_3 = 0;
        }
        kk2[3] = key8_3;
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
        }
      }
    }
    asm volatile("barrier.sync 1, 512;" ::: "memory");
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
        int t0_1_1 = w * 16;
        long long p_2_1 = cbase + (long long)t0_1_1 * nq64;
        cb2[0] = 4286578688u;
        if (lim2x > t0_1_1) {
          cb2[0] = S[p_2_1];
        }
        cb2[1] = 4286578688u;
        if (lim2x > t0_1_1 + 1) {
          cb2[1] = S[p_2_1 + nq64];
        }
        cb2[2] = 4286578688u;
        if (lim2x > t0_1_1 + 2) {
          cb2[2] = S[p_2_1 + 2 * nq64];
        }
        cb2[3] = 4286578688u;
        if (lim2x > t0_1_1 + 3) {
          cb2[3] = S[p_2_1 + 3 * nq64];
        }
        cb2[4] = 4286578688u;
        if (lim2x > t0_1_1 + 4) {
          cb2[4] = S[p_2_1 + 4 * nq64];
        }
        cb2[5] = 4286578688u;
        if (lim2x > t0_1_1 + 5) {
          cb2[5] = S[p_2_1 + 5 * nq64];
        }
        cb2[6] = 4286578688u;
        if (lim2x > t0_1_1 + 6) {
          cb2[6] = S[p_2_1 + 6 * nq64];
        }
        cb2[7] = 4286578688u;
        if (lim2x > t0_1_1 + 7) {
          cb2[7] = S[p_2_1 + 7 * nq64];
        }
        cb2[8] = 4286578688u;
        if (lim2x > t0_1_1 + 8) {
          cb2[8] = S[p_2_1 + 8 * nq64];
        }
        cb2[9] = 4286578688u;
        if (lim2x > t0_1_1 + 9) {
          cb2[9] = S[p_2_1 + 9 * nq64];
        }
        cb2[10] = 4286578688u;
        if (lim2x > t0_1_1 + 10) {
          cb2[10] = S[p_2_1 + 10 * nq64];
        }
        cb2[11] = 4286578688u;
        if (lim2x > t0_1_1 + 11) {
          cb2[11] = S[p_2_1 + 11 * nq64];
        }
        cb2[12] = 4286578688u;
        if (lim2x > t0_1_1 + 12) {
          cb2[12] = S[p_2_1 + 12 * nq64];
        }
        cb2[13] = 4286578688u;
        if (lim2x > t0_1_1 + 13) {
          cb2[13] = S[p_2_1 + 13 * nq64];
        }
        cb2[14] = 4286578688u;
        if (lim2x > t0_1_1 + 14) {
          cb2[14] = S[p_2_1 + 14 * nq64];
        }
        cb2[15] = 4286578688u;
        if (lim2x > t0_1_1 + 15) {
          cb2[15] = S[p_2_1 + 15 * nq64];
        }
        asm volatile("" ::: "memory");
#pragma unroll 1
        for (int j_1 = 0; j_1 < num_chunks; j_1++) {
          int t0_2 = ((j_1 + 1) * 16 + w) * 16;
          long long p_3 = cbase + (long long)t0_2 * nq64;
          nb2[0] = 4286578688u;
          if (lim2x > t0_2) {
            nb2[0] = S[p_3];
          }
          nb2[1] = 4286578688u;
          if (lim2x > t0_2 + 1) {
            nb2[1] = S[p_3 + nq64];
          }
          nb2[2] = 4286578688u;
          if (lim2x > t0_2 + 2) {
            nb2[2] = S[p_3 + 2 * nq64];
          }
          nb2[3] = 4286578688u;
          if (lim2x > t0_2 + 3) {
            nb2[3] = S[p_3 + 3 * nq64];
          }
          nb2[4] = 4286578688u;
          if (lim2x > t0_2 + 4) {
            nb2[4] = S[p_3 + 4 * nq64];
          }
          nb2[5] = 4286578688u;
          if (lim2x > t0_2 + 5) {
            nb2[5] = S[p_3 + 5 * nq64];
          }
          nb2[6] = 4286578688u;
          if (lim2x > t0_2 + 6) {
            nb2[6] = S[p_3 + 6 * nq64];
          }
          nb2[7] = 4286578688u;
          if (lim2x > t0_2 + 7) {
            nb2[7] = S[p_3 + 7 * nq64];
          }
          nb2[8] = 4286578688u;
          if (lim2x > t0_2 + 8) {
            nb2[8] = S[p_3 + 8 * nq64];
          }
          nb2[9] = 4286578688u;
          if (lim2x > t0_2 + 9) {
            nb2[9] = S[p_3 + 9 * nq64];
          }
          nb2[10] = 4286578688u;
          if (lim2x > t0_2 + 10) {
            nb2[10] = S[p_3 + 10 * nq64];
          }
          nb2[11] = 4286578688u;
          if (lim2x > t0_2 + 11) {
            nb2[11] = S[p_3 + 11 * nq64];
          }
          nb2[12] = 4286578688u;
          if (lim2x > t0_2 + 12) {
            nb2[12] = S[p_3 + 12 * nq64];
          }
          nb2[13] = 4286578688u;
          if (lim2x > t0_2 + 13) {
            nb2[13] = S[p_3 + 13 * nq64];
          }
          nb2[14] = 4286578688u;
          if (lim2x > t0_2 + 14) {
            nb2[14] = S[p_3 + 14 * nq64];
          }
          nb2[15] = 4286578688u;
          if (lim2x > t0_2 + 15) {
            nb2[15] = S[p_3 + 15 * nq64];
          }
          asm volatile("" ::: "memory");
          int t0_4 = (j_1 * 16 + w) * 16;
          float qf = __uint_as_float(qc);
          float sc_7 = __uint_as_float(cb2[0]);
          sc_7 = fmaxf(sc_7, -1.7014118346046923e+38f);
          sc_7 = fminf(sc_7, 1.7014118346046923e+38f);
          unsigned int u = __as_u32(sc_7);
          unsigned int cls = u & 4294965248u;
          int f_16 = 0;
          if ((t0_4 < fb || t0_4 >= lim - fe) && lim > t0_4) {
            f_16 = 1;
          }
          if (f_16 != 0) {
            cls = 2139092992;
          }
          unsigned int key_1 = 0;
          if (qf < __uint_as_float(cls) && cls < 4278190080u) {
            key_1 = 1073741824 | (unsigned int)t0_4;
          }
          if (cls == qc) {
            unsigned int lowb = (u ^ (unsigned int)((int)u >> 31) & 2047) & 2047;
            key_1 = 536870912 | lowb << 11 | (unsigned int)t0_4;
          }
          kb2[0] = __uint_as_float(key_1);
          float sc_5_1 = __uint_as_float(cb2[1]);
          sc_5_1 = fmaxf(sc_5_1, -1.7014118346046923e+38f);
          sc_5_1 = fminf(sc_5_1, 1.7014118346046923e+38f);
          unsigned int u_6 = __as_u32(sc_5_1);
          unsigned int cls_7 = u_6 & 4294965248u;
          int f_8_1 = 0;
          if ((t0_4 + 1 < fb || t0_4 + 1 >= lim - fe) && lim > t0_4 + 1) {
            f_8_1 = 1;
          }
          if (f_8_1 != 0) {
            cls_7 = 2139092992;
          }
          unsigned int key_9_1 = 0;
          if (qf < __uint_as_float(cls_7) && cls_7 < 4278190080u) {
            key_9_1 = 1073741824 | (unsigned int)(t0_4 + 1);
          }
          if (cls_7 == qc) {
            unsigned int lowb_1 = (u_6 ^ (unsigned int)((int)u_6 >> 31) & 2047) & 2047;
            key_9_1 = 536870912 | lowb_1 << 11 | (unsigned int)(t0_4 + 1);
          }
          kb2[1] = __uint_as_float(key_9_1);
          float sc_10_1 = __uint_as_float(cb2[2]);
          sc_10_1 = fmaxf(sc_10_1, -1.7014118346046923e+38f);
          sc_10_1 = fminf(sc_10_1, 1.7014118346046923e+38f);
          unsigned int u_11 = __as_u32(sc_10_1);
          unsigned int cls_12 = u_11 & 4294965248u;
          int f_13_1 = 0;
          if ((t0_4 + 2 < fb || t0_4 + 2 >= lim - fe) && lim > t0_4 + 2) {
            f_13_1 = 1;
          }
          if (f_13_1 != 0) {
            cls_12 = 2139092992;
          }
          unsigned int key_14 = 0;
          if (qf < __uint_as_float(cls_12) && cls_12 < 4278190080u) {
            key_14 = 1073741824 | (unsigned int)(t0_4 + 2);
          }
          if (cls_12 == qc) {
            unsigned int lowb_2 = (u_11 ^ (unsigned int)((int)u_11 >> 31) & 2047) & 2047;
            key_14 = 536870912 | lowb_2 << 11 | (unsigned int)(t0_4 + 2);
          }
          kb2[2] = __uint_as_float(key_14);
          float sc_15 = __uint_as_float(cb2[3]);
          sc_15 = fmaxf(sc_15, -1.7014118346046923e+38f);
          sc_15 = fminf(sc_15, 1.7014118346046923e+38f);
          unsigned int u_16 = __as_u32(sc_15);
          unsigned int cls_17 = u_16 & 4294965248u;
          int f_18 = 0;
          if ((t0_4 + 3 < fb || t0_4 + 3 >= lim - fe) && lim > t0_4 + 3) {
            f_18 = 1;
          }
          if (f_18 != 0) {
            cls_17 = 2139092992;
          }
          unsigned int key_19_1 = 0;
          if (qf < __uint_as_float(cls_17) && cls_17 < 4278190080u) {
            key_19_1 = 1073741824 | (unsigned int)(t0_4 + 3);
          }
          if (cls_17 == qc) {
            unsigned int lowb_3 = (u_16 ^ (unsigned int)((int)u_16 >> 31) & 2047) & 2047;
            key_19_1 = 536870912 | lowb_3 << 11 | (unsigned int)(t0_4 + 3);
          }
          kb2[3] = __uint_as_float(key_19_1);
          float sc_20_1 = __uint_as_float(cb2[4]);
          sc_20_1 = fmaxf(sc_20_1, -1.7014118346046923e+38f);
          sc_20_1 = fminf(sc_20_1, 1.7014118346046923e+38f);
          unsigned int u_21 = __as_u32(sc_20_1);
          unsigned int cls_22 = u_21 & 4294965248u;
          int f_23 = 0;
          if ((t0_4 + 4 < fb || t0_4 + 4 >= lim - fe) && lim > t0_4 + 4) {
            f_23 = 1;
          }
          if (f_23 != 0) {
            cls_22 = 2139092992;
          }
          unsigned int key_24 = 0;
          if (qf < __uint_as_float(cls_22) && cls_22 < 4278190080u) {
            key_24 = 1073741824 | (unsigned int)(t0_4 + 4);
          }
          if (cls_22 == qc) {
            unsigned int lowb_4 = (u_21 ^ (unsigned int)((int)u_21 >> 31) & 2047) & 2047;
            key_24 = 536870912 | lowb_4 << 11 | (unsigned int)(t0_4 + 4);
          }
          kb2[4] = __uint_as_float(key_24);
          float sc_25 = __uint_as_float(cb2[5]);
          sc_25 = fmaxf(sc_25, -1.7014118346046923e+38f);
          sc_25 = fminf(sc_25, 1.7014118346046923e+38f);
          unsigned int u_26 = __as_u32(sc_25);
          unsigned int cls_27 = u_26 & 4294965248u;
          int f_28 = 0;
          if ((t0_4 + 5 < fb || t0_4 + 5 >= lim - fe) && lim > t0_4 + 5) {
            f_28 = 1;
          }
          if (f_28 != 0) {
            cls_27 = 2139092992;
          }
          unsigned int key_29_1 = 0;
          if (qf < __uint_as_float(cls_27) && cls_27 < 4278190080u) {
            key_29_1 = 1073741824 | (unsigned int)(t0_4 + 5);
          }
          if (cls_27 == qc) {
            unsigned int lowb_5 = (u_26 ^ (unsigned int)((int)u_26 >> 31) & 2047) & 2047;
            key_29_1 = 536870912 | lowb_5 << 11 | (unsigned int)(t0_4 + 5);
          }
          kb2[5] = __uint_as_float(key_29_1);
          float sc_30_1 = __uint_as_float(cb2[6]);
          sc_30_1 = fmaxf(sc_30_1, -1.7014118346046923e+38f);
          sc_30_1 = fminf(sc_30_1, 1.7014118346046923e+38f);
          unsigned int u_31 = __as_u32(sc_30_1);
          unsigned int cls_32 = u_31 & 4294965248u;
          int f_33 = 0;
          if ((t0_4 + 6 < fb || t0_4 + 6 >= lim - fe) && lim > t0_4 + 6) {
            f_33 = 1;
          }
          if (f_33 != 0) {
            cls_32 = 2139092992;
          }
          unsigned int key_34 = 0;
          if (qf < __uint_as_float(cls_32) && cls_32 < 4278190080u) {
            key_34 = 1073741824 | (unsigned int)(t0_4 + 6);
          }
          if (cls_32 == qc) {
            unsigned int lowb_6 = (u_31 ^ (unsigned int)((int)u_31 >> 31) & 2047) & 2047;
            key_34 = 536870912 | lowb_6 << 11 | (unsigned int)(t0_4 + 6);
          }
          kb2[6] = __uint_as_float(key_34);
          float sc_35 = __uint_as_float(cb2[7]);
          sc_35 = fmaxf(sc_35, -1.7014118346046923e+38f);
          sc_35 = fminf(sc_35, 1.7014118346046923e+38f);
          unsigned int u_36 = __as_u32(sc_35);
          unsigned int cls_37 = u_36 & 4294965248u;
          int f_38 = 0;
          if ((t0_4 + 7 < fb || t0_4 + 7 >= lim - fe) && lim > t0_4 + 7) {
            f_38 = 1;
          }
          if (f_38 != 0) {
            cls_37 = 2139092992;
          }
          unsigned int key_39 = 0;
          if (qf < __uint_as_float(cls_37) && cls_37 < 4278190080u) {
            key_39 = 1073741824 | (unsigned int)(t0_4 + 7);
          }
          if (cls_37 == qc) {
            unsigned int lowb_7 = (u_36 ^ (unsigned int)((int)u_36 >> 31) & 2047) & 2047;
            key_39 = 536870912 | lowb_7 << 11 | (unsigned int)(t0_4 + 7);
          }
          kb2[7] = __uint_as_float(key_39);
          float sc_40 = __uint_as_float(cb2[8]);
          sc_40 = fmaxf(sc_40, -1.7014118346046923e+38f);
          sc_40 = fminf(sc_40, 1.7014118346046923e+38f);
          unsigned int u_41 = __as_u32(sc_40);
          unsigned int cls_42 = u_41 & 4294965248u;
          int f_43 = 0;
          if ((t0_4 + 8 < fb || t0_4 + 8 >= lim - fe) && lim > t0_4 + 8) {
            f_43 = 1;
          }
          if (f_43 != 0) {
            cls_42 = 2139092992;
          }
          unsigned int key_44 = 0;
          if (qf < __uint_as_float(cls_42) && cls_42 < 4278190080u) {
            key_44 = 1073741824 | (unsigned int)(t0_4 + 8);
          }
          if (cls_42 == qc) {
            unsigned int lowb_8 = (u_41 ^ (unsigned int)((int)u_41 >> 31) & 2047) & 2047;
            key_44 = 536870912 | lowb_8 << 11 | (unsigned int)(t0_4 + 8);
          }
          kb2[8] = __uint_as_float(key_44);
          float sc_45 = __uint_as_float(cb2[9]);
          sc_45 = fmaxf(sc_45, -1.7014118346046923e+38f);
          sc_45 = fminf(sc_45, 1.7014118346046923e+38f);
          unsigned int u_46 = __as_u32(sc_45);
          unsigned int cls_47 = u_46 & 4294965248u;
          int f_48 = 0;
          if ((t0_4 + 9 < fb || t0_4 + 9 >= lim - fe) && lim > t0_4 + 9) {
            f_48 = 1;
          }
          if (f_48 != 0) {
            cls_47 = 2139092992;
          }
          unsigned int key_49 = 0;
          if (qf < __uint_as_float(cls_47) && cls_47 < 4278190080u) {
            key_49 = 1073741824 | (unsigned int)(t0_4 + 9);
          }
          if (cls_47 == qc) {
            unsigned int lowb_9 = (u_46 ^ (unsigned int)((int)u_46 >> 31) & 2047) & 2047;
            key_49 = 536870912 | lowb_9 << 11 | (unsigned int)(t0_4 + 9);
          }
          kb2[9] = __uint_as_float(key_49);
          float sc_50 = __uint_as_float(cb2[10]);
          sc_50 = fmaxf(sc_50, -1.7014118346046923e+38f);
          sc_50 = fminf(sc_50, 1.7014118346046923e+38f);
          unsigned int u_51 = __as_u32(sc_50);
          unsigned int cls_52 = u_51 & 4294965248u;
          int f_53 = 0;
          if ((t0_4 + 10 < fb || t0_4 + 10 >= lim - fe) && lim > t0_4 + 10) {
            f_53 = 1;
          }
          if (f_53 != 0) {
            cls_52 = 2139092992;
          }
          unsigned int key_54 = 0;
          if (qf < __uint_as_float(cls_52) && cls_52 < 4278190080u) {
            key_54 = 1073741824 | (unsigned int)(t0_4 + 10);
          }
          if (cls_52 == qc) {
            unsigned int lowb_10 = (u_51 ^ (unsigned int)((int)u_51 >> 31) & 2047) & 2047;
            key_54 = 536870912 | lowb_10 << 11 | (unsigned int)(t0_4 + 10);
          }
          kb2[10] = __uint_as_float(key_54);
          float sc_55 = __uint_as_float(cb2[11]);
          sc_55 = fmaxf(sc_55, -1.7014118346046923e+38f);
          sc_55 = fminf(sc_55, 1.7014118346046923e+38f);
          unsigned int u_56 = __as_u32(sc_55);
          unsigned int cls_57 = u_56 & 4294965248u;
          int f_58 = 0;
          if ((t0_4 + 11 < fb || t0_4 + 11 >= lim - fe) && lim > t0_4 + 11) {
            f_58 = 1;
          }
          if (f_58 != 0) {
            cls_57 = 2139092992;
          }
          unsigned int key_59 = 0;
          if (qf < __uint_as_float(cls_57) && cls_57 < 4278190080u) {
            key_59 = 1073741824 | (unsigned int)(t0_4 + 11);
          }
          if (cls_57 == qc) {
            unsigned int lowb_11 = (u_56 ^ (unsigned int)((int)u_56 >> 31) & 2047) & 2047;
            key_59 = 536870912 | lowb_11 << 11 | (unsigned int)(t0_4 + 11);
          }
          kb2[11] = __uint_as_float(key_59);
          float sc_60 = __uint_as_float(cb2[12]);
          sc_60 = fmaxf(sc_60, -1.7014118346046923e+38f);
          sc_60 = fminf(sc_60, 1.7014118346046923e+38f);
          unsigned int u_61 = __as_u32(sc_60);
          unsigned int cls_62 = u_61 & 4294965248u;
          int f_63 = 0;
          if ((t0_4 + 12 < fb || t0_4 + 12 >= lim - fe) && lim > t0_4 + 12) {
            f_63 = 1;
          }
          if (f_63 != 0) {
            cls_62 = 2139092992;
          }
          unsigned int key_64 = 0;
          if (qf < __uint_as_float(cls_62) && cls_62 < 4278190080u) {
            key_64 = 1073741824 | (unsigned int)(t0_4 + 12);
          }
          if (cls_62 == qc) {
            unsigned int lowb_12 = (u_61 ^ (unsigned int)((int)u_61 >> 31) & 2047) & 2047;
            key_64 = 536870912 | lowb_12 << 11 | (unsigned int)(t0_4 + 12);
          }
          kb2[12] = __uint_as_float(key_64);
          float sc_65 = __uint_as_float(cb2[13]);
          sc_65 = fmaxf(sc_65, -1.7014118346046923e+38f);
          sc_65 = fminf(sc_65, 1.7014118346046923e+38f);
          unsigned int u_66 = __as_u32(sc_65);
          unsigned int cls_67 = u_66 & 4294965248u;
          int f_68 = 0;
          if ((t0_4 + 13 < fb || t0_4 + 13 >= lim - fe) && lim > t0_4 + 13) {
            f_68 = 1;
          }
          if (f_68 != 0) {
            cls_67 = 2139092992;
          }
          unsigned int key_69 = 0;
          if (qf < __uint_as_float(cls_67) && cls_67 < 4278190080u) {
            key_69 = 1073741824 | (unsigned int)(t0_4 + 13);
          }
          if (cls_67 == qc) {
            unsigned int lowb_13 = (u_66 ^ (unsigned int)((int)u_66 >> 31) & 2047) & 2047;
            key_69 = 536870912 | lowb_13 << 11 | (unsigned int)(t0_4 + 13);
          }
          kb2[13] = __uint_as_float(key_69);
          float sc_70 = __uint_as_float(cb2[14]);
          sc_70 = fmaxf(sc_70, -1.7014118346046923e+38f);
          sc_70 = fminf(sc_70, 1.7014118346046923e+38f);
          unsigned int u_71 = __as_u32(sc_70);
          unsigned int cls_72 = u_71 & 4294965248u;
          int f_73 = 0;
          if ((t0_4 + 14 < fb || t0_4 + 14 >= lim - fe) && lim > t0_4 + 14) {
            f_73 = 1;
          }
          if (f_73 != 0) {
            cls_72 = 2139092992;
          }
          unsigned int key_74 = 0;
          if (qf < __uint_as_float(cls_72) && cls_72 < 4278190080u) {
            key_74 = 1073741824 | (unsigned int)(t0_4 + 14);
          }
          if (cls_72 == qc) {
            unsigned int lowb_14 = (u_71 ^ (unsigned int)((int)u_71 >> 31) & 2047) & 2047;
            key_74 = 536870912 | lowb_14 << 11 | (unsigned int)(t0_4 + 14);
          }
          kb2[14] = __uint_as_float(key_74);
          float sc_75 = __uint_as_float(cb2[15]);
          sc_75 = fmaxf(sc_75, -1.7014118346046923e+38f);
          sc_75 = fminf(sc_75, 1.7014118346046923e+38f);
          unsigned int u_76 = __as_u32(sc_75);
          unsigned int cls_77 = u_76 & 4294965248u;
          int f_78 = 0;
          if ((t0_4 + 15 < fb || t0_4 + 15 >= lim - fe) && lim > t0_4 + 15) {
            f_78 = 1;
          }
          if (f_78 != 0) {
            cls_77 = 2139092992;
          }
          unsigned int key_79 = 0;
          if (qf < __uint_as_float(cls_77) && cls_77 < 4278190080u) {
            key_79 = 1073741824 | (unsigned int)(t0_4 + 15);
          }
          if (cls_77 == qc) {
            unsigned int lowb_15 = (u_76 ^ (unsigned int)((int)u_76 >> 31) & 2047) & 2047;
            key_79 = 536870912 | lowb_15 << 11 | (unsigned int)(t0_4 + 15);
          }
          kb2[15] = __uint_as_float(key_79);
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
            float hi_3 = fmaxf(kb2[1], kb2[12]);
            float lo_4 = fminf(kb2[1], kb2[12]);
            kb2[1] = hi_3;
            kb2[12] = lo_4;
            float hi_6 = fmaxf(kb2[2], kb2[15]);
            float lo_7 = fminf(kb2[2], kb2[15]);
            kb2[2] = hi_6;
            kb2[15] = lo_7;
            float hi_9 = fmaxf(kb2[3], kb2[14]);
            float lo_10 = fminf(kb2[3], kb2[14]);
            kb2[3] = hi_9;
            kb2[14] = lo_10;
            float hi_11 = fmaxf(kb2[4], kb2[8]);
            float lo_12 = fminf(kb2[4], kb2[8]);
            kb2[4] = hi_11;
            kb2[8] = lo_12;
            float hi_13 = fmaxf(kb2[5], kb2[6]);
            float lo_14 = fminf(kb2[5], kb2[6]);
            kb2[5] = hi_13;
            kb2[6] = lo_14;
            float hi_15 = fmaxf(kb2[7], kb2[11]);
            float lo_16 = fminf(kb2[7], kb2[11]);
            kb2[7] = hi_15;
            kb2[11] = lo_16;
            float hi_17 = fmaxf(kb2[9], kb2[10]);
            float lo_18 = fminf(kb2[9], kb2[10]);
            kb2[9] = hi_17;
            kb2[10] = lo_18;
            float hi_19 = fmaxf(kb2[0], kb2[5]);
            float lo_20 = fminf(kb2[0], kb2[5]);
            kb2[0] = hi_19;
            kb2[5] = lo_20;
            float hi_22 = fmaxf(kb2[1], kb2[7]);
            float lo_23 = fminf(kb2[1], kb2[7]);
            kb2[1] = hi_22;
            kb2[7] = lo_23;
            float hi_25 = fmaxf(kb2[2], kb2[9]);
            float lo_26 = fminf(kb2[2], kb2[9]);
            kb2[2] = hi_25;
            kb2[9] = lo_26;
            float hi_28 = fmaxf(kb2[3], kb2[4]);
            float lo_29 = fminf(kb2[3], kb2[4]);
            kb2[3] = hi_28;
            kb2[4] = lo_29;
            float hi_30 = fmaxf(kb2[6], kb2[13]);
            float lo_31 = fminf(kb2[6], kb2[13]);
            kb2[6] = hi_30;
            kb2[13] = lo_31;
            float hi_32 = fmaxf(kb2[8], kb2[14]);
            float lo_33 = fminf(kb2[8], kb2[14]);
            kb2[8] = hi_32;
            kb2[14] = lo_33;
            float hi_34_1 = fmaxf(kb2[10], kb2[15]);
            float lo_35_1 = fminf(kb2[10], kb2[15]);
            kb2[10] = hi_34_1;
            kb2[15] = lo_35_1;
            float hi_36_1 = fmaxf(kb2[11], kb2[12]);
            float lo_37_1 = fminf(kb2[11], kb2[12]);
            kb2[11] = hi_36_1;
            kb2[12] = lo_37_1;
            float hi_38_1 = fmaxf(kb2[0], kb2[1]);
            float lo_39_1 = fminf(kb2[0], kb2[1]);
            kb2[0] = hi_38_1;
            kb2[1] = lo_39_1;
            float hi_41 = fmaxf(kb2[2], kb2[3]);
            float lo_42 = fminf(kb2[2], kb2[3]);
            kb2[2] = hi_41;
            kb2[3] = lo_42;
            float hi_44_1 = fmaxf(kb2[4], kb2[5]);
            float lo_45_1 = fminf(kb2[4], kb2[5]);
            kb2[4] = hi_44_1;
            kb2[5] = lo_45_1;
            float hi_47 = fmaxf(kb2[6], kb2[8]);
            float lo_48 = fminf(kb2[6], kb2[8]);
            kb2[6] = hi_47;
            kb2[8] = lo_48;
            float hi_49 = fmaxf(kb2[7], kb2[9]);
            float lo_50 = fminf(kb2[7], kb2[9]);
            kb2[7] = hi_49;
            kb2[9] = lo_50;
            float hi_51 = fmaxf(kb2[10], kb2[11]);
            float lo_52 = fminf(kb2[10], kb2[11]);
            kb2[10] = hi_51;
            kb2[11] = lo_52;
            float hi_53 = fmaxf(kb2[12], kb2[13]);
            float lo_54 = fminf(kb2[12], kb2[13]);
            kb2[12] = hi_53;
            kb2[13] = lo_54;
            float hi_55 = fmaxf(kb2[14], kb2[15]);
            float lo_56 = fminf(kb2[14], kb2[15]);
            kb2[14] = hi_55;
            kb2[15] = lo_56;
            float hi_57 = fmaxf(kb2[0], kb2[2]);
            float lo_58 = fminf(kb2[0], kb2[2]);
            kb2[0] = hi_57;
            kb2[2] = lo_58;
            float hi_60_1 = fmaxf(kb2[1], kb2[3]);
            float lo_61_1 = fminf(kb2[1], kb2[3]);
            kb2[1] = hi_60_1;
            kb2[3] = lo_61_1;
            float hi_63 = fmaxf(kb2[4], kb2[10]);
            float lo_64 = fminf(kb2[4], kb2[10]);
            kb2[4] = hi_63;
            kb2[10] = lo_64;
            float hi_66_1 = fmaxf(kb2[5], kb2[11]);
            float lo_67_1 = fminf(kb2[5], kb2[11]);
            kb2[5] = hi_66_1;
            kb2[11] = lo_67_1;
            float hi_68_1 = fmaxf(kb2[6], kb2[7]);
            float lo_69_1 = fminf(kb2[6], kb2[7]);
            kb2[6] = hi_68_1;
            kb2[7] = lo_69_1;
            float hi_70_1 = fmaxf(kb2[8], kb2[9]);
            float lo_71_1 = fminf(kb2[8], kb2[9]);
            kb2[8] = hi_70_1;
            kb2[9] = lo_71_1;
            float hi_72_1 = fmaxf(kb2[12], kb2[14]);
            float lo_73_1 = fminf(kb2[12], kb2[14]);
            kb2[12] = hi_72_1;
            kb2[14] = lo_73_1;
            float hi_74_1 = fmaxf(kb2[13], kb2[15]);
            float lo_75_1 = fminf(kb2[13], kb2[15]);
            kb2[13] = hi_74_1;
            kb2[15] = lo_75_1;
            float hi_76_1 = fmaxf(kb2[1], kb2[2]);
            float lo_77_1 = fminf(kb2[1], kb2[2]);
            kb2[1] = hi_76_1;
            kb2[2] = lo_77_1;
            float hi_79 = fmaxf(kb2[3], kb2[12]);
            float lo_80 = fminf(kb2[3], kb2[12]);
            kb2[3] = hi_79;
            kb2[12] = lo_80;
            float hi_82_1 = fmaxf(kb2[4], kb2[6]);
            float lo_83_1 = fminf(kb2[4], kb2[6]);
            kb2[4] = hi_82_1;
            kb2[6] = lo_83_1;
            float hi_85 = fmaxf(kb2[5], kb2[7]);
            float lo_86 = fminf(kb2[5], kb2[7]);
            kb2[5] = hi_85;
            kb2[7] = lo_86;
            float hi_87 = fmaxf(kb2[8], kb2[10]);
            float lo_88 = fminf(kb2[8], kb2[10]);
            kb2[8] = hi_87;
            kb2[10] = lo_88;
            float hi_89 = fmaxf(kb2[9], kb2[11]);
            float lo_90 = fminf(kb2[9], kb2[11]);
            kb2[9] = hi_89;
            kb2[11] = lo_90;
            float hi_91 = fmaxf(kb2[13], kb2[14]);
            float lo_92 = fminf(kb2[13], kb2[14]);
            kb2[13] = hi_91;
            kb2[14] = lo_92;
            float hi_93 = fmaxf(kb2[1], kb2[4]);
            float lo_94 = fminf(kb2[1], kb2[4]);
            kb2[1] = hi_93;
            kb2[4] = lo_94;
            float hi_95 = fmaxf(kb2[2], kb2[6]);
            float lo_96 = fminf(kb2[2], kb2[6]);
            kb2[2] = hi_95;
            kb2[6] = lo_96;
            float hi_98_1 = fmaxf(kb2[5], kb2[8]);
            float lo_99_1 = fminf(kb2[5], kb2[8]);
            kb2[5] = hi_98_1;
            kb2[8] = lo_99_1;
            float hi_101 = fmaxf(kb2[7], kb2[10]);
            float lo_102 = fminf(kb2[7], kb2[10]);
            kb2[7] = hi_101;
            kb2[10] = lo_102;
            float hi_104_1 = fmaxf(kb2[9], kb2[13]);
            float lo_105_1 = fminf(kb2[9], kb2[13]);
            kb2[9] = hi_104_1;
            kb2[13] = lo_105_1;
            float hi_106_1 = fmaxf(kb2[11], kb2[14]);
            float lo_107_1 = fminf(kb2[11], kb2[14]);
            kb2[11] = hi_106_1;
            kb2[14] = lo_107_1;
            float hi_108_1 = fmaxf(kb2[2], kb2[4]);
            float lo_109_1 = fminf(kb2[2], kb2[4]);
            kb2[2] = hi_108_1;
            kb2[4] = lo_109_1;
            float hi_110_1 = fmaxf(kb2[3], kb2[6]);
            float lo_111_1 = fminf(kb2[3], kb2[6]);
            kb2[3] = hi_110_1;
            kb2[6] = lo_111_1;
            float hi_112_1 = fmaxf(kb2[9], kb2[12]);
            float lo_113_1 = fminf(kb2[9], kb2[12]);
            kb2[9] = hi_112_1;
            kb2[12] = lo_113_1;
            float hi_114_1 = fmaxf(kb2[11], kb2[13]);
            float lo_115_1 = fminf(kb2[11], kb2[13]);
            kb2[11] = hi_114_1;
            kb2[13] = lo_115_1;
            float hi_117 = fmaxf(kb2[3], kb2[5]);
            float lo_118 = fminf(kb2[3], kb2[5]);
            kb2[3] = hi_117;
            kb2[5] = lo_118;
            float hi_120_1 = fmaxf(kb2[6], kb2[8]);
            float lo_121_1 = fminf(kb2[6], kb2[8]);
            kb2[6] = hi_120_1;
            kb2[8] = lo_121_1;
            float hi_123 = fmaxf(kb2[7], kb2[9]);
            float lo_124 = fminf(kb2[7], kb2[9]);
            kb2[7] = hi_123;
            kb2[9] = lo_124;
            float hi_125 = fmaxf(kb2[10], kb2[12]);
            float lo_126 = fminf(kb2[10], kb2[12]);
            kb2[10] = hi_125;
            kb2[12] = lo_126;
            float hi_127 = fmaxf(kb2[3], kb2[4]);
            float lo_128 = fminf(kb2[3], kb2[4]);
            kb2[3] = hi_127;
            kb2[4] = lo_128;
            float hi_129 = fmaxf(kb2[5], kb2[6]);
            float lo_130 = fminf(kb2[5], kb2[6]);
            kb2[5] = hi_129;
            kb2[6] = lo_130;
            float hi_131 = fmaxf(kb2[7], kb2[8]);
            float lo_132 = fminf(kb2[7], kb2[8]);
            kb2[7] = hi_131;
            kb2[8] = lo_132;
            float hi_133 = fmaxf(kb2[9], kb2[10]);
            float lo_134 = fminf(kb2[9], kb2[10]);
            kb2[9] = hi_133;
            kb2[10] = lo_134;
            float hi_136_1 = fmaxf(kb2[11], kb2[12]);
            float lo_137_1 = fminf(kb2[11], kb2[12]);
            kb2[11] = hi_136_1;
            kb2[12] = lo_137_1;
            float hi_139 = fmaxf(kb2[6], kb2[7]);
            float lo_140 = fminf(kb2[6], kb2[7]);
            kb2[6] = hi_139;
            kb2[7] = lo_140;
            float hi_142_1 = fmaxf(kb2[8], kb2[9]);
            float lo_143_1 = fminf(kb2[8], kb2[9]);
            kb2[8] = hi_142_1;
            kb2[9] = lo_143_1;
            float hi_144_1 = fmaxf(a2x[0], kb2[15]);
            a2x[0] = hi_144_1;
            float hi_146_1 = fmaxf(a2x[1], kb2[14]);
            a2x[1] = hi_146_1;
            float hi_147 = fmaxf(a2x[2], kb2[13]);
            a2x[2] = hi_147;
            float hi_149 = fmaxf(a2x[3], kb2[12]);
            a2x[3] = hi_149;
            float hi_150_1 = fmaxf(a2x[4], kb2[11]);
            a2x[4] = hi_150_1;
            float hi_152_1 = fmaxf(a2x[5], kb2[10]);
            a2x[5] = hi_152_1;
            float hi_153 = fmaxf(a2x[6], kb2[9]);
            a2x[6] = hi_153;
            float hi_155 = fmaxf(a2x[7], kb2[8]);
            a2x[7] = hi_155;
            float hi_156_1 = fmaxf(a2x[8], kb2[7]);
            a2x[8] = hi_156_1;
            float hi_157 = fmaxf(a2x[9], kb2[6]);
            a2x[9] = hi_157;
            float hi_158_1 = fmaxf(a2x[10], kb2[5]);
            a2x[10] = hi_158_1;
            float hi_159 = fmaxf(a2x[11], kb2[4]);
            a2x[11] = hi_159;
            float hi_160_1 = fmaxf(a2x[12], kb2[3]);
            a2x[12] = hi_160_1;
            float hi_162_1 = fmaxf(a2x[13], kb2[2]);
            a2x[13] = hi_162_1;
            float hi_163 = fmaxf(a2x[14], kb2[1]);
            a2x[14] = hi_163;
            float hi_165 = fmaxf(a2x[15], kb2[0]);
            a2x[15] = hi_165;
            float hi_166_1 = fmaxf(a2x[0], a2x[8]);
            float lo_167_1 = fminf(a2x[0], a2x[8]);
            a2x[0] = hi_166_1;
            a2x[8] = lo_167_1;
            float hi_168_1 = fmaxf(a2x[1], a2x[9]);
            float lo_169_1 = fminf(a2x[1], a2x[9]);
            a2x[1] = hi_168_1;
            a2x[9] = lo_169_1;
            float hi_171 = fmaxf(a2x[2], a2x[10]);
            float lo_172 = fminf(a2x[2], a2x[10]);
            a2x[2] = hi_171;
            a2x[10] = lo_172;
            float hi_173 = fmaxf(a2x[3], a2x[11]);
            float lo_174 = fminf(a2x[3], a2x[11]);
            a2x[3] = hi_173;
            a2x[11] = lo_174;
            float hi_175 = fmaxf(a2x[4], a2x[12]);
            float lo_176 = fminf(a2x[4], a2x[12]);
            a2x[4] = hi_175;
            a2x[12] = lo_176;
            float hi_178_1 = fmaxf(a2x[5], a2x[13]);
            float lo_179_1 = fminf(a2x[5], a2x[13]);
            a2x[5] = hi_178_1;
            a2x[13] = lo_179_1;
            float hi_181 = fmaxf(a2x[6], a2x[14]);
            float lo_182 = fminf(a2x[6], a2x[14]);
            a2x[6] = hi_181;
            a2x[14] = lo_182;
            float hi_184_1 = fmaxf(a2x[7], a2x[15]);
            float lo_185_1 = fminf(a2x[7], a2x[15]);
            a2x[7] = hi_184_1;
            a2x[15] = lo_185_1;
            float hi_187 = fmaxf(a2x[0], a2x[4]);
            float lo_188 = fminf(a2x[0], a2x[4]);
            a2x[0] = hi_187;
            a2x[4] = lo_188;
            float hi_189 = fmaxf(a2x[1], a2x[5]);
            float lo_190 = fminf(a2x[1], a2x[5]);
            a2x[1] = hi_189;
            a2x[5] = lo_190;
            float hi_191 = fmaxf(a2x[2], a2x[6]);
            float lo_192 = fminf(a2x[2], a2x[6]);
            a2x[2] = hi_191;
            a2x[6] = lo_192;
            float hi_194_1 = fmaxf(a2x[3], a2x[7]);
            float lo_195_1 = fminf(a2x[3], a2x[7]);
            a2x[3] = hi_194_1;
            a2x[7] = lo_195_1;
            float hi_197 = fmaxf(a2x[8], a2x[12]);
            float lo_198 = fminf(a2x[8], a2x[12]);
            a2x[8] = hi_197;
            a2x[12] = lo_198;
            float hi_200_1 = fmaxf(a2x[9], a2x[13]);
            float lo_201_1 = fminf(a2x[9], a2x[13]);
            a2x[9] = hi_200_1;
            a2x[13] = lo_201_1;
            float hi_203 = fmaxf(a2x[10], a2x[14]);
            float lo_204 = fminf(a2x[10], a2x[14]);
            a2x[10] = hi_203;
            a2x[14] = lo_204;
            float hi_205 = fmaxf(a2x[11], a2x[15]);
            float lo_206 = fminf(a2x[11], a2x[15]);
            a2x[11] = hi_205;
            a2x[15] = lo_206;
            float hi_207 = fmaxf(a2x[0], a2x[2]);
            float lo_208 = fminf(a2x[0], a2x[2]);
            a2x[0] = hi_207;
            a2x[2] = lo_208;
            float hi_210_1 = fmaxf(a2x[1], a2x[3]);
            float lo_211_1 = fminf(a2x[1], a2x[3]);
            a2x[1] = hi_210_1;
            a2x[3] = lo_211_1;
            float hi_213 = fmaxf(a2x[4], a2x[6]);
            float lo_214 = fminf(a2x[4], a2x[6]);
            a2x[4] = hi_213;
            a2x[6] = lo_214;
            float hi_216_1 = fmaxf(a2x[5], a2x[7]);
            float lo_217_1 = fminf(a2x[5], a2x[7]);
            a2x[5] = hi_216_1;
            a2x[7] = lo_217_1;
            float hi_219 = fmaxf(a2x[8], a2x[10]);
            float lo_220 = fminf(a2x[8], a2x[10]);
            a2x[8] = hi_219;
            a2x[10] = lo_220;
            float hi_221 = fmaxf(a2x[9], a2x[11]);
            float lo_222 = fminf(a2x[9], a2x[11]);
            a2x[9] = hi_221;
            a2x[11] = lo_222;
            float hi_223 = fmaxf(a2x[12], a2x[14]);
            float lo_224 = fminf(a2x[12], a2x[14]);
            a2x[12] = hi_223;
            a2x[14] = lo_224;
            float hi_226_1 = fmaxf(a2x[13], a2x[15]);
            float lo_227_1 = fminf(a2x[13], a2x[15]);
            a2x[13] = hi_226_1;
            a2x[15] = lo_227_1;
            float hi_229 = fmaxf(a2x[0], a2x[1]);
            float lo_230 = fminf(a2x[0], a2x[1]);
            a2x[0] = hi_229;
            a2x[1] = lo_230;
            float hi_232_1 = fmaxf(a2x[2], a2x[3]);
            float lo_233_1 = fminf(a2x[2], a2x[3]);
            a2x[2] = hi_232_1;
            a2x[3] = lo_233_1;
            float hi_235 = fmaxf(a2x[4], a2x[5]);
            float lo_236 = fminf(a2x[4], a2x[5]);
            a2x[4] = hi_235;
            a2x[5] = lo_236;
            float hi_237 = fmaxf(a2x[6], a2x[7]);
            float lo_238 = fminf(a2x[6], a2x[7]);
            a2x[6] = hi_237;
            a2x[7] = lo_238;
            float hi_239 = fmaxf(a2x[8], a2x[9]);
            float lo_240 = fminf(a2x[8], a2x[9]);
            a2x[8] = hi_239;
            a2x[9] = lo_240;
            float hi_241 = fmaxf(a2x[10], a2x[11]);
            float lo_242 = fminf(a2x[10], a2x[11]);
            a2x[10] = hi_241;
            a2x[11] = lo_242;
            float hi_243 = fmaxf(a2x[12], a2x[13]);
            float lo_244 = fminf(a2x[12], a2x[13]);
            a2x[12] = hi_243;
            a2x[13] = lo_244;
            float hi_245 = fmaxf(a2x[14], a2x[15]);
            float lo_246 = fminf(a2x[14], a2x[15]);
            a2x[14] = hi_245;
            a2x[15] = lo_246;
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
      asm volatile("barrier.sync 1, 512;" ::: "memory");
      float r_1_1 = neg_inf;
      float V_2[8];
      int s0_3 = (sg * 16 * 32 + cg) * 17;
      int s1_4 = ((sg * 16 + 8) * 32 + cg) * 17;
      float x0_5 = pub[s0_3 + ln];
      float y0_6 = pub[s1_4 + lnr];
      float lo0_7 = fminf(x0_5, y0_6);
      r_1_1 = fmaxf(r_1_1, lo0_7);
      float hi0_8 = fmaxf(x0_5, y0_6);
      float cur_9 = hi0_8;
      float pv_10 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 8);
      float hi_11_1 = fmaxf(cur_9, pv_10);
      float lo_12_1 = fminf(cur_9, pv_10);
      cur_9 = ((up[0] != 0) ? hi_11_1 : lo_12_1);
      float pv_13 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 4);
      float hi_14 = fmaxf(cur_9, pv_13);
      float lo_15 = fminf(cur_9, pv_13);
      cur_9 = ((up[1] != 0) ? hi_14 : lo_15);
      float pv_16 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 2);
      float hi_17_1 = fmaxf(cur_9, pv_16);
      float lo_18_1 = fminf(cur_9, pv_16);
      cur_9 = ((up[2] != 0) ? hi_17_1 : lo_18_1);
      float pv_19 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 1);
      float hi_20 = fmaxf(cur_9, pv_19);
      float lo_21 = fminf(cur_9, pv_19);
      cur_9 = ((up[3] != 0) ? hi_20 : lo_21);
      V_2[0] = cur_9;
      int s0_22 = ((sg * 16 + 1) * 32 + cg) * 17;
      int s1_23 = ((sg * 16 + 1 + 8) * 32 + cg) * 17;
      float x0_24 = pub[s0_22 + ln];
      float y0_25 = pub[s1_23 + lnr];
      float lo0_26 = fminf(x0_24, y0_25);
      r_1_1 = fmaxf(r_1_1, lo0_26);
      float hi0_27 = fmaxf(x0_24, y0_25);
      float cur_28 = hi0_27;
      float pv_29 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 8);
      float hi_30_1 = fmaxf(cur_28, pv_29);
      float lo_31_1 = fminf(cur_28, pv_29);
      cur_28 = ((up[0] != 0) ? hi_30_1 : lo_31_1);
      float pv_32 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 4);
      float hi_33 = fmaxf(cur_28, pv_32);
      float lo_34 = fminf(cur_28, pv_32);
      cur_28 = ((up[1] != 0) ? hi_33 : lo_34);
      float pv_35 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 2);
      float hi_36_2 = fmaxf(cur_28, pv_35);
      float lo_37_2 = fminf(cur_28, pv_35);
      cur_28 = ((up[2] != 0) ? hi_36_2 : lo_37_2);
      float pv_38 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 1);
      float hi_39 = fmaxf(cur_28, pv_38);
      float lo_40 = fminf(cur_28, pv_38);
      cur_28 = ((up[3] != 0) ? hi_39 : lo_40);
      V_2[1] = cur_28;
      int s0_41 = ((sg * 16 + 2) * 32 + cg) * 17;
      int s1_42 = ((sg * 16 + 2 + 8) * 32 + cg) * 17;
      float x0_43 = pub[s0_41 + ln];
      float y0_44 = pub[s1_42 + lnr];
      float lo0_45 = fminf(x0_43, y0_44);
      r_1_1 = fmaxf(r_1_1, lo0_45);
      float hi0_46 = fmaxf(x0_43, y0_44);
      float cur_47 = hi0_46;
      float pv_48 = __shfl_xor_sync(0xFFFFFFFF, cur_47, 8);
      float hi_49_1 = fmaxf(cur_47, pv_48);
      float lo_50_1 = fminf(cur_47, pv_48);
      cur_47 = ((up[0] != 0) ? hi_49_1 : lo_50_1);
      float pv_51 = __shfl_xor_sync(0xFFFFFFFF, cur_47, 4);
      float hi_52_1 = fmaxf(cur_47, pv_51);
      float lo_53_1 = fminf(cur_47, pv_51);
      cur_47 = ((up[1] != 0) ? hi_52_1 : lo_53_1);
      float pv_54 = __shfl_xor_sync(0xFFFFFFFF, cur_47, 2);
      float hi_55_1 = fmaxf(cur_47, pv_54);
      float lo_56_1 = fminf(cur_47, pv_54);
      cur_47 = ((up[2] != 0) ? hi_55_1 : lo_56_1);
      float pv_57 = __shfl_xor_sync(0xFFFFFFFF, cur_47, 1);
      float hi_58_1 = fmaxf(cur_47, pv_57);
      float lo_59_1 = fminf(cur_47, pv_57);
      cur_47 = ((up[3] != 0) ? hi_58_1 : lo_59_1);
      V_2[2] = cur_47;
      int s0_60 = ((sg * 16 + 3) * 32 + cg) * 17;
      int s1_61 = ((sg * 16 + 3 + 8) * 32 + cg) * 17;
      float x0_62 = pub[s0_60 + ln];
      float y0_63 = pub[s1_61 + lnr];
      float lo0_64 = fminf(x0_62, y0_63);
      r_1_1 = fmaxf(r_1_1, lo0_64);
      float hi0_65 = fmaxf(x0_62, y0_63);
      float cur_66 = hi0_65;
      float pv_67 = __shfl_xor_sync(0xFFFFFFFF, cur_66, 8);
      float hi_68_2 = fmaxf(cur_66, pv_67);
      float lo_69_2 = fminf(cur_66, pv_67);
      cur_66 = ((up[0] != 0) ? hi_68_2 : lo_69_2);
      float pv_70 = __shfl_xor_sync(0xFFFFFFFF, cur_66, 4);
      float hi_71 = fmaxf(cur_66, pv_70);
      float lo_72 = fminf(cur_66, pv_70);
      cur_66 = ((up[1] != 0) ? hi_71 : lo_72);
      float pv_73 = __shfl_xor_sync(0xFFFFFFFF, cur_66, 2);
      float hi_74_2 = fmaxf(cur_66, pv_73);
      float lo_75_2 = fminf(cur_66, pv_73);
      cur_66 = ((up[2] != 0) ? hi_74_2 : lo_75_2);
      float pv_76 = __shfl_xor_sync(0xFFFFFFFF, cur_66, 1);
      float hi_77 = fmaxf(cur_66, pv_76);
      float lo_78 = fminf(cur_66, pv_76);
      cur_66 = ((up[3] != 0) ? hi_77 : lo_78);
      V_2[3] = cur_66;
      int s0_79 = ((sg * 16 + 4) * 32 + cg) * 17;
      int s1_80 = ((sg * 16 + 4 + 8) * 32 + cg) * 17;
      float x0_81 = pub[s0_79 + ln];
      float y0_82 = pub[s1_80 + lnr];
      float lo0_83 = fminf(x0_81, y0_82);
      r_1_1 = fmaxf(r_1_1, lo0_83);
      float hi0_84 = fmaxf(x0_81, y0_82);
      float cur_85 = hi0_84;
      float pv_86 = __shfl_xor_sync(0xFFFFFFFF, cur_85, 8);
      float hi_87_1 = fmaxf(cur_85, pv_86);
      float lo_88_1 = fminf(cur_85, pv_86);
      cur_85 = ((up[0] != 0) ? hi_87_1 : lo_88_1);
      float pv_89 = __shfl_xor_sync(0xFFFFFFFF, cur_85, 4);
      float hi_90_1 = fmaxf(cur_85, pv_89);
      float lo_91_1 = fminf(cur_85, pv_89);
      cur_85 = ((up[1] != 0) ? hi_90_1 : lo_91_1);
      float pv_92 = __shfl_xor_sync(0xFFFFFFFF, cur_85, 2);
      float hi_93_1 = fmaxf(cur_85, pv_92);
      float lo_94_1 = fminf(cur_85, pv_92);
      cur_85 = ((up[2] != 0) ? hi_93_1 : lo_94_1);
      float pv_95 = __shfl_xor_sync(0xFFFFFFFF, cur_85, 1);
      float hi_96_1 = fmaxf(cur_85, pv_95);
      float lo_97_1 = fminf(cur_85, pv_95);
      cur_85 = ((up[3] != 0) ? hi_96_1 : lo_97_1);
      V_2[4] = cur_85;
      int s0_98 = ((sg * 16 + 5) * 32 + cg) * 17;
      int s1_99 = ((sg * 16 + 5 + 8) * 32 + cg) * 17;
      float x0_100 = pub[s0_98 + ln];
      float y0_101 = pub[s1_99 + lnr];
      float lo0_102 = fminf(x0_100, y0_101);
      r_1_1 = fmaxf(r_1_1, lo0_102);
      float hi0_103 = fmaxf(x0_100, y0_101);
      float cur_104 = hi0_103;
      float pv_105 = __shfl_xor_sync(0xFFFFFFFF, cur_104, 8);
      float hi_106_2 = fmaxf(cur_104, pv_105);
      float lo_107_2 = fminf(cur_104, pv_105);
      cur_104 = ((up[0] != 0) ? hi_106_2 : lo_107_2);
      float pv_108 = __shfl_xor_sync(0xFFFFFFFF, cur_104, 4);
      float hi_109 = fmaxf(cur_104, pv_108);
      float lo_110 = fminf(cur_104, pv_108);
      cur_104 = ((up[1] != 0) ? hi_109 : lo_110);
      float pv_111 = __shfl_xor_sync(0xFFFFFFFF, cur_104, 2);
      float hi_112_2 = fmaxf(cur_104, pv_111);
      float lo_113_2 = fminf(cur_104, pv_111);
      cur_104 = ((up[2] != 0) ? hi_112_2 : lo_113_2);
      float pv_114 = __shfl_xor_sync(0xFFFFFFFF, cur_104, 1);
      float hi_115 = fmaxf(cur_104, pv_114);
      float lo_116 = fminf(cur_104, pv_114);
      cur_104 = ((up[3] != 0) ? hi_115 : lo_116);
      V_2[5] = cur_104;
      int s0_117 = ((sg * 16 + 6) * 32 + cg) * 17;
      int s1_118 = ((sg * 16 + 6 + 8) * 32 + cg) * 17;
      float x0_119 = pub[s0_117 + ln];
      float y0_120 = pub[s1_118 + lnr];
      float lo0_121 = fminf(x0_119, y0_120);
      r_1_1 = fmaxf(r_1_1, lo0_121);
      float hi0_122 = fmaxf(x0_119, y0_120);
      float cur_123 = hi0_122;
      float pv_124 = __shfl_xor_sync(0xFFFFFFFF, cur_123, 8);
      float hi_125_1 = fmaxf(cur_123, pv_124);
      float lo_126_1 = fminf(cur_123, pv_124);
      cur_123 = ((up[0] != 0) ? hi_125_1 : lo_126_1);
      float pv_127 = __shfl_xor_sync(0xFFFFFFFF, cur_123, 4);
      float hi_128_1 = fmaxf(cur_123, pv_127);
      float lo_129_1 = fminf(cur_123, pv_127);
      cur_123 = ((up[1] != 0) ? hi_128_1 : lo_129_1);
      float pv_130 = __shfl_xor_sync(0xFFFFFFFF, cur_123, 2);
      float hi_131_1 = fmaxf(cur_123, pv_130);
      float lo_132_1 = fminf(cur_123, pv_130);
      cur_123 = ((up[2] != 0) ? hi_131_1 : lo_132_1);
      float pv_133 = __shfl_xor_sync(0xFFFFFFFF, cur_123, 1);
      float hi_134_1 = fmaxf(cur_123, pv_133);
      float lo_135_1 = fminf(cur_123, pv_133);
      cur_123 = ((up[3] != 0) ? hi_134_1 : lo_135_1);
      V_2[6] = cur_123;
      int s0_136 = ((sg * 16 + 7) * 32 + cg) * 17;
      int s1_137 = ((sg * 16 + 7 + 8) * 32 + cg) * 17;
      float x0_138 = pub[s0_136 + ln];
      float y0_139 = pub[s1_137 + lnr];
      float lo0_140 = fminf(x0_138, y0_139);
      r_1_1 = fmaxf(r_1_1, lo0_140);
      float hi0_141 = fmaxf(x0_138, y0_139);
      float cur_142 = hi0_141;
      float pv_143 = __shfl_xor_sync(0xFFFFFFFF, cur_142, 8);
      float hi_144_2 = fmaxf(cur_142, pv_143);
      float lo_145_1 = fminf(cur_142, pv_143);
      cur_142 = ((up[0] != 0) ? hi_144_2 : lo_145_1);
      float pv_146 = __shfl_xor_sync(0xFFFFFFFF, cur_142, 4);
      float hi_147_1 = fmaxf(cur_142, pv_146);
      float lo_148 = fminf(cur_142, pv_146);
      cur_142 = ((up[1] != 0) ? hi_147_1 : lo_148);
      float pv_149 = __shfl_xor_sync(0xFFFFFFFF, cur_142, 2);
      float hi_150_2 = fmaxf(cur_142, pv_149);
      float lo_151_1 = fminf(cur_142, pv_149);
      cur_142 = ((up[2] != 0) ? hi_150_2 : lo_151_1);
      float pv_152 = __shfl_xor_sync(0xFFFFFFFF, cur_142, 1);
      float hi_153_1 = fmaxf(cur_142, pv_152);
      float lo_154 = fminf(cur_142, pv_152);
      cur_142 = ((up[3] != 0) ? hi_153_1 : lo_154);
      V_2[7] = cur_142;
      float rs_155 = pub[((sg * 16 + ln) * 32 + cg) * 17 + 16];
      r_1_1 = fmaxf(r_1_1, rs_155);
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
      float lo_166 = fminf(cur_160, pv_164);
      cur_160 = ((up[1] != 0) ? hi_165_1 : lo_166);
      float pv_167 = __shfl_xor_sync(0xFFFFFFFF, cur_160, 2);
      float hi_168_2 = fmaxf(cur_160, pv_167);
      float lo_169_2 = fminf(cur_160, pv_167);
      cur_160 = ((up[2] != 0) ? hi_168_2 : lo_169_2);
      float pv_170 = __shfl_xor_sync(0xFFFFFFFF, cur_160, 1);
      float hi_171_1 = fmaxf(cur_160, pv_170);
      float lo_172_1 = fminf(cur_160, pv_170);
      cur_160 = ((up[3] != 0) ? hi_171_1 : lo_172_1);
      V_2[0] = cur_160;
      float y1_173 = __shfl_xor_sync(0xFFFFFFFF, V_2[5], 15);
      float lo1_174 = fminf(V_2[1], y1_173);
      r_1_1 = fmaxf(r_1_1, lo1_174);
      float hi1_175 = fmaxf(V_2[1], y1_173);
      float cur_176 = hi1_175;
      float pv_177 = __shfl_xor_sync(0xFFFFFFFF, cur_176, 8);
      float hi_178_2 = fmaxf(cur_176, pv_177);
      float lo_179_2 = fminf(cur_176, pv_177);
      cur_176 = ((up[0] != 0) ? hi_178_2 : lo_179_2);
      float pv_180 = __shfl_xor_sync(0xFFFFFFFF, cur_176, 4);
      float hi_181_1 = fmaxf(cur_176, pv_180);
      float lo_182_1 = fminf(cur_176, pv_180);
      cur_176 = ((up[1] != 0) ? hi_181_1 : lo_182_1);
      float pv_183 = __shfl_xor_sync(0xFFFFFFFF, cur_176, 2);
      float hi_184_2 = fmaxf(cur_176, pv_183);
      float lo_185_2 = fminf(cur_176, pv_183);
      cur_176 = ((up[2] != 0) ? hi_184_2 : lo_185_2);
      float pv_186 = __shfl_xor_sync(0xFFFFFFFF, cur_176, 1);
      float hi_187_1 = fmaxf(cur_176, pv_186);
      float lo_188_1 = fminf(cur_176, pv_186);
      cur_176 = ((up[3] != 0) ? hi_187_1 : lo_188_1);
      V_2[1] = cur_176;
      float y1_189 = __shfl_xor_sync(0xFFFFFFFF, V_2[6], 15);
      float lo1_190 = fminf(V_2[2], y1_189);
      r_1_1 = fmaxf(r_1_1, lo1_190);
      float hi1_191 = fmaxf(V_2[2], y1_189);
      float cur_192 = hi1_191;
      float pv_193 = __shfl_xor_sync(0xFFFFFFFF, cur_192, 8);
      float hi_194_2 = fmaxf(cur_192, pv_193);
      float lo_195_2 = fminf(cur_192, pv_193);
      cur_192 = ((up[0] != 0) ? hi_194_2 : lo_195_2);
      float pv_196 = __shfl_xor_sync(0xFFFFFFFF, cur_192, 4);
      float hi_197_1 = fmaxf(cur_192, pv_196);
      float lo_198_1 = fminf(cur_192, pv_196);
      cur_192 = ((up[1] != 0) ? hi_197_1 : lo_198_1);
      float pv_199 = __shfl_xor_sync(0xFFFFFFFF, cur_192, 2);
      float hi_200_2 = fmaxf(cur_192, pv_199);
      float lo_201_2 = fminf(cur_192, pv_199);
      cur_192 = ((up[2] != 0) ? hi_200_2 : lo_201_2);
      float pv_202 = __shfl_xor_sync(0xFFFFFFFF, cur_192, 1);
      float hi_203_1 = fmaxf(cur_192, pv_202);
      float lo_204_1 = fminf(cur_192, pv_202);
      cur_192 = ((up[3] != 0) ? hi_203_1 : lo_204_1);
      V_2[2] = cur_192;
      float y1_205 = __shfl_xor_sync(0xFFFFFFFF, V_2[7], 15);
      float lo1_206 = fminf(V_2[3], y1_205);
      r_1_1 = fmaxf(r_1_1, lo1_206);
      float hi1_207 = fmaxf(V_2[3], y1_205);
      float cur_208 = hi1_207;
      float pv_209 = __shfl_xor_sync(0xFFFFFFFF, cur_208, 8);
      float hi_210_2 = fmaxf(cur_208, pv_209);
      float lo_211_2 = fminf(cur_208, pv_209);
      cur_208 = ((up[0] != 0) ? hi_210_2 : lo_211_2);
      float pv_212 = __shfl_xor_sync(0xFFFFFFFF, cur_208, 4);
      float hi_213_1 = fmaxf(cur_208, pv_212);
      float lo_214_1 = fminf(cur_208, pv_212);
      cur_208 = ((up[1] != 0) ? hi_213_1 : lo_214_1);
      float pv_215 = __shfl_xor_sync(0xFFFFFFFF, cur_208, 2);
      float hi_216_2 = fmaxf(cur_208, pv_215);
      float lo_217_2 = fminf(cur_208, pv_215);
      cur_208 = ((up[2] != 0) ? hi_216_2 : lo_217_2);
      float pv_218 = __shfl_xor_sync(0xFFFFFFFF, cur_208, 1);
      float hi_219_1 = fmaxf(cur_208, pv_218);
      float lo_220_1 = fminf(cur_208, pv_218);
      cur_208 = ((up[3] != 0) ? hi_219_1 : lo_220_1);
      V_2[3] = cur_208;
      float y1_221 = __shfl_xor_sync(0xFFFFFFFF, V_2[2], 15);
      float lo1_222 = fminf(V_2[0], y1_221);
      r_1_1 = fmaxf(r_1_1, lo1_222);
      float hi1_223 = fmaxf(V_2[0], y1_221);
      float cur_224 = hi1_223;
      float pv_225 = __shfl_xor_sync(0xFFFFFFFF, cur_224, 8);
      float hi_226_2 = fmaxf(cur_224, pv_225);
      float lo_227_2 = fminf(cur_224, pv_225);
      cur_224 = ((up[0] != 0) ? hi_226_2 : lo_227_2);
      float pv_228 = __shfl_xor_sync(0xFFFFFFFF, cur_224, 4);
      float hi_229_1 = fmaxf(cur_224, pv_228);
      float lo_230_1 = fminf(cur_224, pv_228);
      cur_224 = ((up[1] != 0) ? hi_229_1 : lo_230_1);
      float pv_231 = __shfl_xor_sync(0xFFFFFFFF, cur_224, 2);
      float hi_232_2 = fmaxf(cur_224, pv_231);
      float lo_233_2 = fminf(cur_224, pv_231);
      cur_224 = ((up[2] != 0) ? hi_232_2 : lo_233_2);
      float pv_234 = __shfl_xor_sync(0xFFFFFFFF, cur_224, 1);
      float hi_235_1 = fmaxf(cur_224, pv_234);
      float lo_236_1 = fminf(cur_224, pv_234);
      cur_224 = ((up[3] != 0) ? hi_235_1 : lo_236_1);
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
      float hi_245_1 = fmaxf(cur_240, pv_244);
      float lo_246_1 = fminf(cur_240, pv_244);
      cur_240 = ((up[1] != 0) ? hi_245_1 : lo_246_1);
      float pv_247 = __shfl_xor_sync(0xFFFFFFFF, cur_240, 2);
      float hi_248 = fmaxf(cur_240, pv_247);
      float lo_249 = fminf(cur_240, pv_247);
      cur_240 = ((up[2] != 0) ? hi_248 : lo_249);
      float pv_250 = __shfl_xor_sync(0xFFFFFFFF, cur_240, 1);
      float hi_251 = fmaxf(cur_240, pv_250);
      float lo_252 = fminf(cur_240, pv_250);
      cur_240 = ((up[3] != 0) ? hi_251 : lo_252);
      V_2[1] = cur_240;
      float yl_253 = __shfl_xor_sync(0xFFFFFFFF, V_2[1], 15);
      float lol_254 = fminf(V_2[0], yl_253);
      r_1_1 = fmaxf(r_1_1, lol_254);
      float hil_255 = fmaxf(V_2[0], yl_253);
      V_2[0] = hil_255;
      float K_256 = V_2[0];
      rr2x[0] = r_1_1;
      K2 = K_256;
    }
    if (tid_1 < 512) {
      unsigned int cgx = ccnt[cg];
      unsigned int k1x = __as_u32(K);
      int isq = 0;
      if (gflag != 0 && (k1x & 4294965248u) == qg) {
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
          float hi_0_1 = fmaxf(xs[0], xs[13]);
          float lo_1_2 = fminf(xs[0], xs[13]);
          xs[0] = hi_0_1;
          xs[13] = lo_1_2;
          float hi_3_1 = fmaxf(xs[1], xs[12]);
          float lo_4_1 = fminf(xs[1], xs[12]);
          xs[1] = hi_3_1;
          xs[12] = lo_4_1;
          float hi_6_1 = fmaxf(xs[2], xs[15]);
          float lo_7_1 = fminf(xs[2], xs[15]);
          xs[2] = hi_6_1;
          xs[15] = lo_7_1;
          float hi_9_1 = fmaxf(xs[3], xs[14]);
          float lo_10_1 = fminf(xs[3], xs[14]);
          xs[3] = hi_9_1;
          xs[14] = lo_10_1;
          float hi_11_2 = fmaxf(xs[4], xs[8]);
          float lo_12_2 = fminf(xs[4], xs[8]);
          xs[4] = hi_11_2;
          xs[8] = lo_12_2;
          float hi_13_1 = fmaxf(xs[5], xs[6]);
          float lo_14_1 = fminf(xs[5], xs[6]);
          xs[5] = hi_13_1;
          xs[6] = lo_14_1;
          float hi_15_1 = fmaxf(xs[7], xs[11]);
          float lo_16_1 = fminf(xs[7], xs[11]);
          xs[7] = hi_15_1;
          xs[11] = lo_16_1;
          float hi_17_2 = fmaxf(xs[9], xs[10]);
          float lo_18_2 = fminf(xs[9], xs[10]);
          xs[9] = hi_17_2;
          xs[10] = lo_18_2;
          float hi_19_1 = fmaxf(xs[0], xs[5]);
          float lo_20_1 = fminf(xs[0], xs[5]);
          xs[0] = hi_19_1;
          xs[5] = lo_20_1;
          float hi_22_1 = fmaxf(xs[1], xs[7]);
          float lo_23_1 = fminf(xs[1], xs[7]);
          xs[1] = hi_22_1;
          xs[7] = lo_23_1;
          float hi_25_1 = fmaxf(xs[2], xs[9]);
          float lo_26_1 = fminf(xs[2], xs[9]);
          xs[2] = hi_25_1;
          xs[9] = lo_26_1;
          float hi_28_1 = fmaxf(xs[3], xs[4]);
          float lo_29_1 = fminf(xs[3], xs[4]);
          xs[3] = hi_28_1;
          xs[4] = lo_29_1;
          float hi_30_2 = fmaxf(xs[6], xs[13]);
          float lo_31_2 = fminf(xs[6], xs[13]);
          xs[6] = hi_30_2;
          xs[13] = lo_31_2;
          float hi_32_1 = fmaxf(xs[8], xs[14]);
          float lo_33_1 = fminf(xs[8], xs[14]);
          xs[8] = hi_32_1;
          xs[14] = lo_33_1;
          float hi_34_2 = fmaxf(xs[10], xs[15]);
          float lo_35_2 = fminf(xs[10], xs[15]);
          xs[10] = hi_34_2;
          xs[15] = lo_35_2;
          float hi_36_3 = fmaxf(xs[11], xs[12]);
          float lo_37_3 = fminf(xs[11], xs[12]);
          xs[11] = hi_36_3;
          xs[12] = lo_37_3;
          float hi_38_2 = fmaxf(xs[0], xs[1]);
          float lo_39_2 = fminf(xs[0], xs[1]);
          xs[0] = hi_38_2;
          xs[1] = lo_39_2;
          float hi_41_1 = fmaxf(xs[2], xs[3]);
          float lo_42_1 = fminf(xs[2], xs[3]);
          xs[2] = hi_41_1;
          xs[3] = lo_42_1;
          float hi_44_2 = fmaxf(xs[4], xs[5]);
          float lo_45_2 = fminf(xs[4], xs[5]);
          xs[4] = hi_44_2;
          xs[5] = lo_45_2;
          float hi_47_1 = fmaxf(xs[6], xs[8]);
          float lo_48_1 = fminf(xs[6], xs[8]);
          xs[6] = hi_47_1;
          xs[8] = lo_48_1;
          float hi_49_2 = fmaxf(xs[7], xs[9]);
          float lo_50_2 = fminf(xs[7], xs[9]);
          xs[7] = hi_49_2;
          xs[9] = lo_50_2;
          float hi_51_1 = fmaxf(xs[10], xs[11]);
          float lo_52_1 = fminf(xs[10], xs[11]);
          xs[10] = hi_51_1;
          xs[11] = lo_52_1;
          float hi_53_1 = fmaxf(xs[12], xs[13]);
          float lo_54_1 = fminf(xs[12], xs[13]);
          xs[12] = hi_53_1;
          xs[13] = lo_54_1;
          float hi_55_2 = fmaxf(xs[14], xs[15]);
          float lo_56_2 = fminf(xs[14], xs[15]);
          xs[14] = hi_55_2;
          xs[15] = lo_56_2;
          float hi_57_1 = fmaxf(xs[0], xs[2]);
          float lo_58_1 = fminf(xs[0], xs[2]);
          xs[0] = hi_57_1;
          xs[2] = lo_58_1;
          float hi_60_2 = fmaxf(xs[1], xs[3]);
          float lo_61_2 = fminf(xs[1], xs[3]);
          xs[1] = hi_60_2;
          xs[3] = lo_61_2;
          float hi_63_1 = fmaxf(xs[4], xs[10]);
          float lo_64_1 = fminf(xs[4], xs[10]);
          xs[4] = hi_63_1;
          xs[10] = lo_64_1;
          float hi_66_2 = fmaxf(xs[5], xs[11]);
          float lo_67_2 = fminf(xs[5], xs[11]);
          xs[5] = hi_66_2;
          xs[11] = lo_67_2;
          float hi_68_3 = fmaxf(xs[6], xs[7]);
          float lo_69_3 = fminf(xs[6], xs[7]);
          xs[6] = hi_68_3;
          xs[7] = lo_69_3;
          float hi_70_2 = fmaxf(xs[8], xs[9]);
          float lo_71_2 = fminf(xs[8], xs[9]);
          xs[8] = hi_70_2;
          xs[9] = lo_71_2;
          float hi_72_2 = fmaxf(xs[12], xs[14]);
          float lo_73_2 = fminf(xs[12], xs[14]);
          xs[12] = hi_72_2;
          xs[14] = lo_73_2;
          float hi_74_3 = fmaxf(xs[13], xs[15]);
          float lo_75_3 = fminf(xs[13], xs[15]);
          xs[13] = hi_74_3;
          xs[15] = lo_75_3;
          float hi_76_2 = fmaxf(xs[1], xs[2]);
          float lo_77_2 = fminf(xs[1], xs[2]);
          xs[1] = hi_76_2;
          xs[2] = lo_77_2;
          float hi_79_1 = fmaxf(xs[3], xs[12]);
          float lo_80_1 = fminf(xs[3], xs[12]);
          xs[3] = hi_79_1;
          xs[12] = lo_80_1;
          float hi_82_2 = fmaxf(xs[4], xs[6]);
          float lo_83_2 = fminf(xs[4], xs[6]);
          xs[4] = hi_82_2;
          xs[6] = lo_83_2;
          float hi_85_1 = fmaxf(xs[5], xs[7]);
          float lo_86_1 = fminf(xs[5], xs[7]);
          xs[5] = hi_85_1;
          xs[7] = lo_86_1;
          float hi_87_2 = fmaxf(xs[8], xs[10]);
          float lo_88_2 = fminf(xs[8], xs[10]);
          xs[8] = hi_87_2;
          xs[10] = lo_88_2;
          float hi_89_1 = fmaxf(xs[9], xs[11]);
          float lo_90_1 = fminf(xs[9], xs[11]);
          xs[9] = hi_89_1;
          xs[11] = lo_90_1;
          float hi_91_1 = fmaxf(xs[13], xs[14]);
          float lo_92_1 = fminf(xs[13], xs[14]);
          xs[13] = hi_91_1;
          xs[14] = lo_92_1;
          float hi_93_2 = fmaxf(xs[1], xs[4]);
          float lo_94_2 = fminf(xs[1], xs[4]);
          xs[1] = hi_93_2;
          xs[4] = lo_94_2;
          float hi_95_1 = fmaxf(xs[2], xs[6]);
          float lo_96_1 = fminf(xs[2], xs[6]);
          xs[2] = hi_95_1;
          xs[6] = lo_96_1;
          float hi_98_2 = fmaxf(xs[5], xs[8]);
          float lo_99_2 = fminf(xs[5], xs[8]);
          xs[5] = hi_98_2;
          xs[8] = lo_99_2;
          float hi_101_1 = fmaxf(xs[7], xs[10]);
          float lo_102_1 = fminf(xs[7], xs[10]);
          xs[7] = hi_101_1;
          xs[10] = lo_102_1;
          float hi_104_2 = fmaxf(xs[9], xs[13]);
          float lo_105_2 = fminf(xs[9], xs[13]);
          xs[9] = hi_104_2;
          xs[13] = lo_105_2;
          float hi_106_3 = fmaxf(xs[11], xs[14]);
          float lo_107_3 = fminf(xs[11], xs[14]);
          xs[11] = hi_106_3;
          xs[14] = lo_107_3;
          float hi_108_2 = fmaxf(xs[2], xs[4]);
          float lo_109_2 = fminf(xs[2], xs[4]);
          xs[2] = hi_108_2;
          xs[4] = lo_109_2;
          float hi_110_2 = fmaxf(xs[3], xs[6]);
          float lo_111_2 = fminf(xs[3], xs[6]);
          xs[3] = hi_110_2;
          xs[6] = lo_111_2;
          float hi_112_3 = fmaxf(xs[9], xs[12]);
          float lo_113_3 = fminf(xs[9], xs[12]);
          xs[9] = hi_112_3;
          xs[12] = lo_113_3;
          float hi_114_2 = fmaxf(xs[11], xs[13]);
          float lo_115_2 = fminf(xs[11], xs[13]);
          xs[11] = hi_114_2;
          xs[13] = lo_115_2;
          float hi_117_1 = fmaxf(xs[3], xs[5]);
          float lo_118_1 = fminf(xs[3], xs[5]);
          xs[3] = hi_117_1;
          xs[5] = lo_118_1;
          float hi_120_2 = fmaxf(xs[6], xs[8]);
          float lo_121_2 = fminf(xs[6], xs[8]);
          xs[6] = hi_120_2;
          xs[8] = lo_121_2;
          float hi_123_1 = fmaxf(xs[7], xs[9]);
          float lo_124_1 = fminf(xs[7], xs[9]);
          xs[7] = hi_123_1;
          xs[9] = lo_124_1;
          float hi_125_2 = fmaxf(xs[10], xs[12]);
          float lo_126_2 = fminf(xs[10], xs[12]);
          xs[10] = hi_125_2;
          xs[12] = lo_126_2;
          float hi_127_1 = fmaxf(xs[3], xs[4]);
          float lo_128_1 = fminf(xs[3], xs[4]);
          xs[3] = hi_127_1;
          xs[4] = lo_128_1;
          float hi_129_1 = fmaxf(xs[5], xs[6]);
          float lo_130_1 = fminf(xs[5], xs[6]);
          xs[5] = hi_129_1;
          xs[6] = lo_130_1;
          float hi_131_2 = fmaxf(xs[7], xs[8]);
          float lo_132_2 = fminf(xs[7], xs[8]);
          xs[7] = hi_131_2;
          xs[8] = lo_132_2;
          float hi_133_1 = fmaxf(xs[9], xs[10]);
          float lo_134_1 = fminf(xs[9], xs[10]);
          xs[9] = hi_133_1;
          xs[10] = lo_134_1;
          float hi_136_2 = fmaxf(xs[11], xs[12]);
          float lo_137_2 = fminf(xs[11], xs[12]);
          xs[11] = hi_136_2;
          xs[12] = lo_137_2;
          float hi_139_1 = fmaxf(xs[6], xs[7]);
          float lo_140_1 = fminf(xs[6], xs[7]);
          xs[6] = hi_139_1;
          xs[7] = lo_140_1;
          float hi_142_2 = fmaxf(xs[8], xs[9]);
          float lo_143_2 = fminf(xs[8], xs[9]);
          xs[8] = hi_142_2;
          xs[9] = lo_143_2;
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
  if (tid_1 < 512) {
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
    if (idx <= 2047) {
      val = (int)idx;
    }
    if (commit != 0) {
      long long obase = ((long long)ucol * (long long)num_heads + (long long)head) * 16;
      out[obase + (long long)rank] = val;
    }
  }
}

}  // extern "C"
