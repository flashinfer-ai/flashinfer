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

// Shared by every record that names this translation unit in its sources: the record's
// compile_flags set the knob values below (a knob without a -D flag is a compile error) and name
// the kernel symbol (-DCAKE_PROGRAM_KERNEL=<its own symbol>; every JIT library of the family
// exports a distinct kernel).
//   CAKE_INDEX_BITS        index bits l = bit_length(tiles - 1) of the route
//   CAKE_INDEX_LOW_MASK    (1 << l) - 1
//   CAKE_INDEX_HIGH_MASK   ~((1 << l) - 1) as a 32-bit unsigned literal
//   CAKE_INDEX_SCORE_MASK  0x7F7FFFFF & ~((1 << l) - 1)
#ifndef CAKE_INDEX_BITS
#error "CAKE_INDEX_BITS must come from the record's compile_flags"
#endif
#ifndef CAKE_INDEX_LOW_MASK
#error "CAKE_INDEX_LOW_MASK must come from the record's compile_flags"
#endif
#ifndef CAKE_INDEX_HIGH_MASK
#error "CAKE_INDEX_HIGH_MASK must come from the record's compile_flags"
#endif
#ifndef CAKE_INDEX_SCORE_MASK
#error "CAKE_INDEX_SCORE_MASK must come from the record's compile_flags"
#endif
#ifndef CAKE_PROGRAM_KERNEL
#error "CAKE_PROGRAM_KERNEL (the program's kernel symbol) must come from the record's compile_flags"
#endif
#define NUM_MAIN_STAGES 1
#define SMEM_TREE_OFF 0
#define SMEM_TREE_STAGE_BYTES 2176
#define SMEM_TREE_STRIDE 2176
#define SMEM_QCOL_OFF 2176
#define SMEM_QCOL_STAGE_BYTES 128
#define SMEM_QCOL_STRIDE 128
#define SMEM_FLAGW_OFF 2304
#define SMEM_FLAGW_STAGE_BYTES 16
#define SMEM_FLAGW_STRIDE 16
#define SMEM_TOTAL 2432
#define THREADS 32

extern "C" {

__global__ __launch_bounds__(32) void CAKE_PROGRAM_KERNEL(unsigned int* __restrict__ S,
                                                          int* __restrict__ nvp,
                                                          int* __restrict__ out, int tiles,
                                                          int total_q, int num_heads,
                                                          int num_chunks, int fb, int fe) {
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
  unsigned int* qcol = reinterpret_cast<unsigned int*>(smem_raw + 2176);
  const int qcol_addr = smem + 2176;
  unsigned int* flagw = reinterpret_cast<unsigned int*>(smem_raw + 2304);
  const int flagw_addr = smem + 2304;

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
  unsigned int nb[16];
#pragma unroll 1
  for (int j = 0; j < num_chunks; j++) {
    int t0_1 = (j + 1 + w) * 16;
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
    int t0_3 = (j + w) * 16;
    float sc = __uint_as_float(cb[0]);
    sc = fmaxf(sc, -1.7014118346046923e+38f);
    sc = fminf(sc, 1.7014118346046923e+38f);
    unsigned int key = __as_u32(sc) & CAKE_INDEX_HIGH_MASK | (unsigned int)t0_3;
    kb[0] = __uint_as_float(key);
    float sc_4 = __uint_as_float(cb[1]);
    sc_4 = fmaxf(sc_4, -1.7014118346046923e+38f);
    sc_4 = fminf(sc_4, 1.7014118346046923e+38f);
    unsigned int key_5 = __as_u32(sc_4) & CAKE_INDEX_HIGH_MASK | (unsigned int)(t0_3 + 1);
    kb[1] = __uint_as_float(key_5);
    float sc_6 = __uint_as_float(cb[2]);
    sc_6 = fmaxf(sc_6, -1.7014118346046923e+38f);
    sc_6 = fminf(sc_6, 1.7014118346046923e+38f);
    unsigned int key_7 = __as_u32(sc_6) & CAKE_INDEX_HIGH_MASK | (unsigned int)(t0_3 + 2);
    kb[2] = __uint_as_float(key_7);
    float sc_8 = __uint_as_float(cb[3]);
    sc_8 = fmaxf(sc_8, -1.7014118346046923e+38f);
    sc_8 = fminf(sc_8, 1.7014118346046923e+38f);
    unsigned int key_9 = __as_u32(sc_8) & CAKE_INDEX_HIGH_MASK | (unsigned int)(t0_3 + 3);
    kb[3] = __uint_as_float(key_9);
    float sc_10 = __uint_as_float(cb[4]);
    sc_10 = fmaxf(sc_10, -1.7014118346046923e+38f);
    sc_10 = fminf(sc_10, 1.7014118346046923e+38f);
    unsigned int key_11 = __as_u32(sc_10) & CAKE_INDEX_HIGH_MASK | (unsigned int)(t0_3 + 4);
    kb[4] = __uint_as_float(key_11);
    float sc_12 = __uint_as_float(cb[5]);
    sc_12 = fmaxf(sc_12, -1.7014118346046923e+38f);
    sc_12 = fminf(sc_12, 1.7014118346046923e+38f);
    unsigned int key_13 = __as_u32(sc_12) & CAKE_INDEX_HIGH_MASK | (unsigned int)(t0_3 + 5);
    kb[5] = __uint_as_float(key_13);
    float sc_14 = __uint_as_float(cb[6]);
    sc_14 = fmaxf(sc_14, -1.7014118346046923e+38f);
    sc_14 = fminf(sc_14, 1.7014118346046923e+38f);
    unsigned int key_15 = __as_u32(sc_14) & CAKE_INDEX_HIGH_MASK | (unsigned int)(t0_3 + 6);
    kb[6] = __uint_as_float(key_15);
    float sc_16 = __uint_as_float(cb[7]);
    sc_16 = fmaxf(sc_16, -1.7014118346046923e+38f);
    sc_16 = fminf(sc_16, 1.7014118346046923e+38f);
    unsigned int key_17 = __as_u32(sc_16) & CAKE_INDEX_HIGH_MASK | (unsigned int)(t0_3 + 7);
    kb[7] = __uint_as_float(key_17);
    float sc_18 = __uint_as_float(cb[8]);
    sc_18 = fmaxf(sc_18, -1.7014118346046923e+38f);
    sc_18 = fminf(sc_18, 1.7014118346046923e+38f);
    unsigned int key_19 = __as_u32(sc_18) & CAKE_INDEX_HIGH_MASK | (unsigned int)(t0_3 + 8);
    kb[8] = __uint_as_float(key_19);
    float sc_20 = __uint_as_float(cb[9]);
    sc_20 = fmaxf(sc_20, -1.7014118346046923e+38f);
    sc_20 = fminf(sc_20, 1.7014118346046923e+38f);
    unsigned int key_21 = __as_u32(sc_20) & CAKE_INDEX_HIGH_MASK | (unsigned int)(t0_3 + 9);
    kb[9] = __uint_as_float(key_21);
    float sc_22 = __uint_as_float(cb[10]);
    sc_22 = fmaxf(sc_22, -1.7014118346046923e+38f);
    sc_22 = fminf(sc_22, 1.7014118346046923e+38f);
    unsigned int key_23 = __as_u32(sc_22) & CAKE_INDEX_HIGH_MASK | (unsigned int)(t0_3 + 10);
    kb[10] = __uint_as_float(key_23);
    float sc_24 = __uint_as_float(cb[11]);
    sc_24 = fmaxf(sc_24, -1.7014118346046923e+38f);
    sc_24 = fminf(sc_24, 1.7014118346046923e+38f);
    unsigned int key_25 = __as_u32(sc_24) & CAKE_INDEX_HIGH_MASK | (unsigned int)(t0_3 + 11);
    kb[11] = __uint_as_float(key_25);
    float sc_26 = __uint_as_float(cb[12]);
    sc_26 = fmaxf(sc_26, -1.7014118346046923e+38f);
    sc_26 = fminf(sc_26, 1.7014118346046923e+38f);
    unsigned int key_27 = __as_u32(sc_26) & CAKE_INDEX_HIGH_MASK | (unsigned int)(t0_3 + 12);
    kb[12] = __uint_as_float(key_27);
    float sc_28 = __uint_as_float(cb[13]);
    sc_28 = fmaxf(sc_28, -1.7014118346046923e+38f);
    sc_28 = fminf(sc_28, 1.7014118346046923e+38f);
    unsigned int key_29 = __as_u32(sc_28) & CAKE_INDEX_HIGH_MASK | (unsigned int)(t0_3 + 13);
    kb[13] = __uint_as_float(key_29);
    float sc_30 = __uint_as_float(cb[14]);
    sc_30 = fmaxf(sc_30, -1.7014118346046923e+38f);
    sc_30 = fminf(sc_30, 1.7014118346046923e+38f);
    unsigned int key_31 = __as_u32(sc_30) & CAKE_INDEX_HIGH_MASK | (unsigned int)(t0_3 + 14);
    kb[14] = __uint_as_float(key_31);
    float sc_32 = __uint_as_float(cb[15]);
    sc_32 = fmaxf(sc_32, -1.7014118346046923e+38f);
    sc_32 = fminf(sc_32, 1.7014118346046923e+38f);
    unsigned int key_33 = __as_u32(sc_32) & CAKE_INDEX_HIGH_MASK | (unsigned int)(t0_3 + 15);
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
        kb[0] = __uint_as_float(CAKE_INDEX_SCORE_MASK | (unsigned int)t0_3);
      }
      int f_1 = 0;
      if ((t0_3 + 1 < fb || t0_3 + 1 >= lim - fe) && lim > t0_3 + 1) {
        f_1 = 1;
      }
      if (f_1 != 0) {
        kb[1] = __uint_as_float(CAKE_INDEX_SCORE_MASK | (unsigned int)(t0_3 + 1));
      }
      int f_2 = 0;
      if ((t0_3 + 2 < fb || t0_3 + 2 >= lim - fe) && lim > t0_3 + 2) {
        f_2 = 1;
      }
      if (f_2 != 0) {
        kb[2] = __uint_as_float(CAKE_INDEX_SCORE_MASK | (unsigned int)(t0_3 + 2));
      }
      int f_3 = 0;
      if ((t0_3 + 3 < fb || t0_3 + 3 >= lim - fe) && lim > t0_3 + 3) {
        f_3 = 1;
      }
      if (f_3 != 0) {
        kb[3] = __uint_as_float(CAKE_INDEX_SCORE_MASK | (unsigned int)(t0_3 + 3));
      }
      int f_4 = 0;
      if ((t0_3 + 4 < fb || t0_3 + 4 >= lim - fe) && lim > t0_3 + 4) {
        f_4 = 1;
      }
      if (f_4 != 0) {
        kb[4] = __uint_as_float(CAKE_INDEX_SCORE_MASK | (unsigned int)(t0_3 + 4));
      }
      int f_5 = 0;
      if ((t0_3 + 5 < fb || t0_3 + 5 >= lim - fe) && lim > t0_3 + 5) {
        f_5 = 1;
      }
      if (f_5 != 0) {
        kb[5] = __uint_as_float(CAKE_INDEX_SCORE_MASK | (unsigned int)(t0_3 + 5));
      }
      int f_6 = 0;
      if ((t0_3 + 6 < fb || t0_3 + 6 >= lim - fe) && lim > t0_3 + 6) {
        f_6 = 1;
      }
      if (f_6 != 0) {
        kb[6] = __uint_as_float(CAKE_INDEX_SCORE_MASK | (unsigned int)(t0_3 + 6));
      }
      int f_7 = 0;
      if ((t0_3 + 7 < fb || t0_3 + 7 >= lim - fe) && lim > t0_3 + 7) {
        f_7 = 1;
      }
      if (f_7 != 0) {
        kb[7] = __uint_as_float(CAKE_INDEX_SCORE_MASK | (unsigned int)(t0_3 + 7));
      }
      int f_8 = 0;
      if ((t0_3 + 8 < fb || t0_3 + 8 >= lim - fe) && lim > t0_3 + 8) {
        f_8 = 1;
      }
      if (f_8 != 0) {
        kb[8] = __uint_as_float(CAKE_INDEX_SCORE_MASK | (unsigned int)(t0_3 + 8));
      }
      int f_9 = 0;
      if ((t0_3 + 9 < fb || t0_3 + 9 >= lim - fe) && lim > t0_3 + 9) {
        f_9 = 1;
      }
      if (f_9 != 0) {
        kb[9] = __uint_as_float(CAKE_INDEX_SCORE_MASK | (unsigned int)(t0_3 + 9));
      }
      int f_10 = 0;
      if ((t0_3 + 10 < fb || t0_3 + 10 >= lim - fe) && lim > t0_3 + 10) {
        f_10 = 1;
      }
      if (f_10 != 0) {
        kb[10] = __uint_as_float(CAKE_INDEX_SCORE_MASK | (unsigned int)(t0_3 + 10));
      }
      int f_11 = 0;
      if ((t0_3 + 11 < fb || t0_3 + 11 >= lim - fe) && lim > t0_3 + 11) {
        f_11 = 1;
      }
      if (f_11 != 0) {
        kb[11] = __uint_as_float(CAKE_INDEX_SCORE_MASK | (unsigned int)(t0_3 + 11));
      }
      int f_12 = 0;
      if ((t0_3 + 12 < fb || t0_3 + 12 >= lim - fe) && lim > t0_3 + 12) {
        f_12 = 1;
      }
      if (f_12 != 0) {
        kb[12] = __uint_as_float(CAKE_INDEX_SCORE_MASK | (unsigned int)(t0_3 + 12));
      }
      int f_13 = 0;
      if ((t0_3 + 13 < fb || t0_3 + 13 >= lim - fe) && lim > t0_3 + 13) {
        f_13 = 1;
      }
      if (f_13 != 0) {
        kb[13] = __uint_as_float(CAKE_INDEX_SCORE_MASK | (unsigned int)(t0_3 + 13));
      }
      int f_14 = 0;
      if ((t0_3 + 14 < fb || t0_3 + 14 >= lim - fe) && lim > t0_3 + 14) {
        f_14 = 1;
      }
      if (f_14 != 0) {
        kb[14] = __uint_as_float(CAKE_INDEX_SCORE_MASK | (unsigned int)(t0_3 + 14));
      }
      int f_15 = 0;
      if ((t0_3 + 15 < fb || t0_3 + 15 >= lim - fe) && lim > t0_3 + 15) {
        f_15 = 1;
      }
      if (f_15 != 0) {
        kb[15] = __uint_as_float(CAKE_INDEX_SCORE_MASK | (unsigned int)(t0_3 + 15));
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
  if (w == 0) {
    unsigned int u16 = __as_u32(a[15]);
    unsigned int c16 = u16 & CAKE_INDEX_HIGH_MASK;
    unsigned int cr = __as_u32(rej) & CAKE_INDEX_HIGH_MASK;
    unsigned int q = 4294967295u;
    if (c16 == cr && u16 < 4278190080u && c16 != CAKE_INDEX_SCORE_MASK && col < total_q) {
      q = c16;
      flagw[0] = 1;
    }
    qcol[c] = q;
  }
  asm volatile("barrier.sync 1, 32;" ::: "memory");
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
    int lim2 = 0;
    if (flagged != 0) {
      lim2 = lim;
    }
    int _warp_vote_0 = __any_sync(0xFFFFFFFF, flagged != 0);
    if (_warp_vote_0 != 0) {
      unsigned int cb2[16];
      unsigned int nb2[16];
      float kb2[16];
      int t0_1_1 = w * 16;
      long long p_2_1 = cbase + (long long)t0_1_1 * nq64;
      cb2[0] = 4286578688u;
      if (lim2 > t0_1_1) {
        cb2[0] = S[p_2_1];
      }
      cb2[1] = 4286578688u;
      if (lim2 > t0_1_1 + 1) {
        cb2[1] = S[p_2_1 + nq64];
      }
      cb2[2] = 4286578688u;
      if (lim2 > t0_1_1 + 2) {
        cb2[2] = S[p_2_1 + 2 * nq64];
      }
      cb2[3] = 4286578688u;
      if (lim2 > t0_1_1 + 3) {
        cb2[3] = S[p_2_1 + 3 * nq64];
      }
      cb2[4] = 4286578688u;
      if (lim2 > t0_1_1 + 4) {
        cb2[4] = S[p_2_1 + 4 * nq64];
      }
      cb2[5] = 4286578688u;
      if (lim2 > t0_1_1 + 5) {
        cb2[5] = S[p_2_1 + 5 * nq64];
      }
      cb2[6] = 4286578688u;
      if (lim2 > t0_1_1 + 6) {
        cb2[6] = S[p_2_1 + 6 * nq64];
      }
      cb2[7] = 4286578688u;
      if (lim2 > t0_1_1 + 7) {
        cb2[7] = S[p_2_1 + 7 * nq64];
      }
      cb2[8] = 4286578688u;
      if (lim2 > t0_1_1 + 8) {
        cb2[8] = S[p_2_1 + 8 * nq64];
      }
      cb2[9] = 4286578688u;
      if (lim2 > t0_1_1 + 9) {
        cb2[9] = S[p_2_1 + 9 * nq64];
      }
      cb2[10] = 4286578688u;
      if (lim2 > t0_1_1 + 10) {
        cb2[10] = S[p_2_1 + 10 * nq64];
      }
      cb2[11] = 4286578688u;
      if (lim2 > t0_1_1 + 11) {
        cb2[11] = S[p_2_1 + 11 * nq64];
      }
      cb2[12] = 4286578688u;
      if (lim2 > t0_1_1 + 12) {
        cb2[12] = S[p_2_1 + 12 * nq64];
      }
      cb2[13] = 4286578688u;
      if (lim2 > t0_1_1 + 13) {
        cb2[13] = S[p_2_1 + 13 * nq64];
      }
      cb2[14] = 4286578688u;
      if (lim2 > t0_1_1 + 14) {
        cb2[14] = S[p_2_1 + 14 * nq64];
      }
      cb2[15] = 4286578688u;
      if (lim2 > t0_1_1 + 15) {
        cb2[15] = S[p_2_1 + 15 * nq64];
      }
      asm volatile("" ::: "memory");
#pragma unroll 1
      for (int j_1 = 0; j_1 < num_chunks; j_1++) {
        int t0_2 = (j_1 + 1 + w) * 16;
        long long p_3 = cbase + (long long)t0_2 * nq64;
        nb2[0] = 4286578688u;
        if (lim2 > t0_2) {
          nb2[0] = S[p_3];
        }
        nb2[1] = 4286578688u;
        if (lim2 > t0_2 + 1) {
          nb2[1] = S[p_3 + nq64];
        }
        nb2[2] = 4286578688u;
        if (lim2 > t0_2 + 2) {
          nb2[2] = S[p_3 + 2 * nq64];
        }
        nb2[3] = 4286578688u;
        if (lim2 > t0_2 + 3) {
          nb2[3] = S[p_3 + 3 * nq64];
        }
        nb2[4] = 4286578688u;
        if (lim2 > t0_2 + 4) {
          nb2[4] = S[p_3 + 4 * nq64];
        }
        nb2[5] = 4286578688u;
        if (lim2 > t0_2 + 5) {
          nb2[5] = S[p_3 + 5 * nq64];
        }
        nb2[6] = 4286578688u;
        if (lim2 > t0_2 + 6) {
          nb2[6] = S[p_3 + 6 * nq64];
        }
        nb2[7] = 4286578688u;
        if (lim2 > t0_2 + 7) {
          nb2[7] = S[p_3 + 7 * nq64];
        }
        nb2[8] = 4286578688u;
        if (lim2 > t0_2 + 8) {
          nb2[8] = S[p_3 + 8 * nq64];
        }
        nb2[9] = 4286578688u;
        if (lim2 > t0_2 + 9) {
          nb2[9] = S[p_3 + 9 * nq64];
        }
        nb2[10] = 4286578688u;
        if (lim2 > t0_2 + 10) {
          nb2[10] = S[p_3 + 10 * nq64];
        }
        nb2[11] = 4286578688u;
        if (lim2 > t0_2 + 11) {
          nb2[11] = S[p_3 + 11 * nq64];
        }
        nb2[12] = 4286578688u;
        if (lim2 > t0_2 + 12) {
          nb2[12] = S[p_3 + 12 * nq64];
        }
        nb2[13] = 4286578688u;
        if (lim2 > t0_2 + 13) {
          nb2[13] = S[p_3 + 13 * nq64];
        }
        nb2[14] = 4286578688u;
        if (lim2 > t0_2 + 14) {
          nb2[14] = S[p_3 + 14 * nq64];
        }
        nb2[15] = 4286578688u;
        if (lim2 > t0_2 + 15) {
          nb2[15] = S[p_3 + 15 * nq64];
        }
        asm volatile("" ::: "memory");
        int t0_4 = (j_1 + w) * 16;
        float qf = __uint_as_float(qc);
        float sc_1 = __uint_as_float(cb2[0]);
        sc_1 = fmaxf(sc_1, -1.7014118346046923e+38f);
        sc_1 = fminf(sc_1, 1.7014118346046923e+38f);
        unsigned int u = __as_u32(sc_1);
        unsigned int cls = u & CAKE_INDEX_HIGH_MASK;
        int f_16 = 0;
        if ((t0_4 < fb || t0_4 >= lim - fe) && lim > t0_4) {
          f_16 = 1;
        }
        if (f_16 != 0) {
          cls = CAKE_INDEX_SCORE_MASK;
        }
        unsigned int key_1 = 0;
        if (qf < __uint_as_float(cls) && cls < 4278190080u) {
          key_1 = 1073741824 | (unsigned int)t0_4;
        }
        if (cls == qc) {
          unsigned int lowb =
              (u ^ (unsigned int)((int)u >> 31) & CAKE_INDEX_LOW_MASK) & CAKE_INDEX_LOW_MASK;
          key_1 = 536870912 | lowb << CAKE_INDEX_BITS | (unsigned int)t0_4;
        }
        kb2[0] = __uint_as_float(key_1);
        float sc_5 = __uint_as_float(cb2[1]);
        sc_5 = fmaxf(sc_5, -1.7014118346046923e+38f);
        sc_5 = fminf(sc_5, 1.7014118346046923e+38f);
        unsigned int u_6 = __as_u32(sc_5);
        unsigned int cls_7 = u_6 & CAKE_INDEX_HIGH_MASK;
        int f_8_1 = 0;
        if ((t0_4 + 1 < fb || t0_4 + 1 >= lim - fe) && lim > t0_4 + 1) {
          f_8_1 = 1;
        }
        if (f_8_1 != 0) {
          cls_7 = CAKE_INDEX_SCORE_MASK;
        }
        unsigned int key_9_1 = 0;
        if (qf < __uint_as_float(cls_7) && cls_7 < 4278190080u) {
          key_9_1 = 1073741824 | (unsigned int)(t0_4 + 1);
        }
        if (cls_7 == qc) {
          unsigned int lowb_1 =
              (u_6 ^ (unsigned int)((int)u_6 >> 31) & CAKE_INDEX_LOW_MASK) & CAKE_INDEX_LOW_MASK;
          key_9_1 = 536870912 | lowb_1 << CAKE_INDEX_BITS | (unsigned int)(t0_4 + 1);
        }
        kb2[1] = __uint_as_float(key_9_1);
        float sc_10_1 = __uint_as_float(cb2[2]);
        sc_10_1 = fmaxf(sc_10_1, -1.7014118346046923e+38f);
        sc_10_1 = fminf(sc_10_1, 1.7014118346046923e+38f);
        unsigned int u_11 = __as_u32(sc_10_1);
        unsigned int cls_12 = u_11 & CAKE_INDEX_HIGH_MASK;
        int f_13_1 = 0;
        if ((t0_4 + 2 < fb || t0_4 + 2 >= lim - fe) && lim > t0_4 + 2) {
          f_13_1 = 1;
        }
        if (f_13_1 != 0) {
          cls_12 = CAKE_INDEX_SCORE_MASK;
        }
        unsigned int key_14 = 0;
        if (qf < __uint_as_float(cls_12) && cls_12 < 4278190080u) {
          key_14 = 1073741824 | (unsigned int)(t0_4 + 2);
        }
        if (cls_12 == qc) {
          unsigned int lowb_2 =
              (u_11 ^ (unsigned int)((int)u_11 >> 31) & CAKE_INDEX_LOW_MASK) & CAKE_INDEX_LOW_MASK;
          key_14 = 536870912 | lowb_2 << CAKE_INDEX_BITS | (unsigned int)(t0_4 + 2);
        }
        kb2[2] = __uint_as_float(key_14);
        float sc_15 = __uint_as_float(cb2[3]);
        sc_15 = fmaxf(sc_15, -1.7014118346046923e+38f);
        sc_15 = fminf(sc_15, 1.7014118346046923e+38f);
        unsigned int u_16 = __as_u32(sc_15);
        unsigned int cls_17 = u_16 & CAKE_INDEX_HIGH_MASK;
        int f_18 = 0;
        if ((t0_4 + 3 < fb || t0_4 + 3 >= lim - fe) && lim > t0_4 + 3) {
          f_18 = 1;
        }
        if (f_18 != 0) {
          cls_17 = CAKE_INDEX_SCORE_MASK;
        }
        unsigned int key_19_1 = 0;
        if (qf < __uint_as_float(cls_17) && cls_17 < 4278190080u) {
          key_19_1 = 1073741824 | (unsigned int)(t0_4 + 3);
        }
        if (cls_17 == qc) {
          unsigned int lowb_3 =
              (u_16 ^ (unsigned int)((int)u_16 >> 31) & CAKE_INDEX_LOW_MASK) & CAKE_INDEX_LOW_MASK;
          key_19_1 = 536870912 | lowb_3 << CAKE_INDEX_BITS | (unsigned int)(t0_4 + 3);
        }
        kb2[3] = __uint_as_float(key_19_1);
        float sc_20_1 = __uint_as_float(cb2[4]);
        sc_20_1 = fmaxf(sc_20_1, -1.7014118346046923e+38f);
        sc_20_1 = fminf(sc_20_1, 1.7014118346046923e+38f);
        unsigned int u_21 = __as_u32(sc_20_1);
        unsigned int cls_22 = u_21 & CAKE_INDEX_HIGH_MASK;
        int f_23 = 0;
        if ((t0_4 + 4 < fb || t0_4 + 4 >= lim - fe) && lim > t0_4 + 4) {
          f_23 = 1;
        }
        if (f_23 != 0) {
          cls_22 = CAKE_INDEX_SCORE_MASK;
        }
        unsigned int key_24 = 0;
        if (qf < __uint_as_float(cls_22) && cls_22 < 4278190080u) {
          key_24 = 1073741824 | (unsigned int)(t0_4 + 4);
        }
        if (cls_22 == qc) {
          unsigned int lowb_4 =
              (u_21 ^ (unsigned int)((int)u_21 >> 31) & CAKE_INDEX_LOW_MASK) & CAKE_INDEX_LOW_MASK;
          key_24 = 536870912 | lowb_4 << CAKE_INDEX_BITS | (unsigned int)(t0_4 + 4);
        }
        kb2[4] = __uint_as_float(key_24);
        float sc_25 = __uint_as_float(cb2[5]);
        sc_25 = fmaxf(sc_25, -1.7014118346046923e+38f);
        sc_25 = fminf(sc_25, 1.7014118346046923e+38f);
        unsigned int u_26 = __as_u32(sc_25);
        unsigned int cls_27 = u_26 & CAKE_INDEX_HIGH_MASK;
        int f_28 = 0;
        if ((t0_4 + 5 < fb || t0_4 + 5 >= lim - fe) && lim > t0_4 + 5) {
          f_28 = 1;
        }
        if (f_28 != 0) {
          cls_27 = CAKE_INDEX_SCORE_MASK;
        }
        unsigned int key_29_1 = 0;
        if (qf < __uint_as_float(cls_27) && cls_27 < 4278190080u) {
          key_29_1 = 1073741824 | (unsigned int)(t0_4 + 5);
        }
        if (cls_27 == qc) {
          unsigned int lowb_5 =
              (u_26 ^ (unsigned int)((int)u_26 >> 31) & CAKE_INDEX_LOW_MASK) & CAKE_INDEX_LOW_MASK;
          key_29_1 = 536870912 | lowb_5 << CAKE_INDEX_BITS | (unsigned int)(t0_4 + 5);
        }
        kb2[5] = __uint_as_float(key_29_1);
        float sc_30_1 = __uint_as_float(cb2[6]);
        sc_30_1 = fmaxf(sc_30_1, -1.7014118346046923e+38f);
        sc_30_1 = fminf(sc_30_1, 1.7014118346046923e+38f);
        unsigned int u_31 = __as_u32(sc_30_1);
        unsigned int cls_32 = u_31 & CAKE_INDEX_HIGH_MASK;
        int f_33 = 0;
        if ((t0_4 + 6 < fb || t0_4 + 6 >= lim - fe) && lim > t0_4 + 6) {
          f_33 = 1;
        }
        if (f_33 != 0) {
          cls_32 = CAKE_INDEX_SCORE_MASK;
        }
        unsigned int key_34 = 0;
        if (qf < __uint_as_float(cls_32) && cls_32 < 4278190080u) {
          key_34 = 1073741824 | (unsigned int)(t0_4 + 6);
        }
        if (cls_32 == qc) {
          unsigned int lowb_6 =
              (u_31 ^ (unsigned int)((int)u_31 >> 31) & CAKE_INDEX_LOW_MASK) & CAKE_INDEX_LOW_MASK;
          key_34 = 536870912 | lowb_6 << CAKE_INDEX_BITS | (unsigned int)(t0_4 + 6);
        }
        kb2[6] = __uint_as_float(key_34);
        float sc_35 = __uint_as_float(cb2[7]);
        sc_35 = fmaxf(sc_35, -1.7014118346046923e+38f);
        sc_35 = fminf(sc_35, 1.7014118346046923e+38f);
        unsigned int u_36 = __as_u32(sc_35);
        unsigned int cls_37 = u_36 & CAKE_INDEX_HIGH_MASK;
        int f_38 = 0;
        if ((t0_4 + 7 < fb || t0_4 + 7 >= lim - fe) && lim > t0_4 + 7) {
          f_38 = 1;
        }
        if (f_38 != 0) {
          cls_37 = CAKE_INDEX_SCORE_MASK;
        }
        unsigned int key_39 = 0;
        if (qf < __uint_as_float(cls_37) && cls_37 < 4278190080u) {
          key_39 = 1073741824 | (unsigned int)(t0_4 + 7);
        }
        if (cls_37 == qc) {
          unsigned int lowb_7 =
              (u_36 ^ (unsigned int)((int)u_36 >> 31) & CAKE_INDEX_LOW_MASK) & CAKE_INDEX_LOW_MASK;
          key_39 = 536870912 | lowb_7 << CAKE_INDEX_BITS | (unsigned int)(t0_4 + 7);
        }
        kb2[7] = __uint_as_float(key_39);
        float sc_40 = __uint_as_float(cb2[8]);
        sc_40 = fmaxf(sc_40, -1.7014118346046923e+38f);
        sc_40 = fminf(sc_40, 1.7014118346046923e+38f);
        unsigned int u_41 = __as_u32(sc_40);
        unsigned int cls_42 = u_41 & CAKE_INDEX_HIGH_MASK;
        int f_43 = 0;
        if ((t0_4 + 8 < fb || t0_4 + 8 >= lim - fe) && lim > t0_4 + 8) {
          f_43 = 1;
        }
        if (f_43 != 0) {
          cls_42 = CAKE_INDEX_SCORE_MASK;
        }
        unsigned int key_44 = 0;
        if (qf < __uint_as_float(cls_42) && cls_42 < 4278190080u) {
          key_44 = 1073741824 | (unsigned int)(t0_4 + 8);
        }
        if (cls_42 == qc) {
          unsigned int lowb_8 =
              (u_41 ^ (unsigned int)((int)u_41 >> 31) & CAKE_INDEX_LOW_MASK) & CAKE_INDEX_LOW_MASK;
          key_44 = 536870912 | lowb_8 << CAKE_INDEX_BITS | (unsigned int)(t0_4 + 8);
        }
        kb2[8] = __uint_as_float(key_44);
        float sc_45 = __uint_as_float(cb2[9]);
        sc_45 = fmaxf(sc_45, -1.7014118346046923e+38f);
        sc_45 = fminf(sc_45, 1.7014118346046923e+38f);
        unsigned int u_46 = __as_u32(sc_45);
        unsigned int cls_47 = u_46 & CAKE_INDEX_HIGH_MASK;
        int f_48 = 0;
        if ((t0_4 + 9 < fb || t0_4 + 9 >= lim - fe) && lim > t0_4 + 9) {
          f_48 = 1;
        }
        if (f_48 != 0) {
          cls_47 = CAKE_INDEX_SCORE_MASK;
        }
        unsigned int key_49 = 0;
        if (qf < __uint_as_float(cls_47) && cls_47 < 4278190080u) {
          key_49 = 1073741824 | (unsigned int)(t0_4 + 9);
        }
        if (cls_47 == qc) {
          unsigned int lowb_9 =
              (u_46 ^ (unsigned int)((int)u_46 >> 31) & CAKE_INDEX_LOW_MASK) & CAKE_INDEX_LOW_MASK;
          key_49 = 536870912 | lowb_9 << CAKE_INDEX_BITS | (unsigned int)(t0_4 + 9);
        }
        kb2[9] = __uint_as_float(key_49);
        float sc_50 = __uint_as_float(cb2[10]);
        sc_50 = fmaxf(sc_50, -1.7014118346046923e+38f);
        sc_50 = fminf(sc_50, 1.7014118346046923e+38f);
        unsigned int u_51 = __as_u32(sc_50);
        unsigned int cls_52 = u_51 & CAKE_INDEX_HIGH_MASK;
        int f_53 = 0;
        if ((t0_4 + 10 < fb || t0_4 + 10 >= lim - fe) && lim > t0_4 + 10) {
          f_53 = 1;
        }
        if (f_53 != 0) {
          cls_52 = CAKE_INDEX_SCORE_MASK;
        }
        unsigned int key_54 = 0;
        if (qf < __uint_as_float(cls_52) && cls_52 < 4278190080u) {
          key_54 = 1073741824 | (unsigned int)(t0_4 + 10);
        }
        if (cls_52 == qc) {
          unsigned int lowb_10 =
              (u_51 ^ (unsigned int)((int)u_51 >> 31) & CAKE_INDEX_LOW_MASK) & CAKE_INDEX_LOW_MASK;
          key_54 = 536870912 | lowb_10 << CAKE_INDEX_BITS | (unsigned int)(t0_4 + 10);
        }
        kb2[10] = __uint_as_float(key_54);
        float sc_55 = __uint_as_float(cb2[11]);
        sc_55 = fmaxf(sc_55, -1.7014118346046923e+38f);
        sc_55 = fminf(sc_55, 1.7014118346046923e+38f);
        unsigned int u_56 = __as_u32(sc_55);
        unsigned int cls_57 = u_56 & CAKE_INDEX_HIGH_MASK;
        int f_58 = 0;
        if ((t0_4 + 11 < fb || t0_4 + 11 >= lim - fe) && lim > t0_4 + 11) {
          f_58 = 1;
        }
        if (f_58 != 0) {
          cls_57 = CAKE_INDEX_SCORE_MASK;
        }
        unsigned int key_59 = 0;
        if (qf < __uint_as_float(cls_57) && cls_57 < 4278190080u) {
          key_59 = 1073741824 | (unsigned int)(t0_4 + 11);
        }
        if (cls_57 == qc) {
          unsigned int lowb_11 =
              (u_56 ^ (unsigned int)((int)u_56 >> 31) & CAKE_INDEX_LOW_MASK) & CAKE_INDEX_LOW_MASK;
          key_59 = 536870912 | lowb_11 << CAKE_INDEX_BITS | (unsigned int)(t0_4 + 11);
        }
        kb2[11] = __uint_as_float(key_59);
        float sc_60 = __uint_as_float(cb2[12]);
        sc_60 = fmaxf(sc_60, -1.7014118346046923e+38f);
        sc_60 = fminf(sc_60, 1.7014118346046923e+38f);
        unsigned int u_61 = __as_u32(sc_60);
        unsigned int cls_62 = u_61 & CAKE_INDEX_HIGH_MASK;
        int f_63 = 0;
        if ((t0_4 + 12 < fb || t0_4 + 12 >= lim - fe) && lim > t0_4 + 12) {
          f_63 = 1;
        }
        if (f_63 != 0) {
          cls_62 = CAKE_INDEX_SCORE_MASK;
        }
        unsigned int key_64 = 0;
        if (qf < __uint_as_float(cls_62) && cls_62 < 4278190080u) {
          key_64 = 1073741824 | (unsigned int)(t0_4 + 12);
        }
        if (cls_62 == qc) {
          unsigned int lowb_12 =
              (u_61 ^ (unsigned int)((int)u_61 >> 31) & CAKE_INDEX_LOW_MASK) & CAKE_INDEX_LOW_MASK;
          key_64 = 536870912 | lowb_12 << CAKE_INDEX_BITS | (unsigned int)(t0_4 + 12);
        }
        kb2[12] = __uint_as_float(key_64);
        float sc_65 = __uint_as_float(cb2[13]);
        sc_65 = fmaxf(sc_65, -1.7014118346046923e+38f);
        sc_65 = fminf(sc_65, 1.7014118346046923e+38f);
        unsigned int u_66 = __as_u32(sc_65);
        unsigned int cls_67 = u_66 & CAKE_INDEX_HIGH_MASK;
        int f_68 = 0;
        if ((t0_4 + 13 < fb || t0_4 + 13 >= lim - fe) && lim > t0_4 + 13) {
          f_68 = 1;
        }
        if (f_68 != 0) {
          cls_67 = CAKE_INDEX_SCORE_MASK;
        }
        unsigned int key_69 = 0;
        if (qf < __uint_as_float(cls_67) && cls_67 < 4278190080u) {
          key_69 = 1073741824 | (unsigned int)(t0_4 + 13);
        }
        if (cls_67 == qc) {
          unsigned int lowb_13 =
              (u_66 ^ (unsigned int)((int)u_66 >> 31) & CAKE_INDEX_LOW_MASK) & CAKE_INDEX_LOW_MASK;
          key_69 = 536870912 | lowb_13 << CAKE_INDEX_BITS | (unsigned int)(t0_4 + 13);
        }
        kb2[13] = __uint_as_float(key_69);
        float sc_70 = __uint_as_float(cb2[14]);
        sc_70 = fmaxf(sc_70, -1.7014118346046923e+38f);
        sc_70 = fminf(sc_70, 1.7014118346046923e+38f);
        unsigned int u_71 = __as_u32(sc_70);
        unsigned int cls_72 = u_71 & CAKE_INDEX_HIGH_MASK;
        int f_73 = 0;
        if ((t0_4 + 14 < fb || t0_4 + 14 >= lim - fe) && lim > t0_4 + 14) {
          f_73 = 1;
        }
        if (f_73 != 0) {
          cls_72 = CAKE_INDEX_SCORE_MASK;
        }
        unsigned int key_74 = 0;
        if (qf < __uint_as_float(cls_72) && cls_72 < 4278190080u) {
          key_74 = 1073741824 | (unsigned int)(t0_4 + 14);
        }
        if (cls_72 == qc) {
          unsigned int lowb_14 =
              (u_71 ^ (unsigned int)((int)u_71 >> 31) & CAKE_INDEX_LOW_MASK) & CAKE_INDEX_LOW_MASK;
          key_74 = 536870912 | lowb_14 << CAKE_INDEX_BITS | (unsigned int)(t0_4 + 14);
        }
        kb2[14] = __uint_as_float(key_74);
        float sc_75 = __uint_as_float(cb2[15]);
        sc_75 = fmaxf(sc_75, -1.7014118346046923e+38f);
        sc_75 = fminf(sc_75, 1.7014118346046923e+38f);
        unsigned int u_76 = __as_u32(sc_75);
        unsigned int cls_77 = u_76 & CAKE_INDEX_HIGH_MASK;
        int f_78 = 0;
        if ((t0_4 + 15 < fb || t0_4 + 15 >= lim - fe) && lim > t0_4 + 15) {
          f_78 = 1;
        }
        if (f_78 != 0) {
          cls_77 = CAKE_INDEX_SCORE_MASK;
        }
        unsigned int key_79 = 0;
        if (qf < __uint_as_float(cls_77) && cls_77 < 4278190080u) {
          key_79 = 1073741824 | (unsigned int)(t0_4 + 15);
        }
        if (cls_77 == qc) {
          unsigned int lowb_15 =
              (u_76 ^ (unsigned int)((int)u_76 >> 31) & CAKE_INDEX_LOW_MASK) & CAKE_INDEX_LOW_MASK;
          key_79 = 536870912 | lowb_15 << CAKE_INDEX_BITS | (unsigned int)(t0_4 + 15);
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
        int _warp_vote_1 = __any_sync(0xFFFFFFFF, nzc != 0);
        if (_warp_vote_1 != 0) {
          float hi_1 = fmaxf(kb2[0], kb2[13]);
          float lo_1 = fminf(kb2[0], kb2[13]);
          kb2[0] = hi_1;
          kb2[13] = lo_1;
          float hi_0 = fmaxf(kb2[1], kb2[12]);
          float lo_1_1 = fminf(kb2[1], kb2[12]);
          kb2[1] = hi_0;
          kb2[12] = lo_1_1;
          float hi_2 = fmaxf(kb2[2], kb2[15]);
          float lo_3 = fminf(kb2[2], kb2[15]);
          kb2[2] = hi_2;
          kb2[15] = lo_3;
          float hi_4 = fmaxf(kb2[3], kb2[14]);
          float lo_5 = fminf(kb2[3], kb2[14]);
          kb2[3] = hi_4;
          kb2[14] = lo_5;
          float hi_6 = fmaxf(kb2[4], kb2[8]);
          float lo_7 = fminf(kb2[4], kb2[8]);
          kb2[4] = hi_6;
          kb2[8] = lo_7;
          float hi_8 = fmaxf(kb2[5], kb2[6]);
          float lo_9 = fminf(kb2[5], kb2[6]);
          kb2[5] = hi_8;
          kb2[6] = lo_9;
          float hi_10 = fmaxf(kb2[7], kb2[11]);
          float lo_11 = fminf(kb2[7], kb2[11]);
          kb2[7] = hi_10;
          kb2[11] = lo_11;
          float hi_12 = fmaxf(kb2[9], kb2[10]);
          float lo_13 = fminf(kb2[9], kb2[10]);
          kb2[9] = hi_12;
          kb2[10] = lo_13;
          float hi_14 = fmaxf(kb2[0], kb2[5]);
          float lo_15 = fminf(kb2[0], kb2[5]);
          kb2[0] = hi_14;
          kb2[5] = lo_15;
          float hi_16 = fmaxf(kb2[1], kb2[7]);
          float lo_17 = fminf(kb2[1], kb2[7]);
          kb2[1] = hi_16;
          kb2[7] = lo_17;
          float hi_18 = fmaxf(kb2[2], kb2[9]);
          float lo_19 = fminf(kb2[2], kb2[9]);
          kb2[2] = hi_18;
          kb2[9] = lo_19;
          float hi_20 = fmaxf(kb2[3], kb2[4]);
          float lo_21 = fminf(kb2[3], kb2[4]);
          kb2[3] = hi_20;
          kb2[4] = lo_21;
          float hi_22 = fmaxf(kb2[6], kb2[13]);
          float lo_23 = fminf(kb2[6], kb2[13]);
          kb2[6] = hi_22;
          kb2[13] = lo_23;
          float hi_24 = fmaxf(kb2[8], kb2[14]);
          float lo_25 = fminf(kb2[8], kb2[14]);
          kb2[8] = hi_24;
          kb2[14] = lo_25;
          float hi_26 = fmaxf(kb2[10], kb2[15]);
          float lo_27 = fminf(kb2[10], kb2[15]);
          kb2[10] = hi_26;
          kb2[15] = lo_27;
          float hi_28 = fmaxf(kb2[11], kb2[12]);
          float lo_29 = fminf(kb2[11], kb2[12]);
          kb2[11] = hi_28;
          kb2[12] = lo_29;
          float hi_30 = fmaxf(kb2[0], kb2[1]);
          float lo_31 = fminf(kb2[0], kb2[1]);
          kb2[0] = hi_30;
          kb2[1] = lo_31;
          float hi_32 = fmaxf(kb2[2], kb2[3]);
          float lo_33 = fminf(kb2[2], kb2[3]);
          kb2[2] = hi_32;
          kb2[3] = lo_33;
          float hi_34_1 = fmaxf(kb2[4], kb2[5]);
          float lo_35_1 = fminf(kb2[4], kb2[5]);
          kb2[4] = hi_34_1;
          kb2[5] = lo_35_1;
          float hi_36_1 = fmaxf(kb2[6], kb2[8]);
          float lo_37_1 = fminf(kb2[6], kb2[8]);
          kb2[6] = hi_36_1;
          kb2[8] = lo_37_1;
          float hi_38_1 = fmaxf(kb2[7], kb2[9]);
          float lo_39_1 = fminf(kb2[7], kb2[9]);
          kb2[7] = hi_38_1;
          kb2[9] = lo_39_1;
          float hi_40_1 = fmaxf(kb2[10], kb2[11]);
          float lo_41_1 = fminf(kb2[10], kb2[11]);
          kb2[10] = hi_40_1;
          kb2[11] = lo_41_1;
          float hi_42_1 = fmaxf(kb2[12], kb2[13]);
          float lo_43_1 = fminf(kb2[12], kb2[13]);
          kb2[12] = hi_42_1;
          kb2[13] = lo_43_1;
          float hi_44_1 = fmaxf(kb2[14], kb2[15]);
          float lo_45_1 = fminf(kb2[14], kb2[15]);
          kb2[14] = hi_44_1;
          kb2[15] = lo_45_1;
          float hi_46_1 = fmaxf(kb2[0], kb2[2]);
          float lo_47_1 = fminf(kb2[0], kb2[2]);
          kb2[0] = hi_46_1;
          kb2[2] = lo_47_1;
          float hi_48_1 = fmaxf(kb2[1], kb2[3]);
          float lo_49_1 = fminf(kb2[1], kb2[3]);
          kb2[1] = hi_48_1;
          kb2[3] = lo_49_1;
          float hi_50_1 = fmaxf(kb2[4], kb2[10]);
          float lo_51_1 = fminf(kb2[4], kb2[10]);
          kb2[4] = hi_50_1;
          kb2[10] = lo_51_1;
          float hi_52_1 = fmaxf(kb2[5], kb2[11]);
          float lo_53_1 = fminf(kb2[5], kb2[11]);
          kb2[5] = hi_52_1;
          kb2[11] = lo_53_1;
          float hi_54_1 = fmaxf(kb2[6], kb2[7]);
          float lo_55_1 = fminf(kb2[6], kb2[7]);
          kb2[6] = hi_54_1;
          kb2[7] = lo_55_1;
          float hi_56_1 = fmaxf(kb2[8], kb2[9]);
          float lo_57_1 = fminf(kb2[8], kb2[9]);
          kb2[8] = hi_56_1;
          kb2[9] = lo_57_1;
          float hi_58_1 = fmaxf(kb2[12], kb2[14]);
          float lo_59_1 = fminf(kb2[12], kb2[14]);
          kb2[12] = hi_58_1;
          kb2[14] = lo_59_1;
          float hi_60_1 = fmaxf(kb2[13], kb2[15]);
          float lo_61_1 = fminf(kb2[13], kb2[15]);
          kb2[13] = hi_60_1;
          kb2[15] = lo_61_1;
          float hi_62_1 = fmaxf(kb2[1], kb2[2]);
          float lo_63_1 = fminf(kb2[1], kb2[2]);
          kb2[1] = hi_62_1;
          kb2[2] = lo_63_1;
          float hi_64_1 = fmaxf(kb2[3], kb2[12]);
          float lo_65_1 = fminf(kb2[3], kb2[12]);
          kb2[3] = hi_64_1;
          kb2[12] = lo_65_1;
          float hi_66_1 = fmaxf(kb2[4], kb2[6]);
          float lo_67_1 = fminf(kb2[4], kb2[6]);
          kb2[4] = hi_66_1;
          kb2[6] = lo_67_1;
          float hi_68_1 = fmaxf(kb2[5], kb2[7]);
          float lo_69_1 = fminf(kb2[5], kb2[7]);
          kb2[5] = hi_68_1;
          kb2[7] = lo_69_1;
          float hi_70_1 = fmaxf(kb2[8], kb2[10]);
          float lo_71_1 = fminf(kb2[8], kb2[10]);
          kb2[8] = hi_70_1;
          kb2[10] = lo_71_1;
          float hi_72_1 = fmaxf(kb2[9], kb2[11]);
          float lo_73_1 = fminf(kb2[9], kb2[11]);
          kb2[9] = hi_72_1;
          kb2[11] = lo_73_1;
          float hi_74_1 = fmaxf(kb2[13], kb2[14]);
          float lo_75_1 = fminf(kb2[13], kb2[14]);
          kb2[13] = hi_74_1;
          kb2[14] = lo_75_1;
          float hi_76_1 = fmaxf(kb2[1], kb2[4]);
          float lo_77_1 = fminf(kb2[1], kb2[4]);
          kb2[1] = hi_76_1;
          kb2[4] = lo_77_1;
          float hi_78_1 = fmaxf(kb2[2], kb2[6]);
          float lo_79_1 = fminf(kb2[2], kb2[6]);
          kb2[2] = hi_78_1;
          kb2[6] = lo_79_1;
          float hi_80_1 = fmaxf(kb2[5], kb2[8]);
          float lo_81_1 = fminf(kb2[5], kb2[8]);
          kb2[5] = hi_80_1;
          kb2[8] = lo_81_1;
          float hi_82_1 = fmaxf(kb2[7], kb2[10]);
          float lo_83_1 = fminf(kb2[7], kb2[10]);
          kb2[7] = hi_82_1;
          kb2[10] = lo_83_1;
          float hi_84_1 = fmaxf(kb2[9], kb2[13]);
          float lo_85_1 = fminf(kb2[9], kb2[13]);
          kb2[9] = hi_84_1;
          kb2[13] = lo_85_1;
          float hi_86_1 = fmaxf(kb2[11], kb2[14]);
          float lo_87_1 = fminf(kb2[11], kb2[14]);
          kb2[11] = hi_86_1;
          kb2[14] = lo_87_1;
          float hi_88_1 = fmaxf(kb2[2], kb2[4]);
          float lo_89_1 = fminf(kb2[2], kb2[4]);
          kb2[2] = hi_88_1;
          kb2[4] = lo_89_1;
          float hi_90_1 = fmaxf(kb2[3], kb2[6]);
          float lo_91_1 = fminf(kb2[3], kb2[6]);
          kb2[3] = hi_90_1;
          kb2[6] = lo_91_1;
          float hi_92_1 = fmaxf(kb2[9], kb2[12]);
          float lo_93_1 = fminf(kb2[9], kb2[12]);
          kb2[9] = hi_92_1;
          kb2[12] = lo_93_1;
          float hi_94_1 = fmaxf(kb2[11], kb2[13]);
          float lo_95_1 = fminf(kb2[11], kb2[13]);
          kb2[11] = hi_94_1;
          kb2[13] = lo_95_1;
          float hi_96_1 = fmaxf(kb2[3], kb2[5]);
          float lo_97_1 = fminf(kb2[3], kb2[5]);
          kb2[3] = hi_96_1;
          kb2[5] = lo_97_1;
          float hi_98_1 = fmaxf(kb2[6], kb2[8]);
          float lo_99_1 = fminf(kb2[6], kb2[8]);
          kb2[6] = hi_98_1;
          kb2[8] = lo_99_1;
          float hi_100_1 = fmaxf(kb2[7], kb2[9]);
          float lo_101_1 = fminf(kb2[7], kb2[9]);
          kb2[7] = hi_100_1;
          kb2[9] = lo_101_1;
          float hi_102_1 = fmaxf(kb2[10], kb2[12]);
          float lo_103_1 = fminf(kb2[10], kb2[12]);
          kb2[10] = hi_102_1;
          kb2[12] = lo_103_1;
          float hi_104_1 = fmaxf(kb2[3], kb2[4]);
          float lo_105_1 = fminf(kb2[3], kb2[4]);
          kb2[3] = hi_104_1;
          kb2[4] = lo_105_1;
          float hi_106_1 = fmaxf(kb2[5], kb2[6]);
          float lo_107_1 = fminf(kb2[5], kb2[6]);
          kb2[5] = hi_106_1;
          kb2[6] = lo_107_1;
          float hi_108_1 = fmaxf(kb2[7], kb2[8]);
          float lo_109_1 = fminf(kb2[7], kb2[8]);
          kb2[7] = hi_108_1;
          kb2[8] = lo_109_1;
          float hi_110_1 = fmaxf(kb2[9], kb2[10]);
          float lo_111_1 = fminf(kb2[9], kb2[10]);
          kb2[9] = hi_110_1;
          kb2[10] = lo_111_1;
          float hi_112_1 = fmaxf(kb2[11], kb2[12]);
          float lo_113_1 = fminf(kb2[11], kb2[12]);
          kb2[11] = hi_112_1;
          kb2[12] = lo_113_1;
          float hi_114_1 = fmaxf(kb2[6], kb2[7]);
          float lo_115_1 = fminf(kb2[6], kb2[7]);
          kb2[6] = hi_114_1;
          kb2[7] = lo_115_1;
          float hi_116_1 = fmaxf(kb2[8], kb2[9]);
          float lo_117_1 = fminf(kb2[8], kb2[9]);
          kb2[8] = hi_116_1;
          kb2[9] = lo_117_1;
          float hi_118_1 = fmaxf(a2[0], kb2[15]);
          a2[0] = hi_118_1;
          float hi_119 = fmaxf(a2[1], kb2[14]);
          a2[1] = hi_119;
          float hi_120_1 = fmaxf(a2[2], kb2[13]);
          a2[2] = hi_120_1;
          float hi_121 = fmaxf(a2[3], kb2[12]);
          a2[3] = hi_121;
          float hi_122_1 = fmaxf(a2[4], kb2[11]);
          a2[4] = hi_122_1;
          float hi_123 = fmaxf(a2[5], kb2[10]);
          a2[5] = hi_123;
          float hi_124_1 = fmaxf(a2[6], kb2[9]);
          a2[6] = hi_124_1;
          float hi_125 = fmaxf(a2[7], kb2[8]);
          a2[7] = hi_125;
          float hi_126_1 = fmaxf(a2[8], kb2[7]);
          a2[8] = hi_126_1;
          float hi_127 = fmaxf(a2[9], kb2[6]);
          a2[9] = hi_127;
          float hi_128_1 = fmaxf(a2[10], kb2[5]);
          a2[10] = hi_128_1;
          float hi_129 = fmaxf(a2[11], kb2[4]);
          a2[11] = hi_129;
          float hi_130_1 = fmaxf(a2[12], kb2[3]);
          a2[12] = hi_130_1;
          float hi_131 = fmaxf(a2[13], kb2[2]);
          a2[13] = hi_131;
          float hi_132_1 = fmaxf(a2[14], kb2[1]);
          a2[14] = hi_132_1;
          float hi_133 = fmaxf(a2[15], kb2[0]);
          a2[15] = hi_133;
          float hi_134_1 = fmaxf(a2[0], a2[8]);
          float lo_135_1 = fminf(a2[0], a2[8]);
          a2[0] = hi_134_1;
          a2[8] = lo_135_1;
          float hi_136_1 = fmaxf(a2[1], a2[9]);
          float lo_137_1 = fminf(a2[1], a2[9]);
          a2[1] = hi_136_1;
          a2[9] = lo_137_1;
          float hi_138_1 = fmaxf(a2[2], a2[10]);
          float lo_139_1 = fminf(a2[2], a2[10]);
          a2[2] = hi_138_1;
          a2[10] = lo_139_1;
          float hi_140_1 = fmaxf(a2[3], a2[11]);
          float lo_141_1 = fminf(a2[3], a2[11]);
          a2[3] = hi_140_1;
          a2[11] = lo_141_1;
          float hi_142_1 = fmaxf(a2[4], a2[12]);
          float lo_143_1 = fminf(a2[4], a2[12]);
          a2[4] = hi_142_1;
          a2[12] = lo_143_1;
          float hi_144_1 = fmaxf(a2[5], a2[13]);
          float lo_145_1 = fminf(a2[5], a2[13]);
          a2[5] = hi_144_1;
          a2[13] = lo_145_1;
          float hi_146_1 = fmaxf(a2[6], a2[14]);
          float lo_147_1 = fminf(a2[6], a2[14]);
          a2[6] = hi_146_1;
          a2[14] = lo_147_1;
          float hi_148_1 = fmaxf(a2[7], a2[15]);
          float lo_149_1 = fminf(a2[7], a2[15]);
          a2[7] = hi_148_1;
          a2[15] = lo_149_1;
          float hi_150_1 = fmaxf(a2[0], a2[4]);
          float lo_151_1 = fminf(a2[0], a2[4]);
          a2[0] = hi_150_1;
          a2[4] = lo_151_1;
          float hi_152_1 = fmaxf(a2[1], a2[5]);
          float lo_153_1 = fminf(a2[1], a2[5]);
          a2[1] = hi_152_1;
          a2[5] = lo_153_1;
          float hi_154_1 = fmaxf(a2[2], a2[6]);
          float lo_155_1 = fminf(a2[2], a2[6]);
          a2[2] = hi_154_1;
          a2[6] = lo_155_1;
          float hi_156_1 = fmaxf(a2[3], a2[7]);
          float lo_157_1 = fminf(a2[3], a2[7]);
          a2[3] = hi_156_1;
          a2[7] = lo_157_1;
          float hi_158_1 = fmaxf(a2[8], a2[12]);
          float lo_159_1 = fminf(a2[8], a2[12]);
          a2[8] = hi_158_1;
          a2[12] = lo_159_1;
          float hi_160_1 = fmaxf(a2[9], a2[13]);
          float lo_161_1 = fminf(a2[9], a2[13]);
          a2[9] = hi_160_1;
          a2[13] = lo_161_1;
          float hi_162_1 = fmaxf(a2[10], a2[14]);
          float lo_163_1 = fminf(a2[10], a2[14]);
          a2[10] = hi_162_1;
          a2[14] = lo_163_1;
          float hi_164_1 = fmaxf(a2[11], a2[15]);
          float lo_165_1 = fminf(a2[11], a2[15]);
          a2[11] = hi_164_1;
          a2[15] = lo_165_1;
          float hi_166_1 = fmaxf(a2[0], a2[2]);
          float lo_167_1 = fminf(a2[0], a2[2]);
          a2[0] = hi_166_1;
          a2[2] = lo_167_1;
          float hi_168_1 = fmaxf(a2[1], a2[3]);
          float lo_169_1 = fminf(a2[1], a2[3]);
          a2[1] = hi_168_1;
          a2[3] = lo_169_1;
          float hi_170_1 = fmaxf(a2[4], a2[6]);
          float lo_171_1 = fminf(a2[4], a2[6]);
          a2[4] = hi_170_1;
          a2[6] = lo_171_1;
          float hi_172_1 = fmaxf(a2[5], a2[7]);
          float lo_173_1 = fminf(a2[5], a2[7]);
          a2[5] = hi_172_1;
          a2[7] = lo_173_1;
          float hi_174_1 = fmaxf(a2[8], a2[10]);
          float lo_175_1 = fminf(a2[8], a2[10]);
          a2[8] = hi_174_1;
          a2[10] = lo_175_1;
          float hi_176_1 = fmaxf(a2[9], a2[11]);
          float lo_177_1 = fminf(a2[9], a2[11]);
          a2[9] = hi_176_1;
          a2[11] = lo_177_1;
          float hi_178_1 = fmaxf(a2[12], a2[14]);
          float lo_179_1 = fminf(a2[12], a2[14]);
          a2[12] = hi_178_1;
          a2[14] = lo_179_1;
          float hi_180_1 = fmaxf(a2[13], a2[15]);
          float lo_181_1 = fminf(a2[13], a2[15]);
          a2[13] = hi_180_1;
          a2[15] = lo_181_1;
          float hi_182_1 = fmaxf(a2[0], a2[1]);
          float lo_183_1 = fminf(a2[0], a2[1]);
          a2[0] = hi_182_1;
          a2[1] = lo_183_1;
          float hi_184_1 = fmaxf(a2[2], a2[3]);
          float lo_185_1 = fminf(a2[2], a2[3]);
          a2[2] = hi_184_1;
          a2[3] = lo_185_1;
          float hi_186_1 = fmaxf(a2[4], a2[5]);
          float lo_187_1 = fminf(a2[4], a2[5]);
          a2[4] = hi_186_1;
          a2[5] = lo_187_1;
          float hi_188_1 = fmaxf(a2[6], a2[7]);
          float lo_189_1 = fminf(a2[6], a2[7]);
          a2[6] = hi_188_1;
          a2[7] = lo_189_1;
          float hi_190_1 = fmaxf(a2[8], a2[9]);
          float lo_191_1 = fminf(a2[8], a2[9]);
          a2[8] = hi_190_1;
          a2[9] = lo_191_1;
          float hi_192_1 = fmaxf(a2[10], a2[11]);
          float lo_193_1 = fminf(a2[10], a2[11]);
          a2[10] = hi_192_1;
          a2[11] = lo_193_1;
          float hi_194_1 = fmaxf(a2[12], a2[13]);
          float lo_195_1 = fminf(a2[12], a2[13]);
          a2[12] = hi_194_1;
          a2[13] = lo_195_1;
          float hi_196_1 = fmaxf(a2[14], a2[15]);
          float lo_197_1 = fminf(a2[14], a2[15]);
          a2[14] = hi_196_1;
          a2[15] = lo_197_1;
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
  }
  if (w == 0 && col < total_q) {
    unsigned int o[16];
    unsigned int k1 = __as_u32(a[0]);
    unsigned int idx = 4294967295u;
    if (k1 < 4278190080u) {
      idx = k1 & CAKE_INDEX_LOW_MASK;
    }
    if (flagged != 0) {
      unsigned int k2 = __as_u32(a2[0]);
      idx = 4294967295u;
      if (k2 != 0) {
        idx = k2 & CAKE_INDEX_LOW_MASK;
      }
    }
    o[0] = idx;
    unsigned int k1_0 = __as_u32(a[1]);
    unsigned int idx_1 = 4294967295u;
    if (k1_0 < 4278190080u) {
      idx_1 = k1_0 & CAKE_INDEX_LOW_MASK;
    }
    if (flagged != 0) {
      unsigned int k2_1 = __as_u32(a2[1]);
      idx_1 = 4294967295u;
      if (k2_1 != 0) {
        idx_1 = k2_1 & CAKE_INDEX_LOW_MASK;
      }
    }
    o[1] = idx_1;
    unsigned int k1_2 = __as_u32(a[2]);
    unsigned int idx_3 = 4294967295u;
    if (k1_2 < 4278190080u) {
      idx_3 = k1_2 & CAKE_INDEX_LOW_MASK;
    }
    if (flagged != 0) {
      unsigned int k2_2 = __as_u32(a2[2]);
      idx_3 = 4294967295u;
      if (k2_2 != 0) {
        idx_3 = k2_2 & CAKE_INDEX_LOW_MASK;
      }
    }
    o[2] = idx_3;
    unsigned int k1_4 = __as_u32(a[3]);
    unsigned int idx_5 = 4294967295u;
    if (k1_4 < 4278190080u) {
      idx_5 = k1_4 & CAKE_INDEX_LOW_MASK;
    }
    if (flagged != 0) {
      unsigned int k2_3 = __as_u32(a2[3]);
      idx_5 = 4294967295u;
      if (k2_3 != 0) {
        idx_5 = k2_3 & CAKE_INDEX_LOW_MASK;
      }
    }
    o[3] = idx_5;
    unsigned int k1_6 = __as_u32(a[4]);
    unsigned int idx_7 = 4294967295u;
    if (k1_6 < 4278190080u) {
      idx_7 = k1_6 & CAKE_INDEX_LOW_MASK;
    }
    if (flagged != 0) {
      unsigned int k2_4 = __as_u32(a2[4]);
      idx_7 = 4294967295u;
      if (k2_4 != 0) {
        idx_7 = k2_4 & CAKE_INDEX_LOW_MASK;
      }
    }
    o[4] = idx_7;
    unsigned int k1_8 = __as_u32(a[5]);
    unsigned int idx_9 = 4294967295u;
    if (k1_8 < 4278190080u) {
      idx_9 = k1_8 & CAKE_INDEX_LOW_MASK;
    }
    if (flagged != 0) {
      unsigned int k2_5 = __as_u32(a2[5]);
      idx_9 = 4294967295u;
      if (k2_5 != 0) {
        idx_9 = k2_5 & CAKE_INDEX_LOW_MASK;
      }
    }
    o[5] = idx_9;
    unsigned int k1_10 = __as_u32(a[6]);
    unsigned int idx_11 = 4294967295u;
    if (k1_10 < 4278190080u) {
      idx_11 = k1_10 & CAKE_INDEX_LOW_MASK;
    }
    if (flagged != 0) {
      unsigned int k2_6 = __as_u32(a2[6]);
      idx_11 = 4294967295u;
      if (k2_6 != 0) {
        idx_11 = k2_6 & CAKE_INDEX_LOW_MASK;
      }
    }
    o[6] = idx_11;
    unsigned int k1_12 = __as_u32(a[7]);
    unsigned int idx_13 = 4294967295u;
    if (k1_12 < 4278190080u) {
      idx_13 = k1_12 & CAKE_INDEX_LOW_MASK;
    }
    if (flagged != 0) {
      unsigned int k2_7 = __as_u32(a2[7]);
      idx_13 = 4294967295u;
      if (k2_7 != 0) {
        idx_13 = k2_7 & CAKE_INDEX_LOW_MASK;
      }
    }
    o[7] = idx_13;
    unsigned int k1_14 = __as_u32(a[8]);
    unsigned int idx_15 = 4294967295u;
    if (k1_14 < 4278190080u) {
      idx_15 = k1_14 & CAKE_INDEX_LOW_MASK;
    }
    if (flagged != 0) {
      unsigned int k2_8 = __as_u32(a2[8]);
      idx_15 = 4294967295u;
      if (k2_8 != 0) {
        idx_15 = k2_8 & CAKE_INDEX_LOW_MASK;
      }
    }
    o[8] = idx_15;
    unsigned int k1_16 = __as_u32(a[9]);
    unsigned int idx_17 = 4294967295u;
    if (k1_16 < 4278190080u) {
      idx_17 = k1_16 & CAKE_INDEX_LOW_MASK;
    }
    if (flagged != 0) {
      unsigned int k2_9 = __as_u32(a2[9]);
      idx_17 = 4294967295u;
      if (k2_9 != 0) {
        idx_17 = k2_9 & CAKE_INDEX_LOW_MASK;
      }
    }
    o[9] = idx_17;
    unsigned int k1_18 = __as_u32(a[10]);
    unsigned int idx_19 = 4294967295u;
    if (k1_18 < 4278190080u) {
      idx_19 = k1_18 & CAKE_INDEX_LOW_MASK;
    }
    if (flagged != 0) {
      unsigned int k2_10 = __as_u32(a2[10]);
      idx_19 = 4294967295u;
      if (k2_10 != 0) {
        idx_19 = k2_10 & CAKE_INDEX_LOW_MASK;
      }
    }
    o[10] = idx_19;
    unsigned int k1_20 = __as_u32(a[11]);
    unsigned int idx_21 = 4294967295u;
    if (k1_20 < 4278190080u) {
      idx_21 = k1_20 & CAKE_INDEX_LOW_MASK;
    }
    if (flagged != 0) {
      unsigned int k2_11 = __as_u32(a2[11]);
      idx_21 = 4294967295u;
      if (k2_11 != 0) {
        idx_21 = k2_11 & CAKE_INDEX_LOW_MASK;
      }
    }
    o[11] = idx_21;
    unsigned int k1_22 = __as_u32(a[12]);
    unsigned int idx_23 = 4294967295u;
    if (k1_22 < 4278190080u) {
      idx_23 = k1_22 & CAKE_INDEX_LOW_MASK;
    }
    if (flagged != 0) {
      unsigned int k2_12 = __as_u32(a2[12]);
      idx_23 = 4294967295u;
      if (k2_12 != 0) {
        idx_23 = k2_12 & CAKE_INDEX_LOW_MASK;
      }
    }
    o[12] = idx_23;
    unsigned int k1_24 = __as_u32(a[13]);
    unsigned int idx_25 = 4294967295u;
    if (k1_24 < 4278190080u) {
      idx_25 = k1_24 & CAKE_INDEX_LOW_MASK;
    }
    if (flagged != 0) {
      unsigned int k2_13 = __as_u32(a2[13]);
      idx_25 = 4294967295u;
      if (k2_13 != 0) {
        idx_25 = k2_13 & CAKE_INDEX_LOW_MASK;
      }
    }
    o[13] = idx_25;
    unsigned int k1_26 = __as_u32(a[14]);
    unsigned int idx_27 = 4294967295u;
    if (k1_26 < 4278190080u) {
      idx_27 = k1_26 & CAKE_INDEX_LOW_MASK;
    }
    if (flagged != 0) {
      unsigned int k2_14 = __as_u32(a2[14]);
      idx_27 = 4294967295u;
      if (k2_14 != 0) {
        idx_27 = k2_14 & CAKE_INDEX_LOW_MASK;
      }
    }
    o[14] = idx_27;
    unsigned int k1_28 = __as_u32(a[15]);
    unsigned int idx_29 = 4294967295u;
    if (k1_28 < 4278190080u) {
      idx_29 = k1_28 & CAKE_INDEX_LOW_MASK;
    }
    if (flagged != 0) {
      unsigned int k2_15 = __as_u32(a2[15]);
      idx_29 = 4294967295u;
      if (k2_15 != 0) {
        idx_29 = k2_15 & CAKE_INDEX_LOW_MASK;
      }
    }
    o[15] = idx_29;
    unsigned int hi_3 = ((o[0]) > (o[13]) ? (o[0]) : (o[13]));
    unsigned int lo_2 = ((o[0]) < (o[13]) ? (o[0]) : (o[13]));
    o[0] = lo_2;
    o[13] = hi_3;
    unsigned int hi_30_1 = ((o[1]) > (o[12]) ? (o[1]) : (o[12]));
    unsigned int lo_31_1 = ((o[1]) < (o[12]) ? (o[1]) : (o[12]));
    o[1] = lo_31_1;
    o[12] = hi_30_1;
    unsigned int hi_32_1 = ((o[2]) > (o[15]) ? (o[2]) : (o[15]));
    unsigned int lo_33_1 = ((o[2]) < (o[15]) ? (o[2]) : (o[15]));
    o[2] = lo_33_1;
    o[15] = hi_32_1;
    unsigned int hi_34_2 = ((o[3]) > (o[14]) ? (o[3]) : (o[14]));
    unsigned int lo_35_2 = ((o[3]) < (o[14]) ? (o[3]) : (o[14]));
    o[3] = lo_35_2;
    o[14] = hi_34_2;
    unsigned int hi_36_2 = ((o[4]) > (o[8]) ? (o[4]) : (o[8]));
    unsigned int lo_37_2 = ((o[4]) < (o[8]) ? (o[4]) : (o[8]));
    o[4] = lo_37_2;
    o[8] = hi_36_2;
    unsigned int hi_38_2 = ((o[5]) > (o[6]) ? (o[5]) : (o[6]));
    unsigned int lo_39_2 = ((o[5]) < (o[6]) ? (o[5]) : (o[6]));
    o[5] = lo_39_2;
    o[6] = hi_38_2;
    unsigned int hi_40_2 = ((o[7]) > (o[11]) ? (o[7]) : (o[11]));
    unsigned int lo_41_2 = ((o[7]) < (o[11]) ? (o[7]) : (o[11]));
    o[7] = lo_41_2;
    o[11] = hi_40_2;
    unsigned int hi_42_2 = ((o[9]) > (o[10]) ? (o[9]) : (o[10]));
    unsigned int lo_43_2 = ((o[9]) < (o[10]) ? (o[9]) : (o[10]));
    o[9] = lo_43_2;
    o[10] = hi_42_2;
    unsigned int hi_44_2 = ((o[0]) > (o[5]) ? (o[0]) : (o[5]));
    unsigned int lo_45_2 = ((o[0]) < (o[5]) ? (o[0]) : (o[5]));
    o[0] = lo_45_2;
    o[5] = hi_44_2;
    unsigned int hi_46_2 = ((o[1]) > (o[7]) ? (o[1]) : (o[7]));
    unsigned int lo_47_2 = ((o[1]) < (o[7]) ? (o[1]) : (o[7]));
    o[1] = lo_47_2;
    o[7] = hi_46_2;
    unsigned int hi_48_2 = ((o[2]) > (o[9]) ? (o[2]) : (o[9]));
    unsigned int lo_49_2 = ((o[2]) < (o[9]) ? (o[2]) : (o[9]));
    o[2] = lo_49_2;
    o[9] = hi_48_2;
    unsigned int hi_50_2 = ((o[3]) > (o[4]) ? (o[3]) : (o[4]));
    unsigned int lo_51_2 = ((o[3]) < (o[4]) ? (o[3]) : (o[4]));
    o[3] = lo_51_2;
    o[4] = hi_50_2;
    unsigned int hi_52_2 = ((o[6]) > (o[13]) ? (o[6]) : (o[13]));
    unsigned int lo_53_2 = ((o[6]) < (o[13]) ? (o[6]) : (o[13]));
    o[6] = lo_53_2;
    o[13] = hi_52_2;
    unsigned int hi_54_2 = ((o[8]) > (o[14]) ? (o[8]) : (o[14]));
    unsigned int lo_55_2 = ((o[8]) < (o[14]) ? (o[8]) : (o[14]));
    o[8] = lo_55_2;
    o[14] = hi_54_2;
    unsigned int hi_56_2 = ((o[10]) > (o[15]) ? (o[10]) : (o[15]));
    unsigned int lo_57_2 = ((o[10]) < (o[15]) ? (o[10]) : (o[15]));
    o[10] = lo_57_2;
    o[15] = hi_56_2;
    unsigned int hi_58_2 = ((o[11]) > (o[12]) ? (o[11]) : (o[12]));
    unsigned int lo_59_2 = ((o[11]) < (o[12]) ? (o[11]) : (o[12]));
    o[11] = lo_59_2;
    o[12] = hi_58_2;
    unsigned int hi_60_2 = ((o[0]) > (o[1]) ? (o[0]) : (o[1]));
    unsigned int lo_61_2 = ((o[0]) < (o[1]) ? (o[0]) : (o[1]));
    o[0] = lo_61_2;
    o[1] = hi_60_2;
    unsigned int hi_62_2 = ((o[2]) > (o[3]) ? (o[2]) : (o[3]));
    unsigned int lo_63_2 = ((o[2]) < (o[3]) ? (o[2]) : (o[3]));
    o[2] = lo_63_2;
    o[3] = hi_62_2;
    unsigned int hi_64_2 = ((o[4]) > (o[5]) ? (o[4]) : (o[5]));
    unsigned int lo_65_2 = ((o[4]) < (o[5]) ? (o[4]) : (o[5]));
    o[4] = lo_65_2;
    o[5] = hi_64_2;
    unsigned int hi_66_2 = ((o[6]) > (o[8]) ? (o[6]) : (o[8]));
    unsigned int lo_67_2 = ((o[6]) < (o[8]) ? (o[6]) : (o[8]));
    o[6] = lo_67_2;
    o[8] = hi_66_2;
    unsigned int hi_68_2 = ((o[7]) > (o[9]) ? (o[7]) : (o[9]));
    unsigned int lo_69_2 = ((o[7]) < (o[9]) ? (o[7]) : (o[9]));
    o[7] = lo_69_2;
    o[9] = hi_68_2;
    unsigned int hi_70_2 = ((o[10]) > (o[11]) ? (o[10]) : (o[11]));
    unsigned int lo_71_2 = ((o[10]) < (o[11]) ? (o[10]) : (o[11]));
    o[10] = lo_71_2;
    o[11] = hi_70_2;
    unsigned int hi_72_2 = ((o[12]) > (o[13]) ? (o[12]) : (o[13]));
    unsigned int lo_73_2 = ((o[12]) < (o[13]) ? (o[12]) : (o[13]));
    o[12] = lo_73_2;
    o[13] = hi_72_2;
    unsigned int hi_74_2 = ((o[14]) > (o[15]) ? (o[14]) : (o[15]));
    unsigned int lo_75_2 = ((o[14]) < (o[15]) ? (o[14]) : (o[15]));
    o[14] = lo_75_2;
    o[15] = hi_74_2;
    unsigned int hi_76_2 = ((o[0]) > (o[2]) ? (o[0]) : (o[2]));
    unsigned int lo_77_2 = ((o[0]) < (o[2]) ? (o[0]) : (o[2]));
    o[0] = lo_77_2;
    o[2] = hi_76_2;
    unsigned int hi_78_2 = ((o[1]) > (o[3]) ? (o[1]) : (o[3]));
    unsigned int lo_79_2 = ((o[1]) < (o[3]) ? (o[1]) : (o[3]));
    o[1] = lo_79_2;
    o[3] = hi_78_2;
    unsigned int hi_80_2 = ((o[4]) > (o[10]) ? (o[4]) : (o[10]));
    unsigned int lo_81_2 = ((o[4]) < (o[10]) ? (o[4]) : (o[10]));
    o[4] = lo_81_2;
    o[10] = hi_80_2;
    unsigned int hi_82_2 = ((o[5]) > (o[11]) ? (o[5]) : (o[11]));
    unsigned int lo_83_2 = ((o[5]) < (o[11]) ? (o[5]) : (o[11]));
    o[5] = lo_83_2;
    o[11] = hi_82_2;
    unsigned int hi_84_2 = ((o[6]) > (o[7]) ? (o[6]) : (o[7]));
    unsigned int lo_85_2 = ((o[6]) < (o[7]) ? (o[6]) : (o[7]));
    o[6] = lo_85_2;
    o[7] = hi_84_2;
    unsigned int hi_86_2 = ((o[8]) > (o[9]) ? (o[8]) : (o[9]));
    unsigned int lo_87_2 = ((o[8]) < (o[9]) ? (o[8]) : (o[9]));
    o[8] = lo_87_2;
    o[9] = hi_86_2;
    unsigned int hi_88_2 = ((o[12]) > (o[14]) ? (o[12]) : (o[14]));
    unsigned int lo_89_2 = ((o[12]) < (o[14]) ? (o[12]) : (o[14]));
    o[12] = lo_89_2;
    o[14] = hi_88_2;
    unsigned int hi_90_2 = ((o[13]) > (o[15]) ? (o[13]) : (o[15]));
    unsigned int lo_91_2 = ((o[13]) < (o[15]) ? (o[13]) : (o[15]));
    o[13] = lo_91_2;
    o[15] = hi_90_2;
    unsigned int hi_92_2 = ((o[1]) > (o[2]) ? (o[1]) : (o[2]));
    unsigned int lo_93_2 = ((o[1]) < (o[2]) ? (o[1]) : (o[2]));
    o[1] = lo_93_2;
    o[2] = hi_92_2;
    unsigned int hi_94_2 = ((o[3]) > (o[12]) ? (o[3]) : (o[12]));
    unsigned int lo_95_2 = ((o[3]) < (o[12]) ? (o[3]) : (o[12]));
    o[3] = lo_95_2;
    o[12] = hi_94_2;
    unsigned int hi_96_2 = ((o[4]) > (o[6]) ? (o[4]) : (o[6]));
    unsigned int lo_97_2 = ((o[4]) < (o[6]) ? (o[4]) : (o[6]));
    o[4] = lo_97_2;
    o[6] = hi_96_2;
    unsigned int hi_98_2 = ((o[5]) > (o[7]) ? (o[5]) : (o[7]));
    unsigned int lo_99_2 = ((o[5]) < (o[7]) ? (o[5]) : (o[7]));
    o[5] = lo_99_2;
    o[7] = hi_98_2;
    unsigned int hi_100_2 = ((o[8]) > (o[10]) ? (o[8]) : (o[10]));
    unsigned int lo_101_2 = ((o[8]) < (o[10]) ? (o[8]) : (o[10]));
    o[8] = lo_101_2;
    o[10] = hi_100_2;
    unsigned int hi_102_2 = ((o[9]) > (o[11]) ? (o[9]) : (o[11]));
    unsigned int lo_103_2 = ((o[9]) < (o[11]) ? (o[9]) : (o[11]));
    o[9] = lo_103_2;
    o[11] = hi_102_2;
    unsigned int hi_104_2 = ((o[13]) > (o[14]) ? (o[13]) : (o[14]));
    unsigned int lo_105_2 = ((o[13]) < (o[14]) ? (o[13]) : (o[14]));
    o[13] = lo_105_2;
    o[14] = hi_104_2;
    unsigned int hi_106_2 = ((o[1]) > (o[4]) ? (o[1]) : (o[4]));
    unsigned int lo_107_2 = ((o[1]) < (o[4]) ? (o[1]) : (o[4]));
    o[1] = lo_107_2;
    o[4] = hi_106_2;
    unsigned int hi_108_2 = ((o[2]) > (o[6]) ? (o[2]) : (o[6]));
    unsigned int lo_109_2 = ((o[2]) < (o[6]) ? (o[2]) : (o[6]));
    o[2] = lo_109_2;
    o[6] = hi_108_2;
    unsigned int hi_110_2 = ((o[5]) > (o[8]) ? (o[5]) : (o[8]));
    unsigned int lo_111_2 = ((o[5]) < (o[8]) ? (o[5]) : (o[8]));
    o[5] = lo_111_2;
    o[8] = hi_110_2;
    unsigned int hi_112_2 = ((o[7]) > (o[10]) ? (o[7]) : (o[10]));
    unsigned int lo_113_2 = ((o[7]) < (o[10]) ? (o[7]) : (o[10]));
    o[7] = lo_113_2;
    o[10] = hi_112_2;
    unsigned int hi_114_2 = ((o[9]) > (o[13]) ? (o[9]) : (o[13]));
    unsigned int lo_115_2 = ((o[9]) < (o[13]) ? (o[9]) : (o[13]));
    o[9] = lo_115_2;
    o[13] = hi_114_2;
    unsigned int hi_116_2 = ((o[11]) > (o[14]) ? (o[11]) : (o[14]));
    unsigned int lo_117_2 = ((o[11]) < (o[14]) ? (o[11]) : (o[14]));
    o[11] = lo_117_2;
    o[14] = hi_116_2;
    unsigned int hi_118_2 = ((o[2]) > (o[4]) ? (o[2]) : (o[4]));
    unsigned int lo_119_1 = ((o[2]) < (o[4]) ? (o[2]) : (o[4]));
    o[2] = lo_119_1;
    o[4] = hi_118_2;
    unsigned int hi_120_2 = ((o[3]) > (o[6]) ? (o[3]) : (o[6]));
    unsigned int lo_121_1 = ((o[3]) < (o[6]) ? (o[3]) : (o[6]));
    o[3] = lo_121_1;
    o[6] = hi_120_2;
    unsigned int hi_122_2 = ((o[9]) > (o[12]) ? (o[9]) : (o[12]));
    unsigned int lo_123_1 = ((o[9]) < (o[12]) ? (o[9]) : (o[12]));
    o[9] = lo_123_1;
    o[12] = hi_122_2;
    unsigned int hi_124_2 = ((o[11]) > (o[13]) ? (o[11]) : (o[13]));
    unsigned int lo_125_1 = ((o[11]) < (o[13]) ? (o[11]) : (o[13]));
    o[11] = lo_125_1;
    o[13] = hi_124_2;
    unsigned int hi_126_2 = ((o[3]) > (o[5]) ? (o[3]) : (o[5]));
    unsigned int lo_127_1 = ((o[3]) < (o[5]) ? (o[3]) : (o[5]));
    o[3] = lo_127_1;
    o[5] = hi_126_2;
    unsigned int hi_128_2 = ((o[6]) > (o[8]) ? (o[6]) : (o[8]));
    unsigned int lo_129_1 = ((o[6]) < (o[8]) ? (o[6]) : (o[8]));
    o[6] = lo_129_1;
    o[8] = hi_128_2;
    unsigned int hi_130_2 = ((o[7]) > (o[9]) ? (o[7]) : (o[9]));
    unsigned int lo_131_1 = ((o[7]) < (o[9]) ? (o[7]) : (o[9]));
    o[7] = lo_131_1;
    o[9] = hi_130_2;
    unsigned int hi_132_2 = ((o[10]) > (o[12]) ? (o[10]) : (o[12]));
    unsigned int lo_133_1 = ((o[10]) < (o[12]) ? (o[10]) : (o[12]));
    o[10] = lo_133_1;
    o[12] = hi_132_2;
    unsigned int hi_134_2 = ((o[3]) > (o[4]) ? (o[3]) : (o[4]));
    unsigned int lo_135_2 = ((o[3]) < (o[4]) ? (o[3]) : (o[4]));
    o[3] = lo_135_2;
    o[4] = hi_134_2;
    unsigned int hi_136_2 = ((o[5]) > (o[6]) ? (o[5]) : (o[6]));
    unsigned int lo_137_2 = ((o[5]) < (o[6]) ? (o[5]) : (o[6]));
    o[5] = lo_137_2;
    o[6] = hi_136_2;
    unsigned int hi_138_2 = ((o[7]) > (o[8]) ? (o[7]) : (o[8]));
    unsigned int lo_139_2 = ((o[7]) < (o[8]) ? (o[7]) : (o[8]));
    o[7] = lo_139_2;
    o[8] = hi_138_2;
    unsigned int hi_140_2 = ((o[9]) > (o[10]) ? (o[9]) : (o[10]));
    unsigned int lo_141_2 = ((o[9]) < (o[10]) ? (o[9]) : (o[10]));
    o[9] = lo_141_2;
    o[10] = hi_140_2;
    unsigned int hi_142_2 = ((o[11]) > (o[12]) ? (o[11]) : (o[12]));
    unsigned int lo_143_2 = ((o[11]) < (o[12]) ? (o[11]) : (o[12]));
    o[11] = lo_143_2;
    o[12] = hi_142_2;
    unsigned int hi_144_2 = ((o[6]) > (o[7]) ? (o[6]) : (o[7]));
    unsigned int lo_145_2 = ((o[6]) < (o[7]) ? (o[6]) : (o[7]));
    o[6] = lo_145_2;
    o[7] = hi_144_2;
    unsigned int hi_146_2 = ((o[8]) > (o[9]) ? (o[8]) : (o[9]));
    unsigned int lo_147_2 = ((o[8]) < (o[9]) ? (o[8]) : (o[9]));
    o[8] = lo_147_2;
    o[9] = hi_146_2;
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
