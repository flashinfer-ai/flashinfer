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
#define SMEM_QCOL_STAGE_BYTES 8
#define SMEM_QCOL_STRIDE 8
#define SMEM_FLAGW_OFF 8
#define SMEM_FLAGW_STAGE_BYTES 16
#define SMEM_FLAGW_STRIDE 16
#define SMEM_CCNT_OFF 19480
#define SMEM_CCNT_STAGE_BYTES 8
#define SMEM_CCNT_STRIDE 8
#define SMEM_LIMC_OFF 19488
#define SMEM_LIMC_STAGE_BYTES 8
#define SMEM_LIMC_STRIDE 8
#define SMEM_CBUF_OFF 19496
#define SMEM_CBUF_STAGE_BYTES 128
#define SMEM_CBUF_STRIDE 128
#define SMEM_PUB_OFF 24
#define SMEM_PUB_STAGE_BYTES 17408
#define SMEM_PUB_STRIDE 17408
#define SMEM_Q2_OFF 17432
#define SMEM_Q2_STAGE_BYTES 2048
#define SMEM_Q2_STRIDE 2048
#define SMEM_TOTAL 19712
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(256) void kernel_cake_hopper_msa_025cfebce8c49b67ec28(
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
  unsigned int* flagw = reinterpret_cast<unsigned int*>(smem_raw + 8);
  const int flagw_addr = smem + 8;
  unsigned int* ccnt = reinterpret_cast<unsigned int*>(smem_raw + 19480);
  const int ccnt_addr = smem + 19480;
  int* limc = reinterpret_cast<int*>(smem_raw + 19488);
  const int limc_addr = smem + 19488;
  unsigned int* cbuf = reinterpret_cast<unsigned int*>(smem_raw + 19496);
  const int cbuf_addr = smem + 19496;
  float* pub = reinterpret_cast<float*>(smem_raw + 24);
  const int pub_addr = smem + 24;
  float* q2 = reinterpret_cast<float*>(smem_raw + 17432);
  const int q2_addr = smem + 17432;

  // === Task calls (dependency order) ===
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
  float kq[16];
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
  int t0_0 = w * 16;
  float sc = __uint_as_float(cb[0]);
  sc = fmaxf(sc, -1.7014118346046923e+38f);
  sc = fminf(sc, 1.7014118346046923e+38f);
  unsigned int key = __as_u32(sc) & 4294965248u | (unsigned int)t0_0;
  kb[0] = __uint_as_float(key);
  float sc_1 = __uint_as_float(cb[1]);
  sc_1 = fmaxf(sc_1, -1.7014118346046923e+38f);
  sc_1 = fminf(sc_1, 1.7014118346046923e+38f);
  unsigned int key_2 = __as_u32(sc_1) & 4294965248u | (unsigned int)(t0_0 + 1);
  kb[1] = __uint_as_float(key_2);
  float sc_3 = __uint_as_float(cb[2]);
  sc_3 = fmaxf(sc_3, -1.7014118346046923e+38f);
  sc_3 = fminf(sc_3, 1.7014118346046923e+38f);
  unsigned int key_4 = __as_u32(sc_3) & 4294965248u | (unsigned int)(t0_0 + 2);
  kb[2] = __uint_as_float(key_4);
  float sc_5 = __uint_as_float(cb[3]);
  sc_5 = fmaxf(sc_5, -1.7014118346046923e+38f);
  sc_5 = fminf(sc_5, 1.7014118346046923e+38f);
  unsigned int key_6 = __as_u32(sc_5) & 4294965248u | (unsigned int)(t0_0 + 3);
  kb[3] = __uint_as_float(key_6);
  float sc_7 = __uint_as_float(cb[4]);
  sc_7 = fmaxf(sc_7, -1.7014118346046923e+38f);
  sc_7 = fminf(sc_7, 1.7014118346046923e+38f);
  unsigned int key_8 = __as_u32(sc_7) & 4294965248u | (unsigned int)(t0_0 + 4);
  kb[4] = __uint_as_float(key_8);
  float sc_9 = __uint_as_float(cb[5]);
  sc_9 = fmaxf(sc_9, -1.7014118346046923e+38f);
  sc_9 = fminf(sc_9, 1.7014118346046923e+38f);
  unsigned int key_10 = __as_u32(sc_9) & 4294965248u | (unsigned int)(t0_0 + 5);
  kb[5] = __uint_as_float(key_10);
  float sc_11 = __uint_as_float(cb[6]);
  sc_11 = fmaxf(sc_11, -1.7014118346046923e+38f);
  sc_11 = fminf(sc_11, 1.7014118346046923e+38f);
  unsigned int key_12 = __as_u32(sc_11) & 4294965248u | (unsigned int)(t0_0 + 6);
  kb[6] = __uint_as_float(key_12);
  float sc_13 = __uint_as_float(cb[7]);
  sc_13 = fmaxf(sc_13, -1.7014118346046923e+38f);
  sc_13 = fminf(sc_13, 1.7014118346046923e+38f);
  unsigned int key_14 = __as_u32(sc_13) & 4294965248u | (unsigned int)(t0_0 + 7);
  kb[7] = __uint_as_float(key_14);
  float sc_15 = __uint_as_float(cb[8]);
  sc_15 = fmaxf(sc_15, -1.7014118346046923e+38f);
  sc_15 = fminf(sc_15, 1.7014118346046923e+38f);
  unsigned int key_16 = __as_u32(sc_15) & 4294965248u | (unsigned int)(t0_0 + 8);
  kb[8] = __uint_as_float(key_16);
  float sc_17 = __uint_as_float(cb[9]);
  sc_17 = fmaxf(sc_17, -1.7014118346046923e+38f);
  sc_17 = fminf(sc_17, 1.7014118346046923e+38f);
  unsigned int key_18 = __as_u32(sc_17) & 4294965248u | (unsigned int)(t0_0 + 9);
  kb[9] = __uint_as_float(key_18);
  float sc_19 = __uint_as_float(cb[10]);
  sc_19 = fmaxf(sc_19, -1.7014118346046923e+38f);
  sc_19 = fminf(sc_19, 1.7014118346046923e+38f);
  unsigned int key_20 = __as_u32(sc_19) & 4294965248u | (unsigned int)(t0_0 + 10);
  kb[10] = __uint_as_float(key_20);
  float sc_21 = __uint_as_float(cb[11]);
  sc_21 = fmaxf(sc_21, -1.7014118346046923e+38f);
  sc_21 = fminf(sc_21, 1.7014118346046923e+38f);
  unsigned int key_22 = __as_u32(sc_21) & 4294965248u | (unsigned int)(t0_0 + 11);
  kb[11] = __uint_as_float(key_22);
  float sc_23 = __uint_as_float(cb[12]);
  sc_23 = fmaxf(sc_23, -1.7014118346046923e+38f);
  sc_23 = fminf(sc_23, 1.7014118346046923e+38f);
  unsigned int key_24 = __as_u32(sc_23) & 4294965248u | (unsigned int)(t0_0 + 12);
  kb[12] = __uint_as_float(key_24);
  float sc_25 = __uint_as_float(cb[13]);
  sc_25 = fmaxf(sc_25, -1.7014118346046923e+38f);
  sc_25 = fminf(sc_25, 1.7014118346046923e+38f);
  unsigned int key_26 = __as_u32(sc_25) & 4294965248u | (unsigned int)(t0_0 + 13);
  kb[13] = __uint_as_float(key_26);
  float sc_27 = __uint_as_float(cb[14]);
  sc_27 = fmaxf(sc_27, -1.7014118346046923e+38f);
  sc_27 = fminf(sc_27, 1.7014118346046923e+38f);
  unsigned int key_28 = __as_u32(sc_27) & 4294965248u | (unsigned int)(t0_0 + 14);
  kb[14] = __uint_as_float(key_28);
  float sc_29 = __uint_as_float(cb[15]);
  sc_29 = fmaxf(sc_29, -1.7014118346046923e+38f);
  sc_29 = fminf(sc_29, 1.7014118346046923e+38f);
  unsigned int key_30 = __as_u32(sc_29) & 4294965248u | (unsigned int)(t0_0 + 15);
  kb[15] = __uint_as_float(key_30);
  kq[0] = kb[0];
  kq[1] = kb[1];
  kq[2] = kb[2];
  kq[3] = kb[3];
  kq[4] = kb[4];
  kq[5] = kb[5];
  kq[6] = kb[6];
  kq[7] = kb[7];
  kq[8] = kb[8];
  kq[9] = kb[9];
  kq[10] = kb[10];
  kq[11] = kb[11];
  kq[12] = kb[12];
  kq[13] = kb[13];
  kq[14] = kb[14];
  kq[15] = kb[15];
  float hi = fmaxf(kb[0], kb[13]);
  float lo = fminf(kb[0], kb[13]);
  kb[0] = hi;
  kb[13] = lo;
  float hi_31 = fmaxf(kb[1], kb[12]);
  float lo_32 = fminf(kb[1], kb[12]);
  kb[1] = hi_31;
  kb[12] = lo_32;
  float hi_33 = fmaxf(kb[2], kb[15]);
  float lo_34 = fminf(kb[2], kb[15]);
  kb[2] = hi_33;
  kb[15] = lo_34;
  float hi_35 = fmaxf(kb[3], kb[14]);
  float lo_36 = fminf(kb[3], kb[14]);
  kb[3] = hi_35;
  kb[14] = lo_36;
  float hi_37 = fmaxf(kb[4], kb[8]);
  float lo_38 = fminf(kb[4], kb[8]);
  kb[4] = hi_37;
  kb[8] = lo_38;
  float hi_39 = fmaxf(kb[5], kb[6]);
  float lo_40 = fminf(kb[5], kb[6]);
  kb[5] = hi_39;
  kb[6] = lo_40;
  float hi_41 = fmaxf(kb[7], kb[11]);
  float lo_42 = fminf(kb[7], kb[11]);
  kb[7] = hi_41;
  kb[11] = lo_42;
  float hi_43 = fmaxf(kb[9], kb[10]);
  float lo_44 = fminf(kb[9], kb[10]);
  kb[9] = hi_43;
  kb[10] = lo_44;
  float hi_45 = fmaxf(kb[0], kb[5]);
  float lo_46 = fminf(kb[0], kb[5]);
  kb[0] = hi_45;
  kb[5] = lo_46;
  float hi_47 = fmaxf(kb[1], kb[7]);
  float lo_48 = fminf(kb[1], kb[7]);
  kb[1] = hi_47;
  kb[7] = lo_48;
  float hi_49 = fmaxf(kb[2], kb[9]);
  float lo_50 = fminf(kb[2], kb[9]);
  kb[2] = hi_49;
  kb[9] = lo_50;
  float hi_51 = fmaxf(kb[3], kb[4]);
  float lo_52 = fminf(kb[3], kb[4]);
  kb[3] = hi_51;
  kb[4] = lo_52;
  float hi_53 = fmaxf(kb[6], kb[13]);
  float lo_54 = fminf(kb[6], kb[13]);
  kb[6] = hi_53;
  kb[13] = lo_54;
  float hi_55 = fmaxf(kb[8], kb[14]);
  float lo_56 = fminf(kb[8], kb[14]);
  kb[8] = hi_55;
  kb[14] = lo_56;
  float hi_57 = fmaxf(kb[10], kb[15]);
  float lo_58 = fminf(kb[10], kb[15]);
  kb[10] = hi_57;
  kb[15] = lo_58;
  float hi_59 = fmaxf(kb[11], kb[12]);
  float lo_60 = fminf(kb[11], kb[12]);
  kb[11] = hi_59;
  kb[12] = lo_60;
  float hi_61 = fmaxf(kb[0], kb[1]);
  float lo_62 = fminf(kb[0], kb[1]);
  kb[0] = hi_61;
  kb[1] = lo_62;
  float hi_63 = fmaxf(kb[2], kb[3]);
  float lo_64 = fminf(kb[2], kb[3]);
  kb[2] = hi_63;
  kb[3] = lo_64;
  float hi_65 = fmaxf(kb[4], kb[5]);
  float lo_66 = fminf(kb[4], kb[5]);
  kb[4] = hi_65;
  kb[5] = lo_66;
  float hi_67 = fmaxf(kb[6], kb[8]);
  float lo_68 = fminf(kb[6], kb[8]);
  kb[6] = hi_67;
  kb[8] = lo_68;
  float hi_69 = fmaxf(kb[7], kb[9]);
  float lo_70 = fminf(kb[7], kb[9]);
  kb[7] = hi_69;
  kb[9] = lo_70;
  float hi_71 = fmaxf(kb[10], kb[11]);
  float lo_72 = fminf(kb[10], kb[11]);
  kb[10] = hi_71;
  kb[11] = lo_72;
  float hi_73 = fmaxf(kb[12], kb[13]);
  float lo_74 = fminf(kb[12], kb[13]);
  kb[12] = hi_73;
  kb[13] = lo_74;
  float hi_75 = fmaxf(kb[14], kb[15]);
  float lo_76 = fminf(kb[14], kb[15]);
  kb[14] = hi_75;
  kb[15] = lo_76;
  float hi_77 = fmaxf(kb[0], kb[2]);
  float lo_78 = fminf(kb[0], kb[2]);
  kb[0] = hi_77;
  kb[2] = lo_78;
  float hi_79 = fmaxf(kb[1], kb[3]);
  float lo_80 = fminf(kb[1], kb[3]);
  kb[1] = hi_79;
  kb[3] = lo_80;
  float hi_81 = fmaxf(kb[4], kb[10]);
  float lo_82 = fminf(kb[4], kb[10]);
  kb[4] = hi_81;
  kb[10] = lo_82;
  float hi_83 = fmaxf(kb[5], kb[11]);
  float lo_84 = fminf(kb[5], kb[11]);
  kb[5] = hi_83;
  kb[11] = lo_84;
  float hi_85 = fmaxf(kb[6], kb[7]);
  float lo_86 = fminf(kb[6], kb[7]);
  kb[6] = hi_85;
  kb[7] = lo_86;
  float hi_87 = fmaxf(kb[8], kb[9]);
  float lo_88 = fminf(kb[8], kb[9]);
  kb[8] = hi_87;
  kb[9] = lo_88;
  float hi_89 = fmaxf(kb[12], kb[14]);
  float lo_90 = fminf(kb[12], kb[14]);
  kb[12] = hi_89;
  kb[14] = lo_90;
  float hi_91 = fmaxf(kb[13], kb[15]);
  float lo_92 = fminf(kb[13], kb[15]);
  kb[13] = hi_91;
  kb[15] = lo_92;
  float hi_93 = fmaxf(kb[1], kb[2]);
  float lo_94 = fminf(kb[1], kb[2]);
  kb[1] = hi_93;
  kb[2] = lo_94;
  float hi_95 = fmaxf(kb[3], kb[12]);
  float lo_96 = fminf(kb[3], kb[12]);
  kb[3] = hi_95;
  kb[12] = lo_96;
  float hi_97 = fmaxf(kb[4], kb[6]);
  float lo_98 = fminf(kb[4], kb[6]);
  kb[4] = hi_97;
  kb[6] = lo_98;
  float hi_99 = fmaxf(kb[5], kb[7]);
  float lo_100 = fminf(kb[5], kb[7]);
  kb[5] = hi_99;
  kb[7] = lo_100;
  float hi_101 = fmaxf(kb[8], kb[10]);
  float lo_102 = fminf(kb[8], kb[10]);
  kb[8] = hi_101;
  kb[10] = lo_102;
  float hi_103 = fmaxf(kb[9], kb[11]);
  float lo_104 = fminf(kb[9], kb[11]);
  kb[9] = hi_103;
  kb[11] = lo_104;
  float hi_105 = fmaxf(kb[13], kb[14]);
  float lo_106 = fminf(kb[13], kb[14]);
  kb[13] = hi_105;
  kb[14] = lo_106;
  float hi_107 = fmaxf(kb[1], kb[4]);
  float lo_108 = fminf(kb[1], kb[4]);
  kb[1] = hi_107;
  kb[4] = lo_108;
  float hi_109 = fmaxf(kb[2], kb[6]);
  float lo_110 = fminf(kb[2], kb[6]);
  kb[2] = hi_109;
  kb[6] = lo_110;
  float hi_111 = fmaxf(kb[5], kb[8]);
  float lo_112 = fminf(kb[5], kb[8]);
  kb[5] = hi_111;
  kb[8] = lo_112;
  float hi_113 = fmaxf(kb[7], kb[10]);
  float lo_114 = fminf(kb[7], kb[10]);
  kb[7] = hi_113;
  kb[10] = lo_114;
  float hi_115 = fmaxf(kb[9], kb[13]);
  float lo_116 = fminf(kb[9], kb[13]);
  kb[9] = hi_115;
  kb[13] = lo_116;
  float hi_117 = fmaxf(kb[11], kb[14]);
  float lo_118 = fminf(kb[11], kb[14]);
  kb[11] = hi_117;
  kb[14] = lo_118;
  float hi_119 = fmaxf(kb[2], kb[4]);
  float lo_120 = fminf(kb[2], kb[4]);
  kb[2] = hi_119;
  kb[4] = lo_120;
  float hi_121 = fmaxf(kb[3], kb[6]);
  float lo_122 = fminf(kb[3], kb[6]);
  kb[3] = hi_121;
  kb[6] = lo_122;
  float hi_123 = fmaxf(kb[9], kb[12]);
  float lo_124 = fminf(kb[9], kb[12]);
  kb[9] = hi_123;
  kb[12] = lo_124;
  float hi_125 = fmaxf(kb[11], kb[13]);
  float lo_126 = fminf(kb[11], kb[13]);
  kb[11] = hi_125;
  kb[13] = lo_126;
  float hi_127 = fmaxf(kb[3], kb[5]);
  float lo_128 = fminf(kb[3], kb[5]);
  kb[3] = hi_127;
  kb[5] = lo_128;
  float hi_129 = fmaxf(kb[6], kb[8]);
  float lo_130 = fminf(kb[6], kb[8]);
  kb[6] = hi_129;
  kb[8] = lo_130;
  float hi_131 = fmaxf(kb[7], kb[9]);
  float lo_132 = fminf(kb[7], kb[9]);
  kb[7] = hi_131;
  kb[9] = lo_132;
  float hi_133 = fmaxf(kb[10], kb[12]);
  float lo_134 = fminf(kb[10], kb[12]);
  kb[10] = hi_133;
  kb[12] = lo_134;
  float hi_135 = fmaxf(kb[3], kb[4]);
  float lo_136 = fminf(kb[3], kb[4]);
  kb[3] = hi_135;
  kb[4] = lo_136;
  float hi_137 = fmaxf(kb[5], kb[6]);
  float lo_138 = fminf(kb[5], kb[6]);
  kb[5] = hi_137;
  kb[6] = lo_138;
  float hi_139 = fmaxf(kb[7], kb[8]);
  float lo_140 = fminf(kb[7], kb[8]);
  kb[7] = hi_139;
  kb[8] = lo_140;
  float hi_141 = fmaxf(kb[9], kb[10]);
  float lo_142 = fminf(kb[9], kb[10]);
  kb[9] = hi_141;
  kb[10] = lo_142;
  float hi_143 = fmaxf(kb[11], kb[12]);
  float lo_144 = fminf(kb[11], kb[12]);
  kb[11] = hi_143;
  kb[12] = lo_144;
  float hi_145 = fmaxf(kb[6], kb[7]);
  float lo_146 = fminf(kb[6], kb[7]);
  kb[6] = hi_145;
  kb[7] = lo_146;
  float hi_147 = fmaxf(kb[8], kb[9]);
  float lo_148 = fminf(kb[8], kb[9]);
  kb[8] = hi_147;
  kb[9] = lo_148;
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
  asm volatile("barrier.sync 1, 256;" ::: "memory");
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
  float hi_149 = fmaxf(cur, pv);
  float lo_150 = fminf(cur, pv);
  cur = ((up[0] != 0) ? hi_149 : lo_150);
  float pv_151 = __shfl_xor_sync(0xFFFFFFFF, cur, 4);
  float hi_152 = fmaxf(cur, pv_151);
  float lo_153 = fminf(cur, pv_151);
  cur = ((up[1] != 0) ? hi_152 : lo_153);
  float pv_154 = __shfl_xor_sync(0xFFFFFFFF, cur, 2);
  float hi_155 = fmaxf(cur, pv_154);
  float lo_156 = fminf(cur, pv_154);
  cur = ((up[2] != 0) ? hi_155 : lo_156);
  float pv_157 = __shfl_xor_sync(0xFFFFFFFF, cur, 1);
  float hi_158 = fmaxf(cur, pv_157);
  float lo_159 = fminf(cur, pv_157);
  cur = ((up[3] != 0) ? hi_158 : lo_159);
  V[0] = cur;
  int s0_160 = ((sg * 16 + 1) * 2 + cg) * 17;
  int s1_161 = ((sg * 16 + 1 + 8) * 2 + cg) * 17;
  float x0_162 = pub[s0_160 + ln];
  float y0_163 = pub[s1_161 + lnr];
  float lo0_164 = fminf(x0_162, y0_163);
  r = fmaxf(r, lo0_164);
  float hi0_165 = fmaxf(x0_162, y0_163);
  float cur_166 = hi0_165;
  float pv_167 = __shfl_xor_sync(0xFFFFFFFF, cur_166, 8);
  float hi_168 = fmaxf(cur_166, pv_167);
  float lo_169 = fminf(cur_166, pv_167);
  cur_166 = ((up[0] != 0) ? hi_168 : lo_169);
  float pv_170 = __shfl_xor_sync(0xFFFFFFFF, cur_166, 4);
  float hi_171 = fmaxf(cur_166, pv_170);
  float lo_172 = fminf(cur_166, pv_170);
  cur_166 = ((up[1] != 0) ? hi_171 : lo_172);
  float pv_173 = __shfl_xor_sync(0xFFFFFFFF, cur_166, 2);
  float hi_174 = fmaxf(cur_166, pv_173);
  float lo_175 = fminf(cur_166, pv_173);
  cur_166 = ((up[2] != 0) ? hi_174 : lo_175);
  float pv_176 = __shfl_xor_sync(0xFFFFFFFF, cur_166, 1);
  float hi_177 = fmaxf(cur_166, pv_176);
  float lo_178 = fminf(cur_166, pv_176);
  cur_166 = ((up[3] != 0) ? hi_177 : lo_178);
  V[1] = cur_166;
  int s0_179 = ((sg * 16 + 2) * 2 + cg) * 17;
  int s1_180 = ((sg * 16 + 2 + 8) * 2 + cg) * 17;
  float x0_181 = pub[s0_179 + ln];
  float y0_182 = pub[s1_180 + lnr];
  float lo0_183 = fminf(x0_181, y0_182);
  r = fmaxf(r, lo0_183);
  float hi0_184 = fmaxf(x0_181, y0_182);
  float cur_185 = hi0_184;
  float pv_186 = __shfl_xor_sync(0xFFFFFFFF, cur_185, 8);
  float hi_187 = fmaxf(cur_185, pv_186);
  float lo_188 = fminf(cur_185, pv_186);
  cur_185 = ((up[0] != 0) ? hi_187 : lo_188);
  float pv_189 = __shfl_xor_sync(0xFFFFFFFF, cur_185, 4);
  float hi_190 = fmaxf(cur_185, pv_189);
  float lo_191 = fminf(cur_185, pv_189);
  cur_185 = ((up[1] != 0) ? hi_190 : lo_191);
  float pv_192 = __shfl_xor_sync(0xFFFFFFFF, cur_185, 2);
  float hi_193 = fmaxf(cur_185, pv_192);
  float lo_194 = fminf(cur_185, pv_192);
  cur_185 = ((up[2] != 0) ? hi_193 : lo_194);
  float pv_195 = __shfl_xor_sync(0xFFFFFFFF, cur_185, 1);
  float hi_196 = fmaxf(cur_185, pv_195);
  float lo_197 = fminf(cur_185, pv_195);
  cur_185 = ((up[3] != 0) ? hi_196 : lo_197);
  V[2] = cur_185;
  int s0_198 = ((sg * 16 + 3) * 2 + cg) * 17;
  int s1_199 = ((sg * 16 + 3 + 8) * 2 + cg) * 17;
  float x0_200 = pub[s0_198 + ln];
  float y0_201 = pub[s1_199 + lnr];
  float lo0_202 = fminf(x0_200, y0_201);
  r = fmaxf(r, lo0_202);
  float hi0_203 = fmaxf(x0_200, y0_201);
  float cur_204 = hi0_203;
  float pv_205 = __shfl_xor_sync(0xFFFFFFFF, cur_204, 8);
  float hi_206 = fmaxf(cur_204, pv_205);
  float lo_207 = fminf(cur_204, pv_205);
  cur_204 = ((up[0] != 0) ? hi_206 : lo_207);
  float pv_208 = __shfl_xor_sync(0xFFFFFFFF, cur_204, 4);
  float hi_209 = fmaxf(cur_204, pv_208);
  float lo_210 = fminf(cur_204, pv_208);
  cur_204 = ((up[1] != 0) ? hi_209 : lo_210);
  float pv_211 = __shfl_xor_sync(0xFFFFFFFF, cur_204, 2);
  float hi_212 = fmaxf(cur_204, pv_211);
  float lo_213 = fminf(cur_204, pv_211);
  cur_204 = ((up[2] != 0) ? hi_212 : lo_213);
  float pv_214 = __shfl_xor_sync(0xFFFFFFFF, cur_204, 1);
  float hi_215 = fmaxf(cur_204, pv_214);
  float lo_216 = fminf(cur_204, pv_214);
  cur_204 = ((up[3] != 0) ? hi_215 : lo_216);
  V[3] = cur_204;
  int s0_217 = ((sg * 16 + 4) * 2 + cg) * 17;
  int s1_218 = ((sg * 16 + 4 + 8) * 2 + cg) * 17;
  float x0_219 = pub[s0_217 + ln];
  float y0_220 = pub[s1_218 + lnr];
  float lo0_221 = fminf(x0_219, y0_220);
  r = fmaxf(r, lo0_221);
  float hi0_222 = fmaxf(x0_219, y0_220);
  float cur_223 = hi0_222;
  float pv_224 = __shfl_xor_sync(0xFFFFFFFF, cur_223, 8);
  float hi_225 = fmaxf(cur_223, pv_224);
  float lo_226 = fminf(cur_223, pv_224);
  cur_223 = ((up[0] != 0) ? hi_225 : lo_226);
  float pv_227 = __shfl_xor_sync(0xFFFFFFFF, cur_223, 4);
  float hi_228 = fmaxf(cur_223, pv_227);
  float lo_229 = fminf(cur_223, pv_227);
  cur_223 = ((up[1] != 0) ? hi_228 : lo_229);
  float pv_230 = __shfl_xor_sync(0xFFFFFFFF, cur_223, 2);
  float hi_231 = fmaxf(cur_223, pv_230);
  float lo_232 = fminf(cur_223, pv_230);
  cur_223 = ((up[2] != 0) ? hi_231 : lo_232);
  float pv_233 = __shfl_xor_sync(0xFFFFFFFF, cur_223, 1);
  float hi_234 = fmaxf(cur_223, pv_233);
  float lo_235 = fminf(cur_223, pv_233);
  cur_223 = ((up[3] != 0) ? hi_234 : lo_235);
  V[4] = cur_223;
  int s0_236 = ((sg * 16 + 5) * 2 + cg) * 17;
  int s1_237 = ((sg * 16 + 5 + 8) * 2 + cg) * 17;
  float x0_238 = pub[s0_236 + ln];
  float y0_239 = pub[s1_237 + lnr];
  float lo0_240 = fminf(x0_238, y0_239);
  r = fmaxf(r, lo0_240);
  float hi0_241 = fmaxf(x0_238, y0_239);
  float cur_242 = hi0_241;
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
  V[5] = cur_242;
  int s0_255 = ((sg * 16 + 6) * 2 + cg) * 17;
  int s1_256 = ((sg * 16 + 6 + 8) * 2 + cg) * 17;
  float x0_257 = pub[s0_255 + ln];
  float y0_258 = pub[s1_256 + lnr];
  float lo0_259 = fminf(x0_257, y0_258);
  r = fmaxf(r, lo0_259);
  float hi0_260 = fmaxf(x0_257, y0_258);
  float cur_261 = hi0_260;
  float pv_262 = __shfl_xor_sync(0xFFFFFFFF, cur_261, 8);
  float hi_263 = fmaxf(cur_261, pv_262);
  float lo_264 = fminf(cur_261, pv_262);
  cur_261 = ((up[0] != 0) ? hi_263 : lo_264);
  float pv_265 = __shfl_xor_sync(0xFFFFFFFF, cur_261, 4);
  float hi_266 = fmaxf(cur_261, pv_265);
  float lo_267 = fminf(cur_261, pv_265);
  cur_261 = ((up[1] != 0) ? hi_266 : lo_267);
  float pv_268 = __shfl_xor_sync(0xFFFFFFFF, cur_261, 2);
  float hi_269 = fmaxf(cur_261, pv_268);
  float lo_270 = fminf(cur_261, pv_268);
  cur_261 = ((up[2] != 0) ? hi_269 : lo_270);
  float pv_271 = __shfl_xor_sync(0xFFFFFFFF, cur_261, 1);
  float hi_272 = fmaxf(cur_261, pv_271);
  float lo_273 = fminf(cur_261, pv_271);
  cur_261 = ((up[3] != 0) ? hi_272 : lo_273);
  V[6] = cur_261;
  int s0_274 = ((sg * 16 + 7) * 2 + cg) * 17;
  int s1_275 = ((sg * 16 + 7 + 8) * 2 + cg) * 17;
  float x0_276 = pub[s0_274 + ln];
  float y0_277 = pub[s1_275 + lnr];
  float lo0_278 = fminf(x0_276, y0_277);
  r = fmaxf(r, lo0_278);
  float hi0_279 = fmaxf(x0_276, y0_277);
  float cur_280 = hi0_279;
  float pv_281 = __shfl_xor_sync(0xFFFFFFFF, cur_280, 8);
  float hi_282 = fmaxf(cur_280, pv_281);
  float lo_283 = fminf(cur_280, pv_281);
  cur_280 = ((up[0] != 0) ? hi_282 : lo_283);
  float pv_284 = __shfl_xor_sync(0xFFFFFFFF, cur_280, 4);
  float hi_285 = fmaxf(cur_280, pv_284);
  float lo_286 = fminf(cur_280, pv_284);
  cur_280 = ((up[1] != 0) ? hi_285 : lo_286);
  float pv_287 = __shfl_xor_sync(0xFFFFFFFF, cur_280, 2);
  float hi_288 = fmaxf(cur_280, pv_287);
  float lo_289 = fminf(cur_280, pv_287);
  cur_280 = ((up[2] != 0) ? hi_288 : lo_289);
  float pv_290 = __shfl_xor_sync(0xFFFFFFFF, cur_280, 1);
  float hi_291 = fmaxf(cur_280, pv_290);
  float lo_292 = fminf(cur_280, pv_290);
  cur_280 = ((up[3] != 0) ? hi_291 : lo_292);
  V[7] = cur_280;
  float rs = pub[((sg * 16 + ln) * 2 + cg) * 17 + 16];
  r = fmaxf(r, rs);
  float y1 = __shfl_xor_sync(0xFFFFFFFF, V[4], 15);
  float lo1 = fminf(V[0], y1);
  r = fmaxf(r, lo1);
  float hi1 = fmaxf(V[0], y1);
  float cur_293 = hi1;
  float pv_294 = __shfl_xor_sync(0xFFFFFFFF, cur_293, 8);
  float hi_295 = fmaxf(cur_293, pv_294);
  float lo_296 = fminf(cur_293, pv_294);
  cur_293 = ((up[0] != 0) ? hi_295 : lo_296);
  float pv_297 = __shfl_xor_sync(0xFFFFFFFF, cur_293, 4);
  float hi_298 = fmaxf(cur_293, pv_297);
  float lo_299 = fminf(cur_293, pv_297);
  cur_293 = ((up[1] != 0) ? hi_298 : lo_299);
  float pv_300 = __shfl_xor_sync(0xFFFFFFFF, cur_293, 2);
  float hi_301 = fmaxf(cur_293, pv_300);
  float lo_302 = fminf(cur_293, pv_300);
  cur_293 = ((up[2] != 0) ? hi_301 : lo_302);
  float pv_303 = __shfl_xor_sync(0xFFFFFFFF, cur_293, 1);
  float hi_304 = fmaxf(cur_293, pv_303);
  float lo_305 = fminf(cur_293, pv_303);
  cur_293 = ((up[3] != 0) ? hi_304 : lo_305);
  V[0] = cur_293;
  float y1_306 = __shfl_xor_sync(0xFFFFFFFF, V[5], 15);
  float lo1_307 = fminf(V[1], y1_306);
  r = fmaxf(r, lo1_307);
  float hi1_308 = fmaxf(V[1], y1_306);
  float cur_309 = hi1_308;
  float pv_310 = __shfl_xor_sync(0xFFFFFFFF, cur_309, 8);
  float hi_311 = fmaxf(cur_309, pv_310);
  float lo_312 = fminf(cur_309, pv_310);
  cur_309 = ((up[0] != 0) ? hi_311 : lo_312);
  float pv_313 = __shfl_xor_sync(0xFFFFFFFF, cur_309, 4);
  float hi_314 = fmaxf(cur_309, pv_313);
  float lo_315 = fminf(cur_309, pv_313);
  cur_309 = ((up[1] != 0) ? hi_314 : lo_315);
  float pv_316 = __shfl_xor_sync(0xFFFFFFFF, cur_309, 2);
  float hi_317 = fmaxf(cur_309, pv_316);
  float lo_318 = fminf(cur_309, pv_316);
  cur_309 = ((up[2] != 0) ? hi_317 : lo_318);
  float pv_319 = __shfl_xor_sync(0xFFFFFFFF, cur_309, 1);
  float hi_320 = fmaxf(cur_309, pv_319);
  float lo_321 = fminf(cur_309, pv_319);
  cur_309 = ((up[3] != 0) ? hi_320 : lo_321);
  V[1] = cur_309;
  float y1_322 = __shfl_xor_sync(0xFFFFFFFF, V[6], 15);
  float lo1_323 = fminf(V[2], y1_322);
  r = fmaxf(r, lo1_323);
  float hi1_324 = fmaxf(V[2], y1_322);
  float cur_325 = hi1_324;
  float pv_326 = __shfl_xor_sync(0xFFFFFFFF, cur_325, 8);
  float hi_327 = fmaxf(cur_325, pv_326);
  float lo_328 = fminf(cur_325, pv_326);
  cur_325 = ((up[0] != 0) ? hi_327 : lo_328);
  float pv_329 = __shfl_xor_sync(0xFFFFFFFF, cur_325, 4);
  float hi_330 = fmaxf(cur_325, pv_329);
  float lo_331 = fminf(cur_325, pv_329);
  cur_325 = ((up[1] != 0) ? hi_330 : lo_331);
  float pv_332 = __shfl_xor_sync(0xFFFFFFFF, cur_325, 2);
  float hi_333 = fmaxf(cur_325, pv_332);
  float lo_334 = fminf(cur_325, pv_332);
  cur_325 = ((up[2] != 0) ? hi_333 : lo_334);
  float pv_335 = __shfl_xor_sync(0xFFFFFFFF, cur_325, 1);
  float hi_336 = fmaxf(cur_325, pv_335);
  float lo_337 = fminf(cur_325, pv_335);
  cur_325 = ((up[3] != 0) ? hi_336 : lo_337);
  V[2] = cur_325;
  float y1_338 = __shfl_xor_sync(0xFFFFFFFF, V[7], 15);
  float lo1_339 = fminf(V[3], y1_338);
  r = fmaxf(r, lo1_339);
  float hi1_340 = fmaxf(V[3], y1_338);
  float cur_341 = hi1_340;
  float pv_342 = __shfl_xor_sync(0xFFFFFFFF, cur_341, 8);
  float hi_343 = fmaxf(cur_341, pv_342);
  float lo_344 = fminf(cur_341, pv_342);
  cur_341 = ((up[0] != 0) ? hi_343 : lo_344);
  float pv_345 = __shfl_xor_sync(0xFFFFFFFF, cur_341, 4);
  float hi_346 = fmaxf(cur_341, pv_345);
  float lo_347 = fminf(cur_341, pv_345);
  cur_341 = ((up[1] != 0) ? hi_346 : lo_347);
  float pv_348 = __shfl_xor_sync(0xFFFFFFFF, cur_341, 2);
  float hi_349 = fmaxf(cur_341, pv_348);
  float lo_350 = fminf(cur_341, pv_348);
  cur_341 = ((up[2] != 0) ? hi_349 : lo_350);
  float pv_351 = __shfl_xor_sync(0xFFFFFFFF, cur_341, 1);
  float hi_352 = fmaxf(cur_341, pv_351);
  float lo_353 = fminf(cur_341, pv_351);
  cur_341 = ((up[3] != 0) ? hi_352 : lo_353);
  V[3] = cur_341;
  float y1_354 = __shfl_xor_sync(0xFFFFFFFF, V[2], 15);
  float lo1_355 = fminf(V[0], y1_354);
  r = fmaxf(r, lo1_355);
  float hi1_356 = fmaxf(V[0], y1_354);
  float cur_357 = hi1_356;
  float pv_358 = __shfl_xor_sync(0xFFFFFFFF, cur_357, 8);
  float hi_359 = fmaxf(cur_357, pv_358);
  float lo_360 = fminf(cur_357, pv_358);
  cur_357 = ((up[0] != 0) ? hi_359 : lo_360);
  float pv_361 = __shfl_xor_sync(0xFFFFFFFF, cur_357, 4);
  float hi_362 = fmaxf(cur_357, pv_361);
  float lo_363 = fminf(cur_357, pv_361);
  cur_357 = ((up[1] != 0) ? hi_362 : lo_363);
  float pv_364 = __shfl_xor_sync(0xFFFFFFFF, cur_357, 2);
  float hi_365 = fmaxf(cur_357, pv_364);
  float lo_366 = fminf(cur_357, pv_364);
  cur_357 = ((up[2] != 0) ? hi_365 : lo_366);
  float pv_367 = __shfl_xor_sync(0xFFFFFFFF, cur_357, 1);
  float hi_368 = fmaxf(cur_357, pv_367);
  float lo_369 = fminf(cur_357, pv_367);
  cur_357 = ((up[3] != 0) ? hi_368 : lo_369);
  V[0] = cur_357;
  float y1_370 = __shfl_xor_sync(0xFFFFFFFF, V[3], 15);
  float lo1_371 = fminf(V[1], y1_370);
  r = fmaxf(r, lo1_371);
  float hi1_372 = fmaxf(V[1], y1_370);
  float cur_373 = hi1_372;
  float pv_374 = __shfl_xor_sync(0xFFFFFFFF, cur_373, 8);
  float hi_375 = fmaxf(cur_373, pv_374);
  float lo_376 = fminf(cur_373, pv_374);
  cur_373 = ((up[0] != 0) ? hi_375 : lo_376);
  float pv_377 = __shfl_xor_sync(0xFFFFFFFF, cur_373, 4);
  float hi_378 = fmaxf(cur_373, pv_377);
  float lo_379 = fminf(cur_373, pv_377);
  cur_373 = ((up[1] != 0) ? hi_378 : lo_379);
  float pv_380 = __shfl_xor_sync(0xFFFFFFFF, cur_373, 2);
  float hi_381 = fmaxf(cur_373, pv_380);
  float lo_382 = fminf(cur_373, pv_380);
  cur_373 = ((up[2] != 0) ? hi_381 : lo_382);
  float pv_383 = __shfl_xor_sync(0xFFFFFFFF, cur_373, 1);
  float hi_384 = fmaxf(cur_373, pv_383);
  float lo_385 = fminf(cur_373, pv_383);
  cur_373 = ((up[3] != 0) ? hi_384 : lo_385);
  V[1] = cur_373;
  float yl = __shfl_xor_sync(0xFFFFFFFF, V[1], 15);
  float lol = fminf(V[0], yl);
  r = fmaxf(r, lol);
  float hil = fmaxf(V[0], yl);
  float cur_386 = hil;
  float pv_387 = __shfl_xor_sync(0xFFFFFFFF, cur_386, 8);
  float hi_388 = fmaxf(cur_386, pv_387);
  float lo_389 = fminf(cur_386, pv_387);
  cur_386 = ((up[0] != 0) ? hi_388 : lo_389);
  float pv_390 = __shfl_xor_sync(0xFFFFFFFF, cur_386, 4);
  float hi_391 = fmaxf(cur_386, pv_390);
  float lo_392 = fminf(cur_386, pv_390);
  cur_386 = ((up[1] != 0) ? hi_391 : lo_392);
  float pv_393 = __shfl_xor_sync(0xFFFFFFFF, cur_386, 2);
  float hi_394 = fmaxf(cur_386, pv_393);
  float lo_395 = fminf(cur_386, pv_393);
  cur_386 = ((up[2] != 0) ? hi_394 : lo_395);
  float pv_396 = __shfl_xor_sync(0xFFFFFFFF, cur_386, 1);
  float hi_397 = fmaxf(cur_386, pv_396);
  float lo_398 = fminf(cur_386, pv_396);
  cur_386 = ((up[3] != 0) ? hi_397 : lo_398);
  V[0] = cur_386;
  float K = V[0];
  int qb = (sg * 2 + cg) * 32;
  q2[qb + ln] = K;
  q2[qb + 16 + ln] = r;
  asm volatile("barrier.sync 1, 256;" ::: "memory");
  float k2o[1];
  if (tid_1 < 32) {
    float r2v = neg_inf;
    r2v = fmaxf(r2v, q2[cg * 32 + 16 + ln]);
    r2v = fmaxf(r2v, q2[(2 + cg) * 32 + 16 + ln]);
    r2v = fmaxf(r2v, q2[(4 + cg) * 32 + 16 + ln]);
    r2v = fmaxf(r2v, q2[(6 + cg) * 32 + 16 + ln]);
    r2v = fmaxf(r2v, q2[(8 + cg) * 32 + 16 + ln]);
    r2v = fmaxf(r2v, q2[(10 + cg) * 32 + 16 + ln]);
    r2v = fmaxf(r2v, q2[(12 + cg) * 32 + 16 + ln]);
    r2v = fmaxf(r2v, q2[(14 + cg) * 32 + 16 + ln]);
    float V2[4];
    float x2 = q2[cg * 32 + ln];
    float y2 = q2[(8 + cg) * 32 + lnr];
    float lo2 = fminf(x2, y2);
    r2v = fmaxf(r2v, lo2);
    float hi2 = fmaxf(x2, y2);
    float cur_0 = hi2;
    float pv_1 = __shfl_xor_sync(0xFFFFFFFF, cur_0, 8);
    float hi_2 = fmaxf(cur_0, pv_1);
    float lo_3 = fminf(cur_0, pv_1);
    cur_0 = ((up[0] != 0) ? hi_2 : lo_3);
    float pv_4 = __shfl_xor_sync(0xFFFFFFFF, cur_0, 4);
    float hi_5 = fmaxf(cur_0, pv_4);
    float lo_6 = fminf(cur_0, pv_4);
    cur_0 = ((up[1] != 0) ? hi_5 : lo_6);
    float pv_7 = __shfl_xor_sync(0xFFFFFFFF, cur_0, 2);
    float hi_8 = fmaxf(cur_0, pv_7);
    float lo_9 = fminf(cur_0, pv_7);
    cur_0 = ((up[2] != 0) ? hi_8 : lo_9);
    float pv_10 = __shfl_xor_sync(0xFFFFFFFF, cur_0, 1);
    float hi_11 = fmaxf(cur_0, pv_10);
    float lo_12 = fminf(cur_0, pv_10);
    cur_0 = ((up[3] != 0) ? hi_11 : lo_12);
    V2[0] = cur_0;
    float x2_13 = q2[(2 + cg) * 32 + ln];
    float y2_14 = q2[(10 + cg) * 32 + lnr];
    float lo2_15 = fminf(x2_13, y2_14);
    r2v = fmaxf(r2v, lo2_15);
    float hi2_16 = fmaxf(x2_13, y2_14);
    float cur_17 = hi2_16;
    float pv_18 = __shfl_xor_sync(0xFFFFFFFF, cur_17, 8);
    float hi_19 = fmaxf(cur_17, pv_18);
    float lo_20 = fminf(cur_17, pv_18);
    cur_17 = ((up[0] != 0) ? hi_19 : lo_20);
    float pv_21 = __shfl_xor_sync(0xFFFFFFFF, cur_17, 4);
    float hi_22 = fmaxf(cur_17, pv_21);
    float lo_23 = fminf(cur_17, pv_21);
    cur_17 = ((up[1] != 0) ? hi_22 : lo_23);
    float pv_24 = __shfl_xor_sync(0xFFFFFFFF, cur_17, 2);
    float hi_25 = fmaxf(cur_17, pv_24);
    float lo_26 = fminf(cur_17, pv_24);
    cur_17 = ((up[2] != 0) ? hi_25 : lo_26);
    float pv_27 = __shfl_xor_sync(0xFFFFFFFF, cur_17, 1);
    float hi_28 = fmaxf(cur_17, pv_27);
    float lo_29 = fminf(cur_17, pv_27);
    cur_17 = ((up[3] != 0) ? hi_28 : lo_29);
    V2[1] = cur_17;
    float x2_30 = q2[(4 + cg) * 32 + ln];
    float y2_31 = q2[(12 + cg) * 32 + lnr];
    float lo2_32 = fminf(x2_30, y2_31);
    r2v = fmaxf(r2v, lo2_32);
    float hi2_33 = fmaxf(x2_30, y2_31);
    float cur_34 = hi2_33;
    float pv_35 = __shfl_xor_sync(0xFFFFFFFF, cur_34, 8);
    float hi_36 = fmaxf(cur_34, pv_35);
    float lo_37 = fminf(cur_34, pv_35);
    cur_34 = ((up[0] != 0) ? hi_36 : lo_37);
    float pv_38 = __shfl_xor_sync(0xFFFFFFFF, cur_34, 4);
    float hi_40 = fmaxf(cur_34, pv_38);
    float lo_41 = fminf(cur_34, pv_38);
    cur_34 = ((up[1] != 0) ? hi_40 : lo_41);
    float pv_42 = __shfl_xor_sync(0xFFFFFFFF, cur_34, 2);
    float hi_44 = fmaxf(cur_34, pv_42);
    float lo_45 = fminf(cur_34, pv_42);
    cur_34 = ((up[2] != 0) ? hi_44 : lo_45);
    float pv_46 = __shfl_xor_sync(0xFFFFFFFF, cur_34, 1);
    float hi_48 = fmaxf(cur_34, pv_46);
    float lo_49 = fminf(cur_34, pv_46);
    cur_34 = ((up[3] != 0) ? hi_48 : lo_49);
    V2[2] = cur_34;
    float x2_50 = q2[(6 + cg) * 32 + ln];
    float y2_51 = q2[(14 + cg) * 32 + lnr];
    float lo2_52 = fminf(x2_50, y2_51);
    r2v = fmaxf(r2v, lo2_52);
    float hi2_53 = fmaxf(x2_50, y2_51);
    float cur_54 = hi2_53;
    float pv_55 = __shfl_xor_sync(0xFFFFFFFF, cur_54, 8);
    float hi_56 = fmaxf(cur_54, pv_55);
    float lo_57 = fminf(cur_54, pv_55);
    cur_54 = ((up[0] != 0) ? hi_56 : lo_57);
    float pv_58 = __shfl_xor_sync(0xFFFFFFFF, cur_54, 4);
    float hi_60 = fmaxf(cur_54, pv_58);
    float lo_61 = fminf(cur_54, pv_58);
    cur_54 = ((up[1] != 0) ? hi_60 : lo_61);
    float pv_62 = __shfl_xor_sync(0xFFFFFFFF, cur_54, 2);
    float hi_64 = fmaxf(cur_54, pv_62);
    float lo_65 = fminf(cur_54, pv_62);
    cur_54 = ((up[2] != 0) ? hi_64 : lo_65);
    float pv_66 = __shfl_xor_sync(0xFFFFFFFF, cur_54, 1);
    float hi_68 = fmaxf(cur_54, pv_66);
    float lo_69 = fminf(cur_54, pv_66);
    cur_54 = ((up[3] != 0) ? hi_68 : lo_69);
    V2[3] = cur_54;
    float y3 = __shfl_xor_sync(0xFFFFFFFF, V2[2], 15);
    float lo3 = fminf(V2[0], y3);
    r2v = fmaxf(r2v, lo3);
    float hi3 = fmaxf(V2[0], y3);
    float cur_70 = hi3;
    float pv_71 = __shfl_xor_sync(0xFFFFFFFF, cur_70, 8);
    float hi_72 = fmaxf(cur_70, pv_71);
    float lo_73 = fminf(cur_70, pv_71);
    cur_70 = ((up[0] != 0) ? hi_72 : lo_73);
    float pv_74 = __shfl_xor_sync(0xFFFFFFFF, cur_70, 4);
    float hi_76 = fmaxf(cur_70, pv_74);
    float lo_77 = fminf(cur_70, pv_74);
    cur_70 = ((up[1] != 0) ? hi_76 : lo_77);
    float pv_78 = __shfl_xor_sync(0xFFFFFFFF, cur_70, 2);
    float hi_80 = fmaxf(cur_70, pv_78);
    float lo_81 = fminf(cur_70, pv_78);
    cur_70 = ((up[2] != 0) ? hi_80 : lo_81);
    float pv_82 = __shfl_xor_sync(0xFFFFFFFF, cur_70, 1);
    float hi_84 = fmaxf(cur_70, pv_82);
    float lo_85 = fminf(cur_70, pv_82);
    cur_70 = ((up[3] != 0) ? hi_84 : lo_85);
    V2[0] = cur_70;
    float y3_86 = __shfl_xor_sync(0xFFFFFFFF, V2[3], 15);
    float lo3_87 = fminf(V2[1], y3_86);
    r2v = fmaxf(r2v, lo3_87);
    float hi3_88 = fmaxf(V2[1], y3_86);
    float cur_89 = hi3_88;
    float pv_90 = __shfl_xor_sync(0xFFFFFFFF, cur_89, 8);
    float hi_92 = fmaxf(cur_89, pv_90);
    float lo_93 = fminf(cur_89, pv_90);
    cur_89 = ((up[0] != 0) ? hi_92 : lo_93);
    float pv_94 = __shfl_xor_sync(0xFFFFFFFF, cur_89, 4);
    float hi_96 = fmaxf(cur_89, pv_94);
    float lo_97 = fminf(cur_89, pv_94);
    cur_89 = ((up[1] != 0) ? hi_96 : lo_97);
    float pv_98 = __shfl_xor_sync(0xFFFFFFFF, cur_89, 2);
    float hi_100 = fmaxf(cur_89, pv_98);
    float lo_101 = fminf(cur_89, pv_98);
    cur_89 = ((up[2] != 0) ? hi_100 : lo_101);
    float pv_102 = __shfl_xor_sync(0xFFFFFFFF, cur_89, 1);
    float hi_104 = fmaxf(cur_89, pv_102);
    float lo_105 = fminf(cur_89, pv_102);
    cur_89 = ((up[3] != 0) ? hi_104 : lo_105);
    V2[1] = cur_89;
    float yl2 = __shfl_xor_sync(0xFFFFFFFF, V2[1], 15);
    float lol2 = fminf(V2[0], yl2);
    r2v = fmaxf(r2v, lol2);
    V2[0] = fmaxf(V2[0], yl2);
    k2o[0] = V2[0];
    K = k2o[0];
    r = r2v;
  }
  rr1[0] = r;
  int ucol = blockIdx.x * 2 + cg;
  int commit = 0;
  if (g < 2 && ucol < total_q) {
    commit = 1;
  }
  if (tid_1 < 32) {
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
    if (ln == 0 && g < 2) {
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
      int anyq = 0;
      if ((__as_u32(a[0]) & 4294965248u) == qc) {
        anyq = 1;
      }
      if ((__as_u32(a[1]) & 4294965248u) == qc) {
        anyq = 1;
      }
      if ((__as_u32(a[2]) & 4294965248u) == qc) {
        anyq = 1;
      }
      if ((__as_u32(a[3]) & 4294965248u) == qc) {
        anyq = 1;
      }
      if ((__as_u32(a[4]) & 4294965248u) == qc) {
        anyq = 1;
      }
      if ((__as_u32(a[5]) & 4294965248u) == qc) {
        anyq = 1;
      }
      if ((__as_u32(a[6]) & 4294965248u) == qc) {
        anyq = 1;
      }
      if ((__as_u32(a[7]) & 4294965248u) == qc) {
        anyq = 1;
      }
      if ((__as_u32(a[8]) & 4294965248u) == qc) {
        anyq = 1;
      }
      if ((__as_u32(a[9]) & 4294965248u) == qc) {
        anyq = 1;
      }
      if ((__as_u32(a[10]) & 4294965248u) == qc) {
        anyq = 1;
      }
      if ((__as_u32(a[11]) & 4294965248u) == qc) {
        anyq = 1;
      }
      if ((__as_u32(a[12]) & 4294965248u) == qc) {
        anyq = 1;
      }
      if ((__as_u32(a[13]) & 4294965248u) == qc) {
        anyq = 1;
      }
      if ((__as_u32(a[14]) & 4294965248u) == qc) {
        anyq = 1;
      }
      if ((__as_u32(a[15]) & 4294965248u) == qc) {
        anyq = 1;
      }
      if ((__as_u32(rej) & 4294965248u) == qc) {
        anyq = 1;
      }
      if (anyq != 0) {
        int t0_1 = w * 16;
        unsigned int mm = 0;
        if ((__as_u32(kq[0]) & 4294965248u) == qc) {
          mm = mm | 1;
        }
        if ((__as_u32(kq[1]) & 4294965248u) == qc) {
          mm = mm | 2;
        }
        if ((__as_u32(kq[2]) & 4294965248u) == qc) {
          mm = mm | 4;
        }
        if ((__as_u32(kq[3]) & 4294965248u) == qc) {
          mm = mm | 8;
        }
        if ((__as_u32(kq[4]) & 4294965248u) == qc) {
          mm = mm | 16;
        }
        if ((__as_u32(kq[5]) & 4294965248u) == qc) {
          mm = mm | 32;
        }
        if ((__as_u32(kq[6]) & 4294965248u) == qc) {
          mm = mm | 64;
        }
        if ((__as_u32(kq[7]) & 4294965248u) == qc) {
          mm = mm | 128;
        }
        if ((__as_u32(kq[8]) & 4294965248u) == qc) {
          mm = mm | 256;
        }
        if ((__as_u32(kq[9]) & 4294965248u) == qc) {
          mm = mm | 512;
        }
        if ((__as_u32(kq[10]) & 4294965248u) == qc) {
          mm = mm | 1024;
        }
        if ((__as_u32(kq[11]) & 4294965248u) == qc) {
          mm = mm | 2048;
        }
        if ((__as_u32(kq[12]) & 4294965248u) == qc) {
          mm = mm | 4096;
        }
        if ((__as_u32(kq[13]) & 4294965248u) == qc) {
          mm = mm | 8192;
        }
        if ((__as_u32(kq[14]) & 4294965248u) == qc) {
          mm = mm | 16384;
        }
        if ((__as_u32(kq[15]) & 4294965248u) == qc) {
          mm = mm | 32768;
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
              sc_0 = fmaxf(sc_0, -1.7014118346046923e+38f);
              sc_0 = fminf(sc_0, 1.7014118346046923e+38f);
              unsigned int u9 = __as_u32(sc_0);
              unsigned int lowb9 = (u9 ^ (unsigned int)((int)u9 >> 31) & 2047) & 2047;
              unsigned int km9 = 536870912 | lowb9 << 11 | (unsigned int)t0_1;
              cbuf[pos9] = km9;
              pos9 = pos9 + 1;
            }
            if ((mm & 2) != 0) {
              float sc_0_1 = __uint_as_float(cb[1]);
              sc_0_1 = fmaxf(sc_0_1, -1.7014118346046923e+38f);
              sc_0_1 = fminf(sc_0_1, 1.7014118346046923e+38f);
              unsigned int u9_1 = __as_u32(sc_0_1);
              unsigned int lowb9_1 = (u9_1 ^ (unsigned int)((int)u9_1 >> 31) & 2047) & 2047;
              unsigned int km9_1 = 536870912 | lowb9_1 << 11 | (unsigned int)(t0_1 + 1);
              cbuf[pos9] = km9_1;
              pos9 = pos9 + 1;
            }
            if ((mm & 4) != 0) {
              float sc_0_2 = __uint_as_float(cb[2]);
              sc_0_2 = fmaxf(sc_0_2, -1.7014118346046923e+38f);
              sc_0_2 = fminf(sc_0_2, 1.7014118346046923e+38f);
              unsigned int u9_2 = __as_u32(sc_0_2);
              unsigned int lowb9_2 = (u9_2 ^ (unsigned int)((int)u9_2 >> 31) & 2047) & 2047;
              unsigned int km9_2 = 536870912 | lowb9_2 << 11 | (unsigned int)(t0_1 + 2);
              cbuf[pos9] = km9_2;
              pos9 = pos9 + 1;
            }
            if ((mm & 8) != 0) {
              float sc_0_3 = __uint_as_float(cb[3]);
              sc_0_3 = fmaxf(sc_0_3, -1.7014118346046923e+38f);
              sc_0_3 = fminf(sc_0_3, 1.7014118346046923e+38f);
              unsigned int u9_3 = __as_u32(sc_0_3);
              unsigned int lowb9_3 = (u9_3 ^ (unsigned int)((int)u9_3 >> 31) & 2047) & 2047;
              unsigned int km9_3 = 536870912 | lowb9_3 << 11 | (unsigned int)(t0_1 + 3);
              cbuf[pos9] = km9_3;
              pos9 = pos9 + 1;
            }
            if ((mm & 16) != 0) {
              float sc_0_4 = __uint_as_float(cb[4]);
              sc_0_4 = fmaxf(sc_0_4, -1.7014118346046923e+38f);
              sc_0_4 = fminf(sc_0_4, 1.7014118346046923e+38f);
              unsigned int u9_4 = __as_u32(sc_0_4);
              unsigned int lowb9_4 = (u9_4 ^ (unsigned int)((int)u9_4 >> 31) & 2047) & 2047;
              unsigned int km9_4 = 536870912 | lowb9_4 << 11 | (unsigned int)(t0_1 + 4);
              cbuf[pos9] = km9_4;
              pos9 = pos9 + 1;
            }
            if ((mm & 32) != 0) {
              float sc_0_5 = __uint_as_float(cb[5]);
              sc_0_5 = fmaxf(sc_0_5, -1.7014118346046923e+38f);
              sc_0_5 = fminf(sc_0_5, 1.7014118346046923e+38f);
              unsigned int u9_5 = __as_u32(sc_0_5);
              unsigned int lowb9_5 = (u9_5 ^ (unsigned int)((int)u9_5 >> 31) & 2047) & 2047;
              unsigned int km9_5 = 536870912 | lowb9_5 << 11 | (unsigned int)(t0_1 + 5);
              cbuf[pos9] = km9_5;
              pos9 = pos9 + 1;
            }
            if ((mm & 64) != 0) {
              float sc_0_6 = __uint_as_float(cb[6]);
              sc_0_6 = fmaxf(sc_0_6, -1.7014118346046923e+38f);
              sc_0_6 = fminf(sc_0_6, 1.7014118346046923e+38f);
              unsigned int u9_6 = __as_u32(sc_0_6);
              unsigned int lowb9_6 = (u9_6 ^ (unsigned int)((int)u9_6 >> 31) & 2047) & 2047;
              unsigned int km9_6 = 536870912 | lowb9_6 << 11 | (unsigned int)(t0_1 + 6);
              cbuf[pos9] = km9_6;
              pos9 = pos9 + 1;
            }
            if ((mm & 128) != 0) {
              float sc_0_7 = __uint_as_float(cb[7]);
              sc_0_7 = fmaxf(sc_0_7, -1.7014118346046923e+38f);
              sc_0_7 = fminf(sc_0_7, 1.7014118346046923e+38f);
              unsigned int u9_7 = __as_u32(sc_0_7);
              unsigned int lowb9_7 = (u9_7 ^ (unsigned int)((int)u9_7 >> 31) & 2047) & 2047;
              unsigned int km9_7 = 536870912 | lowb9_7 << 11 | (unsigned int)(t0_1 + 7);
              cbuf[pos9] = km9_7;
              pos9 = pos9 + 1;
            }
            if ((mm & 256) != 0) {
              float sc_0_8 = __uint_as_float(cb[8]);
              sc_0_8 = fmaxf(sc_0_8, -1.7014118346046923e+38f);
              sc_0_8 = fminf(sc_0_8, 1.7014118346046923e+38f);
              unsigned int u9_8 = __as_u32(sc_0_8);
              unsigned int lowb9_8 = (u9_8 ^ (unsigned int)((int)u9_8 >> 31) & 2047) & 2047;
              unsigned int km9_8 = 536870912 | lowb9_8 << 11 | (unsigned int)(t0_1 + 8);
              cbuf[pos9] = km9_8;
              pos9 = pos9 + 1;
            }
            if ((mm & 512) != 0) {
              float sc_0_9 = __uint_as_float(cb[9]);
              sc_0_9 = fmaxf(sc_0_9, -1.7014118346046923e+38f);
              sc_0_9 = fminf(sc_0_9, 1.7014118346046923e+38f);
              unsigned int u9_9 = __as_u32(sc_0_9);
              unsigned int lowb9_9 = (u9_9 ^ (unsigned int)((int)u9_9 >> 31) & 2047) & 2047;
              unsigned int km9_9 = 536870912 | lowb9_9 << 11 | (unsigned int)(t0_1 + 9);
              cbuf[pos9] = km9_9;
              pos9 = pos9 + 1;
            }
            if ((mm & 1024) != 0) {
              float sc_0_10 = __uint_as_float(cb[10]);
              sc_0_10 = fmaxf(sc_0_10, -1.7014118346046923e+38f);
              sc_0_10 = fminf(sc_0_10, 1.7014118346046923e+38f);
              unsigned int u9_10 = __as_u32(sc_0_10);
              unsigned int lowb9_10 = (u9_10 ^ (unsigned int)((int)u9_10 >> 31) & 2047) & 2047;
              unsigned int km9_10 = 536870912 | lowb9_10 << 11 | (unsigned int)(t0_1 + 10);
              cbuf[pos9] = km9_10;
              pos9 = pos9 + 1;
            }
            if ((mm & 2048) != 0) {
              float sc_0_11 = __uint_as_float(cb[11]);
              sc_0_11 = fmaxf(sc_0_11, -1.7014118346046923e+38f);
              sc_0_11 = fminf(sc_0_11, 1.7014118346046923e+38f);
              unsigned int u9_11 = __as_u32(sc_0_11);
              unsigned int lowb9_11 = (u9_11 ^ (unsigned int)((int)u9_11 >> 31) & 2047) & 2047;
              unsigned int km9_11 = 536870912 | lowb9_11 << 11 | (unsigned int)(t0_1 + 11);
              cbuf[pos9] = km9_11;
              pos9 = pos9 + 1;
            }
            if ((mm & 4096) != 0) {
              float sc_0_12 = __uint_as_float(cb[12]);
              sc_0_12 = fmaxf(sc_0_12, -1.7014118346046923e+38f);
              sc_0_12 = fminf(sc_0_12, 1.7014118346046923e+38f);
              unsigned int u9_12 = __as_u32(sc_0_12);
              unsigned int lowb9_12 = (u9_12 ^ (unsigned int)((int)u9_12 >> 31) & 2047) & 2047;
              unsigned int km9_12 = 536870912 | lowb9_12 << 11 | (unsigned int)(t0_1 + 12);
              cbuf[pos9] = km9_12;
              pos9 = pos9 + 1;
            }
            if ((mm & 8192) != 0) {
              float sc_0_13 = __uint_as_float(cb[13]);
              sc_0_13 = fmaxf(sc_0_13, -1.7014118346046923e+38f);
              sc_0_13 = fminf(sc_0_13, 1.7014118346046923e+38f);
              unsigned int u9_13 = __as_u32(sc_0_13);
              unsigned int lowb9_13 = (u9_13 ^ (unsigned int)((int)u9_13 >> 31) & 2047) & 2047;
              unsigned int km9_13 = 536870912 | lowb9_13 << 11 | (unsigned int)(t0_1 + 13);
              cbuf[pos9] = km9_13;
              pos9 = pos9 + 1;
            }
            if ((mm & 16384) != 0) {
              float sc_0_14 = __uint_as_float(cb[14]);
              sc_0_14 = fmaxf(sc_0_14, -1.7014118346046923e+38f);
              sc_0_14 = fminf(sc_0_14, 1.7014118346046923e+38f);
              unsigned int u9_14 = __as_u32(sc_0_14);
              unsigned int lowb9_14 = (u9_14 ^ (unsigned int)((int)u9_14 >> 31) & 2047) & 2047;
              unsigned int km9_14 = 536870912 | lowb9_14 << 11 | (unsigned int)(t0_1 + 14);
              cbuf[pos9] = km9_14;
              pos9 = pos9 + 1;
            }
            if ((mm & 32768) != 0) {
              float sc_0_15 = __uint_as_float(cb[15]);
              sc_0_15 = fmaxf(sc_0_15, -1.7014118346046923e+38f);
              sc_0_15 = fminf(sc_0_15, 1.7014118346046923e+38f);
              unsigned int u9_15 = __as_u32(sc_0_15);
              unsigned int lowb9_15 = (u9_15 ^ (unsigned int)((int)u9_15 >> 31) & 2047) & 2047;
              unsigned int km9_15 = 536870912 | lowb9_15 << 11 | (unsigned int)(t0_1 + 15);
              cbuf[pos9] = km9_15;
              pos9 = pos9 + 1;
            }
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
        float kb2o[16];
        int t0_1_1 = w * 16;
        float qf = __uint_as_float(qc);
        float sc_2 = __uint_as_float(cb[0]);
        sc_2 = fmaxf(sc_2, -1.7014118346046923e+38f);
        sc_2 = fminf(sc_2, 1.7014118346046923e+38f);
        unsigned int u = __as_u32(sc_2);
        unsigned int cls = u & 4294965248u;
        unsigned int key_3 = 0;
        if (qf < __uint_as_float(cls) && cls < 4278190080u) {
          key_3 = 1073741824 | (unsigned int)t0_1_1;
        }
        if (cls == qc) {
          unsigned int lowb = (u ^ (unsigned int)((int)u >> 31) & 2047) & 2047;
          key_3 = 536870912 | lowb << 11 | (unsigned int)t0_1_1;
        }
        kb2o[0] = __uint_as_float(key_3);
        float sc_4 = __uint_as_float(cb[1]);
        sc_4 = fmaxf(sc_4, -1.7014118346046923e+38f);
        sc_4 = fminf(sc_4, 1.7014118346046923e+38f);
        unsigned int u_5 = __as_u32(sc_4);
        unsigned int cls_6 = u_5 & 4294965248u;
        unsigned int key_7 = 0;
        if (qf < __uint_as_float(cls_6) && cls_6 < 4278190080u) {
          key_7 = 1073741824 | (unsigned int)(t0_1_1 + 1);
        }
        if (cls_6 == qc) {
          unsigned int lowb_1 = (u_5 ^ (unsigned int)((int)u_5 >> 31) & 2047) & 2047;
          key_7 = 536870912 | lowb_1 << 11 | (unsigned int)(t0_1_1 + 1);
        }
        kb2o[1] = __uint_as_float(key_7);
        float sc_8 = __uint_as_float(cb[2]);
        sc_8 = fmaxf(sc_8, -1.7014118346046923e+38f);
        sc_8 = fminf(sc_8, 1.7014118346046923e+38f);
        unsigned int u_9 = __as_u32(sc_8);
        unsigned int cls_10 = u_9 & 4294965248u;
        unsigned int key_11 = 0;
        if (qf < __uint_as_float(cls_10) && cls_10 < 4278190080u) {
          key_11 = 1073741824 | (unsigned int)(t0_1_1 + 2);
        }
        if (cls_10 == qc) {
          unsigned int lowb_2 = (u_9 ^ (unsigned int)((int)u_9 >> 31) & 2047) & 2047;
          key_11 = 536870912 | lowb_2 << 11 | (unsigned int)(t0_1_1 + 2);
        }
        kb2o[2] = __uint_as_float(key_11);
        float sc_12 = __uint_as_float(cb[3]);
        sc_12 = fmaxf(sc_12, -1.7014118346046923e+38f);
        sc_12 = fminf(sc_12, 1.7014118346046923e+38f);
        unsigned int u_13 = __as_u32(sc_12);
        unsigned int cls_14 = u_13 & 4294965248u;
        unsigned int key_15 = 0;
        if (qf < __uint_as_float(cls_14) && cls_14 < 4278190080u) {
          key_15 = 1073741824 | (unsigned int)(t0_1_1 + 3);
        }
        if (cls_14 == qc) {
          unsigned int lowb_3 = (u_13 ^ (unsigned int)((int)u_13 >> 31) & 2047) & 2047;
          key_15 = 536870912 | lowb_3 << 11 | (unsigned int)(t0_1_1 + 3);
        }
        kb2o[3] = __uint_as_float(key_15);
        float sc_16 = __uint_as_float(cb[4]);
        sc_16 = fmaxf(sc_16, -1.7014118346046923e+38f);
        sc_16 = fminf(sc_16, 1.7014118346046923e+38f);
        unsigned int u_17 = __as_u32(sc_16);
        unsigned int cls_18 = u_17 & 4294965248u;
        unsigned int key_19 = 0;
        if (qf < __uint_as_float(cls_18) && cls_18 < 4278190080u) {
          key_19 = 1073741824 | (unsigned int)(t0_1_1 + 4);
        }
        if (cls_18 == qc) {
          unsigned int lowb_4 = (u_17 ^ (unsigned int)((int)u_17 >> 31) & 2047) & 2047;
          key_19 = 536870912 | lowb_4 << 11 | (unsigned int)(t0_1_1 + 4);
        }
        kb2o[4] = __uint_as_float(key_19);
        float sc_20 = __uint_as_float(cb[5]);
        sc_20 = fmaxf(sc_20, -1.7014118346046923e+38f);
        sc_20 = fminf(sc_20, 1.7014118346046923e+38f);
        unsigned int u_21 = __as_u32(sc_20);
        unsigned int cls_22 = u_21 & 4294965248u;
        unsigned int key_23 = 0;
        if (qf < __uint_as_float(cls_22) && cls_22 < 4278190080u) {
          key_23 = 1073741824 | (unsigned int)(t0_1_1 + 5);
        }
        if (cls_22 == qc) {
          unsigned int lowb_5 = (u_21 ^ (unsigned int)((int)u_21 >> 31) & 2047) & 2047;
          key_23 = 536870912 | lowb_5 << 11 | (unsigned int)(t0_1_1 + 5);
        }
        kb2o[5] = __uint_as_float(key_23);
        float sc_24 = __uint_as_float(cb[6]);
        sc_24 = fmaxf(sc_24, -1.7014118346046923e+38f);
        sc_24 = fminf(sc_24, 1.7014118346046923e+38f);
        unsigned int u_25 = __as_u32(sc_24);
        unsigned int cls_26 = u_25 & 4294965248u;
        unsigned int key_27 = 0;
        if (qf < __uint_as_float(cls_26) && cls_26 < 4278190080u) {
          key_27 = 1073741824 | (unsigned int)(t0_1_1 + 6);
        }
        if (cls_26 == qc) {
          unsigned int lowb_6 = (u_25 ^ (unsigned int)((int)u_25 >> 31) & 2047) & 2047;
          key_27 = 536870912 | lowb_6 << 11 | (unsigned int)(t0_1_1 + 6);
        }
        kb2o[6] = __uint_as_float(key_27);
        float sc_28 = __uint_as_float(cb[7]);
        sc_28 = fmaxf(sc_28, -1.7014118346046923e+38f);
        sc_28 = fminf(sc_28, 1.7014118346046923e+38f);
        unsigned int u_29 = __as_u32(sc_28);
        unsigned int cls_30 = u_29 & 4294965248u;
        unsigned int key_31 = 0;
        if (qf < __uint_as_float(cls_30) && cls_30 < 4278190080u) {
          key_31 = 1073741824 | (unsigned int)(t0_1_1 + 7);
        }
        if (cls_30 == qc) {
          unsigned int lowb_7 = (u_29 ^ (unsigned int)((int)u_29 >> 31) & 2047) & 2047;
          key_31 = 536870912 | lowb_7 << 11 | (unsigned int)(t0_1_1 + 7);
        }
        kb2o[7] = __uint_as_float(key_31);
        float sc_32 = __uint_as_float(cb[8]);
        sc_32 = fmaxf(sc_32, -1.7014118346046923e+38f);
        sc_32 = fminf(sc_32, 1.7014118346046923e+38f);
        unsigned int u_33 = __as_u32(sc_32);
        unsigned int cls_34 = u_33 & 4294965248u;
        unsigned int key_35 = 0;
        if (qf < __uint_as_float(cls_34) && cls_34 < 4278190080u) {
          key_35 = 1073741824 | (unsigned int)(t0_1_1 + 8);
        }
        if (cls_34 == qc) {
          unsigned int lowb_8 = (u_33 ^ (unsigned int)((int)u_33 >> 31) & 2047) & 2047;
          key_35 = 536870912 | lowb_8 << 11 | (unsigned int)(t0_1_1 + 8);
        }
        kb2o[8] = __uint_as_float(key_35);
        float sc_36 = __uint_as_float(cb[9]);
        sc_36 = fmaxf(sc_36, -1.7014118346046923e+38f);
        sc_36 = fminf(sc_36, 1.7014118346046923e+38f);
        unsigned int u_37 = __as_u32(sc_36);
        unsigned int cls_38 = u_37 & 4294965248u;
        unsigned int key_39 = 0;
        if (qf < __uint_as_float(cls_38) && cls_38 < 4278190080u) {
          key_39 = 1073741824 | (unsigned int)(t0_1_1 + 9);
        }
        if (cls_38 == qc) {
          unsigned int lowb_9 = (u_37 ^ (unsigned int)((int)u_37 >> 31) & 2047) & 2047;
          key_39 = 536870912 | lowb_9 << 11 | (unsigned int)(t0_1_1 + 9);
        }
        kb2o[9] = __uint_as_float(key_39);
        float sc_40 = __uint_as_float(cb[10]);
        sc_40 = fmaxf(sc_40, -1.7014118346046923e+38f);
        sc_40 = fminf(sc_40, 1.7014118346046923e+38f);
        unsigned int u_41 = __as_u32(sc_40);
        unsigned int cls_42 = u_41 & 4294965248u;
        unsigned int key_43 = 0;
        if (qf < __uint_as_float(cls_42) && cls_42 < 4278190080u) {
          key_43 = 1073741824 | (unsigned int)(t0_1_1 + 10);
        }
        if (cls_42 == qc) {
          unsigned int lowb_10 = (u_41 ^ (unsigned int)((int)u_41 >> 31) & 2047) & 2047;
          key_43 = 536870912 | lowb_10 << 11 | (unsigned int)(t0_1_1 + 10);
        }
        kb2o[10] = __uint_as_float(key_43);
        float sc_44 = __uint_as_float(cb[11]);
        sc_44 = fmaxf(sc_44, -1.7014118346046923e+38f);
        sc_44 = fminf(sc_44, 1.7014118346046923e+38f);
        unsigned int u_45 = __as_u32(sc_44);
        unsigned int cls_46 = u_45 & 4294965248u;
        unsigned int key_47 = 0;
        if (qf < __uint_as_float(cls_46) && cls_46 < 4278190080u) {
          key_47 = 1073741824 | (unsigned int)(t0_1_1 + 11);
        }
        if (cls_46 == qc) {
          unsigned int lowb_11 = (u_45 ^ (unsigned int)((int)u_45 >> 31) & 2047) & 2047;
          key_47 = 536870912 | lowb_11 << 11 | (unsigned int)(t0_1_1 + 11);
        }
        kb2o[11] = __uint_as_float(key_47);
        float sc_48 = __uint_as_float(cb[12]);
        sc_48 = fmaxf(sc_48, -1.7014118346046923e+38f);
        sc_48 = fminf(sc_48, 1.7014118346046923e+38f);
        unsigned int u_49 = __as_u32(sc_48);
        unsigned int cls_50 = u_49 & 4294965248u;
        unsigned int key_51 = 0;
        if (qf < __uint_as_float(cls_50) && cls_50 < 4278190080u) {
          key_51 = 1073741824 | (unsigned int)(t0_1_1 + 12);
        }
        if (cls_50 == qc) {
          unsigned int lowb_12 = (u_49 ^ (unsigned int)((int)u_49 >> 31) & 2047) & 2047;
          key_51 = 536870912 | lowb_12 << 11 | (unsigned int)(t0_1_1 + 12);
        }
        kb2o[12] = __uint_as_float(key_51);
        float sc_52 = __uint_as_float(cb[13]);
        sc_52 = fmaxf(sc_52, -1.7014118346046923e+38f);
        sc_52 = fminf(sc_52, 1.7014118346046923e+38f);
        unsigned int u_53 = __as_u32(sc_52);
        unsigned int cls_54 = u_53 & 4294965248u;
        unsigned int key_55 = 0;
        if (qf < __uint_as_float(cls_54) && cls_54 < 4278190080u) {
          key_55 = 1073741824 | (unsigned int)(t0_1_1 + 13);
        }
        if (cls_54 == qc) {
          unsigned int lowb_13 = (u_53 ^ (unsigned int)((int)u_53 >> 31) & 2047) & 2047;
          key_55 = 536870912 | lowb_13 << 11 | (unsigned int)(t0_1_1 + 13);
        }
        kb2o[13] = __uint_as_float(key_55);
        float sc_56 = __uint_as_float(cb[14]);
        sc_56 = fmaxf(sc_56, -1.7014118346046923e+38f);
        sc_56 = fminf(sc_56, 1.7014118346046923e+38f);
        unsigned int u_57 = __as_u32(sc_56);
        unsigned int cls_58 = u_57 & 4294965248u;
        unsigned int key_59 = 0;
        if (qf < __uint_as_float(cls_58) && cls_58 < 4278190080u) {
          key_59 = 1073741824 | (unsigned int)(t0_1_1 + 14);
        }
        if (cls_58 == qc) {
          unsigned int lowb_14 = (u_57 ^ (unsigned int)((int)u_57 >> 31) & 2047) & 2047;
          key_59 = 536870912 | lowb_14 << 11 | (unsigned int)(t0_1_1 + 14);
        }
        kb2o[14] = __uint_as_float(key_59);
        float sc_60 = __uint_as_float(cb[15]);
        sc_60 = fmaxf(sc_60, -1.7014118346046923e+38f);
        sc_60 = fminf(sc_60, 1.7014118346046923e+38f);
        unsigned int u_61 = __as_u32(sc_60);
        unsigned int cls_62 = u_61 & 4294965248u;
        unsigned int key_63 = 0;
        if (qf < __uint_as_float(cls_62) && cls_62 < 4278190080u) {
          key_63 = 1073741824 | (unsigned int)(t0_1_1 + 15);
        }
        if (cls_62 == qc) {
          unsigned int lowb_15 = (u_61 ^ (unsigned int)((int)u_61 >> 31) & 2047) & 2047;
          key_63 = 536870912 | lowb_15 << 11 | (unsigned int)(t0_1_1 + 15);
        }
        kb2o[15] = __uint_as_float(key_63);
        float hi_64_1 = fmaxf(kb2o[0], kb2o[13]);
        float lo_65_1 = fminf(kb2o[0], kb2o[13]);
        kb2o[0] = hi_64_1;
        kb2o[13] = lo_65_1;
        float hi_66 = fmaxf(kb2o[1], kb2o[12]);
        float lo_67 = fminf(kb2o[1], kb2o[12]);
        kb2o[1] = hi_66;
        kb2o[12] = lo_67;
        float hi_68_1 = fmaxf(kb2o[2], kb2o[15]);
        float lo_69_1 = fminf(kb2o[2], kb2o[15]);
        kb2o[2] = hi_68_1;
        kb2o[15] = lo_69_1;
        float hi_70 = fmaxf(kb2o[3], kb2o[14]);
        float lo_71 = fminf(kb2o[3], kb2o[14]);
        kb2o[3] = hi_70;
        kb2o[14] = lo_71;
        float hi_72_1 = fmaxf(kb2o[4], kb2o[8]);
        float lo_73_1 = fminf(kb2o[4], kb2o[8]);
        kb2o[4] = hi_72_1;
        kb2o[8] = lo_73_1;
        float hi_74 = fmaxf(kb2o[5], kb2o[6]);
        float lo_75 = fminf(kb2o[5], kb2o[6]);
        kb2o[5] = hi_74;
        kb2o[6] = lo_75;
        float hi_76_1 = fmaxf(kb2o[7], kb2o[11]);
        float lo_77_1 = fminf(kb2o[7], kb2o[11]);
        kb2o[7] = hi_76_1;
        kb2o[11] = lo_77_1;
        float hi_78 = fmaxf(kb2o[9], kb2o[10]);
        float lo_79 = fminf(kb2o[9], kb2o[10]);
        kb2o[9] = hi_78;
        kb2o[10] = lo_79;
        float hi_80_1 = fmaxf(kb2o[0], kb2o[5]);
        float lo_81_1 = fminf(kb2o[0], kb2o[5]);
        kb2o[0] = hi_80_1;
        kb2o[5] = lo_81_1;
        float hi_82 = fmaxf(kb2o[1], kb2o[7]);
        float lo_83 = fminf(kb2o[1], kb2o[7]);
        kb2o[1] = hi_82;
        kb2o[7] = lo_83;
        float hi_84_1 = fmaxf(kb2o[2], kb2o[9]);
        float lo_85_1 = fminf(kb2o[2], kb2o[9]);
        kb2o[2] = hi_84_1;
        kb2o[9] = lo_85_1;
        float hi_86 = fmaxf(kb2o[3], kb2o[4]);
        float lo_87 = fminf(kb2o[3], kb2o[4]);
        kb2o[3] = hi_86;
        kb2o[4] = lo_87;
        float hi_88 = fmaxf(kb2o[6], kb2o[13]);
        float lo_89 = fminf(kb2o[6], kb2o[13]);
        kb2o[6] = hi_88;
        kb2o[13] = lo_89;
        float hi_90 = fmaxf(kb2o[8], kb2o[14]);
        float lo_91 = fminf(kb2o[8], kb2o[14]);
        kb2o[8] = hi_90;
        kb2o[14] = lo_91;
        float hi_92_1 = fmaxf(kb2o[10], kb2o[15]);
        float lo_93_1 = fminf(kb2o[10], kb2o[15]);
        kb2o[10] = hi_92_1;
        kb2o[15] = lo_93_1;
        float hi_94 = fmaxf(kb2o[11], kb2o[12]);
        float lo_95 = fminf(kb2o[11], kb2o[12]);
        kb2o[11] = hi_94;
        kb2o[12] = lo_95;
        float hi_96_1 = fmaxf(kb2o[0], kb2o[1]);
        float lo_97_1 = fminf(kb2o[0], kb2o[1]);
        kb2o[0] = hi_96_1;
        kb2o[1] = lo_97_1;
        float hi_98 = fmaxf(kb2o[2], kb2o[3]);
        float lo_99 = fminf(kb2o[2], kb2o[3]);
        kb2o[2] = hi_98;
        kb2o[3] = lo_99;
        float hi_100_1 = fmaxf(kb2o[4], kb2o[5]);
        float lo_101_1 = fminf(kb2o[4], kb2o[5]);
        kb2o[4] = hi_100_1;
        kb2o[5] = lo_101_1;
        float hi_102 = fmaxf(kb2o[6], kb2o[8]);
        float lo_103 = fminf(kb2o[6], kb2o[8]);
        kb2o[6] = hi_102;
        kb2o[8] = lo_103;
        float hi_104_1 = fmaxf(kb2o[7], kb2o[9]);
        float lo_105_1 = fminf(kb2o[7], kb2o[9]);
        kb2o[7] = hi_104_1;
        kb2o[9] = lo_105_1;
        float hi_106 = fmaxf(kb2o[10], kb2o[11]);
        float lo_107 = fminf(kb2o[10], kb2o[11]);
        kb2o[10] = hi_106;
        kb2o[11] = lo_107;
        float hi_108 = fmaxf(kb2o[12], kb2o[13]);
        float lo_109 = fminf(kb2o[12], kb2o[13]);
        kb2o[12] = hi_108;
        kb2o[13] = lo_109;
        float hi_110 = fmaxf(kb2o[14], kb2o[15]);
        float lo_111 = fminf(kb2o[14], kb2o[15]);
        kb2o[14] = hi_110;
        kb2o[15] = lo_111;
        float hi_112 = fmaxf(kb2o[0], kb2o[2]);
        float lo_113 = fminf(kb2o[0], kb2o[2]);
        kb2o[0] = hi_112;
        kb2o[2] = lo_113;
        float hi_114 = fmaxf(kb2o[1], kb2o[3]);
        float lo_115 = fminf(kb2o[1], kb2o[3]);
        kb2o[1] = hi_114;
        kb2o[3] = lo_115;
        float hi_116 = fmaxf(kb2o[4], kb2o[10]);
        float lo_117 = fminf(kb2o[4], kb2o[10]);
        kb2o[4] = hi_116;
        kb2o[10] = lo_117;
        float hi_118 = fmaxf(kb2o[5], kb2o[11]);
        float lo_119 = fminf(kb2o[5], kb2o[11]);
        kb2o[5] = hi_118;
        kb2o[11] = lo_119;
        float hi_120 = fmaxf(kb2o[6], kb2o[7]);
        float lo_121 = fminf(kb2o[6], kb2o[7]);
        kb2o[6] = hi_120;
        kb2o[7] = lo_121;
        float hi_122 = fmaxf(kb2o[8], kb2o[9]);
        float lo_123 = fminf(kb2o[8], kb2o[9]);
        kb2o[8] = hi_122;
        kb2o[9] = lo_123;
        float hi_124 = fmaxf(kb2o[12], kb2o[14]);
        float lo_125 = fminf(kb2o[12], kb2o[14]);
        kb2o[12] = hi_124;
        kb2o[14] = lo_125;
        float hi_126 = fmaxf(kb2o[13], kb2o[15]);
        float lo_127 = fminf(kb2o[13], kb2o[15]);
        kb2o[13] = hi_126;
        kb2o[15] = lo_127;
        float hi_128 = fmaxf(kb2o[1], kb2o[2]);
        float lo_129 = fminf(kb2o[1], kb2o[2]);
        kb2o[1] = hi_128;
        kb2o[2] = lo_129;
        float hi_130 = fmaxf(kb2o[3], kb2o[12]);
        float lo_131 = fminf(kb2o[3], kb2o[12]);
        kb2o[3] = hi_130;
        kb2o[12] = lo_131;
        float hi_132 = fmaxf(kb2o[4], kb2o[6]);
        float lo_133 = fminf(kb2o[4], kb2o[6]);
        kb2o[4] = hi_132;
        kb2o[6] = lo_133;
        float hi_134 = fmaxf(kb2o[5], kb2o[7]);
        float lo_135 = fminf(kb2o[5], kb2o[7]);
        kb2o[5] = hi_134;
        kb2o[7] = lo_135;
        float hi_136 = fmaxf(kb2o[8], kb2o[10]);
        float lo_137 = fminf(kb2o[8], kb2o[10]);
        kb2o[8] = hi_136;
        kb2o[10] = lo_137;
        float hi_138 = fmaxf(kb2o[9], kb2o[11]);
        float lo_139 = fminf(kb2o[9], kb2o[11]);
        kb2o[9] = hi_138;
        kb2o[11] = lo_139;
        float hi_140 = fmaxf(kb2o[13], kb2o[14]);
        float lo_141 = fminf(kb2o[13], kb2o[14]);
        kb2o[13] = hi_140;
        kb2o[14] = lo_141;
        float hi_142 = fmaxf(kb2o[1], kb2o[4]);
        float lo_143 = fminf(kb2o[1], kb2o[4]);
        kb2o[1] = hi_142;
        kb2o[4] = lo_143;
        float hi_144 = fmaxf(kb2o[2], kb2o[6]);
        float lo_145 = fminf(kb2o[2], kb2o[6]);
        kb2o[2] = hi_144;
        kb2o[6] = lo_145;
        float hi_146 = fmaxf(kb2o[5], kb2o[8]);
        float lo_147 = fminf(kb2o[5], kb2o[8]);
        kb2o[5] = hi_146;
        kb2o[8] = lo_147;
        float hi_148 = fmaxf(kb2o[7], kb2o[10]);
        float lo_149 = fminf(kb2o[7], kb2o[10]);
        kb2o[7] = hi_148;
        kb2o[10] = lo_149;
        float hi_150 = fmaxf(kb2o[9], kb2o[13]);
        float lo_151 = fminf(kb2o[9], kb2o[13]);
        kb2o[9] = hi_150;
        kb2o[13] = lo_151;
        float hi_153 = fmaxf(kb2o[11], kb2o[14]);
        float lo_154 = fminf(kb2o[11], kb2o[14]);
        kb2o[11] = hi_153;
        kb2o[14] = lo_154;
        float hi_156 = fmaxf(kb2o[2], kb2o[4]);
        float lo_157 = fminf(kb2o[2], kb2o[4]);
        kb2o[2] = hi_156;
        kb2o[4] = lo_157;
        float hi_159 = fmaxf(kb2o[3], kb2o[6]);
        float lo_160 = fminf(kb2o[3], kb2o[6]);
        kb2o[3] = hi_159;
        kb2o[6] = lo_160;
        float hi_161 = fmaxf(kb2o[9], kb2o[12]);
        float lo_162 = fminf(kb2o[9], kb2o[12]);
        kb2o[9] = hi_161;
        kb2o[12] = lo_162;
        float hi_163 = fmaxf(kb2o[11], kb2o[13]);
        float lo_164 = fminf(kb2o[11], kb2o[13]);
        kb2o[11] = hi_163;
        kb2o[13] = lo_164;
        float hi_165 = fmaxf(kb2o[3], kb2o[5]);
        float lo_166 = fminf(kb2o[3], kb2o[5]);
        kb2o[3] = hi_165;
        kb2o[5] = lo_166;
        float hi_167 = fmaxf(kb2o[6], kb2o[8]);
        float lo_168 = fminf(kb2o[6], kb2o[8]);
        kb2o[6] = hi_167;
        kb2o[8] = lo_168;
        float hi_169 = fmaxf(kb2o[7], kb2o[9]);
        float lo_170 = fminf(kb2o[7], kb2o[9]);
        kb2o[7] = hi_169;
        kb2o[9] = lo_170;
        float hi_172 = fmaxf(kb2o[10], kb2o[12]);
        float lo_173 = fminf(kb2o[10], kb2o[12]);
        kb2o[10] = hi_172;
        kb2o[12] = lo_173;
        float hi_175 = fmaxf(kb2o[3], kb2o[4]);
        float lo_176 = fminf(kb2o[3], kb2o[4]);
        kb2o[3] = hi_175;
        kb2o[4] = lo_176;
        float hi_178 = fmaxf(kb2o[5], kb2o[6]);
        float lo_179 = fminf(kb2o[5], kb2o[6]);
        kb2o[5] = hi_178;
        kb2o[6] = lo_179;
        float hi_180 = fmaxf(kb2o[7], kb2o[8]);
        float lo_181 = fminf(kb2o[7], kb2o[8]);
        kb2o[7] = hi_180;
        kb2o[8] = lo_181;
        float hi_182 = fmaxf(kb2o[9], kb2o[10]);
        float lo_183 = fminf(kb2o[9], kb2o[10]);
        kb2o[9] = hi_182;
        kb2o[10] = lo_183;
        float hi_184 = fmaxf(kb2o[11], kb2o[12]);
        float lo_185 = fminf(kb2o[11], kb2o[12]);
        kb2o[11] = hi_184;
        kb2o[12] = lo_185;
        float hi_186 = fmaxf(kb2o[6], kb2o[7]);
        float lo_187 = fminf(kb2o[6], kb2o[7]);
        kb2o[6] = hi_186;
        kb2o[7] = lo_187;
        float hi_188 = fmaxf(kb2o[8], kb2o[9]);
        float lo_189 = fminf(kb2o[8], kb2o[9]);
        kb2o[8] = hi_188;
        kb2o[9] = lo_189;
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
      float pv_10_1 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 8);
      float hi_11_1 = fmaxf(cur_9, pv_10_1);
      float lo_12_1 = fminf(cur_9, pv_10_1);
      cur_9 = ((up[0] != 0) ? hi_11_1 : lo_12_1);
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
      float hi_34 = fmaxf(cur_28, pv_32);
      float lo_35 = fminf(cur_28, pv_32);
      cur_28 = ((up[1] != 0) ? hi_34 : lo_35);
      float pv_36 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 2);
      float hi_38 = fmaxf(cur_28, pv_36);
      float lo_39 = fminf(cur_28, pv_36);
      cur_28 = ((up[2] != 0) ? hi_38 : lo_39);
      float pv_40 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 1);
      float hi_42 = fmaxf(cur_28, pv_40);
      float lo_43 = fminf(cur_28, pv_40);
      cur_28 = ((up[3] != 0) ? hi_42 : lo_43);
      V_2[1] = cur_28;
      int s0_44 = ((sg * 16 + 2) * 2 + cg) * 17;
      int s1_45 = ((sg * 16 + 2 + 8) * 2 + cg) * 17;
      float x0_46 = pub[s0_44 + ln];
      float y0_47 = pub[s1_45 + lnr];
      float lo0_48 = fminf(x0_46, y0_47);
      r_1 = fmaxf(r_1, lo0_48);
      float hi0_49 = fmaxf(x0_46, y0_47);
      float cur_50 = hi0_49;
      float pv_51 = __shfl_xor_sync(0xFFFFFFFF, cur_50, 8);
      float hi_52 = fmaxf(cur_50, pv_51);
      float lo_53 = fminf(cur_50, pv_51);
      cur_50 = ((up[0] != 0) ? hi_52 : lo_53);
      float pv_54 = __shfl_xor_sync(0xFFFFFFFF, cur_50, 4);
      float hi_56_1 = fmaxf(cur_50, pv_54);
      float lo_57_1 = fminf(cur_50, pv_54);
      cur_50 = ((up[1] != 0) ? hi_56_1 : lo_57_1);
      float pv_58_1 = __shfl_xor_sync(0xFFFFFFFF, cur_50, 2);
      float hi_60_1 = fmaxf(cur_50, pv_58_1);
      float lo_61_1 = fminf(cur_50, pv_58_1);
      cur_50 = ((up[2] != 0) ? hi_60_1 : lo_61_1);
      float pv_62_1 = __shfl_xor_sync(0xFFFFFFFF, cur_50, 1);
      float hi_64_2 = fmaxf(cur_50, pv_62_1);
      float lo_65_2 = fminf(cur_50, pv_62_1);
      cur_50 = ((up[3] != 0) ? hi_64_2 : lo_65_2);
      V_2[2] = cur_50;
      int s0_66 = ((sg * 16 + 3) * 2 + cg) * 17;
      int s1_67 = ((sg * 16 + 3 + 8) * 2 + cg) * 17;
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
      float hi_78_1 = fmaxf(cur_72, pv_76);
      float lo_79_1 = fminf(cur_72, pv_76);
      cur_72 = ((up[1] != 0) ? hi_78_1 : lo_79_1);
      float pv_80 = __shfl_xor_sync(0xFFFFFFFF, cur_72, 2);
      float hi_82_1 = fmaxf(cur_72, pv_80);
      float lo_83_1 = fminf(cur_72, pv_80);
      cur_72 = ((up[2] != 0) ? hi_82_1 : lo_83_1);
      float pv_84 = __shfl_xor_sync(0xFFFFFFFF, cur_72, 1);
      float hi_86_1 = fmaxf(cur_72, pv_84);
      float lo_87_1 = fminf(cur_72, pv_84);
      cur_72 = ((up[3] != 0) ? hi_86_1 : lo_87_1);
      V_2[3] = cur_72;
      int s0_88 = ((sg * 16 + 4) * 2 + cg) * 17;
      int s1_89 = ((sg * 16 + 4 + 8) * 2 + cg) * 17;
      float x0_90 = pub[s0_88 + ln];
      float y0_91 = pub[s1_89 + lnr];
      float lo0_92 = fminf(x0_90, y0_91);
      r_1 = fmaxf(r_1, lo0_92);
      float hi0_93 = fmaxf(x0_90, y0_91);
      float cur_94 = hi0_93;
      float pv_95 = __shfl_xor_sync(0xFFFFFFFF, cur_94, 8);
      float hi_96_2 = fmaxf(cur_94, pv_95);
      float lo_97_2 = fminf(cur_94, pv_95);
      cur_94 = ((up[0] != 0) ? hi_96_2 : lo_97_2);
      float pv_98_1 = __shfl_xor_sync(0xFFFFFFFF, cur_94, 4);
      float hi_100_2 = fmaxf(cur_94, pv_98_1);
      float lo_101_2 = fminf(cur_94, pv_98_1);
      cur_94 = ((up[1] != 0) ? hi_100_2 : lo_101_2);
      float pv_102_1 = __shfl_xor_sync(0xFFFFFFFF, cur_94, 2);
      float hi_104_2 = fmaxf(cur_94, pv_102_1);
      float lo_105_2 = fminf(cur_94, pv_102_1);
      cur_94 = ((up[2] != 0) ? hi_104_2 : lo_105_2);
      float pv_106 = __shfl_xor_sync(0xFFFFFFFF, cur_94, 1);
      float hi_108_1 = fmaxf(cur_94, pv_106);
      float lo_109_1 = fminf(cur_94, pv_106);
      cur_94 = ((up[3] != 0) ? hi_108_1 : lo_109_1);
      V_2[4] = cur_94;
      int s0_110 = ((sg * 16 + 5) * 2 + cg) * 17;
      int s1_111 = ((sg * 16 + 5 + 8) * 2 + cg) * 17;
      float x0_112 = pub[s0_110 + ln];
      float y0_113 = pub[s1_111 + lnr];
      float lo0_114 = fminf(x0_112, y0_113);
      r_1 = fmaxf(r_1, lo0_114);
      float hi0_115 = fmaxf(x0_112, y0_113);
      float cur_116 = hi0_115;
      float pv_117 = __shfl_xor_sync(0xFFFFFFFF, cur_116, 8);
      float hi_118_1 = fmaxf(cur_116, pv_117);
      float lo_119_1 = fminf(cur_116, pv_117);
      cur_116 = ((up[0] != 0) ? hi_118_1 : lo_119_1);
      float pv_120 = __shfl_xor_sync(0xFFFFFFFF, cur_116, 4);
      float hi_122_1 = fmaxf(cur_116, pv_120);
      float lo_123_1 = fminf(cur_116, pv_120);
      cur_116 = ((up[1] != 0) ? hi_122_1 : lo_123_1);
      float pv_124 = __shfl_xor_sync(0xFFFFFFFF, cur_116, 2);
      float hi_126_1 = fmaxf(cur_116, pv_124);
      float lo_127_1 = fminf(cur_116, pv_124);
      cur_116 = ((up[2] != 0) ? hi_126_1 : lo_127_1);
      float pv_128 = __shfl_xor_sync(0xFFFFFFFF, cur_116, 1);
      float hi_130_1 = fmaxf(cur_116, pv_128);
      float lo_131_1 = fminf(cur_116, pv_128);
      cur_116 = ((up[3] != 0) ? hi_130_1 : lo_131_1);
      V_2[5] = cur_116;
      int s0_132 = ((sg * 16 + 6) * 2 + cg) * 17;
      int s1_133 = ((sg * 16 + 6 + 8) * 2 + cg) * 17;
      float x0_134 = pub[s0_132 + ln];
      float y0_135 = pub[s1_133 + lnr];
      float lo0_136 = fminf(x0_134, y0_135);
      r_1 = fmaxf(r_1, lo0_136);
      float hi0_137 = fmaxf(x0_134, y0_135);
      float cur_138 = hi0_137;
      float pv_139 = __shfl_xor_sync(0xFFFFFFFF, cur_138, 8);
      float hi_140_1 = fmaxf(cur_138, pv_139);
      float lo_141_1 = fminf(cur_138, pv_139);
      cur_138 = ((up[0] != 0) ? hi_140_1 : lo_141_1);
      float pv_142 = __shfl_xor_sync(0xFFFFFFFF, cur_138, 4);
      float hi_144_1 = fmaxf(cur_138, pv_142);
      float lo_145_1 = fminf(cur_138, pv_142);
      cur_138 = ((up[1] != 0) ? hi_144_1 : lo_145_1);
      float pv_146 = __shfl_xor_sync(0xFFFFFFFF, cur_138, 2);
      float hi_148_1 = fmaxf(cur_138, pv_146);
      float lo_149_1 = fminf(cur_138, pv_146);
      cur_138 = ((up[2] != 0) ? hi_148_1 : lo_149_1);
      float pv_150 = __shfl_xor_sync(0xFFFFFFFF, cur_138, 1);
      float hi_151 = fmaxf(cur_138, pv_150);
      float lo_152 = fminf(cur_138, pv_150);
      cur_138 = ((up[3] != 0) ? hi_151 : lo_152);
      V_2[6] = cur_138;
      int s0_153 = ((sg * 16 + 7) * 2 + cg) * 17;
      int s1_154 = ((sg * 16 + 7 + 8) * 2 + cg) * 17;
      float x0_155 = pub[s0_153 + ln];
      float y0_156 = pub[s1_154 + lnr];
      float lo0_157 = fminf(x0_155, y0_156);
      r_1 = fmaxf(r_1, lo0_157);
      float hi0_158 = fmaxf(x0_155, y0_156);
      float cur_159 = hi0_158;
      float pv_160 = __shfl_xor_sync(0xFFFFFFFF, cur_159, 8);
      float hi_161_1 = fmaxf(cur_159, pv_160);
      float lo_162_1 = fminf(cur_159, pv_160);
      cur_159 = ((up[0] != 0) ? hi_161_1 : lo_162_1);
      float pv_163 = __shfl_xor_sync(0xFFFFFFFF, cur_159, 4);
      float hi_164 = fmaxf(cur_159, pv_163);
      float lo_165 = fminf(cur_159, pv_163);
      cur_159 = ((up[1] != 0) ? hi_164 : lo_165);
      float pv_166 = __shfl_xor_sync(0xFFFFFFFF, cur_159, 2);
      float hi_167_1 = fmaxf(cur_159, pv_166);
      float lo_168_1 = fminf(cur_159, pv_166);
      cur_159 = ((up[2] != 0) ? hi_167_1 : lo_168_1);
      float pv_169 = __shfl_xor_sync(0xFFFFFFFF, cur_159, 1);
      float hi_170 = fmaxf(cur_159, pv_169);
      float lo_171 = fminf(cur_159, pv_169);
      cur_159 = ((up[3] != 0) ? hi_170 : lo_171);
      V_2[7] = cur_159;
      float rs_172 = pub[((sg * 16 + ln) * 2 + cg) * 17 + 16];
      r_1 = fmaxf(r_1, rs_172);
      float y1_173 = __shfl_xor_sync(0xFFFFFFFF, V_2[4], 15);
      float lo1_174 = fminf(V_2[0], y1_173);
      r_1 = fmaxf(r_1, lo1_174);
      float hi1_175 = fmaxf(V_2[0], y1_173);
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
      float pv_187 = __shfl_xor_sync(0xFFFFFFFF, cur_176, 1);
      float hi_188_1 = fmaxf(cur_176, pv_187);
      float lo_189_1 = fminf(cur_176, pv_187);
      cur_176 = ((up[3] != 0) ? hi_188_1 : lo_189_1);
      V_2[0] = cur_176;
      float y1_190 = __shfl_xor_sync(0xFFFFFFFF, V_2[5], 15);
      float lo1_191 = fminf(V_2[1], y1_190);
      r_1 = fmaxf(r_1, lo1_191);
      float hi1_192 = fmaxf(V_2[1], y1_190);
      float cur_193 = hi1_192;
      float pv_194 = __shfl_xor_sync(0xFFFFFFFF, cur_193, 8);
      float hi_195 = fmaxf(cur_193, pv_194);
      float lo_196 = fminf(cur_193, pv_194);
      cur_193 = ((up[0] != 0) ? hi_195 : lo_196);
      float pv_197 = __shfl_xor_sync(0xFFFFFFFF, cur_193, 4);
      float hi_198 = fmaxf(cur_193, pv_197);
      float lo_199 = fminf(cur_193, pv_197);
      cur_193 = ((up[1] != 0) ? hi_198 : lo_199);
      float pv_200 = __shfl_xor_sync(0xFFFFFFFF, cur_193, 2);
      float hi_201 = fmaxf(cur_193, pv_200);
      float lo_202 = fminf(cur_193, pv_200);
      cur_193 = ((up[2] != 0) ? hi_201 : lo_202);
      float pv_203 = __shfl_xor_sync(0xFFFFFFFF, cur_193, 1);
      float hi_204 = fmaxf(cur_193, pv_203);
      float lo_205 = fminf(cur_193, pv_203);
      cur_193 = ((up[3] != 0) ? hi_204 : lo_205);
      V_2[1] = cur_193;
      float y1_206 = __shfl_xor_sync(0xFFFFFFFF, V_2[6], 15);
      float lo1_207 = fminf(V_2[2], y1_206);
      r_1 = fmaxf(r_1, lo1_207);
      float hi1_208 = fmaxf(V_2[2], y1_206);
      float cur_209 = hi1_208;
      float pv_210 = __shfl_xor_sync(0xFFFFFFFF, cur_209, 8);
      float hi_211 = fmaxf(cur_209, pv_210);
      float lo_212 = fminf(cur_209, pv_210);
      cur_209 = ((up[0] != 0) ? hi_211 : lo_212);
      float pv_213 = __shfl_xor_sync(0xFFFFFFFF, cur_209, 4);
      float hi_214 = fmaxf(cur_209, pv_213);
      float lo_215 = fminf(cur_209, pv_213);
      cur_209 = ((up[1] != 0) ? hi_214 : lo_215);
      float pv_216 = __shfl_xor_sync(0xFFFFFFFF, cur_209, 2);
      float hi_217 = fmaxf(cur_209, pv_216);
      float lo_218 = fminf(cur_209, pv_216);
      cur_209 = ((up[2] != 0) ? hi_217 : lo_218);
      float pv_219 = __shfl_xor_sync(0xFFFFFFFF, cur_209, 1);
      float hi_220 = fmaxf(cur_209, pv_219);
      float lo_221 = fminf(cur_209, pv_219);
      cur_209 = ((up[3] != 0) ? hi_220 : lo_221);
      V_2[2] = cur_209;
      float y1_222 = __shfl_xor_sync(0xFFFFFFFF, V_2[7], 15);
      float lo1_223 = fminf(V_2[3], y1_222);
      r_1 = fmaxf(r_1, lo1_223);
      float hi1_224 = fmaxf(V_2[3], y1_222);
      float cur_225 = hi1_224;
      float pv_226 = __shfl_xor_sync(0xFFFFFFFF, cur_225, 8);
      float hi_227 = fmaxf(cur_225, pv_226);
      float lo_228 = fminf(cur_225, pv_226);
      cur_225 = ((up[0] != 0) ? hi_227 : lo_228);
      float pv_229 = __shfl_xor_sync(0xFFFFFFFF, cur_225, 4);
      float hi_230 = fmaxf(cur_225, pv_229);
      float lo_231 = fminf(cur_225, pv_229);
      cur_225 = ((up[1] != 0) ? hi_230 : lo_231);
      float pv_232 = __shfl_xor_sync(0xFFFFFFFF, cur_225, 2);
      float hi_233 = fmaxf(cur_225, pv_232);
      float lo_234 = fminf(cur_225, pv_232);
      cur_225 = ((up[2] != 0) ? hi_233 : lo_234);
      float pv_235 = __shfl_xor_sync(0xFFFFFFFF, cur_225, 1);
      float hi_236 = fmaxf(cur_225, pv_235);
      float lo_237 = fminf(cur_225, pv_235);
      cur_225 = ((up[3] != 0) ? hi_236 : lo_237);
      V_2[3] = cur_225;
      float y1_238 = __shfl_xor_sync(0xFFFFFFFF, V_2[2], 15);
      float lo1_239 = fminf(V_2[0], y1_238);
      r_1 = fmaxf(r_1, lo1_239);
      float hi1_240 = fmaxf(V_2[0], y1_238);
      float cur_241 = hi1_240;
      float pv_242 = __shfl_xor_sync(0xFFFFFFFF, cur_241, 8);
      float hi_243 = fmaxf(cur_241, pv_242);
      float lo_244 = fminf(cur_241, pv_242);
      cur_241 = ((up[0] != 0) ? hi_243 : lo_244);
      float pv_245 = __shfl_xor_sync(0xFFFFFFFF, cur_241, 4);
      float hi_246 = fmaxf(cur_241, pv_245);
      float lo_247 = fminf(cur_241, pv_245);
      cur_241 = ((up[1] != 0) ? hi_246 : lo_247);
      float pv_248 = __shfl_xor_sync(0xFFFFFFFF, cur_241, 2);
      float hi_249 = fmaxf(cur_241, pv_248);
      float lo_250 = fminf(cur_241, pv_248);
      cur_241 = ((up[2] != 0) ? hi_249 : lo_250);
      float pv_251 = __shfl_xor_sync(0xFFFFFFFF, cur_241, 1);
      float hi_252 = fmaxf(cur_241, pv_251);
      float lo_253 = fminf(cur_241, pv_251);
      cur_241 = ((up[3] != 0) ? hi_252 : lo_253);
      V_2[0] = cur_241;
      float y1_254 = __shfl_xor_sync(0xFFFFFFFF, V_2[3], 15);
      float lo1_255 = fminf(V_2[1], y1_254);
      r_1 = fmaxf(r_1, lo1_255);
      float hi1_256 = fmaxf(V_2[1], y1_254);
      float cur_257 = hi1_256;
      float pv_258 = __shfl_xor_sync(0xFFFFFFFF, cur_257, 8);
      float hi_259 = fmaxf(cur_257, pv_258);
      float lo_260 = fminf(cur_257, pv_258);
      cur_257 = ((up[0] != 0) ? hi_259 : lo_260);
      float pv_261 = __shfl_xor_sync(0xFFFFFFFF, cur_257, 4);
      float hi_262 = fmaxf(cur_257, pv_261);
      float lo_263 = fminf(cur_257, pv_261);
      cur_257 = ((up[1] != 0) ? hi_262 : lo_263);
      float pv_264 = __shfl_xor_sync(0xFFFFFFFF, cur_257, 2);
      float hi_265 = fmaxf(cur_257, pv_264);
      float lo_266 = fminf(cur_257, pv_264);
      cur_257 = ((up[2] != 0) ? hi_265 : lo_266);
      float pv_267 = __shfl_xor_sync(0xFFFFFFFF, cur_257, 1);
      float hi_268 = fmaxf(cur_257, pv_267);
      float lo_269 = fminf(cur_257, pv_267);
      cur_257 = ((up[3] != 0) ? hi_268 : lo_269);
      V_2[1] = cur_257;
      float yl_270 = __shfl_xor_sync(0xFFFFFFFF, V_2[1], 15);
      float lol_271 = fminf(V_2[0], yl_270);
      r_1 = fmaxf(r_1, lol_271);
      float hil_272 = fmaxf(V_2[0], yl_270);
      float cur_273 = hil_272;
      float pv_274 = __shfl_xor_sync(0xFFFFFFFF, cur_273, 8);
      float hi_275 = fmaxf(cur_273, pv_274);
      float lo_276 = fminf(cur_273, pv_274);
      cur_273 = ((up[0] != 0) ? hi_275 : lo_276);
      float pv_277 = __shfl_xor_sync(0xFFFFFFFF, cur_273, 4);
      float hi_278 = fmaxf(cur_273, pv_277);
      float lo_279 = fminf(cur_273, pv_277);
      cur_273 = ((up[1] != 0) ? hi_278 : lo_279);
      float pv_280 = __shfl_xor_sync(0xFFFFFFFF, cur_273, 2);
      float hi_281 = fmaxf(cur_273, pv_280);
      float lo_282 = fminf(cur_273, pv_280);
      cur_273 = ((up[2] != 0) ? hi_281 : lo_282);
      float pv_283 = __shfl_xor_sync(0xFFFFFFFF, cur_273, 1);
      float hi_284 = fmaxf(cur_273, pv_283);
      float lo_285 = fminf(cur_273, pv_283);
      cur_273 = ((up[3] != 0) ? hi_284 : lo_285);
      V_2[0] = cur_273;
      float K_286 = V_2[0];
      int qb_287 = (sg * 2 + cg) * 32;
      q2[qb_287 + ln] = K_286;
      q2[qb_287 + 16 + ln] = r_1;
      asm volatile("barrier.sync 1, 256;" ::: "memory");
      float k2o_288[1];
      if (tid_1 < 32) {
        float r2v_1 = neg_inf;
        r2v_1 = fmaxf(r2v_1, q2[cg * 32 + 16 + ln]);
        r2v_1 = fmaxf(r2v_1, q2[(2 + cg) * 32 + 16 + ln]);
        r2v_1 = fmaxf(r2v_1, q2[(4 + cg) * 32 + 16 + ln]);
        r2v_1 = fmaxf(r2v_1, q2[(6 + cg) * 32 + 16 + ln]);
        r2v_1 = fmaxf(r2v_1, q2[(8 + cg) * 32 + 16 + ln]);
        r2v_1 = fmaxf(r2v_1, q2[(10 + cg) * 32 + 16 + ln]);
        r2v_1 = fmaxf(r2v_1, q2[(12 + cg) * 32 + 16 + ln]);
        r2v_1 = fmaxf(r2v_1, q2[(14 + cg) * 32 + 16 + ln]);
        float V2_1[4];
        float x2_1 = q2[cg * 32 + ln];
        float y2_1 = q2[(8 + cg) * 32 + lnr];
        float lo2_1 = fminf(x2_1, y2_1);
        r2v_1 = fmaxf(r2v_1, lo2_1);
        float hi2_1 = fmaxf(x2_1, y2_1);
        float cur_0_1 = hi2_1;
        float pv_1_1 = __shfl_xor_sync(0xFFFFFFFF, cur_0_1, 8);
        float hi_2_1 = fmaxf(cur_0_1, pv_1_1);
        float lo_3_1 = fminf(cur_0_1, pv_1_1);
        cur_0_1 = ((up[0] != 0) ? hi_2_1 : lo_3_1);
        float pv_4_1 = __shfl_xor_sync(0xFFFFFFFF, cur_0_1, 4);
        float hi_5_1 = fmaxf(cur_0_1, pv_4_1);
        float lo_6_1 = fminf(cur_0_1, pv_4_1);
        cur_0_1 = ((up[1] != 0) ? hi_5_1 : lo_6_1);
        float pv_7_1 = __shfl_xor_sync(0xFFFFFFFF, cur_0_1, 2);
        float hi_8_1 = fmaxf(cur_0_1, pv_7_1);
        float lo_9_1 = fminf(cur_0_1, pv_7_1);
        cur_0_1 = ((up[2] != 0) ? hi_8_1 : lo_9_1);
        float pv_11 = __shfl_xor_sync(0xFFFFFFFF, cur_0_1, 1);
        float hi_12 = fmaxf(cur_0_1, pv_11);
        float lo_13 = fminf(cur_0_1, pv_11);
        cur_0_1 = ((up[3] != 0) ? hi_12 : lo_13);
        V2_1[0] = cur_0_1;
        float x2_14 = q2[(2 + cg) * 32 + ln];
        float y2_15 = q2[(10 + cg) * 32 + lnr];
        float lo2_16 = fminf(x2_14, y2_15);
        r2v_1 = fmaxf(r2v_1, lo2_16);
        float hi2_17 = fmaxf(x2_14, y2_15);
        float cur_18 = hi2_17;
        float pv_20 = __shfl_xor_sync(0xFFFFFFFF, cur_18, 8);
        float hi_21 = fmaxf(cur_18, pv_20);
        float lo_22 = fminf(cur_18, pv_20);
        cur_18 = ((up[0] != 0) ? hi_21 : lo_22);
        float pv_23 = __shfl_xor_sync(0xFFFFFFFF, cur_18, 4);
        float hi_24 = fmaxf(cur_18, pv_23);
        float lo_25 = fminf(cur_18, pv_23);
        cur_18 = ((up[1] != 0) ? hi_24 : lo_25);
        float pv_26 = __shfl_xor_sync(0xFFFFFFFF, cur_18, 2);
        float hi_27 = fmaxf(cur_18, pv_26);
        float lo_28 = fminf(cur_18, pv_26);
        cur_18 = ((up[2] != 0) ? hi_27 : lo_28);
        float pv_30 = __shfl_xor_sync(0xFFFFFFFF, cur_18, 1);
        float hi_32 = fmaxf(cur_18, pv_30);
        float lo_33 = fminf(cur_18, pv_30);
        cur_18 = ((up[3] != 0) ? hi_32 : lo_33);
        V2_1[1] = cur_18;
        float x2_34 = q2[(4 + cg) * 32 + ln];
        float y2_35 = q2[(12 + cg) * 32 + lnr];
        float lo2_36 = fminf(x2_34, y2_35);
        r2v_1 = fmaxf(r2v_1, lo2_36);
        float hi2_37 = fmaxf(x2_34, y2_35);
        float cur_38 = hi2_37;
        float pv_39 = __shfl_xor_sync(0xFFFFFFFF, cur_38, 8);
        float hi_40_1 = fmaxf(cur_38, pv_39);
        float lo_41_1 = fminf(cur_38, pv_39);
        cur_38 = ((up[0] != 0) ? hi_40_1 : lo_41_1);
        float pv_42_1 = __shfl_xor_sync(0xFFFFFFFF, cur_38, 4);
        float hi_44_1 = fmaxf(cur_38, pv_42_1);
        float lo_45_1 = fminf(cur_38, pv_42_1);
        cur_38 = ((up[1] != 0) ? hi_44_1 : lo_45_1);
        float pv_46_1 = __shfl_xor_sync(0xFFFFFFFF, cur_38, 2);
        float hi_48_1 = fmaxf(cur_38, pv_46_1);
        float lo_49_1 = fminf(cur_38, pv_46_1);
        cur_38 = ((up[2] != 0) ? hi_48_1 : lo_49_1);
        float pv_50 = __shfl_xor_sync(0xFFFFFFFF, cur_38, 1);
        float hi_54 = fmaxf(cur_38, pv_50);
        float lo_55 = fminf(cur_38, pv_50);
        cur_38 = ((up[3] != 0) ? hi_54 : lo_55);
        V2_1[2] = cur_38;
        float x2_56 = q2[(6 + cg) * 32 + ln];
        float y2_57 = q2[(14 + cg) * 32 + lnr];
        float lo2_58 = fminf(x2_56, y2_57);
        r2v_1 = fmaxf(r2v_1, lo2_58);
        float hi2_59 = fmaxf(x2_56, y2_57);
        float cur_60 = hi2_59;
        float pv_61 = __shfl_xor_sync(0xFFFFFFFF, cur_60, 8);
        float hi_62 = fmaxf(cur_60, pv_61);
        float lo_63 = fminf(cur_60, pv_61);
        cur_60 = ((up[0] != 0) ? hi_62 : lo_63);
        float pv_64 = __shfl_xor_sync(0xFFFFFFFF, cur_60, 4);
        float hi_66_1 = fmaxf(cur_60, pv_64);
        float lo_67_1 = fminf(cur_60, pv_64);
        cur_60 = ((up[1] != 0) ? hi_66_1 : lo_67_1);
        float pv_68 = __shfl_xor_sync(0xFFFFFFFF, cur_60, 2);
        float hi_70_1 = fmaxf(cur_60, pv_68);
        float lo_71_1 = fminf(cur_60, pv_68);
        cur_60 = ((up[2] != 0) ? hi_70_1 : lo_71_1);
        float pv_72 = __shfl_xor_sync(0xFFFFFFFF, cur_60, 1);
        float hi_76_2 = fmaxf(cur_60, pv_72);
        float lo_77_2 = fminf(cur_60, pv_72);
        cur_60 = ((up[3] != 0) ? hi_76_2 : lo_77_2);
        V2_1[3] = cur_60;
        float y3_1 = __shfl_xor_sync(0xFFFFFFFF, V2_1[2], 15);
        float lo3_1 = fminf(V2_1[0], y3_1);
        r2v_1 = fmaxf(r2v_1, lo3_1);
        float hi3_1 = fmaxf(V2_1[0], y3_1);
        float cur_78 = hi3_1;
        float pv_79 = __shfl_xor_sync(0xFFFFFFFF, cur_78, 8);
        float hi_80_2 = fmaxf(cur_78, pv_79);
        float lo_81_2 = fminf(cur_78, pv_79);
        cur_78 = ((up[0] != 0) ? hi_80_2 : lo_81_2);
        float pv_82_1 = __shfl_xor_sync(0xFFFFFFFF, cur_78, 4);
        float hi_84_2 = fmaxf(cur_78, pv_82_1);
        float lo_85_2 = fminf(cur_78, pv_82_1);
        cur_78 = ((up[1] != 0) ? hi_84_2 : lo_85_2);
        float pv_86 = __shfl_xor_sync(0xFFFFFFFF, cur_78, 2);
        float hi_88_1 = fmaxf(cur_78, pv_86);
        float lo_89_1 = fminf(cur_78, pv_86);
        cur_78 = ((up[2] != 0) ? hi_88_1 : lo_89_1);
        float pv_90_1 = __shfl_xor_sync(0xFFFFFFFF, cur_78, 1);
        float hi_92_2 = fmaxf(cur_78, pv_90_1);
        float lo_93_2 = fminf(cur_78, pv_90_1);
        cur_78 = ((up[3] != 0) ? hi_92_2 : lo_93_2);
        V2_1[0] = cur_78;
        float y3_94 = __shfl_xor_sync(0xFFFFFFFF, V2_1[3], 15);
        float lo3_95 = fminf(V2_1[1], y3_94);
        r2v_1 = fmaxf(r2v_1, lo3_95);
        float hi3_96 = fmaxf(V2_1[1], y3_94);
        float cur_97 = hi3_96;
        float pv_99 = __shfl_xor_sync(0xFFFFFFFF, cur_97, 8);
        float hi_102_1 = fmaxf(cur_97, pv_99);
        float lo_103_1 = fminf(cur_97, pv_99);
        cur_97 = ((up[0] != 0) ? hi_102_1 : lo_103_1);
        float pv_104 = __shfl_xor_sync(0xFFFFFFFF, cur_97, 4);
        float hi_106_1 = fmaxf(cur_97, pv_104);
        float lo_107_1 = fminf(cur_97, pv_104);
        cur_97 = ((up[1] != 0) ? hi_106_1 : lo_107_1);
        float pv_108 = __shfl_xor_sync(0xFFFFFFFF, cur_97, 2);
        float hi_110_1 = fmaxf(cur_97, pv_108);
        float lo_111_1 = fminf(cur_97, pv_108);
        cur_97 = ((up[2] != 0) ? hi_110_1 : lo_111_1);
        float pv_112 = __shfl_xor_sync(0xFFFFFFFF, cur_97, 1);
        float hi_114_1 = fmaxf(cur_97, pv_112);
        float lo_115_1 = fminf(cur_97, pv_112);
        cur_97 = ((up[3] != 0) ? hi_114_1 : lo_115_1);
        V2_1[1] = cur_97;
        float yl2_1 = __shfl_xor_sync(0xFFFFFFFF, V2_1[1], 15);
        float lol2_1 = fminf(V2_1[0], yl2_1);
        r2v_1 = fmaxf(r2v_1, lol2_1);
        V2_1[0] = fmaxf(V2_1[0], yl2_1);
        k2o_288[0] = V2_1[0];
        K_286 = k2o_288[0];
        r_1 = r2v_1;
      }
      rr2x[0] = r_1;
      K2 = K_286;
    }
    if (tid_1 < 32) {
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
            float hi_2_2 = fmaxf(xsl[1], xsl[12]);
            float lo_3_2 = fminf(xsl[1], xsl[12]);
            xsl[1] = hi_2_2;
            xsl[12] = lo_3_2;
            float hi_4 = fmaxf(xsl[2], xsl[15]);
            float lo_5 = fminf(xsl[2], xsl[15]);
            xsl[2] = hi_4;
            xsl[15] = lo_5;
            float hi_6 = fmaxf(xsl[3], xsl[14]);
            float lo_7 = fminf(xsl[3], xsl[14]);
            xsl[3] = hi_6;
            xsl[14] = lo_7;
            float hi_8_2 = fmaxf(xsl[4], xsl[8]);
            float lo_9_2 = fminf(xsl[4], xsl[8]);
            xsl[4] = hi_8_2;
            xsl[8] = lo_9_2;
            float hi_10 = fmaxf(xsl[5], xsl[6]);
            float lo_11 = fminf(xsl[5], xsl[6]);
            xsl[5] = hi_10;
            xsl[6] = lo_11;
            float hi_12_1 = fmaxf(xsl[7], xsl[11]);
            float lo_13_1 = fminf(xsl[7], xsl[11]);
            xsl[7] = hi_12_1;
            xsl[11] = lo_13_1;
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
            float hi_22_1 = fmaxf(xsl[3], xsl[4]);
            float lo_23_1 = fminf(xsl[3], xsl[4]);
            xsl[3] = hi_22_1;
            xsl[4] = lo_23_1;
            float hi_24_1 = fmaxf(xsl[6], xsl[13]);
            float lo_25_1 = fminf(xsl[6], xsl[13]);
            xsl[6] = hi_24_1;
            xsl[13] = lo_25_1;
            float hi_26 = fmaxf(xsl[8], xsl[14]);
            float lo_27 = fminf(xsl[8], xsl[14]);
            xsl[8] = hi_26;
            xsl[14] = lo_27;
            float hi_28_1 = fmaxf(xsl[10], xsl[15]);
            float lo_29_1 = fminf(xsl[10], xsl[15]);
            xsl[10] = hi_28_1;
            xsl[15] = lo_29_1;
            float hi_30_1 = fmaxf(xsl[11], xsl[12]);
            float lo_31_1 = fminf(xsl[11], xsl[12]);
            xsl[11] = hi_30_1;
            xsl[12] = lo_31_1;
            float hi_32_1 = fmaxf(xsl[0], xsl[1]);
            float lo_33_1 = fminf(xsl[0], xsl[1]);
            xsl[0] = hi_32_1;
            xsl[1] = lo_33_1;
            float hi_34_1 = fmaxf(xsl[2], xsl[3]);
            float lo_35_1 = fminf(xsl[2], xsl[3]);
            xsl[2] = hi_34_1;
            xsl[3] = lo_35_1;
            float hi_36_1 = fmaxf(xsl[4], xsl[5]);
            float lo_37_1 = fminf(xsl[4], xsl[5]);
            xsl[4] = hi_36_1;
            xsl[5] = lo_37_1;
            float hi_38_1 = fmaxf(xsl[6], xsl[8]);
            float lo_39_1 = fminf(xsl[6], xsl[8]);
            xsl[6] = hi_38_1;
            xsl[8] = lo_39_1;
            float hi_40_2 = fmaxf(xsl[7], xsl[9]);
            float lo_41_2 = fminf(xsl[7], xsl[9]);
            xsl[7] = hi_40_2;
            xsl[9] = lo_41_2;
            float hi_42_1 = fmaxf(xsl[10], xsl[11]);
            float lo_43_1 = fminf(xsl[10], xsl[11]);
            xsl[10] = hi_42_1;
            xsl[11] = lo_43_1;
            float hi_44_2 = fmaxf(xsl[12], xsl[13]);
            float lo_45_2 = fminf(xsl[12], xsl[13]);
            xsl[12] = hi_44_2;
            xsl[13] = lo_45_2;
            float hi_46 = fmaxf(xsl[14], xsl[15]);
            float lo_47 = fminf(xsl[14], xsl[15]);
            xsl[14] = hi_46;
            xsl[15] = lo_47;
            float hi_48_2 = fmaxf(xsl[0], xsl[2]);
            float lo_49_2 = fminf(xsl[0], xsl[2]);
            xsl[0] = hi_48_2;
            xsl[2] = lo_49_2;
            float hi_50 = fmaxf(xsl[1], xsl[3]);
            float lo_51 = fminf(xsl[1], xsl[3]);
            xsl[1] = hi_50;
            xsl[3] = lo_51;
            float hi_52_1 = fmaxf(xsl[4], xsl[10]);
            float lo_53_1 = fminf(xsl[4], xsl[10]);
            xsl[4] = hi_52_1;
            xsl[10] = lo_53_1;
            float hi_54_1 = fmaxf(xsl[5], xsl[11]);
            float lo_55_1 = fminf(xsl[5], xsl[11]);
            xsl[5] = hi_54_1;
            xsl[11] = lo_55_1;
            float hi_56_2 = fmaxf(xsl[6], xsl[7]);
            float lo_57_2 = fminf(xsl[6], xsl[7]);
            xsl[6] = hi_56_2;
            xsl[7] = lo_57_2;
            float hi_58 = fmaxf(xsl[8], xsl[9]);
            float lo_59 = fminf(xsl[8], xsl[9]);
            xsl[8] = hi_58;
            xsl[9] = lo_59;
            float hi_60_2 = fmaxf(xsl[12], xsl[14]);
            float lo_61_2 = fminf(xsl[12], xsl[14]);
            xsl[12] = hi_60_2;
            xsl[14] = lo_61_2;
            float hi_62_1 = fmaxf(xsl[13], xsl[15]);
            float lo_63_1 = fminf(xsl[13], xsl[15]);
            xsl[13] = hi_62_1;
            xsl[15] = lo_63_1;
            float hi_64_3 = fmaxf(xsl[1], xsl[2]);
            float lo_65_3 = fminf(xsl[1], xsl[2]);
            xsl[1] = hi_64_3;
            xsl[2] = lo_65_3;
            float hi_66_2 = fmaxf(xsl[3], xsl[12]);
            float lo_67_2 = fminf(xsl[3], xsl[12]);
            xsl[3] = hi_66_2;
            xsl[12] = lo_67_2;
            float hi_68_2 = fmaxf(xsl[4], xsl[6]);
            float lo_69_2 = fminf(xsl[4], xsl[6]);
            xsl[4] = hi_68_2;
            xsl[6] = lo_69_2;
            float hi_70_2 = fmaxf(xsl[5], xsl[7]);
            float lo_71_2 = fminf(xsl[5], xsl[7]);
            xsl[5] = hi_70_2;
            xsl[7] = lo_71_2;
            float hi_72_2 = fmaxf(xsl[8], xsl[10]);
            float lo_73_2 = fminf(xsl[8], xsl[10]);
            xsl[8] = hi_72_2;
            xsl[10] = lo_73_2;
            float hi_74_2 = fmaxf(xsl[9], xsl[11]);
            float lo_75_2 = fminf(xsl[9], xsl[11]);
            xsl[9] = hi_74_2;
            xsl[11] = lo_75_2;
            float hi_76_3 = fmaxf(xsl[13], xsl[14]);
            float lo_77_3 = fminf(xsl[13], xsl[14]);
            xsl[13] = hi_76_3;
            xsl[14] = lo_77_3;
            float hi_78_2 = fmaxf(xsl[1], xsl[4]);
            float lo_79_2 = fminf(xsl[1], xsl[4]);
            xsl[1] = hi_78_2;
            xsl[4] = lo_79_2;
            float hi_80_3 = fmaxf(xsl[2], xsl[6]);
            float lo_81_3 = fminf(xsl[2], xsl[6]);
            xsl[2] = hi_80_3;
            xsl[6] = lo_81_3;
            float hi_82_2 = fmaxf(xsl[5], xsl[8]);
            float lo_83_2 = fminf(xsl[5], xsl[8]);
            xsl[5] = hi_82_2;
            xsl[8] = lo_83_2;
            float hi_84_3 = fmaxf(xsl[7], xsl[10]);
            float lo_85_3 = fminf(xsl[7], xsl[10]);
            xsl[7] = hi_84_3;
            xsl[10] = lo_85_3;
            float hi_86_2 = fmaxf(xsl[9], xsl[13]);
            float lo_87_2 = fminf(xsl[9], xsl[13]);
            xsl[9] = hi_86_2;
            xsl[13] = lo_87_2;
            float hi_88_2 = fmaxf(xsl[11], xsl[14]);
            float lo_89_2 = fminf(xsl[11], xsl[14]);
            xsl[11] = hi_88_2;
            xsl[14] = lo_89_2;
            float hi_90_1 = fmaxf(xsl[2], xsl[4]);
            float lo_91_1 = fminf(xsl[2], xsl[4]);
            xsl[2] = hi_90_1;
            xsl[4] = lo_91_1;
            float hi_92_3 = fmaxf(xsl[3], xsl[6]);
            float lo_93_3 = fminf(xsl[3], xsl[6]);
            xsl[3] = hi_92_3;
            xsl[6] = lo_93_3;
            float hi_94_1 = fmaxf(xsl[9], xsl[12]);
            float lo_95_1 = fminf(xsl[9], xsl[12]);
            xsl[9] = hi_94_1;
            xsl[12] = lo_95_1;
            float hi_96_3 = fmaxf(xsl[11], xsl[13]);
            float lo_97_3 = fminf(xsl[11], xsl[13]);
            xsl[11] = hi_96_3;
            xsl[13] = lo_97_3;
            float hi_98_1 = fmaxf(xsl[3], xsl[5]);
            float lo_99_1 = fminf(xsl[3], xsl[5]);
            xsl[3] = hi_98_1;
            xsl[5] = lo_99_1;
            float hi_100_3 = fmaxf(xsl[6], xsl[8]);
            float lo_101_3 = fminf(xsl[6], xsl[8]);
            xsl[6] = hi_100_3;
            xsl[8] = lo_101_3;
            float hi_102_2 = fmaxf(xsl[7], xsl[9]);
            float lo_103_2 = fminf(xsl[7], xsl[9]);
            xsl[7] = hi_102_2;
            xsl[9] = lo_103_2;
            float hi_104_3 = fmaxf(xsl[10], xsl[12]);
            float lo_105_3 = fminf(xsl[10], xsl[12]);
            xsl[10] = hi_104_3;
            xsl[12] = lo_105_3;
            float hi_106_2 = fmaxf(xsl[3], xsl[4]);
            float lo_107_2 = fminf(xsl[3], xsl[4]);
            xsl[3] = hi_106_2;
            xsl[4] = lo_107_2;
            float hi_108_2 = fmaxf(xsl[5], xsl[6]);
            float lo_109_2 = fminf(xsl[5], xsl[6]);
            xsl[5] = hi_108_2;
            xsl[6] = lo_109_2;
            float hi_110_2 = fmaxf(xsl[7], xsl[8]);
            float lo_111_2 = fminf(xsl[7], xsl[8]);
            xsl[7] = hi_110_2;
            xsl[8] = lo_111_2;
            float hi_112_1 = fmaxf(xsl[9], xsl[10]);
            float lo_113_1 = fminf(xsl[9], xsl[10]);
            xsl[9] = hi_112_1;
            xsl[10] = lo_113_1;
            float hi_114_2 = fmaxf(xsl[11], xsl[12]);
            float lo_115_2 = fminf(xsl[11], xsl[12]);
            xsl[11] = hi_114_2;
            xsl[12] = lo_115_2;
            float hi_116_1 = fmaxf(xsl[6], xsl[7]);
            float lo_117_1 = fminf(xsl[6], xsl[7]);
            xsl[6] = hi_116_1;
            xsl[7] = lo_117_1;
            float hi_118_2 = fmaxf(xsl[8], xsl[9]);
            float lo_119_2 = fminf(xsl[8], xsl[9]);
            xsl[8] = hi_118_2;
            xsl[9] = lo_119_2;
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
