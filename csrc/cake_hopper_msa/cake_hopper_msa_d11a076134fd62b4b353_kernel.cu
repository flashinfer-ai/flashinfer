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
#define SMEM_TREE_STAGE_BYTES 544
#define SMEM_TREE_STRIDE 544
#define SMEM_QCOL_OFF 544
#define SMEM_QCOL_STAGE_BYTES 32
#define SMEM_QCOL_STRIDE 32
#define SMEM_FLAGW_OFF 576
#define SMEM_FLAGW_STAGE_BYTES 16
#define SMEM_FLAGW_STRIDE 16
#define SMEM_TOTAL 640
#define THREADS 32

extern "C" {

__global__ __launch_bounds__(32) void kernel_cake_hopper_msa_d11a076134fd62b4b353(
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
  unsigned int* qcol = reinterpret_cast<unsigned int*>(smem_raw + 544);
  const int qcol_addr = smem + 544;
  unsigned int* flagw = reinterpret_cast<unsigned int*>(smem_raw + 576);
  const int flagw_addr = smem + 576;

  // === Task calls (dependency order) ===
  int tid_1 = threadIdx.x;
  int w = tid_1 / 8;
  int c = tid_1 - w * 8;
  int whi = w / 4;
  int col = blockIdx.x * 8 + c;
  int head = blockIdx.y;
  if (tid_1 == 0) {
    flagw[0] = 0;
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
  int t0_0 = w * 16;
  float sc = __uint_as_float(cb[0]);
  sc = fmaxf(sc, -1.7014118346046923e+38f);
  sc = fminf(sc, 1.7014118346046923e+38f);
  unsigned int key = __as_u32(sc) & 4294967232u | (unsigned int)t0_0;
  kb[0] = __uint_as_float(key);
  float sc_1 = __uint_as_float(cb[1]);
  sc_1 = fmaxf(sc_1, -1.7014118346046923e+38f);
  sc_1 = fminf(sc_1, 1.7014118346046923e+38f);
  unsigned int key_2 = __as_u32(sc_1) & 4294967232u | (unsigned int)(t0_0 + 1);
  kb[1] = __uint_as_float(key_2);
  float sc_3 = __uint_as_float(cb[2]);
  sc_3 = fmaxf(sc_3, -1.7014118346046923e+38f);
  sc_3 = fminf(sc_3, 1.7014118346046923e+38f);
  unsigned int key_4 = __as_u32(sc_3) & 4294967232u | (unsigned int)(t0_0 + 2);
  kb[2] = __uint_as_float(key_4);
  float sc_5 = __uint_as_float(cb[3]);
  sc_5 = fmaxf(sc_5, -1.7014118346046923e+38f);
  sc_5 = fminf(sc_5, 1.7014118346046923e+38f);
  unsigned int key_6 = __as_u32(sc_5) & 4294967232u | (unsigned int)(t0_0 + 3);
  kb[3] = __uint_as_float(key_6);
  float sc_7 = __uint_as_float(cb[4]);
  sc_7 = fmaxf(sc_7, -1.7014118346046923e+38f);
  sc_7 = fminf(sc_7, 1.7014118346046923e+38f);
  unsigned int key_8 = __as_u32(sc_7) & 4294967232u | (unsigned int)(t0_0 + 4);
  kb[4] = __uint_as_float(key_8);
  float sc_9 = __uint_as_float(cb[5]);
  sc_9 = fmaxf(sc_9, -1.7014118346046923e+38f);
  sc_9 = fminf(sc_9, 1.7014118346046923e+38f);
  unsigned int key_10 = __as_u32(sc_9) & 4294967232u | (unsigned int)(t0_0 + 5);
  kb[5] = __uint_as_float(key_10);
  float sc_11 = __uint_as_float(cb[6]);
  sc_11 = fmaxf(sc_11, -1.7014118346046923e+38f);
  sc_11 = fminf(sc_11, 1.7014118346046923e+38f);
  unsigned int key_12 = __as_u32(sc_11) & 4294967232u | (unsigned int)(t0_0 + 6);
  kb[6] = __uint_as_float(key_12);
  float sc_13 = __uint_as_float(cb[7]);
  sc_13 = fmaxf(sc_13, -1.7014118346046923e+38f);
  sc_13 = fminf(sc_13, 1.7014118346046923e+38f);
  unsigned int key_14 = __as_u32(sc_13) & 4294967232u | (unsigned int)(t0_0 + 7);
  kb[7] = __uint_as_float(key_14);
  float sc_15 = __uint_as_float(cb[8]);
  sc_15 = fmaxf(sc_15, -1.7014118346046923e+38f);
  sc_15 = fminf(sc_15, 1.7014118346046923e+38f);
  unsigned int key_16 = __as_u32(sc_15) & 4294967232u | (unsigned int)(t0_0 + 8);
  kb[8] = __uint_as_float(key_16);
  float sc_17 = __uint_as_float(cb[9]);
  sc_17 = fmaxf(sc_17, -1.7014118346046923e+38f);
  sc_17 = fminf(sc_17, 1.7014118346046923e+38f);
  unsigned int key_18 = __as_u32(sc_17) & 4294967232u | (unsigned int)(t0_0 + 9);
  kb[9] = __uint_as_float(key_18);
  float sc_19 = __uint_as_float(cb[10]);
  sc_19 = fmaxf(sc_19, -1.7014118346046923e+38f);
  sc_19 = fminf(sc_19, 1.7014118346046923e+38f);
  unsigned int key_20 = __as_u32(sc_19) & 4294967232u | (unsigned int)(t0_0 + 10);
  kb[10] = __uint_as_float(key_20);
  float sc_21 = __uint_as_float(cb[11]);
  sc_21 = fmaxf(sc_21, -1.7014118346046923e+38f);
  sc_21 = fminf(sc_21, 1.7014118346046923e+38f);
  unsigned int key_22 = __as_u32(sc_21) & 4294967232u | (unsigned int)(t0_0 + 11);
  kb[11] = __uint_as_float(key_22);
  float sc_23 = __uint_as_float(cb[12]);
  sc_23 = fmaxf(sc_23, -1.7014118346046923e+38f);
  sc_23 = fminf(sc_23, 1.7014118346046923e+38f);
  unsigned int key_24 = __as_u32(sc_23) & 4294967232u | (unsigned int)(t0_0 + 12);
  kb[12] = __uint_as_float(key_24);
  float sc_25 = __uint_as_float(cb[13]);
  sc_25 = fmaxf(sc_25, -1.7014118346046923e+38f);
  sc_25 = fminf(sc_25, 1.7014118346046923e+38f);
  unsigned int key_26 = __as_u32(sc_25) & 4294967232u | (unsigned int)(t0_0 + 13);
  kb[13] = __uint_as_float(key_26);
  float sc_27 = __uint_as_float(cb[14]);
  sc_27 = fmaxf(sc_27, -1.7014118346046923e+38f);
  sc_27 = fminf(sc_27, 1.7014118346046923e+38f);
  unsigned int key_28 = __as_u32(sc_27) & 4294967232u | (unsigned int)(t0_0 + 14);
  kb[14] = __uint_as_float(key_28);
  float sc_29 = __uint_as_float(cb[15]);
  sc_29 = fmaxf(sc_29, -1.7014118346046923e+38f);
  sc_29 = fminf(sc_29, 1.7014118346046923e+38f);
  unsigned int key_30 = __as_u32(sc_29) & 4294967232u | (unsigned int)(t0_0 + 15);
  kb[15] = __uint_as_float(key_30);
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
#pragma unroll 1
  for (int k = 0; k < 2; k++) {
    int m = 8 << k;
    float ob[16];
    float _shfl_xor_sync_0 = __shfl_xor_sync(0xFFFFFFFF, a[0], m);
    ob[0] = _shfl_xor_sync_0;
    float _shfl_xor_sync_1 = __shfl_xor_sync(0xFFFFFFFF, a[1], m);
    ob[1] = _shfl_xor_sync_1;
    float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, a[2], m);
    ob[2] = _shfl_xor_sync_2;
    float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, a[3], m);
    ob[3] = _shfl_xor_sync_3;
    float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, a[4], m);
    ob[4] = _shfl_xor_sync_4;
    float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, a[5], m);
    ob[5] = _shfl_xor_sync_5;
    float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, a[6], m);
    ob[6] = _shfl_xor_sync_6;
    float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, a[7], m);
    ob[7] = _shfl_xor_sync_7;
    float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, a[8], m);
    ob[8] = _shfl_xor_sync_8;
    float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, a[9], m);
    ob[9] = _shfl_xor_sync_9;
    float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, a[10], m);
    ob[10] = _shfl_xor_sync_10;
    float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, a[11], m);
    ob[11] = _shfl_xor_sync_11;
    float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, a[12], m);
    ob[12] = _shfl_xor_sync_12;
    float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, a[13], m);
    ob[13] = _shfl_xor_sync_13;
    float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, a[14], m);
    ob[14] = _shfl_xor_sync_14;
    float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, a[15], m);
    ob[15] = _shfl_xor_sync_15;
    float orej = __shfl_xor_sync(0xFFFFFFFF, rej, m);
    rej = fmaxf(rej, orej);
    float r = rej;
    float hi_0 = fmaxf(a[0], ob[15]);
    float lo_1 = fminf(a[0], ob[15]);
    a[0] = hi_0;
    r = fmaxf(r, lo_1);
    float hi_2 = fmaxf(a[1], ob[14]);
    float lo_3 = fminf(a[1], ob[14]);
    a[1] = hi_2;
    r = fmaxf(r, lo_3);
    float hi_4 = fmaxf(a[2], ob[13]);
    float lo_5 = fminf(a[2], ob[13]);
    a[2] = hi_4;
    r = fmaxf(r, lo_5);
    float hi_6 = fmaxf(a[3], ob[12]);
    float lo_7 = fminf(a[3], ob[12]);
    a[3] = hi_6;
    r = fmaxf(r, lo_7);
    float hi_8 = fmaxf(a[4], ob[11]);
    float lo_9 = fminf(a[4], ob[11]);
    a[4] = hi_8;
    r = fmaxf(r, lo_9);
    float hi_10 = fmaxf(a[5], ob[10]);
    float lo_11 = fminf(a[5], ob[10]);
    a[5] = hi_10;
    r = fmaxf(r, lo_11);
    float hi_12 = fmaxf(a[6], ob[9]);
    float lo_13 = fminf(a[6], ob[9]);
    a[6] = hi_12;
    r = fmaxf(r, lo_13);
    float hi_14 = fmaxf(a[7], ob[8]);
    float lo_15 = fminf(a[7], ob[8]);
    a[7] = hi_14;
    r = fmaxf(r, lo_15);
    float hi_16 = fmaxf(a[8], ob[7]);
    float lo_17 = fminf(a[8], ob[7]);
    a[8] = hi_16;
    r = fmaxf(r, lo_17);
    float hi_18 = fmaxf(a[9], ob[6]);
    float lo_19 = fminf(a[9], ob[6]);
    a[9] = hi_18;
    r = fmaxf(r, lo_19);
    float hi_20 = fmaxf(a[10], ob[5]);
    float lo_21 = fminf(a[10], ob[5]);
    a[10] = hi_20;
    r = fmaxf(r, lo_21);
    float hi_22 = fmaxf(a[11], ob[4]);
    float lo_23 = fminf(a[11], ob[4]);
    a[11] = hi_22;
    r = fmaxf(r, lo_23);
    float hi_24 = fmaxf(a[12], ob[3]);
    float lo_25 = fminf(a[12], ob[3]);
    a[12] = hi_24;
    r = fmaxf(r, lo_25);
    float hi_26 = fmaxf(a[13], ob[2]);
    float lo_27 = fminf(a[13], ob[2]);
    a[13] = hi_26;
    r = fmaxf(r, lo_27);
    float hi_28 = fmaxf(a[14], ob[1]);
    float lo_29 = fminf(a[14], ob[1]);
    a[14] = hi_28;
    r = fmaxf(r, lo_29);
    float hi_30 = fmaxf(a[15], ob[0]);
    float lo_31 = fminf(a[15], ob[0]);
    a[15] = hi_30;
    r = fmaxf(r, lo_31);
    float hi_32 = fmaxf(a[0], a[8]);
    float lo_33 = fminf(a[0], a[8]);
    a[0] = hi_32;
    a[8] = lo_33;
    float hi_34 = fmaxf(a[1], a[9]);
    float lo_35 = fminf(a[1], a[9]);
    a[1] = hi_34;
    a[9] = lo_35;
    float hi_36 = fmaxf(a[2], a[10]);
    float lo_37 = fminf(a[2], a[10]);
    a[2] = hi_36;
    a[10] = lo_37;
    float hi_38 = fmaxf(a[3], a[11]);
    float lo_39 = fminf(a[3], a[11]);
    a[3] = hi_38;
    a[11] = lo_39;
    float hi_40 = fmaxf(a[4], a[12]);
    float lo_41 = fminf(a[4], a[12]);
    a[4] = hi_40;
    a[12] = lo_41;
    float hi_42 = fmaxf(a[5], a[13]);
    float lo_43 = fminf(a[5], a[13]);
    a[5] = hi_42;
    a[13] = lo_43;
    float hi_44 = fmaxf(a[6], a[14]);
    float lo_45 = fminf(a[6], a[14]);
    a[6] = hi_44;
    a[14] = lo_45;
    float hi_46 = fmaxf(a[7], a[15]);
    float lo_47 = fminf(a[7], a[15]);
    a[7] = hi_46;
    a[15] = lo_47;
    float hi_48 = fmaxf(a[0], a[4]);
    float lo_49 = fminf(a[0], a[4]);
    a[0] = hi_48;
    a[4] = lo_49;
    float hi_50 = fmaxf(a[1], a[5]);
    float lo_51 = fminf(a[1], a[5]);
    a[1] = hi_50;
    a[5] = lo_51;
    float hi_52 = fmaxf(a[2], a[6]);
    float lo_53 = fminf(a[2], a[6]);
    a[2] = hi_52;
    a[6] = lo_53;
    float hi_54 = fmaxf(a[3], a[7]);
    float lo_55 = fminf(a[3], a[7]);
    a[3] = hi_54;
    a[7] = lo_55;
    float hi_56 = fmaxf(a[8], a[12]);
    float lo_57 = fminf(a[8], a[12]);
    a[8] = hi_56;
    a[12] = lo_57;
    float hi_58 = fmaxf(a[9], a[13]);
    float lo_59 = fminf(a[9], a[13]);
    a[9] = hi_58;
    a[13] = lo_59;
    float hi_60 = fmaxf(a[10], a[14]);
    float lo_61 = fminf(a[10], a[14]);
    a[10] = hi_60;
    a[14] = lo_61;
    float hi_62 = fmaxf(a[11], a[15]);
    float lo_63 = fminf(a[11], a[15]);
    a[11] = hi_62;
    a[15] = lo_63;
    float hi_64 = fmaxf(a[0], a[2]);
    float lo_65 = fminf(a[0], a[2]);
    a[0] = hi_64;
    a[2] = lo_65;
    float hi_66 = fmaxf(a[1], a[3]);
    float lo_67 = fminf(a[1], a[3]);
    a[1] = hi_66;
    a[3] = lo_67;
    float hi_68 = fmaxf(a[4], a[6]);
    float lo_69 = fminf(a[4], a[6]);
    a[4] = hi_68;
    a[6] = lo_69;
    float hi_70 = fmaxf(a[5], a[7]);
    float lo_71 = fminf(a[5], a[7]);
    a[5] = hi_70;
    a[7] = lo_71;
    float hi_72 = fmaxf(a[8], a[10]);
    float lo_73 = fminf(a[8], a[10]);
    a[8] = hi_72;
    a[10] = lo_73;
    float hi_74 = fmaxf(a[9], a[11]);
    float lo_75 = fminf(a[9], a[11]);
    a[9] = hi_74;
    a[11] = lo_75;
    float hi_76 = fmaxf(a[12], a[14]);
    float lo_77 = fminf(a[12], a[14]);
    a[12] = hi_76;
    a[14] = lo_77;
    float hi_78 = fmaxf(a[13], a[15]);
    float lo_79 = fminf(a[13], a[15]);
    a[13] = hi_78;
    a[15] = lo_79;
    float hi_80 = fmaxf(a[0], a[1]);
    float lo_81 = fminf(a[0], a[1]);
    a[0] = hi_80;
    a[1] = lo_81;
    float hi_82 = fmaxf(a[2], a[3]);
    float lo_83 = fminf(a[2], a[3]);
    a[2] = hi_82;
    a[3] = lo_83;
    float hi_84 = fmaxf(a[4], a[5]);
    float lo_85 = fminf(a[4], a[5]);
    a[4] = hi_84;
    a[5] = lo_85;
    float hi_86 = fmaxf(a[6], a[7]);
    float lo_87 = fminf(a[6], a[7]);
    a[6] = hi_86;
    a[7] = lo_87;
    float hi_88 = fmaxf(a[8], a[9]);
    float lo_89 = fminf(a[8], a[9]);
    a[8] = hi_88;
    a[9] = lo_89;
    float hi_90 = fmaxf(a[10], a[11]);
    float lo_91 = fminf(a[10], a[11]);
    a[10] = hi_90;
    a[11] = lo_91;
    float hi_92 = fmaxf(a[12], a[13]);
    float lo_93 = fminf(a[12], a[13]);
    a[12] = hi_92;
    a[13] = lo_93;
    float hi_94 = fmaxf(a[14], a[15]);
    float lo_95 = fminf(a[14], a[15]);
    a[14] = hi_94;
    a[15] = lo_95;
    rej = r;
  }
  unsigned int u16 = __as_u32(a[15]);
  unsigned int c16 = u16 & 4294967232u;
  unsigned int cr = __as_u32(rej) & 4294967232u;
  int own = 0;
  if (w == 0 && c16 == cr && u16 < 4278190080u && c16 != 2139094976 && col < total_q) {
    own = 1;
  }
  unsigned int bal = __ballot_sync(0xFFFFFFFF, own != 0);
  unsigned int need2 = 0;
  if (bal != 0) {
    need2 = 1;
  }
  unsigned int qc = 4294967295u;
  if (own != 0) {
    qc = c16;
  }
  unsigned int _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, qc, 8);
  unsigned int _min_16 = ((qc) < (_shfl_xor_sync_17) ? (qc) : (_shfl_xor_sync_17));
  qc = _min_16;
  unsigned int _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, qc, 16);
  unsigned int _min_18 = ((qc) < (_shfl_xor_sync_19) ? (qc) : (_shfl_xor_sync_19));
  qc = _min_18;
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
    if (flagged != 0) {
      float kb2o[16];
      int t0_1 = w * 16;
      float qf = __uint_as_float(qc);
      float sc_2 = __uint_as_float(cb[0]);
      sc_2 = fmaxf(sc_2, -1.7014118346046923e+38f);
      sc_2 = fminf(sc_2, 1.7014118346046923e+38f);
      unsigned int u = __as_u32(sc_2);
      unsigned int cls = u & 4294967232u;
      unsigned int key_3 = 0;
      if (qf < __uint_as_float(cls) && cls < 4278190080u) {
        key_3 = 1073741824 | (unsigned int)t0_1;
      }
      if (cls == qc) {
        unsigned int lowb = (u ^ (unsigned int)((int)u >> 31) & 63) & 63;
        key_3 = 536870912 | lowb << 6 | (unsigned int)t0_1;
      }
      kb2o[0] = __uint_as_float(key_3);
      float sc_4 = __uint_as_float(cb[1]);
      sc_4 = fmaxf(sc_4, -1.7014118346046923e+38f);
      sc_4 = fminf(sc_4, 1.7014118346046923e+38f);
      unsigned int u_5 = __as_u32(sc_4);
      unsigned int cls_6 = u_5 & 4294967232u;
      unsigned int key_7 = 0;
      if (qf < __uint_as_float(cls_6) && cls_6 < 4278190080u) {
        key_7 = 1073741824 | (unsigned int)(t0_1 + 1);
      }
      if (cls_6 == qc) {
        unsigned int lowb_1 = (u_5 ^ (unsigned int)((int)u_5 >> 31) & 63) & 63;
        key_7 = 536870912 | lowb_1 << 6 | (unsigned int)(t0_1 + 1);
      }
      kb2o[1] = __uint_as_float(key_7);
      float sc_8 = __uint_as_float(cb[2]);
      sc_8 = fmaxf(sc_8, -1.7014118346046923e+38f);
      sc_8 = fminf(sc_8, 1.7014118346046923e+38f);
      unsigned int u_9 = __as_u32(sc_8);
      unsigned int cls_10 = u_9 & 4294967232u;
      unsigned int key_11 = 0;
      if (qf < __uint_as_float(cls_10) && cls_10 < 4278190080u) {
        key_11 = 1073741824 | (unsigned int)(t0_1 + 2);
      }
      if (cls_10 == qc) {
        unsigned int lowb_2 = (u_9 ^ (unsigned int)((int)u_9 >> 31) & 63) & 63;
        key_11 = 536870912 | lowb_2 << 6 | (unsigned int)(t0_1 + 2);
      }
      kb2o[2] = __uint_as_float(key_11);
      float sc_12 = __uint_as_float(cb[3]);
      sc_12 = fmaxf(sc_12, -1.7014118346046923e+38f);
      sc_12 = fminf(sc_12, 1.7014118346046923e+38f);
      unsigned int u_13 = __as_u32(sc_12);
      unsigned int cls_14 = u_13 & 4294967232u;
      unsigned int key_15 = 0;
      if (qf < __uint_as_float(cls_14) && cls_14 < 4278190080u) {
        key_15 = 1073741824 | (unsigned int)(t0_1 + 3);
      }
      if (cls_14 == qc) {
        unsigned int lowb_3 = (u_13 ^ (unsigned int)((int)u_13 >> 31) & 63) & 63;
        key_15 = 536870912 | lowb_3 << 6 | (unsigned int)(t0_1 + 3);
      }
      kb2o[3] = __uint_as_float(key_15);
      float sc_16 = __uint_as_float(cb[4]);
      sc_16 = fmaxf(sc_16, -1.7014118346046923e+38f);
      sc_16 = fminf(sc_16, 1.7014118346046923e+38f);
      unsigned int u_17 = __as_u32(sc_16);
      unsigned int cls_18 = u_17 & 4294967232u;
      unsigned int key_19 = 0;
      if (qf < __uint_as_float(cls_18) && cls_18 < 4278190080u) {
        key_19 = 1073741824 | (unsigned int)(t0_1 + 4);
      }
      if (cls_18 == qc) {
        unsigned int lowb_4 = (u_17 ^ (unsigned int)((int)u_17 >> 31) & 63) & 63;
        key_19 = 536870912 | lowb_4 << 6 | (unsigned int)(t0_1 + 4);
      }
      kb2o[4] = __uint_as_float(key_19);
      float sc_20 = __uint_as_float(cb[5]);
      sc_20 = fmaxf(sc_20, -1.7014118346046923e+38f);
      sc_20 = fminf(sc_20, 1.7014118346046923e+38f);
      unsigned int u_21 = __as_u32(sc_20);
      unsigned int cls_22 = u_21 & 4294967232u;
      unsigned int key_23 = 0;
      if (qf < __uint_as_float(cls_22) && cls_22 < 4278190080u) {
        key_23 = 1073741824 | (unsigned int)(t0_1 + 5);
      }
      if (cls_22 == qc) {
        unsigned int lowb_5 = (u_21 ^ (unsigned int)((int)u_21 >> 31) & 63) & 63;
        key_23 = 536870912 | lowb_5 << 6 | (unsigned int)(t0_1 + 5);
      }
      kb2o[5] = __uint_as_float(key_23);
      float sc_24 = __uint_as_float(cb[6]);
      sc_24 = fmaxf(sc_24, -1.7014118346046923e+38f);
      sc_24 = fminf(sc_24, 1.7014118346046923e+38f);
      unsigned int u_25 = __as_u32(sc_24);
      unsigned int cls_26 = u_25 & 4294967232u;
      unsigned int key_27 = 0;
      if (qf < __uint_as_float(cls_26) && cls_26 < 4278190080u) {
        key_27 = 1073741824 | (unsigned int)(t0_1 + 6);
      }
      if (cls_26 == qc) {
        unsigned int lowb_6 = (u_25 ^ (unsigned int)((int)u_25 >> 31) & 63) & 63;
        key_27 = 536870912 | lowb_6 << 6 | (unsigned int)(t0_1 + 6);
      }
      kb2o[6] = __uint_as_float(key_27);
      float sc_28 = __uint_as_float(cb[7]);
      sc_28 = fmaxf(sc_28, -1.7014118346046923e+38f);
      sc_28 = fminf(sc_28, 1.7014118346046923e+38f);
      unsigned int u_29 = __as_u32(sc_28);
      unsigned int cls_30 = u_29 & 4294967232u;
      unsigned int key_31 = 0;
      if (qf < __uint_as_float(cls_30) && cls_30 < 4278190080u) {
        key_31 = 1073741824 | (unsigned int)(t0_1 + 7);
      }
      if (cls_30 == qc) {
        unsigned int lowb_7 = (u_29 ^ (unsigned int)((int)u_29 >> 31) & 63) & 63;
        key_31 = 536870912 | lowb_7 << 6 | (unsigned int)(t0_1 + 7);
      }
      kb2o[7] = __uint_as_float(key_31);
      float sc_32 = __uint_as_float(cb[8]);
      sc_32 = fmaxf(sc_32, -1.7014118346046923e+38f);
      sc_32 = fminf(sc_32, 1.7014118346046923e+38f);
      unsigned int u_33 = __as_u32(sc_32);
      unsigned int cls_34 = u_33 & 4294967232u;
      unsigned int key_35 = 0;
      if (qf < __uint_as_float(cls_34) && cls_34 < 4278190080u) {
        key_35 = 1073741824 | (unsigned int)(t0_1 + 8);
      }
      if (cls_34 == qc) {
        unsigned int lowb_8 = (u_33 ^ (unsigned int)((int)u_33 >> 31) & 63) & 63;
        key_35 = 536870912 | lowb_8 << 6 | (unsigned int)(t0_1 + 8);
      }
      kb2o[8] = __uint_as_float(key_35);
      float sc_36 = __uint_as_float(cb[9]);
      sc_36 = fmaxf(sc_36, -1.7014118346046923e+38f);
      sc_36 = fminf(sc_36, 1.7014118346046923e+38f);
      unsigned int u_37 = __as_u32(sc_36);
      unsigned int cls_38 = u_37 & 4294967232u;
      unsigned int key_39 = 0;
      if (qf < __uint_as_float(cls_38) && cls_38 < 4278190080u) {
        key_39 = 1073741824 | (unsigned int)(t0_1 + 9);
      }
      if (cls_38 == qc) {
        unsigned int lowb_9 = (u_37 ^ (unsigned int)((int)u_37 >> 31) & 63) & 63;
        key_39 = 536870912 | lowb_9 << 6 | (unsigned int)(t0_1 + 9);
      }
      kb2o[9] = __uint_as_float(key_39);
      float sc_40 = __uint_as_float(cb[10]);
      sc_40 = fmaxf(sc_40, -1.7014118346046923e+38f);
      sc_40 = fminf(sc_40, 1.7014118346046923e+38f);
      unsigned int u_41 = __as_u32(sc_40);
      unsigned int cls_42 = u_41 & 4294967232u;
      unsigned int key_43 = 0;
      if (qf < __uint_as_float(cls_42) && cls_42 < 4278190080u) {
        key_43 = 1073741824 | (unsigned int)(t0_1 + 10);
      }
      if (cls_42 == qc) {
        unsigned int lowb_10 = (u_41 ^ (unsigned int)((int)u_41 >> 31) & 63) & 63;
        key_43 = 536870912 | lowb_10 << 6 | (unsigned int)(t0_1 + 10);
      }
      kb2o[10] = __uint_as_float(key_43);
      float sc_44 = __uint_as_float(cb[11]);
      sc_44 = fmaxf(sc_44, -1.7014118346046923e+38f);
      sc_44 = fminf(sc_44, 1.7014118346046923e+38f);
      unsigned int u_45 = __as_u32(sc_44);
      unsigned int cls_46 = u_45 & 4294967232u;
      unsigned int key_47 = 0;
      if (qf < __uint_as_float(cls_46) && cls_46 < 4278190080u) {
        key_47 = 1073741824 | (unsigned int)(t0_1 + 11);
      }
      if (cls_46 == qc) {
        unsigned int lowb_11 = (u_45 ^ (unsigned int)((int)u_45 >> 31) & 63) & 63;
        key_47 = 536870912 | lowb_11 << 6 | (unsigned int)(t0_1 + 11);
      }
      kb2o[11] = __uint_as_float(key_47);
      float sc_48 = __uint_as_float(cb[12]);
      sc_48 = fmaxf(sc_48, -1.7014118346046923e+38f);
      sc_48 = fminf(sc_48, 1.7014118346046923e+38f);
      unsigned int u_49 = __as_u32(sc_48);
      unsigned int cls_50 = u_49 & 4294967232u;
      unsigned int key_51 = 0;
      if (qf < __uint_as_float(cls_50) && cls_50 < 4278190080u) {
        key_51 = 1073741824 | (unsigned int)(t0_1 + 12);
      }
      if (cls_50 == qc) {
        unsigned int lowb_12 = (u_49 ^ (unsigned int)((int)u_49 >> 31) & 63) & 63;
        key_51 = 536870912 | lowb_12 << 6 | (unsigned int)(t0_1 + 12);
      }
      kb2o[12] = __uint_as_float(key_51);
      float sc_52 = __uint_as_float(cb[13]);
      sc_52 = fmaxf(sc_52, -1.7014118346046923e+38f);
      sc_52 = fminf(sc_52, 1.7014118346046923e+38f);
      unsigned int u_53 = __as_u32(sc_52);
      unsigned int cls_54 = u_53 & 4294967232u;
      unsigned int key_55 = 0;
      if (qf < __uint_as_float(cls_54) && cls_54 < 4278190080u) {
        key_55 = 1073741824 | (unsigned int)(t0_1 + 13);
      }
      if (cls_54 == qc) {
        unsigned int lowb_13 = (u_53 ^ (unsigned int)((int)u_53 >> 31) & 63) & 63;
        key_55 = 536870912 | lowb_13 << 6 | (unsigned int)(t0_1 + 13);
      }
      kb2o[13] = __uint_as_float(key_55);
      float sc_56 = __uint_as_float(cb[14]);
      sc_56 = fmaxf(sc_56, -1.7014118346046923e+38f);
      sc_56 = fminf(sc_56, 1.7014118346046923e+38f);
      unsigned int u_57 = __as_u32(sc_56);
      unsigned int cls_58 = u_57 & 4294967232u;
      unsigned int key_59 = 0;
      if (qf < __uint_as_float(cls_58) && cls_58 < 4278190080u) {
        key_59 = 1073741824 | (unsigned int)(t0_1 + 14);
      }
      if (cls_58 == qc) {
        unsigned int lowb_14 = (u_57 ^ (unsigned int)((int)u_57 >> 31) & 63) & 63;
        key_59 = 536870912 | lowb_14 << 6 | (unsigned int)(t0_1 + 14);
      }
      kb2o[14] = __uint_as_float(key_59);
      float sc_60 = __uint_as_float(cb[15]);
      sc_60 = fmaxf(sc_60, -1.7014118346046923e+38f);
      sc_60 = fminf(sc_60, 1.7014118346046923e+38f);
      unsigned int u_61 = __as_u32(sc_60);
      unsigned int cls_62 = u_61 & 4294967232u;
      unsigned int key_63 = 0;
      if (qf < __uint_as_float(cls_62) && cls_62 < 4278190080u) {
        key_63 = 1073741824 | (unsigned int)(t0_1 + 15);
      }
      if (cls_62 == qc) {
        unsigned int lowb_15 = (u_61 ^ (unsigned int)((int)u_61 >> 31) & 63) & 63;
        key_63 = 536870912 | lowb_15 << 6 | (unsigned int)(t0_1 + 15);
      }
      kb2o[15] = __uint_as_float(key_63);
      float hi_64_1 = fmaxf(kb2o[0], kb2o[13]);
      float lo_65_1 = fminf(kb2o[0], kb2o[13]);
      kb2o[0] = hi_64_1;
      kb2o[13] = lo_65_1;
      float hi_66_1 = fmaxf(kb2o[1], kb2o[12]);
      float lo_67_1 = fminf(kb2o[1], kb2o[12]);
      kb2o[1] = hi_66_1;
      kb2o[12] = lo_67_1;
      float hi_68_1 = fmaxf(kb2o[2], kb2o[15]);
      float lo_69_1 = fminf(kb2o[2], kb2o[15]);
      kb2o[2] = hi_68_1;
      kb2o[15] = lo_69_1;
      float hi_70_1 = fmaxf(kb2o[3], kb2o[14]);
      float lo_71_1 = fminf(kb2o[3], kb2o[14]);
      kb2o[3] = hi_70_1;
      kb2o[14] = lo_71_1;
      float hi_72_1 = fmaxf(kb2o[4], kb2o[8]);
      float lo_73_1 = fminf(kb2o[4], kb2o[8]);
      kb2o[4] = hi_72_1;
      kb2o[8] = lo_73_1;
      float hi_74_1 = fmaxf(kb2o[5], kb2o[6]);
      float lo_75_1 = fminf(kb2o[5], kb2o[6]);
      kb2o[5] = hi_74_1;
      kb2o[6] = lo_75_1;
      float hi_76_1 = fmaxf(kb2o[7], kb2o[11]);
      float lo_77_1 = fminf(kb2o[7], kb2o[11]);
      kb2o[7] = hi_76_1;
      kb2o[11] = lo_77_1;
      float hi_78_1 = fmaxf(kb2o[9], kb2o[10]);
      float lo_79_1 = fminf(kb2o[9], kb2o[10]);
      kb2o[9] = hi_78_1;
      kb2o[10] = lo_79_1;
      float hi_80_1 = fmaxf(kb2o[0], kb2o[5]);
      float lo_81_1 = fminf(kb2o[0], kb2o[5]);
      kb2o[0] = hi_80_1;
      kb2o[5] = lo_81_1;
      float hi_82_1 = fmaxf(kb2o[1], kb2o[7]);
      float lo_83_1 = fminf(kb2o[1], kb2o[7]);
      kb2o[1] = hi_82_1;
      kb2o[7] = lo_83_1;
      float hi_84_1 = fmaxf(kb2o[2], kb2o[9]);
      float lo_85_1 = fminf(kb2o[2], kb2o[9]);
      kb2o[2] = hi_84_1;
      kb2o[9] = lo_85_1;
      float hi_86_1 = fmaxf(kb2o[3], kb2o[4]);
      float lo_87_1 = fminf(kb2o[3], kb2o[4]);
      kb2o[3] = hi_86_1;
      kb2o[4] = lo_87_1;
      float hi_88_1 = fmaxf(kb2o[6], kb2o[13]);
      float lo_89_1 = fminf(kb2o[6], kb2o[13]);
      kb2o[6] = hi_88_1;
      kb2o[13] = lo_89_1;
      float hi_90_1 = fmaxf(kb2o[8], kb2o[14]);
      float lo_91_1 = fminf(kb2o[8], kb2o[14]);
      kb2o[8] = hi_90_1;
      kb2o[14] = lo_91_1;
      float hi_92_1 = fmaxf(kb2o[10], kb2o[15]);
      float lo_93_1 = fminf(kb2o[10], kb2o[15]);
      kb2o[10] = hi_92_1;
      kb2o[15] = lo_93_1;
      float hi_94_1 = fmaxf(kb2o[11], kb2o[12]);
      float lo_95_1 = fminf(kb2o[11], kb2o[12]);
      kb2o[11] = hi_94_1;
      kb2o[12] = lo_95_1;
      float hi_96 = fmaxf(kb2o[0], kb2o[1]);
      float lo_97 = fminf(kb2o[0], kb2o[1]);
      kb2o[0] = hi_96;
      kb2o[1] = lo_97;
      float hi_98 = fmaxf(kb2o[2], kb2o[3]);
      float lo_99 = fminf(kb2o[2], kb2o[3]);
      kb2o[2] = hi_98;
      kb2o[3] = lo_99;
      float hi_100 = fmaxf(kb2o[4], kb2o[5]);
      float lo_101 = fminf(kb2o[4], kb2o[5]);
      kb2o[4] = hi_100;
      kb2o[5] = lo_101;
      float hi_102 = fmaxf(kb2o[6], kb2o[8]);
      float lo_103 = fminf(kb2o[6], kb2o[8]);
      kb2o[6] = hi_102;
      kb2o[8] = lo_103;
      float hi_104 = fmaxf(kb2o[7], kb2o[9]);
      float lo_105 = fminf(kb2o[7], kb2o[9]);
      kb2o[7] = hi_104;
      kb2o[9] = lo_105;
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
      float hi_152 = fmaxf(kb2o[11], kb2o[14]);
      float lo_153 = fminf(kb2o[11], kb2o[14]);
      kb2o[11] = hi_152;
      kb2o[14] = lo_153;
      float hi_154 = fmaxf(kb2o[2], kb2o[4]);
      float lo_155 = fminf(kb2o[2], kb2o[4]);
      kb2o[2] = hi_154;
      kb2o[4] = lo_155;
      float hi_156 = fmaxf(kb2o[3], kb2o[6]);
      float lo_157 = fminf(kb2o[3], kb2o[6]);
      kb2o[3] = hi_156;
      kb2o[6] = lo_157;
      float hi_158 = fmaxf(kb2o[9], kb2o[12]);
      float lo_159 = fminf(kb2o[9], kb2o[12]);
      kb2o[9] = hi_158;
      kb2o[12] = lo_159;
      float hi_160 = fmaxf(kb2o[11], kb2o[13]);
      float lo_161 = fminf(kb2o[11], kb2o[13]);
      kb2o[11] = hi_160;
      kb2o[13] = lo_161;
      float hi_162 = fmaxf(kb2o[3], kb2o[5]);
      float lo_163 = fminf(kb2o[3], kb2o[5]);
      kb2o[3] = hi_162;
      kb2o[5] = lo_163;
      float hi_164 = fmaxf(kb2o[6], kb2o[8]);
      float lo_165 = fminf(kb2o[6], kb2o[8]);
      kb2o[6] = hi_164;
      kb2o[8] = lo_165;
      float hi_166 = fmaxf(kb2o[7], kb2o[9]);
      float lo_167 = fminf(kb2o[7], kb2o[9]);
      kb2o[7] = hi_166;
      kb2o[9] = lo_167;
      float hi_168 = fmaxf(kb2o[10], kb2o[12]);
      float lo_169 = fminf(kb2o[10], kb2o[12]);
      kb2o[10] = hi_168;
      kb2o[12] = lo_169;
      float hi_170 = fmaxf(kb2o[3], kb2o[4]);
      float lo_171 = fminf(kb2o[3], kb2o[4]);
      kb2o[3] = hi_170;
      kb2o[4] = lo_171;
      float hi_172 = fmaxf(kb2o[5], kb2o[6]);
      float lo_173 = fminf(kb2o[5], kb2o[6]);
      kb2o[5] = hi_172;
      kb2o[6] = lo_173;
      float hi_174 = fmaxf(kb2o[7], kb2o[8]);
      float lo_175 = fminf(kb2o[7], kb2o[8]);
      kb2o[7] = hi_174;
      kb2o[8] = lo_175;
      float hi_176 = fmaxf(kb2o[9], kb2o[10]);
      float lo_177 = fminf(kb2o[9], kb2o[10]);
      kb2o[9] = hi_176;
      kb2o[10] = lo_177;
      float hi_178 = fmaxf(kb2o[11], kb2o[12]);
      float lo_179 = fminf(kb2o[11], kb2o[12]);
      kb2o[11] = hi_178;
      kb2o[12] = lo_179;
      float hi_180 = fmaxf(kb2o[6], kb2o[7]);
      float lo_181 = fminf(kb2o[6], kb2o[7]);
      kb2o[6] = hi_180;
      kb2o[7] = lo_181;
      float hi_182 = fmaxf(kb2o[8], kb2o[9]);
      float lo_183 = fminf(kb2o[8], kb2o[9]);
      kb2o[8] = hi_182;
      kb2o[9] = lo_183;
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
      int m2 = 8 << k_1;
      float ob2[16];
      float _shfl_xor_sync_20 = __shfl_xor_sync(0xFFFFFFFF, a2[0], m2);
      ob2[0] = _shfl_xor_sync_20;
      float _shfl_xor_sync_21 = __shfl_xor_sync(0xFFFFFFFF, a2[1], m2);
      ob2[1] = _shfl_xor_sync_21;
      float _shfl_xor_sync_22 = __shfl_xor_sync(0xFFFFFFFF, a2[2], m2);
      ob2[2] = _shfl_xor_sync_22;
      float _shfl_xor_sync_23 = __shfl_xor_sync(0xFFFFFFFF, a2[3], m2);
      ob2[3] = _shfl_xor_sync_23;
      float _shfl_xor_sync_24 = __shfl_xor_sync(0xFFFFFFFF, a2[4], m2);
      ob2[4] = _shfl_xor_sync_24;
      float _shfl_xor_sync_25 = __shfl_xor_sync(0xFFFFFFFF, a2[5], m2);
      ob2[5] = _shfl_xor_sync_25;
      float _shfl_xor_sync_26 = __shfl_xor_sync(0xFFFFFFFF, a2[6], m2);
      ob2[6] = _shfl_xor_sync_26;
      float _shfl_xor_sync_27 = __shfl_xor_sync(0xFFFFFFFF, a2[7], m2);
      ob2[7] = _shfl_xor_sync_27;
      float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, a2[8], m2);
      ob2[8] = _shfl_xor_sync_28;
      float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, a2[9], m2);
      ob2[9] = _shfl_xor_sync_29;
      float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, a2[10], m2);
      ob2[10] = _shfl_xor_sync_30;
      float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, a2[11], m2);
      ob2[11] = _shfl_xor_sync_31;
      float _shfl_xor_sync_32 = __shfl_xor_sync(0xFFFFFFFF, a2[12], m2);
      ob2[12] = _shfl_xor_sync_32;
      float _shfl_xor_sync_33 = __shfl_xor_sync(0xFFFFFFFF, a2[13], m2);
      ob2[13] = _shfl_xor_sync_33;
      float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, a2[14], m2);
      ob2[14] = _shfl_xor_sync_34;
      float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, a2[15], m2);
      ob2[15] = _shfl_xor_sync_35;
      float hi_0_1 = fmaxf(a2[0], ob2[15]);
      a2[0] = hi_0_1;
      float hi_1 = fmaxf(a2[1], ob2[14]);
      a2[1] = hi_1;
      float hi_2_1 = fmaxf(a2[2], ob2[13]);
      a2[2] = hi_2_1;
      float hi_3 = fmaxf(a2[3], ob2[12]);
      a2[3] = hi_3;
      float hi_4_1 = fmaxf(a2[4], ob2[11]);
      a2[4] = hi_4_1;
      float hi_5 = fmaxf(a2[5], ob2[10]);
      a2[5] = hi_5;
      float hi_6_1 = fmaxf(a2[6], ob2[9]);
      a2[6] = hi_6_1;
      float hi_7 = fmaxf(a2[7], ob2[8]);
      a2[7] = hi_7;
      float hi_8_1 = fmaxf(a2[8], ob2[7]);
      a2[8] = hi_8_1;
      float hi_9 = fmaxf(a2[9], ob2[6]);
      a2[9] = hi_9;
      float hi_10_1 = fmaxf(a2[10], ob2[5]);
      a2[10] = hi_10_1;
      float hi_11 = fmaxf(a2[11], ob2[4]);
      a2[11] = hi_11;
      float hi_12_1 = fmaxf(a2[12], ob2[3]);
      a2[12] = hi_12_1;
      float hi_13 = fmaxf(a2[13], ob2[2]);
      a2[13] = hi_13;
      float hi_14_1 = fmaxf(a2[14], ob2[1]);
      a2[14] = hi_14_1;
      float hi_15 = fmaxf(a2[15], ob2[0]);
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
      float hi_32_1 = fmaxf(a2[0], a2[4]);
      float lo_33_1 = fminf(a2[0], a2[4]);
      a2[0] = hi_32_1;
      a2[4] = lo_33_1;
      float hi_34_1 = fmaxf(a2[1], a2[5]);
      float lo_35_1 = fminf(a2[1], a2[5]);
      a2[1] = hi_34_1;
      a2[5] = lo_35_1;
      float hi_36_1 = fmaxf(a2[2], a2[6]);
      float lo_37_1 = fminf(a2[2], a2[6]);
      a2[2] = hi_36_1;
      a2[6] = lo_37_1;
      float hi_38_1 = fmaxf(a2[3], a2[7]);
      float lo_39_1 = fminf(a2[3], a2[7]);
      a2[3] = hi_38_1;
      a2[7] = lo_39_1;
      float hi_40_1 = fmaxf(a2[8], a2[12]);
      float lo_41_1 = fminf(a2[8], a2[12]);
      a2[8] = hi_40_1;
      a2[12] = lo_41_1;
      float hi_42_1 = fmaxf(a2[9], a2[13]);
      float lo_43_1 = fminf(a2[9], a2[13]);
      a2[9] = hi_42_1;
      a2[13] = lo_43_1;
      float hi_44_1 = fmaxf(a2[10], a2[14]);
      float lo_45_1 = fminf(a2[10], a2[14]);
      a2[10] = hi_44_1;
      a2[14] = lo_45_1;
      float hi_46_1 = fmaxf(a2[11], a2[15]);
      float lo_47_1 = fminf(a2[11], a2[15]);
      a2[11] = hi_46_1;
      a2[15] = lo_47_1;
      float hi_48_1 = fmaxf(a2[0], a2[2]);
      float lo_49_1 = fminf(a2[0], a2[2]);
      a2[0] = hi_48_1;
      a2[2] = lo_49_1;
      float hi_50_1 = fmaxf(a2[1], a2[3]);
      float lo_51_1 = fminf(a2[1], a2[3]);
      a2[1] = hi_50_1;
      a2[3] = lo_51_1;
      float hi_52_1 = fmaxf(a2[4], a2[6]);
      float lo_53_1 = fminf(a2[4], a2[6]);
      a2[4] = hi_52_1;
      a2[6] = lo_53_1;
      float hi_54_1 = fmaxf(a2[5], a2[7]);
      float lo_55_1 = fminf(a2[5], a2[7]);
      a2[5] = hi_54_1;
      a2[7] = lo_55_1;
      float hi_56_1 = fmaxf(a2[8], a2[10]);
      float lo_57_1 = fminf(a2[8], a2[10]);
      a2[8] = hi_56_1;
      a2[10] = lo_57_1;
      float hi_58_1 = fmaxf(a2[9], a2[11]);
      float lo_59_1 = fminf(a2[9], a2[11]);
      a2[9] = hi_58_1;
      a2[11] = lo_59_1;
      float hi_60_1 = fmaxf(a2[12], a2[14]);
      float lo_61_1 = fminf(a2[12], a2[14]);
      a2[12] = hi_60_1;
      a2[14] = lo_61_1;
      float hi_62_1 = fmaxf(a2[13], a2[15]);
      float lo_63_1 = fminf(a2[13], a2[15]);
      a2[13] = hi_62_1;
      a2[15] = lo_63_1;
      float hi_64_2 = fmaxf(a2[0], a2[1]);
      float lo_65_2 = fminf(a2[0], a2[1]);
      a2[0] = hi_64_2;
      a2[1] = lo_65_2;
      float hi_66_2 = fmaxf(a2[2], a2[3]);
      float lo_67_2 = fminf(a2[2], a2[3]);
      a2[2] = hi_66_2;
      a2[3] = lo_67_2;
      float hi_68_2 = fmaxf(a2[4], a2[5]);
      float lo_69_2 = fminf(a2[4], a2[5]);
      a2[4] = hi_68_2;
      a2[5] = lo_69_2;
      float hi_70_2 = fmaxf(a2[6], a2[7]);
      float lo_71_2 = fminf(a2[6], a2[7]);
      a2[6] = hi_70_2;
      a2[7] = lo_71_2;
      float hi_72_2 = fmaxf(a2[8], a2[9]);
      float lo_73_2 = fminf(a2[8], a2[9]);
      a2[8] = hi_72_2;
      a2[9] = lo_73_2;
      float hi_74_2 = fmaxf(a2[10], a2[11]);
      float lo_75_2 = fminf(a2[10], a2[11]);
      a2[10] = hi_74_2;
      a2[11] = lo_75_2;
      float hi_76_2 = fmaxf(a2[12], a2[13]);
      float lo_77_2 = fminf(a2[12], a2[13]);
      a2[12] = hi_76_2;
      a2[13] = lo_77_2;
      float hi_78_2 = fmaxf(a2[14], a2[15]);
      float lo_79_2 = fminf(a2[14], a2[15]);
      a2[14] = hi_78_2;
      a2[15] = lo_79_2;
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
    unsigned int hi_32_2 = ((o[1]) > (o[12]) ? (o[1]) : (o[12]));
    unsigned int lo_33_2 = ((o[1]) < (o[12]) ? (o[1]) : (o[12]));
    o[1] = lo_33_2;
    o[12] = hi_32_2;
    unsigned int hi_34_2 = ((o[2]) > (o[15]) ? (o[2]) : (o[15]));
    unsigned int lo_35_2 = ((o[2]) < (o[15]) ? (o[2]) : (o[15]));
    o[2] = lo_35_2;
    o[15] = hi_34_2;
    unsigned int hi_36_2 = ((o[3]) > (o[14]) ? (o[3]) : (o[14]));
    unsigned int lo_37_2 = ((o[3]) < (o[14]) ? (o[3]) : (o[14]));
    o[3] = lo_37_2;
    o[14] = hi_36_2;
    unsigned int hi_38_2 = ((o[4]) > (o[8]) ? (o[4]) : (o[8]));
    unsigned int lo_39_2 = ((o[4]) < (o[8]) ? (o[4]) : (o[8]));
    o[4] = lo_39_2;
    o[8] = hi_38_2;
    unsigned int hi_40_2 = ((o[5]) > (o[6]) ? (o[5]) : (o[6]));
    unsigned int lo_41_2 = ((o[5]) < (o[6]) ? (o[5]) : (o[6]));
    o[5] = lo_41_2;
    o[6] = hi_40_2;
    unsigned int hi_42_2 = ((o[7]) > (o[11]) ? (o[7]) : (o[11]));
    unsigned int lo_43_2 = ((o[7]) < (o[11]) ? (o[7]) : (o[11]));
    o[7] = lo_43_2;
    o[11] = hi_42_2;
    unsigned int hi_44_2 = ((o[9]) > (o[10]) ? (o[9]) : (o[10]));
    unsigned int lo_45_2 = ((o[9]) < (o[10]) ? (o[9]) : (o[10]));
    o[9] = lo_45_2;
    o[10] = hi_44_2;
    unsigned int hi_46_2 = ((o[0]) > (o[5]) ? (o[0]) : (o[5]));
    unsigned int lo_47_2 = ((o[0]) < (o[5]) ? (o[0]) : (o[5]));
    o[0] = lo_47_2;
    o[5] = hi_46_2;
    unsigned int hi_48_2 = ((o[1]) > (o[7]) ? (o[1]) : (o[7]));
    unsigned int lo_49_2 = ((o[1]) < (o[7]) ? (o[1]) : (o[7]));
    o[1] = lo_49_2;
    o[7] = hi_48_2;
    unsigned int hi_50_2 = ((o[2]) > (o[9]) ? (o[2]) : (o[9]));
    unsigned int lo_51_2 = ((o[2]) < (o[9]) ? (o[2]) : (o[9]));
    o[2] = lo_51_2;
    o[9] = hi_50_2;
    unsigned int hi_52_2 = ((o[3]) > (o[4]) ? (o[3]) : (o[4]));
    unsigned int lo_53_2 = ((o[3]) < (o[4]) ? (o[3]) : (o[4]));
    o[3] = lo_53_2;
    o[4] = hi_52_2;
    unsigned int hi_54_2 = ((o[6]) > (o[13]) ? (o[6]) : (o[13]));
    unsigned int lo_55_2 = ((o[6]) < (o[13]) ? (o[6]) : (o[13]));
    o[6] = lo_55_2;
    o[13] = hi_54_2;
    unsigned int hi_56_2 = ((o[8]) > (o[14]) ? (o[8]) : (o[14]));
    unsigned int lo_57_2 = ((o[8]) < (o[14]) ? (o[8]) : (o[14]));
    o[8] = lo_57_2;
    o[14] = hi_56_2;
    unsigned int hi_58_2 = ((o[10]) > (o[15]) ? (o[10]) : (o[15]));
    unsigned int lo_59_2 = ((o[10]) < (o[15]) ? (o[10]) : (o[15]));
    o[10] = lo_59_2;
    o[15] = hi_58_2;
    unsigned int hi_60_2 = ((o[11]) > (o[12]) ? (o[11]) : (o[12]));
    unsigned int lo_61_2 = ((o[11]) < (o[12]) ? (o[11]) : (o[12]));
    o[11] = lo_61_2;
    o[12] = hi_60_2;
    unsigned int hi_62_2 = ((o[0]) > (o[1]) ? (o[0]) : (o[1]));
    unsigned int lo_63_2 = ((o[0]) < (o[1]) ? (o[0]) : (o[1]));
    o[0] = lo_63_2;
    o[1] = hi_62_2;
    unsigned int hi_64_3 = ((o[2]) > (o[3]) ? (o[2]) : (o[3]));
    unsigned int lo_65_3 = ((o[2]) < (o[3]) ? (o[2]) : (o[3]));
    o[2] = lo_65_3;
    o[3] = hi_64_3;
    unsigned int hi_66_3 = ((o[4]) > (o[5]) ? (o[4]) : (o[5]));
    unsigned int lo_67_3 = ((o[4]) < (o[5]) ? (o[4]) : (o[5]));
    o[4] = lo_67_3;
    o[5] = hi_66_3;
    unsigned int hi_68_3 = ((o[6]) > (o[8]) ? (o[6]) : (o[8]));
    unsigned int lo_69_3 = ((o[6]) < (o[8]) ? (o[6]) : (o[8]));
    o[6] = lo_69_3;
    o[8] = hi_68_3;
    unsigned int hi_70_3 = ((o[7]) > (o[9]) ? (o[7]) : (o[9]));
    unsigned int lo_71_3 = ((o[7]) < (o[9]) ? (o[7]) : (o[9]));
    o[7] = lo_71_3;
    o[9] = hi_70_3;
    unsigned int hi_72_3 = ((o[10]) > (o[11]) ? (o[10]) : (o[11]));
    unsigned int lo_73_3 = ((o[10]) < (o[11]) ? (o[10]) : (o[11]));
    o[10] = lo_73_3;
    o[11] = hi_72_3;
    unsigned int hi_74_3 = ((o[12]) > (o[13]) ? (o[12]) : (o[13]));
    unsigned int lo_75_3 = ((o[12]) < (o[13]) ? (o[12]) : (o[13]));
    o[12] = lo_75_3;
    o[13] = hi_74_3;
    unsigned int hi_76_3 = ((o[14]) > (o[15]) ? (o[14]) : (o[15]));
    unsigned int lo_77_3 = ((o[14]) < (o[15]) ? (o[14]) : (o[15]));
    o[14] = lo_77_3;
    o[15] = hi_76_3;
    unsigned int hi_78_3 = ((o[0]) > (o[2]) ? (o[0]) : (o[2]));
    unsigned int lo_79_3 = ((o[0]) < (o[2]) ? (o[0]) : (o[2]));
    o[0] = lo_79_3;
    o[2] = hi_78_3;
    unsigned int hi_80_2 = ((o[1]) > (o[3]) ? (o[1]) : (o[3]));
    unsigned int lo_81_2 = ((o[1]) < (o[3]) ? (o[1]) : (o[3]));
    o[1] = lo_81_2;
    o[3] = hi_80_2;
    unsigned int hi_82_2 = ((o[4]) > (o[10]) ? (o[4]) : (o[10]));
    unsigned int lo_83_2 = ((o[4]) < (o[10]) ? (o[4]) : (o[10]));
    o[4] = lo_83_2;
    o[10] = hi_82_2;
    unsigned int hi_84_2 = ((o[5]) > (o[11]) ? (o[5]) : (o[11]));
    unsigned int lo_85_2 = ((o[5]) < (o[11]) ? (o[5]) : (o[11]));
    o[5] = lo_85_2;
    o[11] = hi_84_2;
    unsigned int hi_86_2 = ((o[6]) > (o[7]) ? (o[6]) : (o[7]));
    unsigned int lo_87_2 = ((o[6]) < (o[7]) ? (o[6]) : (o[7]));
    o[6] = lo_87_2;
    o[7] = hi_86_2;
    unsigned int hi_88_2 = ((o[8]) > (o[9]) ? (o[8]) : (o[9]));
    unsigned int lo_89_2 = ((o[8]) < (o[9]) ? (o[8]) : (o[9]));
    o[8] = lo_89_2;
    o[9] = hi_88_2;
    unsigned int hi_90_2 = ((o[12]) > (o[14]) ? (o[12]) : (o[14]));
    unsigned int lo_91_2 = ((o[12]) < (o[14]) ? (o[12]) : (o[14]));
    o[12] = lo_91_2;
    o[14] = hi_90_2;
    unsigned int hi_92_2 = ((o[13]) > (o[15]) ? (o[13]) : (o[15]));
    unsigned int lo_93_2 = ((o[13]) < (o[15]) ? (o[13]) : (o[15]));
    o[13] = lo_93_2;
    o[15] = hi_92_2;
    unsigned int hi_94_2 = ((o[1]) > (o[2]) ? (o[1]) : (o[2]));
    unsigned int lo_95_2 = ((o[1]) < (o[2]) ? (o[1]) : (o[2]));
    o[1] = lo_95_2;
    o[2] = hi_94_2;
    unsigned int hi_96_1 = ((o[3]) > (o[12]) ? (o[3]) : (o[12]));
    unsigned int lo_97_1 = ((o[3]) < (o[12]) ? (o[3]) : (o[12]));
    o[3] = lo_97_1;
    o[12] = hi_96_1;
    unsigned int hi_98_1 = ((o[4]) > (o[6]) ? (o[4]) : (o[6]));
    unsigned int lo_99_1 = ((o[4]) < (o[6]) ? (o[4]) : (o[6]));
    o[4] = lo_99_1;
    o[6] = hi_98_1;
    unsigned int hi_100_1 = ((o[5]) > (o[7]) ? (o[5]) : (o[7]));
    unsigned int lo_101_1 = ((o[5]) < (o[7]) ? (o[5]) : (o[7]));
    o[5] = lo_101_1;
    o[7] = hi_100_1;
    unsigned int hi_102_1 = ((o[8]) > (o[10]) ? (o[8]) : (o[10]));
    unsigned int lo_103_1 = ((o[8]) < (o[10]) ? (o[8]) : (o[10]));
    o[8] = lo_103_1;
    o[10] = hi_102_1;
    unsigned int hi_104_1 = ((o[9]) > (o[11]) ? (o[9]) : (o[11]));
    unsigned int lo_105_1 = ((o[9]) < (o[11]) ? (o[9]) : (o[11]));
    o[9] = lo_105_1;
    o[11] = hi_104_1;
    unsigned int hi_106_1 = ((o[13]) > (o[14]) ? (o[13]) : (o[14]));
    unsigned int lo_107_1 = ((o[13]) < (o[14]) ? (o[13]) : (o[14]));
    o[13] = lo_107_1;
    o[14] = hi_106_1;
    unsigned int hi_108_1 = ((o[1]) > (o[4]) ? (o[1]) : (o[4]));
    unsigned int lo_109_1 = ((o[1]) < (o[4]) ? (o[1]) : (o[4]));
    o[1] = lo_109_1;
    o[4] = hi_108_1;
    unsigned int hi_110_1 = ((o[2]) > (o[6]) ? (o[2]) : (o[6]));
    unsigned int lo_111_1 = ((o[2]) < (o[6]) ? (o[2]) : (o[6]));
    o[2] = lo_111_1;
    o[6] = hi_110_1;
    unsigned int hi_112_1 = ((o[5]) > (o[8]) ? (o[5]) : (o[8]));
    unsigned int lo_113_1 = ((o[5]) < (o[8]) ? (o[5]) : (o[8]));
    o[5] = lo_113_1;
    o[8] = hi_112_1;
    unsigned int hi_114_1 = ((o[7]) > (o[10]) ? (o[7]) : (o[10]));
    unsigned int lo_115_1 = ((o[7]) < (o[10]) ? (o[7]) : (o[10]));
    o[7] = lo_115_1;
    o[10] = hi_114_1;
    unsigned int hi_116_1 = ((o[9]) > (o[13]) ? (o[9]) : (o[13]));
    unsigned int lo_117_1 = ((o[9]) < (o[13]) ? (o[9]) : (o[13]));
    o[9] = lo_117_1;
    o[13] = hi_116_1;
    unsigned int hi_118_1 = ((o[11]) > (o[14]) ? (o[11]) : (o[14]));
    unsigned int lo_119_1 = ((o[11]) < (o[14]) ? (o[11]) : (o[14]));
    o[11] = lo_119_1;
    o[14] = hi_118_1;
    unsigned int hi_120_1 = ((o[2]) > (o[4]) ? (o[2]) : (o[4]));
    unsigned int lo_121_1 = ((o[2]) < (o[4]) ? (o[2]) : (o[4]));
    o[2] = lo_121_1;
    o[4] = hi_120_1;
    unsigned int hi_122_1 = ((o[3]) > (o[6]) ? (o[3]) : (o[6]));
    unsigned int lo_123_1 = ((o[3]) < (o[6]) ? (o[3]) : (o[6]));
    o[3] = lo_123_1;
    o[6] = hi_122_1;
    unsigned int hi_124_1 = ((o[9]) > (o[12]) ? (o[9]) : (o[12]));
    unsigned int lo_125_1 = ((o[9]) < (o[12]) ? (o[9]) : (o[12]));
    o[9] = lo_125_1;
    o[12] = hi_124_1;
    unsigned int hi_126_1 = ((o[11]) > (o[13]) ? (o[11]) : (o[13]));
    unsigned int lo_127_1 = ((o[11]) < (o[13]) ? (o[11]) : (o[13]));
    o[11] = lo_127_1;
    o[13] = hi_126_1;
    unsigned int hi_128_1 = ((o[3]) > (o[5]) ? (o[3]) : (o[5]));
    unsigned int lo_129_1 = ((o[3]) < (o[5]) ? (o[3]) : (o[5]));
    o[3] = lo_129_1;
    o[5] = hi_128_1;
    unsigned int hi_130_1 = ((o[6]) > (o[8]) ? (o[6]) : (o[8]));
    unsigned int lo_131_1 = ((o[6]) < (o[8]) ? (o[6]) : (o[8]));
    o[6] = lo_131_1;
    o[8] = hi_130_1;
    unsigned int hi_132_1 = ((o[7]) > (o[9]) ? (o[7]) : (o[9]));
    unsigned int lo_133_1 = ((o[7]) < (o[9]) ? (o[7]) : (o[9]));
    o[7] = lo_133_1;
    o[9] = hi_132_1;
    unsigned int hi_134_1 = ((o[10]) > (o[12]) ? (o[10]) : (o[12]));
    unsigned int lo_135_1 = ((o[10]) < (o[12]) ? (o[10]) : (o[12]));
    o[10] = lo_135_1;
    o[12] = hi_134_1;
    unsigned int hi_136_1 = ((o[3]) > (o[4]) ? (o[3]) : (o[4]));
    unsigned int lo_137_1 = ((o[3]) < (o[4]) ? (o[3]) : (o[4]));
    o[3] = lo_137_1;
    o[4] = hi_136_1;
    unsigned int hi_138_1 = ((o[5]) > (o[6]) ? (o[5]) : (o[6]));
    unsigned int lo_139_1 = ((o[5]) < (o[6]) ? (o[5]) : (o[6]));
    o[5] = lo_139_1;
    o[6] = hi_138_1;
    unsigned int hi_140_1 = ((o[7]) > (o[8]) ? (o[7]) : (o[8]));
    unsigned int lo_141_1 = ((o[7]) < (o[8]) ? (o[7]) : (o[8]));
    o[7] = lo_141_1;
    o[8] = hi_140_1;
    unsigned int hi_142_1 = ((o[9]) > (o[10]) ? (o[9]) : (o[10]));
    unsigned int lo_143_1 = ((o[9]) < (o[10]) ? (o[9]) : (o[10]));
    o[9] = lo_143_1;
    o[10] = hi_142_1;
    unsigned int hi_144_1 = ((o[11]) > (o[12]) ? (o[11]) : (o[12]));
    unsigned int lo_145_1 = ((o[11]) < (o[12]) ? (o[11]) : (o[12]));
    o[11] = lo_145_1;
    o[12] = hi_144_1;
    unsigned int hi_146_1 = ((o[6]) > (o[7]) ? (o[6]) : (o[7]));
    unsigned int lo_147_1 = ((o[6]) < (o[7]) ? (o[6]) : (o[7]));
    o[6] = lo_147_1;
    o[7] = hi_146_1;
    unsigned int hi_148_1 = ((o[8]) > (o[9]) ? (o[8]) : (o[9]));
    unsigned int lo_149_1 = ((o[8]) < (o[9]) ? (o[8]) : (o[9]));
    o[8] = lo_149_1;
    o[9] = hi_148_1;
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
