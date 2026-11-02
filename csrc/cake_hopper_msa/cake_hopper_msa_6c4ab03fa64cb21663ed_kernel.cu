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

#include "cake_hopper_msa_device.h"

// Shared by every record that names this translation unit in its sources: the record's
// compile_flags set the knob values below (a knob without a -D flag is a compile error) and name
// the kernel symbol (-DCAKE_PROGRAM_KERNEL=<its own symbol>; every JIT library of the family
// exports a distinct kernel).
//   CAKE_GROUP  query heads per KV head (the route's g<G>); selects the literal table below
#ifndef CAKE_GROUP
#error "CAKE_GROUP must come from the record's compile_flags"
#endif
#ifndef CAKE_PROGRAM_KERNEL
#error "CAKE_PROGRAM_KERNEL (the program's kernel symbol) must come from the record's compile_flags"
#endif
#define NUM_MAIN_STAGES 1
#define SMEM_KV8_OFF 1024
#define SMEM_KV8_STAGE_BYTES 32768
#define SMEM_KV8_STRIDE 32768
#define SMEM_IDENT8_OFF 66560
#define SMEM_IDENT8_STAGE_BYTES 8192
#define SMEM_IDENT8_STRIDE 8192
#define SMEM_Q16_OFF 74752
#define SMEM_Q16_STAGE_BYTES 8192
#define SMEM_Q16_STRIDE 8192
#define SMEM_P16_OFF 82944
#define SMEM_P16_STAGE_BYTES 4096
#define SMEM_P16_STRIDE 4096
#define SMEM_MARK_OFF 82944
#define SMEM_MARK_STAGE_BYTES 4096
#define SMEM_MARK_STRIDE 4096
#define SMEM_RED_OFF 91136
#define SMEM_RED_STAGE_BYTES 512
#define SMEM_RED_STRIDE 512
#define SMEM_LRED_OFF 91648
#define SMEM_LRED_STAGE_BYTES 512
#define SMEM_LRED_STRIDE 512
#define SMEM_PREF_OFF 92160
#define SMEM_PREF_STAGE_BYTES 512
#define SMEM_PREF_STRIDE 512
#define SMEM_CNT_OFF 92672
#define SMEM_CNT_STAGE_BYTES 32
#define SMEM_CNT_STRIDE 32
#define SMEM_HDR_OFF 92704
#define SMEM_HDR_STAGE_BYTES 32
#define SMEM_HDR_STRIDE 32
#define SMEM_META_PG_OFF 92736
#if CAKE_GROUP == 16
#define CAKE_TILE_TOKENS 2
#define CAKE_LOG2_GROUP 4
#define CAKE_META_ENTRIES 32
#define SMEM_META_PG_STAGE_BYTES 128
#define SMEM_META_PG_STRIDE 128
#define SMEM_META_BLK_OFF 92864
#define SMEM_META_BLK_STAGE_BYTES 128
#define SMEM_META_BLK_STRIDE 128
#define SMEM_META_MASK_OFF 92992
#define SMEM_META_MASK_STAGE_BYTES 128
#define SMEM_META_MASK_STRIDE 128
#define SMEM_TOTAL 93184
#elif CAKE_GROUP == 8
#define CAKE_TILE_TOKENS 4
#define CAKE_LOG2_GROUP 3
#define CAKE_META_ENTRIES 64
#define SMEM_META_PG_STAGE_BYTES 256
#define SMEM_META_PG_STRIDE 256
#define SMEM_META_BLK_OFF 92992
#define SMEM_META_BLK_STAGE_BYTES 256
#define SMEM_META_BLK_STRIDE 256
#define SMEM_META_MASK_OFF 93248
#define SMEM_META_MASK_STAGE_BYTES 256
#define SMEM_META_MASK_STRIDE 256
#define SMEM_TOTAL 93568
#elif CAKE_GROUP == 4
#define CAKE_TILE_TOKENS 8
#define CAKE_LOG2_GROUP 2
#define CAKE_META_ENTRIES 128
#define SMEM_META_PG_STAGE_BYTES 512
#define SMEM_META_PG_STRIDE 512
#define SMEM_META_BLK_OFF 93248
#define SMEM_META_BLK_STAGE_BYTES 512
#define SMEM_META_BLK_STRIDE 512
#define SMEM_META_MASK_OFF 93760
#define SMEM_META_MASK_STAGE_BYTES 512
#define SMEM_META_MASK_STRIDE 512
#define SMEM_TOTAL 94336
#else
#error "unsupported CAKE_GROUP"
#endif
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 2

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void CAKE_PROGRAM_KERNEL(
    __nv_bfloat16* __restrict__ Q, const __grid_constant__ CUtensorMap K,
    const __grid_constant__ CUtensorMap V, __nv_bfloat16* __restrict__ O,
    int* __restrict__ q2k_indices, int* __restrict__ cu_seqlens_q, int* __restrict__ page_table,
    int* __restrict__ seqused_k, int* __restrict__ q_offset, int total_q, int batch,
    int num_q_heads, int num_kv_heads, int max_pages, int use_q_offset, float softmax_scale_log2,
    unsigned int zero_u32) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

  const int mbar_base = smem;
#define kv_full_addr (mbar_base + 0)

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  const int cta_rank = 0;

  // Kernel setup ops
  uint8_t* kv8 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
  const int kv8_addr = smem + 1024;
  uint8_t* ident8 = reinterpret_cast<uint8_t*>(smem_raw + 66560);
  const int ident8_addr = smem + 66560;
  __half* q16 = reinterpret_cast<__half*>(smem_raw + 74752);
  const int q16_addr = smem + 74752;
  __half* p16 = reinterpret_cast<__half*>(smem_raw + 82944);
  const int p16_addr = smem + 82944;
  uint8_t* mark = reinterpret_cast<uint8_t*>(smem_raw + 82944);
  const int mark_addr = smem + 82944;
  float* red = reinterpret_cast<float*>(smem_raw + 91136);
  const int red_addr = smem + 91136;
  float* lred = reinterpret_cast<float*>(smem_raw + 91648);
  const int lred_addr = smem + 91648;
  int* pref = reinterpret_cast<int*>(smem_raw + 92160);
  const int pref_addr = smem + 92160;
  int* cnt = reinterpret_cast<int*>(smem_raw + 92672);
  const int cnt_addr = smem + 92672;
  int* hdr = reinterpret_cast<int*>(smem_raw + 92704);
  const int hdr_addr = smem + 92704;
  int* meta_pg = reinterpret_cast<int*>(smem_raw + 92736);
  const int meta_pg_addr = smem + 92736;
  int* meta_blk = reinterpret_cast<int*>(smem_raw + SMEM_META_BLK_OFF);
  const int meta_blk_addr = smem + SMEM_META_BLK_OFF;
  unsigned int* meta_mask = reinterpret_cast<unsigned int*>(smem_raw + SMEM_META_MASK_OFF);
  const int meta_mask_addr = smem + SMEM_META_MASK_OFF;

  // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 2 barriers)
  // Mbarriers at smem_raw[0..16)

  if (warp == 0) {
    uint32_t leader = elect_sync();
    if (leader) {
      // kv_full: 2 barriers, init_count=1
      mbarrier_init(smem + 0, 1);
      mbarrier_init(smem + 8, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
  }

  __syncthreads();

  // === Task calls (dependency order) ===
  if (warp == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)((&K))) : "memory");
  }
  if (warp == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)((&V))) : "memory");
  }
  int gi = blockIdx.x;
  int kv_head = blockIdx.y;
  int tid_1 = threadIdx.x;
  int lane_0 = lane;
  int warp_1 = warp;
  int g_ = lane_0 / 4;
  int cq = lane_0 - g_ * 4;
  int key0 = warp_1 * 16 + g_;
  int key1 = key0 + 8;
  if (warp_1 == 0) {
    int found = 0;
    int base_g = 0;
    int sel_b = 0;
    int sel_tok0 = 0;
    int sel_ntok = 0;
    int sel_cu = 0;
    int sel_qlen = 0;
    int nchunks = (batch + 31) / 32;
#pragma unroll 1
    for (int c = 0; c < nchunks; c++) {
      if (found == 0) {
        int bb = c * 32 + lane_0;
        int cnt_b = 0;
        int cu_lo = 0;
        int cu_hi = 0;
        if (bb < batch) {
          cu_lo = cu_seqlens_q[bb];
          cu_hi = cu_seqlens_q[bb + 1];
          cnt_b = (cu_hi - cu_lo + CAKE_TILE_TOKENS - 1) / CAKE_TILE_TOKENS;
        }
        int incl = cnt_b;
        int up = __shfl_up_sync(0xFFFFFFFF, incl, 1, 32);
        if (lane_0 >= 1) {
          incl = incl + up;
        }
        int up_0 = __shfl_up_sync(0xFFFFFFFF, incl, 2, 32);
        if (lane_0 >= 2) {
          incl = incl + up_0;
        }
        int up_1 = __shfl_up_sync(0xFFFFFFFF, incl, 4, 32);
        if (lane_0 >= 4) {
          incl = incl + up_1;
        }
        int up_2 = __shfl_up_sync(0xFFFFFFFF, incl, 8, 32);
        if (lane_0 >= 8) {
          incl = incl + up_2;
        }
        int up_3 = __shfl_up_sync(0xFFFFFFFF, incl, 16, 32);
        if (lane_0 >= 16) {
          incl = incl + up_3;
        }
        int excl = incl - cnt_b;
        int hit = 0;
        if (bb < batch) {
          if (gi >= base_g + excl) {
            if (gi < base_g + incl) {
              hit = 1;
            }
          }
        }
        unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, hit != 0);
        if (_vote_0 != 0) {
          int _find_first_set_0 = __ffs(_vote_0);
          int src_l = _find_first_set_0 - 1;
          sel_b = c * 32 + src_l;
          int _shfl_sync_1 = __shfl_sync(0xFFFFFFFF, cu_lo, src_l);
          sel_cu = _shfl_sync_1;
          int hi_s = __shfl_sync(0xFFFFFFFF, cu_hi, src_l);
          int ex_s = __shfl_sync(0xFFFFFFFF, excl, src_l);
          int cnt_s = __shfl_sync(0xFFFFFFFF, cnt_b, src_l);
          int lg = gi - base_g - ex_s;
          lg = cnt_s - 1 - lg;
          sel_tok0 = lg * CAKE_TILE_TOKENS;
          sel_qlen = hi_s - sel_cu;
          sel_ntok = sel_qlen - sel_tok0;
          if (sel_ntok > CAKE_TILE_TOKENS) {
            sel_ntok = CAKE_TILE_TOKENS;
          }
          found = 1;
        }
        int _shfl_sync_2 = __shfl_sync(0xFFFFFFFF, incl, 31);
        base_g = base_g + _shfl_sync_2;
      }
    }
    if (lane_0 == 0) {
      int qoff = 0;
      int sk_sel = seqused_k[sel_b];
      if (found != 0) {
        if (use_q_offset != 0) {
          qoff = q_offset[sel_b];
        }
        if (use_q_offset == 0) {
          qoff = sk_sel - sel_qlen;
        }
      }
      hdr[0] = sel_b;
      hdr[1] = sel_tok0;
      hdr[2] = sel_ntok;
      hdr[3] = sel_cu;
      hdr[4] = qoff + sel_tok0;
      hdr[5] = sk_sel;
    }
  }
  if (warp_1 != 0) {
    int sid = tid_1 - 32;
    int ngrp_z = (max_pages + 31) / 32;
    unsigned int zz[4] = {0};
    int grp_z = sid;
    if (grp_z < ngrp_z) {
      asm volatile(
          "st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(mark_addr + (unsigned int)(grp_z * 32)),
          "r"(zz[0]), "r"(zz[1]), "r"(zz[2]), "r"(zz[3])
          : "memory");
      asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(mark_addr +
                                                                 (unsigned int)(grp_z * 32 + 16)),
                   "r"(zz[0]), "r"(zz[1]), "r"(zz[2]), "r"(zz[3])
                   : "memory");
    }
    int grp_z_0 = sid + 96;
    if (grp_z_0 < ngrp_z) {
      asm volatile(
          "st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(mark_addr + (unsigned int)(grp_z_0 * 32)),
          "r"(zz[0]), "r"(zz[1]), "r"(zz[2]), "r"(zz[3])
          : "memory");
      asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(mark_addr +
                                                                 (unsigned int)(grp_z_0 * 32 + 16)),
                   "r"(zz[0]), "r"(zz[1]), "r"(zz[2]), "r"(zz[3])
                   : "memory");
    }
    int ich_lin = sid;
    if (ich_lin < 512) {
      int irow = ich_lin / 8;
      int ich = ich_lin - irow * 8;
      int iword = irow / 4;
      unsigned int ione = 56 << (unsigned int)(8 * (irow & 3));
      unsigned int wv[4] = {0};
      if (ich * 4 == iword) {
        wv[0] = ione;
      }
      if (ich * 4 + 1 == iword) {
        wv[1] = ione;
      }
      if (ich * 4 + 2 == iword) {
        wv[2] = ione;
      }
      if (ich * 4 + 3 == iword) {
        wv[3] = ione;
      }
      asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
                       ident8_addr + (unsigned int)(irow * 128 + (ich * 16 ^ (irow & 7) << 4))),
                   "r"(wv[0]), "r"(wv[1]), "r"(wv[2]), "r"(wv[3])
                   : "memory");
    }
    int ich_lin_1 = sid + 96;
    if (ich_lin_1 < 512) {
      int irow_1 = ich_lin_1 / 8;
      int ich_1 = ich_lin_1 - irow_1 * 8;
      int iword_1 = irow_1 / 4;
      unsigned int ione_1 = 56 << (unsigned int)(8 * (irow_1 & 3));
      unsigned int wv_1[4] = {0};
      if (ich_1 * 4 == iword_1) {
        wv_1[0] = ione_1;
      }
      if (ich_1 * 4 + 1 == iword_1) {
        wv_1[1] = ione_1;
      }
      if (ich_1 * 4 + 2 == iword_1) {
        wv_1[2] = ione_1;
      }
      if (ich_1 * 4 + 3 == iword_1) {
        wv_1[3] = ione_1;
      }
      asm volatile(
          "st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
              ident8_addr + (unsigned int)(irow_1 * 128 + (ich_1 * 16 ^ (irow_1 & 7) << 4))),
          "r"(wv_1[0]), "r"(wv_1[1]), "r"(wv_1[2]), "r"(wv_1[3])
          : "memory");
    }
    int ich_lin_2 = sid + 192;
    if (ich_lin_2 < 512) {
      int irow_2 = ich_lin_2 / 8;
      int ich_2 = ich_lin_2 - irow_2 * 8;
      int iword_2 = irow_2 / 4;
      unsigned int ione_2 = 56 << (unsigned int)(8 * (irow_2 & 3));
      unsigned int wv_2[4] = {0};
      if (ich_2 * 4 == iword_2) {
        wv_2[0] = ione_2;
      }
      if (ich_2 * 4 + 1 == iword_2) {
        wv_2[1] = ione_2;
      }
      if (ich_2 * 4 + 2 == iword_2) {
        wv_2[2] = ione_2;
      }
      if (ich_2 * 4 + 3 == iword_2) {
        wv_2[3] = ione_2;
      }
      asm volatile(
          "st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
              ident8_addr + (unsigned int)(irow_2 * 128 + (ich_2 * 16 ^ (irow_2 & 7) << 4))),
          "r"(wv_2[0]), "r"(wv_2[1]), "r"(wv_2[2]), "r"(wv_2[3])
          : "memory");
    }
    int ich_lin_3 = sid + 288;
    if (ich_lin_3 < 512) {
      int irow_3 = ich_lin_3 / 8;
      int ich_3 = ich_lin_3 - irow_3 * 8;
      int iword_3 = irow_3 / 4;
      unsigned int ione_3 = 56 << (unsigned int)(8 * (irow_3 & 3));
      unsigned int wv_3[4] = {0};
      if (ich_3 * 4 == iword_3) {
        wv_3[0] = ione_3;
      }
      if (ich_3 * 4 + 1 == iword_3) {
        wv_3[1] = ione_3;
      }
      if (ich_3 * 4 + 2 == iword_3) {
        wv_3[2] = ione_3;
      }
      if (ich_3 * 4 + 3 == iword_3) {
        wv_3[3] = ione_3;
      }
      asm volatile(
          "st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
              ident8_addr + (unsigned int)(irow_3 * 128 + (ich_3 * 16 ^ (irow_3 & 7) << 4))),
          "r"(wv_3[0]), "r"(wv_3[1]), "r"(wv_3[2]), "r"(wv_3[3])
          : "memory");
    }
    int ich_lin_4 = sid + 384;
    if (ich_lin_4 < 512) {
      int irow_4 = ich_lin_4 / 8;
      int ich_4 = ich_lin_4 - irow_4 * 8;
      int iword_4 = irow_4 / 4;
      unsigned int ione_4 = 56 << (unsigned int)(8 * (irow_4 & 3));
      unsigned int wv_4[4] = {0};
      if (ich_4 * 4 == iword_4) {
        wv_4[0] = ione_4;
      }
      if (ich_4 * 4 + 1 == iword_4) {
        wv_4[1] = ione_4;
      }
      if (ich_4 * 4 + 2 == iword_4) {
        wv_4[2] = ione_4;
      }
      if (ich_4 * 4 + 3 == iword_4) {
        wv_4[3] = ione_4;
      }
      asm volatile(
          "st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
              ident8_addr + (unsigned int)(irow_4 * 128 + (ich_4 * 16 ^ (irow_4 & 7) << 4))),
          "r"(wv_4[0]), "r"(wv_4[1]), "r"(wv_4[2]), "r"(wv_4[3])
          : "memory");
    }
    int ich_lin_5 = sid + 480;
    if (ich_lin_5 < 512) {
      int irow_5 = ich_lin_5 / 8;
      int ich_5 = ich_lin_5 - irow_5 * 8;
      int iword_5 = irow_5 / 4;
      unsigned int ione_5 = 56 << (unsigned int)(8 * (irow_5 & 3));
      unsigned int wv_5[4] = {0};
      if (ich_5 * 4 == iword_5) {
        wv_5[0] = ione_5;
      }
      if (ich_5 * 4 + 1 == iword_5) {
        wv_5[1] = ione_5;
      }
      if (ich_5 * 4 + 2 == iword_5) {
        wv_5[2] = ione_5;
      }
      if (ich_5 * 4 + 3 == iword_5) {
        wv_5[3] = ione_5;
      }
      asm volatile(
          "st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
              ident8_addr + (unsigned int)(irow_5 * 128 + (ich_5 * 16 ^ (irow_5 & 7) << 4))),
          "r"(wv_5[0]), "r"(wv_5[1]), "r"(wv_5[2]), "r"(wv_5[3])
          : "memory");
    }
  }
  asm volatile("barrier.sync 1, 128;" ::: "memory");
  int b_ = hdr[0];
  int tok0 = hdr[1];
  int n_tok = hdr[2];
  int cu_b = hdr[3];
  int qpos0 = hdr[4];
  int sk_b = hdr[5];
  int blk_e[1];
  int e = tid_1;
  blk_e[0] = -1;
  if (e < CAKE_META_ENTRIES) {
    int tl_e = e / 16;
    int k_e = e - tl_e * 16;
    if (tl_e < n_tok) {
      int bv = q2k_indices[(kv_head * total_q + cu_b + tok0 + tl_e) * 16 + k_e];
      if (bv >= 0) {
        if (bv < max_pages) {
          blk_e[0] = bv;
        }
      }
    }
  }
  if (blk_e[0] >= 0) {
    {
      uint32_t _byte_3 = static_cast<uint32_t>(1) & 0xFFu;
      uint32_t _addr_3 = static_cast<uint32_t>(mark_addr + (unsigned int)blk_e[0]);
      asm volatile("st.shared.u8 [%0], %1;" ::"r"(_addr_3), "r"(_byte_3) : "memory");
    }
  }
  float qf[32] = {0};
  int vid = tid_1;
  int qn = vid / 16;
  int qv = vid - qn * 16;
  int tl_q = qn >> CAKE_LOG2_GROUP;
  if (tl_q < n_tok) {
    int hg_q = qn - (tl_q << CAKE_LOG2_GROUP);
    int qrow = (cu_b + tok0 + tl_q) * num_q_heads + kv_head * CAKE_GROUP + hg_q;
    long long qbase = (long long)qrow * 128 + (long long)(qv * 8);
    {
      const uint4* _vptr_4 = reinterpret_cast<const uint4*>(Q + qbase);
      uint4 _vld_4[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_4[_blk] = _vptr_4[_blk];
        uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&qf[0 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
          (&qf[0 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
        }
      }
    }
  }
  int vid_2 = tid_1 + 128;
  int qn_3 = vid_2 / 16;
  int qv_4 = vid_2 - qn_3 * 16;
  int tl_q_5 = qn_3 >> CAKE_LOG2_GROUP;
  if (tl_q_5 < n_tok) {
    int hg_q_1 = qn_3 - (tl_q_5 << CAKE_LOG2_GROUP);
    int qrow_1 = (cu_b + tok0 + tl_q_5) * num_q_heads + kv_head * CAKE_GROUP + hg_q_1;
    long long qbase_1 = (long long)qrow_1 * 128 + (long long)(qv_4 * 8);
    {
      const uint4* _vptr_5 = reinterpret_cast<const uint4*>(Q + qbase_1);
      uint4 _vld_5[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_5[_blk] = _vptr_5[_blk];
        uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&qf[8 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
          (&qf[8 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
        }
      }
    }
  }
  int vid_6 = tid_1 + 256;
  int qn_7 = vid_6 / 16;
  int qv_8 = vid_6 - qn_7 * 16;
  int tl_q_9 = qn_7 >> CAKE_LOG2_GROUP;
  if (tl_q_9 < n_tok) {
    int hg_q_2 = qn_7 - (tl_q_9 << CAKE_LOG2_GROUP);
    int qrow_2 = (cu_b + tok0 + tl_q_9) * num_q_heads + kv_head * CAKE_GROUP + hg_q_2;
    long long qbase_2 = (long long)qrow_2 * 128 + (long long)(qv_8 * 8);
    {
      const uint4* _vptr_6 = reinterpret_cast<const uint4*>(Q + qbase_2);
      uint4 _vld_6[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_6[_blk] = _vptr_6[_blk];
        uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&qf[16 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
          (&qf[16 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
        }
      }
    }
  }
  int vid_10 = tid_1 + 384;
  int qn_11 = vid_10 / 16;
  int qv_12 = vid_10 - qn_11 * 16;
  int tl_q_13 = qn_11 >> CAKE_LOG2_GROUP;
  if (tl_q_13 < n_tok) {
    int hg_q_3 = qn_11 - (tl_q_13 << CAKE_LOG2_GROUP);
    int qrow_3 = (cu_b + tok0 + tl_q_13) * num_q_heads + kv_head * CAKE_GROUP + hg_q_3;
    long long qbase_3 = (long long)qrow_3 * 128 + (long long)(qv_12 * 8);
    {
      const uint4* _vptr_7 = reinterpret_cast<const uint4*>(Q + qbase_3);
      uint4 _vld_7[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_7[_blk] = _vptr_7[_blk];
        uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&qf[24 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
          (&qf[24 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
        }
      }
    }
  }
  int vid2 = tid_1;
  int qn2 = vid2 / 16;
  int qv2 = vid2 - qn2 * 16;
  int qslab = qv2 / 8;
  int qcol = (qv2 - qslab * 8) * 16;
  int qoff2 = qslab * 4096 + qn2 * 128 + (qcol ^ (qn2 & 7) << 4);
  unsigned int qw[4];
  uint32_t _cvt_rn_f16x2_f32_8;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_8) : "f"(qf[1]), "f"(qf[0]));
  qw[0] = _cvt_rn_f16x2_f32_8;
  uint32_t _cvt_rn_f16x2_f32_9;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_9) : "f"(qf[3]), "f"(qf[2]));
  qw[1] = _cvt_rn_f16x2_f32_9;
  uint32_t _cvt_rn_f16x2_f32_10;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_10) : "f"(qf[5]), "f"(qf[4]));
  qw[2] = _cvt_rn_f16x2_f32_10;
  uint32_t _cvt_rn_f16x2_f32_11;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_11) : "f"(qf[7]), "f"(qf[6]));
  qw[3] = _cvt_rn_f16x2_f32_11;
  asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(q16_addr + (unsigned int)qoff2),
               "r"(qw[0]), "r"(qw[1]), "r"(qw[2]), "r"(qw[3])
               : "memory");
  int vid2_14 = tid_1 + 128;
  int qn2_15 = vid2_14 / 16;
  int qv2_16 = vid2_14 - qn2_15 * 16;
  int qslab_17 = qv2_16 / 8;
  int qcol_18 = (qv2_16 - qslab_17 * 8) * 16;
  int qoff2_19 = qslab_17 * 4096 + qn2_15 * 128 + (qcol_18 ^ (qn2_15 & 7) << 4);
  unsigned int qw_20[4];
  uint32_t _cvt_rn_f16x2_f32_12;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_12) : "f"(qf[9]), "f"(qf[8]));
  qw_20[0] = _cvt_rn_f16x2_f32_12;
  uint32_t _cvt_rn_f16x2_f32_13;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_13) : "f"(qf[11]), "f"(qf[10]));
  qw_20[1] = _cvt_rn_f16x2_f32_13;
  uint32_t _cvt_rn_f16x2_f32_14;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_14) : "f"(qf[13]), "f"(qf[12]));
  qw_20[2] = _cvt_rn_f16x2_f32_14;
  uint32_t _cvt_rn_f16x2_f32_15;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_15) : "f"(qf[15]), "f"(qf[14]));
  qw_20[3] = _cvt_rn_f16x2_f32_15;
  asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(q16_addr + (unsigned int)qoff2_19),
               "r"(qw_20[0]), "r"(qw_20[1]), "r"(qw_20[2]), "r"(qw_20[3])
               : "memory");
  int vid2_21 = tid_1 + 256;
  int qn2_22 = vid2_21 / 16;
  int qv2_23 = vid2_21 - qn2_22 * 16;
  int qslab_24 = qv2_23 / 8;
  int qcol_25 = (qv2_23 - qslab_24 * 8) * 16;
  int qoff2_26 = qslab_24 * 4096 + qn2_22 * 128 + (qcol_25 ^ (qn2_22 & 7) << 4);
  unsigned int qw_27[4];
  uint32_t _cvt_rn_f16x2_f32_16;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_16) : "f"(qf[17]), "f"(qf[16]));
  qw_27[0] = _cvt_rn_f16x2_f32_16;
  uint32_t _cvt_rn_f16x2_f32_17;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_17) : "f"(qf[19]), "f"(qf[18]));
  qw_27[1] = _cvt_rn_f16x2_f32_17;
  uint32_t _cvt_rn_f16x2_f32_18;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_18) : "f"(qf[21]), "f"(qf[20]));
  qw_27[2] = _cvt_rn_f16x2_f32_18;
  uint32_t _cvt_rn_f16x2_f32_19;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_19) : "f"(qf[23]), "f"(qf[22]));
  qw_27[3] = _cvt_rn_f16x2_f32_19;
  asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(q16_addr + (unsigned int)qoff2_26),
               "r"(qw_27[0]), "r"(qw_27[1]), "r"(qw_27[2]), "r"(qw_27[3])
               : "memory");
  int vid2_28 = tid_1 + 384;
  int qn2_29 = vid2_28 / 16;
  int qv2_30 = vid2_28 - qn2_29 * 16;
  int qslab_31 = qv2_30 / 8;
  int qcol_32 = (qv2_30 - qslab_31 * 8) * 16;
  int qoff2_33 = qslab_31 * 4096 + qn2_29 * 128 + (qcol_32 ^ (qn2_29 & 7) << 4);
  unsigned int qw_34[4];
  uint32_t _cvt_rn_f16x2_f32_20;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_20) : "f"(qf[25]), "f"(qf[24]));
  qw_34[0] = _cvt_rn_f16x2_f32_20;
  uint32_t _cvt_rn_f16x2_f32_21;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_21) : "f"(qf[27]), "f"(qf[26]));
  qw_34[1] = _cvt_rn_f16x2_f32_21;
  uint32_t _cvt_rn_f16x2_f32_22;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_22) : "f"(qf[29]), "f"(qf[28]));
  qw_34[2] = _cvt_rn_f16x2_f32_22;
  uint32_t _cvt_rn_f16x2_f32_23;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_23) : "f"(qf[31]), "f"(qf[30]));
  qw_34[3] = _cvt_rn_f16x2_f32_23;
  asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(q16_addr + (unsigned int)qoff2_33),
               "r"(qw_34[0]), "r"(qw_34[1]), "r"(qw_34[2]), "r"(qw_34[3])
               : "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  asm volatile("barrier.sync 1, 128;" ::: "memory");
  int ngrp = (max_pages + 31) / 32;
  unsigned int mw[8] = {0};
  if (tid_1 < ngrp) {
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&mw[0])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mw[(0) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mw[(0) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mw[(0) + 3]))
                 : "r"(mark_addr + (unsigned int)(tid_1 * 32)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&mw[4])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mw[(4) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mw[(4) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mw[(4) + 3]))
                 : "r"(mark_addr + (unsigned int)(tid_1 * 32 + 16)));
  }
  int c_t = 0;
  c_t = c_t + (int)(unsigned int)__popc(mw[0]);
  c_t = c_t + (int)(unsigned int)__popc(mw[1]);
  c_t = c_t + (int)(unsigned int)__popc(mw[2]);
  c_t = c_t + (int)(unsigned int)__popc(mw[3]);
  c_t = c_t + (int)(unsigned int)__popc(mw[4]);
  c_t = c_t + (int)(unsigned int)__popc(mw[5]);
  c_t = c_t + (int)(unsigned int)__popc(mw[6]);
  c_t = c_t + (int)(unsigned int)__popc(mw[7]);
  int incl_t = c_t;
  int up2 = __shfl_up_sync(0xFFFFFFFF, incl_t, 1, 32);
  if (lane_0 >= 1) {
    incl_t = incl_t + up2;
  }
  int up2_35 = __shfl_up_sync(0xFFFFFFFF, incl_t, 2, 32);
  if (lane_0 >= 2) {
    incl_t = incl_t + up2_35;
  }
  int up2_36 = __shfl_up_sync(0xFFFFFFFF, incl_t, 4, 32);
  if (lane_0 >= 4) {
    incl_t = incl_t + up2_36;
  }
  int up2_37 = __shfl_up_sync(0xFFFFFFFF, incl_t, 8, 32);
  if (lane_0 >= 8) {
    incl_t = incl_t + up2_37;
  }
  int up2_38 = __shfl_up_sync(0xFFFFFFFF, incl_t, 16, 32);
  if (lane_0 >= 16) {
    incl_t = incl_t + up2_38;
  }
  if (lane_0 == 31) {
    cnt[warp_1] = incl_t;
  }
  asm volatile("barrier.sync 1, 128;" ::: "memory");
  int pos_t = incl_t - c_t;
  int nblk_u = 0;
  int cw_ = cnt[0];
  nblk_u = nblk_u + cw_;
  if (warp_1 > 0) {
    pos_t = pos_t + cw_;
  }
  int cw__39 = cnt[1];
  nblk_u = nblk_u + cw__39;
  if (warp_1 > 1) {
    pos_t = pos_t + cw__39;
  }
  int cw__40 = cnt[2];
  nblk_u = nblk_u + cw__40;
  if (warp_1 > 2) {
    pos_t = pos_t + cw__40;
  }
  int cw__41 = cnt[3];
  nblk_u = nblk_u + cw__41;
  if (warp_1 > 3) {
    pos_t = pos_t + cw__41;
  }
  pref[tid_1] = pos_t;
  if (c_t != 0) {
    if ((mw[0] & 1) != 0) {
      int blk_u = tid_1 * 32;
      int pg_u = page_table[b_ * max_pages + blk_u];
      int blk_st = blk_u;
      if (pg_u < 0) {
        pg_u = 0;
        blk_st = 4194304;
      }
      meta_blk[pos_t] = blk_st;
      meta_pg[pos_t] = pg_u * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
    if ((mw[0] >> 8 & 1) != 0) {
      int blk_u_1 = tid_1 * 32 + 1;
      int pg_u_1 = page_table[b_ * max_pages + blk_u_1];
      int blk_st_1 = blk_u_1;
      if (pg_u_1 < 0) {
        pg_u_1 = 0;
        blk_st_1 = 4194304;
      }
      meta_blk[pos_t] = blk_st_1;
      meta_pg[pos_t] = pg_u_1 * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
    if ((mw[0] >> 16 & 1) != 0) {
      int blk_u_2 = tid_1 * 32 + 2;
      int pg_u_2 = page_table[b_ * max_pages + blk_u_2];
      int blk_st_2 = blk_u_2;
      if (pg_u_2 < 0) {
        pg_u_2 = 0;
        blk_st_2 = 4194304;
      }
      meta_blk[pos_t] = blk_st_2;
      meta_pg[pos_t] = pg_u_2 * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
    if ((mw[0] >> 24 & 1) != 0) {
      int blk_u_3 = tid_1 * 32 + 3;
      int pg_u_3 = page_table[b_ * max_pages + blk_u_3];
      int blk_st_3 = blk_u_3;
      if (pg_u_3 < 0) {
        pg_u_3 = 0;
        blk_st_3 = 4194304;
      }
      meta_blk[pos_t] = blk_st_3;
      meta_pg[pos_t] = pg_u_3 * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
    if ((mw[1] & 1) != 0) {
      int blk_u_4 = tid_1 * 32 + 4;
      int pg_u_4 = page_table[b_ * max_pages + blk_u_4];
      int blk_st_4 = blk_u_4;
      if (pg_u_4 < 0) {
        pg_u_4 = 0;
        blk_st_4 = 4194304;
      }
      meta_blk[pos_t] = blk_st_4;
      meta_pg[pos_t] = pg_u_4 * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
    if ((mw[1] >> 8 & 1) != 0) {
      int blk_u_5 = tid_1 * 32 + 4 + 1;
      int pg_u_5 = page_table[b_ * max_pages + blk_u_5];
      int blk_st_5 = blk_u_5;
      if (pg_u_5 < 0) {
        pg_u_5 = 0;
        blk_st_5 = 4194304;
      }
      meta_blk[pos_t] = blk_st_5;
      meta_pg[pos_t] = pg_u_5 * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
    if ((mw[1] >> 16 & 1) != 0) {
      int blk_u_6 = tid_1 * 32 + 4 + 2;
      int pg_u_6 = page_table[b_ * max_pages + blk_u_6];
      int blk_st_6 = blk_u_6;
      if (pg_u_6 < 0) {
        pg_u_6 = 0;
        blk_st_6 = 4194304;
      }
      meta_blk[pos_t] = blk_st_6;
      meta_pg[pos_t] = pg_u_6 * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
    if ((mw[1] >> 24 & 1) != 0) {
      int blk_u_7 = tid_1 * 32 + 4 + 3;
      int pg_u_7 = page_table[b_ * max_pages + blk_u_7];
      int blk_st_7 = blk_u_7;
      if (pg_u_7 < 0) {
        pg_u_7 = 0;
        blk_st_7 = 4194304;
      }
      meta_blk[pos_t] = blk_st_7;
      meta_pg[pos_t] = pg_u_7 * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
    if ((mw[2] & 1) != 0) {
      int blk_u_8 = tid_1 * 32 + 8;
      int pg_u_8 = page_table[b_ * max_pages + blk_u_8];
      int blk_st_8 = blk_u_8;
      if (pg_u_8 < 0) {
        pg_u_8 = 0;
        blk_st_8 = 4194304;
      }
      meta_blk[pos_t] = blk_st_8;
      meta_pg[pos_t] = pg_u_8 * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
    if ((mw[2] >> 8 & 1) != 0) {
      int blk_u_9 = tid_1 * 32 + 8 + 1;
      int pg_u_9 = page_table[b_ * max_pages + blk_u_9];
      int blk_st_9 = blk_u_9;
      if (pg_u_9 < 0) {
        pg_u_9 = 0;
        blk_st_9 = 4194304;
      }
      meta_blk[pos_t] = blk_st_9;
      meta_pg[pos_t] = pg_u_9 * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
    if ((mw[2] >> 16 & 1) != 0) {
      int blk_u_10 = tid_1 * 32 + 8 + 2;
      int pg_u_10 = page_table[b_ * max_pages + blk_u_10];
      int blk_st_10 = blk_u_10;
      if (pg_u_10 < 0) {
        pg_u_10 = 0;
        blk_st_10 = 4194304;
      }
      meta_blk[pos_t] = blk_st_10;
      meta_pg[pos_t] = pg_u_10 * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
    if ((mw[2] >> 24 & 1) != 0) {
      int blk_u_11 = tid_1 * 32 + 8 + 3;
      int pg_u_11 = page_table[b_ * max_pages + blk_u_11];
      int blk_st_11 = blk_u_11;
      if (pg_u_11 < 0) {
        pg_u_11 = 0;
        blk_st_11 = 4194304;
      }
      meta_blk[pos_t] = blk_st_11;
      meta_pg[pos_t] = pg_u_11 * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
    if ((mw[3] & 1) != 0) {
      int blk_u_12 = tid_1 * 32 + 12;
      int pg_u_12 = page_table[b_ * max_pages + blk_u_12];
      int blk_st_12 = blk_u_12;
      if (pg_u_12 < 0) {
        pg_u_12 = 0;
        blk_st_12 = 4194304;
      }
      meta_blk[pos_t] = blk_st_12;
      meta_pg[pos_t] = pg_u_12 * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
    if ((mw[3] >> 8 & 1) != 0) {
      int blk_u_13 = tid_1 * 32 + 12 + 1;
      int pg_u_13 = page_table[b_ * max_pages + blk_u_13];
      int blk_st_13 = blk_u_13;
      if (pg_u_13 < 0) {
        pg_u_13 = 0;
        blk_st_13 = 4194304;
      }
      meta_blk[pos_t] = blk_st_13;
      meta_pg[pos_t] = pg_u_13 * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
    if ((mw[3] >> 16 & 1) != 0) {
      int blk_u_14 = tid_1 * 32 + 12 + 2;
      int pg_u_14 = page_table[b_ * max_pages + blk_u_14];
      int blk_st_14 = blk_u_14;
      if (pg_u_14 < 0) {
        pg_u_14 = 0;
        blk_st_14 = 4194304;
      }
      meta_blk[pos_t] = blk_st_14;
      meta_pg[pos_t] = pg_u_14 * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
    if ((mw[3] >> 24 & 1) != 0) {
      int blk_u_15 = tid_1 * 32 + 12 + 3;
      int pg_u_15 = page_table[b_ * max_pages + blk_u_15];
      int blk_st_15 = blk_u_15;
      if (pg_u_15 < 0) {
        pg_u_15 = 0;
        blk_st_15 = 4194304;
      }
      meta_blk[pos_t] = blk_st_15;
      meta_pg[pos_t] = pg_u_15 * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
    if ((mw[4] & 1) != 0) {
      int blk_u_16 = tid_1 * 32 + 16;
      int pg_u_16 = page_table[b_ * max_pages + blk_u_16];
      int blk_st_16 = blk_u_16;
      if (pg_u_16 < 0) {
        pg_u_16 = 0;
        blk_st_16 = 4194304;
      }
      meta_blk[pos_t] = blk_st_16;
      meta_pg[pos_t] = pg_u_16 * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
    if ((mw[4] >> 8 & 1) != 0) {
      int blk_u_17 = tid_1 * 32 + 16 + 1;
      int pg_u_17 = page_table[b_ * max_pages + blk_u_17];
      int blk_st_17 = blk_u_17;
      if (pg_u_17 < 0) {
        pg_u_17 = 0;
        blk_st_17 = 4194304;
      }
      meta_blk[pos_t] = blk_st_17;
      meta_pg[pos_t] = pg_u_17 * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
    if ((mw[4] >> 16 & 1) != 0) {
      int blk_u_18 = tid_1 * 32 + 16 + 2;
      int pg_u_18 = page_table[b_ * max_pages + blk_u_18];
      int blk_st_18 = blk_u_18;
      if (pg_u_18 < 0) {
        pg_u_18 = 0;
        blk_st_18 = 4194304;
      }
      meta_blk[pos_t] = blk_st_18;
      meta_pg[pos_t] = pg_u_18 * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
    if ((mw[4] >> 24 & 1) != 0) {
      int blk_u_19 = tid_1 * 32 + 16 + 3;
      int pg_u_19 = page_table[b_ * max_pages + blk_u_19];
      int blk_st_19 = blk_u_19;
      if (pg_u_19 < 0) {
        pg_u_19 = 0;
        blk_st_19 = 4194304;
      }
      meta_blk[pos_t] = blk_st_19;
      meta_pg[pos_t] = pg_u_19 * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
    if ((mw[5] & 1) != 0) {
      int blk_u_20 = tid_1 * 32 + 20;
      int pg_u_20 = page_table[b_ * max_pages + blk_u_20];
      int blk_st_20 = blk_u_20;
      if (pg_u_20 < 0) {
        pg_u_20 = 0;
        blk_st_20 = 4194304;
      }
      meta_blk[pos_t] = blk_st_20;
      meta_pg[pos_t] = pg_u_20 * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
    if ((mw[5] >> 8 & 1) != 0) {
      int blk_u_21 = tid_1 * 32 + 20 + 1;
      int pg_u_21 = page_table[b_ * max_pages + blk_u_21];
      int blk_st_21 = blk_u_21;
      if (pg_u_21 < 0) {
        pg_u_21 = 0;
        blk_st_21 = 4194304;
      }
      meta_blk[pos_t] = blk_st_21;
      meta_pg[pos_t] = pg_u_21 * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
    if ((mw[5] >> 16 & 1) != 0) {
      int blk_u_22 = tid_1 * 32 + 20 + 2;
      int pg_u_22 = page_table[b_ * max_pages + blk_u_22];
      int blk_st_22 = blk_u_22;
      if (pg_u_22 < 0) {
        pg_u_22 = 0;
        blk_st_22 = 4194304;
      }
      meta_blk[pos_t] = blk_st_22;
      meta_pg[pos_t] = pg_u_22 * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
    if ((mw[5] >> 24 & 1) != 0) {
      int blk_u_23 = tid_1 * 32 + 20 + 3;
      int pg_u_23 = page_table[b_ * max_pages + blk_u_23];
      int blk_st_23 = blk_u_23;
      if (pg_u_23 < 0) {
        pg_u_23 = 0;
        blk_st_23 = 4194304;
      }
      meta_blk[pos_t] = blk_st_23;
      meta_pg[pos_t] = pg_u_23 * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
    if ((mw[6] & 1) != 0) {
      int blk_u_24 = tid_1 * 32 + 24;
      int pg_u_24 = page_table[b_ * max_pages + blk_u_24];
      int blk_st_24 = blk_u_24;
      if (pg_u_24 < 0) {
        pg_u_24 = 0;
        blk_st_24 = 4194304;
      }
      meta_blk[pos_t] = blk_st_24;
      meta_pg[pos_t] = pg_u_24 * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
    if ((mw[6] >> 8 & 1) != 0) {
      int blk_u_25 = tid_1 * 32 + 24 + 1;
      int pg_u_25 = page_table[b_ * max_pages + blk_u_25];
      int blk_st_25 = blk_u_25;
      if (pg_u_25 < 0) {
        pg_u_25 = 0;
        blk_st_25 = 4194304;
      }
      meta_blk[pos_t] = blk_st_25;
      meta_pg[pos_t] = pg_u_25 * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
    if ((mw[6] >> 16 & 1) != 0) {
      int blk_u_26 = tid_1 * 32 + 24 + 2;
      int pg_u_26 = page_table[b_ * max_pages + blk_u_26];
      int blk_st_26 = blk_u_26;
      if (pg_u_26 < 0) {
        pg_u_26 = 0;
        blk_st_26 = 4194304;
      }
      meta_blk[pos_t] = blk_st_26;
      meta_pg[pos_t] = pg_u_26 * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
    if ((mw[6] >> 24 & 1) != 0) {
      int blk_u_27 = tid_1 * 32 + 24 + 3;
      int pg_u_27 = page_table[b_ * max_pages + blk_u_27];
      int blk_st_27 = blk_u_27;
      if (pg_u_27 < 0) {
        pg_u_27 = 0;
        blk_st_27 = 4194304;
      }
      meta_blk[pos_t] = blk_st_27;
      meta_pg[pos_t] = pg_u_27 * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
    if ((mw[7] & 1) != 0) {
      int blk_u_28 = tid_1 * 32 + 28;
      int pg_u_28 = page_table[b_ * max_pages + blk_u_28];
      int blk_st_28 = blk_u_28;
      if (pg_u_28 < 0) {
        pg_u_28 = 0;
        blk_st_28 = 4194304;
      }
      meta_blk[pos_t] = blk_st_28;
      meta_pg[pos_t] = pg_u_28 * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
    if ((mw[7] >> 8 & 1) != 0) {
      int blk_u_29 = tid_1 * 32 + 28 + 1;
      int pg_u_29 = page_table[b_ * max_pages + blk_u_29];
      int blk_st_29 = blk_u_29;
      if (pg_u_29 < 0) {
        pg_u_29 = 0;
        blk_st_29 = 4194304;
      }
      meta_blk[pos_t] = blk_st_29;
      meta_pg[pos_t] = pg_u_29 * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
    if ((mw[7] >> 16 & 1) != 0) {
      int blk_u_30 = tid_1 * 32 + 28 + 2;
      int pg_u_30 = page_table[b_ * max_pages + blk_u_30];
      int blk_st_30 = blk_u_30;
      if (pg_u_30 < 0) {
        pg_u_30 = 0;
        blk_st_30 = 4194304;
      }
      meta_blk[pos_t] = blk_st_30;
      meta_pg[pos_t] = pg_u_30 * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
    if ((mw[7] >> 24 & 1) != 0) {
      int blk_u_31 = tid_1 * 32 + 28 + 3;
      int pg_u_31 = page_table[b_ * max_pages + blk_u_31];
      int blk_st_31 = blk_u_31;
      if (pg_u_31 < 0) {
        pg_u_31 = 0;
        blk_st_31 = 4194304;
      }
      meta_blk[pos_t] = blk_st_31;
      meta_pg[pos_t] = pg_u_31 * num_kv_heads + kv_head;
      meta_mask[pos_t] = 0;
      pos_t = pos_t + 1;
    }
  }
  asm volatile("barrier.sync 1, 128;" ::: "memory");
  if (blk_e[0] >= 0) {
    int grp_e = blk_e[0] >> 5;
    int o_e = blk_e[0] & 31;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&mw[0])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mw[(0) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mw[(0) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mw[(0) + 3]))
                 : "r"(mark_addr + (unsigned int)(grp_e * 32)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&mw[4])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mw[(4) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mw[(4) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mw[(4) + 3]))
                 : "r"(mark_addr + (unsigned int)(grp_e * 32 + 16)));
    int rank = pref[grp_e];
    int nb_ = o_e;
    if (nb_ > 0) {
      if (nb_ >= 4) {
        rank = rank + (int)(unsigned int)__popc(mw[0]);
      }
      if (nb_ < 4) {
        unsigned int lowmask = (1 << (unsigned int)(8 * (nb_ & 3))) - 1;
        rank = rank + (int)(unsigned int)__popc(mw[0] & lowmask);
      }
    }
    int nb__0 = o_e - 4;
    if (nb__0 > 0) {
      if (nb__0 >= 4) {
        rank = rank + (int)(unsigned int)__popc(mw[1]);
      }
      if (nb__0 < 4) {
        unsigned int lowmask_1 = (1 << (unsigned int)(8 * (nb__0 & 3))) - 1;
        rank = rank + (int)(unsigned int)__popc(mw[1] & lowmask_1);
      }
    }
    int nb__1 = o_e - 8;
    if (nb__1 > 0) {
      if (nb__1 >= 4) {
        rank = rank + (int)(unsigned int)__popc(mw[2]);
      }
      if (nb__1 < 4) {
        unsigned int lowmask_2 = (1 << (unsigned int)(8 * (nb__1 & 3))) - 1;
        rank = rank + (int)(unsigned int)__popc(mw[2] & lowmask_2);
      }
    }
    int nb__2 = o_e - 12;
    if (nb__2 > 0) {
      if (nb__2 >= 4) {
        rank = rank + (int)(unsigned int)__popc(mw[3]);
      }
      if (nb__2 < 4) {
        unsigned int lowmask_3 = (1 << (unsigned int)(8 * (nb__2 & 3))) - 1;
        rank = rank + (int)(unsigned int)__popc(mw[3] & lowmask_3);
      }
    }
    int nb__3 = o_e - 16;
    if (nb__3 > 0) {
      if (nb__3 >= 4) {
        rank = rank + (int)(unsigned int)__popc(mw[4]);
      }
      if (nb__3 < 4) {
        unsigned int lowmask_4 = (1 << (unsigned int)(8 * (nb__3 & 3))) - 1;
        rank = rank + (int)(unsigned int)__popc(mw[4] & lowmask_4);
      }
    }
    int nb__4 = o_e - 20;
    if (nb__4 > 0) {
      if (nb__4 >= 4) {
        rank = rank + (int)(unsigned int)__popc(mw[5]);
      }
      if (nb__4 < 4) {
        unsigned int lowmask_5 = (1 << (unsigned int)(8 * (nb__4 & 3))) - 1;
        rank = rank + (int)(unsigned int)__popc(mw[5] & lowmask_5);
      }
    }
    int nb__5 = o_e - 24;
    if (nb__5 > 0) {
      if (nb__5 >= 4) {
        rank = rank + (int)(unsigned int)__popc(mw[6]);
      }
      if (nb__5 < 4) {
        unsigned int lowmask_6 = (1 << (unsigned int)(8 * (nb__5 & 3))) - 1;
        rank = rank + (int)(unsigned int)__popc(mw[6] & lowmask_6);
      }
    }
    int nb__6 = o_e - 28;
    if (nb__6 > 0) {
      if (nb__6 >= 4) {
        rank = rank + (int)(unsigned int)__popc(mw[7]);
      }
      if (nb__6 < 4) {
        unsigned int lowmask_7 = (1 << (unsigned int)(8 * (nb__6 & 3))) - 1;
        rank = rank + (int)(unsigned int)__popc(mw[7] & lowmask_7);
      }
    }
    int tl_i = tid_1 / 16;
    unsigned int bit = 1 << (unsigned int)tl_i;
    atomicAdd_block(&meta_mask[rank], bit);
  }
  asm volatile("barrier.sync 1, 128;" ::: "memory");
  if (nblk_u > 0) {
    int page_head = meta_pg[0];
    if (warp == 1) {
      if (elect_sync()) {
        mbarrier_arrive_expect_tx(kv_full_addr, 32768);
        unsigned int base = kv8_addr;
        tma_3d_gmem2smem(base, (&K), 0, 0, page_head, kv_full_addr);
        tma_3d_gmem2smem(base + 8192, (&K), 0, 64, page_head, kv_full_addr);
        tma_3d_gmem2smem(base + 16384, (&V), 0, 0, page_head, kv_full_addr);
        tma_3d_gmem2smem(base + 24576, (&V), 0, 64, page_head, kv_full_addr);
      }
    }
  }
  if (nblk_u > 1) {
    int page_head_1 = meta_pg[1];
    if (warp == 1) {
      if (elect_sync()) {
        mbarrier_arrive_expect_tx(kv_full_addr + 8, 32768);
        unsigned int base_1 = kv8_addr + 32768;
        tma_3d_gmem2smem(base_1, (&K), 0, 0, page_head_1, kv_full_addr + 8);
        tma_3d_gmem2smem(base_1 + 8192, (&K), 0, 64, page_head_1, kv_full_addr + 8);
        tma_3d_gmem2smem(base_1 + 16384, (&V), 0, 0, page_head_1, kv_full_addr + 8);
        tma_3d_gmem2smem(base_1 + 24576, (&V), 0, 64, page_head_1, kv_full_addr + 8);
      }
    }
  }
  int qpos1 = qpos0 + 1;
  unsigned int prev_bl = 15;
  unsigned int dk0_0[8];
  unsigned int dk0_1[8];
  unsigned int dk0_2[8];
  unsigned int dk0_3[8];
  unsigned int dk1_0[8];
  unsigned int dk1_1[8];
  unsigned int dk1_2[8];
  unsigned int dk1_3[8];
  unsigned int dv0_0[16];
  unsigned int dv0_1[16];
  unsigned int dv1_0[16];
  unsigned int dv1_1[16];
  unsigned int ak0[32];
  unsigned int ak1[32];
  unsigned int av0[32];
  unsigned int av1[32];
  float d_s0[16];
  float d_s1[16];
  float d_o0[16] = {0};
  float d_o1[16] = {0};
  float mrun[8];
  float lp[8] = {0};
  mrun[0] = -CUDART_INF_F;
  mrun[1] = -CUDART_INF_F;
  mrun[2] = -CUDART_INF_F;
  mrun[3] = -CUDART_INF_F;
  mrun[4] = -CUDART_INF_F;
  mrun[5] = -CUDART_INF_F;
  mrun[6] = -CUDART_INF_F;
  mrun[7] = -CUDART_INF_F;
  float tmax[8];
  float rv[4];
  uint64_t _wgmma_desc_24 = (((uint64_t)(((ident8_addr)) >> 4) & 0x3FFFULL) |
                             ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
  uint64_t _wgmma_b_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_24 >> 32)) << 32) |
                        (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_24);
  uint64_t _wgmma_desc_25 = (((uint64_t)(((q16_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) |
                             ((uint64_t)(64) << 32) | (1ULL << 62));
  uint64_t _wgmma_b_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_25 >> 32)) << 32) |
                        (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_25);
  uint64_t _wgmma_desc_26 = (((uint64_t)(((q16_addr + 4096)) >> 4) & 0x3FFFULL) |
                             ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
  uint64_t _wgmma_b_2 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_26 >> 32)) << 32) |
                        (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_26);
  uint64_t _wgmma_desc_27 = (((uint64_t)(((p16_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) |
                             ((uint64_t)(64) << 32) | (1ULL << 62));
  uint64_t _wgmma_b_3 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_27 >> 32)) << 32) |
                        (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_27);
  uint64_t _wgmma_desc_28 = (((uint64_t)(((p16_addr + 4096)) >> 4) & 0x3FFFULL) |
                             ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
  uint64_t _wgmma_b_4 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_28 >> 32)) << 32) |
                        (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_28);
#pragma unroll 1
  for (int j = 0; j < nblk_u; j++) {
    int stage = j % 2;
    int phase = j / 2 % 2;
    int blk_j = meta_blk[j];
    unsigned int mask_j = meta_mask[j];
    int pbase = blk_j * 128;
    int vlim = sk_b - pbase;
    unsigned int bl = 0;
#if CAKE_GROUP == 16
    if ((mask_j & 1) != 0) {
#elif CAKE_GROUP == 8
    if ((mask_j & 1) != 0) {
#elif CAKE_GROUP == 4
    if ((mask_j & 3) != 0) {
#else
#error "unsupported CAKE_GROUP"
#endif
      bl = bl | 1;
    }
#if CAKE_GROUP == 16
    if ((mask_j & 1) != 0) {
#elif CAKE_GROUP == 8
    if ((mask_j >> 1 & 1) != 0) {
#elif CAKE_GROUP == 4
    if ((mask_j >> 2 & 3) != 0) {
#else
#error "unsupported CAKE_GROUP"
#endif
      bl = bl | 2;
    }
#if CAKE_GROUP == 16
    if ((mask_j >> 1 & 1) != 0) {
#elif CAKE_GROUP == 8
    if ((mask_j >> 2 & 1) != 0) {
#elif CAKE_GROUP == 4
    if ((mask_j >> 4 & 3) != 0) {
#else
#error "unsupported CAKE_GROUP"
#endif
      bl = bl | 4;
    }
#if CAKE_GROUP == 16
    if ((mask_j >> 1 & 1) != 0) {
#elif CAKE_GROUP == 8
    if ((mask_j >> 3 & 1) != 0) {
#elif CAKE_GROUP == 4
    if ((mask_j >> 6 & 3) != 0) {
#else
#error "unsupported CAKE_GROUP"
#endif
      bl = bl | 8;
    }
    mbarrier_wait(kv_full_addr + (stage) * 8, phase);
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    uint64_t _wgmma_desc_29 =
        (((uint64_t)(((kv8_addr + (unsigned int)(stage * 32768))) >> 4) & 0x3FFFULL) |
         ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_a_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_29 >> 32)) << 32) |
                          (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_29);
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "%8, %9, 0, 1, 1;\n}\n"
        : "+r"(dk0_0[0]), "+r"(dk0_0[1]), "+r"(dk0_0[2]), "+r"(dk0_0[3]), "+r"(dk0_0[4]),
          "+r"(dk0_0[5]), "+r"(dk0_0[6]), "+r"(dk0_0[7])
        : "l"(_wgmma_a_0), "l"(_wgmma_b_0)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "%8, %9, 0, 1, 1;\n}\n"
        : "+r"(dk0_1[0]), "+r"(dk0_1[1]), "+r"(dk0_1[2]), "+r"(dk0_1[3]), "+r"(dk0_1[4]),
          "+r"(dk0_1[5]), "+r"(dk0_1[6]), "+r"(dk0_1[7])
        : "l"(_wgmma_a_0 + 2), "l"(_wgmma_b_0)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "%8, %9, 0, 1, 1;\n}\n"
        : "+r"(dk0_2[0]), "+r"(dk0_2[1]), "+r"(dk0_2[2]), "+r"(dk0_2[3]), "+r"(dk0_2[4]),
          "+r"(dk0_2[5]), "+r"(dk0_2[6]), "+r"(dk0_2[7])
        : "l"(_wgmma_a_0 + 4), "l"(_wgmma_b_0)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "%8, %9, 0, 1, 1;\n}\n"
        : "+r"(dk0_3[0]), "+r"(dk0_3[1]), "+r"(dk0_3[2]), "+r"(dk0_3[3]), "+r"(dk0_3[4]),
          "+r"(dk0_3[5]), "+r"(dk0_3[6]), "+r"(dk0_3[7])
        : "l"(_wgmma_a_0 + 6), "l"(_wgmma_b_0)
        : "memory");
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    uint64_t _wgmma_desc_30 =
        (((uint64_t)(((kv8_addr + (unsigned int)(stage * 32768) + 8192)) >> 4) & 0x3FFFULL) |
         ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_a_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_30 >> 32)) << 32) |
                          (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_30);
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "%8, %9, 0, 1, 1;\n}\n"
        : "+r"(dk1_0[0]), "+r"(dk1_0[1]), "+r"(dk1_0[2]), "+r"(dk1_0[3]), "+r"(dk1_0[4]),
          "+r"(dk1_0[5]), "+r"(dk1_0[6]), "+r"(dk1_0[7])
        : "l"(_wgmma_a_1), "l"(_wgmma_b_0)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "%8, %9, 0, 1, 1;\n}\n"
        : "+r"(dk1_1[0]), "+r"(dk1_1[1]), "+r"(dk1_1[2]), "+r"(dk1_1[3]), "+r"(dk1_1[4]),
          "+r"(dk1_1[5]), "+r"(dk1_1[6]), "+r"(dk1_1[7])
        : "l"(_wgmma_a_1 + 2), "l"(_wgmma_b_0)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "%8, %9, 0, 1, 1;\n}\n"
        : "+r"(dk1_2[0]), "+r"(dk1_2[1]), "+r"(dk1_2[2]), "+r"(dk1_2[3]), "+r"(dk1_2[4]),
          "+r"(dk1_2[5]), "+r"(dk1_2[6]), "+r"(dk1_2[7])
        : "l"(_wgmma_a_1 + 4), "l"(_wgmma_b_0)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "%8, %9, 0, 1, 1;\n}\n"
        : "+r"(dk1_3[0]), "+r"(dk1_3[1]), "+r"(dk1_3[2]), "+r"(dk1_3[3]), "+r"(dk1_3[4]),
          "+r"(dk1_3[5]), "+r"(dk1_3[6]), "+r"(dk1_3[7])
        : "l"(_wgmma_a_1 + 6), "l"(_wgmma_b_0)
        : "memory");
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
    ak0[0] = dk0_0[0] + zero_u32;
    ak0[1] = dk0_0[1] + zero_u32;
    ak0[2] = dk0_0[2] + zero_u32;
    ak0[3] = dk0_0[3] + zero_u32;
    ak0[4] = dk0_0[4] + zero_u32;
    ak0[5] = dk0_0[5] + zero_u32;
    ak0[6] = dk0_0[6] + zero_u32;
    ak0[7] = dk0_0[7] + zero_u32;
    ak0[8] = dk0_1[0] + zero_u32;
    ak0[9] = dk0_1[1] + zero_u32;
    ak0[10] = dk0_1[2] + zero_u32;
    ak0[11] = dk0_1[3] + zero_u32;
    ak0[12] = dk0_1[4] + zero_u32;
    ak0[13] = dk0_1[5] + zero_u32;
    ak0[14] = dk0_1[6] + zero_u32;
    ak0[15] = dk0_1[7] + zero_u32;
    ak0[16] = dk0_2[0] + zero_u32;
    ak0[17] = dk0_2[1] + zero_u32;
    ak0[18] = dk0_2[2] + zero_u32;
    ak0[19] = dk0_2[3] + zero_u32;
    ak0[20] = dk0_2[4] + zero_u32;
    ak0[21] = dk0_2[5] + zero_u32;
    ak0[22] = dk0_2[6] + zero_u32;
    ak0[23] = dk0_2[7] + zero_u32;
    ak0[24] = dk0_3[0] + zero_u32;
    ak0[25] = dk0_3[1] + zero_u32;
    ak0[26] = dk0_3[2] + zero_u32;
    ak0[27] = dk0_3[3] + zero_u32;
    ak0[28] = dk0_3[4] + zero_u32;
    ak0[29] = dk0_3[5] + zero_u32;
    ak0[30] = dk0_3[6] + zero_u32;
    ak0[31] = dk0_3[7] + zero_u32;
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 0, 1, 1, 0;\n}\n"
        : "+f"(d_s0[0]), "+f"(d_s0[1]), "+f"(d_s0[2]), "+f"(d_s0[3]), "+f"(d_s0[4]), "+f"(d_s0[5]),
          "+f"(d_s0[6]), "+f"(d_s0[7]), "+f"(d_s0[8]), "+f"(d_s0[9]), "+f"(d_s0[10]),
          "+f"(d_s0[11]), "+f"(d_s0[12]), "+f"(d_s0[13]), "+f"(d_s0[14]), "+f"(d_s0[15])
        : "r"(ak0[0]), "r"(ak0[1]), "r"(ak0[2]), "r"(ak0[3]), "l"(_wgmma_b_1)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s0[0]), "+f"(d_s0[1]), "+f"(d_s0[2]), "+f"(d_s0[3]), "+f"(d_s0[4]), "+f"(d_s0[5]),
          "+f"(d_s0[6]), "+f"(d_s0[7]), "+f"(d_s0[8]), "+f"(d_s0[9]), "+f"(d_s0[10]),
          "+f"(d_s0[11]), "+f"(d_s0[12]), "+f"(d_s0[13]), "+f"(d_s0[14]), "+f"(d_s0[15])
        : "r"(ak0[4]), "r"(ak0[(4) + 1]), "r"(ak0[(4) + 2]), "r"(ak0[(4) + 3]), "l"(_wgmma_b_1 + 2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s0[0]), "+f"(d_s0[1]), "+f"(d_s0[2]), "+f"(d_s0[3]), "+f"(d_s0[4]), "+f"(d_s0[5]),
          "+f"(d_s0[6]), "+f"(d_s0[7]), "+f"(d_s0[8]), "+f"(d_s0[9]), "+f"(d_s0[10]),
          "+f"(d_s0[11]), "+f"(d_s0[12]), "+f"(d_s0[13]), "+f"(d_s0[14]), "+f"(d_s0[15])
        : "r"(ak0[8]), "r"(ak0[(8) + 1]), "r"(ak0[(8) + 2]), "r"(ak0[(8) + 3]), "l"(_wgmma_b_1 + 4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s0[0]), "+f"(d_s0[1]), "+f"(d_s0[2]), "+f"(d_s0[3]), "+f"(d_s0[4]), "+f"(d_s0[5]),
          "+f"(d_s0[6]), "+f"(d_s0[7]), "+f"(d_s0[8]), "+f"(d_s0[9]), "+f"(d_s0[10]),
          "+f"(d_s0[11]), "+f"(d_s0[12]), "+f"(d_s0[13]), "+f"(d_s0[14]), "+f"(d_s0[15])
        : "r"(ak0[12]), "r"(ak0[(12) + 1]), "r"(ak0[(12) + 2]), "r"(ak0[(12) + 3]),
          "l"(_wgmma_b_1 + 6)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s0[0]), "+f"(d_s0[1]), "+f"(d_s0[2]), "+f"(d_s0[3]), "+f"(d_s0[4]), "+f"(d_s0[5]),
          "+f"(d_s0[6]), "+f"(d_s0[7]), "+f"(d_s0[8]), "+f"(d_s0[9]), "+f"(d_s0[10]),
          "+f"(d_s0[11]), "+f"(d_s0[12]), "+f"(d_s0[13]), "+f"(d_s0[14]), "+f"(d_s0[15])
        : "r"(ak0[16]), "r"(ak0[(16) + 1]), "r"(ak0[(16) + 2]), "r"(ak0[(16) + 3]), "l"(_wgmma_b_2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s0[0]), "+f"(d_s0[1]), "+f"(d_s0[2]), "+f"(d_s0[3]), "+f"(d_s0[4]), "+f"(d_s0[5]),
          "+f"(d_s0[6]), "+f"(d_s0[7]), "+f"(d_s0[8]), "+f"(d_s0[9]), "+f"(d_s0[10]),
          "+f"(d_s0[11]), "+f"(d_s0[12]), "+f"(d_s0[13]), "+f"(d_s0[14]), "+f"(d_s0[15])
        : "r"(ak0[20]), "r"(ak0[(20) + 1]), "r"(ak0[(20) + 2]), "r"(ak0[(20) + 3]),
          "l"(_wgmma_b_2 + 2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s0[0]), "+f"(d_s0[1]), "+f"(d_s0[2]), "+f"(d_s0[3]), "+f"(d_s0[4]), "+f"(d_s0[5]),
          "+f"(d_s0[6]), "+f"(d_s0[7]), "+f"(d_s0[8]), "+f"(d_s0[9]), "+f"(d_s0[10]),
          "+f"(d_s0[11]), "+f"(d_s0[12]), "+f"(d_s0[13]), "+f"(d_s0[14]), "+f"(d_s0[15])
        : "r"(ak0[24]), "r"(ak0[(24) + 1]), "r"(ak0[(24) + 2]), "r"(ak0[(24) + 3]),
          "l"(_wgmma_b_2 + 4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s0[0]), "+f"(d_s0[1]), "+f"(d_s0[2]), "+f"(d_s0[3]), "+f"(d_s0[4]), "+f"(d_s0[5]),
          "+f"(d_s0[6]), "+f"(d_s0[7]), "+f"(d_s0[8]), "+f"(d_s0[9]), "+f"(d_s0[10]),
          "+f"(d_s0[11]), "+f"(d_s0[12]), "+f"(d_s0[13]), "+f"(d_s0[14]), "+f"(d_s0[15])
        : "r"(ak0[28]), "r"(ak0[(28) + 1]), "r"(ak0[(28) + 2]), "r"(ak0[(28) + 3]),
          "l"(_wgmma_b_2 + 6)
        : "memory");
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
    ak1[0] = dk1_0[0] + zero_u32;
    ak1[1] = dk1_0[1] + zero_u32;
    ak1[2] = dk1_0[2] + zero_u32;
    ak1[3] = dk1_0[3] + zero_u32;
    ak1[4] = dk1_0[4] + zero_u32;
    ak1[5] = dk1_0[5] + zero_u32;
    ak1[6] = dk1_0[6] + zero_u32;
    ak1[7] = dk1_0[7] + zero_u32;
    ak1[8] = dk1_1[0] + zero_u32;
    ak1[9] = dk1_1[1] + zero_u32;
    ak1[10] = dk1_1[2] + zero_u32;
    ak1[11] = dk1_1[3] + zero_u32;
    ak1[12] = dk1_1[4] + zero_u32;
    ak1[13] = dk1_1[5] + zero_u32;
    ak1[14] = dk1_1[6] + zero_u32;
    ak1[15] = dk1_1[7] + zero_u32;
    ak1[16] = dk1_2[0] + zero_u32;
    ak1[17] = dk1_2[1] + zero_u32;
    ak1[18] = dk1_2[2] + zero_u32;
    ak1[19] = dk1_2[3] + zero_u32;
    ak1[20] = dk1_2[4] + zero_u32;
    ak1[21] = dk1_2[5] + zero_u32;
    ak1[22] = dk1_2[6] + zero_u32;
    ak1[23] = dk1_2[7] + zero_u32;
    ak1[24] = dk1_3[0] + zero_u32;
    ak1[25] = dk1_3[1] + zero_u32;
    ak1[26] = dk1_3[2] + zero_u32;
    ak1[27] = dk1_3[3] + zero_u32;
    ak1[28] = dk1_3[4] + zero_u32;
    ak1[29] = dk1_3[5] + zero_u32;
    ak1[30] = dk1_3[6] + zero_u32;
    ak1[31] = dk1_3[7] + zero_u32;
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 0, 1, 1, 0;\n}\n"
        : "+f"(d_s1[0]), "+f"(d_s1[1]), "+f"(d_s1[2]), "+f"(d_s1[3]), "+f"(d_s1[4]), "+f"(d_s1[5]),
          "+f"(d_s1[6]), "+f"(d_s1[7]), "+f"(d_s1[8]), "+f"(d_s1[9]), "+f"(d_s1[10]),
          "+f"(d_s1[11]), "+f"(d_s1[12]), "+f"(d_s1[13]), "+f"(d_s1[14]), "+f"(d_s1[15])
        : "r"(ak1[0]), "r"(ak1[1]), "r"(ak1[2]), "r"(ak1[3]), "l"(_wgmma_b_1)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s1[0]), "+f"(d_s1[1]), "+f"(d_s1[2]), "+f"(d_s1[3]), "+f"(d_s1[4]), "+f"(d_s1[5]),
          "+f"(d_s1[6]), "+f"(d_s1[7]), "+f"(d_s1[8]), "+f"(d_s1[9]), "+f"(d_s1[10]),
          "+f"(d_s1[11]), "+f"(d_s1[12]), "+f"(d_s1[13]), "+f"(d_s1[14]), "+f"(d_s1[15])
        : "r"(ak1[4]), "r"(ak1[(4) + 1]), "r"(ak1[(4) + 2]), "r"(ak1[(4) + 3]), "l"(_wgmma_b_1 + 2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s1[0]), "+f"(d_s1[1]), "+f"(d_s1[2]), "+f"(d_s1[3]), "+f"(d_s1[4]), "+f"(d_s1[5]),
          "+f"(d_s1[6]), "+f"(d_s1[7]), "+f"(d_s1[8]), "+f"(d_s1[9]), "+f"(d_s1[10]),
          "+f"(d_s1[11]), "+f"(d_s1[12]), "+f"(d_s1[13]), "+f"(d_s1[14]), "+f"(d_s1[15])
        : "r"(ak1[8]), "r"(ak1[(8) + 1]), "r"(ak1[(8) + 2]), "r"(ak1[(8) + 3]), "l"(_wgmma_b_1 + 4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s1[0]), "+f"(d_s1[1]), "+f"(d_s1[2]), "+f"(d_s1[3]), "+f"(d_s1[4]), "+f"(d_s1[5]),
          "+f"(d_s1[6]), "+f"(d_s1[7]), "+f"(d_s1[8]), "+f"(d_s1[9]), "+f"(d_s1[10]),
          "+f"(d_s1[11]), "+f"(d_s1[12]), "+f"(d_s1[13]), "+f"(d_s1[14]), "+f"(d_s1[15])
        : "r"(ak1[12]), "r"(ak1[(12) + 1]), "r"(ak1[(12) + 2]), "r"(ak1[(12) + 3]),
          "l"(_wgmma_b_1 + 6)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s1[0]), "+f"(d_s1[1]), "+f"(d_s1[2]), "+f"(d_s1[3]), "+f"(d_s1[4]), "+f"(d_s1[5]),
          "+f"(d_s1[6]), "+f"(d_s1[7]), "+f"(d_s1[8]), "+f"(d_s1[9]), "+f"(d_s1[10]),
          "+f"(d_s1[11]), "+f"(d_s1[12]), "+f"(d_s1[13]), "+f"(d_s1[14]), "+f"(d_s1[15])
        : "r"(ak1[16]), "r"(ak1[(16) + 1]), "r"(ak1[(16) + 2]), "r"(ak1[(16) + 3]), "l"(_wgmma_b_2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s1[0]), "+f"(d_s1[1]), "+f"(d_s1[2]), "+f"(d_s1[3]), "+f"(d_s1[4]), "+f"(d_s1[5]),
          "+f"(d_s1[6]), "+f"(d_s1[7]), "+f"(d_s1[8]), "+f"(d_s1[9]), "+f"(d_s1[10]),
          "+f"(d_s1[11]), "+f"(d_s1[12]), "+f"(d_s1[13]), "+f"(d_s1[14]), "+f"(d_s1[15])
        : "r"(ak1[20]), "r"(ak1[(20) + 1]), "r"(ak1[(20) + 2]), "r"(ak1[(20) + 3]),
          "l"(_wgmma_b_2 + 2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s1[0]), "+f"(d_s1[1]), "+f"(d_s1[2]), "+f"(d_s1[3]), "+f"(d_s1[4]), "+f"(d_s1[5]),
          "+f"(d_s1[6]), "+f"(d_s1[7]), "+f"(d_s1[8]), "+f"(d_s1[9]), "+f"(d_s1[10]),
          "+f"(d_s1[11]), "+f"(d_s1[12]), "+f"(d_s1[13]), "+f"(d_s1[14]), "+f"(d_s1[15])
        : "r"(ak1[24]), "r"(ak1[(24) + 1]), "r"(ak1[(24) + 2]), "r"(ak1[(24) + 3]),
          "l"(_wgmma_b_2 + 4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s1[0]), "+f"(d_s1[1]), "+f"(d_s1[2]), "+f"(d_s1[3]), "+f"(d_s1[4]), "+f"(d_s1[5]),
          "+f"(d_s1[6]), "+f"(d_s1[7]), "+f"(d_s1[8]), "+f"(d_s1[9]), "+f"(d_s1[10]),
          "+f"(d_s1[11]), "+f"(d_s1[12]), "+f"(d_s1[13]), "+f"(d_s1[14]), "+f"(d_s1[15])
        : "r"(ak1[28]), "r"(ak1[(28) + 1]), "r"(ak1[(28) + 2]), "r"(ak1[(28) + 3]),
          "l"(_wgmma_b_2 + 6)
        : "memory");
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    uint64_t _wgmma_desc_31 =
        (((uint64_t)(((kv8_addr + (unsigned int)(stage * 32768) + 16384)) >> 4) & 0x3FFFULL) |
         ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_5 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_31 >> 32)) << 32) |
                          (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_31);
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 0, 1, 1;\n}\n"
        : "+r"(dv0_0[0]), "+r"(dv0_0[1]), "+r"(dv0_0[2]), "+r"(dv0_0[3]), "+r"(dv0_0[4]),
          "+r"(dv0_0[5]), "+r"(dv0_0[6]), "+r"(dv0_0[7]), "+r"(dv0_0[8]), "+r"(dv0_0[9]),
          "+r"(dv0_0[10]), "+r"(dv0_0[11]), "+r"(dv0_0[12]), "+r"(dv0_0[13]), "+r"(dv0_0[14]),
          "+r"(dv0_0[15])
        : "l"(_wgmma_b_0), "l"(_wgmma_b_5)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1;\n}\n"
        : "+r"(dv0_0[0]), "+r"(dv0_0[1]), "+r"(dv0_0[2]), "+r"(dv0_0[3]), "+r"(dv0_0[4]),
          "+r"(dv0_0[5]), "+r"(dv0_0[6]), "+r"(dv0_0[7]), "+r"(dv0_0[8]), "+r"(dv0_0[9]),
          "+r"(dv0_0[10]), "+r"(dv0_0[11]), "+r"(dv0_0[12]), "+r"(dv0_0[13]), "+r"(dv0_0[14]),
          "+r"(dv0_0[15])
        : "l"(_wgmma_b_0 + 2), "l"(_wgmma_b_5 + 2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 0, 1, 1;\n}\n"
        : "+r"(dv0_1[0]), "+r"(dv0_1[1]), "+r"(dv0_1[2]), "+r"(dv0_1[3]), "+r"(dv0_1[4]),
          "+r"(dv0_1[5]), "+r"(dv0_1[6]), "+r"(dv0_1[7]), "+r"(dv0_1[8]), "+r"(dv0_1[9]),
          "+r"(dv0_1[10]), "+r"(dv0_1[11]), "+r"(dv0_1[12]), "+r"(dv0_1[13]), "+r"(dv0_1[14]),
          "+r"(dv0_1[15])
        : "l"(_wgmma_b_0), "l"(_wgmma_b_5 + 4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1;\n}\n"
        : "+r"(dv0_1[0]), "+r"(dv0_1[1]), "+r"(dv0_1[2]), "+r"(dv0_1[3]), "+r"(dv0_1[4]),
          "+r"(dv0_1[5]), "+r"(dv0_1[6]), "+r"(dv0_1[7]), "+r"(dv0_1[8]), "+r"(dv0_1[9]),
          "+r"(dv0_1[10]), "+r"(dv0_1[11]), "+r"(dv0_1[12]), "+r"(dv0_1[13]), "+r"(dv0_1[14]),
          "+r"(dv0_1[15])
        : "l"(_wgmma_b_0 + 2), "l"(_wgmma_b_5 + 6)
        : "memory");
    uint64_t _wgmma_desc_32 =
        (((uint64_t)(((kv8_addr + (unsigned int)(stage * 32768) + 24576)) >> 4) & 0x3FFFULL) |
         ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_6 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_32 >> 32)) << 32) |
                          (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_32);
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 0, 1, 1;\n}\n"
        : "+r"(dv1_0[0]), "+r"(dv1_0[1]), "+r"(dv1_0[2]), "+r"(dv1_0[3]), "+r"(dv1_0[4]),
          "+r"(dv1_0[5]), "+r"(dv1_0[6]), "+r"(dv1_0[7]), "+r"(dv1_0[8]), "+r"(dv1_0[9]),
          "+r"(dv1_0[10]), "+r"(dv1_0[11]), "+r"(dv1_0[12]), "+r"(dv1_0[13]), "+r"(dv1_0[14]),
          "+r"(dv1_0[15])
        : "l"(_wgmma_b_0), "l"(_wgmma_b_6)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1;\n}\n"
        : "+r"(dv1_0[0]), "+r"(dv1_0[1]), "+r"(dv1_0[2]), "+r"(dv1_0[3]), "+r"(dv1_0[4]),
          "+r"(dv1_0[5]), "+r"(dv1_0[6]), "+r"(dv1_0[7]), "+r"(dv1_0[8]), "+r"(dv1_0[9]),
          "+r"(dv1_0[10]), "+r"(dv1_0[11]), "+r"(dv1_0[12]), "+r"(dv1_0[13]), "+r"(dv1_0[14]),
          "+r"(dv1_0[15])
        : "l"(_wgmma_b_0 + 2), "l"(_wgmma_b_6 + 2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 0, 1, 1;\n}\n"
        : "+r"(dv1_1[0]), "+r"(dv1_1[1]), "+r"(dv1_1[2]), "+r"(dv1_1[3]), "+r"(dv1_1[4]),
          "+r"(dv1_1[5]), "+r"(dv1_1[6]), "+r"(dv1_1[7]), "+r"(dv1_1[8]), "+r"(dv1_1[9]),
          "+r"(dv1_1[10]), "+r"(dv1_1[11]), "+r"(dv1_1[12]), "+r"(dv1_1[13]), "+r"(dv1_1[14]),
          "+r"(dv1_1[15])
        : "l"(_wgmma_b_0), "l"(_wgmma_b_6 + 4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1;\n}\n"
        : "+r"(dv1_1[0]), "+r"(dv1_1[1]), "+r"(dv1_1[2]), "+r"(dv1_1[3]), "+r"(dv1_1[4]),
          "+r"(dv1_1[5]), "+r"(dv1_1[6]), "+r"(dv1_1[7]), "+r"(dv1_1[8]), "+r"(dv1_1[9]),
          "+r"(dv1_1[10]), "+r"(dv1_1[11]), "+r"(dv1_1[12]), "+r"(dv1_1[13]), "+r"(dv1_1[14]),
          "+r"(dv1_1[15])
        : "l"(_wgmma_b_0 + 2), "l"(_wgmma_b_6 + 6)
        : "memory");
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    if ((bl & 1) != 0) {
      int limv[2];
      int r_c = 2 * cq;
      int lim_ = qpos1 + (r_c >> CAKE_LOG2_GROUP) - pbase;
      if (lim_ > 128) {
        lim_ = 128;
      }
      if (lim_ < 0) {
        lim_ = 0;
      }
      if ((mask_j >> (unsigned int)(r_c >> CAKE_LOG2_GROUP) & 1) == 0) {
        lim_ = 0;
      }
      limv[0] = lim_;
      tmax[0] = -CUDART_INF_F;
      int r_c_0 = 2 * cq + 1;
      int lim__1 = qpos1 + (r_c_0 >> CAKE_LOG2_GROUP) - pbase;
      if (lim__1 > 128) {
        lim__1 = 128;
      }
      if (lim__1 < 0) {
        lim__1 = 0;
      }
      if ((mask_j >> (unsigned int)(r_c_0 >> CAKE_LOG2_GROUP) & 1) == 0) {
        lim__1 = 0;
      }
      limv[1] = lim__1;
      tmax[1] = -CUDART_INF_F;
      float v0 = -CUDART_INF_F;
      if (key0 < limv[0]) {
        v0 = d_s0[0] * softmax_scale_log2;
      }
      d_s0[0] = v0;
      float v1 = -CUDART_INF_F;
      if (limv[0] > 64 + key0) {
        v1 = d_s1[0] * softmax_scale_log2;
      }
      d_s1[0] = v1;
      tmax[0] = max_noftz(tmax[0], max_noftz(v0, v1));
      float v0_2 = -CUDART_INF_F;
      if (key0 < limv[1]) {
        v0_2 = d_s0[1] * softmax_scale_log2;
      }
      d_s0[1] = v0_2;
      float v1_3 = -CUDART_INF_F;
      if (limv[1] > 64 + key0) {
        v1_3 = d_s1[1] * softmax_scale_log2;
      }
      d_s1[1] = v1_3;
      tmax[1] = max_noftz(tmax[1], max_noftz(v0_2, v1_3));
      float v0_4 = -CUDART_INF_F;
      if (key1 < limv[0]) {
        v0_4 = d_s0[2] * softmax_scale_log2;
      }
      d_s0[2] = v0_4;
      float v1_5 = -CUDART_INF_F;
      if (limv[0] > 64 + key1) {
        v1_5 = d_s1[2] * softmax_scale_log2;
      }
      d_s1[2] = v1_5;
      tmax[0] = max_noftz(tmax[0], max_noftz(v0_4, v1_5));
      float v0_6 = -CUDART_INF_F;
      if (key1 < limv[1]) {
        v0_6 = d_s0[3] * softmax_scale_log2;
      }
      d_s0[3] = v0_6;
      float v1_7 = -CUDART_INF_F;
      if (limv[1] > 64 + key1) {
        v1_7 = d_s1[3] * softmax_scale_log2;
      }
      d_s1[3] = v1_7;
      tmax[1] = max_noftz(tmax[1], max_noftz(v0_6, v1_7));
      float t = tmax[0];
      float _shfl_xor_sync_33 = __shfl_xor_sync(0xFFFFFFFF, t, 4);
      t = max_noftz(t, _shfl_xor_sync_33);
      float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, t, 8);
      t = max_noftz(t, _shfl_xor_sync_34);
      float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, t, 16);
      t = max_noftz(t, _shfl_xor_sync_35);
      tmax[0] = t;
      float t_8 = tmax[1];
      float _shfl_xor_sync_36 = __shfl_xor_sync(0xFFFFFFFF, t_8, 4);
      t_8 = max_noftz(t_8, _shfl_xor_sync_36);
      float _shfl_xor_sync_37 = __shfl_xor_sync(0xFFFFFFFF, t_8, 8);
      t_8 = max_noftz(t_8, _shfl_xor_sync_37);
      float _shfl_xor_sync_38 = __shfl_xor_sync(0xFFFFFFFF, t_8, 16);
      t_8 = max_noftz(t_8, _shfl_xor_sync_38);
      tmax[1] = t_8;
      if (lane_0 < 4) {
        int hh = 2 * cq;
        red[hh * 4 + warp_1] = tmax[0];
        int hh_0 = 2 * cq + 1;
        red[hh_0 * 4 + warp_1] = tmax[1];
      }
    }
    if ((bl >> 1 & 1) != 0) {
      int limv_1[2];
      int r_c_1 = 8 + 2 * cq;
      int lim__2 = qpos1 + (r_c_1 >> CAKE_LOG2_GROUP) - pbase;
      if (lim__2 > 128) {
        lim__2 = 128;
      }
      if (lim__2 < 0) {
        lim__2 = 0;
      }
      if ((mask_j >> (unsigned int)(r_c_1 >> CAKE_LOG2_GROUP) & 1) == 0) {
        lim__2 = 0;
      }
      limv_1[0] = lim__2;
      tmax[2] = -CUDART_INF_F;
      int r_c_0_1 = 8 + 2 * cq + 1;
      int lim__1_1 = qpos1 + (r_c_0_1 >> CAKE_LOG2_GROUP) - pbase;
      if (lim__1_1 > 128) {
        lim__1_1 = 128;
      }
      if (lim__1_1 < 0) {
        lim__1_1 = 0;
      }
      if ((mask_j >> (unsigned int)(r_c_0_1 >> CAKE_LOG2_GROUP) & 1) == 0) {
        lim__1_1 = 0;
      }
      limv_1[1] = lim__1_1;
      tmax[3] = -CUDART_INF_F;
      float v0_1 = -CUDART_INF_F;
      if (key0 < limv_1[0]) {
        v0_1 = d_s0[4] * softmax_scale_log2;
      }
      d_s0[4] = v0_1;
      float v1_1 = -CUDART_INF_F;
      if (limv_1[0] > 64 + key0) {
        v1_1 = d_s1[4] * softmax_scale_log2;
      }
      d_s1[4] = v1_1;
      tmax[2] = max_noftz(tmax[2], max_noftz(v0_1, v1_1));
      float v0_2_1 = -CUDART_INF_F;
      if (key0 < limv_1[1]) {
        v0_2_1 = d_s0[5] * softmax_scale_log2;
      }
      d_s0[5] = v0_2_1;
      float v1_3_1 = -CUDART_INF_F;
      if (limv_1[1] > 64 + key0) {
        v1_3_1 = d_s1[5] * softmax_scale_log2;
      }
      d_s1[5] = v1_3_1;
      tmax[3] = max_noftz(tmax[3], max_noftz(v0_2_1, v1_3_1));
      float v0_4_1 = -CUDART_INF_F;
      if (key1 < limv_1[0]) {
        v0_4_1 = d_s0[6] * softmax_scale_log2;
      }
      d_s0[6] = v0_4_1;
      float v1_5_1 = -CUDART_INF_F;
      if (limv_1[0] > 64 + key1) {
        v1_5_1 = d_s1[6] * softmax_scale_log2;
      }
      d_s1[6] = v1_5_1;
      tmax[2] = max_noftz(tmax[2], max_noftz(v0_4_1, v1_5_1));
      float v0_6_1 = -CUDART_INF_F;
      if (key1 < limv_1[1]) {
        v0_6_1 = d_s0[7] * softmax_scale_log2;
      }
      d_s0[7] = v0_6_1;
      float v1_7_1 = -CUDART_INF_F;
      if (limv_1[1] > 64 + key1) {
        v1_7_1 = d_s1[7] * softmax_scale_log2;
      }
      d_s1[7] = v1_7_1;
      tmax[3] = max_noftz(tmax[3], max_noftz(v0_6_1, v1_7_1));
      float t_1 = tmax[2];
      float _shfl_xor_sync_39 = __shfl_xor_sync(0xFFFFFFFF, t_1, 4);
      t_1 = max_noftz(t_1, _shfl_xor_sync_39);
      float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, t_1, 8);
      t_1 = max_noftz(t_1, _shfl_xor_sync_40);
      float _shfl_xor_sync_41 = __shfl_xor_sync(0xFFFFFFFF, t_1, 16);
      t_1 = max_noftz(t_1, _shfl_xor_sync_41);
      tmax[2] = t_1;
      float t_8_1 = tmax[3];
      float _shfl_xor_sync_42 = __shfl_xor_sync(0xFFFFFFFF, t_8_1, 4);
      t_8_1 = max_noftz(t_8_1, _shfl_xor_sync_42);
      float _shfl_xor_sync_43 = __shfl_xor_sync(0xFFFFFFFF, t_8_1, 8);
      t_8_1 = max_noftz(t_8_1, _shfl_xor_sync_43);
      float _shfl_xor_sync_44 = __shfl_xor_sync(0xFFFFFFFF, t_8_1, 16);
      t_8_1 = max_noftz(t_8_1, _shfl_xor_sync_44);
      tmax[3] = t_8_1;
      if (lane_0 < 4) {
        int hh_1 = 8 + 2 * cq;
        red[hh_1 * 4 + warp_1] = tmax[2];
        int hh_0_1 = 8 + 2 * cq + 1;
        red[hh_0_1 * 4 + warp_1] = tmax[3];
      }
    }
    if ((bl >> 2 & 1) != 0) {
      int limv_2[2];
      int r_c_2 = 16 + 2 * cq;
      int lim__3 = qpos1 + (r_c_2 >> CAKE_LOG2_GROUP) - pbase;
      if (lim__3 > 128) {
        lim__3 = 128;
      }
      if (lim__3 < 0) {
        lim__3 = 0;
      }
      if ((mask_j >> (unsigned int)(r_c_2 >> CAKE_LOG2_GROUP) & 1) == 0) {
        lim__3 = 0;
      }
      limv_2[0] = lim__3;
      tmax[4] = -CUDART_INF_F;
      int r_c_0_2 = 16 + 2 * cq + 1;
      int lim__1_2 = qpos1 + (r_c_0_2 >> CAKE_LOG2_GROUP) - pbase;
      if (lim__1_2 > 128) {
        lim__1_2 = 128;
      }
      if (lim__1_2 < 0) {
        lim__1_2 = 0;
      }
      if ((mask_j >> (unsigned int)(r_c_0_2 >> CAKE_LOG2_GROUP) & 1) == 0) {
        lim__1_2 = 0;
      }
      limv_2[1] = lim__1_2;
      tmax[5] = -CUDART_INF_F;
      float v0_3 = -CUDART_INF_F;
      if (key0 < limv_2[0]) {
        v0_3 = d_s0[8] * softmax_scale_log2;
      }
      d_s0[8] = v0_3;
      float v1_2 = -CUDART_INF_F;
      if (limv_2[0] > 64 + key0) {
        v1_2 = d_s1[8] * softmax_scale_log2;
      }
      d_s1[8] = v1_2;
      tmax[4] = max_noftz(tmax[4], max_noftz(v0_3, v1_2));
      float v0_2_2 = -CUDART_INF_F;
      if (key0 < limv_2[1]) {
        v0_2_2 = d_s0[9] * softmax_scale_log2;
      }
      d_s0[9] = v0_2_2;
      float v1_3_2 = -CUDART_INF_F;
      if (limv_2[1] > 64 + key0) {
        v1_3_2 = d_s1[9] * softmax_scale_log2;
      }
      d_s1[9] = v1_3_2;
      tmax[5] = max_noftz(tmax[5], max_noftz(v0_2_2, v1_3_2));
      float v0_4_2 = -CUDART_INF_F;
      if (key1 < limv_2[0]) {
        v0_4_2 = d_s0[10] * softmax_scale_log2;
      }
      d_s0[10] = v0_4_2;
      float v1_5_2 = -CUDART_INF_F;
      if (limv_2[0] > 64 + key1) {
        v1_5_2 = d_s1[10] * softmax_scale_log2;
      }
      d_s1[10] = v1_5_2;
      tmax[4] = max_noftz(tmax[4], max_noftz(v0_4_2, v1_5_2));
      float v0_6_2 = -CUDART_INF_F;
      if (key1 < limv_2[1]) {
        v0_6_2 = d_s0[11] * softmax_scale_log2;
      }
      d_s0[11] = v0_6_2;
      float v1_7_2 = -CUDART_INF_F;
      if (limv_2[1] > 64 + key1) {
        v1_7_2 = d_s1[11] * softmax_scale_log2;
      }
      d_s1[11] = v1_7_2;
      tmax[5] = max_noftz(tmax[5], max_noftz(v0_6_2, v1_7_2));
      float t_2 = tmax[4];
      float _shfl_xor_sync_45 = __shfl_xor_sync(0xFFFFFFFF, t_2, 4);
      t_2 = max_noftz(t_2, _shfl_xor_sync_45);
      float _shfl_xor_sync_46 = __shfl_xor_sync(0xFFFFFFFF, t_2, 8);
      t_2 = max_noftz(t_2, _shfl_xor_sync_46);
      float _shfl_xor_sync_47 = __shfl_xor_sync(0xFFFFFFFF, t_2, 16);
      t_2 = max_noftz(t_2, _shfl_xor_sync_47);
      tmax[4] = t_2;
      float t_8_2 = tmax[5];
      float _shfl_xor_sync_48 = __shfl_xor_sync(0xFFFFFFFF, t_8_2, 4);
      t_8_2 = max_noftz(t_8_2, _shfl_xor_sync_48);
      float _shfl_xor_sync_49 = __shfl_xor_sync(0xFFFFFFFF, t_8_2, 8);
      t_8_2 = max_noftz(t_8_2, _shfl_xor_sync_49);
      float _shfl_xor_sync_50 = __shfl_xor_sync(0xFFFFFFFF, t_8_2, 16);
      t_8_2 = max_noftz(t_8_2, _shfl_xor_sync_50);
      tmax[5] = t_8_2;
      if (lane_0 < 4) {
        int hh_2 = 16 + 2 * cq;
        red[hh_2 * 4 + warp_1] = tmax[4];
        int hh_0_2 = 16 + 2 * cq + 1;
        red[hh_0_2 * 4 + warp_1] = tmax[5];
      }
    }
    if ((bl >> 3 & 1) != 0) {
      int limv_3[2];
      int r_c_3 = 24 + 2 * cq;
      int lim__4 = qpos1 + (r_c_3 >> CAKE_LOG2_GROUP) - pbase;
      if (lim__4 > 128) {
        lim__4 = 128;
      }
      if (lim__4 < 0) {
        lim__4 = 0;
      }
      if ((mask_j >> (unsigned int)(r_c_3 >> CAKE_LOG2_GROUP) & 1) == 0) {
        lim__4 = 0;
      }
      limv_3[0] = lim__4;
      tmax[6] = -CUDART_INF_F;
      int r_c_0_3 = 24 + 2 * cq + 1;
      int lim__1_3 = qpos1 + (r_c_0_3 >> CAKE_LOG2_GROUP) - pbase;
      if (lim__1_3 > 128) {
        lim__1_3 = 128;
      }
      if (lim__1_3 < 0) {
        lim__1_3 = 0;
      }
      if ((mask_j >> (unsigned int)(r_c_0_3 >> CAKE_LOG2_GROUP) & 1) == 0) {
        lim__1_3 = 0;
      }
      limv_3[1] = lim__1_3;
      tmax[7] = -CUDART_INF_F;
      float v0_5 = -CUDART_INF_F;
      if (key0 < limv_3[0]) {
        v0_5 = d_s0[12] * softmax_scale_log2;
      }
      d_s0[12] = v0_5;
      float v1_4 = -CUDART_INF_F;
      if (limv_3[0] > 64 + key0) {
        v1_4 = d_s1[12] * softmax_scale_log2;
      }
      d_s1[12] = v1_4;
      tmax[6] = max_noftz(tmax[6], max_noftz(v0_5, v1_4));
      float v0_2_3 = -CUDART_INF_F;
      if (key0 < limv_3[1]) {
        v0_2_3 = d_s0[13] * softmax_scale_log2;
      }
      d_s0[13] = v0_2_3;
      float v1_3_3 = -CUDART_INF_F;
      if (limv_3[1] > 64 + key0) {
        v1_3_3 = d_s1[13] * softmax_scale_log2;
      }
      d_s1[13] = v1_3_3;
      tmax[7] = max_noftz(tmax[7], max_noftz(v0_2_3, v1_3_3));
      float v0_4_3 = -CUDART_INF_F;
      if (key1 < limv_3[0]) {
        v0_4_3 = d_s0[14] * softmax_scale_log2;
      }
      d_s0[14] = v0_4_3;
      float v1_5_3 = -CUDART_INF_F;
      if (limv_3[0] > 64 + key1) {
        v1_5_3 = d_s1[14] * softmax_scale_log2;
      }
      d_s1[14] = v1_5_3;
      tmax[6] = max_noftz(tmax[6], max_noftz(v0_4_3, v1_5_3));
      float v0_6_3 = -CUDART_INF_F;
      if (key1 < limv_3[1]) {
        v0_6_3 = d_s0[15] * softmax_scale_log2;
      }
      d_s0[15] = v0_6_3;
      float v1_7_3 = -CUDART_INF_F;
      if (limv_3[1] > 64 + key1) {
        v1_7_3 = d_s1[15] * softmax_scale_log2;
      }
      d_s1[15] = v1_7_3;
      tmax[7] = max_noftz(tmax[7], max_noftz(v0_6_3, v1_7_3));
      float t_3 = tmax[6];
      float _shfl_xor_sync_51 = __shfl_xor_sync(0xFFFFFFFF, t_3, 4);
      t_3 = max_noftz(t_3, _shfl_xor_sync_51);
      float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, t_3, 8);
      t_3 = max_noftz(t_3, _shfl_xor_sync_52);
      float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, t_3, 16);
      t_3 = max_noftz(t_3, _shfl_xor_sync_53);
      tmax[6] = t_3;
      float t_8_3 = tmax[7];
      float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, t_8_3, 4);
      t_8_3 = max_noftz(t_8_3, _shfl_xor_sync_54);
      float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, t_8_3, 8);
      t_8_3 = max_noftz(t_8_3, _shfl_xor_sync_55);
      float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, t_8_3, 16);
      t_8_3 = max_noftz(t_8_3, _shfl_xor_sync_56);
      tmax[7] = t_8_3;
      if (lane_0 < 4) {
        int hh_3 = 24 + 2 * cq;
        red[hh_3 * 4 + warp_1] = tmax[6];
        int hh_0_3 = 24 + 2 * cq + 1;
        red[hh_0_3 * 4 + warp_1] = tmax[7];
      }
    }
    asm volatile("barrier.sync 1, 128;" ::: "memory");
    if ((bl & 1) != 0) {
      int hh2 = 2 * cq;
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])),
                     "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
                   : "r"(red_addr + (unsigned int)(hh2 * 16)));
      float page_max;
      asm("max.f32 %0, %1, %2;"
          : "=f"(page_max)
          : "f"(max_noftz(rv[0], rv[1])), "f"(max_noftz(rv[2], rv[3])));
      float mnew;
      asm("max.f32 %0, %1, %2;" : "=f"(mnew) : "f"(mrun[0]), "f"(page_max));
      float corr = 0.0f;
      if (mrun[0] != -CUDART_INF_F) {
        float _exp2_57 = approx_exp2(mrun[0] - mnew);
        corr = _exp2_57;
      }
      d_o0[0] = d_o0[0] * corr;
      d_o0[2] = d_o0[2] * corr;
      d_o1[0] = d_o1[0] * corr;
      d_o1[2] = d_o1[2] * corr;
      lp[0] = lp[0] * corr;
      mrun[0] = mnew;
      int hh2_0 = 2 * cq + 1;
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])),
                     "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
                   : "r"(red_addr + (unsigned int)(hh2_0 * 16)));
      float page_max_1;
      asm("max.f32 %0, %1, %2;"
          : "=f"(page_max_1)
          : "f"(max_noftz(rv[0], rv[1])), "f"(max_noftz(rv[2], rv[3])));
      float mnew_2;
      asm("max.f32 %0, %1, %2;" : "=f"(mnew_2) : "f"(mrun[1]), "f"(page_max_1));
      float corr_3 = 0.0f;
      if (mrun[1] != -CUDART_INF_F) {
        float _exp2_58 = approx_exp2(mrun[1] - mnew_2);
        corr_3 = _exp2_58;
      }
      d_o0[1] = d_o0[1] * corr_3;
      d_o0[3] = d_o0[3] * corr_3;
      d_o1[1] = d_o1[1] * corr_3;
      d_o1[3] = d_o1[3] * corr_3;
      lp[1] = lp[1] * corr_3;
      mrun[1] = mnew_2;
    }
    if ((bl >> 1 & 1) != 0) {
      int hh2_1 = 8 + 2 * cq;
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])),
                     "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
                   : "r"(red_addr + (unsigned int)(hh2_1 * 16)));
      float page_max_2;
      asm("max.f32 %0, %1, %2;"
          : "=f"(page_max_2)
          : "f"(max_noftz(rv[0], rv[1])), "f"(max_noftz(rv[2], rv[3])));
      float mnew_1;
      asm("max.f32 %0, %1, %2;" : "=f"(mnew_1) : "f"(mrun[2]), "f"(page_max_2));
      float corr_1 = 0.0f;
      if (mrun[2] != -CUDART_INF_F) {
        float _exp2_59 = approx_exp2(mrun[2] - mnew_1);
        corr_1 = _exp2_59;
      }
      d_o0[4] = d_o0[4] * corr_1;
      d_o0[6] = d_o0[6] * corr_1;
      d_o1[4] = d_o1[4] * corr_1;
      d_o1[6] = d_o1[6] * corr_1;
      lp[2] = lp[2] * corr_1;
      mrun[2] = mnew_1;
      int hh2_0_1 = 8 + 2 * cq + 1;
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])),
                     "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
                   : "r"(red_addr + (unsigned int)(hh2_0_1 * 16)));
      float page_max_1_1;
      asm("max.f32 %0, %1, %2;"
          : "=f"(page_max_1_1)
          : "f"(max_noftz(rv[0], rv[1])), "f"(max_noftz(rv[2], rv[3])));
      float mnew_2_1;
      asm("max.f32 %0, %1, %2;" : "=f"(mnew_2_1) : "f"(mrun[3]), "f"(page_max_1_1));
      float corr_3_1 = 0.0f;
      if (mrun[3] != -CUDART_INF_F) {
        float _exp2_60 = approx_exp2(mrun[3] - mnew_2_1);
        corr_3_1 = _exp2_60;
      }
      d_o0[5] = d_o0[5] * corr_3_1;
      d_o0[7] = d_o0[7] * corr_3_1;
      d_o1[5] = d_o1[5] * corr_3_1;
      d_o1[7] = d_o1[7] * corr_3_1;
      lp[3] = lp[3] * corr_3_1;
      mrun[3] = mnew_2_1;
    }
    if ((bl >> 2 & 1) != 0) {
      int hh2_2 = 16 + 2 * cq;
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])),
                     "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
                   : "r"(red_addr + (unsigned int)(hh2_2 * 16)));
      float page_max_3;
      asm("max.f32 %0, %1, %2;"
          : "=f"(page_max_3)
          : "f"(max_noftz(rv[0], rv[1])), "f"(max_noftz(rv[2], rv[3])));
      float mnew_3;
      asm("max.f32 %0, %1, %2;" : "=f"(mnew_3) : "f"(mrun[4]), "f"(page_max_3));
      float corr_2 = 0.0f;
      if (mrun[4] != -CUDART_INF_F) {
        float _exp2_61 = approx_exp2(mrun[4] - mnew_3);
        corr_2 = _exp2_61;
      }
      d_o0[8] = d_o0[8] * corr_2;
      d_o0[10] = d_o0[10] * corr_2;
      d_o1[8] = d_o1[8] * corr_2;
      d_o1[10] = d_o1[10] * corr_2;
      lp[4] = lp[4] * corr_2;
      mrun[4] = mnew_3;
      int hh2_0_2 = 16 + 2 * cq + 1;
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])),
                     "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
                   : "r"(red_addr + (unsigned int)(hh2_0_2 * 16)));
      float page_max_1_2;
      asm("max.f32 %0, %1, %2;"
          : "=f"(page_max_1_2)
          : "f"(max_noftz(rv[0], rv[1])), "f"(max_noftz(rv[2], rv[3])));
      float mnew_2_2;
      asm("max.f32 %0, %1, %2;" : "=f"(mnew_2_2) : "f"(mrun[5]), "f"(page_max_1_2));
      float corr_3_2 = 0.0f;
      if (mrun[5] != -CUDART_INF_F) {
        float _exp2_62 = approx_exp2(mrun[5] - mnew_2_2);
        corr_3_2 = _exp2_62;
      }
      d_o0[9] = d_o0[9] * corr_3_2;
      d_o0[11] = d_o0[11] * corr_3_2;
      d_o1[9] = d_o1[9] * corr_3_2;
      d_o1[11] = d_o1[11] * corr_3_2;
      lp[5] = lp[5] * corr_3_2;
      mrun[5] = mnew_2_2;
    }
    if ((bl >> 3 & 1) != 0) {
      int hh2_3 = 24 + 2 * cq;
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])),
                     "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
                   : "r"(red_addr + (unsigned int)(hh2_3 * 16)));
      float page_max_4;
      asm("max.f32 %0, %1, %2;"
          : "=f"(page_max_4)
          : "f"(max_noftz(rv[0], rv[1])), "f"(max_noftz(rv[2], rv[3])));
      float mnew_4;
      asm("max.f32 %0, %1, %2;" : "=f"(mnew_4) : "f"(mrun[6]), "f"(page_max_4));
      float corr_4 = 0.0f;
      if (mrun[6] != -CUDART_INF_F) {
        float _exp2_63 = approx_exp2(mrun[6] - mnew_4);
        corr_4 = _exp2_63;
      }
      d_o0[12] = d_o0[12] * corr_4;
      d_o0[14] = d_o0[14] * corr_4;
      d_o1[12] = d_o1[12] * corr_4;
      d_o1[14] = d_o1[14] * corr_4;
      lp[6] = lp[6] * corr_4;
      mrun[6] = mnew_4;
      int hh2_0_3 = 24 + 2 * cq + 1;
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])),
                     "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
                   : "r"(red_addr + (unsigned int)(hh2_0_3 * 16)));
      float page_max_1_3;
      asm("max.f32 %0, %1, %2;"
          : "=f"(page_max_1_3)
          : "f"(max_noftz(rv[0], rv[1])), "f"(max_noftz(rv[2], rv[3])));
      float mnew_2_3;
      asm("max.f32 %0, %1, %2;" : "=f"(mnew_2_3) : "f"(mrun[7]), "f"(page_max_1_3));
      float corr_3_3 = 0.0f;
      if (mrun[7] != -CUDART_INF_F) {
        float _exp2_64 = approx_exp2(mrun[7] - mnew_2_3);
        corr_3_3 = _exp2_64;
      }
      d_o0[13] = d_o0[13] * corr_3_3;
      d_o0[15] = d_o0[15] * corr_3_3;
      d_o1[13] = d_o1[13] * corr_3_3;
      d_o1[15] = d_o1[15] * corr_3_3;
      lp[7] = lp[7] * corr_3_3;
      mrun[7] = mnew_2_3;
    }
    if ((bl & 1) != 0) {
      int hh3 = 2 * cq;
      int pcol = hh3 * 128 + (key0 * 2 ^ (hh3 & 7) << 4);
      float msub = mrun[0];
      if (msub == -CUDART_INF_F) {
        msub = 0.0f;
      }
      float p0 = approx_exp2(d_s0[0] - msub);
      float p1 = approx_exp2(d_s1[0] - msub);
      lp[0] = lp[0] + (p0 + p1);
      {
        __half _hval_65 = __float2half_rn(p0);
        uint16_t _bits_65 = *(uint16_t*)&_hval_65;
        uint32_t _addr_65 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol);
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_65), "h"(_bits_65) : "memory");
      }
      {
        __half _hval_66 = __float2half_rn(p1);
        uint16_t _bits_66 = *(uint16_t*)&_hval_66;
        uint32_t _addr_66 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol));
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_66), "h"(_bits_66) : "memory");
      }
      int hh3_0 = 2 * cq + 1;
      int pcol_1 = hh3_0 * 128 + (key0 * 2 ^ (hh3_0 & 7) << 4);
      float msub_2 = mrun[1];
      if (msub_2 == -CUDART_INF_F) {
        msub_2 = 0.0f;
      }
      float p0_3 = approx_exp2(d_s0[1] - msub_2);
      float p1_4 = approx_exp2(d_s1[1] - msub_2);
      lp[1] = lp[1] + (p0_3 + p1_4);
      {
        __half _hval_67 = __float2half_rn(p0_3);
        uint16_t _bits_67 = *(uint16_t*)&_hval_67;
        uint32_t _addr_67 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol_1);
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_67), "h"(_bits_67) : "memory");
      }
      {
        __half _hval_68 = __float2half_rn(p1_4);
        uint16_t _bits_68 = *(uint16_t*)&_hval_68;
        uint32_t _addr_68 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol_1));
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_68), "h"(_bits_68) : "memory");
      }
      int hh3_5 = 2 * cq;
      int pcol_6 = hh3_5 * 128 + (key1 * 2 ^ (hh3_5 & 7) << 4);
      float msub_7 = mrun[0];
      if (msub_7 == -CUDART_INF_F) {
        msub_7 = 0.0f;
      }
      float p0_8 = approx_exp2(d_s0[2] - msub_7);
      float p1_9 = approx_exp2(d_s1[2] - msub_7);
      lp[0] = lp[0] + (p0_8 + p1_9);
      {
        __half _hval_69 = __float2half_rn(p0_8);
        uint16_t _bits_69 = *(uint16_t*)&_hval_69;
        uint32_t _addr_69 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol_6);
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_69), "h"(_bits_69) : "memory");
      }
      {
        __half _hval_70 = __float2half_rn(p1_9);
        uint16_t _bits_70 = *(uint16_t*)&_hval_70;
        uint32_t _addr_70 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol_6));
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_70), "h"(_bits_70) : "memory");
      }
      int hh3_10 = 2 * cq + 1;
      int pcol_11 = hh3_10 * 128 + (key1 * 2 ^ (hh3_10 & 7) << 4);
      float msub_12 = mrun[1];
      if (msub_12 == -CUDART_INF_F) {
        msub_12 = 0.0f;
      }
      float p0_13 = approx_exp2(d_s0[3] - msub_12);
      float p1_14 = approx_exp2(d_s1[3] - msub_12);
      lp[1] = lp[1] + (p0_13 + p1_14);
      {
        __half _hval_71 = __float2half_rn(p0_13);
        uint16_t _bits_71 = *(uint16_t*)&_hval_71;
        uint32_t _addr_71 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol_11);
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_71), "h"(_bits_71) : "memory");
      }
      {
        __half _hval_72 = __float2half_rn(p1_14);
        uint16_t _bits_72 = *(uint16_t*)&_hval_72;
        uint32_t _addr_72 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol_11));
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_72), "h"(_bits_72) : "memory");
      }
    }
    if ((bl & 1) == 0) {
      if ((prev_bl & 1) != 0) {
        int hh3z = 2 * cq;
        int pcolz = hh3z * 128 + (key0 * 2 ^ (hh3z & 7) << 4);
        {
          __half _hval_73 = __float2half_rn(0.0f);
          uint16_t _bits_73 = *(uint16_t*)&_hval_73;
          uint32_t _addr_73 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz);
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_73), "h"(_bits_73) : "memory");
        }
        {
          __half _hval_74 = __float2half_rn(0.0f);
          uint16_t _bits_74 = *(uint16_t*)&_hval_74;
          uint32_t _addr_74 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz));
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_74), "h"(_bits_74) : "memory");
        }
        int hh3z_0 = 2 * cq + 1;
        int pcolz_1 = hh3z_0 * 128 + (key0 * 2 ^ (hh3z_0 & 7) << 4);
        {
          __half _hval_75 = __float2half_rn(0.0f);
          uint16_t _bits_75 = *(uint16_t*)&_hval_75;
          uint32_t _addr_75 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz_1);
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_75), "h"(_bits_75) : "memory");
        }
        {
          __half _hval_76 = __float2half_rn(0.0f);
          uint16_t _bits_76 = *(uint16_t*)&_hval_76;
          uint32_t _addr_76 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz_1));
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_76), "h"(_bits_76) : "memory");
        }
        int hh3z_2 = 2 * cq;
        int pcolz_3 = hh3z_2 * 128 + (key1 * 2 ^ (hh3z_2 & 7) << 4);
        {
          __half _hval_77 = __float2half_rn(0.0f);
          uint16_t _bits_77 = *(uint16_t*)&_hval_77;
          uint32_t _addr_77 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz_3);
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_77), "h"(_bits_77) : "memory");
        }
        {
          __half _hval_78 = __float2half_rn(0.0f);
          uint16_t _bits_78 = *(uint16_t*)&_hval_78;
          uint32_t _addr_78 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz_3));
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_78), "h"(_bits_78) : "memory");
        }
        int hh3z_4 = 2 * cq + 1;
        int pcolz_5 = hh3z_4 * 128 + (key1 * 2 ^ (hh3z_4 & 7) << 4);
        {
          __half _hval_79 = __float2half_rn(0.0f);
          uint16_t _bits_79 = *(uint16_t*)&_hval_79;
          uint32_t _addr_79 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz_5);
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_79), "h"(_bits_79) : "memory");
        }
        {
          __half _hval_80 = __float2half_rn(0.0f);
          uint16_t _bits_80 = *(uint16_t*)&_hval_80;
          uint32_t _addr_80 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz_5));
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_80), "h"(_bits_80) : "memory");
        }
      }
    }
    if ((bl >> 1 & 1) != 0) {
      int hh3_1 = 8 + 2 * cq;
      int pcol_2 = hh3_1 * 128 + (key0 * 2 ^ (hh3_1 & 7) << 4);
      float msub_1 = mrun[2];
      if (msub_1 == -CUDART_INF_F) {
        msub_1 = 0.0f;
      }
      float p0_1 = approx_exp2(d_s0[4] - msub_1);
      float p1_1 = approx_exp2(d_s1[4] - msub_1);
      lp[2] = lp[2] + (p0_1 + p1_1);
      {
        __half _hval_81 = __float2half_rn(p0_1);
        uint16_t _bits_81 = *(uint16_t*)&_hval_81;
        uint32_t _addr_81 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol_2);
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_81), "h"(_bits_81) : "memory");
      }
      {
        __half _hval_82 = __float2half_rn(p1_1);
        uint16_t _bits_82 = *(uint16_t*)&_hval_82;
        uint32_t _addr_82 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol_2));
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_82), "h"(_bits_82) : "memory");
      }
      int hh3_0_1 = 8 + 2 * cq + 1;
      int pcol_1_1 = hh3_0_1 * 128 + (key0 * 2 ^ (hh3_0_1 & 7) << 4);
      float msub_2_1 = mrun[3];
      if (msub_2_1 == -CUDART_INF_F) {
        msub_2_1 = 0.0f;
      }
      float p0_3_1 = approx_exp2(d_s0[5] - msub_2_1);
      float p1_4_1 = approx_exp2(d_s1[5] - msub_2_1);
      lp[3] = lp[3] + (p0_3_1 + p1_4_1);
      {
        __half _hval_83 = __float2half_rn(p0_3_1);
        uint16_t _bits_83 = *(uint16_t*)&_hval_83;
        uint32_t _addr_83 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol_1_1);
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_83), "h"(_bits_83) : "memory");
      }
      {
        __half _hval_84 = __float2half_rn(p1_4_1);
        uint16_t _bits_84 = *(uint16_t*)&_hval_84;
        uint32_t _addr_84 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol_1_1));
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_84), "h"(_bits_84) : "memory");
      }
      int hh3_5_1 = 8 + 2 * cq;
      int pcol_6_1 = hh3_5_1 * 128 + (key1 * 2 ^ (hh3_5_1 & 7) << 4);
      float msub_7_1 = mrun[2];
      if (msub_7_1 == -CUDART_INF_F) {
        msub_7_1 = 0.0f;
      }
      float p0_8_1 = approx_exp2(d_s0[6] - msub_7_1);
      float p1_9_1 = approx_exp2(d_s1[6] - msub_7_1);
      lp[2] = lp[2] + (p0_8_1 + p1_9_1);
      {
        __half _hval_85 = __float2half_rn(p0_8_1);
        uint16_t _bits_85 = *(uint16_t*)&_hval_85;
        uint32_t _addr_85 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol_6_1);
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_85), "h"(_bits_85) : "memory");
      }
      {
        __half _hval_86 = __float2half_rn(p1_9_1);
        uint16_t _bits_86 = *(uint16_t*)&_hval_86;
        uint32_t _addr_86 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol_6_1));
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_86), "h"(_bits_86) : "memory");
      }
      int hh3_10_1 = 8 + 2 * cq + 1;
      int pcol_11_1 = hh3_10_1 * 128 + (key1 * 2 ^ (hh3_10_1 & 7) << 4);
      float msub_12_1 = mrun[3];
      if (msub_12_1 == -CUDART_INF_F) {
        msub_12_1 = 0.0f;
      }
      float p0_13_1 = approx_exp2(d_s0[7] - msub_12_1);
      float p1_14_1 = approx_exp2(d_s1[7] - msub_12_1);
      lp[3] = lp[3] + (p0_13_1 + p1_14_1);
      {
        __half _hval_87 = __float2half_rn(p0_13_1);
        uint16_t _bits_87 = *(uint16_t*)&_hval_87;
        uint32_t _addr_87 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol_11_1);
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_87), "h"(_bits_87) : "memory");
      }
      {
        __half _hval_88 = __float2half_rn(p1_14_1);
        uint16_t _bits_88 = *(uint16_t*)&_hval_88;
        uint32_t _addr_88 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol_11_1));
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_88), "h"(_bits_88) : "memory");
      }
    }
    if ((bl >> 1 & 1) == 0) {
      if ((prev_bl >> 1 & 1) != 0) {
        int hh3z_1 = 8 + 2 * cq;
        int pcolz_2 = hh3z_1 * 128 + (key0 * 2 ^ (hh3z_1 & 7) << 4);
        {
          __half _hval_89 = __float2half_rn(0.0f);
          uint16_t _bits_89 = *(uint16_t*)&_hval_89;
          uint32_t _addr_89 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz_2);
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_89), "h"(_bits_89) : "memory");
        }
        {
          __half _hval_90 = __float2half_rn(0.0f);
          uint16_t _bits_90 = *(uint16_t*)&_hval_90;
          uint32_t _addr_90 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz_2));
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_90), "h"(_bits_90) : "memory");
        }
        int hh3z_0_1 = 8 + 2 * cq + 1;
        int pcolz_1_1 = hh3z_0_1 * 128 + (key0 * 2 ^ (hh3z_0_1 & 7) << 4);
        {
          __half _hval_91 = __float2half_rn(0.0f);
          uint16_t _bits_91 = *(uint16_t*)&_hval_91;
          uint32_t _addr_91 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz_1_1);
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_91), "h"(_bits_91) : "memory");
        }
        {
          __half _hval_92 = __float2half_rn(0.0f);
          uint16_t _bits_92 = *(uint16_t*)&_hval_92;
          uint32_t _addr_92 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz_1_1));
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_92), "h"(_bits_92) : "memory");
        }
        int hh3z_2_1 = 8 + 2 * cq;
        int pcolz_3_1 = hh3z_2_1 * 128 + (key1 * 2 ^ (hh3z_2_1 & 7) << 4);
        {
          __half _hval_93 = __float2half_rn(0.0f);
          uint16_t _bits_93 = *(uint16_t*)&_hval_93;
          uint32_t _addr_93 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz_3_1);
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_93), "h"(_bits_93) : "memory");
        }
        {
          __half _hval_94 = __float2half_rn(0.0f);
          uint16_t _bits_94 = *(uint16_t*)&_hval_94;
          uint32_t _addr_94 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz_3_1));
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_94), "h"(_bits_94) : "memory");
        }
        int hh3z_4_1 = 8 + 2 * cq + 1;
        int pcolz_5_1 = hh3z_4_1 * 128 + (key1 * 2 ^ (hh3z_4_1 & 7) << 4);
        {
          __half _hval_95 = __float2half_rn(0.0f);
          uint16_t _bits_95 = *(uint16_t*)&_hval_95;
          uint32_t _addr_95 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz_5_1);
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_95), "h"(_bits_95) : "memory");
        }
        {
          __half _hval_96 = __float2half_rn(0.0f);
          uint16_t _bits_96 = *(uint16_t*)&_hval_96;
          uint32_t _addr_96 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz_5_1));
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_96), "h"(_bits_96) : "memory");
        }
      }
    }
    if ((bl >> 2 & 1) != 0) {
      int hh3_2 = 16 + 2 * cq;
      int pcol_3 = hh3_2 * 128 + (key0 * 2 ^ (hh3_2 & 7) << 4);
      float msub_3 = mrun[4];
      if (msub_3 == -CUDART_INF_F) {
        msub_3 = 0.0f;
      }
      float p0_2 = approx_exp2(d_s0[8] - msub_3);
      float p1_2 = approx_exp2(d_s1[8] - msub_3);
      lp[4] = lp[4] + (p0_2 + p1_2);
      {
        __half _hval_97 = __float2half_rn(p0_2);
        uint16_t _bits_97 = *(uint16_t*)&_hval_97;
        uint32_t _addr_97 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol_3);
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_97), "h"(_bits_97) : "memory");
      }
      {
        __half _hval_98 = __float2half_rn(p1_2);
        uint16_t _bits_98 = *(uint16_t*)&_hval_98;
        uint32_t _addr_98 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol_3));
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_98), "h"(_bits_98) : "memory");
      }
      int hh3_0_2 = 16 + 2 * cq + 1;
      int pcol_1_2 = hh3_0_2 * 128 + (key0 * 2 ^ (hh3_0_2 & 7) << 4);
      float msub_2_2 = mrun[5];
      if (msub_2_2 == -CUDART_INF_F) {
        msub_2_2 = 0.0f;
      }
      float p0_3_2 = approx_exp2(d_s0[9] - msub_2_2);
      float p1_4_2 = approx_exp2(d_s1[9] - msub_2_2);
      lp[5] = lp[5] + (p0_3_2 + p1_4_2);
      {
        __half _hval_99 = __float2half_rn(p0_3_2);
        uint16_t _bits_99 = *(uint16_t*)&_hval_99;
        uint32_t _addr_99 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol_1_2);
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_99), "h"(_bits_99) : "memory");
      }
      {
        __half _hval_100 = __float2half_rn(p1_4_2);
        uint16_t _bits_100 = *(uint16_t*)&_hval_100;
        uint32_t _addr_100 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol_1_2));
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_100), "h"(_bits_100) : "memory");
      }
      int hh3_5_2 = 16 + 2 * cq;
      int pcol_6_2 = hh3_5_2 * 128 + (key1 * 2 ^ (hh3_5_2 & 7) << 4);
      float msub_7_2 = mrun[4];
      if (msub_7_2 == -CUDART_INF_F) {
        msub_7_2 = 0.0f;
      }
      float p0_8_2 = approx_exp2(d_s0[10] - msub_7_2);
      float p1_9_2 = approx_exp2(d_s1[10] - msub_7_2);
      lp[4] = lp[4] + (p0_8_2 + p1_9_2);
      {
        __half _hval_101 = __float2half_rn(p0_8_2);
        uint16_t _bits_101 = *(uint16_t*)&_hval_101;
        uint32_t _addr_101 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol_6_2);
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_101), "h"(_bits_101) : "memory");
      }
      {
        __half _hval_102 = __float2half_rn(p1_9_2);
        uint16_t _bits_102 = *(uint16_t*)&_hval_102;
        uint32_t _addr_102 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol_6_2));
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_102), "h"(_bits_102) : "memory");
      }
      int hh3_10_2 = 16 + 2 * cq + 1;
      int pcol_11_2 = hh3_10_2 * 128 + (key1 * 2 ^ (hh3_10_2 & 7) << 4);
      float msub_12_2 = mrun[5];
      if (msub_12_2 == -CUDART_INF_F) {
        msub_12_2 = 0.0f;
      }
      float p0_13_2 = approx_exp2(d_s0[11] - msub_12_2);
      float p1_14_2 = approx_exp2(d_s1[11] - msub_12_2);
      lp[5] = lp[5] + (p0_13_2 + p1_14_2);
      {
        __half _hval_103 = __float2half_rn(p0_13_2);
        uint16_t _bits_103 = *(uint16_t*)&_hval_103;
        uint32_t _addr_103 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol_11_2);
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_103), "h"(_bits_103) : "memory");
      }
      {
        __half _hval_104 = __float2half_rn(p1_14_2);
        uint16_t _bits_104 = *(uint16_t*)&_hval_104;
        uint32_t _addr_104 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol_11_2));
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_104), "h"(_bits_104) : "memory");
      }
    }
    if ((bl >> 2 & 1) == 0) {
      if ((prev_bl >> 2 & 1) != 0) {
        int hh3z_3 = 16 + 2 * cq;
        int pcolz_4 = hh3z_3 * 128 + (key0 * 2 ^ (hh3z_3 & 7) << 4);
        {
          __half _hval_105 = __float2half_rn(0.0f);
          uint16_t _bits_105 = *(uint16_t*)&_hval_105;
          uint32_t _addr_105 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz_4);
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_105), "h"(_bits_105) : "memory");
        }
        {
          __half _hval_106 = __float2half_rn(0.0f);
          uint16_t _bits_106 = *(uint16_t*)&_hval_106;
          uint32_t _addr_106 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz_4));
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_106), "h"(_bits_106) : "memory");
        }
        int hh3z_0_2 = 16 + 2 * cq + 1;
        int pcolz_1_2 = hh3z_0_2 * 128 + (key0 * 2 ^ (hh3z_0_2 & 7) << 4);
        {
          __half _hval_107 = __float2half_rn(0.0f);
          uint16_t _bits_107 = *(uint16_t*)&_hval_107;
          uint32_t _addr_107 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz_1_2);
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_107), "h"(_bits_107) : "memory");
        }
        {
          __half _hval_108 = __float2half_rn(0.0f);
          uint16_t _bits_108 = *(uint16_t*)&_hval_108;
          uint32_t _addr_108 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz_1_2));
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_108), "h"(_bits_108) : "memory");
        }
        int hh3z_2_2 = 16 + 2 * cq;
        int pcolz_3_2 = hh3z_2_2 * 128 + (key1 * 2 ^ (hh3z_2_2 & 7) << 4);
        {
          __half _hval_109 = __float2half_rn(0.0f);
          uint16_t _bits_109 = *(uint16_t*)&_hval_109;
          uint32_t _addr_109 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz_3_2);
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_109), "h"(_bits_109) : "memory");
        }
        {
          __half _hval_110 = __float2half_rn(0.0f);
          uint16_t _bits_110 = *(uint16_t*)&_hval_110;
          uint32_t _addr_110 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz_3_2));
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_110), "h"(_bits_110) : "memory");
        }
        int hh3z_4_2 = 16 + 2 * cq + 1;
        int pcolz_5_2 = hh3z_4_2 * 128 + (key1 * 2 ^ (hh3z_4_2 & 7) << 4);
        {
          __half _hval_111 = __float2half_rn(0.0f);
          uint16_t _bits_111 = *(uint16_t*)&_hval_111;
          uint32_t _addr_111 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz_5_2);
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_111), "h"(_bits_111) : "memory");
        }
        {
          __half _hval_112 = __float2half_rn(0.0f);
          uint16_t _bits_112 = *(uint16_t*)&_hval_112;
          uint32_t _addr_112 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz_5_2));
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_112), "h"(_bits_112) : "memory");
        }
      }
    }
    if ((bl >> 3 & 1) != 0) {
      int hh3_3 = 24 + 2 * cq;
      int pcol_4 = hh3_3 * 128 + (key0 * 2 ^ (hh3_3 & 7) << 4);
      float msub_4 = mrun[6];
      if (msub_4 == -CUDART_INF_F) {
        msub_4 = 0.0f;
      }
      float p0_4 = approx_exp2(d_s0[12] - msub_4);
      float p1_3 = approx_exp2(d_s1[12] - msub_4);
      lp[6] = lp[6] + (p0_4 + p1_3);
      {
        __half _hval_113 = __float2half_rn(p0_4);
        uint16_t _bits_113 = *(uint16_t*)&_hval_113;
        uint32_t _addr_113 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol_4);
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_113), "h"(_bits_113) : "memory");
      }
      {
        __half _hval_114 = __float2half_rn(p1_3);
        uint16_t _bits_114 = *(uint16_t*)&_hval_114;
        uint32_t _addr_114 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol_4));
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_114), "h"(_bits_114) : "memory");
      }
      int hh3_0_3 = 24 + 2 * cq + 1;
      int pcol_1_3 = hh3_0_3 * 128 + (key0 * 2 ^ (hh3_0_3 & 7) << 4);
      float msub_2_3 = mrun[7];
      if (msub_2_3 == -CUDART_INF_F) {
        msub_2_3 = 0.0f;
      }
      float p0_3_3 = approx_exp2(d_s0[13] - msub_2_3);
      float p1_4_3 = approx_exp2(d_s1[13] - msub_2_3);
      lp[7] = lp[7] + (p0_3_3 + p1_4_3);
      {
        __half _hval_115 = __float2half_rn(p0_3_3);
        uint16_t _bits_115 = *(uint16_t*)&_hval_115;
        uint32_t _addr_115 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol_1_3);
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_115), "h"(_bits_115) : "memory");
      }
      {
        __half _hval_116 = __float2half_rn(p1_4_3);
        uint16_t _bits_116 = *(uint16_t*)&_hval_116;
        uint32_t _addr_116 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol_1_3));
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_116), "h"(_bits_116) : "memory");
      }
      int hh3_5_3 = 24 + 2 * cq;
      int pcol_6_3 = hh3_5_3 * 128 + (key1 * 2 ^ (hh3_5_3 & 7) << 4);
      float msub_7_3 = mrun[6];
      if (msub_7_3 == -CUDART_INF_F) {
        msub_7_3 = 0.0f;
      }
      float p0_8_3 = approx_exp2(d_s0[14] - msub_7_3);
      float p1_9_3 = approx_exp2(d_s1[14] - msub_7_3);
      lp[6] = lp[6] + (p0_8_3 + p1_9_3);
      {
        __half _hval_117 = __float2half_rn(p0_8_3);
        uint16_t _bits_117 = *(uint16_t*)&_hval_117;
        uint32_t _addr_117 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol_6_3);
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_117), "h"(_bits_117) : "memory");
      }
      {
        __half _hval_118 = __float2half_rn(p1_9_3);
        uint16_t _bits_118 = *(uint16_t*)&_hval_118;
        uint32_t _addr_118 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol_6_3));
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_118), "h"(_bits_118) : "memory");
      }
      int hh3_10_3 = 24 + 2 * cq + 1;
      int pcol_11_3 = hh3_10_3 * 128 + (key1 * 2 ^ (hh3_10_3 & 7) << 4);
      float msub_12_3 = mrun[7];
      if (msub_12_3 == -CUDART_INF_F) {
        msub_12_3 = 0.0f;
      }
      float p0_13_3 = approx_exp2(d_s0[15] - msub_12_3);
      float p1_14_3 = approx_exp2(d_s1[15] - msub_12_3);
      lp[7] = lp[7] + (p0_13_3 + p1_14_3);
      {
        __half _hval_119 = __float2half_rn(p0_13_3);
        uint16_t _bits_119 = *(uint16_t*)&_hval_119;
        uint32_t _addr_119 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol_11_3);
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_119), "h"(_bits_119) : "memory");
      }
      {
        __half _hval_120 = __float2half_rn(p1_14_3);
        uint16_t _bits_120 = *(uint16_t*)&_hval_120;
        uint32_t _addr_120 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol_11_3));
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_120), "h"(_bits_120) : "memory");
      }
    }
    if ((bl >> 3 & 1) == 0) {
      if ((prev_bl >> 3 & 1) != 0) {
        int hh3z_5 = 24 + 2 * cq;
        int pcolz_6 = hh3z_5 * 128 + (key0 * 2 ^ (hh3z_5 & 7) << 4);
        {
          __half _hval_121 = __float2half_rn(0.0f);
          uint16_t _bits_121 = *(uint16_t*)&_hval_121;
          uint32_t _addr_121 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz_6);
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_121), "h"(_bits_121) : "memory");
        }
        {
          __half _hval_122 = __float2half_rn(0.0f);
          uint16_t _bits_122 = *(uint16_t*)&_hval_122;
          uint32_t _addr_122 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz_6));
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_122), "h"(_bits_122) : "memory");
        }
        int hh3z_0_3 = 24 + 2 * cq + 1;
        int pcolz_1_3 = hh3z_0_3 * 128 + (key0 * 2 ^ (hh3z_0_3 & 7) << 4);
        {
          __half _hval_123 = __float2half_rn(0.0f);
          uint16_t _bits_123 = *(uint16_t*)&_hval_123;
          uint32_t _addr_123 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz_1_3);
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_123), "h"(_bits_123) : "memory");
        }
        {
          __half _hval_124 = __float2half_rn(0.0f);
          uint16_t _bits_124 = *(uint16_t*)&_hval_124;
          uint32_t _addr_124 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz_1_3));
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_124), "h"(_bits_124) : "memory");
        }
        int hh3z_2_3 = 24 + 2 * cq;
        int pcolz_3_3 = hh3z_2_3 * 128 + (key1 * 2 ^ (hh3z_2_3 & 7) << 4);
        {
          __half _hval_125 = __float2half_rn(0.0f);
          uint16_t _bits_125 = *(uint16_t*)&_hval_125;
          uint32_t _addr_125 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz_3_3);
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_125), "h"(_bits_125) : "memory");
        }
        {
          __half _hval_126 = __float2half_rn(0.0f);
          uint16_t _bits_126 = *(uint16_t*)&_hval_126;
          uint32_t _addr_126 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz_3_3));
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_126), "h"(_bits_126) : "memory");
        }
        int hh3z_4_3 = 24 + 2 * cq + 1;
        int pcolz_5_3 = hh3z_4_3 * 128 + (key1 * 2 ^ (hh3z_4_3 & 7) << 4);
        {
          __half _hval_127 = __float2half_rn(0.0f);
          uint16_t _bits_127 = *(uint16_t*)&_hval_127;
          uint32_t _addr_127 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz_5_3);
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_127), "h"(_bits_127) : "memory");
        }
        {
          __half _hval_128 = __float2half_rn(0.0f);
          uint16_t _bits_128 = *(uint16_t*)&_hval_128;
          uint32_t _addr_128 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz_5_3));
          asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_128), "h"(_bits_128) : "memory");
        }
      }
    }
    prev_bl = bl;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("barrier.sync 1, 128;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    if (vlim < 128) {
      int tv = vlim - 2 * cq;
      int u0 = tv;
      unsigned int m0 = 4294967295;
      if (u0 < 2) {
        m0 = 65535;
      }
      if (u0 < 1) {
        m0 = 0;
      }
      int u1 = u0 - 64;
      unsigned int m1 = 4294967295;
      if (u1 < 2) {
        m1 = 65535;
      }
      if (u1 < 1) {
        m1 = 0;
      }
      dv0_0[0] = dv0_0[0] & m0;
      dv1_0[0] = dv1_0[0] & m1;
      dv0_0[1] = dv0_0[1] & m0;
      dv1_0[1] = dv1_0[1] & m1;
      dv0_1[0] = dv0_1[0] & m0;
      dv1_1[0] = dv1_1[0] & m1;
      dv0_1[1] = dv0_1[1] & m0;
      dv1_1[1] = dv1_1[1] & m1;
      int u0_0 = tv - 8;
      unsigned int m0_1 = 4294967295;
      if (u0_0 < 2) {
        m0_1 = 65535;
      }
      if (u0_0 < 1) {
        m0_1 = 0;
      }
      int u1_2 = u0_0 - 64;
      unsigned int m1_3 = 4294967295;
      if (u1_2 < 2) {
        m1_3 = 65535;
      }
      if (u1_2 < 1) {
        m1_3 = 0;
      }
      dv0_0[2] = dv0_0[2] & m0_1;
      dv1_0[2] = dv1_0[2] & m1_3;
      dv0_0[3] = dv0_0[3] & m0_1;
      dv1_0[3] = dv1_0[3] & m1_3;
      dv0_1[2] = dv0_1[2] & m0_1;
      dv1_1[2] = dv1_1[2] & m1_3;
      dv0_1[3] = dv0_1[3] & m0_1;
      dv1_1[3] = dv1_1[3] & m1_3;
      int u0_4 = tv - 16;
      unsigned int m0_5 = 4294967295;
      if (u0_4 < 2) {
        m0_5 = 65535;
      }
      if (u0_4 < 1) {
        m0_5 = 0;
      }
      int u1_6 = u0_4 - 64;
      unsigned int m1_7 = 4294967295;
      if (u1_6 < 2) {
        m1_7 = 65535;
      }
      if (u1_6 < 1) {
        m1_7 = 0;
      }
      dv0_0[4] = dv0_0[4] & m0_5;
      dv1_0[4] = dv1_0[4] & m1_7;
      dv0_0[5] = dv0_0[5] & m0_5;
      dv1_0[5] = dv1_0[5] & m1_7;
      dv0_1[4] = dv0_1[4] & m0_5;
      dv1_1[4] = dv1_1[4] & m1_7;
      dv0_1[5] = dv0_1[5] & m0_5;
      dv1_1[5] = dv1_1[5] & m1_7;
      int u0_8 = tv - 24;
      unsigned int m0_9 = 4294967295;
      if (u0_8 < 2) {
        m0_9 = 65535;
      }
      if (u0_8 < 1) {
        m0_9 = 0;
      }
      int u1_10 = u0_8 - 64;
      unsigned int m1_11 = 4294967295;
      if (u1_10 < 2) {
        m1_11 = 65535;
      }
      if (u1_10 < 1) {
        m1_11 = 0;
      }
      dv0_0[6] = dv0_0[6] & m0_9;
      dv1_0[6] = dv1_0[6] & m1_11;
      dv0_0[7] = dv0_0[7] & m0_9;
      dv1_0[7] = dv1_0[7] & m1_11;
      dv0_1[6] = dv0_1[6] & m0_9;
      dv1_1[6] = dv1_1[6] & m1_11;
      dv0_1[7] = dv0_1[7] & m0_9;
      dv1_1[7] = dv1_1[7] & m1_11;
      int u0_12 = tv - 32;
      unsigned int m0_13 = 4294967295;
      if (u0_12 < 2) {
        m0_13 = 65535;
      }
      if (u0_12 < 1) {
        m0_13 = 0;
      }
      int u1_14 = u0_12 - 64;
      unsigned int m1_15 = 4294967295;
      if (u1_14 < 2) {
        m1_15 = 65535;
      }
      if (u1_14 < 1) {
        m1_15 = 0;
      }
      dv0_0[8] = dv0_0[8] & m0_13;
      dv1_0[8] = dv1_0[8] & m1_15;
      dv0_0[9] = dv0_0[9] & m0_13;
      dv1_0[9] = dv1_0[9] & m1_15;
      dv0_1[8] = dv0_1[8] & m0_13;
      dv1_1[8] = dv1_1[8] & m1_15;
      dv0_1[9] = dv0_1[9] & m0_13;
      dv1_1[9] = dv1_1[9] & m1_15;
      int u0_16 = tv - 40;
      unsigned int m0_17 = 4294967295;
      if (u0_16 < 2) {
        m0_17 = 65535;
      }
      if (u0_16 < 1) {
        m0_17 = 0;
      }
      int u1_18 = u0_16 - 64;
      unsigned int m1_19 = 4294967295;
      if (u1_18 < 2) {
        m1_19 = 65535;
      }
      if (u1_18 < 1) {
        m1_19 = 0;
      }
      dv0_0[10] = dv0_0[10] & m0_17;
      dv1_0[10] = dv1_0[10] & m1_19;
      dv0_0[11] = dv0_0[11] & m0_17;
      dv1_0[11] = dv1_0[11] & m1_19;
      dv0_1[10] = dv0_1[10] & m0_17;
      dv1_1[10] = dv1_1[10] & m1_19;
      dv0_1[11] = dv0_1[11] & m0_17;
      dv1_1[11] = dv1_1[11] & m1_19;
      int u0_20 = tv - 48;
      unsigned int m0_21 = 4294967295;
      if (u0_20 < 2) {
        m0_21 = 65535;
      }
      if (u0_20 < 1) {
        m0_21 = 0;
      }
      int u1_22 = u0_20 - 64;
      unsigned int m1_23 = 4294967295;
      if (u1_22 < 2) {
        m1_23 = 65535;
      }
      if (u1_22 < 1) {
        m1_23 = 0;
      }
      dv0_0[12] = dv0_0[12] & m0_21;
      dv1_0[12] = dv1_0[12] & m1_23;
      dv0_0[13] = dv0_0[13] & m0_21;
      dv1_0[13] = dv1_0[13] & m1_23;
      dv0_1[12] = dv0_1[12] & m0_21;
      dv1_1[12] = dv1_1[12] & m1_23;
      dv0_1[13] = dv0_1[13] & m0_21;
      dv1_1[13] = dv1_1[13] & m1_23;
      int u0_24 = tv - 56;
      unsigned int m0_25 = 4294967295;
      if (u0_24 < 2) {
        m0_25 = 65535;
      }
      if (u0_24 < 1) {
        m0_25 = 0;
      }
      int u1_26 = u0_24 - 64;
      unsigned int m1_27 = 4294967295;
      if (u1_26 < 2) {
        m1_27 = 65535;
      }
      if (u1_26 < 1) {
        m1_27 = 0;
      }
      dv0_0[14] = dv0_0[14] & m0_25;
      dv1_0[14] = dv1_0[14] & m1_27;
      dv0_0[15] = dv0_0[15] & m0_25;
      dv1_0[15] = dv1_0[15] & m1_27;
      dv0_1[14] = dv0_1[14] & m0_25;
      dv1_1[14] = dv1_1[14] & m1_27;
      dv0_1[15] = dv0_1[15] & m0_25;
      dv1_1[15] = dv1_1[15] & m1_27;
    }
    av0[0] = dv0_0[0] + zero_u32;
    av1[0] = dv1_0[0] + zero_u32;
    av0[1] = dv0_0[1] + zero_u32;
    av1[1] = dv1_0[1] + zero_u32;
    av0[2] = dv0_0[2] + zero_u32;
    av1[2] = dv1_0[2] + zero_u32;
    av0[3] = dv0_0[3] + zero_u32;
    av1[3] = dv1_0[3] + zero_u32;
    av0[4] = dv0_0[4] + zero_u32;
    av1[4] = dv1_0[4] + zero_u32;
    av0[5] = dv0_0[5] + zero_u32;
    av1[5] = dv1_0[5] + zero_u32;
    av0[6] = dv0_0[6] + zero_u32;
    av1[6] = dv1_0[6] + zero_u32;
    av0[7] = dv0_0[7] + zero_u32;
    av1[7] = dv1_0[7] + zero_u32;
    av0[8] = dv0_0[8] + zero_u32;
    av1[8] = dv1_0[8] + zero_u32;
    av0[9] = dv0_0[9] + zero_u32;
    av1[9] = dv1_0[9] + zero_u32;
    av0[10] = dv0_0[10] + zero_u32;
    av1[10] = dv1_0[10] + zero_u32;
    av0[11] = dv0_0[11] + zero_u32;
    av1[11] = dv1_0[11] + zero_u32;
    av0[12] = dv0_0[12] + zero_u32;
    av1[12] = dv1_0[12] + zero_u32;
    av0[13] = dv0_0[13] + zero_u32;
    av1[13] = dv1_0[13] + zero_u32;
    av0[14] = dv0_0[14] + zero_u32;
    av1[14] = dv1_0[14] + zero_u32;
    av0[15] = dv0_0[15] + zero_u32;
    av1[15] = dv1_0[15] + zero_u32;
    av0[16] = dv0_1[0] + zero_u32;
    av1[16] = dv1_1[0] + zero_u32;
    av0[17] = dv0_1[1] + zero_u32;
    av1[17] = dv1_1[1] + zero_u32;
    av0[18] = dv0_1[2] + zero_u32;
    av1[18] = dv1_1[2] + zero_u32;
    av0[19] = dv0_1[3] + zero_u32;
    av1[19] = dv1_1[3] + zero_u32;
    av0[20] = dv0_1[4] + zero_u32;
    av1[20] = dv1_1[4] + zero_u32;
    av0[21] = dv0_1[5] + zero_u32;
    av1[21] = dv1_1[5] + zero_u32;
    av0[22] = dv0_1[6] + zero_u32;
    av1[22] = dv1_1[6] + zero_u32;
    av0[23] = dv0_1[7] + zero_u32;
    av1[23] = dv1_1[7] + zero_u32;
    av0[24] = dv0_1[8] + zero_u32;
    av1[24] = dv1_1[8] + zero_u32;
    av0[25] = dv0_1[9] + zero_u32;
    av1[25] = dv1_1[9] + zero_u32;
    av0[26] = dv0_1[10] + zero_u32;
    av1[26] = dv1_1[10] + zero_u32;
    av0[27] = dv0_1[11] + zero_u32;
    av1[27] = dv1_1[11] + zero_u32;
    av0[28] = dv0_1[12] + zero_u32;
    av1[28] = dv1_1[12] + zero_u32;
    av0[29] = dv0_1[13] + zero_u32;
    av1[29] = dv1_1[13] + zero_u32;
    av0[30] = dv0_1[14] + zero_u32;
    av1[30] = dv1_1[14] + zero_u32;
    av0[31] = dv0_1[15] + zero_u32;
    av1[31] = dv1_1[15] + zero_u32;
    int nxt = j + 2;
    if (nxt < nblk_u) {
      int page_head_2 = meta_pg[nxt];
      if (warp == 1) {
        if (elect_sync()) {
          mbarrier_arrive_expect_tx(kv_full_addr + (stage) * 8, 32768);
          unsigned int base_2 = kv8_addr + (unsigned int)(stage * 32768);
          tma_3d_gmem2smem(base_2, (&K), 0, 0, page_head_2, kv_full_addr + (stage) * 8);
          tma_3d_gmem2smem(base_2 + 8192, (&K), 0, 64, page_head_2, kv_full_addr + (stage) * 8);
          tma_3d_gmem2smem(base_2 + 16384, (&V), 0, 0, page_head_2, kv_full_addr + (stage) * 8);
          tma_3d_gmem2smem(base_2 + 24576, (&V), 0, 64, page_head_2, kv_full_addr + (stage) * 8);
        }
      }
    }
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o0[0]), "+f"(d_o0[1]), "+f"(d_o0[2]), "+f"(d_o0[3]), "+f"(d_o0[4]), "+f"(d_o0[5]),
          "+f"(d_o0[6]), "+f"(d_o0[7]), "+f"(d_o0[8]), "+f"(d_o0[9]), "+f"(d_o0[10]),
          "+f"(d_o0[11]), "+f"(d_o0[12]), "+f"(d_o0[13]), "+f"(d_o0[14]), "+f"(d_o0[15])
        : "r"(av0[0]), "r"(av0[1]), "r"(av0[2]), "r"(av0[3]), "l"(_wgmma_b_3)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o1[0]), "+f"(d_o1[1]), "+f"(d_o1[2]), "+f"(d_o1[3]), "+f"(d_o1[4]), "+f"(d_o1[5]),
          "+f"(d_o1[6]), "+f"(d_o1[7]), "+f"(d_o1[8]), "+f"(d_o1[9]), "+f"(d_o1[10]),
          "+f"(d_o1[11]), "+f"(d_o1[12]), "+f"(d_o1[13]), "+f"(d_o1[14]), "+f"(d_o1[15])
        : "r"(av0[16]), "r"(av0[(16) + 1]), "r"(av0[(16) + 2]), "r"(av0[(16) + 3]), "l"(_wgmma_b_3)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o0[0]), "+f"(d_o0[1]), "+f"(d_o0[2]), "+f"(d_o0[3]), "+f"(d_o0[4]), "+f"(d_o0[5]),
          "+f"(d_o0[6]), "+f"(d_o0[7]), "+f"(d_o0[8]), "+f"(d_o0[9]), "+f"(d_o0[10]),
          "+f"(d_o0[11]), "+f"(d_o0[12]), "+f"(d_o0[13]), "+f"(d_o0[14]), "+f"(d_o0[15])
        : "r"(av1[0]), "r"(av1[1]), "r"(av1[2]), "r"(av1[3]), "l"(_wgmma_b_4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o1[0]), "+f"(d_o1[1]), "+f"(d_o1[2]), "+f"(d_o1[3]), "+f"(d_o1[4]), "+f"(d_o1[5]),
          "+f"(d_o1[6]), "+f"(d_o1[7]), "+f"(d_o1[8]), "+f"(d_o1[9]), "+f"(d_o1[10]),
          "+f"(d_o1[11]), "+f"(d_o1[12]), "+f"(d_o1[13]), "+f"(d_o1[14]), "+f"(d_o1[15])
        : "r"(av1[16]), "r"(av1[(16) + 1]), "r"(av1[(16) + 2]), "r"(av1[(16) + 3]), "l"(_wgmma_b_4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o0[0]), "+f"(d_o0[1]), "+f"(d_o0[2]), "+f"(d_o0[3]), "+f"(d_o0[4]), "+f"(d_o0[5]),
          "+f"(d_o0[6]), "+f"(d_o0[7]), "+f"(d_o0[8]), "+f"(d_o0[9]), "+f"(d_o0[10]),
          "+f"(d_o0[11]), "+f"(d_o0[12]), "+f"(d_o0[13]), "+f"(d_o0[14]), "+f"(d_o0[15])
        : "r"(av0[4]), "r"(av0[(4) + 1]), "r"(av0[(4) + 2]), "r"(av0[(4) + 3]), "l"(_wgmma_b_3 + 2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o1[0]), "+f"(d_o1[1]), "+f"(d_o1[2]), "+f"(d_o1[3]), "+f"(d_o1[4]), "+f"(d_o1[5]),
          "+f"(d_o1[6]), "+f"(d_o1[7]), "+f"(d_o1[8]), "+f"(d_o1[9]), "+f"(d_o1[10]),
          "+f"(d_o1[11]), "+f"(d_o1[12]), "+f"(d_o1[13]), "+f"(d_o1[14]), "+f"(d_o1[15])
        : "r"(av0[20]), "r"(av0[(20) + 1]), "r"(av0[(20) + 2]), "r"(av0[(20) + 3]),
          "l"(_wgmma_b_3 + 2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o0[0]), "+f"(d_o0[1]), "+f"(d_o0[2]), "+f"(d_o0[3]), "+f"(d_o0[4]), "+f"(d_o0[5]),
          "+f"(d_o0[6]), "+f"(d_o0[7]), "+f"(d_o0[8]), "+f"(d_o0[9]), "+f"(d_o0[10]),
          "+f"(d_o0[11]), "+f"(d_o0[12]), "+f"(d_o0[13]), "+f"(d_o0[14]), "+f"(d_o0[15])
        : "r"(av1[4]), "r"(av1[(4) + 1]), "r"(av1[(4) + 2]), "r"(av1[(4) + 3]), "l"(_wgmma_b_4 + 2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o1[0]), "+f"(d_o1[1]), "+f"(d_o1[2]), "+f"(d_o1[3]), "+f"(d_o1[4]), "+f"(d_o1[5]),
          "+f"(d_o1[6]), "+f"(d_o1[7]), "+f"(d_o1[8]), "+f"(d_o1[9]), "+f"(d_o1[10]),
          "+f"(d_o1[11]), "+f"(d_o1[12]), "+f"(d_o1[13]), "+f"(d_o1[14]), "+f"(d_o1[15])
        : "r"(av1[20]), "r"(av1[(20) + 1]), "r"(av1[(20) + 2]), "r"(av1[(20) + 3]),
          "l"(_wgmma_b_4 + 2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o0[0]), "+f"(d_o0[1]), "+f"(d_o0[2]), "+f"(d_o0[3]), "+f"(d_o0[4]), "+f"(d_o0[5]),
          "+f"(d_o0[6]), "+f"(d_o0[7]), "+f"(d_o0[8]), "+f"(d_o0[9]), "+f"(d_o0[10]),
          "+f"(d_o0[11]), "+f"(d_o0[12]), "+f"(d_o0[13]), "+f"(d_o0[14]), "+f"(d_o0[15])
        : "r"(av0[8]), "r"(av0[(8) + 1]), "r"(av0[(8) + 2]), "r"(av0[(8) + 3]), "l"(_wgmma_b_3 + 4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o1[0]), "+f"(d_o1[1]), "+f"(d_o1[2]), "+f"(d_o1[3]), "+f"(d_o1[4]), "+f"(d_o1[5]),
          "+f"(d_o1[6]), "+f"(d_o1[7]), "+f"(d_o1[8]), "+f"(d_o1[9]), "+f"(d_o1[10]),
          "+f"(d_o1[11]), "+f"(d_o1[12]), "+f"(d_o1[13]), "+f"(d_o1[14]), "+f"(d_o1[15])
        : "r"(av0[24]), "r"(av0[(24) + 1]), "r"(av0[(24) + 2]), "r"(av0[(24) + 3]),
          "l"(_wgmma_b_3 + 4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o0[0]), "+f"(d_o0[1]), "+f"(d_o0[2]), "+f"(d_o0[3]), "+f"(d_o0[4]), "+f"(d_o0[5]),
          "+f"(d_o0[6]), "+f"(d_o0[7]), "+f"(d_o0[8]), "+f"(d_o0[9]), "+f"(d_o0[10]),
          "+f"(d_o0[11]), "+f"(d_o0[12]), "+f"(d_o0[13]), "+f"(d_o0[14]), "+f"(d_o0[15])
        : "r"(av1[8]), "r"(av1[(8) + 1]), "r"(av1[(8) + 2]), "r"(av1[(8) + 3]), "l"(_wgmma_b_4 + 4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o1[0]), "+f"(d_o1[1]), "+f"(d_o1[2]), "+f"(d_o1[3]), "+f"(d_o1[4]), "+f"(d_o1[5]),
          "+f"(d_o1[6]), "+f"(d_o1[7]), "+f"(d_o1[8]), "+f"(d_o1[9]), "+f"(d_o1[10]),
          "+f"(d_o1[11]), "+f"(d_o1[12]), "+f"(d_o1[13]), "+f"(d_o1[14]), "+f"(d_o1[15])
        : "r"(av1[24]), "r"(av1[(24) + 1]), "r"(av1[(24) + 2]), "r"(av1[(24) + 3]),
          "l"(_wgmma_b_4 + 4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o0[0]), "+f"(d_o0[1]), "+f"(d_o0[2]), "+f"(d_o0[3]), "+f"(d_o0[4]), "+f"(d_o0[5]),
          "+f"(d_o0[6]), "+f"(d_o0[7]), "+f"(d_o0[8]), "+f"(d_o0[9]), "+f"(d_o0[10]),
          "+f"(d_o0[11]), "+f"(d_o0[12]), "+f"(d_o0[13]), "+f"(d_o0[14]), "+f"(d_o0[15])
        : "r"(av0[12]), "r"(av0[(12) + 1]), "r"(av0[(12) + 2]), "r"(av0[(12) + 3]),
          "l"(_wgmma_b_3 + 6)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o1[0]), "+f"(d_o1[1]), "+f"(d_o1[2]), "+f"(d_o1[3]), "+f"(d_o1[4]), "+f"(d_o1[5]),
          "+f"(d_o1[6]), "+f"(d_o1[7]), "+f"(d_o1[8]), "+f"(d_o1[9]), "+f"(d_o1[10]),
          "+f"(d_o1[11]), "+f"(d_o1[12]), "+f"(d_o1[13]), "+f"(d_o1[14]), "+f"(d_o1[15])
        : "r"(av0[28]), "r"(av0[(28) + 1]), "r"(av0[(28) + 2]), "r"(av0[(28) + 3]),
          "l"(_wgmma_b_3 + 6)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o0[0]), "+f"(d_o0[1]), "+f"(d_o0[2]), "+f"(d_o0[3]), "+f"(d_o0[4]), "+f"(d_o0[5]),
          "+f"(d_o0[6]), "+f"(d_o0[7]), "+f"(d_o0[8]), "+f"(d_o0[9]), "+f"(d_o0[10]),
          "+f"(d_o0[11]), "+f"(d_o0[12]), "+f"(d_o0[13]), "+f"(d_o0[14]), "+f"(d_o0[15])
        : "r"(av1[12]), "r"(av1[(12) + 1]), "r"(av1[(12) + 2]), "r"(av1[(12) + 3]),
          "l"(_wgmma_b_4 + 6)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o1[0]), "+f"(d_o1[1]), "+f"(d_o1[2]), "+f"(d_o1[3]), "+f"(d_o1[4]), "+f"(d_o1[5]),
          "+f"(d_o1[6]), "+f"(d_o1[7]), "+f"(d_o1[8]), "+f"(d_o1[9]), "+f"(d_o1[10]),
          "+f"(d_o1[11]), "+f"(d_o1[12]), "+f"(d_o1[13]), "+f"(d_o1[14]), "+f"(d_o1[15])
        : "r"(av1[28]), "r"(av1[(28) + 1]), "r"(av1[(28) + 2]), "r"(av1[(28) + 3]),
          "l"(_wgmma_b_4 + 6)
        : "memory");
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  float t2 = lp[0];
  float _shfl_xor_sync_129 = __shfl_xor_sync(0xFFFFFFFF, t2, 4);
  t2 = t2 + _shfl_xor_sync_129;
  float _shfl_xor_sync_130 = __shfl_xor_sync(0xFFFFFFFF, t2, 8);
  t2 = t2 + _shfl_xor_sync_130;
  float _shfl_xor_sync_131 = __shfl_xor_sync(0xFFFFFFFF, t2, 16);
  t2 = t2 + _shfl_xor_sync_131;
  lp[0] = t2;
  float t2_42 = lp[1];
  float _shfl_xor_sync_132 = __shfl_xor_sync(0xFFFFFFFF, t2_42, 4);
  t2_42 = t2_42 + _shfl_xor_sync_132;
  float _shfl_xor_sync_133 = __shfl_xor_sync(0xFFFFFFFF, t2_42, 8);
  t2_42 = t2_42 + _shfl_xor_sync_133;
  float _shfl_xor_sync_134 = __shfl_xor_sync(0xFFFFFFFF, t2_42, 16);
  t2_42 = t2_42 + _shfl_xor_sync_134;
  lp[1] = t2_42;
  float t2_43 = lp[2];
  float _shfl_xor_sync_135 = __shfl_xor_sync(0xFFFFFFFF, t2_43, 4);
  t2_43 = t2_43 + _shfl_xor_sync_135;
  float _shfl_xor_sync_136 = __shfl_xor_sync(0xFFFFFFFF, t2_43, 8);
  t2_43 = t2_43 + _shfl_xor_sync_136;
  float _shfl_xor_sync_137 = __shfl_xor_sync(0xFFFFFFFF, t2_43, 16);
  t2_43 = t2_43 + _shfl_xor_sync_137;
  lp[2] = t2_43;
  float t2_44 = lp[3];
  float _shfl_xor_sync_138 = __shfl_xor_sync(0xFFFFFFFF, t2_44, 4);
  t2_44 = t2_44 + _shfl_xor_sync_138;
  float _shfl_xor_sync_139 = __shfl_xor_sync(0xFFFFFFFF, t2_44, 8);
  t2_44 = t2_44 + _shfl_xor_sync_139;
  float _shfl_xor_sync_140 = __shfl_xor_sync(0xFFFFFFFF, t2_44, 16);
  t2_44 = t2_44 + _shfl_xor_sync_140;
  lp[3] = t2_44;
  float t2_45 = lp[4];
  float _shfl_xor_sync_141 = __shfl_xor_sync(0xFFFFFFFF, t2_45, 4);
  t2_45 = t2_45 + _shfl_xor_sync_141;
  float _shfl_xor_sync_142 = __shfl_xor_sync(0xFFFFFFFF, t2_45, 8);
  t2_45 = t2_45 + _shfl_xor_sync_142;
  float _shfl_xor_sync_143 = __shfl_xor_sync(0xFFFFFFFF, t2_45, 16);
  t2_45 = t2_45 + _shfl_xor_sync_143;
  lp[4] = t2_45;
  float t2_46 = lp[5];
  float _shfl_xor_sync_144 = __shfl_xor_sync(0xFFFFFFFF, t2_46, 4);
  t2_46 = t2_46 + _shfl_xor_sync_144;
  float _shfl_xor_sync_145 = __shfl_xor_sync(0xFFFFFFFF, t2_46, 8);
  t2_46 = t2_46 + _shfl_xor_sync_145;
  float _shfl_xor_sync_146 = __shfl_xor_sync(0xFFFFFFFF, t2_46, 16);
  t2_46 = t2_46 + _shfl_xor_sync_146;
  lp[5] = t2_46;
  float t2_47 = lp[6];
  float _shfl_xor_sync_147 = __shfl_xor_sync(0xFFFFFFFF, t2_47, 4);
  t2_47 = t2_47 + _shfl_xor_sync_147;
  float _shfl_xor_sync_148 = __shfl_xor_sync(0xFFFFFFFF, t2_47, 8);
  t2_47 = t2_47 + _shfl_xor_sync_148;
  float _shfl_xor_sync_149 = __shfl_xor_sync(0xFFFFFFFF, t2_47, 16);
  t2_47 = t2_47 + _shfl_xor_sync_149;
  lp[6] = t2_47;
  float t2_48 = lp[7];
  float _shfl_xor_sync_150 = __shfl_xor_sync(0xFFFFFFFF, t2_48, 4);
  t2_48 = t2_48 + _shfl_xor_sync_150;
  float _shfl_xor_sync_151 = __shfl_xor_sync(0xFFFFFFFF, t2_48, 8);
  t2_48 = t2_48 + _shfl_xor_sync_151;
  float _shfl_xor_sync_152 = __shfl_xor_sync(0xFFFFFFFF, t2_48, 16);
  t2_48 = t2_48 + _shfl_xor_sync_152;
  lp[7] = t2_48;
  if (lane_0 < 4) {
    int hh4 = 2 * cq;
    lred[hh4 * 4 + warp_1] = lp[0];
    int hh4_0 = 2 * cq + 1;
    lred[hh4_0 * 4 + warp_1] = lp[1];
    int hh4_1 = 8 + 2 * cq;
    lred[hh4_1 * 4 + warp_1] = lp[2];
    int hh4_2 = 8 + 2 * cq + 1;
    lred[hh4_2 * 4 + warp_1] = lp[3];
    int hh4_3 = 16 + 2 * cq;
    lred[hh4_3 * 4 + warp_1] = lp[4];
    int hh4_4 = 16 + 2 * cq + 1;
    lred[hh4_4 * 4 + warp_1] = lp[5];
    int hh4_5 = 24 + 2 * cq;
    lred[hh4_5 * 4 + warp_1] = lp[6];
    int hh4_6 = 24 + 2 * cq + 1;
    lred[hh4_6 * 4 + warp_1] = lp[7];
  }
  asm volatile("barrier.sync 1, 128;" ::: "memory");
  float lsum[8] = {0};
  int hh5 = 2 * cq;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
               : "r"(lred_addr + (unsigned int)(hh5 * 16)));
  float ls = rv[0] + rv[1] + (rv[2] + rv[3]);
  if (ls != 0.0f) {
    float _rcp_153 = approx_rcp(ls);
    lsum[0] = _rcp_153;
  }
  int hh5_49 = 2 * cq + 1;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
               : "r"(lred_addr + (unsigned int)(hh5_49 * 16)));
  float ls_50 = rv[0] + rv[1] + (rv[2] + rv[3]);
  if (ls_50 != 0.0f) {
    float _rcp_154 = approx_rcp(ls_50);
    lsum[1] = _rcp_154;
  }
  int hh5_51 = 8 + 2 * cq;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
               : "r"(lred_addr + (unsigned int)(hh5_51 * 16)));
  float ls_52 = rv[0] + rv[1] + (rv[2] + rv[3]);
  if (ls_52 != 0.0f) {
    float _rcp_155 = approx_rcp(ls_52);
    lsum[2] = _rcp_155;
  }
  int hh5_53 = 8 + 2 * cq + 1;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
               : "r"(lred_addr + (unsigned int)(hh5_53 * 16)));
  float ls_54 = rv[0] + rv[1] + (rv[2] + rv[3]);
  if (ls_54 != 0.0f) {
    float _rcp_156 = approx_rcp(ls_54);
    lsum[3] = _rcp_156;
  }
  int hh5_55 = 16 + 2 * cq;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
               : "r"(lred_addr + (unsigned int)(hh5_55 * 16)));
  float ls_56 = rv[0] + rv[1] + (rv[2] + rv[3]);
  if (ls_56 != 0.0f) {
    float _rcp_157 = approx_rcp(ls_56);
    lsum[4] = _rcp_157;
  }
  int hh5_57 = 16 + 2 * cq + 1;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
               : "r"(lred_addr + (unsigned int)(hh5_57 * 16)));
  float ls_58 = rv[0] + rv[1] + (rv[2] + rv[3]);
  if (ls_58 != 0.0f) {
    float _rcp_158 = approx_rcp(ls_58);
    lsum[5] = _rcp_158;
  }
  int hh5_59 = 24 + 2 * cq;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
               : "r"(lred_addr + (unsigned int)(hh5_59 * 16)));
  float ls_60 = rv[0] + rv[1] + (rv[2] + rv[3]);
  if (ls_60 != 0.0f) {
    float _rcp_159 = approx_rcp(ls_60);
    lsum[6] = _rcp_159;
  }
  int hh5_61 = 24 + 2 * cq + 1;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
               : "r"(lred_addr + (unsigned int)(hh5_61 * 16)));
  float ls_62 = rv[0] + rv[1] + (rv[2] + rv[3]);
  if (ls_62 != 0.0f) {
    float _rcp_160 = approx_rcp(ls_62);
    lsum[7] = _rcp_160;
  }
  int hh6 = 2 * cq;
  int dd = warp_1 * 16 + g_;
  {
    uint32_t _addr_161 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6 * 128 + dd) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_161), "f"(d_o0[0] * lsum[0]) : "memory");
  }
  {
    uint32_t _addr_162 =
        static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6 * 128 + dd + 64) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_162), "f"(d_o1[0] * lsum[0]) : "memory");
  }
  int hh6_63 = 2 * cq + 1;
  int dd_64 = warp_1 * 16 + g_;
  {
    uint32_t _addr_163 =
        static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_63 * 128 + dd_64) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_163), "f"(d_o0[1] * lsum[1]) : "memory");
  }
  {
    uint32_t _addr_164 =
        static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_63 * 128 + dd_64 + 64) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_164), "f"(d_o1[1] * lsum[1]) : "memory");
  }
  int hh6_65 = 2 * cq;
  int dd_66 = warp_1 * 16 + g_ + 8;
  {
    uint32_t _addr_165 =
        static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_65 * 128 + dd_66) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_165), "f"(d_o0[2] * lsum[0]) : "memory");
  }
  {
    uint32_t _addr_166 =
        static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_65 * 128 + dd_66 + 64) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_166), "f"(d_o1[2] * lsum[0]) : "memory");
  }
  int hh6_67 = 2 * cq + 1;
  int dd_68 = warp_1 * 16 + g_ + 8;
  {
    uint32_t _addr_167 =
        static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_67 * 128 + dd_68) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_167), "f"(d_o0[3] * lsum[1]) : "memory");
  }
  {
    uint32_t _addr_168 =
        static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_67 * 128 + dd_68 + 64) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_168), "f"(d_o1[3] * lsum[1]) : "memory");
  }
  int hh6_69 = 8 + 2 * cq;
  int dd_70 = warp_1 * 16 + g_;
  {
    uint32_t _addr_169 =
        static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_69 * 128 + dd_70) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_169), "f"(d_o0[4] * lsum[2]) : "memory");
  }
  {
    uint32_t _addr_170 =
        static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_69 * 128 + dd_70 + 64) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_170), "f"(d_o1[4] * lsum[2]) : "memory");
  }
  int hh6_71 = 8 + 2 * cq + 1;
  int dd_72 = warp_1 * 16 + g_;
  {
    uint32_t _addr_171 =
        static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_71 * 128 + dd_72) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_171), "f"(d_o0[5] * lsum[3]) : "memory");
  }
  {
    uint32_t _addr_172 =
        static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_71 * 128 + dd_72 + 64) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_172), "f"(d_o1[5] * lsum[3]) : "memory");
  }
  int hh6_73 = 8 + 2 * cq;
  int dd_74 = warp_1 * 16 + g_ + 8;
  {
    uint32_t _addr_173 =
        static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_73 * 128 + dd_74) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_173), "f"(d_o0[6] * lsum[2]) : "memory");
  }
  {
    uint32_t _addr_174 =
        static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_73 * 128 + dd_74 + 64) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_174), "f"(d_o1[6] * lsum[2]) : "memory");
  }
  int hh6_75 = 8 + 2 * cq + 1;
  int dd_76 = warp_1 * 16 + g_ + 8;
  {
    uint32_t _addr_175 =
        static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_75 * 128 + dd_76) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_175), "f"(d_o0[7] * lsum[3]) : "memory");
  }
  {
    uint32_t _addr_176 =
        static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_75 * 128 + dd_76 + 64) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_176), "f"(d_o1[7] * lsum[3]) : "memory");
  }
  int hh6_77 = 16 + 2 * cq;
  int dd_78 = warp_1 * 16 + g_;
  {
    uint32_t _addr_177 =
        static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_77 * 128 + dd_78) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_177), "f"(d_o0[8] * lsum[4]) : "memory");
  }
  {
    uint32_t _addr_178 =
        static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_77 * 128 + dd_78 + 64) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_178), "f"(d_o1[8] * lsum[4]) : "memory");
  }
  int hh6_79 = 16 + 2 * cq + 1;
  int dd_80 = warp_1 * 16 + g_;
  {
    uint32_t _addr_179 =
        static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_79 * 128 + dd_80) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_179), "f"(d_o0[9] * lsum[5]) : "memory");
  }
  {
    uint32_t _addr_180 =
        static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_79 * 128 + dd_80 + 64) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_180), "f"(d_o1[9] * lsum[5]) : "memory");
  }
  int hh6_81 = 16 + 2 * cq;
  int dd_82 = warp_1 * 16 + g_ + 8;
  {
    uint32_t _addr_181 =
        static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_81 * 128 + dd_82) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_181), "f"(d_o0[10] * lsum[4]) : "memory");
  }
  {
    uint32_t _addr_182 =
        static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_81 * 128 + dd_82 + 64) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_182), "f"(d_o1[10] * lsum[4]) : "memory");
  }
  int hh6_83 = 16 + 2 * cq + 1;
  int dd_84 = warp_1 * 16 + g_ + 8;
  {
    uint32_t _addr_183 =
        static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_83 * 128 + dd_84) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_183), "f"(d_o0[11] * lsum[5]) : "memory");
  }
  {
    uint32_t _addr_184 =
        static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_83 * 128 + dd_84 + 64) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_184), "f"(d_o1[11] * lsum[5]) : "memory");
  }
  int hh6_85 = 24 + 2 * cq;
  int dd_86 = warp_1 * 16 + g_;
  {
    uint32_t _addr_185 =
        static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_85 * 128 + dd_86) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_185), "f"(d_o0[12] * lsum[6]) : "memory");
  }
  {
    uint32_t _addr_186 =
        static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_85 * 128 + dd_86 + 64) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_186), "f"(d_o1[12] * lsum[6]) : "memory");
  }
  int hh6_87 = 24 + 2 * cq + 1;
  int dd_88 = warp_1 * 16 + g_;
  {
    uint32_t _addr_187 =
        static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_87 * 128 + dd_88) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_187), "f"(d_o0[13] * lsum[7]) : "memory");
  }
  {
    uint32_t _addr_188 =
        static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_87 * 128 + dd_88 + 64) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_188), "f"(d_o1[13] * lsum[7]) : "memory");
  }
  int hh6_89 = 24 + 2 * cq;
  int dd_90 = warp_1 * 16 + g_ + 8;
  {
    uint32_t _addr_189 =
        static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_89 * 128 + dd_90) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_189), "f"(d_o0[14] * lsum[6]) : "memory");
  }
  {
    uint32_t _addr_190 =
        static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_89 * 128 + dd_90 + 64) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_190), "f"(d_o1[14] * lsum[6]) : "memory");
  }
  int hh6_91 = 24 + 2 * cq + 1;
  int dd_92 = warp_1 * 16 + g_ + 8;
  {
    uint32_t _addr_191 =
        static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_91 * 128 + dd_92) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_191), "f"(d_o0[15] * lsum[7]) : "memory");
  }
  {
    uint32_t _addr_192 =
        static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_91 * 128 + dd_92 + 64) * 4));
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_192), "f"(d_o1[15] * lsum[7]) : "memory");
  }
  asm volatile("barrier.sync 1, 128;" ::: "memory");
  float ov[8];
  int vid3 = tid_1;
  int on = vid3 / 16;
  int oseg = vid3 - on * 16;
  int od0 = oseg * 8;
  int tl_o = on >> CAKE_LOG2_GROUP;
  if (tl_o < n_tok) {
    int hg_o = on - (tl_o << CAKE_LOG2_GROUP);
    int orow = (cu_b + tok0 + tl_o) * num_q_heads + kv_head * CAKE_GROUP + hg_o;
    long long obase = (long long)orow * 128 + (long long)od0;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&ov[0])),
                   "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 3]))
                 : "r"(kv8_addr + (unsigned int)((on * 128 + od0) * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&ov[4])),
                   "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 3]))
                 : "r"(kv8_addr + (unsigned int)((on * 128 + od0 + 4) * 4)));
    {
      __nv_bfloat162 _pk[4];
      _pk[0] = __floats2bfloat162_rn(ov[0 + 0], ov[0 + 1]);
      _pk[1] = __floats2bfloat162_rn(ov[0 + 2], ov[0 + 3]);
      _pk[2] = __floats2bfloat162_rn(ov[0 + 4], ov[0 + 5]);
      _pk[3] = __floats2bfloat162_rn(ov[0 + 6], ov[0 + 7]);
      *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + obase))[0]) =
          *reinterpret_cast<uint4*>(&_pk[0]);
    }
  }
  int vid3_93 = tid_1 + 128;
  int on_94 = vid3_93 / 16;
  int oseg_95 = vid3_93 - on_94 * 16;
  int od0_96 = oseg_95 * 8;
  int tl_o_97 = on_94 >> CAKE_LOG2_GROUP;
  if (tl_o_97 < n_tok) {
    int hg_o_1 = on_94 - (tl_o_97 << CAKE_LOG2_GROUP);
    int orow_1 = (cu_b + tok0 + tl_o_97) * num_q_heads + kv_head * CAKE_GROUP + hg_o_1;
    long long obase_1 = (long long)orow_1 * 128 + (long long)od0_96;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&ov[0])),
                   "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 3]))
                 : "r"(kv8_addr + (unsigned int)((on_94 * 128 + od0_96) * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&ov[4])),
                   "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 3]))
                 : "r"(kv8_addr + (unsigned int)((on_94 * 128 + od0_96 + 4) * 4)));
    {
      __nv_bfloat162 _pk[4];
      _pk[0] = __floats2bfloat162_rn(ov[0 + 0], ov[0 + 1]);
      _pk[1] = __floats2bfloat162_rn(ov[0 + 2], ov[0 + 3]);
      _pk[2] = __floats2bfloat162_rn(ov[0 + 4], ov[0 + 5]);
      _pk[3] = __floats2bfloat162_rn(ov[0 + 6], ov[0 + 7]);
      *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + obase_1))[0]) =
          *reinterpret_cast<uint4*>(&_pk[0]);
    }
  }
  int vid3_98 = tid_1 + 256;
  int on_99 = vid3_98 / 16;
  int oseg_100 = vid3_98 - on_99 * 16;
  int od0_101 = oseg_100 * 8;
  int tl_o_102 = on_99 >> CAKE_LOG2_GROUP;
  if (tl_o_102 < n_tok) {
    int hg_o_2 = on_99 - (tl_o_102 << CAKE_LOG2_GROUP);
    int orow_2 = (cu_b + tok0 + tl_o_102) * num_q_heads + kv_head * CAKE_GROUP + hg_o_2;
    long long obase_2 = (long long)orow_2 * 128 + (long long)od0_101;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&ov[0])),
                   "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 3]))
                 : "r"(kv8_addr + (unsigned int)((on_99 * 128 + od0_101) * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&ov[4])),
                   "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 3]))
                 : "r"(kv8_addr + (unsigned int)((on_99 * 128 + od0_101 + 4) * 4)));
    {
      __nv_bfloat162 _pk[4];
      _pk[0] = __floats2bfloat162_rn(ov[0 + 0], ov[0 + 1]);
      _pk[1] = __floats2bfloat162_rn(ov[0 + 2], ov[0 + 3]);
      _pk[2] = __floats2bfloat162_rn(ov[0 + 4], ov[0 + 5]);
      _pk[3] = __floats2bfloat162_rn(ov[0 + 6], ov[0 + 7]);
      *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + obase_2))[0]) =
          *reinterpret_cast<uint4*>(&_pk[0]);
    }
  }
  int vid3_103 = tid_1 + 384;
  int on_104 = vid3_103 / 16;
  int oseg_105 = vid3_103 - on_104 * 16;
  int od0_106 = oseg_105 * 8;
  int tl_o_107 = on_104 >> CAKE_LOG2_GROUP;
  if (tl_o_107 < n_tok) {
    int hg_o_3 = on_104 - (tl_o_107 << CAKE_LOG2_GROUP);
    int orow_3 = (cu_b + tok0 + tl_o_107) * num_q_heads + kv_head * CAKE_GROUP + hg_o_3;
    long long obase_3 = (long long)orow_3 * 128 + (long long)od0_106;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&ov[0])),
                   "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 3]))
                 : "r"(kv8_addr + (unsigned int)((on_104 * 128 + od0_106) * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&ov[4])),
                   "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 3]))
                 : "r"(kv8_addr + (unsigned int)((on_104 * 128 + od0_106 + 4) * 4)));
    {
      __nv_bfloat162 _pk[4];
      _pk[0] = __floats2bfloat162_rn(ov[0 + 0], ov[0 + 1]);
      _pk[1] = __floats2bfloat162_rn(ov[0 + 2], ov[0 + 3]);
      _pk[2] = __floats2bfloat162_rn(ov[0 + 4], ov[0 + 5]);
      _pk[3] = __floats2bfloat162_rn(ov[0 + 6], ov[0 + 7]);
      *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + obase_3))[0]) =
          *reinterpret_cast<uint4*>(&_pk[0]);
    }
  }

  // Cleanup
  __syncthreads();
}

}  // extern "C"
