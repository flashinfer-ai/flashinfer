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
//   CAKE_L2_EVICT_FIRST  1 = K/V TMA loads carry the L2 evict-first cache hint; 0 = plain loads
#ifndef CAKE_L2_EVICT_FIRST
#error "CAKE_L2_EVICT_FIRST must come from the record's compile_flags"
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
#define SMEM_Q16_STAGE_BYTES 2048
#define SMEM_Q16_STRIDE 2048
#define SMEM_P16_OFF 76800
#define SMEM_P16_STAGE_BYTES 1024
#define SMEM_P16_STRIDE 1024
#define SMEM_RED_OFF 80896
#define SMEM_RED_STAGE_BYTES 128
#define SMEM_RED_STRIDE 128
#define SMEM_LRED_OFF 81024
#define SMEM_LRED_STAGE_BYTES 128
#define SMEM_LRED_STRIDE 128
#define SMEM_FLAG_OFF 81152
#define SMEM_FLAG_STAGE_BYTES 16
#define SMEM_FLAG_STRIDE 16
#define SMEM_META_OFF 81168
#define SMEM_META_STAGE_BYTES 128
#define SMEM_META_STRIDE 128
#define SMEM_TOTAL 81408
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 2

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void CAKE_PROGRAM_KERNEL(
    unsigned int* __restrict__ Q32, const __grid_constant__ CUtensorMap K,
    const __grid_constant__ CUtensorMap V, __nv_bfloat16* __restrict__ O,
    int* __restrict__ q2k_indices, int* __restrict__ page_table, int* __restrict__ seqused_k,
    unsigned int* __restrict__ part_o, float* __restrict__ part_ml,
    unsigned int* __restrict__ counters, int total_q, int seqlen_q, int num_q_heads,
    int num_kv_heads, int max_pages, int num_chunks, float softmax_scale_log2,
    unsigned int zero_u32) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

  const int mbar_base = smem;
#define kv_full_addr (mbar_base + 0)
#define mfull_addr (mbar_base + 16)

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
  __half* p16 = reinterpret_cast<__half*>(smem_raw + 76800);
  const int p16_addr = smem + 76800;
  float* red = reinterpret_cast<float*>(smem_raw + 80896);
  const int red_addr = smem + 80896;
  float* lred = reinterpret_cast<float*>(smem_raw + 81024);
  const int lred_addr = smem + 81024;
  unsigned int* flag = reinterpret_cast<unsigned int*>(smem_raw + 81152);
  const int flag_addr = smem + 81152;
  int* meta = reinterpret_cast<int*>(smem_raw + 81168);
  const int meta_addr = smem + 81168;

  // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 3 barriers)
  // Mbarriers at smem_raw[0..24)

  if (warp == 0) {
    uint32_t leader = elect_sync();
    if (leader) {
      // kv_full: 2 barriers, init_count=1
      mbarrier_init(smem + 0, 1);
      mbarrier_init(smem + 8, 1);
      // mfull: 1 barriers, init_count=1
      mbarrier_init(smem + 16, 1);
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
  int tok = blockIdx.y;
  int gx = blockIdx.x;
  int gc = gx / 16;
  int split = gx - gc * 16;
  int kv_head = gc / num_chunks;
  int chunk = gc - kv_head * num_chunks;
  int batch = tok / seqlen_q;
  int q_in_batch = tok - batch * seqlen_q;
  int kv_len = seqused_k[batch];
  int q_pos = kv_len - seqlen_q + q_in_batch;
  int group = num_q_heads / num_kv_heads;
  int head0 = kv_head * group + chunk * 8;
  int q_row0 = tok * num_q_heads + head0;
  int heads_valid = group - chunk * 8;
  if (heads_valid > 8) {
    heads_valid = 8;
  }
  int item = (tok * num_kv_heads + kv_head) * num_chunks + chunk;
  int tid_1 = threadIdx.x;
  int lane_0 = lane;
  int warp_1 = warp;
  int g = lane_0 / 4;
  int cq = lane_0 - g * 4;
  int key0 = warp_1 * 16 + g;
  int key1 = key0 + 8;
  if (warp_1 == 1) {
    int pte = lane_0 * 32;
    if (pte < max_pages) {
      asm volatile(
          "prefetch.global.L2 [%0];" ::"l"((uint64_t)(page_table + (batch * max_pages + pte))));
    }
    int pte_0 = (lane_0 + 32) * 32;
    if (pte_0 < max_pages) {
      asm volatile(
          "prefetch.global.L2 [%0];" ::"l"((uint64_t)(page_table + (batch * max_pages + pte_0))));
    }
    int pte_1 = (lane_0 + 64) * 32;
    if (pte_1 < max_pages) {
      asm volatile(
          "prefetch.global.L2 [%0];" ::"l"((uint64_t)(page_table + (batch * max_pages + pte_1))));
    }
    int pte_2 = (lane_0 + 96) * 32;
    if (pte_2 < max_pages) {
      asm volatile(
          "prefetch.global.L2 [%0];" ::"l"((uint64_t)(page_table + (batch * max_pages + pte_2))));
    }
    if (lane_0 < 16) {
      int blk = q2k_indices[(kv_head * total_q + tok) * 16 + lane_0];
      int page = 0;
      if (blk >= 0) {
        page = page_table[batch * max_pages + blk];
        if (page < 0) {
          blk = -1;
          page = 0;
        }
      }
      meta[lane_0] = blk;
      meta[16 + lane_0] = page * num_kv_heads + kv_head;
    }
    __syncwarp();
    int slot = split;
    int blk_1 = meta[slot];
    int page_head = meta[16 + slot];
    int valid = 0;
    if (blk_1 >= 0) {
      valid = q_pos + 1 - blk_1 * 128;
      if (valid > 128) {
        valid = 128;
      }
      if (valid < 0) {
        valid = 0;
      }
    }
    if (warp == 1) {
      if (elect_sync()) {
        mbarrier_arrive_expect_tx(kv_full_addr, 32768);
#if CAKE_L2_EVICT_FIRST
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(kv8_addr),
            "l"((&K)), "r"(0), "r"(0), "r"(page_head), "r"(kv_full_addr), "l"(0x12F0000000000000ULL)
            : "memory");
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(kv8_addr + 8192),
            "l"((&K)), "r"(0), "r"(64), "r"(page_head), "r"(kv_full_addr),
            "l"(0x12F0000000000000ULL)
            : "memory");
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(kv8_addr + 16384),
            "l"((&V)), "r"(0), "r"(0), "r"(page_head), "r"(kv_full_addr), "l"(0x12F0000000000000ULL)
            : "memory");
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(kv8_addr + 24576),
            "l"((&V)), "r"(0), "r"(64), "r"(page_head), "r"(kv_full_addr),
            "l"(0x12F0000000000000ULL)
            : "memory");
#else
        tma_3d_gmem2smem(kv8_addr, (&K), 0, 0, page_head, kv_full_addr);
        tma_3d_gmem2smem(kv8_addr + 8192, (&K), 0, 64, page_head, kv_full_addr);
        tma_3d_gmem2smem(kv8_addr + 16384, (&V), 0, 0, page_head, kv_full_addr);
        tma_3d_gmem2smem(kv8_addr + 24576, (&V), 0, 64, page_head, kv_full_addr);
#endif
      }
    }
  }
  if (warp_1 != 1) {
    int sid = tid_1;
    if (warp_1 != 0) {
      sid = tid_1 - 32;
    }
    int ich_lin = sid;
    if (ich_lin < 512) {
      int irow = ich_lin / 8;
      int ich = ich_lin - irow * 8;
      int iword = irow / 4;
      unsigned int ione = 56 << 8 * (irow & 3);
      unsigned int wv[4];
      wv[0] = 0;
      if (ich * 4 == iword) {
        wv[0] = ione;
      }
      wv[1] = 0;
      if (ich * 4 + 1 == iword) {
        wv[1] = ione;
      }
      wv[2] = 0;
      if (ich * 4 + 2 == iword) {
        wv[2] = ione;
      }
      wv[3] = 0;
      if (ich * 4 + 3 == iword) {
        wv[3] = ione;
      }
      asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
                       ident8_addr + (unsigned int)(irow * 128 + (ich * 16 ^ (irow & 7) << 4))),
                   "r"(wv[0]), "r"(wv[1]), "r"(wv[2]), "r"(wv[3])
                   : "memory");
    }
    int ich_lin_0 = sid + 96;
    if (ich_lin_0 < 512) {
      int irow_1 = ich_lin_0 / 8;
      int ich_1 = ich_lin_0 - irow_1 * 8;
      int iword_1 = irow_1 / 4;
      unsigned int ione_1 = 56 << 8 * (irow_1 & 3);
      unsigned int wv_1[4];
      wv_1[0] = 0;
      if (ich_1 * 4 == iword_1) {
        wv_1[0] = ione_1;
      }
      wv_1[1] = 0;
      if (ich_1 * 4 + 1 == iword_1) {
        wv_1[1] = ione_1;
      }
      wv_1[2] = 0;
      if (ich_1 * 4 + 2 == iword_1) {
        wv_1[2] = ione_1;
      }
      wv_1[3] = 0;
      if (ich_1 * 4 + 3 == iword_1) {
        wv_1[3] = ione_1;
      }
      asm volatile(
          "st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
              ident8_addr + (unsigned int)(irow_1 * 128 + (ich_1 * 16 ^ (irow_1 & 7) << 4))),
          "r"(wv_1[0]), "r"(wv_1[1]), "r"(wv_1[2]), "r"(wv_1[3])
          : "memory");
    }
    int ich_lin_1 = sid + 192;
    if (ich_lin_1 < 512) {
      int irow_2 = ich_lin_1 / 8;
      int ich_2 = ich_lin_1 - irow_2 * 8;
      int iword_2 = irow_2 / 4;
      unsigned int ione_2 = 56 << 8 * (irow_2 & 3);
      unsigned int wv_2[4];
      wv_2[0] = 0;
      if (ich_2 * 4 == iword_2) {
        wv_2[0] = ione_2;
      }
      wv_2[1] = 0;
      if (ich_2 * 4 + 1 == iword_2) {
        wv_2[1] = ione_2;
      }
      wv_2[2] = 0;
      if (ich_2 * 4 + 2 == iword_2) {
        wv_2[2] = ione_2;
      }
      wv_2[3] = 0;
      if (ich_2 * 4 + 3 == iword_2) {
        wv_2[3] = ione_2;
      }
      asm volatile(
          "st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
              ident8_addr + (unsigned int)(irow_2 * 128 + (ich_2 * 16 ^ (irow_2 & 7) << 4))),
          "r"(wv_2[0]), "r"(wv_2[1]), "r"(wv_2[2]), "r"(wv_2[3])
          : "memory");
    }
    int ich_lin_2 = sid + 288;
    if (ich_lin_2 < 512) {
      int irow_3 = ich_lin_2 / 8;
      int ich_3 = ich_lin_2 - irow_3 * 8;
      int iword_3 = irow_3 / 4;
      unsigned int ione_3 = 56 << 8 * (irow_3 & 3);
      unsigned int wv_3[4];
      wv_3[0] = 0;
      if (ich_3 * 4 == iword_3) {
        wv_3[0] = ione_3;
      }
      wv_3[1] = 0;
      if (ich_3 * 4 + 1 == iword_3) {
        wv_3[1] = ione_3;
      }
      wv_3[2] = 0;
      if (ich_3 * 4 + 2 == iword_3) {
        wv_3[2] = ione_3;
      }
      wv_3[3] = 0;
      if (ich_3 * 4 + 3 == iword_3) {
        wv_3[3] = ione_3;
      }
      asm volatile(
          "st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
              ident8_addr + (unsigned int)(irow_3 * 128 + (ich_3 * 16 ^ (irow_3 & 7) << 4))),
          "r"(wv_3[0]), "r"(wv_3[1]), "r"(wv_3[2]), "r"(wv_3[3])
          : "memory");
    }
    int ich_lin_3 = sid + 384;
    if (ich_lin_3 < 512) {
      int irow_4 = ich_lin_3 / 8;
      int ich_4 = ich_lin_3 - irow_4 * 8;
      int iword_4 = irow_4 / 4;
      unsigned int ione_4 = 56 << 8 * (irow_4 & 3);
      unsigned int wv_4[4];
      wv_4[0] = 0;
      if (ich_4 * 4 == iword_4) {
        wv_4[0] = ione_4;
      }
      wv_4[1] = 0;
      if (ich_4 * 4 + 1 == iword_4) {
        wv_4[1] = ione_4;
      }
      wv_4[2] = 0;
      if (ich_4 * 4 + 2 == iword_4) {
        wv_4[2] = ione_4;
      }
      wv_4[3] = 0;
      if (ich_4 * 4 + 3 == iword_4) {
        wv_4[3] = ione_4;
      }
      asm volatile(
          "st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
              ident8_addr + (unsigned int)(irow_4 * 128 + (ich_4 * 16 ^ (irow_4 & 7) << 4))),
          "r"(wv_4[0]), "r"(wv_4[1]), "r"(wv_4[2]), "r"(wv_4[3])
          : "memory");
    }
    int ich_lin_4 = sid + 480;
    if (ich_lin_4 < 512) {
      int irow_5 = ich_lin_4 / 8;
      int ich_5 = ich_lin_4 - irow_5 * 8;
      int iword_5 = irow_5 / 4;
      unsigned int ione_5 = 56 << 8 * (irow_5 & 3);
      unsigned int wv_5[4];
      wv_5[0] = 0;
      if (ich_5 * 4 == iword_5) {
        wv_5[0] = ione_5;
      }
      wv_5[1] = 0;
      if (ich_5 * 4 + 1 == iword_5) {
        wv_5[1] = ione_5;
      }
      wv_5[2] = 0;
      if (ich_5 * 4 + 2 == iword_5) {
        wv_5[2] = ione_5;
      }
      wv_5[3] = 0;
      if (ich_5 * 4 + 3 == iword_5) {
        wv_5[3] = ione_5;
      }
      asm volatile(
          "st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
              ident8_addr + (unsigned int)(irow_5 * 128 + (ich_5 * 16 ^ (irow_5 & 7) << 4))),
          "r"(wv_5[0]), "r"(wv_5[1]), "r"(wv_5[2]), "r"(wv_5[3])
          : "memory");
    }
    unsigned int qw[8];
    int vid = sid;
    int qn = vid / 16;
    int qv = vid - qn * 16;
    qw[0] = 0;
    qw[1] = 0;
    qw[2] = 0;
    qw[3] = 0;
    if (qn < heads_valid) {
      int qbase = (q_row0 + qn) * 64 + qv * 4;
      qw[0] = Q32[qbase];
      qw[1] = Q32[qbase + 1];
      qw[2] = Q32[qbase + 2];
      qw[3] = Q32[qbase + 3];
    }
    int vid_5 = sid + 96;
    int qn_6 = vid_5 / 16;
    int qv_7 = vid_5 - qn_6 * 16;
    qw[4] = 0;
    qw[5] = 0;
    qw[6] = 0;
    qw[7] = 0;
    if (qn_6 < heads_valid) {
      int qbase_1 = (q_row0 + qn_6) * 64 + qv_7 * 4;
      qw[4] = Q32[qbase_1];
      qw[5] = Q32[qbase_1 + 1];
      qw[6] = Q32[qbase_1 + 2];
      qw[7] = Q32[qbase_1 + 3];
    }
    int vid2 = sid;
    if (vid2 < 128) {
      int qn2 = vid2 / 16;
      int qv2 = vid2 - qn2 * 16;
      int qslab = qv2 / 8;
      int qcol = (qv2 - qslab * 8) * 16;
      int qoff = qslab * 1024 + qn2 * 128 + (qcol ^ (qn2 & 7) << 4);
      float _cvt_f32_bf16_1;
      asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_1) : "h"((uint16_t)(qw[0])));
      float _cvt_f32_bf16_2;
      asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_2) : "h"((uint16_t)(qw[0] >> 16)));
      uint32_t _cvt_rn_f16x2_f32_0;
      asm("cvt.rn.f16x2.f32 %0, %1, %2;"
          : "=r"(_cvt_rn_f16x2_f32_0)
          : "f"(_cvt_f32_bf16_2), "f"(_cvt_f32_bf16_1));
      qw[0] = _cvt_rn_f16x2_f32_0;
      float _cvt_f32_bf16_4;
      asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_4) : "h"((uint16_t)(qw[1])));
      float _cvt_f32_bf16_5;
      asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_5) : "h"((uint16_t)(qw[1] >> 16)));
      uint32_t _cvt_rn_f16x2_f32_3;
      asm("cvt.rn.f16x2.f32 %0, %1, %2;"
          : "=r"(_cvt_rn_f16x2_f32_3)
          : "f"(_cvt_f32_bf16_5), "f"(_cvt_f32_bf16_4));
      qw[1] = _cvt_rn_f16x2_f32_3;
      float _cvt_f32_bf16_7;
      asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_7) : "h"((uint16_t)(qw[2])));
      float _cvt_f32_bf16_8;
      asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_8) : "h"((uint16_t)(qw[2] >> 16)));
      uint32_t _cvt_rn_f16x2_f32_6;
      asm("cvt.rn.f16x2.f32 %0, %1, %2;"
          : "=r"(_cvt_rn_f16x2_f32_6)
          : "f"(_cvt_f32_bf16_8), "f"(_cvt_f32_bf16_7));
      qw[2] = _cvt_rn_f16x2_f32_6;
      float _cvt_f32_bf16_10;
      asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_10) : "h"((uint16_t)(qw[3])));
      float _cvt_f32_bf16_11;
      asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_11) : "h"((uint16_t)(qw[3] >> 16)));
      uint32_t _cvt_rn_f16x2_f32_9;
      asm("cvt.rn.f16x2.f32 %0, %1, %2;"
          : "=r"(_cvt_rn_f16x2_f32_9)
          : "f"(_cvt_f32_bf16_11), "f"(_cvt_f32_bf16_10));
      qw[3] = _cvt_rn_f16x2_f32_9;
      asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(q16_addr + (unsigned int)qoff),
                   "r"(qw[0]), "r"(qw[1]), "r"(qw[2]), "r"(qw[3])
                   : "memory");
    }
    int vid2_8 = sid + 96;
    if (vid2_8 < 128) {
      int qn2_1 = vid2_8 / 16;
      int qv2_1 = vid2_8 - qn2_1 * 16;
      int qslab_1 = qv2_1 / 8;
      int qcol_1 = (qv2_1 - qslab_1 * 8) * 16;
      int qoff_1 = qslab_1 * 1024 + qn2_1 * 128 + (qcol_1 ^ (qn2_1 & 7) << 4);
      float _cvt_f32_bf16_13;
      asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_13) : "h"((uint16_t)(qw[4])));
      float _cvt_f32_bf16_14;
      asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_14) : "h"((uint16_t)(qw[4] >> 16)));
      uint32_t _cvt_rn_f16x2_f32_12;
      asm("cvt.rn.f16x2.f32 %0, %1, %2;"
          : "=r"(_cvt_rn_f16x2_f32_12)
          : "f"(_cvt_f32_bf16_14), "f"(_cvt_f32_bf16_13));
      qw[4] = _cvt_rn_f16x2_f32_12;
      float _cvt_f32_bf16_16;
      asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_16) : "h"((uint16_t)(qw[5])));
      float _cvt_f32_bf16_17;
      asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_17) : "h"((uint16_t)(qw[5] >> 16)));
      uint32_t _cvt_rn_f16x2_f32_15;
      asm("cvt.rn.f16x2.f32 %0, %1, %2;"
          : "=r"(_cvt_rn_f16x2_f32_15)
          : "f"(_cvt_f32_bf16_17), "f"(_cvt_f32_bf16_16));
      qw[5] = _cvt_rn_f16x2_f32_15;
      float _cvt_f32_bf16_19;
      asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_19) : "h"((uint16_t)(qw[6])));
      float _cvt_f32_bf16_20;
      asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_20) : "h"((uint16_t)(qw[6] >> 16)));
      uint32_t _cvt_rn_f16x2_f32_18;
      asm("cvt.rn.f16x2.f32 %0, %1, %2;"
          : "=r"(_cvt_rn_f16x2_f32_18)
          : "f"(_cvt_f32_bf16_20), "f"(_cvt_f32_bf16_19));
      qw[6] = _cvt_rn_f16x2_f32_18;
      float _cvt_f32_bf16_22;
      asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_22) : "h"((uint16_t)(qw[7])));
      float _cvt_f32_bf16_23;
      asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_23) : "h"((uint16_t)(qw[7] >> 16)));
      uint32_t _cvt_rn_f16x2_f32_21;
      asm("cvt.rn.f16x2.f32 %0, %1, %2;"
          : "=r"(_cvt_rn_f16x2_f32_21)
          : "f"(_cvt_f32_bf16_23), "f"(_cvt_f32_bf16_22));
      qw[7] = _cvt_rn_f16x2_f32_21;
      asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(q16_addr + (unsigned int)qoff_1),
                   "r"(qw[4]), "r"(qw[5]), "r"(qw[6]), "r"(qw[7])
                   : "memory");
    }
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  asm volatile("barrier.sync 1, 128;" ::: "memory");
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
  float d_s0[4];
  float d_s1[4];
  float d_s0a[4];
  float d_s0b[4];
  float d_s1a[4];
  float d_s1b[4];
  float d_o0[4];
  float d_o1[4];
  float d_o0x[4];
  float d_o1x[4];
  d_o0[0] = 0.0f;
  d_o1[0] = 0.0f;
  d_o0x[0] = 0.0f;
  d_o1x[0] = 0.0f;
  d_o0[1] = 0.0f;
  d_o1[1] = 0.0f;
  d_o0x[1] = 0.0f;
  d_o1x[1] = 0.0f;
  d_o0[2] = 0.0f;
  d_o1[2] = 0.0f;
  d_o0x[2] = 0.0f;
  d_o1x[2] = 0.0f;
  d_o0[3] = 0.0f;
  d_o1[3] = 0.0f;
  d_o0x[3] = 0.0f;
  d_o1x[3] = 0.0f;
  float mrun[2];
  float lp[2];
  mrun[0] = -CUDART_INF_F;
  lp[0] = 0.0f;
  mrun[1] = -CUDART_INF_F;
  lp[1] = 0.0f;
  float sv0[4];
  float sv1[4];
  float tmax[2];
  float rv[4];
  float msub[2];
  uint64_t _wgmma_desc_24 = (((uint64_t)(((ident8_addr)) >> 4) & 0x3FFFULL) |
                             ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
  uint64_t _wgmma_b_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_24 >> 32)) << 32) |
                        (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_24);
  uint64_t _wgmma_desc_25 = (((uint64_t)(((q16_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) |
                             ((uint64_t)(64) << 32) | (1ULL << 62));
  uint64_t _wgmma_b_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_25 >> 32)) << 32) |
                        (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_25);
  uint64_t _wgmma_desc_26 = (((uint64_t)(((q16_addr + 1024)) >> 4) & 0x3FFFULL) |
                             ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
  uint64_t _wgmma_b_2 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_26 >> 32)) << 32) |
                        (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_26);
#pragma unroll 1
  for (int j = 0; j < 1; j++) {
    int stage = j % 2;
    int phase = j / 2 % 2;
    int par = j - j / 2 * 2;
    mbarrier_wait(kv_full_addr + (stage) * 8, phase);
    int slot_1 = split + j;
    int blk_2 = meta[slot_1];
    int page_head_1 = meta[16 + slot_1];
    int valid_1 = 0;
    if (blk_2 >= 0) {
      valid_1 = q_pos + 1 - blk_2 * 128;
      if (valid_1 > 128) {
        valid_1 = 128;
      }
      if (valid_1 < 0) {
        valid_1 = 0;
      }
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    uint64_t _wgmma_desc_27 =
        (((uint64_t)(((kv8_addr + (unsigned int)(stage * 32768))) >> 4) & 0x3FFFULL) |
         ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_a_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_27 >> 32)) << 32) |
                          (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_27);
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
    uint64_t _wgmma_desc_28 =
        (((uint64_t)(((kv8_addr + (unsigned int)(stage * 32768) + 8192)) >> 4) & 0x3FFFULL) |
         ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_a_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_28 >> 32)) << 32) |
                          (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_28);
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
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    ak0[0] = dk0_0[0] + zero_u32;
    ak1[0] = dk1_0[0] + zero_u32;
    ak0[1] = dk0_0[1] + zero_u32;
    ak1[1] = dk1_0[1] + zero_u32;
    ak0[2] = dk0_0[2] + zero_u32;
    ak1[2] = dk1_0[2] + zero_u32;
    ak0[3] = dk0_0[3] + zero_u32;
    ak1[3] = dk1_0[3] + zero_u32;
    ak0[4] = dk0_0[4] + zero_u32;
    ak1[4] = dk1_0[4] + zero_u32;
    ak0[5] = dk0_0[5] + zero_u32;
    ak1[5] = dk1_0[5] + zero_u32;
    ak0[6] = dk0_0[6] + zero_u32;
    ak1[6] = dk1_0[6] + zero_u32;
    ak0[7] = dk0_0[7] + zero_u32;
    ak1[7] = dk1_0[7] + zero_u32;
    ak0[8] = dk0_1[0] + zero_u32;
    ak1[8] = dk1_1[0] + zero_u32;
    ak0[9] = dk0_1[1] + zero_u32;
    ak1[9] = dk1_1[1] + zero_u32;
    ak0[10] = dk0_1[2] + zero_u32;
    ak1[10] = dk1_1[2] + zero_u32;
    ak0[11] = dk0_1[3] + zero_u32;
    ak1[11] = dk1_1[3] + zero_u32;
    ak0[12] = dk0_1[4] + zero_u32;
    ak1[12] = dk1_1[4] + zero_u32;
    ak0[13] = dk0_1[5] + zero_u32;
    ak1[13] = dk1_1[5] + zero_u32;
    ak0[14] = dk0_1[6] + zero_u32;
    ak1[14] = dk1_1[6] + zero_u32;
    ak0[15] = dk0_1[7] + zero_u32;
    ak1[15] = dk1_1[7] + zero_u32;
    ak0[16] = dk0_2[0] + zero_u32;
    ak1[16] = dk1_2[0] + zero_u32;
    ak0[17] = dk0_2[1] + zero_u32;
    ak1[17] = dk1_2[1] + zero_u32;
    ak0[18] = dk0_2[2] + zero_u32;
    ak1[18] = dk1_2[2] + zero_u32;
    ak0[19] = dk0_2[3] + zero_u32;
    ak1[19] = dk1_2[3] + zero_u32;
    ak0[20] = dk0_2[4] + zero_u32;
    ak1[20] = dk1_2[4] + zero_u32;
    ak0[21] = dk0_2[5] + zero_u32;
    ak1[21] = dk1_2[5] + zero_u32;
    ak0[22] = dk0_2[6] + zero_u32;
    ak1[22] = dk1_2[6] + zero_u32;
    ak0[23] = dk0_2[7] + zero_u32;
    ak1[23] = dk1_2[7] + zero_u32;
    ak0[24] = dk0_3[0] + zero_u32;
    ak1[24] = dk1_3[0] + zero_u32;
    ak0[25] = dk0_3[1] + zero_u32;
    ak1[25] = dk1_3[1] + zero_u32;
    ak0[26] = dk0_3[2] + zero_u32;
    ak1[26] = dk1_3[2] + zero_u32;
    ak0[27] = dk0_3[3] + zero_u32;
    ak1[27] = dk1_3[3] + zero_u32;
    ak0[28] = dk0_3[4] + zero_u32;
    ak1[28] = dk1_3[4] + zero_u32;
    ak0[29] = dk0_3[5] + zero_u32;
    ak1[29] = dk1_3[5] + zero_u32;
    ak0[30] = dk0_3[6] + zero_u32;
    ak1[30] = dk1_3[6] + zero_u32;
    ak0[31] = dk0_3[7] + zero_u32;
    ak1[31] = dk1_3[7] + zero_u32;
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 0, 1, 1, 0;\n}\n"
        : "+f"(d_s0a[0]), "+f"(d_s0a[1]), "+f"(d_s0a[2]), "+f"(d_s0a[3])
        : "r"(ak0[0]), "r"(ak0[1]), "r"(ak0[2]), "r"(ak0[3]), "l"(_wgmma_b_1)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 0, 1, 1, 0;\n}\n"
        : "+f"(d_s1a[0]), "+f"(d_s1a[1]), "+f"(d_s1a[2]), "+f"(d_s1a[3])
        : "r"(ak1[0]), "r"(ak1[1]), "r"(ak1[2]), "r"(ak1[3]), "l"(_wgmma_b_1)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 0, 1, 1, 0;\n}\n"
        : "+f"(d_s0b[0]), "+f"(d_s0b[1]), "+f"(d_s0b[2]), "+f"(d_s0b[3])
        : "r"(ak0[16]), "r"(ak0[(16) + 1]), "r"(ak0[(16) + 2]), "r"(ak0[(16) + 3]), "l"(_wgmma_b_2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 0, 1, 1, 0;\n}\n"
        : "+f"(d_s1b[0]), "+f"(d_s1b[1]), "+f"(d_s1b[2]), "+f"(d_s1b[3])
        : "r"(ak1[16]), "r"(ak1[(16) + 1]), "r"(ak1[(16) + 2]), "r"(ak1[(16) + 3]), "l"(_wgmma_b_2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s0a[0]), "+f"(d_s0a[1]), "+f"(d_s0a[2]), "+f"(d_s0a[3])
        : "r"(ak0[4]), "r"(ak0[(4) + 1]), "r"(ak0[(4) + 2]), "r"(ak0[(4) + 3]), "l"(_wgmma_b_1 + 2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s1a[0]), "+f"(d_s1a[1]), "+f"(d_s1a[2]), "+f"(d_s1a[3])
        : "r"(ak1[4]), "r"(ak1[(4) + 1]), "r"(ak1[(4) + 2]), "r"(ak1[(4) + 3]), "l"(_wgmma_b_1 + 2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s0b[0]), "+f"(d_s0b[1]), "+f"(d_s0b[2]), "+f"(d_s0b[3])
        : "r"(ak0[20]), "r"(ak0[(20) + 1]), "r"(ak0[(20) + 2]), "r"(ak0[(20) + 3]),
          "l"(_wgmma_b_2 + 2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s1b[0]), "+f"(d_s1b[1]), "+f"(d_s1b[2]), "+f"(d_s1b[3])
        : "r"(ak1[20]), "r"(ak1[(20) + 1]), "r"(ak1[(20) + 2]), "r"(ak1[(20) + 3]),
          "l"(_wgmma_b_2 + 2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s0a[0]), "+f"(d_s0a[1]), "+f"(d_s0a[2]), "+f"(d_s0a[3])
        : "r"(ak0[8]), "r"(ak0[(8) + 1]), "r"(ak0[(8) + 2]), "r"(ak0[(8) + 3]), "l"(_wgmma_b_1 + 4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s1a[0]), "+f"(d_s1a[1]), "+f"(d_s1a[2]), "+f"(d_s1a[3])
        : "r"(ak1[8]), "r"(ak1[(8) + 1]), "r"(ak1[(8) + 2]), "r"(ak1[(8) + 3]), "l"(_wgmma_b_1 + 4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s0b[0]), "+f"(d_s0b[1]), "+f"(d_s0b[2]), "+f"(d_s0b[3])
        : "r"(ak0[24]), "r"(ak0[(24) + 1]), "r"(ak0[(24) + 2]), "r"(ak0[(24) + 3]),
          "l"(_wgmma_b_2 + 4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s1b[0]), "+f"(d_s1b[1]), "+f"(d_s1b[2]), "+f"(d_s1b[3])
        : "r"(ak1[24]), "r"(ak1[(24) + 1]), "r"(ak1[(24) + 2]), "r"(ak1[(24) + 3]),
          "l"(_wgmma_b_2 + 4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s0a[0]), "+f"(d_s0a[1]), "+f"(d_s0a[2]), "+f"(d_s0a[3])
        : "r"(ak0[12]), "r"(ak0[(12) + 1]), "r"(ak0[(12) + 2]), "r"(ak0[(12) + 3]),
          "l"(_wgmma_b_1 + 6)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s1a[0]), "+f"(d_s1a[1]), "+f"(d_s1a[2]), "+f"(d_s1a[3])
        : "r"(ak1[12]), "r"(ak1[(12) + 1]), "r"(ak1[(12) + 2]), "r"(ak1[(12) + 3]),
          "l"(_wgmma_b_1 + 6)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s0b[0]), "+f"(d_s0b[1]), "+f"(d_s0b[2]), "+f"(d_s0b[3])
        : "r"(ak0[28]), "r"(ak0[(28) + 1]), "r"(ak0[(28) + 2]), "r"(ak0[(28) + 3]),
          "l"(_wgmma_b_2 + 6)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s1b[0]), "+f"(d_s1b[1]), "+f"(d_s1b[2]), "+f"(d_s1b[3])
        : "r"(ak1[28]), "r"(ak1[(28) + 1]), "r"(ak1[(28) + 2]), "r"(ak1[(28) + 3]),
          "l"(_wgmma_b_2 + 6)
        : "memory");
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    d_s0[0] = d_s0a[0] + d_s0b[0];
    d_s1[0] = d_s1a[0] + d_s1b[0];
    d_s0[1] = d_s0a[1] + d_s0b[1];
    d_s1[1] = d_s1a[1] + d_s1b[1];
    d_s0[2] = d_s0a[2] + d_s0b[2];
    d_s1[2] = d_s1a[2] + d_s1b[2];
    d_s0[3] = d_s0a[3] + d_s0b[3];
    d_s1[3] = d_s1a[3] + d_s1b[3];
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    uint64_t _wgmma_desc_29 =
        (((uint64_t)(((kv8_addr + (unsigned int)(stage * 32768) + 16384)) >> 4) & 0x3FFFULL) |
         ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_3 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_29 >> 32)) << 32) |
                          (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_29);
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 0, 1, 1;\n}\n"
        : "+r"(dv0_0[0]), "+r"(dv0_0[1]), "+r"(dv0_0[2]), "+r"(dv0_0[3]), "+r"(dv0_0[4]),
          "+r"(dv0_0[5]), "+r"(dv0_0[6]), "+r"(dv0_0[7]), "+r"(dv0_0[8]), "+r"(dv0_0[9]),
          "+r"(dv0_0[10]), "+r"(dv0_0[11]), "+r"(dv0_0[12]), "+r"(dv0_0[13]), "+r"(dv0_0[14]),
          "+r"(dv0_0[15])
        : "l"(_wgmma_b_0), "l"(_wgmma_b_3)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1;\n}\n"
        : "+r"(dv0_0[0]), "+r"(dv0_0[1]), "+r"(dv0_0[2]), "+r"(dv0_0[3]), "+r"(dv0_0[4]),
          "+r"(dv0_0[5]), "+r"(dv0_0[6]), "+r"(dv0_0[7]), "+r"(dv0_0[8]), "+r"(dv0_0[9]),
          "+r"(dv0_0[10]), "+r"(dv0_0[11]), "+r"(dv0_0[12]), "+r"(dv0_0[13]), "+r"(dv0_0[14]),
          "+r"(dv0_0[15])
        : "l"(_wgmma_b_0 + 2), "l"(_wgmma_b_3 + 2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 0, 1, 1;\n}\n"
        : "+r"(dv0_1[0]), "+r"(dv0_1[1]), "+r"(dv0_1[2]), "+r"(dv0_1[3]), "+r"(dv0_1[4]),
          "+r"(dv0_1[5]), "+r"(dv0_1[6]), "+r"(dv0_1[7]), "+r"(dv0_1[8]), "+r"(dv0_1[9]),
          "+r"(dv0_1[10]), "+r"(dv0_1[11]), "+r"(dv0_1[12]), "+r"(dv0_1[13]), "+r"(dv0_1[14]),
          "+r"(dv0_1[15])
        : "l"(_wgmma_b_0), "l"(_wgmma_b_3 + 4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1;\n}\n"
        : "+r"(dv0_1[0]), "+r"(dv0_1[1]), "+r"(dv0_1[2]), "+r"(dv0_1[3]), "+r"(dv0_1[4]),
          "+r"(dv0_1[5]), "+r"(dv0_1[6]), "+r"(dv0_1[7]), "+r"(dv0_1[8]), "+r"(dv0_1[9]),
          "+r"(dv0_1[10]), "+r"(dv0_1[11]), "+r"(dv0_1[12]), "+r"(dv0_1[13]), "+r"(dv0_1[14]),
          "+r"(dv0_1[15])
        : "l"(_wgmma_b_0 + 2), "l"(_wgmma_b_3 + 6)
        : "memory");
    uint64_t _wgmma_desc_30 =
        (((uint64_t)(((kv8_addr + (unsigned int)(stage * 32768) + 24576)) >> 4) & 0x3FFFULL) |
         ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_4 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_30 >> 32)) << 32) |
                          (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_30);
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 0, 1, 1;\n}\n"
        : "+r"(dv1_0[0]), "+r"(dv1_0[1]), "+r"(dv1_0[2]), "+r"(dv1_0[3]), "+r"(dv1_0[4]),
          "+r"(dv1_0[5]), "+r"(dv1_0[6]), "+r"(dv1_0[7]), "+r"(dv1_0[8]), "+r"(dv1_0[9]),
          "+r"(dv1_0[10]), "+r"(dv1_0[11]), "+r"(dv1_0[12]), "+r"(dv1_0[13]), "+r"(dv1_0[14]),
          "+r"(dv1_0[15])
        : "l"(_wgmma_b_0), "l"(_wgmma_b_4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1;\n}\n"
        : "+r"(dv1_0[0]), "+r"(dv1_0[1]), "+r"(dv1_0[2]), "+r"(dv1_0[3]), "+r"(dv1_0[4]),
          "+r"(dv1_0[5]), "+r"(dv1_0[6]), "+r"(dv1_0[7]), "+r"(dv1_0[8]), "+r"(dv1_0[9]),
          "+r"(dv1_0[10]), "+r"(dv1_0[11]), "+r"(dv1_0[12]), "+r"(dv1_0[13]), "+r"(dv1_0[14]),
          "+r"(dv1_0[15])
        : "l"(_wgmma_b_0 + 2), "l"(_wgmma_b_4 + 2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 0, 1, 1;\n}\n"
        : "+r"(dv1_1[0]), "+r"(dv1_1[1]), "+r"(dv1_1[2]), "+r"(dv1_1[3]), "+r"(dv1_1[4]),
          "+r"(dv1_1[5]), "+r"(dv1_1[6]), "+r"(dv1_1[7]), "+r"(dv1_1[8]), "+r"(dv1_1[9]),
          "+r"(dv1_1[10]), "+r"(dv1_1[11]), "+r"(dv1_1[12]), "+r"(dv1_1[13]), "+r"(dv1_1[14]),
          "+r"(dv1_1[15])
        : "l"(_wgmma_b_0), "l"(_wgmma_b_4 + 4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1;\n}\n"
        : "+r"(dv1_1[0]), "+r"(dv1_1[1]), "+r"(dv1_1[2]), "+r"(dv1_1[3]), "+r"(dv1_1[4]),
          "+r"(dv1_1[5]), "+r"(dv1_1[6]), "+r"(dv1_1[7]), "+r"(dv1_1[8]), "+r"(dv1_1[9]),
          "+r"(dv1_1[10]), "+r"(dv1_1[11]), "+r"(dv1_1[12]), "+r"(dv1_1[13]), "+r"(dv1_1[14]),
          "+r"(dv1_1[15])
        : "l"(_wgmma_b_0 + 2), "l"(_wgmma_b_4 + 6)
        : "memory");
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    tmax[0] = -CUDART_INF_F;
    tmax[1] = -CUDART_INF_F;
    sv0[0] = -CUDART_INF_F;
    if (key0 < valid_1) {
      sv0[0] = d_s0[0] * softmax_scale_log2;
    }
    sv1[0] = -CUDART_INF_F;
    if (valid_1 > 64 + key0) {
      sv1[0] = d_s1[0] * softmax_scale_log2;
    }
    tmax[0] = max_noftz(tmax[0], max_noftz(sv0[0], sv1[0]));
    sv0[1] = -CUDART_INF_F;
    if (key0 < valid_1) {
      sv0[1] = d_s0[1] * softmax_scale_log2;
    }
    sv1[1] = -CUDART_INF_F;
    if (valid_1 > 64 + key0) {
      sv1[1] = d_s1[1] * softmax_scale_log2;
    }
    tmax[1] = max_noftz(tmax[1], max_noftz(sv0[1], sv1[1]));
    sv0[2] = -CUDART_INF_F;
    if (key1 < valid_1) {
      sv0[2] = d_s0[2] * softmax_scale_log2;
    }
    sv1[2] = -CUDART_INF_F;
    if (valid_1 > 64 + key1) {
      sv1[2] = d_s1[2] * softmax_scale_log2;
    }
    tmax[0] = max_noftz(tmax[0], max_noftz(sv0[2], sv1[2]));
    sv0[3] = -CUDART_INF_F;
    if (key1 < valid_1) {
      sv0[3] = d_s0[3] * softmax_scale_log2;
    }
    sv1[3] = -CUDART_INF_F;
    if (valid_1 > 64 + key1) {
      sv1[3] = d_s1[3] * softmax_scale_log2;
    }
    tmax[1] = max_noftz(tmax[1], max_noftz(sv0[3], sv1[3]));
    float t = tmax[0];
    float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, t, 4);
    t = max_noftz(t, _shfl_xor_sync_31);
    float _shfl_xor_sync_32 = __shfl_xor_sync(0xFFFFFFFF, t, 8);
    t = max_noftz(t, _shfl_xor_sync_32);
    float _shfl_xor_sync_33 = __shfl_xor_sync(0xFFFFFFFF, t, 16);
    t = max_noftz(t, _shfl_xor_sync_33);
    tmax[0] = t;
    float t_0 = tmax[1];
    float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, t_0, 4);
    t_0 = max_noftz(t_0, _shfl_xor_sync_34);
    float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, t_0, 8);
    t_0 = max_noftz(t_0, _shfl_xor_sync_35);
    float _shfl_xor_sync_36 = __shfl_xor_sync(0xFFFFFFFF, t_0, 16);
    t_0 = max_noftz(t_0, _shfl_xor_sync_36);
    tmax[1] = t_0;
    if (lane_0 < 4) {
      int hh = 2 * cq;
      red[hh * 4 + warp_1] = tmax[0];
      int hh_0 = 2 * cq + 1;
      red[hh_0 * 4 + warp_1] = tmax[1];
    }
    asm volatile("barrier.sync 1, 128;" ::: "memory");
    int hh2 = 2 * cq;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])),
                   "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
                 : "r"(red_addr + (unsigned int)(hh2 * 4 * 4)));
    float page_max;
    asm("max.f32 %0, %1, %2;"
        : "=f"(page_max)
        : "f"(max_noftz(rv[0], rv[1])), "f"(max_noftz(rv[2], rv[3])));
    float mnew;
    asm("max.f32 %0, %1, %2;" : "=f"(mnew) : "f"(mrun[0]), "f"(page_max));
    float corr = 0.0f;
    if (mrun[0] != -CUDART_INF_F) {
      float _exp2_37 = approx_exp2(mrun[0] - mnew);
      corr = _exp2_37;
    }
    d_o0[0] = d_o0[0] * corr;
    d_o0[2] = d_o0[2] * corr;
    d_o1[0] = d_o1[0] * corr;
    d_o1[2] = d_o1[2] * corr;
    d_o0x[0] = d_o0x[0] * corr;
    d_o0x[2] = d_o0x[2] * corr;
    d_o1x[0] = d_o1x[0] * corr;
    d_o1x[2] = d_o1x[2] * corr;
    lp[0] = lp[0] * corr;
    mrun[0] = mnew;
    msub[0] = mnew;
    if (mnew == -CUDART_INF_F) {
      msub[0] = 0.0f;
    }
    int hh2_1 = 2 * cq + 1;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])),
                   "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
                 : "r"(red_addr + (unsigned int)(hh2_1 * 4 * 4)));
    float page_max_2;
    asm("max.f32 %0, %1, %2;"
        : "=f"(page_max_2)
        : "f"(max_noftz(rv[0], rv[1])), "f"(max_noftz(rv[2], rv[3])));
    float mnew_3;
    asm("max.f32 %0, %1, %2;" : "=f"(mnew_3) : "f"(mrun[1]), "f"(page_max_2));
    float corr_4 = 0.0f;
    if (mrun[1] != -CUDART_INF_F) {
      float _exp2_38 = approx_exp2(mrun[1] - mnew_3);
      corr_4 = _exp2_38;
    }
    d_o0[1] = d_o0[1] * corr_4;
    d_o0[3] = d_o0[3] * corr_4;
    d_o1[1] = d_o1[1] * corr_4;
    d_o1[3] = d_o1[3] * corr_4;
    d_o0x[1] = d_o0x[1] * corr_4;
    d_o0x[3] = d_o0x[3] * corr_4;
    d_o1x[1] = d_o1x[1] * corr_4;
    d_o1x[3] = d_o1x[3] * corr_4;
    lp[1] = lp[1] * corr_4;
    mrun[1] = mnew_3;
    msub[1] = mnew_3;
    if (mnew_3 == -CUDART_INF_F) {
      msub[1] = 0.0f;
    }
    int hh3 = 2 * cq;
    int kk = key0;
    int pcol = hh3 * 128 + (kk * 2 ^ (hh3 & 7) << 4);
    float p0 = approx_exp2(sv0[0] - msub[0]);
    float p1 = approx_exp2(sv1[0] - msub[0]);
    lp[0] = lp[0] + (p0 + p1);
    {
      __half _hval_39 = __float2half_rn(p0);
      uint16_t _bits_39 = *(uint16_t*)&_hval_39;
      uint32_t _addr_39 = static_cast<uint32_t>(p16_addr + (unsigned int)(par * 2 * 1024 + pcol));
      asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_39), "h"(_bits_39) : "memory");
    }
    {
      __half _hval_40 = __float2half_rn(p1);
      uint16_t _bits_40 = *(uint16_t*)&_hval_40;
      uint32_t _addr_40 =
          static_cast<uint32_t>(p16_addr + (unsigned int)((par * 2 + 1) * 1024 + pcol));
      asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_40), "h"(_bits_40) : "memory");
    }
    int hh3_5 = 2 * cq + 1;
    int kk_6 = key0;
    int pcol_7 = hh3_5 * 128 + (kk_6 * 2 ^ (hh3_5 & 7) << 4);
    float p0_8 = approx_exp2(sv0[1] - msub[1]);
    float p1_9 = approx_exp2(sv1[1] - msub[1]);
    lp[1] = lp[1] + (p0_8 + p1_9);
    {
      __half _hval_41 = __float2half_rn(p0_8);
      uint16_t _bits_41 = *(uint16_t*)&_hval_41;
      uint32_t _addr_41 = static_cast<uint32_t>(p16_addr + (unsigned int)(par * 2 * 1024 + pcol_7));
      asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_41), "h"(_bits_41) : "memory");
    }
    {
      __half _hval_42 = __float2half_rn(p1_9);
      uint16_t _bits_42 = *(uint16_t*)&_hval_42;
      uint32_t _addr_42 =
          static_cast<uint32_t>(p16_addr + (unsigned int)((par * 2 + 1) * 1024 + pcol_7));
      asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_42), "h"(_bits_42) : "memory");
    }
    int hh3_10 = 2 * cq;
    int kk_11 = key1;
    int pcol_12 = hh3_10 * 128 + (kk_11 * 2 ^ (hh3_10 & 7) << 4);
    float p0_13 = approx_exp2(sv0[2] - msub[0]);
    float p1_14 = approx_exp2(sv1[2] - msub[0]);
    lp[0] = lp[0] + (p0_13 + p1_14);
    {
      __half _hval_43 = __float2half_rn(p0_13);
      uint16_t _bits_43 = *(uint16_t*)&_hval_43;
      uint32_t _addr_43 =
          static_cast<uint32_t>(p16_addr + (unsigned int)(par * 2 * 1024 + pcol_12));
      asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_43), "h"(_bits_43) : "memory");
    }
    {
      __half _hval_44 = __float2half_rn(p1_14);
      uint16_t _bits_44 = *(uint16_t*)&_hval_44;
      uint32_t _addr_44 =
          static_cast<uint32_t>(p16_addr + (unsigned int)((par * 2 + 1) * 1024 + pcol_12));
      asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_44), "h"(_bits_44) : "memory");
    }
    int hh3_15 = 2 * cq + 1;
    int kk_16 = key1;
    int pcol_17 = hh3_15 * 128 + (kk_16 * 2 ^ (hh3_15 & 7) << 4);
    float p0_18 = approx_exp2(sv0[3] - msub[1]);
    float p1_19 = approx_exp2(sv1[3] - msub[1]);
    lp[1] = lp[1] + (p0_18 + p1_19);
    {
      __half _hval_45 = __float2half_rn(p0_18);
      uint16_t _bits_45 = *(uint16_t*)&_hval_45;
      uint32_t _addr_45 =
          static_cast<uint32_t>(p16_addr + (unsigned int)(par * 2 * 1024 + pcol_17));
      asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_45), "h"(_bits_45) : "memory");
    }
    {
      __half _hval_46 = __float2half_rn(p1_19);
      uint16_t _bits_46 = *(uint16_t*)&_hval_46;
      uint32_t _addr_46 =
          static_cast<uint32_t>(p16_addr + (unsigned int)((par * 2 + 1) * 1024 + pcol_17));
      asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_46), "h"(_bits_46) : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("barrier.sync 1, 128;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    if (valid_1 < 128) {
      int thr = valid_1 - 2 * cq;
      if (thr <= 1) {
        dv0_0[0] = dv0_0[0] & 65535;
        dv0_1[0] = dv0_1[0] & 65535;
      }
      if (thr <= 0) {
        dv0_0[0] = 0;
        dv0_1[0] = 0;
      }
      if (thr <= 1) {
        dv0_0[1] = dv0_0[1] & 65535;
        dv0_1[1] = dv0_1[1] & 65535;
      }
      if (thr <= 0) {
        dv0_0[1] = 0;
        dv0_1[1] = 0;
      }
      if (thr <= 9) {
        dv0_0[2] = dv0_0[2] & 65535;
        dv0_1[2] = dv0_1[2] & 65535;
      }
      if (thr <= 8) {
        dv0_0[2] = 0;
        dv0_1[2] = 0;
      }
      if (thr <= 9) {
        dv0_0[3] = dv0_0[3] & 65535;
        dv0_1[3] = dv0_1[3] & 65535;
      }
      if (thr <= 8) {
        dv0_0[3] = 0;
        dv0_1[3] = 0;
      }
      if (thr <= 17) {
        dv0_0[4] = dv0_0[4] & 65535;
        dv0_1[4] = dv0_1[4] & 65535;
      }
      if (thr <= 16) {
        dv0_0[4] = 0;
        dv0_1[4] = 0;
      }
      if (thr <= 17) {
        dv0_0[5] = dv0_0[5] & 65535;
        dv0_1[5] = dv0_1[5] & 65535;
      }
      if (thr <= 16) {
        dv0_0[5] = 0;
        dv0_1[5] = 0;
      }
      if (thr <= 25) {
        dv0_0[6] = dv0_0[6] & 65535;
        dv0_1[6] = dv0_1[6] & 65535;
      }
      if (thr <= 24) {
        dv0_0[6] = 0;
        dv0_1[6] = 0;
      }
      if (thr <= 25) {
        dv0_0[7] = dv0_0[7] & 65535;
        dv0_1[7] = dv0_1[7] & 65535;
      }
      if (thr <= 24) {
        dv0_0[7] = 0;
        dv0_1[7] = 0;
      }
      if (thr <= 33) {
        dv0_0[8] = dv0_0[8] & 65535;
        dv0_1[8] = dv0_1[8] & 65535;
      }
      if (thr <= 32) {
        dv0_0[8] = 0;
        dv0_1[8] = 0;
      }
      if (thr <= 33) {
        dv0_0[9] = dv0_0[9] & 65535;
        dv0_1[9] = dv0_1[9] & 65535;
      }
      if (thr <= 32) {
        dv0_0[9] = 0;
        dv0_1[9] = 0;
      }
      if (thr <= 41) {
        dv0_0[10] = dv0_0[10] & 65535;
        dv0_1[10] = dv0_1[10] & 65535;
      }
      if (thr <= 40) {
        dv0_0[10] = 0;
        dv0_1[10] = 0;
      }
      if (thr <= 41) {
        dv0_0[11] = dv0_0[11] & 65535;
        dv0_1[11] = dv0_1[11] & 65535;
      }
      if (thr <= 40) {
        dv0_0[11] = 0;
        dv0_1[11] = 0;
      }
      if (thr <= 49) {
        dv0_0[12] = dv0_0[12] & 65535;
        dv0_1[12] = dv0_1[12] & 65535;
      }
      if (thr <= 48) {
        dv0_0[12] = 0;
        dv0_1[12] = 0;
      }
      if (thr <= 49) {
        dv0_0[13] = dv0_0[13] & 65535;
        dv0_1[13] = dv0_1[13] & 65535;
      }
      if (thr <= 48) {
        dv0_0[13] = 0;
        dv0_1[13] = 0;
      }
      if (thr <= 57) {
        dv0_0[14] = dv0_0[14] & 65535;
        dv0_1[14] = dv0_1[14] & 65535;
      }
      if (thr <= 56) {
        dv0_0[14] = 0;
        dv0_1[14] = 0;
      }
      if (thr <= 57) {
        dv0_0[15] = dv0_0[15] & 65535;
        dv0_1[15] = dv0_1[15] & 65535;
      }
      if (thr <= 56) {
        dv0_0[15] = 0;
        dv0_1[15] = 0;
      }
      int thr_0 = valid_1 - 64 - 2 * cq;
      if (thr_0 <= 1) {
        dv1_0[0] = dv1_0[0] & 65535;
        dv1_1[0] = dv1_1[0] & 65535;
      }
      if (thr_0 <= 0) {
        dv1_0[0] = 0;
        dv1_1[0] = 0;
      }
      if (thr_0 <= 1) {
        dv1_0[1] = dv1_0[1] & 65535;
        dv1_1[1] = dv1_1[1] & 65535;
      }
      if (thr_0 <= 0) {
        dv1_0[1] = 0;
        dv1_1[1] = 0;
      }
      if (thr_0 <= 9) {
        dv1_0[2] = dv1_0[2] & 65535;
        dv1_1[2] = dv1_1[2] & 65535;
      }
      if (thr_0 <= 8) {
        dv1_0[2] = 0;
        dv1_1[2] = 0;
      }
      if (thr_0 <= 9) {
        dv1_0[3] = dv1_0[3] & 65535;
        dv1_1[3] = dv1_1[3] & 65535;
      }
      if (thr_0 <= 8) {
        dv1_0[3] = 0;
        dv1_1[3] = 0;
      }
      if (thr_0 <= 17) {
        dv1_0[4] = dv1_0[4] & 65535;
        dv1_1[4] = dv1_1[4] & 65535;
      }
      if (thr_0 <= 16) {
        dv1_0[4] = 0;
        dv1_1[4] = 0;
      }
      if (thr_0 <= 17) {
        dv1_0[5] = dv1_0[5] & 65535;
        dv1_1[5] = dv1_1[5] & 65535;
      }
      if (thr_0 <= 16) {
        dv1_0[5] = 0;
        dv1_1[5] = 0;
      }
      if (thr_0 <= 25) {
        dv1_0[6] = dv1_0[6] & 65535;
        dv1_1[6] = dv1_1[6] & 65535;
      }
      if (thr_0 <= 24) {
        dv1_0[6] = 0;
        dv1_1[6] = 0;
      }
      if (thr_0 <= 25) {
        dv1_0[7] = dv1_0[7] & 65535;
        dv1_1[7] = dv1_1[7] & 65535;
      }
      if (thr_0 <= 24) {
        dv1_0[7] = 0;
        dv1_1[7] = 0;
      }
      if (thr_0 <= 33) {
        dv1_0[8] = dv1_0[8] & 65535;
        dv1_1[8] = dv1_1[8] & 65535;
      }
      if (thr_0 <= 32) {
        dv1_0[8] = 0;
        dv1_1[8] = 0;
      }
      if (thr_0 <= 33) {
        dv1_0[9] = dv1_0[9] & 65535;
        dv1_1[9] = dv1_1[9] & 65535;
      }
      if (thr_0 <= 32) {
        dv1_0[9] = 0;
        dv1_1[9] = 0;
      }
      if (thr_0 <= 41) {
        dv1_0[10] = dv1_0[10] & 65535;
        dv1_1[10] = dv1_1[10] & 65535;
      }
      if (thr_0 <= 40) {
        dv1_0[10] = 0;
        dv1_1[10] = 0;
      }
      if (thr_0 <= 41) {
        dv1_0[11] = dv1_0[11] & 65535;
        dv1_1[11] = dv1_1[11] & 65535;
      }
      if (thr_0 <= 40) {
        dv1_0[11] = 0;
        dv1_1[11] = 0;
      }
      if (thr_0 <= 49) {
        dv1_0[12] = dv1_0[12] & 65535;
        dv1_1[12] = dv1_1[12] & 65535;
      }
      if (thr_0 <= 48) {
        dv1_0[12] = 0;
        dv1_1[12] = 0;
      }
      if (thr_0 <= 49) {
        dv1_0[13] = dv1_0[13] & 65535;
        dv1_1[13] = dv1_1[13] & 65535;
      }
      if (thr_0 <= 48) {
        dv1_0[13] = 0;
        dv1_1[13] = 0;
      }
      if (thr_0 <= 57) {
        dv1_0[14] = dv1_0[14] & 65535;
        dv1_1[14] = dv1_1[14] & 65535;
      }
      if (thr_0 <= 56) {
        dv1_0[14] = 0;
        dv1_1[14] = 0;
      }
      if (thr_0 <= 57) {
        dv1_0[15] = dv1_0[15] & 65535;
        dv1_1[15] = dv1_1[15] & 65535;
      }
      if (thr_0 <= 56) {
        dv1_0[15] = 0;
        dv1_1[15] = 0;
      }
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
    if (nxt < 1) {
      int slot_0 = split + nxt;
      int blk_1_1 = meta[slot_0];
      int page_head_2 = meta[16 + slot_0];
      int valid_3 = 0;
      if (blk_1_1 >= 0) {
        valid_3 = q_pos + 1 - blk_1_1 * 128;
        if (valid_3 > 128) {
          valid_3 = 128;
        }
        if (valid_3 < 0) {
          valid_3 = 0;
        }
      }
      if (warp == 1) {
        if (elect_sync()) {
          mbarrier_arrive_expect_tx(kv_full_addr + (stage) * 8, 32768);
#if CAKE_L2_EVICT_FIRST
          asm volatile(
              "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_"
              "hint"
              " [%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(kv8_addr +
                                                           (unsigned int)(stage * 32768)),
              "l"((&K)), "r"(0), "r"(0), "r"(page_head_2), "r"(kv_full_addr + (stage) * 8),
              "l"(0x12F0000000000000ULL)
              : "memory");
          asm volatile(
              "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_"
              "hint"
              " [%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(kv8_addr +
                                                           (unsigned int)(stage * 32768) + 8192),
              "l"((&K)), "r"(0), "r"(64), "r"(page_head_2), "r"(kv_full_addr + (stage) * 8),
              "l"(0x12F0000000000000ULL)
              : "memory");
          asm volatile(
              "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_"
              "hint"
              " [%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(kv8_addr +
                                                           (unsigned int)(stage * 32768) + 16384),
              "l"((&V)), "r"(0), "r"(0), "r"(page_head_2), "r"(kv_full_addr + (stage) * 8),
              "l"(0x12F0000000000000ULL)
              : "memory");
          asm volatile(
              "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_"
              "hint"
              " [%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(kv8_addr +
                                                           (unsigned int)(stage * 32768) + 24576),
              "l"((&V)), "r"(0), "r"(64), "r"(page_head_2), "r"(kv_full_addr + (stage) * 8),
              "l"(0x12F0000000000000ULL)
              : "memory");
#else
          tma_3d_gmem2smem(kv8_addr + (unsigned int)(stage * 32768), (&K), 0, 0, page_head_2,
                           kv_full_addr + (stage) * 8);
          tma_3d_gmem2smem(kv8_addr + (unsigned int)(stage * 32768) + 8192, (&K), 0, 64,
                           page_head_2, kv_full_addr + (stage) * 8);
          tma_3d_gmem2smem(kv8_addr + (unsigned int)(stage * 32768) + 16384, (&V), 0, 0,
                           page_head_2, kv_full_addr + (stage) * 8);
          tma_3d_gmem2smem(kv8_addr + (unsigned int)(stage * 32768) + 24576, (&V), 0, 64,
                           page_head_2, kv_full_addr + (stage) * 8);
#endif
        }
      }
    }
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    uint64_t _wgmma_desc_47 =
        (((uint64_t)(((p16_addr + (unsigned int)(par * 2 * 1024))) >> 4) & 0x3FFFULL) |
         ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_5 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_47 >> 32)) << 32) |
                          (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_47);
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o0[0]), "+f"(d_o0[1]), "+f"(d_o0[2]), "+f"(d_o0[3])
        : "r"(av0[0]), "r"(av0[1]), "r"(av0[2]), "r"(av0[3]), "l"(_wgmma_b_5)
        : "memory");
    uint64_t _wgmma_desc_48 =
        (((uint64_t)(((p16_addr + (unsigned int)((par * 2 + 1) * 1024))) >> 4) & 0x3FFFULL) |
         ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_6 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_48 >> 32)) << 32) |
                          (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_48);
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o0x[0]), "+f"(d_o0x[1]), "+f"(d_o0x[2]), "+f"(d_o0x[3])
        : "r"(av1[0]), "r"(av1[1]), "r"(av1[2]), "r"(av1[3]), "l"(_wgmma_b_6)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o1[0]), "+f"(d_o1[1]), "+f"(d_o1[2]), "+f"(d_o1[3])
        : "r"(av0[16]), "r"(av0[(16) + 1]), "r"(av0[(16) + 2]), "r"(av0[(16) + 3]), "l"(_wgmma_b_5)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o1x[0]), "+f"(d_o1x[1]), "+f"(d_o1x[2]), "+f"(d_o1x[3])
        : "r"(av1[16]), "r"(av1[(16) + 1]), "r"(av1[(16) + 2]), "r"(av1[(16) + 3]), "l"(_wgmma_b_6)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o0[0]), "+f"(d_o0[1]), "+f"(d_o0[2]), "+f"(d_o0[3])
        : "r"(av0[4]), "r"(av0[(4) + 1]), "r"(av0[(4) + 2]), "r"(av0[(4) + 3]), "l"(_wgmma_b_5 + 2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o0x[0]), "+f"(d_o0x[1]), "+f"(d_o0x[2]), "+f"(d_o0x[3])
        : "r"(av1[4]), "r"(av1[(4) + 1]), "r"(av1[(4) + 2]), "r"(av1[(4) + 3]), "l"(_wgmma_b_6 + 2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o1[0]), "+f"(d_o1[1]), "+f"(d_o1[2]), "+f"(d_o1[3])
        : "r"(av0[20]), "r"(av0[(20) + 1]), "r"(av0[(20) + 2]), "r"(av0[(20) + 3]),
          "l"(_wgmma_b_5 + 2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o1x[0]), "+f"(d_o1x[1]), "+f"(d_o1x[2]), "+f"(d_o1x[3])
        : "r"(av1[20]), "r"(av1[(20) + 1]), "r"(av1[(20) + 2]), "r"(av1[(20) + 3]),
          "l"(_wgmma_b_6 + 2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o0[0]), "+f"(d_o0[1]), "+f"(d_o0[2]), "+f"(d_o0[3])
        : "r"(av0[8]), "r"(av0[(8) + 1]), "r"(av0[(8) + 2]), "r"(av0[(8) + 3]), "l"(_wgmma_b_5 + 4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o0x[0]), "+f"(d_o0x[1]), "+f"(d_o0x[2]), "+f"(d_o0x[3])
        : "r"(av1[8]), "r"(av1[(8) + 1]), "r"(av1[(8) + 2]), "r"(av1[(8) + 3]), "l"(_wgmma_b_6 + 4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o1[0]), "+f"(d_o1[1]), "+f"(d_o1[2]), "+f"(d_o1[3])
        : "r"(av0[24]), "r"(av0[(24) + 1]), "r"(av0[(24) + 2]), "r"(av0[(24) + 3]),
          "l"(_wgmma_b_5 + 4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o1x[0]), "+f"(d_o1x[1]), "+f"(d_o1x[2]), "+f"(d_o1x[3])
        : "r"(av1[24]), "r"(av1[(24) + 1]), "r"(av1[(24) + 2]), "r"(av1[(24) + 3]),
          "l"(_wgmma_b_6 + 4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o0[0]), "+f"(d_o0[1]), "+f"(d_o0[2]), "+f"(d_o0[3])
        : "r"(av0[12]), "r"(av0[(12) + 1]), "r"(av0[(12) + 2]), "r"(av0[(12) + 3]),
          "l"(_wgmma_b_5 + 6)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o0x[0]), "+f"(d_o0x[1]), "+f"(d_o0x[2]), "+f"(d_o0x[3])
        : "r"(av1[12]), "r"(av1[(12) + 1]), "r"(av1[(12) + 2]), "r"(av1[(12) + 3]),
          "l"(_wgmma_b_6 + 6)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o1[0]), "+f"(d_o1[1]), "+f"(d_o1[2]), "+f"(d_o1[3])
        : "r"(av0[28]), "r"(av0[(28) + 1]), "r"(av0[(28) + 2]), "r"(av0[(28) + 3]),
          "l"(_wgmma_b_5 + 6)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "%8, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o1x[0]), "+f"(d_o1x[1]), "+f"(d_o1x[2]), "+f"(d_o1x[3])
        : "r"(av1[28]), "r"(av1[(28) + 1]), "r"(av1[(28) + 2]), "r"(av1[(28) + 3]),
          "l"(_wgmma_b_6 + 6)
        : "memory");
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  d_o0[0] = d_o0[0] + d_o0x[0];
  d_o1[0] = d_o1[0] + d_o1x[0];
  d_o0[1] = d_o0[1] + d_o0x[1];
  d_o1[1] = d_o1[1] + d_o1x[1];
  d_o0[2] = d_o0[2] + d_o0x[2];
  d_o1[2] = d_o1[2] + d_o1x[2];
  d_o0[3] = d_o0[3] + d_o0x[3];
  d_o1[3] = d_o1[3] + d_o1x[3];
  float t2 = lp[0];
  float _shfl_xor_sync_49 = __shfl_xor_sync(0xFFFFFFFF, t2, 4);
  t2 = t2 + _shfl_xor_sync_49;
  float _shfl_xor_sync_50 = __shfl_xor_sync(0xFFFFFFFF, t2, 8);
  t2 = t2 + _shfl_xor_sync_50;
  float _shfl_xor_sync_51 = __shfl_xor_sync(0xFFFFFFFF, t2, 16);
  t2 = t2 + _shfl_xor_sync_51;
  lp[0] = t2;
  float t2_2 = lp[1];
  float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, t2_2, 4);
  t2_2 = t2_2 + _shfl_xor_sync_52;
  float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, t2_2, 8);
  t2_2 = t2_2 + _shfl_xor_sync_53;
  float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, t2_2, 16);
  t2_2 = t2_2 + _shfl_xor_sync_54;
  lp[1] = t2_2;
  if (lane_0 < 4) {
    int hh4 = 2 * cq;
    lred[hh4 * 4 + warp_1] = lp[0];
    int hh4_0 = 2 * cq + 1;
    lred[hh4_0 * 4 + warp_1] = lp[1];
  }
  asm volatile("barrier.sync 1, 128;" ::: "memory");
  float lsum[2];
  int hh5 = 2 * cq;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
               : "r"(lred_addr + (unsigned int)(hh5 * 4 * 4)));
  lsum[0] = rv[0] + rv[1] + (rv[2] + rv[3]);
  int hh5_3 = 2 * cq + 1;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
               : "r"(lred_addr + (unsigned int)(hh5_3 * 4 * 4)));
  lsum[1] = rv[0] + rv[1] + (rv[2] + rv[3]);
  int merge = 1;
  int pbase = item * 16 + split;
  int orow = (pbase * 128 + tid_1) * 4;
  float invl[2];
  invl[0] = 0.0f;
  if (lsum[0] != 0.0f) {
    float _rcp_55 = approx_rcp(lsum[0]);
    invl[0] = _rcp_55;
  }
  invl[1] = 0.0f;
  if (lsum[1] != 0.0f) {
    float _rcp_56 = approx_rcp(lsum[1]);
    invl[1] = _rcp_56;
  }
  unsigned int pk[4];
  uint32_t _cvt_rn_f16x2_f32_57;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;"
      : "=r"(_cvt_rn_f16x2_f32_57)
      : "f"(d_o0[1] * invl[1]), "f"(d_o0[0] * invl[0]));
  pk[0] = _cvt_rn_f16x2_f32_57;
  uint32_t _cvt_rn_f16x2_f32_58;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;"
      : "=r"(_cvt_rn_f16x2_f32_58)
      : "f"(d_o0[3] * invl[1]), "f"(d_o0[2] * invl[0]));
  pk[1] = _cvt_rn_f16x2_f32_58;
  uint32_t _cvt_rn_f16x2_f32_59;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;"
      : "=r"(_cvt_rn_f16x2_f32_59)
      : "f"(d_o1[1] * invl[1]), "f"(d_o1[0] * invl[0]));
  pk[2] = _cvt_rn_f16x2_f32_59;
  uint32_t _cvt_rn_f16x2_f32_60;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;"
      : "=r"(_cvt_rn_f16x2_f32_60)
      : "f"(d_o1[3] * invl[1]), "f"(d_o1[2] * invl[0]));
  pk[3] = _cvt_rn_f16x2_f32_60;
  asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(part_o + orow + (0)),
               "r"(pk[(0) + 0]), "r"(pk[(0) + 1]), "r"(pk[(0) + 2]), "r"(pk[(0) + 3])
               : "memory");
  if (warp_1 == 0) {
    if (lane_0 < 4) {
      float mlv[4];
      mlv[0] = mrun[0];
      mlv[1] = lsum[0];
      mlv[2] = mrun[1];
      mlv[3] = lsum[1];
      {
        float4 _v4 = make_float4(mlv[0 + 0], mlv[0 + 1], mlv[0 + 2], mlv[0 + 3]);
        *reinterpret_cast<float4*>((part_ml + ((pbase * 4 + cq) * 4)) + 0) = _v4;
      }
    }
  }
  asm volatile("barrier.sync 1, 128;" ::: "memory");
  if (warp == 0) {
    if (elect_sync()) {
      unsigned int _atomic_fetch_add_61;
      asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                   : "=r"(_atomic_fetch_add_61)
                   : "l"(&counters[item]), "r"(static_cast<uint32_t>(1))
                   : "memory");
      flag[0] = _atomic_fetch_add_61;
    }
  }
  asm volatile("barrier.sync 1, 128;" ::: "memory");
  unsigned int oldv = flag[0];
  merge = 0;
  if ((oldv & 15) == 15) {
    merge = 1;
    if (warp == 0) {
      if (elect_sync()) {
        asm volatile("fence.proxy.async.global;" ::: "memory");
        mbarrier_arrive_expect_tx(mfull_addr, 33792);
        cp_async_bulk_gmem2smem(kv8_addr,
                                reinterpret_cast<const void*>(
                                    reinterpret_cast<const uint8_t*>(part_o) +
                                    ((unsigned long long)(item * 8192) * (unsigned long long)4)),
                                32768, mfull_addr);
        cp_async_bulk_gmem2smem(p16_addr,
                                reinterpret_cast<const void*>(
                                    reinterpret_cast<const uint8_t*>(part_ml) +
                                    ((unsigned long long)(item * 256) * (unsigned long long)4)),
                                1024, mfull_addr);
      }
    }
    mbarrier_wait(mfull_addr, 0);
    d_o0[0] = 0.0f;
    d_o1[0] = 0.0f;
    d_o0[1] = 0.0f;
    d_o1[1] = 0.0f;
    d_o0[2] = 0.0f;
    d_o1[2] = 0.0f;
    d_o0[3] = 0.0f;
    d_o1[3] = 0.0f;
    lsum[0] = 0.0f;
    lsum[1] = 0.0f;
    float mlall[64];
    unsigned int pall[64];
    float cw[32];
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&mlall[0])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(0) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(0) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(0) + 3]))
                 : "r"(p16_addr + (unsigned int)(cq * 4 * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&mlall[4])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(4) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(4) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(4) + 3]))
                 : "r"(p16_addr + (unsigned int)((4 + cq) * 4 * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&mlall[8])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(8) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(8) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(8) + 3]))
                 : "r"(p16_addr + (unsigned int)((8 + cq) * 4 * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&mlall[12])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(12) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(12) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(12) + 3]))
                 : "r"(p16_addr + (unsigned int)((12 + cq) * 4 * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&mlall[16])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(16) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(16) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(16) + 3]))
                 : "r"(p16_addr + (unsigned int)((16 + cq) * 4 * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&mlall[20])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(20) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(20) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(20) + 3]))
                 : "r"(p16_addr + (unsigned int)((20 + cq) * 4 * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&mlall[24])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(24) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(24) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(24) + 3]))
                 : "r"(p16_addr + (unsigned int)((24 + cq) * 4 * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&mlall[28])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(28) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(28) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(28) + 3]))
                 : "r"(p16_addr + (unsigned int)((28 + cq) * 4 * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&mlall[32])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(32) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(32) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(32) + 3]))
                 : "r"(p16_addr + (unsigned int)((32 + cq) * 4 * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&mlall[36])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(36) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(36) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(36) + 3]))
                 : "r"(p16_addr + (unsigned int)((36 + cq) * 4 * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&mlall[40])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(40) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(40) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(40) + 3]))
                 : "r"(p16_addr + (unsigned int)((40 + cq) * 4 * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&mlall[44])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(44) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(44) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(44) + 3]))
                 : "r"(p16_addr + (unsigned int)((44 + cq) * 4 * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&mlall[48])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(48) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(48) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(48) + 3]))
                 : "r"(p16_addr + (unsigned int)((48 + cq) * 4 * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&mlall[52])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(52) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(52) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(52) + 3]))
                 : "r"(p16_addr + (unsigned int)((52 + cq) * 4 * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&mlall[56])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(56) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(56) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(56) + 3]))
                 : "r"(p16_addr + (unsigned int)((56 + cq) * 4 * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&mlall[60])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(60) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(60) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&mlall[(60) + 3]))
                 : "r"(p16_addr + (unsigned int)((60 + cq) * 4 * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&pall[0])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(0) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(0) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(0) + 3]))
                 : "r"(kv8_addr + (unsigned int)(tid_1 * 4 * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&pall[4])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(4) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(4) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(4) + 3]))
                 : "r"(kv8_addr + (unsigned int)((128 + tid_1) * 4 * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&pall[8])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(8) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(8) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(8) + 3]))
                 : "r"(kv8_addr + (unsigned int)((256 + tid_1) * 4 * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&pall[12])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(12) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(12) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(12) + 3]))
                 : "r"(kv8_addr + (unsigned int)((384 + tid_1) * 4 * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&pall[16])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(16) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(16) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(16) + 3]))
                 : "r"(kv8_addr + (unsigned int)((512 + tid_1) * 4 * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&pall[20])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(20) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(20) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(20) + 3]))
                 : "r"(kv8_addr + (unsigned int)((640 + tid_1) * 4 * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&pall[24])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(24) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(24) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(24) + 3]))
                 : "r"(kv8_addr + (unsigned int)((768 + tid_1) * 4 * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&pall[28])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(28) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(28) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(28) + 3]))
                 : "r"(kv8_addr + (unsigned int)((896 + tid_1) * 4 * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&pall[32])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(32) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(32) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(32) + 3]))
                 : "r"(kv8_addr + (unsigned int)((1024 + tid_1) * 4 * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&pall[36])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(36) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(36) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(36) + 3]))
                 : "r"(kv8_addr + (unsigned int)((1152 + tid_1) * 4 * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&pall[40])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(40) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(40) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(40) + 3]))
                 : "r"(kv8_addr + (unsigned int)((1280 + tid_1) * 4 * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&pall[44])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(44) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(44) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(44) + 3]))
                 : "r"(kv8_addr + (unsigned int)((1408 + tid_1) * 4 * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&pall[48])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(48) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(48) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(48) + 3]))
                 : "r"(kv8_addr + (unsigned int)((1536 + tid_1) * 4 * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&pall[52])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(52) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(52) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(52) + 3]))
                 : "r"(kv8_addr + (unsigned int)((1664 + tid_1) * 4 * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&pall[56])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(56) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(56) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(56) + 3]))
                 : "r"(kv8_addr + (unsigned int)((1792 + tid_1) * 4 * 4)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&pall[60])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(60) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(60) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&pall[(60) + 3]))
                 : "r"(kv8_addr + (unsigned int)((1920 + tid_1) * 4 * 4)));
    float m0 = -CUDART_INF_F;
    float m1 = -CUDART_INF_F;
    m0 = max_noftz(m0, mlall[0]);
    m1 = max_noftz(m1, mlall[2]);
    m0 = max_noftz(m0, mlall[4]);
    m1 = max_noftz(m1, mlall[6]);
    m0 = max_noftz(m0, mlall[8]);
    m1 = max_noftz(m1, mlall[10]);
    m0 = max_noftz(m0, mlall[12]);
    m1 = max_noftz(m1, mlall[14]);
    m0 = max_noftz(m0, mlall[16]);
    m1 = max_noftz(m1, mlall[18]);
    m0 = max_noftz(m0, mlall[20]);
    m1 = max_noftz(m1, mlall[22]);
    m0 = max_noftz(m0, mlall[24]);
    m1 = max_noftz(m1, mlall[26]);
    m0 = max_noftz(m0, mlall[28]);
    m1 = max_noftz(m1, mlall[30]);
    m0 = max_noftz(m0, mlall[32]);
    m1 = max_noftz(m1, mlall[34]);
    m0 = max_noftz(m0, mlall[36]);
    m1 = max_noftz(m1, mlall[38]);
    m0 = max_noftz(m0, mlall[40]);
    m1 = max_noftz(m1, mlall[42]);
    m0 = max_noftz(m0, mlall[44]);
    m1 = max_noftz(m1, mlall[46]);
    m0 = max_noftz(m0, mlall[48]);
    m1 = max_noftz(m1, mlall[50]);
    m0 = max_noftz(m0, mlall[52]);
    m1 = max_noftz(m1, mlall[54]);
    m0 = max_noftz(m0, mlall[56]);
    m1 = max_noftz(m1, mlall[58]);
    m0 = max_noftz(m0, mlall[60]);
    m1 = max_noftz(m1, mlall[62]);
    cw[0] = 0.0f;
    if (mlall[0] != -CUDART_INF_F) {
      float _exp2_62 = approx_exp2(mlall[0] - m0);
      cw[0] = _exp2_62 * mlall[1];
    }
    cw[1] = 0.0f;
    if (mlall[2] != -CUDART_INF_F) {
      float _exp2_63 = approx_exp2(mlall[2] - m1);
      cw[1] = _exp2_63 * mlall[3];
    }
    lsum[0] = lsum[0] + cw[0];
    lsum[1] = lsum[1] + cw[1];
    cw[2] = 0.0f;
    if (mlall[4] != -CUDART_INF_F) {
      float _exp2_64 = approx_exp2(mlall[4] - m0);
      cw[2] = _exp2_64 * mlall[5];
    }
    cw[3] = 0.0f;
    if (mlall[6] != -CUDART_INF_F) {
      float _exp2_65 = approx_exp2(mlall[6] - m1);
      cw[3] = _exp2_65 * mlall[7];
    }
    lsum[0] = lsum[0] + cw[2];
    lsum[1] = lsum[1] + cw[3];
    cw[4] = 0.0f;
    if (mlall[8] != -CUDART_INF_F) {
      float _exp2_66 = approx_exp2(mlall[8] - m0);
      cw[4] = _exp2_66 * mlall[9];
    }
    cw[5] = 0.0f;
    if (mlall[10] != -CUDART_INF_F) {
      float _exp2_67 = approx_exp2(mlall[10] - m1);
      cw[5] = _exp2_67 * mlall[11];
    }
    lsum[0] = lsum[0] + cw[4];
    lsum[1] = lsum[1] + cw[5];
    cw[6] = 0.0f;
    if (mlall[12] != -CUDART_INF_F) {
      float _exp2_68 = approx_exp2(mlall[12] - m0);
      cw[6] = _exp2_68 * mlall[13];
    }
    cw[7] = 0.0f;
    if (mlall[14] != -CUDART_INF_F) {
      float _exp2_69 = approx_exp2(mlall[14] - m1);
      cw[7] = _exp2_69 * mlall[15];
    }
    lsum[0] = lsum[0] + cw[6];
    lsum[1] = lsum[1] + cw[7];
    cw[8] = 0.0f;
    if (mlall[16] != -CUDART_INF_F) {
      float _exp2_70 = approx_exp2(mlall[16] - m0);
      cw[8] = _exp2_70 * mlall[17];
    }
    cw[9] = 0.0f;
    if (mlall[18] != -CUDART_INF_F) {
      float _exp2_71 = approx_exp2(mlall[18] - m1);
      cw[9] = _exp2_71 * mlall[19];
    }
    lsum[0] = lsum[0] + cw[8];
    lsum[1] = lsum[1] + cw[9];
    cw[10] = 0.0f;
    if (mlall[20] != -CUDART_INF_F) {
      float _exp2_72 = approx_exp2(mlall[20] - m0);
      cw[10] = _exp2_72 * mlall[21];
    }
    cw[11] = 0.0f;
    if (mlall[22] != -CUDART_INF_F) {
      float _exp2_73 = approx_exp2(mlall[22] - m1);
      cw[11] = _exp2_73 * mlall[23];
    }
    lsum[0] = lsum[0] + cw[10];
    lsum[1] = lsum[1] + cw[11];
    cw[12] = 0.0f;
    if (mlall[24] != -CUDART_INF_F) {
      float _exp2_74 = approx_exp2(mlall[24] - m0);
      cw[12] = _exp2_74 * mlall[25];
    }
    cw[13] = 0.0f;
    if (mlall[26] != -CUDART_INF_F) {
      float _exp2_75 = approx_exp2(mlall[26] - m1);
      cw[13] = _exp2_75 * mlall[27];
    }
    lsum[0] = lsum[0] + cw[12];
    lsum[1] = lsum[1] + cw[13];
    cw[14] = 0.0f;
    if (mlall[28] != -CUDART_INF_F) {
      float _exp2_76 = approx_exp2(mlall[28] - m0);
      cw[14] = _exp2_76 * mlall[29];
    }
    cw[15] = 0.0f;
    if (mlall[30] != -CUDART_INF_F) {
      float _exp2_77 = approx_exp2(mlall[30] - m1);
      cw[15] = _exp2_77 * mlall[31];
    }
    lsum[0] = lsum[0] + cw[14];
    lsum[1] = lsum[1] + cw[15];
    cw[16] = 0.0f;
    if (mlall[32] != -CUDART_INF_F) {
      float _exp2_78 = approx_exp2(mlall[32] - m0);
      cw[16] = _exp2_78 * mlall[33];
    }
    cw[17] = 0.0f;
    if (mlall[34] != -CUDART_INF_F) {
      float _exp2_79 = approx_exp2(mlall[34] - m1);
      cw[17] = _exp2_79 * mlall[35];
    }
    lsum[0] = lsum[0] + cw[16];
    lsum[1] = lsum[1] + cw[17];
    cw[18] = 0.0f;
    if (mlall[36] != -CUDART_INF_F) {
      float _exp2_80 = approx_exp2(mlall[36] - m0);
      cw[18] = _exp2_80 * mlall[37];
    }
    cw[19] = 0.0f;
    if (mlall[38] != -CUDART_INF_F) {
      float _exp2_81 = approx_exp2(mlall[38] - m1);
      cw[19] = _exp2_81 * mlall[39];
    }
    lsum[0] = lsum[0] + cw[18];
    lsum[1] = lsum[1] + cw[19];
    cw[20] = 0.0f;
    if (mlall[40] != -CUDART_INF_F) {
      float _exp2_82 = approx_exp2(mlall[40] - m0);
      cw[20] = _exp2_82 * mlall[41];
    }
    cw[21] = 0.0f;
    if (mlall[42] != -CUDART_INF_F) {
      float _exp2_83 = approx_exp2(mlall[42] - m1);
      cw[21] = _exp2_83 * mlall[43];
    }
    lsum[0] = lsum[0] + cw[20];
    lsum[1] = lsum[1] + cw[21];
    cw[22] = 0.0f;
    if (mlall[44] != -CUDART_INF_F) {
      float _exp2_84 = approx_exp2(mlall[44] - m0);
      cw[22] = _exp2_84 * mlall[45];
    }
    cw[23] = 0.0f;
    if (mlall[46] != -CUDART_INF_F) {
      float _exp2_85 = approx_exp2(mlall[46] - m1);
      cw[23] = _exp2_85 * mlall[47];
    }
    lsum[0] = lsum[0] + cw[22];
    lsum[1] = lsum[1] + cw[23];
    cw[24] = 0.0f;
    if (mlall[48] != -CUDART_INF_F) {
      float _exp2_86 = approx_exp2(mlall[48] - m0);
      cw[24] = _exp2_86 * mlall[49];
    }
    cw[25] = 0.0f;
    if (mlall[50] != -CUDART_INF_F) {
      float _exp2_87 = approx_exp2(mlall[50] - m1);
      cw[25] = _exp2_87 * mlall[51];
    }
    lsum[0] = lsum[0] + cw[24];
    lsum[1] = lsum[1] + cw[25];
    cw[26] = 0.0f;
    if (mlall[52] != -CUDART_INF_F) {
      float _exp2_88 = approx_exp2(mlall[52] - m0);
      cw[26] = _exp2_88 * mlall[53];
    }
    cw[27] = 0.0f;
    if (mlall[54] != -CUDART_INF_F) {
      float _exp2_89 = approx_exp2(mlall[54] - m1);
      cw[27] = _exp2_89 * mlall[55];
    }
    lsum[0] = lsum[0] + cw[26];
    lsum[1] = lsum[1] + cw[27];
    cw[28] = 0.0f;
    if (mlall[56] != -CUDART_INF_F) {
      float _exp2_90 = approx_exp2(mlall[56] - m0);
      cw[28] = _exp2_90 * mlall[57];
    }
    cw[29] = 0.0f;
    if (mlall[58] != -CUDART_INF_F) {
      float _exp2_91 = approx_exp2(mlall[58] - m1);
      cw[29] = _exp2_91 * mlall[59];
    }
    lsum[0] = lsum[0] + cw[28];
    lsum[1] = lsum[1] + cw[29];
    cw[30] = 0.0f;
    if (mlall[60] != -CUDART_INF_F) {
      float _exp2_92 = approx_exp2(mlall[60] - m0);
      cw[30] = _exp2_92 * mlall[61];
    }
    cw[31] = 0.0f;
    if (mlall[62] != -CUDART_INF_F) {
      float _exp2_93 = approx_exp2(mlall[62] - m1);
      cw[31] = _exp2_93 * mlall[63];
    }
    lsum[0] = lsum[0] + cw[30];
    lsum[1] = lsum[1] + cw[31];
    float _cvt_f32_f16_94;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_94) : "h"((uint16_t)(pall[0])));
    d_o0[0] = __fmaf_rn(cw[0], _cvt_f32_f16_94, d_o0[0]);
    float _cvt_f32_f16_95;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_95) : "h"((uint16_t)(pall[0] >> 16)));
    d_o0[1] = __fmaf_rn(cw[1], _cvt_f32_f16_95, d_o0[1]);
    float _cvt_f32_f16_96;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_96) : "h"((uint16_t)(pall[1])));
    d_o0[2] = __fmaf_rn(cw[0], _cvt_f32_f16_96, d_o0[2]);
    float _cvt_f32_f16_97;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_97) : "h"((uint16_t)(pall[1] >> 16)));
    d_o0[3] = __fmaf_rn(cw[1], _cvt_f32_f16_97, d_o0[3]);
    float _cvt_f32_f16_98;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_98) : "h"((uint16_t)(pall[2])));
    d_o1[0] = __fmaf_rn(cw[0], _cvt_f32_f16_98, d_o1[0]);
    float _cvt_f32_f16_99;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_99) : "h"((uint16_t)(pall[2] >> 16)));
    d_o1[1] = __fmaf_rn(cw[1], _cvt_f32_f16_99, d_o1[1]);
    float _cvt_f32_f16_100;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_100) : "h"((uint16_t)(pall[3])));
    d_o1[2] = __fmaf_rn(cw[0], _cvt_f32_f16_100, d_o1[2]);
    float _cvt_f32_f16_101;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_101) : "h"((uint16_t)(pall[3] >> 16)));
    d_o1[3] = __fmaf_rn(cw[1], _cvt_f32_f16_101, d_o1[3]);
    float _cvt_f32_f16_102;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_102) : "h"((uint16_t)(pall[4])));
    d_o0[0] = __fmaf_rn(cw[2], _cvt_f32_f16_102, d_o0[0]);
    float _cvt_f32_f16_103;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_103) : "h"((uint16_t)(pall[4] >> 16)));
    d_o0[1] = __fmaf_rn(cw[3], _cvt_f32_f16_103, d_o0[1]);
    float _cvt_f32_f16_104;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_104) : "h"((uint16_t)(pall[5])));
    d_o0[2] = __fmaf_rn(cw[2], _cvt_f32_f16_104, d_o0[2]);
    float _cvt_f32_f16_105;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_105) : "h"((uint16_t)(pall[5] >> 16)));
    d_o0[3] = __fmaf_rn(cw[3], _cvt_f32_f16_105, d_o0[3]);
    float _cvt_f32_f16_106;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_106) : "h"((uint16_t)(pall[6])));
    d_o1[0] = __fmaf_rn(cw[2], _cvt_f32_f16_106, d_o1[0]);
    float _cvt_f32_f16_107;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_107) : "h"((uint16_t)(pall[6] >> 16)));
    d_o1[1] = __fmaf_rn(cw[3], _cvt_f32_f16_107, d_o1[1]);
    float _cvt_f32_f16_108;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_108) : "h"((uint16_t)(pall[7])));
    d_o1[2] = __fmaf_rn(cw[2], _cvt_f32_f16_108, d_o1[2]);
    float _cvt_f32_f16_109;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_109) : "h"((uint16_t)(pall[7] >> 16)));
    d_o1[3] = __fmaf_rn(cw[3], _cvt_f32_f16_109, d_o1[3]);
    float _cvt_f32_f16_110;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_110) : "h"((uint16_t)(pall[8])));
    d_o0[0] = __fmaf_rn(cw[4], _cvt_f32_f16_110, d_o0[0]);
    float _cvt_f32_f16_111;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_111) : "h"((uint16_t)(pall[8] >> 16)));
    d_o0[1] = __fmaf_rn(cw[5], _cvt_f32_f16_111, d_o0[1]);
    float _cvt_f32_f16_112;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_112) : "h"((uint16_t)(pall[9])));
    d_o0[2] = __fmaf_rn(cw[4], _cvt_f32_f16_112, d_o0[2]);
    float _cvt_f32_f16_113;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_113) : "h"((uint16_t)(pall[9] >> 16)));
    d_o0[3] = __fmaf_rn(cw[5], _cvt_f32_f16_113, d_o0[3]);
    float _cvt_f32_f16_114;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_114) : "h"((uint16_t)(pall[10])));
    d_o1[0] = __fmaf_rn(cw[4], _cvt_f32_f16_114, d_o1[0]);
    float _cvt_f32_f16_115;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_115) : "h"((uint16_t)(pall[10] >> 16)));
    d_o1[1] = __fmaf_rn(cw[5], _cvt_f32_f16_115, d_o1[1]);
    float _cvt_f32_f16_116;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_116) : "h"((uint16_t)(pall[11])));
    d_o1[2] = __fmaf_rn(cw[4], _cvt_f32_f16_116, d_o1[2]);
    float _cvt_f32_f16_117;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_117) : "h"((uint16_t)(pall[11] >> 16)));
    d_o1[3] = __fmaf_rn(cw[5], _cvt_f32_f16_117, d_o1[3]);
    float _cvt_f32_f16_118;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_118) : "h"((uint16_t)(pall[12])));
    d_o0[0] = __fmaf_rn(cw[6], _cvt_f32_f16_118, d_o0[0]);
    float _cvt_f32_f16_119;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_119) : "h"((uint16_t)(pall[12] >> 16)));
    d_o0[1] = __fmaf_rn(cw[7], _cvt_f32_f16_119, d_o0[1]);
    float _cvt_f32_f16_120;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_120) : "h"((uint16_t)(pall[13])));
    d_o0[2] = __fmaf_rn(cw[6], _cvt_f32_f16_120, d_o0[2]);
    float _cvt_f32_f16_121;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_121) : "h"((uint16_t)(pall[13] >> 16)));
    d_o0[3] = __fmaf_rn(cw[7], _cvt_f32_f16_121, d_o0[3]);
    float _cvt_f32_f16_122;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_122) : "h"((uint16_t)(pall[14])));
    d_o1[0] = __fmaf_rn(cw[6], _cvt_f32_f16_122, d_o1[0]);
    float _cvt_f32_f16_123;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_123) : "h"((uint16_t)(pall[14] >> 16)));
    d_o1[1] = __fmaf_rn(cw[7], _cvt_f32_f16_123, d_o1[1]);
    float _cvt_f32_f16_124;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_124) : "h"((uint16_t)(pall[15])));
    d_o1[2] = __fmaf_rn(cw[6], _cvt_f32_f16_124, d_o1[2]);
    float _cvt_f32_f16_125;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_125) : "h"((uint16_t)(pall[15] >> 16)));
    d_o1[3] = __fmaf_rn(cw[7], _cvt_f32_f16_125, d_o1[3]);
    float _cvt_f32_f16_126;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_126) : "h"((uint16_t)(pall[16])));
    d_o0[0] = __fmaf_rn(cw[8], _cvt_f32_f16_126, d_o0[0]);
    float _cvt_f32_f16_127;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_127) : "h"((uint16_t)(pall[16] >> 16)));
    d_o0[1] = __fmaf_rn(cw[9], _cvt_f32_f16_127, d_o0[1]);
    float _cvt_f32_f16_128;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_128) : "h"((uint16_t)(pall[17])));
    d_o0[2] = __fmaf_rn(cw[8], _cvt_f32_f16_128, d_o0[2]);
    float _cvt_f32_f16_129;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_129) : "h"((uint16_t)(pall[17] >> 16)));
    d_o0[3] = __fmaf_rn(cw[9], _cvt_f32_f16_129, d_o0[3]);
    float _cvt_f32_f16_130;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_130) : "h"((uint16_t)(pall[18])));
    d_o1[0] = __fmaf_rn(cw[8], _cvt_f32_f16_130, d_o1[0]);
    float _cvt_f32_f16_131;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_131) : "h"((uint16_t)(pall[18] >> 16)));
    d_o1[1] = __fmaf_rn(cw[9], _cvt_f32_f16_131, d_o1[1]);
    float _cvt_f32_f16_132;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_132) : "h"((uint16_t)(pall[19])));
    d_o1[2] = __fmaf_rn(cw[8], _cvt_f32_f16_132, d_o1[2]);
    float _cvt_f32_f16_133;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_133) : "h"((uint16_t)(pall[19] >> 16)));
    d_o1[3] = __fmaf_rn(cw[9], _cvt_f32_f16_133, d_o1[3]);
    float _cvt_f32_f16_134;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_134) : "h"((uint16_t)(pall[20])));
    d_o0[0] = __fmaf_rn(cw[10], _cvt_f32_f16_134, d_o0[0]);
    float _cvt_f32_f16_135;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_135) : "h"((uint16_t)(pall[20] >> 16)));
    d_o0[1] = __fmaf_rn(cw[11], _cvt_f32_f16_135, d_o0[1]);
    float _cvt_f32_f16_136;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_136) : "h"((uint16_t)(pall[21])));
    d_o0[2] = __fmaf_rn(cw[10], _cvt_f32_f16_136, d_o0[2]);
    float _cvt_f32_f16_137;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_137) : "h"((uint16_t)(pall[21] >> 16)));
    d_o0[3] = __fmaf_rn(cw[11], _cvt_f32_f16_137, d_o0[3]);
    float _cvt_f32_f16_138;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_138) : "h"((uint16_t)(pall[22])));
    d_o1[0] = __fmaf_rn(cw[10], _cvt_f32_f16_138, d_o1[0]);
    float _cvt_f32_f16_139;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_139) : "h"((uint16_t)(pall[22] >> 16)));
    d_o1[1] = __fmaf_rn(cw[11], _cvt_f32_f16_139, d_o1[1]);
    float _cvt_f32_f16_140;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_140) : "h"((uint16_t)(pall[23])));
    d_o1[2] = __fmaf_rn(cw[10], _cvt_f32_f16_140, d_o1[2]);
    float _cvt_f32_f16_141;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_141) : "h"((uint16_t)(pall[23] >> 16)));
    d_o1[3] = __fmaf_rn(cw[11], _cvt_f32_f16_141, d_o1[3]);
    float _cvt_f32_f16_142;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_142) : "h"((uint16_t)(pall[24])));
    d_o0[0] = __fmaf_rn(cw[12], _cvt_f32_f16_142, d_o0[0]);
    float _cvt_f32_f16_143;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_143) : "h"((uint16_t)(pall[24] >> 16)));
    d_o0[1] = __fmaf_rn(cw[13], _cvt_f32_f16_143, d_o0[1]);
    float _cvt_f32_f16_144;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_144) : "h"((uint16_t)(pall[25])));
    d_o0[2] = __fmaf_rn(cw[12], _cvt_f32_f16_144, d_o0[2]);
    float _cvt_f32_f16_145;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_145) : "h"((uint16_t)(pall[25] >> 16)));
    d_o0[3] = __fmaf_rn(cw[13], _cvt_f32_f16_145, d_o0[3]);
    float _cvt_f32_f16_146;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_146) : "h"((uint16_t)(pall[26])));
    d_o1[0] = __fmaf_rn(cw[12], _cvt_f32_f16_146, d_o1[0]);
    float _cvt_f32_f16_147;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_147) : "h"((uint16_t)(pall[26] >> 16)));
    d_o1[1] = __fmaf_rn(cw[13], _cvt_f32_f16_147, d_o1[1]);
    float _cvt_f32_f16_148;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_148) : "h"((uint16_t)(pall[27])));
    d_o1[2] = __fmaf_rn(cw[12], _cvt_f32_f16_148, d_o1[2]);
    float _cvt_f32_f16_149;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_149) : "h"((uint16_t)(pall[27] >> 16)));
    d_o1[3] = __fmaf_rn(cw[13], _cvt_f32_f16_149, d_o1[3]);
    float _cvt_f32_f16_150;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_150) : "h"((uint16_t)(pall[28])));
    d_o0[0] = __fmaf_rn(cw[14], _cvt_f32_f16_150, d_o0[0]);
    float _cvt_f32_f16_151;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_151) : "h"((uint16_t)(pall[28] >> 16)));
    d_o0[1] = __fmaf_rn(cw[15], _cvt_f32_f16_151, d_o0[1]);
    float _cvt_f32_f16_152;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_152) : "h"((uint16_t)(pall[29])));
    d_o0[2] = __fmaf_rn(cw[14], _cvt_f32_f16_152, d_o0[2]);
    float _cvt_f32_f16_153;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_153) : "h"((uint16_t)(pall[29] >> 16)));
    d_o0[3] = __fmaf_rn(cw[15], _cvt_f32_f16_153, d_o0[3]);
    float _cvt_f32_f16_154;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_154) : "h"((uint16_t)(pall[30])));
    d_o1[0] = __fmaf_rn(cw[14], _cvt_f32_f16_154, d_o1[0]);
    float _cvt_f32_f16_155;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_155) : "h"((uint16_t)(pall[30] >> 16)));
    d_o1[1] = __fmaf_rn(cw[15], _cvt_f32_f16_155, d_o1[1]);
    float _cvt_f32_f16_156;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_156) : "h"((uint16_t)(pall[31])));
    d_o1[2] = __fmaf_rn(cw[14], _cvt_f32_f16_156, d_o1[2]);
    float _cvt_f32_f16_157;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_157) : "h"((uint16_t)(pall[31] >> 16)));
    d_o1[3] = __fmaf_rn(cw[15], _cvt_f32_f16_157, d_o1[3]);
    float _cvt_f32_f16_158;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_158) : "h"((uint16_t)(pall[32])));
    d_o0[0] = __fmaf_rn(cw[16], _cvt_f32_f16_158, d_o0[0]);
    float _cvt_f32_f16_159;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_159) : "h"((uint16_t)(pall[32] >> 16)));
    d_o0[1] = __fmaf_rn(cw[17], _cvt_f32_f16_159, d_o0[1]);
    float _cvt_f32_f16_160;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_160) : "h"((uint16_t)(pall[33])));
    d_o0[2] = __fmaf_rn(cw[16], _cvt_f32_f16_160, d_o0[2]);
    float _cvt_f32_f16_161;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_161) : "h"((uint16_t)(pall[33] >> 16)));
    d_o0[3] = __fmaf_rn(cw[17], _cvt_f32_f16_161, d_o0[3]);
    float _cvt_f32_f16_162;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_162) : "h"((uint16_t)(pall[34])));
    d_o1[0] = __fmaf_rn(cw[16], _cvt_f32_f16_162, d_o1[0]);
    float _cvt_f32_f16_163;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_163) : "h"((uint16_t)(pall[34] >> 16)));
    d_o1[1] = __fmaf_rn(cw[17], _cvt_f32_f16_163, d_o1[1]);
    float _cvt_f32_f16_164;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_164) : "h"((uint16_t)(pall[35])));
    d_o1[2] = __fmaf_rn(cw[16], _cvt_f32_f16_164, d_o1[2]);
    float _cvt_f32_f16_165;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_165) : "h"((uint16_t)(pall[35] >> 16)));
    d_o1[3] = __fmaf_rn(cw[17], _cvt_f32_f16_165, d_o1[3]);
    float _cvt_f32_f16_166;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_166) : "h"((uint16_t)(pall[36])));
    d_o0[0] = __fmaf_rn(cw[18], _cvt_f32_f16_166, d_o0[0]);
    float _cvt_f32_f16_167;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_167) : "h"((uint16_t)(pall[36] >> 16)));
    d_o0[1] = __fmaf_rn(cw[19], _cvt_f32_f16_167, d_o0[1]);
    float _cvt_f32_f16_168;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_168) : "h"((uint16_t)(pall[37])));
    d_o0[2] = __fmaf_rn(cw[18], _cvt_f32_f16_168, d_o0[2]);
    float _cvt_f32_f16_169;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_169) : "h"((uint16_t)(pall[37] >> 16)));
    d_o0[3] = __fmaf_rn(cw[19], _cvt_f32_f16_169, d_o0[3]);
    float _cvt_f32_f16_170;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_170) : "h"((uint16_t)(pall[38])));
    d_o1[0] = __fmaf_rn(cw[18], _cvt_f32_f16_170, d_o1[0]);
    float _cvt_f32_f16_171;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_171) : "h"((uint16_t)(pall[38] >> 16)));
    d_o1[1] = __fmaf_rn(cw[19], _cvt_f32_f16_171, d_o1[1]);
    float _cvt_f32_f16_172;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_172) : "h"((uint16_t)(pall[39])));
    d_o1[2] = __fmaf_rn(cw[18], _cvt_f32_f16_172, d_o1[2]);
    float _cvt_f32_f16_173;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_173) : "h"((uint16_t)(pall[39] >> 16)));
    d_o1[3] = __fmaf_rn(cw[19], _cvt_f32_f16_173, d_o1[3]);
    float _cvt_f32_f16_174;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_174) : "h"((uint16_t)(pall[40])));
    d_o0[0] = __fmaf_rn(cw[20], _cvt_f32_f16_174, d_o0[0]);
    float _cvt_f32_f16_175;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_175) : "h"((uint16_t)(pall[40] >> 16)));
    d_o0[1] = __fmaf_rn(cw[21], _cvt_f32_f16_175, d_o0[1]);
    float _cvt_f32_f16_176;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_176) : "h"((uint16_t)(pall[41])));
    d_o0[2] = __fmaf_rn(cw[20], _cvt_f32_f16_176, d_o0[2]);
    float _cvt_f32_f16_177;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_177) : "h"((uint16_t)(pall[41] >> 16)));
    d_o0[3] = __fmaf_rn(cw[21], _cvt_f32_f16_177, d_o0[3]);
    float _cvt_f32_f16_178;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_178) : "h"((uint16_t)(pall[42])));
    d_o1[0] = __fmaf_rn(cw[20], _cvt_f32_f16_178, d_o1[0]);
    float _cvt_f32_f16_179;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_179) : "h"((uint16_t)(pall[42] >> 16)));
    d_o1[1] = __fmaf_rn(cw[21], _cvt_f32_f16_179, d_o1[1]);
    float _cvt_f32_f16_180;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_180) : "h"((uint16_t)(pall[43])));
    d_o1[2] = __fmaf_rn(cw[20], _cvt_f32_f16_180, d_o1[2]);
    float _cvt_f32_f16_181;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_181) : "h"((uint16_t)(pall[43] >> 16)));
    d_o1[3] = __fmaf_rn(cw[21], _cvt_f32_f16_181, d_o1[3]);
    float _cvt_f32_f16_182;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_182) : "h"((uint16_t)(pall[44])));
    d_o0[0] = __fmaf_rn(cw[22], _cvt_f32_f16_182, d_o0[0]);
    float _cvt_f32_f16_183;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_183) : "h"((uint16_t)(pall[44] >> 16)));
    d_o0[1] = __fmaf_rn(cw[23], _cvt_f32_f16_183, d_o0[1]);
    float _cvt_f32_f16_184;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_184) : "h"((uint16_t)(pall[45])));
    d_o0[2] = __fmaf_rn(cw[22], _cvt_f32_f16_184, d_o0[2]);
    float _cvt_f32_f16_185;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_185) : "h"((uint16_t)(pall[45] >> 16)));
    d_o0[3] = __fmaf_rn(cw[23], _cvt_f32_f16_185, d_o0[3]);
    float _cvt_f32_f16_186;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_186) : "h"((uint16_t)(pall[46])));
    d_o1[0] = __fmaf_rn(cw[22], _cvt_f32_f16_186, d_o1[0]);
    float _cvt_f32_f16_187;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_187) : "h"((uint16_t)(pall[46] >> 16)));
    d_o1[1] = __fmaf_rn(cw[23], _cvt_f32_f16_187, d_o1[1]);
    float _cvt_f32_f16_188;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_188) : "h"((uint16_t)(pall[47])));
    d_o1[2] = __fmaf_rn(cw[22], _cvt_f32_f16_188, d_o1[2]);
    float _cvt_f32_f16_189;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_189) : "h"((uint16_t)(pall[47] >> 16)));
    d_o1[3] = __fmaf_rn(cw[23], _cvt_f32_f16_189, d_o1[3]);
    float _cvt_f32_f16_190;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_190) : "h"((uint16_t)(pall[48])));
    d_o0[0] = __fmaf_rn(cw[24], _cvt_f32_f16_190, d_o0[0]);
    float _cvt_f32_f16_191;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_191) : "h"((uint16_t)(pall[48] >> 16)));
    d_o0[1] = __fmaf_rn(cw[25], _cvt_f32_f16_191, d_o0[1]);
    float _cvt_f32_f16_192;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_192) : "h"((uint16_t)(pall[49])));
    d_o0[2] = __fmaf_rn(cw[24], _cvt_f32_f16_192, d_o0[2]);
    float _cvt_f32_f16_193;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_193) : "h"((uint16_t)(pall[49] >> 16)));
    d_o0[3] = __fmaf_rn(cw[25], _cvt_f32_f16_193, d_o0[3]);
    float _cvt_f32_f16_194;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_194) : "h"((uint16_t)(pall[50])));
    d_o1[0] = __fmaf_rn(cw[24], _cvt_f32_f16_194, d_o1[0]);
    float _cvt_f32_f16_195;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_195) : "h"((uint16_t)(pall[50] >> 16)));
    d_o1[1] = __fmaf_rn(cw[25], _cvt_f32_f16_195, d_o1[1]);
    float _cvt_f32_f16_196;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_196) : "h"((uint16_t)(pall[51])));
    d_o1[2] = __fmaf_rn(cw[24], _cvt_f32_f16_196, d_o1[2]);
    float _cvt_f32_f16_197;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_197) : "h"((uint16_t)(pall[51] >> 16)));
    d_o1[3] = __fmaf_rn(cw[25], _cvt_f32_f16_197, d_o1[3]);
    float _cvt_f32_f16_198;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_198) : "h"((uint16_t)(pall[52])));
    d_o0[0] = __fmaf_rn(cw[26], _cvt_f32_f16_198, d_o0[0]);
    float _cvt_f32_f16_199;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_199) : "h"((uint16_t)(pall[52] >> 16)));
    d_o0[1] = __fmaf_rn(cw[27], _cvt_f32_f16_199, d_o0[1]);
    float _cvt_f32_f16_200;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_200) : "h"((uint16_t)(pall[53])));
    d_o0[2] = __fmaf_rn(cw[26], _cvt_f32_f16_200, d_o0[2]);
    float _cvt_f32_f16_201;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_201) : "h"((uint16_t)(pall[53] >> 16)));
    d_o0[3] = __fmaf_rn(cw[27], _cvt_f32_f16_201, d_o0[3]);
    float _cvt_f32_f16_202;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_202) : "h"((uint16_t)(pall[54])));
    d_o1[0] = __fmaf_rn(cw[26], _cvt_f32_f16_202, d_o1[0]);
    float _cvt_f32_f16_203;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_203) : "h"((uint16_t)(pall[54] >> 16)));
    d_o1[1] = __fmaf_rn(cw[27], _cvt_f32_f16_203, d_o1[1]);
    float _cvt_f32_f16_204;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_204) : "h"((uint16_t)(pall[55])));
    d_o1[2] = __fmaf_rn(cw[26], _cvt_f32_f16_204, d_o1[2]);
    float _cvt_f32_f16_205;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_205) : "h"((uint16_t)(pall[55] >> 16)));
    d_o1[3] = __fmaf_rn(cw[27], _cvt_f32_f16_205, d_o1[3]);
    float _cvt_f32_f16_206;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_206) : "h"((uint16_t)(pall[56])));
    d_o0[0] = __fmaf_rn(cw[28], _cvt_f32_f16_206, d_o0[0]);
    float _cvt_f32_f16_207;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_207) : "h"((uint16_t)(pall[56] >> 16)));
    d_o0[1] = __fmaf_rn(cw[29], _cvt_f32_f16_207, d_o0[1]);
    float _cvt_f32_f16_208;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_208) : "h"((uint16_t)(pall[57])));
    d_o0[2] = __fmaf_rn(cw[28], _cvt_f32_f16_208, d_o0[2]);
    float _cvt_f32_f16_209;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_209) : "h"((uint16_t)(pall[57] >> 16)));
    d_o0[3] = __fmaf_rn(cw[29], _cvt_f32_f16_209, d_o0[3]);
    float _cvt_f32_f16_210;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_210) : "h"((uint16_t)(pall[58])));
    d_o1[0] = __fmaf_rn(cw[28], _cvt_f32_f16_210, d_o1[0]);
    float _cvt_f32_f16_211;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_211) : "h"((uint16_t)(pall[58] >> 16)));
    d_o1[1] = __fmaf_rn(cw[29], _cvt_f32_f16_211, d_o1[1]);
    float _cvt_f32_f16_212;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_212) : "h"((uint16_t)(pall[59])));
    d_o1[2] = __fmaf_rn(cw[28], _cvt_f32_f16_212, d_o1[2]);
    float _cvt_f32_f16_213;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_213) : "h"((uint16_t)(pall[59] >> 16)));
    d_o1[3] = __fmaf_rn(cw[29], _cvt_f32_f16_213, d_o1[3]);
    float _cvt_f32_f16_214;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_214) : "h"((uint16_t)(pall[60])));
    d_o0[0] = __fmaf_rn(cw[30], _cvt_f32_f16_214, d_o0[0]);
    float _cvt_f32_f16_215;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_215) : "h"((uint16_t)(pall[60] >> 16)));
    d_o0[1] = __fmaf_rn(cw[31], _cvt_f32_f16_215, d_o0[1]);
    float _cvt_f32_f16_216;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_216) : "h"((uint16_t)(pall[61])));
    d_o0[2] = __fmaf_rn(cw[30], _cvt_f32_f16_216, d_o0[2]);
    float _cvt_f32_f16_217;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_217) : "h"((uint16_t)(pall[61] >> 16)));
    d_o0[3] = __fmaf_rn(cw[31], _cvt_f32_f16_217, d_o0[3]);
    float _cvt_f32_f16_218;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_218) : "h"((uint16_t)(pall[62])));
    d_o1[0] = __fmaf_rn(cw[30], _cvt_f32_f16_218, d_o1[0]);
    float _cvt_f32_f16_219;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_219) : "h"((uint16_t)(pall[62] >> 16)));
    d_o1[1] = __fmaf_rn(cw[31], _cvt_f32_f16_219, d_o1[1]);
    float _cvt_f32_f16_220;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_220) : "h"((uint16_t)(pall[63])));
    d_o1[2] = __fmaf_rn(cw[30], _cvt_f32_f16_220, d_o1[2]);
    float _cvt_f32_f16_221;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_221) : "h"((uint16_t)(pall[63] >> 16)));
    d_o1[3] = __fmaf_rn(cw[31], _cvt_f32_f16_221, d_o1[3]);
  }
  if (merge != 0) {
    float inv = 0.0f;
    if (lsum[0] != 0.0f) {
      float _rcp_222 = approx_rcp(lsum[0]);
      inv = _rcp_222;
    }
    lsum[0] = inv;
    float inv_0 = 0.0f;
    if (lsum[1] != 0.0f) {
      float _rcp_223 = approx_rcp(lsum[1]);
      inv_0 = _rcp_223;
    }
    lsum[1] = inv_0;
    int hh6 = 2 * cq;
    int dd = warp_1 * 16 + g;
    {
      uint32_t _addr_224 =
          static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6 * 128 + dd) * 4));
      asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_224), "f"(d_o0[0] * lsum[0]) : "memory");
    }
    {
      uint32_t _addr_225 =
          static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6 * 128 + dd + 64) * 4));
      asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_225), "f"(d_o1[0] * lsum[0]) : "memory");
    }
    int hh6_1 = 2 * cq + 1;
    int dd_2 = warp_1 * 16 + g;
    {
      uint32_t _addr_226 =
          static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6_1 * 128 + dd_2) * 4));
      asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_226), "f"(d_o0[1] * lsum[1]) : "memory");
    }
    {
      uint32_t _addr_227 =
          static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6_1 * 128 + dd_2 + 64) * 4));
      asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_227), "f"(d_o1[1] * lsum[1]) : "memory");
    }
    int hh6_3 = 2 * cq;
    int dd_4 = warp_1 * 16 + g + 8;
    {
      uint32_t _addr_228 =
          static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6_3 * 128 + dd_4) * 4));
      asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_228), "f"(d_o0[2] * lsum[0]) : "memory");
    }
    {
      uint32_t _addr_229 =
          static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6_3 * 128 + dd_4 + 64) * 4));
      asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_229), "f"(d_o1[2] * lsum[0]) : "memory");
    }
    int hh6_5 = 2 * cq + 1;
    int dd_6 = warp_1 * 16 + g + 8;
    {
      uint32_t _addr_230 =
          static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6_5 * 128 + dd_6) * 4));
      asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_230), "f"(d_o0[3] * lsum[1]) : "memory");
    }
    {
      uint32_t _addr_231 =
          static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6_5 * 128 + dd_6 + 64) * 4));
      asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_231), "f"(d_o1[3] * lsum[1]) : "memory");
    }
    asm volatile("barrier.sync 1, 128;" ::: "memory");
    float ov[8];
    int vid3 = tid_1;
    int on = vid3 / 16;
    int oseg = vid3 - on * 16;
    int od0 = oseg * 8;
    if (on < heads_valid) {
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&ov[0])),
                     "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 3]))
                   : "r"(ident8_addr + (unsigned int)((on * 128 + od0) * 4)));
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&ov[4])),
                     "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 3]))
                   : "r"(ident8_addr + (unsigned int)((on * 128 + od0 + 4) * 4)));
      {
        __nv_bfloat162 _pk[4];
        _pk[0] = __floats2bfloat162_rn(ov[0 + 0], ov[0 + 1]);
        _pk[1] = __floats2bfloat162_rn(ov[0 + 2], ov[0 + 3]);
        _pk[2] = __floats2bfloat162_rn(ov[0 + 4], ov[0 + 5]);
        _pk[3] = __floats2bfloat162_rn(ov[0 + 6], ov[0 + 7]);
        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + ((q_row0 + on) * 128 + od0)))[0]) =
            *reinterpret_cast<uint4*>(&_pk[0]);
      }
    }
  }

  // Cleanup
  __syncthreads();
}

}  // extern "C"
