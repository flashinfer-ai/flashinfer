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

#define NUM_MAIN_STAGES 1
#define SMEM_KV8_OFF 1024
#define SMEM_KV8_STAGE_BYTES 16384
#define SMEM_KV8_STRIDE 16384
#define SMEM_IDENT8_OFF 33792
#define SMEM_IDENT8_STAGE_BYTES 8192
#define SMEM_IDENT8_STRIDE 8192
#define SMEM_Q16_OFF 41984
#define SMEM_Q16_STAGE_BYTES 4096
#define SMEM_Q16_STRIDE 4096
#define SMEM_P16_OFF 46080
#define SMEM_P16_STAGE_BYTES 2048
#define SMEM_P16_STRIDE 2048
#define SMEM_RED_OFF 54272
#define SMEM_RED_STAGE_BYTES 256
#define SMEM_RED_STRIDE 256
#define SMEM_LRED_OFF 54528
#define SMEM_LRED_STAGE_BYTES 256
#define SMEM_LRED_STRIDE 256
#define SMEM_FLAG_OFF 54784
#define SMEM_FLAG_STAGE_BYTES 16
#define SMEM_FLAG_STRIDE 16
#define SMEM_META_OFF 54800
#define SMEM_META_STAGE_BYTES 128
#define SMEM_META_STRIDE 128
#define SMEM_TOTAL 55040
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 4

extern "C" {

__global__
__launch_bounds__(128, LAUNCH_MIN_BLOCKS) void kernel_cake_hopper_msa_56dcad5539ad9283e4c4(
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
  uint8_t* ident8 = reinterpret_cast<uint8_t*>(smem_raw + 33792);
  const int ident8_addr = smem + 33792;
  __half* q16 = reinterpret_cast<__half*>(smem_raw + 41984);
  const int q16_addr = smem + 41984;
  __half* p16 = reinterpret_cast<__half*>(smem_raw + 46080);
  const int p16_addr = smem + 46080;
  float* red = reinterpret_cast<float*>(smem_raw + 54272);
  const int red_addr = smem + 54272;
  float* lred = reinterpret_cast<float*>(smem_raw + 54528);
  const int lred_addr = smem + 54528;
  unsigned int* flag = reinterpret_cast<unsigned int*>(smem_raw + 54784);
  const int flag_addr = smem + 54784;
  int* meta = reinterpret_cast<int*>(smem_raw + 54800);
  const int meta_addr = smem + 54800;

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
  int gc = gx;
  int split = gx - gc;
  int kv_head = gc / num_chunks;
  int chunk = gc - kv_head * num_chunks;
  int batch = tok / seqlen_q;
  int q_in_batch = tok - batch * seqlen_q;
  int kv_len = seqused_k[batch];
  int q_pos = kv_len - seqlen_q + q_in_batch;
  int group = num_q_heads / num_kv_heads;
  int head0 = kv_head * group + chunk * 16;
  int q_row0 = tok * num_q_heads + head0;
  int heads_valid = group - chunk * 16;
  if (heads_valid > 16) {
    heads_valid = 16;
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
    int slot = split * 16;
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
        mbarrier_arrive_expect_tx(kv_full_addr, 16384);
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(kv8_addr),
            "l"((&K)), "r"(0), "r"(0), "r"(page_head), "r"(kv_full_addr), "l"(0x12F0000000000000ULL)
            : "memory");
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(kv8_addr + 8192),
            "l"((&V)), "r"(0), "r"(0), "r"(page_head), "r"(kv_full_addr), "l"(0x12F0000000000000ULL)
            : "memory");
      }
    }
    int slot_3 = split * 16;
    int blk_4 = meta[slot_3];
    int page_head_5 = meta[16 + slot_3];
    int valid_6 = 0;
    if (blk_4 >= 0) {
      valid_6 = q_pos + 1 - blk_4 * 128;
      if (valid_6 > 128) {
        valid_6 = 128;
      }
      if (valid_6 < 0) {
        valid_6 = 0;
      }
    }
    if (warp == 1) {
      if (elect_sync()) {
        mbarrier_arrive_expect_tx(kv_full_addr + 8, 16384);
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(kv8_addr + 16384),
            "l"((&K)), "r"(0), "r"(64), "r"(page_head_5), "r"(kv_full_addr + 8),
            "l"(0x12F0000000000000ULL)
            : "memory");
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(kv8_addr + 16384 + 8192),
            "l"((&V)), "r"(0), "r"(64), "r"(page_head_5), "r"(kv_full_addr + 8),
            "l"(0x12F0000000000000ULL)
            : "memory");
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
    unsigned int qw[12];
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
    int vid_8 = sid + 192;
    int qn_9 = vid_8 / 16;
    int qv_10 = vid_8 - qn_9 * 16;
    qw[8] = 0;
    qw[9] = 0;
    qw[10] = 0;
    qw[11] = 0;
    if (qn_9 < heads_valid) {
      int qbase_2 = (q_row0 + qn_9) * 64 + qv_10 * 4;
      qw[8] = Q32[qbase_2];
      qw[9] = Q32[qbase_2 + 1];
      qw[10] = Q32[qbase_2 + 2];
      qw[11] = Q32[qbase_2 + 3];
    }
    int vid2 = sid;
    if (vid2 < 256) {
      int qn2 = vid2 / 16;
      int qv2 = vid2 - qn2 * 16;
      int qslab = qv2 / 8;
      int qcol = (qv2 - qslab * 8) * 16;
      int qoff = qslab * 2048 + qn2 * 128 + (qcol ^ (qn2 & 7) << 4);
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
    int vid2_11 = sid + 96;
    if (vid2_11 < 256) {
      int qn2_1 = vid2_11 / 16;
      int qv2_1 = vid2_11 - qn2_1 * 16;
      int qslab_1 = qv2_1 / 8;
      int qcol_1 = (qv2_1 - qslab_1 * 8) * 16;
      int qoff_1 = qslab_1 * 2048 + qn2_1 * 128 + (qcol_1 ^ (qn2_1 & 7) << 4);
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
    int vid2_12 = sid + 192;
    if (vid2_12 < 256) {
      int qn2_2 = vid2_12 / 16;
      int qv2_2 = vid2_12 - qn2_2 * 16;
      int qslab_2 = qv2_2 / 8;
      int qcol_2 = (qv2_2 - qslab_2 * 8) * 16;
      int qoff_2 = qslab_2 * 2048 + qn2_2 * 128 + (qcol_2 ^ (qn2_2 & 7) << 4);
      float _cvt_f32_bf16_25;
      asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_25) : "h"((uint16_t)(qw[8])));
      float _cvt_f32_bf16_26;
      asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_26) : "h"((uint16_t)(qw[8] >> 16)));
      uint32_t _cvt_rn_f16x2_f32_24;
      asm("cvt.rn.f16x2.f32 %0, %1, %2;"
          : "=r"(_cvt_rn_f16x2_f32_24)
          : "f"(_cvt_f32_bf16_26), "f"(_cvt_f32_bf16_25));
      qw[8] = _cvt_rn_f16x2_f32_24;
      float _cvt_f32_bf16_28;
      asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_28) : "h"((uint16_t)(qw[9])));
      float _cvt_f32_bf16_29;
      asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_29) : "h"((uint16_t)(qw[9] >> 16)));
      uint32_t _cvt_rn_f16x2_f32_27;
      asm("cvt.rn.f16x2.f32 %0, %1, %2;"
          : "=r"(_cvt_rn_f16x2_f32_27)
          : "f"(_cvt_f32_bf16_29), "f"(_cvt_f32_bf16_28));
      qw[9] = _cvt_rn_f16x2_f32_27;
      float _cvt_f32_bf16_31;
      asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_31) : "h"((uint16_t)(qw[10])));
      float _cvt_f32_bf16_32;
      asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_32) : "h"((uint16_t)(qw[10] >> 16)));
      uint32_t _cvt_rn_f16x2_f32_30;
      asm("cvt.rn.f16x2.f32 %0, %1, %2;"
          : "=r"(_cvt_rn_f16x2_f32_30)
          : "f"(_cvt_f32_bf16_32), "f"(_cvt_f32_bf16_31));
      qw[10] = _cvt_rn_f16x2_f32_30;
      float _cvt_f32_bf16_34;
      asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_34) : "h"((uint16_t)(qw[11])));
      float _cvt_f32_bf16_35;
      asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_35) : "h"((uint16_t)(qw[11] >> 16)));
      uint32_t _cvt_rn_f16x2_f32_33;
      asm("cvt.rn.f16x2.f32 %0, %1, %2;"
          : "=r"(_cvt_rn_f16x2_f32_33)
          : "f"(_cvt_f32_bf16_35), "f"(_cvt_f32_bf16_34));
      qw[11] = _cvt_rn_f16x2_f32_33;
      asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(q16_addr + (unsigned int)qoff_2),
                   "r"(qw[8]), "r"(qw[9]), "r"(qw[10]), "r"(qw[11])
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
  float d_s0[8];
  float d_s1[8];
  float d_s0a[8];
  float d_s0b[8];
  float d_s1a[8];
  float d_s1b[8];
  float d_o0[8];
  float d_o1[8];
  float d_o0x[8];
  float d_o1x[8];
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
  d_o0[4] = 0.0f;
  d_o1[4] = 0.0f;
  d_o0x[4] = 0.0f;
  d_o1x[4] = 0.0f;
  d_o0[5] = 0.0f;
  d_o1[5] = 0.0f;
  d_o0x[5] = 0.0f;
  d_o1x[5] = 0.0f;
  d_o0[6] = 0.0f;
  d_o1[6] = 0.0f;
  d_o0x[6] = 0.0f;
  d_o1x[6] = 0.0f;
  d_o0[7] = 0.0f;
  d_o1[7] = 0.0f;
  d_o0x[7] = 0.0f;
  d_o1x[7] = 0.0f;
  float mrun[4];
  float lp[4];
  mrun[0] = -CUDART_INF_F;
  lp[0] = 0.0f;
  mrun[1] = -CUDART_INF_F;
  lp[1] = 0.0f;
  mrun[2] = -CUDART_INF_F;
  lp[2] = 0.0f;
  mrun[3] = -CUDART_INF_F;
  lp[3] = 0.0f;
  float sv0[8];
  float sv1[8];
  float tmax[4];
  float rv[4];
  float msub[4];
  uint64_t _wgmma_desc_36 = (((uint64_t)(((ident8_addr)) >> 4) & 0x3FFFULL) |
                             ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
  uint64_t _wgmma_b_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_36 >> 32)) << 32) |
                        (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_36);
  uint64_t _wgmma_desc_37 = (((uint64_t)(((q16_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) |
                             ((uint64_t)(64) << 32) | (1ULL << 62));
  uint64_t _wgmma_b_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_37 >> 32)) << 32) |
                        (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_37);
  uint64_t _wgmma_desc_38 = (((uint64_t)(((q16_addr + 2048)) >> 4) & 0x3FFFULL) |
                             ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
  uint64_t _wgmma_b_2 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_38 >> 32)) << 32) |
                        (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_38);
#pragma unroll 1
  for (int j = 0; j < 16; j++) {
    int stage = 2 * j % 2;
    int phase = 2 * j / 2 % 2;
    int stage1 = (2 * j + 1) % 2;
    int phase1 = (2 * j + 1) / 2 % 2;
    int par = j - j / 2 * 2;
    mbarrier_wait(kv_full_addr + (stage) * 8, phase);
    int slot_1 = split * 16 + j;
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
    uint64_t _wgmma_desc_39 =
        (((uint64_t)(((kv8_addr + (unsigned int)(stage * 16384))) >> 4) & 0x3FFFULL) |
         ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_a_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_39 >> 32)) << 32) |
                          (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_39);
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
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
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
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 0, 1, 1, 0;\n}\n"
        : "+f"(d_s0a[0]), "+f"(d_s0a[1]), "+f"(d_s0a[2]), "+f"(d_s0a[3]), "+f"(d_s0a[4]),
          "+f"(d_s0a[5]), "+f"(d_s0a[6]), "+f"(d_s0a[7])
        : "r"(ak0[0]), "r"(ak0[1]), "r"(ak0[2]), "r"(ak0[3]), "l"(_wgmma_b_1)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 0, 1, 1, 0;\n}\n"
        : "+f"(d_s0b[0]), "+f"(d_s0b[1]), "+f"(d_s0b[2]), "+f"(d_s0b[3]), "+f"(d_s0b[4]),
          "+f"(d_s0b[5]), "+f"(d_s0b[6]), "+f"(d_s0b[7])
        : "r"(ak0[16]), "r"(ak0[(16) + 1]), "r"(ak0[(16) + 2]), "r"(ak0[(16) + 3]), "l"(_wgmma_b_2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s0a[0]), "+f"(d_s0a[1]), "+f"(d_s0a[2]), "+f"(d_s0a[3]), "+f"(d_s0a[4]),
          "+f"(d_s0a[5]), "+f"(d_s0a[6]), "+f"(d_s0a[7])
        : "r"(ak0[4]), "r"(ak0[(4) + 1]), "r"(ak0[(4) + 2]), "r"(ak0[(4) + 3]), "l"(_wgmma_b_1 + 2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s0b[0]), "+f"(d_s0b[1]), "+f"(d_s0b[2]), "+f"(d_s0b[3]), "+f"(d_s0b[4]),
          "+f"(d_s0b[5]), "+f"(d_s0b[6]), "+f"(d_s0b[7])
        : "r"(ak0[20]), "r"(ak0[(20) + 1]), "r"(ak0[(20) + 2]), "r"(ak0[(20) + 3]),
          "l"(_wgmma_b_2 + 2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s0a[0]), "+f"(d_s0a[1]), "+f"(d_s0a[2]), "+f"(d_s0a[3]), "+f"(d_s0a[4]),
          "+f"(d_s0a[5]), "+f"(d_s0a[6]), "+f"(d_s0a[7])
        : "r"(ak0[8]), "r"(ak0[(8) + 1]), "r"(ak0[(8) + 2]), "r"(ak0[(8) + 3]), "l"(_wgmma_b_1 + 4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s0b[0]), "+f"(d_s0b[1]), "+f"(d_s0b[2]), "+f"(d_s0b[3]), "+f"(d_s0b[4]),
          "+f"(d_s0b[5]), "+f"(d_s0b[6]), "+f"(d_s0b[7])
        : "r"(ak0[24]), "r"(ak0[(24) + 1]), "r"(ak0[(24) + 2]), "r"(ak0[(24) + 3]),
          "l"(_wgmma_b_2 + 4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s0a[0]), "+f"(d_s0a[1]), "+f"(d_s0a[2]), "+f"(d_s0a[3]), "+f"(d_s0a[4]),
          "+f"(d_s0a[5]), "+f"(d_s0a[6]), "+f"(d_s0a[7])
        : "r"(ak0[12]), "r"(ak0[(12) + 1]), "r"(ak0[(12) + 2]), "r"(ak0[(12) + 3]),
          "l"(_wgmma_b_1 + 6)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s0b[0]), "+f"(d_s0b[1]), "+f"(d_s0b[2]), "+f"(d_s0b[3]), "+f"(d_s0b[4]),
          "+f"(d_s0b[5]), "+f"(d_s0b[6]), "+f"(d_s0b[7])
        : "r"(ak0[28]), "r"(ak0[(28) + 1]), "r"(ak0[(28) + 2]), "r"(ak0[(28) + 3]),
          "l"(_wgmma_b_2 + 6)
        : "memory");
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    d_s0[0] = d_s0a[0] + d_s0b[0];
    d_s0[1] = d_s0a[1] + d_s0b[1];
    d_s0[2] = d_s0a[2] + d_s0b[2];
    d_s0[3] = d_s0a[3] + d_s0b[3];
    d_s0[4] = d_s0a[4] + d_s0b[4];
    d_s0[5] = d_s0a[5] + d_s0b[5];
    d_s0[6] = d_s0a[6] + d_s0b[6];
    d_s0[7] = d_s0a[7] + d_s0b[7];
    mbarrier_wait(kv_full_addr + (stage1) * 8, phase1);
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    uint64_t _wgmma_desc_40 =
        (((uint64_t)(((kv8_addr + (unsigned int)(stage1 * 16384))) >> 4) & 0x3FFFULL) |
         ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_a_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_40 >> 32)) << 32) |
                          (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_40);
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
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 0, 1, 1, 0;\n}\n"
        : "+f"(d_s1a[0]), "+f"(d_s1a[1]), "+f"(d_s1a[2]), "+f"(d_s1a[3]), "+f"(d_s1a[4]),
          "+f"(d_s1a[5]), "+f"(d_s1a[6]), "+f"(d_s1a[7])
        : "r"(ak1[0]), "r"(ak1[1]), "r"(ak1[2]), "r"(ak1[3]), "l"(_wgmma_b_1)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 0, 1, 1, 0;\n}\n"
        : "+f"(d_s1b[0]), "+f"(d_s1b[1]), "+f"(d_s1b[2]), "+f"(d_s1b[3]), "+f"(d_s1b[4]),
          "+f"(d_s1b[5]), "+f"(d_s1b[6]), "+f"(d_s1b[7])
        : "r"(ak1[16]), "r"(ak1[(16) + 1]), "r"(ak1[(16) + 2]), "r"(ak1[(16) + 3]), "l"(_wgmma_b_2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s1a[0]), "+f"(d_s1a[1]), "+f"(d_s1a[2]), "+f"(d_s1a[3]), "+f"(d_s1a[4]),
          "+f"(d_s1a[5]), "+f"(d_s1a[6]), "+f"(d_s1a[7])
        : "r"(ak1[4]), "r"(ak1[(4) + 1]), "r"(ak1[(4) + 2]), "r"(ak1[(4) + 3]), "l"(_wgmma_b_1 + 2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s1b[0]), "+f"(d_s1b[1]), "+f"(d_s1b[2]), "+f"(d_s1b[3]), "+f"(d_s1b[4]),
          "+f"(d_s1b[5]), "+f"(d_s1b[6]), "+f"(d_s1b[7])
        : "r"(ak1[20]), "r"(ak1[(20) + 1]), "r"(ak1[(20) + 2]), "r"(ak1[(20) + 3]),
          "l"(_wgmma_b_2 + 2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s1a[0]), "+f"(d_s1a[1]), "+f"(d_s1a[2]), "+f"(d_s1a[3]), "+f"(d_s1a[4]),
          "+f"(d_s1a[5]), "+f"(d_s1a[6]), "+f"(d_s1a[7])
        : "r"(ak1[8]), "r"(ak1[(8) + 1]), "r"(ak1[(8) + 2]), "r"(ak1[(8) + 3]), "l"(_wgmma_b_1 + 4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s1b[0]), "+f"(d_s1b[1]), "+f"(d_s1b[2]), "+f"(d_s1b[3]), "+f"(d_s1b[4]),
          "+f"(d_s1b[5]), "+f"(d_s1b[6]), "+f"(d_s1b[7])
        : "r"(ak1[24]), "r"(ak1[(24) + 1]), "r"(ak1[(24) + 2]), "r"(ak1[(24) + 3]),
          "l"(_wgmma_b_2 + 4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s1a[0]), "+f"(d_s1a[1]), "+f"(d_s1a[2]), "+f"(d_s1a[3]), "+f"(d_s1a[4]),
          "+f"(d_s1a[5]), "+f"(d_s1a[6]), "+f"(d_s1a[7])
        : "r"(ak1[12]), "r"(ak1[(12) + 1]), "r"(ak1[(12) + 2]), "r"(ak1[(12) + 3]),
          "l"(_wgmma_b_1 + 6)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n"
        : "+f"(d_s1b[0]), "+f"(d_s1b[1]), "+f"(d_s1b[2]), "+f"(d_s1b[3]), "+f"(d_s1b[4]),
          "+f"(d_s1b[5]), "+f"(d_s1b[6]), "+f"(d_s1b[7])
        : "r"(ak1[28]), "r"(ak1[(28) + 1]), "r"(ak1[(28) + 2]), "r"(ak1[(28) + 3]),
          "l"(_wgmma_b_2 + 6)
        : "memory");
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    d_s1[0] = d_s1a[0] + d_s1b[0];
    d_s1[1] = d_s1a[1] + d_s1b[1];
    d_s1[2] = d_s1a[2] + d_s1b[2];
    d_s1[3] = d_s1a[3] + d_s1b[3];
    d_s1[4] = d_s1a[4] + d_s1b[4];
    d_s1[5] = d_s1a[5] + d_s1b[5];
    d_s1[6] = d_s1a[6] + d_s1b[6];
    d_s1[7] = d_s1a[7] + d_s1b[7];
    tmax[0] = -CUDART_INF_F;
    tmax[1] = -CUDART_INF_F;
    tmax[2] = -CUDART_INF_F;
    tmax[3] = -CUDART_INF_F;
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
    sv0[4] = -CUDART_INF_F;
    if (key0 < valid_1) {
      sv0[4] = d_s0[4] * softmax_scale_log2;
    }
    sv1[4] = -CUDART_INF_F;
    if (valid_1 > 64 + key0) {
      sv1[4] = d_s1[4] * softmax_scale_log2;
    }
    tmax[2] = max_noftz(tmax[2], max_noftz(sv0[4], sv1[4]));
    sv0[5] = -CUDART_INF_F;
    if (key0 < valid_1) {
      sv0[5] = d_s0[5] * softmax_scale_log2;
    }
    sv1[5] = -CUDART_INF_F;
    if (valid_1 > 64 + key0) {
      sv1[5] = d_s1[5] * softmax_scale_log2;
    }
    tmax[3] = max_noftz(tmax[3], max_noftz(sv0[5], sv1[5]));
    sv0[6] = -CUDART_INF_F;
    if (key1 < valid_1) {
      sv0[6] = d_s0[6] * softmax_scale_log2;
    }
    sv1[6] = -CUDART_INF_F;
    if (valid_1 > 64 + key1) {
      sv1[6] = d_s1[6] * softmax_scale_log2;
    }
    tmax[2] = max_noftz(tmax[2], max_noftz(sv0[6], sv1[6]));
    sv0[7] = -CUDART_INF_F;
    if (key1 < valid_1) {
      sv0[7] = d_s0[7] * softmax_scale_log2;
    }
    sv1[7] = -CUDART_INF_F;
    if (valid_1 > 64 + key1) {
      sv1[7] = d_s1[7] * softmax_scale_log2;
    }
    tmax[3] = max_noftz(tmax[3], max_noftz(sv0[7], sv1[7]));
    float t = tmax[0];
    float _shfl_xor_sync_41 = __shfl_xor_sync(0xFFFFFFFF, t, 4);
    t = max_noftz(t, _shfl_xor_sync_41);
    float _shfl_xor_sync_42 = __shfl_xor_sync(0xFFFFFFFF, t, 8);
    t = max_noftz(t, _shfl_xor_sync_42);
    float _shfl_xor_sync_43 = __shfl_xor_sync(0xFFFFFFFF, t, 16);
    t = max_noftz(t, _shfl_xor_sync_43);
    tmax[0] = t;
    float t_0 = tmax[1];
    float _shfl_xor_sync_44 = __shfl_xor_sync(0xFFFFFFFF, t_0, 4);
    t_0 = max_noftz(t_0, _shfl_xor_sync_44);
    float _shfl_xor_sync_45 = __shfl_xor_sync(0xFFFFFFFF, t_0, 8);
    t_0 = max_noftz(t_0, _shfl_xor_sync_45);
    float _shfl_xor_sync_46 = __shfl_xor_sync(0xFFFFFFFF, t_0, 16);
    t_0 = max_noftz(t_0, _shfl_xor_sync_46);
    tmax[1] = t_0;
    float t_1 = tmax[2];
    float _shfl_xor_sync_47 = __shfl_xor_sync(0xFFFFFFFF, t_1, 4);
    t_1 = max_noftz(t_1, _shfl_xor_sync_47);
    float _shfl_xor_sync_48 = __shfl_xor_sync(0xFFFFFFFF, t_1, 8);
    t_1 = max_noftz(t_1, _shfl_xor_sync_48);
    float _shfl_xor_sync_49 = __shfl_xor_sync(0xFFFFFFFF, t_1, 16);
    t_1 = max_noftz(t_1, _shfl_xor_sync_49);
    tmax[2] = t_1;
    float t_2 = tmax[3];
    float _shfl_xor_sync_50 = __shfl_xor_sync(0xFFFFFFFF, t_2, 4);
    t_2 = max_noftz(t_2, _shfl_xor_sync_50);
    float _shfl_xor_sync_51 = __shfl_xor_sync(0xFFFFFFFF, t_2, 8);
    t_2 = max_noftz(t_2, _shfl_xor_sync_51);
    float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, t_2, 16);
    t_2 = max_noftz(t_2, _shfl_xor_sync_52);
    tmax[3] = t_2;
    if (lane_0 < 4) {
      int hh = 2 * cq;
      red[hh * 4 + warp_1] = tmax[0];
      int hh_0 = 2 * cq + 1;
      red[hh_0 * 4 + warp_1] = tmax[1];
      int hh_1 = 8 + 2 * cq;
      red[hh_1 * 4 + warp_1] = tmax[2];
      int hh_2 = 8 + 2 * cq + 1;
      red[hh_2 * 4 + warp_1] = tmax[3];
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
      float _exp2_53 = approx_exp2(mrun[0] - mnew);
      corr = _exp2_53;
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
    int hh2_3 = 2 * cq + 1;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])),
                   "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
                 : "r"(red_addr + (unsigned int)(hh2_3 * 4 * 4)));
    float page_max_4;
    asm("max.f32 %0, %1, %2;"
        : "=f"(page_max_4)
        : "f"(max_noftz(rv[0], rv[1])), "f"(max_noftz(rv[2], rv[3])));
    float mnew_5;
    asm("max.f32 %0, %1, %2;" : "=f"(mnew_5) : "f"(mrun[1]), "f"(page_max_4));
    float corr_6 = 0.0f;
    if (mrun[1] != -CUDART_INF_F) {
      float _exp2_54 = approx_exp2(mrun[1] - mnew_5);
      corr_6 = _exp2_54;
    }
    d_o0[1] = d_o0[1] * corr_6;
    d_o0[3] = d_o0[3] * corr_6;
    d_o1[1] = d_o1[1] * corr_6;
    d_o1[3] = d_o1[3] * corr_6;
    d_o0x[1] = d_o0x[1] * corr_6;
    d_o0x[3] = d_o0x[3] * corr_6;
    d_o1x[1] = d_o1x[1] * corr_6;
    d_o1x[3] = d_o1x[3] * corr_6;
    lp[1] = lp[1] * corr_6;
    mrun[1] = mnew_5;
    msub[1] = mnew_5;
    if (mnew_5 == -CUDART_INF_F) {
      msub[1] = 0.0f;
    }
    int hh2_7 = 8 + 2 * cq;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])),
                   "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
                 : "r"(red_addr + (unsigned int)(hh2_7 * 4 * 4)));
    float page_max_8;
    asm("max.f32 %0, %1, %2;"
        : "=f"(page_max_8)
        : "f"(max_noftz(rv[0], rv[1])), "f"(max_noftz(rv[2], rv[3])));
    float mnew_9;
    asm("max.f32 %0, %1, %2;" : "=f"(mnew_9) : "f"(mrun[2]), "f"(page_max_8));
    float corr_10 = 0.0f;
    if (mrun[2] != -CUDART_INF_F) {
      float _exp2_55 = approx_exp2(mrun[2] - mnew_9);
      corr_10 = _exp2_55;
    }
    d_o0[4] = d_o0[4] * corr_10;
    d_o0[6] = d_o0[6] * corr_10;
    d_o1[4] = d_o1[4] * corr_10;
    d_o1[6] = d_o1[6] * corr_10;
    d_o0x[4] = d_o0x[4] * corr_10;
    d_o0x[6] = d_o0x[6] * corr_10;
    d_o1x[4] = d_o1x[4] * corr_10;
    d_o1x[6] = d_o1x[6] * corr_10;
    lp[2] = lp[2] * corr_10;
    mrun[2] = mnew_9;
    msub[2] = mnew_9;
    if (mnew_9 == -CUDART_INF_F) {
      msub[2] = 0.0f;
    }
    int hh2_11 = 8 + 2 * cq + 1;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])),
                   "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
                 : "r"(red_addr + (unsigned int)(hh2_11 * 4 * 4)));
    float page_max_12;
    asm("max.f32 %0, %1, %2;"
        : "=f"(page_max_12)
        : "f"(max_noftz(rv[0], rv[1])), "f"(max_noftz(rv[2], rv[3])));
    float mnew_13;
    asm("max.f32 %0, %1, %2;" : "=f"(mnew_13) : "f"(mrun[3]), "f"(page_max_12));
    float corr_14 = 0.0f;
    if (mrun[3] != -CUDART_INF_F) {
      float _exp2_56 = approx_exp2(mrun[3] - mnew_13);
      corr_14 = _exp2_56;
    }
    d_o0[5] = d_o0[5] * corr_14;
    d_o0[7] = d_o0[7] * corr_14;
    d_o1[5] = d_o1[5] * corr_14;
    d_o1[7] = d_o1[7] * corr_14;
    d_o0x[5] = d_o0x[5] * corr_14;
    d_o0x[7] = d_o0x[7] * corr_14;
    d_o1x[5] = d_o1x[5] * corr_14;
    d_o1x[7] = d_o1x[7] * corr_14;
    lp[3] = lp[3] * corr_14;
    mrun[3] = mnew_13;
    msub[3] = mnew_13;
    if (mnew_13 == -CUDART_INF_F) {
      msub[3] = 0.0f;
    }
    int hh3 = 2 * cq;
    int kk = key0;
    int pcol = hh3 * 128 + (kk * 2 ^ (hh3 & 7) << 4);
    float p0 = approx_exp2(sv0[0] - msub[0]);
    float p1 = approx_exp2(sv1[0] - msub[0]);
    lp[0] = lp[0] + (p0 + p1);
    {
      __half _hval_57 = __float2half_rn(p0);
      uint16_t _bits_57 = *(uint16_t*)&_hval_57;
      uint32_t _addr_57 = static_cast<uint32_t>(p16_addr + (unsigned int)(par * 2 * 2048 + pcol));
      asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_57), "h"(_bits_57) : "memory");
    }
    {
      __half _hval_58 = __float2half_rn(p1);
      uint16_t _bits_58 = *(uint16_t*)&_hval_58;
      uint32_t _addr_58 =
          static_cast<uint32_t>(p16_addr + (unsigned int)((par * 2 + 1) * 2048 + pcol));
      asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_58), "h"(_bits_58) : "memory");
    }
    int hh3_15 = 2 * cq + 1;
    int kk_16 = key0;
    int pcol_17 = hh3_15 * 128 + (kk_16 * 2 ^ (hh3_15 & 7) << 4);
    float p0_18 = approx_exp2(sv0[1] - msub[1]);
    float p1_19 = approx_exp2(sv1[1] - msub[1]);
    lp[1] = lp[1] + (p0_18 + p1_19);
    {
      __half _hval_59 = __float2half_rn(p0_18);
      uint16_t _bits_59 = *(uint16_t*)&_hval_59;
      uint32_t _addr_59 =
          static_cast<uint32_t>(p16_addr + (unsigned int)(par * 2 * 2048 + pcol_17));
      asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_59), "h"(_bits_59) : "memory");
    }
    {
      __half _hval_60 = __float2half_rn(p1_19);
      uint16_t _bits_60 = *(uint16_t*)&_hval_60;
      uint32_t _addr_60 =
          static_cast<uint32_t>(p16_addr + (unsigned int)((par * 2 + 1) * 2048 + pcol_17));
      asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_60), "h"(_bits_60) : "memory");
    }
    int hh3_20 = 2 * cq;
    int kk_21 = key1;
    int pcol_22 = hh3_20 * 128 + (kk_21 * 2 ^ (hh3_20 & 7) << 4);
    float p0_23 = approx_exp2(sv0[2] - msub[0]);
    float p1_24 = approx_exp2(sv1[2] - msub[0]);
    lp[0] = lp[0] + (p0_23 + p1_24);
    {
      __half _hval_61 = __float2half_rn(p0_23);
      uint16_t _bits_61 = *(uint16_t*)&_hval_61;
      uint32_t _addr_61 =
          static_cast<uint32_t>(p16_addr + (unsigned int)(par * 2 * 2048 + pcol_22));
      asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_61), "h"(_bits_61) : "memory");
    }
    {
      __half _hval_62 = __float2half_rn(p1_24);
      uint16_t _bits_62 = *(uint16_t*)&_hval_62;
      uint32_t _addr_62 =
          static_cast<uint32_t>(p16_addr + (unsigned int)((par * 2 + 1) * 2048 + pcol_22));
      asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_62), "h"(_bits_62) : "memory");
    }
    int hh3_25 = 2 * cq + 1;
    int kk_26 = key1;
    int pcol_27 = hh3_25 * 128 + (kk_26 * 2 ^ (hh3_25 & 7) << 4);
    float p0_28 = approx_exp2(sv0[3] - msub[1]);
    float p1_29 = approx_exp2(sv1[3] - msub[1]);
    lp[1] = lp[1] + (p0_28 + p1_29);
    {
      __half _hval_63 = __float2half_rn(p0_28);
      uint16_t _bits_63 = *(uint16_t*)&_hval_63;
      uint32_t _addr_63 =
          static_cast<uint32_t>(p16_addr + (unsigned int)(par * 2 * 2048 + pcol_27));
      asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_63), "h"(_bits_63) : "memory");
    }
    {
      __half _hval_64 = __float2half_rn(p1_29);
      uint16_t _bits_64 = *(uint16_t*)&_hval_64;
      uint32_t _addr_64 =
          static_cast<uint32_t>(p16_addr + (unsigned int)((par * 2 + 1) * 2048 + pcol_27));
      asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_64), "h"(_bits_64) : "memory");
    }
    int hh3_30 = 8 + 2 * cq;
    int kk_31 = key0;
    int pcol_32 = hh3_30 * 128 + (kk_31 * 2 ^ (hh3_30 & 7) << 4);
    float p0_33 = approx_exp2(sv0[4] - msub[2]);
    float p1_34 = approx_exp2(sv1[4] - msub[2]);
    lp[2] = lp[2] + (p0_33 + p1_34);
    {
      __half _hval_65 = __float2half_rn(p0_33);
      uint16_t _bits_65 = *(uint16_t*)&_hval_65;
      uint32_t _addr_65 =
          static_cast<uint32_t>(p16_addr + (unsigned int)(par * 2 * 2048 + pcol_32));
      asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_65), "h"(_bits_65) : "memory");
    }
    {
      __half _hval_66 = __float2half_rn(p1_34);
      uint16_t _bits_66 = *(uint16_t*)&_hval_66;
      uint32_t _addr_66 =
          static_cast<uint32_t>(p16_addr + (unsigned int)((par * 2 + 1) * 2048 + pcol_32));
      asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_66), "h"(_bits_66) : "memory");
    }
    int hh3_35 = 8 + 2 * cq + 1;
    int kk_36 = key0;
    int pcol_37 = hh3_35 * 128 + (kk_36 * 2 ^ (hh3_35 & 7) << 4);
    float p0_38 = approx_exp2(sv0[5] - msub[3]);
    float p1_39 = approx_exp2(sv1[5] - msub[3]);
    lp[3] = lp[3] + (p0_38 + p1_39);
    {
      __half _hval_67 = __float2half_rn(p0_38);
      uint16_t _bits_67 = *(uint16_t*)&_hval_67;
      uint32_t _addr_67 =
          static_cast<uint32_t>(p16_addr + (unsigned int)(par * 2 * 2048 + pcol_37));
      asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_67), "h"(_bits_67) : "memory");
    }
    {
      __half _hval_68 = __float2half_rn(p1_39);
      uint16_t _bits_68 = *(uint16_t*)&_hval_68;
      uint32_t _addr_68 =
          static_cast<uint32_t>(p16_addr + (unsigned int)((par * 2 + 1) * 2048 + pcol_37));
      asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_68), "h"(_bits_68) : "memory");
    }
    int hh3_40 = 8 + 2 * cq;
    int kk_41 = key1;
    int pcol_42 = hh3_40 * 128 + (kk_41 * 2 ^ (hh3_40 & 7) << 4);
    float p0_43 = approx_exp2(sv0[6] - msub[2]);
    float p1_44 = approx_exp2(sv1[6] - msub[2]);
    lp[2] = lp[2] + (p0_43 + p1_44);
    {
      __half _hval_69 = __float2half_rn(p0_43);
      uint16_t _bits_69 = *(uint16_t*)&_hval_69;
      uint32_t _addr_69 =
          static_cast<uint32_t>(p16_addr + (unsigned int)(par * 2 * 2048 + pcol_42));
      asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_69), "h"(_bits_69) : "memory");
    }
    {
      __half _hval_70 = __float2half_rn(p1_44);
      uint16_t _bits_70 = *(uint16_t*)&_hval_70;
      uint32_t _addr_70 =
          static_cast<uint32_t>(p16_addr + (unsigned int)((par * 2 + 1) * 2048 + pcol_42));
      asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_70), "h"(_bits_70) : "memory");
    }
    int hh3_45 = 8 + 2 * cq + 1;
    int kk_46 = key1;
    int pcol_47 = hh3_45 * 128 + (kk_46 * 2 ^ (hh3_45 & 7) << 4);
    float p0_48 = approx_exp2(sv0[7] - msub[3]);
    float p1_49 = approx_exp2(sv1[7] - msub[3]);
    lp[3] = lp[3] + (p0_48 + p1_49);
    {
      __half _hval_71 = __float2half_rn(p0_48);
      uint16_t _bits_71 = *(uint16_t*)&_hval_71;
      uint32_t _addr_71 =
          static_cast<uint32_t>(p16_addr + (unsigned int)(par * 2 * 2048 + pcol_47));
      asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_71), "h"(_bits_71) : "memory");
    }
    {
      __half _hval_72 = __float2half_rn(p1_49);
      uint16_t _bits_72 = *(uint16_t*)&_hval_72;
      uint32_t _addr_72 =
          static_cast<uint32_t>(p16_addr + (unsigned int)((par * 2 + 1) * 2048 + pcol_47));
      asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_72), "h"(_bits_72) : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("barrier.sync 1, 128;" ::: "memory");
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    uint64_t _wgmma_desc_73 =
        (((uint64_t)(((kv8_addr + (unsigned int)(stage * 16384) + 8192)) >> 4) & 0x3FFFULL) |
         ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_3 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_73 >> 32)) << 32) |
                          (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_73);
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
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
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
    }
    av0[0] = dv0_0[0] + zero_u32;
    av0[1] = dv0_0[1] + zero_u32;
    av0[2] = dv0_0[2] + zero_u32;
    av0[3] = dv0_0[3] + zero_u32;
    av0[4] = dv0_0[4] + zero_u32;
    av0[5] = dv0_0[5] + zero_u32;
    av0[6] = dv0_0[6] + zero_u32;
    av0[7] = dv0_0[7] + zero_u32;
    av0[8] = dv0_0[8] + zero_u32;
    av0[9] = dv0_0[9] + zero_u32;
    av0[10] = dv0_0[10] + zero_u32;
    av0[11] = dv0_0[11] + zero_u32;
    av0[12] = dv0_0[12] + zero_u32;
    av0[13] = dv0_0[13] + zero_u32;
    av0[14] = dv0_0[14] + zero_u32;
    av0[15] = dv0_0[15] + zero_u32;
    av0[16] = dv0_1[0] + zero_u32;
    av0[17] = dv0_1[1] + zero_u32;
    av0[18] = dv0_1[2] + zero_u32;
    av0[19] = dv0_1[3] + zero_u32;
    av0[20] = dv0_1[4] + zero_u32;
    av0[21] = dv0_1[5] + zero_u32;
    av0[22] = dv0_1[6] + zero_u32;
    av0[23] = dv0_1[7] + zero_u32;
    av0[24] = dv0_1[8] + zero_u32;
    av0[25] = dv0_1[9] + zero_u32;
    av0[26] = dv0_1[10] + zero_u32;
    av0[27] = dv0_1[11] + zero_u32;
    av0[28] = dv0_1[12] + zero_u32;
    av0[29] = dv0_1[13] + zero_u32;
    av0[30] = dv0_1[14] + zero_u32;
    av0[31] = dv0_1[15] + zero_u32;
    int nxt_s = j + 1;
    if (nxt_s < 16) {
      int slot_0 = split * 16 + nxt_s;
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
          mbarrier_arrive_expect_tx(kv_full_addr + (stage) * 8, 16384);
          asm volatile(
              "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_"
              "hint"
              " [%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(kv8_addr +
                                                           (unsigned int)(stage * 16384)),
              "l"((&K)), "r"(0), "r"(0), "r"(page_head_2), "r"(kv_full_addr + (stage) * 8),
              "l"(0x12F0000000000000ULL)
              : "memory");
          asm volatile(
              "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_"
              "hint"
              " [%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(kv8_addr +
                                                           (unsigned int)(stage * 16384) + 8192),
              "l"((&V)), "r"(0), "r"(0), "r"(page_head_2), "r"(kv_full_addr + (stage) * 8),
              "l"(0x12F0000000000000ULL)
              : "memory");
        }
      }
    }
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    uint64_t _wgmma_desc_74 =
        (((uint64_t)(((p16_addr + (unsigned int)(par * 2 * 2048))) >> 4) & 0x3FFFULL) |
         ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_4 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_74 >> 32)) << 32) |
                          (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_74);
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o0[0]), "+f"(d_o0[1]), "+f"(d_o0[2]), "+f"(d_o0[3]), "+f"(d_o0[4]), "+f"(d_o0[5]),
          "+f"(d_o0[6]), "+f"(d_o0[7])
        : "r"(av0[0]), "r"(av0[1]), "r"(av0[2]), "r"(av0[3]), "l"(_wgmma_b_4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o1[0]), "+f"(d_o1[1]), "+f"(d_o1[2]), "+f"(d_o1[3]), "+f"(d_o1[4]), "+f"(d_o1[5]),
          "+f"(d_o1[6]), "+f"(d_o1[7])
        : "r"(av0[16]), "r"(av0[(16) + 1]), "r"(av0[(16) + 2]), "r"(av0[(16) + 3]), "l"(_wgmma_b_4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o0[0]), "+f"(d_o0[1]), "+f"(d_o0[2]), "+f"(d_o0[3]), "+f"(d_o0[4]), "+f"(d_o0[5]),
          "+f"(d_o0[6]), "+f"(d_o0[7])
        : "r"(av0[4]), "r"(av0[(4) + 1]), "r"(av0[(4) + 2]), "r"(av0[(4) + 3]), "l"(_wgmma_b_4 + 2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o1[0]), "+f"(d_o1[1]), "+f"(d_o1[2]), "+f"(d_o1[3]), "+f"(d_o1[4]), "+f"(d_o1[5]),
          "+f"(d_o1[6]), "+f"(d_o1[7])
        : "r"(av0[20]), "r"(av0[(20) + 1]), "r"(av0[(20) + 2]), "r"(av0[(20) + 3]),
          "l"(_wgmma_b_4 + 2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o0[0]), "+f"(d_o0[1]), "+f"(d_o0[2]), "+f"(d_o0[3]), "+f"(d_o0[4]), "+f"(d_o0[5]),
          "+f"(d_o0[6]), "+f"(d_o0[7])
        : "r"(av0[8]), "r"(av0[(8) + 1]), "r"(av0[(8) + 2]), "r"(av0[(8) + 3]), "l"(_wgmma_b_4 + 4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o1[0]), "+f"(d_o1[1]), "+f"(d_o1[2]), "+f"(d_o1[3]), "+f"(d_o1[4]), "+f"(d_o1[5]),
          "+f"(d_o1[6]), "+f"(d_o1[7])
        : "r"(av0[24]), "r"(av0[(24) + 1]), "r"(av0[(24) + 2]), "r"(av0[(24) + 3]),
          "l"(_wgmma_b_4 + 4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o0[0]), "+f"(d_o0[1]), "+f"(d_o0[2]), "+f"(d_o0[3]), "+f"(d_o0[4]), "+f"(d_o0[5]),
          "+f"(d_o0[6]), "+f"(d_o0[7])
        : "r"(av0[12]), "r"(av0[(12) + 1]), "r"(av0[(12) + 2]), "r"(av0[(12) + 3]),
          "l"(_wgmma_b_4 + 6)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o1[0]), "+f"(d_o1[1]), "+f"(d_o1[2]), "+f"(d_o1[3]), "+f"(d_o1[4]), "+f"(d_o1[5]),
          "+f"(d_o1[6]), "+f"(d_o1[7])
        : "r"(av0[28]), "r"(av0[(28) + 1]), "r"(av0[(28) + 2]), "r"(av0[(28) + 3]),
          "l"(_wgmma_b_4 + 6)
        : "memory");
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    uint64_t _wgmma_desc_75 =
        (((uint64_t)(((kv8_addr + (unsigned int)(stage1 * 16384) + 8192)) >> 4) & 0x3FFFULL) |
         ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_5 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_75 >> 32)) << 32) |
                          (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_75);
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 0, 1, 1;\n}\n"
        : "+r"(dv1_0[0]), "+r"(dv1_0[1]), "+r"(dv1_0[2]), "+r"(dv1_0[3]), "+r"(dv1_0[4]),
          "+r"(dv1_0[5]), "+r"(dv1_0[6]), "+r"(dv1_0[7]), "+r"(dv1_0[8]), "+r"(dv1_0[9]),
          "+r"(dv1_0[10]), "+r"(dv1_0[11]), "+r"(dv1_0[12]), "+r"(dv1_0[13]), "+r"(dv1_0[14]),
          "+r"(dv1_0[15])
        : "l"(_wgmma_b_0), "l"(_wgmma_b_5)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1;\n}\n"
        : "+r"(dv1_0[0]), "+r"(dv1_0[1]), "+r"(dv1_0[2]), "+r"(dv1_0[3]), "+r"(dv1_0[4]),
          "+r"(dv1_0[5]), "+r"(dv1_0[6]), "+r"(dv1_0[7]), "+r"(dv1_0[8]), "+r"(dv1_0[9]),
          "+r"(dv1_0[10]), "+r"(dv1_0[11]), "+r"(dv1_0[12]), "+r"(dv1_0[13]), "+r"(dv1_0[14]),
          "+r"(dv1_0[15])
        : "l"(_wgmma_b_0 + 2), "l"(_wgmma_b_5 + 2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 0, 1, 1;\n}\n"
        : "+r"(dv1_1[0]), "+r"(dv1_1[1]), "+r"(dv1_1[2]), "+r"(dv1_1[3]), "+r"(dv1_1[4]),
          "+r"(dv1_1[5]), "+r"(dv1_1[6]), "+r"(dv1_1[7]), "+r"(dv1_1[8]), "+r"(dv1_1[9]),
          "+r"(dv1_1[10]), "+r"(dv1_1[11]), "+r"(dv1_1[12]), "+r"(dv1_1[13]), "+r"(dv1_1[14]),
          "+r"(dv1_1[15])
        : "l"(_wgmma_b_0), "l"(_wgmma_b_5 + 4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1;\n}\n"
        : "+r"(dv1_1[0]), "+r"(dv1_1[1]), "+r"(dv1_1[2]), "+r"(dv1_1[3]), "+r"(dv1_1[4]),
          "+r"(dv1_1[5]), "+r"(dv1_1[6]), "+r"(dv1_1[7]), "+r"(dv1_1[8]), "+r"(dv1_1[9]),
          "+r"(dv1_1[10]), "+r"(dv1_1[11]), "+r"(dv1_1[12]), "+r"(dv1_1[13]), "+r"(dv1_1[14]),
          "+r"(dv1_1[15])
        : "l"(_wgmma_b_0 + 2), "l"(_wgmma_b_5 + 6)
        : "memory");
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    if (valid_1 < 128) {
      int thr_1 = valid_1 - 64 - 2 * cq;
      if (thr_1 <= 1) {
        dv1_0[0] = dv1_0[0] & 65535;
        dv1_1[0] = dv1_1[0] & 65535;
      }
      if (thr_1 <= 0) {
        dv1_0[0] = 0;
        dv1_1[0] = 0;
      }
      if (thr_1 <= 1) {
        dv1_0[1] = dv1_0[1] & 65535;
        dv1_1[1] = dv1_1[1] & 65535;
      }
      if (thr_1 <= 0) {
        dv1_0[1] = 0;
        dv1_1[1] = 0;
      }
      if (thr_1 <= 9) {
        dv1_0[2] = dv1_0[2] & 65535;
        dv1_1[2] = dv1_1[2] & 65535;
      }
      if (thr_1 <= 8) {
        dv1_0[2] = 0;
        dv1_1[2] = 0;
      }
      if (thr_1 <= 9) {
        dv1_0[3] = dv1_0[3] & 65535;
        dv1_1[3] = dv1_1[3] & 65535;
      }
      if (thr_1 <= 8) {
        dv1_0[3] = 0;
        dv1_1[3] = 0;
      }
      if (thr_1 <= 17) {
        dv1_0[4] = dv1_0[4] & 65535;
        dv1_1[4] = dv1_1[4] & 65535;
      }
      if (thr_1 <= 16) {
        dv1_0[4] = 0;
        dv1_1[4] = 0;
      }
      if (thr_1 <= 17) {
        dv1_0[5] = dv1_0[5] & 65535;
        dv1_1[5] = dv1_1[5] & 65535;
      }
      if (thr_1 <= 16) {
        dv1_0[5] = 0;
        dv1_1[5] = 0;
      }
      if (thr_1 <= 25) {
        dv1_0[6] = dv1_0[6] & 65535;
        dv1_1[6] = dv1_1[6] & 65535;
      }
      if (thr_1 <= 24) {
        dv1_0[6] = 0;
        dv1_1[6] = 0;
      }
      if (thr_1 <= 25) {
        dv1_0[7] = dv1_0[7] & 65535;
        dv1_1[7] = dv1_1[7] & 65535;
      }
      if (thr_1 <= 24) {
        dv1_0[7] = 0;
        dv1_1[7] = 0;
      }
      if (thr_1 <= 33) {
        dv1_0[8] = dv1_0[8] & 65535;
        dv1_1[8] = dv1_1[8] & 65535;
      }
      if (thr_1 <= 32) {
        dv1_0[8] = 0;
        dv1_1[8] = 0;
      }
      if (thr_1 <= 33) {
        dv1_0[9] = dv1_0[9] & 65535;
        dv1_1[9] = dv1_1[9] & 65535;
      }
      if (thr_1 <= 32) {
        dv1_0[9] = 0;
        dv1_1[9] = 0;
      }
      if (thr_1 <= 41) {
        dv1_0[10] = dv1_0[10] & 65535;
        dv1_1[10] = dv1_1[10] & 65535;
      }
      if (thr_1 <= 40) {
        dv1_0[10] = 0;
        dv1_1[10] = 0;
      }
      if (thr_1 <= 41) {
        dv1_0[11] = dv1_0[11] & 65535;
        dv1_1[11] = dv1_1[11] & 65535;
      }
      if (thr_1 <= 40) {
        dv1_0[11] = 0;
        dv1_1[11] = 0;
      }
      if (thr_1 <= 49) {
        dv1_0[12] = dv1_0[12] & 65535;
        dv1_1[12] = dv1_1[12] & 65535;
      }
      if (thr_1 <= 48) {
        dv1_0[12] = 0;
        dv1_1[12] = 0;
      }
      if (thr_1 <= 49) {
        dv1_0[13] = dv1_0[13] & 65535;
        dv1_1[13] = dv1_1[13] & 65535;
      }
      if (thr_1 <= 48) {
        dv1_0[13] = 0;
        dv1_1[13] = 0;
      }
      if (thr_1 <= 57) {
        dv1_0[14] = dv1_0[14] & 65535;
        dv1_1[14] = dv1_1[14] & 65535;
      }
      if (thr_1 <= 56) {
        dv1_0[14] = 0;
        dv1_1[14] = 0;
      }
      if (thr_1 <= 57) {
        dv1_0[15] = dv1_0[15] & 65535;
        dv1_1[15] = dv1_1[15] & 65535;
      }
      if (thr_1 <= 56) {
        dv1_0[15] = 0;
        dv1_1[15] = 0;
      }
    }
    av1[0] = dv1_0[0] + zero_u32;
    av1[1] = dv1_0[1] + zero_u32;
    av1[2] = dv1_0[2] + zero_u32;
    av1[3] = dv1_0[3] + zero_u32;
    av1[4] = dv1_0[4] + zero_u32;
    av1[5] = dv1_0[5] + zero_u32;
    av1[6] = dv1_0[6] + zero_u32;
    av1[7] = dv1_0[7] + zero_u32;
    av1[8] = dv1_0[8] + zero_u32;
    av1[9] = dv1_0[9] + zero_u32;
    av1[10] = dv1_0[10] + zero_u32;
    av1[11] = dv1_0[11] + zero_u32;
    av1[12] = dv1_0[12] + zero_u32;
    av1[13] = dv1_0[13] + zero_u32;
    av1[14] = dv1_0[14] + zero_u32;
    av1[15] = dv1_0[15] + zero_u32;
    av1[16] = dv1_1[0] + zero_u32;
    av1[17] = dv1_1[1] + zero_u32;
    av1[18] = dv1_1[2] + zero_u32;
    av1[19] = dv1_1[3] + zero_u32;
    av1[20] = dv1_1[4] + zero_u32;
    av1[21] = dv1_1[5] + zero_u32;
    av1[22] = dv1_1[6] + zero_u32;
    av1[23] = dv1_1[7] + zero_u32;
    av1[24] = dv1_1[8] + zero_u32;
    av1[25] = dv1_1[9] + zero_u32;
    av1[26] = dv1_1[10] + zero_u32;
    av1[27] = dv1_1[11] + zero_u32;
    av1[28] = dv1_1[12] + zero_u32;
    av1[29] = dv1_1[13] + zero_u32;
    av1[30] = dv1_1[14] + zero_u32;
    av1[31] = dv1_1[15] + zero_u32;
    if (nxt_s < 16) {
      int slot_0_1 = split * 16 + nxt_s;
      int blk_1_2 = meta[slot_0_1];
      int page_head_2_1 = meta[16 + slot_0_1];
      int valid_3_1 = 0;
      if (blk_1_2 >= 0) {
        valid_3_1 = q_pos + 1 - blk_1_2 * 128;
        if (valid_3_1 > 128) {
          valid_3_1 = 128;
        }
        if (valid_3_1 < 0) {
          valid_3_1 = 0;
        }
      }
      if (warp == 1) {
        if (elect_sync()) {
          mbarrier_arrive_expect_tx(kv_full_addr + (stage1) * 8, 16384);
          asm volatile(
              "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_"
              "hint"
              " [%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(kv8_addr +
                                                           (unsigned int)(stage1 * 16384)),
              "l"((&K)), "r"(0), "r"(64), "r"(page_head_2_1), "r"(kv_full_addr + (stage1) * 8),
              "l"(0x12F0000000000000ULL)
              : "memory");
          asm volatile(
              "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_"
              "hint"
              " [%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(kv8_addr +
                                                           (unsigned int)(stage1 * 16384) + 8192),
              "l"((&V)), "r"(0), "r"(64), "r"(page_head_2_1), "r"(kv_full_addr + (stage1) * 8),
              "l"(0x12F0000000000000ULL)
              : "memory");
        }
      }
    }
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    uint64_t _wgmma_desc_76 =
        (((uint64_t)(((p16_addr + (unsigned int)((par * 2 + 1) * 2048))) >> 4) & 0x3FFFULL) |
         ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_6 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_76 >> 32)) << 32) |
                          (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_76);
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o0x[0]), "+f"(d_o0x[1]), "+f"(d_o0x[2]), "+f"(d_o0x[3]), "+f"(d_o0x[4]),
          "+f"(d_o0x[5]), "+f"(d_o0x[6]), "+f"(d_o0x[7])
        : "r"(av1[0]), "r"(av1[1]), "r"(av1[2]), "r"(av1[3]), "l"(_wgmma_b_6)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o1x[0]), "+f"(d_o1x[1]), "+f"(d_o1x[2]), "+f"(d_o1x[3]), "+f"(d_o1x[4]),
          "+f"(d_o1x[5]), "+f"(d_o1x[6]), "+f"(d_o1x[7])
        : "r"(av1[16]), "r"(av1[(16) + 1]), "r"(av1[(16) + 2]), "r"(av1[(16) + 3]), "l"(_wgmma_b_6)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o0x[0]), "+f"(d_o0x[1]), "+f"(d_o0x[2]), "+f"(d_o0x[3]), "+f"(d_o0x[4]),
          "+f"(d_o0x[5]), "+f"(d_o0x[6]), "+f"(d_o0x[7])
        : "r"(av1[4]), "r"(av1[(4) + 1]), "r"(av1[(4) + 2]), "r"(av1[(4) + 3]), "l"(_wgmma_b_6 + 2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o1x[0]), "+f"(d_o1x[1]), "+f"(d_o1x[2]), "+f"(d_o1x[3]), "+f"(d_o1x[4]),
          "+f"(d_o1x[5]), "+f"(d_o1x[6]), "+f"(d_o1x[7])
        : "r"(av1[20]), "r"(av1[(20) + 1]), "r"(av1[(20) + 2]), "r"(av1[(20) + 3]),
          "l"(_wgmma_b_6 + 2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o0x[0]), "+f"(d_o0x[1]), "+f"(d_o0x[2]), "+f"(d_o0x[3]), "+f"(d_o0x[4]),
          "+f"(d_o0x[5]), "+f"(d_o0x[6]), "+f"(d_o0x[7])
        : "r"(av1[8]), "r"(av1[(8) + 1]), "r"(av1[(8) + 2]), "r"(av1[(8) + 3]), "l"(_wgmma_b_6 + 4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o1x[0]), "+f"(d_o1x[1]), "+f"(d_o1x[2]), "+f"(d_o1x[3]), "+f"(d_o1x[4]),
          "+f"(d_o1x[5]), "+f"(d_o1x[6]), "+f"(d_o1x[7])
        : "r"(av1[24]), "r"(av1[(24) + 1]), "r"(av1[(24) + 2]), "r"(av1[(24) + 3]),
          "l"(_wgmma_b_6 + 4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o0x[0]), "+f"(d_o0x[1]), "+f"(d_o0x[2]), "+f"(d_o0x[3]), "+f"(d_o0x[4]),
          "+f"(d_o0x[5]), "+f"(d_o0x[6]), "+f"(d_o0x[7])
        : "r"(av1[12]), "r"(av1[(12) + 1]), "r"(av1[(12) + 2]), "r"(av1[(12) + 3]),
          "l"(_wgmma_b_6 + 6)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n"
        : "+f"(d_o1x[0]), "+f"(d_o1x[1]), "+f"(d_o1x[2]), "+f"(d_o1x[3]), "+f"(d_o1x[4]),
          "+f"(d_o1x[5]), "+f"(d_o1x[6]), "+f"(d_o1x[7])
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
  d_o0[4] = d_o0[4] + d_o0x[4];
  d_o1[4] = d_o1[4] + d_o1x[4];
  d_o0[5] = d_o0[5] + d_o0x[5];
  d_o1[5] = d_o1[5] + d_o1x[5];
  d_o0[6] = d_o0[6] + d_o0x[6];
  d_o1[6] = d_o1[6] + d_o1x[6];
  d_o0[7] = d_o0[7] + d_o0x[7];
  d_o1[7] = d_o1[7] + d_o1x[7];
  float t2 = lp[0];
  float _shfl_xor_sync_77 = __shfl_xor_sync(0xFFFFFFFF, t2, 4);
  t2 = t2 + _shfl_xor_sync_77;
  float _shfl_xor_sync_78 = __shfl_xor_sync(0xFFFFFFFF, t2, 8);
  t2 = t2 + _shfl_xor_sync_78;
  float _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, t2, 16);
  t2 = t2 + _shfl_xor_sync_79;
  lp[0] = t2;
  float t2_2 = lp[1];
  float _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, t2_2, 4);
  t2_2 = t2_2 + _shfl_xor_sync_80;
  float _shfl_xor_sync_81 = __shfl_xor_sync(0xFFFFFFFF, t2_2, 8);
  t2_2 = t2_2 + _shfl_xor_sync_81;
  float _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, t2_2, 16);
  t2_2 = t2_2 + _shfl_xor_sync_82;
  lp[1] = t2_2;
  float t2_3 = lp[2];
  float _shfl_xor_sync_83 = __shfl_xor_sync(0xFFFFFFFF, t2_3, 4);
  t2_3 = t2_3 + _shfl_xor_sync_83;
  float _shfl_xor_sync_84 = __shfl_xor_sync(0xFFFFFFFF, t2_3, 8);
  t2_3 = t2_3 + _shfl_xor_sync_84;
  float _shfl_xor_sync_85 = __shfl_xor_sync(0xFFFFFFFF, t2_3, 16);
  t2_3 = t2_3 + _shfl_xor_sync_85;
  lp[2] = t2_3;
  float t2_4 = lp[3];
  float _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, t2_4, 4);
  t2_4 = t2_4 + _shfl_xor_sync_86;
  float _shfl_xor_sync_87 = __shfl_xor_sync(0xFFFFFFFF, t2_4, 8);
  t2_4 = t2_4 + _shfl_xor_sync_87;
  float _shfl_xor_sync_88 = __shfl_xor_sync(0xFFFFFFFF, t2_4, 16);
  t2_4 = t2_4 + _shfl_xor_sync_88;
  lp[3] = t2_4;
  if (lane_0 < 4) {
    int hh4 = 2 * cq;
    lred[hh4 * 4 + warp_1] = lp[0];
    int hh4_0 = 2 * cq + 1;
    lred[hh4_0 * 4 + warp_1] = lp[1];
    int hh4_1 = 8 + 2 * cq;
    lred[hh4_1 * 4 + warp_1] = lp[2];
    int hh4_2 = 8 + 2 * cq + 1;
    lred[hh4_2 * 4 + warp_1] = lp[3];
  }
  asm volatile("barrier.sync 1, 128;" ::: "memory");
  float lsum[4];
  int hh5 = 2 * cq;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
               : "r"(lred_addr + (unsigned int)(hh5 * 4 * 4)));
  lsum[0] = rv[0] + rv[1] + (rv[2] + rv[3]);
  int hh5_5 = 2 * cq + 1;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
               : "r"(lred_addr + (unsigned int)(hh5_5 * 4 * 4)));
  lsum[1] = rv[0] + rv[1] + (rv[2] + rv[3]);
  int hh5_6 = 8 + 2 * cq;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
               : "r"(lred_addr + (unsigned int)(hh5_6 * 4 * 4)));
  lsum[2] = rv[0] + rv[1] + (rv[2] + rv[3]);
  int hh5_7 = 8 + 2 * cq + 1;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])),
                 "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
               : "r"(lred_addr + (unsigned int)(hh5_7 * 4 * 4)));
  lsum[3] = rv[0] + rv[1] + (rv[2] + rv[3]);
  int merge = 1;
  if (merge != 0) {
    float inv = 0.0f;
    if (lsum[0] != 0.0f) {
      float _rcp_89 = approx_rcp(lsum[0]);
      inv = _rcp_89;
    }
    lsum[0] = inv;
    float inv_0 = 0.0f;
    if (lsum[1] != 0.0f) {
      float _rcp_90 = approx_rcp(lsum[1]);
      inv_0 = _rcp_90;
    }
    lsum[1] = inv_0;
    float inv_1 = 0.0f;
    if (lsum[2] != 0.0f) {
      float _rcp_91 = approx_rcp(lsum[2]);
      inv_1 = _rcp_91;
    }
    lsum[2] = inv_1;
    float inv_2 = 0.0f;
    if (lsum[3] != 0.0f) {
      float _rcp_92 = approx_rcp(lsum[3]);
      inv_2 = _rcp_92;
    }
    lsum[3] = inv_2;
    int hh6 = 2 * cq;
    int dd = warp_1 * 16 + g;
    {
      uint32_t _addr_93 = static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6 * 128 + dd) * 4));
      asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_93), "f"(d_o0[0] * lsum[0]) : "memory");
    }
    {
      uint32_t _addr_94 =
          static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6 * 128 + dd + 64) * 4));
      asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_94), "f"(d_o1[0] * lsum[0]) : "memory");
    }
    int hh6_3 = 2 * cq + 1;
    int dd_4 = warp_1 * 16 + g;
    {
      uint32_t _addr_95 =
          static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6_3 * 128 + dd_4) * 4));
      asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_95), "f"(d_o0[1] * lsum[1]) : "memory");
    }
    {
      uint32_t _addr_96 =
          static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6_3 * 128 + dd_4 + 64) * 4));
      asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_96), "f"(d_o1[1] * lsum[1]) : "memory");
    }
    int hh6_5 = 2 * cq;
    int dd_6 = warp_1 * 16 + g + 8;
    {
      uint32_t _addr_97 =
          static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6_5 * 128 + dd_6) * 4));
      asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_97), "f"(d_o0[2] * lsum[0]) : "memory");
    }
    {
      uint32_t _addr_98 =
          static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6_5 * 128 + dd_6 + 64) * 4));
      asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_98), "f"(d_o1[2] * lsum[0]) : "memory");
    }
    int hh6_7 = 2 * cq + 1;
    int dd_8 = warp_1 * 16 + g + 8;
    {
      uint32_t _addr_99 =
          static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6_7 * 128 + dd_8) * 4));
      asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_99), "f"(d_o0[3] * lsum[1]) : "memory");
    }
    {
      uint32_t _addr_100 =
          static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6_7 * 128 + dd_8 + 64) * 4));
      asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_100), "f"(d_o1[3] * lsum[1]) : "memory");
    }
    int hh6_9 = 8 + 2 * cq;
    int dd_10 = warp_1 * 16 + g;
    {
      uint32_t _addr_101 =
          static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6_9 * 128 + dd_10) * 4));
      asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_101), "f"(d_o0[4] * lsum[2]) : "memory");
    }
    {
      uint32_t _addr_102 =
          static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6_9 * 128 + dd_10 + 64) * 4));
      asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_102), "f"(d_o1[4] * lsum[2]) : "memory");
    }
    int hh6_11 = 8 + 2 * cq + 1;
    int dd_12 = warp_1 * 16 + g;
    {
      uint32_t _addr_103 =
          static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6_11 * 128 + dd_12) * 4));
      asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_103), "f"(d_o0[5] * lsum[3]) : "memory");
    }
    {
      uint32_t _addr_104 =
          static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6_11 * 128 + dd_12 + 64) * 4));
      asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_104), "f"(d_o1[5] * lsum[3]) : "memory");
    }
    int hh6_13 = 8 + 2 * cq;
    int dd_14 = warp_1 * 16 + g + 8;
    {
      uint32_t _addr_105 =
          static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6_13 * 128 + dd_14) * 4));
      asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_105), "f"(d_o0[6] * lsum[2]) : "memory");
    }
    {
      uint32_t _addr_106 =
          static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6_13 * 128 + dd_14 + 64) * 4));
      asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_106), "f"(d_o1[6] * lsum[2]) : "memory");
    }
    int hh6_15 = 8 + 2 * cq + 1;
    int dd_16 = warp_1 * 16 + g + 8;
    {
      uint32_t _addr_107 =
          static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6_15 * 128 + dd_16) * 4));
      asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_107), "f"(d_o0[7] * lsum[3]) : "memory");
    }
    {
      uint32_t _addr_108 =
          static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6_15 * 128 + dd_16 + 64) * 4));
      asm volatile("st.shared.f32 [%0], %1;" ::"r"(_addr_108), "f"(d_o1[7] * lsum[3]) : "memory");
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
    int vid3_17 = tid_1 + 128;
    int on_18 = vid3_17 / 16;
    int oseg_19 = vid3_17 - on_18 * 16;
    int od0_20 = oseg_19 * 8;
    if (on_18 < heads_valid) {
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&ov[0])),
                     "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 3]))
                   : "r"(ident8_addr + (unsigned int)((on_18 * 128 + od0_20) * 4)));
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&ov[4])),
                     "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 3]))
                   : "r"(ident8_addr + (unsigned int)((on_18 * 128 + od0_20 + 4) * 4)));
      {
        __nv_bfloat162 _pk[4];
        _pk[0] = __floats2bfloat162_rn(ov[0 + 0], ov[0 + 1]);
        _pk[1] = __floats2bfloat162_rn(ov[0 + 2], ov[0 + 3]);
        _pk[2] = __floats2bfloat162_rn(ov[0 + 4], ov[0 + 5]);
        _pk[3] = __floats2bfloat162_rn(ov[0 + 6], ov[0 + 7]);
        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + ((q_row0 + on_18) * 128 + od0_20)))[0]) =
            *reinterpret_cast<uint4*>(&_pk[0]);
      }
    }
  }

  // Cleanup
  __syncthreads();
}

}  // extern "C"
