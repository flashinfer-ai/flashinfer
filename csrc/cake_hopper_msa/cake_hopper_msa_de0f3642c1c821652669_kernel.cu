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
#define SMEM_Q_S_OFF 1024
#define SMEM_Q_S_STAGE_BYTES 32768
#define SMEM_Q_S_STRIDE 32768
#define SMEM_K_S_OFF 33792
#define SMEM_K_S_STAGE_BYTES 16384
#define SMEM_K_S_STRIDE 16384
#define SMEM_TOTAL 99328
#define THREADS 544
#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__
__launch_bounds__(544, LAUNCH_MIN_BLOCKS) void kernel_cake_hopper_msa_de0f3642c1c821652669(
    const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K,
    float* __restrict__ out, int* __restrict__ cu_seqlens_q, int* __restrict__ page_table,
    int* __restrict__ seqused_k, int* __restrict__ q_offset, int has_qoff, int pt_stride,
    int max_k_tiles, int total_q, int num_heads, int nsplit) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

  const int mbar_base = smem;
#define full_addr (mbar_base + 0)
#define empty_addr (mbar_base + 32)

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  const int cta_rank = 0;

  // Kernel setup ops
  uint8_t* q_s = reinterpret_cast<uint8_t*>(smem_raw + 1024);
  const int q_s_addr = smem + 1024;
  uint8_t* k_s = reinterpret_cast<uint8_t*>(smem_raw + 33792);
  const int k_s_addr = smem + 33792;

  // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 8 barriers)
  // Mbarriers at smem_raw[0..64)

  if (warp == 0) {
    uint32_t leader = elect_sync();
    if (leader) {
      // full: 4 barriers, init_count=1
      mbarrier_init(smem + 0, 1);
      mbarrier_init(smem + 8, 1);
      mbarrier_init(smem + 16, 1);
      mbarrier_init(smem + 24, 1);
      // empty: 4 barriers, init_count=16
      mbarrier_init(smem + 32, 16);
      mbarrier_init(smem + 40, 16);
      mbarrier_init(smem + 48, 16);
      mbarrier_init(smem + 56, 16);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
  }

  __syncthreads();

  // ---- Role: producer ----
  if (warp == 16) {
    {  // producer_main
      if (warp == 16) {
        asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)((&K))) : "memory");
        asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)((&Q))) : "memory");
        int bx = blockIdx.x;
        int split = blockIdx.y;
        int b = blockIdx.z;
        int mt = gridDim.x / num_heads - 1 - bx / num_heads;
        int pt_base = b * pt_stride + split;
        int pg_first = page_table[pt_base];
        int h = bx % num_heads;
        int qlo = cu_seqlens_q[b];
        int sqb = cu_seqlens_q[b + 1] - qlo;
        int sk = seqused_k[b];
        int pfx = sk - sqb;
        if (has_qoff != 0) {
          pfx = q_offset[b];
        }
        int m0 = mt * 256;
        int mlim = sqb - 256;
        if (m0 > mlim) {
          m0 = mlim;
        }
        if (m0 < 0) {
          m0 = 0;
        }
        int nb = (sk + 127) / 128;
        int qmax = m0 + 255;
        if (qmax > sqb - 1) {
          qmax = sqb - 1;
        }
        qmax = qmax + pfx;
        int t_lim = qmax / 128 + 1;
        if (t_lim > nb) {
          t_lim = nb;
        }
        if (t_lim > max_k_tiles) {
          t_lim = max_k_tiles;
        }
        int n_comp = (t_lim - split + nsplit - 1) / nsplit;
        if (t_lim <= split) {
          n_comp = 0;
        }
        if (elect_sync()) {
          if (n_comp > 0) {
            int pg_next = pg_first;
#pragma unroll 1
            for (int i = 0; i < n_comp; i++) {
              int stage = i % 4;
              int pphase = i / 4 + 1 & 1;
              int page = pg_next;
              if (n_comp > i + 1) {
                pg_next = page_table[pt_base + (i + 1) * nsplit];
              }
              mbarrier_wait(empty_addr + (stage) * 8, pphase);
              if (i == 0) {
                mbarrier_arrive_expect_tx(full_addr + (stage) * 8, 49152);
                tma_3d_gmem2smem(q_s_addr, (&Q), 0, h, qlo + m0, full_addr + (stage) * 8);
              } else {
                mbarrier_arrive_expect_tx(full_addr + (stage) * 8, 16384);
              }
              tma_2d_gmem2smem(k_s_addr + (unsigned int)(stage * 16384), (&K), 0, page * 128,
                               full_addr + (stage) * 8);
            }
          }
        }
      }
    }
  }
  // ---- Role: math ----
  if (warp <= 15) {
    {  // math_main
      int bx_1 = blockIdx.x;
      int split_1 = blockIdx.y;
      int b_1 = blockIdx.z;
      int mt_1 = gridDim.x / num_heads - 1 - bx_1 / num_heads;
      int h_1 = bx_1 % num_heads;
      int qlo_1 = cu_seqlens_q[b_1];
      int sqb_1 = cu_seqlens_q[b_1 + 1] - qlo_1;
      int sk_1 = seqused_k[b_1];
      int pfx_1 = sk_1 - sqb_1;
      if (has_qoff != 0) {
        pfx_1 = q_offset[b_1];
      }
      int m0_1 = mt_1 * 256;
      int mlim_1 = sqb_1 - 256;
      if (m0_1 > mlim_1) {
        m0_1 = mlim_1;
      }
      if (m0_1 < 0) {
        m0_1 = 0;
      }
      int nb_1 = (sk_1 + 127) / 128;
      int qmax_1 = m0_1 + 255;
      if (qmax_1 > sqb_1 - 1) {
        qmax_1 = sqb_1 - 1;
      }
      qmax_1 = qmax_1 + pfx_1;
      int t_lim_1 = qmax_1 / 128 + 1;
      if (t_lim_1 > nb_1) {
        t_lim_1 = nb_1;
      }
      if (t_lim_1 > max_k_tiles) {
        t_lim_1 = max_k_tiles;
      }
      int n_comp_1 = (t_lim_1 - split_1 + nsplit - 1) / nsplit;
      if (t_lim_1 <= split_1) {
        n_comp_1 = 0;
      }
      int n_dead = (max_k_tiles - t_lim_1 - split_1 + nsplit - 1) / nsplit;
      if (split_1 >= max_k_tiles - t_lim_1) {
        n_dead = 0;
      }
      int tid_1 = threadIdx.x;
      int lane_0 = lane;
      int wgi = warp / 4;
      int wrp = warp - wgi * 4;
      int g = lane_0 / 4;
      int cq = lane_0 - g * 4;
      int r0 = wgi * 64 + wrp * 16 + g;
      int rows_valid = sqb_1 - m0_1;
      int obase = h_1 * max_k_tiles * total_q + qlo_1 + m0_1;
      int sk1 = sk_1 - 1;
      int lim0 = pfx_1 + m0_1 + r0;
      if (lim0 > sk1) {
        lim0 = sk1;
      }
      int lim1 = pfx_1 + m0_1 + r0 + 8;
      if (lim1 > sk1) {
        lim1 = sk1;
      }
      int mask_thr = pfx_1 + m0_1;
      if (mask_thr > sk1) {
        mask_thr = sk1;
      }
      float neg = -CUDART_INF_F;
      unsigned int acc[32];
      unsigned int acc1[32];
      uint64_t _wgmma_desc_0 =
          (((uint64_t)(((q_s_addr + (unsigned int)(wgi * 64) * 128)) >> 4) & 0x3FFFULL) |
           ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
      uint64_t _wgmma_a_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_0 >> 32)) << 32) |
                            (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_0);
      uint64_t _wgmma_desc_1 = (((uint64_t)(((k_s_addr)) >> 4) & 0x3FFFULL) |
                                ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
      uint64_t _wgmma_b_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_1 >> 32)) << 32) |
                            (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_1);
      if (n_comp_1 > 0) {
        mbarrier_wait(full_addr, 0);
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile(
            "{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, "
            "%7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, "
            "%24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1;\n}\n"
            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]),
              "+r"(acc[6]), "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]),
              "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15]), "+r"(acc[16]),
              "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]), "+r"(acc[20]), "+r"(acc[21]),
              "+r"(acc[22]), "+r"(acc[23]), "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]),
              "+r"(acc[27]), "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
            : "l"(_wgmma_a_0), "l"(_wgmma_b_0)
            : "memory");
        asm volatile(
            "{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, "
            "%7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, "
            "%24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]),
              "+r"(acc[6]), "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]),
              "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15]), "+r"(acc[16]),
              "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]), "+r"(acc[20]), "+r"(acc[21]),
              "+r"(acc[22]), "+r"(acc[23]), "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]),
              "+r"(acc[27]), "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
            : "l"(_wgmma_a_0 + 2), "l"(_wgmma_b_0 + 2)
            : "memory");
        asm volatile(
            "{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, "
            "%7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, "
            "%24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]),
              "+r"(acc[6]), "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]),
              "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15]), "+r"(acc[16]),
              "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]), "+r"(acc[20]), "+r"(acc[21]),
              "+r"(acc[22]), "+r"(acc[23]), "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]),
              "+r"(acc[27]), "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
            : "l"(_wgmma_a_0 + 4), "l"(_wgmma_b_0 + 4)
            : "memory");
        asm volatile(
            "{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, "
            "%7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, "
            "%24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]),
              "+r"(acc[6]), "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]),
              "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15]), "+r"(acc[16]),
              "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]), "+r"(acc[20]), "+r"(acc[21]),
              "+r"(acc[22]), "+r"(acc[23]), "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]),
              "+r"(acc[27]), "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
            : "l"(_wgmma_a_0 + 6), "l"(_wgmma_b_0 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        int npair = (n_comp_1 - 1) / 2;
#pragma unroll 1
        for (int jp = 0; jp < npair; jp++) {
          int ia = 2 * jp;
          int ib = ia + 1;
          int ic = ia + 2;
          int sa = ia % 4;
          int sb = ib % 4;
          int sc = ic % 4;
          mbarrier_wait(full_addr + (sb) * 8, ib / 4 & 1);
          asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
          uint64_t _wgmma_desc_2 =
              (((uint64_t)(((k_s_addr + (unsigned int)(sb * 16384))) >> 4) & 0x3FFFULL) |
               ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
          uint64_t _wgmma_b_1 =
              ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_2 >> 32)) << 32) |
              (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_2);
          asm volatile(
              "{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, "
              "%6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
              "%23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1;\n}\n"
              : "+r"(acc1[0]), "+r"(acc1[1]), "+r"(acc1[2]), "+r"(acc1[3]), "+r"(acc1[4]),
                "+r"(acc1[5]), "+r"(acc1[6]), "+r"(acc1[7]), "+r"(acc1[8]), "+r"(acc1[9]),
                "+r"(acc1[10]), "+r"(acc1[11]), "+r"(acc1[12]), "+r"(acc1[13]), "+r"(acc1[14]),
                "+r"(acc1[15]), "+r"(acc1[16]), "+r"(acc1[17]), "+r"(acc1[18]), "+r"(acc1[19]),
                "+r"(acc1[20]), "+r"(acc1[21]), "+r"(acc1[22]), "+r"(acc1[23]), "+r"(acc1[24]),
                "+r"(acc1[25]), "+r"(acc1[26]), "+r"(acc1[27]), "+r"(acc1[28]), "+r"(acc1[29]),
                "+r"(acc1[30]), "+r"(acc1[31])
              : "l"(_wgmma_a_0), "l"(_wgmma_b_1)
              : "memory");
          asm volatile(
              "{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, "
              "%6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
              "%23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
              : "+r"(acc1[0]), "+r"(acc1[1]), "+r"(acc1[2]), "+r"(acc1[3]), "+r"(acc1[4]),
                "+r"(acc1[5]), "+r"(acc1[6]), "+r"(acc1[7]), "+r"(acc1[8]), "+r"(acc1[9]),
                "+r"(acc1[10]), "+r"(acc1[11]), "+r"(acc1[12]), "+r"(acc1[13]), "+r"(acc1[14]),
                "+r"(acc1[15]), "+r"(acc1[16]), "+r"(acc1[17]), "+r"(acc1[18]), "+r"(acc1[19]),
                "+r"(acc1[20]), "+r"(acc1[21]), "+r"(acc1[22]), "+r"(acc1[23]), "+r"(acc1[24]),
                "+r"(acc1[25]), "+r"(acc1[26]), "+r"(acc1[27]), "+r"(acc1[28]), "+r"(acc1[29]),
                "+r"(acc1[30]), "+r"(acc1[31])
              : "l"(_wgmma_a_0 + 2), "l"(_wgmma_b_1 + 2)
              : "memory");
          asm volatile(
              "{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, "
              "%6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
              "%23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
              : "+r"(acc1[0]), "+r"(acc1[1]), "+r"(acc1[2]), "+r"(acc1[3]), "+r"(acc1[4]),
                "+r"(acc1[5]), "+r"(acc1[6]), "+r"(acc1[7]), "+r"(acc1[8]), "+r"(acc1[9]),
                "+r"(acc1[10]), "+r"(acc1[11]), "+r"(acc1[12]), "+r"(acc1[13]), "+r"(acc1[14]),
                "+r"(acc1[15]), "+r"(acc1[16]), "+r"(acc1[17]), "+r"(acc1[18]), "+r"(acc1[19]),
                "+r"(acc1[20]), "+r"(acc1[21]), "+r"(acc1[22]), "+r"(acc1[23]), "+r"(acc1[24]),
                "+r"(acc1[25]), "+r"(acc1[26]), "+r"(acc1[27]), "+r"(acc1[28]), "+r"(acc1[29]),
                "+r"(acc1[30]), "+r"(acc1[31])
              : "l"(_wgmma_a_0 + 4), "l"(_wgmma_b_1 + 4)
              : "memory");
          asm volatile(
              "{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, "
              "%6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
              "%23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
              : "+r"(acc1[0]), "+r"(acc1[1]), "+r"(acc1[2]), "+r"(acc1[3]), "+r"(acc1[4]),
                "+r"(acc1[5]), "+r"(acc1[6]), "+r"(acc1[7]), "+r"(acc1[8]), "+r"(acc1[9]),
                "+r"(acc1[10]), "+r"(acc1[11]), "+r"(acc1[12]), "+r"(acc1[13]), "+r"(acc1[14]),
                "+r"(acc1[15]), "+r"(acc1[16]), "+r"(acc1[17]), "+r"(acc1[18]), "+r"(acc1[19]),
                "+r"(acc1[20]), "+r"(acc1[21]), "+r"(acc1[22]), "+r"(acc1[23]), "+r"(acc1[24]),
                "+r"(acc1[25]), "+r"(acc1[26]), "+r"(acc1[27]), "+r"(acc1[28]), "+r"(acc1[29]),
                "+r"(acc1[30]), "+r"(acc1[31])
              : "l"(_wgmma_a_0 + 6), "l"(_wgmma_b_1 + 6)
              : "memory");
          asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
          asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
          if (elect_sync()) {
            mbarrier_arrive(empty_addr + (sa) * 8);
          }
          int k0 = (split_1 + ia * nsplit) * 128;
          float m0v = neg;
          float m1v = neg;
          if (mask_thr < k0 + 127) {
            int c0 = lim0 - k0 - 2 * cq;
            int c1 = lim1 - k0 - 2 * cq;
            float v = neg;
            if (c0 >= 0) {
              float _cvt_f32_f16_3;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_3) : "h"((uint16_t)(acc[0] & 65535)));
              v = _cvt_f32_f16_3;
            }
            m0v = fmaxf(m0v, v);
            float v_0 = neg;
            if (c0 >= 1) {
              float _cvt_f32_f16_4;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_4) : "h"((uint16_t)(acc[0] >> 16)));
              v_0 = _cvt_f32_f16_4;
            }
            m0v = fmaxf(m0v, v_0);
            float v_1 = neg;
            if (c1 >= 0) {
              float _cvt_f32_f16_5;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_5) : "h"((uint16_t)(acc[1] & 65535)));
              v_1 = _cvt_f32_f16_5;
            }
            m1v = fmaxf(m1v, v_1);
            float v_2 = neg;
            if (c1 >= 1) {
              float _cvt_f32_f16_6;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_6) : "h"((uint16_t)(acc[1] >> 16)));
              v_2 = _cvt_f32_f16_6;
            }
            m1v = fmaxf(m1v, v_2);
            float v_3 = neg;
            if (c0 >= 8) {
              float _cvt_f32_f16_7;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_7) : "h"((uint16_t)(acc[2] & 65535)));
              v_3 = _cvt_f32_f16_7;
            }
            m0v = fmaxf(m0v, v_3);
            float v_4 = neg;
            if (c0 >= 9) {
              float _cvt_f32_f16_8;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_8) : "h"((uint16_t)(acc[2] >> 16)));
              v_4 = _cvt_f32_f16_8;
            }
            m0v = fmaxf(m0v, v_4);
            float v_5 = neg;
            if (c1 >= 8) {
              float _cvt_f32_f16_9;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_9) : "h"((uint16_t)(acc[3] & 65535)));
              v_5 = _cvt_f32_f16_9;
            }
            m1v = fmaxf(m1v, v_5);
            float v_6 = neg;
            if (c1 >= 9) {
              float _cvt_f32_f16_10;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_10) : "h"((uint16_t)(acc[3] >> 16)));
              v_6 = _cvt_f32_f16_10;
            }
            m1v = fmaxf(m1v, v_6);
            float v_7 = neg;
            if (c0 >= 16) {
              float _cvt_f32_f16_11;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_11) : "h"((uint16_t)(acc[4] & 65535)));
              v_7 = _cvt_f32_f16_11;
            }
            m0v = fmaxf(m0v, v_7);
            float v_8 = neg;
            if (c0 >= 17) {
              float _cvt_f32_f16_12;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_12) : "h"((uint16_t)(acc[4] >> 16)));
              v_8 = _cvt_f32_f16_12;
            }
            m0v = fmaxf(m0v, v_8);
            float v_9 = neg;
            if (c1 >= 16) {
              float _cvt_f32_f16_13;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_13) : "h"((uint16_t)(acc[5] & 65535)));
              v_9 = _cvt_f32_f16_13;
            }
            m1v = fmaxf(m1v, v_9);
            float v_10 = neg;
            if (c1 >= 17) {
              float _cvt_f32_f16_14;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_14) : "h"((uint16_t)(acc[5] >> 16)));
              v_10 = _cvt_f32_f16_14;
            }
            m1v = fmaxf(m1v, v_10);
            float v_11 = neg;
            if (c0 >= 24) {
              float _cvt_f32_f16_15;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_15) : "h"((uint16_t)(acc[6] & 65535)));
              v_11 = _cvt_f32_f16_15;
            }
            m0v = fmaxf(m0v, v_11);
            float v_12 = neg;
            if (c0 >= 25) {
              float _cvt_f32_f16_16;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_16) : "h"((uint16_t)(acc[6] >> 16)));
              v_12 = _cvt_f32_f16_16;
            }
            m0v = fmaxf(m0v, v_12);
            float v_13 = neg;
            if (c1 >= 24) {
              float _cvt_f32_f16_17;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_17) : "h"((uint16_t)(acc[7] & 65535)));
              v_13 = _cvt_f32_f16_17;
            }
            m1v = fmaxf(m1v, v_13);
            float v_14 = neg;
            if (c1 >= 25) {
              float _cvt_f32_f16_18;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_18) : "h"((uint16_t)(acc[7] >> 16)));
              v_14 = _cvt_f32_f16_18;
            }
            m1v = fmaxf(m1v, v_14);
            float v_15 = neg;
            if (c0 >= 32) {
              float _cvt_f32_f16_19;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_19) : "h"((uint16_t)(acc[8] & 65535)));
              v_15 = _cvt_f32_f16_19;
            }
            m0v = fmaxf(m0v, v_15);
            float v_16 = neg;
            if (c0 >= 33) {
              float _cvt_f32_f16_20;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_20) : "h"((uint16_t)(acc[8] >> 16)));
              v_16 = _cvt_f32_f16_20;
            }
            m0v = fmaxf(m0v, v_16);
            float v_17 = neg;
            if (c1 >= 32) {
              float _cvt_f32_f16_21;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_21) : "h"((uint16_t)(acc[9] & 65535)));
              v_17 = _cvt_f32_f16_21;
            }
            m1v = fmaxf(m1v, v_17);
            float v_18 = neg;
            if (c1 >= 33) {
              float _cvt_f32_f16_22;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_22) : "h"((uint16_t)(acc[9] >> 16)));
              v_18 = _cvt_f32_f16_22;
            }
            m1v = fmaxf(m1v, v_18);
            float v_19 = neg;
            if (c0 >= 40) {
              float _cvt_f32_f16_23;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_23) : "h"((uint16_t)(acc[10] & 65535)));
              v_19 = _cvt_f32_f16_23;
            }
            m0v = fmaxf(m0v, v_19);
            float v_20 = neg;
            if (c0 >= 41) {
              float _cvt_f32_f16_24;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_24) : "h"((uint16_t)(acc[10] >> 16)));
              v_20 = _cvt_f32_f16_24;
            }
            m0v = fmaxf(m0v, v_20);
            float v_21 = neg;
            if (c1 >= 40) {
              float _cvt_f32_f16_25;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_25) : "h"((uint16_t)(acc[11] & 65535)));
              v_21 = _cvt_f32_f16_25;
            }
            m1v = fmaxf(m1v, v_21);
            float v_22 = neg;
            if (c1 >= 41) {
              float _cvt_f32_f16_26;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_26) : "h"((uint16_t)(acc[11] >> 16)));
              v_22 = _cvt_f32_f16_26;
            }
            m1v = fmaxf(m1v, v_22);
            float v_23 = neg;
            if (c0 >= 48) {
              float _cvt_f32_f16_27;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_27) : "h"((uint16_t)(acc[12] & 65535)));
              v_23 = _cvt_f32_f16_27;
            }
            m0v = fmaxf(m0v, v_23);
            float v_24 = neg;
            if (c0 >= 49) {
              float _cvt_f32_f16_28;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_28) : "h"((uint16_t)(acc[12] >> 16)));
              v_24 = _cvt_f32_f16_28;
            }
            m0v = fmaxf(m0v, v_24);
            float v_25 = neg;
            if (c1 >= 48) {
              float _cvt_f32_f16_29;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_29) : "h"((uint16_t)(acc[13] & 65535)));
              v_25 = _cvt_f32_f16_29;
            }
            m1v = fmaxf(m1v, v_25);
            float v_26 = neg;
            if (c1 >= 49) {
              float _cvt_f32_f16_30;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_30) : "h"((uint16_t)(acc[13] >> 16)));
              v_26 = _cvt_f32_f16_30;
            }
            m1v = fmaxf(m1v, v_26);
            float v_27 = neg;
            if (c0 >= 56) {
              float _cvt_f32_f16_31;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_31) : "h"((uint16_t)(acc[14] & 65535)));
              v_27 = _cvt_f32_f16_31;
            }
            m0v = fmaxf(m0v, v_27);
            float v_28 = neg;
            if (c0 >= 57) {
              float _cvt_f32_f16_32;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_32) : "h"((uint16_t)(acc[14] >> 16)));
              v_28 = _cvt_f32_f16_32;
            }
            m0v = fmaxf(m0v, v_28);
            float v_29 = neg;
            if (c1 >= 56) {
              float _cvt_f32_f16_33;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_33) : "h"((uint16_t)(acc[15] & 65535)));
              v_29 = _cvt_f32_f16_33;
            }
            m1v = fmaxf(m1v, v_29);
            float v_30 = neg;
            if (c1 >= 57) {
              float _cvt_f32_f16_34;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_34) : "h"((uint16_t)(acc[15] >> 16)));
              v_30 = _cvt_f32_f16_34;
            }
            m1v = fmaxf(m1v, v_30);
            float v_31 = neg;
            if (c0 >= 64) {
              float _cvt_f32_f16_35;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_35) : "h"((uint16_t)(acc[16] & 65535)));
              v_31 = _cvt_f32_f16_35;
            }
            m0v = fmaxf(m0v, v_31);
            float v_32 = neg;
            if (c0 >= 65) {
              float _cvt_f32_f16_36;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_36) : "h"((uint16_t)(acc[16] >> 16)));
              v_32 = _cvt_f32_f16_36;
            }
            m0v = fmaxf(m0v, v_32);
            float v_33 = neg;
            if (c1 >= 64) {
              float _cvt_f32_f16_37;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_37) : "h"((uint16_t)(acc[17] & 65535)));
              v_33 = _cvt_f32_f16_37;
            }
            m1v = fmaxf(m1v, v_33);
            float v_34 = neg;
            if (c1 >= 65) {
              float _cvt_f32_f16_38;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_38) : "h"((uint16_t)(acc[17] >> 16)));
              v_34 = _cvt_f32_f16_38;
            }
            m1v = fmaxf(m1v, v_34);
            float v_35 = neg;
            if (c0 >= 72) {
              float _cvt_f32_f16_39;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_39) : "h"((uint16_t)(acc[18] & 65535)));
              v_35 = _cvt_f32_f16_39;
            }
            m0v = fmaxf(m0v, v_35);
            float v_36 = neg;
            if (c0 >= 73) {
              float _cvt_f32_f16_40;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_40) : "h"((uint16_t)(acc[18] >> 16)));
              v_36 = _cvt_f32_f16_40;
            }
            m0v = fmaxf(m0v, v_36);
            float v_37 = neg;
            if (c1 >= 72) {
              float _cvt_f32_f16_41;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_41) : "h"((uint16_t)(acc[19] & 65535)));
              v_37 = _cvt_f32_f16_41;
            }
            m1v = fmaxf(m1v, v_37);
            float v_38 = neg;
            if (c1 >= 73) {
              float _cvt_f32_f16_42;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_42) : "h"((uint16_t)(acc[19] >> 16)));
              v_38 = _cvt_f32_f16_42;
            }
            m1v = fmaxf(m1v, v_38);
            float v_39 = neg;
            if (c0 >= 80) {
              float _cvt_f32_f16_43;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_43) : "h"((uint16_t)(acc[20] & 65535)));
              v_39 = _cvt_f32_f16_43;
            }
            m0v = fmaxf(m0v, v_39);
            float v_40 = neg;
            if (c0 >= 81) {
              float _cvt_f32_f16_44;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_44) : "h"((uint16_t)(acc[20] >> 16)));
              v_40 = _cvt_f32_f16_44;
            }
            m0v = fmaxf(m0v, v_40);
            float v_41 = neg;
            if (c1 >= 80) {
              float _cvt_f32_f16_45;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_45) : "h"((uint16_t)(acc[21] & 65535)));
              v_41 = _cvt_f32_f16_45;
            }
            m1v = fmaxf(m1v, v_41);
            float v_42 = neg;
            if (c1 >= 81) {
              float _cvt_f32_f16_46;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_46) : "h"((uint16_t)(acc[21] >> 16)));
              v_42 = _cvt_f32_f16_46;
            }
            m1v = fmaxf(m1v, v_42);
            float v_43 = neg;
            if (c0 >= 88) {
              float _cvt_f32_f16_47;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_47) : "h"((uint16_t)(acc[22] & 65535)));
              v_43 = _cvt_f32_f16_47;
            }
            m0v = fmaxf(m0v, v_43);
            float v_44 = neg;
            if (c0 >= 89) {
              float _cvt_f32_f16_48;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_48) : "h"((uint16_t)(acc[22] >> 16)));
              v_44 = _cvt_f32_f16_48;
            }
            m0v = fmaxf(m0v, v_44);
            float v_45 = neg;
            if (c1 >= 88) {
              float _cvt_f32_f16_49;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_49) : "h"((uint16_t)(acc[23] & 65535)));
              v_45 = _cvt_f32_f16_49;
            }
            m1v = fmaxf(m1v, v_45);
            float v_46 = neg;
            if (c1 >= 89) {
              float _cvt_f32_f16_50;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_50) : "h"((uint16_t)(acc[23] >> 16)));
              v_46 = _cvt_f32_f16_50;
            }
            m1v = fmaxf(m1v, v_46);
            float v_47 = neg;
            if (c0 >= 96) {
              float _cvt_f32_f16_51;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_51) : "h"((uint16_t)(acc[24] & 65535)));
              v_47 = _cvt_f32_f16_51;
            }
            m0v = fmaxf(m0v, v_47);
            float v_48 = neg;
            if (c0 >= 97) {
              float _cvt_f32_f16_52;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_52) : "h"((uint16_t)(acc[24] >> 16)));
              v_48 = _cvt_f32_f16_52;
            }
            m0v = fmaxf(m0v, v_48);
            float v_49 = neg;
            if (c1 >= 96) {
              float _cvt_f32_f16_53;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_53) : "h"((uint16_t)(acc[25] & 65535)));
              v_49 = _cvt_f32_f16_53;
            }
            m1v = fmaxf(m1v, v_49);
            float v_50 = neg;
            if (c1 >= 97) {
              float _cvt_f32_f16_54;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_54) : "h"((uint16_t)(acc[25] >> 16)));
              v_50 = _cvt_f32_f16_54;
            }
            m1v = fmaxf(m1v, v_50);
            float v_51 = neg;
            if (c0 >= 104) {
              float _cvt_f32_f16_55;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_55) : "h"((uint16_t)(acc[26] & 65535)));
              v_51 = _cvt_f32_f16_55;
            }
            m0v = fmaxf(m0v, v_51);
            float v_52 = neg;
            if (c0 >= 105) {
              float _cvt_f32_f16_56;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_56) : "h"((uint16_t)(acc[26] >> 16)));
              v_52 = _cvt_f32_f16_56;
            }
            m0v = fmaxf(m0v, v_52);
            float v_53 = neg;
            if (c1 >= 104) {
              float _cvt_f32_f16_57;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_57) : "h"((uint16_t)(acc[27] & 65535)));
              v_53 = _cvt_f32_f16_57;
            }
            m1v = fmaxf(m1v, v_53);
            float v_54 = neg;
            if (c1 >= 105) {
              float _cvt_f32_f16_58;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_58) : "h"((uint16_t)(acc[27] >> 16)));
              v_54 = _cvt_f32_f16_58;
            }
            m1v = fmaxf(m1v, v_54);
            float v_55 = neg;
            if (c0 >= 112) {
              float _cvt_f32_f16_59;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_59) : "h"((uint16_t)(acc[28] & 65535)));
              v_55 = _cvt_f32_f16_59;
            }
            m0v = fmaxf(m0v, v_55);
            float v_56 = neg;
            if (c0 >= 113) {
              float _cvt_f32_f16_60;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_60) : "h"((uint16_t)(acc[28] >> 16)));
              v_56 = _cvt_f32_f16_60;
            }
            m0v = fmaxf(m0v, v_56);
            float v_57 = neg;
            if (c1 >= 112) {
              float _cvt_f32_f16_61;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_61) : "h"((uint16_t)(acc[29] & 65535)));
              v_57 = _cvt_f32_f16_61;
            }
            m1v = fmaxf(m1v, v_57);
            float v_58 = neg;
            if (c1 >= 113) {
              float _cvt_f32_f16_62;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_62) : "h"((uint16_t)(acc[29] >> 16)));
              v_58 = _cvt_f32_f16_62;
            }
            m1v = fmaxf(m1v, v_58);
            float v_59 = neg;
            if (c0 >= 120) {
              float _cvt_f32_f16_63;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_63) : "h"((uint16_t)(acc[30] & 65535)));
              v_59 = _cvt_f32_f16_63;
            }
            m0v = fmaxf(m0v, v_59);
            float v_60 = neg;
            if (c0 >= 121) {
              float _cvt_f32_f16_64;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_64) : "h"((uint16_t)(acc[30] >> 16)));
              v_60 = _cvt_f32_f16_64;
            }
            m0v = fmaxf(m0v, v_60);
            float v_61 = neg;
            if (c1 >= 120) {
              float _cvt_f32_f16_65;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_65) : "h"((uint16_t)(acc[31] & 65535)));
              v_61 = _cvt_f32_f16_65;
            }
            m1v = fmaxf(m1v, v_61);
            float v_62 = neg;
            if (c1 >= 121) {
              float _cvt_f32_f16_66;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_66) : "h"((uint16_t)(acc[31] >> 16)));
              v_62 = _cvt_f32_f16_66;
            }
            m1v = fmaxf(m1v, v_62);
          } else {
            unsigned int w0 = acc[0];
            unsigned int w1 = acc[1];
            uint32_t _f16x2_max_0;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_0) : "r"(w0), "r"(acc[2]));
            w0 = _f16x2_max_0;
            uint32_t _f16x2_max_1;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_1) : "r"(w1), "r"(acc[3]));
            w1 = _f16x2_max_1;
            uint32_t _f16x2_max_2;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_2) : "r"(w0), "r"(acc[4]));
            w0 = _f16x2_max_2;
            uint32_t _f16x2_max_3;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_3) : "r"(w1), "r"(acc[5]));
            w1 = _f16x2_max_3;
            uint32_t _f16x2_max_4;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_4) : "r"(w0), "r"(acc[6]));
            w0 = _f16x2_max_4;
            uint32_t _f16x2_max_5;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_5) : "r"(w1), "r"(acc[7]));
            w1 = _f16x2_max_5;
            uint32_t _f16x2_max_6;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_6) : "r"(w0), "r"(acc[8]));
            w0 = _f16x2_max_6;
            uint32_t _f16x2_max_7;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_7) : "r"(w1), "r"(acc[9]));
            w1 = _f16x2_max_7;
            uint32_t _f16x2_max_8;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_8) : "r"(w0), "r"(acc[10]));
            w0 = _f16x2_max_8;
            uint32_t _f16x2_max_9;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_9) : "r"(w1), "r"(acc[11]));
            w1 = _f16x2_max_9;
            uint32_t _f16x2_max_10;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_10) : "r"(w0), "r"(acc[12]));
            w0 = _f16x2_max_10;
            uint32_t _f16x2_max_11;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_11) : "r"(w1), "r"(acc[13]));
            w1 = _f16x2_max_11;
            uint32_t _f16x2_max_12;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_12) : "r"(w0), "r"(acc[14]));
            w0 = _f16x2_max_12;
            uint32_t _f16x2_max_13;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_13) : "r"(w1), "r"(acc[15]));
            w1 = _f16x2_max_13;
            uint32_t _f16x2_max_14;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_14) : "r"(w0), "r"(acc[16]));
            w0 = _f16x2_max_14;
            uint32_t _f16x2_max_15;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_15) : "r"(w1), "r"(acc[17]));
            w1 = _f16x2_max_15;
            uint32_t _f16x2_max_16;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_16) : "r"(w0), "r"(acc[18]));
            w0 = _f16x2_max_16;
            uint32_t _f16x2_max_17;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_17) : "r"(w1), "r"(acc[19]));
            w1 = _f16x2_max_17;
            uint32_t _f16x2_max_18;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_18) : "r"(w0), "r"(acc[20]));
            w0 = _f16x2_max_18;
            uint32_t _f16x2_max_19;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_19) : "r"(w1), "r"(acc[21]));
            w1 = _f16x2_max_19;
            uint32_t _f16x2_max_20;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_20) : "r"(w0), "r"(acc[22]));
            w0 = _f16x2_max_20;
            uint32_t _f16x2_max_21;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_21) : "r"(w1), "r"(acc[23]));
            w1 = _f16x2_max_21;
            uint32_t _f16x2_max_22;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_22) : "r"(w0), "r"(acc[24]));
            w0 = _f16x2_max_22;
            uint32_t _f16x2_max_23;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_23) : "r"(w1), "r"(acc[25]));
            w1 = _f16x2_max_23;
            uint32_t _f16x2_max_24;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_24) : "r"(w0), "r"(acc[26]));
            w0 = _f16x2_max_24;
            uint32_t _f16x2_max_25;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_25) : "r"(w1), "r"(acc[27]));
            w1 = _f16x2_max_25;
            uint32_t _f16x2_max_26;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_26) : "r"(w0), "r"(acc[28]));
            w0 = _f16x2_max_26;
            uint32_t _f16x2_max_27;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_27) : "r"(w1), "r"(acc[29]));
            w1 = _f16x2_max_27;
            uint32_t _f16x2_max_28;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_28) : "r"(w0), "r"(acc[30]));
            w0 = _f16x2_max_28;
            uint32_t _f16x2_max_29;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_29) : "r"(w1), "r"(acc[31]));
            w1 = _f16x2_max_29;
            uint16_t _f16_max_0;
            asm("max.f16 %0, %1, %2;"
                : "=h"(_f16_max_0)
                : "h"((uint16_t)(w0 & 65535)), "h"((uint16_t)(w0 >> 16)));
            float _cvt_f32_f16_67;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_67) : "h"((uint16_t)(_f16_max_0)));
            m0v = _cvt_f32_f16_67;
            uint16_t _f16_max_1;
            asm("max.f16 %0, %1, %2;"
                : "=h"(_f16_max_1)
                : "h"((uint16_t)(w1 & 65535)), "h"((uint16_t)(w1 >> 16)));
            float _cvt_f32_f16_68;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_68) : "h"((uint16_t)(_f16_max_1)));
            m1v = _cvt_f32_f16_68;
          }
          float _shfl_xor_sync_69 = __shfl_xor_sync(0xFFFFFFFF, m0v, 1);
          m0v = fmaxf(m0v, _shfl_xor_sync_69);
          float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, m0v, 2);
          m0v = fmaxf(m0v, _shfl_xor_sync_70);
          float _shfl_xor_sync_71 = __shfl_xor_sync(0xFFFFFFFF, m1v, 1);
          m1v = fmaxf(m1v, _shfl_xor_sync_71);
          float _shfl_xor_sync_72 = __shfl_xor_sync(0xFFFFFFFF, m1v, 2);
          m1v = fmaxf(m1v, _shfl_xor_sync_72);
          if (cq == 0) {
            int orow = obase + (split_1 + ia * nsplit) * total_q + r0;
            if (r0 < rows_valid) {
              out[orow] = m0v;
            }
            if (rows_valid > r0 + 8) {
              out[orow + 8] = m1v;
            }
          }
          mbarrier_wait(full_addr + (sc) * 8, ic / 4 & 1);
          asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
          uint64_t _wgmma_desc_73 =
              (((uint64_t)(((k_s_addr + (unsigned int)(sc * 16384))) >> 4) & 0x3FFFULL) |
               ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
          uint64_t _wgmma_b_2 =
              ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_73 >> 32)) << 32) |
              (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_73);
          asm volatile(
              "{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, "
              "%6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
              "%23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1;\n}\n"
              : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]),
                "+r"(acc[6]), "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]),
                "+r"(acc[11]), "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15]),
                "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]), "+r"(acc[20]),
                "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23]), "+r"(acc[24]), "+r"(acc[25]),
                "+r"(acc[26]), "+r"(acc[27]), "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]),
                "+r"(acc[31])
              : "l"(_wgmma_a_0), "l"(_wgmma_b_2)
              : "memory");
          asm volatile(
              "{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, "
              "%6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
              "%23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
              : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]),
                "+r"(acc[6]), "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]),
                "+r"(acc[11]), "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15]),
                "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]), "+r"(acc[20]),
                "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23]), "+r"(acc[24]), "+r"(acc[25]),
                "+r"(acc[26]), "+r"(acc[27]), "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]),
                "+r"(acc[31])
              : "l"(_wgmma_a_0 + 2), "l"(_wgmma_b_2 + 2)
              : "memory");
          asm volatile(
              "{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, "
              "%6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
              "%23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
              : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]),
                "+r"(acc[6]), "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]),
                "+r"(acc[11]), "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15]),
                "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]), "+r"(acc[20]),
                "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23]), "+r"(acc[24]), "+r"(acc[25]),
                "+r"(acc[26]), "+r"(acc[27]), "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]),
                "+r"(acc[31])
              : "l"(_wgmma_a_0 + 4), "l"(_wgmma_b_2 + 4)
              : "memory");
          asm volatile(
              "{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, "
              "%6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
              "%23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
              : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]),
                "+r"(acc[6]), "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]),
                "+r"(acc[11]), "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15]),
                "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]), "+r"(acc[20]),
                "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23]), "+r"(acc[24]), "+r"(acc[25]),
                "+r"(acc[26]), "+r"(acc[27]), "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]),
                "+r"(acc[31])
              : "l"(_wgmma_a_0 + 6), "l"(_wgmma_b_2 + 6)
              : "memory");
          asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
          asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
          if (elect_sync()) {
            mbarrier_arrive(empty_addr + (sb) * 8);
          }
          int k0_0 = (split_1 + ib * nsplit) * 128;
          float m0v_1 = neg;
          float m1v_2 = neg;
          if (mask_thr < k0_0 + 127) {
            int c0_1 = lim0 - k0_0 - 2 * cq;
            int c1_1 = lim1 - k0_0 - 2 * cq;
            float v_63 = neg;
            if (c0_1 >= 0) {
              float _cvt_f32_f16_74;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_74) : "h"((uint16_t)(acc1[0] & 65535)));
              v_63 = _cvt_f32_f16_74;
            }
            m0v_1 = fmaxf(m0v_1, v_63);
            float v_0_1 = neg;
            if (c0_1 >= 1) {
              float _cvt_f32_f16_75;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_75) : "h"((uint16_t)(acc1[0] >> 16)));
              v_0_1 = _cvt_f32_f16_75;
            }
            m0v_1 = fmaxf(m0v_1, v_0_1);
            float v_1_1 = neg;
            if (c1_1 >= 0) {
              float _cvt_f32_f16_76;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_76) : "h"((uint16_t)(acc1[1] & 65535)));
              v_1_1 = _cvt_f32_f16_76;
            }
            m1v_2 = fmaxf(m1v_2, v_1_1);
            float v_2_1 = neg;
            if (c1_1 >= 1) {
              float _cvt_f32_f16_77;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_77) : "h"((uint16_t)(acc1[1] >> 16)));
              v_2_1 = _cvt_f32_f16_77;
            }
            m1v_2 = fmaxf(m1v_2, v_2_1);
            float v_3_1 = neg;
            if (c0_1 >= 8) {
              float _cvt_f32_f16_78;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_78) : "h"((uint16_t)(acc1[2] & 65535)));
              v_3_1 = _cvt_f32_f16_78;
            }
            m0v_1 = fmaxf(m0v_1, v_3_1);
            float v_4_1 = neg;
            if (c0_1 >= 9) {
              float _cvt_f32_f16_79;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_79) : "h"((uint16_t)(acc1[2] >> 16)));
              v_4_1 = _cvt_f32_f16_79;
            }
            m0v_1 = fmaxf(m0v_1, v_4_1);
            float v_5_1 = neg;
            if (c1_1 >= 8) {
              float _cvt_f32_f16_80;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_80) : "h"((uint16_t)(acc1[3] & 65535)));
              v_5_1 = _cvt_f32_f16_80;
            }
            m1v_2 = fmaxf(m1v_2, v_5_1);
            float v_6_1 = neg;
            if (c1_1 >= 9) {
              float _cvt_f32_f16_81;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_81) : "h"((uint16_t)(acc1[3] >> 16)));
              v_6_1 = _cvt_f32_f16_81;
            }
            m1v_2 = fmaxf(m1v_2, v_6_1);
            float v_7_1 = neg;
            if (c0_1 >= 16) {
              float _cvt_f32_f16_82;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_82) : "h"((uint16_t)(acc1[4] & 65535)));
              v_7_1 = _cvt_f32_f16_82;
            }
            m0v_1 = fmaxf(m0v_1, v_7_1);
            float v_8_1 = neg;
            if (c0_1 >= 17) {
              float _cvt_f32_f16_83;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_83) : "h"((uint16_t)(acc1[4] >> 16)));
              v_8_1 = _cvt_f32_f16_83;
            }
            m0v_1 = fmaxf(m0v_1, v_8_1);
            float v_9_1 = neg;
            if (c1_1 >= 16) {
              float _cvt_f32_f16_84;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_84) : "h"((uint16_t)(acc1[5] & 65535)));
              v_9_1 = _cvt_f32_f16_84;
            }
            m1v_2 = fmaxf(m1v_2, v_9_1);
            float v_10_1 = neg;
            if (c1_1 >= 17) {
              float _cvt_f32_f16_85;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_85) : "h"((uint16_t)(acc1[5] >> 16)));
              v_10_1 = _cvt_f32_f16_85;
            }
            m1v_2 = fmaxf(m1v_2, v_10_1);
            float v_11_1 = neg;
            if (c0_1 >= 24) {
              float _cvt_f32_f16_86;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_86) : "h"((uint16_t)(acc1[6] & 65535)));
              v_11_1 = _cvt_f32_f16_86;
            }
            m0v_1 = fmaxf(m0v_1, v_11_1);
            float v_12_1 = neg;
            if (c0_1 >= 25) {
              float _cvt_f32_f16_87;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_87) : "h"((uint16_t)(acc1[6] >> 16)));
              v_12_1 = _cvt_f32_f16_87;
            }
            m0v_1 = fmaxf(m0v_1, v_12_1);
            float v_13_1 = neg;
            if (c1_1 >= 24) {
              float _cvt_f32_f16_88;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_88) : "h"((uint16_t)(acc1[7] & 65535)));
              v_13_1 = _cvt_f32_f16_88;
            }
            m1v_2 = fmaxf(m1v_2, v_13_1);
            float v_14_1 = neg;
            if (c1_1 >= 25) {
              float _cvt_f32_f16_89;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_89) : "h"((uint16_t)(acc1[7] >> 16)));
              v_14_1 = _cvt_f32_f16_89;
            }
            m1v_2 = fmaxf(m1v_2, v_14_1);
            float v_15_1 = neg;
            if (c0_1 >= 32) {
              float _cvt_f32_f16_90;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_90) : "h"((uint16_t)(acc1[8] & 65535)));
              v_15_1 = _cvt_f32_f16_90;
            }
            m0v_1 = fmaxf(m0v_1, v_15_1);
            float v_16_1 = neg;
            if (c0_1 >= 33) {
              float _cvt_f32_f16_91;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_91) : "h"((uint16_t)(acc1[8] >> 16)));
              v_16_1 = _cvt_f32_f16_91;
            }
            m0v_1 = fmaxf(m0v_1, v_16_1);
            float v_17_1 = neg;
            if (c1_1 >= 32) {
              float _cvt_f32_f16_92;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_92) : "h"((uint16_t)(acc1[9] & 65535)));
              v_17_1 = _cvt_f32_f16_92;
            }
            m1v_2 = fmaxf(m1v_2, v_17_1);
            float v_18_1 = neg;
            if (c1_1 >= 33) {
              float _cvt_f32_f16_93;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_93) : "h"((uint16_t)(acc1[9] >> 16)));
              v_18_1 = _cvt_f32_f16_93;
            }
            m1v_2 = fmaxf(m1v_2, v_18_1);
            float v_19_1 = neg;
            if (c0_1 >= 40) {
              float _cvt_f32_f16_94;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_94)
                  : "h"((uint16_t)(acc1[10] & 65535)));
              v_19_1 = _cvt_f32_f16_94;
            }
            m0v_1 = fmaxf(m0v_1, v_19_1);
            float v_20_1 = neg;
            if (c0_1 >= 41) {
              float _cvt_f32_f16_95;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_95) : "h"((uint16_t)(acc1[10] >> 16)));
              v_20_1 = _cvt_f32_f16_95;
            }
            m0v_1 = fmaxf(m0v_1, v_20_1);
            float v_21_1 = neg;
            if (c1_1 >= 40) {
              float _cvt_f32_f16_96;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_96)
                  : "h"((uint16_t)(acc1[11] & 65535)));
              v_21_1 = _cvt_f32_f16_96;
            }
            m1v_2 = fmaxf(m1v_2, v_21_1);
            float v_22_1 = neg;
            if (c1_1 >= 41) {
              float _cvt_f32_f16_97;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_97) : "h"((uint16_t)(acc1[11] >> 16)));
              v_22_1 = _cvt_f32_f16_97;
            }
            m1v_2 = fmaxf(m1v_2, v_22_1);
            float v_23_1 = neg;
            if (c0_1 >= 48) {
              float _cvt_f32_f16_98;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_98)
                  : "h"((uint16_t)(acc1[12] & 65535)));
              v_23_1 = _cvt_f32_f16_98;
            }
            m0v_1 = fmaxf(m0v_1, v_23_1);
            float v_24_1 = neg;
            if (c0_1 >= 49) {
              float _cvt_f32_f16_99;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_99) : "h"((uint16_t)(acc1[12] >> 16)));
              v_24_1 = _cvt_f32_f16_99;
            }
            m0v_1 = fmaxf(m0v_1, v_24_1);
            float v_25_1 = neg;
            if (c1_1 >= 48) {
              float _cvt_f32_f16_100;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_100)
                  : "h"((uint16_t)(acc1[13] & 65535)));
              v_25_1 = _cvt_f32_f16_100;
            }
            m1v_2 = fmaxf(m1v_2, v_25_1);
            float v_26_1 = neg;
            if (c1_1 >= 49) {
              float _cvt_f32_f16_101;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_101) : "h"((uint16_t)(acc1[13] >> 16)));
              v_26_1 = _cvt_f32_f16_101;
            }
            m1v_2 = fmaxf(m1v_2, v_26_1);
            float v_27_1 = neg;
            if (c0_1 >= 56) {
              float _cvt_f32_f16_102;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_102)
                  : "h"((uint16_t)(acc1[14] & 65535)));
              v_27_1 = _cvt_f32_f16_102;
            }
            m0v_1 = fmaxf(m0v_1, v_27_1);
            float v_28_1 = neg;
            if (c0_1 >= 57) {
              float _cvt_f32_f16_103;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_103) : "h"((uint16_t)(acc1[14] >> 16)));
              v_28_1 = _cvt_f32_f16_103;
            }
            m0v_1 = fmaxf(m0v_1, v_28_1);
            float v_29_1 = neg;
            if (c1_1 >= 56) {
              float _cvt_f32_f16_104;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_104)
                  : "h"((uint16_t)(acc1[15] & 65535)));
              v_29_1 = _cvt_f32_f16_104;
            }
            m1v_2 = fmaxf(m1v_2, v_29_1);
            float v_30_1 = neg;
            if (c1_1 >= 57) {
              float _cvt_f32_f16_105;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_105) : "h"((uint16_t)(acc1[15] >> 16)));
              v_30_1 = _cvt_f32_f16_105;
            }
            m1v_2 = fmaxf(m1v_2, v_30_1);
            float v_31_1 = neg;
            if (c0_1 >= 64) {
              float _cvt_f32_f16_106;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_106)
                  : "h"((uint16_t)(acc1[16] & 65535)));
              v_31_1 = _cvt_f32_f16_106;
            }
            m0v_1 = fmaxf(m0v_1, v_31_1);
            float v_32_1 = neg;
            if (c0_1 >= 65) {
              float _cvt_f32_f16_107;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_107) : "h"((uint16_t)(acc1[16] >> 16)));
              v_32_1 = _cvt_f32_f16_107;
            }
            m0v_1 = fmaxf(m0v_1, v_32_1);
            float v_33_1 = neg;
            if (c1_1 >= 64) {
              float _cvt_f32_f16_108;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_108)
                  : "h"((uint16_t)(acc1[17] & 65535)));
              v_33_1 = _cvt_f32_f16_108;
            }
            m1v_2 = fmaxf(m1v_2, v_33_1);
            float v_34_1 = neg;
            if (c1_1 >= 65) {
              float _cvt_f32_f16_109;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_109) : "h"((uint16_t)(acc1[17] >> 16)));
              v_34_1 = _cvt_f32_f16_109;
            }
            m1v_2 = fmaxf(m1v_2, v_34_1);
            float v_35_1 = neg;
            if (c0_1 >= 72) {
              float _cvt_f32_f16_110;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_110)
                  : "h"((uint16_t)(acc1[18] & 65535)));
              v_35_1 = _cvt_f32_f16_110;
            }
            m0v_1 = fmaxf(m0v_1, v_35_1);
            float v_36_1 = neg;
            if (c0_1 >= 73) {
              float _cvt_f32_f16_111;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_111) : "h"((uint16_t)(acc1[18] >> 16)));
              v_36_1 = _cvt_f32_f16_111;
            }
            m0v_1 = fmaxf(m0v_1, v_36_1);
            float v_37_1 = neg;
            if (c1_1 >= 72) {
              float _cvt_f32_f16_112;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_112)
                  : "h"((uint16_t)(acc1[19] & 65535)));
              v_37_1 = _cvt_f32_f16_112;
            }
            m1v_2 = fmaxf(m1v_2, v_37_1);
            float v_38_1 = neg;
            if (c1_1 >= 73) {
              float _cvt_f32_f16_113;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_113) : "h"((uint16_t)(acc1[19] >> 16)));
              v_38_1 = _cvt_f32_f16_113;
            }
            m1v_2 = fmaxf(m1v_2, v_38_1);
            float v_39_1 = neg;
            if (c0_1 >= 80) {
              float _cvt_f32_f16_114;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_114)
                  : "h"((uint16_t)(acc1[20] & 65535)));
              v_39_1 = _cvt_f32_f16_114;
            }
            m0v_1 = fmaxf(m0v_1, v_39_1);
            float v_40_1 = neg;
            if (c0_1 >= 81) {
              float _cvt_f32_f16_115;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_115) : "h"((uint16_t)(acc1[20] >> 16)));
              v_40_1 = _cvt_f32_f16_115;
            }
            m0v_1 = fmaxf(m0v_1, v_40_1);
            float v_41_1 = neg;
            if (c1_1 >= 80) {
              float _cvt_f32_f16_116;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_116)
                  : "h"((uint16_t)(acc1[21] & 65535)));
              v_41_1 = _cvt_f32_f16_116;
            }
            m1v_2 = fmaxf(m1v_2, v_41_1);
            float v_42_1 = neg;
            if (c1_1 >= 81) {
              float _cvt_f32_f16_117;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_117) : "h"((uint16_t)(acc1[21] >> 16)));
              v_42_1 = _cvt_f32_f16_117;
            }
            m1v_2 = fmaxf(m1v_2, v_42_1);
            float v_43_1 = neg;
            if (c0_1 >= 88) {
              float _cvt_f32_f16_118;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_118)
                  : "h"((uint16_t)(acc1[22] & 65535)));
              v_43_1 = _cvt_f32_f16_118;
            }
            m0v_1 = fmaxf(m0v_1, v_43_1);
            float v_44_1 = neg;
            if (c0_1 >= 89) {
              float _cvt_f32_f16_119;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_119) : "h"((uint16_t)(acc1[22] >> 16)));
              v_44_1 = _cvt_f32_f16_119;
            }
            m0v_1 = fmaxf(m0v_1, v_44_1);
            float v_45_1 = neg;
            if (c1_1 >= 88) {
              float _cvt_f32_f16_120;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_120)
                  : "h"((uint16_t)(acc1[23] & 65535)));
              v_45_1 = _cvt_f32_f16_120;
            }
            m1v_2 = fmaxf(m1v_2, v_45_1);
            float v_46_1 = neg;
            if (c1_1 >= 89) {
              float _cvt_f32_f16_121;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_121) : "h"((uint16_t)(acc1[23] >> 16)));
              v_46_1 = _cvt_f32_f16_121;
            }
            m1v_2 = fmaxf(m1v_2, v_46_1);
            float v_47_1 = neg;
            if (c0_1 >= 96) {
              float _cvt_f32_f16_122;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_122)
                  : "h"((uint16_t)(acc1[24] & 65535)));
              v_47_1 = _cvt_f32_f16_122;
            }
            m0v_1 = fmaxf(m0v_1, v_47_1);
            float v_48_1 = neg;
            if (c0_1 >= 97) {
              float _cvt_f32_f16_123;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_123) : "h"((uint16_t)(acc1[24] >> 16)));
              v_48_1 = _cvt_f32_f16_123;
            }
            m0v_1 = fmaxf(m0v_1, v_48_1);
            float v_49_1 = neg;
            if (c1_1 >= 96) {
              float _cvt_f32_f16_124;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_124)
                  : "h"((uint16_t)(acc1[25] & 65535)));
              v_49_1 = _cvt_f32_f16_124;
            }
            m1v_2 = fmaxf(m1v_2, v_49_1);
            float v_50_1 = neg;
            if (c1_1 >= 97) {
              float _cvt_f32_f16_125;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_125) : "h"((uint16_t)(acc1[25] >> 16)));
              v_50_1 = _cvt_f32_f16_125;
            }
            m1v_2 = fmaxf(m1v_2, v_50_1);
            float v_51_1 = neg;
            if (c0_1 >= 104) {
              float _cvt_f32_f16_126;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_126)
                  : "h"((uint16_t)(acc1[26] & 65535)));
              v_51_1 = _cvt_f32_f16_126;
            }
            m0v_1 = fmaxf(m0v_1, v_51_1);
            float v_52_1 = neg;
            if (c0_1 >= 105) {
              float _cvt_f32_f16_127;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_127) : "h"((uint16_t)(acc1[26] >> 16)));
              v_52_1 = _cvt_f32_f16_127;
            }
            m0v_1 = fmaxf(m0v_1, v_52_1);
            float v_53_1 = neg;
            if (c1_1 >= 104) {
              float _cvt_f32_f16_128;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_128)
                  : "h"((uint16_t)(acc1[27] & 65535)));
              v_53_1 = _cvt_f32_f16_128;
            }
            m1v_2 = fmaxf(m1v_2, v_53_1);
            float v_54_1 = neg;
            if (c1_1 >= 105) {
              float _cvt_f32_f16_129;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_129) : "h"((uint16_t)(acc1[27] >> 16)));
              v_54_1 = _cvt_f32_f16_129;
            }
            m1v_2 = fmaxf(m1v_2, v_54_1);
            float v_55_1 = neg;
            if (c0_1 >= 112) {
              float _cvt_f32_f16_130;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_130)
                  : "h"((uint16_t)(acc1[28] & 65535)));
              v_55_1 = _cvt_f32_f16_130;
            }
            m0v_1 = fmaxf(m0v_1, v_55_1);
            float v_56_1 = neg;
            if (c0_1 >= 113) {
              float _cvt_f32_f16_131;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_131) : "h"((uint16_t)(acc1[28] >> 16)));
              v_56_1 = _cvt_f32_f16_131;
            }
            m0v_1 = fmaxf(m0v_1, v_56_1);
            float v_57_1 = neg;
            if (c1_1 >= 112) {
              float _cvt_f32_f16_132;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_132)
                  : "h"((uint16_t)(acc1[29] & 65535)));
              v_57_1 = _cvt_f32_f16_132;
            }
            m1v_2 = fmaxf(m1v_2, v_57_1);
            float v_58_1 = neg;
            if (c1_1 >= 113) {
              float _cvt_f32_f16_133;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_133) : "h"((uint16_t)(acc1[29] >> 16)));
              v_58_1 = _cvt_f32_f16_133;
            }
            m1v_2 = fmaxf(m1v_2, v_58_1);
            float v_59_1 = neg;
            if (c0_1 >= 120) {
              float _cvt_f32_f16_134;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_134)
                  : "h"((uint16_t)(acc1[30] & 65535)));
              v_59_1 = _cvt_f32_f16_134;
            }
            m0v_1 = fmaxf(m0v_1, v_59_1);
            float v_60_1 = neg;
            if (c0_1 >= 121) {
              float _cvt_f32_f16_135;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_135) : "h"((uint16_t)(acc1[30] >> 16)));
              v_60_1 = _cvt_f32_f16_135;
            }
            m0v_1 = fmaxf(m0v_1, v_60_1);
            float v_61_1 = neg;
            if (c1_1 >= 120) {
              float _cvt_f32_f16_136;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_136)
                  : "h"((uint16_t)(acc1[31] & 65535)));
              v_61_1 = _cvt_f32_f16_136;
            }
            m1v_2 = fmaxf(m1v_2, v_61_1);
            float v_62_1 = neg;
            if (c1_1 >= 121) {
              float _cvt_f32_f16_137;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_137) : "h"((uint16_t)(acc1[31] >> 16)));
              v_62_1 = _cvt_f32_f16_137;
            }
            m1v_2 = fmaxf(m1v_2, v_62_1);
          } else {
            unsigned int w0_1 = acc1[0];
            unsigned int w1_1 = acc1[1];
            uint32_t _f16x2_max_30;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_30) : "r"(w0_1), "r"(acc1[2]));
            w0_1 = _f16x2_max_30;
            uint32_t _f16x2_max_31;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_31) : "r"(w1_1), "r"(acc1[3]));
            w1_1 = _f16x2_max_31;
            uint32_t _f16x2_max_32;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_32) : "r"(w0_1), "r"(acc1[4]));
            w0_1 = _f16x2_max_32;
            uint32_t _f16x2_max_33;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_33) : "r"(w1_1), "r"(acc1[5]));
            w1_1 = _f16x2_max_33;
            uint32_t _f16x2_max_34;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_34) : "r"(w0_1), "r"(acc1[6]));
            w0_1 = _f16x2_max_34;
            uint32_t _f16x2_max_35;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_35) : "r"(w1_1), "r"(acc1[7]));
            w1_1 = _f16x2_max_35;
            uint32_t _f16x2_max_36;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_36) : "r"(w0_1), "r"(acc1[8]));
            w0_1 = _f16x2_max_36;
            uint32_t _f16x2_max_37;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_37) : "r"(w1_1), "r"(acc1[9]));
            w1_1 = _f16x2_max_37;
            uint32_t _f16x2_max_38;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_38) : "r"(w0_1), "r"(acc1[10]));
            w0_1 = _f16x2_max_38;
            uint32_t _f16x2_max_39;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_39) : "r"(w1_1), "r"(acc1[11]));
            w1_1 = _f16x2_max_39;
            uint32_t _f16x2_max_40;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_40) : "r"(w0_1), "r"(acc1[12]));
            w0_1 = _f16x2_max_40;
            uint32_t _f16x2_max_41;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_41) : "r"(w1_1), "r"(acc1[13]));
            w1_1 = _f16x2_max_41;
            uint32_t _f16x2_max_42;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_42) : "r"(w0_1), "r"(acc1[14]));
            w0_1 = _f16x2_max_42;
            uint32_t _f16x2_max_43;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_43) : "r"(w1_1), "r"(acc1[15]));
            w1_1 = _f16x2_max_43;
            uint32_t _f16x2_max_44;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_44) : "r"(w0_1), "r"(acc1[16]));
            w0_1 = _f16x2_max_44;
            uint32_t _f16x2_max_45;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_45) : "r"(w1_1), "r"(acc1[17]));
            w1_1 = _f16x2_max_45;
            uint32_t _f16x2_max_46;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_46) : "r"(w0_1), "r"(acc1[18]));
            w0_1 = _f16x2_max_46;
            uint32_t _f16x2_max_47;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_47) : "r"(w1_1), "r"(acc1[19]));
            w1_1 = _f16x2_max_47;
            uint32_t _f16x2_max_48;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_48) : "r"(w0_1), "r"(acc1[20]));
            w0_1 = _f16x2_max_48;
            uint32_t _f16x2_max_49;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_49) : "r"(w1_1), "r"(acc1[21]));
            w1_1 = _f16x2_max_49;
            uint32_t _f16x2_max_50;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_50) : "r"(w0_1), "r"(acc1[22]));
            w0_1 = _f16x2_max_50;
            uint32_t _f16x2_max_51;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_51) : "r"(w1_1), "r"(acc1[23]));
            w1_1 = _f16x2_max_51;
            uint32_t _f16x2_max_52;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_52) : "r"(w0_1), "r"(acc1[24]));
            w0_1 = _f16x2_max_52;
            uint32_t _f16x2_max_53;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_53) : "r"(w1_1), "r"(acc1[25]));
            w1_1 = _f16x2_max_53;
            uint32_t _f16x2_max_54;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_54) : "r"(w0_1), "r"(acc1[26]));
            w0_1 = _f16x2_max_54;
            uint32_t _f16x2_max_55;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_55) : "r"(w1_1), "r"(acc1[27]));
            w1_1 = _f16x2_max_55;
            uint32_t _f16x2_max_56;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_56) : "r"(w0_1), "r"(acc1[28]));
            w0_1 = _f16x2_max_56;
            uint32_t _f16x2_max_57;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_57) : "r"(w1_1), "r"(acc1[29]));
            w1_1 = _f16x2_max_57;
            uint32_t _f16x2_max_58;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_58) : "r"(w0_1), "r"(acc1[30]));
            w0_1 = _f16x2_max_58;
            uint32_t _f16x2_max_59;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_59) : "r"(w1_1), "r"(acc1[31]));
            w1_1 = _f16x2_max_59;
            uint16_t _f16_max_2;
            asm("max.f16 %0, %1, %2;"
                : "=h"(_f16_max_2)
                : "h"((uint16_t)(w0_1 & 65535)), "h"((uint16_t)(w0_1 >> 16)));
            float _cvt_f32_f16_138;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_138) : "h"((uint16_t)(_f16_max_2)));
            m0v_1 = _cvt_f32_f16_138;
            uint16_t _f16_max_3;
            asm("max.f16 %0, %1, %2;"
                : "=h"(_f16_max_3)
                : "h"((uint16_t)(w1_1 & 65535)), "h"((uint16_t)(w1_1 >> 16)));
            float _cvt_f32_f16_139;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_139) : "h"((uint16_t)(_f16_max_3)));
            m1v_2 = _cvt_f32_f16_139;
          }
          float _shfl_xor_sync_140 = __shfl_xor_sync(0xFFFFFFFF, m0v_1, 1);
          m0v_1 = fmaxf(m0v_1, _shfl_xor_sync_140);
          float _shfl_xor_sync_141 = __shfl_xor_sync(0xFFFFFFFF, m0v_1, 2);
          m0v_1 = fmaxf(m0v_1, _shfl_xor_sync_141);
          float _shfl_xor_sync_142 = __shfl_xor_sync(0xFFFFFFFF, m1v_2, 1);
          m1v_2 = fmaxf(m1v_2, _shfl_xor_sync_142);
          float _shfl_xor_sync_143 = __shfl_xor_sync(0xFFFFFFFF, m1v_2, 2);
          m1v_2 = fmaxf(m1v_2, _shfl_xor_sync_143);
          if (cq == 0) {
            int orow_1 = obase + (split_1 + ib * nsplit) * total_q + r0;
            if (r0 < rows_valid) {
              out[orow_1] = m0v_1;
            }
            if (rows_valid > r0 + 8) {
              out[orow_1 + 8] = m1v_2;
            }
          }
        }
        int il = 2 * npair;
        int sl = il % 4;
        if (n_comp_1 - il == 2) {
          int im = il + 1;
          int sm = im % 4;
          mbarrier_wait(full_addr + (sm) * 8, im / 4 & 1);
          asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
          uint64_t _wgmma_desc_144 =
              (((uint64_t)(((k_s_addr + (unsigned int)(sm * 16384))) >> 4) & 0x3FFFULL) |
               ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
          uint64_t _wgmma_b_3 =
              ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_144 >> 32)) << 32) |
              (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_144);
          asm volatile(
              "{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, "
              "%6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
              "%23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1;\n}\n"
              : "+r"(acc1[0]), "+r"(acc1[1]), "+r"(acc1[2]), "+r"(acc1[3]), "+r"(acc1[4]),
                "+r"(acc1[5]), "+r"(acc1[6]), "+r"(acc1[7]), "+r"(acc1[8]), "+r"(acc1[9]),
                "+r"(acc1[10]), "+r"(acc1[11]), "+r"(acc1[12]), "+r"(acc1[13]), "+r"(acc1[14]),
                "+r"(acc1[15]), "+r"(acc1[16]), "+r"(acc1[17]), "+r"(acc1[18]), "+r"(acc1[19]),
                "+r"(acc1[20]), "+r"(acc1[21]), "+r"(acc1[22]), "+r"(acc1[23]), "+r"(acc1[24]),
                "+r"(acc1[25]), "+r"(acc1[26]), "+r"(acc1[27]), "+r"(acc1[28]), "+r"(acc1[29]),
                "+r"(acc1[30]), "+r"(acc1[31])
              : "l"(_wgmma_a_0), "l"(_wgmma_b_3)
              : "memory");
          asm volatile(
              "{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, "
              "%6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
              "%23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
              : "+r"(acc1[0]), "+r"(acc1[1]), "+r"(acc1[2]), "+r"(acc1[3]), "+r"(acc1[4]),
                "+r"(acc1[5]), "+r"(acc1[6]), "+r"(acc1[7]), "+r"(acc1[8]), "+r"(acc1[9]),
                "+r"(acc1[10]), "+r"(acc1[11]), "+r"(acc1[12]), "+r"(acc1[13]), "+r"(acc1[14]),
                "+r"(acc1[15]), "+r"(acc1[16]), "+r"(acc1[17]), "+r"(acc1[18]), "+r"(acc1[19]),
                "+r"(acc1[20]), "+r"(acc1[21]), "+r"(acc1[22]), "+r"(acc1[23]), "+r"(acc1[24]),
                "+r"(acc1[25]), "+r"(acc1[26]), "+r"(acc1[27]), "+r"(acc1[28]), "+r"(acc1[29]),
                "+r"(acc1[30]), "+r"(acc1[31])
              : "l"(_wgmma_a_0 + 2), "l"(_wgmma_b_3 + 2)
              : "memory");
          asm volatile(
              "{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, "
              "%6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
              "%23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
              : "+r"(acc1[0]), "+r"(acc1[1]), "+r"(acc1[2]), "+r"(acc1[3]), "+r"(acc1[4]),
                "+r"(acc1[5]), "+r"(acc1[6]), "+r"(acc1[7]), "+r"(acc1[8]), "+r"(acc1[9]),
                "+r"(acc1[10]), "+r"(acc1[11]), "+r"(acc1[12]), "+r"(acc1[13]), "+r"(acc1[14]),
                "+r"(acc1[15]), "+r"(acc1[16]), "+r"(acc1[17]), "+r"(acc1[18]), "+r"(acc1[19]),
                "+r"(acc1[20]), "+r"(acc1[21]), "+r"(acc1[22]), "+r"(acc1[23]), "+r"(acc1[24]),
                "+r"(acc1[25]), "+r"(acc1[26]), "+r"(acc1[27]), "+r"(acc1[28]), "+r"(acc1[29]),
                "+r"(acc1[30]), "+r"(acc1[31])
              : "l"(_wgmma_a_0 + 4), "l"(_wgmma_b_3 + 4)
              : "memory");
          asm volatile(
              "{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, "
              "%6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
              "%23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
              : "+r"(acc1[0]), "+r"(acc1[1]), "+r"(acc1[2]), "+r"(acc1[3]), "+r"(acc1[4]),
                "+r"(acc1[5]), "+r"(acc1[6]), "+r"(acc1[7]), "+r"(acc1[8]), "+r"(acc1[9]),
                "+r"(acc1[10]), "+r"(acc1[11]), "+r"(acc1[12]), "+r"(acc1[13]), "+r"(acc1[14]),
                "+r"(acc1[15]), "+r"(acc1[16]), "+r"(acc1[17]), "+r"(acc1[18]), "+r"(acc1[19]),
                "+r"(acc1[20]), "+r"(acc1[21]), "+r"(acc1[22]), "+r"(acc1[23]), "+r"(acc1[24]),
                "+r"(acc1[25]), "+r"(acc1[26]), "+r"(acc1[27]), "+r"(acc1[28]), "+r"(acc1[29]),
                "+r"(acc1[30]), "+r"(acc1[31])
              : "l"(_wgmma_a_0 + 6), "l"(_wgmma_b_3 + 6)
              : "memory");
          asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
          asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
          if (elect_sync()) {
            mbarrier_arrive(empty_addr + (sl) * 8);
          }
          int k0_1 = (split_1 + il * nsplit) * 128;
          float m0v_2 = neg;
          float m1v_1 = neg;
          if (mask_thr < k0_1 + 127) {
            int c0_2 = lim0 - k0_1 - 2 * cq;
            int c1_2 = lim1 - k0_1 - 2 * cq;
            float v_64 = neg;
            if (c0_2 >= 0) {
              float _cvt_f32_f16_145;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_145) : "h"((uint16_t)(acc[0] & 65535)));
              v_64 = _cvt_f32_f16_145;
            }
            m0v_2 = fmaxf(m0v_2, v_64);
            float v_0_2 = neg;
            if (c0_2 >= 1) {
              float _cvt_f32_f16_146;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_146) : "h"((uint16_t)(acc[0] >> 16)));
              v_0_2 = _cvt_f32_f16_146;
            }
            m0v_2 = fmaxf(m0v_2, v_0_2);
            float v_1_2 = neg;
            if (c1_2 >= 0) {
              float _cvt_f32_f16_147;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_147) : "h"((uint16_t)(acc[1] & 65535)));
              v_1_2 = _cvt_f32_f16_147;
            }
            m1v_1 = fmaxf(m1v_1, v_1_2);
            float v_2_2 = neg;
            if (c1_2 >= 1) {
              float _cvt_f32_f16_148;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_148) : "h"((uint16_t)(acc[1] >> 16)));
              v_2_2 = _cvt_f32_f16_148;
            }
            m1v_1 = fmaxf(m1v_1, v_2_2);
            float v_3_2 = neg;
            if (c0_2 >= 8) {
              float _cvt_f32_f16_149;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_149) : "h"((uint16_t)(acc[2] & 65535)));
              v_3_2 = _cvt_f32_f16_149;
            }
            m0v_2 = fmaxf(m0v_2, v_3_2);
            float v_4_2 = neg;
            if (c0_2 >= 9) {
              float _cvt_f32_f16_150;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_150) : "h"((uint16_t)(acc[2] >> 16)));
              v_4_2 = _cvt_f32_f16_150;
            }
            m0v_2 = fmaxf(m0v_2, v_4_2);
            float v_5_2 = neg;
            if (c1_2 >= 8) {
              float _cvt_f32_f16_151;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_151) : "h"((uint16_t)(acc[3] & 65535)));
              v_5_2 = _cvt_f32_f16_151;
            }
            m1v_1 = fmaxf(m1v_1, v_5_2);
            float v_6_2 = neg;
            if (c1_2 >= 9) {
              float _cvt_f32_f16_152;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_152) : "h"((uint16_t)(acc[3] >> 16)));
              v_6_2 = _cvt_f32_f16_152;
            }
            m1v_1 = fmaxf(m1v_1, v_6_2);
            float v_7_2 = neg;
            if (c0_2 >= 16) {
              float _cvt_f32_f16_153;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_153) : "h"((uint16_t)(acc[4] & 65535)));
              v_7_2 = _cvt_f32_f16_153;
            }
            m0v_2 = fmaxf(m0v_2, v_7_2);
            float v_8_2 = neg;
            if (c0_2 >= 17) {
              float _cvt_f32_f16_154;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_154) : "h"((uint16_t)(acc[4] >> 16)));
              v_8_2 = _cvt_f32_f16_154;
            }
            m0v_2 = fmaxf(m0v_2, v_8_2);
            float v_9_2 = neg;
            if (c1_2 >= 16) {
              float _cvt_f32_f16_155;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_155) : "h"((uint16_t)(acc[5] & 65535)));
              v_9_2 = _cvt_f32_f16_155;
            }
            m1v_1 = fmaxf(m1v_1, v_9_2);
            float v_10_2 = neg;
            if (c1_2 >= 17) {
              float _cvt_f32_f16_156;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_156) : "h"((uint16_t)(acc[5] >> 16)));
              v_10_2 = _cvt_f32_f16_156;
            }
            m1v_1 = fmaxf(m1v_1, v_10_2);
            float v_11_2 = neg;
            if (c0_2 >= 24) {
              float _cvt_f32_f16_157;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_157) : "h"((uint16_t)(acc[6] & 65535)));
              v_11_2 = _cvt_f32_f16_157;
            }
            m0v_2 = fmaxf(m0v_2, v_11_2);
            float v_12_2 = neg;
            if (c0_2 >= 25) {
              float _cvt_f32_f16_158;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_158) : "h"((uint16_t)(acc[6] >> 16)));
              v_12_2 = _cvt_f32_f16_158;
            }
            m0v_2 = fmaxf(m0v_2, v_12_2);
            float v_13_2 = neg;
            if (c1_2 >= 24) {
              float _cvt_f32_f16_159;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_159) : "h"((uint16_t)(acc[7] & 65535)));
              v_13_2 = _cvt_f32_f16_159;
            }
            m1v_1 = fmaxf(m1v_1, v_13_2);
            float v_14_2 = neg;
            if (c1_2 >= 25) {
              float _cvt_f32_f16_160;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_160) : "h"((uint16_t)(acc[7] >> 16)));
              v_14_2 = _cvt_f32_f16_160;
            }
            m1v_1 = fmaxf(m1v_1, v_14_2);
            float v_15_2 = neg;
            if (c0_2 >= 32) {
              float _cvt_f32_f16_161;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_161) : "h"((uint16_t)(acc[8] & 65535)));
              v_15_2 = _cvt_f32_f16_161;
            }
            m0v_2 = fmaxf(m0v_2, v_15_2);
            float v_16_2 = neg;
            if (c0_2 >= 33) {
              float _cvt_f32_f16_162;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_162) : "h"((uint16_t)(acc[8] >> 16)));
              v_16_2 = _cvt_f32_f16_162;
            }
            m0v_2 = fmaxf(m0v_2, v_16_2);
            float v_17_2 = neg;
            if (c1_2 >= 32) {
              float _cvt_f32_f16_163;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_163) : "h"((uint16_t)(acc[9] & 65535)));
              v_17_2 = _cvt_f32_f16_163;
            }
            m1v_1 = fmaxf(m1v_1, v_17_2);
            float v_18_2 = neg;
            if (c1_2 >= 33) {
              float _cvt_f32_f16_164;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_164) : "h"((uint16_t)(acc[9] >> 16)));
              v_18_2 = _cvt_f32_f16_164;
            }
            m1v_1 = fmaxf(m1v_1, v_18_2);
            float v_19_2 = neg;
            if (c0_2 >= 40) {
              float _cvt_f32_f16_165;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_165)
                  : "h"((uint16_t)(acc[10] & 65535)));
              v_19_2 = _cvt_f32_f16_165;
            }
            m0v_2 = fmaxf(m0v_2, v_19_2);
            float v_20_2 = neg;
            if (c0_2 >= 41) {
              float _cvt_f32_f16_166;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_166) : "h"((uint16_t)(acc[10] >> 16)));
              v_20_2 = _cvt_f32_f16_166;
            }
            m0v_2 = fmaxf(m0v_2, v_20_2);
            float v_21_2 = neg;
            if (c1_2 >= 40) {
              float _cvt_f32_f16_167;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_167)
                  : "h"((uint16_t)(acc[11] & 65535)));
              v_21_2 = _cvt_f32_f16_167;
            }
            m1v_1 = fmaxf(m1v_1, v_21_2);
            float v_22_2 = neg;
            if (c1_2 >= 41) {
              float _cvt_f32_f16_168;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_168) : "h"((uint16_t)(acc[11] >> 16)));
              v_22_2 = _cvt_f32_f16_168;
            }
            m1v_1 = fmaxf(m1v_1, v_22_2);
            float v_23_2 = neg;
            if (c0_2 >= 48) {
              float _cvt_f32_f16_169;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_169)
                  : "h"((uint16_t)(acc[12] & 65535)));
              v_23_2 = _cvt_f32_f16_169;
            }
            m0v_2 = fmaxf(m0v_2, v_23_2);
            float v_24_2 = neg;
            if (c0_2 >= 49) {
              float _cvt_f32_f16_170;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_170) : "h"((uint16_t)(acc[12] >> 16)));
              v_24_2 = _cvt_f32_f16_170;
            }
            m0v_2 = fmaxf(m0v_2, v_24_2);
            float v_25_2 = neg;
            if (c1_2 >= 48) {
              float _cvt_f32_f16_171;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_171)
                  : "h"((uint16_t)(acc[13] & 65535)));
              v_25_2 = _cvt_f32_f16_171;
            }
            m1v_1 = fmaxf(m1v_1, v_25_2);
            float v_26_2 = neg;
            if (c1_2 >= 49) {
              float _cvt_f32_f16_172;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_172) : "h"((uint16_t)(acc[13] >> 16)));
              v_26_2 = _cvt_f32_f16_172;
            }
            m1v_1 = fmaxf(m1v_1, v_26_2);
            float v_27_2 = neg;
            if (c0_2 >= 56) {
              float _cvt_f32_f16_173;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_173)
                  : "h"((uint16_t)(acc[14] & 65535)));
              v_27_2 = _cvt_f32_f16_173;
            }
            m0v_2 = fmaxf(m0v_2, v_27_2);
            float v_28_2 = neg;
            if (c0_2 >= 57) {
              float _cvt_f32_f16_174;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_174) : "h"((uint16_t)(acc[14] >> 16)));
              v_28_2 = _cvt_f32_f16_174;
            }
            m0v_2 = fmaxf(m0v_2, v_28_2);
            float v_29_2 = neg;
            if (c1_2 >= 56) {
              float _cvt_f32_f16_175;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_175)
                  : "h"((uint16_t)(acc[15] & 65535)));
              v_29_2 = _cvt_f32_f16_175;
            }
            m1v_1 = fmaxf(m1v_1, v_29_2);
            float v_30_2 = neg;
            if (c1_2 >= 57) {
              float _cvt_f32_f16_176;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_176) : "h"((uint16_t)(acc[15] >> 16)));
              v_30_2 = _cvt_f32_f16_176;
            }
            m1v_1 = fmaxf(m1v_1, v_30_2);
            float v_31_2 = neg;
            if (c0_2 >= 64) {
              float _cvt_f32_f16_177;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_177)
                  : "h"((uint16_t)(acc[16] & 65535)));
              v_31_2 = _cvt_f32_f16_177;
            }
            m0v_2 = fmaxf(m0v_2, v_31_2);
            float v_32_2 = neg;
            if (c0_2 >= 65) {
              float _cvt_f32_f16_178;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_178) : "h"((uint16_t)(acc[16] >> 16)));
              v_32_2 = _cvt_f32_f16_178;
            }
            m0v_2 = fmaxf(m0v_2, v_32_2);
            float v_33_2 = neg;
            if (c1_2 >= 64) {
              float _cvt_f32_f16_179;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_179)
                  : "h"((uint16_t)(acc[17] & 65535)));
              v_33_2 = _cvt_f32_f16_179;
            }
            m1v_1 = fmaxf(m1v_1, v_33_2);
            float v_34_2 = neg;
            if (c1_2 >= 65) {
              float _cvt_f32_f16_180;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_180) : "h"((uint16_t)(acc[17] >> 16)));
              v_34_2 = _cvt_f32_f16_180;
            }
            m1v_1 = fmaxf(m1v_1, v_34_2);
            float v_35_2 = neg;
            if (c0_2 >= 72) {
              float _cvt_f32_f16_181;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_181)
                  : "h"((uint16_t)(acc[18] & 65535)));
              v_35_2 = _cvt_f32_f16_181;
            }
            m0v_2 = fmaxf(m0v_2, v_35_2);
            float v_36_2 = neg;
            if (c0_2 >= 73) {
              float _cvt_f32_f16_182;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_182) : "h"((uint16_t)(acc[18] >> 16)));
              v_36_2 = _cvt_f32_f16_182;
            }
            m0v_2 = fmaxf(m0v_2, v_36_2);
            float v_37_2 = neg;
            if (c1_2 >= 72) {
              float _cvt_f32_f16_183;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_183)
                  : "h"((uint16_t)(acc[19] & 65535)));
              v_37_2 = _cvt_f32_f16_183;
            }
            m1v_1 = fmaxf(m1v_1, v_37_2);
            float v_38_2 = neg;
            if (c1_2 >= 73) {
              float _cvt_f32_f16_184;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_184) : "h"((uint16_t)(acc[19] >> 16)));
              v_38_2 = _cvt_f32_f16_184;
            }
            m1v_1 = fmaxf(m1v_1, v_38_2);
            float v_39_2 = neg;
            if (c0_2 >= 80) {
              float _cvt_f32_f16_185;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_185)
                  : "h"((uint16_t)(acc[20] & 65535)));
              v_39_2 = _cvt_f32_f16_185;
            }
            m0v_2 = fmaxf(m0v_2, v_39_2);
            float v_40_2 = neg;
            if (c0_2 >= 81) {
              float _cvt_f32_f16_186;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_186) : "h"((uint16_t)(acc[20] >> 16)));
              v_40_2 = _cvt_f32_f16_186;
            }
            m0v_2 = fmaxf(m0v_2, v_40_2);
            float v_41_2 = neg;
            if (c1_2 >= 80) {
              float _cvt_f32_f16_187;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_187)
                  : "h"((uint16_t)(acc[21] & 65535)));
              v_41_2 = _cvt_f32_f16_187;
            }
            m1v_1 = fmaxf(m1v_1, v_41_2);
            float v_42_2 = neg;
            if (c1_2 >= 81) {
              float _cvt_f32_f16_188;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_188) : "h"((uint16_t)(acc[21] >> 16)));
              v_42_2 = _cvt_f32_f16_188;
            }
            m1v_1 = fmaxf(m1v_1, v_42_2);
            float v_43_2 = neg;
            if (c0_2 >= 88) {
              float _cvt_f32_f16_189;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_189)
                  : "h"((uint16_t)(acc[22] & 65535)));
              v_43_2 = _cvt_f32_f16_189;
            }
            m0v_2 = fmaxf(m0v_2, v_43_2);
            float v_44_2 = neg;
            if (c0_2 >= 89) {
              float _cvt_f32_f16_190;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_190) : "h"((uint16_t)(acc[22] >> 16)));
              v_44_2 = _cvt_f32_f16_190;
            }
            m0v_2 = fmaxf(m0v_2, v_44_2);
            float v_45_2 = neg;
            if (c1_2 >= 88) {
              float _cvt_f32_f16_191;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_191)
                  : "h"((uint16_t)(acc[23] & 65535)));
              v_45_2 = _cvt_f32_f16_191;
            }
            m1v_1 = fmaxf(m1v_1, v_45_2);
            float v_46_2 = neg;
            if (c1_2 >= 89) {
              float _cvt_f32_f16_192;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_192) : "h"((uint16_t)(acc[23] >> 16)));
              v_46_2 = _cvt_f32_f16_192;
            }
            m1v_1 = fmaxf(m1v_1, v_46_2);
            float v_47_2 = neg;
            if (c0_2 >= 96) {
              float _cvt_f32_f16_193;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_193)
                  : "h"((uint16_t)(acc[24] & 65535)));
              v_47_2 = _cvt_f32_f16_193;
            }
            m0v_2 = fmaxf(m0v_2, v_47_2);
            float v_48_2 = neg;
            if (c0_2 >= 97) {
              float _cvt_f32_f16_194;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_194) : "h"((uint16_t)(acc[24] >> 16)));
              v_48_2 = _cvt_f32_f16_194;
            }
            m0v_2 = fmaxf(m0v_2, v_48_2);
            float v_49_2 = neg;
            if (c1_2 >= 96) {
              float _cvt_f32_f16_195;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_195)
                  : "h"((uint16_t)(acc[25] & 65535)));
              v_49_2 = _cvt_f32_f16_195;
            }
            m1v_1 = fmaxf(m1v_1, v_49_2);
            float v_50_2 = neg;
            if (c1_2 >= 97) {
              float _cvt_f32_f16_196;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_196) : "h"((uint16_t)(acc[25] >> 16)));
              v_50_2 = _cvt_f32_f16_196;
            }
            m1v_1 = fmaxf(m1v_1, v_50_2);
            float v_51_2 = neg;
            if (c0_2 >= 104) {
              float _cvt_f32_f16_197;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_197)
                  : "h"((uint16_t)(acc[26] & 65535)));
              v_51_2 = _cvt_f32_f16_197;
            }
            m0v_2 = fmaxf(m0v_2, v_51_2);
            float v_52_2 = neg;
            if (c0_2 >= 105) {
              float _cvt_f32_f16_198;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_198) : "h"((uint16_t)(acc[26] >> 16)));
              v_52_2 = _cvt_f32_f16_198;
            }
            m0v_2 = fmaxf(m0v_2, v_52_2);
            float v_53_2 = neg;
            if (c1_2 >= 104) {
              float _cvt_f32_f16_199;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_199)
                  : "h"((uint16_t)(acc[27] & 65535)));
              v_53_2 = _cvt_f32_f16_199;
            }
            m1v_1 = fmaxf(m1v_1, v_53_2);
            float v_54_2 = neg;
            if (c1_2 >= 105) {
              float _cvt_f32_f16_200;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_200) : "h"((uint16_t)(acc[27] >> 16)));
              v_54_2 = _cvt_f32_f16_200;
            }
            m1v_1 = fmaxf(m1v_1, v_54_2);
            float v_55_2 = neg;
            if (c0_2 >= 112) {
              float _cvt_f32_f16_201;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_201)
                  : "h"((uint16_t)(acc[28] & 65535)));
              v_55_2 = _cvt_f32_f16_201;
            }
            m0v_2 = fmaxf(m0v_2, v_55_2);
            float v_56_2 = neg;
            if (c0_2 >= 113) {
              float _cvt_f32_f16_202;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_202) : "h"((uint16_t)(acc[28] >> 16)));
              v_56_2 = _cvt_f32_f16_202;
            }
            m0v_2 = fmaxf(m0v_2, v_56_2);
            float v_57_2 = neg;
            if (c1_2 >= 112) {
              float _cvt_f32_f16_203;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_203)
                  : "h"((uint16_t)(acc[29] & 65535)));
              v_57_2 = _cvt_f32_f16_203;
            }
            m1v_1 = fmaxf(m1v_1, v_57_2);
            float v_58_2 = neg;
            if (c1_2 >= 113) {
              float _cvt_f32_f16_204;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_204) : "h"((uint16_t)(acc[29] >> 16)));
              v_58_2 = _cvt_f32_f16_204;
            }
            m1v_1 = fmaxf(m1v_1, v_58_2);
            float v_59_2 = neg;
            if (c0_2 >= 120) {
              float _cvt_f32_f16_205;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_205)
                  : "h"((uint16_t)(acc[30] & 65535)));
              v_59_2 = _cvt_f32_f16_205;
            }
            m0v_2 = fmaxf(m0v_2, v_59_2);
            float v_60_2 = neg;
            if (c0_2 >= 121) {
              float _cvt_f32_f16_206;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_206) : "h"((uint16_t)(acc[30] >> 16)));
              v_60_2 = _cvt_f32_f16_206;
            }
            m0v_2 = fmaxf(m0v_2, v_60_2);
            float v_61_2 = neg;
            if (c1_2 >= 120) {
              float _cvt_f32_f16_207;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_207)
                  : "h"((uint16_t)(acc[31] & 65535)));
              v_61_2 = _cvt_f32_f16_207;
            }
            m1v_1 = fmaxf(m1v_1, v_61_2);
            float v_62_2 = neg;
            if (c1_2 >= 121) {
              float _cvt_f32_f16_208;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_208) : "h"((uint16_t)(acc[31] >> 16)));
              v_62_2 = _cvt_f32_f16_208;
            }
            m1v_1 = fmaxf(m1v_1, v_62_2);
          } else {
            unsigned int w0_2 = acc[0];
            unsigned int w1_2 = acc[1];
            uint32_t _f16x2_max_60;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_60) : "r"(w0_2), "r"(acc[2]));
            w0_2 = _f16x2_max_60;
            uint32_t _f16x2_max_61;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_61) : "r"(w1_2), "r"(acc[3]));
            w1_2 = _f16x2_max_61;
            uint32_t _f16x2_max_62;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_62) : "r"(w0_2), "r"(acc[4]));
            w0_2 = _f16x2_max_62;
            uint32_t _f16x2_max_63;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_63) : "r"(w1_2), "r"(acc[5]));
            w1_2 = _f16x2_max_63;
            uint32_t _f16x2_max_64;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_64) : "r"(w0_2), "r"(acc[6]));
            w0_2 = _f16x2_max_64;
            uint32_t _f16x2_max_65;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_65) : "r"(w1_2), "r"(acc[7]));
            w1_2 = _f16x2_max_65;
            uint32_t _f16x2_max_66;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_66) : "r"(w0_2), "r"(acc[8]));
            w0_2 = _f16x2_max_66;
            uint32_t _f16x2_max_67;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_67) : "r"(w1_2), "r"(acc[9]));
            w1_2 = _f16x2_max_67;
            uint32_t _f16x2_max_68;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_68) : "r"(w0_2), "r"(acc[10]));
            w0_2 = _f16x2_max_68;
            uint32_t _f16x2_max_69;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_69) : "r"(w1_2), "r"(acc[11]));
            w1_2 = _f16x2_max_69;
            uint32_t _f16x2_max_70;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_70) : "r"(w0_2), "r"(acc[12]));
            w0_2 = _f16x2_max_70;
            uint32_t _f16x2_max_71;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_71) : "r"(w1_2), "r"(acc[13]));
            w1_2 = _f16x2_max_71;
            uint32_t _f16x2_max_72;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_72) : "r"(w0_2), "r"(acc[14]));
            w0_2 = _f16x2_max_72;
            uint32_t _f16x2_max_73;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_73) : "r"(w1_2), "r"(acc[15]));
            w1_2 = _f16x2_max_73;
            uint32_t _f16x2_max_74;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_74) : "r"(w0_2), "r"(acc[16]));
            w0_2 = _f16x2_max_74;
            uint32_t _f16x2_max_75;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_75) : "r"(w1_2), "r"(acc[17]));
            w1_2 = _f16x2_max_75;
            uint32_t _f16x2_max_76;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_76) : "r"(w0_2), "r"(acc[18]));
            w0_2 = _f16x2_max_76;
            uint32_t _f16x2_max_77;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_77) : "r"(w1_2), "r"(acc[19]));
            w1_2 = _f16x2_max_77;
            uint32_t _f16x2_max_78;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_78) : "r"(w0_2), "r"(acc[20]));
            w0_2 = _f16x2_max_78;
            uint32_t _f16x2_max_79;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_79) : "r"(w1_2), "r"(acc[21]));
            w1_2 = _f16x2_max_79;
            uint32_t _f16x2_max_80;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_80) : "r"(w0_2), "r"(acc[22]));
            w0_2 = _f16x2_max_80;
            uint32_t _f16x2_max_81;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_81) : "r"(w1_2), "r"(acc[23]));
            w1_2 = _f16x2_max_81;
            uint32_t _f16x2_max_82;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_82) : "r"(w0_2), "r"(acc[24]));
            w0_2 = _f16x2_max_82;
            uint32_t _f16x2_max_83;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_83) : "r"(w1_2), "r"(acc[25]));
            w1_2 = _f16x2_max_83;
            uint32_t _f16x2_max_84;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_84) : "r"(w0_2), "r"(acc[26]));
            w0_2 = _f16x2_max_84;
            uint32_t _f16x2_max_85;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_85) : "r"(w1_2), "r"(acc[27]));
            w1_2 = _f16x2_max_85;
            uint32_t _f16x2_max_86;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_86) : "r"(w0_2), "r"(acc[28]));
            w0_2 = _f16x2_max_86;
            uint32_t _f16x2_max_87;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_87) : "r"(w1_2), "r"(acc[29]));
            w1_2 = _f16x2_max_87;
            uint32_t _f16x2_max_88;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_88) : "r"(w0_2), "r"(acc[30]));
            w0_2 = _f16x2_max_88;
            uint32_t _f16x2_max_89;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_89) : "r"(w1_2), "r"(acc[31]));
            w1_2 = _f16x2_max_89;
            uint16_t _f16_max_4;
            asm("max.f16 %0, %1, %2;"
                : "=h"(_f16_max_4)
                : "h"((uint16_t)(w0_2 & 65535)), "h"((uint16_t)(w0_2 >> 16)));
            float _cvt_f32_f16_209;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_209) : "h"((uint16_t)(_f16_max_4)));
            m0v_2 = _cvt_f32_f16_209;
            uint16_t _f16_max_5;
            asm("max.f16 %0, %1, %2;"
                : "=h"(_f16_max_5)
                : "h"((uint16_t)(w1_2 & 65535)), "h"((uint16_t)(w1_2 >> 16)));
            float _cvt_f32_f16_210;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_210) : "h"((uint16_t)(_f16_max_5)));
            m1v_1 = _cvt_f32_f16_210;
          }
          float _shfl_xor_sync_211 = __shfl_xor_sync(0xFFFFFFFF, m0v_2, 1);
          m0v_2 = fmaxf(m0v_2, _shfl_xor_sync_211);
          float _shfl_xor_sync_212 = __shfl_xor_sync(0xFFFFFFFF, m0v_2, 2);
          m0v_2 = fmaxf(m0v_2, _shfl_xor_sync_212);
          float _shfl_xor_sync_213 = __shfl_xor_sync(0xFFFFFFFF, m1v_1, 1);
          m1v_1 = fmaxf(m1v_1, _shfl_xor_sync_213);
          float _shfl_xor_sync_214 = __shfl_xor_sync(0xFFFFFFFF, m1v_1, 2);
          m1v_1 = fmaxf(m1v_1, _shfl_xor_sync_214);
          if (cq == 0) {
            int orow_2 = obase + (split_1 + il * nsplit) * total_q + r0;
            if (r0 < rows_valid) {
              out[orow_2] = m0v_2;
            }
            if (rows_valid > r0 + 8) {
              out[orow_2 + 8] = m1v_1;
            }
          }
          asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
          if (elect_sync()) {
            mbarrier_arrive(empty_addr + (sm) * 8);
          }
          int k0_0_1 = (split_1 + im * nsplit) * 128;
          float m0v_1_1 = neg;
          float m1v_2_1 = neg;
          if (mask_thr < k0_0_1 + 127) {
            int c0_3 = lim0 - k0_0_1 - 2 * cq;
            int c1_3 = lim1 - k0_0_1 - 2 * cq;
            float v_65 = neg;
            if (c0_3 >= 0) {
              float _cvt_f32_f16_215;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_215)
                  : "h"((uint16_t)(acc1[0] & 65535)));
              v_65 = _cvt_f32_f16_215;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_65);
            float v_0_3 = neg;
            if (c0_3 >= 1) {
              float _cvt_f32_f16_216;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_216) : "h"((uint16_t)(acc1[0] >> 16)));
              v_0_3 = _cvt_f32_f16_216;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_0_3);
            float v_1_3 = neg;
            if (c1_3 >= 0) {
              float _cvt_f32_f16_217;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_217)
                  : "h"((uint16_t)(acc1[1] & 65535)));
              v_1_3 = _cvt_f32_f16_217;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_1_3);
            float v_2_3 = neg;
            if (c1_3 >= 1) {
              float _cvt_f32_f16_218;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_218) : "h"((uint16_t)(acc1[1] >> 16)));
              v_2_3 = _cvt_f32_f16_218;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_2_3);
            float v_3_3 = neg;
            if (c0_3 >= 8) {
              float _cvt_f32_f16_219;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_219)
                  : "h"((uint16_t)(acc1[2] & 65535)));
              v_3_3 = _cvt_f32_f16_219;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_3_3);
            float v_4_3 = neg;
            if (c0_3 >= 9) {
              float _cvt_f32_f16_220;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_220) : "h"((uint16_t)(acc1[2] >> 16)));
              v_4_3 = _cvt_f32_f16_220;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_4_3);
            float v_5_3 = neg;
            if (c1_3 >= 8) {
              float _cvt_f32_f16_221;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_221)
                  : "h"((uint16_t)(acc1[3] & 65535)));
              v_5_3 = _cvt_f32_f16_221;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_5_3);
            float v_6_3 = neg;
            if (c1_3 >= 9) {
              float _cvt_f32_f16_222;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_222) : "h"((uint16_t)(acc1[3] >> 16)));
              v_6_3 = _cvt_f32_f16_222;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_6_3);
            float v_7_3 = neg;
            if (c0_3 >= 16) {
              float _cvt_f32_f16_223;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_223)
                  : "h"((uint16_t)(acc1[4] & 65535)));
              v_7_3 = _cvt_f32_f16_223;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_7_3);
            float v_8_3 = neg;
            if (c0_3 >= 17) {
              float _cvt_f32_f16_224;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_224) : "h"((uint16_t)(acc1[4] >> 16)));
              v_8_3 = _cvt_f32_f16_224;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_8_3);
            float v_9_3 = neg;
            if (c1_3 >= 16) {
              float _cvt_f32_f16_225;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_225)
                  : "h"((uint16_t)(acc1[5] & 65535)));
              v_9_3 = _cvt_f32_f16_225;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_9_3);
            float v_10_3 = neg;
            if (c1_3 >= 17) {
              float _cvt_f32_f16_226;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_226) : "h"((uint16_t)(acc1[5] >> 16)));
              v_10_3 = _cvt_f32_f16_226;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_10_3);
            float v_11_3 = neg;
            if (c0_3 >= 24) {
              float _cvt_f32_f16_227;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_227)
                  : "h"((uint16_t)(acc1[6] & 65535)));
              v_11_3 = _cvt_f32_f16_227;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_11_3);
            float v_12_3 = neg;
            if (c0_3 >= 25) {
              float _cvt_f32_f16_228;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_228) : "h"((uint16_t)(acc1[6] >> 16)));
              v_12_3 = _cvt_f32_f16_228;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_12_3);
            float v_13_3 = neg;
            if (c1_3 >= 24) {
              float _cvt_f32_f16_229;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_229)
                  : "h"((uint16_t)(acc1[7] & 65535)));
              v_13_3 = _cvt_f32_f16_229;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_13_3);
            float v_14_3 = neg;
            if (c1_3 >= 25) {
              float _cvt_f32_f16_230;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_230) : "h"((uint16_t)(acc1[7] >> 16)));
              v_14_3 = _cvt_f32_f16_230;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_14_3);
            float v_15_3 = neg;
            if (c0_3 >= 32) {
              float _cvt_f32_f16_231;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_231)
                  : "h"((uint16_t)(acc1[8] & 65535)));
              v_15_3 = _cvt_f32_f16_231;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_15_3);
            float v_16_3 = neg;
            if (c0_3 >= 33) {
              float _cvt_f32_f16_232;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_232) : "h"((uint16_t)(acc1[8] >> 16)));
              v_16_3 = _cvt_f32_f16_232;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_16_3);
            float v_17_3 = neg;
            if (c1_3 >= 32) {
              float _cvt_f32_f16_233;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_233)
                  : "h"((uint16_t)(acc1[9] & 65535)));
              v_17_3 = _cvt_f32_f16_233;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_17_3);
            float v_18_3 = neg;
            if (c1_3 >= 33) {
              float _cvt_f32_f16_234;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_234) : "h"((uint16_t)(acc1[9] >> 16)));
              v_18_3 = _cvt_f32_f16_234;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_18_3);
            float v_19_3 = neg;
            if (c0_3 >= 40) {
              float _cvt_f32_f16_235;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_235)
                  : "h"((uint16_t)(acc1[10] & 65535)));
              v_19_3 = _cvt_f32_f16_235;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_19_3);
            float v_20_3 = neg;
            if (c0_3 >= 41) {
              float _cvt_f32_f16_236;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_236) : "h"((uint16_t)(acc1[10] >> 16)));
              v_20_3 = _cvt_f32_f16_236;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_20_3);
            float v_21_3 = neg;
            if (c1_3 >= 40) {
              float _cvt_f32_f16_237;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_237)
                  : "h"((uint16_t)(acc1[11] & 65535)));
              v_21_3 = _cvt_f32_f16_237;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_21_3);
            float v_22_3 = neg;
            if (c1_3 >= 41) {
              float _cvt_f32_f16_238;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_238) : "h"((uint16_t)(acc1[11] >> 16)));
              v_22_3 = _cvt_f32_f16_238;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_22_3);
            float v_23_3 = neg;
            if (c0_3 >= 48) {
              float _cvt_f32_f16_239;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_239)
                  : "h"((uint16_t)(acc1[12] & 65535)));
              v_23_3 = _cvt_f32_f16_239;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_23_3);
            float v_24_3 = neg;
            if (c0_3 >= 49) {
              float _cvt_f32_f16_240;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_240) : "h"((uint16_t)(acc1[12] >> 16)));
              v_24_3 = _cvt_f32_f16_240;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_24_3);
            float v_25_3 = neg;
            if (c1_3 >= 48) {
              float _cvt_f32_f16_241;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_241)
                  : "h"((uint16_t)(acc1[13] & 65535)));
              v_25_3 = _cvt_f32_f16_241;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_25_3);
            float v_26_3 = neg;
            if (c1_3 >= 49) {
              float _cvt_f32_f16_242;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_242) : "h"((uint16_t)(acc1[13] >> 16)));
              v_26_3 = _cvt_f32_f16_242;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_26_3);
            float v_27_3 = neg;
            if (c0_3 >= 56) {
              float _cvt_f32_f16_243;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_243)
                  : "h"((uint16_t)(acc1[14] & 65535)));
              v_27_3 = _cvt_f32_f16_243;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_27_3);
            float v_28_3 = neg;
            if (c0_3 >= 57) {
              float _cvt_f32_f16_244;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_244) : "h"((uint16_t)(acc1[14] >> 16)));
              v_28_3 = _cvt_f32_f16_244;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_28_3);
            float v_29_3 = neg;
            if (c1_3 >= 56) {
              float _cvt_f32_f16_245;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_245)
                  : "h"((uint16_t)(acc1[15] & 65535)));
              v_29_3 = _cvt_f32_f16_245;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_29_3);
            float v_30_3 = neg;
            if (c1_3 >= 57) {
              float _cvt_f32_f16_246;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_246) : "h"((uint16_t)(acc1[15] >> 16)));
              v_30_3 = _cvt_f32_f16_246;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_30_3);
            float v_31_3 = neg;
            if (c0_3 >= 64) {
              float _cvt_f32_f16_247;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_247)
                  : "h"((uint16_t)(acc1[16] & 65535)));
              v_31_3 = _cvt_f32_f16_247;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_31_3);
            float v_32_3 = neg;
            if (c0_3 >= 65) {
              float _cvt_f32_f16_248;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_248) : "h"((uint16_t)(acc1[16] >> 16)));
              v_32_3 = _cvt_f32_f16_248;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_32_3);
            float v_33_3 = neg;
            if (c1_3 >= 64) {
              float _cvt_f32_f16_249;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_249)
                  : "h"((uint16_t)(acc1[17] & 65535)));
              v_33_3 = _cvt_f32_f16_249;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_33_3);
            float v_34_3 = neg;
            if (c1_3 >= 65) {
              float _cvt_f32_f16_250;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_250) : "h"((uint16_t)(acc1[17] >> 16)));
              v_34_3 = _cvt_f32_f16_250;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_34_3);
            float v_35_3 = neg;
            if (c0_3 >= 72) {
              float _cvt_f32_f16_251;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_251)
                  : "h"((uint16_t)(acc1[18] & 65535)));
              v_35_3 = _cvt_f32_f16_251;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_35_3);
            float v_36_3 = neg;
            if (c0_3 >= 73) {
              float _cvt_f32_f16_252;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_252) : "h"((uint16_t)(acc1[18] >> 16)));
              v_36_3 = _cvt_f32_f16_252;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_36_3);
            float v_37_3 = neg;
            if (c1_3 >= 72) {
              float _cvt_f32_f16_253;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_253)
                  : "h"((uint16_t)(acc1[19] & 65535)));
              v_37_3 = _cvt_f32_f16_253;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_37_3);
            float v_38_3 = neg;
            if (c1_3 >= 73) {
              float _cvt_f32_f16_254;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_254) : "h"((uint16_t)(acc1[19] >> 16)));
              v_38_3 = _cvt_f32_f16_254;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_38_3);
            float v_39_3 = neg;
            if (c0_3 >= 80) {
              float _cvt_f32_f16_255;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_255)
                  : "h"((uint16_t)(acc1[20] & 65535)));
              v_39_3 = _cvt_f32_f16_255;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_39_3);
            float v_40_3 = neg;
            if (c0_3 >= 81) {
              float _cvt_f32_f16_256;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_256) : "h"((uint16_t)(acc1[20] >> 16)));
              v_40_3 = _cvt_f32_f16_256;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_40_3);
            float v_41_3 = neg;
            if (c1_3 >= 80) {
              float _cvt_f32_f16_257;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_257)
                  : "h"((uint16_t)(acc1[21] & 65535)));
              v_41_3 = _cvt_f32_f16_257;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_41_3);
            float v_42_3 = neg;
            if (c1_3 >= 81) {
              float _cvt_f32_f16_258;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_258) : "h"((uint16_t)(acc1[21] >> 16)));
              v_42_3 = _cvt_f32_f16_258;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_42_3);
            float v_43_3 = neg;
            if (c0_3 >= 88) {
              float _cvt_f32_f16_259;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_259)
                  : "h"((uint16_t)(acc1[22] & 65535)));
              v_43_3 = _cvt_f32_f16_259;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_43_3);
            float v_44_3 = neg;
            if (c0_3 >= 89) {
              float _cvt_f32_f16_260;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_260) : "h"((uint16_t)(acc1[22] >> 16)));
              v_44_3 = _cvt_f32_f16_260;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_44_3);
            float v_45_3 = neg;
            if (c1_3 >= 88) {
              float _cvt_f32_f16_261;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_261)
                  : "h"((uint16_t)(acc1[23] & 65535)));
              v_45_3 = _cvt_f32_f16_261;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_45_3);
            float v_46_3 = neg;
            if (c1_3 >= 89) {
              float _cvt_f32_f16_262;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_262) : "h"((uint16_t)(acc1[23] >> 16)));
              v_46_3 = _cvt_f32_f16_262;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_46_3);
            float v_47_3 = neg;
            if (c0_3 >= 96) {
              float _cvt_f32_f16_263;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_263)
                  : "h"((uint16_t)(acc1[24] & 65535)));
              v_47_3 = _cvt_f32_f16_263;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_47_3);
            float v_48_3 = neg;
            if (c0_3 >= 97) {
              float _cvt_f32_f16_264;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_264) : "h"((uint16_t)(acc1[24] >> 16)));
              v_48_3 = _cvt_f32_f16_264;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_48_3);
            float v_49_3 = neg;
            if (c1_3 >= 96) {
              float _cvt_f32_f16_265;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_265)
                  : "h"((uint16_t)(acc1[25] & 65535)));
              v_49_3 = _cvt_f32_f16_265;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_49_3);
            float v_50_3 = neg;
            if (c1_3 >= 97) {
              float _cvt_f32_f16_266;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_266) : "h"((uint16_t)(acc1[25] >> 16)));
              v_50_3 = _cvt_f32_f16_266;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_50_3);
            float v_51_3 = neg;
            if (c0_3 >= 104) {
              float _cvt_f32_f16_267;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_267)
                  : "h"((uint16_t)(acc1[26] & 65535)));
              v_51_3 = _cvt_f32_f16_267;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_51_3);
            float v_52_3 = neg;
            if (c0_3 >= 105) {
              float _cvt_f32_f16_268;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_268) : "h"((uint16_t)(acc1[26] >> 16)));
              v_52_3 = _cvt_f32_f16_268;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_52_3);
            float v_53_3 = neg;
            if (c1_3 >= 104) {
              float _cvt_f32_f16_269;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_269)
                  : "h"((uint16_t)(acc1[27] & 65535)));
              v_53_3 = _cvt_f32_f16_269;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_53_3);
            float v_54_3 = neg;
            if (c1_3 >= 105) {
              float _cvt_f32_f16_270;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_270) : "h"((uint16_t)(acc1[27] >> 16)));
              v_54_3 = _cvt_f32_f16_270;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_54_3);
            float v_55_3 = neg;
            if (c0_3 >= 112) {
              float _cvt_f32_f16_271;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_271)
                  : "h"((uint16_t)(acc1[28] & 65535)));
              v_55_3 = _cvt_f32_f16_271;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_55_3);
            float v_56_3 = neg;
            if (c0_3 >= 113) {
              float _cvt_f32_f16_272;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_272) : "h"((uint16_t)(acc1[28] >> 16)));
              v_56_3 = _cvt_f32_f16_272;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_56_3);
            float v_57_3 = neg;
            if (c1_3 >= 112) {
              float _cvt_f32_f16_273;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_273)
                  : "h"((uint16_t)(acc1[29] & 65535)));
              v_57_3 = _cvt_f32_f16_273;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_57_3);
            float v_58_3 = neg;
            if (c1_3 >= 113) {
              float _cvt_f32_f16_274;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_274) : "h"((uint16_t)(acc1[29] >> 16)));
              v_58_3 = _cvt_f32_f16_274;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_58_3);
            float v_59_3 = neg;
            if (c0_3 >= 120) {
              float _cvt_f32_f16_275;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_275)
                  : "h"((uint16_t)(acc1[30] & 65535)));
              v_59_3 = _cvt_f32_f16_275;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_59_3);
            float v_60_3 = neg;
            if (c0_3 >= 121) {
              float _cvt_f32_f16_276;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_276) : "h"((uint16_t)(acc1[30] >> 16)));
              v_60_3 = _cvt_f32_f16_276;
            }
            m0v_1_1 = fmaxf(m0v_1_1, v_60_3);
            float v_61_3 = neg;
            if (c1_3 >= 120) {
              float _cvt_f32_f16_277;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_277)
                  : "h"((uint16_t)(acc1[31] & 65535)));
              v_61_3 = _cvt_f32_f16_277;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_61_3);
            float v_62_3 = neg;
            if (c1_3 >= 121) {
              float _cvt_f32_f16_278;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_278) : "h"((uint16_t)(acc1[31] >> 16)));
              v_62_3 = _cvt_f32_f16_278;
            }
            m1v_2_1 = fmaxf(m1v_2_1, v_62_3);
          } else {
            unsigned int w0_3 = acc1[0];
            unsigned int w1_3 = acc1[1];
            uint32_t _f16x2_max_90;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_90) : "r"(w0_3), "r"(acc1[2]));
            w0_3 = _f16x2_max_90;
            uint32_t _f16x2_max_91;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_91) : "r"(w1_3), "r"(acc1[3]));
            w1_3 = _f16x2_max_91;
            uint32_t _f16x2_max_92;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_92) : "r"(w0_3), "r"(acc1[4]));
            w0_3 = _f16x2_max_92;
            uint32_t _f16x2_max_93;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_93) : "r"(w1_3), "r"(acc1[5]));
            w1_3 = _f16x2_max_93;
            uint32_t _f16x2_max_94;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_94) : "r"(w0_3), "r"(acc1[6]));
            w0_3 = _f16x2_max_94;
            uint32_t _f16x2_max_95;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_95) : "r"(w1_3), "r"(acc1[7]));
            w1_3 = _f16x2_max_95;
            uint32_t _f16x2_max_96;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_96) : "r"(w0_3), "r"(acc1[8]));
            w0_3 = _f16x2_max_96;
            uint32_t _f16x2_max_97;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_97) : "r"(w1_3), "r"(acc1[9]));
            w1_3 = _f16x2_max_97;
            uint32_t _f16x2_max_98;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_98) : "r"(w0_3), "r"(acc1[10]));
            w0_3 = _f16x2_max_98;
            uint32_t _f16x2_max_99;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_99) : "r"(w1_3), "r"(acc1[11]));
            w1_3 = _f16x2_max_99;
            uint32_t _f16x2_max_100;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_100) : "r"(w0_3), "r"(acc1[12]));
            w0_3 = _f16x2_max_100;
            uint32_t _f16x2_max_101;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_101) : "r"(w1_3), "r"(acc1[13]));
            w1_3 = _f16x2_max_101;
            uint32_t _f16x2_max_102;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_102) : "r"(w0_3), "r"(acc1[14]));
            w0_3 = _f16x2_max_102;
            uint32_t _f16x2_max_103;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_103) : "r"(w1_3), "r"(acc1[15]));
            w1_3 = _f16x2_max_103;
            uint32_t _f16x2_max_104;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_104) : "r"(w0_3), "r"(acc1[16]));
            w0_3 = _f16x2_max_104;
            uint32_t _f16x2_max_105;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_105) : "r"(w1_3), "r"(acc1[17]));
            w1_3 = _f16x2_max_105;
            uint32_t _f16x2_max_106;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_106) : "r"(w0_3), "r"(acc1[18]));
            w0_3 = _f16x2_max_106;
            uint32_t _f16x2_max_107;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_107) : "r"(w1_3), "r"(acc1[19]));
            w1_3 = _f16x2_max_107;
            uint32_t _f16x2_max_108;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_108) : "r"(w0_3), "r"(acc1[20]));
            w0_3 = _f16x2_max_108;
            uint32_t _f16x2_max_109;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_109) : "r"(w1_3), "r"(acc1[21]));
            w1_3 = _f16x2_max_109;
            uint32_t _f16x2_max_110;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_110) : "r"(w0_3), "r"(acc1[22]));
            w0_3 = _f16x2_max_110;
            uint32_t _f16x2_max_111;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_111) : "r"(w1_3), "r"(acc1[23]));
            w1_3 = _f16x2_max_111;
            uint32_t _f16x2_max_112;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_112) : "r"(w0_3), "r"(acc1[24]));
            w0_3 = _f16x2_max_112;
            uint32_t _f16x2_max_113;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_113) : "r"(w1_3), "r"(acc1[25]));
            w1_3 = _f16x2_max_113;
            uint32_t _f16x2_max_114;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_114) : "r"(w0_3), "r"(acc1[26]));
            w0_3 = _f16x2_max_114;
            uint32_t _f16x2_max_115;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_115) : "r"(w1_3), "r"(acc1[27]));
            w1_3 = _f16x2_max_115;
            uint32_t _f16x2_max_116;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_116) : "r"(w0_3), "r"(acc1[28]));
            w0_3 = _f16x2_max_116;
            uint32_t _f16x2_max_117;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_117) : "r"(w1_3), "r"(acc1[29]));
            w1_3 = _f16x2_max_117;
            uint32_t _f16x2_max_118;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_118) : "r"(w0_3), "r"(acc1[30]));
            w0_3 = _f16x2_max_118;
            uint32_t _f16x2_max_119;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_119) : "r"(w1_3), "r"(acc1[31]));
            w1_3 = _f16x2_max_119;
            uint16_t _f16_max_6;
            asm("max.f16 %0, %1, %2;"
                : "=h"(_f16_max_6)
                : "h"((uint16_t)(w0_3 & 65535)), "h"((uint16_t)(w0_3 >> 16)));
            float _cvt_f32_f16_279;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_279) : "h"((uint16_t)(_f16_max_6)));
            m0v_1_1 = _cvt_f32_f16_279;
            uint16_t _f16_max_7;
            asm("max.f16 %0, %1, %2;"
                : "=h"(_f16_max_7)
                : "h"((uint16_t)(w1_3 & 65535)), "h"((uint16_t)(w1_3 >> 16)));
            float _cvt_f32_f16_280;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_280) : "h"((uint16_t)(_f16_max_7)));
            m1v_2_1 = _cvt_f32_f16_280;
          }
          float _shfl_xor_sync_281 = __shfl_xor_sync(0xFFFFFFFF, m0v_1_1, 1);
          m0v_1_1 = fmaxf(m0v_1_1, _shfl_xor_sync_281);
          float _shfl_xor_sync_282 = __shfl_xor_sync(0xFFFFFFFF, m0v_1_1, 2);
          m0v_1_1 = fmaxf(m0v_1_1, _shfl_xor_sync_282);
          float _shfl_xor_sync_283 = __shfl_xor_sync(0xFFFFFFFF, m1v_2_1, 1);
          m1v_2_1 = fmaxf(m1v_2_1, _shfl_xor_sync_283);
          float _shfl_xor_sync_284 = __shfl_xor_sync(0xFFFFFFFF, m1v_2_1, 2);
          m1v_2_1 = fmaxf(m1v_2_1, _shfl_xor_sync_284);
          if (cq == 0) {
            int orow_3 = obase + (split_1 + im * nsplit) * total_q + r0;
            if (r0 < rows_valid) {
              out[orow_3] = m0v_1_1;
            }
            if (rows_valid > r0 + 8) {
              out[orow_3 + 8] = m1v_2_1;
            }
          }
        } else {
          asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
          if (elect_sync()) {
            mbarrier_arrive(empty_addr + (sl) * 8);
          }
          int k0_2 = (split_1 + il * nsplit) * 128;
          float m0v_3 = neg;
          float m1v_3 = neg;
          if (mask_thr < k0_2 + 127) {
            int c0_4 = lim0 - k0_2 - 2 * cq;
            int c1_4 = lim1 - k0_2 - 2 * cq;
            float v_66 = neg;
            if (c0_4 >= 0) {
              float _cvt_f32_f16_285;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_285) : "h"((uint16_t)(acc[0] & 65535)));
              v_66 = _cvt_f32_f16_285;
            }
            m0v_3 = fmaxf(m0v_3, v_66);
            float v_0_4 = neg;
            if (c0_4 >= 1) {
              float _cvt_f32_f16_286;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_286) : "h"((uint16_t)(acc[0] >> 16)));
              v_0_4 = _cvt_f32_f16_286;
            }
            m0v_3 = fmaxf(m0v_3, v_0_4);
            float v_1_4 = neg;
            if (c1_4 >= 0) {
              float _cvt_f32_f16_287;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_287) : "h"((uint16_t)(acc[1] & 65535)));
              v_1_4 = _cvt_f32_f16_287;
            }
            m1v_3 = fmaxf(m1v_3, v_1_4);
            float v_2_4 = neg;
            if (c1_4 >= 1) {
              float _cvt_f32_f16_288;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_288) : "h"((uint16_t)(acc[1] >> 16)));
              v_2_4 = _cvt_f32_f16_288;
            }
            m1v_3 = fmaxf(m1v_3, v_2_4);
            float v_3_4 = neg;
            if (c0_4 >= 8) {
              float _cvt_f32_f16_289;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_289) : "h"((uint16_t)(acc[2] & 65535)));
              v_3_4 = _cvt_f32_f16_289;
            }
            m0v_3 = fmaxf(m0v_3, v_3_4);
            float v_4_4 = neg;
            if (c0_4 >= 9) {
              float _cvt_f32_f16_290;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_290) : "h"((uint16_t)(acc[2] >> 16)));
              v_4_4 = _cvt_f32_f16_290;
            }
            m0v_3 = fmaxf(m0v_3, v_4_4);
            float v_5_4 = neg;
            if (c1_4 >= 8) {
              float _cvt_f32_f16_291;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_291) : "h"((uint16_t)(acc[3] & 65535)));
              v_5_4 = _cvt_f32_f16_291;
            }
            m1v_3 = fmaxf(m1v_3, v_5_4);
            float v_6_4 = neg;
            if (c1_4 >= 9) {
              float _cvt_f32_f16_292;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_292) : "h"((uint16_t)(acc[3] >> 16)));
              v_6_4 = _cvt_f32_f16_292;
            }
            m1v_3 = fmaxf(m1v_3, v_6_4);
            float v_7_4 = neg;
            if (c0_4 >= 16) {
              float _cvt_f32_f16_293;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_293) : "h"((uint16_t)(acc[4] & 65535)));
              v_7_4 = _cvt_f32_f16_293;
            }
            m0v_3 = fmaxf(m0v_3, v_7_4);
            float v_8_4 = neg;
            if (c0_4 >= 17) {
              float _cvt_f32_f16_294;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_294) : "h"((uint16_t)(acc[4] >> 16)));
              v_8_4 = _cvt_f32_f16_294;
            }
            m0v_3 = fmaxf(m0v_3, v_8_4);
            float v_9_4 = neg;
            if (c1_4 >= 16) {
              float _cvt_f32_f16_295;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_295) : "h"((uint16_t)(acc[5] & 65535)));
              v_9_4 = _cvt_f32_f16_295;
            }
            m1v_3 = fmaxf(m1v_3, v_9_4);
            float v_10_4 = neg;
            if (c1_4 >= 17) {
              float _cvt_f32_f16_296;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_296) : "h"((uint16_t)(acc[5] >> 16)));
              v_10_4 = _cvt_f32_f16_296;
            }
            m1v_3 = fmaxf(m1v_3, v_10_4);
            float v_11_4 = neg;
            if (c0_4 >= 24) {
              float _cvt_f32_f16_297;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_297) : "h"((uint16_t)(acc[6] & 65535)));
              v_11_4 = _cvt_f32_f16_297;
            }
            m0v_3 = fmaxf(m0v_3, v_11_4);
            float v_12_4 = neg;
            if (c0_4 >= 25) {
              float _cvt_f32_f16_298;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_298) : "h"((uint16_t)(acc[6] >> 16)));
              v_12_4 = _cvt_f32_f16_298;
            }
            m0v_3 = fmaxf(m0v_3, v_12_4);
            float v_13_4 = neg;
            if (c1_4 >= 24) {
              float _cvt_f32_f16_299;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_299) : "h"((uint16_t)(acc[7] & 65535)));
              v_13_4 = _cvt_f32_f16_299;
            }
            m1v_3 = fmaxf(m1v_3, v_13_4);
            float v_14_4 = neg;
            if (c1_4 >= 25) {
              float _cvt_f32_f16_300;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_300) : "h"((uint16_t)(acc[7] >> 16)));
              v_14_4 = _cvt_f32_f16_300;
            }
            m1v_3 = fmaxf(m1v_3, v_14_4);
            float v_15_4 = neg;
            if (c0_4 >= 32) {
              float _cvt_f32_f16_301;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_301) : "h"((uint16_t)(acc[8] & 65535)));
              v_15_4 = _cvt_f32_f16_301;
            }
            m0v_3 = fmaxf(m0v_3, v_15_4);
            float v_16_4 = neg;
            if (c0_4 >= 33) {
              float _cvt_f32_f16_302;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_302) : "h"((uint16_t)(acc[8] >> 16)));
              v_16_4 = _cvt_f32_f16_302;
            }
            m0v_3 = fmaxf(m0v_3, v_16_4);
            float v_17_4 = neg;
            if (c1_4 >= 32) {
              float _cvt_f32_f16_303;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_303) : "h"((uint16_t)(acc[9] & 65535)));
              v_17_4 = _cvt_f32_f16_303;
            }
            m1v_3 = fmaxf(m1v_3, v_17_4);
            float v_18_4 = neg;
            if (c1_4 >= 33) {
              float _cvt_f32_f16_304;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_304) : "h"((uint16_t)(acc[9] >> 16)));
              v_18_4 = _cvt_f32_f16_304;
            }
            m1v_3 = fmaxf(m1v_3, v_18_4);
            float v_19_4 = neg;
            if (c0_4 >= 40) {
              float _cvt_f32_f16_305;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_305)
                  : "h"((uint16_t)(acc[10] & 65535)));
              v_19_4 = _cvt_f32_f16_305;
            }
            m0v_3 = fmaxf(m0v_3, v_19_4);
            float v_20_4 = neg;
            if (c0_4 >= 41) {
              float _cvt_f32_f16_306;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_306) : "h"((uint16_t)(acc[10] >> 16)));
              v_20_4 = _cvt_f32_f16_306;
            }
            m0v_3 = fmaxf(m0v_3, v_20_4);
            float v_21_4 = neg;
            if (c1_4 >= 40) {
              float _cvt_f32_f16_307;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_307)
                  : "h"((uint16_t)(acc[11] & 65535)));
              v_21_4 = _cvt_f32_f16_307;
            }
            m1v_3 = fmaxf(m1v_3, v_21_4);
            float v_22_4 = neg;
            if (c1_4 >= 41) {
              float _cvt_f32_f16_308;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_308) : "h"((uint16_t)(acc[11] >> 16)));
              v_22_4 = _cvt_f32_f16_308;
            }
            m1v_3 = fmaxf(m1v_3, v_22_4);
            float v_23_4 = neg;
            if (c0_4 >= 48) {
              float _cvt_f32_f16_309;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_309)
                  : "h"((uint16_t)(acc[12] & 65535)));
              v_23_4 = _cvt_f32_f16_309;
            }
            m0v_3 = fmaxf(m0v_3, v_23_4);
            float v_24_4 = neg;
            if (c0_4 >= 49) {
              float _cvt_f32_f16_310;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_310) : "h"((uint16_t)(acc[12] >> 16)));
              v_24_4 = _cvt_f32_f16_310;
            }
            m0v_3 = fmaxf(m0v_3, v_24_4);
            float v_25_4 = neg;
            if (c1_4 >= 48) {
              float _cvt_f32_f16_311;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_311)
                  : "h"((uint16_t)(acc[13] & 65535)));
              v_25_4 = _cvt_f32_f16_311;
            }
            m1v_3 = fmaxf(m1v_3, v_25_4);
            float v_26_4 = neg;
            if (c1_4 >= 49) {
              float _cvt_f32_f16_312;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_312) : "h"((uint16_t)(acc[13] >> 16)));
              v_26_4 = _cvt_f32_f16_312;
            }
            m1v_3 = fmaxf(m1v_3, v_26_4);
            float v_27_4 = neg;
            if (c0_4 >= 56) {
              float _cvt_f32_f16_313;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_313)
                  : "h"((uint16_t)(acc[14] & 65535)));
              v_27_4 = _cvt_f32_f16_313;
            }
            m0v_3 = fmaxf(m0v_3, v_27_4);
            float v_28_4 = neg;
            if (c0_4 >= 57) {
              float _cvt_f32_f16_314;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_314) : "h"((uint16_t)(acc[14] >> 16)));
              v_28_4 = _cvt_f32_f16_314;
            }
            m0v_3 = fmaxf(m0v_3, v_28_4);
            float v_29_4 = neg;
            if (c1_4 >= 56) {
              float _cvt_f32_f16_315;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_315)
                  : "h"((uint16_t)(acc[15] & 65535)));
              v_29_4 = _cvt_f32_f16_315;
            }
            m1v_3 = fmaxf(m1v_3, v_29_4);
            float v_30_4 = neg;
            if (c1_4 >= 57) {
              float _cvt_f32_f16_316;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_316) : "h"((uint16_t)(acc[15] >> 16)));
              v_30_4 = _cvt_f32_f16_316;
            }
            m1v_3 = fmaxf(m1v_3, v_30_4);
            float v_31_4 = neg;
            if (c0_4 >= 64) {
              float _cvt_f32_f16_317;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_317)
                  : "h"((uint16_t)(acc[16] & 65535)));
              v_31_4 = _cvt_f32_f16_317;
            }
            m0v_3 = fmaxf(m0v_3, v_31_4);
            float v_32_4 = neg;
            if (c0_4 >= 65) {
              float _cvt_f32_f16_318;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_318) : "h"((uint16_t)(acc[16] >> 16)));
              v_32_4 = _cvt_f32_f16_318;
            }
            m0v_3 = fmaxf(m0v_3, v_32_4);
            float v_33_4 = neg;
            if (c1_4 >= 64) {
              float _cvt_f32_f16_319;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_319)
                  : "h"((uint16_t)(acc[17] & 65535)));
              v_33_4 = _cvt_f32_f16_319;
            }
            m1v_3 = fmaxf(m1v_3, v_33_4);
            float v_34_4 = neg;
            if (c1_4 >= 65) {
              float _cvt_f32_f16_320;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_320) : "h"((uint16_t)(acc[17] >> 16)));
              v_34_4 = _cvt_f32_f16_320;
            }
            m1v_3 = fmaxf(m1v_3, v_34_4);
            float v_35_4 = neg;
            if (c0_4 >= 72) {
              float _cvt_f32_f16_321;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_321)
                  : "h"((uint16_t)(acc[18] & 65535)));
              v_35_4 = _cvt_f32_f16_321;
            }
            m0v_3 = fmaxf(m0v_3, v_35_4);
            float v_36_4 = neg;
            if (c0_4 >= 73) {
              float _cvt_f32_f16_322;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_322) : "h"((uint16_t)(acc[18] >> 16)));
              v_36_4 = _cvt_f32_f16_322;
            }
            m0v_3 = fmaxf(m0v_3, v_36_4);
            float v_37_4 = neg;
            if (c1_4 >= 72) {
              float _cvt_f32_f16_323;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_323)
                  : "h"((uint16_t)(acc[19] & 65535)));
              v_37_4 = _cvt_f32_f16_323;
            }
            m1v_3 = fmaxf(m1v_3, v_37_4);
            float v_38_4 = neg;
            if (c1_4 >= 73) {
              float _cvt_f32_f16_324;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_324) : "h"((uint16_t)(acc[19] >> 16)));
              v_38_4 = _cvt_f32_f16_324;
            }
            m1v_3 = fmaxf(m1v_3, v_38_4);
            float v_39_4 = neg;
            if (c0_4 >= 80) {
              float _cvt_f32_f16_325;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_325)
                  : "h"((uint16_t)(acc[20] & 65535)));
              v_39_4 = _cvt_f32_f16_325;
            }
            m0v_3 = fmaxf(m0v_3, v_39_4);
            float v_40_4 = neg;
            if (c0_4 >= 81) {
              float _cvt_f32_f16_326;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_326) : "h"((uint16_t)(acc[20] >> 16)));
              v_40_4 = _cvt_f32_f16_326;
            }
            m0v_3 = fmaxf(m0v_3, v_40_4);
            float v_41_4 = neg;
            if (c1_4 >= 80) {
              float _cvt_f32_f16_327;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_327)
                  : "h"((uint16_t)(acc[21] & 65535)));
              v_41_4 = _cvt_f32_f16_327;
            }
            m1v_3 = fmaxf(m1v_3, v_41_4);
            float v_42_4 = neg;
            if (c1_4 >= 81) {
              float _cvt_f32_f16_328;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_328) : "h"((uint16_t)(acc[21] >> 16)));
              v_42_4 = _cvt_f32_f16_328;
            }
            m1v_3 = fmaxf(m1v_3, v_42_4);
            float v_43_4 = neg;
            if (c0_4 >= 88) {
              float _cvt_f32_f16_329;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_329)
                  : "h"((uint16_t)(acc[22] & 65535)));
              v_43_4 = _cvt_f32_f16_329;
            }
            m0v_3 = fmaxf(m0v_3, v_43_4);
            float v_44_4 = neg;
            if (c0_4 >= 89) {
              float _cvt_f32_f16_330;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_330) : "h"((uint16_t)(acc[22] >> 16)));
              v_44_4 = _cvt_f32_f16_330;
            }
            m0v_3 = fmaxf(m0v_3, v_44_4);
            float v_45_4 = neg;
            if (c1_4 >= 88) {
              float _cvt_f32_f16_331;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_331)
                  : "h"((uint16_t)(acc[23] & 65535)));
              v_45_4 = _cvt_f32_f16_331;
            }
            m1v_3 = fmaxf(m1v_3, v_45_4);
            float v_46_4 = neg;
            if (c1_4 >= 89) {
              float _cvt_f32_f16_332;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_332) : "h"((uint16_t)(acc[23] >> 16)));
              v_46_4 = _cvt_f32_f16_332;
            }
            m1v_3 = fmaxf(m1v_3, v_46_4);
            float v_47_4 = neg;
            if (c0_4 >= 96) {
              float _cvt_f32_f16_333;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_333)
                  : "h"((uint16_t)(acc[24] & 65535)));
              v_47_4 = _cvt_f32_f16_333;
            }
            m0v_3 = fmaxf(m0v_3, v_47_4);
            float v_48_4 = neg;
            if (c0_4 >= 97) {
              float _cvt_f32_f16_334;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_334) : "h"((uint16_t)(acc[24] >> 16)));
              v_48_4 = _cvt_f32_f16_334;
            }
            m0v_3 = fmaxf(m0v_3, v_48_4);
            float v_49_4 = neg;
            if (c1_4 >= 96) {
              float _cvt_f32_f16_335;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_335)
                  : "h"((uint16_t)(acc[25] & 65535)));
              v_49_4 = _cvt_f32_f16_335;
            }
            m1v_3 = fmaxf(m1v_3, v_49_4);
            float v_50_4 = neg;
            if (c1_4 >= 97) {
              float _cvt_f32_f16_336;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_336) : "h"((uint16_t)(acc[25] >> 16)));
              v_50_4 = _cvt_f32_f16_336;
            }
            m1v_3 = fmaxf(m1v_3, v_50_4);
            float v_51_4 = neg;
            if (c0_4 >= 104) {
              float _cvt_f32_f16_337;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_337)
                  : "h"((uint16_t)(acc[26] & 65535)));
              v_51_4 = _cvt_f32_f16_337;
            }
            m0v_3 = fmaxf(m0v_3, v_51_4);
            float v_52_4 = neg;
            if (c0_4 >= 105) {
              float _cvt_f32_f16_338;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_338) : "h"((uint16_t)(acc[26] >> 16)));
              v_52_4 = _cvt_f32_f16_338;
            }
            m0v_3 = fmaxf(m0v_3, v_52_4);
            float v_53_4 = neg;
            if (c1_4 >= 104) {
              float _cvt_f32_f16_339;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_339)
                  : "h"((uint16_t)(acc[27] & 65535)));
              v_53_4 = _cvt_f32_f16_339;
            }
            m1v_3 = fmaxf(m1v_3, v_53_4);
            float v_54_4 = neg;
            if (c1_4 >= 105) {
              float _cvt_f32_f16_340;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_340) : "h"((uint16_t)(acc[27] >> 16)));
              v_54_4 = _cvt_f32_f16_340;
            }
            m1v_3 = fmaxf(m1v_3, v_54_4);
            float v_55_4 = neg;
            if (c0_4 >= 112) {
              float _cvt_f32_f16_341;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_341)
                  : "h"((uint16_t)(acc[28] & 65535)));
              v_55_4 = _cvt_f32_f16_341;
            }
            m0v_3 = fmaxf(m0v_3, v_55_4);
            float v_56_4 = neg;
            if (c0_4 >= 113) {
              float _cvt_f32_f16_342;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_342) : "h"((uint16_t)(acc[28] >> 16)));
              v_56_4 = _cvt_f32_f16_342;
            }
            m0v_3 = fmaxf(m0v_3, v_56_4);
            float v_57_4 = neg;
            if (c1_4 >= 112) {
              float _cvt_f32_f16_343;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_343)
                  : "h"((uint16_t)(acc[29] & 65535)));
              v_57_4 = _cvt_f32_f16_343;
            }
            m1v_3 = fmaxf(m1v_3, v_57_4);
            float v_58_4 = neg;
            if (c1_4 >= 113) {
              float _cvt_f32_f16_344;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_344) : "h"((uint16_t)(acc[29] >> 16)));
              v_58_4 = _cvt_f32_f16_344;
            }
            m1v_3 = fmaxf(m1v_3, v_58_4);
            float v_59_4 = neg;
            if (c0_4 >= 120) {
              float _cvt_f32_f16_345;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_345)
                  : "h"((uint16_t)(acc[30] & 65535)));
              v_59_4 = _cvt_f32_f16_345;
            }
            m0v_3 = fmaxf(m0v_3, v_59_4);
            float v_60_4 = neg;
            if (c0_4 >= 121) {
              float _cvt_f32_f16_346;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_346) : "h"((uint16_t)(acc[30] >> 16)));
              v_60_4 = _cvt_f32_f16_346;
            }
            m0v_3 = fmaxf(m0v_3, v_60_4);
            float v_61_4 = neg;
            if (c1_4 >= 120) {
              float _cvt_f32_f16_347;
              asm("cvt.f32.f16 %0, %1;"
                  : "=f"(_cvt_f32_f16_347)
                  : "h"((uint16_t)(acc[31] & 65535)));
              v_61_4 = _cvt_f32_f16_347;
            }
            m1v_3 = fmaxf(m1v_3, v_61_4);
            float v_62_4 = neg;
            if (c1_4 >= 121) {
              float _cvt_f32_f16_348;
              asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_348) : "h"((uint16_t)(acc[31] >> 16)));
              v_62_4 = _cvt_f32_f16_348;
            }
            m1v_3 = fmaxf(m1v_3, v_62_4);
          } else {
            unsigned int w0_4 = acc[0];
            unsigned int w1_4 = acc[1];
            uint32_t _f16x2_max_120;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_120) : "r"(w0_4), "r"(acc[2]));
            w0_4 = _f16x2_max_120;
            uint32_t _f16x2_max_121;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_121) : "r"(w1_4), "r"(acc[3]));
            w1_4 = _f16x2_max_121;
            uint32_t _f16x2_max_122;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_122) : "r"(w0_4), "r"(acc[4]));
            w0_4 = _f16x2_max_122;
            uint32_t _f16x2_max_123;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_123) : "r"(w1_4), "r"(acc[5]));
            w1_4 = _f16x2_max_123;
            uint32_t _f16x2_max_124;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_124) : "r"(w0_4), "r"(acc[6]));
            w0_4 = _f16x2_max_124;
            uint32_t _f16x2_max_125;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_125) : "r"(w1_4), "r"(acc[7]));
            w1_4 = _f16x2_max_125;
            uint32_t _f16x2_max_126;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_126) : "r"(w0_4), "r"(acc[8]));
            w0_4 = _f16x2_max_126;
            uint32_t _f16x2_max_127;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_127) : "r"(w1_4), "r"(acc[9]));
            w1_4 = _f16x2_max_127;
            uint32_t _f16x2_max_128;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_128) : "r"(w0_4), "r"(acc[10]));
            w0_4 = _f16x2_max_128;
            uint32_t _f16x2_max_129;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_129) : "r"(w1_4), "r"(acc[11]));
            w1_4 = _f16x2_max_129;
            uint32_t _f16x2_max_130;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_130) : "r"(w0_4), "r"(acc[12]));
            w0_4 = _f16x2_max_130;
            uint32_t _f16x2_max_131;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_131) : "r"(w1_4), "r"(acc[13]));
            w1_4 = _f16x2_max_131;
            uint32_t _f16x2_max_132;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_132) : "r"(w0_4), "r"(acc[14]));
            w0_4 = _f16x2_max_132;
            uint32_t _f16x2_max_133;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_133) : "r"(w1_4), "r"(acc[15]));
            w1_4 = _f16x2_max_133;
            uint32_t _f16x2_max_134;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_134) : "r"(w0_4), "r"(acc[16]));
            w0_4 = _f16x2_max_134;
            uint32_t _f16x2_max_135;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_135) : "r"(w1_4), "r"(acc[17]));
            w1_4 = _f16x2_max_135;
            uint32_t _f16x2_max_136;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_136) : "r"(w0_4), "r"(acc[18]));
            w0_4 = _f16x2_max_136;
            uint32_t _f16x2_max_137;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_137) : "r"(w1_4), "r"(acc[19]));
            w1_4 = _f16x2_max_137;
            uint32_t _f16x2_max_138;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_138) : "r"(w0_4), "r"(acc[20]));
            w0_4 = _f16x2_max_138;
            uint32_t _f16x2_max_139;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_139) : "r"(w1_4), "r"(acc[21]));
            w1_4 = _f16x2_max_139;
            uint32_t _f16x2_max_140;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_140) : "r"(w0_4), "r"(acc[22]));
            w0_4 = _f16x2_max_140;
            uint32_t _f16x2_max_141;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_141) : "r"(w1_4), "r"(acc[23]));
            w1_4 = _f16x2_max_141;
            uint32_t _f16x2_max_142;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_142) : "r"(w0_4), "r"(acc[24]));
            w0_4 = _f16x2_max_142;
            uint32_t _f16x2_max_143;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_143) : "r"(w1_4), "r"(acc[25]));
            w1_4 = _f16x2_max_143;
            uint32_t _f16x2_max_144;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_144) : "r"(w0_4), "r"(acc[26]));
            w0_4 = _f16x2_max_144;
            uint32_t _f16x2_max_145;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_145) : "r"(w1_4), "r"(acc[27]));
            w1_4 = _f16x2_max_145;
            uint32_t _f16x2_max_146;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_146) : "r"(w0_4), "r"(acc[28]));
            w0_4 = _f16x2_max_146;
            uint32_t _f16x2_max_147;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_147) : "r"(w1_4), "r"(acc[29]));
            w1_4 = _f16x2_max_147;
            uint32_t _f16x2_max_148;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_148) : "r"(w0_4), "r"(acc[30]));
            w0_4 = _f16x2_max_148;
            uint32_t _f16x2_max_149;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_149) : "r"(w1_4), "r"(acc[31]));
            w1_4 = _f16x2_max_149;
            uint16_t _f16_max_8;
            asm("max.f16 %0, %1, %2;"
                : "=h"(_f16_max_8)
                : "h"((uint16_t)(w0_4 & 65535)), "h"((uint16_t)(w0_4 >> 16)));
            float _cvt_f32_f16_349;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_349) : "h"((uint16_t)(_f16_max_8)));
            m0v_3 = _cvt_f32_f16_349;
            uint16_t _f16_max_9;
            asm("max.f16 %0, %1, %2;"
                : "=h"(_f16_max_9)
                : "h"((uint16_t)(w1_4 & 65535)), "h"((uint16_t)(w1_4 >> 16)));
            float _cvt_f32_f16_350;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_350) : "h"((uint16_t)(_f16_max_9)));
            m1v_3 = _cvt_f32_f16_350;
          }
          float _shfl_xor_sync_351 = __shfl_xor_sync(0xFFFFFFFF, m0v_3, 1);
          m0v_3 = fmaxf(m0v_3, _shfl_xor_sync_351);
          float _shfl_xor_sync_352 = __shfl_xor_sync(0xFFFFFFFF, m0v_3, 2);
          m0v_3 = fmaxf(m0v_3, _shfl_xor_sync_352);
          float _shfl_xor_sync_353 = __shfl_xor_sync(0xFFFFFFFF, m1v_3, 1);
          m1v_3 = fmaxf(m1v_3, _shfl_xor_sync_353);
          float _shfl_xor_sync_354 = __shfl_xor_sync(0xFFFFFFFF, m1v_3, 2);
          m1v_3 = fmaxf(m1v_3, _shfl_xor_sync_354);
          if (cq == 0) {
            int orow_4 = obase + (split_1 + il * nsplit) * total_q + r0;
            if (r0 < rows_valid) {
              out[orow_4] = m0v_3;
            }
            if (rows_valid > r0 + 8) {
              out[orow_4 + 8] = m1v_3;
            }
          }
        }
      }
      float negs[4];
      negs[0] = neg;
      negs[1] = neg;
      negs[2] = neg;
      negs[3] = neg;
      int rv = rows_valid;
      if (rv > 256) {
        rv = 256;
      }
      int tq = (tid_1 & 31) * 4;
      int tw = tid_1 >> 5;
      int n_dead_it = (n_dead + 16 - 1) / 16;
#pragma unroll 1
      for (int d = 0; d < n_dead_it; d++) {
        int di = 16 * d + tw;
        if (di < n_dead) {
          int td = t_lim_1 + split_1 + di * nsplit;
          int ob = obase + td * total_q;
          int al = ob & 3;
          int tqh = tq;
          if (al == 0) {
            if (rv > tqh + 3) {
              {
                float4 _v4 = make_float4(negs[0 + 0], negs[0 + 1], negs[0 + 2], negs[0 + 3]);
                *reinterpret_cast<float4*>(out + ob + tqh) = _v4;
              }
            } else {
              if (rv > tqh) {
                out[ob + tqh] = neg;
              }
              if (rv > tqh + 1) {
                out[ob + tqh + 1] = neg;
              }
              if (rv > tqh + 2) {
                out[ob + tqh + 2] = neg;
              }
              if (rv > tqh + 3) {
                out[ob + tqh + 3] = neg;
              }
            }
          } else {
            if (rv > tqh) {
              out[ob + tqh] = neg;
            }
            if (rv > tqh + 1) {
              out[ob + tqh + 1] = neg;
            }
            if (rv > tqh + 2) {
              out[ob + tqh + 2] = neg;
            }
            if (rv > tqh + 3) {
              out[ob + tqh + 3] = neg;
            }
          }
          int tqh_0 = tq + 128;
          if (al == 0) {
            if (rv > tqh_0 + 3) {
              {
                float4 _v4 = make_float4(negs[0 + 0], negs[0 + 1], negs[0 + 2], negs[0 + 3]);
                *reinterpret_cast<float4*>(out + ob + tqh_0) = _v4;
              }
            } else {
              if (rv > tqh_0) {
                out[ob + tqh_0] = neg;
              }
              if (rv > tqh_0 + 1) {
                out[ob + tqh_0 + 1] = neg;
              }
              if (rv > tqh_0 + 2) {
                out[ob + tqh_0 + 2] = neg;
              }
              if (rv > tqh_0 + 3) {
                out[ob + tqh_0 + 3] = neg;
              }
            }
          } else {
            if (rv > tqh_0) {
              out[ob + tqh_0] = neg;
            }
            if (rv > tqh_0 + 1) {
              out[ob + tqh_0 + 1] = neg;
            }
            if (rv > tqh_0 + 2) {
              out[ob + tqh_0 + 2] = neg;
            }
            if (rv > tqh_0 + 3) {
              out[ob + tqh_0 + 3] = neg;
            }
          }
        }
      }
    }
  }

  // Cleanup
}

}  // extern "C"
