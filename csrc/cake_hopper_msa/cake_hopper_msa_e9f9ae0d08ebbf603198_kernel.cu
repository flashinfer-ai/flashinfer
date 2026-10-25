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
//   CAKE_HQ     query heads per KV head of the route (hq1 / hq2 / hq4)
//   CAKE_SQ     max query length of the route (sq1 .. sq4)
//   CAKE_HQ_SQ  CAKE_HQ * CAKE_SQ, the query rows per batch element
#ifndef CAKE_HQ
#error "CAKE_HQ must come from the record's compile_flags"
#endif
#ifndef CAKE_SQ
#error "CAKE_SQ must come from the record's compile_flags"
#endif
#ifndef CAKE_HQ_SQ
#error "CAKE_HQ_SQ must come from the record's compile_flags"
#endif
#ifndef CAKE_PROGRAM_KERNEL
#error "CAKE_PROGRAM_KERNEL (the program's kernel symbol) must come from the record's compile_flags"
#endif
#define NUM_MAIN_STAGES 1
#define SMEM_K_S_OFF 1024
#define SMEM_K_S_STAGE_BYTES 8192
#define SMEM_K_S_STRIDE 8192
#define SMEM_Q_S_OFF 17408
#define SMEM_Q_S_STAGE_BYTES 1024
#define SMEM_Q_S_STRIDE 1024
#define SMEM_RED_OFF 18432
#define SMEM_RED_STAGE_BYTES 128
#define SMEM_RED_STRIDE 128
#define SMEM_TOTAL 18560
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 4

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void CAKE_PROGRAM_KERNEL(
    const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K,
    float* __restrict__ out, int* __restrict__ page_table, int* __restrict__ seqused_k,
    int* __restrict__ q_offset, int has_qoff, int pt_stride, int max_k_tiles, int total_q,
    int batch_fast, int evict_first) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

  const int mbar_base = smem;
#define full_addr (mbar_base + 0)

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  const int cta_rank = 0;

  // Kernel setup ops
  uint8_t* k_s = reinterpret_cast<uint8_t*>(smem_raw + 1024);
  const int k_s_addr = smem + 1024;
  uint8_t* q_s = reinterpret_cast<uint8_t*>(smem_raw + 17408);
  const int q_s_addr = smem + 17408;
  float* red = reinterpret_cast<float*>(smem_raw + 18432);
  const int red_addr = smem + 18432;

  // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 1 barriers)
  // Mbarriers at smem_raw[0..8)

  if (warp == 0) {
    uint32_t leader = elect_sync();
    if (leader) {
      // full: 1 barriers, init_count=1
      mbarrier_init(smem + 0, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
  }

  __syncthreads();

  // === Task calls (dependency order) ===
  if (warp == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)((&K))) : "memory");
  }
  if (warp == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)((&Q))) : "memory");
  }
  int b = blockIdx.x;
  int t = blockIdx.y;
  if (batch_fast == 0) {
    b = blockIdx.y;
    t = blockIdx.x;
  }
  int tid_1 = threadIdx.x;
  int lane_0 = lane;
  int warp_1 = warp;
  int g = lane_0 / 4;
  int cq = lane_0 - g * 4;
  int sk = seqused_k[b];
  int page = page_table[b * pt_stride + t];
  int nb = (sk + 127) / 128;
  int qoff = sk - CAKE_SQ;
  if (has_qoff != 0) {
    qoff = q_offset[b];
  }
#if CAKE_HQ == 1 && CAKE_SQ == 1
  int oidx = ((tid_1 - tid_1) * max_k_tiles + t) * total_q + b + tid_1;
#elif CAKE_HQ == 1
  int oidx = ((tid_1 - tid_1) * max_k_tiles + t) * total_q + b * CAKE_SQ + tid_1;
#elif CAKE_SQ == 1
  int oidx =
      ((tid_1 - tid_1 / CAKE_HQ * CAKE_HQ) * max_k_tiles + t) * total_q + b + tid_1 / CAKE_HQ;
#else
  int oidx = ((tid_1 - tid_1 / CAKE_HQ * CAKE_HQ) * max_k_tiles + t) * total_q + b * CAKE_SQ +
             tid_1 / CAKE_HQ;
#endif
  if (t >= nb) {
    if (tid_1 < CAKE_HQ_SQ) {
      out[oidx] = -CUDART_INF_F;
    }
  }
  uint64_t _wgmma_desc_0 = (((uint64_t)(((k_s_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) |
                            ((uint64_t)(64) << 32) | (1ULL << 62));
  uint64_t _wgmma_a_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_0 >> 32)) << 32) |
                        (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_0);
  uint64_t _wgmma_desc_1 = (((uint64_t)(((q_s_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) |
                            ((uint64_t)(64) << 32) | (1ULL << 62));
  uint64_t _wgmma_b_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_1 >> 32)) << 32) |
                        (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_1);
  uint64_t _wgmma_desc_2 = (((uint64_t)(((k_s_addr + 8192)) >> 4) & 0x3FFFULL) |
                            ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
  uint64_t _wgmma_a_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_2 >> 32)) << 32) |
                        (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_2);
  if (t < nb) {
    if (warp == 0) {
      if (elect_sync()) {
        mbarrier_arrive_expect_tx(full_addr, 17408);
        int krow = page * 128;
        if (evict_first != 0) {
          asm volatile(
              "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_"
              "hint"
              " [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(k_s_addr),
              "l"((&K)), "r"(0), "r"(krow), "r"(full_addr), "l"(0x12F0000000000000ULL)
              : "memory");
          asm volatile(
              "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_"
              "hint"
              " [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(k_s_addr + 8192),
              "l"((&K)), "r"(0), "r"(krow + 64), "r"(full_addr), "l"(0x12F0000000000000ULL)
              : "memory");
        } else {
          tma_2d_gmem2smem(k_s_addr, (&K), 0, krow, full_addr);
          tma_2d_gmem2smem(k_s_addr + 8192, (&K), 0, krow + 64, full_addr);
        }
#if CAKE_HQ == 1 && CAKE_SQ == 1
        tma_2d_gmem2smem(q_s_addr, (&Q), 0, b, full_addr);
#else
        tma_2d_gmem2smem(q_s_addr, (&Q), 0, b * CAKE_HQ_SQ, full_addr);
#endif
      }
    }
    int sk1 = sk - 1;
    int limc[2];
#if CAKE_HQ == 1
    int pos = qoff + 2 * cq;
#else
    int pos = qoff + 2 * cq / CAKE_HQ;
#endif
    if (pos > sk1) {
      pos = sk1;
    }
    limc[0] = pos - t * 128;
#if CAKE_HQ == 1
    int pos_0 = qoff + (2 * cq + 1);
#else
    int pos_0 = qoff + (2 * cq + 1) / CAKE_HQ;
#endif
    if (pos_0 > sk1) {
      pos_0 = sk1;
    }
    limc[1] = pos_0 - t * 128;
    mbarrier_wait(full_addr, 0);
    float acc0[4];
    float acc1[4];
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, %4, %5, 0, 1, "
        "1;\n}\n"
        : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
        : "l"(_wgmma_a_0), "l"(_wgmma_b_0)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, %4, %5, 0, 1, "
        "1;\n}\n"
        : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
        : "l"(_wgmma_a_1), "l"(_wgmma_b_0)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, %4, %5, 1, 1, "
        "1;\n}\n"
        : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
        : "l"(_wgmma_a_0 + 2), "l"(_wgmma_b_0 + 2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, %4, %5, 1, 1, "
        "1;\n}\n"
        : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
        : "l"(_wgmma_a_1 + 2), "l"(_wgmma_b_0 + 2)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, %4, %5, 1, 1, "
        "1;\n}\n"
        : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
        : "l"(_wgmma_a_0 + 4), "l"(_wgmma_b_0 + 4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, %4, %5, 1, 1, "
        "1;\n}\n"
        : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
        : "l"(_wgmma_a_1 + 4), "l"(_wgmma_b_0 + 4)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, %4, %5, 1, 1, "
        "1;\n}\n"
        : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
        : "l"(_wgmma_a_0 + 6), "l"(_wgmma_b_0 + 6)
        : "memory");
    asm volatile(
        "{\nwgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, %4, %5, 1, 1, "
        "1;\n}\n"
        : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
        : "l"(_wgmma_a_1 + 6), "l"(_wgmma_b_0 + 6)
        : "memory");
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    float cm[2];
    cm[0] = -CUDART_INF_F;
    cm[1] = -CUDART_INF_F;
    int r0 = warp_1 * 16 + g;
    float v0 = -CUDART_INF_F;
    if (limc[0] >= r0) {
      v0 = acc0[0];
    }
    float v1 = -CUDART_INF_F;
    if (limc[0] >= r0 + 64) {
      v1 = acc1[0];
    }
    cm[0] = fmaxf(cm[0], fmaxf(v0, v1));
    float v0_1 = -CUDART_INF_F;
    if (limc[1] >= r0) {
      v0_1 = acc0[1];
    }
    float v1_2 = -CUDART_INF_F;
    if (limc[1] >= r0 + 64) {
      v1_2 = acc1[1];
    }
    cm[1] = fmaxf(cm[1], fmaxf(v0_1, v1_2));
    float v0_3 = -CUDART_INF_F;
    if (limc[0] >= r0 + 8) {
      v0_3 = acc0[2];
    }
    float v1_4 = -CUDART_INF_F;
    if (limc[0] >= r0 + 72) {
      v1_4 = acc1[2];
    }
    cm[0] = fmaxf(cm[0], fmaxf(v0_3, v1_4));
    float v0_5 = -CUDART_INF_F;
    if (limc[1] >= r0 + 8) {
      v0_5 = acc0[3];
    }
    float v1_6 = -CUDART_INF_F;
    if (limc[1] >= r0 + 72) {
      v1_6 = acc1[3];
    }
    cm[1] = fmaxf(cm[1], fmaxf(v0_5, v1_6));
    float tt = cm[0];
    float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, tt, 4);
    tt = fmaxf(tt, _shfl_xor_sync_3);
    float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, tt, 8);
    tt = fmaxf(tt, _shfl_xor_sync_4);
    float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, tt, 16);
    tt = fmaxf(tt, _shfl_xor_sync_5);
    cm[0] = tt;
    float tt_7 = cm[1];
    float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, tt_7, 4);
    tt_7 = fmaxf(tt_7, _shfl_xor_sync_6);
    float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, tt_7, 8);
    tt_7 = fmaxf(tt_7, _shfl_xor_sync_7);
    float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, tt_7, 16);
    tt_7 = fmaxf(tt_7, _shfl_xor_sync_8);
    cm[1] = tt_7;
    if (lane_0 < 4) {
      red[2 * lane_0 * 4 + warp_1] = cm[0];
      red[(2 * lane_0 + 1) * 4 + warp_1] = cm[1];
    }
    asm volatile("barrier.sync 1, 128;" ::: "memory");
    if (tid_1 < CAKE_HQ_SQ) {
      float rv[4];
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])),
                     "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
                   : "r"(red_addr + (unsigned int)tid_1 * 16));
      out[oidx] = fmaxf(fmaxf(rv[0], rv[1]), fmaxf(rv[2], rv[3]));
    }
  }

  // Cleanup
  __syncthreads();
}

}  // extern "C"
