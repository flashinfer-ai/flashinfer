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

// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_fp8_rowwise_gemm_allreduce_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define NUM_TMA_PIPE_STAGES 5
#define SMEM_SMEM_A_OFF 0
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 16384
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_STAGING_OFF 163840
#define SMEM_STAGING_STAGE_BYTES 16384
#define SMEM_STAGING_STRIDE 16384
#define SMEM_COMM_OFF 196608
#define SMEM_COMM_STAGE_BYTES 28672
#define SMEM_COMM_STRIDE 28672
#define SMEM_TOTAL 225376
#define THREADS 512
#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__
__launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) void kernel_cake_fp8_rowwise_gemm_allreduce_82fa3df922ed493ea553(
    const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap W,
    const __grid_constant__ CUtensorMap OUT, __nv_bfloat16* __restrict__ out_ptr,
    float* __restrict__ sa, float* __restrict__ sw, unsigned int* __restrict__ ctl,
    unsigned int* __restrict__ S_mc, int M, int N, int K, int grid_n, int tiles, int K_tiles,
    int group_m, int C, int C_cap, unsigned long long S_bytes, unsigned int flag_words,
    int32_t pg_world, int32_t pg_rank, CakePeerPointerTable<unsigned int> pg_flags,
    uint8_t* __restrict__ S, CakePeerPointerTable<uint8_t> S_peers,
    unsigned int* __restrict__ flags, CakePeerPointerTable<unsigned int> flags_peers) {
  const int tid = threadIdx.x;
  const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
  const uint32_t lane = static_cast<uint32_t>(tid) & 31u;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

  const int mbar_base = smem + 225280;
#define full_addr (mbar_base + 0)
#define empty_addr (mbar_base + 40)
#define pull_addr (mbar_base + 80)

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  const int cta_rank = 0;

  // Kernel setup ops
  uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_A_OFF);
  const int smem_a_addr = smem + SMEM_SMEM_A_OFF;
  uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_B_OFF);
  const int smem_b_addr = smem + SMEM_SMEM_B_OFF;
  __nv_bfloat16* staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_STAGING_OFF);
  const int staging_addr = smem + SMEM_STAGING_OFF;
  uint8_t* comm = reinterpret_cast<uint8_t*>(smem_raw + SMEM_COMM_OFF);
  const int comm_addr = smem + SMEM_COMM_OFF;
  if (warp == 0) {
    if (elect_sync()) {
      asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)((&A))) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)((&W))) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)((&OUT))) : "memory");
    }
  }
  __syncwarp();

  // Mbarrier init (3 pipeline groups, 0 ordered-sequence groups, 11 barriers)
  // Mbarriers at smem_raw[225280..225368)

  if (warp == 1) {
    uint32_t leader = elect_sync();
    if (leader) {
      // Stage-major initialization across barrier declarations.
      mbarrier_init(smem + 225280, 1);
      mbarrier_init(smem + 225320, 256);
      mbarrier_init(smem + 225360, 1);
      mbarrier_init(smem + 225288, 1);
      mbarrier_init(smem + 225328, 256);
      mbarrier_init(smem + 225296, 1);
      mbarrier_init(smem + 225336, 256);
      mbarrier_init(smem + 225304, 1);
      mbarrier_init(smem + 225344, 256);
      mbarrier_init(smem + 225312, 1);
      mbarrier_init(smem + 225352, 256);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
  }

  __syncthreads();

  // ---- Ordered hardware-WG register redistribution ----
  // Dec phase frees registers before any WG attempts inc.
  if (warp >= 8 && warp <= 15) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
  }

  // ---- Role: producer ----
  if (warp == 8) {
    {  // producer_main
      unsigned int grid_n_p = (unsigned int)grid_n;
      unsigned int K_tiles_p = (unsigned int)K_tiles;
      unsigned int group_m_p = (unsigned int)group_m;
      unsigned int c_count_p = (unsigned int)C;
      if (warp == 8) {
        if (elect_sync()) {
          unsigned int it_p = 0;
#pragma unroll 1
          for (unsigned int t_p = bid; t_p < tiles; t_p += num_bids) {
            unsigned int gsz = group_m_p * grid_n_p;
            unsigned int g = t_p / gsz;
            unsigned int r = t_p - g * gsz;
            unsigned int c0 = g * group_m_p;
            unsigned int rows_left = c_count_p - c0;
            unsigned int rows_eff = ((group_m_p < rows_left) ? group_m_p : rows_left);
            unsigned int bn = r / rows_eff;
            unsigned int c = c0 + (r - bn * rows_eff);
            for (unsigned int k_p = 0; k_p < K_tiles_p; k_p++) {
              unsigned int stage_p = it_p % 5;
              mbarrier_wait_hint(empty_addr + (stage_p) * 8, it_p / 5 + 1 & 1, 10000000);
              {
                const uint64_t _tma_operand_pack_0_desc = (uint64_t)((&A));
                const uint32_t _tma_operand_pack_0_mbar = (uint32_t)(full_addr + (stage_p) * 8);
                const uint32_t _tma_operand_pack_0_dst = (uint32_t)(smem_a_addr + stage_p * 32768);
                const int32_t _tma_operand_pack_0_coord_0 = (int32_t)(k_p * 128);
                const int32_t _tma_operand_pack_0_coord_1 = (int32_t)(c * 128);
                const uint64_t _tma_operand_pack_0_cache = (uint64_t)(0x1000000000000000ULL);
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::"
                    "cache_hint"
                    " [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(_tma_operand_pack_0_dst),
                    "l"(_tma_operand_pack_0_desc), "r"(_tma_operand_pack_0_mbar),
                    "r"(_tma_operand_pack_0_coord_0), "r"(_tma_operand_pack_0_coord_1),
                    "l"(_tma_operand_pack_0_cache)
                    : "memory");
              }
              {
                const uint64_t _tma_operand_pack_1_desc = (uint64_t)((&W));
                const uint32_t _tma_operand_pack_1_mbar = (uint32_t)(full_addr + (stage_p) * 8);
                const uint32_t _tma_operand_pack_1_dst = (uint32_t)(smem_b_addr + stage_p * 32768);
                const int32_t _tma_operand_pack_1_coord_0 = (int32_t)(k_p * 128);
                const int32_t _tma_operand_pack_1_coord_1 = (int32_t)(bn * 128);
                const uint64_t _tma_operand_pack_1_cache = (uint64_t)(0x1000000000000000ULL);
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::"
                    "cache_hint"
                    " [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(_tma_operand_pack_1_dst),
                    "l"(_tma_operand_pack_1_desc), "r"(_tma_operand_pack_1_mbar),
                    "r"(_tma_operand_pack_1_coord_0), "r"(_tma_operand_pack_1_coord_1),
                    "l"(_tma_operand_pack_1_cache)
                    : "memory");
              }
              mbarrier_arrive_expect_tx(full_addr + (stage_p) * 8, 32768);
              it_p = it_p + 1;
            }
          }
        }
      }
      __syncwarp();
      __threadfence();
      if (elect_sync()) {
        int cta_slot = 2 * C_cap + blockIdx.x;
        uint32_t _atomic_inc_old_0;
        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                     : "=r"(_atomic_inc_old_0)
                     : "l"(&ctl[cta_slot]), "r"(static_cast<uint32_t>(15))
                     : "memory");
        if (_atomic_inc_old_0 == 15) {
          uint32_t _atomic_inc_old_1;
          asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                       : "=r"(_atomic_inc_old_1)
                       : "l"(&ctl[2 * C_cap + 1024]), "r"(static_cast<uint32_t>(gridDim.x - 1))
                       : "memory");
          if (_atomic_inc_old_1 == (unsigned int)(gridDim.x - 1)) {
            uint32_t _relaxed_ld_0;
            asm volatile("ld.relaxed.gpu.u32 %0, [%1];"
                         : "=r"(_relaxed_ld_0)
                         : "l"(ctl + (2 * C_cap + 1024 + 1))
                         : "memory");
            unsigned int e_fin = _relaxed_ld_0 + 1;
            asm volatile("st.relaxed.sys.u32 [%0], %1;" ::"l"(
                             (reinterpret_cast<unsigned int*>(ctl) + (2 * C_cap + 1024 + 1))),
                         "r"(static_cast<unsigned int>(e_fin))
                         : "memory");
          }
        }
      }
    }
    // ---- Role: math ----
  } else if (warp <= 7) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 216;");
    {  // math_main
      int rank_e = pg_rank;
      int world_e = pg_world;
      int slots_e = ((world_e > 2) ? world_e : 1);
      unsigned int grid_n_u = (unsigned int)grid_n;
      unsigned int K_tiles_u = (unsigned int)K_tiles;
      unsigned int group_m_u = (unsigned int)group_m;
      unsigned int c_count_u = (unsigned int)C;
      unsigned int rows_u = (unsigned int)M;
      uint32_t warp_group_idx = static_cast<uint32_t>(tid) / 128u;
      unsigned int math_wg_idx = make_warp_uniform(warp_group_idx);
      int epi_tid = warp * 32 + lane;
      float accum[64] = {0};
      float accum1[64] = {0};
      float final[64] = {0};
      unsigned int pk[4] = {0};
      unsigned int it_c = 0;
      unsigned int warp_in_wg = (unsigned int)warp % 4;
      unsigned int lane_row = lane / 4;
      unsigned int q_lane = lane % 4;
      unsigned int m_odd = -(q_lane & 1);
      unsigned int m_even = m_odd ^ 4294967295u;
      unsigned int m_hi = -(q_lane >> 1 & 1);
      unsigned int m_lo = m_hi ^ 4294967295u;
#pragma unroll 1
      for (unsigned int t_c = bid; t_c < tiles; t_c += num_bids) {
        unsigned int gsz_1 = group_m_u * grid_n_u;
        unsigned int g_1 = t_c / gsz_1;
        unsigned int r_1 = t_c - g_1 * gsz_1;
        unsigned int c0_1 = g_1 * group_m_u;
        unsigned int rows_left_1 = c_count_u - c0_1;
        unsigned int rows_eff_1 = ((group_m_u < rows_left_1) ? group_m_u : rows_left_1);
        unsigned int bn_1 = r_1 / rows_eff_1;
        unsigned int c_1 = c0_1 + (r_1 - bn_1 * rows_eff_1);
        unsigned int off_m_u = c_1 * 128;
        unsigned int off_n_u = bn_1 * 128;
        int c_tile = (int)c_1;
        int bid_n = (int)bn_1;
        int owner_t = c_tile % world_e;
        int k_e = ((rank_e < owner_t) ? rank_e : rank_e - 1);
        int s_row_e = c_tile / world_e;
        final[0] = 0.0f;
        final[1] = 0.0f;
        final[2] = 0.0f;
        final[3] = 0.0f;
        final[4] = 0.0f;
        final[5] = 0.0f;
        final[6] = 0.0f;
        final[7] = 0.0f;
        final[8] = 0.0f;
        final[9] = 0.0f;
        final[10] = 0.0f;
        final[11] = 0.0f;
        final[12] = 0.0f;
        final[13] = 0.0f;
        final[14] = 0.0f;
        final[15] = 0.0f;
        final[16] = 0.0f;
        final[17] = 0.0f;
        final[18] = 0.0f;
        final[19] = 0.0f;
        final[20] = 0.0f;
        final[21] = 0.0f;
        final[22] = 0.0f;
        final[23] = 0.0f;
        final[24] = 0.0f;
        final[25] = 0.0f;
        final[26] = 0.0f;
        final[27] = 0.0f;
        final[28] = 0.0f;
        final[29] = 0.0f;
        final[30] = 0.0f;
        final[31] = 0.0f;
        final[32] = 0.0f;
        final[33] = 0.0f;
        final[34] = 0.0f;
        final[35] = 0.0f;
        final[36] = 0.0f;
        final[37] = 0.0f;
        final[38] = 0.0f;
        final[39] = 0.0f;
        final[40] = 0.0f;
        final[41] = 0.0f;
        final[42] = 0.0f;
        final[43] = 0.0f;
        final[44] = 0.0f;
        final[45] = 0.0f;
        final[46] = 0.0f;
        final[47] = 0.0f;
        final[48] = 0.0f;
        final[49] = 0.0f;
        final[50] = 0.0f;
        final[51] = 0.0f;
        final[52] = 0.0f;
        final[53] = 0.0f;
        final[54] = 0.0f;
        final[55] = 0.0f;
        final[56] = 0.0f;
        final[57] = 0.0f;
        final[58] = 0.0f;
        final[59] = 0.0f;
        final[60] = 0.0f;
        final[61] = 0.0f;
        final[62] = 0.0f;
        final[63] = 0.0f;
        for (unsigned int k_c = 0; k_c < K_tiles_u; k_c++) {
          unsigned int stage_c = it_c % 5;
          mbarrier_wait_relaxed_hint(full_addr + (stage_c) * 8, it_c / 5 & 1, 10000000);
          uint32_t _wgmma_stage_base_0 = (unsigned int)smem + (unsigned int)stage_c * 32768;
          asm volatile("" : "+f"(accum[0])::"memory");
          asm volatile("" : "+f"(accum[1])::"memory");
          asm volatile("" : "+f"(accum[2])::"memory");
          asm volatile("" : "+f"(accum[3])::"memory");
          asm volatile("" : "+f"(accum[4])::"memory");
          asm volatile("" : "+f"(accum[5])::"memory");
          asm volatile("" : "+f"(accum[6])::"memory");
          asm volatile("" : "+f"(accum[7])::"memory");
          asm volatile("" : "+f"(accum[8])::"memory");
          asm volatile("" : "+f"(accum[9])::"memory");
          asm volatile("" : "+f"(accum[10])::"memory");
          asm volatile("" : "+f"(accum[11])::"memory");
          asm volatile("" : "+f"(accum[12])::"memory");
          asm volatile("" : "+f"(accum[13])::"memory");
          asm volatile("" : "+f"(accum[14])::"memory");
          asm volatile("" : "+f"(accum[15])::"memory");
          asm volatile("" : "+f"(accum[16])::"memory");
          asm volatile("" : "+f"(accum[17])::"memory");
          asm volatile("" : "+f"(accum[18])::"memory");
          asm volatile("" : "+f"(accum[19])::"memory");
          asm volatile("" : "+f"(accum[20])::"memory");
          asm volatile("" : "+f"(accum[21])::"memory");
          asm volatile("" : "+f"(accum[22])::"memory");
          asm volatile("" : "+f"(accum[23])::"memory");
          asm volatile("" : "+f"(accum[24])::"memory");
          asm volatile("" : "+f"(accum[25])::"memory");
          asm volatile("" : "+f"(accum[26])::"memory");
          asm volatile("" : "+f"(accum[27])::"memory");
          asm volatile("" : "+f"(accum[28])::"memory");
          asm volatile("" : "+f"(accum[29])::"memory");
          asm volatile("" : "+f"(accum[30])::"memory");
          asm volatile("" : "+f"(accum[31])::"memory");
          asm volatile("" : "+f"(accum[32])::"memory");
          asm volatile("" : "+f"(accum[33])::"memory");
          asm volatile("" : "+f"(accum[34])::"memory");
          asm volatile("" : "+f"(accum[35])::"memory");
          asm volatile("" : "+f"(accum[36])::"memory");
          asm volatile("" : "+f"(accum[37])::"memory");
          asm volatile("" : "+f"(accum[38])::"memory");
          asm volatile("" : "+f"(accum[39])::"memory");
          asm volatile("" : "+f"(accum[40])::"memory");
          asm volatile("" : "+f"(accum[41])::"memory");
          asm volatile("" : "+f"(accum[42])::"memory");
          asm volatile("" : "+f"(accum[43])::"memory");
          asm volatile("" : "+f"(accum[44])::"memory");
          asm volatile("" : "+f"(accum[45])::"memory");
          asm volatile("" : "+f"(accum[46])::"memory");
          asm volatile("" : "+f"(accum[47])::"memory");
          asm volatile("" : "+f"(accum[48])::"memory");
          asm volatile("" : "+f"(accum[49])::"memory");
          asm volatile("" : "+f"(accum[50])::"memory");
          asm volatile("" : "+f"(accum[51])::"memory");
          asm volatile("" : "+f"(accum[52])::"memory");
          asm volatile("" : "+f"(accum[53])::"memory");
          asm volatile("" : "+f"(accum[54])::"memory");
          asm volatile("" : "+f"(accum[55])::"memory");
          asm volatile("" : "+f"(accum[56])::"memory");
          asm volatile("" : "+f"(accum[57])::"memory");
          asm volatile("" : "+f"(accum[58])::"memory");
          asm volatile("" : "+f"(accum[59])::"memory");
          asm volatile("" : "+f"(accum[60])::"memory");
          asm volatile("" : "+f"(accum[61])::"memory");
          asm volatile("" : "+f"(accum[62])::"memory");
          asm volatile("" : "+f"(accum[63])::"memory");
          asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
          {
            uint64_t _wgmma_a_0_0 =
                (((uint64_t)(((_wgmma_stage_base_0 +
                               (0u + static_cast<uint32_t>(math_wg_idx * 64) * 128u +
                                static_cast<uint32_t>(0) * 1u))) >>
                             4) &
                  0x3FFFULL) |
                 ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            uint64_t _wgmma_b_0_1 =
                (((uint64_t)(((_wgmma_stage_base_0 + (16384u + static_cast<uint32_t>(0) * 128u +
                                                      static_cast<uint32_t>(0) * 1u))) >>
                             4) &
                  0x3FFFULL) |
                 ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            asm volatile(
                "{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, "
                "%6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
                "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, "
                "%39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, "
                "%55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 0, 1, 1;\n}\n"
                : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]), "+f"(accum[4]),
                  "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7]), "+f"(accum[8]), "+f"(accum[9]),
                  "+f"(accum[10]), "+f"(accum[11]), "+f"(accum[12]), "+f"(accum[13]),
                  "+f"(accum[14]), "+f"(accum[15]), "+f"(accum[16]), "+f"(accum[17]),
                  "+f"(accum[18]), "+f"(accum[19]), "+f"(accum[20]), "+f"(accum[21]),
                  "+f"(accum[22]), "+f"(accum[23]), "+f"(accum[24]), "+f"(accum[25]),
                  "+f"(accum[26]), "+f"(accum[27]), "+f"(accum[28]), "+f"(accum[29]),
                  "+f"(accum[30]), "+f"(accum[31]), "+f"(accum[32]), "+f"(accum[33]),
                  "+f"(accum[34]), "+f"(accum[35]), "+f"(accum[36]), "+f"(accum[37]),
                  "+f"(accum[38]), "+f"(accum[39]), "+f"(accum[40]), "+f"(accum[41]),
                  "+f"(accum[42]), "+f"(accum[43]), "+f"(accum[44]), "+f"(accum[45]),
                  "+f"(accum[46]), "+f"(accum[47]), "+f"(accum[48]), "+f"(accum[49]),
                  "+f"(accum[50]), "+f"(accum[51]), "+f"(accum[52]), "+f"(accum[53]),
                  "+f"(accum[54]), "+f"(accum[55]), "+f"(accum[56]), "+f"(accum[57]),
                  "+f"(accum[58]), "+f"(accum[59]), "+f"(accum[60]), "+f"(accum[61]),
                  "+f"(accum[62]), "+f"(accum[63])
                : "l"(_wgmma_a_0_0), "l"(_wgmma_b_0_1)
                : "memory");
          }
          {
            uint64_t _wgmma_a_0_2 =
                (((uint64_t)(((_wgmma_stage_base_0 +
                               (0u + static_cast<uint32_t>(math_wg_idx * 64) * 128u +
                                static_cast<uint32_t>(32) * 1u))) >>
                             4) &
                  0x3FFFULL) |
                 ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            uint64_t _wgmma_b_0_3 =
                (((uint64_t)(((_wgmma_stage_base_0 + (16384u + static_cast<uint32_t>(0) * 128u +
                                                      static_cast<uint32_t>(32) * 1u))) >>
                             4) &
                  0x3FFFULL) |
                 ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            asm volatile(
                "{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, "
                "%6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
                "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, "
                "%39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, "
                "%55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 1, 1, 1;\n}\n"
                : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]), "+f"(accum[4]),
                  "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7]), "+f"(accum[8]), "+f"(accum[9]),
                  "+f"(accum[10]), "+f"(accum[11]), "+f"(accum[12]), "+f"(accum[13]),
                  "+f"(accum[14]), "+f"(accum[15]), "+f"(accum[16]), "+f"(accum[17]),
                  "+f"(accum[18]), "+f"(accum[19]), "+f"(accum[20]), "+f"(accum[21]),
                  "+f"(accum[22]), "+f"(accum[23]), "+f"(accum[24]), "+f"(accum[25]),
                  "+f"(accum[26]), "+f"(accum[27]), "+f"(accum[28]), "+f"(accum[29]),
                  "+f"(accum[30]), "+f"(accum[31]), "+f"(accum[32]), "+f"(accum[33]),
                  "+f"(accum[34]), "+f"(accum[35]), "+f"(accum[36]), "+f"(accum[37]),
                  "+f"(accum[38]), "+f"(accum[39]), "+f"(accum[40]), "+f"(accum[41]),
                  "+f"(accum[42]), "+f"(accum[43]), "+f"(accum[44]), "+f"(accum[45]),
                  "+f"(accum[46]), "+f"(accum[47]), "+f"(accum[48]), "+f"(accum[49]),
                  "+f"(accum[50]), "+f"(accum[51]), "+f"(accum[52]), "+f"(accum[53]),
                  "+f"(accum[54]), "+f"(accum[55]), "+f"(accum[56]), "+f"(accum[57]),
                  "+f"(accum[58]), "+f"(accum[59]), "+f"(accum[60]), "+f"(accum[61]),
                  "+f"(accum[62]), "+f"(accum[63])
                : "l"(_wgmma_a_0_2), "l"(_wgmma_b_0_3)
                : "memory");
          }
          {
            uint64_t _wgmma_a_0_4 =
                (((uint64_t)(((_wgmma_stage_base_0 +
                               (0u + static_cast<uint32_t>(math_wg_idx * 64) * 128u +
                                static_cast<uint32_t>(64) * 1u))) >>
                             4) &
                  0x3FFFULL) |
                 ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            uint64_t _wgmma_b_0_5 =
                (((uint64_t)(((_wgmma_stage_base_0 + (16384u + static_cast<uint32_t>(0) * 128u +
                                                      static_cast<uint32_t>(64) * 1u))) >>
                             4) &
                  0x3FFFULL) |
                 ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            asm volatile(
                "{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, "
                "%6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
                "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, "
                "%39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, "
                "%55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 1, 1, 1;\n}\n"
                : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]), "+f"(accum[4]),
                  "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7]), "+f"(accum[8]), "+f"(accum[9]),
                  "+f"(accum[10]), "+f"(accum[11]), "+f"(accum[12]), "+f"(accum[13]),
                  "+f"(accum[14]), "+f"(accum[15]), "+f"(accum[16]), "+f"(accum[17]),
                  "+f"(accum[18]), "+f"(accum[19]), "+f"(accum[20]), "+f"(accum[21]),
                  "+f"(accum[22]), "+f"(accum[23]), "+f"(accum[24]), "+f"(accum[25]),
                  "+f"(accum[26]), "+f"(accum[27]), "+f"(accum[28]), "+f"(accum[29]),
                  "+f"(accum[30]), "+f"(accum[31]), "+f"(accum[32]), "+f"(accum[33]),
                  "+f"(accum[34]), "+f"(accum[35]), "+f"(accum[36]), "+f"(accum[37]),
                  "+f"(accum[38]), "+f"(accum[39]), "+f"(accum[40]), "+f"(accum[41]),
                  "+f"(accum[42]), "+f"(accum[43]), "+f"(accum[44]), "+f"(accum[45]),
                  "+f"(accum[46]), "+f"(accum[47]), "+f"(accum[48]), "+f"(accum[49]),
                  "+f"(accum[50]), "+f"(accum[51]), "+f"(accum[52]), "+f"(accum[53]),
                  "+f"(accum[54]), "+f"(accum[55]), "+f"(accum[56]), "+f"(accum[57]),
                  "+f"(accum[58]), "+f"(accum[59]), "+f"(accum[60]), "+f"(accum[61]),
                  "+f"(accum[62]), "+f"(accum[63])
                : "l"(_wgmma_a_0_4), "l"(_wgmma_b_0_5)
                : "memory");
          }
          {
            uint64_t _wgmma_a_0_6 =
                (((uint64_t)(((_wgmma_stage_base_0 +
                               (0u + static_cast<uint32_t>(math_wg_idx * 64) * 128u +
                                static_cast<uint32_t>(96) * 1u))) >>
                             4) &
                  0x3FFFULL) |
                 ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            uint64_t _wgmma_b_0_7 =
                (((uint64_t)(((_wgmma_stage_base_0 + (16384u + static_cast<uint32_t>(0) * 128u +
                                                      static_cast<uint32_t>(96) * 1u))) >>
                             4) &
                  0x3FFFULL) |
                 ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            asm volatile(
                "{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, "
                "%6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
                "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, "
                "%39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, "
                "%55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 1, 1, 1;\n}\n"
                : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]), "+f"(accum[4]),
                  "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7]), "+f"(accum[8]), "+f"(accum[9]),
                  "+f"(accum[10]), "+f"(accum[11]), "+f"(accum[12]), "+f"(accum[13]),
                  "+f"(accum[14]), "+f"(accum[15]), "+f"(accum[16]), "+f"(accum[17]),
                  "+f"(accum[18]), "+f"(accum[19]), "+f"(accum[20]), "+f"(accum[21]),
                  "+f"(accum[22]), "+f"(accum[23]), "+f"(accum[24]), "+f"(accum[25]),
                  "+f"(accum[26]), "+f"(accum[27]), "+f"(accum[28]), "+f"(accum[29]),
                  "+f"(accum[30]), "+f"(accum[31]), "+f"(accum[32]), "+f"(accum[33]),
                  "+f"(accum[34]), "+f"(accum[35]), "+f"(accum[36]), "+f"(accum[37]),
                  "+f"(accum[38]), "+f"(accum[39]), "+f"(accum[40]), "+f"(accum[41]),
                  "+f"(accum[42]), "+f"(accum[43]), "+f"(accum[44]), "+f"(accum[45]),
                  "+f"(accum[46]), "+f"(accum[47]), "+f"(accum[48]), "+f"(accum[49]),
                  "+f"(accum[50]), "+f"(accum[51]), "+f"(accum[52]), "+f"(accum[53]),
                  "+f"(accum[54]), "+f"(accum[55]), "+f"(accum[56]), "+f"(accum[57]),
                  "+f"(accum[58]), "+f"(accum[59]), "+f"(accum[60]), "+f"(accum[61]),
                  "+f"(accum[62]), "+f"(accum[63])
                : "l"(_wgmma_a_0_6), "l"(_wgmma_b_0_7)
                : "memory");
          }
          asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
          asm volatile("" : "+f"(accum[0])::"memory");
          asm volatile("" : "+f"(accum[1])::"memory");
          asm volatile("" : "+f"(accum[2])::"memory");
          asm volatile("" : "+f"(accum[3])::"memory");
          asm volatile("" : "+f"(accum[4])::"memory");
          asm volatile("" : "+f"(accum[5])::"memory");
          asm volatile("" : "+f"(accum[6])::"memory");
          asm volatile("" : "+f"(accum[7])::"memory");
          asm volatile("" : "+f"(accum[8])::"memory");
          asm volatile("" : "+f"(accum[9])::"memory");
          asm volatile("" : "+f"(accum[10])::"memory");
          asm volatile("" : "+f"(accum[11])::"memory");
          asm volatile("" : "+f"(accum[12])::"memory");
          asm volatile("" : "+f"(accum[13])::"memory");
          asm volatile("" : "+f"(accum[14])::"memory");
          asm volatile("" : "+f"(accum[15])::"memory");
          asm volatile("" : "+f"(accum[16])::"memory");
          asm volatile("" : "+f"(accum[17])::"memory");
          asm volatile("" : "+f"(accum[18])::"memory");
          asm volatile("" : "+f"(accum[19])::"memory");
          asm volatile("" : "+f"(accum[20])::"memory");
          asm volatile("" : "+f"(accum[21])::"memory");
          asm volatile("" : "+f"(accum[22])::"memory");
          asm volatile("" : "+f"(accum[23])::"memory");
          asm volatile("" : "+f"(accum[24])::"memory");
          asm volatile("" : "+f"(accum[25])::"memory");
          asm volatile("" : "+f"(accum[26])::"memory");
          asm volatile("" : "+f"(accum[27])::"memory");
          asm volatile("" : "+f"(accum[28])::"memory");
          asm volatile("" : "+f"(accum[29])::"memory");
          asm volatile("" : "+f"(accum[30])::"memory");
          asm volatile("" : "+f"(accum[31])::"memory");
          asm volatile("" : "+f"(accum[32])::"memory");
          asm volatile("" : "+f"(accum[33])::"memory");
          asm volatile("" : "+f"(accum[34])::"memory");
          asm volatile("" : "+f"(accum[35])::"memory");
          asm volatile("" : "+f"(accum[36])::"memory");
          asm volatile("" : "+f"(accum[37])::"memory");
          asm volatile("" : "+f"(accum[38])::"memory");
          asm volatile("" : "+f"(accum[39])::"memory");
          asm volatile("" : "+f"(accum[40])::"memory");
          asm volatile("" : "+f"(accum[41])::"memory");
          asm volatile("" : "+f"(accum[42])::"memory");
          asm volatile("" : "+f"(accum[43])::"memory");
          asm volatile("" : "+f"(accum[44])::"memory");
          asm volatile("" : "+f"(accum[45])::"memory");
          asm volatile("" : "+f"(accum[46])::"memory");
          asm volatile("" : "+f"(accum[47])::"memory");
          asm volatile("" : "+f"(accum[48])::"memory");
          asm volatile("" : "+f"(accum[49])::"memory");
          asm volatile("" : "+f"(accum[50])::"memory");
          asm volatile("" : "+f"(accum[51])::"memory");
          asm volatile("" : "+f"(accum[52])::"memory");
          asm volatile("" : "+f"(accum[53])::"memory");
          asm volatile("" : "+f"(accum[54])::"memory");
          asm volatile("" : "+f"(accum[55])::"memory");
          asm volatile("" : "+f"(accum[56])::"memory");
          asm volatile("" : "+f"(accum[57])::"memory");
          asm volatile("" : "+f"(accum[58])::"memory");
          asm volatile("" : "+f"(accum[59])::"memory");
          asm volatile("" : "+f"(accum[60])::"memory");
          asm volatile("" : "+f"(accum[61])::"memory");
          asm volatile("" : "+f"(accum[62])::"memory");
          asm volatile("" : "+f"(accum[63])::"memory");
          asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
          final[0] = final[0] + accum[0];
          final[1] = final[1] + accum[1];
          final[2] = final[2] + accum[2];
          final[3] = final[3] + accum[3];
          final[4] = final[4] + accum[4];
          final[5] = final[5] + accum[5];
          final[6] = final[6] + accum[6];
          final[7] = final[7] + accum[7];
          final[8] = final[8] + accum[8];
          final[9] = final[9] + accum[9];
          final[10] = final[10] + accum[10];
          final[11] = final[11] + accum[11];
          final[12] = final[12] + accum[12];
          final[13] = final[13] + accum[13];
          final[14] = final[14] + accum[14];
          final[15] = final[15] + accum[15];
          final[16] = final[16] + accum[16];
          final[17] = final[17] + accum[17];
          final[18] = final[18] + accum[18];
          final[19] = final[19] + accum[19];
          final[20] = final[20] + accum[20];
          final[21] = final[21] + accum[21];
          final[22] = final[22] + accum[22];
          final[23] = final[23] + accum[23];
          final[24] = final[24] + accum[24];
          final[25] = final[25] + accum[25];
          final[26] = final[26] + accum[26];
          final[27] = final[27] + accum[27];
          final[28] = final[28] + accum[28];
          final[29] = final[29] + accum[29];
          final[30] = final[30] + accum[30];
          final[31] = final[31] + accum[31];
          final[32] = final[32] + accum[32];
          final[33] = final[33] + accum[33];
          final[34] = final[34] + accum[34];
          final[35] = final[35] + accum[35];
          final[36] = final[36] + accum[36];
          final[37] = final[37] + accum[37];
          final[38] = final[38] + accum[38];
          final[39] = final[39] + accum[39];
          final[40] = final[40] + accum[40];
          final[41] = final[41] + accum[41];
          final[42] = final[42] + accum[42];
          final[43] = final[43] + accum[43];
          final[44] = final[44] + accum[44];
          final[45] = final[45] + accum[45];
          final[46] = final[46] + accum[46];
          final[47] = final[47] + accum[47];
          final[48] = final[48] + accum[48];
          final[49] = final[49] + accum[49];
          final[50] = final[50] + accum[50];
          final[51] = final[51] + accum[51];
          final[52] = final[52] + accum[52];
          final[53] = final[53] + accum[53];
          final[54] = final[54] + accum[54];
          final[55] = final[55] + accum[55];
          final[56] = final[56] + accum[56];
          final[57] = final[57] + accum[57];
          final[58] = final[58] + accum[58];
          final[59] = final[59] + accum[59];
          final[60] = final[60] + accum[60];
          final[61] = final[61] + accum[61];
          final[62] = final[62] + accum[62];
          final[63] = final[63] + accum[63];
          mbarrier_arrive(empty_addr + (stage_c) * 8);
          it_c = it_c + 1;
        }
        unsigned int local_lo = math_wg_idx * 64 + warp_in_wg * 16 + lane_row;
        unsigned int row_lo = off_m_u + local_lo;
        unsigned int row_hi = row_lo + 8;
        unsigned int last_row = rows_u - 1;
        unsigned int _min_0 = ((row_lo) < (last_row) ? (row_lo) : (last_row));
        float _vec_load_0[1];
        {
          _vec_load_0[0] = *reinterpret_cast<const float*>(sa + _min_0);
        }
        unsigned int _min_1 = ((row_hi) < (last_row) ? (row_hi) : (last_row));
        float _vec_load_1[1];
        {
          _vec_load_1[0] = *reinterpret_cast<const float*>(sa + _min_1);
        }
        float sa_lo = _vec_load_0[0];
        float sa_hi = _vec_load_1[0];
        unsigned int col_base = off_n_u + q_lane * 2;
        unsigned int cb = q_lane * 16;
        float _vec_load_2[2];
        {
          float2 _v2_0 = *reinterpret_cast<const float2*>(sw + col_base);
          _vec_load_2[0] = _v2_0.x;
          _vec_load_2[0 + 1] = _v2_0.y;
        }
        float v0 = final[0] * ((1) ? sa_lo : sa_hi) * _vec_load_2[0];
        float v1 = final[1] * ((1) ? sa_lo : sa_hi) * _vec_load_2[1];
        __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(make_float2(v0, v1));
        pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_0)[0];
        float _vec_load_3[2];
        {
          float2 _v2_1 = *reinterpret_cast<const float2*>(sw + col_base + 8);
          _vec_load_3[0] = _v2_1.x;
          _vec_load_3[0 + 1] = _v2_1.y;
        }
        float v0_0 = final[4] * ((1) ? sa_lo : sa_hi) * _vec_load_3[0];
        float v1_1 = final[5] * ((1) ? sa_lo : sa_hi) * _vec_load_3[1];
        __nv_bfloat162 _bf16x2_1 = __float22bfloat162_rn(make_float2(v0_0, v1_1));
        pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_1)[0];
        float _vec_load_4[2];
        {
          float2 _v2_2 = *reinterpret_cast<const float2*>(sw + col_base + 16);
          _vec_load_4[0] = _v2_2.x;
          _vec_load_4[0 + 1] = _v2_2.y;
        }
        float v0_2 = final[8] * ((1) ? sa_lo : sa_hi) * _vec_load_4[0];
        float v1_3 = final[9] * ((1) ? sa_lo : sa_hi) * _vec_load_4[1];
        __nv_bfloat162 _bf16x2_2 = __float22bfloat162_rn(make_float2(v0_2, v1_3));
        pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_2)[0];
        float _vec_load_5[2];
        {
          float2 _v2_3 = *reinterpret_cast<const float2*>(sw + col_base + 24);
          _vec_load_5[0] = _v2_3.x;
          _vec_load_5[0 + 1] = _v2_3.y;
        }
        float v0_4 = final[12] * ((1) ? sa_lo : sa_hi) * _vec_load_5[0];
        float v1_5 = final[13] * ((1) ? sa_lo : sa_hi) * _vec_load_5[1];
        __nv_bfloat162 _bf16x2_3 = __float22bfloat162_rn(make_float2(v0_4, v1_5));
        pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_3)[0];
        unsigned int x0 = pk[1] & m_even | pk[0] & m_odd;
        unsigned int x1 = pk[3] & m_even | pk[2] & m_odd;
        unsigned int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, x0, 1);
        unsigned int y0 = _shfl_xor_0;
        unsigned int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, x1, 1);
        unsigned int y1 = _shfl_xor_1;
        pk[0] = y0 & m_odd | pk[0] & m_even;
        pk[1] = y0 & m_even | pk[1] & m_odd;
        pk[2] = y1 & m_odd | pk[2] & m_even;
        pk[3] = y1 & m_even | pk[3] & m_odd;
        unsigned int x2 = pk[2] & m_lo | pk[0] & m_hi;
        unsigned int x3 = pk[3] & m_lo | pk[1] & m_hi;
        unsigned int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, x2, 2);
        unsigned int y2 = _shfl_xor_2;
        unsigned int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, x3, 2);
        unsigned int y3 = _shfl_xor_3;
        pk[0] = y2 & m_hi | pk[0] & m_lo;
        pk[2] = y2 & m_lo | pk[2] & m_hi;
        pk[1] = y3 & m_hi | pk[1] & m_lo;
        pk[3] = y3 & m_lo | pk[3] & m_hi;
        asm volatile(
            "st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
                (staging_addr + (local_lo * 128 + cb ^ (local_lo * 128 + cb >> 7 & 7) << 4))),
            "r"(pk[0]), "r"(pk[1]), "r"(pk[2]), "r"(pk[3])
            : "memory");
        float _vec_load_6[2];
        {
          float2 _v2_4 = *reinterpret_cast<const float2*>(sw + col_base);
          _vec_load_6[0] = _v2_4.x;
          _vec_load_6[0 + 1] = _v2_4.y;
        }
        float v0_6 = final[2] * ((0) ? sa_lo : sa_hi) * _vec_load_6[0];
        float v1_7 = final[3] * ((0) ? sa_lo : sa_hi) * _vec_load_6[1];
        __nv_bfloat162 _bf16x2_4 = __float22bfloat162_rn(make_float2(v0_6, v1_7));
        pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_4)[0];
        float _vec_load_7[2];
        {
          float2 _v2_5 = *reinterpret_cast<const float2*>(sw + col_base + 8);
          _vec_load_7[0] = _v2_5.x;
          _vec_load_7[0 + 1] = _v2_5.y;
        }
        float v0_8 = final[6] * ((0) ? sa_lo : sa_hi) * _vec_load_7[0];
        float v1_9 = final[7] * ((0) ? sa_lo : sa_hi) * _vec_load_7[1];
        __nv_bfloat162 _bf16x2_5 = __float22bfloat162_rn(make_float2(v0_8, v1_9));
        pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_5)[0];
        float _vec_load_8[2];
        {
          float2 _v2_6 = *reinterpret_cast<const float2*>(sw + col_base + 16);
          _vec_load_8[0] = _v2_6.x;
          _vec_load_8[0 + 1] = _v2_6.y;
        }
        float v0_10 = final[10] * ((0) ? sa_lo : sa_hi) * _vec_load_8[0];
        float v1_11 = final[11] * ((0) ? sa_lo : sa_hi) * _vec_load_8[1];
        __nv_bfloat162 _bf16x2_6 = __float22bfloat162_rn(make_float2(v0_10, v1_11));
        pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_6)[0];
        float _vec_load_9[2];
        {
          float2 _v2_7 = *reinterpret_cast<const float2*>(sw + col_base + 24);
          _vec_load_9[0] = _v2_7.x;
          _vec_load_9[0 + 1] = _v2_7.y;
        }
        float v0_12 = final[14] * ((0) ? sa_lo : sa_hi) * _vec_load_9[0];
        float v1_13 = final[15] * ((0) ? sa_lo : sa_hi) * _vec_load_9[1];
        __nv_bfloat162 _bf16x2_7 = __float22bfloat162_rn(make_float2(v0_12, v1_13));
        pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_7)[0];
        unsigned int x0_14 = pk[1] & m_even | pk[0] & m_odd;
        unsigned int x1_15 = pk[3] & m_even | pk[2] & m_odd;
        unsigned int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, x0_14, 1);
        unsigned int y0_16 = _shfl_xor_4;
        unsigned int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, x1_15, 1);
        unsigned int y1_17 = _shfl_xor_5;
        pk[0] = y0_16 & m_odd | pk[0] & m_even;
        pk[1] = y0_16 & m_even | pk[1] & m_odd;
        pk[2] = y1_17 & m_odd | pk[2] & m_even;
        pk[3] = y1_17 & m_even | pk[3] & m_odd;
        unsigned int x2_18 = pk[2] & m_lo | pk[0] & m_hi;
        unsigned int x3_19 = pk[3] & m_lo | pk[1] & m_hi;
        unsigned int _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, x2_18, 2);
        unsigned int y2_20 = _shfl_xor_6;
        unsigned int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, x3_19, 2);
        unsigned int y3_21 = _shfl_xor_7;
        pk[0] = y2_20 & m_hi | pk[0] & m_lo;
        pk[2] = y2_20 & m_lo | pk[2] & m_hi;
        pk[1] = y3_21 & m_hi | pk[1] & m_lo;
        pk[3] = y3_21 & m_lo | pk[3] & m_hi;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
                         (staging_addr +
                          ((local_lo + 8) * 128 + cb ^ ((local_lo + 8) * 128 + cb >> 7 & 7) << 4))),
                     "r"(pk[0]), "r"(pk[1]), "r"(pk[2]), "r"(pk[3])
                     : "memory");
        unsigned int cb_22 = 64 + q_lane * 16;
        float _vec_load_10[2];
        {
          float2 _v2_8 = *reinterpret_cast<const float2*>(sw + col_base + 32);
          _vec_load_10[0] = _v2_8.x;
          _vec_load_10[0 + 1] = _v2_8.y;
        }
        float v0_23 = final[16] * ((1) ? sa_lo : sa_hi) * _vec_load_10[0];
        float v1_24 = final[17] * ((1) ? sa_lo : sa_hi) * _vec_load_10[1];
        __nv_bfloat162 _bf16x2_8 = __float22bfloat162_rn(make_float2(v0_23, v1_24));
        pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_8)[0];
        float _vec_load_11[2];
        {
          float2 _v2_9 = *reinterpret_cast<const float2*>(sw + col_base + 40);
          _vec_load_11[0] = _v2_9.x;
          _vec_load_11[0 + 1] = _v2_9.y;
        }
        float v0_25 = final[20] * ((1) ? sa_lo : sa_hi) * _vec_load_11[0];
        float v1_26 = final[21] * ((1) ? sa_lo : sa_hi) * _vec_load_11[1];
        __nv_bfloat162 _bf16x2_9 = __float22bfloat162_rn(make_float2(v0_25, v1_26));
        pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_9)[0];
        float _vec_load_12[2];
        {
          float2 _v2_10 = *reinterpret_cast<const float2*>(sw + col_base + 48);
          _vec_load_12[0] = _v2_10.x;
          _vec_load_12[0 + 1] = _v2_10.y;
        }
        float v0_27 = final[24] * ((1) ? sa_lo : sa_hi) * _vec_load_12[0];
        float v1_28 = final[25] * ((1) ? sa_lo : sa_hi) * _vec_load_12[1];
        __nv_bfloat162 _bf16x2_10 = __float22bfloat162_rn(make_float2(v0_27, v1_28));
        pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_10)[0];
        float _vec_load_13[2];
        {
          float2 _v2_11 = *reinterpret_cast<const float2*>(sw + col_base + 56);
          _vec_load_13[0] = _v2_11.x;
          _vec_load_13[0 + 1] = _v2_11.y;
        }
        float v0_29 = final[28] * ((1) ? sa_lo : sa_hi) * _vec_load_13[0];
        float v1_30 = final[29] * ((1) ? sa_lo : sa_hi) * _vec_load_13[1];
        __nv_bfloat162 _bf16x2_11 = __float22bfloat162_rn(make_float2(v0_29, v1_30));
        pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_11)[0];
        unsigned int x0_31 = pk[1] & m_even | pk[0] & m_odd;
        unsigned int x1_32 = pk[3] & m_even | pk[2] & m_odd;
        unsigned int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, x0_31, 1);
        unsigned int y0_33 = _shfl_xor_8;
        unsigned int _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, x1_32, 1);
        unsigned int y1_34 = _shfl_xor_9;
        pk[0] = y0_33 & m_odd | pk[0] & m_even;
        pk[1] = y0_33 & m_even | pk[1] & m_odd;
        pk[2] = y1_34 & m_odd | pk[2] & m_even;
        pk[3] = y1_34 & m_even | pk[3] & m_odd;
        unsigned int x2_35 = pk[2] & m_lo | pk[0] & m_hi;
        unsigned int x3_36 = pk[3] & m_lo | pk[1] & m_hi;
        unsigned int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, x2_35, 2);
        unsigned int y2_37 = _shfl_xor_10;
        unsigned int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, x3_36, 2);
        unsigned int y3_38 = _shfl_xor_11;
        pk[0] = y2_37 & m_hi | pk[0] & m_lo;
        pk[2] = y2_37 & m_lo | pk[2] & m_hi;
        pk[1] = y3_38 & m_hi | pk[1] & m_lo;
        pk[3] = y3_38 & m_lo | pk[3] & m_hi;
        asm volatile(
            "st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
                (staging_addr + (local_lo * 128 + cb_22 ^ (local_lo * 128 + cb_22 >> 7 & 7) << 4))),
            "r"(pk[0]), "r"(pk[1]), "r"(pk[2]), "r"(pk[3])
            : "memory");
        float _vec_load_14[2];
        {
          float2 _v2_12 = *reinterpret_cast<const float2*>(sw + col_base + 32);
          _vec_load_14[0] = _v2_12.x;
          _vec_load_14[0 + 1] = _v2_12.y;
        }
        float v0_39 = final[18] * ((0) ? sa_lo : sa_hi) * _vec_load_14[0];
        float v1_40 = final[19] * ((0) ? sa_lo : sa_hi) * _vec_load_14[1];
        __nv_bfloat162 _bf16x2_12 = __float22bfloat162_rn(make_float2(v0_39, v1_40));
        pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_12)[0];
        float _vec_load_15[2];
        {
          float2 _v2_13 = *reinterpret_cast<const float2*>(sw + col_base + 40);
          _vec_load_15[0] = _v2_13.x;
          _vec_load_15[0 + 1] = _v2_13.y;
        }
        float v0_41 = final[22] * ((0) ? sa_lo : sa_hi) * _vec_load_15[0];
        float v1_42 = final[23] * ((0) ? sa_lo : sa_hi) * _vec_load_15[1];
        __nv_bfloat162 _bf16x2_13 = __float22bfloat162_rn(make_float2(v0_41, v1_42));
        pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_13)[0];
        float _vec_load_16[2];
        {
          float2 _v2_14 = *reinterpret_cast<const float2*>(sw + col_base + 48);
          _vec_load_16[0] = _v2_14.x;
          _vec_load_16[0 + 1] = _v2_14.y;
        }
        float v0_43 = final[26] * ((0) ? sa_lo : sa_hi) * _vec_load_16[0];
        float v1_44 = final[27] * ((0) ? sa_lo : sa_hi) * _vec_load_16[1];
        __nv_bfloat162 _bf16x2_14 = __float22bfloat162_rn(make_float2(v0_43, v1_44));
        pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_14)[0];
        float _vec_load_17[2];
        {
          float2 _v2_15 = *reinterpret_cast<const float2*>(sw + col_base + 56);
          _vec_load_17[0] = _v2_15.x;
          _vec_load_17[0 + 1] = _v2_15.y;
        }
        float v0_45 = final[30] * ((0) ? sa_lo : sa_hi) * _vec_load_17[0];
        float v1_46 = final[31] * ((0) ? sa_lo : sa_hi) * _vec_load_17[1];
        __nv_bfloat162 _bf16x2_15 = __float22bfloat162_rn(make_float2(v0_45, v1_46));
        pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_15)[0];
        unsigned int x0_47 = pk[1] & m_even | pk[0] & m_odd;
        unsigned int x1_48 = pk[3] & m_even | pk[2] & m_odd;
        unsigned int _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, x0_47, 1);
        unsigned int y0_49 = _shfl_xor_12;
        unsigned int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, x1_48, 1);
        unsigned int y1_50 = _shfl_xor_13;
        pk[0] = y0_49 & m_odd | pk[0] & m_even;
        pk[1] = y0_49 & m_even | pk[1] & m_odd;
        pk[2] = y1_50 & m_odd | pk[2] & m_even;
        pk[3] = y1_50 & m_even | pk[3] & m_odd;
        unsigned int x2_51 = pk[2] & m_lo | pk[0] & m_hi;
        unsigned int x3_52 = pk[3] & m_lo | pk[1] & m_hi;
        unsigned int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, x2_51, 2);
        unsigned int y2_53 = _shfl_xor_14;
        unsigned int _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, x3_52, 2);
        unsigned int y3_54 = _shfl_xor_15;
        pk[0] = y2_53 & m_hi | pk[0] & m_lo;
        pk[2] = y2_53 & m_lo | pk[2] & m_hi;
        pk[1] = y3_54 & m_hi | pk[1] & m_lo;
        pk[3] = y3_54 & m_lo | pk[3] & m_hi;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
                         (staging_addr + ((local_lo + 8) * 128 + cb_22 ^
                                          ((local_lo + 8) * 128 + cb_22 >> 7 & 7) << 4))),
                     "r"(pk[0]), "r"(pk[1]), "r"(pk[2]), "r"(pk[3])
                     : "memory");
        unsigned int cb_55 = q_lane * 16;
        float _vec_load_18[2];
        {
          float2 _v2_16 = *reinterpret_cast<const float2*>(sw + col_base + 64);
          _vec_load_18[0] = _v2_16.x;
          _vec_load_18[0 + 1] = _v2_16.y;
        }
        float v0_56 = final[32] * ((1) ? sa_lo : sa_hi) * _vec_load_18[0];
        float v1_57 = final[33] * ((1) ? sa_lo : sa_hi) * _vec_load_18[1];
        __nv_bfloat162 _bf16x2_16 = __float22bfloat162_rn(make_float2(v0_56, v1_57));
        pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_16)[0];
        float _vec_load_19[2];
        {
          float2 _v2_17 = *reinterpret_cast<const float2*>(sw + col_base + 72);
          _vec_load_19[0] = _v2_17.x;
          _vec_load_19[0 + 1] = _v2_17.y;
        }
        float v0_58 = final[36] * ((1) ? sa_lo : sa_hi) * _vec_load_19[0];
        float v1_59 = final[37] * ((1) ? sa_lo : sa_hi) * _vec_load_19[1];
        __nv_bfloat162 _bf16x2_17 = __float22bfloat162_rn(make_float2(v0_58, v1_59));
        pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_17)[0];
        float _vec_load_20[2];
        {
          float2 _v2_18 = *reinterpret_cast<const float2*>(sw + col_base + 80);
          _vec_load_20[0] = _v2_18.x;
          _vec_load_20[0 + 1] = _v2_18.y;
        }
        float v0_60 = final[40] * ((1) ? sa_lo : sa_hi) * _vec_load_20[0];
        float v1_61 = final[41] * ((1) ? sa_lo : sa_hi) * _vec_load_20[1];
        __nv_bfloat162 _bf16x2_18 = __float22bfloat162_rn(make_float2(v0_60, v1_61));
        pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_18)[0];
        float _vec_load_21[2];
        {
          float2 _v2_19 = *reinterpret_cast<const float2*>(sw + col_base + 88);
          _vec_load_21[0] = _v2_19.x;
          _vec_load_21[0 + 1] = _v2_19.y;
        }
        float v0_62 = final[44] * ((1) ? sa_lo : sa_hi) * _vec_load_21[0];
        float v1_63 = final[45] * ((1) ? sa_lo : sa_hi) * _vec_load_21[1];
        __nv_bfloat162 _bf16x2_19 = __float22bfloat162_rn(make_float2(v0_62, v1_63));
        pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_19)[0];
        unsigned int x0_64 = pk[1] & m_even | pk[0] & m_odd;
        unsigned int x1_65 = pk[3] & m_even | pk[2] & m_odd;
        unsigned int _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, x0_64, 1);
        unsigned int y0_66 = _shfl_xor_16;
        unsigned int _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, x1_65, 1);
        unsigned int y1_67 = _shfl_xor_17;
        pk[0] = y0_66 & m_odd | pk[0] & m_even;
        pk[1] = y0_66 & m_even | pk[1] & m_odd;
        pk[2] = y1_67 & m_odd | pk[2] & m_even;
        pk[3] = y1_67 & m_even | pk[3] & m_odd;
        unsigned int x2_68 = pk[2] & m_lo | pk[0] & m_hi;
        unsigned int x3_69 = pk[3] & m_lo | pk[1] & m_hi;
        unsigned int _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, x2_68, 2);
        unsigned int y2_70 = _shfl_xor_18;
        unsigned int _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, x3_69, 2);
        unsigned int y3_71 = _shfl_xor_19;
        pk[0] = y2_70 & m_hi | pk[0] & m_lo;
        pk[2] = y2_70 & m_lo | pk[2] & m_hi;
        pk[1] = y3_71 & m_hi | pk[1] & m_lo;
        pk[3] = y3_71 & m_lo | pk[3] & m_hi;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
                         (staging_addr + ((128 + local_lo) * 128 + cb_55 ^
                                          ((128 + local_lo) * 128 + cb_55 >> 7 & 7) << 4))),
                     "r"(pk[0]), "r"(pk[1]), "r"(pk[2]), "r"(pk[3])
                     : "memory");
        float _vec_load_22[2];
        {
          float2 _v2_20 = *reinterpret_cast<const float2*>(sw + col_base + 64);
          _vec_load_22[0] = _v2_20.x;
          _vec_load_22[0 + 1] = _v2_20.y;
        }
        float v0_72 = final[34] * ((0) ? sa_lo : sa_hi) * _vec_load_22[0];
        float v1_73 = final[35] * ((0) ? sa_lo : sa_hi) * _vec_load_22[1];
        __nv_bfloat162 _bf16x2_20 = __float22bfloat162_rn(make_float2(v0_72, v1_73));
        pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_20)[0];
        float _vec_load_23[2];
        {
          float2 _v2_21 = *reinterpret_cast<const float2*>(sw + col_base + 72);
          _vec_load_23[0] = _v2_21.x;
          _vec_load_23[0 + 1] = _v2_21.y;
        }
        float v0_74 = final[38] * ((0) ? sa_lo : sa_hi) * _vec_load_23[0];
        float v1_75 = final[39] * ((0) ? sa_lo : sa_hi) * _vec_load_23[1];
        __nv_bfloat162 _bf16x2_21 = __float22bfloat162_rn(make_float2(v0_74, v1_75));
        pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_21)[0];
        float _vec_load_24[2];
        {
          float2 _v2_22 = *reinterpret_cast<const float2*>(sw + col_base + 80);
          _vec_load_24[0] = _v2_22.x;
          _vec_load_24[0 + 1] = _v2_22.y;
        }
        float v0_76 = final[42] * ((0) ? sa_lo : sa_hi) * _vec_load_24[0];
        float v1_77 = final[43] * ((0) ? sa_lo : sa_hi) * _vec_load_24[1];
        __nv_bfloat162 _bf16x2_22 = __float22bfloat162_rn(make_float2(v0_76, v1_77));
        pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_22)[0];
        float _vec_load_25[2];
        {
          float2 _v2_23 = *reinterpret_cast<const float2*>(sw + col_base + 88);
          _vec_load_25[0] = _v2_23.x;
          _vec_load_25[0 + 1] = _v2_23.y;
        }
        float v0_78 = final[46] * ((0) ? sa_lo : sa_hi) * _vec_load_25[0];
        float v1_79 = final[47] * ((0) ? sa_lo : sa_hi) * _vec_load_25[1];
        __nv_bfloat162 _bf16x2_23 = __float22bfloat162_rn(make_float2(v0_78, v1_79));
        pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_23)[0];
        unsigned int x0_80 = pk[1] & m_even | pk[0] & m_odd;
        unsigned int x1_81 = pk[3] & m_even | pk[2] & m_odd;
        unsigned int _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, x0_80, 1);
        unsigned int y0_82 = _shfl_xor_20;
        unsigned int _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, x1_81, 1);
        unsigned int y1_83 = _shfl_xor_21;
        pk[0] = y0_82 & m_odd | pk[0] & m_even;
        pk[1] = y0_82 & m_even | pk[1] & m_odd;
        pk[2] = y1_83 & m_odd | pk[2] & m_even;
        pk[3] = y1_83 & m_even | pk[3] & m_odd;
        unsigned int x2_84 = pk[2] & m_lo | pk[0] & m_hi;
        unsigned int x3_85 = pk[3] & m_lo | pk[1] & m_hi;
        unsigned int _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, x2_84, 2);
        unsigned int y2_86 = _shfl_xor_22;
        unsigned int _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, x3_85, 2);
        unsigned int y3_87 = _shfl_xor_23;
        pk[0] = y2_86 & m_hi | pk[0] & m_lo;
        pk[2] = y2_86 & m_lo | pk[2] & m_hi;
        pk[1] = y3_87 & m_hi | pk[1] & m_lo;
        pk[3] = y3_87 & m_lo | pk[3] & m_hi;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
                         (staging_addr + ((128 + local_lo + 8) * 128 + cb_55 ^
                                          ((128 + local_lo + 8) * 128 + cb_55 >> 7 & 7) << 4))),
                     "r"(pk[0]), "r"(pk[1]), "r"(pk[2]), "r"(pk[3])
                     : "memory");
        unsigned int cb_88 = 64 + q_lane * 16;
        float _vec_load_26[2];
        {
          float2 _v2_24 = *reinterpret_cast<const float2*>(sw + col_base + 96);
          _vec_load_26[0] = _v2_24.x;
          _vec_load_26[0 + 1] = _v2_24.y;
        }
        float v0_89 = final[48] * ((1) ? sa_lo : sa_hi) * _vec_load_26[0];
        float v1_90 = final[49] * ((1) ? sa_lo : sa_hi) * _vec_load_26[1];
        __nv_bfloat162 _bf16x2_24 = __float22bfloat162_rn(make_float2(v0_89, v1_90));
        pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_24)[0];
        float _vec_load_27[2];
        {
          float2 _v2_25 = *reinterpret_cast<const float2*>(sw + col_base + 104);
          _vec_load_27[0] = _v2_25.x;
          _vec_load_27[0 + 1] = _v2_25.y;
        }
        float v0_91 = final[52] * ((1) ? sa_lo : sa_hi) * _vec_load_27[0];
        float v1_92 = final[53] * ((1) ? sa_lo : sa_hi) * _vec_load_27[1];
        __nv_bfloat162 _bf16x2_25 = __float22bfloat162_rn(make_float2(v0_91, v1_92));
        pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_25)[0];
        float _vec_load_28[2];
        {
          float2 _v2_26 = *reinterpret_cast<const float2*>(sw + col_base + 112);
          _vec_load_28[0] = _v2_26.x;
          _vec_load_28[0 + 1] = _v2_26.y;
        }
        float v0_93 = final[56] * ((1) ? sa_lo : sa_hi) * _vec_load_28[0];
        float v1_94 = final[57] * ((1) ? sa_lo : sa_hi) * _vec_load_28[1];
        __nv_bfloat162 _bf16x2_26 = __float22bfloat162_rn(make_float2(v0_93, v1_94));
        pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_26)[0];
        float _vec_load_29[2];
        {
          float2 _v2_27 = *reinterpret_cast<const float2*>(sw + col_base + 120);
          _vec_load_29[0] = _v2_27.x;
          _vec_load_29[0 + 1] = _v2_27.y;
        }
        float v0_95 = final[60] * ((1) ? sa_lo : sa_hi) * _vec_load_29[0];
        float v1_96 = final[61] * ((1) ? sa_lo : sa_hi) * _vec_load_29[1];
        __nv_bfloat162 _bf16x2_27 = __float22bfloat162_rn(make_float2(v0_95, v1_96));
        pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_27)[0];
        unsigned int x0_97 = pk[1] & m_even | pk[0] & m_odd;
        unsigned int x1_98 = pk[3] & m_even | pk[2] & m_odd;
        unsigned int _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, x0_97, 1);
        unsigned int y0_99 = _shfl_xor_24;
        unsigned int _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, x1_98, 1);
        unsigned int y1_100 = _shfl_xor_25;
        pk[0] = y0_99 & m_odd | pk[0] & m_even;
        pk[1] = y0_99 & m_even | pk[1] & m_odd;
        pk[2] = y1_100 & m_odd | pk[2] & m_even;
        pk[3] = y1_100 & m_even | pk[3] & m_odd;
        unsigned int x2_101 = pk[2] & m_lo | pk[0] & m_hi;
        unsigned int x3_102 = pk[3] & m_lo | pk[1] & m_hi;
        unsigned int _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, x2_101, 2);
        unsigned int y2_103 = _shfl_xor_26;
        unsigned int _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, x3_102, 2);
        unsigned int y3_104 = _shfl_xor_27;
        pk[0] = y2_103 & m_hi | pk[0] & m_lo;
        pk[2] = y2_103 & m_lo | pk[2] & m_hi;
        pk[1] = y3_104 & m_hi | pk[1] & m_lo;
        pk[3] = y3_104 & m_lo | pk[3] & m_hi;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
                         (staging_addr + ((128 + local_lo) * 128 + cb_88 ^
                                          ((128 + local_lo) * 128 + cb_88 >> 7 & 7) << 4))),
                     "r"(pk[0]), "r"(pk[1]), "r"(pk[2]), "r"(pk[3])
                     : "memory");
        float _vec_load_30[2];
        {
          float2 _v2_28 = *reinterpret_cast<const float2*>(sw + col_base + 96);
          _vec_load_30[0] = _v2_28.x;
          _vec_load_30[0 + 1] = _v2_28.y;
        }
        float v0_105 = final[50] * ((0) ? sa_lo : sa_hi) * _vec_load_30[0];
        float v1_106 = final[51] * ((0) ? sa_lo : sa_hi) * _vec_load_30[1];
        __nv_bfloat162 _bf16x2_28 = __float22bfloat162_rn(make_float2(v0_105, v1_106));
        pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_28)[0];
        float _vec_load_31[2];
        {
          float2 _v2_29 = *reinterpret_cast<const float2*>(sw + col_base + 104);
          _vec_load_31[0] = _v2_29.x;
          _vec_load_31[0 + 1] = _v2_29.y;
        }
        float v0_107 = final[54] * ((0) ? sa_lo : sa_hi) * _vec_load_31[0];
        float v1_108 = final[55] * ((0) ? sa_lo : sa_hi) * _vec_load_31[1];
        __nv_bfloat162 _bf16x2_29 = __float22bfloat162_rn(make_float2(v0_107, v1_108));
        pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_29)[0];
        float _vec_load_32[2];
        {
          float2 _v2_30 = *reinterpret_cast<const float2*>(sw + col_base + 112);
          _vec_load_32[0] = _v2_30.x;
          _vec_load_32[0 + 1] = _v2_30.y;
        }
        float v0_109 = final[58] * ((0) ? sa_lo : sa_hi) * _vec_load_32[0];
        float v1_110 = final[59] * ((0) ? sa_lo : sa_hi) * _vec_load_32[1];
        __nv_bfloat162 _bf16x2_30 = __float22bfloat162_rn(make_float2(v0_109, v1_110));
        pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_30)[0];
        float _vec_load_33[2];
        {
          float2 _v2_31 = *reinterpret_cast<const float2*>(sw + col_base + 120);
          _vec_load_33[0] = _v2_31.x;
          _vec_load_33[0 + 1] = _v2_31.y;
        }
        float v0_111 = final[62] * ((0) ? sa_lo : sa_hi) * _vec_load_33[0];
        float v1_112 = final[63] * ((0) ? sa_lo : sa_hi) * _vec_load_33[1];
        __nv_bfloat162 _bf16x2_31 = __float22bfloat162_rn(make_float2(v0_111, v1_112));
        pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_31)[0];
        unsigned int x0_113 = pk[1] & m_even | pk[0] & m_odd;
        unsigned int x1_114 = pk[3] & m_even | pk[2] & m_odd;
        unsigned int _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, x0_113, 1);
        unsigned int y0_115 = _shfl_xor_28;
        unsigned int _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, x1_114, 1);
        unsigned int y1_116 = _shfl_xor_29;
        pk[0] = y0_115 & m_odd | pk[0] & m_even;
        pk[1] = y0_115 & m_even | pk[1] & m_odd;
        pk[2] = y1_116 & m_odd | pk[2] & m_even;
        pk[3] = y1_116 & m_even | pk[3] & m_odd;
        unsigned int x2_117 = pk[2] & m_lo | pk[0] & m_hi;
        unsigned int x3_118 = pk[3] & m_lo | pk[1] & m_hi;
        unsigned int _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, x2_117, 2);
        unsigned int y2_119 = _shfl_xor_30;
        unsigned int _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, x3_118, 2);
        unsigned int y3_120 = _shfl_xor_31;
        pk[0] = y2_119 & m_hi | pk[0] & m_lo;
        pk[2] = y2_119 & m_lo | pk[2] & m_hi;
        pk[1] = y3_120 & m_hi | pk[1] & m_lo;
        pk[3] = y3_120 & m_lo | pk[3] & m_hi;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
                         (staging_addr + ((128 + local_lo + 8) * 128 + cb_88 ^
                                          ((128 + local_lo + 8) * 128 + cb_88 >> 7 & 7) << 4))),
                     "r"(pk[0]), "r"(pk[1]), "r"(pk[2]), "r"(pk[3])
                     : "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("bar.sync 1, 256;" ::: "memory");
        if (warp == 0) {
          if (elect_sync()) {
            if (owner_t == rank_e) {
              tma_store_2d((&OUT), off_n_u, off_m_u, staging_addr);
              tma_store_2d((&OUT), off_n_u + 64, off_m_u, staging_addr + 16384);
              asm volatile("cp.async.bulk.commit_group;");
              asm volatile("cp.async.bulk.wait_group 0;");
              asm volatile("fence.proxy.async;");
            } else {
              // nvlink_push: peers.ptrs[owner_t] + (((unsigned long long)s_row_e * (unsigned long
              // long)slots_e + (unsigned long long)k_e) * (unsigned long long)grid_n + (unsigned
              // long long)bid_n) * 32768 <- smem(staging_addr), 32768B
              {
                void* __remote =
                    (void*)((char*)((S_peers).ptrs[owner_t]) +
                            (uint64_t)((((unsigned long long)s_row_e * (unsigned long long)slots_e +
                                         (unsigned long long)k_e) *
                                            (unsigned long long)grid_n +
                                        (unsigned long long)bid_n) *
                                       32768));
                asm volatile(
                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(__remote),
                    "r"(staging_addr), "r"((uint32_t)(32768))
                    : "memory");
              }
              asm volatile("cp.async.bulk.commit_group;");
              asm volatile("cp.async.bulk.wait_group 0;");
              asm volatile("fence.proxy.async;");
            }
            uint32_t _atomic_inc_old_2;
            asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                         : "=r"(_atomic_inc_old_2)
                         : "l"(&ctl[c_tile]), "r"(static_cast<uint32_t>(grid_n - 1))
                         : "memory");
            if (_atomic_inc_old_2 == (unsigned int)(grid_n - 1)) {
              __threadfence_system();
              asm volatile("red.release.sys.global.add.u32 [%0], %1;" ::"l"(
                               reinterpret_cast<unsigned int*>(flags_peers.ptrs[owner_t] + c_tile)),
                           "r"(static_cast<unsigned int>(1))
                           : "memory");
            }
          }
        }
        asm volatile("bar.sync 1, 256;" ::: "memory");
      }
      if (warp == 0) {
        if (elect_sync()) {
          asm volatile("cp.async.bulk.wait_group 0;");
        }
      }
      __syncwarp();
      __threadfence();
      if (elect_sync()) {
        int cta_slot_1 = 2 * C_cap + blockIdx.x;
        uint32_t _atomic_inc_old_3;
        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                     : "=r"(_atomic_inc_old_3)
                     : "l"(&ctl[cta_slot_1]), "r"(static_cast<uint32_t>(15))
                     : "memory");
        if (_atomic_inc_old_3 == 15) {
          uint32_t _atomic_inc_old_4;
          asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                       : "=r"(_atomic_inc_old_4)
                       : "l"(&ctl[2 * C_cap + 1024]), "r"(static_cast<uint32_t>(gridDim.x - 1))
                       : "memory");
          if (_atomic_inc_old_4 == (unsigned int)(gridDim.x - 1)) {
            uint32_t _relaxed_ld_1;
            asm volatile("ld.relaxed.gpu.u32 %0, [%1];"
                         : "=r"(_relaxed_ld_1)
                         : "l"(ctl + (2 * C_cap + 1024 + 1))
                         : "memory");
            unsigned int e_fin_1 = _relaxed_ld_1 + 1;
            asm volatile("st.relaxed.sys.u32 [%0], %1;" ::"l"(
                             (reinterpret_cast<unsigned int*>(ctl) + (2 * C_cap + 1024 + 1))),
                         "r"(static_cast<unsigned int>(e_fin_1))
                         : "memory");
          }
        }
      }
    }
    // ---- Role: reducers ----
  } else if (warp >= 9 && warp <= 14) {
    {  // reducers_main
      int rank_r = pg_rank;
      int world_r = pg_world;
      unsigned int world_ru = (unsigned int)world_r;
      int slots_r = ((world_r > 2) ? world_r : 1);
      uint32_t _relaxed_ld_2;
      asm volatile("ld.relaxed.gpu.u32 %0, [%1];"
                   : "=r"(_relaxed_ld_2)
                   : "l"(ctl + (2 * C_cap + 1024 + 1))
                   : "memory");
      unsigned int e_r = _relaxed_ld_2 + 1;
      int g_eff = ((gridDim.x < 8) ? gridDim.x : 8);
      int g_2 = blockIdx.x % g_eff;
      int cta_in_g = blockIdx.x / g_eff;
      int n_g = (gridDim.x - 1 - g_2) / g_eff + 1;
      int limit_r = 6 * n_g - 1;
      const int rw = warp - 9;
      int warp_g = cta_in_g * 6 + rw;
      int warps_g = n_g * 6;
      int n_blocks = grid_n * 8;
      unsigned long long slot_elems = (unsigned long long)grid_n * 16384;
      int slot_r = comm_addr + (unsigned int)(rw * 4096);
      int y_slot_r = slots_r - 1;
      unsigned long long y_off = (unsigned long long)y_slot_r * slot_elems;
#pragma unroll 1
      for (int c_2 = rank_r + g_2 * world_r; c_2 < C; c_2 += g_eff * world_r) {
        if (warp == 9) {
          if (elect_sync()) {
            {
              unsigned int* _sre_ptr_0 =
                  (reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(flags)) + (c_2));
              const unsigned int _sre_expected_0 = static_cast<unsigned int>(world_ru);
              const unsigned long long _sre_start_0 = clock64();
              bool _sre_matched_0 = false;
              do {
                unsigned int _sre_value_0;
                asm volatile("ld.relaxed.sys.u32 %0, [%1];"
                             : "=r"(_sre_value_0)
                             : "l"(_sre_ptr_0)
                             : "memory");
                _sre_matched_0 = (_sre_value_0 == _sre_expected_0);
              } while (!_sre_matched_0 && ((clock64() - _sre_start_0) <=
                                           static_cast<unsigned long long>(17179869184)));
              if (__builtin_expect(!_sre_matched_0, 0)) {
                asm volatile("trap;");
                return;
              }
            }
            asm volatile("fence.acquire.sys;" ::: "memory");
          }
        }
        asm volatile("bar.sync 2, 192;" ::: "memory");
        asm volatile("fence.acquire.sys;" ::: "memory");
        int s_row_r = c_2 / world_r;
        unsigned long long chunk_elem = (unsigned long long)s_row_r * (unsigned long long)slots_r *
                                        (unsigned long long)grid_n * 16384;
#pragma unroll 1
        for (int b = warp_g; b < n_blocks; b += warps_g) {
#pragma unroll
          for (int j = 0; j < 8; j++) {
            int v = (unsigned int)(b * 256 + j * 32) + lane;
            int n_t = v / 2048;
            int w_t = v % 2048;
            int slab = w_t / 1024;
            int r_2 = w_t % 1024 / 8;
            int swz = w_t % 8;
            int col = n_t * 128 + slab * 64 + (swz ^ r_2 & 7) * 8;
            int row_c = c_2 * 128 + r_2;
            unsigned long long s_elem = chunk_elem + (unsigned long long)v * 8;
            unsigned long long o_elem =
                (unsigned long long)row_c * (unsigned long long)N + (unsigned long long)col;
            if (row_c < M) {
              float y[8];
              float pv[8];
#pragma unroll 1
              for (int s = 0; s < world_r; s++) {
                if (s == rank_r) {
                  float _vec_load_34[8];
                  {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(out_ptr + o_elem + 0);
                    uint4 _vld_1[1];
#pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                      asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                   : "=r"(_vld_1[_blk].x), "=r"(_vld_1[_blk].y),
                                     "=r"(_vld_1[_blk].z), "=r"(_vld_1[_blk].w)
                                   : "l"((const void*)(_vptr_1 + _blk))
                                   : "memory");
                      uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
#pragma unroll
                      for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_34[0 + _blk * 8 + _pair * 2])[0] =
                            __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                        (&_vec_load_34[0 + _blk * 8 + _pair * 2])[1] =
                            __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                      }
                    }
                  }
#pragma unroll
                  for (int k = 0; k < 8; k++) {
                    pv[k] = _vec_load_34[k];
                  }
                } else {
                  int k_s = ((rank_r > s) ? s : s - 1);
                  float _vec_load_35[8];
                  {
                    const uint4* _vptr_2 = reinterpret_cast<const uint4*>(
                        reinterpret_cast<__nv_bfloat16*>(S) +
                        (chunk_elem + (unsigned long long)k_s * slot_elems +
                         (unsigned long long)v * 8) +
                        0);
                    uint4 _vld_2[1];
#pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                      asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                   : "=r"(_vld_2[_blk].x), "=r"(_vld_2[_blk].y),
                                     "=r"(_vld_2[_blk].z), "=r"(_vld_2[_blk].w)
                                   : "l"((const void*)(_vptr_2 + _blk))
                                   : "memory");
                      uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
#pragma unroll
                      for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_35[0 + _blk * 8 + _pair * 2])[0] =
                            __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                        (&_vec_load_35[0 + _blk * 8 + _pair * 2])[1] =
                            __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                      }
                    }
                  }
#pragma unroll
                  for (int k_1 = 0; k_1 < 8; k_1++) {
                    pv[k_1] = _vec_load_35[k_1];
                  }
                }
                if (s == 0) {
#pragma unroll
                  for (int k_2 = 0; k_2 < 8; k_2++) {
                    y[k_2] = pv[k_2];
                  }
                } else {
#pragma unroll
                  for (int k_3 = 0; k_3 < 8; k_3++) {
                    y[k_3] = y[k_3] + pv[k_3];
                  }
                }
              }
              {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(y[0 + 0], y[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(y[0 + 2], y[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(y[0 + 4], y[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(y[0 + 6], y[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out_ptr + o_elem))[0]) =
                    *reinterpret_cast<uint4*>(&_pk[0]);
              }
              if (world_r > 1) {
                uint32_t y_bf16[4];
#pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                  __nv_bfloat162 _bf2 =
                      __float22bfloat162_rn(make_float2(y[_lp * 2 + 0], y[_lp * 2 + 1 + 0]));
                  y_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                {
                  __nv_bfloat162 _pk[4];
                  _pk[0] = __floats2bfloat162_rn(y[0 + 0], y[0 + 1]);
                  _pk[1] = __floats2bfloat162_rn(y[0 + 2], y[0 + 3]);
                  _pk[2] = __floats2bfloat162_rn(y[0 + 4], y[0 + 5]);
                  _pk[3] = __floats2bfloat162_rn(y[0 + 6], y[0 + 7]);
                  *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(reinterpret_cast<__nv_bfloat16*>(S) +
                                                               (s_elem + y_off)))[0]) =
                      *reinterpret_cast<uint4*>(&_pk[0]);
                }
              }
            }
          }
        }
        __threadfence_system();
        __syncwarp();
        if (elect_sync()) {
          uint32_t _atomic_inc_old_5;
          asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                       : "=r"(_atomic_inc_old_5)
                       : "l"(&ctl[C_cap + c_2]), "r"(static_cast<uint32_t>(limit_r))
                       : "memory");
          if (_atomic_inc_old_5 == (unsigned int)limit_r) {
            asm volatile(
                "st.relaxed.sys.u32 [%0], %1;" ::"l"(
                    (reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(flags)) +
                     (c_2))),
                "r"(static_cast<unsigned int>(0))
                : "memory");
            __threadfence_system();
            int bc_word = C_cap + c_2;
#pragma unroll 1
            for (int p = 0; p < world_r; p++) {
              if (p != rank_r) {
                asm volatile(
                    "st.release.sys.global.u32 [%0], %1;" ::"l"(flags_peers.ptrs[p] + bc_word),
                    "r"(e_r)
                    : "memory");
              }
            }
          }
        }
      }
      __syncwarp();
      __threadfence();
      if (elect_sync()) {
        int cta_slot_2 = 2 * C_cap + blockIdx.x;
        uint32_t _atomic_inc_old_6;
        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                     : "=r"(_atomic_inc_old_6)
                     : "l"(&ctl[cta_slot_2]), "r"(static_cast<uint32_t>(15))
                     : "memory");
        if (_atomic_inc_old_6 == 15) {
          uint32_t _atomic_inc_old_7;
          asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                       : "=r"(_atomic_inc_old_7)
                       : "l"(&ctl[2 * C_cap + 1024]), "r"(static_cast<uint32_t>(gridDim.x - 1))
                       : "memory");
          if (_atomic_inc_old_7 == (unsigned int)(gridDim.x - 1)) {
            uint32_t _relaxed_ld_3;
            asm volatile("ld.relaxed.gpu.u32 %0, [%1];"
                         : "=r"(_relaxed_ld_3)
                         : "l"(ctl + (2 * C_cap + 1024 + 1))
                         : "memory");
            unsigned int e_fin_2 = _relaxed_ld_3 + 1;
            asm volatile("st.relaxed.sys.u32 [%0], %1;" ::"l"(
                             (reinterpret_cast<unsigned int*>(ctl) + (2 * C_cap + 1024 + 1))),
                         "r"(static_cast<unsigned int>(e_fin_2))
                         : "memory");
          }
        }
      }
    }
    // ---- Role: drainers ----
  } else if (warp == 15) {
    {  // drainers_main
      int rank_d = pg_rank;
      int world_d = pg_world;
      int slots_d = ((world_d > 2) ? world_d : 1);
      uint32_t _relaxed_ld_4;
      asm volatile("ld.relaxed.gpu.u32 %0, [%1];"
                   : "=r"(_relaxed_ld_4)
                   : "l"(ctl + (2 * C_cap + 1024 + 1))
                   : "memory");
      unsigned int e_d = _relaxed_ld_4 + 1;
      const int dw = warp - 15;
      int chunk_vecs = grid_n * 2048;
      int lane_d = (unsigned int)((blockIdx.x + dw) * 32) + lane;
      int lanes_d = gridDim.x * 32;
      int n_blocks_d = grid_n * 8;
      int y_slot_d = slots_d - 1;
      int slot_d = comm_addr + 24576 + (unsigned int)(dw * 4096);
      unsigned int pull_phase = 0;
#pragma unroll 1
      for (int c_3 = 0; c_3 < C; c_3++) {
        if (c_3 % world_d != rank_d) {
          if (warp == 15) {
            if (elect_sync()) {
              {
                unsigned int* _sre_ptr_0 =
                    (reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(flags)) +
                     (C_cap + c_3));
                const unsigned int _sre_expected_0 = static_cast<unsigned int>(e_d);
                const unsigned long long _sre_start_0 = clock64();
                bool _sre_matched_0 = false;
                do {
                  unsigned int _sre_value_0;
                  asm volatile("ld.relaxed.sys.u32 %0, [%1];"
                               : "=r"(_sre_value_0)
                               : "l"(_sre_ptr_0)
                               : "memory");
                  _sre_matched_0 = (_sre_value_0 == _sre_expected_0);
                } while (!_sre_matched_0 && ((clock64() - _sre_start_0) <=
                                             static_cast<unsigned long long>(17179869184)));
                if (__builtin_expect(!_sre_matched_0, 0)) {
                  asm volatile("trap;");
                  return;
                }
              }
              asm volatile("fence.acquire.sys;" ::: "memory");
            }
          }
          __syncwarp();
          asm volatile("fence.acquire.sys;" ::: "memory");
          int owner_d = c_3 % world_d;
          int s_row_d = c_3 / world_d;
          unsigned long long chunk_elem_d =
              ((unsigned long long)s_row_d * (unsigned long long)slots_d +
               (unsigned long long)y_slot_d) *
              (unsigned long long)grid_n * 16384;
#pragma unroll 1
          for (int b_1 = blockIdx.x + dw; b_1 < n_blocks_d; b_1 += gridDim.x) {
            if (elect_sync()) {
              mbarrier_arrive_expect_tx(pull_addr + (dw) * 8, 4096);
              // nvlink_pull: smem(slot_d) <- peers.ptrs[owner_d] + chunk_elem_d * 2 + (unsigned
              // long long)b_1 * 4096, 4096B
              {
                const void* __remote =
                    (const void*)((const char*)((S_peers).ptrs[owner_d]) +
                                  (uint64_t)(chunk_elem_d * 2 + (unsigned long long)b_1 * 4096));
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], %2, [%3];" ::"r"(slot_d),
                    "l"(__remote), "r"((uint32_t)(4096)), "r"(pull_addr + (dw) * 8)
                    : "memory");
              }
            }
            mbarrier_wait(pull_addr + (dw) * 8, pull_phase);
            pull_phase ^= 1;
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#pragma unroll
            for (int j_1 = 0; j_1 < 8; j_1++) {
              int v_1 = (unsigned int)(b_1 * 256 + j_1 * 32) + lane;
              int n_t_1 = v_1 / 2048;
              int w_t_1 = v_1 % 2048;
              int slab_1 = w_t_1 / 1024;
              int r_3 = w_t_1 % 1024 / 8;
              int swz_1 = w_t_1 % 8;
              int col_1 = n_t_1 * 128 + slab_1 * 64 + (swz_1 ^ r_3 & 7) * 8;
              int row_c_1 = c_3 * 128 + r_3;
              if (row_c_1 < M) {
                int words_d[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(*reinterpret_cast<uint32_t*>(&words_d[0])),
                               "=r"(*reinterpret_cast<uint32_t*>(&words_d[(0) + 1])),
                               "=r"(*reinterpret_cast<uint32_t*>(&words_d[(0) + 2])),
                               "=r"(*reinterpret_cast<uint32_t*>(&words_d[(0) + 3]))
                             : "r"((unsigned int)slot_d + ((unsigned int)(j_1 * 32) + lane) * 16));
                unsigned long long o_word = ((unsigned long long)row_c_1 * (unsigned long long)N +
                                             (unsigned long long)col_1) /
                                            2;
                {
                  int4 _iv4 =
                      make_int4(words_d[0 + 0], words_d[0 + 1], words_d[0 + 2], words_d[0 + 3]);
                  *reinterpret_cast<int4*>(reinterpret_cast<int*>(out_ptr) + o_word + 0) = _iv4;
                }
              }
            }
            __syncwarp();
          }
        }
      }
      __syncwarp();
      __threadfence();
      if (elect_sync()) {
        int cta_slot_3 = 2 * C_cap + blockIdx.x;
        uint32_t _atomic_inc_old_8;
        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                     : "=r"(_atomic_inc_old_8)
                     : "l"(&ctl[cta_slot_3]), "r"(static_cast<uint32_t>(15))
                     : "memory");
        if (_atomic_inc_old_8 == 15) {
          uint32_t _atomic_inc_old_9;
          asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                       : "=r"(_atomic_inc_old_9)
                       : "l"(&ctl[2 * C_cap + 1024]), "r"(static_cast<uint32_t>(gridDim.x - 1))
                       : "memory");
          if (_atomic_inc_old_9 == (unsigned int)(gridDim.x - 1)) {
            uint32_t _relaxed_ld_5;
            asm volatile("ld.relaxed.gpu.u32 %0, [%1];"
                         : "=r"(_relaxed_ld_5)
                         : "l"(ctl + (2 * C_cap + 1024 + 1))
                         : "memory");
            unsigned int e_fin_3 = _relaxed_ld_5 + 1;
            asm volatile("st.relaxed.sys.u32 [%0], %1;" ::"l"(
                             (reinterpret_cast<unsigned int*>(ctl) + (2 * C_cap + 1024 + 1))),
                         "r"(static_cast<unsigned int>(e_fin_3))
                         : "memory");
          }
        }
      }
    }
  }

  // Cleanup
}

}  // extern "C"
