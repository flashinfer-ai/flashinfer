/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
#define TMEM_NCOLS 512
#define TMEM_ACCUM_OFFSET 0
#define NUM_TMA_PIPE_STAGES 4
#define NUM_ACC_PIPE_STAGES 2
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_W_OFF 17408
#define SMEM_SMEM_W_STAGE_BYTES 16384
#define SMEM_SMEM_W_STRIDE 32768
#define SMEM_STAGING_OFF 132096
#define SMEM_STAGING_STAGE_BYTES 16384
#define SMEM_STAGING_STRIDE 16384
#define SMEM_COMM_OFF 197632
#define SMEM_COMM_STAGE_BYTES 24576
#define SMEM_COMM_STRIDE 24576
#define SMEM_TOTAL 222208
#define THREADS 384

extern "C" {

__global__ __launch_bounds__(THREADS)
    __cluster_dims__(2, 1, 1) void kernel_cake_fp8_rowwise_gemm_allreduce_40d34f41bf331a3a97a7(
        const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap W,
        const __grid_constant__ CUtensorMap OUT, __nv_bfloat16* __restrict__ out_ptr,
        float* __restrict__ sa, float* __restrict__ sw, unsigned int* __restrict__ ctl,
        unsigned int* __restrict__ S_mc, int M, int N, int K, int grid_n, int pair_tiles,
        int K_tiles, int C, int C_cap, unsigned long long S_bytes, unsigned int flag_words,
        int32_t pg_world, int32_t pg_rank, CakePeerPointerTable<unsigned int> pg_flags,
        uint8_t* __restrict__ S, CakePeerPointerTable<uint8_t> S_peers,
        unsigned int* __restrict__ flags, CakePeerPointerTable<unsigned int> flags_peers) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
#if __CUDA_ARCH__ == 1000
  asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }"
               : "=r"(smem)
               : "l"(smem_raw));
  smem = make_warp_uniform(smem);
#else
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

  const int mbar_base = smem;
#define tma_full_addr (mbar_base + 0)
#define tma_free_addr (mbar_base + 32)
#define mainloop_done_addr (mbar_base + 64)
#define epilogue_done_addr (mbar_base + 80)
#define pull_bar_addr (mbar_base + 96)

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;
  const unsigned int clusters_x = gridDim.x / 2;
  const unsigned int cluster_id =
      ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
  const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

  int cta_rank;
  asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

  // Kernel setup ops
  uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_A_OFF);
  const int smem_a_addr = smem + SMEM_SMEM_A_OFF;
  uint8_t* smem_w = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_W_OFF);
  const int smem_w_addr = smem + SMEM_SMEM_W_OFF;
  __nv_bfloat16* staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_STAGING_OFF);
  const int staging_addr = smem + SMEM_STAGING_OFF;
  uint8_t* comm = reinterpret_cast<uint8_t*>(smem_raw + SMEM_COMM_OFF);
  const int comm_addr = smem + SMEM_COMM_OFF;

  // Mbarrier init (5 pipeline groups, 0 ordered-sequence groups, 14 barriers)
  // Mbarriers at smem_raw[0..112)

  if (warp == 0) {
    uint32_t leader = elect_sync();
    if (leader) {
      // --- pipeline 'tma_pipe' ---
      // tma_full: 4 barriers, init_count=2
      mbarrier_init(smem + 0, 2);
      mbarrier_init(smem + 8, 2);
      mbarrier_init(smem + 16, 2);
      mbarrier_init(smem + 24, 2);
      // tma_free: 4 barriers, init_count=1
      mbarrier_init(smem + 32, 1);
      mbarrier_init(smem + 40, 1);
      mbarrier_init(smem + 48, 1);
      mbarrier_init(smem + 56, 1);
      // --- pipeline 'acc_pipe' ---
      // mainloop_done: 2 barriers, init_count=1
      mbarrier_init(smem + 64, 1);
      mbarrier_init(smem + 72, 1);
      // epilogue_done: 2 barriers, init_count=8
      mbarrier_init(smem + 80, 8);
      mbarrier_init(smem + 88, 8);
      // pull_bar: 2 barriers, init_count=1
      mbarrier_init(smem + 96, 1);
      mbarrier_init(smem + 104, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
  }

  __syncwarp();

  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

  // TMEM alloc (512 columns, 512 used)
  volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 112);
  if (warp == 0) {
    int _tmem_hold = smem + 112;
    asm volatile(
        "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(_tmem_hold),
        "r"(512)
        : "memory");
    __syncwarp();
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
  }

  __syncthreads();
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  const int taddr = tmem_addr_storage[0];

  // Kernel post-init ops
  const int tmem_accum = taddr;

  // ---- Role: epilogue ----
  if (warp <= 3) {
    {  // epilogue_main
      unsigned int acc_stage = 0;
      int cta_rank_e = cta_rank;
      unsigned int it0_e = cluster_id;
      unsigned int step_e = num_clusters;
      int rank_e = pg_rank;
      int world_e = pg_world;
      int slots_e = ((world_e > 2) ? world_e : 1);
      const int epi_warp = warp % 4;
      const int local_row = epi_warp * 32 + lane;
      const int epi_tid = epi_warp * 32 + lane;
      unsigned int _phase_mainloop_done = 0;
#pragma unroll 1
      for (unsigned int t = it0_e; t < pair_tiles; t += step_e) {
        int pair_row = t / (unsigned int)grid_n;
        int bid_n = t % (unsigned int)grid_n;
        int c_tile = pair_row * 2 + cta_rank_e;
        int off_m = pair_row * 256 + cta_rank_e * 128;
        int off_n = bid_n * 256;
        int owner_t = c_tile % world_e;
        int k_e = ((owner_t < rank_e) ? rank_e - 1 : rank_e);
        int row_g = off_m + local_row;
        int safe_row = ((row_g < M) ? row_g : M - 1);
        float sa_row = sa[safe_row];
        mbarrier_wait(mainloop_done_addr + (acc_stage) * 8, _phase_mainloop_done);
        asm volatile("tcgen05.fence::after_thread_sync;");
#pragma unroll
        for (int batch = 0; batch < 4; batch++) {
          int tmem_addr = taddr + (unsigned int)(epi_warp * 32 << 16) +
                          (unsigned int)((int)acc_stage * 256) + (unsigned int)(batch * 64);
          float _tmem_load_0[64];
          asm volatile(
              "tcgen05.ld.sync.aligned.32x32b.x64.b32"
              " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, "
              "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, "
              "%34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, "
              "%50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
              : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]),
                "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]),
                "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]),
                "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]),
                "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]),
                "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]),
                "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]),
                "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]),
                "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]),
                "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]),
                "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31]), "=f"(_tmem_load_0[32]),
                "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]),
                "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]),
                "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]),
                "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]),
                "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]),
                "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]),
                "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]),
                "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]),
                "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]),
                "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]),
                "=f"(_tmem_load_0[63])
              : "r"(tmem_addr));
          asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
          if (batch == 3) {
            asm volatile("tcgen05.fence::before_thread_sync;");
            if (elect_sync()) {
              asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];" ::"r"(
                               (epilogue_done_addr + (acc_stage) * 8) & 0xFEFFFFFF)
                           : "memory");
            }
          }
#pragma unroll
          for (int q = 0; q < 4; q++) {
            int sw_base = off_n + batch * 64 + q * 16;
            float _vec_load_0[4];
            {
              float4 _v4 = *reinterpret_cast<const float4*>(sw + sw_base);
              _vec_load_0[0 + 0] = _v4.x;
              _vec_load_0[0 + 1] = _v4.y;
              _vec_load_0[0 + 2] = _v4.z;
              _vec_load_0[0 + 3] = _v4.w;
            }
            float _vec_load_1[4];
            {
              float4 _v4 = *reinterpret_cast<const float4*>(sw + sw_base + 4);
              _vec_load_1[0 + 0] = _v4.x;
              _vec_load_1[0 + 1] = _v4.y;
              _vec_load_1[0 + 2] = _v4.z;
              _vec_load_1[0 + 3] = _v4.w;
            }
            float _vec_load_2[4];
            {
              float4 _v4 = *reinterpret_cast<const float4*>(sw + sw_base + 8);
              _vec_load_2[0 + 0] = _v4.x;
              _vec_load_2[0 + 1] = _v4.y;
              _vec_load_2[0 + 2] = _v4.z;
              _vec_load_2[0 + 3] = _v4.w;
            }
            float _vec_load_3[4];
            {
              float4 _v4 = *reinterpret_cast<const float4*>(sw + sw_base + 12);
              _vec_load_3[0 + 0] = _v4.x;
              _vec_load_3[0 + 1] = _v4.y;
              _vec_load_3[0 + 2] = _v4.z;
              _vec_load_3[0 + 3] = _v4.w;
            }
            float vals[16];
#pragma unroll
            for (int j = 0; j < 4; j++) {
              vals[j] = _tmem_load_0[q * 16 + j] * sa_row * _vec_load_0[j];
              vals[4 + j] = _tmem_load_0[q * 16 + 4 + j] * sa_row * _vec_load_1[j];
              vals[8 + j] = _tmem_load_0[q * 16 + 8 + j] * sa_row * _vec_load_2[j];
              vals[12 + j] = _tmem_load_0[q * 16 + 12 + j] * sa_row * _vec_load_3[j];
            }
            uint32_t vals_bf16[8];
#pragma unroll
            for (int _lp = 0; _lp < 8; _lp++) {
              __nv_bfloat162 _bf2 =
                  __float22bfloat162_rn(make_float2(vals[_lp * 2 + 0], vals[_lp * 2 + 1 + 0]));
              vals_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile(
                "st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
                    (staging_addr +
                     (unsigned int)((batch * 128 + local_row) * 128 + q * 32 ^
                                    ((batch * 128 + local_row) * 128 + q * 32 >> 7 & 7) << 4))),
                "r"(vals_bf16[0]), "r"(vals_bf16[1]), "r"(vals_bf16[2]), "r"(vals_bf16[3])
                : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
                             (staging_addr +
                              (unsigned int)((batch * 128 + local_row) * 128 + (q * 32 + 16) ^
                                             ((batch * 128 + local_row) * 128 + (q * 32 + 16) >> 7 &
                                              7) << 4))),
                         "r"(vals_bf16[4]), "r"(vals_bf16[5]), "r"(vals_bf16[6]), "r"(vals_bf16[7])
                         : "memory");
          }
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("bar.sync 1, 128;" ::: "memory");
        if (warp == 0) {
          if (elect_sync()) {
            if (c_tile < C) {
              if (owner_t == rank_e) {
#pragma unroll
                for (int j_1 = 0; j_1 < 4; j_1++) {
                  tma_store_3d((&OUT), 0, off_m, off_n / 64 + j_1,
                               staging_addr + (unsigned int)(j_1 * 16384));
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("cp.async.bulk.wait_group 0;");
                asm volatile("fence.proxy.async;");
              } else {
                // nvlink_push: peers.ptrs[owner_t] + (((unsigned long long)c_tile * (unsigned long
                // long)slots_e + (unsigned long long)k_e) * (unsigned long long)grid_n + (unsigned
                // long long)bid_n) * 65536 <- smem(staging_addr), 65536B
                {
                  void* __remote = (void*)((char*)((S_peers).ptrs[owner_t]) +
                                           (uint64_t)((((unsigned long long)c_tile *
                                                            (unsigned long long)slots_e +
                                                        (unsigned long long)k_e) *
                                                           (unsigned long long)grid_n +
                                                       (unsigned long long)bid_n) *
                                                      65536));
                  asm volatile(
                      "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(__remote),
                      "r"(staging_addr), "r"((uint32_t)(65536))
                      : "memory");
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("cp.async.bulk.wait_group 0;");
                asm volatile("fence.proxy.async;");
              }
              uint32_t _atomic_inc_old_4;
              asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                           : "=r"(_atomic_inc_old_4)
                           : "l"(&ctl[c_tile]), "r"(static_cast<uint32_t>(grid_n - 1))
                           : "memory");
              if (_atomic_inc_old_4 == (unsigned int)(grid_n - 1)) {
                __threadfence_system();
                asm volatile(
                    "red.release.sys.global.add.u32 [%0], %1;" ::"l"(
                        reinterpret_cast<unsigned int*>(flags_peers.ptrs[owner_t] + c_tile)),
                    "r"(static_cast<unsigned int>(1))
                    : "memory");
              }
            }
          }
        }
        asm volatile("bar.sync 1, 128;" ::: "memory");
        acc_stage += 1;
        if (acc_stage == 2) {
          acc_stage = 0;
          _phase_mainloop_done ^= 1;
        }
      }
      if (warp == 0) {
        if (elect_sync()) {
          asm volatile("cp.async.bulk.wait_group 0;");
        }
      }
      __syncwarp();
      __threadfence();
      if (elect_sync()) {
        int cta_slot = 2 * C_cap + blockIdx.x;
        uint32_t _atomic_inc_old_5;
        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                     : "=r"(_atomic_inc_old_5)
                     : "l"(&ctl[cta_slot]), "r"(static_cast<uint32_t>(11))
                     : "memory");
        if (_atomic_inc_old_5 == 11) {
          uint32_t _atomic_inc_old_6;
          asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                       : "=r"(_atomic_inc_old_6)
                       : "l"(&ctl[2 * C_cap + 1024]), "r"(static_cast<uint32_t>(gridDim.x - 1))
                       : "memory");
          if (_atomic_inc_old_6 == (unsigned int)(gridDim.x - 1)) {
            uint32_t _relaxed_ld_2;
            asm volatile("ld.relaxed.gpu.u32 %0, [%1];"
                         : "=r"(_relaxed_ld_2)
                         : "l"(ctl + (2 * C_cap + 1024 + 1))
                         : "memory");
            unsigned int e_fin = _relaxed_ld_2 + 1;
            asm volatile("st.relaxed.sys.u32 [%0], %1;" ::"l"(
                             (reinterpret_cast<unsigned int*>(ctl) + (2 * C_cap + 1024 + 1))),
                         "r"(static_cast<unsigned int>(e_fin))
                         : "memory");
          }
        }
      }
    }
  }
  // ---- Role: mma ----
  if (warp == 4) {
    {  // mma_main
      unsigned int tma_stage = 0;
      unsigned int acc_stage_1 = 0;
      unsigned int _phase_epilogue_done = 1;
      unsigned int _phase_tma_full = 0;
      if (cta_rank == 0) {
#pragma unroll 1
        for (unsigned int t_1 = cluster_id; t_1 < pair_tiles; t_1 += num_clusters) {
          mbarrier_wait(epilogue_done_addr + (acc_stage_1) * 8, _phase_epilogue_done);
#pragma unroll 1
          for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
            mbarrier_wait(tma_full_addr + (tma_stage) * 8, _phase_tma_full);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int init_flag = ((iter_k == 0) ? 1 : 0);
            if (elect_sync()) {
              int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (tma_stage) * 2048;
              int _mma_b_lo_0 = (((smem_w_addr) >> 4) & 0x3FFF) + (tma_stage) * 2048;
              asm volatile(
                  "{\n\t"
                  ".reg .pred p0, p1;\n\t"
                  ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                  ".reg .b64 da, db;\n\t"
                  ""
                  "setp.ne.b32 p0, %3, 0;\n\t"
                  "setp.ne.b32 p1, 1, 0;\n\t"
                  "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; "
                  "mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                  "mov.b32 adhi, 0x40004040;\n\t"
                  "mov.b32 bdhi, 0x40004040;\n\t"
                  "mov.b32 id, 272629776;\n\t"
                  "mov.b32 alo, %0;\n\t"
                  "mov.b32 blo, %1;\n\t"
                  "mov.b64 da, {alo, adhi};\n\t"
                  "mov.b64 db, {blo, bdhi};\n\t"
                  "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, "
                  "m5, m6, m7}, p0;\n\t"
                  "add.u32 alo, alo, 2;\n\t"
                  "add.u32 blo, blo, 2;\n\t"
                  "mov.b64 da, {alo, adhi};\n\t"
                  "mov.b64 db, {blo, bdhi};\n\t"
                  "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, "
                  "m5, m6, m7}, p1;\n\t"
                  "add.u32 alo, alo, 2;\n\t"
                  "add.u32 blo, blo, 2;\n\t"
                  "mov.b64 da, {alo, adhi};\n\t"
                  "mov.b64 db, {blo, bdhi};\n\t"
                  "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, "
                  "m5, m6, m7}, p1;\n\t"
                  "add.u32 alo, alo, 2;\n\t"
                  "add.u32 blo, blo, 2;\n\t"
                  "mov.b64 da, {alo, adhi};\n\t"
                  "mov.b64 db, {blo, bdhi};\n\t"
                  "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, "
                  "m5, m6, m7}, p1;\n\t"
                  "}\n" ::"r"(_mma_a_lo_0),
                  "r"(_mma_b_lo_0), "r"((tmem_accum + (acc_stage_1 * 256))),
                  "r"(((init_flag) ? 0 : 1)));
            }
            elect_commit_cg2_multicast(tma_free_addr + (tma_stage) * 8, (uint16_t)(3));
            tma_stage += 1;
            if (tma_stage == 4) {
              tma_stage = 0;
              _phase_tma_full ^= 1;
            }
          }
          elect_commit_cg2_multicast(mainloop_done_addr + (acc_stage_1) * 8, (uint16_t)(3));
          acc_stage_1 += 1;
          if (acc_stage_1 == 2) {
            acc_stage_1 = 0;
            _phase_epilogue_done ^= 1;
          }
        }
      }
      __syncwarp();
      __threadfence();
      if (elect_sync()) {
        int cta_slot_1 = 2 * C_cap + blockIdx.x;
        uint32_t _atomic_inc_old_2;
        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                     : "=r"(_atomic_inc_old_2)
                     : "l"(&ctl[cta_slot_1]), "r"(static_cast<uint32_t>(11))
                     : "memory");
        if (_atomic_inc_old_2 == 11) {
          uint32_t _atomic_inc_old_3;
          asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                       : "=r"(_atomic_inc_old_3)
                       : "l"(&ctl[2 * C_cap + 1024]), "r"(static_cast<uint32_t>(gridDim.x - 1))
                       : "memory");
          if (_atomic_inc_old_3 == (unsigned int)(gridDim.x - 1)) {
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
  }
  // ---- Role: load ----
  if (warp == 5) {
    {  // load_main
      int cta_rank_l = cta_rank;
      unsigned int it0_l = cluster_id;
      unsigned int step_l = num_clusters;
      unsigned int load_stage = 0;
      if (elect_sync()) {
        asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)((&A))) : "memory");
        asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)((&W))) : "memory");
        asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)((&OUT))) : "memory");
      }
      unsigned int _phase_tma_free = 1;
#pragma unroll 1
      for (unsigned int t_2 = it0_l; t_2 < pair_tiles; t_2 += step_l) {
        int pair_row_1 = t_2 / (unsigned int)grid_n;
        int bid_n_1 = t_2 % (unsigned int)grid_n;
        int c_tile_1 = pair_row_1 * 2 + cta_rank_l;
        int off_m_1 = pair_row_1 * 256 + cta_rank_l * 128;
        int off_w = bid_n_1 * 256 + cta_rank_l * 128;
#pragma unroll 1
        for (int iter_k_1 = 0; iter_k_1 < K_tiles; iter_k_1++) {
          mbarrier_wait(tma_free_addr + (load_stage) * 8, _phase_tma_free);
          if (elect_sync()) {
            tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 32768, (&A), 0, off_m_1, iter_k_1,
                                  ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
            tma_3d_gmem2smem_cta2(smem_w_addr + load_stage * 32768, (&W), 0, off_w, iter_k_1,
                                  ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
            asm volatile(
                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;" ::"r"(
                    (tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF),
                "r"((uint32_t)(32768))
                : "memory");
          }
          load_stage += 1;
          if (load_stage == 4) {
            load_stage = 0;
            _phase_tma_free ^= 1;
          }
        }
      }
      __syncwarp();
      __threadfence();
      if (elect_sync()) {
        int cta_slot_2 = 2 * C_cap + blockIdx.x;
        uint32_t _atomic_inc_old_0;
        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                     : "=r"(_atomic_inc_old_0)
                     : "l"(&ctl[cta_slot_2]), "r"(static_cast<uint32_t>(11))
                     : "memory");
        if (_atomic_inc_old_0 == 11) {
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
            unsigned int e_fin_2 = _relaxed_ld_0 + 1;
            asm volatile("st.relaxed.sys.u32 [%0], %1;" ::"l"(
                             (reinterpret_cast<unsigned int*>(ctl) + (2 * C_cap + 1024 + 1))),
                         "r"(static_cast<unsigned int>(e_fin_2))
                         : "memory");
          }
        }
      }
    }
  }
  // ---- Role: reducers ----
  if (warp >= 6 && warp <= 9) {
    {  // reducers_main
      int rank_r = pg_rank;
      int world_r = pg_world;
      unsigned int world_ru = (unsigned int)world_r;
      int slots_r = ((world_r > 2) ? world_r : 1);
      uint32_t _relaxed_ld_3;
      asm volatile("ld.relaxed.gpu.u32 %0, [%1];"
                   : "=r"(_relaxed_ld_3)
                   : "l"(ctl + (2 * C_cap + 1024 + 1))
                   : "memory");
      unsigned int e_r = _relaxed_ld_3 + 1;
      int g_eff = ((gridDim.x < 4) ? gridDim.x : 4);
      int g = blockIdx.x % g_eff;
      int cta_in_g = blockIdx.x / g_eff;
      int n_g = (gridDim.x - 1 - g) / g_eff + 1;
      int limit_r = 4 * n_g - 1;
      const int rw = warp - 6;
      int warp_g = cta_in_g * 4 + rw;
      int warps_g = n_g * 4;
      int n_blocks = grid_n * 16;
      unsigned long long slot_elems = (unsigned long long)grid_n * 32768;
      int slot_r = comm_addr + (unsigned int)(rw * 4096);
      int y_slot_r = slots_r - 1;
      unsigned long long y_off = (unsigned long long)y_slot_r * slot_elems;
#pragma unroll 1
      for (int c = rank_r + g * world_r; c < C; c += g_eff * world_r) {
        if (warp == 6) {
          if (elect_sync()) {
            {
              unsigned int* _sre_ptr_0 =
                  (reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(flags)) + (c));
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
        asm volatile("bar.sync 2, 128;" ::: "memory");
        asm volatile("fence.acquire.sys;" ::: "memory");
        unsigned long long chunk_elem = (unsigned long long)c * (unsigned long long)slots_r *
                                        (unsigned long long)grid_n * 32768;
#pragma unroll 1
        for (int b = warp_g; b < n_blocks; b += warps_g) {
#pragma unroll
          for (int w = 0; w < 4; w++) {
            int rows_w[2];
            unsigned long long o_w[2];
            unsigned long long s_w[2];
            float y[16];
            float pv[16];
#pragma unroll
            for (int i = 0; i < 2; i++) {
              int v = b * 256 + (w * 2 + i) * 32 + lane;
              int n_t = v / 4096;
              int w_t = v % 4096;
              int slab = w_t / 1024;
              int r = w_t % 1024 / 8;
              int swz = w_t % 8;
              int col = n_t * 256 + slab * 64 + (swz ^ r & 7) * 8;
              int row_c = c * 128 + r;
              rows_w[i] = row_c;
              s_w[i] = chunk_elem + (unsigned long long)v * 8;
              o_w[i] = (unsigned long long)row_c * (unsigned long long)N + (unsigned long long)col;
            }
            if (0 == rank_r) {
#pragma unroll
              for (int i_1 = 0; i_1 < 2; i_1++) {
                int row_i = rows_w[i_1];
                if (row_i < M) {
                  unsigned long long o_i = o_w[i_1];
                  float _vec_load_4[8];
                  {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(out_ptr + o_i + 0);
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
                        (&_vec_load_4[0 + _blk * 8 + _pair * 2])[0] =
                            __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                        (&_vec_load_4[0 + _blk * 8 + _pair * 2])[1] =
                            __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                      }
                    }
                  }
#pragma unroll
                  for (int k = 0; k < 8; k++) {
                    y[i_1 * 8 + k] = _vec_load_4[k];
                  }
                }
              }
            } else {
              int k_s = ((rank_r > 0) ? 0 : -1);
#pragma unroll
              for (int i_2 = 0; i_2 < 2; i_2++) {
                int row_i_1 = rows_w[i_2];
                if (row_i_1 < M) {
                  unsigned long long s_i = s_w[i_2];
                  float _vec_load_5[8];
                  {
                    const uint4* _vptr_2 = reinterpret_cast<const uint4*>(
                        reinterpret_cast<__nv_bfloat16*>(S) +
                        (s_i + (unsigned long long)k_s * slot_elems) + 0);
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
                        (&_vec_load_5[0 + _blk * 8 + _pair * 2])[0] =
                            __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                        (&_vec_load_5[0 + _blk * 8 + _pair * 2])[1] =
                            __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                      }
                    }
                  }
#pragma unroll
                  for (int k_1 = 0; k_1 < 8; k_1++) {
                    y[i_2 * 8 + k_1] = _vec_load_5[k_1];
                  }
                }
              }
            }
#pragma unroll 1
            for (int s = 1; s < world_r; s++) {
              if (s == rank_r) {
#pragma unroll
                for (int i_3 = 0; i_3 < 2; i_3++) {
                  int row_i_2 = rows_w[i_3];
                  if (row_i_2 < M) {
                    unsigned long long o_i_1 = o_w[i_3];
                    float _vec_load_6[8];
                    {
                      const uint4* _vptr_3 = reinterpret_cast<const uint4*>(out_ptr + o_i_1 + 0);
                      uint4 _vld_3[1];
#pragma unroll
                      for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                     : "=r"(_vld_3[_blk].x), "=r"(_vld_3[_blk].y),
                                       "=r"(_vld_3[_blk].z), "=r"(_vld_3[_blk].w)
                                     : "l"((const void*)(_vptr_3 + _blk))
                                     : "memory");
                        uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
#pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                          (&_vec_load_6[0 + _blk * 8 + _pair * 2])[0] =
                              __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                          (&_vec_load_6[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(
                              static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                        }
                      }
                    }
#pragma unroll
                    for (int k_2 = 0; k_2 < 8; k_2++) {
                      pv[i_3 * 8 + k_2] = _vec_load_6[k_2];
                    }
                  }
                }
              } else {
                int k_s_1 = ((rank_r > s) ? s : s - 1);
#pragma unroll
                for (int i_4 = 0; i_4 < 2; i_4++) {
                  int row_i_3 = rows_w[i_4];
                  if (row_i_3 < M) {
                    unsigned long long s_i_1 = s_w[i_4];
                    float _vec_load_7[8];
                    {
                      const uint4* _vptr_4 = reinterpret_cast<const uint4*>(
                          reinterpret_cast<__nv_bfloat16*>(S) +
                          (s_i_1 + (unsigned long long)k_s_1 * slot_elems) + 0);
                      uint4 _vld_4[1];
#pragma unroll
                      for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                     : "=r"(_vld_4[_blk].x), "=r"(_vld_4[_blk].y),
                                       "=r"(_vld_4[_blk].z), "=r"(_vld_4[_blk].w)
                                     : "l"((const void*)(_vptr_4 + _blk))
                                     : "memory");
                        uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
#pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                          (&_vec_load_7[0 + _blk * 8 + _pair * 2])[0] =
                              __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                          (&_vec_load_7[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(
                              static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                        }
                      }
                    }
#pragma unroll
                    for (int k_3 = 0; k_3 < 8; k_3++) {
                      pv[i_4 * 8 + k_3] = _vec_load_7[k_3];
                    }
                  }
                }
              }
#pragma unroll
              for (int i_5 = 0; i_5 < 2; i_5++) {
                int row_a = rows_w[i_5];
                if (row_a < M) {
#pragma unroll
                  for (int k_4 = 0; k_4 < 8; k_4++) {
                    y[i_5 * 8 + k_4] = y[i_5 * 8 + k_4] + pv[i_5 * 8 + k_4];
                  }
                }
              }
            }
#pragma unroll
            for (int i_6 = 0; i_6 < 2; i_6++) {
              int row_s = rows_w[i_6];
              if (row_s < M) {
                unsigned long long o_s = o_w[i_6];
                unsigned long long s_s = s_w[i_6];
                float y_i[8];
#pragma unroll
                for (int k_5 = 0; k_5 < 8; k_5++) {
                  y_i[k_5] = y[i_6 * 8 + k_5];
                }
                {
                  __nv_bfloat162 _pk[4];
                  _pk[0] = __floats2bfloat162_rn(y_i[0 + 0], y_i[0 + 1]);
                  _pk[1] = __floats2bfloat162_rn(y_i[0 + 2], y_i[0 + 3]);
                  _pk[2] = __floats2bfloat162_rn(y_i[0 + 4], y_i[0 + 5]);
                  _pk[3] = __floats2bfloat162_rn(y_i[0 + 6], y_i[0 + 7]);
                  *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out_ptr + o_s))[0]) =
                      *reinterpret_cast<uint4*>(&_pk[0]);
                }
                if (world_r > 1) {
                  uint32_t y_i_bf16[4];
#pragma unroll
                  for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 =
                        __float22bfloat162_rn(make_float2(y_i[_lp * 2 + 0], y_i[_lp * 2 + 1 + 0]));
                    y_i_bf16[_lp] = *(uint32_t*)&_bf2;
                  }
                  {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(y_i[0 + 0], y_i[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(y_i[0 + 2], y_i[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(y_i[0 + 4], y_i[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(y_i[0 + 6], y_i[0 + 7]);
                    *reinterpret_cast<uint4*>(&(
                        (__nv_bfloat16*)(reinterpret_cast<__nv_bfloat16*>(S) + (s_s + y_off)))[0]) =
                        *reinterpret_cast<uint4*>(&_pk[0]);
                  }
                }
              }
            }
          }
        }
        __threadfence_system();
        __syncwarp();
        if (elect_sync()) {
          uint32_t _atomic_inc_old_7;
          asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                       : "=r"(_atomic_inc_old_7)
                       : "l"(&ctl[C_cap + c]), "r"(static_cast<uint32_t>(limit_r))
                       : "memory");
          if (_atomic_inc_old_7 == (unsigned int)limit_r) {
            asm volatile(
                "st.relaxed.sys.u32 [%0], %1;" ::"l"((
                    reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(flags)) + (c))),
                "r"(static_cast<unsigned int>(0))
                : "memory");
            __threadfence_system();
            int bc_word = C_cap + c;
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
        int cta_slot_3 = 2 * C_cap + blockIdx.x;
        uint32_t _atomic_inc_old_8;
        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                     : "=r"(_atomic_inc_old_8)
                     : "l"(&ctl[cta_slot_3]), "r"(static_cast<uint32_t>(11))
                     : "memory");
        if (_atomic_inc_old_8 == 11) {
          uint32_t _atomic_inc_old_9;
          asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                       : "=r"(_atomic_inc_old_9)
                       : "l"(&ctl[2 * C_cap + 1024]), "r"(static_cast<uint32_t>(gridDim.x - 1))
                       : "memory");
          if (_atomic_inc_old_9 == (unsigned int)(gridDim.x - 1)) {
            uint32_t _relaxed_ld_4;
            asm volatile("ld.relaxed.gpu.u32 %0, [%1];"
                         : "=r"(_relaxed_ld_4)
                         : "l"(ctl + (2 * C_cap + 1024 + 1))
                         : "memory");
            unsigned int e_fin_3 = _relaxed_ld_4 + 1;
            asm volatile("st.relaxed.sys.u32 [%0], %1;" ::"l"(
                             (reinterpret_cast<unsigned int*>(ctl) + (2 * C_cap + 1024 + 1))),
                         "r"(static_cast<unsigned int>(e_fin_3))
                         : "memory");
          }
        }
      }
    }
  }
  // ---- Role: drainers ----
  if (warp >= 10 && warp <= 11) {
    {  // drainers_main
      int rank_d = pg_rank;
      int world_d = pg_world;
      int slots_d = ((world_d > 2) ? world_d : 1);
      uint32_t _relaxed_ld_5;
      asm volatile("ld.relaxed.gpu.u32 %0, [%1];"
                   : "=r"(_relaxed_ld_5)
                   : "l"(ctl + (2 * C_cap + 1024 + 1))
                   : "memory");
      unsigned int e_d = _relaxed_ld_5 + 1;
      const int dw = warp - 10;
      int chunk_vecs = grid_n * 4096;
      int lane_d = (blockIdx.x * 2 + dw) * 32 + lane;
      int lanes_d = gridDim.x * 64;
      int n_blocks_d = grid_n * 16;
      int slot_d = comm_addr + 16384 + (unsigned int)(dw * 4096);
      int y_slot_d = slots_d - 1;
      unsigned int pull_phase = 0;
#pragma unroll 1
      for (int c_1 = 0; c_1 < C; c_1++) {
        if (c_1 % world_d != rank_d) {
          if (warp == 10) {
            if (elect_sync()) {
              {
                unsigned int* _sre_ptr_0 =
                    (reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(flags)) +
                     (C_cap + c_1));
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
          asm volatile("bar.sync 3, 64;" ::: "memory");
          asm volatile("fence.acquire.sys;" ::: "memory");
          unsigned long long chunk_elem_d = ((unsigned long long)c_1 * (unsigned long long)slots_d +
                                             (unsigned long long)y_slot_d) *
                                            (unsigned long long)grid_n * 32768;
          int owner_d = c_1 % world_d;
#pragma unroll 1
          for (int b_1 = blockIdx.x * 2 + dw; b_1 < n_blocks_d; b_1 += gridDim.x * 2) {
            if (elect_sync()) {
              mbarrier_arrive_expect_tx(pull_bar_addr + (dw) * 8, 4096);
              // nvlink_pull: smem(slot_d) <- peers.ptrs[owner_d] + chunk_elem_d * 2 + (unsigned
              // long long)b_1 * 4096, 4096B
              {
                const void* __remote =
                    (const void*)((const char*)((S_peers).ptrs[owner_d]) +
                                  (uint64_t)(chunk_elem_d * 2 + (unsigned long long)b_1 * 4096));
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], %2, [%3];" ::"r"(slot_d),
                    "l"(__remote), "r"((uint32_t)(4096)), "r"(pull_bar_addr + (dw) * 8)
                    : "memory");
              }
            }
            mbarrier_wait(pull_bar_addr + (dw) * 8, pull_phase);
            pull_phase ^= 1;
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#pragma unroll
            for (int j_2 = 0; j_2 < 8; j_2++) {
              int v_1 = b_1 * 256 + j_2 * 32 + lane;
              int n_t_1 = v_1 / 4096;
              int w_t_1 = v_1 % 4096;
              int slab_1 = w_t_1 / 1024;
              int r_1 = w_t_1 % 1024 / 8;
              int swz_1 = w_t_1 % 8;
              int col_1 = n_t_1 * 256 + slab_1 * 64 + (swz_1 ^ r_1 & 7) * 8;
              int row_c_1 = c_1 * 128 + r_1;
              if (row_c_1 < M) {
                int words_d[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(*reinterpret_cast<uint32_t*>(&words_d[0])),
                               "=r"(*reinterpret_cast<uint32_t*>(&words_d[(0) + 1])),
                               "=r"(*reinterpret_cast<uint32_t*>(&words_d[(0) + 2])),
                               "=r"(*reinterpret_cast<uint32_t*>(&words_d[(0) + 3]))
                             : "r"(slot_d + (j_2 * 32 + lane) * 16));
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
        int cta_slot_4 = 2 * C_cap + blockIdx.x;
        uint32_t _atomic_inc_old_10;
        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                     : "=r"(_atomic_inc_old_10)
                     : "l"(&ctl[cta_slot_4]), "r"(static_cast<uint32_t>(11))
                     : "memory");
        if (_atomic_inc_old_10 == 11) {
          uint32_t _atomic_inc_old_11;
          asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                       : "=r"(_atomic_inc_old_11)
                       : "l"(&ctl[2 * C_cap + 1024]), "r"(static_cast<uint32_t>(gridDim.x - 1))
                       : "memory");
          if (_atomic_inc_old_11 == (unsigned int)(gridDim.x - 1)) {
            uint32_t _relaxed_ld_6;
            asm volatile("ld.relaxed.gpu.u32 %0, [%1];"
                         : "=r"(_relaxed_ld_6)
                         : "l"(ctl + (2 * C_cap + 1024 + 1))
                         : "memory");
            unsigned int e_fin_4 = _relaxed_ld_6 + 1;
            asm volatile("st.relaxed.sys.u32 [%0], %1;" ::"l"(
                             (reinterpret_cast<unsigned int*>(ctl) + (2 * C_cap + 1024 + 1))),
                         "r"(static_cast<unsigned int>(e_fin_4))
                         : "memory");
          }
        }
      }
    }
  }

  // Cleanup
  asm volatile("barrier.cluster.arrive.release.aligned;");
  asm volatile("barrier.cluster.wait.acquire.aligned;");

  if (warp == 0) {
    asm volatile(
        "tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(tmem_addr_storage[0]),
        "r"(512));
  }
}

}  // extern "C"
