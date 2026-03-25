#include <cudaTypedefs.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cassert>
#include <cstdint>
#include <flashinfer/dsv3_ops/common.cuh>
#include <flashinfer/dsv3_ops/tcgen05_utils.cuh>

#define KBLOCK 128

// The algorithm is as follows:
// Each threadblock handles a tile Q[q_next * 64, 128] and a range of pages
// [page_offset, page_offset + num_pages) in the kv cache First Q is loaded into
// shared memory Each k_group, which gets a warp_group, a mma_warp, a ldg_k warp
// and a k_scales warp processes 2 pages of K. The ldg_k warp loads 2 pages of K
// into shared memory, packed contiguously to form a K-tile of shape [128, 128].
// (the page size is 64) Then the mma warp computes K @ Q to get tile of shape
// [128, q_next * 64] finally the epilogue reduces over each q_next tile of
// shape [128, 64] along the second dim to get the logits

template <bool PDL_ENABLED, int K_LDG_STAGES, int K_LDG_SCALES_STAGES, int K_MMA_STAGES,
          int EPILOGUE_WARPGRPS, int VECTORIZE_TCGEN_LD, int PRELOAD_TCGEN_REGS, int Q_NEXT>
__device__ void mqa_v2_kernel(
    const CUtensorMap* Q_tmap,          // f8[BatchSize, q_next * 64, 128],
                                        // strides=[q_next * 64 * 128, 128]
    const CUtensorMap* K_tmap,          // f8[page_table_pages, 64, 128], strides=[64 * 132, 128, 1]
    const uint8_t* K_ptr,               // [page_table_pages, 64 * 132]
    const float* __restrict__ weights,  // [q_next * 64]
    const int q_idx, const int seq_len, const int page_offset,
    const int num_pages,  // the number of pages in kv_cache that this
                          // threadblock handles
    const int max_num_pages, const int page_table_pages,
    const int* __restrict__ block_table,     // [max_num_pages]
    float* __restrict__ out_logits,          // [max_num_pages * 64]
    uint32_t* __restrict__ global_histogram  // [q_next, 256]

) {
  constexpr int THREADS = EPILOGUE_WARPGRPS * 7 * 32;
  static_assert(THREADS <= 1024, "too many threads");
  constexpr int TMEM_COLS = 64 * Q_NEXT * K_MMA_STAGES * EPILOGUE_WARPGRPS;
  static_assert(TMEM_COLS <= 512, "too much tensor memory");
  constexpr bool DEBUG = false;

  const int seq_end = min(seq_len, (page_offset + num_pages) * 64);

  // put on uniform registers
  const auto& warp_id = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
  const auto& warp_grp_id = warp_id / 4;
  const auto& lane_id = threadIdx.x % 32;

  __shared__ uint64_t K_prod_mbars[EPILOGUE_WARPGRPS][K_LDG_STAGES];  // signals that we are done
                                                                      // with loading into K
  __shared__ uint64_t
      K_scales_prod_mbars[EPILOGUE_WARPGRPS][K_LDG_SCALES_STAGES];  // signals that we are done
                                                                    // with loading into k scales

  __shared__ uint64_t MMA_mbars[EPILOGUE_WARPGRPS][K_MMA_STAGES];  // signals we are done computing
                                                                   // mma, eg, done with tmem
  __shared__ uint64_t
      Epilogue_mbars[EPILOGUE_WARPGRPS][K_MMA_STAGES];  // signals we are done with tmem
  __shared__ uint64_t q_bar;

  alignas(128) __shared__ float q_weights[Q_NEXT][64];
  __shared__ uint32_t histogram[Q_NEXT][256];

  alignas(1024) extern __shared__ uint8_t
      s_buf[];  // Q: 128 * 64 * Q_NEXT + K: 128 * 128 * EPILOGUE_WARPGRPS *
                // K_LDG_STAGES + scales: sizeof(float) * 128 * K_LDG_STAGES *
                // EPILOGUE_WARPGRPS;

  uint8_t* s_q_buf = s_buf;
  uint8_t* s_k_bufs = s_buf + 128 * 64 * Q_NEXT;
  float* k_scales = (float*)(s_k_bufs + 128 * 128 * EPILOGUE_WARPGRPS * K_LDG_STAGES);

  // epilogue warps: 0 to EPILOGUE_WARPGRPS * 4
  // ldg warps: EPILOGUE_WARPGRPS * 4 to EPILOGUE_WARPGRPS * 4 +
  // EPILOGUE_WARPGRPS mma warps: EPILOGUE_WARPGRPS * 4 + EPILOGUE_WARPGRPS to
  // EPILOGUE_WARPGRPS * 4 + 2 * EPILOGUE_WARPGRPS

  constexpr int ldg_warp_offset = EPILOGUE_WARPGRPS * 4;
  constexpr int ldg_scale_warp_offset = EPILOGUE_WARPGRPS * 5;
  constexpr int mma_warp_offset = EPILOGUE_WARPGRPS * 6;

  const bool is_epilogue_warp = warp_id < ldg_warp_offset;
  const bool is_ldg_warp = ldg_warp_offset <= warp_id && warp_id < ldg_scale_warp_offset;
  const bool is_ldg_scales_warp = ldg_scale_warp_offset <= warp_id && warp_id < mma_warp_offset;
  const bool is_mma_warp = mma_warp_offset <= warp_id && warp_id < EPILOGUE_WARPGRPS * 7;

  // always between 0 and EPILOGUE_WARPGRPS
  const auto& k_group = is_epilogue_warp
                            ? warp_grp_id
                            : (is_ldg_warp ? warp_id - ldg_warp_offset
                                           : (is_ldg_scales_warp ? warp_id - ldg_scale_warp_offset
                                                                 : warp_id - mma_warp_offset));

  __shared__ int tmem_addr[1];

  if (is_ldg_scales_warp) {
    if (lane_id < K_LDG_STAGES) {
      mbarrier_init(&K_prod_mbars[k_group][lane_id], 1);
      asm volatile("fence.mbarrier_init.release.cluster;");
    } else if (lane_id < K_LDG_STAGES + K_LDG_SCALES_STAGES) {
      mbarrier_init(&K_scales_prod_mbars[k_group][lane_id - K_LDG_STAGES], 1);
      asm volatile("fence.mbarrier_init.release.cluster;");
    }

  } else if (is_mma_warp) {
    if (lane_id < K_MMA_STAGES) {
      mbarrier_init(&MMA_mbars[k_group][lane_id], 1);
      mbarrier_init(&Epilogue_mbars[k_group][lane_id],
                    32 * 4);  // a warp group arrives
      asm volatile("fence.mbarrier_init.release.cluster;");
    }
  } else if (warp_id == 0) {
    if (elect_sync()) {
      mbarrier_init(&q_bar, 1);
      tma_3d_gmem2smem(s_q_buf, Q_tmap, 0, 0, q_idx, &q_bar);
      tma_1d_gmem2smem(weights, &q_weights[0][0], 64 * Q_NEXT * sizeof(float), &q_bar);
      asm volatile("fence.mbarrier_init.release.cluster;");
      mbarrier_arrive(&q_bar, Q_NEXT * (64 * 128 + 64 * sizeof(float)));
    }

  } else if (warp_id == 1) {
    const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(addr),
                 "r"(TMEM_COLS));
  } else if (warp_id >= 2 && warp_id < ldg_scale_warp_offset) {
    int local_tid = threadIdx.x - 2 * 32;
    constexpr int local_stride = (ldg_scale_warp_offset - 2) * 32;
    for (int i = local_tid; i < 256 * Q_NEXT; i += local_stride) {
      histogram[i / 256][i % 256] = 0;
    }
  }

  constexpr int num_epilogue_threads = 32 * EPILOGUE_WARPGRPS * 4;
  constexpr int num_special_warps = 32 * EPILOGUE_WARPGRPS * 3;
  constexpr int math_nreg = 8 * 22;
  constexpr int special_nreg = 32;
  static_assert(num_epilogue_threads * math_nreg + num_special_warps * special_nreg <= 65536,
                "too many registers");

  __syncthreads();  // first make qbar visible to all threads, also sets
  // histogram to 0
  const int taddr = tmem_addr[0];

  const int k_grp_taddr = taddr + k_group * K_MMA_STAGES * Q_NEXT * 64;
  const auto& k_group_s_buf = s_k_bufs + k_group * 128 * 128 * K_LDG_STAGES;
  const auto& k_group_scales_buf = k_scales + k_group * K_LDG_STAGES * 128;

  constexpr uint32_t idesc = (1U << 4U)  // dtype f32
                                         // | (1U << 16U)          // transpose B
                             | ((((uint32_t)Q_NEXT * 64) >> 3U) << 17U)  // dim B = Q_NEXT * 64
                             | ((128U >> 4U) << 24U)                     // dim A = 128
                                                      // f8 type is encoded by 0, so no shifts
      ;
  auto make_desc_a = [](int addr) -> uint64_t {
    // K-major, 128B swizzling
    // `((8,8),(16,8)):((128,SBO)(1,16))` (f8, K=128, T=16, m=64, SBO=128x8, )
    const int SBO = 128 * 8;
    return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
  };

  // each cta takes care of [page_offset, page_offset+num_pages) pages

  const int max_logical_pages = divup(seq_len, 64);
  if (is_ldg_warp) {
    if (elect_sync()) {
      int mma_phase = 0;
      for (int i = 0; i < divup(num_pages, EPILOGUE_WARPGRPS * 2); i++) {
        int ldg_stage = i % K_LDG_STAGES;
        int logical_page = page_offset + i * EPILOGUE_WARPGRPS * 2 + k_group * 2;

        int2 physical_pages;
        if (logical_page + 1 < max_logical_pages) {
          // TODO: ensure page-offset is aligned to 2
          // physical_pages = *(int2 *)(block_table + logical_page);
          physical_pages.x = block_table[logical_page];
          physical_pages.y = block_table[logical_page + 1];
        } else if (logical_page < max_logical_pages) {
          physical_pages.x = block_table[logical_page];
          physical_pages.y = page_table_pages;
        } else {
          physical_pages.x = page_table_pages;
          physical_pages.y = page_table_pages;
        }

        if (DEBUG)
          printf(
              "pre K producer: loading %d, ldg_stage %d, mma_phase %d, "
              "physical_pages %d %d\n",
              i, ldg_stage, mma_phase, physical_pages.x, physical_pages.y);
        if (i >= K_LDG_STAGES) {
          int mma_stage = (i - K_LDG_STAGES) % K_MMA_STAGES;
          mbarrier_wait(&MMA_mbars[k_group][mma_stage], mma_phase);
          mma_stage = (mma_stage + 1) % K_MMA_STAGES;
          if (mma_stage == 0) {
            mma_phase ^= 1;
          }
        }

        tma_3d_gmem2smem((k_group_s_buf + ldg_stage * 128 * 128), K_tmap, 0, 0, physical_pages.x,
                         &K_prod_mbars[k_group][ldg_stage]);
        tma_3d_gmem2smem((k_group_s_buf + ldg_stage * 128 * 128 + 64 * 128), K_tmap, 0, 0,
                         physical_pages.y, &K_prod_mbars[k_group][ldg_stage]);

        asm volatile(
            "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, "
            "[%0], %1;" ::"l"(&K_prod_mbars[k_group][ldg_stage]),
            "r"(128 * 128)
            : "memory");

        if (DEBUG)
          printf(
              "post K producer: loading %d, ldg_stage %d, mma_phase %d, "
              "physical_pages %d %d\n",
              i, ldg_stage, mma_phase, physical_pages.x, physical_pages.y);
      }
    }
  } else if (is_ldg_scales_warp) {  // ldg warp k scales

    if (elect_sync()) {
      int epilogue_phase = 0;

      for (int i = 0; i < divup(num_pages, EPILOGUE_WARPGRPS * 2); i++) {
        int ldg_stage = i % K_LDG_SCALES_STAGES;
        int logical_page = page_offset + i * EPILOGUE_WARPGRPS * 2 + k_group * 2;

        if (DEBUG)
          printf(
              "pre K scales producer: loading %d, ldg_stage %d, "
              "epilogue_phase %d, logical_page %d\n",
              i, ldg_stage, epilogue_phase, logical_page);

        if (i >= K_LDG_SCALES_STAGES) {
          int epilogue_stage = (i - K_LDG_SCALES_STAGES) % K_MMA_STAGES;
          mbarrier_wait(&Epilogue_mbars[k_group][epilogue_stage], epilogue_phase);
          epilogue_stage = (epilogue_stage + 1) % K_MMA_STAGES;
          if (epilogue_stage == 0) epilogue_phase ^= 1;
        }

        int num_bytes;
        if (logical_page + 1 < max_logical_pages) {
          // TODO: ensure page_offset is aligned
          // int2 physical_pages = *(int2 *)(block_table + logical_page);
          int2 physical_pages;
          physical_pages.x = block_table[logical_page];
          physical_pages.y = block_table[logical_page + 1];
          num_bytes = 2 * 64 * sizeof(int);
          tma_1d_gmem2smem((float*)(K_ptr + physical_pages.x * 64 * 132 + 64 * 128),
                           k_group_scales_buf + ldg_stage * 128, 64 * sizeof(float),
                           &K_scales_prod_mbars[k_group][ldg_stage]);

          tma_1d_gmem2smem((float*)(K_ptr + physical_pages.y * 64 * 132 + 64 * 128),
                           k_group_scales_buf + ldg_stage * 128 + 64, 64 * sizeof(float),
                           &K_scales_prod_mbars[k_group][ldg_stage]);
        } else if (logical_page < max_logical_pages) {
          int physical_pages = block_table[logical_page];
          num_bytes = 64 * sizeof(int);
          tma_1d_gmem2smem((float*)(K_ptr + physical_pages * 64 * 132 + 64 * 128),
                           k_group_scales_buf + ldg_stage * 128, 64 * sizeof(float),
                           &K_scales_prod_mbars[k_group][ldg_stage]);
        } else {
          num_bytes = 0;
        }

        asm volatile(
            "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, "
            "[%0], %1;" ::"l"(&K_scales_prod_mbars[k_group][ldg_stage]),
            "r"(num_bytes)
            : "memory");

        if (DEBUG)
          printf(
              "post K scales producer: loading %d, epilogue_phase %d, "
              "logical_page %d\n",
              i, epilogue_phase, logical_page);
      }
    }
  } else if (is_mma_warp) {
    if (elect_sync()) {
      mbarrier_wait(&q_bar, 0);  // make weights and q visible
      int ldg_phase = 0;
      int epilogue_phase = 1;

      for (int i = 0; i < divup(num_pages, EPILOGUE_WARPGRPS * 2); i++) {
        int ldg_stage = i % K_LDG_STAGES;
        int epilogue_stage = i % K_MMA_STAGES;

        if (DEBUG)
          printf("pre mma: loading %d, ldg_phase %d, epilogue_phase %d\n", i, ldg_phase,
                 epilogue_phase);

        mbarrier_wait(&K_prod_mbars[k_group][ldg_stage], ldg_phase);
        if (DEBUG) printf("pre mma: done waiting for k_prod\n");
        mbarrier_wait(&Epilogue_mbars[k_group][epilogue_stage], epilogue_phase);
        asm volatile("tcgen05.fence::after_thread_sync;");

        int cur_k_buf = cvt_addr(k_group_s_buf + ldg_stage * 128 * 128);
        int cur_q_buf = cvt_addr(s_q_buf);
        auto a_desc = make_desc_a(cur_k_buf);  // [128, 128]
        auto b_desc = make_desc_a(cur_q_buf);  // [64 * q_next, 128]
        int cur_taddr = k_grp_taddr + epilogue_stage * 64 * Q_NEXT;
        tcgen05_mma_f8(cur_taddr, a_desc, b_desc, idesc, 0);
        for (int k = 1; k < 128 / 32; k++) {
          auto a_desc = make_desc_a(cur_k_buf + 32 * k);  // [128, 128]
          auto b_desc = make_desc_a(cur_q_buf + 32 * k);  // [64 * q_next, 128]
          tcgen05_mma_f8(cur_taddr, a_desc, b_desc, idesc, 1);
        }

        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" ::"l"(
                         &MMA_mbars[k_group][epilogue_stage])
                     : "memory");

        if ((i + 1) % K_LDG_STAGES == 0) ldg_phase ^= 1;

        if ((i + 1) % K_MMA_STAGES == 0) epilogue_phase ^= 1;

        if (DEBUG)
          printf("post mma: loading %d, ldg_phase %d, epilogue_phase %d\n", i, ldg_phase,
                 epilogue_phase);
      }
    }
  } else if (is_epilogue_warp) {  // math warps

    int mma_phase = 0;
    int ldg_phase = 0;

    mbarrier_wait(&q_bar, 0);         // make weights and q visible
    float q_weights_reg[Q_NEXT][64];  // only used if PRELOAD_TCGEN_REGS is true
    if (PRELOAD_TCGEN_REGS) {
      for (int q_next = 0; q_next < Q_NEXT; q_next++) {
        for (int off = 0; off < 64; off++) {
          q_weights_reg[q_next][off] = q_weights[q_next][off];
        }
      }
    }

    for (int idx = 0; idx < divup(num_pages, EPILOGUE_WARPGRPS * 2); idx++) {
      int mma_stage = idx % K_MMA_STAGES;
      int ldg_stage = idx % K_LDG_SCALES_STAGES;

      if (DEBUG && (threadIdx.x % 128) == 0)
        printf("pre epilogue: loading %d, mma_phase %d, ldg_phase %d\n", idx, mma_phase, ldg_phase);

      mbarrier_wait(&K_scales_prod_mbars[k_group][ldg_stage], ldg_phase);
      if (DEBUG && (threadIdx.x % 128) == 0)
        printf("pre epilogue: done waiting for k_scales_prod\n");

      float* scales_addr = k_group_scales_buf + ldg_stage * 128;
      float k_seq_scale = scales_addr[threadIdx.x % 128];  // each warp group accesses the full
                                                           // 128 cols along the seq dim

      mbarrier_wait(&MMA_mbars[k_group][mma_stage], mma_phase);
      asm volatile("tcgen05.fence::after_thread_sync;");

      int cur_taddr = k_grp_taddr + mma_stage * 64 * Q_NEXT;

      constexpr int accum_num = 2;  // ffma pipeline ilp
      float2 accum[Q_NEXT][accum_num];
      for (int q_next = 0; q_next < Q_NEXT; q_next++) {
        for (int v = 0; v < accum_num; v++) {
          accum[q_next][v].x = 0;
          accum[q_next][v].y = 0;
        }

        for (int off = 0; off < 64; off += VECTORIZE_TCGEN_LD) {
          float tmp[VECTORIZE_TCGEN_LD];
          int row_off = (warp_id % 4) * 32;
          int col_off = off + q_next * 64;

          int ldg_taddr = cur_taddr + (row_off << 16) + col_off;
          tcgen05_ld_32x32b<VECTORIZE_TCGEN_LD>(ldg_taddr, tmp);
          static_assert(VECTORIZE_TCGEN_LD % 2 == 0, "VECTORIZE_TCGEN_LD must be multiple of 2");
          for (int v = 0; v < VECTORIZE_TCGEN_LD; v += 2) {
            float2 qw;
            if (PRELOAD_TCGEN_REGS) {
              qw.x = q_weights_reg[q_next][off + v];
              qw.y = q_weights_reg[q_next][off + 1 + v];
            } else {
              qw.x = q_weights[q_next][off + v];
              qw.y = q_weights[q_next][off + 1 + v];
            }

            float2 tmp2 = make_float2(max(0.0f, tmp[v]), max(0.0f, tmp[v + 1]));

            int accum_ind = (v / 2) % accum_num;
            accum[q_next][accum_ind] = __ffma2_rn(tmp2, qw, accum[q_next][accum_ind]);
          }
        }
      }
      asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"l"(
                       &Epilogue_mbars[k_group][mma_stage])
                   : "memory");
      for (int q_next = 0; q_next < Q_NEXT; q_next++) {
        int logical_page = page_offset + idx * EPILOGUE_WARPGRPS * 2 + k_group * 2;
        int offset = logical_page * 64 + (threadIdx.x % 128);
        if (offset < seq_end) {
          float final_sum = 0.0f;

          for (int v = 0; v < accum_num; v++) {
            final_sum += accum[q_next][v].x + accum[q_next][v].y;
          }
          float final_logit = final_sum * k_seq_scale;
          out_logits[q_next * max_num_pages * 64 + offset] = final_logit;

          uint8_t bin = convert_to_uint32_v2(final_logit) >> 24;
          atomicAdd(histogram[q_next] + bin, 1);
        }
      }

      if (DEBUG && (threadIdx.x % 128) == 0)
        printf("post epilogue: loading %d, mma_phase %d, ldg_phase %d\n", idx, mma_phase,
               ldg_phase);
      if ((idx + 1) % K_MMA_STAGES == 0) mma_phase ^= 1;

      if ((idx + 1) % K_LDG_SCALES_STAGES == 0) ldg_phase ^= 1;
    }
  }

  if (DEBUG) {
    int mask = __activemask();
    if (lane_id == 0) {
      printf("warp end %d, %x\n", warp_id, mask);
    }
  }
  __syncthreads();

  for (int id = threadIdx.x; id < Q_NEXT * 256; id += THREADS) {
    atomicAdd(global_histogram + id, histogram[id / 256][id % 256]);
  }
  if (PDL_ENABLED) {
    __syncthreads();
    if (threadIdx.x == 0) cudaTriggerProgrammaticLaunchCompletion();
  }

  if (warp_id == 0) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr),
                 "r"(TMEM_COLS));
  }
}

template <bool PDL_ENABLED, int K_LDG_STAGES, int K_LDG_SCALES_STAGES, int K_MMA_STAGES,
          int EPILOGUE_WARPGRPS, int VECTORIZE_TCGEN_LD, int PRELOAD_TCGEN_REGS, int Q_NEXT>
__global__ __launch_bounds__(EPILOGUE_WARPGRPS * 7 * 32, 1) void mqa_v3_fused_epilogue_kernel(
    const __grid_constant__ CUtensorMap
        Q_tmap,  // f8[BatchSize, q_next * 64, 128], strides=[64 * 128, 128]
    const __grid_constant__ CUtensorMap K_tmap,  // f8[Pages, 64, 128], strides=[64 * 132, 128, 1]
    const uint8_t* K_ptr,                        // [Pages, 64 * 132]
    const float* __restrict__ weights,           // [BatchSize, q_next, 64]
    const int* __restrict__ seq_lens,            // [BatchSize]
    const int* __restrict__ block_table,         // [BatchSize, max_num_pages]
    const int4* __restrict__ sm_mapping, const int max_num_pages, const int page_table_pages,
    const int sm_multiple,
    const int logit_batch_stride,            // stride in floats between batch rows of out_logits
    float* __restrict__ out_logits,          // [BatchSize, logit_batch_stride]
    uint32_t* __restrict__ global_histogram  // [BatchSize, Q_NEXT * 256] (contiguous)

) {
  int sm_id = blockIdx.x / sm_multiple;
  int sm_loc = blockIdx.x % sm_multiple;

  int4 sm_map;
  if (threadIdx.x % 32 == 0) {
    sm_map = sm_mapping[sm_id];
  }
  sm_map.x = __shfl_sync(0xffffffff, sm_map.x, 0, 32);
  sm_map.y = __shfl_sync(0xffffffff, sm_map.y, 0, 32);
  sm_map.z = __shfl_sync(0xffffffff, sm_map.z, 0, 32);
  sm_map.w = __shfl_sync(0xffffffff, sm_map.w, 0, 32);
  int batch_idx = sm_map.x;
  int num_pages_per_sm = divup(sm_map.w, sm_multiple);
  int num_pages = min(num_pages_per_sm, sm_map.w - num_pages_per_sm * sm_loc);
  int page_offset = sm_map.z + num_pages_per_sm * sm_loc;
  if (num_pages > 0) {
    int seq_len = seq_lens[batch_idx];

    mqa_v2_kernel<PDL_ENABLED, K_LDG_STAGES, K_LDG_SCALES_STAGES, K_MMA_STAGES, EPILOGUE_WARPGRPS,
                  VECTORIZE_TCGEN_LD, PRELOAD_TCGEN_REGS, Q_NEXT>(
        &Q_tmap, &K_tmap, K_ptr, weights + Q_NEXT * 64 * batch_idx, batch_idx, seq_len, page_offset,
        num_pages, max_num_pages, page_table_pages, block_table + batch_idx * max_num_pages,
        out_logits + batch_idx * logit_batch_stride, global_histogram + batch_idx * Q_NEXT * 256);
  } else if (PDL_ENABLED) {
    if (threadIdx.x == 0) {
      cudaTriggerProgrammaticLaunchCompletion();
    }
  }
}

// launches the fused epilogue kernel, where we compute the histogram of logits
// bits [0-8) for the first instance of the topK kernel
void launch_mqa_v3_fused_epilogue(uint8_t* q_ptr, uint8_t* k_ptr, float* weights, int* seq_lens,
                                  int* block_table, float* logits, uint32_t* histogram,
                                  int4* sm_map, int max_num_pages, int num_pages, int batch_size,
                                  int num_sms, int sm_multiple, int logit_batch_stride,
                                  bool pdl_enabled, cudaStream_t stream) {
  constexpr int Q_NEXT = 1;
  constexpr int EPILOGUE_WARPGRPS = 2;
  constexpr int VECTORIZE_TCGEN_LD = 16;

  constexpr int K_LDG_STAGES = 3;
  constexpr int K_LDG_SCALES_STAGES = 3;
  constexpr int K_MMA_STAGES = 4;
  constexpr int PRELOAD_TCGEN_REGS = 1;

  constexpr int THREADS = EPILOGUE_WARPGRPS * 7 * 32;

  CUtensorMap q_tmap;
  CUtensorMap k_tmap;
  prep_tmaps<Q_NEXT>(&q_tmap, &k_tmap, q_ptr, k_ptr, batch_size, num_pages);
  constexpr int smem_bytes =
      (128 * 128 * EPILOGUE_WARPGRPS * K_LDG_STAGES + Q_NEXT * 64 * 128)  // K + Q
      + sizeof(float) * 128 * K_LDG_SCALES_STAGES * EPILOGUE_WARPGRPS;
  if (pdl_enabled) {
    setup_kernel_smem_once<mqa_v3_fused_epilogue_kernel<
                               true, K_LDG_STAGES, K_LDG_SCALES_STAGES, K_MMA_STAGES,
                               EPILOGUE_WARPGRPS, VECTORIZE_TCGEN_LD, PRELOAD_TCGEN_REGS, Q_NEXT>,
                           131 * 1000>();

    mqa_v3_fused_epilogue_kernel<true, K_LDG_STAGES, K_LDG_SCALES_STAGES, K_MMA_STAGES,
                                 EPILOGUE_WARPGRPS, VECTORIZE_TCGEN_LD, PRELOAD_TCGEN_REGS, Q_NEXT>
        <<<sm_multiple * num_sms, THREADS, smem_bytes, stream>>>(
            q_tmap, k_tmap, k_ptr, weights, seq_lens, block_table, sm_map, max_num_pages, num_pages,
            sm_multiple, logit_batch_stride, logits, histogram);
  } else {
    setup_kernel_smem_once<mqa_v3_fused_epilogue_kernel<
                               false, K_LDG_STAGES, K_LDG_SCALES_STAGES, K_MMA_STAGES,
                               EPILOGUE_WARPGRPS, VECTORIZE_TCGEN_LD, PRELOAD_TCGEN_REGS, Q_NEXT>,
                           131 * 1000>();

    mqa_v3_fused_epilogue_kernel<false, K_LDG_STAGES, K_LDG_SCALES_STAGES, K_MMA_STAGES,
                                 EPILOGUE_WARPGRPS, VECTORIZE_TCGEN_LD, PRELOAD_TCGEN_REGS, Q_NEXT>
        <<<sm_multiple * num_sms, THREADS, smem_bytes, stream>>>(
            q_tmap, k_tmap, k_ptr, weights, seq_lens, block_table, sm_map, max_num_pages, num_pages,
            sm_multiple, logit_batch_stride, logits, histogram);
  }
}
