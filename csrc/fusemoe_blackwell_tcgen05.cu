/*
 * Copyright (c) 2026 by the IFKernel team.
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
 *
 * Originally contributed to the FlashInfer-Bench MLSys'26 kernel-generation
 * contest (https://github.com/flashinfer-ai/mlsys26-contest), then adapted
 * into flashinfer's JIT pipeline.
 */
// Extracted from the embedded g_tcgen05_src in fusemoe_blackwell.cu.
// Compiled directly via flashinfer JIT (was previously built into a .so via
// system("nvcc ...") + dlopen at first call).

// tcgen05 FP8 Grouped GEMM for MoE pipeline
// Single TMA descriptor: A=[buf_rows,K], B=[num_experts*N,K]
// Per-row scale_a, per-expert/block scale_b, partial-tile masking
// Compatible with CutlassBwArgs interface
//
// Optimizations R19:
// - Linear scan tile decode (replaces binary search, fewer branch mispredictions for G<=32)
// - Tile swizzle (swizzle_size=4) for L2 locality on B matrix columns

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

// ============================================================================
// Constants
// ============================================================================
static constexpr int WARP_SIZE = 32;
static constexpr int NUM_WARPS = 6;
static constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

static constexpr int BM = 128;
static constexpr int BN = 128;
static constexpr int BK = 128;
static constexpr int MMA_K = 32;

static constexpr int CTA_GROUP = 1;
static constexpr int NUM_STAGES = 7;

static constexpr int A_SIZE = BM * BK;
static constexpr int B_SIZE = BN * BK;
static constexpr int AB_SIZE = A_SIZE + B_SIZE;

// ============================================================================
// Device helpers
// ============================================================================
template <typename T>
__device__ __forceinline__ T warp_uniform(T x) {
  return __shfl_sync(0xFFFFFFFF, x, 0);
}

__device__ __forceinline__ uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile(
      "{\n\t.reg .pred %%px;\n\t"
      "elect.sync _|%%px, %1;\n\t@%%px mov.s32 %0, 1;\n\t}"
      : "+r"(pred)
      : "r"(0xFFFFFFFF));
  return pred;
}

__device__ __forceinline__ void mbar_init(int a, int c) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(c));
}
__device__ __forceinline__ void mbar_wait(int a, int p) {
  uint32_t t = 0x989680;
  asm volatile(
      "{\n\t.reg .pred P1;\n\tLAB_WAIT:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
      "@P1 bra.uni DONE;\n\tbra.uni LAB_WAIT;\n\tDONE:\n\t}" ::"r"(a),
      "r"(p), "r"(t));
}
__device__ __forceinline__ void mbar_arrive_tx(int a, int s) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;" ::"r"(a),
               "r"(s)
               : "memory");
}
__device__ __forceinline__ void mbar_arrive(int a) {
  asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];" ::"r"(a) : "memory");
}

template <int G = 1>
__device__ __forceinline__ void tma_g2s(int dst, const void* t, int x, int y, int z, int m) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::%6 "
      "[%0], [%1, {%2, %3, %4}], [%5];" ::"r"(dst),
      "l"(t), "r"(x), "r"(y), "r"(z), "r"(m), "n"(G)
      : "memory");
}

template <int G = 1>
__device__ __forceinline__ void tc_alloc(int a, int s) {
  asm volatile("tcgen05.alloc.cta_group::%2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(a),
               "r"(s), "n"(G));
}
template <int G = 1>
__device__ __forceinline__ void tc_dealloc(int a, int s) {
  asm volatile("tcgen05.dealloc.cta_group::%2.sync.aligned.b32 %0, %1;" ::"r"(a), "r"(s), "n"(G));
}
template <int G = 1>
__device__ __forceinline__ void tc_mma_f8(int t, uint64_t a, uint64_t b, uint32_t i, int d) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::%5.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}" ::"r"(t),
      "l"(a), "l"(b), "r"(i), "r"(d), "n"(G));
}
template <int G = 1>
__device__ __forceinline__ void tc_commit(int m) {
  asm volatile(
      "tcgen05.commit.cta_group::%1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(m),
      "n"(G)
      : "memory");
}

__device__ __forceinline__ constexpr uint64_t desc_enc(uint64_t x) {
  return (x & 0x3'FFFFULL) >> 4ULL;
}

// ============================================================================
// Tile decode with linear scan + swizzle
// ============================================================================
struct TileInfo {
  int group_id, m_tile, n_tile, m_offset;
  int group_M;
};

__device__ __forceinline__ TileInfo decode_tile(int tile_idx, const int* __restrict__ tile_offsets,
                                                const int* __restrict__ m_indptr, int num_groups,
                                                int grid_n) {
  // Linear scan: for G<=32, fewer branch mispredictions than binary search
  int g = 0;
  for (int gg = 1; gg <= num_groups; gg++) {
    if (tile_offsets[gg] <= tile_idx)
      g = gg;
    else
      break;
  }

  int local = tile_idx - tile_offsets[g];
  int m_off = __ldg(m_indptr + g);
  int gM = __ldg(m_indptr + g + 1) - m_off;
  int m_blocks = (gM + BM - 1) / BM;

  // Swizzle for L2 locality: group 4 M-tiles together, iterate N within
  // This keeps B-matrix columns in L2 across consecutive tiles
  int m_tile, n_tile;
  constexpr int S = 4;
  if (m_blocks > S) {
    int tpsg = S * grid_n;
    int sg = local / tpsg;
    int within = local % tpsg;
    n_tile = within / S;
    m_tile = sg * S + (within % S);
    // Fallback for last partial swizzle group
    if (m_tile >= m_blocks) {
      m_tile = local / grid_n;
      n_tile = local % grid_n;
    }
  } else {
    m_tile = local / grid_n;
    n_tile = local % grid_n;
  }

  TileInfo i;
  i.group_id = g;
  i.m_tile = m_tile;
  i.n_tile = n_tile;
  i.m_offset = m_off;
  i.group_M = gM;
  return i;
}

// ============================================================================
// Main kernel
// ============================================================================
__global__ __launch_bounds__(TB_SIZE, 1) void tcgen05_gemm_kernel(
    const CUtensorMap* __restrict__ A_tmap, const CUtensorMap* __restrict__ B_tmap,
    __nv_bfloat16* D_ptr, const float* scale_a, const float* scale_b, const int* m_indptr,
    const int* expert_ids, int G, int N, int K) {
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;");
#endif
  const int tid = threadIdx.x;
  const int bid = warp_uniform(blockIdx.x);
  const int nbids = warp_uniform(gridDim.x);
  const int wid = warp_uniform(tid / WARP_SIZE);

  const int gn = N / BN;
  const int nkb = K / BK;
  const int nb = N / BN;
  const int kb = K / BK;

  extern __shared__ __align__(1024) char smem_raw[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_raw));

  const int tma_mb = smem + AB_SIZE * NUM_STAGES;
  const int mma_mb = tma_mb + NUM_STAGES * 8;
  const int ml_mb = mma_mb + NUM_STAGES * 8;
  const int ep_mb = ml_mb + 2 * 8;
  int* s_tile_offsets = reinterpret_cast<int*>(smem_raw + AB_SIZE * NUM_STAGES + NUM_STAGES * 8 +
                                               NUM_STAGES * 8 + 2 * 8 + 2 * 8);

  // Compute tile_offsets in smem (warp 0, thread 0)
  if (wid == 0) {
    if (tid == 0) {
      int acc = 0;
      s_tile_offsets[0] = 0;
      for (int g = 0; g < G; g++) {
        int Mg = __ldg(m_indptr + g + 1) - __ldg(m_indptr + g);
        int mt = (Mg + BM - 1) / BM;
        acc += mt * gn;
        s_tile_offsets[g + 1] = acc;
      }
    }
    if (elect_sync()) {
      for (int i = 0; i < NUM_STAGES; i++) {
        mbar_init(tma_mb + i * 8, 1);
        mbar_init(mma_mb + i * 8, 1);
      }
      for (int i = 0; i < 2; i++) {
        mbar_init(ml_mb + i * 8, 1);
        mbar_init(ep_mb + i * 8, 4);
      }
      asm volatile("fence.mbarrier_init.release.cluster;");
    }
  }
  __syncthreads();

  const int total_tiles = s_tile_offsets[G];
  if (total_tiles <= 0) {
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    return;
  }
  if (bid >= total_tiles) {
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    return;
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
  if (bid == 0 && wid == 0 && tid == 0) asm volatile("griddepcontrol.launch_dependents;");
#endif

  if (wid == NUM_WARPS - 2) {
    // TMA warp
    if (elect_sync()) {
      int s = 0, mp = 1;
      for (int tb = bid; tb < total_tiles; tb += nbids) {
        TileInfo ti = decode_tile(tb, s_tile_offsets, m_indptr, G, gn);
        int ar = ti.m_offset + ti.m_tile * BM;
        int ex = __ldg(expert_ids + ti.group_id);
        int br = ex * N + ti.n_tile * BN;
        for (int ik = 0; ik < nkb; ik++) {
          int mb = tma_mb + s * 8;
          int As = smem + s * AB_SIZE, Bs = As + A_SIZE;
          mbar_wait(mma_mb + s * 8, mp);
          tma_g2s<CTA_GROUP>(As, A_tmap, 0, ar, ik, mb);
          tma_g2s<CTA_GROUP>(Bs, B_tmap, 0, br, ik, mb);
          mbar_arrive_tx(mb, A_SIZE + B_SIZE);
          s = (s + 1) % NUM_STAGES;
          if (s == 0) mp ^= 1;
        }
      }
    }
  } else if (wid == NUM_WARPS - 1) {
    // MMA warp
    tc_alloc<CTA_GROUP>(ep_mb + 8 * 2, BN * 2);
    constexpr uint32_t idesc =
        (1U << 4) | (0U << 7) | (0U << 10) | (BN >> 3 << 17) | (BM >> 4 << 24);
    constexpr uint64_t ABd = (desc_enc(8 * 128) << 32ULL) | (1ULL << 46) | (2ULL << 61);
    if (elect_sync()) {
      int s = 0, tp = 0, ms = 0, ep = 1;
      for (int tb = bid; tb < total_tiles; tb += nbids) {
        for (int ik = 0; ik < nkb; ik++) {
          mbar_wait(ep_mb + ms * 8, ep);
          int As = smem + s * AB_SIZE, Bs = As + A_SIZE;
          int tm = ms * BN;
          uint64_t ad = ABd | (As >> 4), bd = ABd | (Bs >> 4);
          mbar_wait(tma_mb + s * 8, tp);
          asm volatile("tcgen05.fence::after_thread_sync;");
          tc_mma_f8<CTA_GROUP>(tm, ad, bd, idesc, 0);
          for (int k = 1; k < BK / MMA_K; k++) {
            ad += (32 >> 4);
            bd += (32 >> 4);
            tc_mma_f8<CTA_GROUP>(tm, ad, bd, idesc, 1);
          }
          tc_commit<CTA_GROUP>(mma_mb + s * 8);
          tc_commit<CTA_GROUP>(ml_mb + ms * 8);
          s = (s + 1) % NUM_STAGES;
          if (s == 0) tp ^= 1;
          ms = (ms + 1) % 2;
          if (ms == 0) ep ^= 1;
        }
      }
    }
  } else {
    // Epilogue warps 0-3
    int ms = 0, mp = 0;
    auto esync = []() { asm volatile("bar.sync %0, %1;" ::"r"(1), "r"(4 * WARP_SIZE) : "memory"); };
    constexpr int NR = BN / 16;
    float acc[NR][16];

    for (int tb = bid; tb < total_tiles; tb += nbids) {
      TileInfo ti = decode_tile(tb, s_tile_offsets, m_indptr, G, gn);
      int ex = __ldg(expert_ids + ti.group_id);
      int my_row = ti.m_offset + ti.m_tile * BM + tid;
      int row_in_group = ti.m_tile * BM + tid;
      bool valid = (row_in_group < ti.group_M);

      // Precompute base pointers to avoid repeated 64-bit index math in inner loop
      const float* sa_base = valid ? (scale_a + (int64_t)my_row * kb) : nullptr;
      const float* sb_base = scale_b + ((int64_t)ex * nb * kb + (int64_t)ti.n_tile * kb);

#pragma unroll
      for (int n = 0; n < NR; n++)
#pragma unroll
        for (int i = 0; i < 16; i++) acc[n][i] = 0.0f;

      for (int ik = 0; ik < nkb; ik++) {
        float sa = sa_base ? sa_base[ik] : 0.0f;
        float sb = sb_base[ik];
        float cs = sa * sb;

        if (wid == 0) mbar_wait(ml_mb + ms * 8, mp);
        esync();
        asm volatile("tcgen05.fence::after_thread_sync;");

#pragma unroll
        for (int n = 0; n < NR; n++) {
          int ta = (wid * 32 << 16) + ms * BN + n * 16;
          float t[16];
          asm volatile(
              "tcgen05.ld.sync.aligned.32x32b.x16.b32\n"
              "  {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n"
              "tcgen05.wait::ld.sync.aligned;\n"
              : "=f"(t[0]), "=f"(t[1]), "=f"(t[2]), "=f"(t[3]), "=f"(t[4]), "=f"(t[5]), "=f"(t[6]),
                "=f"(t[7]), "=f"(t[8]), "=f"(t[9]), "=f"(t[10]), "=f"(t[11]), "=f"(t[12]),
                "=f"(t[13]), "=f"(t[14]), "=f"(t[15])
              : "r"(ta));
#pragma unroll
          for (int i = 0; i < 16; i++) acc[n][i] += t[i] * cs;
        }
        // Only lane 0 of each epilogue warp arrives (4 arrivals total, matching init count)
        if ((tid % WARP_SIZE) == 0) mbar_arrive(ep_mb + ms * 8);
        ms = (ms + 1) % 2;
        if (ms == 0) mp ^= 1;
      }

      if (valid) {
#pragma unroll
        for (int n = 0; n < NR; n++) {
          int gc = ti.n_tile * BN + n * 16;
          __nv_bfloat16* out = D_ptr + (int64_t)my_row * N + gc;
          __nv_bfloat16 v[16];
#pragma unroll
          for (int i = 0; i < 16; i++) v[i] = __float2bfloat16(acc[n][i]);
          *reinterpret_cast<int4*>(out) = *reinterpret_cast<int4*>(&v[0]);
          *reinterpret_cast<int4*>(out + 8) = *reinterpret_cast<int4*>(&v[8]);
        }
      }
    }
    esync();
    if (wid == 0) tc_dealloc<CTA_GROUP>(0, BN * 2);
  }
}

// ============================================================================
// Host
// ============================================================================
#define CHK_CUDA(c)                                                         \
  do {                                                                      \
    cudaError_t e = (c);                                                    \
    if (e != cudaSuccess) {                                                 \
      fprintf(stderr, "[tcgen05] CUDA err %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(e));                                       \
      return -10;                                                           \
    }                                                                       \
  } while (0)
#define CHK_CU(c)                                                             \
  do {                                                                        \
    CUresult e = (c);                                                         \
    if (e != CUDA_SUCCESS) {                                                  \
      const char* m;                                                          \
      cuGetErrorString(e, &m);                                                \
      fprintf(stderr, "[tcgen05] CU err %s:%d: %s\n", __FILE__, __LINE__, m); \
      return -11;                                                             \
    }                                                                         \
  } while (0)

static int mk_tmap(CUtensorMap* t, const void* p, uint64_t h, uint64_t w, uint32_t sh,
                   uint32_t sw) {
  uint64_t gd[3] = {128, h, w / 128};
  uint64_t gs[2] = {w, 128};
  uint32_t bd[3] = {128, sh, sw / 128};
  uint32_t es[3] = {1, 1, 1};
  CHK_CU(cuTensorMapEncodeTiled(t, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, (void*)p, gd, gs, bd, es,
                                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return 0;
}

#include "fusemoe_blackwell_gemm_args.h"
#include "fusemoe_blackwell_log.h"

static CUtensorMap *d_Atm = nullptr, *d_Btm = nullptr;
static CUtensorMap *d_Atm2 = nullptr, *d_Btm2 = nullptr;  // GEMM2 descriptors
static int s_maxg = 0;
static bool s_smem_done = false;
static int s_smem_bytes = 0;
static void* s_cA = nullptr;
static int s_cAh = 0;
static void* s_cB = nullptr;
static int s_cBh = 0, s_cK = 0;
static int s_hint_total_rows = 0;

extern "C" void tcgen05_set_total_rows(int r) { s_hint_total_rows = r; }

static inline int compute_smem_bytes(int G) {
  int mbar_bytes = (2 * NUM_STAGES + 4) * 8;
  int toff_bytes = (G + 1) * 4 + 4;
  return NUM_STAGES * AB_SIZE + mbar_bytes + toff_bytes;
}

extern "C" int tcgen05_setup_tma(void* Ap, int mr, void* Bp, int ne, int N, int K) {
  if (!d_Atm && FUSEMOE_CUDA_MALLOC(d_Atm, sizeof(CUtensorMap)) != cudaSuccess) return -1;
  if (!d_Btm && FUSEMOE_CUDA_MALLOC(d_Btm, sizeof(CUtensorMap)) != cudaSuccess) return -1;

  int pr = ((mr + 127) / 128) * 128;
  CUtensorMap hA, hB;
  int r;
  r = mk_tmap(&hA, Ap, pr, K, BM, BK);
  if (r) return r;
  r = mk_tmap(&hB, Bp, (uint64_t)ne * N, K, BN, BK);
  if (r) return r;
  CHK_CUDA(cudaMemcpy(d_Atm, &hA, sizeof(CUtensorMap), cudaMemcpyHostToDevice));
  CHK_CUDA(cudaMemcpy(d_Btm, &hB, sizeof(CUtensorMap), cudaMemcpyHostToDevice));
  s_cA = Ap;
  s_cAh = pr;
  s_cB = Bp;
  s_cBh = ne * N;
  s_cK = K;

  if (!s_smem_done) {
    int ss = compute_smem_bytes(512);
    CHK_CUDA(
        cudaFuncSetAttribute(tcgen05_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ss));
    s_smem_bytes = ss;
    s_smem_done = true;
  }
  FUSEMOE_LOG_INFO("[tcgen05] TMA: A=[%d,%d] B=[%d,%d]\n", pr, K, ne * N, K);
  return 0;
}

extern "C" int tcgen05_grouped_gemm(GemmArgs* a, cudaStream_t stream) {
  int G = a->num_groups, N = a->N, K = a->K;
  if (!d_Atm || !d_Btm) return -1;

  int gn = N / BN;
  int grid = 148;
  if (s_hint_total_rows > 0) {
    int max_tiles = ((s_hint_total_rows + G * (BM - 1)) / BM) * gn;
    if (max_tiles < 1) max_tiles = 1;
    if (max_tiles < grid) grid = max_tiles;
  } else {
    int cap = G * gn * 8;
    if (cap < grid) grid = cap;
  }
  if (grid < 1) grid = 1;

  int ss = compute_smem_bytes(G);
  if (ss > s_smem_bytes) {
    CHK_CUDA(
        cudaFuncSetAttribute(tcgen05_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ss));
    s_smem_bytes = ss;
  }

  cudaLaunchConfig_t lc{};
  lc.gridDim = grid;
  lc.blockDim = TB_SIZE;
  lc.dynamicSmemBytes = ss;
  lc.stream = stream;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  lc.numAttrs = 1;
  lc.attrs = at;

  cudaLaunchKernelEx(&lc, tcgen05_gemm_kernel, (const CUtensorMap*)d_Atm, (const CUtensorMap*)d_Btm,
                     (__nv_bfloat16*)a->D, (const float*)a->SFA, (const float*)a->SFB,
                     (const int*)a->m_indptr, (const int*)a->expert_ids, G, N, K);

  return 0;
}

// ============================================================================
// GEMM2 TMA setup and dispatch (separate descriptors for different N,K)
// ============================================================================
extern "C" int tcgen05_setup_tma2(void* Ap, int mr, void* Bp, int ne, int N, int K) {
  if (!d_Atm2 && FUSEMOE_CUDA_MALLOC(d_Atm2, sizeof(CUtensorMap)) != cudaSuccess) return -1;
  if (!d_Btm2 && FUSEMOE_CUDA_MALLOC(d_Btm2, sizeof(CUtensorMap)) != cudaSuccess) return -1;

  int pr = ((mr + 127) / 128) * 128;
  CUtensorMap hA2, hB2;
  int r;
  r = mk_tmap(&hA2, Ap, pr, K, BM, BK);
  if (r) return r;
  r = mk_tmap(&hB2, Bp, (uint64_t)ne * N, K, BN, BK);
  if (r) return r;
  CHK_CUDA(cudaMemcpy(d_Atm2, &hA2, sizeof(CUtensorMap), cudaMemcpyHostToDevice));
  CHK_CUDA(cudaMemcpy(d_Btm2, &hB2, sizeof(CUtensorMap), cudaMemcpyHostToDevice));

  if (!s_smem_done) {
    int ss = compute_smem_bytes(512);
    CHK_CUDA(
        cudaFuncSetAttribute(tcgen05_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ss));
    s_smem_bytes = ss;
    s_smem_done = true;
  }
  FUSEMOE_LOG_INFO("[tcgen05] TMA2: A=[%d,%d] B=[%d,%d]\n", pr, K, ne * N, K);
  return 0;
}

extern "C" int tcgen05_grouped_gemm2(GemmArgs* a, cudaStream_t stream) {
  int G = a->num_groups, N = a->N, K = a->K;
  if (!d_Atm2 || !d_Btm2) return -1;

  int gn = N / BN;
  int grid = 148;
  if (s_hint_total_rows > 0) {
    int max_tiles = ((s_hint_total_rows + G * (BM - 1)) / BM) * gn;
    if (max_tiles < 1) max_tiles = 1;
    if (max_tiles < grid) grid = max_tiles;
  } else {
    int cap = G * gn * 8;
    if (cap < grid) grid = cap;
  }
  if (grid < 1) grid = 1;

  int ss = compute_smem_bytes(G);
  if (ss > s_smem_bytes) {
    CHK_CUDA(
        cudaFuncSetAttribute(tcgen05_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ss));
    s_smem_bytes = ss;
  }

  cudaLaunchConfig_t lc{};
  lc.gridDim = grid;
  lc.blockDim = TB_SIZE;
  lc.dynamicSmemBytes = ss;
  lc.stream = stream;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  lc.numAttrs = 1;
  lc.attrs = at;

  cudaLaunchKernelEx(&lc, tcgen05_gemm_kernel, (const CUtensorMap*)d_Atm2,
                     (const CUtensorMap*)d_Btm2, (__nv_bfloat16*)a->D, (const float*)a->SFA,
                     (const float*)a->SFB, (const int*)a->m_indptr, (const int*)a->expert_ids, G, N,
                     K);

  return 0;
}
