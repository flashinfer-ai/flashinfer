/*
 * Copyright (c) 2023 by FlashInfer team.
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

typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int int32_t;
typedef short int int16_t;
struct __align__(128) CakeTensorMap {
  uint64_t opaque[16];
};
template <int N>
struct __align__(128) CakeTensorMapPack {
  CakeTensorMap maps[N];
};

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) {
  uint64_t opaque[16];
} CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128-byte aligned");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
  int result;
  asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;" : "=r"(result) : "r"(x));
  return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_Q_SMEM_OFF 1024
#define SMEM_Q_SMEM_STAGE_BYTES 8192
#define SMEM_Q_SMEM_STRIDE 8192
#define SMEM_K_SMEM_OFF 9216
#define SMEM_K_SMEM_STAGE_BYTES 8192
#define SMEM_K_SMEM_STRIDE 8192
#define SMEM_V_TMA_SMEM_OFF 17408
#define SMEM_V_TMA_SMEM_STAGE_BYTES 8192
#define SMEM_V_TMA_SMEM_STRIDE 8192
#define SMEM_V_SMEM_OFF 17408
#define SMEM_V_SMEM_STAGE_BYTES 8192
#define SMEM_V_SMEM_STRIDE 8192
#define SMEM_O_SMEM_OFF 1024
#define SMEM_O_SMEM_STAGE_BYTES 16384
#define SMEM_O_SMEM_STRIDE 16384
#define SMEM_TOTAL 25600
#define THREADS 128
#define HAS_BLOCK_NUMS 0
#define BLOCK_SIZES_MODE 0
#define FULL_K64_TILES 0
#define UNIFORM_NONEMPTY 1
#define CONTIGUOUS_BLOCK_INDICES 1

#include <cuda_awbarrier_primitives.h>
#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred %%px;\n\t"
      "elect.sync _|%%px, %1;\n\t"
      "@%%px mov.s32 %0, 1;\n\t"
      "}\n"
      : "+r"(pred)
      : "r"(0xFFFFFFFF));
  return pred;
}

__device__ __forceinline__ void mbarrier_init_owner_lane(void* mbar_ptr, uint32_t count) {
  __mbarrier_init(reinterpret_cast<__mbarrier_t*>(mbar_ptr), count);
}

__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
  uint32_t token;
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
      " P1, [%1], %2;\n\t"
      "selp.u32 %0, 1, 0, P1;\n\t"
      "}\n"
      : "=r"(token)
      : "r"(mbar_addr), "r"(phase)
      : "memory");
  return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait_cluster(int mbar_addr, int phase) {
  uint32_t token;
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
      " P1, [%1], %2;\n\t"
      "selp.u32 %0, 1, 0, P1;\n\t"
      "}\n"
      : "=r"(token)
      : "r"(mbar_addr), "r"(phase)
      : "memory");
  return token;
}

// CTA-local pipelines have short, resident producer/consumer edges.  Omitting
// suspendTimeHint keeps a miss on the lightweight TRYWAIT retry path; the
// explicit loop still makes this helper blocking until acquire succeeds.
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "LAB_WAIT:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
      " P1, [%0], %1;\n\t"
      "@P1 bra.uni DONE;\n\t"
      "bra.uni LAB_WAIT;\n\t"
      "DONE:\n\t"
      "}\n" ::"r"(mbar_addr),
      "r"(phase)
      : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
  uint32_t ticks = 0x989680;
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "LAB_WAIT_CLUSTER:\n\t"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
      " P1, [%0], %1, %2;\n\t"
      "@P1 bra.uni DONE_CLUSTER;\n\t"
      "bra.uni LAB_WAIT_CLUSTER;\n\t"
      "DONE_CLUSTER:\n\t"
      "}\n" ::"r"(mbar_addr),
      "r"(phase), "r"(ticks)
      : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
  if (token == 0) {
    mbarrier_wait(mbar_addr, phase);
  }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase,
                                                            uint32_t token) {
  if (token == 0) {
    mbarrier_wait_cluster(mbar_addr, phase);
  }
}

__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(mbar_addr) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(mbar_addr),
      "r"(bytes)
      : "memory");
}

__device__ __forceinline__ float approx_exp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float approx_rcp(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float max_noftz(float a, float b) {
  float c;
  asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
  return c;
}

__device__ __forceinline__ void fence_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ uint64_t desc_encode(uint64_t x) { return (x & 0x3FFFFULL) >> 4ULL; }

__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
  const int SBO = 1024;
  return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
}

__device__ __forceinline__ void tma_4d_gmem2smem(int dst, const void* tmap_ptr, int x, int y, int z,
                                                 int w, int mbar_addr) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cta.global"
      ".mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4, %5}], [%6];" ::"r"(dst),
      "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(mbar_addr)
      : "memory");
}

__device__ __forceinline__ void tma_5d_gmem2smem(int dst, const void* tmap_ptr, int x, int y, int z,
                                                 int w, int v, int mbar_addr) {
  asm volatile(
      "cp.async.bulk.tensor.5d.shared::cta.global"
      ".mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];" ::"r"(dst),
      "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v), "r"(mbar_addr)
      : "memory");
}

__device__ __forceinline__ void tma_store_5d(const void* tmap, int x, int y, int z, int w, int v,
                                             unsigned smem_addr) {
  asm volatile(
      "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group"
      " [%0, {%1, %2, %3, %4, %5}], [%6];" ::"l"(tmap),
      "r"(x), "r"(y), "r"(z), "r"(w), "r"(v), "r"(smem_addr)
      : "memory");
}

extern "C" {

__global__
__launch_bounds__(128, 3) void kernel_cake_sage_block_sparse_attention_2c6e4042e52ef5d5e924(
    CakeTensorMap const* Q_map, CakeTensorMap const* K_map, CakeTensorMap const* V_map,
    CakeTensorMap const* O_map, float* __restrict__ q_scale, float* __restrict__ k_scale,
    float* __restrict__ v_scale, int* __restrict__ q2k_block_index,
    int* __restrict__ q2k_block_nums, int* __restrict__ block_sizes, int seqlen_q, int seqlen_k,
    int num_heads, int q2k_capacity, int num_k_blocks, int block_sparse_num, float softmax_scale) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

  const int mbar_base = smem;
#define q_full_addr (mbar_base + 0)
#define k_full_addr (mbar_base + 8)
#define v_full_addr (mbar_base + 16)

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;
  if (tid == 0) {
    asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" ::"l"((uint64_t)(Q_map))
                 : "memory");
    asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" ::"l"((uint64_t)(K_map))
                 : "memory");
    asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" ::"l"((uint64_t)(V_map))
                 : "memory");
    asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" ::"l"((uint64_t)(O_map))
                 : "memory");
  }
  __syncthreads();

  // Kernel setup ops
  int8_t* q_smem = reinterpret_cast<int8_t*>(smem_raw + 1024);
  const int q_smem_addr = smem + 1024;
  int8_t* k_smem = reinterpret_cast<int8_t*>(smem_raw + 9216);
  const int k_smem_addr = smem + 9216;
  uint8_t* v_tma_smem = reinterpret_cast<uint8_t*>(smem_raw + 17408);
  const int v_tma_smem_addr = smem + 17408;
  uint8_t* v_smem = reinterpret_cast<uint8_t*>(smem_raw + 17408);
  const int v_smem_addr = smem + 17408;
  __nv_bfloat16* o_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
  const int o_smem_addr = smem + 1024;

  // Mbarrier init (3 groups, 3 barriers)
  // Mbarriers at smem_raw[0..24)

  if (tid == 0) {
    // q_full: 1 barriers, init_count=1
    mbarrier_init_owner_lane(smem_raw + 0, 1);
    // k_full: 1 barriers, init_count=1
    mbarrier_init_owner_lane(smem_raw + 8, 1);
    // v_full: 1 barriers, init_count=1
    mbarrier_init_owner_lane(smem_raw + 16, 1);
  }

  // CUTLASS owner-lane publication sequence
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");

  __syncthreads();

  // === Task calls (dependency order) ===
  int q_block = blockIdx.x;
  int head = blockIdx.y;
  int batch = blockIdx.z;
  int q_base = q_block * 64;
  int metadata_row = (batch * num_heads + head) * ((seqlen_q + 64 - 1) / 64) + q_block;
  int selected_count = block_sparse_num;
  int index_base = metadata_row * q2k_capacity;
  if (warp == 0 && lane == 0 && q_block == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)(Q_map)) : "memory");
  }
  if (warp == 0 && lane == 0 && q_block == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)(K_map)) : "memory");
  }
  if (warp == 0 && lane == 0 && q_block == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)(V_map)) : "memory");
  }
  if (warp == 0 && lane == 0 && q_block == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)(O_map)) : "memory");
  }
  if (warp == 0) {
    if (elect_sync()) {
      mbarrier_arrive_expect_tx(q_full_addr, 8192);
      asm volatile(
          "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
          " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;" ::"r"(q_smem_addr),
          "l"(Q_map), "r"(0), "r"(q_base), "r"(head), "r"(batch), "r"(q_full_addr),
          "l"(0x14F0000000000000ULL)
          : "memory");
    }
  }
  int first_physical = 0;
  {
    {
      if (lane == 0) {
        first_physical = q2k_block_index[index_base + selected_count - 1];
      }
      int _shfl_0 = __shfl_sync(0xFFFFFFFF, first_physical, 0);
      first_physical = _shfl_0;
    }
    if (warp == 0) {
      if (elect_sync()) {
        mbarrier_arrive_expect_tx(k_full_addr, 8192);
        asm volatile(
            "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;" ::"r"(k_smem_addr),
            "l"(K_map), "r"(0), "r"(first_physical * 64), "r"(head), "r"(batch), "r"(k_full_addr),
            "l"(0x14F0000000000000ULL)
            : "memory");
        mbarrier_arrive_expect_tx(v_full_addr, 8192);
        asm volatile(
            "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;" ::"r"(v_tma_smem_addr),
            "l"(V_map), "r"(first_physical * 64), "r"(0), "r"(head), "r"(batch), "r"(v_full_addr),
            "l"(0x14F0000000000000ULL)
            : "memory");
      }
    }
  }
  unsigned int q_frag[4];
  unsigned int q_prefetch[4];
  unsigned int k_frag[4];
  unsigned int v_frag[4];
  int scores_i32[32];
  float scores[32];
  unsigned int probabilities[8];
  float output[64];
  float row_max[2];
  float row_sum[2];
  float pack_values[4];
  for (int row = 0; row < 2; row++) {
    row_max[row] = -CAKE_INF;
    row_sum[row] = 0.0f;
  }
  for (int feature = 0; feature < 8; feature++) {
    for (int elem = 0; elem < 8; elem++) {
      output[feature * 8 + elem] = 0.0f;
    }
  }
  int q_scale_index = q_base / 128 * 4 + (q_base % 128 + warp * 16) / 32;
  int q_scale_base = (batch * num_heads + head) * ((seqlen_q + 127) / 128 * 4);
  float q_scale_lane = 0.0f;
  if (lane == 0) {
    q_scale_lane = q_scale[q_scale_base + q_scale_index];
  }
  float _shfl_1 = __shfl_sync(0xFFFFFFFF, q_scale_lane, 0);
  float q_scale_value = _shfl_1;
  float q_softmax_log2 = q_scale_value * softmax_scale * 1.4426950408889634f;
  int k_scale_base = (batch * num_heads + head) * num_k_blocks;
  int v_scale_base = (batch * num_heads + head) * 128;
  unsigned int _phase_q_full_0 = 0;
  mbarrier_wait(q_full_addr, _phase_q_full_0);
  _phase_q_full_0 ^= 1;
  int current_physical = first_physical;
  unsigned int _phase_k_full_0 = 0;
  unsigned int _phase_v_full_0 = 0;
#pragma unroll 2
  for (int load_count = 0; load_count < selected_count; load_count++) {
    mbarrier_wait(k_full_addr, _phase_k_full_0);
    _phase_k_full_0 ^= 1;
    unsigned int q_row_base = warp * 16 + lane % 16;
    unsigned int q_col_base = lane / 16;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(q_frag[0]), "=r"(q_frag[1]), "=r"(q_frag[2]), "=r"(q_frag[3])
        : "r"(q_smem_addr +
              (q_row_base * 8 + (unsigned int)((q_col_base * 16 ^ (q_row_base & 7) << 4) / 16)) *
                  16)
        : "memory");
    unsigned int q_row_base_0 = warp * 16 + lane % 16;
    unsigned int q_col_base_1 = lane / 16;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(q_prefetch[0]), "=r"(q_prefetch[1]), "=r"(q_prefetch[2]),
                   "=r"(q_prefetch[3])
                 : "r"(q_smem_addr +
                       (q_row_base_0 * 8 +
                        (unsigned int)(((q_col_base_1 + 2) * 16 ^ (q_row_base_0 & 7) << 4) / 16)) *
                           16)
                 : "memory");
    unsigned int k_row_base = lane % 8 + lane / 16 * 8;
    unsigned int k_col_base = lane / 8 % 2;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
        : "r"(k_smem_addr +
              (k_row_base * 8 + (unsigned int)((k_col_base * 16 ^ (k_row_base & 7) << 4) / 16)) *
                  16)
        : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%10, %11, %12, %13};\n"
        : "=r"(scores_i32[0]), "=r"(scores_i32[1]), "=r"(scores_i32[2]), "=r"(scores_i32[3])
        : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[0]),
          "r"(k_frag[1]), "r"(0), "r"(0), "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%10, %11, %12, %13};\n"
        : "=r"(scores_i32[4]), "=r"(scores_i32[(4) + 1]), "=r"(scores_i32[(4) + 2]),
          "=r"(scores_i32[(4) + 3])
        : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[2]),
          "r"(k_frag[(2) + 1]), "r"(0), "r"(0), "r"(0), "r"(0));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
        : "r"(k_smem_addr + ((k_row_base + 16) * 8 +
                             (unsigned int)((k_col_base * 16 ^ (k_row_base + 16 & 7) << 4) / 16)) *
                                16)
        : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%10, %11, %12, %13};\n"
        : "=r"(scores_i32[8]), "=r"(scores_i32[(8) + 1]), "=r"(scores_i32[(8) + 2]),
          "=r"(scores_i32[(8) + 3])
        : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[0]),
          "r"(k_frag[1]), "r"(0), "r"(0), "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%10, %11, %12, %13};\n"
        : "=r"(scores_i32[12]), "=r"(scores_i32[(12) + 1]), "=r"(scores_i32[(12) + 2]),
          "=r"(scores_i32[(12) + 3])
        : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[2]),
          "r"(k_frag[(2) + 1]), "r"(0), "r"(0), "r"(0), "r"(0));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
        : "r"(k_smem_addr + ((k_row_base + 32) * 8 +
                             (unsigned int)((k_col_base * 16 ^ (k_row_base + 32 & 7) << 4) / 16)) *
                                16)
        : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%10, %11, %12, %13};\n"
        : "=r"(scores_i32[16]), "=r"(scores_i32[(16) + 1]), "=r"(scores_i32[(16) + 2]),
          "=r"(scores_i32[(16) + 3])
        : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[0]),
          "r"(k_frag[1]), "r"(0), "r"(0), "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%10, %11, %12, %13};\n"
        : "=r"(scores_i32[20]), "=r"(scores_i32[(20) + 1]), "=r"(scores_i32[(20) + 2]),
          "=r"(scores_i32[(20) + 3])
        : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[2]),
          "r"(k_frag[(2) + 1]), "r"(0), "r"(0), "r"(0), "r"(0));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
        : "r"(k_smem_addr + ((k_row_base + 48) * 8 +
                             (unsigned int)((k_col_base * 16 ^ (k_row_base + 48 & 7) << 4) / 16)) *
                                16)
        : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%10, %11, %12, %13};\n"
        : "=r"(scores_i32[24]), "=r"(scores_i32[(24) + 1]), "=r"(scores_i32[(24) + 2]),
          "=r"(scores_i32[(24) + 3])
        : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[0]),
          "r"(k_frag[1]), "r"(0), "r"(0), "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%10, %11, %12, %13};\n"
        : "=r"(scores_i32[28]), "=r"(scores_i32[(28) + 1]), "=r"(scores_i32[(28) + 2]),
          "=r"(scores_i32[(28) + 3])
        : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[2]),
          "r"(k_frag[(2) + 1]), "r"(0), "r"(0), "r"(0), "r"(0));
    unsigned int q_row_base_2 = warp * 16 + lane % 16;
    unsigned int q_col_base_3 = lane / 16;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(q_frag[0]), "=r"(q_frag[1]), "=r"(q_frag[2]), "=r"(q_frag[3])
                 : "r"(q_smem_addr +
                       (q_row_base_2 * 8 +
                        (unsigned int)(((q_col_base_3 + 4) * 16 ^ (q_row_base_2 & 7) << 4) / 16)) *
                           16)
                 : "memory");
    unsigned int k_row_base_4 = lane % 8 + lane / 16 * 8;
    unsigned int k_col_base_5 = lane / 8 % 2;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                 : "r"(k_smem_addr +
                       (k_row_base_4 * 8 +
                        (unsigned int)(((k_col_base_5 + 2) * 16 ^ (k_row_base_4 & 7) << 4) / 16)) *
                           16)
                 : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%0, %1, %2, %3};\n"
        : "+r"(scores_i32[0]), "+r"(scores_i32[1]), "+r"(scores_i32[2]), "+r"(scores_i32[3])
        : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]),
          "r"(k_frag[0]), "r"(k_frag[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%0, %1, %2, %3};\n"
        : "+r"(scores_i32[4]), "+r"(scores_i32[(4) + 1]), "+r"(scores_i32[(4) + 2]),
          "+r"(scores_i32[(4) + 3])
        : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]),
          "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
        : "r"(k_smem_addr +
              ((k_row_base_4 + 16) * 8 +
               (unsigned int)(((k_col_base_5 + 2) * 16 ^ (k_row_base_4 + 16 & 7) << 4) / 16)) *
                  16)
        : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%0, %1, %2, %3};\n"
        : "+r"(scores_i32[8]), "+r"(scores_i32[(8) + 1]), "+r"(scores_i32[(8) + 2]),
          "+r"(scores_i32[(8) + 3])
        : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]),
          "r"(k_frag[0]), "r"(k_frag[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%0, %1, %2, %3};\n"
        : "+r"(scores_i32[12]), "+r"(scores_i32[(12) + 1]), "+r"(scores_i32[(12) + 2]),
          "+r"(scores_i32[(12) + 3])
        : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]),
          "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
        : "r"(k_smem_addr +
              ((k_row_base_4 + 32) * 8 +
               (unsigned int)(((k_col_base_5 + 2) * 16 ^ (k_row_base_4 + 32 & 7) << 4) / 16)) *
                  16)
        : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%0, %1, %2, %3};\n"
        : "+r"(scores_i32[16]), "+r"(scores_i32[(16) + 1]), "+r"(scores_i32[(16) + 2]),
          "+r"(scores_i32[(16) + 3])
        : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]),
          "r"(k_frag[0]), "r"(k_frag[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%0, %1, %2, %3};\n"
        : "+r"(scores_i32[20]), "+r"(scores_i32[(20) + 1]), "+r"(scores_i32[(20) + 2]),
          "+r"(scores_i32[(20) + 3])
        : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]),
          "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
        : "r"(k_smem_addr +
              ((k_row_base_4 + 48) * 8 +
               (unsigned int)(((k_col_base_5 + 2) * 16 ^ (k_row_base_4 + 48 & 7) << 4) / 16)) *
                  16)
        : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%0, %1, %2, %3};\n"
        : "+r"(scores_i32[24]), "+r"(scores_i32[(24) + 1]), "+r"(scores_i32[(24) + 2]),
          "+r"(scores_i32[(24) + 3])
        : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]),
          "r"(k_frag[0]), "r"(k_frag[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%0, %1, %2, %3};\n"
        : "+r"(scores_i32[28]), "+r"(scores_i32[(28) + 1]), "+r"(scores_i32[(28) + 2]),
          "+r"(scores_i32[(28) + 3])
        : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]),
          "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
    unsigned int q_row_base_6 = warp * 16 + lane % 16;
    unsigned int q_col_base_7 = lane / 16;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(q_prefetch[0]), "=r"(q_prefetch[1]), "=r"(q_prefetch[2]),
                   "=r"(q_prefetch[3])
                 : "r"(q_smem_addr +
                       (q_row_base_6 * 8 +
                        (unsigned int)(((q_col_base_7 + 6) * 16 ^ (q_row_base_6 & 7) << 4) / 16)) *
                           16)
                 : "memory");
    unsigned int k_row_base_8 = lane % 8 + lane / 16 * 8;
    unsigned int k_col_base_9 = lane / 8 % 2;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                 : "r"(k_smem_addr +
                       (k_row_base_8 * 8 +
                        (unsigned int)(((k_col_base_9 + 4) * 16 ^ (k_row_base_8 & 7) << 4) / 16)) *
                           16)
                 : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%0, %1, %2, %3};\n"
        : "+r"(scores_i32[0]), "+r"(scores_i32[1]), "+r"(scores_i32[2]), "+r"(scores_i32[3])
        : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[0]),
          "r"(k_frag[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%0, %1, %2, %3};\n"
        : "+r"(scores_i32[4]), "+r"(scores_i32[(4) + 1]), "+r"(scores_i32[(4) + 2]),
          "+r"(scores_i32[(4) + 3])
        : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[2]),
          "r"(k_frag[(2) + 1]));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
        : "r"(k_smem_addr +
              ((k_row_base_8 + 16) * 8 +
               (unsigned int)(((k_col_base_9 + 4) * 16 ^ (k_row_base_8 + 16 & 7) << 4) / 16)) *
                  16)
        : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%0, %1, %2, %3};\n"
        : "+r"(scores_i32[8]), "+r"(scores_i32[(8) + 1]), "+r"(scores_i32[(8) + 2]),
          "+r"(scores_i32[(8) + 3])
        : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[0]),
          "r"(k_frag[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%0, %1, %2, %3};\n"
        : "+r"(scores_i32[12]), "+r"(scores_i32[(12) + 1]), "+r"(scores_i32[(12) + 2]),
          "+r"(scores_i32[(12) + 3])
        : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[2]),
          "r"(k_frag[(2) + 1]));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
        : "r"(k_smem_addr +
              ((k_row_base_8 + 32) * 8 +
               (unsigned int)(((k_col_base_9 + 4) * 16 ^ (k_row_base_8 + 32 & 7) << 4) / 16)) *
                  16)
        : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%0, %1, %2, %3};\n"
        : "+r"(scores_i32[16]), "+r"(scores_i32[(16) + 1]), "+r"(scores_i32[(16) + 2]),
          "+r"(scores_i32[(16) + 3])
        : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[0]),
          "r"(k_frag[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%0, %1, %2, %3};\n"
        : "+r"(scores_i32[20]), "+r"(scores_i32[(20) + 1]), "+r"(scores_i32[(20) + 2]),
          "+r"(scores_i32[(20) + 3])
        : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[2]),
          "r"(k_frag[(2) + 1]));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
        : "r"(k_smem_addr +
              ((k_row_base_8 + 48) * 8 +
               (unsigned int)(((k_col_base_9 + 4) * 16 ^ (k_row_base_8 + 48 & 7) << 4) / 16)) *
                  16)
        : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%0, %1, %2, %3};\n"
        : "+r"(scores_i32[24]), "+r"(scores_i32[(24) + 1]), "+r"(scores_i32[(24) + 2]),
          "+r"(scores_i32[(24) + 3])
        : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[0]),
          "r"(k_frag[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%0, %1, %2, %3};\n"
        : "+r"(scores_i32[28]), "+r"(scores_i32[(28) + 1]), "+r"(scores_i32[(28) + 2]),
          "+r"(scores_i32[(28) + 3])
        : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[2]),
          "r"(k_frag[(2) + 1]));
    unsigned int k_row_base_10 = lane % 8 + lane / 16 * 8;
    unsigned int k_col_base_11 = lane / 8 % 2;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
        : "r"(k_smem_addr +
              (k_row_base_10 * 8 +
               (unsigned int)(((k_col_base_11 + 6) * 16 ^ (k_row_base_10 & 7) << 4) / 16)) *
                  16)
        : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%0, %1, %2, %3};\n"
        : "+r"(scores_i32[0]), "+r"(scores_i32[1]), "+r"(scores_i32[2]), "+r"(scores_i32[3])
        : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]),
          "r"(k_frag[0]), "r"(k_frag[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%0, %1, %2, %3};\n"
        : "+r"(scores_i32[4]), "+r"(scores_i32[(4) + 1]), "+r"(scores_i32[(4) + 2]),
          "+r"(scores_i32[(4) + 3])
        : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]),
          "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
        : "r"(k_smem_addr +
              ((k_row_base_10 + 16) * 8 +
               (unsigned int)(((k_col_base_11 + 6) * 16 ^ (k_row_base_10 + 16 & 7) << 4) / 16)) *
                  16)
        : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%0, %1, %2, %3};\n"
        : "+r"(scores_i32[8]), "+r"(scores_i32[(8) + 1]), "+r"(scores_i32[(8) + 2]),
          "+r"(scores_i32[(8) + 3])
        : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]),
          "r"(k_frag[0]), "r"(k_frag[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%0, %1, %2, %3};\n"
        : "+r"(scores_i32[12]), "+r"(scores_i32[(12) + 1]), "+r"(scores_i32[(12) + 2]),
          "+r"(scores_i32[(12) + 3])
        : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]),
          "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
        : "r"(k_smem_addr +
              ((k_row_base_10 + 32) * 8 +
               (unsigned int)(((k_col_base_11 + 6) * 16 ^ (k_row_base_10 + 32 & 7) << 4) / 16)) *
                  16)
        : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%0, %1, %2, %3};\n"
        : "+r"(scores_i32[16]), "+r"(scores_i32[(16) + 1]), "+r"(scores_i32[(16) + 2]),
          "+r"(scores_i32[(16) + 3])
        : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]),
          "r"(k_frag[0]), "r"(k_frag[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%0, %1, %2, %3};\n"
        : "+r"(scores_i32[20]), "+r"(scores_i32[(20) + 1]), "+r"(scores_i32[(20) + 2]),
          "+r"(scores_i32[(20) + 3])
        : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]),
          "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
        : "r"(k_smem_addr +
              ((k_row_base_10 + 48) * 8 +
               (unsigned int)(((k_col_base_11 + 6) * 16 ^ (k_row_base_10 + 48 & 7) << 4) / 16)) *
                  16)
        : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%0, %1, %2, %3};\n"
        : "+r"(scores_i32[24]), "+r"(scores_i32[(24) + 1]), "+r"(scores_i32[(24) + 2]),
          "+r"(scores_i32[(24) + 3])
        : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]),
          "r"(k_frag[0]), "r"(k_frag[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
        "%9}, {%0, %1, %2, %3};\n"
        : "+r"(scores_i32[28]), "+r"(scores_i32[(28) + 1]), "+r"(scores_i32[(28) + 2]),
          "+r"(scores_i32[(28) + 3])
        : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]),
          "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
    float k_scale_lane = 0.0f;
    if (lane == 0) {
      k_scale_lane = k_scale[k_scale_base + current_physical];
    }
    float _shfl_2 = __shfl_sync(0xFFFFFFFF, k_scale_lane, 0);
    float k_scale_value = _shfl_2;
    float score_scale_log2 = q_softmax_log2 * k_scale_value;
    int valid_tokens = 64;
    {
      valid_tokens = seqlen_k - current_physical * 64;
      if (valid_tokens > 64) {
        valid_tokens = 64;
      }
      if (valid_tokens < 0) {
        valid_tokens = 0;
      }
      {
        {
        }
      }
      if (valid_tokens < 0) {
        valid_tokens = 0;
      }
    }
    int preload_count = load_count + 1;
    int next_physical = 0;
    if (preload_count < selected_count) {
      {
        next_physical = current_physical - 1;
        if (next_physical < 0) {
          next_physical = next_physical + num_k_blocks;
        }
      }
      __syncthreads();
      if (warp == 0) {
        if (elect_sync()) {
          mbarrier_arrive_expect_tx(k_full_addr, 8192);
          asm volatile(
              "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_"
              "hint"
              " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;" ::"r"(k_smem_addr),
              "l"(K_map), "r"(0), "r"(next_physical * 64), "r"(head), "r"(batch), "r"(k_full_addr),
              "l"(0x14F0000000000000ULL)
              : "memory");
        }
      }
    }
    scores[0] = (float)scores_i32[0];
    scores[1] = (float)scores_i32[1];
    scores[2] = (float)scores_i32[2];
    scores[3] = (float)scores_i32[3];
    scores[4] = (float)scores_i32[4];
    scores[5] = (float)scores_i32[5];
    scores[6] = (float)scores_i32[6];
    scores[7] = (float)scores_i32[7];
    scores[8] = (float)scores_i32[8];
    scores[9] = (float)scores_i32[9];
    scores[10] = (float)scores_i32[10];
    scores[11] = (float)scores_i32[11];
    scores[12] = (float)scores_i32[12];
    scores[13] = (float)scores_i32[13];
    scores[14] = (float)scores_i32[14];
    scores[15] = (float)scores_i32[15];
    scores[16] = (float)scores_i32[16];
    scores[17] = (float)scores_i32[17];
    scores[18] = (float)scores_i32[18];
    scores[19] = (float)scores_i32[19];
    scores[20] = (float)scores_i32[20];
    scores[21] = (float)scores_i32[21];
    scores[22] = (float)scores_i32[22];
    scores[23] = (float)scores_i32[23];
    scores[24] = (float)scores_i32[24];
    scores[25] = (float)scores_i32[25];
    scores[26] = (float)scores_i32[26];
    scores[27] = (float)scores_i32[27];
    scores[28] = (float)scores_i32[28];
    scores[29] = (float)scores_i32[29];
    scores[30] = (float)scores_i32[30];
    scores[31] = (float)scores_i32[31];
    if (valid_tokens < 64) {
      int lane_col_base = 2 * (lane % 4);
      int token_in_tile = lane_col_base;
      if (token_in_tile >= valid_tokens) {
        scores[0] = -CAKE_INF;
      }
      int token_in_tile_0 = lane_col_base + 1;
      if (token_in_tile_0 >= valid_tokens) {
        scores[1] = -CAKE_INF;
      }
      int token_in_tile_1 = lane_col_base;
      if (token_in_tile_1 >= valid_tokens) {
        scores[2] = -CAKE_INF;
      }
      int token_in_tile_2 = lane_col_base + 1;
      if (token_in_tile_2 >= valid_tokens) {
        scores[3] = -CAKE_INF;
      }
      int token_in_tile_3 = lane_col_base + 8;
      if (token_in_tile_3 >= valid_tokens) {
        scores[4] = -CAKE_INF;
      }
      int token_in_tile_4 = lane_col_base + 8 + 1;
      if (token_in_tile_4 >= valid_tokens) {
        scores[5] = -CAKE_INF;
      }
      int token_in_tile_5 = lane_col_base + 8;
      if (token_in_tile_5 >= valid_tokens) {
        scores[6] = -CAKE_INF;
      }
      int token_in_tile_6 = lane_col_base + 8 + 1;
      if (token_in_tile_6 >= valid_tokens) {
        scores[7] = -CAKE_INF;
      }
      int token_in_tile_7 = 16 + lane_col_base;
      if (token_in_tile_7 >= valid_tokens) {
        scores[8] = -CAKE_INF;
      }
      int token_in_tile_8 = 16 + lane_col_base + 1;
      if (token_in_tile_8 >= valid_tokens) {
        scores[9] = -CAKE_INF;
      }
      int token_in_tile_9 = 16 + lane_col_base;
      if (token_in_tile_9 >= valid_tokens) {
        scores[10] = -CAKE_INF;
      }
      int token_in_tile_10 = 16 + lane_col_base + 1;
      if (token_in_tile_10 >= valid_tokens) {
        scores[11] = -CAKE_INF;
      }
      int token_in_tile_11 = 16 + lane_col_base + 8;
      if (token_in_tile_11 >= valid_tokens) {
        scores[12] = -CAKE_INF;
      }
      int token_in_tile_12 = 16 + lane_col_base + 8 + 1;
      if (token_in_tile_12 >= valid_tokens) {
        scores[13] = -CAKE_INF;
      }
      int token_in_tile_13 = 16 + lane_col_base + 8;
      if (token_in_tile_13 >= valid_tokens) {
        scores[14] = -CAKE_INF;
      }
      int token_in_tile_14 = 16 + lane_col_base + 8 + 1;
      if (token_in_tile_14 >= valid_tokens) {
        scores[15] = -CAKE_INF;
      }
      int token_in_tile_15 = 32 + lane_col_base;
      if (token_in_tile_15 >= valid_tokens) {
        scores[16] = -CAKE_INF;
      }
      int token_in_tile_16 = 32 + lane_col_base + 1;
      if (token_in_tile_16 >= valid_tokens) {
        scores[17] = -CAKE_INF;
      }
      int token_in_tile_17 = 32 + lane_col_base;
      if (token_in_tile_17 >= valid_tokens) {
        scores[18] = -CAKE_INF;
      }
      int token_in_tile_18 = 32 + lane_col_base + 1;
      if (token_in_tile_18 >= valid_tokens) {
        scores[19] = -CAKE_INF;
      }
      int token_in_tile_19 = 32 + lane_col_base + 8;
      if (token_in_tile_19 >= valid_tokens) {
        scores[20] = -CAKE_INF;
      }
      int token_in_tile_20 = 32 + lane_col_base + 8 + 1;
      if (token_in_tile_20 >= valid_tokens) {
        scores[21] = -CAKE_INF;
      }
      int token_in_tile_21 = 32 + lane_col_base + 8;
      if (token_in_tile_21 >= valid_tokens) {
        scores[22] = -CAKE_INF;
      }
      int token_in_tile_22 = 32 + lane_col_base + 8 + 1;
      if (token_in_tile_22 >= valid_tokens) {
        scores[23] = -CAKE_INF;
      }
      int token_in_tile_23 = 48 + lane_col_base;
      if (token_in_tile_23 >= valid_tokens) {
        scores[24] = -CAKE_INF;
      }
      int token_in_tile_24 = 48 + lane_col_base + 1;
      if (token_in_tile_24 >= valid_tokens) {
        scores[25] = -CAKE_INF;
      }
      int token_in_tile_25 = 48 + lane_col_base;
      if (token_in_tile_25 >= valid_tokens) {
        scores[26] = -CAKE_INF;
      }
      int token_in_tile_26 = 48 + lane_col_base + 1;
      if (token_in_tile_26 >= valid_tokens) {
        scores[27] = -CAKE_INF;
      }
      int token_in_tile_27 = 48 + lane_col_base + 8;
      if (token_in_tile_27 >= valid_tokens) {
        scores[28] = -CAKE_INF;
      }
      int token_in_tile_28 = 48 + lane_col_base + 8 + 1;
      if (token_in_tile_28 >= valid_tokens) {
        scores[29] = -CAKE_INF;
      }
      int token_in_tile_29 = 48 + lane_col_base + 8;
      if (token_in_tile_29 >= valid_tokens) {
        scores[30] = -CAKE_INF;
      }
      int token_in_tile_30 = 48 + lane_col_base + 8 + 1;
      if (token_in_tile_30 >= valid_tokens) {
        scores[31] = -CAKE_INF;
      }
    }
    float row_scale[2];
    float shifted_max[2];
    float previous_max = row_max[0];
    float group_max = -CAKE_INF;
    float _max_0 = max_noftz(scores[0], scores[1]);
    float _max_1 = max_noftz(scores[4], scores[5]);
    float _max_2 = max_noftz(_max_0, _max_1);
    float _max_3 = max_noftz(group_max, _max_2);
    group_max = _max_3;
    float _max_4 = max_noftz(scores[8], scores[9]);
    float _max_5 = max_noftz(scores[12], scores[13]);
    float _max_6 = max_noftz(_max_4, _max_5);
    float _max_7 = max_noftz(group_max, _max_6);
    group_max = _max_7;
    float _max_8 = max_noftz(scores[16], scores[17]);
    float _max_9 = max_noftz(scores[20], scores[21]);
    float _max_10 = max_noftz(_max_8, _max_9);
    float _max_11 = max_noftz(group_max, _max_10);
    group_max = _max_11;
    float _max_12 = max_noftz(scores[24], scores[25]);
    float _max_13 = max_noftz(scores[28], scores[29]);
    float _max_14 = max_noftz(_max_12, _max_13);
    float _max_15 = max_noftz(group_max, _max_14);
    group_max = _max_15;
    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, group_max, 1);
    float _max_16 = max_noftz(group_max, _shfl_xor_0);
    group_max = _max_16;
    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, group_max, 2);
    float _max_17 = max_noftz(group_max, _shfl_xor_1);
    group_max = _max_17;
    float _max_18 = max_noftz(previous_max, group_max * score_scale_log2);
    float current_max = _max_18;
    float safe_max = ((current_max == -CAKE_INF) ? 0.0f : current_max);
    float delta = previous_max - safe_max;
    if (delta >= -0.8073549220576041f) {
      current_max = previous_max;
      safe_max = previous_max;
      row_scale[0] = 1.0f;
    } else {
      float _exp2_0 = approx_exp2(delta);
      row_scale[0] = _exp2_0;
    }
    row_max[0] = current_max;
    shifted_max[0] = safe_max - 8.0f;
    float previous_max_12 = row_max[1];
    float group_max_13 = -CAKE_INF;
    float _max_19 = max_noftz(scores[2], scores[3]);
    float _max_20 = max_noftz(scores[6], scores[7]);
    float _max_21 = max_noftz(_max_19, _max_20);
    float _max_22 = max_noftz(group_max_13, _max_21);
    group_max_13 = _max_22;
    float _max_23 = max_noftz(scores[10], scores[11]);
    float _max_24 = max_noftz(scores[14], scores[15]);
    float _max_25 = max_noftz(_max_23, _max_24);
    float _max_26 = max_noftz(group_max_13, _max_25);
    group_max_13 = _max_26;
    float _max_27 = max_noftz(scores[18], scores[19]);
    float _max_28 = max_noftz(scores[22], scores[23]);
    float _max_29 = max_noftz(_max_27, _max_28);
    float _max_30 = max_noftz(group_max_13, _max_29);
    group_max_13 = _max_30;
    float _max_31 = max_noftz(scores[26], scores[27]);
    float _max_32 = max_noftz(scores[30], scores[31]);
    float _max_33 = max_noftz(_max_31, _max_32);
    float _max_34 = max_noftz(group_max_13, _max_33);
    group_max_13 = _max_34;
    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, group_max_13, 1);
    float _max_35 = max_noftz(group_max_13, _shfl_xor_2);
    group_max_13 = _max_35;
    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, group_max_13, 2);
    float _max_36 = max_noftz(group_max_13, _shfl_xor_3);
    group_max_13 = _max_36;
    float _max_37 = max_noftz(previous_max_12, group_max_13 * score_scale_log2);
    float current_max_14 = _max_37;
    float safe_max_15 = ((current_max_14 == -CAKE_INF) ? 0.0f : current_max_14);
    float delta_16 = previous_max_12 - safe_max_15;
    if (delta_16 >= -0.8073549220576041f) {
      current_max_14 = previous_max_12;
      safe_max_15 = previous_max_12;
      row_scale[1] = 1.0f;
    } else {
      float _exp2_1 = approx_exp2(delta_16);
      row_scale[1] = _exp2_1;
    }
    row_max[1] = current_max_14;
    shifted_max[1] = safe_max_15 - 8.0f;
    float local_sum[2];
    local_sum[0] = 0.0f;
    local_sum[1] = 0.0f;
    float _fma_0 = __fmaf_rn(scores[0], score_scale_log2, -shifted_max[0]);
    float _exp2_2 = approx_exp2(_fma_0);
    float value0 = _exp2_2;
    scores[0] = value0;
    local_sum[0] = local_sum[0] + value0;
    float _fma_1 = __fmaf_rn(scores[2], score_scale_log2, -shifted_max[1]);
    float _exp2_3 = approx_exp2(_fma_1);
    float value1 = _exp2_3;
    scores[2] = value1;
    local_sum[1] = local_sum[1] + value1;
    float _fma_2 = __fmaf_rn(scores[1], score_scale_log2, -shifted_max[0]);
    float _exp2_4 = approx_exp2(_fma_2);
    float value0_17 = _exp2_4;
    scores[1] = value0_17;
    local_sum[0] = local_sum[0] + value0_17;
    float _fma_3 = __fmaf_rn(scores[3], score_scale_log2, -shifted_max[1]);
    float _exp2_5 = approx_exp2(_fma_3);
    float value1_18 = _exp2_5;
    scores[3] = value1_18;
    local_sum[1] = local_sum[1] + value1_18;
    float _fma_4 = __fmaf_rn(scores[4], score_scale_log2, -shifted_max[0]);
    float _exp2_6 = approx_exp2(_fma_4);
    float value0_19 = _exp2_6;
    scores[4] = value0_19;
    local_sum[0] = local_sum[0] + value0_19;
    float _fma_5 = __fmaf_rn(scores[6], score_scale_log2, -shifted_max[1]);
    float _exp2_7 = approx_exp2(_fma_5);
    float value1_20 = _exp2_7;
    scores[6] = value1_20;
    local_sum[1] = local_sum[1] + value1_20;
    float _fma_6 = __fmaf_rn(scores[5], score_scale_log2, -shifted_max[0]);
    float _exp2_8 = approx_exp2(_fma_6);
    float value0_21 = _exp2_8;
    scores[5] = value0_21;
    local_sum[0] = local_sum[0] + value0_21;
    float _fma_7 = __fmaf_rn(scores[7], score_scale_log2, -shifted_max[1]);
    float _exp2_9 = approx_exp2(_fma_7);
    float value1_22 = _exp2_9;
    scores[7] = value1_22;
    local_sum[1] = local_sum[1] + value1_22;
    float _fma_8 = __fmaf_rn(scores[8], score_scale_log2, -shifted_max[0]);
    float _exp2_10 = approx_exp2(_fma_8);
    float value0_23 = _exp2_10;
    scores[8] = value0_23;
    local_sum[0] = local_sum[0] + value0_23;
    float _fma_9 = __fmaf_rn(scores[10], score_scale_log2, -shifted_max[1]);
    float _exp2_11 = approx_exp2(_fma_9);
    float value1_24 = _exp2_11;
    scores[10] = value1_24;
    local_sum[1] = local_sum[1] + value1_24;
    float _fma_10 = __fmaf_rn(scores[9], score_scale_log2, -shifted_max[0]);
    float _exp2_12 = approx_exp2(_fma_10);
    float value0_25 = _exp2_12;
    scores[9] = value0_25;
    local_sum[0] = local_sum[0] + value0_25;
    float _fma_11 = __fmaf_rn(scores[11], score_scale_log2, -shifted_max[1]);
    float _exp2_13 = approx_exp2(_fma_11);
    float value1_26 = _exp2_13;
    scores[11] = value1_26;
    local_sum[1] = local_sum[1] + value1_26;
    float _fma_12 = __fmaf_rn(scores[12], score_scale_log2, -shifted_max[0]);
    float _exp2_14 = approx_exp2(_fma_12);
    float value0_27 = _exp2_14;
    scores[12] = value0_27;
    local_sum[0] = local_sum[0] + value0_27;
    float _fma_13 = __fmaf_rn(scores[14], score_scale_log2, -shifted_max[1]);
    float _exp2_15 = approx_exp2(_fma_13);
    float value1_28 = _exp2_15;
    scores[14] = value1_28;
    local_sum[1] = local_sum[1] + value1_28;
    float _fma_14 = __fmaf_rn(scores[13], score_scale_log2, -shifted_max[0]);
    float _exp2_16 = approx_exp2(_fma_14);
    float value0_29 = _exp2_16;
    scores[13] = value0_29;
    local_sum[0] = local_sum[0] + value0_29;
    float _fma_15 = __fmaf_rn(scores[15], score_scale_log2, -shifted_max[1]);
    float _exp2_17 = approx_exp2(_fma_15);
    float value1_30 = _exp2_17;
    scores[15] = value1_30;
    local_sum[1] = local_sum[1] + value1_30;
    float _fma_16 = __fmaf_rn(scores[16], score_scale_log2, -shifted_max[0]);
    float _exp2_18 = approx_exp2(_fma_16);
    float value0_31 = _exp2_18;
    scores[16] = value0_31;
    local_sum[0] = local_sum[0] + value0_31;
    float _fma_17 = __fmaf_rn(scores[18], score_scale_log2, -shifted_max[1]);
    float _exp2_19 = approx_exp2(_fma_17);
    float value1_32 = _exp2_19;
    scores[18] = value1_32;
    local_sum[1] = local_sum[1] + value1_32;
    float _fma_18 = __fmaf_rn(scores[17], score_scale_log2, -shifted_max[0]);
    float _exp2_20 = approx_exp2(_fma_18);
    float value0_33 = _exp2_20;
    scores[17] = value0_33;
    local_sum[0] = local_sum[0] + value0_33;
    float _fma_19 = __fmaf_rn(scores[19], score_scale_log2, -shifted_max[1]);
    float _exp2_21 = approx_exp2(_fma_19);
    float value1_34 = _exp2_21;
    scores[19] = value1_34;
    local_sum[1] = local_sum[1] + value1_34;
    float _fma_20 = __fmaf_rn(scores[20], score_scale_log2, -shifted_max[0]);
    float _exp2_22 = approx_exp2(_fma_20);
    float value0_35 = _exp2_22;
    scores[20] = value0_35;
    local_sum[0] = local_sum[0] + value0_35;
    float _fma_21 = __fmaf_rn(scores[22], score_scale_log2, -shifted_max[1]);
    float _exp2_23 = approx_exp2(_fma_21);
    float value1_36 = _exp2_23;
    scores[22] = value1_36;
    local_sum[1] = local_sum[1] + value1_36;
    float _fma_22 = __fmaf_rn(scores[21], score_scale_log2, -shifted_max[0]);
    float _exp2_24 = approx_exp2(_fma_22);
    float value0_37 = _exp2_24;
    scores[21] = value0_37;
    local_sum[0] = local_sum[0] + value0_37;
    float _fma_23 = __fmaf_rn(scores[23], score_scale_log2, -shifted_max[1]);
    float _exp2_25 = approx_exp2(_fma_23);
    float value1_38 = _exp2_25;
    scores[23] = value1_38;
    local_sum[1] = local_sum[1] + value1_38;
    float _fma_24 = __fmaf_rn(scores[24], score_scale_log2, -shifted_max[0]);
    float _exp2_26 = approx_exp2(_fma_24);
    float value0_39 = _exp2_26;
    scores[24] = value0_39;
    local_sum[0] = local_sum[0] + value0_39;
    float _fma_25 = __fmaf_rn(scores[26], score_scale_log2, -shifted_max[1]);
    float _exp2_27 = approx_exp2(_fma_25);
    float value1_40 = _exp2_27;
    scores[26] = value1_40;
    local_sum[1] = local_sum[1] + value1_40;
    float _fma_26 = __fmaf_rn(scores[25], score_scale_log2, -shifted_max[0]);
    float _exp2_28 = approx_exp2(_fma_26);
    float value0_41 = _exp2_28;
    scores[25] = value0_41;
    local_sum[0] = local_sum[0] + value0_41;
    float _fma_27 = __fmaf_rn(scores[27], score_scale_log2, -shifted_max[1]);
    float _exp2_29 = approx_exp2(_fma_27);
    float value1_42 = _exp2_29;
    scores[27] = value1_42;
    local_sum[1] = local_sum[1] + value1_42;
    float _fma_28 = __fmaf_rn(scores[28], score_scale_log2, -shifted_max[0]);
    float _exp2_30 = approx_exp2(_fma_28);
    float value0_43 = _exp2_30;
    scores[28] = value0_43;
    local_sum[0] = local_sum[0] + value0_43;
    float _fma_29 = __fmaf_rn(scores[30], score_scale_log2, -shifted_max[1]);
    float _exp2_31 = approx_exp2(_fma_29);
    float value1_44 = _exp2_31;
    scores[30] = value1_44;
    local_sum[1] = local_sum[1] + value1_44;
    float _fma_30 = __fmaf_rn(scores[29], score_scale_log2, -shifted_max[0]);
    float _exp2_32 = approx_exp2(_fma_30);
    float value0_45 = _exp2_32;
    scores[29] = value0_45;
    local_sum[0] = local_sum[0] + value0_45;
    float _fma_31 = __fmaf_rn(scores[31], score_scale_log2, -shifted_max[1]);
    float _exp2_33 = approx_exp2(_fma_31);
    float value1_46 = _exp2_33;
    scores[31] = value1_46;
    local_sum[1] = local_sum[1] + value1_46;
    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, local_sum[0], 1);
    local_sum[0] = local_sum[0] + _shfl_xor_4;
    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, local_sum[0], 2);
    local_sum[0] = local_sum[0] + _shfl_xor_5;
    float _fma_32 = __fmaf_rn(row_sum[0], row_scale[0], local_sum[0]);
    row_sum[0] = _fma_32;
    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, local_sum[1], 1);
    local_sum[1] = local_sum[1] + _shfl_xor_6;
    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, local_sum[1], 2);
    local_sum[1] = local_sum[1] + _shfl_xor_7;
    float _fma_33 = __fmaf_rn(row_sum[1], row_scale[1], local_sum[1]);
    row_sum[1] = _fma_33;
    int _vote_0 = __any_sync(0xFFFFFFFF, row_scale[0] < 1.0f || row_scale[1] < 1.0f);
    if (_vote_0 != 0) {
      {
        float2 _pair_scale_even2_0 = make_float2(row_scale[0], row_scale[0]);
        float2 _pair_scale_odd2_0 = make_float2(row_scale[1], row_scale[1]);
        float2* _pair_scale_src2_0 = reinterpret_cast<float2*>(&(output + 0)[0]);
#if __CUDA_ARCH__ >= 1000
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_0[0])
                     : "l"(*(unsigned long long*)&_pair_scale_even2_0));
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_0[1])
                     : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_0[2])
                     : "l"(*(unsigned long long*)&_pair_scale_even2_0));
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_0[3])
                     : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
#else
        (output + 0)[0] *= row_scale[0];
        (output + 0)[1] *= row_scale[0];
        (output + 0)[2] *= row_scale[1];
        (output + 0)[3] *= row_scale[1];
        (output + 0)[4] *= row_scale[0];
        (output + 0)[5] *= row_scale[0];
        (output + 0)[6] *= row_scale[1];
        (output + 0)[7] *= row_scale[1];
#endif
      }
      {
        float2 _pair_scale_even2_1 = make_float2(row_scale[0], row_scale[0]);
        float2 _pair_scale_odd2_1 = make_float2(row_scale[1], row_scale[1]);
        float2* _pair_scale_src2_1 = reinterpret_cast<float2*>(&(output + 8)[0]);
#if __CUDA_ARCH__ >= 1000
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_1[0])
                     : "l"(*(unsigned long long*)&_pair_scale_even2_1));
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_1[1])
                     : "l"(*(unsigned long long*)&_pair_scale_odd2_1));
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_1[2])
                     : "l"(*(unsigned long long*)&_pair_scale_even2_1));
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_1[3])
                     : "l"(*(unsigned long long*)&_pair_scale_odd2_1));
#else
        (output + 8)[0] *= row_scale[0];
        (output + 8)[1] *= row_scale[0];
        (output + 8)[2] *= row_scale[1];
        (output + 8)[3] *= row_scale[1];
        (output + 8)[4] *= row_scale[0];
        (output + 8)[5] *= row_scale[0];
        (output + 8)[6] *= row_scale[1];
        (output + 8)[7] *= row_scale[1];
#endif
      }
      {
        float2 _pair_scale_even2_2 = make_float2(row_scale[0], row_scale[0]);
        float2 _pair_scale_odd2_2 = make_float2(row_scale[1], row_scale[1]);
        float2* _pair_scale_src2_2 = reinterpret_cast<float2*>(&(output + 16)[0]);
#if __CUDA_ARCH__ >= 1000
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_2[0])
                     : "l"(*(unsigned long long*)&_pair_scale_even2_2));
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_2[1])
                     : "l"(*(unsigned long long*)&_pair_scale_odd2_2));
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_2[2])
                     : "l"(*(unsigned long long*)&_pair_scale_even2_2));
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_2[3])
                     : "l"(*(unsigned long long*)&_pair_scale_odd2_2));
#else
        (output + 16)[0] *= row_scale[0];
        (output + 16)[1] *= row_scale[0];
        (output + 16)[2] *= row_scale[1];
        (output + 16)[3] *= row_scale[1];
        (output + 16)[4] *= row_scale[0];
        (output + 16)[5] *= row_scale[0];
        (output + 16)[6] *= row_scale[1];
        (output + 16)[7] *= row_scale[1];
#endif
      }
      {
        float2 _pair_scale_even2_3 = make_float2(row_scale[0], row_scale[0]);
        float2 _pair_scale_odd2_3 = make_float2(row_scale[1], row_scale[1]);
        float2* _pair_scale_src2_3 = reinterpret_cast<float2*>(&(output + 24)[0]);
#if __CUDA_ARCH__ >= 1000
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_3[0])
                     : "l"(*(unsigned long long*)&_pair_scale_even2_3));
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_3[1])
                     : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_3[2])
                     : "l"(*(unsigned long long*)&_pair_scale_even2_3));
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_3[3])
                     : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
#else
        (output + 24)[0] *= row_scale[0];
        (output + 24)[1] *= row_scale[0];
        (output + 24)[2] *= row_scale[1];
        (output + 24)[3] *= row_scale[1];
        (output + 24)[4] *= row_scale[0];
        (output + 24)[5] *= row_scale[0];
        (output + 24)[6] *= row_scale[1];
        (output + 24)[7] *= row_scale[1];
#endif
      }
      {
        float2 _pair_scale_even2_4 = make_float2(row_scale[0], row_scale[0]);
        float2 _pair_scale_odd2_4 = make_float2(row_scale[1], row_scale[1]);
        float2* _pair_scale_src2_4 = reinterpret_cast<float2*>(&(output + 32)[0]);
#if __CUDA_ARCH__ >= 1000
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_4[0])
                     : "l"(*(unsigned long long*)&_pair_scale_even2_4));
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_4[1])
                     : "l"(*(unsigned long long*)&_pair_scale_odd2_4));
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_4[2])
                     : "l"(*(unsigned long long*)&_pair_scale_even2_4));
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_4[3])
                     : "l"(*(unsigned long long*)&_pair_scale_odd2_4));
#else
        (output + 32)[0] *= row_scale[0];
        (output + 32)[1] *= row_scale[0];
        (output + 32)[2] *= row_scale[1];
        (output + 32)[3] *= row_scale[1];
        (output + 32)[4] *= row_scale[0];
        (output + 32)[5] *= row_scale[0];
        (output + 32)[6] *= row_scale[1];
        (output + 32)[7] *= row_scale[1];
#endif
      }
      {
        float2 _pair_scale_even2_5 = make_float2(row_scale[0], row_scale[0]);
        float2 _pair_scale_odd2_5 = make_float2(row_scale[1], row_scale[1]);
        float2* _pair_scale_src2_5 = reinterpret_cast<float2*>(&(output + 40)[0]);
#if __CUDA_ARCH__ >= 1000
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_5[0])
                     : "l"(*(unsigned long long*)&_pair_scale_even2_5));
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_5[1])
                     : "l"(*(unsigned long long*)&_pair_scale_odd2_5));
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_5[2])
                     : "l"(*(unsigned long long*)&_pair_scale_even2_5));
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_5[3])
                     : "l"(*(unsigned long long*)&_pair_scale_odd2_5));
#else
        (output + 40)[0] *= row_scale[0];
        (output + 40)[1] *= row_scale[0];
        (output + 40)[2] *= row_scale[1];
        (output + 40)[3] *= row_scale[1];
        (output + 40)[4] *= row_scale[0];
        (output + 40)[5] *= row_scale[0];
        (output + 40)[6] *= row_scale[1];
        (output + 40)[7] *= row_scale[1];
#endif
      }
      {
        float2 _pair_scale_even2_6 = make_float2(row_scale[0], row_scale[0]);
        float2 _pair_scale_odd2_6 = make_float2(row_scale[1], row_scale[1]);
        float2* _pair_scale_src2_6 = reinterpret_cast<float2*>(&(output + 48)[0]);
#if __CUDA_ARCH__ >= 1000
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_6[0])
                     : "l"(*(unsigned long long*)&_pair_scale_even2_6));
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_6[1])
                     : "l"(*(unsigned long long*)&_pair_scale_odd2_6));
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_6[2])
                     : "l"(*(unsigned long long*)&_pair_scale_even2_6));
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_6[3])
                     : "l"(*(unsigned long long*)&_pair_scale_odd2_6));
#else
        (output + 48)[0] *= row_scale[0];
        (output + 48)[1] *= row_scale[0];
        (output + 48)[2] *= row_scale[1];
        (output + 48)[3] *= row_scale[1];
        (output + 48)[4] *= row_scale[0];
        (output + 48)[5] *= row_scale[0];
        (output + 48)[6] *= row_scale[1];
        (output + 48)[7] *= row_scale[1];
#endif
      }
      {
        float2 _pair_scale_even2_7 = make_float2(row_scale[0], row_scale[0]);
        float2 _pair_scale_odd2_7 = make_float2(row_scale[1], row_scale[1]);
        float2* _pair_scale_src2_7 = reinterpret_cast<float2*>(&(output + 56)[0]);
#if __CUDA_ARCH__ >= 1000
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_7[0])
                     : "l"(*(unsigned long long*)&_pair_scale_even2_7));
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_7[1])
                     : "l"(*(unsigned long long*)&_pair_scale_odd2_7));
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_7[2])
                     : "l"(*(unsigned long long*)&_pair_scale_even2_7));
        asm volatile("mul.rn.ftz.f32x2 %0, %0, %1;"
                     : "+l"(*(unsigned long long*)&_pair_scale_src2_7[3])
                     : "l"(*(unsigned long long*)&_pair_scale_odd2_7));
#else
        (output + 56)[0] *= row_scale[0];
        (output + 56)[1] *= row_scale[0];
        (output + 56)[2] *= row_scale[1];
        (output + 56)[3] *= row_scale[1];
        (output + 56)[4] *= row_scale[0];
        (output + 56)[5] *= row_scale[0];
        (output + 56)[6] *= row_scale[1];
        (output + 56)[7] *= row_scale[1];
#endif
      }
    }
    pack_values[0] = scores[0];
    pack_values[1] = scores[1];
    pack_values[2] = scores[2];
    pack_values[3] = scores[3];
    {
      uint32_t _packed;
      asm volatile(
          "{\n\t"
          ".reg .b16 _lo;\n\t"
          ".reg .b16 _hi;\n\t"
          "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
          "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
          "mov.b32 %0, {_lo, _hi};\n\t"
          "}"
          : "=r"(_packed)
          : "f"(pack_values[0]), "f"(pack_values[1]), "f"(pack_values[2]), "f"(pack_values[3]));
      probabilities[0] = _packed;
    }
    pack_values[0] = scores[4];
    pack_values[1] = scores[5];
    pack_values[2] = scores[6];
    pack_values[3] = scores[7];
    {
      uint32_t _packed;
      asm volatile(
          "{\n\t"
          ".reg .b16 _lo;\n\t"
          ".reg .b16 _hi;\n\t"
          "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
          "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
          "mov.b32 %0, {_lo, _hi};\n\t"
          "}"
          : "=r"(_packed)
          : "f"(pack_values[0]), "f"(pack_values[1]), "f"(pack_values[2]), "f"(pack_values[3]));
      probabilities[1] = _packed;
    }
    unsigned int natural_lo = probabilities[0];
    unsigned int natural_hi = probabilities[1];
    uint32_t _prmt_b32_0;
    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_0) : "r"(natural_lo), "r"(natural_hi));
    unsigned int permuted_lo = _prmt_b32_0;
    uint32_t _prmt_b32_1;
    asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_b32_1) : "r"(natural_lo), "r"(natural_hi));
    unsigned int permuted_hi = _prmt_b32_1;
    probabilities[0] = permuted_lo;
    probabilities[1] = permuted_hi;
    pack_values[0] = scores[8];
    pack_values[1] = scores[9];
    pack_values[2] = scores[10];
    pack_values[3] = scores[11];
    {
      uint32_t _packed;
      asm volatile(
          "{\n\t"
          ".reg .b16 _lo;\n\t"
          ".reg .b16 _hi;\n\t"
          "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
          "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
          "mov.b32 %0, {_lo, _hi};\n\t"
          "}"
          : "=r"(_packed)
          : "f"(pack_values[0]), "f"(pack_values[1]), "f"(pack_values[2]), "f"(pack_values[3]));
      probabilities[2] = _packed;
    }
    pack_values[0] = scores[12];
    pack_values[1] = scores[13];
    pack_values[2] = scores[14];
    pack_values[3] = scores[15];
    {
      uint32_t _packed;
      asm volatile(
          "{\n\t"
          ".reg .b16 _lo;\n\t"
          ".reg .b16 _hi;\n\t"
          "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
          "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
          "mov.b32 %0, {_lo, _hi};\n\t"
          "}"
          : "=r"(_packed)
          : "f"(pack_values[0]), "f"(pack_values[1]), "f"(pack_values[2]), "f"(pack_values[3]));
      probabilities[3] = _packed;
    }
    unsigned int natural_lo_47 = probabilities[2];
    unsigned int natural_hi_48 = probabilities[3];
    uint32_t _prmt_b32_2;
    asm("prmt.b32 %0, %1, %2, 0x5410;"
        : "=r"(_prmt_b32_2)
        : "r"(natural_lo_47), "r"(natural_hi_48));
    unsigned int permuted_lo_49 = _prmt_b32_2;
    uint32_t _prmt_b32_3;
    asm("prmt.b32 %0, %1, %2, 0x7632;"
        : "=r"(_prmt_b32_3)
        : "r"(natural_lo_47), "r"(natural_hi_48));
    unsigned int permuted_hi_50 = _prmt_b32_3;
    probabilities[2] = permuted_lo_49;
    probabilities[3] = permuted_hi_50;
    pack_values[0] = scores[16];
    pack_values[1] = scores[17];
    pack_values[2] = scores[18];
    pack_values[3] = scores[19];
    {
      uint32_t _packed;
      asm volatile(
          "{\n\t"
          ".reg .b16 _lo;\n\t"
          ".reg .b16 _hi;\n\t"
          "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
          "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
          "mov.b32 %0, {_lo, _hi};\n\t"
          "}"
          : "=r"(_packed)
          : "f"(pack_values[0]), "f"(pack_values[1]), "f"(pack_values[2]), "f"(pack_values[3]));
      probabilities[4] = _packed;
    }
    pack_values[0] = scores[20];
    pack_values[1] = scores[21];
    pack_values[2] = scores[22];
    pack_values[3] = scores[23];
    {
      uint32_t _packed;
      asm volatile(
          "{\n\t"
          ".reg .b16 _lo;\n\t"
          ".reg .b16 _hi;\n\t"
          "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
          "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
          "mov.b32 %0, {_lo, _hi};\n\t"
          "}"
          : "=r"(_packed)
          : "f"(pack_values[0]), "f"(pack_values[1]), "f"(pack_values[2]), "f"(pack_values[3]));
      probabilities[5] = _packed;
    }
    unsigned int natural_lo_51 = probabilities[4];
    unsigned int natural_hi_52 = probabilities[5];
    uint32_t _prmt_b32_4;
    asm("prmt.b32 %0, %1, %2, 0x5410;"
        : "=r"(_prmt_b32_4)
        : "r"(natural_lo_51), "r"(natural_hi_52));
    unsigned int permuted_lo_53 = _prmt_b32_4;
    uint32_t _prmt_b32_5;
    asm("prmt.b32 %0, %1, %2, 0x7632;"
        : "=r"(_prmt_b32_5)
        : "r"(natural_lo_51), "r"(natural_hi_52));
    unsigned int permuted_hi_54 = _prmt_b32_5;
    probabilities[4] = permuted_lo_53;
    probabilities[5] = permuted_hi_54;
    pack_values[0] = scores[24];
    pack_values[1] = scores[25];
    pack_values[2] = scores[26];
    pack_values[3] = scores[27];
    {
      uint32_t _packed;
      asm volatile(
          "{\n\t"
          ".reg .b16 _lo;\n\t"
          ".reg .b16 _hi;\n\t"
          "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
          "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
          "mov.b32 %0, {_lo, _hi};\n\t"
          "}"
          : "=r"(_packed)
          : "f"(pack_values[0]), "f"(pack_values[1]), "f"(pack_values[2]), "f"(pack_values[3]));
      probabilities[6] = _packed;
    }
    pack_values[0] = scores[28];
    pack_values[1] = scores[29];
    pack_values[2] = scores[30];
    pack_values[3] = scores[31];
    {
      uint32_t _packed;
      asm volatile(
          "{\n\t"
          ".reg .b16 _lo;\n\t"
          ".reg .b16 _hi;\n\t"
          "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
          "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
          "mov.b32 %0, {_lo, _hi};\n\t"
          "}"
          : "=r"(_packed)
          : "f"(pack_values[0]), "f"(pack_values[1]), "f"(pack_values[2]), "f"(pack_values[3]));
      probabilities[7] = _packed;
    }
    unsigned int natural_lo_55 = probabilities[6];
    unsigned int natural_hi_56 = probabilities[7];
    uint32_t _prmt_b32_6;
    asm("prmt.b32 %0, %1, %2, 0x5410;"
        : "=r"(_prmt_b32_6)
        : "r"(natural_lo_55), "r"(natural_hi_56));
    unsigned int permuted_lo_57 = _prmt_b32_6;
    uint32_t _prmt_b32_7;
    asm("prmt.b32 %0, %1, %2, 0x7632;"
        : "=r"(_prmt_b32_7)
        : "r"(natural_lo_55), "r"(natural_hi_56));
    unsigned int permuted_hi_58 = _prmt_b32_7;
    probabilities[6] = permuted_lo_57;
    probabilities[7] = permuted_hi_58;
    mbarrier_wait(v_full_addr, _phase_v_full_0);
    _phase_v_full_0 ^= 1;
    unsigned int v_row_base = lane % 8 + lane / 16 * 8;
    unsigned int v_col_base = lane / 8 % 2;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
        : "r"(v_smem_addr + (v_row_base * (unsigned int)(((0) ? 8 : 4)) +
                             (unsigned int)((v_col_base * 16 ^ (v_row_base >> 1 & 3) << 4) / 16)) *
                                16)
        : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[0]), "+f"(output[1]), "+f"(output[2]), "+f"(output[3])
        : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]),
          "r"(probabilities[3]), "r"(v_frag[0]), "r"(v_frag[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[4]), "+f"(output[(4) + 1]), "+f"(output[(4) + 2]), "+f"(output[(4) + 3])
        : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]),
          "r"(probabilities[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                 : "r"(v_smem_addr +
                       ((v_row_base + 16) * (unsigned int)(((0) ? 8 : 4)) +
                        (unsigned int)((v_col_base * 16 ^ (v_row_base + 16 >> 1 & 3) << 4) / 16)) *
                           16)
                 : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[8]), "+f"(output[(8) + 1]), "+f"(output[(8) + 2]), "+f"(output[(8) + 3])
        : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]),
          "r"(probabilities[3]), "r"(v_frag[0]), "r"(v_frag[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[12]), "+f"(output[(12) + 1]), "+f"(output[(12) + 2]), "+f"(output[(12) + 3])
        : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]),
          "r"(probabilities[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                 : "r"(v_smem_addr +
                       ((v_row_base + 32) * (unsigned int)(((0) ? 8 : 4)) +
                        (unsigned int)((v_col_base * 16 ^ (v_row_base + 32 >> 1 & 3) << 4) / 16)) *
                           16)
                 : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[16]), "+f"(output[(16) + 1]), "+f"(output[(16) + 2]), "+f"(output[(16) + 3])
        : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]),
          "r"(probabilities[3]), "r"(v_frag[0]), "r"(v_frag[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[20]), "+f"(output[(20) + 1]), "+f"(output[(20) + 2]), "+f"(output[(20) + 3])
        : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]),
          "r"(probabilities[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                 : "r"(v_smem_addr +
                       ((v_row_base + 48) * (unsigned int)(((0) ? 8 : 4)) +
                        (unsigned int)((v_col_base * 16 ^ (v_row_base + 48 >> 1 & 3) << 4) / 16)) *
                           16)
                 : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[24]), "+f"(output[(24) + 1]), "+f"(output[(24) + 2]), "+f"(output[(24) + 3])
        : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]),
          "r"(probabilities[3]), "r"(v_frag[0]), "r"(v_frag[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[28]), "+f"(output[(28) + 1]), "+f"(output[(28) + 2]), "+f"(output[(28) + 3])
        : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]),
          "r"(probabilities[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                 : "r"(v_smem_addr +
                       ((v_row_base + 64) * (unsigned int)(((0) ? 8 : 4)) +
                        (unsigned int)((v_col_base * 16 ^ (v_row_base + 64 >> 1 & 3) << 4) / 16)) *
                           16)
                 : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[32]), "+f"(output[(32) + 1]), "+f"(output[(32) + 2]), "+f"(output[(32) + 3])
        : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]),
          "r"(probabilities[3]), "r"(v_frag[0]), "r"(v_frag[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[36]), "+f"(output[(36) + 1]), "+f"(output[(36) + 2]), "+f"(output[(36) + 3])
        : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]),
          "r"(probabilities[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                 : "r"(v_smem_addr +
                       ((v_row_base + 80) * (unsigned int)(((0) ? 8 : 4)) +
                        (unsigned int)((v_col_base * 16 ^ (v_row_base + 80 >> 1 & 3) << 4) / 16)) *
                           16)
                 : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[40]), "+f"(output[(40) + 1]), "+f"(output[(40) + 2]), "+f"(output[(40) + 3])
        : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]),
          "r"(probabilities[3]), "r"(v_frag[0]), "r"(v_frag[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[44]), "+f"(output[(44) + 1]), "+f"(output[(44) + 2]), "+f"(output[(44) + 3])
        : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]),
          "r"(probabilities[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                 : "r"(v_smem_addr +
                       ((v_row_base + 96) * (unsigned int)(((0) ? 8 : 4)) +
                        (unsigned int)((v_col_base * 16 ^ (v_row_base + 96 >> 1 & 3) << 4) / 16)) *
                           16)
                 : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[48]), "+f"(output[(48) + 1]), "+f"(output[(48) + 2]), "+f"(output[(48) + 3])
        : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]),
          "r"(probabilities[3]), "r"(v_frag[0]), "r"(v_frag[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[52]), "+f"(output[(52) + 1]), "+f"(output[(52) + 2]), "+f"(output[(52) + 3])
        : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]),
          "r"(probabilities[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                 : "r"(v_smem_addr +
                       ((v_row_base + 112) * (unsigned int)(((0) ? 8 : 4)) +
                        (unsigned int)((v_col_base * 16 ^ (v_row_base + 112 >> 1 & 3) << 4) / 16)) *
                           16)
                 : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[56]), "+f"(output[(56) + 1]), "+f"(output[(56) + 2]), "+f"(output[(56) + 3])
        : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]),
          "r"(probabilities[3]), "r"(v_frag[0]), "r"(v_frag[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[60]), "+f"(output[(60) + 1]), "+f"(output[(60) + 2]), "+f"(output[(60) + 3])
        : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]),
          "r"(probabilities[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                 : "r"(v_smem_addr +
                       (v_row_base * (unsigned int)(((0) ? 8 : 4)) +
                        (unsigned int)(((v_col_base + 2) * 16 ^ (v_row_base >> 1 & 3) << 4) / 16)) *
                           16)
                 : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[0]), "+f"(output[1]), "+f"(output[2]), "+f"(output[3])
        : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]),
          "r"(probabilities[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[4]), "+f"(output[(4) + 1]), "+f"(output[(4) + 2]), "+f"(output[(4) + 3])
        : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]),
          "r"(probabilities[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
        : "r"(v_smem_addr +
              ((v_row_base + 16) * (unsigned int)(((0) ? 8 : 4)) +
               (unsigned int)(((v_col_base + 2) * 16 ^ (v_row_base + 16 >> 1 & 3) << 4) / 16)) *
                  16)
        : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[8]), "+f"(output[(8) + 1]), "+f"(output[(8) + 2]), "+f"(output[(8) + 3])
        : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]),
          "r"(probabilities[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[12]), "+f"(output[(12) + 1]), "+f"(output[(12) + 2]), "+f"(output[(12) + 3])
        : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]),
          "r"(probabilities[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
        : "r"(v_smem_addr +
              ((v_row_base + 32) * (unsigned int)(((0) ? 8 : 4)) +
               (unsigned int)(((v_col_base + 2) * 16 ^ (v_row_base + 32 >> 1 & 3) << 4) / 16)) *
                  16)
        : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[16]), "+f"(output[(16) + 1]), "+f"(output[(16) + 2]), "+f"(output[(16) + 3])
        : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]),
          "r"(probabilities[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[20]), "+f"(output[(20) + 1]), "+f"(output[(20) + 2]), "+f"(output[(20) + 3])
        : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]),
          "r"(probabilities[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
        : "r"(v_smem_addr +
              ((v_row_base + 48) * (unsigned int)(((0) ? 8 : 4)) +
               (unsigned int)(((v_col_base + 2) * 16 ^ (v_row_base + 48 >> 1 & 3) << 4) / 16)) *
                  16)
        : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[24]), "+f"(output[(24) + 1]), "+f"(output[(24) + 2]), "+f"(output[(24) + 3])
        : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]),
          "r"(probabilities[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[28]), "+f"(output[(28) + 1]), "+f"(output[(28) + 2]), "+f"(output[(28) + 3])
        : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]),
          "r"(probabilities[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
        : "r"(v_smem_addr +
              ((v_row_base + 64) * (unsigned int)(((0) ? 8 : 4)) +
               (unsigned int)(((v_col_base + 2) * 16 ^ (v_row_base + 64 >> 1 & 3) << 4) / 16)) *
                  16)
        : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[32]), "+f"(output[(32) + 1]), "+f"(output[(32) + 2]), "+f"(output[(32) + 3])
        : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]),
          "r"(probabilities[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[36]), "+f"(output[(36) + 1]), "+f"(output[(36) + 2]), "+f"(output[(36) + 3])
        : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]),
          "r"(probabilities[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
        : "r"(v_smem_addr +
              ((v_row_base + 80) * (unsigned int)(((0) ? 8 : 4)) +
               (unsigned int)(((v_col_base + 2) * 16 ^ (v_row_base + 80 >> 1 & 3) << 4) / 16)) *
                  16)
        : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[40]), "+f"(output[(40) + 1]), "+f"(output[(40) + 2]), "+f"(output[(40) + 3])
        : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]),
          "r"(probabilities[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[44]), "+f"(output[(44) + 1]), "+f"(output[(44) + 2]), "+f"(output[(44) + 3])
        : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]),
          "r"(probabilities[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
        : "r"(v_smem_addr +
              ((v_row_base + 96) * (unsigned int)(((0) ? 8 : 4)) +
               (unsigned int)(((v_col_base + 2) * 16 ^ (v_row_base + 96 >> 1 & 3) << 4) / 16)) *
                  16)
        : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[48]), "+f"(output[(48) + 1]), "+f"(output[(48) + 2]), "+f"(output[(48) + 3])
        : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]),
          "r"(probabilities[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[52]), "+f"(output[(52) + 1]), "+f"(output[(52) + 2]), "+f"(output[(52) + 3])
        : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]),
          "r"(probabilities[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
        : "r"(v_smem_addr +
              ((v_row_base + 112) * (unsigned int)(((0) ? 8 : 4)) +
               (unsigned int)(((v_col_base + 2) * 16 ^ (v_row_base + 112 >> 1 & 3) << 4) / 16)) *
                  16)
        : "memory");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[56]), "+f"(output[(56) + 1]), "+f"(output[(56) + 2]), "+f"(output[(56) + 3])
        : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]),
          "r"(probabilities[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(output[60]), "+f"(output[(60) + 1]), "+f"(output[(60) + 2]), "+f"(output[(60) + 3])
        : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]),
          "r"(probabilities[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
    if (preload_count < selected_count) {
      __syncthreads();
      if (warp == 0) {
        if (elect_sync()) {
          mbarrier_arrive_expect_tx(v_full_addr, 8192);
          asm volatile(
              "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_"
              "hint"
              " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;" ::"r"(v_tma_smem_addr),
              "l"(V_map), "r"(next_physical * 64), "r"(0), "r"(head), "r"(batch), "r"(v_full_addr),
              "l"(0x14F0000000000000ULL)
              : "memory");
        }
      }
    }
    current_physical = next_physical;
  }
  __syncthreads();
  float reciprocal[2];
  float total = row_sum[0];
  float _rcp_0 = approx_rcp(((total != 0.0f && total == total) ? total : 1.0f));
  reciprocal[0] = _rcp_0;
  float total_0 = row_sum[1];
  float _rcp_1 = approx_rcp(((total_0 != 0.0f && total_0 == total_0) ? total_0 : 1.0f));
  reciprocal[1] = _rcp_1;
  unsigned int scale_base = v_scale_base + lane % 4 * 2;
  float _vec_load_0[2];
  {
    float2 _v2_8 = *reinterpret_cast<const float2*>(v_scale + scale_base);
    _vec_load_0[0] = _v2_8.x;
    _vec_load_0[0 + 1] = _v2_8.y;
  }
  float _vec_load_1[2];
  {
    float2 _v2_9 = *reinterpret_cast<const float2*>(v_scale + scale_base + 8);
    _vec_load_1[0] = _v2_9.x;
    _vec_load_1[0 + 1] = _v2_9.y;
  }
  float scale0 = _vec_load_0[0];
  float scale1 = _vec_load_0[1];
  float scale2 = _vec_load_1[0];
  float scale3 = _vec_load_1[1];
  output[0] = output[0] * reciprocal[0] * scale0;
  output[1] = output[1] * reciprocal[0] * scale1;
  output[2] = output[2] * reciprocal[1] * scale0;
  output[3] = output[3] * reciprocal[1] * scale1;
  output[4] = output[4] * reciprocal[0] * scale2;
  output[5] = output[5] * reciprocal[0] * scale3;
  output[6] = output[6] * reciprocal[1] * scale2;
  output[7] = output[7] * reciprocal[1] * scale3;
  unsigned int packed[4];
#pragma unroll
  for (int _lp = 0; _lp < 4; _lp++) {
    __nv_bfloat162 _bf2 =
        __float22bfloat162_rn(make_float2(output[_lp * 2 + 0], output[_lp * 2 + 1 + 0]));
    packed[_lp] = *(uint32_t*)&_bf2;
  }
  uint32_t _stmatrix_addr_10 = static_cast<uint32_t>(
      o_smem_addr + (unsigned int)((lane / 16 / 8 * 512 + (warp * 16 + lane % 16) * 8 +
                                    (lane / 16 % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) *
                                   16));
  asm volatile(
      "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(_stmatrix_addr_10),
      "r"(*reinterpret_cast<const uint32_t*>(&packed[0])),
      "r"(*reinterpret_cast<const uint32_t*>(&packed[1])),
      "r"(*reinterpret_cast<const uint32_t*>(&packed[2])),
      "r"(*reinterpret_cast<const uint32_t*>(&packed[3]))
      : "memory");
  unsigned int scale_base_1 = v_scale_base + 16 + lane % 4 * 2;
  float _vec_load_2[2];
  {
    float2 _v2_11 = *reinterpret_cast<const float2*>(v_scale + scale_base_1);
    _vec_load_2[0] = _v2_11.x;
    _vec_load_2[0 + 1] = _v2_11.y;
  }
  float _vec_load_3[2];
  {
    float2 _v2_12 = *reinterpret_cast<const float2*>(v_scale + scale_base_1 + 8);
    _vec_load_3[0] = _v2_12.x;
    _vec_load_3[0 + 1] = _v2_12.y;
  }
  float scale0_2 = _vec_load_2[0];
  float scale1_3 = _vec_load_2[1];
  float scale2_4 = _vec_load_3[0];
  float scale3_5 = _vec_load_3[1];
  output[8] = output[8] * reciprocal[0] * scale0_2;
  output[9] = output[9] * reciprocal[0] * scale1_3;
  output[10] = output[10] * reciprocal[1] * scale0_2;
  output[11] = output[11] * reciprocal[1] * scale1_3;
  output[12] = output[12] * reciprocal[0] * scale2_4;
  output[13] = output[13] * reciprocal[0] * scale3_5;
  output[14] = output[14] * reciprocal[1] * scale2_4;
  output[15] = output[15] * reciprocal[1] * scale3_5;
  unsigned int packed_6[4];
#pragma unroll
  for (int _lp = 0; _lp < 4; _lp++) {
    __nv_bfloat162 _bf2 =
        __float22bfloat162_rn(make_float2(output[_lp * 2 + 8], output[_lp * 2 + 1 + 8]));
    packed_6[_lp] = *(uint32_t*)&_bf2;
  }
  uint32_t _stmatrix_addr_13 = static_cast<uint32_t>(
      o_smem_addr +
      (unsigned int)(((2 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 +
                      ((2 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) *
                     16));
  asm volatile(
      "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(_stmatrix_addr_13),
      "r"(*reinterpret_cast<const uint32_t*>(&packed_6[0])),
      "r"(*reinterpret_cast<const uint32_t*>(&packed_6[1])),
      "r"(*reinterpret_cast<const uint32_t*>(&packed_6[2])),
      "r"(*reinterpret_cast<const uint32_t*>(&packed_6[3]))
      : "memory");
  unsigned int scale_base_7 = v_scale_base + 32 + lane % 4 * 2;
  float _vec_load_4[2];
  {
    float2 _v2_14 = *reinterpret_cast<const float2*>(v_scale + scale_base_7);
    _vec_load_4[0] = _v2_14.x;
    _vec_load_4[0 + 1] = _v2_14.y;
  }
  float _vec_load_5[2];
  {
    float2 _v2_15 = *reinterpret_cast<const float2*>(v_scale + scale_base_7 + 8);
    _vec_load_5[0] = _v2_15.x;
    _vec_load_5[0 + 1] = _v2_15.y;
  }
  float scale0_8 = _vec_load_4[0];
  float scale1_9 = _vec_load_4[1];
  float scale2_10 = _vec_load_5[0];
  float scale3_11 = _vec_load_5[1];
  output[16] = output[16] * reciprocal[0] * scale0_8;
  output[17] = output[17] * reciprocal[0] * scale1_9;
  output[18] = output[18] * reciprocal[1] * scale0_8;
  output[19] = output[19] * reciprocal[1] * scale1_9;
  output[20] = output[20] * reciprocal[0] * scale2_10;
  output[21] = output[21] * reciprocal[0] * scale3_11;
  output[22] = output[22] * reciprocal[1] * scale2_10;
  output[23] = output[23] * reciprocal[1] * scale3_11;
  unsigned int packed_12[4];
#pragma unroll
  for (int _lp = 0; _lp < 4; _lp++) {
    __nv_bfloat162 _bf2 =
        __float22bfloat162_rn(make_float2(output[_lp * 2 + 16], output[_lp * 2 + 1 + 16]));
    packed_12[_lp] = *(uint32_t*)&_bf2;
  }
  uint32_t _stmatrix_addr_16 = static_cast<uint32_t>(
      o_smem_addr +
      (unsigned int)(((4 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 +
                      ((4 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) *
                     16));
  asm volatile(
      "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(_stmatrix_addr_16),
      "r"(*reinterpret_cast<const uint32_t*>(&packed_12[0])),
      "r"(*reinterpret_cast<const uint32_t*>(&packed_12[1])),
      "r"(*reinterpret_cast<const uint32_t*>(&packed_12[2])),
      "r"(*reinterpret_cast<const uint32_t*>(&packed_12[3]))
      : "memory");
  unsigned int scale_base_13 = v_scale_base + 48 + lane % 4 * 2;
  float _vec_load_6[2];
  {
    float2 _v2_17 = *reinterpret_cast<const float2*>(v_scale + scale_base_13);
    _vec_load_6[0] = _v2_17.x;
    _vec_load_6[0 + 1] = _v2_17.y;
  }
  float _vec_load_7[2];
  {
    float2 _v2_18 = *reinterpret_cast<const float2*>(v_scale + scale_base_13 + 8);
    _vec_load_7[0] = _v2_18.x;
    _vec_load_7[0 + 1] = _v2_18.y;
  }
  float scale0_14 = _vec_load_6[0];
  float scale1_15 = _vec_load_6[1];
  float scale2_16 = _vec_load_7[0];
  float scale3_17 = _vec_load_7[1];
  output[24] = output[24] * reciprocal[0] * scale0_14;
  output[25] = output[25] * reciprocal[0] * scale1_15;
  output[26] = output[26] * reciprocal[1] * scale0_14;
  output[27] = output[27] * reciprocal[1] * scale1_15;
  output[28] = output[28] * reciprocal[0] * scale2_16;
  output[29] = output[29] * reciprocal[0] * scale3_17;
  output[30] = output[30] * reciprocal[1] * scale2_16;
  output[31] = output[31] * reciprocal[1] * scale3_17;
  unsigned int packed_18[4];
#pragma unroll
  for (int _lp = 0; _lp < 4; _lp++) {
    __nv_bfloat162 _bf2 =
        __float22bfloat162_rn(make_float2(output[_lp * 2 + 24], output[_lp * 2 + 1 + 24]));
    packed_18[_lp] = *(uint32_t*)&_bf2;
  }
  uint32_t _stmatrix_addr_19 = static_cast<uint32_t>(
      o_smem_addr +
      (unsigned int)(((6 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 +
                      ((6 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) *
                     16));
  asm volatile(
      "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(_stmatrix_addr_19),
      "r"(*reinterpret_cast<const uint32_t*>(&packed_18[0])),
      "r"(*reinterpret_cast<const uint32_t*>(&packed_18[1])),
      "r"(*reinterpret_cast<const uint32_t*>(&packed_18[2])),
      "r"(*reinterpret_cast<const uint32_t*>(&packed_18[3]))
      : "memory");
  unsigned int scale_base_19 = v_scale_base + 64 + lane % 4 * 2;
  float _vec_load_8[2];
  {
    float2 _v2_20 = *reinterpret_cast<const float2*>(v_scale + scale_base_19);
    _vec_load_8[0] = _v2_20.x;
    _vec_load_8[0 + 1] = _v2_20.y;
  }
  float _vec_load_9[2];
  {
    float2 _v2_21 = *reinterpret_cast<const float2*>(v_scale + scale_base_19 + 8);
    _vec_load_9[0] = _v2_21.x;
    _vec_load_9[0 + 1] = _v2_21.y;
  }
  float scale0_20 = _vec_load_8[0];
  float scale1_21 = _vec_load_8[1];
  float scale2_22 = _vec_load_9[0];
  float scale3_23 = _vec_load_9[1];
  output[32] = output[32] * reciprocal[0] * scale0_20;
  output[33] = output[33] * reciprocal[0] * scale1_21;
  output[34] = output[34] * reciprocal[1] * scale0_20;
  output[35] = output[35] * reciprocal[1] * scale1_21;
  output[36] = output[36] * reciprocal[0] * scale2_22;
  output[37] = output[37] * reciprocal[0] * scale3_23;
  output[38] = output[38] * reciprocal[1] * scale2_22;
  output[39] = output[39] * reciprocal[1] * scale3_23;
  unsigned int packed_24[4];
#pragma unroll
  for (int _lp = 0; _lp < 4; _lp++) {
    __nv_bfloat162 _bf2 =
        __float22bfloat162_rn(make_float2(output[_lp * 2 + 32], output[_lp * 2 + 1 + 32]));
    packed_24[_lp] = *(uint32_t*)&_bf2;
  }
  uint32_t _stmatrix_addr_22 = static_cast<uint32_t>(
      o_smem_addr +
      (unsigned int)(((8 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 +
                      ((8 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) *
                     16));
  asm volatile(
      "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(_stmatrix_addr_22),
      "r"(*reinterpret_cast<const uint32_t*>(&packed_24[0])),
      "r"(*reinterpret_cast<const uint32_t*>(&packed_24[1])),
      "r"(*reinterpret_cast<const uint32_t*>(&packed_24[2])),
      "r"(*reinterpret_cast<const uint32_t*>(&packed_24[3]))
      : "memory");
  unsigned int scale_base_25 = v_scale_base + 80 + lane % 4 * 2;
  float _vec_load_10[2];
  {
    float2 _v2_23 = *reinterpret_cast<const float2*>(v_scale + scale_base_25);
    _vec_load_10[0] = _v2_23.x;
    _vec_load_10[0 + 1] = _v2_23.y;
  }
  float _vec_load_11[2];
  {
    float2 _v2_24 = *reinterpret_cast<const float2*>(v_scale + scale_base_25 + 8);
    _vec_load_11[0] = _v2_24.x;
    _vec_load_11[0 + 1] = _v2_24.y;
  }
  float scale0_26 = _vec_load_10[0];
  float scale1_27 = _vec_load_10[1];
  float scale2_28 = _vec_load_11[0];
  float scale3_29 = _vec_load_11[1];
  output[40] = output[40] * reciprocal[0] * scale0_26;
  output[41] = output[41] * reciprocal[0] * scale1_27;
  output[42] = output[42] * reciprocal[1] * scale0_26;
  output[43] = output[43] * reciprocal[1] * scale1_27;
  output[44] = output[44] * reciprocal[0] * scale2_28;
  output[45] = output[45] * reciprocal[0] * scale3_29;
  output[46] = output[46] * reciprocal[1] * scale2_28;
  output[47] = output[47] * reciprocal[1] * scale3_29;
  unsigned int packed_30[4];
#pragma unroll
  for (int _lp = 0; _lp < 4; _lp++) {
    __nv_bfloat162 _bf2 =
        __float22bfloat162_rn(make_float2(output[_lp * 2 + 40], output[_lp * 2 + 1 + 40]));
    packed_30[_lp] = *(uint32_t*)&_bf2;
  }
  uint32_t _stmatrix_addr_25 = static_cast<uint32_t>(
      o_smem_addr +
      (unsigned int)(((10 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 +
                      ((10 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) *
                     16));
  asm volatile(
      "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(_stmatrix_addr_25),
      "r"(*reinterpret_cast<const uint32_t*>(&packed_30[0])),
      "r"(*reinterpret_cast<const uint32_t*>(&packed_30[1])),
      "r"(*reinterpret_cast<const uint32_t*>(&packed_30[2])),
      "r"(*reinterpret_cast<const uint32_t*>(&packed_30[3]))
      : "memory");
  unsigned int scale_base_31 = v_scale_base + 96 + lane % 4 * 2;
  float _vec_load_12[2];
  {
    float2 _v2_26 = *reinterpret_cast<const float2*>(v_scale + scale_base_31);
    _vec_load_12[0] = _v2_26.x;
    _vec_load_12[0 + 1] = _v2_26.y;
  }
  float _vec_load_13[2];
  {
    float2 _v2_27 = *reinterpret_cast<const float2*>(v_scale + scale_base_31 + 8);
    _vec_load_13[0] = _v2_27.x;
    _vec_load_13[0 + 1] = _v2_27.y;
  }
  float scale0_32 = _vec_load_12[0];
  float scale1_33 = _vec_load_12[1];
  float scale2_34 = _vec_load_13[0];
  float scale3_35 = _vec_load_13[1];
  output[48] = output[48] * reciprocal[0] * scale0_32;
  output[49] = output[49] * reciprocal[0] * scale1_33;
  output[50] = output[50] * reciprocal[1] * scale0_32;
  output[51] = output[51] * reciprocal[1] * scale1_33;
  output[52] = output[52] * reciprocal[0] * scale2_34;
  output[53] = output[53] * reciprocal[0] * scale3_35;
  output[54] = output[54] * reciprocal[1] * scale2_34;
  output[55] = output[55] * reciprocal[1] * scale3_35;
  unsigned int packed_36[4];
#pragma unroll
  for (int _lp = 0; _lp < 4; _lp++) {
    __nv_bfloat162 _bf2 =
        __float22bfloat162_rn(make_float2(output[_lp * 2 + 48], output[_lp * 2 + 1 + 48]));
    packed_36[_lp] = *(uint32_t*)&_bf2;
  }
  uint32_t _stmatrix_addr_28 = static_cast<uint32_t>(
      o_smem_addr +
      (unsigned int)(((12 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 +
                      ((12 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) *
                     16));
  asm volatile(
      "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(_stmatrix_addr_28),
      "r"(*reinterpret_cast<const uint32_t*>(&packed_36[0])),
      "r"(*reinterpret_cast<const uint32_t*>(&packed_36[1])),
      "r"(*reinterpret_cast<const uint32_t*>(&packed_36[2])),
      "r"(*reinterpret_cast<const uint32_t*>(&packed_36[3]))
      : "memory");
  unsigned int scale_base_37 = v_scale_base + 112 + lane % 4 * 2;
  float _vec_load_14[2];
  {
    float2 _v2_29 = *reinterpret_cast<const float2*>(v_scale + scale_base_37);
    _vec_load_14[0] = _v2_29.x;
    _vec_load_14[0 + 1] = _v2_29.y;
  }
  float _vec_load_15[2];
  {
    float2 _v2_30 = *reinterpret_cast<const float2*>(v_scale + scale_base_37 + 8);
    _vec_load_15[0] = _v2_30.x;
    _vec_load_15[0 + 1] = _v2_30.y;
  }
  float scale0_38 = _vec_load_14[0];
  float scale1_39 = _vec_load_14[1];
  float scale2_40 = _vec_load_15[0];
  float scale3_41 = _vec_load_15[1];
  output[56] = output[56] * reciprocal[0] * scale0_38;
  output[57] = output[57] * reciprocal[0] * scale1_39;
  output[58] = output[58] * reciprocal[1] * scale0_38;
  output[59] = output[59] * reciprocal[1] * scale1_39;
  output[60] = output[60] * reciprocal[0] * scale2_40;
  output[61] = output[61] * reciprocal[0] * scale3_41;
  output[62] = output[62] * reciprocal[1] * scale2_40;
  output[63] = output[63] * reciprocal[1] * scale3_41;
  unsigned int packed_42[4];
#pragma unroll
  for (int _lp = 0; _lp < 4; _lp++) {
    __nv_bfloat162 _bf2 =
        __float22bfloat162_rn(make_float2(output[_lp * 2 + 56], output[_lp * 2 + 1 + 56]));
    packed_42[_lp] = *(uint32_t*)&_bf2;
  }
  uint32_t _stmatrix_addr_31 = static_cast<uint32_t>(
      o_smem_addr +
      (unsigned int)(((14 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 +
                      ((14 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) *
                     16));
  asm volatile(
      "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(_stmatrix_addr_31),
      "r"(*reinterpret_cast<const uint32_t*>(&packed_42[0])),
      "r"(*reinterpret_cast<const uint32_t*>(&packed_42[1])),
      "r"(*reinterpret_cast<const uint32_t*>(&packed_42[2])),
      "r"(*reinterpret_cast<const uint32_t*>(&packed_42[3]))
      : "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (warp == 0) {
    if (elect_sync()) {
      tma_store_5d(O_map, 0, q_base, head, batch, 0, o_smem_addr);
      asm volatile("cp.async.bulk.commit_group;");
      asm volatile("cp.async.bulk.wait_group.read 0;");
    }
  }

  // Cleanup
  __syncthreads();
}

}  // extern "C"
