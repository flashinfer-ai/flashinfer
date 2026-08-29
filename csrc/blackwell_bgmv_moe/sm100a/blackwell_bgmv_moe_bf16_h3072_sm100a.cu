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
// Generated source for FlashInfer.
// Bundle: Blackwell BGMV MoE shrink and expand benchmark portfolio.
// Target: sm_100a; compile flags: none.
// Generated file; do not edit manually.
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef signed int int32_t;
typedef short int int16_t;
struct __align__(128) BlackwellTensorMap {
  uint64_t opaque[16];
};
template <int N> struct __align__(128) BlackwellTensorMapPack {
  BlackwellTensorMap maps[N];
};

typedef struct __align__(64) {
  uint64_t opaque[16];
} CUtensorMap;

#include <cuda_bf16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
  int result;
  asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
               : "=r"(result)
               : "r"(x));
  return result;
}

#include <math_constants.h>

#define BLACKWELL_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_X_SMEM_OFF 0
#define SMEM_X_SMEM_STAGE_BYTES 24576
#define SMEM_X_SMEM_STRIDE 24576
#define SMEM_W_SMEM_OFF 24576
#define SMEM_W_SMEM_STAGE_BYTES 196608
#define SMEM_W_SMEM_STRIDE 196608
#define SMEM_WARP_PARTIALS_OFF 221184
#define SMEM_WARP_PARTIALS_STAGE_BYTES 512
#define SMEM_WARP_PARTIALS_STRIDE 512
#define SMEM_TOTAL 221696
#define THREADS 128

extern "C" {

__global__
__launch_bounds__(128, 1) void kernel_flashinfer_bgmv_moe_shrink_bf16_h3072_r32_p4_s3(
    uint16_t *__restrict__ shrink_out_raw, uint16_t *__restrict__ x_raw,
    uint16_t *__restrict__ lora_a_raw, long long *__restrict__ sorted_token_ids,
    long long *__restrict__ expert_ids, long long *__restrict__ lora_indices,
    int num_pairs, int num_experts, int num_tokens) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  // Kernel setup ops
  __nv_bfloat16 *x_smem = reinterpret_cast<__nv_bfloat16 *>(smem_raw + 0);
  const int x_smem_addr = smem + 0;
  __nv_bfloat16 *w_smem = reinterpret_cast<__nv_bfloat16 *>(smem_raw + 24576);
  const int w_smem_addr = smem + 24576;
  float *warp_partials = reinterpret_cast<float *>(smem_raw + 221184);
  const int warp_partials_addr = smem + 221184;

  // === Task calls (dependency order) ===
  int pair_block = blockIdx.x;
  int rank_block = blockIdx.y;
  int rank_base = rank_block * 8;
  long long tokens[4];
  long long experts[4];
  long long loras[4];
  int valid[4];
#pragma unroll
  for (int pp = 0; pp < 4; pp++) {
    int pair = pair_block * 4 + pp;
    tokens[pp] = -1;
    experts[pp] = -1;
    loras[pp] = -1;
    valid[pp] = 0;
    if (pair < num_pairs) {
      tokens[pp] = sorted_token_ids[pair];
      experts[pp] = expert_ids[pair];
      if (tokens[pp] >= 0) {
        if (tokens[pp] < (long long)num_tokens) {
          loras[pp] = lora_indices[tokens[pp]];
          if (loras[pp] >= 0) {
            valid[pp] = 1;
          }
        }
      }
    }
  }
#pragma unroll
  for (int tile = 0; tile < ((1) ? 3 : 3); tile++) {
    int k_base = tile * 1024 + tid * 8;
#pragma unroll
    for (int pp_1 = 0; pp_1 < 4; pp_1++) {
      if (valid[pp_1] != 0) {
        if (k_base < 3072) {
          asm volatile(
              "cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                  x_smem_addr +
                  (unsigned int)((tile % 3 * 4 * 1024 + pp_1 * 1024 + tid * 8) *
                                 2)),
              "l"(reinterpret_cast<const __nv_bfloat16 *>(x_raw) +
                  (tokens[pp_1] * 3072 + (long long)k_base)));
#pragma unroll
          for (int rr = 0; rr < 8; rr++) {
            int rank_row = rank_base + rr;
            long long weight_index =
                ((loras[pp_1] * (long long)num_experts + experts[pp_1]) * 32 +
                 (long long)rank_row) *
                    3072 +
                (long long)k_base;
            asm volatile(
                "cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                    w_smem_addr +
                    (unsigned int)((tile % 3 * 4 * 8 * 1024 + pp_1 * 8 * 1024 +
                                    rr * 1024 + tid * 8) *
                                   2)),
                "l"(reinterpret_cast<const __nv_bfloat16 *>(lora_a_raw) +
                    weight_index));
          }
        }
      }
    }
    asm volatile("cp.async.commit_group;");
  }
  float owned_accum = 0.0f;
  unsigned int x_carriers[4];
  unsigned int w_carriers[4];
  float x_values[8];
  float w_values[8];
#pragma unroll
  for (int tile_1 = 0; tile_1 < 3; tile_1++) {
    if (3 - tile_1 - 1 == 0) {
      asm volatile("cp.async.wait_group 0;");
    } else if (3 - tile_1 - 1 == 1) {
      asm volatile("cp.async.wait_group 1;");
    } else {
      asm volatile("cp.async.wait_group 2;");
    }
    __syncthreads();
    int k_base_1 = tile_1 * 1024 + tid * 8;
#pragma unroll
    for (int pp_2 = 0; pp_2 < 4; pp_2++) {
      int x_thread_base = tile_1 % 3 * 4 * 1024 + pp_2 * 1024 + tid * 8;
      if (valid[pp_2] != 0) {
        if (k_base_1 < 3072) {
          asm volatile(
              "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
              : "=r"(*reinterpret_cast<uint32_t *>(&x_carriers[0])),
                "=r"(*reinterpret_cast<uint32_t *>(&x_carriers[(0) + 1])),
                "=r"(*reinterpret_cast<uint32_t *>(&x_carriers[(0) + 2])),
                "=r"(*reinterpret_cast<uint32_t *>(&x_carriers[(0) + 3]))
              : "r"(x_smem_addr + (unsigned int)(x_thread_base * 2)));
          {
#pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
              asm volatile("{\n\t"
                           "shl.b32 %0, %2, 16;\n\t"
                           "and.b32 %1, %2, 0xffff0000;\n\t"
                           "}\n"
                           : "=f"((&x_values[_pair * 2])[0]),
                             "=f"((&x_values[_pair * 2])[1])
                           : "r"(x_carriers[_pair]));
            }
          }
        }
      }
#pragma unroll
      for (int rr_1 = 0; rr_1 < 8; rr_1++) {
        float partial = 0.0f;
        if (valid[pp_2] != 0) {
          if (k_base_1 < 3072) {
            int w_thread_base = tile_1 % 3 * 4 * 8 * 1024 + pp_2 * 8 * 1024 +
                                rr_1 * 1024 + tid * 8;
            asm volatile(
                "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t *>(&w_carriers[0])),
                  "=r"(*reinterpret_cast<uint32_t *>(&w_carriers[(0) + 1])),
                  "=r"(*reinterpret_cast<uint32_t *>(&w_carriers[(0) + 2])),
                  "=r"(*reinterpret_cast<uint32_t *>(&w_carriers[(0) + 3]))
                : "r"(w_smem_addr + (unsigned int)(w_thread_base * 2)));
            {
#pragma unroll
              for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile("{\n\t"
                             "shl.b32 %0, %2, 16;\n\t"
                             "and.b32 %1, %2, 0xffff0000;\n\t"
                             "}\n"
                             : "=f"((&w_values[_pair * 2])[0]),
                               "=f"((&w_values[_pair * 2])[1])
                             : "r"(w_carriers[_pair]));
              }
            }
#pragma unroll
            for (int element = 0; element < 8; element++) {
              float _fma_0 =
                  __fmaf_rn(x_values[element], w_values[element], partial);
              partial = _fma_0;
            }
          }
        }
        float _warp_reduce_0 = partial;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
          _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
        partial = _warp_reduce_0;
        if (lane == 0) {
          warp_partials[(pp_2 * 8 + rr_1) * 4 + warp] = partial;
        }
      }
    }
    __syncthreads();
    if (warp == 0) {
      if (lane < 32) {
        int owner_index = lane;
        float cta_partial = 0.0f;
#pragma unroll
        for (int source_warp = 0; source_warp < 4; source_warp++) {
          cta_partial += warp_partials[owner_index * 4 + source_warp];
        }
        owned_accum += cta_partial;
      }
    }
    __syncthreads();
    if (tile_1 + ((1) ? 3 : 3) < 3) {
      int refill_k = (tile_1 + ((1) ? 3 : 3)) * 1024 + tid * 8;
#pragma unroll
      for (int pp_3 = 0; pp_3 < 4; pp_3++) {
        if (valid[pp_3] != 0) {
          if (refill_k < 3072) {
            asm volatile(
                "cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                    x_smem_addr +
                    (unsigned int)(((tile_1 + ((1) ? 3 : 3)) % 3 * 4 * 1024 +
                                    pp_3 * 1024 + tid * 8) *
                                   2)),
                "l"(reinterpret_cast<const __nv_bfloat16 *>(x_raw) +
                    (tokens[pp_3] * 3072 + (long long)refill_k)));
#pragma unroll
            for (int rr_2 = 0; rr_2 < 8; rr_2++) {
              int rank_row_1 = rank_base + rr_2;
              long long weight_index_1 =
                  ((loras[pp_3] * (long long)num_experts + experts[pp_3]) * 32 +
                   (long long)rank_row_1) *
                      3072 +
                  (long long)refill_k;
              asm volatile(
                  "cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                      w_smem_addr +
                      (unsigned int)(((tile_1 + ((1) ? 3 : 3)) % 3 * 4 * 8 *
                                          1024 +
                                      pp_3 * 8 * 1024 + rr_2 * 1024 + tid * 8) *
                                     2)),
                  "l"(reinterpret_cast<const __nv_bfloat16 *>(lora_a_raw) +
                      weight_index_1));
            }
          }
        }
      }
      asm volatile("cp.async.commit_group;");
    }
  }
  if (warp == 0) {
    if (lane < 32) {
      int owner_pp = lane / 8;
      int owner_rr = lane % 8;
      int pair_1 = pair_block * 4 + owner_pp;
      if (pair_1 < num_pairs) {
        *(reinterpret_cast<__nv_bfloat16 *>(
              reinterpret_cast<__nv_bfloat16 *>(shrink_out_raw) +
              (pair_1 * 32 + rank_base + owner_rr)) +
          (0)) = __float2bfloat16_rn(owned_accum);
      }
    }
  }
}

} // extern "C"

#undef BLACKWELL_INF
#undef NUM_MAIN_STAGES
#undef SMEM_TOTAL
#undef SMEM_WARP_PARTIALS_OFF
#undef SMEM_WARP_PARTIALS_STAGE_BYTES
#undef SMEM_WARP_PARTIALS_STRIDE
#undef SMEM_W_SMEM_OFF
#undef SMEM_W_SMEM_STAGE_BYTES
#undef SMEM_W_SMEM_STRIDE
#undef SMEM_X_SMEM_OFF
#undef SMEM_X_SMEM_STAGE_BYTES
#undef SMEM_X_SMEM_STRIDE
#undef THREADS
#undef w_smem_addr
#undef warp_partials_addr
#undef x_smem_addr

#define BLACKWELL_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_X_SMEM_OFF 0
#define SMEM_X_SMEM_STAGE_BYTES 4096
#define SMEM_X_SMEM_STRIDE 4096
#define SMEM_W_SMEM_OFF 4096
#define SMEM_W_SMEM_STAGE_BYTES 32768
#define SMEM_W_SMEM_STRIDE 32768
#define SMEM_WARP_PARTIALS_OFF 36864
#define SMEM_WARP_PARTIALS_STAGE_BYTES 128
#define SMEM_WARP_PARTIALS_STRIDE 128
#define SMEM_TOTAL 36992
#define THREADS 128

extern "C" {

__global__
__launch_bounds__(128, 1) void kernel_flashinfer_bgmv_moe_shrink_bf16_h3072_r32_p1_s2(
    uint16_t *__restrict__ shrink_out_raw, uint16_t *__restrict__ x_raw,
    uint16_t *__restrict__ lora_a_raw, long long *__restrict__ sorted_token_ids,
    long long *__restrict__ expert_ids, long long *__restrict__ lora_indices,
    int num_pairs, int num_experts, int num_tokens) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  // Kernel setup ops
  __nv_bfloat16 *x_smem = reinterpret_cast<__nv_bfloat16 *>(smem_raw + 0);
  const int x_smem_addr = smem + 0;
  __nv_bfloat16 *w_smem = reinterpret_cast<__nv_bfloat16 *>(smem_raw + 4096);
  const int w_smem_addr = smem + 4096;
  float *warp_partials = reinterpret_cast<float *>(smem_raw + 36864);
  const int warp_partials_addr = smem + 36864;

  // === Task calls (dependency order) ===
  int pair_block = blockIdx.x;
  int rank_block = blockIdx.y;
  int rank_base = rank_block * 8;
  long long tokens[1];
  long long experts[1];
  long long loras[1];
  int valid[1];
#pragma unroll
  for (int pp = 0; pp < 1; pp++) {
    int pair = pair_block + pp;
    tokens[pp] = -1;
    experts[pp] = -1;
    loras[pp] = -1;
    valid[pp] = 0;
    if (pair < num_pairs) {
      tokens[pp] = sorted_token_ids[pair];
      experts[pp] = expert_ids[pair];
      if (tokens[pp] >= 0) {
        if (tokens[pp] < (long long)num_tokens) {
          loras[pp] = lora_indices[tokens[pp]];
          if (loras[pp] >= 0) {
            valid[pp] = 1;
          }
        }
      }
    }
  }
#pragma unroll
  for (int tile = 0; tile < ((1) ? 2 : 3); tile++) {
    int k_base = tile * 1024 + tid * 8;
#pragma unroll
    for (int pp_1 = 0; pp_1 < 1; pp_1++) {
      if (valid[pp_1] != 0) {
        if (k_base < 3072) {
          asm volatile(
              "cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                  x_smem_addr +
                  (unsigned int)((tile % 2 * 1024 + pp_1 * 1024 + tid * 8) *
                                 2)),
              "l"(reinterpret_cast<const __nv_bfloat16 *>(x_raw) +
                  (tokens[pp_1] * 3072 + (long long)k_base)));
#pragma unroll
          for (int rr = 0; rr < 8; rr++) {
            int rank_row = rank_base + rr;
            long long weight_index =
                ((loras[pp_1] * (long long)num_experts + experts[pp_1]) * 32 +
                 (long long)rank_row) *
                    3072 +
                (long long)k_base;
            asm volatile(
                "cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                    w_smem_addr +
                    (unsigned int)((tile % 2 * 8 * 1024 + pp_1 * 8 * 1024 +
                                    rr * 1024 + tid * 8) *
                                   2)),
                "l"(reinterpret_cast<const __nv_bfloat16 *>(lora_a_raw) +
                    weight_index));
          }
        }
      }
    }
    asm volatile("cp.async.commit_group;");
  }
  float owned_accum = 0.0f;
  unsigned int x_carriers[4];
  unsigned int w_carriers[4];
  float x_values[8];
  float w_values[8];
#pragma unroll
  for (int tile_1 = 0; tile_1 < 3; tile_1++) {
    if (3 - tile_1 - 1 == 0) {
      asm volatile("cp.async.wait_group 0;");
    } else if (3 - tile_1 - 1 == 1) {
      asm volatile("cp.async.wait_group 1;");
    } else {
      asm volatile("cp.async.wait_group 1;");
    }
    __syncthreads();
    int k_base_1 = tile_1 * 1024 + tid * 8;
#pragma unroll
    for (int pp_2 = 0; pp_2 < 1; pp_2++) {
      int x_thread_base = tile_1 % 2 * 1024 + pp_2 * 1024 + tid * 8;
      if (valid[pp_2] != 0) {
        if (k_base_1 < 3072) {
          asm volatile(
              "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
              : "=r"(*reinterpret_cast<uint32_t *>(&x_carriers[0])),
                "=r"(*reinterpret_cast<uint32_t *>(&x_carriers[(0) + 1])),
                "=r"(*reinterpret_cast<uint32_t *>(&x_carriers[(0) + 2])),
                "=r"(*reinterpret_cast<uint32_t *>(&x_carriers[(0) + 3]))
              : "r"(x_smem_addr + (unsigned int)(x_thread_base * 2)));
          {
#pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
              asm volatile("{\n\t"
                           "shl.b32 %0, %2, 16;\n\t"
                           "and.b32 %1, %2, 0xffff0000;\n\t"
                           "}\n"
                           : "=f"((&x_values[_pair * 2])[0]),
                             "=f"((&x_values[_pair * 2])[1])
                           : "r"(x_carriers[_pair]));
            }
          }
        }
      }
#pragma unroll
      for (int rr_1 = 0; rr_1 < 8; rr_1++) {
        float partial = 0.0f;
        if (valid[pp_2] != 0) {
          if (k_base_1 < 3072) {
            int w_thread_base =
                tile_1 % 2 * 8 * 1024 + pp_2 * 8 * 1024 + rr_1 * 1024 + tid * 8;
            asm volatile(
                "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t *>(&w_carriers[0])),
                  "=r"(*reinterpret_cast<uint32_t *>(&w_carriers[(0) + 1])),
                  "=r"(*reinterpret_cast<uint32_t *>(&w_carriers[(0) + 2])),
                  "=r"(*reinterpret_cast<uint32_t *>(&w_carriers[(0) + 3]))
                : "r"(w_smem_addr + (unsigned int)(w_thread_base * 2)));
            {
#pragma unroll
              for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile("{\n\t"
                             "shl.b32 %0, %2, 16;\n\t"
                             "and.b32 %1, %2, 0xffff0000;\n\t"
                             "}\n"
                             : "=f"((&w_values[_pair * 2])[0]),
                               "=f"((&w_values[_pair * 2])[1])
                             : "r"(w_carriers[_pair]));
              }
            }
#pragma unroll
            for (int element = 0; element < 8; element++) {
              float _fma_0 =
                  __fmaf_rn(x_values[element], w_values[element], partial);
              partial = _fma_0;
            }
          }
        }
        float _warp_reduce_0 = partial;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
          _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
        partial = _warp_reduce_0;
        if (lane == 0) {
          warp_partials[(pp_2 * 8 + rr_1) * 4 + warp] = partial;
        }
      }
    }
    __syncthreads();
    if (warp == 0) {
      if (lane < 8) {
        int owner_index = lane;
        float cta_partial = 0.0f;
#pragma unroll
        for (int source_warp = 0; source_warp < 4; source_warp++) {
          cta_partial += warp_partials[owner_index * 4 + source_warp];
        }
        owned_accum += cta_partial;
      }
    }
    __syncthreads();
    if (tile_1 + ((1) ? 2 : 3) < 3) {
      int refill_k = (tile_1 + ((1) ? 2 : 3)) * 1024 + tid * 8;
#pragma unroll
      for (int pp_3 = 0; pp_3 < 1; pp_3++) {
        if (valid[pp_3] != 0) {
          if (refill_k < 3072) {
            asm volatile(
                "cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                    x_smem_addr +
                    (unsigned int)(((tile_1 + ((1) ? 2 : 3)) % 2 * 1024 +
                                    pp_3 * 1024 + tid * 8) *
                                   2)),
                "l"(reinterpret_cast<const __nv_bfloat16 *>(x_raw) +
                    (tokens[pp_3] * 3072 + (long long)refill_k)));
#pragma unroll
            for (int rr_2 = 0; rr_2 < 8; rr_2++) {
              int rank_row_1 = rank_base + rr_2;
              long long weight_index_1 =
                  ((loras[pp_3] * (long long)num_experts + experts[pp_3]) * 32 +
                   (long long)rank_row_1) *
                      3072 +
                  (long long)refill_k;
              asm volatile(
                  "cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                      w_smem_addr +
                      (unsigned int)(((tile_1 + ((1) ? 2 : 3)) % 2 * 8 * 1024 +
                                      pp_3 * 8 * 1024 + rr_2 * 1024 + tid * 8) *
                                     2)),
                  "l"(reinterpret_cast<const __nv_bfloat16 *>(lora_a_raw) +
                      weight_index_1));
            }
          }
        }
      }
      asm volatile("cp.async.commit_group;");
    }
  }
  if (warp == 0) {
    if (lane < 8) {
      int owner_pp = lane / 8;
      int owner_rr = lane % 8;
      int pair_1 = pair_block + owner_pp;
      if (pair_1 < num_pairs) {
        *(reinterpret_cast<__nv_bfloat16 *>(
              reinterpret_cast<__nv_bfloat16 *>(shrink_out_raw) +
              (pair_1 * 32 + rank_base + owner_rr)) +
          (0)) = __float2bfloat16_rn(owned_accum);
      }
    }
  }
}

} // extern "C"

#undef BLACKWELL_INF
#undef NUM_MAIN_STAGES
#undef SMEM_TOTAL
#undef SMEM_WARP_PARTIALS_OFF
#undef SMEM_WARP_PARTIALS_STAGE_BYTES
#undef SMEM_WARP_PARTIALS_STRIDE
#undef SMEM_W_SMEM_OFF
#undef SMEM_W_SMEM_STAGE_BYTES
#undef SMEM_W_SMEM_STRIDE
#undef SMEM_X_SMEM_OFF
#undef SMEM_X_SMEM_STAGE_BYTES
#undef SMEM_X_SMEM_STRIDE
#undef THREADS
#undef w_smem_addr
#undef warp_partials_addr
#undef x_smem_addr

#define BLACKWELL_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SHRINK_STAGE_OFF 0
#define SMEM_SHRINK_STAGE_STAGE_BYTES 128
#define SMEM_SHRINK_STAGE_STRIDE 128
#define SMEM_TOTAL 128
#define THREADS 64

extern "C" {

__global__
__launch_bounds__(64, 1) void kernel_flashinfer_bgmv_moe_expand_token_t64_bf16_h3072_r32(
    float *__restrict__ y_accum, uint16_t *__restrict__ shrink_raw,
    uint16_t *__restrict__ lora_b_raw, long long *__restrict__ sorted_token_ids,
    long long *__restrict__ expert_ids, long long *__restrict__ lora_indices,
    float *__restrict__ topk_weights, int num_pairs, int num_experts,
    int num_tokens, int output_stride, int output_offset) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  // Kernel setup ops
  __nv_bfloat16 *shrink_stage = reinterpret_cast<__nv_bfloat16 *>(smem_raw + 0);
  const int shrink_stage_addr = smem + 0;

  // === Task calls (dependency order) ===
  int token = blockIdx.x;
  int output_col = blockIdx.y * 64 + tid;
  unsigned int activation_carriers[4];
  float activation_values[8];
  if (token < num_tokens) {
    long long lora_id = lora_indices[token];
    if (lora_id >= 0) {
      int pair_base = token * 2;
      int contiguous = 0;
      if (num_pairs == num_tokens * 2) {
        if (pair_base + 1 < num_pairs) {
          if (sorted_token_ids[pair_base] == (long long)token) {
            if (sorted_token_ids[pair_base + 1] == (long long)token) {
              contiguous = 1;
            }
          }
        }
      }
      if (contiguous != 0) {
        if (tid < 8) {
          int stage_route = tid / 4;
          int stage_rank_block = tid % 4;
          asm volatile(
              "cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                  shrink_stage_addr +
                  (unsigned int)((stage_route * 32 + stage_rank_block * 8) *
                                 2)),
              "l"(reinterpret_cast<const __nv_bfloat16 *>(shrink_raw) +
                  ((pair_base + stage_route) * 32 + stage_rank_block * 8)));
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        if (output_col < 3072) {
          float total = 0.0f;
#pragma unroll
          for (int route = 0; route < 2; route++) {
            int pair = pair_base + route;
            long long expert = expert_ids[pair];
            float route_partial = 0.0f;
#pragma unroll
            for (int rank_block = 0; rank_block < 4; rank_block++) {
              int rank_col = rank_block * 8;
              asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                           : "=r"(*reinterpret_cast<uint32_t *>(
                                 &activation_carriers[0])),
                             "=r"(*reinterpret_cast<uint32_t *>(
                                 &activation_carriers[(0) + 1])),
                             "=r"(*reinterpret_cast<uint32_t *>(
                                 &activation_carriers[(0) + 2])),
                             "=r"(*reinterpret_cast<uint32_t *>(
                                 &activation_carriers[(0) + 3]))
                           : "r"(shrink_stage_addr +
                                 (unsigned int)((route * 32 + rank_col) * 2)));
              {
#pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                  asm volatile("{\n\t"
                               "shl.b32 %0, %2, 16;\n\t"
                               "and.b32 %1, %2, 0xffff0000;\n\t"
                               "}\n"
                               : "=f"((&activation_values[_pair * 2])[0]),
                                 "=f"((&activation_values[_pair * 2])[1])
                               : "r"(activation_carriers[_pair]));
                }
              }
              long long weight_index =
                  ((lora_id * (long long)num_experts + expert) * 3072 +
                   (long long)output_col) *
                      32 +
                  (long long)rank_col;
              float _vec_load_0[8];
              {
                const uint4 *_vptr_0 = reinterpret_cast<const uint4 *>(
                    reinterpret_cast<const __nv_bfloat16 *>(lora_b_raw) +
                    weight_index + 0);
                uint4 _vld_0[1];
#pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                  _vld_0[_blk] = _vptr_0[_blk];
                  uint32_t *_vpairs_0 =
                      reinterpret_cast<uint32_t *>(&_vld_0[_blk]);
#pragma unroll
                  for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]),
                          "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_0[_pair]));
                  }
                }
              }
#pragma unroll
              for (int element = 0; element < 8; element++) {
                float _fma_0 = __fmaf_rn(activation_values[element],
                                         _vec_load_0[element], route_partial);
                route_partial = _fma_0;
              }
            }
            float _fma_1 = __fmaf_rn(route_partial, topk_weights[pair], total);
            total = _fma_1;
          }
          *(reinterpret_cast<float *>(y_accum + (token * output_stride +
                                                 output_offset + output_col)) +
            (0)) = total;
        }
      } else if (output_col < 3072) {
        float total_1 = 0.0f;
#pragma unroll 1
        for (int pair_1 = 0; pair_1 < num_pairs; pair_1++) {
          if (sorted_token_ids[pair_1] == (long long)token) {
            long long expert_1 = expert_ids[pair_1];
            float route_partial_1 = 0.0f;
#pragma unroll
            for (int rank_block_1 = 0; rank_block_1 < 4; rank_block_1++) {
              int rank_col_1 = rank_block_1 * 8;
              float _vec_load_1[8];
              {
                const uint4 *_vptr_1 = reinterpret_cast<const uint4 *>(
                    reinterpret_cast<const __nv_bfloat16 *>(shrink_raw) +
                    (pair_1 * 32 + rank_col_1) + 0);
                uint4 _vld_1[1];
#pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                  _vld_1[_blk] = _vptr_1[_blk];
                  uint32_t *_vpairs_1 =
                      reinterpret_cast<uint32_t *>(&_vld_1[_blk]);
#pragma unroll
                  for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]),
                          "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_1[_pair]));
                  }
                }
              }
              long long weight_index_1 =
                  ((lora_id * (long long)num_experts + expert_1) * 3072 +
                   (long long)output_col) *
                      32 +
                  (long long)rank_col_1;
              float _vec_load_2[8];
              {
                const uint4 *_vptr_2 = reinterpret_cast<const uint4 *>(
                    reinterpret_cast<const __nv_bfloat16 *>(lora_b_raw) +
                    weight_index_1 + 0);
                uint4 _vld_2[1];
#pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                  _vld_2[_blk] = _vptr_2[_blk];
                  uint32_t *_vpairs_2 =
                      reinterpret_cast<uint32_t *>(&_vld_2[_blk]);
#pragma unroll
                  for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]),
                          "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_2[_pair]));
                  }
                }
              }
#pragma unroll
              for (int element_1 = 0; element_1 < 8; element_1++) {
                float _fma_2 =
                    __fmaf_rn(_vec_load_1[element_1], _vec_load_2[element_1],
                              route_partial_1);
                route_partial_1 = _fma_2;
              }
            }
            float _fma_3 =
                __fmaf_rn(route_partial_1, topk_weights[pair_1], total_1);
            total_1 = _fma_3;
          }
        }
        *(reinterpret_cast<float *>(
              y_accum + (token * output_stride + output_offset + output_col)) +
          (0)) = total_1;
      }
    } else if (output_col < 3072) {
      *(reinterpret_cast<float *>(
            y_accum + (token * output_stride + output_offset + output_col)) +
        (0)) = 0.0f;
    }
  }
}

} // extern "C"

#undef BLACKWELL_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SHRINK_STAGE_OFF
#undef SMEM_SHRINK_STAGE_STAGE_BYTES
#undef SMEM_SHRINK_STAGE_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef shrink_stage_addr

#define BLACKWELL_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 128

extern "C" {

__global__
__launch_bounds__(128, 1) void kernel_flashinfer_bgmv_moe_expand_pair_owned_bf16_h3072_r32_t128(
    float *__restrict__ y_accum, uint16_t *__restrict__ shrink_raw,
    uint16_t *__restrict__ lora_b_raw, long long *__restrict__ sorted_token_ids,
    long long *__restrict__ expert_ids, long long *__restrict__ lora_indices,
    float *__restrict__ topk_weights, int num_pairs, int num_experts,
    int num_tokens, int output_stride, int output_offset) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  // === Task calls (dependency order) ===
  int pair = blockIdx.x;
  int output_col = blockIdx.y * 128 + tid;
  if (pair < num_pairs) {
    if (output_col < 3072) {
      long long token = sorted_token_ids[pair];
      if (token >= 0) {
        if (token < (long long)num_tokens) {
          long long lora_id = lora_indices[token];
          if (lora_id >= 0) {
            long long expert = expert_ids[pair];
            float partial = 0.0f;
#pragma unroll
            for (int rank_block = 0; rank_block < 4; rank_block++) {
              int rank_col = rank_block * 8;
              float _vec_load_0[8];
              {
                const uint4 *_vptr_0 = reinterpret_cast<const uint4 *>(
                    reinterpret_cast<const __nv_bfloat16 *>(shrink_raw) +
                    (pair * 32 + rank_col) + 0);
                uint4 _vld_0[1];
#pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                  _vld_0[_blk] = _vptr_0[_blk];
                  uint32_t *_vpairs_0 =
                      reinterpret_cast<uint32_t *>(&_vld_0[_blk]);
#pragma unroll
                  for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]),
                          "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_0[_pair]));
                  }
                }
              }
              long long weight_index =
                  ((lora_id * (long long)num_experts + expert) * 3072 +
                   (long long)output_col) *
                      32 +
                  (long long)rank_col;
              float _vec_load_1[8];
              {
                const uint4 *_vptr_1 = reinterpret_cast<const uint4 *>(
                    reinterpret_cast<const __nv_bfloat16 *>(lora_b_raw) +
                    weight_index + 0);
                uint4 _vld_1[1];
#pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                  _vld_1[_blk] = _vptr_1[_blk];
                  uint32_t *_vpairs_1 =
                      reinterpret_cast<uint32_t *>(&_vld_1[_blk]);
#pragma unroll
                  for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]),
                          "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_1[_pair]));
                  }
                }
              }
#pragma unroll
              for (int element = 0; element < 8; element++) {
                float _fma_0 = __fmaf_rn(_vec_load_0[element],
                                         _vec_load_1[element], partial);
                partial = _fma_0;
              }
            }
            atomicAdd(
                &y_accum[token * (long long)output_stride +
                         (long long)output_offset + (long long)output_col],
                partial * topk_weights[pair]);
          }
        }
      }
    }
  }
}

} // extern "C"

#undef BLACKWELL_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define BLACKWELL_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SHRINK_STAGE_OFF 0
#define SMEM_SHRINK_STAGE_STAGE_BYTES 128
#define SMEM_SHRINK_STAGE_STRIDE 128
#define SMEM_TOTAL 128
#define THREADS 128

extern "C" {

__global__
__launch_bounds__(128, 1) void kernel_flashinfer_bgmv_moe_expand_token_bf16_h3072_r32(
    float *__restrict__ y_accum, uint16_t *__restrict__ shrink_raw,
    uint16_t *__restrict__ lora_b_raw, long long *__restrict__ sorted_token_ids,
    long long *__restrict__ expert_ids, long long *__restrict__ lora_indices,
    float *__restrict__ topk_weights, int num_pairs, int num_experts,
    int num_tokens, int output_stride, int output_offset) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  // Kernel setup ops
  __nv_bfloat16 *shrink_stage = reinterpret_cast<__nv_bfloat16 *>(smem_raw + 0);
  const int shrink_stage_addr = smem + 0;

  // === Task calls (dependency order) ===
  int token = blockIdx.x;
  int output_col = blockIdx.y * 128 + tid;
  unsigned int activation_carriers[4];
  float activation_values[8];
  if (token < num_tokens) {
    if (output_col < 3072) {
      long long lora_id = lora_indices[token];
      if (lora_id >= 0) {
        int pair_base = token * 2;
        int contiguous = 0;
        if (pair_base + 1 < num_pairs) {
          if (sorted_token_ids[pair_base] == (long long)token) {
            if (sorted_token_ids[pair_base + 1] == (long long)token) {
              contiguous = 1;
            }
          }
        }
        float total = 0.0f;
        if (contiguous != 0) {
          if (tid < 8) {
            int stage_route = tid / 4;
            int stage_rank_block = tid % 4;
            asm volatile(
                "cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                    shrink_stage_addr +
                    (unsigned int)((stage_route * 32 + stage_rank_block * 8) *
                                   2)),
                "l"(reinterpret_cast<const __nv_bfloat16 *>(shrink_raw) +
                    ((pair_base + stage_route) * 32 + stage_rank_block * 8)));
          }
          asm volatile("cp.async.commit_group;");
          asm volatile("cp.async.wait_group 0;");
          __syncthreads();
#pragma unroll
          for (int route = 0; route < 2; route++) {
            int pair = pair_base + route;
            long long expert = expert_ids[pair];
            float route_partial = 0.0f;
#pragma unroll
            for (int rank_block = 0; rank_block < 4; rank_block++) {
              int rank_col = rank_block * 8;
              asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                           : "=r"(*reinterpret_cast<uint32_t *>(
                                 &activation_carriers[0])),
                             "=r"(*reinterpret_cast<uint32_t *>(
                                 &activation_carriers[(0) + 1])),
                             "=r"(*reinterpret_cast<uint32_t *>(
                                 &activation_carriers[(0) + 2])),
                             "=r"(*reinterpret_cast<uint32_t *>(
                                 &activation_carriers[(0) + 3]))
                           : "r"(shrink_stage_addr +
                                 (unsigned int)((route * 32 + rank_col) * 2)));
              {
#pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                  asm volatile("{\n\t"
                               "shl.b32 %0, %2, 16;\n\t"
                               "and.b32 %1, %2, 0xffff0000;\n\t"
                               "}\n"
                               : "=f"((&activation_values[_pair * 2])[0]),
                                 "=f"((&activation_values[_pair * 2])[1])
                               : "r"(activation_carriers[_pair]));
                }
              }
              long long weight_index =
                  ((lora_id * (long long)num_experts + expert) * 3072 +
                   (long long)output_col) *
                      32 +
                  (long long)rank_col;
              float _vec_load_0[8];
              {
                const uint4 *_vptr_0 = reinterpret_cast<const uint4 *>(
                    reinterpret_cast<const __nv_bfloat16 *>(lora_b_raw) +
                    weight_index + 0);
                uint4 _vld_0[1];
#pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                  _vld_0[_blk] = _vptr_0[_blk];
                  uint32_t *_vpairs_0 =
                      reinterpret_cast<uint32_t *>(&_vld_0[_blk]);
#pragma unroll
                  for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]),
                          "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_0[_pair]));
                  }
                }
              }
#pragma unroll
              for (int element = 0; element < 8; element++) {
                float _fma_0 = __fmaf_rn(activation_values[element],
                                         _vec_load_0[element], route_partial);
                route_partial = _fma_0;
              }
            }
            float _fma_1 = __fmaf_rn(route_partial, topk_weights[pair], total);
            total = _fma_1;
          }
        } else {
#pragma unroll 1
          for (int pair_1 = 0; pair_1 < num_pairs; pair_1++) {
            if (sorted_token_ids[pair_1] == (long long)token) {
              long long expert_1 = expert_ids[pair_1];
              float route_partial_1 = 0.0f;
#pragma unroll
              for (int rank_block_1 = 0; rank_block_1 < 4; rank_block_1++) {
                int rank_col_1 = rank_block_1 * 8;
                float _vec_load_1[8];
                {
                  const uint4 *_vptr_1 = reinterpret_cast<const uint4 *>(
                      reinterpret_cast<const __nv_bfloat16 *>(shrink_raw) +
                      (pair_1 * 32 + rank_col_1) + 0);
                  uint4 _vld_1[1];
#pragma unroll
                  for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_1[_blk] = _vptr_1[_blk];
                    uint32_t *_vpairs_1 =
                        reinterpret_cast<uint32_t *>(&_vld_1[_blk]);
#pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                      asm volatile(
                          "{\n\t"
                          "shl.b32 %0, %2, 16;\n\t"
                          "and.b32 %1, %2, 0xffff0000;\n\t"
                          "}\n"
                          : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]),
                            "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                          : "r"(_vpairs_1[_pair]));
                    }
                  }
                }
                long long weight_index_1 =
                    ((lora_id * (long long)num_experts + expert_1) * 3072 +
                     (long long)output_col) *
                        32 +
                    (long long)rank_col_1;
                float _vec_load_2[8];
                {
                  const uint4 *_vptr_2 = reinterpret_cast<const uint4 *>(
                      reinterpret_cast<const __nv_bfloat16 *>(lora_b_raw) +
                      weight_index_1 + 0);
                  uint4 _vld_2[1];
#pragma unroll
                  for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_2[_blk] = _vptr_2[_blk];
                    uint32_t *_vpairs_2 =
                        reinterpret_cast<uint32_t *>(&_vld_2[_blk]);
#pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                      asm volatile(
                          "{\n\t"
                          "shl.b32 %0, %2, 16;\n\t"
                          "and.b32 %1, %2, 0xffff0000;\n\t"
                          "}\n"
                          : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]),
                            "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                          : "r"(_vpairs_2[_pair]));
                    }
                  }
                }
#pragma unroll
                for (int element_1 = 0; element_1 < 8; element_1++) {
                  float _fma_2 =
                      __fmaf_rn(_vec_load_1[element_1], _vec_load_2[element_1],
                                route_partial_1);
                  route_partial_1 = _fma_2;
                }
              }
              float _fma_3 =
                  __fmaf_rn(route_partial_1, topk_weights[pair_1], total);
              total = _fma_3;
            }
          }
        }
        *(reinterpret_cast<float *>(
              y_accum + (token * output_stride + output_offset + output_col)) +
          (0)) = total;
      } else {
        *(reinterpret_cast<float *>(
              y_accum + (token * output_stride + output_offset + output_col)) +
          (0)) = 0.0f;
      }
    }
  }
}

} // extern "C"

#undef BLACKWELL_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SHRINK_STAGE_OFF
#undef SMEM_SHRINK_STAGE_STAGE_BYTES
#undef SMEM_SHRINK_STAGE_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef shrink_stage_addr

#define BLACKWELL_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SHRINK_STAGE_OFF 0
#define SMEM_SHRINK_STAGE_STAGE_BYTES 128
#define SMEM_SHRINK_STAGE_STRIDE 128
#define SMEM_TOTAL 128
#define THREADS 128

extern "C" {

__global__
__launch_bounds__(128, 1) void kernel_flashinfer_bgmv_moe_expand_token_dual_col_bf16_h3072_r32(
    float *__restrict__ y_accum, uint16_t *__restrict__ shrink_raw,
    uint16_t *__restrict__ lora_b_raw, long long *__restrict__ sorted_token_ids,
    long long *__restrict__ expert_ids, long long *__restrict__ lora_indices,
    float *__restrict__ topk_weights, int num_pairs, int num_experts,
    int num_tokens, int output_stride, int output_offset) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  // Kernel setup ops
  __nv_bfloat16 *shrink_stage = reinterpret_cast<__nv_bfloat16 *>(smem_raw + 0);
  const int shrink_stage_addr = smem + 0;

  // === Task calls (dependency order) ===
  int token = blockIdx.x;
  int output_col0 = blockIdx.y * 256 + tid;
  int output_col1 = output_col0 + 128;
  unsigned int activation_carriers[4];
  float activation_values[8];
  if (token < num_tokens) {
    long long lora_id = lora_indices[token];
    if (lora_id >= 0) {
      int pair_base = token * 2;
      int contiguous = 0;
      if (num_pairs == num_tokens * 2) {
        if (pair_base + 1 < num_pairs) {
          if (sorted_token_ids[pair_base] == (long long)token) {
            if (sorted_token_ids[pair_base + 1] == (long long)token) {
              contiguous = 1;
            }
          }
        }
      }
      int valid0 = 0;
      int valid1 = 0;
      if (output_col0 < 3072) {
        valid0 = 1;
      }
      if (output_col1 < 3072) {
        valid1 = 1;
      }
      float total0 = 0.0f;
      float total1 = 0.0f;
      if (contiguous != 0) {
        if (tid < 8) {
          int stage_route = tid / 4;
          int stage_rank_block = tid % 4;
          asm volatile(
              "cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                  shrink_stage_addr +
                  (unsigned int)((stage_route * 32 + stage_rank_block * 8) *
                                 2)),
              "l"(reinterpret_cast<const __nv_bfloat16 *>(shrink_raw) +
                  ((pair_base + stage_route) * 32 + stage_rank_block * 8)));
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
#pragma unroll
        for (int route = 0; route < 2; route++) {
          int pair = pair_base + route;
          long long expert = expert_ids[pair];
          float route_partial0 = 0.0f;
          float route_partial1 = 0.0f;
#pragma unroll
          for (int rank_block = 0; rank_block < 4; rank_block++) {
            int rank_col = rank_block * 8;
            asm volatile(
                "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t *>(&activation_carriers[0])),
                  "=r"(*reinterpret_cast<uint32_t *>(
                      &activation_carriers[(0) + 1])),
                  "=r"(*reinterpret_cast<uint32_t *>(
                      &activation_carriers[(0) + 2])),
                  "=r"(*reinterpret_cast<uint32_t *>(
                      &activation_carriers[(0) + 3]))
                : "r"(shrink_stage_addr +
                      (unsigned int)((route * 32 + rank_col) * 2)));
            {
#pragma unroll
              for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile("{\n\t"
                             "shl.b32 %0, %2, 16;\n\t"
                             "and.b32 %1, %2, 0xffff0000;\n\t"
                             "}\n"
                             : "=f"((&activation_values[_pair * 2])[0]),
                               "=f"((&activation_values[_pair * 2])[1])
                             : "r"(activation_carriers[_pair]));
              }
            }
            if (valid0 != 0) {
              long long weight_index0 =
                  ((lora_id * (long long)num_experts + expert) * 3072 +
                   (long long)output_col0) *
                      32 +
                  (long long)rank_col;
              float _vec_load_0[8];
              {
                const uint4 *_vptr_0 = reinterpret_cast<const uint4 *>(
                    reinterpret_cast<const __nv_bfloat16 *>(lora_b_raw) +
                    weight_index0 + 0);
                uint4 _vld_0[1];
#pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                  _vld_0[_blk] = _vptr_0[_blk];
                  uint32_t *_vpairs_0 =
                      reinterpret_cast<uint32_t *>(&_vld_0[_blk]);
#pragma unroll
                  for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]),
                          "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_0[_pair]));
                  }
                }
              }
#pragma unroll
              for (int element = 0; element < 8; element++) {
                float _fma_0 = __fmaf_rn(activation_values[element],
                                         _vec_load_0[element], route_partial0);
                route_partial0 = _fma_0;
              }
            }
            if (valid1 != 0) {
              long long weight_index1 =
                  ((lora_id * (long long)num_experts + expert) * 3072 +
                   (long long)output_col1) *
                      32 +
                  (long long)rank_col;
              float _vec_load_1[8];
              {
                const uint4 *_vptr_1 = reinterpret_cast<const uint4 *>(
                    reinterpret_cast<const __nv_bfloat16 *>(lora_b_raw) +
                    weight_index1 + 0);
                uint4 _vld_1[1];
#pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                  _vld_1[_blk] = _vptr_1[_blk];
                  uint32_t *_vpairs_1 =
                      reinterpret_cast<uint32_t *>(&_vld_1[_blk]);
#pragma unroll
                  for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]),
                          "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_1[_pair]));
                  }
                }
              }
#pragma unroll
              for (int element_1 = 0; element_1 < 8; element_1++) {
                float _fma_1 =
                    __fmaf_rn(activation_values[element_1],
                              _vec_load_1[element_1], route_partial1);
                route_partial1 = _fma_1;
              }
            }
          }
          if (valid0 != 0) {
            float _fma_2 =
                __fmaf_rn(route_partial0, topk_weights[pair], total0);
            total0 = _fma_2;
          }
          if (valid1 != 0) {
            float _fma_3 =
                __fmaf_rn(route_partial1, topk_weights[pair], total1);
            total1 = _fma_3;
          }
        }
      } else {
#pragma unroll 1
        for (int pair_1 = 0; pair_1 < num_pairs; pair_1++) {
          if (sorted_token_ids[pair_1] == (long long)token) {
            long long expert_1 = expert_ids[pair_1];
            float route_partial0_1 = 0.0f;
            float route_partial1_1 = 0.0f;
#pragma unroll
            for (int rank_block_1 = 0; rank_block_1 < 4; rank_block_1++) {
              int rank_col_1 = rank_block_1 * 8;
              float _vec_load_2[8];
              {
                const uint4 *_vptr_2 = reinterpret_cast<const uint4 *>(
                    reinterpret_cast<const __nv_bfloat16 *>(shrink_raw) +
                    (pair_1 * 32 + rank_col_1) + 0);
                uint4 _vld_2[1];
#pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                  _vld_2[_blk] = _vptr_2[_blk];
                  uint32_t *_vpairs_2 =
                      reinterpret_cast<uint32_t *>(&_vld_2[_blk]);
#pragma unroll
                  for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]),
                          "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_2[_pair]));
                  }
                }
              }
              if (valid0 != 0) {
                long long weight_index0_1 =
                    ((lora_id * (long long)num_experts + expert_1) * 3072 +
                     (long long)output_col0) *
                        32 +
                    (long long)rank_col_1;
                float _vec_load_3[8];
                {
                  const uint4 *_vptr_3 = reinterpret_cast<const uint4 *>(
                      reinterpret_cast<const __nv_bfloat16 *>(lora_b_raw) +
                      weight_index0_1 + 0);
                  uint4 _vld_3[1];
#pragma unroll
                  for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_3[_blk] = _vptr_3[_blk];
                    uint32_t *_vpairs_3 =
                        reinterpret_cast<uint32_t *>(&_vld_3[_blk]);
#pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                      asm volatile(
                          "{\n\t"
                          "shl.b32 %0, %2, 16;\n\t"
                          "and.b32 %1, %2, 0xffff0000;\n\t"
                          "}\n"
                          : "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[0]),
                            "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[1])
                          : "r"(_vpairs_3[_pair]));
                    }
                  }
                }
#pragma unroll
                for (int element_2 = 0; element_2 < 8; element_2++) {
                  float _fma_4 =
                      __fmaf_rn(_vec_load_2[element_2], _vec_load_3[element_2],
                                route_partial0_1);
                  route_partial0_1 = _fma_4;
                }
              }
              if (valid1 != 0) {
                long long weight_index1_1 =
                    ((lora_id * (long long)num_experts + expert_1) * 3072 +
                     (long long)output_col1) *
                        32 +
                    (long long)rank_col_1;
                float _vec_load_4[8];
                {
                  const uint4 *_vptr_4 = reinterpret_cast<const uint4 *>(
                      reinterpret_cast<const __nv_bfloat16 *>(lora_b_raw) +
                      weight_index1_1 + 0);
                  uint4 _vld_4[1];
#pragma unroll
                  for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_4[_blk] = _vptr_4[_blk];
                    uint32_t *_vpairs_4 =
                        reinterpret_cast<uint32_t *>(&_vld_4[_blk]);
#pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                      asm volatile(
                          "{\n\t"
                          "shl.b32 %0, %2, 16;\n\t"
                          "and.b32 %1, %2, 0xffff0000;\n\t"
                          "}\n"
                          : "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[0]),
                            "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[1])
                          : "r"(_vpairs_4[_pair]));
                    }
                  }
                }
#pragma unroll
                for (int element_3 = 0; element_3 < 8; element_3++) {
                  float _fma_5 =
                      __fmaf_rn(_vec_load_2[element_3], _vec_load_4[element_3],
                                route_partial1_1);
                  route_partial1_1 = _fma_5;
                }
              }
            }
            if (valid0 != 0) {
              float _fma_6 =
                  __fmaf_rn(route_partial0_1, topk_weights[pair_1], total0);
              total0 = _fma_6;
            }
            if (valid1 != 0) {
              float _fma_7 =
                  __fmaf_rn(route_partial1_1, topk_weights[pair_1], total1);
              total1 = _fma_7;
            }
          }
        }
      }
      if (valid0 != 0) {
        *(reinterpret_cast<float *>(
              y_accum + (token * output_stride + output_offset + output_col0)) +
          (0)) = total0;
      }
      if (valid1 != 0) {
        *(reinterpret_cast<float *>(
              y_accum + (token * output_stride + output_offset + output_col1)) +
          (0)) = total1;
      }
    } else {
      if (output_col0 < 3072) {
        *(reinterpret_cast<float *>(
              y_accum + (token * output_stride + output_offset + output_col0)) +
          (0)) = 0.0f;
      }
      if (output_col1 < 3072) {
        *(reinterpret_cast<float *>(
              y_accum + (token * output_stride + output_offset + output_col1)) +
          (0)) = 0.0f;
      }
    }
  }
}

} // extern "C"

#undef BLACKWELL_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SHRINK_STAGE_OFF
#undef SMEM_SHRINK_STAGE_STAGE_BYTES
#undef SMEM_SHRINK_STAGE_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef shrink_stage_addr
