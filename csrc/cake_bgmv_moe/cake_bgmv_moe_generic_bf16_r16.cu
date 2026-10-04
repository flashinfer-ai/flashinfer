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
// Bundle: Cake BGMV MoE generic-shape shrink and deterministic expand, bf16 rank 16.
// Target: sm_90a, sm_100a, sm_103a (cp.async, shuffles, atomics, FMA only); compile flags: none.
// Generated file; do not edit manually.
#include <cuda_bf16.h>

#include <cstdint>

__device__ __forceinline__ int make_warp_uniform(int x) {
  int result;
  asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;" : "=r"(result) : "r"(x));
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
#define SMEM_SPLIT_FLAG_OFF 221696
#define SMEM_SPLIT_FLAG_STAGE_BYTES 16
#define SMEM_SPLIT_FLAG_STRIDE 16
#define SMEM_TOTAL 221824
#define THREADS 128

extern "C" {

__global__ __launch_bounds__(128, 1) void kernel_flashinfer_bgmv_moe_shrink_generic_bf16_r16_p4_s3(
    uint16_t* __restrict__ shrink_out_raw, uint16_t* __restrict__ x_raw,
    uint16_t* __restrict__ lora_a_raw, long long* __restrict__ sorted_token_ids,
    long long* __restrict__ expert_ids, long long* __restrict__ lora_indices, int num_pairs,
    int num_experts, int num_tokens, unsigned int* __restrict__ route_index_raw, int route_build,
    int hidden, int num_tiles, float* __restrict__ split_partials_raw,
    unsigned int* __restrict__ split_counters_raw, int num_splits) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  const int cta_rank = 0;

  // Kernel setup ops
  __nv_bfloat16* x_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
  const int x_smem_addr = smem + 0;
  __nv_bfloat16* w_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 24576);
  const int w_smem_addr = smem + 24576;
  float* warp_partials = reinterpret_cast<float*>(smem_raw + 221184);
  const int warp_partials_addr = smem + 221184;
  int* split_flag = reinterpret_cast<int*>(smem_raw + 221696);
  const int split_flag_addr = smem + 221696;

  // === Task calls (dependency order) ===
  int pair_block = blockIdx.x;
  int rank_block = blockIdx.y;
  int rank_base = rank_block * 8;
  int split = blockIdx.z;
  int tiles_per_split = (num_tiles + num_splits - 1) / num_splits;
  int tile_begin = split * tiles_per_split;
  int local_tiles = num_tiles - tile_begin;
  if (local_tiles > tiles_per_split) {
    local_tiles = tiles_per_split;
  }
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
  for (int tile = 0; tile < 2; tile++) {
    if (local_tiles > tile) {
      int k_base = (tile_begin + tile) * 1024 + tid * 8;
#pragma unroll
      for (int pp_1 = 0; pp_1 < 4; pp_1++) {
        if (valid[pp_1] != 0) {
          if (k_base < hidden) {
            asm volatile(
                "cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                    x_smem_addr + (unsigned int)((tile * 4 * 1024 + pp_1 * 1024 + tid * 8) * 2)),
                "l"(reinterpret_cast<const __nv_bfloat16*>(x_raw) +
                    (tokens[pp_1] * (long long)hidden + (long long)k_base)));
#pragma unroll
            for (int rr = 0; rr < 8; rr++) {
              int rank_row = rank_base + rr;
              long long weight_index =
                  ((loras[pp_1] * (long long)num_experts + experts[pp_1]) * 16 +
                   (long long)rank_row) *
                      (long long)hidden +
                  (long long)k_base;
              asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                               w_smem_addr + (unsigned int)((tile * 4 * 8 * 1024 + pp_1 * 8 * 1024 +
                                                             rr * 1024 + tid * 8) *
                                                            2)),
                           "l"(reinterpret_cast<const __nv_bfloat16*>(lora_a_raw) + weight_index));
            }
          }
        }
      }
    }
    asm volatile("cp.async.commit_group;");
  }
  unsigned int route_launch = 0;
  unsigned int route_old[4];
  unsigned int route_base_even[4];
  unsigned int route_base_odd[4];
  int route_publish[4];
#pragma unroll
  for (int pp_2 = 0; pp_2 < 4; pp_2++) {
    route_publish[pp_2] = 0;
  }
  if (route_build != 0) {
    if (blockIdx.y + blockIdx.z == 0) {
      if (tid == 0) {
        route_launch = route_index_raw[0];
#pragma unroll
        for (int pp_3 = 0; pp_3 < 4; pp_3++) {
          if (valid[pp_3] != 0) {
            int route_token = (int)tokens[pp_3];
            route_publish[pp_3] = 1;
            if (num_pairs == num_tokens * 2) {
              int route_offset = pair_block * 4 + pp_3 - route_token * 2;
              if (route_offset >= 0) {
                if (route_offset < 2) {
                  route_publish[pp_3] = 0;
                }
              }
            }
          }
          if (route_publish[pp_3] != 0) {
            int publish_token = (int)tokens[pp_3];
            unsigned int _atomic_old_0 =
                atomicAdd(&reinterpret_cast<unsigned int*>(route_index_raw)[4 + publish_token], 1);
            route_old[pp_3] = _atomic_old_0;
            route_base_even[pp_3] = route_index_raw[4 + num_tokens + publish_token];
            route_base_odd[pp_3] = route_index_raw[4 + 2 * num_tokens + publish_token];
          }
        }
      }
    }
  }
  float acc[32];
#pragma unroll
  for (int owner = 0; owner < 32; owner++) {
    acc[owner] = 0.0f;
  }
  unsigned int x_carriers[4];
  unsigned int w_carriers[4];
  float x_values[8];
  float w_values[8];
#pragma unroll 1
  for (int local = 0; local < local_tiles; local++) {
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();
    int stage = local % 3;
    int refill_local = local + 3 - 1;
    if (refill_local < local_tiles) {
      int refill_stage = refill_local % 3;
      int refill_k = (tile_begin + refill_local) * 1024 + tid * 8;
#pragma unroll
      for (int pp_4 = 0; pp_4 < 4; pp_4++) {
        if (valid[pp_4] != 0) {
          if (refill_k < hidden) {
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                             x_smem_addr +
                             (unsigned int)((refill_stage * 4 * 1024 + pp_4 * 1024 + tid * 8) * 2)),
                         "l"(reinterpret_cast<const __nv_bfloat16*>(x_raw) +
                             (tokens[pp_4] * (long long)hidden + (long long)refill_k)));
#pragma unroll
            for (int rr_1 = 0; rr_1 < 8; rr_1++) {
              int rank_row_1 = rank_base + rr_1;
              long long weight_index_1 =
                  ((loras[pp_4] * (long long)num_experts + experts[pp_4]) * 16 +
                   (long long)rank_row_1) *
                      (long long)hidden +
                  (long long)refill_k;
              asm volatile(
                  "cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                      w_smem_addr + (unsigned int)((refill_stage * 4 * 8 * 1024 + pp_4 * 8 * 1024 +
                                                    rr_1 * 1024 + tid * 8) *
                                                   2)),
                  "l"(reinterpret_cast<const __nv_bfloat16*>(lora_a_raw) + weight_index_1));
            }
          }
        }
      }
    }
    asm volatile("cp.async.commit_group;");
    int k_base_1 = (tile_begin + local) * 1024 + tid * 8;
#pragma unroll
    for (int pp_5 = 0; pp_5 < 4; pp_5++) {
      int x_thread_base = stage * 4 * 1024 + pp_5 * 1024 + tid * 8;
      if (valid[pp_5] != 0) {
        if (k_base_1 < hidden) {
          asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                       : "=r"(*reinterpret_cast<uint32_t*>(&x_carriers[0])),
                         "=r"(*reinterpret_cast<uint32_t*>(&x_carriers[(0) + 1])),
                         "=r"(*reinterpret_cast<uint32_t*>(&x_carriers[(0) + 2])),
                         "=r"(*reinterpret_cast<uint32_t*>(&x_carriers[(0) + 3]))
                       : "r"(x_smem_addr + (unsigned int)(x_thread_base * 2)));
          {
#pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
              asm volatile(
                  "{\n\t"
                  "shl.b32 %0, %2, 16;\n\t"
                  "and.b32 %1, %2, 0xffff0000;\n\t"
                  "}\n"
                  : "=f"((&x_values[_pair * 2])[0]), "=f"((&x_values[_pair * 2])[1])
                  : "r"(x_carriers[_pair]));
            }
          }
        }
      }
#pragma unroll
      for (int rr_2 = 0; rr_2 < 8; rr_2++) {
        if (valid[pp_5] != 0) {
          if (k_base_1 < hidden) {
            int w_thread_base = stage * 4 * 8 * 1024 + pp_5 * 8 * 1024 + rr_2 * 1024 + tid * 8;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(*reinterpret_cast<uint32_t*>(&w_carriers[0])),
                           "=r"(*reinterpret_cast<uint32_t*>(&w_carriers[(0) + 1])),
                           "=r"(*reinterpret_cast<uint32_t*>(&w_carriers[(0) + 2])),
                           "=r"(*reinterpret_cast<uint32_t*>(&w_carriers[(0) + 3]))
                         : "r"(w_smem_addr + (unsigned int)(w_thread_base * 2)));
            {
#pragma unroll
              for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&w_values[_pair * 2])[0]), "=f"((&w_values[_pair * 2])[1])
                    : "r"(w_carriers[_pair]));
              }
            }
#pragma unroll
            for (int element = 0; element < 8; element++) {
              float _fma_0 = __fmaf_rn(x_values[element], w_values[element], acc[pp_5 * 8 + rr_2]);
              acc[pp_5 * 8 + rr_2] = _fma_0;
            }
          }
        }
      }
    }
  }
#pragma unroll
  for (int owner_1 = 0; owner_1 < 32; owner_1++) {
    float _warp_reduce_0 = acc[owner_1];
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
    float warp_sum = _warp_reduce_0;
    if (lane == 0) {
      warp_partials[owner_1 * 4 + warp] = warp_sum;
    }
  }
  __syncthreads();
  float owned_accum = 0.0f;
  int owner_pp = lane / 8;
  int owner_rr = lane % 8;
  int owner_pair = pair_block * 4 + owner_pp;
  if (warp == 0) {
    if (lane < 32) {
#pragma unroll
      for (int source_warp = 0; source_warp < 4; source_warp++) {
        owned_accum += warp_partials[lane * 4 + source_warp];
      }
    }
  }
  if (num_splits == 1) {
    if (warp == 0) {
      if (lane < 32) {
        if (owner_pair < num_pairs) {
          *(reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<__nv_bfloat16*>(shrink_out_raw) +
                                             (owner_pair * 16 + rank_base + owner_rr)) +
            (0)) = __float2bfloat16_rn(owned_accum);
        }
      }
    }
  } else {
    if (warp == 0) {
      if (lane < 32) {
        if (owner_pair < num_pairs) {
          *(reinterpret_cast<float*>(
                reinterpret_cast<float*>(split_partials_raw) +
                ((split * num_pairs + owner_pair) * 16 + rank_base + owner_rr)) +
            (0)) = owned_accum;
        }
      }
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
      unsigned int _atomic_old_1;
      asm volatile(
          "atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
          : "=r"(_atomic_old_1)
          : "l"(&reinterpret_cast<unsigned int*>(split_counters_raw)[pair_block * 2 + rank_block]),
            "r"(static_cast<uint32_t>(1))
          : "memory");
      unsigned int ticket = _atomic_old_1;
      int last_arrival = 0;
      if (ticket == (unsigned int)(num_splits - 1)) {
        last_arrival = 1;
        *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_counters_raw) +
                                          (pair_block * 2 + rank_block)) +
          (0)) = 0;
      }
      split_flag[0] = last_arrival;
    }
    __syncthreads();
    if (split_flag[0] != 0) {
      __threadfence();
      if (warp == 0) {
        if (lane < 32) {
          if (owner_pair < num_pairs) {
            float split_total = 0.0f;
#pragma unroll 1
            for (int source_split = 0; source_split < num_splits; source_split++) {
              split_total += reinterpret_cast<float*>(
                  split_partials_raw)[(source_split * num_pairs + owner_pair) * 16 + rank_base +
                                      owner_rr];
            }
            *(reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<__nv_bfloat16*>(shrink_out_raw) +
                                               (owner_pair * 16 + rank_base + owner_rr)) +
              (0)) = __float2bfloat16_rn(split_total);
          }
        }
      }
    }
  }
  if (route_build != 0) {
    if (blockIdx.y + blockIdx.z == 0) {
      if (tid == 0) {
        int route_parity = (int)(route_launch & 1);
        if (blockIdx.x == 0) {
          *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(route_index_raw) + 1) +
            (0)) = (unsigned int)route_parity;
        }
#pragma unroll
        for (int pp_6 = 0; pp_6 < 4; pp_6++) {
          if (route_publish[pp_6] != 0) {
            int publish_token_1 = (int)tokens[pp_6];
            unsigned int route_base = route_base_even[pp_6];
            if (route_parity != 0) {
              route_base = route_base_odd[pp_6];
            }
            unsigned int route_slot = route_old[pp_6] - route_base;
            if (route_slot < 16) {
              *(reinterpret_cast<unsigned int*>(
                    reinterpret_cast<unsigned int*>(route_index_raw) +
                    (4 + 3 * num_tokens + publish_token_1 * 16 + (int)route_slot)) +
                (0)) = (unsigned int)(pair_block * 4 + pp_6);
            }
          }
        }
      }
    }
  }
}

}  // extern "C"

#undef BLACKWELL_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SPLIT_FLAG_OFF
#undef SMEM_SPLIT_FLAG_STAGE_BYTES
#undef SMEM_SPLIT_FLAG_STRIDE
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
#define SMEM_SPLIT_FLAG_OFF 36992
#define SMEM_SPLIT_FLAG_STAGE_BYTES 16
#define SMEM_SPLIT_FLAG_STRIDE 16
#define SMEM_TOTAL 37120
#define THREADS 128

extern "C" {

__global__ __launch_bounds__(128, 1) void kernel_flashinfer_bgmv_moe_shrink_generic_bf16_r16_p1_s2(
    uint16_t* __restrict__ shrink_out_raw, uint16_t* __restrict__ x_raw,
    uint16_t* __restrict__ lora_a_raw, long long* __restrict__ sorted_token_ids,
    long long* __restrict__ expert_ids, long long* __restrict__ lora_indices, int num_pairs,
    int num_experts, int num_tokens, unsigned int* __restrict__ route_index_raw, int route_build,
    int hidden, int num_tiles, float* __restrict__ split_partials_raw,
    unsigned int* __restrict__ split_counters_raw, int num_splits) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  const int cta_rank = 0;

  // Kernel setup ops
  __nv_bfloat16* x_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
  const int x_smem_addr = smem + 0;
  __nv_bfloat16* w_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 4096);
  const int w_smem_addr = smem + 4096;
  float* warp_partials = reinterpret_cast<float*>(smem_raw + 36864);
  const int warp_partials_addr = smem + 36864;
  int* split_flag = reinterpret_cast<int*>(smem_raw + 36992);
  const int split_flag_addr = smem + 36992;

  // === Task calls (dependency order) ===
  int pair_block = blockIdx.x;
  int rank_block = blockIdx.y;
  int rank_base = rank_block * 8;
  int split = blockIdx.z;
  int tiles_per_split = (num_tiles + num_splits - 1) / num_splits;
  int tile_begin = split * tiles_per_split;
  int local_tiles = num_tiles - tile_begin;
  if (local_tiles > tiles_per_split) {
    local_tiles = tiles_per_split;
  }
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
  for (int tile = 0; tile < 1; tile++) {
    if (local_tiles > tile) {
      int k_base = (tile_begin + tile) * 1024 + tid * 8;
#pragma unroll
      for (int pp_1 = 0; pp_1 < 1; pp_1++) {
        if (valid[pp_1] != 0) {
          if (k_base < hidden) {
            asm volatile(
                "cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                    x_smem_addr + (unsigned int)((tile * 1024 + pp_1 * 1024 + tid * 8) * 2)),
                "l"(reinterpret_cast<const __nv_bfloat16*>(x_raw) +
                    (tokens[pp_1] * (long long)hidden + (long long)k_base)));
#pragma unroll
            for (int rr = 0; rr < 8; rr++) {
              int rank_row = rank_base + rr;
              long long weight_index =
                  ((loras[pp_1] * (long long)num_experts + experts[pp_1]) * 16 +
                   (long long)rank_row) *
                      (long long)hidden +
                  (long long)k_base;
              asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                               w_smem_addr + (unsigned int)((tile * 8 * 1024 + pp_1 * 8 * 1024 +
                                                             rr * 1024 + tid * 8) *
                                                            2)),
                           "l"(reinterpret_cast<const __nv_bfloat16*>(lora_a_raw) + weight_index));
            }
          }
        }
      }
    }
    asm volatile("cp.async.commit_group;");
  }
  unsigned int route_launch = 0;
  unsigned int route_old[1];
  unsigned int route_base_even[1];
  unsigned int route_base_odd[1];
  int route_publish[1];
#pragma unroll
  for (int pp_2 = 0; pp_2 < 1; pp_2++) {
    route_publish[pp_2] = 0;
  }
  if (route_build != 0) {
    if (blockIdx.y + blockIdx.z == 0) {
      if (tid == 0) {
        route_launch = route_index_raw[0];
#pragma unroll
        for (int pp_3 = 0; pp_3 < 1; pp_3++) {
          if (valid[pp_3] != 0) {
            int route_token = (int)tokens[pp_3];
            route_publish[pp_3] = 1;
            if (num_pairs == num_tokens * 2) {
              int route_offset = pair_block + pp_3 - route_token * 2;
              if (route_offset >= 0) {
                if (route_offset < 2) {
                  route_publish[pp_3] = 0;
                }
              }
            }
          }
          if (route_publish[pp_3] != 0) {
            int publish_token = (int)tokens[pp_3];
            unsigned int _atomic_old_0 =
                atomicAdd(&reinterpret_cast<unsigned int*>(route_index_raw)[4 + publish_token], 1);
            route_old[pp_3] = _atomic_old_0;
            route_base_even[pp_3] = route_index_raw[4 + num_tokens + publish_token];
            route_base_odd[pp_3] = route_index_raw[4 + 2 * num_tokens + publish_token];
          }
        }
      }
    }
  }
  float acc[8];
#pragma unroll
  for (int owner = 0; owner < 8; owner++) {
    acc[owner] = 0.0f;
  }
  unsigned int x_carriers[4];
  unsigned int w_carriers[4];
  float x_values[8];
  float w_values[8];
#pragma unroll 1
  for (int local = 0; local < local_tiles; local++) {
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    int stage = local % 2;
    int refill_local = local + 2 - 1;
    if (refill_local < local_tiles) {
      int refill_stage = refill_local % 2;
      int refill_k = (tile_begin + refill_local) * 1024 + tid * 8;
#pragma unroll
      for (int pp_4 = 0; pp_4 < 1; pp_4++) {
        if (valid[pp_4] != 0) {
          if (refill_k < hidden) {
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                             x_smem_addr +
                             (unsigned int)((refill_stage * 1024 + pp_4 * 1024 + tid * 8) * 2)),
                         "l"(reinterpret_cast<const __nv_bfloat16*>(x_raw) +
                             (tokens[pp_4] * (long long)hidden + (long long)refill_k)));
#pragma unroll
            for (int rr_1 = 0; rr_1 < 8; rr_1++) {
              int rank_row_1 = rank_base + rr_1;
              long long weight_index_1 =
                  ((loras[pp_4] * (long long)num_experts + experts[pp_4]) * 16 +
                   (long long)rank_row_1) *
                      (long long)hidden +
                  (long long)refill_k;
              asm volatile(
                  "cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                      w_smem_addr + (unsigned int)((refill_stage * 8 * 1024 + pp_4 * 8 * 1024 +
                                                    rr_1 * 1024 + tid * 8) *
                                                   2)),
                  "l"(reinterpret_cast<const __nv_bfloat16*>(lora_a_raw) + weight_index_1));
            }
          }
        }
      }
    }
    asm volatile("cp.async.commit_group;");
    int k_base_1 = (tile_begin + local) * 1024 + tid * 8;
#pragma unroll
    for (int pp_5 = 0; pp_5 < 1; pp_5++) {
      int x_thread_base = stage * 1024 + pp_5 * 1024 + tid * 8;
      if (valid[pp_5] != 0) {
        if (k_base_1 < hidden) {
          asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                       : "=r"(*reinterpret_cast<uint32_t*>(&x_carriers[0])),
                         "=r"(*reinterpret_cast<uint32_t*>(&x_carriers[(0) + 1])),
                         "=r"(*reinterpret_cast<uint32_t*>(&x_carriers[(0) + 2])),
                         "=r"(*reinterpret_cast<uint32_t*>(&x_carriers[(0) + 3]))
                       : "r"(x_smem_addr + (unsigned int)(x_thread_base * 2)));
          {
#pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
              asm volatile(
                  "{\n\t"
                  "shl.b32 %0, %2, 16;\n\t"
                  "and.b32 %1, %2, 0xffff0000;\n\t"
                  "}\n"
                  : "=f"((&x_values[_pair * 2])[0]), "=f"((&x_values[_pair * 2])[1])
                  : "r"(x_carriers[_pair]));
            }
          }
        }
      }
#pragma unroll
      for (int rr_2 = 0; rr_2 < 8; rr_2++) {
        if (valid[pp_5] != 0) {
          if (k_base_1 < hidden) {
            int w_thread_base = stage * 8 * 1024 + pp_5 * 8 * 1024 + rr_2 * 1024 + tid * 8;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(*reinterpret_cast<uint32_t*>(&w_carriers[0])),
                           "=r"(*reinterpret_cast<uint32_t*>(&w_carriers[(0) + 1])),
                           "=r"(*reinterpret_cast<uint32_t*>(&w_carriers[(0) + 2])),
                           "=r"(*reinterpret_cast<uint32_t*>(&w_carriers[(0) + 3]))
                         : "r"(w_smem_addr + (unsigned int)(w_thread_base * 2)));
            {
#pragma unroll
              for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&w_values[_pair * 2])[0]), "=f"((&w_values[_pair * 2])[1])
                    : "r"(w_carriers[_pair]));
              }
            }
#pragma unroll
            for (int element = 0; element < 8; element++) {
              float _fma_0 = __fmaf_rn(x_values[element], w_values[element], acc[pp_5 * 8 + rr_2]);
              acc[pp_5 * 8 + rr_2] = _fma_0;
            }
          }
        }
      }
    }
  }
#pragma unroll
  for (int owner_1 = 0; owner_1 < 8; owner_1++) {
    float _warp_reduce_0 = acc[owner_1];
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
    float warp_sum = _warp_reduce_0;
    if (lane == 0) {
      warp_partials[owner_1 * 4 + warp] = warp_sum;
    }
  }
  __syncthreads();
  float owned_accum = 0.0f;
  int owner_pp = lane / 8;
  int owner_rr = lane % 8;
  int owner_pair = pair_block + owner_pp;
  if (warp == 0) {
    if (lane < 8) {
#pragma unroll
      for (int source_warp = 0; source_warp < 4; source_warp++) {
        owned_accum += warp_partials[lane * 4 + source_warp];
      }
    }
  }
  if (num_splits == 1) {
    if (warp == 0) {
      if (lane < 8) {
        if (owner_pair < num_pairs) {
          *(reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<__nv_bfloat16*>(shrink_out_raw) +
                                             (owner_pair * 16 + rank_base + owner_rr)) +
            (0)) = __float2bfloat16_rn(owned_accum);
        }
      }
    }
  } else {
    if (warp == 0) {
      if (lane < 8) {
        if (owner_pair < num_pairs) {
          *(reinterpret_cast<float*>(
                reinterpret_cast<float*>(split_partials_raw) +
                ((split * num_pairs + owner_pair) * 16 + rank_base + owner_rr)) +
            (0)) = owned_accum;
        }
      }
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
      unsigned int _atomic_old_1;
      asm volatile(
          "atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
          : "=r"(_atomic_old_1)
          : "l"(&reinterpret_cast<unsigned int*>(split_counters_raw)[pair_block * 2 + rank_block]),
            "r"(static_cast<uint32_t>(1))
          : "memory");
      unsigned int ticket = _atomic_old_1;
      int last_arrival = 0;
      if (ticket == (unsigned int)(num_splits - 1)) {
        last_arrival = 1;
        *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_counters_raw) +
                                          (pair_block * 2 + rank_block)) +
          (0)) = 0;
      }
      split_flag[0] = last_arrival;
    }
    __syncthreads();
    if (split_flag[0] != 0) {
      __threadfence();
      if (warp == 0) {
        if (lane < 8) {
          if (owner_pair < num_pairs) {
            float split_total = 0.0f;
#pragma unroll 1
            for (int source_split = 0; source_split < num_splits; source_split++) {
              split_total += reinterpret_cast<float*>(
                  split_partials_raw)[(source_split * num_pairs + owner_pair) * 16 + rank_base +
                                      owner_rr];
            }
            *(reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<__nv_bfloat16*>(shrink_out_raw) +
                                               (owner_pair * 16 + rank_base + owner_rr)) +
              (0)) = __float2bfloat16_rn(split_total);
          }
        }
      }
    }
  }
  if (route_build != 0) {
    if (blockIdx.y + blockIdx.z == 0) {
      if (tid == 0) {
        int route_parity = (int)(route_launch & 1);
        if (blockIdx.x == 0) {
          *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(route_index_raw) + 1) +
            (0)) = (unsigned int)route_parity;
        }
#pragma unroll
        for (int pp_6 = 0; pp_6 < 1; pp_6++) {
          if (route_publish[pp_6] != 0) {
            int publish_token_1 = (int)tokens[pp_6];
            unsigned int route_base = route_base_even[pp_6];
            if (route_parity != 0) {
              route_base = route_base_odd[pp_6];
            }
            unsigned int route_slot = route_old[pp_6] - route_base;
            if (route_slot < 16) {
              *(reinterpret_cast<unsigned int*>(
                    reinterpret_cast<unsigned int*>(route_index_raw) +
                    (4 + 3 * num_tokens + publish_token_1 * 16 + (int)route_slot)) +
                (0)) = (unsigned int)(pair_block + pp_6);
            }
          }
        }
      }
    }
  }
}

}  // extern "C"

#undef BLACKWELL_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SPLIT_FLAG_OFF
#undef SMEM_SPLIT_FLAG_STAGE_BYTES
#undef SMEM_SPLIT_FLAG_STRIDE
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

#define BLACKWELL_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_ROUTE_LIST_OFF 0
#define SMEM_ROUTE_LIST_STAGE_BYTES 72
#define SMEM_ROUTE_LIST_STRIDE 72
#define SMEM_TOTAL 128
#define THREADS 64

extern "C" {

__global__
__launch_bounds__(64, 1) void kernel_flashinfer_bgmv_moe_expand_generic_token_t64_bf16_r16(
    float* __restrict__ y_accum, uint16_t* __restrict__ shrink_raw,
    uint16_t* __restrict__ lora_b_raw, long long* __restrict__ sorted_token_ids,
    long long* __restrict__ expert_ids, long long* __restrict__ lora_indices,
    float* __restrict__ topk_weights, int num_pairs, int num_experts, int num_tokens,
    int output_stride, int output_offset, unsigned int* __restrict__ route_index_raw,
    int route_lookup, int route_advance, int hidden) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  const int cta_rank = 0;

  // Kernel setup ops
  int* route_list = reinterpret_cast<int*>(smem_raw + 0);
  const int route_list_addr = smem + 0;

  // === Task calls (dependency order) ===
  int token = blockIdx.x;
  int warp_col_base = blockIdx.y * 64 + warp * 32;
  int rank_lane = lane % 2;
  int col_lane = lane / 2;
  int rank_col = rank_lane * 8;
  int slot_col[2];
  float slot_acc[2];
#pragma unroll
  for (int slot = 0; slot < 2; slot++) {
    slot_col[slot] = warp_col_base + slot * 16 + col_lane;
    slot_acc[slot] = 0.0f;
  }
  if (token < num_tokens) {
    int advance_parity = 0;
    unsigned int advance_count = 0;
    unsigned int advance_launch = 0;
    if (route_advance != 0) {
      if (blockIdx.y == 0) {
        if (tid == 0) {
          advance_parity = (int)route_index_raw[1];
          advance_count = route_index_raw[4 + token];
          advance_launch = route_index_raw[0];
        }
      }
    }
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
#pragma unroll
        for (int route = 0; route < 2; route++) {
          int direct_pair = pair_base + route;
          long long expert = expert_ids[direct_pair];
          float pair_weight = topk_weights[direct_pair];
          float _vec_load_0[8];
          {
            const uint4* _vptr_0 =
                reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(shrink_raw) +
                                               (direct_pair * 16 + rank_col) + 0);
            uint4 _vld_0[1];
#pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
              _vld_0[_blk] = _vptr_0[_blk];
              uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
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
          long long row_base = (lora_id * (long long)num_experts + expert) * (long long)hidden;
#pragma unroll
          for (int slot_1 = 0; slot_1 < 2; slot_1++) {
            if (slot_col[slot_1] < hidden) {
              float _vec_load_1[8];
              {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(
                    reinterpret_cast<const __nv_bfloat16*>(lora_b_raw) +
                    ((row_base + (long long)slot_col[slot_1]) * 16 + (long long)rank_col) + 0);
                uint4 _vld_1[1];
#pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                  _vld_1[_blk] = _vptr_1[_blk];
                  uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
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
              float partial = 0.0f;
#pragma unroll
              for (int element = 0; element < 8; element++) {
                float _fma_0 = __fmaf_rn(_vec_load_0[element], _vec_load_1[element], partial);
                partial = _fma_0;
              }
              float _fma_1 = __fmaf_rn(partial, pair_weight, slot_acc[slot_1]);
              slot_acc[slot_1] = _fma_1;
            }
          }
        }
      } else {
        int route_count = num_pairs;
        int use_index = 0;
        if (route_lookup != 0) {
          int indexed_parity = (int)route_index_raw[1];
          int indexed_count =
              (int)(route_index_raw[4 + token] -
                    route_index_raw[4 + num_tokens + indexed_parity * num_tokens + token]);
          if (indexed_count <= 16) {
            use_index = 1;
            int direct_first = -1;
            int direct_second = -1;
            int direct_count = 0;
            if (num_pairs == num_tokens * 2) {
              if (sorted_token_ids[pair_base] == (long long)token) {
                direct_first = pair_base;
                direct_count = 1;
              }
              if (sorted_token_ids[pair_base + 1] == (long long)token) {
                if (direct_count == 0) {
                  direct_first = pair_base + 1;
                } else {
                  direct_second = pair_base + 1;
                }
                direct_count += 1;
              }
            }
            route_count = indexed_count + direct_count;
            int table_base = 4 + 3 * num_tokens + token * 16;
            if (route_count > tid) {
              int own_pair = direct_first;
              if (indexed_count > tid) {
                own_pair = (int)route_index_raw[table_base + tid];
              } else if (tid == indexed_count + 1) {
                own_pair = direct_second;
              }
              int own_order = 0;
#pragma unroll 1
              for (int probe = 0; probe < indexed_count; probe++) {
                if (own_pair > (int)route_index_raw[table_base + probe]) {
                  own_order += 1;
                }
              }
              if (direct_count > 0) {
                if (direct_first < own_pair) {
                  own_order += 1;
                }
              }
              if (direct_count > 1) {
                if (direct_second < own_pair) {
                  own_order += 1;
                }
              }
              route_list[own_order] = own_pair;
            }
            __syncthreads();
          }
        }
#pragma unroll 1
        for (int route_step = 0; route_step < route_count; route_step++) {
          int pair = route_step;
          int route_match = 1;
          if (use_index != 0) {
            pair = route_list[route_step];
          } else if (sorted_token_ids[route_step] != (long long)token) {
            route_match = 0;
          }
          if (route_match != 0) {
            long long expert_1 = expert_ids[pair];
            float pair_weight_1 = topk_weights[pair];
            float _vec_load_2[8];
            {
              const uint4* _vptr_2 = reinterpret_cast<const uint4*>(
                  reinterpret_cast<const __nv_bfloat16*>(shrink_raw) + (pair * 16 + rank_col) + 0);
              uint4 _vld_2[1];
#pragma unroll
              for (int _blk = 0; _blk < 1; _blk++) {
                _vld_2[_blk] = _vptr_2[_blk];
                uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
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
            long long row_base_1 =
                (lora_id * (long long)num_experts + expert_1) * (long long)hidden;
#pragma unroll
            for (int slot_2 = 0; slot_2 < 2; slot_2++) {
              if (slot_col[slot_2] < hidden) {
                float _vec_load_3[8];
                {
                  const uint4* _vptr_3 = reinterpret_cast<const uint4*>(
                      reinterpret_cast<const __nv_bfloat16*>(lora_b_raw) +
                      ((row_base_1 + (long long)slot_col[slot_2]) * 16 + (long long)rank_col) + 0);
                  uint4 _vld_3[1];
#pragma unroll
                  for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_3[_blk] = _vptr_3[_blk];
                    uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
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
                float partial_1 = 0.0f;
#pragma unroll
                for (int element_1 = 0; element_1 < 8; element_1++) {
                  float _fma_2 =
                      __fmaf_rn(_vec_load_2[element_1], _vec_load_3[element_1], partial_1);
                  partial_1 = _fma_2;
                }
                float _fma_3 = __fmaf_rn(partial_1, pair_weight_1, slot_acc[slot_2]);
                slot_acc[slot_2] = _fma_3;
              }
            }
          }
        }
      }
#pragma unroll
      for (int slot_3 = 0; slot_3 < 2; slot_3++) {
        float column_sum = slot_acc[slot_3];
#pragma unroll
        for (int step = 0; step < 1; step++) {
          float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, column_sum, 2 >> step + 1);
          column_sum += _shfl_xor_0;
        }
        if (rank_lane == 0) {
          if (slot_col[slot_3] < hidden) {
            *(reinterpret_cast<float*>(y_accum +
                                       (token * output_stride + output_offset + slot_col[slot_3])) +
              (0)) = column_sum;
          }
        }
      }
    } else {
      int zero_col = warp_col_base + lane;
      if (zero_col < hidden) {
        *(reinterpret_cast<float*>(y_accum + (token * output_stride + output_offset + zero_col)) +
          (0)) = 0.0f;
      }
    }
    if (route_advance != 0) {
      if (blockIdx.y == 0) {
        if (tid == 0) {
          *(reinterpret_cast<unsigned int*>(
                reinterpret_cast<unsigned int*>(route_index_raw) +
                (4 + num_tokens + (1 - advance_parity) * num_tokens + token)) +
            (0)) = advance_count;
          if (token == 0) {
            *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(route_index_raw)) +
              (0)) = advance_launch + 1;
          }
        }
      }
    }
  }
}

}  // extern "C"

#undef BLACKWELL_INF
#undef NUM_MAIN_STAGES
#undef SMEM_ROUTE_LIST_OFF
#undef SMEM_ROUTE_LIST_STAGE_BYTES
#undef SMEM_ROUTE_LIST_STRIDE
#undef SMEM_TOTAL
#undef THREADS

#define BLACKWELL_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_ROUTE_LIST_OFF 0
#define SMEM_ROUTE_LIST_STAGE_BYTES 72
#define SMEM_ROUTE_LIST_STRIDE 72
#define SMEM_TOTAL 128
#define THREADS 128

extern "C" {

__global__
__launch_bounds__(128, 1) void kernel_flashinfer_bgmv_moe_expand_generic_token_t128_bf16_r16(
    float* __restrict__ y_accum, uint16_t* __restrict__ shrink_raw,
    uint16_t* __restrict__ lora_b_raw, long long* __restrict__ sorted_token_ids,
    long long* __restrict__ expert_ids, long long* __restrict__ lora_indices,
    float* __restrict__ topk_weights, int num_pairs, int num_experts, int num_tokens,
    int output_stride, int output_offset, unsigned int* __restrict__ route_index_raw,
    int route_lookup, int route_advance, int hidden) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  const int cta_rank = 0;

  // Kernel setup ops
  int* route_list = reinterpret_cast<int*>(smem_raw + 0);
  const int route_list_addr = smem + 0;

  // === Task calls (dependency order) ===
  int token = blockIdx.x;
  int warp_col_base = blockIdx.y * 128 + warp * 32;
  int rank_lane = lane % 2;
  int col_lane = lane / 2;
  int rank_col = rank_lane * 8;
  int slot_col[2];
  float slot_acc[2];
#pragma unroll
  for (int slot = 0; slot < 2; slot++) {
    slot_col[slot] = warp_col_base + slot * 16 + col_lane;
    slot_acc[slot] = 0.0f;
  }
  if (token < num_tokens) {
    int advance_parity = 0;
    unsigned int advance_count = 0;
    unsigned int advance_launch = 0;
    if (route_advance != 0) {
      if (blockIdx.y == 0) {
        if (tid == 0) {
          advance_parity = (int)route_index_raw[1];
          advance_count = route_index_raw[4 + token];
          advance_launch = route_index_raw[0];
        }
      }
    }
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
#pragma unroll
        for (int route = 0; route < 2; route++) {
          int direct_pair = pair_base + route;
          long long expert = expert_ids[direct_pair];
          float pair_weight = topk_weights[direct_pair];
          float _vec_load_0[8];
          {
            const uint4* _vptr_0 =
                reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(shrink_raw) +
                                               (direct_pair * 16 + rank_col) + 0);
            uint4 _vld_0[1];
#pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
              _vld_0[_blk] = _vptr_0[_blk];
              uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
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
          long long row_base = (lora_id * (long long)num_experts + expert) * (long long)hidden;
#pragma unroll
          for (int slot_1 = 0; slot_1 < 2; slot_1++) {
            if (slot_col[slot_1] < hidden) {
              float _vec_load_1[8];
              {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(
                    reinterpret_cast<const __nv_bfloat16*>(lora_b_raw) +
                    ((row_base + (long long)slot_col[slot_1]) * 16 + (long long)rank_col) + 0);
                uint4 _vld_1[1];
#pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                  _vld_1[_blk] = _vptr_1[_blk];
                  uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
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
              float partial = 0.0f;
#pragma unroll
              for (int element = 0; element < 8; element++) {
                float _fma_0 = __fmaf_rn(_vec_load_0[element], _vec_load_1[element], partial);
                partial = _fma_0;
              }
              float _fma_1 = __fmaf_rn(partial, pair_weight, slot_acc[slot_1]);
              slot_acc[slot_1] = _fma_1;
            }
          }
        }
      } else {
        int route_count = num_pairs;
        int use_index = 0;
        if (route_lookup != 0) {
          int indexed_parity = (int)route_index_raw[1];
          int indexed_count =
              (int)(route_index_raw[4 + token] -
                    route_index_raw[4 + num_tokens + indexed_parity * num_tokens + token]);
          if (indexed_count <= 16) {
            use_index = 1;
            int direct_first = -1;
            int direct_second = -1;
            int direct_count = 0;
            if (num_pairs == num_tokens * 2) {
              if (sorted_token_ids[pair_base] == (long long)token) {
                direct_first = pair_base;
                direct_count = 1;
              }
              if (sorted_token_ids[pair_base + 1] == (long long)token) {
                if (direct_count == 0) {
                  direct_first = pair_base + 1;
                } else {
                  direct_second = pair_base + 1;
                }
                direct_count += 1;
              }
            }
            route_count = indexed_count + direct_count;
            int table_base = 4 + 3 * num_tokens + token * 16;
            if (route_count > tid) {
              int own_pair = direct_first;
              if (indexed_count > tid) {
                own_pair = (int)route_index_raw[table_base + tid];
              } else if (tid == indexed_count + 1) {
                own_pair = direct_second;
              }
              int own_order = 0;
#pragma unroll 1
              for (int probe = 0; probe < indexed_count; probe++) {
                if (own_pair > (int)route_index_raw[table_base + probe]) {
                  own_order += 1;
                }
              }
              if (direct_count > 0) {
                if (direct_first < own_pair) {
                  own_order += 1;
                }
              }
              if (direct_count > 1) {
                if (direct_second < own_pair) {
                  own_order += 1;
                }
              }
              route_list[own_order] = own_pair;
            }
            __syncthreads();
          }
        }
#pragma unroll 1
        for (int route_step = 0; route_step < route_count; route_step++) {
          int pair = route_step;
          int route_match = 1;
          if (use_index != 0) {
            pair = route_list[route_step];
          } else if (sorted_token_ids[route_step] != (long long)token) {
            route_match = 0;
          }
          if (route_match != 0) {
            long long expert_1 = expert_ids[pair];
            float pair_weight_1 = topk_weights[pair];
            float _vec_load_2[8];
            {
              const uint4* _vptr_2 = reinterpret_cast<const uint4*>(
                  reinterpret_cast<const __nv_bfloat16*>(shrink_raw) + (pair * 16 + rank_col) + 0);
              uint4 _vld_2[1];
#pragma unroll
              for (int _blk = 0; _blk < 1; _blk++) {
                _vld_2[_blk] = _vptr_2[_blk];
                uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
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
            long long row_base_1 =
                (lora_id * (long long)num_experts + expert_1) * (long long)hidden;
#pragma unroll
            for (int slot_2 = 0; slot_2 < 2; slot_2++) {
              if (slot_col[slot_2] < hidden) {
                float _vec_load_3[8];
                {
                  const uint4* _vptr_3 = reinterpret_cast<const uint4*>(
                      reinterpret_cast<const __nv_bfloat16*>(lora_b_raw) +
                      ((row_base_1 + (long long)slot_col[slot_2]) * 16 + (long long)rank_col) + 0);
                  uint4 _vld_3[1];
#pragma unroll
                  for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_3[_blk] = _vptr_3[_blk];
                    uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
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
                float partial_1 = 0.0f;
#pragma unroll
                for (int element_1 = 0; element_1 < 8; element_1++) {
                  float _fma_2 =
                      __fmaf_rn(_vec_load_2[element_1], _vec_load_3[element_1], partial_1);
                  partial_1 = _fma_2;
                }
                float _fma_3 = __fmaf_rn(partial_1, pair_weight_1, slot_acc[slot_2]);
                slot_acc[slot_2] = _fma_3;
              }
            }
          }
        }
      }
#pragma unroll
      for (int slot_3 = 0; slot_3 < 2; slot_3++) {
        float column_sum = slot_acc[slot_3];
#pragma unroll
        for (int step = 0; step < 1; step++) {
          float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, column_sum, 2 >> step + 1);
          column_sum += _shfl_xor_0;
        }
        if (rank_lane == 0) {
          if (slot_col[slot_3] < hidden) {
            *(reinterpret_cast<float*>(y_accum +
                                       (token * output_stride + output_offset + slot_col[slot_3])) +
              (0)) = column_sum;
          }
        }
      }
    } else {
      int zero_col = warp_col_base + lane;
      if (zero_col < hidden) {
        *(reinterpret_cast<float*>(y_accum + (token * output_stride + output_offset + zero_col)) +
          (0)) = 0.0f;
      }
    }
    if (route_advance != 0) {
      if (blockIdx.y == 0) {
        if (tid == 0) {
          *(reinterpret_cast<unsigned int*>(
                reinterpret_cast<unsigned int*>(route_index_raw) +
                (4 + num_tokens + (1 - advance_parity) * num_tokens + token)) +
            (0)) = advance_count;
          if (token == 0) {
            *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(route_index_raw)) +
              (0)) = advance_launch + 1;
          }
        }
      }
    }
  }
}

}  // extern "C"

#undef BLACKWELL_INF
#undef NUM_MAIN_STAGES
#undef SMEM_ROUTE_LIST_OFF
#undef SMEM_ROUTE_LIST_STAGE_BYTES
#undef SMEM_ROUTE_LIST_STRIDE
#undef SMEM_TOTAL
#undef THREADS

// Dynamic shared memory per launch, in bytes.
#define CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_DECODE 221824
#define CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_PREFILL 37120
#define CAKE_BGMV_MOE_GENERIC_SMEM_EXPAND_T64 128
#define CAKE_BGMV_MOE_GENERIC_SMEM_EXPAND_T128 128
