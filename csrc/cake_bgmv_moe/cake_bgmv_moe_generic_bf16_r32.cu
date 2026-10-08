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
// Bundle: Cake BGMV MoE generic-shape shrink, deterministic expand and pair-grouped pipeline, bf16
// rank 32. Target: sm_90a, sm_100a, sm_103a (cp.async, shuffles, atomics, FMA only); compile flags:
// none. Generated file; do not edit manually.
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

__global__ __launch_bounds__(128, 1) void kernel_flashinfer_bgmv_moe_shrink_generic_bf16_r32_p4_s3(
    uint16_t* __restrict__ shrink_out_raw, uint16_t* __restrict__ x_raw,
    uint16_t* __restrict__ lora_a_raw, long long* __restrict__ sorted_token_ids,
    long long* __restrict__ expert_ids, long long* __restrict__ lora_indices, int num_pairs,
    int num_experts, int num_tokens, unsigned int* __restrict__ route_index_raw, int route_build,
    int hidden, int num_tiles, float* __restrict__ split_partials_raw,
    unsigned int* __restrict__ split_counters_raw, int num_splits, int pdl_early,
    unsigned int* __restrict__ order_raw, int off_route_order, int route_remap) {
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
                  ((loras[pp_1] * (long long)num_experts + experts[pp_1]) * 32 +
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
                  ((loras[pp_4] * (long long)num_experts + experts[pp_4]) * 32 +
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
                                             (owner_pair * 32 + rank_base + owner_rr)) +
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
                ((split * num_pairs + owner_pair) * 32 + rank_base + owner_rr)) +
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
          : "l"(&reinterpret_cast<unsigned int*>(split_counters_raw)[pair_block * 4 + rank_block]),
            "r"(static_cast<uint32_t>(1))
          : "memory");
      unsigned int ticket = _atomic_old_1;
      int last_arrival = 0;
      if (ticket == (unsigned int)(num_splits - 1)) {
        last_arrival = 1;
        *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_counters_raw) +
                                          (pair_block * 4 + rank_block)) +
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
                  split_partials_raw)[(source_split * num_pairs + owner_pair) * 32 + rank_base +
                                      owner_rr];
            }
            *(reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<__nv_bfloat16*>(shrink_out_raw) +
                                               (owner_pair * 32 + rank_base + owner_rr)) +
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

__global__ __launch_bounds__(128, 1) void kernel_flashinfer_bgmv_moe_shrink_generic_bf16_r32_p1_s2(
    uint16_t* __restrict__ shrink_out_raw, uint16_t* __restrict__ x_raw,
    uint16_t* __restrict__ lora_a_raw, long long* __restrict__ sorted_token_ids,
    long long* __restrict__ expert_ids, long long* __restrict__ lora_indices, int num_pairs,
    int num_experts, int num_tokens, unsigned int* __restrict__ route_index_raw, int route_build,
    int hidden, int num_tiles, float* __restrict__ split_partials_raw,
    unsigned int* __restrict__ split_counters_raw, int num_splits, int pdl_early,
    unsigned int* __restrict__ order_raw, int off_route_order, int route_remap) {
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
                  ((loras[pp_1] * (long long)num_experts + experts[pp_1]) * 32 +
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
                  ((loras[pp_4] * (long long)num_experts + experts[pp_4]) * 32 +
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
                                             (owner_pair * 32 + rank_base + owner_rr)) +
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
                ((split * num_pairs + owner_pair) * 32 + rank_base + owner_rr)) +
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
          : "l"(&reinterpret_cast<unsigned int*>(split_counters_raw)[pair_block * 4 + rank_block]),
            "r"(static_cast<uint32_t>(1))
          : "memory");
      unsigned int ticket = _atomic_old_1;
      int last_arrival = 0;
      if (ticket == (unsigned int)(num_splits - 1)) {
        last_arrival = 1;
        *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_counters_raw) +
                                          (pair_block * 4 + rank_block)) +
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
                  split_partials_raw)[(source_split * num_pairs + owner_pair) * 32 + rank_base +
                                      owner_rr];
            }
            *(reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<__nv_bfloat16*>(shrink_out_raw) +
                                               (owner_pair * 32 + rank_base + owner_rr)) +
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

__global__
__launch_bounds__(128, 1) void kernel_flashinfer_bgmv_moe_shrink_generic_bf16_r32_p4_s3_pdl(
    uint16_t* __restrict__ shrink_out_raw, uint16_t* __restrict__ x_raw,
    uint16_t* __restrict__ lora_a_raw, long long* __restrict__ sorted_token_ids,
    long long* __restrict__ expert_ids, long long* __restrict__ lora_indices, int num_pairs,
    int num_experts, int num_tokens, unsigned int* __restrict__ route_index_raw, int route_build,
    int hidden, int num_tiles, float* __restrict__ split_partials_raw,
    unsigned int* __restrict__ split_counters_raw, int num_splits, int pdl_early,
    unsigned int* __restrict__ order_raw, int off_route_order, int route_remap) {
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
  if (pdl_early != 0) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
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
                  ((loras[pp_1] * (long long)num_experts + experts[pp_1]) * 32 +
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
                  ((loras[pp_4] * (long long)num_experts + experts[pp_4]) * 32 +
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
  if (pdl_early == 0) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
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
                                             (owner_pair * 32 + rank_base + owner_rr)) +
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
                ((split * num_pairs + owner_pair) * 32 + rank_base + owner_rr)) +
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
          : "l"(&reinterpret_cast<unsigned int*>(split_counters_raw)[pair_block * 4 + rank_block]),
            "r"(static_cast<uint32_t>(1))
          : "memory");
      unsigned int ticket = _atomic_old_1;
      int last_arrival = 0;
      if (ticket == (unsigned int)(num_splits - 1)) {
        last_arrival = 1;
        *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_counters_raw) +
                                          (pair_block * 4 + rank_block)) +
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
                  split_partials_raw)[(source_split * num_pairs + owner_pair) * 32 + rank_base +
                                      owner_rr];
            }
            *(reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<__nv_bfloat16*>(shrink_out_raw) +
                                               (owner_pair * 32 + rank_base + owner_rr)) +
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

__global__
__launch_bounds__(128, 1) void kernel_flashinfer_bgmv_moe_shrink_generic_bf16_r32_p1_s2_pdl(
    uint16_t* __restrict__ shrink_out_raw, uint16_t* __restrict__ x_raw,
    uint16_t* __restrict__ lora_a_raw, long long* __restrict__ sorted_token_ids,
    long long* __restrict__ expert_ids, long long* __restrict__ lora_indices, int num_pairs,
    int num_experts, int num_tokens, unsigned int* __restrict__ route_index_raw, int route_build,
    int hidden, int num_tiles, float* __restrict__ split_partials_raw,
    unsigned int* __restrict__ split_counters_raw, int num_splits, int pdl_early,
    unsigned int* __restrict__ order_raw, int off_route_order, int route_remap) {
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
  if (pdl_early != 0) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
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
                  ((loras[pp_1] * (long long)num_experts + experts[pp_1]) * 32 +
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
                  ((loras[pp_4] * (long long)num_experts + experts[pp_4]) * 32 +
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
  if (pdl_early == 0) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
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
                                             (owner_pair * 32 + rank_base + owner_rr)) +
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
                ((split * num_pairs + owner_pair) * 32 + rank_base + owner_rr)) +
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
          : "l"(&reinterpret_cast<unsigned int*>(split_counters_raw)[pair_block * 4 + rank_block]),
            "r"(static_cast<uint32_t>(1))
          : "memory");
      unsigned int ticket = _atomic_old_1;
      int last_arrival = 0;
      if (ticket == (unsigned int)(num_splits - 1)) {
        last_arrival = 1;
        *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_counters_raw) +
                                          (pair_block * 4 + rank_block)) +
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
                  split_partials_raw)[(source_split * num_pairs + owner_pair) * 32 + rank_base +
                                      owner_rr];
            }
            *(reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<__nv_bfloat16*>(shrink_out_raw) +
                                               (owner_pair * 32 + rank_base + owner_rr)) +
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
#define SMEM_X_SMEM_OFF 0
#define SMEM_X_SMEM_STAGE_BYTES 6144
#define SMEM_X_SMEM_STRIDE 6144
#define SMEM_W_SMEM_OFF 6144
#define SMEM_W_SMEM_STAGE_BYTES 49152
#define SMEM_W_SMEM_STRIDE 49152
#define SMEM_WARP_PARTIALS_OFF 55296
#define SMEM_WARP_PARTIALS_STAGE_BYTES 128
#define SMEM_WARP_PARTIALS_STRIDE 128
#define SMEM_SPLIT_FLAG_OFF 55424
#define SMEM_SPLIT_FLAG_STAGE_BYTES 16
#define SMEM_SPLIT_FLAG_STRIDE 16
#define SMEM_TOTAL 55552
#define THREADS 128

extern "C" {

__global__ __launch_bounds__(128, 1) void kernel_flashinfer_bgmv_moe_shrink_generic_bf16_r32_p1_s3(
    uint16_t* __restrict__ shrink_out_raw, uint16_t* __restrict__ x_raw,
    uint16_t* __restrict__ lora_a_raw, long long* __restrict__ sorted_token_ids,
    long long* __restrict__ expert_ids, long long* __restrict__ lora_indices, int num_pairs,
    int num_experts, int num_tokens, unsigned int* __restrict__ route_index_raw, int route_build,
    int hidden, int num_tiles, float* __restrict__ split_partials_raw,
    unsigned int* __restrict__ split_counters_raw, int num_splits, int pdl_early,
    unsigned int* __restrict__ order_raw, int off_route_order, int route_remap) {
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
  __nv_bfloat16* w_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
  const int w_smem_addr = smem + 6144;
  float* warp_partials = reinterpret_cast<float*>(smem_raw + 55296);
  const int warp_partials_addr = smem + 55296;
  int* split_flag = reinterpret_cast<int*>(smem_raw + 55424);
  const int split_flag_addr = smem + 55424;

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
  for (int tile = 0; tile < 2; tile++) {
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
                  ((loras[pp_1] * (long long)num_experts + experts[pp_1]) * 32 +
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
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();
    int stage = local % 3;
    int refill_local = local + 3 - 1;
    if (refill_local < local_tiles) {
      int refill_stage = refill_local % 3;
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
                  ((loras[pp_4] * (long long)num_experts + experts[pp_4]) * 32 +
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
                                             (owner_pair * 32 + rank_base + owner_rr)) +
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
                ((split * num_pairs + owner_pair) * 32 + rank_base + owner_rr)) +
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
          : "l"(&reinterpret_cast<unsigned int*>(split_counters_raw)[pair_block * 4 + rank_block]),
            "r"(static_cast<uint32_t>(1))
          : "memory");
      unsigned int ticket = _atomic_old_1;
      int last_arrival = 0;
      if (ticket == (unsigned int)(num_splits - 1)) {
        last_arrival = 1;
        *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_counters_raw) +
                                          (pair_block * 4 + rank_block)) +
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
                  split_partials_raw)[(source_split * num_pairs + owner_pair) * 32 + rank_base +
                                      owner_rr];
            }
            *(reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<__nv_bfloat16*>(shrink_out_raw) +
                                               (owner_pair * 32 + rank_base + owner_rr)) +
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
#define SMEM_X_SMEM_OFF 0
#define SMEM_X_SMEM_STAGE_BYTES 6144
#define SMEM_X_SMEM_STRIDE 6144
#define SMEM_W_SMEM_OFF 6144
#define SMEM_W_SMEM_STAGE_BYTES 49152
#define SMEM_W_SMEM_STRIDE 49152
#define SMEM_WARP_PARTIALS_OFF 55296
#define SMEM_WARP_PARTIALS_STAGE_BYTES 128
#define SMEM_WARP_PARTIALS_STRIDE 128
#define SMEM_SPLIT_FLAG_OFF 55424
#define SMEM_SPLIT_FLAG_STAGE_BYTES 16
#define SMEM_SPLIT_FLAG_STRIDE 16
#define SMEM_TOTAL 55552
#define THREADS 128

extern "C" {

__global__
__launch_bounds__(128, 1) void kernel_flashinfer_bgmv_moe_shrink_generic_bf16_r32_p1_s3_pdl(
    uint16_t* __restrict__ shrink_out_raw, uint16_t* __restrict__ x_raw,
    uint16_t* __restrict__ lora_a_raw, long long* __restrict__ sorted_token_ids,
    long long* __restrict__ expert_ids, long long* __restrict__ lora_indices, int num_pairs,
    int num_experts, int num_tokens, unsigned int* __restrict__ route_index_raw, int route_build,
    int hidden, int num_tiles, float* __restrict__ split_partials_raw,
    unsigned int* __restrict__ split_counters_raw, int num_splits, int pdl_early,
    unsigned int* __restrict__ order_raw, int off_route_order, int route_remap) {
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
  __nv_bfloat16* w_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
  const int w_smem_addr = smem + 6144;
  float* warp_partials = reinterpret_cast<float*>(smem_raw + 55296);
  const int warp_partials_addr = smem + 55296;
  int* split_flag = reinterpret_cast<int*>(smem_raw + 55424);
  const int split_flag_addr = smem + 55424;

  // === Task calls (dependency order) ===
  if (pdl_early != 0) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
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
  for (int tile = 0; tile < 2; tile++) {
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
                  ((loras[pp_1] * (long long)num_experts + experts[pp_1]) * 32 +
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
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();
    int stage = local % 3;
    int refill_local = local + 3 - 1;
    if (refill_local < local_tiles) {
      int refill_stage = refill_local % 3;
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
                  ((loras[pp_4] * (long long)num_experts + experts[pp_4]) * 32 +
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
  if (pdl_early == 0) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
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
                                             (owner_pair * 32 + rank_base + owner_rr)) +
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
                ((split * num_pairs + owner_pair) * 32 + rank_base + owner_rr)) +
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
          : "l"(&reinterpret_cast<unsigned int*>(split_counters_raw)[pair_block * 4 + rank_block]),
            "r"(static_cast<uint32_t>(1))
          : "memory");
      unsigned int ticket = _atomic_old_1;
      int last_arrival = 0;
      if (ticket == (unsigned int)(num_splits - 1)) {
        last_arrival = 1;
        *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_counters_raw) +
                                          (pair_block * 4 + rank_block)) +
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
                  split_partials_raw)[(source_split * num_pairs + owner_pair) * 32 + rank_base +
                                      owner_rr];
            }
            *(reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<__nv_bfloat16*>(shrink_out_raw) +
                                               (owner_pair * 32 + rank_base + owner_rr)) +
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

__global__
__launch_bounds__(128, 1) void kernel_flashinfer_bgmv_moe_shrink_generic_bf16_r32_p1_s2_remap(
    uint16_t* __restrict__ shrink_out_raw, uint16_t* __restrict__ x_raw,
    uint16_t* __restrict__ lora_a_raw, long long* __restrict__ sorted_token_ids,
    long long* __restrict__ expert_ids, long long* __restrict__ lora_indices, int num_pairs,
    int num_experts, int num_tokens, unsigned int* __restrict__ route_index_raw, int route_build,
    int hidden, int num_tiles, float* __restrict__ split_partials_raw,
    unsigned int* __restrict__ split_counters_raw, int num_splits, int pdl_early,
    unsigned int* __restrict__ order_raw, int off_route_order, int route_remap) {
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
  int pairs_r[1];
#pragma unroll
  for (int pp = 0; pp < 1; pp++) {
    int pair = pair_block + pp;
    if (pair < num_pairs) {
      pair = (int)order_raw[off_route_order + pair];
    }
    pairs_r[pp] = pair;
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
                  ((loras[pp_1] * (long long)num_experts + experts[pp_1]) * 32 +
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
              route_offset = pairs_r[pp_3] - route_token * 2;
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
                  ((loras[pp_4] * (long long)num_experts + experts[pp_4]) * 32 +
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
#pragma unroll
  for (int sel_pp = 0; sel_pp < 1; sel_pp++) {
    if (owner_pp == sel_pp) {
      owner_pair = pairs_r[sel_pp];
    }
  }
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
                                             (owner_pair * 32 + rank_base + owner_rr)) +
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
                ((split * num_pairs + owner_pair) * 32 + rank_base + owner_rr)) +
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
          : "l"(&reinterpret_cast<unsigned int*>(split_counters_raw)[pair_block * 4 + rank_block]),
            "r"(static_cast<uint32_t>(1))
          : "memory");
      unsigned int ticket = _atomic_old_1;
      int last_arrival = 0;
      if (ticket == (unsigned int)(num_splits - 1)) {
        last_arrival = 1;
        *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_counters_raw) +
                                          (pair_block * 4 + rank_block)) +
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
                  split_partials_raw)[(source_split * num_pairs + owner_pair) * 32 + rank_base +
                                      owner_rr];
            }
            *(reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<__nv_bfloat16*>(shrink_out_raw) +
                                               (owner_pair * 32 + rank_base + owner_rr)) +
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
                (0)) = (unsigned int)pairs_r[pp_6];
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

__global__
__launch_bounds__(128, 1) void kernel_flashinfer_bgmv_moe_shrink_generic_bf16_r32_p1_s2_remap_pdl(
    uint16_t* __restrict__ shrink_out_raw, uint16_t* __restrict__ x_raw,
    uint16_t* __restrict__ lora_a_raw, long long* __restrict__ sorted_token_ids,
    long long* __restrict__ expert_ids, long long* __restrict__ lora_indices, int num_pairs,
    int num_experts, int num_tokens, unsigned int* __restrict__ route_index_raw, int route_build,
    int hidden, int num_tiles, float* __restrict__ split_partials_raw,
    unsigned int* __restrict__ split_counters_raw, int num_splits, int pdl_early,
    unsigned int* __restrict__ order_raw, int off_route_order, int route_remap) {
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
  if (pdl_early != 0) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
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
  int pairs_r[1];
#pragma unroll
  for (int pp = 0; pp < 1; pp++) {
    int pair = pair_block + pp;
    if (pair < num_pairs) {
      pair = (int)order_raw[off_route_order + pair];
    }
    pairs_r[pp] = pair;
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
                  ((loras[pp_1] * (long long)num_experts + experts[pp_1]) * 32 +
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
              route_offset = pairs_r[pp_3] - route_token * 2;
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
                  ((loras[pp_4] * (long long)num_experts + experts[pp_4]) * 32 +
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
  if (pdl_early == 0) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
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
#pragma unroll
  for (int sel_pp = 0; sel_pp < 1; sel_pp++) {
    if (owner_pp == sel_pp) {
      owner_pair = pairs_r[sel_pp];
    }
  }
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
                                             (owner_pair * 32 + rank_base + owner_rr)) +
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
                ((split * num_pairs + owner_pair) * 32 + rank_base + owner_rr)) +
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
          : "l"(&reinterpret_cast<unsigned int*>(split_counters_raw)[pair_block * 4 + rank_block]),
            "r"(static_cast<uint32_t>(1))
          : "memory");
      unsigned int ticket = _atomic_old_1;
      int last_arrival = 0;
      if (ticket == (unsigned int)(num_splits - 1)) {
        last_arrival = 1;
        *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_counters_raw) +
                                          (pair_block * 4 + rank_block)) +
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
                  split_partials_raw)[(source_split * num_pairs + owner_pair) * 32 + rank_base +
                                      owner_rr];
            }
            *(reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<__nv_bfloat16*>(shrink_out_raw) +
                                               (owner_pair * 32 + rank_base + owner_rr)) +
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
                (0)) = (unsigned int)pairs_r[pp_6];
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
__launch_bounds__(64, 1) void kernel_flashinfer_bgmv_moe_expand_generic_token_t64_bf16_r32(
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
  int rank_lane = lane % 4;
  int col_lane = lane / 4;
  int rank_col = rank_lane * 8;
  int slot_col[4];
  float slot_acc[4];
#pragma unroll
  for (int slot = 0; slot < 4; slot++) {
    slot_col[slot] = warp_col_base + slot * 8 + col_lane;
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
                                               (direct_pair * 32 + rank_col) + 0);
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
          for (int slot_1 = 0; slot_1 < 4; slot_1++) {
            if (slot_col[slot_1] < hidden) {
              float _vec_load_1[8];
              {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(
                    reinterpret_cast<const __nv_bfloat16*>(lora_b_raw) +
                    ((row_base + (long long)slot_col[slot_1]) * 32 + (long long)rank_col) + 0);
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
                  reinterpret_cast<const __nv_bfloat16*>(shrink_raw) + (pair * 32 + rank_col) + 0);
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
            for (int slot_2 = 0; slot_2 < 4; slot_2++) {
              if (slot_col[slot_2] < hidden) {
                float _vec_load_3[8];
                {
                  const uint4* _vptr_3 = reinterpret_cast<const uint4*>(
                      reinterpret_cast<const __nv_bfloat16*>(lora_b_raw) +
                      ((row_base_1 + (long long)slot_col[slot_2]) * 32 + (long long)rank_col) + 0);
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
      for (int slot_3 = 0; slot_3 < 4; slot_3++) {
        float column_sum = slot_acc[slot_3];
#pragma unroll
        for (int step = 0; step < 2; step++) {
          float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, column_sum, 4 >> step + 1);
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
__launch_bounds__(128, 1) void kernel_flashinfer_bgmv_moe_expand_generic_token_t128_bf16_r32(
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
  int rank_lane = lane % 4;
  int col_lane = lane / 4;
  int rank_col = rank_lane * 8;
  int slot_col[4];
  float slot_acc[4];
#pragma unroll
  for (int slot = 0; slot < 4; slot++) {
    slot_col[slot] = warp_col_base + slot * 8 + col_lane;
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
                                               (direct_pair * 32 + rank_col) + 0);
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
          for (int slot_1 = 0; slot_1 < 4; slot_1++) {
            if (slot_col[slot_1] < hidden) {
              float _vec_load_1[8];
              {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(
                    reinterpret_cast<const __nv_bfloat16*>(lora_b_raw) +
                    ((row_base + (long long)slot_col[slot_1]) * 32 + (long long)rank_col) + 0);
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
                  reinterpret_cast<const __nv_bfloat16*>(shrink_raw) + (pair * 32 + rank_col) + 0);
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
            for (int slot_2 = 0; slot_2 < 4; slot_2++) {
              if (slot_col[slot_2] < hidden) {
                float _vec_load_3[8];
                {
                  const uint4* _vptr_3 = reinterpret_cast<const uint4*>(
                      reinterpret_cast<const __nv_bfloat16*>(lora_b_raw) +
                      ((row_base_1 + (long long)slot_col[slot_2]) * 32 + (long long)rank_col) + 0);
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
      for (int slot_3 = 0; slot_3 < 4; slot_3++) {
        float column_sum = slot_acc[slot_3];
#pragma unroll
        for (int step = 0; step < 2; step++) {
          float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, column_sum, 4 >> step + 1);
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
#define THREADS 64

extern "C" {

__global__
__launch_bounds__(64, 1) void kernel_flashinfer_bgmv_moe_expand_generic_token_t64_pf_bf16_r32(
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
  int rank_lane = lane % 4;
  int col_lane = lane / 4;
  int rank_col = rank_lane * 8;
  int slot_col[4];
  float slot_acc[4];
#pragma unroll
  for (int slot = 0; slot < 4; slot++) {
    slot_col[slot] = warp_col_base + slot * 8 + col_lane;
    slot_acc[slot] = 0.0f;
  }
  if (token < num_tokens) {
    int advance_parity = 0;
    unsigned int advance_count = 0;
    unsigned int advance_launch = 0;
    long long lora_id = lora_indices[token];
    int pair_base = token * 2;
    int contiguous = 0;
    unsigned int w_car[32];
    float w_values[8];
    float pre_weight[2];
    if (lora_id >= 0) {
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
          pre_weight[route] = topk_weights[direct_pair];
          long long row_base = (lora_id * (long long)num_experts + expert) * (long long)hidden;
#pragma unroll
          for (int slot_1 = 0; slot_1 < 4; slot_1++) {
            if (slot_col[slot_1] < hidden) {
              {
                const uint4* _ivptr_0 = reinterpret_cast<const uint4*>(
                    reinterpret_cast<const unsigned int*>(lora_b_raw) +
                    (row_base + (long long)slot_col[slot_1]) * 16 + (long long)(rank_lane * 4));
                uint4 _ivld_0;
                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                             : "=r"(_ivld_0.x), "=r"(_ivld_0.y), "=r"(_ivld_0.z), "=r"(_ivld_0.w)
                             : "l"((const void*)(_ivptr_0))
                             : "memory");
                (w_car + (route * 4 + slot_1) * 4)[0 + 0] = _ivld_0.x;
                (w_car + (route * 4 + slot_1) * 4)[0 + 1] = _ivld_0.y;
                (w_car + (route * 4 + slot_1) * 4)[0 + 2] = _ivld_0.z;
                (w_car + (route * 4 + slot_1) * 4)[0 + 3] = _ivld_0.w;
              }
            }
          }
        }
      }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (route_advance != 0) {
      if (blockIdx.y == 0) {
        if (tid == 0) {
          advance_parity = (int)route_index_raw[1];
          advance_count = route_index_raw[4 + token];
          advance_launch = route_index_raw[0];
        }
      }
    }
    int route_count = num_pairs;
    int use_index = 0;
    if (lora_id >= 0) {
      if (contiguous == 0) {
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
      }
    }
    if (lora_id >= 0) {
      if (contiguous != 0) {
#pragma unroll
        for (int route_1 = 0; route_1 < 2; route_1++) {
          int direct_pair_1 = pair_base + route_1;
          float _vec_load_0[8];
          {
            const uint4* _vptr_1 =
                reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(shrink_raw) +
                                               (direct_pair_1 * 32 + rank_col) + 0);
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
                    : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]),
                      "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                    : "r"(_vpairs_1[_pair]));
              }
            }
          }
#pragma unroll
          for (int slot_2 = 0; slot_2 < 4; slot_2++) {
            if (slot_col[slot_2] < hidden) {
              {
#pragma unroll
                for (int pair = 0; pair < 4; pair++) {
                  w_values[2 * pair] =
                      __uint_as_float(w_car[(route_1 * 4 + slot_2) * 4 + pair] << 16);
                  w_values[2 * pair + 1] =
                      __uint_as_float(w_car[(route_1 * 4 + slot_2) * 4 + pair] & 4294901760u);
                }
              }
              float partial = 0.0f;
#pragma unroll
              for (int element = 0; element < 8; element++) {
                float _fma_0 = __fmaf_rn(_vec_load_0[element], w_values[element], partial);
                partial = _fma_0;
              }
              float _fma_1 = __fmaf_rn(partial, pre_weight[route_1], slot_acc[slot_2]);
              slot_acc[slot_2] = _fma_1;
            }
          }
        }
      } else {
#pragma unroll 1
        for (int route_step = 0; route_step < route_count; route_step++) {
          int pair_1 = route_step;
          int route_match = 1;
          if (use_index != 0) {
            pair_1 = route_list[route_step];
          } else if (sorted_token_ids[route_step] != (long long)token) {
            route_match = 0;
          }
          if (route_match != 0) {
            long long expert_1 = expert_ids[pair_1];
            float pair_weight = topk_weights[pair_1];
            float _vec_load_1[8];
            {
              const uint4* _vptr_2 = reinterpret_cast<const uint4*>(
                  reinterpret_cast<const __nv_bfloat16*>(shrink_raw) + (pair_1 * 32 + rank_col) +
                  0);
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
                      : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]),
                        "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                      : "r"(_vpairs_2[_pair]));
                }
              }
            }
            long long row_base_1 =
                (lora_id * (long long)num_experts + expert_1) * (long long)hidden;
#pragma unroll
            for (int slot_3 = 0; slot_3 < 4; slot_3++) {
              if (slot_col[slot_3] < hidden) {
                float _vec_load_2[8];
                {
                  const uint4* _vptr_3 = reinterpret_cast<const uint4*>(
                      reinterpret_cast<const __nv_bfloat16*>(lora_b_raw) +
                      ((row_base_1 + (long long)slot_col[slot_3]) * 32 + (long long)rank_col) + 0);
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
                          : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]),
                            "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                          : "r"(_vpairs_3[_pair]));
                    }
                  }
                }
                float partial_1 = 0.0f;
#pragma unroll
                for (int element_1 = 0; element_1 < 8; element_1++) {
                  float _fma_2 =
                      __fmaf_rn(_vec_load_1[element_1], _vec_load_2[element_1], partial_1);
                  partial_1 = _fma_2;
                }
                float _fma_3 = __fmaf_rn(partial_1, pair_weight, slot_acc[slot_3]);
                slot_acc[slot_3] = _fma_3;
              }
            }
          }
        }
      }
#pragma unroll
      for (int slot_4 = 0; slot_4 < 4; slot_4++) {
        float column_sum = slot_acc[slot_4];
#pragma unroll
        for (int step = 0; step < 2; step++) {
          float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, column_sum, 4 >> step + 1);
          column_sum += _shfl_xor_0;
        }
        if (rank_lane == 0) {
          if (slot_col[slot_4] < hidden) {
            *(reinterpret_cast<float*>(y_accum +
                                       (token * output_stride + output_offset + slot_col[slot_4])) +
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
__launch_bounds__(128, 1) void kernel_flashinfer_bgmv_moe_expand_generic_token_t128_pf_bf16_r32(
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
  int rank_lane = lane % 4;
  int col_lane = lane / 4;
  int rank_col = rank_lane * 8;
  int slot_col[4];
  float slot_acc[4];
#pragma unroll
  for (int slot = 0; slot < 4; slot++) {
    slot_col[slot] = warp_col_base + slot * 8 + col_lane;
    slot_acc[slot] = 0.0f;
  }
  if (token < num_tokens) {
    int advance_parity = 0;
    unsigned int advance_count = 0;
    unsigned int advance_launch = 0;
    long long lora_id = lora_indices[token];
    int pair_base = token * 2;
    int contiguous = 0;
    unsigned int w_car[32];
    float w_values[8];
    float pre_weight[2];
    if (lora_id >= 0) {
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
          pre_weight[route] = topk_weights[direct_pair];
          long long row_base = (lora_id * (long long)num_experts + expert) * (long long)hidden;
#pragma unroll
          for (int slot_1 = 0; slot_1 < 4; slot_1++) {
            if (slot_col[slot_1] < hidden) {
              {
                const uint4* _ivptr_0 = reinterpret_cast<const uint4*>(
                    reinterpret_cast<const unsigned int*>(lora_b_raw) +
                    (row_base + (long long)slot_col[slot_1]) * 16 + (long long)(rank_lane * 4));
                uint4 _ivld_0;
                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                             : "=r"(_ivld_0.x), "=r"(_ivld_0.y), "=r"(_ivld_0.z), "=r"(_ivld_0.w)
                             : "l"((const void*)(_ivptr_0))
                             : "memory");
                (w_car + (route * 4 + slot_1) * 4)[0 + 0] = _ivld_0.x;
                (w_car + (route * 4 + slot_1) * 4)[0 + 1] = _ivld_0.y;
                (w_car + (route * 4 + slot_1) * 4)[0 + 2] = _ivld_0.z;
                (w_car + (route * 4 + slot_1) * 4)[0 + 3] = _ivld_0.w;
              }
            }
          }
        }
      }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (route_advance != 0) {
      if (blockIdx.y == 0) {
        if (tid == 0) {
          advance_parity = (int)route_index_raw[1];
          advance_count = route_index_raw[4 + token];
          advance_launch = route_index_raw[0];
        }
      }
    }
    int route_count = num_pairs;
    int use_index = 0;
    if (lora_id >= 0) {
      if (contiguous == 0) {
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
      }
    }
    if (lora_id >= 0) {
      if (contiguous != 0) {
#pragma unroll
        for (int route_1 = 0; route_1 < 2; route_1++) {
          int direct_pair_1 = pair_base + route_1;
          float _vec_load_0[8];
          {
            const uint4* _vptr_1 =
                reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(shrink_raw) +
                                               (direct_pair_1 * 32 + rank_col) + 0);
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
                    : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]),
                      "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                    : "r"(_vpairs_1[_pair]));
              }
            }
          }
#pragma unroll
          for (int slot_2 = 0; slot_2 < 4; slot_2++) {
            if (slot_col[slot_2] < hidden) {
              {
#pragma unroll
                for (int pair = 0; pair < 4; pair++) {
                  w_values[2 * pair] =
                      __uint_as_float(w_car[(route_1 * 4 + slot_2) * 4 + pair] << 16);
                  w_values[2 * pair + 1] =
                      __uint_as_float(w_car[(route_1 * 4 + slot_2) * 4 + pair] & 4294901760u);
                }
              }
              float partial = 0.0f;
#pragma unroll
              for (int element = 0; element < 8; element++) {
                float _fma_0 = __fmaf_rn(_vec_load_0[element], w_values[element], partial);
                partial = _fma_0;
              }
              float _fma_1 = __fmaf_rn(partial, pre_weight[route_1], slot_acc[slot_2]);
              slot_acc[slot_2] = _fma_1;
            }
          }
        }
      } else {
#pragma unroll 1
        for (int route_step = 0; route_step < route_count; route_step++) {
          int pair_1 = route_step;
          int route_match = 1;
          if (use_index != 0) {
            pair_1 = route_list[route_step];
          } else if (sorted_token_ids[route_step] != (long long)token) {
            route_match = 0;
          }
          if (route_match != 0) {
            long long expert_1 = expert_ids[pair_1];
            float pair_weight = topk_weights[pair_1];
            float _vec_load_1[8];
            {
              const uint4* _vptr_2 = reinterpret_cast<const uint4*>(
                  reinterpret_cast<const __nv_bfloat16*>(shrink_raw) + (pair_1 * 32 + rank_col) +
                  0);
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
                      : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]),
                        "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                      : "r"(_vpairs_2[_pair]));
                }
              }
            }
            long long row_base_1 =
                (lora_id * (long long)num_experts + expert_1) * (long long)hidden;
#pragma unroll
            for (int slot_3 = 0; slot_3 < 4; slot_3++) {
              if (slot_col[slot_3] < hidden) {
                float _vec_load_2[8];
                {
                  const uint4* _vptr_3 = reinterpret_cast<const uint4*>(
                      reinterpret_cast<const __nv_bfloat16*>(lora_b_raw) +
                      ((row_base_1 + (long long)slot_col[slot_3]) * 32 + (long long)rank_col) + 0);
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
                          : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]),
                            "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                          : "r"(_vpairs_3[_pair]));
                    }
                  }
                }
                float partial_1 = 0.0f;
#pragma unroll
                for (int element_1 = 0; element_1 < 8; element_1++) {
                  float _fma_2 =
                      __fmaf_rn(_vec_load_1[element_1], _vec_load_2[element_1], partial_1);
                  partial_1 = _fma_2;
                }
                float _fma_3 = __fmaf_rn(partial_1, pair_weight, slot_acc[slot_3]);
                slot_acc[slot_3] = _fma_3;
              }
            }
          }
        }
      }
#pragma unroll
      for (int slot_4 = 0; slot_4 < 4; slot_4++) {
        float column_sum = slot_acc[slot_4];
#pragma unroll
        for (int step = 0; step < 2; step++) {
          float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, column_sum, 4 >> step + 1);
          column_sum += _shfl_xor_0;
        }
        if (rank_lane == 0) {
          if (slot_col[slot_4] < hidden) {
            *(reinterpret_cast<float*>(y_accum +
                                       (token * output_stride + output_offset + slot_col[slot_4])) +
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
#define SMEM_HIST_S_OFF 0
#define SMEM_HIST_S_STAGE_BYTES 16384
#define SMEM_HIST_S_STRIDE 16384
#define SMEM_TOTAL 16384
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(256, 1) void kernel_flashinfer_bgmv_moe_group_hist_bf16_r32(
    long long* __restrict__ sorted_token_ids, long long* __restrict__ expert_ids,
    long long* __restrict__ lora_indices, uint16_t* __restrict__ shrink_out_raw, int num_pairs,
    int num_tokens, int num_experts, int num_loras, int num_ctas,
    unsigned int* __restrict__ workspace_raw, int off_hist) {
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
  int* hist_s = reinterpret_cast<int*>(smem_raw + 0);
  const int hist_s_addr = smem + 0;

  // === Task calls (dependency order) ===
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  int cta = blockIdx.x;
  int bins = num_loras * num_experts;
  int bins_per_thread = (bins + 256 - 1) / 256;
  int passes = (num_pairs + num_ctas * 1024 - 1) / (num_ctas * 1024);
#pragma unroll 1
  for (int i = 0; i < bins_per_thread; i++) {
    int b = i * 256 + tid;
    if (b < bins) {
      hist_s[b] = 0;
    }
  }
  __syncthreads();
  int keys[4];
#pragma unroll 1
  for (int ps = 0; ps < passes; ps++) {
#pragma unroll
    for (int slot = 0; slot < 4; slot++) {
      keys[slot] = -1;
      int p = (ps * num_ctas + cta) * 1024 + slot * 256 + tid;
      if (p < num_pairs) {
        long long token = sorted_token_ids[p];
        long long expert = expert_ids[p];
        keys[slot] = -2;
        if (token >= 0) {
          if (token < (long long)num_tokens) {
            long long lora = lora_indices[token];
            if (lora >= 0) {
              if (expert >= 0) {
                if (expert < (long long)num_experts) {
                  keys[slot] = (int)lora * num_experts + (int)expert;
                }
              }
            }
          }
        }
      }
    }
#pragma unroll
    for (int slot_1 = 0; slot_1 < 4; slot_1++) {
      int p_1 = (ps * num_ctas + cta) * 1024 + slot_1 * 256 + tid;
      if (keys[slot_1] >= 0) {
        int _atomic_old_0 = atomicAdd(&hist_s[keys[slot_1]], 1);
      }
      if (keys[slot_1] == -2) {
#pragma unroll
        for (int rr = 0; rr < 32; rr++) {
          *(reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<__nv_bfloat16*>(shrink_out_raw) +
                                             (p_1 * 32 + rr)) +
            (0)) = __float2bfloat16_rn(0.0f);
        }
      }
    }
  }
  __syncthreads();
#pragma unroll 1
  for (int i_1 = 0; i_1 < bins_per_thread; i_1++) {
    int b_1 = i_1 * 256 + tid;
    if (b_1 < bins) {
      *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(workspace_raw) +
                                        (off_hist + cta * bins + b_1)) +
        (0)) = (unsigned int)hist_s[b_1];
    }
  }
}

}  // extern "C"

#undef BLACKWELL_INF
#undef NUM_MAIN_STAGES
#undef SMEM_HIST_S_OFF
#undef SMEM_HIST_S_STAGE_BYTES
#undef SMEM_HIST_S_STRIDE
#undef SMEM_TOTAL
#undef THREADS

#define BLACKWELL_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_WARP_COUNTS_OFF 0
#define SMEM_WARP_COUNTS_STAGE_BYTES 128
#define SMEM_WARP_COUNTS_STRIDE 128
#define SMEM_WARP_TILES_OFF 128
#define SMEM_WARP_TILES_STAGE_BYTES 128
#define SMEM_WARP_TILES_STRIDE 128
#define SMEM_TOTAL 384
#define THREADS 1024

extern "C" {

__global__ __launch_bounds__(1024, 1) void kernel_flashinfer_bgmv_moe_group_scan_bf16_r32(
    int num_pairs, int num_tokens, int num_experts, int num_loras, int num_ctas,
    unsigned int* __restrict__ workspace_raw, int off_group_offset, int off_tile_table,
    int off_token_count, int off_hist, int off_base) {
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
  int* warp_counts = reinterpret_cast<int*>(smem_raw + 0);
  const int warp_counts_addr = smem + 0;
  int* warp_tiles = reinterpret_cast<int*>(smem_raw + 128);
  const int warp_tiles_addr = smem + 128;

  // === Task calls (dependency order) ===
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  int bins = num_loras * num_experts;
  int bins_per_thread = (bins + 1024 - 1) / 1024;
  int tokens_per_thread = (num_tokens + 1024 - 1) / 1024;
#pragma unroll 1
  for (int i = 0; i < tokens_per_thread; i++) {
    int t = i * 1024 + tid;
    if (t < num_tokens) {
      *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(workspace_raw) +
                                        (off_token_count + t)) +
        (0)) = 0;
    }
  }
  int local_counts[4];
  int local_tiles[4];
  int my_count = 0;
  int my_tiles = 0;
#pragma unroll
  for (int i_1 = 0; i_1 < 4; i_1++) {
    local_counts[i_1] = 0;
    local_tiles[i_1] = 0;
    if (bins_per_thread > i_1) {
      int b = tid * bins_per_thread + i_1;
      if (b < bins) {
#pragma unroll 8
        for (int c = 0; c < num_ctas; c++) {
          local_counts[i_1] = local_counts[i_1] + (int)reinterpret_cast<unsigned int*>(
                                                      workspace_raw)[off_hist + c * bins + b];
        }
        local_tiles[i_1] = (local_counts[i_1] + 16 - 1) / 16;
        my_count += local_counts[i_1];
        my_tiles += local_tiles[i_1];
      }
    }
  }
  int incl_count = my_count;
  int incl_tiles = my_tiles;
#pragma unroll
  for (int step = 0; step < 5; step++) {
    int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, incl_count, 1 << step, 32);
    int up_count = _shfl_up_0;
    int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, incl_tiles, 1 << step, 32);
    int up_tiles = _shfl_up_1;
    if (lane >= 1 << step) {
      incl_count += up_count;
      incl_tiles += up_tiles;
    }
  }
  if (lane == 31) {
    warp_counts[warp] = incl_count;
    warp_tiles[warp] = incl_tiles;
  }
  __syncthreads();
  int warp_base_count = 0;
  int warp_base_tiles = 0;
#pragma unroll
  for (int w = 0; w < 32; w++) {
    if (w < warp) {
      warp_base_count += warp_counts[w];
      warp_base_tiles += warp_tiles[w];
    }
  }
  int excl_count = warp_base_count + incl_count - my_count;
  int excl_tiles = warp_base_tiles + incl_tiles - my_tiles;
#pragma unroll
  for (int i_2 = 0; i_2 < 4; i_2++) {
    if (bins_per_thread > i_2) {
      int b_1 = tid * bins_per_thread + i_2;
      if (b_1 < bins) {
        *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(workspace_raw) +
                                          (off_group_offset + b_1)) +
          (0)) = (unsigned int)excl_count;
#pragma unroll 1
        for (int c_1 = 0; c_1 < local_tiles[i_2]; c_1++) {
          *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(workspace_raw) +
                                            (off_tile_table + excl_tiles + c_1)) +
            (0)) = (unsigned int)(b_1 * 65536 + c_1);
        }
        int cta_base = excl_count;
#pragma unroll 8
        for (int c_2 = 0; c_2 < num_ctas; c_2++) {
          *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(workspace_raw) +
                                            (off_base + c_2 * bins + b_1)) +
            (0)) = (unsigned int)cta_base;
          cta_base +=
              (int)reinterpret_cast<unsigned int*>(workspace_raw)[off_hist + c_2 * bins + b_1];
        }
        excl_count += local_counts[i_2];
        excl_tiles += local_tiles[i_2];
      }
    }
  }
  if (tid == 1023) {
    *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(workspace_raw) +
                                      (off_group_offset + bins)) +
      (0)) = (unsigned int)excl_count;
    *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(workspace_raw)) + (0)) =
        (unsigned int)excl_tiles;
  }
}

}  // extern "C"

#undef BLACKWELL_INF
#undef NUM_MAIN_STAGES
#undef SMEM_TOTAL
#undef SMEM_WARP_COUNTS_OFF
#undef SMEM_WARP_COUNTS_STAGE_BYTES
#undef SMEM_WARP_COUNTS_STRIDE
#undef SMEM_WARP_TILES_OFF
#undef SMEM_WARP_TILES_STAGE_BYTES
#undef SMEM_WARP_TILES_STRIDE
#undef THREADS

#define BLACKWELL_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_FILL_S_OFF 0
#define SMEM_FILL_S_STAGE_BYTES 16384
#define SMEM_FILL_S_STRIDE 16384
#define SMEM_TOTAL 16384
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(256, 1) void kernel_flashinfer_bgmv_moe_group_scatter_bf16_r32(
    long long* __restrict__ sorted_token_ids, long long* __restrict__ expert_ids,
    long long* __restrict__ lora_indices, int num_pairs, int num_tokens, int num_experts,
    int num_loras, int num_ctas, unsigned int* __restrict__ workspace_raw, int off_sorted_routes,
    int off_token_count, int off_token_routes, int off_base) {
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
  int* fill_s = reinterpret_cast<int*>(smem_raw + 0);
  const int fill_s_addr = smem + 0;

  // === Task calls (dependency order) ===
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  int cta = blockIdx.x;
  int bins = num_loras * num_experts;
  int bins_per_thread = (bins + 256 - 1) / 256;
  int passes = (num_pairs + num_ctas * 1024 - 1) / (num_ctas * 1024);
  int contiguous_shape = 0;
  if (num_pairs == num_tokens * 2) {
    contiguous_shape = 1;
  }
#pragma unroll 1
  for (int i = 0; i < bins_per_thread; i++) {
    int b = i * 256 + tid;
    if (b < bins) {
      fill_s[b] = 0;
    }
  }
  __syncthreads();
  int keys[4];
  int tok32[4];
  int listed[4];
#pragma unroll 1
  for (int ps = 0; ps < passes; ps++) {
#pragma unroll
    for (int slot = 0; slot < 4; slot++) {
      keys[slot] = -1;
      tok32[slot] = -1;
      listed[slot] = 0;
      int p = (ps * num_ctas + cta) * 1024 + slot * 256 + tid;
      if (p < num_pairs) {
        long long token = sorted_token_ids[p];
        long long expert = expert_ids[p];
        if (token >= 0) {
          if (token < (long long)num_tokens) {
            tok32[slot] = (int)token;
            listed[slot] = 1;
            if (contiguous_shape != 0) {
              if (p / 2 == tok32[slot]) {
                listed[slot] = 0;
              }
            }
            long long lora = lora_indices[token];
            if (lora >= 0) {
              if (expert >= 0) {
                if (expert < (long long)num_experts) {
                  keys[slot] = (int)lora * num_experts + (int)expert;
                }
              }
            }
          }
        }
      }
    }
#pragma unroll
    for (int slot_1 = 0; slot_1 < 4; slot_1++) {
      int p_1 = (ps * num_ctas + cta) * 1024 + slot_1 * 256 + tid;
      if (listed[slot_1] != 0) {
        unsigned int _atomic_old_0 = atomicAdd(
            &reinterpret_cast<unsigned int*>(workspace_raw)[off_token_count + tok32[slot_1]], 1);
        unsigned int route_slot = _atomic_old_0;
        if (route_slot < 16) {
          *(reinterpret_cast<unsigned int*>(
                reinterpret_cast<unsigned int*>(workspace_raw) +
                (off_token_routes + tok32[slot_1] * 16 + (int)route_slot)) +
            (0)) = (unsigned int)p_1;
        }
      }
      if (keys[slot_1] >= 0) {
        int _atomic_old_1 = atomicAdd(&fill_s[keys[slot_1]], 1);
        int fill_slot = _atomic_old_1;
        int pos = (int)reinterpret_cast<unsigned int*>(
                      workspace_raw)[off_base + cta * bins + keys[slot_1]] +
                  fill_slot;
        *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(workspace_raw) +
                                          (off_sorted_routes + pos)) +
          (0)) = (unsigned int)p_1;
      }
    }
  }
}

}  // extern "C"

#undef BLACKWELL_INF
#undef NUM_MAIN_STAGES
#undef SMEM_FILL_S_OFF
#undef SMEM_FILL_S_STAGE_BYTES
#undef SMEM_FILL_S_STRIDE
#undef SMEM_TOTAL
#undef THREADS

#define BLACKWELL_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_WARP_PARTIALS_OFF 0
#define SMEM_WARP_PARTIALS_STAGE_BYTES 512
#define SMEM_WARP_PARTIALS_STRIDE 512
#define SMEM_X_RING_OFF 0
#define SMEM_X_RING_STAGE_BYTES 16
#define SMEM_X_RING_STRIDE 16
#define SMEM_W_RING_OFF 0
#define SMEM_W_RING_STAGE_BYTES 16
#define SMEM_W_RING_STRIDE 16
#define SMEM_TOTAL 512
#define THREADS 128

extern "C" {

__global__ __launch_bounds__(128, 4) void kernel_flashinfer_bgmv_moe_shrink_grouped_bf16_r32(
    uint16_t* __restrict__ shrink_out_raw, uint16_t* __restrict__ x_raw,
    uint16_t* __restrict__ lora_a_raw, long long* __restrict__ sorted_token_ids, int num_pairs,
    int num_experts, int hidden, int num_tiles, int rt_per_cta, int rt_groups,
    unsigned int* __restrict__ workspace_raw, int off_group_offset, int off_tile_table,
    int off_sorted_routes) {
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
  float* warp_partials = reinterpret_cast<float*>(smem_raw + 0);
  const int warp_partials_addr = smem + 0;
  __nv_bfloat16* x_ring = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
  const int x_ring_addr = smem + 0;
  __nv_bfloat16* w_ring = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
  const int w_ring_addr = smem + 0;

  // === Task calls (dependency order) ===
  asm volatile("griddepcontrol.wait;" ::: "memory");
  int rt_group = blockIdx.x % ((0) ? 4 : rt_groups);
  int half = blockIdx.x / ((0) ? 4 : rt_groups) % 4;
  int tile = blockIdx.x / (((0) ? 4 : rt_groups) * 4);
  int n_tiles = (int)reinterpret_cast<const unsigned int*>(workspace_raw)[0];
  if (tile < n_tiles) {
    int entry = (int)reinterpret_cast<const unsigned int*>(workspace_raw)[off_tile_table + tile];
    int group = entry / 65536;
    int chunk = entry % 65536;
    int start =
        (int)reinterpret_cast<const unsigned int*>(workspace_raw)[off_group_offset + group] +
        chunk * 16 + half * 4;
    int group_end =
        (int)reinterpret_cast<const unsigned int*>(workspace_raw)[off_group_offset + group + 1];
    int count = group_end - start;
    if (count > 4) {
      count = 4;
    }
    if (count > 0) {
      int lora = group / num_experts;
      int expert = group % num_experts;
      int hidden_words = hidden / 2;
      long long weight_row_base = (long long)(lora * num_experts + expert) * 32;
      int routes[4];
      long long tokens[4];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        routes[j] = -1;
        tokens[j] = 0;
        if (count > j) {
          routes[j] = (int)reinterpret_cast<const unsigned int*>(
              workspace_raw)[off_sorted_routes + start + j];
          tokens[j] = sorted_token_ids[routes[j]];
        }
      }
      int rank_tile0 = rt_group * rt_per_cta;
      int rank_tile_end = rank_tile0 + rt_per_cta;
      int rank_base_it = rank_tile0 * 8;
      long long weight_words_it =
          (weight_row_base + (long long)rank_base_it) * (long long)hidden_words;
      long long weight_words_step = hidden_words * 8;
#pragma unroll 1
      for (int rank_tile = rank_tile0; rank_tile < rank_tile_end; rank_tile++) {
        float acc[32];
        unsigned int x_car[16];
        unsigned int xr_car[4];
        unsigned int w_car[4];
        float x_values[32];
        float w_values[8];
        float w_t[16];
        int lane_0 = lane;
        float red_a[16];
        float red_b[8];
        float red_c[4];
        float red_d[2];
        float red_e[1];
#pragma unroll
        for (int owner = 0; owner < 32; owner++) {
          acc[owner] = 0.0f;
        }
#pragma unroll 1
        for (int local = 0; local < num_tiles; local++) {
          int k_base = local * 1024 + tid * 8;
          if (k_base < hidden) {
            int k_words = k_base / 2;
#pragma unroll
            for (int j_1 = 0; j_1 < 4; j_1++) {
              {
                const uint4* _ivptr_0 = reinterpret_cast<const uint4*>(
                    reinterpret_cast<const unsigned int*>(x_raw) +
                    tokens[j_1] * (long long)hidden_words + (long long)k_words);
                uint4 _ivld_0;
                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                             : "=r"(_ivld_0.x), "=r"(_ivld_0.y), "=r"(_ivld_0.z), "=r"(_ivld_0.w)
                             : "l"((const void*)(_ivptr_0))
                             : "memory");
                (x_car + j_1 * 4)[0 + 0] = _ivld_0.x;
                (x_car + j_1 * 4)[0 + 1] = _ivld_0.y;
                (x_car + j_1 * 4)[0 + 2] = _ivld_0.z;
                (x_car + j_1 * 4)[0 + 3] = _ivld_0.w;
              }
            }
#pragma unroll
            for (int j_2 = 0; j_2 < 4; j_2++) {
#pragma unroll
              for (int pair = 0; pair < 4; pair++) {
                x_values[j_2 * 8 + 2 * pair] = __uint_as_float(x_car[j_2 * 4 + pair] << 16);
                x_values[j_2 * 8 + 2 * pair + 1] =
                    __uint_as_float(x_car[j_2 * 4 + pair] & 4294901760u);
              }
            }
            {
              uint32_t _uv4_1_0;
              uint32_t _uv4_1_1;
              uint32_t _uv4_1_2;
              uint32_t _uv4_1_3;
              asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                           : "=r"(_uv4_1_0), "=r"(_uv4_1_1), "=r"(_uv4_1_2), "=r"(_uv4_1_3)
                           : "l"((const void*)(reinterpret_cast<const unsigned int*>(lora_a_raw) +
                                               (weight_words_it + (long long)k_words)))
                           : "memory");
              w_car[0 + 0] = _uv4_1_0;
              w_car[0 + 1] = _uv4_1_1;
              w_car[0 + 2] = _uv4_1_2;
              w_car[0 + 3] = _uv4_1_3;
            }
            w_t[0] = __uint_as_float(w_car[0] << 16);
            w_t[2] = __uint_as_float(w_car[0] & 4294901760u);
            w_t[4] = __uint_as_float(w_car[1] << 16);
            w_t[6] = __uint_as_float(w_car[1] & 4294901760u);
            w_t[8] = __uint_as_float(w_car[2] << 16);
            w_t[10] = __uint_as_float(w_car[2] & 4294901760u);
            w_t[12] = __uint_as_float(w_car[3] << 16);
            w_t[14] = __uint_as_float(w_car[3] & 4294901760u);
            {
              uint32_t _uv4_2_0;
              uint32_t _uv4_2_1;
              uint32_t _uv4_2_2;
              uint32_t _uv4_2_3;
              asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                           : "=r"(_uv4_2_0), "=r"(_uv4_2_1), "=r"(_uv4_2_2), "=r"(_uv4_2_3)
                           : "l"((const void*)(reinterpret_cast<const unsigned int*>(lora_a_raw) +
                                               (weight_words_it + (long long)hidden_words +
                                                (long long)k_words)))
                           : "memory");
              w_car[0 + 0] = _uv4_2_0;
              w_car[0 + 1] = _uv4_2_1;
              w_car[0 + 2] = _uv4_2_2;
              w_car[0 + 3] = _uv4_2_3;
            }
            w_t[1] = __uint_as_float(w_car[0] << 16);
            w_t[3] = __uint_as_float(w_car[0] & 4294901760u);
            w_t[5] = __uint_as_float(w_car[1] << 16);
            w_t[7] = __uint_as_float(w_car[1] & 4294901760u);
            w_t[9] = __uint_as_float(w_car[2] << 16);
            w_t[11] = __uint_as_float(w_car[2] & 4294901760u);
            w_t[13] = __uint_as_float(w_car[3] << 16);
            w_t[15] = __uint_as_float(w_car[3] & 4294901760u);
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_3;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_3) : "f"(x_values[0]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[0]), "+f"(acc[1])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_3));
#else
              acc[0] = __fmaf_rn(w_t[0], x_values[0], acc[0]);
              acc[1] = __fmaf_rn(w_t[1], x_values[0], acc[1]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_4;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_4) : "f"(x_values[1]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[0]), "+f"(acc[1])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_4));
#else
              acc[0] = __fmaf_rn(w_t[2], x_values[1], acc[0]);
              acc[1] = __fmaf_rn(w_t[3], x_values[1], acc[1]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_5;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_5) : "f"(x_values[2]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[0]), "+f"(acc[1])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_5));
#else
              acc[0] = __fmaf_rn(w_t[4], x_values[2], acc[0]);
              acc[1] = __fmaf_rn(w_t[5], x_values[2], acc[1]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_6;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_6) : "f"(x_values[3]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[0]), "+f"(acc[1])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_6));
#else
              acc[0] = __fmaf_rn(w_t[6], x_values[3], acc[0]);
              acc[1] = __fmaf_rn(w_t[7], x_values[3], acc[1]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_7;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_7) : "f"(x_values[4]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[0]), "+f"(acc[1])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_7));
#else
              acc[0] = __fmaf_rn(w_t[8], x_values[4], acc[0]);
              acc[1] = __fmaf_rn(w_t[9], x_values[4], acc[1]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_8;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_8) : "f"(x_values[5]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[0]), "+f"(acc[1])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_8));
#else
              acc[0] = __fmaf_rn(w_t[10], x_values[5], acc[0]);
              acc[1] = __fmaf_rn(w_t[11], x_values[5], acc[1]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_9;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_9) : "f"(x_values[6]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[0]), "+f"(acc[1])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_9));
#else
              acc[0] = __fmaf_rn(w_t[12], x_values[6], acc[0]);
              acc[1] = __fmaf_rn(w_t[13], x_values[6], acc[1]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_10;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_10) : "f"(x_values[7]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[0]), "+f"(acc[1])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_10));
#else
              acc[0] = __fmaf_rn(w_t[14], x_values[7], acc[0]);
              acc[1] = __fmaf_rn(w_t[15], x_values[7], acc[1]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_11;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_11) : "f"(x_values[8]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[8]), "+f"(acc[9])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_11));
#else
              acc[8] = __fmaf_rn(w_t[0], x_values[8], acc[8]);
              acc[9] = __fmaf_rn(w_t[1], x_values[8], acc[9]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_12;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_12) : "f"(x_values[9]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[8]), "+f"(acc[9])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_12));
#else
              acc[8] = __fmaf_rn(w_t[2], x_values[9], acc[8]);
              acc[9] = __fmaf_rn(w_t[3], x_values[9], acc[9]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_13;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_13) : "f"(x_values[10]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[8]), "+f"(acc[9])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_13));
#else
              acc[8] = __fmaf_rn(w_t[4], x_values[10], acc[8]);
              acc[9] = __fmaf_rn(w_t[5], x_values[10], acc[9]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_14;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_14) : "f"(x_values[11]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[8]), "+f"(acc[9])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_14));
#else
              acc[8] = __fmaf_rn(w_t[6], x_values[11], acc[8]);
              acc[9] = __fmaf_rn(w_t[7], x_values[11], acc[9]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_15;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_15) : "f"(x_values[12]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[8]), "+f"(acc[9])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_15));
#else
              acc[8] = __fmaf_rn(w_t[8], x_values[12], acc[8]);
              acc[9] = __fmaf_rn(w_t[9], x_values[12], acc[9]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_16;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_16) : "f"(x_values[13]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[8]), "+f"(acc[9])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_16));
#else
              acc[8] = __fmaf_rn(w_t[10], x_values[13], acc[8]);
              acc[9] = __fmaf_rn(w_t[11], x_values[13], acc[9]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_17;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_17) : "f"(x_values[14]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[8]), "+f"(acc[9])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_17));
#else
              acc[8] = __fmaf_rn(w_t[12], x_values[14], acc[8]);
              acc[9] = __fmaf_rn(w_t[13], x_values[14], acc[9]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_18;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_18) : "f"(x_values[15]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[8]), "+f"(acc[9])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_18));
#else
              acc[8] = __fmaf_rn(w_t[14], x_values[15], acc[8]);
              acc[9] = __fmaf_rn(w_t[15], x_values[15], acc[9]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_19;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_19) : "f"(x_values[16]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[16]), "+f"(acc[17])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_19));
#else
              acc[16] = __fmaf_rn(w_t[0], x_values[16], acc[16]);
              acc[17] = __fmaf_rn(w_t[1], x_values[16], acc[17]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_20;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_20) : "f"(x_values[17]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[16]), "+f"(acc[17])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_20));
#else
              acc[16] = __fmaf_rn(w_t[2], x_values[17], acc[16]);
              acc[17] = __fmaf_rn(w_t[3], x_values[17], acc[17]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_21;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_21) : "f"(x_values[18]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[16]), "+f"(acc[17])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_21));
#else
              acc[16] = __fmaf_rn(w_t[4], x_values[18], acc[16]);
              acc[17] = __fmaf_rn(w_t[5], x_values[18], acc[17]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_22;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_22) : "f"(x_values[19]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[16]), "+f"(acc[17])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_22));
#else
              acc[16] = __fmaf_rn(w_t[6], x_values[19], acc[16]);
              acc[17] = __fmaf_rn(w_t[7], x_values[19], acc[17]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_23;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_23) : "f"(x_values[20]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[16]), "+f"(acc[17])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_23));
#else
              acc[16] = __fmaf_rn(w_t[8], x_values[20], acc[16]);
              acc[17] = __fmaf_rn(w_t[9], x_values[20], acc[17]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_24;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_24) : "f"(x_values[21]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[16]), "+f"(acc[17])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_24));
#else
              acc[16] = __fmaf_rn(w_t[10], x_values[21], acc[16]);
              acc[17] = __fmaf_rn(w_t[11], x_values[21], acc[17]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_25;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_25) : "f"(x_values[22]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[16]), "+f"(acc[17])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_25));
#else
              acc[16] = __fmaf_rn(w_t[12], x_values[22], acc[16]);
              acc[17] = __fmaf_rn(w_t[13], x_values[22], acc[17]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_26;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_26) : "f"(x_values[23]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[16]), "+f"(acc[17])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_26));
#else
              acc[16] = __fmaf_rn(w_t[14], x_values[23], acc[16]);
              acc[17] = __fmaf_rn(w_t[15], x_values[23], acc[17]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_27;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_27) : "f"(x_values[24]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[24]), "+f"(acc[25])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_27));
#else
              acc[24] = __fmaf_rn(w_t[0], x_values[24], acc[24]);
              acc[25] = __fmaf_rn(w_t[1], x_values[24], acc[25]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_28;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_28) : "f"(x_values[25]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[24]), "+f"(acc[25])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_28));
#else
              acc[24] = __fmaf_rn(w_t[2], x_values[25], acc[24]);
              acc[25] = __fmaf_rn(w_t[3], x_values[25], acc[25]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_29;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_29) : "f"(x_values[26]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[24]), "+f"(acc[25])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_29));
#else
              acc[24] = __fmaf_rn(w_t[4], x_values[26], acc[24]);
              acc[25] = __fmaf_rn(w_t[5], x_values[26], acc[25]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_30;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_30) : "f"(x_values[27]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[24]), "+f"(acc[25])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_30));
#else
              acc[24] = __fmaf_rn(w_t[6], x_values[27], acc[24]);
              acc[25] = __fmaf_rn(w_t[7], x_values[27], acc[25]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_31;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_31) : "f"(x_values[28]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[24]), "+f"(acc[25])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_31));
#else
              acc[24] = __fmaf_rn(w_t[8], x_values[28], acc[24]);
              acc[25] = __fmaf_rn(w_t[9], x_values[28], acc[25]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_32;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_32) : "f"(x_values[29]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[24]), "+f"(acc[25])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_32));
#else
              acc[24] = __fmaf_rn(w_t[10], x_values[29], acc[24]);
              acc[25] = __fmaf_rn(w_t[11], x_values[29], acc[25]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_33;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_33) : "f"(x_values[30]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[24]), "+f"(acc[25])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_33));
#else
              acc[24] = __fmaf_rn(w_t[12], x_values[30], acc[24]);
              acc[25] = __fmaf_rn(w_t[13], x_values[30], acc[25]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_34;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_34) : "f"(x_values[31]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[24]), "+f"(acc[25])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_34));
#else
              acc[24] = __fmaf_rn(w_t[14], x_values[31], acc[24]);
              acc[25] = __fmaf_rn(w_t[15], x_values[31], acc[25]);
#endif
            }
            {
              uint32_t _uv4_35_0;
              uint32_t _uv4_35_1;
              uint32_t _uv4_35_2;
              uint32_t _uv4_35_3;
              asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                           : "=r"(_uv4_35_0), "=r"(_uv4_35_1), "=r"(_uv4_35_2), "=r"(_uv4_35_3)
                           : "l"((const void*)(reinterpret_cast<const unsigned int*>(lora_a_raw) +
                                               (weight_words_it + (long long)(2 * hidden_words) +
                                                (long long)k_words)))
                           : "memory");
              w_car[0 + 0] = _uv4_35_0;
              w_car[0 + 1] = _uv4_35_1;
              w_car[0 + 2] = _uv4_35_2;
              w_car[0 + 3] = _uv4_35_3;
            }
            w_t[0] = __uint_as_float(w_car[0] << 16);
            w_t[2] = __uint_as_float(w_car[0] & 4294901760u);
            w_t[4] = __uint_as_float(w_car[1] << 16);
            w_t[6] = __uint_as_float(w_car[1] & 4294901760u);
            w_t[8] = __uint_as_float(w_car[2] << 16);
            w_t[10] = __uint_as_float(w_car[2] & 4294901760u);
            w_t[12] = __uint_as_float(w_car[3] << 16);
            w_t[14] = __uint_as_float(w_car[3] & 4294901760u);
            {
              uint32_t _uv4_36_0;
              uint32_t _uv4_36_1;
              uint32_t _uv4_36_2;
              uint32_t _uv4_36_3;
              asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                           : "=r"(_uv4_36_0), "=r"(_uv4_36_1), "=r"(_uv4_36_2), "=r"(_uv4_36_3)
                           : "l"((const void*)(reinterpret_cast<const unsigned int*>(lora_a_raw) +
                                               (weight_words_it + (long long)(3 * hidden_words) +
                                                (long long)k_words)))
                           : "memory");
              w_car[0 + 0] = _uv4_36_0;
              w_car[0 + 1] = _uv4_36_1;
              w_car[0 + 2] = _uv4_36_2;
              w_car[0 + 3] = _uv4_36_3;
            }
            w_t[1] = __uint_as_float(w_car[0] << 16);
            w_t[3] = __uint_as_float(w_car[0] & 4294901760u);
            w_t[5] = __uint_as_float(w_car[1] << 16);
            w_t[7] = __uint_as_float(w_car[1] & 4294901760u);
            w_t[9] = __uint_as_float(w_car[2] << 16);
            w_t[11] = __uint_as_float(w_car[2] & 4294901760u);
            w_t[13] = __uint_as_float(w_car[3] << 16);
            w_t[15] = __uint_as_float(w_car[3] & 4294901760u);
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_37;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_37) : "f"(x_values[0]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[2]), "+f"(acc[3])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_37));
#else
              acc[2] = __fmaf_rn(w_t[0], x_values[0], acc[2]);
              acc[3] = __fmaf_rn(w_t[1], x_values[0], acc[3]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_38;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_38) : "f"(x_values[1]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[2]), "+f"(acc[3])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_38));
#else
              acc[2] = __fmaf_rn(w_t[2], x_values[1], acc[2]);
              acc[3] = __fmaf_rn(w_t[3], x_values[1], acc[3]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_39;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_39) : "f"(x_values[2]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[2]), "+f"(acc[3])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_39));
#else
              acc[2] = __fmaf_rn(w_t[4], x_values[2], acc[2]);
              acc[3] = __fmaf_rn(w_t[5], x_values[2], acc[3]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_40;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_40) : "f"(x_values[3]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[2]), "+f"(acc[3])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_40));
#else
              acc[2] = __fmaf_rn(w_t[6], x_values[3], acc[2]);
              acc[3] = __fmaf_rn(w_t[7], x_values[3], acc[3]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_41;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_41) : "f"(x_values[4]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[2]), "+f"(acc[3])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_41));
#else
              acc[2] = __fmaf_rn(w_t[8], x_values[4], acc[2]);
              acc[3] = __fmaf_rn(w_t[9], x_values[4], acc[3]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_42;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_42) : "f"(x_values[5]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[2]), "+f"(acc[3])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_42));
#else
              acc[2] = __fmaf_rn(w_t[10], x_values[5], acc[2]);
              acc[3] = __fmaf_rn(w_t[11], x_values[5], acc[3]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_43;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_43) : "f"(x_values[6]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[2]), "+f"(acc[3])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_43));
#else
              acc[2] = __fmaf_rn(w_t[12], x_values[6], acc[2]);
              acc[3] = __fmaf_rn(w_t[13], x_values[6], acc[3]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_44;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_44) : "f"(x_values[7]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[2]), "+f"(acc[3])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_44));
#else
              acc[2] = __fmaf_rn(w_t[14], x_values[7], acc[2]);
              acc[3] = __fmaf_rn(w_t[15], x_values[7], acc[3]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_45;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_45) : "f"(x_values[8]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[10]), "+f"(acc[11])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_45));
#else
              acc[10] = __fmaf_rn(w_t[0], x_values[8], acc[10]);
              acc[11] = __fmaf_rn(w_t[1], x_values[8], acc[11]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_46;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_46) : "f"(x_values[9]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[10]), "+f"(acc[11])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_46));
#else
              acc[10] = __fmaf_rn(w_t[2], x_values[9], acc[10]);
              acc[11] = __fmaf_rn(w_t[3], x_values[9], acc[11]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_47;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_47) : "f"(x_values[10]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[10]), "+f"(acc[11])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_47));
#else
              acc[10] = __fmaf_rn(w_t[4], x_values[10], acc[10]);
              acc[11] = __fmaf_rn(w_t[5], x_values[10], acc[11]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_48;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_48) : "f"(x_values[11]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[10]), "+f"(acc[11])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_48));
#else
              acc[10] = __fmaf_rn(w_t[6], x_values[11], acc[10]);
              acc[11] = __fmaf_rn(w_t[7], x_values[11], acc[11]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_49;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_49) : "f"(x_values[12]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[10]), "+f"(acc[11])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_49));
#else
              acc[10] = __fmaf_rn(w_t[8], x_values[12], acc[10]);
              acc[11] = __fmaf_rn(w_t[9], x_values[12], acc[11]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_50;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_50) : "f"(x_values[13]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[10]), "+f"(acc[11])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_50));
#else
              acc[10] = __fmaf_rn(w_t[10], x_values[13], acc[10]);
              acc[11] = __fmaf_rn(w_t[11], x_values[13], acc[11]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_51;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_51) : "f"(x_values[14]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[10]), "+f"(acc[11])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_51));
#else
              acc[10] = __fmaf_rn(w_t[12], x_values[14], acc[10]);
              acc[11] = __fmaf_rn(w_t[13], x_values[14], acc[11]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_52;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_52) : "f"(x_values[15]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[10]), "+f"(acc[11])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_52));
#else
              acc[10] = __fmaf_rn(w_t[14], x_values[15], acc[10]);
              acc[11] = __fmaf_rn(w_t[15], x_values[15], acc[11]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_53;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_53) : "f"(x_values[16]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[18]), "+f"(acc[19])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_53));
#else
              acc[18] = __fmaf_rn(w_t[0], x_values[16], acc[18]);
              acc[19] = __fmaf_rn(w_t[1], x_values[16], acc[19]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_54;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_54) : "f"(x_values[17]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[18]), "+f"(acc[19])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_54));
#else
              acc[18] = __fmaf_rn(w_t[2], x_values[17], acc[18]);
              acc[19] = __fmaf_rn(w_t[3], x_values[17], acc[19]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_55;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_55) : "f"(x_values[18]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[18]), "+f"(acc[19])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_55));
#else
              acc[18] = __fmaf_rn(w_t[4], x_values[18], acc[18]);
              acc[19] = __fmaf_rn(w_t[5], x_values[18], acc[19]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_56;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_56) : "f"(x_values[19]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[18]), "+f"(acc[19])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_56));
#else
              acc[18] = __fmaf_rn(w_t[6], x_values[19], acc[18]);
              acc[19] = __fmaf_rn(w_t[7], x_values[19], acc[19]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_57;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_57) : "f"(x_values[20]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[18]), "+f"(acc[19])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_57));
#else
              acc[18] = __fmaf_rn(w_t[8], x_values[20], acc[18]);
              acc[19] = __fmaf_rn(w_t[9], x_values[20], acc[19]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_58;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_58) : "f"(x_values[21]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[18]), "+f"(acc[19])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_58));
#else
              acc[18] = __fmaf_rn(w_t[10], x_values[21], acc[18]);
              acc[19] = __fmaf_rn(w_t[11], x_values[21], acc[19]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_59;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_59) : "f"(x_values[22]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[18]), "+f"(acc[19])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_59));
#else
              acc[18] = __fmaf_rn(w_t[12], x_values[22], acc[18]);
              acc[19] = __fmaf_rn(w_t[13], x_values[22], acc[19]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_60;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_60) : "f"(x_values[23]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[18]), "+f"(acc[19])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_60));
#else
              acc[18] = __fmaf_rn(w_t[14], x_values[23], acc[18]);
              acc[19] = __fmaf_rn(w_t[15], x_values[23], acc[19]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_61;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_61) : "f"(x_values[24]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[26]), "+f"(acc[27])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_61));
#else
              acc[26] = __fmaf_rn(w_t[0], x_values[24], acc[26]);
              acc[27] = __fmaf_rn(w_t[1], x_values[24], acc[27]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_62;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_62) : "f"(x_values[25]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[26]), "+f"(acc[27])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_62));
#else
              acc[26] = __fmaf_rn(w_t[2], x_values[25], acc[26]);
              acc[27] = __fmaf_rn(w_t[3], x_values[25], acc[27]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_63;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_63) : "f"(x_values[26]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[26]), "+f"(acc[27])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_63));
#else
              acc[26] = __fmaf_rn(w_t[4], x_values[26], acc[26]);
              acc[27] = __fmaf_rn(w_t[5], x_values[26], acc[27]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_64;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_64) : "f"(x_values[27]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[26]), "+f"(acc[27])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_64));
#else
              acc[26] = __fmaf_rn(w_t[6], x_values[27], acc[26]);
              acc[27] = __fmaf_rn(w_t[7], x_values[27], acc[27]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_65;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_65) : "f"(x_values[28]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[26]), "+f"(acc[27])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_65));
#else
              acc[26] = __fmaf_rn(w_t[8], x_values[28], acc[26]);
              acc[27] = __fmaf_rn(w_t[9], x_values[28], acc[27]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_66;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_66) : "f"(x_values[29]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[26]), "+f"(acc[27])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_66));
#else
              acc[26] = __fmaf_rn(w_t[10], x_values[29], acc[26]);
              acc[27] = __fmaf_rn(w_t[11], x_values[29], acc[27]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_67;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_67) : "f"(x_values[30]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[26]), "+f"(acc[27])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_67));
#else
              acc[26] = __fmaf_rn(w_t[12], x_values[30], acc[26]);
              acc[27] = __fmaf_rn(w_t[13], x_values[30], acc[27]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_68;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_68) : "f"(x_values[31]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[26]), "+f"(acc[27])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_68));
#else
              acc[26] = __fmaf_rn(w_t[14], x_values[31], acc[26]);
              acc[27] = __fmaf_rn(w_t[15], x_values[31], acc[27]);
#endif
            }
            {
              uint32_t _uv4_69_0;
              uint32_t _uv4_69_1;
              uint32_t _uv4_69_2;
              uint32_t _uv4_69_3;
              asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                           : "=r"(_uv4_69_0), "=r"(_uv4_69_1), "=r"(_uv4_69_2), "=r"(_uv4_69_3)
                           : "l"((const void*)(reinterpret_cast<const unsigned int*>(lora_a_raw) +
                                               (weight_words_it + (long long)(4 * hidden_words) +
                                                (long long)k_words)))
                           : "memory");
              w_car[0 + 0] = _uv4_69_0;
              w_car[0 + 1] = _uv4_69_1;
              w_car[0 + 2] = _uv4_69_2;
              w_car[0 + 3] = _uv4_69_3;
            }
            w_t[0] = __uint_as_float(w_car[0] << 16);
            w_t[2] = __uint_as_float(w_car[0] & 4294901760u);
            w_t[4] = __uint_as_float(w_car[1] << 16);
            w_t[6] = __uint_as_float(w_car[1] & 4294901760u);
            w_t[8] = __uint_as_float(w_car[2] << 16);
            w_t[10] = __uint_as_float(w_car[2] & 4294901760u);
            w_t[12] = __uint_as_float(w_car[3] << 16);
            w_t[14] = __uint_as_float(w_car[3] & 4294901760u);
            {
              uint32_t _uv4_70_0;
              uint32_t _uv4_70_1;
              uint32_t _uv4_70_2;
              uint32_t _uv4_70_3;
              asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                           : "=r"(_uv4_70_0), "=r"(_uv4_70_1), "=r"(_uv4_70_2), "=r"(_uv4_70_3)
                           : "l"((const void*)(reinterpret_cast<const unsigned int*>(lora_a_raw) +
                                               (weight_words_it + (long long)(5 * hidden_words) +
                                                (long long)k_words)))
                           : "memory");
              w_car[0 + 0] = _uv4_70_0;
              w_car[0 + 1] = _uv4_70_1;
              w_car[0 + 2] = _uv4_70_2;
              w_car[0 + 3] = _uv4_70_3;
            }
            w_t[1] = __uint_as_float(w_car[0] << 16);
            w_t[3] = __uint_as_float(w_car[0] & 4294901760u);
            w_t[5] = __uint_as_float(w_car[1] << 16);
            w_t[7] = __uint_as_float(w_car[1] & 4294901760u);
            w_t[9] = __uint_as_float(w_car[2] << 16);
            w_t[11] = __uint_as_float(w_car[2] & 4294901760u);
            w_t[13] = __uint_as_float(w_car[3] << 16);
            w_t[15] = __uint_as_float(w_car[3] & 4294901760u);
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_71;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_71) : "f"(x_values[0]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[4]), "+f"(acc[5])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_71));
#else
              acc[4] = __fmaf_rn(w_t[0], x_values[0], acc[4]);
              acc[5] = __fmaf_rn(w_t[1], x_values[0], acc[5]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_72;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_72) : "f"(x_values[1]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[4]), "+f"(acc[5])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_72));
#else
              acc[4] = __fmaf_rn(w_t[2], x_values[1], acc[4]);
              acc[5] = __fmaf_rn(w_t[3], x_values[1], acc[5]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_73;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_73) : "f"(x_values[2]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[4]), "+f"(acc[5])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_73));
#else
              acc[4] = __fmaf_rn(w_t[4], x_values[2], acc[4]);
              acc[5] = __fmaf_rn(w_t[5], x_values[2], acc[5]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_74;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_74) : "f"(x_values[3]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[4]), "+f"(acc[5])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_74));
#else
              acc[4] = __fmaf_rn(w_t[6], x_values[3], acc[4]);
              acc[5] = __fmaf_rn(w_t[7], x_values[3], acc[5]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_75;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_75) : "f"(x_values[4]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[4]), "+f"(acc[5])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_75));
#else
              acc[4] = __fmaf_rn(w_t[8], x_values[4], acc[4]);
              acc[5] = __fmaf_rn(w_t[9], x_values[4], acc[5]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_76;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_76) : "f"(x_values[5]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[4]), "+f"(acc[5])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_76));
#else
              acc[4] = __fmaf_rn(w_t[10], x_values[5], acc[4]);
              acc[5] = __fmaf_rn(w_t[11], x_values[5], acc[5]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_77;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_77) : "f"(x_values[6]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[4]), "+f"(acc[5])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_77));
#else
              acc[4] = __fmaf_rn(w_t[12], x_values[6], acc[4]);
              acc[5] = __fmaf_rn(w_t[13], x_values[6], acc[5]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_78;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_78) : "f"(x_values[7]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[4]), "+f"(acc[5])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_78));
#else
              acc[4] = __fmaf_rn(w_t[14], x_values[7], acc[4]);
              acc[5] = __fmaf_rn(w_t[15], x_values[7], acc[5]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_79;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_79) : "f"(x_values[8]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[12]), "+f"(acc[13])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_79));
#else
              acc[12] = __fmaf_rn(w_t[0], x_values[8], acc[12]);
              acc[13] = __fmaf_rn(w_t[1], x_values[8], acc[13]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_80;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_80) : "f"(x_values[9]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[12]), "+f"(acc[13])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_80));
#else
              acc[12] = __fmaf_rn(w_t[2], x_values[9], acc[12]);
              acc[13] = __fmaf_rn(w_t[3], x_values[9], acc[13]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_81;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_81) : "f"(x_values[10]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[12]), "+f"(acc[13])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_81));
#else
              acc[12] = __fmaf_rn(w_t[4], x_values[10], acc[12]);
              acc[13] = __fmaf_rn(w_t[5], x_values[10], acc[13]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_82;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_82) : "f"(x_values[11]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[12]), "+f"(acc[13])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_82));
#else
              acc[12] = __fmaf_rn(w_t[6], x_values[11], acc[12]);
              acc[13] = __fmaf_rn(w_t[7], x_values[11], acc[13]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_83;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_83) : "f"(x_values[12]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[12]), "+f"(acc[13])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_83));
#else
              acc[12] = __fmaf_rn(w_t[8], x_values[12], acc[12]);
              acc[13] = __fmaf_rn(w_t[9], x_values[12], acc[13]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_84;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_84) : "f"(x_values[13]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[12]), "+f"(acc[13])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_84));
#else
              acc[12] = __fmaf_rn(w_t[10], x_values[13], acc[12]);
              acc[13] = __fmaf_rn(w_t[11], x_values[13], acc[13]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_85;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_85) : "f"(x_values[14]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[12]), "+f"(acc[13])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_85));
#else
              acc[12] = __fmaf_rn(w_t[12], x_values[14], acc[12]);
              acc[13] = __fmaf_rn(w_t[13], x_values[14], acc[13]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_86;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_86) : "f"(x_values[15]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[12]), "+f"(acc[13])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_86));
#else
              acc[12] = __fmaf_rn(w_t[14], x_values[15], acc[12]);
              acc[13] = __fmaf_rn(w_t[15], x_values[15], acc[13]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_87;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_87) : "f"(x_values[16]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[20]), "+f"(acc[21])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_87));
#else
              acc[20] = __fmaf_rn(w_t[0], x_values[16], acc[20]);
              acc[21] = __fmaf_rn(w_t[1], x_values[16], acc[21]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_88;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_88) : "f"(x_values[17]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[20]), "+f"(acc[21])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_88));
#else
              acc[20] = __fmaf_rn(w_t[2], x_values[17], acc[20]);
              acc[21] = __fmaf_rn(w_t[3], x_values[17], acc[21]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_89;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_89) : "f"(x_values[18]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[20]), "+f"(acc[21])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_89));
#else
              acc[20] = __fmaf_rn(w_t[4], x_values[18], acc[20]);
              acc[21] = __fmaf_rn(w_t[5], x_values[18], acc[21]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_90;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_90) : "f"(x_values[19]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[20]), "+f"(acc[21])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_90));
#else
              acc[20] = __fmaf_rn(w_t[6], x_values[19], acc[20]);
              acc[21] = __fmaf_rn(w_t[7], x_values[19], acc[21]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_91;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_91) : "f"(x_values[20]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[20]), "+f"(acc[21])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_91));
#else
              acc[20] = __fmaf_rn(w_t[8], x_values[20], acc[20]);
              acc[21] = __fmaf_rn(w_t[9], x_values[20], acc[21]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_92;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_92) : "f"(x_values[21]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[20]), "+f"(acc[21])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_92));
#else
              acc[20] = __fmaf_rn(w_t[10], x_values[21], acc[20]);
              acc[21] = __fmaf_rn(w_t[11], x_values[21], acc[21]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_93;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_93) : "f"(x_values[22]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[20]), "+f"(acc[21])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_93));
#else
              acc[20] = __fmaf_rn(w_t[12], x_values[22], acc[20]);
              acc[21] = __fmaf_rn(w_t[13], x_values[22], acc[21]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_94;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_94) : "f"(x_values[23]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[20]), "+f"(acc[21])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_94));
#else
              acc[20] = __fmaf_rn(w_t[14], x_values[23], acc[20]);
              acc[21] = __fmaf_rn(w_t[15], x_values[23], acc[21]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_95;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_95) : "f"(x_values[24]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[28]), "+f"(acc[29])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_95));
#else
              acc[28] = __fmaf_rn(w_t[0], x_values[24], acc[28]);
              acc[29] = __fmaf_rn(w_t[1], x_values[24], acc[29]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_96;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_96) : "f"(x_values[25]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[28]), "+f"(acc[29])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_96));
#else
              acc[28] = __fmaf_rn(w_t[2], x_values[25], acc[28]);
              acc[29] = __fmaf_rn(w_t[3], x_values[25], acc[29]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_97;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_97) : "f"(x_values[26]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[28]), "+f"(acc[29])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_97));
#else
              acc[28] = __fmaf_rn(w_t[4], x_values[26], acc[28]);
              acc[29] = __fmaf_rn(w_t[5], x_values[26], acc[29]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_98;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_98) : "f"(x_values[27]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[28]), "+f"(acc[29])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_98));
#else
              acc[28] = __fmaf_rn(w_t[6], x_values[27], acc[28]);
              acc[29] = __fmaf_rn(w_t[7], x_values[27], acc[29]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_99;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_99) : "f"(x_values[28]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[28]), "+f"(acc[29])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_99));
#else
              acc[28] = __fmaf_rn(w_t[8], x_values[28], acc[28]);
              acc[29] = __fmaf_rn(w_t[9], x_values[28], acc[29]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_100;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_100) : "f"(x_values[29]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[28]), "+f"(acc[29])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_100));
#else
              acc[28] = __fmaf_rn(w_t[10], x_values[29], acc[28]);
              acc[29] = __fmaf_rn(w_t[11], x_values[29], acc[29]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_101;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_101) : "f"(x_values[30]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[28]), "+f"(acc[29])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_101));
#else
              acc[28] = __fmaf_rn(w_t[12], x_values[30], acc[28]);
              acc[29] = __fmaf_rn(w_t[13], x_values[30], acc[29]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_102;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_102) : "f"(x_values[31]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[28]), "+f"(acc[29])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_102));
#else
              acc[28] = __fmaf_rn(w_t[14], x_values[31], acc[28]);
              acc[29] = __fmaf_rn(w_t[15], x_values[31], acc[29]);
#endif
            }
            {
              uint32_t _uv4_103_0;
              uint32_t _uv4_103_1;
              uint32_t _uv4_103_2;
              uint32_t _uv4_103_3;
              asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                           : "=r"(_uv4_103_0), "=r"(_uv4_103_1), "=r"(_uv4_103_2), "=r"(_uv4_103_3)
                           : "l"((const void*)(reinterpret_cast<const unsigned int*>(lora_a_raw) +
                                               (weight_words_it + (long long)(6 * hidden_words) +
                                                (long long)k_words)))
                           : "memory");
              w_car[0 + 0] = _uv4_103_0;
              w_car[0 + 1] = _uv4_103_1;
              w_car[0 + 2] = _uv4_103_2;
              w_car[0 + 3] = _uv4_103_3;
            }
            w_t[0] = __uint_as_float(w_car[0] << 16);
            w_t[2] = __uint_as_float(w_car[0] & 4294901760u);
            w_t[4] = __uint_as_float(w_car[1] << 16);
            w_t[6] = __uint_as_float(w_car[1] & 4294901760u);
            w_t[8] = __uint_as_float(w_car[2] << 16);
            w_t[10] = __uint_as_float(w_car[2] & 4294901760u);
            w_t[12] = __uint_as_float(w_car[3] << 16);
            w_t[14] = __uint_as_float(w_car[3] & 4294901760u);
            {
              uint32_t _uv4_104_0;
              uint32_t _uv4_104_1;
              uint32_t _uv4_104_2;
              uint32_t _uv4_104_3;
              asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                           : "=r"(_uv4_104_0), "=r"(_uv4_104_1), "=r"(_uv4_104_2), "=r"(_uv4_104_3)
                           : "l"((const void*)(reinterpret_cast<const unsigned int*>(lora_a_raw) +
                                               (weight_words_it + (long long)(7 * hidden_words) +
                                                (long long)k_words)))
                           : "memory");
              w_car[0 + 0] = _uv4_104_0;
              w_car[0 + 1] = _uv4_104_1;
              w_car[0 + 2] = _uv4_104_2;
              w_car[0 + 3] = _uv4_104_3;
            }
            w_t[1] = __uint_as_float(w_car[0] << 16);
            w_t[3] = __uint_as_float(w_car[0] & 4294901760u);
            w_t[5] = __uint_as_float(w_car[1] << 16);
            w_t[7] = __uint_as_float(w_car[1] & 4294901760u);
            w_t[9] = __uint_as_float(w_car[2] << 16);
            w_t[11] = __uint_as_float(w_car[2] & 4294901760u);
            w_t[13] = __uint_as_float(w_car[3] << 16);
            w_t[15] = __uint_as_float(w_car[3] & 4294901760u);
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_105;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_105) : "f"(x_values[0]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[6]), "+f"(acc[7])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_105));
#else
              acc[6] = __fmaf_rn(w_t[0], x_values[0], acc[6]);
              acc[7] = __fmaf_rn(w_t[1], x_values[0], acc[7]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_106;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_106) : "f"(x_values[1]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[6]), "+f"(acc[7])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_106));
#else
              acc[6] = __fmaf_rn(w_t[2], x_values[1], acc[6]);
              acc[7] = __fmaf_rn(w_t[3], x_values[1], acc[7]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_107;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_107) : "f"(x_values[2]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[6]), "+f"(acc[7])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_107));
#else
              acc[6] = __fmaf_rn(w_t[4], x_values[2], acc[6]);
              acc[7] = __fmaf_rn(w_t[5], x_values[2], acc[7]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_108;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_108) : "f"(x_values[3]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[6]), "+f"(acc[7])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_108));
#else
              acc[6] = __fmaf_rn(w_t[6], x_values[3], acc[6]);
              acc[7] = __fmaf_rn(w_t[7], x_values[3], acc[7]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_109;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_109) : "f"(x_values[4]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[6]), "+f"(acc[7])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_109));
#else
              acc[6] = __fmaf_rn(w_t[8], x_values[4], acc[6]);
              acc[7] = __fmaf_rn(w_t[9], x_values[4], acc[7]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_110;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_110) : "f"(x_values[5]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[6]), "+f"(acc[7])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_110));
#else
              acc[6] = __fmaf_rn(w_t[10], x_values[5], acc[6]);
              acc[7] = __fmaf_rn(w_t[11], x_values[5], acc[7]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_111;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_111) : "f"(x_values[6]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[6]), "+f"(acc[7])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_111));
#else
              acc[6] = __fmaf_rn(w_t[12], x_values[6], acc[6]);
              acc[7] = __fmaf_rn(w_t[13], x_values[6], acc[7]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_112;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_112) : "f"(x_values[7]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[6]), "+f"(acc[7])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_112));
#else
              acc[6] = __fmaf_rn(w_t[14], x_values[7], acc[6]);
              acc[7] = __fmaf_rn(w_t[15], x_values[7], acc[7]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_113;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_113) : "f"(x_values[8]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[14]), "+f"(acc[15])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_113));
#else
              acc[14] = __fmaf_rn(w_t[0], x_values[8], acc[14]);
              acc[15] = __fmaf_rn(w_t[1], x_values[8], acc[15]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_114;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_114) : "f"(x_values[9]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[14]), "+f"(acc[15])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_114));
#else
              acc[14] = __fmaf_rn(w_t[2], x_values[9], acc[14]);
              acc[15] = __fmaf_rn(w_t[3], x_values[9], acc[15]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_115;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_115) : "f"(x_values[10]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[14]), "+f"(acc[15])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_115));
#else
              acc[14] = __fmaf_rn(w_t[4], x_values[10], acc[14]);
              acc[15] = __fmaf_rn(w_t[5], x_values[10], acc[15]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_116;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_116) : "f"(x_values[11]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[14]), "+f"(acc[15])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_116));
#else
              acc[14] = __fmaf_rn(w_t[6], x_values[11], acc[14]);
              acc[15] = __fmaf_rn(w_t[7], x_values[11], acc[15]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_117;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_117) : "f"(x_values[12]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[14]), "+f"(acc[15])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_117));
#else
              acc[14] = __fmaf_rn(w_t[8], x_values[12], acc[14]);
              acc[15] = __fmaf_rn(w_t[9], x_values[12], acc[15]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_118;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_118) : "f"(x_values[13]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[14]), "+f"(acc[15])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_118));
#else
              acc[14] = __fmaf_rn(w_t[10], x_values[13], acc[14]);
              acc[15] = __fmaf_rn(w_t[11], x_values[13], acc[15]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_119;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_119) : "f"(x_values[14]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[14]), "+f"(acc[15])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_119));
#else
              acc[14] = __fmaf_rn(w_t[12], x_values[14], acc[14]);
              acc[15] = __fmaf_rn(w_t[13], x_values[14], acc[15]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_120;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_120) : "f"(x_values[15]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[14]), "+f"(acc[15])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_120));
#else
              acc[14] = __fmaf_rn(w_t[14], x_values[15], acc[14]);
              acc[15] = __fmaf_rn(w_t[15], x_values[15], acc[15]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_121;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_121) : "f"(x_values[16]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[22]), "+f"(acc[23])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_121));
#else
              acc[22] = __fmaf_rn(w_t[0], x_values[16], acc[22]);
              acc[23] = __fmaf_rn(w_t[1], x_values[16], acc[23]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_122;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_122) : "f"(x_values[17]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[22]), "+f"(acc[23])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_122));
#else
              acc[22] = __fmaf_rn(w_t[2], x_values[17], acc[22]);
              acc[23] = __fmaf_rn(w_t[3], x_values[17], acc[23]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_123;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_123) : "f"(x_values[18]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[22]), "+f"(acc[23])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_123));
#else
              acc[22] = __fmaf_rn(w_t[4], x_values[18], acc[22]);
              acc[23] = __fmaf_rn(w_t[5], x_values[18], acc[23]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_124;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_124) : "f"(x_values[19]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[22]), "+f"(acc[23])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_124));
#else
              acc[22] = __fmaf_rn(w_t[6], x_values[19], acc[22]);
              acc[23] = __fmaf_rn(w_t[7], x_values[19], acc[23]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_125;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_125) : "f"(x_values[20]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[22]), "+f"(acc[23])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_125));
#else
              acc[22] = __fmaf_rn(w_t[8], x_values[20], acc[22]);
              acc[23] = __fmaf_rn(w_t[9], x_values[20], acc[23]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_126;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_126) : "f"(x_values[21]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[22]), "+f"(acc[23])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_126));
#else
              acc[22] = __fmaf_rn(w_t[10], x_values[21], acc[22]);
              acc[23] = __fmaf_rn(w_t[11], x_values[21], acc[23]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_127;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_127) : "f"(x_values[22]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[22]), "+f"(acc[23])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_127));
#else
              acc[22] = __fmaf_rn(w_t[12], x_values[22], acc[22]);
              acc[23] = __fmaf_rn(w_t[13], x_values[22], acc[23]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_128;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_128) : "f"(x_values[23]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[22]), "+f"(acc[23])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_128));
#else
              acc[22] = __fmaf_rn(w_t[14], x_values[23], acc[22]);
              acc[23] = __fmaf_rn(w_t[15], x_values[23], acc[23]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_129;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_129) : "f"(x_values[24]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[30]), "+f"(acc[31])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_129));
#else
              acc[30] = __fmaf_rn(w_t[0], x_values[24], acc[30]);
              acc[31] = __fmaf_rn(w_t[1], x_values[24], acc[31]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_130;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_130) : "f"(x_values[25]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[30]), "+f"(acc[31])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_130));
#else
              acc[30] = __fmaf_rn(w_t[2], x_values[25], acc[30]);
              acc[31] = __fmaf_rn(w_t[3], x_values[25], acc[31]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_131;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_131) : "f"(x_values[26]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[30]), "+f"(acc[31])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_131));
#else
              acc[30] = __fmaf_rn(w_t[4], x_values[26], acc[30]);
              acc[31] = __fmaf_rn(w_t[5], x_values[26], acc[31]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_132;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_132) : "f"(x_values[27]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[30]), "+f"(acc[31])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_132));
#else
              acc[30] = __fmaf_rn(w_t[6], x_values[27], acc[30]);
              acc[31] = __fmaf_rn(w_t[7], x_values[27], acc[31]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_133;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_133) : "f"(x_values[28]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[30]), "+f"(acc[31])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_133));
#else
              acc[30] = __fmaf_rn(w_t[8], x_values[28], acc[30]);
              acc[31] = __fmaf_rn(w_t[9], x_values[28], acc[31]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_134;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_134) : "f"(x_values[29]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[30]), "+f"(acc[31])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_134));
#else
              acc[30] = __fmaf_rn(w_t[10], x_values[29], acc[30]);
              acc[31] = __fmaf_rn(w_t[11], x_values[29], acc[31]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_135;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_135) : "f"(x_values[30]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[30]), "+f"(acc[31])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_135));
#else
              acc[30] = __fmaf_rn(w_t[12], x_values[30], acc[30]);
              acc[31] = __fmaf_rn(w_t[13], x_values[30], acc[31]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_136;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_136) : "f"(x_values[31]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[30]), "+f"(acc[31])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_136));
#else
              acc[30] = __fmaf_rn(w_t[14], x_values[31], acc[30]);
              acc[31] = __fmaf_rn(w_t[15], x_values[31], acc[31]);
#endif
            }
          }
        }
#pragma unroll
        for (int i = 0; i < 16; i++) {
          float _shfl_xor_0 =
              __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) != 0) ? acc[i] : acc[i + 16]), 16);
          red_a[i] = (((lane_0 & 16) != 0) ? acc[i + 16] : acc[i]) + _shfl_xor_0;
        }
#pragma unroll
        for (int i_1 = 0; i_1 < 8; i_1++) {
          float _shfl_xor_1 =
              __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) != 0) ? red_a[i_1] : red_a[i_1 + 8]), 8);
          red_b[i_1] = (((lane_0 & 8) != 0) ? red_a[i_1 + 8] : red_a[i_1]) + _shfl_xor_1;
        }
#pragma unroll
        for (int i_2 = 0; i_2 < 4; i_2++) {
          float _shfl_xor_2 =
              __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 4) != 0) ? red_b[i_2] : red_b[i_2 + 4]), 4);
          red_c[i_2] = (((lane_0 & 4) != 0) ? red_b[i_2 + 4] : red_b[i_2]) + _shfl_xor_2;
        }
#pragma unroll
        for (int i_3 = 0; i_3 < 2; i_3++) {
          float _shfl_xor_3 =
              __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 2) != 0) ? red_c[i_3] : red_c[i_3 + 2]), 2);
          red_d[i_3] = (((lane_0 & 2) != 0) ? red_c[i_3 + 2] : red_c[i_3]) + _shfl_xor_3;
        }
#pragma unroll
        for (int i_4 = 0; i_4 < 1; i_4++) {
          float _shfl_xor_4 =
              __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 1) != 0) ? red_d[i_4] : red_d[i_4 + 1]), 1);
          red_e[i_4] = (((lane_0 & 1) != 0) ? red_d[i_4 + 1] : red_d[i_4]) + _shfl_xor_4;
        }
#pragma unroll
        for (int i_5 = 0; i_5 < 1; i_5++) {
          warp_partials[(lane_0 + i_5) * 4 + warp] = red_e[i_5];
        }
        __syncthreads();
        if (tid < 32) {
          float owned_accum = 0.0f;
#pragma unroll
          for (int source_warp = 0; source_warp < 4; source_warp++) {
            owned_accum += warp_partials[tid * 4 + source_warp];
          }
          int owner_j = tid / 8;
          int owner_rr = tid % 8;
          if (owner_j < count) {
            int owner_route = (int)reinterpret_cast<const unsigned int*>(
                workspace_raw)[off_sorted_routes + start + owner_j];
            *(reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<__nv_bfloat16*>(shrink_out_raw) +
                                               (owner_route * 32 + rank_base_it + owner_rr)) +
              (0)) = __float2bfloat16_rn(owned_accum);
          }
        }
        __syncthreads();
        rank_base_it = rank_base_it + 8;
        weight_words_it = weight_words_it + weight_words_step;
      }
    }
  }
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

}  // extern "C"

#undef BLACKWELL_INF
#undef NUM_MAIN_STAGES
#undef SMEM_TOTAL
#undef SMEM_WARP_PARTIALS_OFF
#undef SMEM_WARP_PARTIALS_STAGE_BYTES
#undef SMEM_WARP_PARTIALS_STRIDE
#undef SMEM_W_RING_OFF
#undef SMEM_W_RING_STAGE_BYTES
#undef SMEM_W_RING_STRIDE
#undef SMEM_X_RING_OFF
#undef SMEM_X_RING_STAGE_BYTES
#undef SMEM_X_RING_STRIDE
#undef THREADS

#define BLACKWELL_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_ROUTE_SMEM_OFF 0
#define SMEM_ROUTE_SMEM_STAGE_BYTES 64
#define SMEM_ROUTE_SMEM_STRIDE 64
#define SMEM_S_SMEM_OFF 64
#define SMEM_S_SMEM_STAGE_BYTES 1024
#define SMEM_S_SMEM_STRIDE 1024
#define SMEM_W_SMEM_OFF 0
#define SMEM_W_SMEM_STAGE_BYTES 16
#define SMEM_W_SMEM_STRIDE 16
#define SMEM_W_U32_OFF 0
#define SMEM_W_U32_STAGE_BYTES 16
#define SMEM_W_U32_STRIDE 16
#define SMEM_TOTAL 1152
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(256, 1) void kernel_flashinfer_bgmv_moe_expand_grouped_bf16_r32(
    float* __restrict__ partials_raw, uint16_t* __restrict__ shrink_raw,
    uint16_t* __restrict__ lora_b_raw, int num_pairs, int num_experts, int hidden,
    unsigned int* __restrict__ workspace_raw, int off_group_offset, int off_tile_table,
    int off_sorted_routes, int col_blocks_per_cta) {
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
  int* route_smem = reinterpret_cast<int*>(smem_raw + 0);
  const int route_smem_addr = smem + 0;
  __nv_bfloat16* s_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 64);
  const int s_smem_addr = smem + 64;
  __nv_bfloat16* w_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
  const int w_smem_addr = smem + 0;
  int* w_u32 = reinterpret_cast<int*>(smem_raw + 0);
  const int w_u32_addr = smem + 0;

  // === Task calls (dependency order) ===
  asm volatile("griddepcontrol.wait;" ::: "memory");
  int tile = blockIdx.x;
  int n_tiles = (int)reinterpret_cast<const unsigned int*>(workspace_raw)[0];
  if (tile < n_tiles) {
    int entry = (int)reinterpret_cast<const unsigned int*>(workspace_raw)[off_tile_table + tile];
    int group = entry / 65536;
    int chunk = entry % 65536;
    int start =
        (int)reinterpret_cast<const unsigned int*>(workspace_raw)[off_group_offset + group] +
        chunk * 16;
    int group_end =
        (int)reinterpret_cast<const unsigned int*>(workspace_raw)[off_group_offset + group + 1];
    int count = group_end - start;
    if (count > 16) {
      count = 16;
    }
    int lora = group / num_experts;
    int expert = group % num_experts;
    int group_id = lane / 4;
    int tig = lane % 4;
    long long weight_row_base = (long long)(lora * num_experts + expert) * (long long)hidden;
    unsigned int a_frag[16];
    unsigned int a_next[16];
    int col_blocks = (hidden + 256 - 1) / 256;
    int cb_first = blockIdx.y * col_blocks_per_cta;
    int first_base = cb_first * 256;
#pragma unroll
    for (int m_tile = 0; m_tile < 2; m_tile++) {
      int col_lo = first_base + warp * 32 + m_tile * 16 + group_id;
      int col_hi = col_lo + 8;
      long long row_lo = (weight_row_base + (long long)col_lo) * 16;
      long long row_hi = (weight_row_base + (long long)col_hi) * 16;
#pragma unroll
      for (int ks = 0; ks < 2; ks++) {
        a_frag[(m_tile * 2 + ks) * 4] = 0;
        a_frag[(m_tile * 2 + ks) * 4 + 1] = 0;
        a_frag[(m_tile * 2 + ks) * 4 + 2] = 0;
        a_frag[(m_tile * 2 + ks) * 4 + 3] = 0;
        if (col_lo < hidden) {
          a_frag[(m_tile * 2 + ks) * 4] = reinterpret_cast<const unsigned int*>(
              lora_b_raw)[row_lo + (long long)(ks * 8) + (long long)tig];
          a_frag[(m_tile * 2 + ks) * 4 + 2] = reinterpret_cast<const unsigned int*>(
              lora_b_raw)[row_lo + (long long)(ks * 8) + 4 + (long long)tig];
        }
        if (col_hi < hidden) {
          a_frag[(m_tile * 2 + ks) * 4 + 1] = reinterpret_cast<const unsigned int*>(
              lora_b_raw)[row_hi + (long long)(ks * 8) + (long long)tig];
          a_frag[(m_tile * 2 + ks) * 4 + 3] = reinterpret_cast<const unsigned int*>(
              lora_b_raw)[row_hi + (long long)(ks * 8) + 4 + (long long)tig];
        }
      }
    }
    if (tid < 16) {
      int staged_route = 0;
      if (count > tid) {
        staged_route = (int)reinterpret_cast<const unsigned int*>(
            workspace_raw)[off_sorted_routes + start + tid];
      }
      route_smem[tid] = staged_route;
    }
    __syncthreads();
    if (tid < 64) {
      int chunk_route_slot = tid / 4;
      int chunk_part = tid % 4;
      if (chunk_route_slot < count) {
        int chunk_route = route_smem[chunk_route_slot];
        asm volatile(
            "cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                s_smem_addr + (unsigned int)((chunk_route_slot * 32 + chunk_part * 8) * 2)),
            "l"(reinterpret_cast<const __nv_bfloat16*>(shrink_raw) +
                (chunk_route * 32 + chunk_part * 8)));
      }
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    float acc[16];
    unsigned int b_lo[1];
    unsigned int b_hi[1];
#pragma unroll 1
    for (int cb = 0; cb < col_blocks_per_cta; cb++) {
      int col_block = cb_first + cb;
      if (col_block < col_blocks) {
        int col_base = col_block * 256;
        int w_stage = 0;
        int next_block = col_block + 1;
        int next_base = next_block * 256;
#pragma unroll
        for (int m_tile_1 = 0; m_tile_1 < 2; m_tile_1++) {
          int n_col_lo = next_base + warp * 32 + m_tile_1 * 16 + group_id;
          int n_col_hi = n_col_lo + 8;
          long long n_row_lo = (weight_row_base + (long long)n_col_lo) * 16;
          long long n_row_hi = (weight_row_base + (long long)n_col_hi) * 16;
#pragma unroll
          for (int ks_1 = 0; ks_1 < 2; ks_1++) {
            a_next[(m_tile_1 * 2 + ks_1) * 4] = 0;
            a_next[(m_tile_1 * 2 + ks_1) * 4 + 1] = 0;
            a_next[(m_tile_1 * 2 + ks_1) * 4 + 2] = 0;
            a_next[(m_tile_1 * 2 + ks_1) * 4 + 3] = 0;
            if (cb + 1 < col_blocks_per_cta) {
              if (n_col_lo < hidden) {
                a_next[(m_tile_1 * 2 + ks_1) * 4] = reinterpret_cast<const unsigned int*>(
                    lora_b_raw)[n_row_lo + (long long)(ks_1 * 8) + (long long)tig];
                a_next[(m_tile_1 * 2 + ks_1) * 4 + 2] = reinterpret_cast<const unsigned int*>(
                    lora_b_raw)[n_row_lo + (long long)(ks_1 * 8) + 4 + (long long)tig];
              }
              if (n_col_hi < hidden) {
                a_next[(m_tile_1 * 2 + ks_1) * 4 + 1] = reinterpret_cast<const unsigned int*>(
                    lora_b_raw)[n_row_hi + (long long)(ks_1 * 8) + (long long)tig];
                a_next[(m_tile_1 * 2 + ks_1) * 4 + 3] = reinterpret_cast<const unsigned int*>(
                    lora_b_raw)[n_row_hi + (long long)(ks_1 * 8) + 4 + (long long)tig];
              }
            }
          }
        }
#pragma unroll
        for (int i = 0; i < 16; i++) {
          acc[i] = 0.0f;
        }
#pragma unroll
        for (int ks_2 = 0; ks_2 < 2; ks_2++) {
          unsigned int a0[4];
          unsigned int a1[4];
#pragma unroll
          for (int r = 0; r < 4; r++) {
            a0[r] = a_frag[ks_2 * 4 + r];
            a1[r] = a_frag[(2 + ks_2) * 4 + r];
          }
#pragma unroll
          for (int nt = 0; nt < 2; nt++) {
            int slot = nt * 8 + group_id;
            b_lo[0] = 0;
            b_hi[0] = 0;
            if (slot < count) {
              asm volatile(
                  "ld.shared.b32 %0, [%1];"
                  : "=r"(*reinterpret_cast<uint32_t*>(&b_lo[0]))
                  : "r"(s_smem_addr + (unsigned int)((slot * 32 + ks_2 * 16 + tig * 2) * 2)));
              asm volatile(
                  "ld.shared.b32 %0, [%1];"
                  : "=r"(*reinterpret_cast<uint32_t*>(&b_hi[0]))
                  : "r"(s_smem_addr + (unsigned int)((slot * 32 + ks_2 * 16 + 8 + tig * 2) * 2)));
            }
            uint32_t _mma_sync_m16n8k16_b_0[2];
            _mma_sync_m16n8k16_b_0[0] = b_lo[0];
            _mma_sync_m16n8k16_b_0[1] = b_hi[0];
#pragma unroll
            for (int m_tile_2 = 0; m_tile_2 < 2; m_tile_2++) {
              float grp[4];
#pragma unroll
              for (int i_1 = 0; i_1 < 4; i_1++) {
                grp[i_1] = 0.0f;
              }
              if (m_tile_2 == 0) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                    "%5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(grp[0]), "+f"(grp[1]), "+f"(grp[2]), "+f"(grp[3])
                    : "r"(a0[0]), "r"(a0[1]), "r"(a0[2]), "r"(a0[3]),
                      "r"(_mma_sync_m16n8k16_b_0[0]), "r"(_mma_sync_m16n8k16_b_0[1]));
              } else {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                    "%5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(grp[0]), "+f"(grp[1]), "+f"(grp[2]), "+f"(grp[3])
                    : "r"(a1[0]), "r"(a1[1]), "r"(a1[2]), "r"(a1[3]),
                      "r"(_mma_sync_m16n8k16_b_0[0]), "r"(_mma_sync_m16n8k16_b_0[1]));
              }
#pragma unroll
              for (int i_2 = 0; i_2 < 4; i_2++) {
                acc[(m_tile_2 * 2 + nt) * 4 + i_2] = acc[(m_tile_2 * 2 + nt) * 4 + i_2] + grp[i_2];
              }
            }
          }
        }
#pragma unroll
        for (int nt_1 = 0; nt_1 < 2; nt_1++) {
          int route_a = nt_1 * 8 + tig * 2;
          int route_b = nt_1 * 8 + tig * 2 + 1;
#pragma unroll
          for (int m_tile_3 = 0; m_tile_3 < 2; m_tile_3++) {
            int col_lo_1 = col_base + warp * 32 + m_tile_3 * 16 + group_id;
            int col_hi_1 = col_lo_1 + 8;
            if (route_a < count) {
              long long pair_a = (long long)route_smem[route_a] * (long long)hidden;
              if (col_lo_1 < hidden) {
                *(reinterpret_cast<float*>(reinterpret_cast<float*>(partials_raw) +
                                           (pair_a + (long long)col_lo_1)) +
                  (0)) = acc[(m_tile_3 * 2 + nt_1) * 4];
              }
              if (col_hi_1 < hidden) {
                *(reinterpret_cast<float*>(reinterpret_cast<float*>(partials_raw) +
                                           (pair_a + (long long)col_hi_1)) +
                  (0)) = acc[(m_tile_3 * 2 + nt_1) * 4 + 2];
              }
            }
            if (route_b < count) {
              long long pair_b = (long long)route_smem[route_b] * (long long)hidden;
              if (col_lo_1 < hidden) {
                *(reinterpret_cast<float*>(reinterpret_cast<float*>(partials_raw) +
                                           (pair_b + (long long)col_lo_1)) +
                  (0)) = acc[(m_tile_3 * 2 + nt_1) * 4 + 1];
              }
              if (col_hi_1 < hidden) {
                *(reinterpret_cast<float*>(reinterpret_cast<float*>(partials_raw) +
                                           (pair_b + (long long)col_hi_1)) +
                  (0)) = acc[(m_tile_3 * 2 + nt_1) * 4 + 3];
              }
            }
          }
        }
#pragma unroll
        for (int i_3 = 0; i_3 < 16; i_3++) {
          a_frag[i_3] = a_next[i_3];
        }
      }
    }
  }
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

}  // extern "C"

#undef BLACKWELL_INF
#undef NUM_MAIN_STAGES
#undef SMEM_ROUTE_SMEM_OFF
#undef SMEM_ROUTE_SMEM_STAGE_BYTES
#undef SMEM_ROUTE_SMEM_STRIDE
#undef SMEM_S_SMEM_OFF
#undef SMEM_S_SMEM_STAGE_BYTES
#undef SMEM_S_SMEM_STRIDE
#undef SMEM_TOTAL
#undef SMEM_W_SMEM_OFF
#undef SMEM_W_SMEM_STAGE_BYTES
#undef SMEM_W_SMEM_STRIDE
#undef SMEM_W_U32_OFF
#undef SMEM_W_U32_STAGE_BYTES
#undef SMEM_W_U32_STRIDE
#undef THREADS

#define BLACKWELL_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_ROUTE_LIST_OFF 0
#define SMEM_ROUTE_LIST_STAGE_BYTES 72
#define SMEM_ROUTE_LIST_STRIDE 72
#define SMEM_TOTAL 128
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(256, 1) void kernel_flashinfer_bgmv_moe_combine_grouped_bf16_r32(
    float* __restrict__ y_accum, float* __restrict__ partials_raw,
    long long* __restrict__ sorted_token_ids, long long* __restrict__ lora_indices,
    float* __restrict__ topk_weights, int num_pairs, int num_tokens, int hidden, int output_stride,
    int output_offset, unsigned int* __restrict__ workspace_raw, int off_token_count,
    int off_token_routes) {
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
  asm volatile("griddepcontrol.wait;" ::: "memory");
  int token = blockIdx.x;
  if (token < num_tokens) {
    long long lora_id = lora_indices[token];
    int route_count = 0;
    int scan_all = 0;
    if (lora_id >= 0) {
      int pair_base = token * 2;
      int contiguous = 0;
      if (num_pairs == num_tokens * 2) {
        if (sorted_token_ids[pair_base] == (long long)token) {
          if (sorted_token_ids[pair_base + 1] == (long long)token) {
            contiguous = 1;
          }
        }
      }
      if (contiguous != 0) {
        route_count = 2;
        if (tid < 2) {
          route_list[tid] = pair_base + tid;
        }
      } else {
        int listed =
            (int)reinterpret_cast<const unsigned int*>(workspace_raw)[off_token_count + token];
        if (listed <= 16) {
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
          route_count = listed + direct_count;
          int table_base = off_token_routes + token * 16;
          if (route_count > tid) {
            int own_pair = direct_first;
            if (listed > tid) {
              own_pair =
                  (int)reinterpret_cast<const unsigned int*>(workspace_raw)[table_base + tid];
            } else if (tid == listed + 1) {
              own_pair = direct_second;
            }
            int own_order = 0;
#pragma unroll 1
            for (int probe = 0; probe < listed; probe++) {
              if (own_pair >
                  (int)reinterpret_cast<const unsigned int*>(workspace_raw)[table_base + probe]) {
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
        } else {
          scan_all = 1;
          route_count = num_pairs;
        }
      }
      __syncthreads();
    }
    int sweeps = (hidden + 1024 - 1) / 1024;
    int vec_ok = 0;
    if (hidden % 4 == 0) {
      if (output_stride % 4 == 0) {
        if (output_offset % 4 == 0) {
          vec_ok = 1;
        }
      }
    }
    float pv[4];
    int fast = 0;
    if (vec_ok != 0) {
      if (scan_all == 0) {
        if (route_count == 2) {
          if (lora_id >= 0) {
            fast = 1;
          }
        }
      }
    }
    if (fast != 0) {
      int fp0 = route_list[0];
      int fp1 = route_list[1];
      float fw0 = topk_weights[fp0];
      float fw1 = topk_weights[fp1];
      long long fb0 = (long long)fp0 * (long long)hidden;
      long long fb1 = (long long)fp1 * (long long)hidden;
      float pva0[4];
      float pva1[4];
      float acca[4];
#pragma unroll 1
      for (int st = 0; st < sweeps; st++) {
        int colt = st * 1024 + tid * 4;
        if (colt < hidden) {
          {
            unsigned _v4_0_0;
            unsigned _v4_0_1;
            unsigned _v4_0_2;
            unsigned _v4_0_3;
            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(_v4_0_0), "=r"(_v4_0_1), "=r"(_v4_0_2), "=r"(_v4_0_3)
                         : "l"((const void*)(reinterpret_cast<const float*>(partials_raw) +
                                             (fb0 + (long long)colt)))
                         : "memory");
            pva0[0 + 0] = __uint_as_float(_v4_0_0);
            pva0[0 + 1] = __uint_as_float(_v4_0_1);
            pva0[0 + 2] = __uint_as_float(_v4_0_2);
            pva0[0 + 3] = __uint_as_float(_v4_0_3);
          }
          {
            unsigned _v4_1_0;
            unsigned _v4_1_1;
            unsigned _v4_1_2;
            unsigned _v4_1_3;
            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(_v4_1_0), "=r"(_v4_1_1), "=r"(_v4_1_2), "=r"(_v4_1_3)
                         : "l"((const void*)(reinterpret_cast<const float*>(partials_raw) +
                                             (fb1 + (long long)colt)))
                         : "memory");
            pva1[0 + 0] = __uint_as_float(_v4_1_0);
            pva1[0 + 1] = __uint_as_float(_v4_1_1);
            pva1[0 + 2] = __uint_as_float(_v4_1_2);
            pva1[0 + 3] = __uint_as_float(_v4_1_3);
          }
#pragma unroll
          for (int cc = 0; cc < 4; cc++) {
            acca[cc] = 0.0f;
            float _fma_0 = __fmaf_rn(pva0[cc], fw0, acca[cc]);
            acca[cc] = _fma_0;
            float _fma_1 = __fmaf_rn(pva1[cc], fw1, acca[cc]);
            acca[cc] = _fma_1;
          }
          {
            float4 _v4 = make_float4(acca[0 + 0], acca[0 + 1], acca[0 + 2], acca[0 + 3]);
            *reinterpret_cast<float4*>((y_accum + (token * output_stride + output_offset + colt)) +
                                       0) = _v4;
          }
        }
      }
    }
    if (fast == 0) {
#pragma unroll 1
      for (int s = 0; s < sweeps; s++) {
        float acc[4];
        int col0 = s * 1024 + tid * 4;
#pragma unroll
        for (int cc_1 = 0; cc_1 < 4; cc_1++) {
          acc[cc_1] = 0.0f;
        }
        if (vec_ok != 0) {
          if (col0 < hidden) {
            if (lora_id >= 0) {
#pragma unroll 1
              for (int step = 0; step < route_count; step++) {
                int pair = step;
                int route_match = 1;
                if (scan_all != 0) {
                  if (sorted_token_ids[step] != (long long)token) {
                    route_match = 0;
                  }
                } else {
                  pair = route_list[step];
                }
                if (route_match != 0) {
                  float pair_weight = topk_weights[pair];
                  {
                    unsigned _v4_2_0;
                    unsigned _v4_2_1;
                    unsigned _v4_2_2;
                    unsigned _v4_2_3;
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                                 : "=r"(_v4_2_0), "=r"(_v4_2_1), "=r"(_v4_2_2), "=r"(_v4_2_3)
                                 : "l"((const void*)(reinterpret_cast<const float*>(partials_raw) +
                                                     ((long long)pair * (long long)hidden +
                                                      (long long)col0)))
                                 : "memory");
                    pv[0 + 0] = __uint_as_float(_v4_2_0);
                    pv[0 + 1] = __uint_as_float(_v4_2_1);
                    pv[0 + 2] = __uint_as_float(_v4_2_2);
                    pv[0 + 3] = __uint_as_float(_v4_2_3);
                  }
#pragma unroll
                  for (int cc_2 = 0; cc_2 < 4; cc_2++) {
                    float _fma_2 = __fmaf_rn(pv[cc_2], pair_weight, acc[cc_2]);
                    acc[cc_2] = _fma_2;
                  }
                }
              }
            }
            {
              float4 _v4 = make_float4(acc[0 + 0], acc[0 + 1], acc[0 + 2], acc[0 + 3]);
              *reinterpret_cast<float4*>(
                  (y_accum + (token * output_stride + output_offset + col0)) + 0) = _v4;
            }
          }
        } else {
          if (lora_id >= 0) {
#pragma unroll 1
            for (int step_1 = 0; step_1 < route_count; step_1++) {
              int pair_s = step_1;
              int route_match_s = 1;
              if (scan_all != 0) {
                if (sorted_token_ids[step_1] != (long long)token) {
                  route_match_s = 0;
                }
              } else {
                pair_s = route_list[step_1];
              }
              if (route_match_s != 0) {
                float pair_weight_s = topk_weights[pair_s];
#pragma unroll
                for (int cc_3 = 0; cc_3 < 4; cc_3++) {
                  if (col0 + cc_3 < hidden) {
                    float _fma_3 =
                        __fmaf_rn(reinterpret_cast<const float*>(
                                      partials_raw)[(long long)pair_s * (long long)hidden +
                                                    (long long)col0 + (long long)cc_3],
                                  pair_weight_s, acc[cc_3]);
                    acc[cc_3] = _fma_3;
                  }
                }
              }
            }
          }
#pragma unroll
          for (int cc_4 = 0; cc_4 < 4; cc_4++) {
            if (col0 + cc_4 < hidden) {
              *(reinterpret_cast<float*>(y_accum +
                                         (token * output_stride + output_offset + col0 + cc_4)) +
                (0)) = acc[cc_4];
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
#undef SMEM_ROUTE_LIST_OFF
#undef SMEM_ROUTE_LIST_STAGE_BYTES
#undef SMEM_ROUTE_LIST_STRIDE
#undef SMEM_TOTAL
#undef THREADS

#define BLACKWELL_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_CNT_OFF 0
#define SMEM_CNT_STAGE_BYTES 16388
#define SMEM_CNT_STRIDE 16388
#define SMEM_WARP_SUM_OFF 16388
#define SMEM_WARP_SUM_STAGE_BYTES 128
#define SMEM_WARP_SUM_STRIDE 128
#define SMEM_TOTAL 16640
#define THREADS 1024

extern "C" {

__global__ __launch_bounds__(1024, 1) void kernel_flashinfer_bgmv_moe_order_build_bf16_r32(
    long long* __restrict__ sorted_token_ids, long long* __restrict__ expert_ids,
    long long* __restrict__ lora_indices, int num_pairs, int num_tokens, int num_experts,
    int num_loras, unsigned int* __restrict__ workspace_raw, int off_route_order) {
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
  int* cnt = reinterpret_cast<int*>(smem_raw + 0);
  const int cnt_addr = smem + 0;
  int* warp_sum = reinterpret_cast<int*>(smem_raw + 16388);
  const int warp_sum_addr = smem + 16388;

  // === Task calls (dependency order) ===
  int bins = num_loras * num_experts;
  int buckets = bins + 1;
  int bpt = (buckets + 1024 - 1) / 1024;
#pragma unroll 1
  for (int i = 0; i < bpt; i++) {
    int zb = i * 1024 + tid;
    if (zb < buckets) {
      cnt[zb] = 0;
    }
  }
  int keys[4];
#pragma unroll
  for (int i_1 = 0; i_1 < 4; i_1++) {
    keys[i_1] = bins;
    int p = i_1 * 1024 + tid;
    if (p < num_pairs) {
      long long token = sorted_token_ids[p];
      if (token >= 0) {
        if (token < (long long)num_tokens) {
          long long lora = lora_indices[token];
          long long expert = expert_ids[p];
          if (lora >= 0) {
            if (expert >= 0) {
              if (expert < (long long)num_experts) {
                keys[i_1] = (int)lora * num_experts + (int)expert;
              }
            }
          }
        }
      }
    }
  }
  __syncthreads();
#pragma unroll
  for (int i_2 = 0; i_2 < 4; i_2++) {
    int pc = i_2 * 1024 + tid;
    if (pc < num_pairs) {
      int _atomic_old_0 = atomicAdd(&cnt[keys[i_2]], 1);
      int counted = _atomic_old_0;
    }
  }
  __syncthreads();
  int local[5];
  int mine = 0;
#pragma unroll
  for (int i_3 = 0; i_3 < 5; i_3++) {
    local[i_3] = 0;
    if (bpt > i_3) {
      int b = tid * bpt + i_3;
      if (b < buckets) {
        local[i_3] = cnt[b];
        mine += local[i_3];
      }
    }
  }
  int incl = mine;
#pragma unroll
  for (int step = 0; step < 5; step++) {
    int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, incl, 1 << step, 32);
    int up = _shfl_up_0;
    if (lane >= 1 << step) {
      incl += up;
    }
  }
  if (lane == 31) {
    warp_sum[warp] = incl;
  }
  __syncthreads();
  int base = 0;
#pragma unroll
  for (int w = 0; w < 32; w++) {
    if (w < warp) {
      base += warp_sum[w];
    }
  }
  int excl = base + incl - mine;
#pragma unroll
  for (int i_4 = 0; i_4 < 5; i_4++) {
    if (bpt > i_4) {
      int b2 = tid * bpt + i_4;
      if (b2 < buckets) {
        cnt[b2] = excl;
        excl += local[i_4];
      }
    }
  }
  __syncthreads();
#pragma unroll
  for (int i_5 = 0; i_5 < 4; i_5++) {
    int ps = i_5 * 1024 + tid;
    if (ps < num_pairs) {
      int _atomic_old_1 = atomicAdd(&cnt[keys[i_5]], 1);
      int pos = _atomic_old_1;
      *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(workspace_raw) +
                                        (off_route_order + pos)) +
        (0)) = (unsigned int)ps;
    }
  }
}

}  // extern "C"

#undef BLACKWELL_INF
#undef NUM_MAIN_STAGES
#undef SMEM_CNT_OFF
#undef SMEM_CNT_STAGE_BYTES
#undef SMEM_CNT_STRIDE
#undef SMEM_TOTAL
#undef SMEM_WARP_SUM_OFF
#undef SMEM_WARP_SUM_STAGE_BYTES
#undef SMEM_WARP_SUM_STRIDE
#undef THREADS

#define BLACKWELL_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_WARP_PARTIALS_OFF 0
#define SMEM_WARP_PARTIALS_STAGE_BYTES 512
#define SMEM_WARP_PARTIALS_STRIDE 512
#define SMEM_X_RING_OFF 512
#define SMEM_X_RING_STAGE_BYTES 16384
#define SMEM_X_RING_STRIDE 16384
#define SMEM_W_RING_OFF 16896
#define SMEM_W_RING_STAGE_BYTES 32768
#define SMEM_W_RING_STRIDE 32768
#define SMEM_TOTAL 49664
#define THREADS 128

extern "C" {

__global__ __launch_bounds__(128, 4) void kernel_flashinfer_bgmv_moe_shrink_grouped_ring_bf16_r32(
    uint16_t* __restrict__ shrink_out_raw, uint16_t* __restrict__ x_raw,
    uint16_t* __restrict__ lora_a_raw, long long* __restrict__ sorted_token_ids, int num_pairs,
    int num_experts, int hidden, int num_tiles, int rt_per_cta, int rt_groups,
    unsigned int* __restrict__ workspace_raw, int off_group_offset, int off_tile_table,
    int off_sorted_routes) {
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
  float* warp_partials = reinterpret_cast<float*>(smem_raw + 0);
  const int warp_partials_addr = smem + 0;
  __nv_bfloat16* x_ring = reinterpret_cast<__nv_bfloat16*>(smem_raw + 512);
  const int x_ring_addr = smem + 512;
  __nv_bfloat16* w_ring = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16896);
  const int w_ring_addr = smem + 16896;

  // === Task calls (dependency order) ===
  asm volatile("griddepcontrol.wait;" ::: "memory");
  int rt_group = blockIdx.x % ((0) ? 4 : rt_groups);
  int half = blockIdx.x / ((0) ? 4 : rt_groups) % 4;
  int tile = blockIdx.x / (((0) ? 4 : rt_groups) * 4);
  int n_tiles = (int)reinterpret_cast<const unsigned int*>(workspace_raw)[0];
  if (tile < n_tiles) {
    int entry = (int)reinterpret_cast<const unsigned int*>(workspace_raw)[off_tile_table + tile];
    int group = entry / 65536;
    int chunk = entry % 65536;
    int start =
        (int)reinterpret_cast<const unsigned int*>(workspace_raw)[off_group_offset + group] +
        chunk * 16 + half * 4;
    int group_end =
        (int)reinterpret_cast<const unsigned int*>(workspace_raw)[off_group_offset + group + 1];
    int count = group_end - start;
    if (count > 4) {
      count = 4;
    }
    if (count > 0) {
      int lora = group / num_experts;
      int expert = group % num_experts;
      int hidden_words = hidden / 2;
      long long weight_row_base = (long long)(lora * num_experts + expert) * 32;
      int routes[4];
      long long tokens[4];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        routes[j] = -1;
        tokens[j] = 0;
        if (count > j) {
          routes[j] = (int)reinterpret_cast<const unsigned int*>(
              workspace_raw)[off_sorted_routes + start + j];
          tokens[j] = sorted_token_ids[routes[j]];
        }
      }
      int rank_tile0 = rt_group * rt_per_cta;
      int rank_tile_end = rank_tile0 + rt_per_cta;
      int rank_base_it = rank_tile0 * 8;
      long long weight_words_it =
          (weight_row_base + (long long)rank_base_it) * (long long)hidden_words;
      long long weight_words_step = hidden_words * 8;
#pragma unroll 1
      for (int rank_tile = rank_tile0; rank_tile < rank_tile_end; rank_tile++) {
        float acc[32];
        unsigned int x_car[16];
        unsigned int xr_car[4];
        unsigned int w_car[4];
        float x_values[32];
        float w_values[8];
        float w_t[16];
        int lane_0 = lane;
        float red_a[16];
        float red_b[8];
        float red_c[4];
        float red_d[2];
        float red_e[1];
#pragma unroll
        for (int owner = 0; owner < 32; owner++) {
          acc[owner] = 0.0f;
        }
        int tid_vec = tid * 8;
#pragma unroll
        for (int d = 0; d < 1; d++) {
          int k_p = d * 1024 + tid_vec;
          if (k_p < hidden) {
            int kw_p = k_p / 2;
            {
#pragma unroll
              for (int j_1 = 0; j_1 < 4; j_1++) {
                asm volatile(
                    "cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                        x_ring_addr + (unsigned int)(((d * 4 + j_1) * 1024 + tid_vec) * 2)),
                    "l"(reinterpret_cast<const unsigned int*>(x_raw) +
                        (tokens[j_1] * (long long)hidden_words + (long long)kw_p)));
              }
            }
#pragma unroll
            for (int r = 0; r < 8; r++) {
              asm volatile(
                  "cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                      w_ring_addr + (unsigned int)(((d * 8 + r) * 1024 + tid_vec) * 2)),
                  "l"(reinterpret_cast<const unsigned int*>(lora_a_raw) +
                      (weight_words_it + (long long)(r * hidden_words) + (long long)kw_p)));
            }
          }
          asm volatile("cp.async.commit_group;");
        }
#pragma unroll 1
        for (int local = 0; local < num_tiles; local++) {
          int stage = local % 2;
          int nxt = local + 1;
          if (nxt < num_tiles) {
            int nstage = nxt % 2;
            int k_n = nxt * 1024 + tid_vec;
            if (k_n < hidden) {
              int kw_n = k_n / 2;
              {
#pragma unroll
                for (int j_2 = 0; j_2 < 4; j_2++) {
                  asm volatile(
                      "cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                          x_ring_addr + (unsigned int)(((nstage * 4 + j_2) * 1024 + tid_vec) * 2)),
                      "l"(reinterpret_cast<const unsigned int*>(x_raw) +
                          (tokens[j_2] * (long long)hidden_words + (long long)kw_n)));
                }
              }
#pragma unroll
              for (int r_1 = 0; r_1 < 8; r_1++) {
                asm volatile(
                    "cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                        w_ring_addr + (unsigned int)(((nstage * 8 + r_1) * 1024 + tid_vec) * 2)),
                    "l"(reinterpret_cast<const unsigned int*>(lora_a_raw) +
                        (weight_words_it + (long long)(r_1 * hidden_words) + (long long)kw_n)));
              }
            }
          }
          asm volatile("cp.async.commit_group;");
          asm volatile("cp.async.wait_group 1;");
          int k_base_r = local * 1024 + tid_vec;
          if (k_base_r < hidden) {
#pragma unroll
            for (int j_3 = 0; j_3 < 4; j_3++) {
              asm volatile(
                  "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                  : "=r"(*reinterpret_cast<uint32_t*>(&xr_car[0])),
                    "=r"(*reinterpret_cast<uint32_t*>(&xr_car[(0) + 1])),
                    "=r"(*reinterpret_cast<uint32_t*>(&xr_car[(0) + 2])),
                    "=r"(*reinterpret_cast<uint32_t*>(&xr_car[(0) + 3]))
                  : "r"(x_ring_addr + (unsigned int)(((stage * 4 + j_3) * 1024 + tid_vec) * 2)));
#pragma unroll
              for (int pair = 0; pair < 4; pair++) {
                x_values[j_3 * 8 + 2 * pair] = __uint_as_float(xr_car[pair] << 16);
                x_values[j_3 * 8 + 2 * pair + 1] = __uint_as_float(xr_car[pair] & 4294901760u);
              }
            }
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                           "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                           "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                           "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
                         : "r"(w_ring_addr + (unsigned int)((stage * 8 * 1024 + tid_vec) * 2)));
            w_t[0] = __uint_as_float(w_car[0] << 16);
            w_t[2] = __uint_as_float(w_car[0] & 4294901760u);
            w_t[4] = __uint_as_float(w_car[1] << 16);
            w_t[6] = __uint_as_float(w_car[1] & 4294901760u);
            w_t[8] = __uint_as_float(w_car[2] << 16);
            w_t[10] = __uint_as_float(w_car[2] & 4294901760u);
            w_t[12] = __uint_as_float(w_car[3] << 16);
            w_t[14] = __uint_as_float(w_car[3] & 4294901760u);
            asm volatile(
                "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
                : "r"(w_ring_addr + (unsigned int)(((stage * 8 + 1) * 1024 + tid_vec) * 2)));
            w_t[1] = __uint_as_float(w_car[0] << 16);
            w_t[3] = __uint_as_float(w_car[0] & 4294901760u);
            w_t[5] = __uint_as_float(w_car[1] << 16);
            w_t[7] = __uint_as_float(w_car[1] & 4294901760u);
            w_t[9] = __uint_as_float(w_car[2] << 16);
            w_t[11] = __uint_as_float(w_car[2] & 4294901760u);
            w_t[13] = __uint_as_float(w_car[3] << 16);
            w_t[15] = __uint_as_float(w_car[3] & 4294901760u);
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_0;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_0) : "f"(x_values[0]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[0]), "+f"(acc[1])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_0));
#else
              acc[0] = __fmaf_rn(w_t[0], x_values[0], acc[0]);
              acc[1] = __fmaf_rn(w_t[1], x_values[0], acc[1]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_1;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_1) : "f"(x_values[1]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[0]), "+f"(acc[1])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_1));
#else
              acc[0] = __fmaf_rn(w_t[2], x_values[1], acc[0]);
              acc[1] = __fmaf_rn(w_t[3], x_values[1], acc[1]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_2;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_2) : "f"(x_values[2]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[0]), "+f"(acc[1])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_2));
#else
              acc[0] = __fmaf_rn(w_t[4], x_values[2], acc[0]);
              acc[1] = __fmaf_rn(w_t[5], x_values[2], acc[1]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_3;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_3) : "f"(x_values[3]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[0]), "+f"(acc[1])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_3));
#else
              acc[0] = __fmaf_rn(w_t[6], x_values[3], acc[0]);
              acc[1] = __fmaf_rn(w_t[7], x_values[3], acc[1]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_4;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_4) : "f"(x_values[4]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[0]), "+f"(acc[1])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_4));
#else
              acc[0] = __fmaf_rn(w_t[8], x_values[4], acc[0]);
              acc[1] = __fmaf_rn(w_t[9], x_values[4], acc[1]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_5;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_5) : "f"(x_values[5]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[0]), "+f"(acc[1])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_5));
#else
              acc[0] = __fmaf_rn(w_t[10], x_values[5], acc[0]);
              acc[1] = __fmaf_rn(w_t[11], x_values[5], acc[1]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_6;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_6) : "f"(x_values[6]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[0]), "+f"(acc[1])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_6));
#else
              acc[0] = __fmaf_rn(w_t[12], x_values[6], acc[0]);
              acc[1] = __fmaf_rn(w_t[13], x_values[6], acc[1]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_7;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_7) : "f"(x_values[7]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[0]), "+f"(acc[1])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_7));
#else
              acc[0] = __fmaf_rn(w_t[14], x_values[7], acc[0]);
              acc[1] = __fmaf_rn(w_t[15], x_values[7], acc[1]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_8;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_8) : "f"(x_values[8]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[8]), "+f"(acc[9])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_8));
#else
              acc[8] = __fmaf_rn(w_t[0], x_values[8], acc[8]);
              acc[9] = __fmaf_rn(w_t[1], x_values[8], acc[9]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_9;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_9) : "f"(x_values[9]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[8]), "+f"(acc[9])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_9));
#else
              acc[8] = __fmaf_rn(w_t[2], x_values[9], acc[8]);
              acc[9] = __fmaf_rn(w_t[3], x_values[9], acc[9]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_10;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_10) : "f"(x_values[10]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[8]), "+f"(acc[9])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_10));
#else
              acc[8] = __fmaf_rn(w_t[4], x_values[10], acc[8]);
              acc[9] = __fmaf_rn(w_t[5], x_values[10], acc[9]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_11;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_11) : "f"(x_values[11]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[8]), "+f"(acc[9])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_11));
#else
              acc[8] = __fmaf_rn(w_t[6], x_values[11], acc[8]);
              acc[9] = __fmaf_rn(w_t[7], x_values[11], acc[9]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_12;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_12) : "f"(x_values[12]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[8]), "+f"(acc[9])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_12));
#else
              acc[8] = __fmaf_rn(w_t[8], x_values[12], acc[8]);
              acc[9] = __fmaf_rn(w_t[9], x_values[12], acc[9]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_13;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_13) : "f"(x_values[13]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[8]), "+f"(acc[9])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_13));
#else
              acc[8] = __fmaf_rn(w_t[10], x_values[13], acc[8]);
              acc[9] = __fmaf_rn(w_t[11], x_values[13], acc[9]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_14;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_14) : "f"(x_values[14]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[8]), "+f"(acc[9])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_14));
#else
              acc[8] = __fmaf_rn(w_t[12], x_values[14], acc[8]);
              acc[9] = __fmaf_rn(w_t[13], x_values[14], acc[9]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_15;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_15) : "f"(x_values[15]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[8]), "+f"(acc[9])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_15));
#else
              acc[8] = __fmaf_rn(w_t[14], x_values[15], acc[8]);
              acc[9] = __fmaf_rn(w_t[15], x_values[15], acc[9]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_16;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_16) : "f"(x_values[16]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[16]), "+f"(acc[17])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_16));
#else
              acc[16] = __fmaf_rn(w_t[0], x_values[16], acc[16]);
              acc[17] = __fmaf_rn(w_t[1], x_values[16], acc[17]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_17;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_17) : "f"(x_values[17]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[16]), "+f"(acc[17])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_17));
#else
              acc[16] = __fmaf_rn(w_t[2], x_values[17], acc[16]);
              acc[17] = __fmaf_rn(w_t[3], x_values[17], acc[17]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_18;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_18) : "f"(x_values[18]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[16]), "+f"(acc[17])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_18));
#else
              acc[16] = __fmaf_rn(w_t[4], x_values[18], acc[16]);
              acc[17] = __fmaf_rn(w_t[5], x_values[18], acc[17]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_19;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_19) : "f"(x_values[19]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[16]), "+f"(acc[17])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_19));
#else
              acc[16] = __fmaf_rn(w_t[6], x_values[19], acc[16]);
              acc[17] = __fmaf_rn(w_t[7], x_values[19], acc[17]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_20;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_20) : "f"(x_values[20]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[16]), "+f"(acc[17])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_20));
#else
              acc[16] = __fmaf_rn(w_t[8], x_values[20], acc[16]);
              acc[17] = __fmaf_rn(w_t[9], x_values[20], acc[17]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_21;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_21) : "f"(x_values[21]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[16]), "+f"(acc[17])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_21));
#else
              acc[16] = __fmaf_rn(w_t[10], x_values[21], acc[16]);
              acc[17] = __fmaf_rn(w_t[11], x_values[21], acc[17]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_22;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_22) : "f"(x_values[22]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[16]), "+f"(acc[17])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_22));
#else
              acc[16] = __fmaf_rn(w_t[12], x_values[22], acc[16]);
              acc[17] = __fmaf_rn(w_t[13], x_values[22], acc[17]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_23;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_23) : "f"(x_values[23]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[16]), "+f"(acc[17])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_23));
#else
              acc[16] = __fmaf_rn(w_t[14], x_values[23], acc[16]);
              acc[17] = __fmaf_rn(w_t[15], x_values[23], acc[17]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_24;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_24) : "f"(x_values[24]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[24]), "+f"(acc[25])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_24));
#else
              acc[24] = __fmaf_rn(w_t[0], x_values[24], acc[24]);
              acc[25] = __fmaf_rn(w_t[1], x_values[24], acc[25]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_25;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_25) : "f"(x_values[25]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[24]), "+f"(acc[25])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_25));
#else
              acc[24] = __fmaf_rn(w_t[2], x_values[25], acc[24]);
              acc[25] = __fmaf_rn(w_t[3], x_values[25], acc[25]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_26;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_26) : "f"(x_values[26]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[24]), "+f"(acc[25])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_26));
#else
              acc[24] = __fmaf_rn(w_t[4], x_values[26], acc[24]);
              acc[25] = __fmaf_rn(w_t[5], x_values[26], acc[25]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_27;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_27) : "f"(x_values[27]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[24]), "+f"(acc[25])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_27));
#else
              acc[24] = __fmaf_rn(w_t[6], x_values[27], acc[24]);
              acc[25] = __fmaf_rn(w_t[7], x_values[27], acc[25]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_28;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_28) : "f"(x_values[28]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[24]), "+f"(acc[25])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_28));
#else
              acc[24] = __fmaf_rn(w_t[8], x_values[28], acc[24]);
              acc[25] = __fmaf_rn(w_t[9], x_values[28], acc[25]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_29;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_29) : "f"(x_values[29]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[24]), "+f"(acc[25])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_29));
#else
              acc[24] = __fmaf_rn(w_t[10], x_values[29], acc[24]);
              acc[25] = __fmaf_rn(w_t[11], x_values[29], acc[25]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_30;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_30) : "f"(x_values[30]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[24]), "+f"(acc[25])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_30));
#else
              acc[24] = __fmaf_rn(w_t[12], x_values[30], acc[24]);
              acc[25] = __fmaf_rn(w_t[13], x_values[30], acc[25]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_31;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_31) : "f"(x_values[31]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[24]), "+f"(acc[25])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_31));
#else
              acc[24] = __fmaf_rn(w_t[14], x_values[31], acc[24]);
              acc[25] = __fmaf_rn(w_t[15], x_values[31], acc[25]);
#endif
            }
            asm volatile(
                "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
                : "r"(w_ring_addr + (unsigned int)(((stage * 8 + 2) * 1024 + tid_vec) * 2)));
            w_t[0] = __uint_as_float(w_car[0] << 16);
            w_t[2] = __uint_as_float(w_car[0] & 4294901760u);
            w_t[4] = __uint_as_float(w_car[1] << 16);
            w_t[6] = __uint_as_float(w_car[1] & 4294901760u);
            w_t[8] = __uint_as_float(w_car[2] << 16);
            w_t[10] = __uint_as_float(w_car[2] & 4294901760u);
            w_t[12] = __uint_as_float(w_car[3] << 16);
            w_t[14] = __uint_as_float(w_car[3] & 4294901760u);
            asm volatile(
                "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
                : "r"(w_ring_addr + (unsigned int)(((stage * 8 + 2 + 1) * 1024 + tid_vec) * 2)));
            w_t[1] = __uint_as_float(w_car[0] << 16);
            w_t[3] = __uint_as_float(w_car[0] & 4294901760u);
            w_t[5] = __uint_as_float(w_car[1] << 16);
            w_t[7] = __uint_as_float(w_car[1] & 4294901760u);
            w_t[9] = __uint_as_float(w_car[2] << 16);
            w_t[11] = __uint_as_float(w_car[2] & 4294901760u);
            w_t[13] = __uint_as_float(w_car[3] << 16);
            w_t[15] = __uint_as_float(w_car[3] & 4294901760u);
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_32;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_32) : "f"(x_values[0]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[2]), "+f"(acc[3])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_32));
#else
              acc[2] = __fmaf_rn(w_t[0], x_values[0], acc[2]);
              acc[3] = __fmaf_rn(w_t[1], x_values[0], acc[3]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_33;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_33) : "f"(x_values[1]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[2]), "+f"(acc[3])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_33));
#else
              acc[2] = __fmaf_rn(w_t[2], x_values[1], acc[2]);
              acc[3] = __fmaf_rn(w_t[3], x_values[1], acc[3]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_34;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_34) : "f"(x_values[2]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[2]), "+f"(acc[3])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_34));
#else
              acc[2] = __fmaf_rn(w_t[4], x_values[2], acc[2]);
              acc[3] = __fmaf_rn(w_t[5], x_values[2], acc[3]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_35;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_35) : "f"(x_values[3]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[2]), "+f"(acc[3])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_35));
#else
              acc[2] = __fmaf_rn(w_t[6], x_values[3], acc[2]);
              acc[3] = __fmaf_rn(w_t[7], x_values[3], acc[3]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_36;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_36) : "f"(x_values[4]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[2]), "+f"(acc[3])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_36));
#else
              acc[2] = __fmaf_rn(w_t[8], x_values[4], acc[2]);
              acc[3] = __fmaf_rn(w_t[9], x_values[4], acc[3]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_37;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_37) : "f"(x_values[5]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[2]), "+f"(acc[3])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_37));
#else
              acc[2] = __fmaf_rn(w_t[10], x_values[5], acc[2]);
              acc[3] = __fmaf_rn(w_t[11], x_values[5], acc[3]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_38;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_38) : "f"(x_values[6]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[2]), "+f"(acc[3])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_38));
#else
              acc[2] = __fmaf_rn(w_t[12], x_values[6], acc[2]);
              acc[3] = __fmaf_rn(w_t[13], x_values[6], acc[3]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_39;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_39) : "f"(x_values[7]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[2]), "+f"(acc[3])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_39));
#else
              acc[2] = __fmaf_rn(w_t[14], x_values[7], acc[2]);
              acc[3] = __fmaf_rn(w_t[15], x_values[7], acc[3]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_40;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_40) : "f"(x_values[8]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[10]), "+f"(acc[11])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_40));
#else
              acc[10] = __fmaf_rn(w_t[0], x_values[8], acc[10]);
              acc[11] = __fmaf_rn(w_t[1], x_values[8], acc[11]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_41;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_41) : "f"(x_values[9]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[10]), "+f"(acc[11])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_41));
#else
              acc[10] = __fmaf_rn(w_t[2], x_values[9], acc[10]);
              acc[11] = __fmaf_rn(w_t[3], x_values[9], acc[11]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_42;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_42) : "f"(x_values[10]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[10]), "+f"(acc[11])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_42));
#else
              acc[10] = __fmaf_rn(w_t[4], x_values[10], acc[10]);
              acc[11] = __fmaf_rn(w_t[5], x_values[10], acc[11]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_43;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_43) : "f"(x_values[11]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[10]), "+f"(acc[11])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_43));
#else
              acc[10] = __fmaf_rn(w_t[6], x_values[11], acc[10]);
              acc[11] = __fmaf_rn(w_t[7], x_values[11], acc[11]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_44;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_44) : "f"(x_values[12]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[10]), "+f"(acc[11])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_44));
#else
              acc[10] = __fmaf_rn(w_t[8], x_values[12], acc[10]);
              acc[11] = __fmaf_rn(w_t[9], x_values[12], acc[11]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_45;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_45) : "f"(x_values[13]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[10]), "+f"(acc[11])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_45));
#else
              acc[10] = __fmaf_rn(w_t[10], x_values[13], acc[10]);
              acc[11] = __fmaf_rn(w_t[11], x_values[13], acc[11]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_46;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_46) : "f"(x_values[14]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[10]), "+f"(acc[11])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_46));
#else
              acc[10] = __fmaf_rn(w_t[12], x_values[14], acc[10]);
              acc[11] = __fmaf_rn(w_t[13], x_values[14], acc[11]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_47;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_47) : "f"(x_values[15]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[10]), "+f"(acc[11])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_47));
#else
              acc[10] = __fmaf_rn(w_t[14], x_values[15], acc[10]);
              acc[11] = __fmaf_rn(w_t[15], x_values[15], acc[11]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_48;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_48) : "f"(x_values[16]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[18]), "+f"(acc[19])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_48));
#else
              acc[18] = __fmaf_rn(w_t[0], x_values[16], acc[18]);
              acc[19] = __fmaf_rn(w_t[1], x_values[16], acc[19]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_49;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_49) : "f"(x_values[17]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[18]), "+f"(acc[19])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_49));
#else
              acc[18] = __fmaf_rn(w_t[2], x_values[17], acc[18]);
              acc[19] = __fmaf_rn(w_t[3], x_values[17], acc[19]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_50;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_50) : "f"(x_values[18]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[18]), "+f"(acc[19])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_50));
#else
              acc[18] = __fmaf_rn(w_t[4], x_values[18], acc[18]);
              acc[19] = __fmaf_rn(w_t[5], x_values[18], acc[19]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_51;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_51) : "f"(x_values[19]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[18]), "+f"(acc[19])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_51));
#else
              acc[18] = __fmaf_rn(w_t[6], x_values[19], acc[18]);
              acc[19] = __fmaf_rn(w_t[7], x_values[19], acc[19]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_52;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_52) : "f"(x_values[20]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[18]), "+f"(acc[19])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_52));
#else
              acc[18] = __fmaf_rn(w_t[8], x_values[20], acc[18]);
              acc[19] = __fmaf_rn(w_t[9], x_values[20], acc[19]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_53;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_53) : "f"(x_values[21]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[18]), "+f"(acc[19])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_53));
#else
              acc[18] = __fmaf_rn(w_t[10], x_values[21], acc[18]);
              acc[19] = __fmaf_rn(w_t[11], x_values[21], acc[19]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_54;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_54) : "f"(x_values[22]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[18]), "+f"(acc[19])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_54));
#else
              acc[18] = __fmaf_rn(w_t[12], x_values[22], acc[18]);
              acc[19] = __fmaf_rn(w_t[13], x_values[22], acc[19]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_55;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_55) : "f"(x_values[23]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[18]), "+f"(acc[19])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_55));
#else
              acc[18] = __fmaf_rn(w_t[14], x_values[23], acc[18]);
              acc[19] = __fmaf_rn(w_t[15], x_values[23], acc[19]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_56;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_56) : "f"(x_values[24]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[26]), "+f"(acc[27])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_56));
#else
              acc[26] = __fmaf_rn(w_t[0], x_values[24], acc[26]);
              acc[27] = __fmaf_rn(w_t[1], x_values[24], acc[27]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_57;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_57) : "f"(x_values[25]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[26]), "+f"(acc[27])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_57));
#else
              acc[26] = __fmaf_rn(w_t[2], x_values[25], acc[26]);
              acc[27] = __fmaf_rn(w_t[3], x_values[25], acc[27]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_58;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_58) : "f"(x_values[26]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[26]), "+f"(acc[27])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_58));
#else
              acc[26] = __fmaf_rn(w_t[4], x_values[26], acc[26]);
              acc[27] = __fmaf_rn(w_t[5], x_values[26], acc[27]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_59;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_59) : "f"(x_values[27]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[26]), "+f"(acc[27])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_59));
#else
              acc[26] = __fmaf_rn(w_t[6], x_values[27], acc[26]);
              acc[27] = __fmaf_rn(w_t[7], x_values[27], acc[27]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_60;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_60) : "f"(x_values[28]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[26]), "+f"(acc[27])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_60));
#else
              acc[26] = __fmaf_rn(w_t[8], x_values[28], acc[26]);
              acc[27] = __fmaf_rn(w_t[9], x_values[28], acc[27]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_61;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_61) : "f"(x_values[29]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[26]), "+f"(acc[27])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_61));
#else
              acc[26] = __fmaf_rn(w_t[10], x_values[29], acc[26]);
              acc[27] = __fmaf_rn(w_t[11], x_values[29], acc[27]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_62;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_62) : "f"(x_values[30]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[26]), "+f"(acc[27])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_62));
#else
              acc[26] = __fmaf_rn(w_t[12], x_values[30], acc[26]);
              acc[27] = __fmaf_rn(w_t[13], x_values[30], acc[27]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_63;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_63) : "f"(x_values[31]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[26]), "+f"(acc[27])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_63));
#else
              acc[26] = __fmaf_rn(w_t[14], x_values[31], acc[26]);
              acc[27] = __fmaf_rn(w_t[15], x_values[31], acc[27]);
#endif
            }
            asm volatile(
                "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
                : "r"(w_ring_addr + (unsigned int)(((stage * 8 + 4) * 1024 + tid_vec) * 2)));
            w_t[0] = __uint_as_float(w_car[0] << 16);
            w_t[2] = __uint_as_float(w_car[0] & 4294901760u);
            w_t[4] = __uint_as_float(w_car[1] << 16);
            w_t[6] = __uint_as_float(w_car[1] & 4294901760u);
            w_t[8] = __uint_as_float(w_car[2] << 16);
            w_t[10] = __uint_as_float(w_car[2] & 4294901760u);
            w_t[12] = __uint_as_float(w_car[3] << 16);
            w_t[14] = __uint_as_float(w_car[3] & 4294901760u);
            asm volatile(
                "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
                : "r"(w_ring_addr + (unsigned int)(((stage * 8 + 4 + 1) * 1024 + tid_vec) * 2)));
            w_t[1] = __uint_as_float(w_car[0] << 16);
            w_t[3] = __uint_as_float(w_car[0] & 4294901760u);
            w_t[5] = __uint_as_float(w_car[1] << 16);
            w_t[7] = __uint_as_float(w_car[1] & 4294901760u);
            w_t[9] = __uint_as_float(w_car[2] << 16);
            w_t[11] = __uint_as_float(w_car[2] & 4294901760u);
            w_t[13] = __uint_as_float(w_car[3] << 16);
            w_t[15] = __uint_as_float(w_car[3] & 4294901760u);
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_64;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_64) : "f"(x_values[0]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[4]), "+f"(acc[5])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_64));
#else
              acc[4] = __fmaf_rn(w_t[0], x_values[0], acc[4]);
              acc[5] = __fmaf_rn(w_t[1], x_values[0], acc[5]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_65;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_65) : "f"(x_values[1]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[4]), "+f"(acc[5])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_65));
#else
              acc[4] = __fmaf_rn(w_t[2], x_values[1], acc[4]);
              acc[5] = __fmaf_rn(w_t[3], x_values[1], acc[5]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_66;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_66) : "f"(x_values[2]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[4]), "+f"(acc[5])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_66));
#else
              acc[4] = __fmaf_rn(w_t[4], x_values[2], acc[4]);
              acc[5] = __fmaf_rn(w_t[5], x_values[2], acc[5]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_67;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_67) : "f"(x_values[3]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[4]), "+f"(acc[5])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_67));
#else
              acc[4] = __fmaf_rn(w_t[6], x_values[3], acc[4]);
              acc[5] = __fmaf_rn(w_t[7], x_values[3], acc[5]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_68;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_68) : "f"(x_values[4]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[4]), "+f"(acc[5])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_68));
#else
              acc[4] = __fmaf_rn(w_t[8], x_values[4], acc[4]);
              acc[5] = __fmaf_rn(w_t[9], x_values[4], acc[5]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_69;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_69) : "f"(x_values[5]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[4]), "+f"(acc[5])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_69));
#else
              acc[4] = __fmaf_rn(w_t[10], x_values[5], acc[4]);
              acc[5] = __fmaf_rn(w_t[11], x_values[5], acc[5]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_70;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_70) : "f"(x_values[6]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[4]), "+f"(acc[5])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_70));
#else
              acc[4] = __fmaf_rn(w_t[12], x_values[6], acc[4]);
              acc[5] = __fmaf_rn(w_t[13], x_values[6], acc[5]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_71;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_71) : "f"(x_values[7]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[4]), "+f"(acc[5])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_71));
#else
              acc[4] = __fmaf_rn(w_t[14], x_values[7], acc[4]);
              acc[5] = __fmaf_rn(w_t[15], x_values[7], acc[5]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_72;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_72) : "f"(x_values[8]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[12]), "+f"(acc[13])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_72));
#else
              acc[12] = __fmaf_rn(w_t[0], x_values[8], acc[12]);
              acc[13] = __fmaf_rn(w_t[1], x_values[8], acc[13]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_73;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_73) : "f"(x_values[9]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[12]), "+f"(acc[13])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_73));
#else
              acc[12] = __fmaf_rn(w_t[2], x_values[9], acc[12]);
              acc[13] = __fmaf_rn(w_t[3], x_values[9], acc[13]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_74;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_74) : "f"(x_values[10]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[12]), "+f"(acc[13])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_74));
#else
              acc[12] = __fmaf_rn(w_t[4], x_values[10], acc[12]);
              acc[13] = __fmaf_rn(w_t[5], x_values[10], acc[13]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_75;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_75) : "f"(x_values[11]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[12]), "+f"(acc[13])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_75));
#else
              acc[12] = __fmaf_rn(w_t[6], x_values[11], acc[12]);
              acc[13] = __fmaf_rn(w_t[7], x_values[11], acc[13]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_76;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_76) : "f"(x_values[12]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[12]), "+f"(acc[13])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_76));
#else
              acc[12] = __fmaf_rn(w_t[8], x_values[12], acc[12]);
              acc[13] = __fmaf_rn(w_t[9], x_values[12], acc[13]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_77;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_77) : "f"(x_values[13]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[12]), "+f"(acc[13])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_77));
#else
              acc[12] = __fmaf_rn(w_t[10], x_values[13], acc[12]);
              acc[13] = __fmaf_rn(w_t[11], x_values[13], acc[13]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_78;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_78) : "f"(x_values[14]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[12]), "+f"(acc[13])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_78));
#else
              acc[12] = __fmaf_rn(w_t[12], x_values[14], acc[12]);
              acc[13] = __fmaf_rn(w_t[13], x_values[14], acc[13]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_79;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_79) : "f"(x_values[15]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[12]), "+f"(acc[13])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_79));
#else
              acc[12] = __fmaf_rn(w_t[14], x_values[15], acc[12]);
              acc[13] = __fmaf_rn(w_t[15], x_values[15], acc[13]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_80;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_80) : "f"(x_values[16]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[20]), "+f"(acc[21])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_80));
#else
              acc[20] = __fmaf_rn(w_t[0], x_values[16], acc[20]);
              acc[21] = __fmaf_rn(w_t[1], x_values[16], acc[21]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_81;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_81) : "f"(x_values[17]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[20]), "+f"(acc[21])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_81));
#else
              acc[20] = __fmaf_rn(w_t[2], x_values[17], acc[20]);
              acc[21] = __fmaf_rn(w_t[3], x_values[17], acc[21]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_82;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_82) : "f"(x_values[18]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[20]), "+f"(acc[21])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_82));
#else
              acc[20] = __fmaf_rn(w_t[4], x_values[18], acc[20]);
              acc[21] = __fmaf_rn(w_t[5], x_values[18], acc[21]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_83;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_83) : "f"(x_values[19]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[20]), "+f"(acc[21])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_83));
#else
              acc[20] = __fmaf_rn(w_t[6], x_values[19], acc[20]);
              acc[21] = __fmaf_rn(w_t[7], x_values[19], acc[21]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_84;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_84) : "f"(x_values[20]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[20]), "+f"(acc[21])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_84));
#else
              acc[20] = __fmaf_rn(w_t[8], x_values[20], acc[20]);
              acc[21] = __fmaf_rn(w_t[9], x_values[20], acc[21]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_85;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_85) : "f"(x_values[21]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[20]), "+f"(acc[21])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_85));
#else
              acc[20] = __fmaf_rn(w_t[10], x_values[21], acc[20]);
              acc[21] = __fmaf_rn(w_t[11], x_values[21], acc[21]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_86;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_86) : "f"(x_values[22]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[20]), "+f"(acc[21])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_86));
#else
              acc[20] = __fmaf_rn(w_t[12], x_values[22], acc[20]);
              acc[21] = __fmaf_rn(w_t[13], x_values[22], acc[21]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_87;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_87) : "f"(x_values[23]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[20]), "+f"(acc[21])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_87));
#else
              acc[20] = __fmaf_rn(w_t[14], x_values[23], acc[20]);
              acc[21] = __fmaf_rn(w_t[15], x_values[23], acc[21]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_88;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_88) : "f"(x_values[24]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[28]), "+f"(acc[29])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_88));
#else
              acc[28] = __fmaf_rn(w_t[0], x_values[24], acc[28]);
              acc[29] = __fmaf_rn(w_t[1], x_values[24], acc[29]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_89;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_89) : "f"(x_values[25]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[28]), "+f"(acc[29])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_89));
#else
              acc[28] = __fmaf_rn(w_t[2], x_values[25], acc[28]);
              acc[29] = __fmaf_rn(w_t[3], x_values[25], acc[29]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_90;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_90) : "f"(x_values[26]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[28]), "+f"(acc[29])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_90));
#else
              acc[28] = __fmaf_rn(w_t[4], x_values[26], acc[28]);
              acc[29] = __fmaf_rn(w_t[5], x_values[26], acc[29]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_91;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_91) : "f"(x_values[27]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[28]), "+f"(acc[29])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_91));
#else
              acc[28] = __fmaf_rn(w_t[6], x_values[27], acc[28]);
              acc[29] = __fmaf_rn(w_t[7], x_values[27], acc[29]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_92;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_92) : "f"(x_values[28]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[28]), "+f"(acc[29])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_92));
#else
              acc[28] = __fmaf_rn(w_t[8], x_values[28], acc[28]);
              acc[29] = __fmaf_rn(w_t[9], x_values[28], acc[29]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_93;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_93) : "f"(x_values[29]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[28]), "+f"(acc[29])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_93));
#else
              acc[28] = __fmaf_rn(w_t[10], x_values[29], acc[28]);
              acc[29] = __fmaf_rn(w_t[11], x_values[29], acc[29]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_94;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_94) : "f"(x_values[30]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[28]), "+f"(acc[29])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_94));
#else
              acc[28] = __fmaf_rn(w_t[12], x_values[30], acc[28]);
              acc[29] = __fmaf_rn(w_t[13], x_values[30], acc[29]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_95;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_95) : "f"(x_values[31]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[28]), "+f"(acc[29])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_95));
#else
              acc[28] = __fmaf_rn(w_t[14], x_values[31], acc[28]);
              acc[29] = __fmaf_rn(w_t[15], x_values[31], acc[29]);
#endif
            }
            asm volatile(
                "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
                : "r"(w_ring_addr + (unsigned int)(((stage * 8 + 6) * 1024 + tid_vec) * 2)));
            w_t[0] = __uint_as_float(w_car[0] << 16);
            w_t[2] = __uint_as_float(w_car[0] & 4294901760u);
            w_t[4] = __uint_as_float(w_car[1] << 16);
            w_t[6] = __uint_as_float(w_car[1] & 4294901760u);
            w_t[8] = __uint_as_float(w_car[2] << 16);
            w_t[10] = __uint_as_float(w_car[2] & 4294901760u);
            w_t[12] = __uint_as_float(w_car[3] << 16);
            w_t[14] = __uint_as_float(w_car[3] & 4294901760u);
            asm volatile(
                "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
                : "r"(w_ring_addr + (unsigned int)(((stage * 8 + 6 + 1) * 1024 + tid_vec) * 2)));
            w_t[1] = __uint_as_float(w_car[0] << 16);
            w_t[3] = __uint_as_float(w_car[0] & 4294901760u);
            w_t[5] = __uint_as_float(w_car[1] << 16);
            w_t[7] = __uint_as_float(w_car[1] & 4294901760u);
            w_t[9] = __uint_as_float(w_car[2] << 16);
            w_t[11] = __uint_as_float(w_car[2] & 4294901760u);
            w_t[13] = __uint_as_float(w_car[3] << 16);
            w_t[15] = __uint_as_float(w_car[3] & 4294901760u);
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_96;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_96) : "f"(x_values[0]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[6]), "+f"(acc[7])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_96));
#else
              acc[6] = __fmaf_rn(w_t[0], x_values[0], acc[6]);
              acc[7] = __fmaf_rn(w_t[1], x_values[0], acc[7]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_97;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_97) : "f"(x_values[1]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[6]), "+f"(acc[7])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_97));
#else
              acc[6] = __fmaf_rn(w_t[2], x_values[1], acc[6]);
              acc[7] = __fmaf_rn(w_t[3], x_values[1], acc[7]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_98;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_98) : "f"(x_values[2]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[6]), "+f"(acc[7])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_98));
#else
              acc[6] = __fmaf_rn(w_t[4], x_values[2], acc[6]);
              acc[7] = __fmaf_rn(w_t[5], x_values[2], acc[7]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_99;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_99) : "f"(x_values[3]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[6]), "+f"(acc[7])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_99));
#else
              acc[6] = __fmaf_rn(w_t[6], x_values[3], acc[6]);
              acc[7] = __fmaf_rn(w_t[7], x_values[3], acc[7]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_100;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_100) : "f"(x_values[4]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[6]), "+f"(acc[7])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_100));
#else
              acc[6] = __fmaf_rn(w_t[8], x_values[4], acc[6]);
              acc[7] = __fmaf_rn(w_t[9], x_values[4], acc[7]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_101;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_101) : "f"(x_values[5]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[6]), "+f"(acc[7])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_101));
#else
              acc[6] = __fmaf_rn(w_t[10], x_values[5], acc[6]);
              acc[7] = __fmaf_rn(w_t[11], x_values[5], acc[7]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_102;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_102) : "f"(x_values[6]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[6]), "+f"(acc[7])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_102));
#else
              acc[6] = __fmaf_rn(w_t[12], x_values[6], acc[6]);
              acc[7] = __fmaf_rn(w_t[13], x_values[6], acc[7]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_103;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_103) : "f"(x_values[7]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[6]), "+f"(acc[7])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_103));
#else
              acc[6] = __fmaf_rn(w_t[14], x_values[7], acc[6]);
              acc[7] = __fmaf_rn(w_t[15], x_values[7], acc[7]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_104;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_104) : "f"(x_values[8]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[14]), "+f"(acc[15])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_104));
#else
              acc[14] = __fmaf_rn(w_t[0], x_values[8], acc[14]);
              acc[15] = __fmaf_rn(w_t[1], x_values[8], acc[15]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_105;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_105) : "f"(x_values[9]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[14]), "+f"(acc[15])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_105));
#else
              acc[14] = __fmaf_rn(w_t[2], x_values[9], acc[14]);
              acc[15] = __fmaf_rn(w_t[3], x_values[9], acc[15]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_106;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_106) : "f"(x_values[10]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[14]), "+f"(acc[15])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_106));
#else
              acc[14] = __fmaf_rn(w_t[4], x_values[10], acc[14]);
              acc[15] = __fmaf_rn(w_t[5], x_values[10], acc[15]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_107;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_107) : "f"(x_values[11]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[14]), "+f"(acc[15])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_107));
#else
              acc[14] = __fmaf_rn(w_t[6], x_values[11], acc[14]);
              acc[15] = __fmaf_rn(w_t[7], x_values[11], acc[15]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_108;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_108) : "f"(x_values[12]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[14]), "+f"(acc[15])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_108));
#else
              acc[14] = __fmaf_rn(w_t[8], x_values[12], acc[14]);
              acc[15] = __fmaf_rn(w_t[9], x_values[12], acc[15]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_109;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_109) : "f"(x_values[13]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[14]), "+f"(acc[15])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_109));
#else
              acc[14] = __fmaf_rn(w_t[10], x_values[13], acc[14]);
              acc[15] = __fmaf_rn(w_t[11], x_values[13], acc[15]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_110;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_110) : "f"(x_values[14]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[14]), "+f"(acc[15])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_110));
#else
              acc[14] = __fmaf_rn(w_t[12], x_values[14], acc[14]);
              acc[15] = __fmaf_rn(w_t[13], x_values[14], acc[15]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_111;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_111) : "f"(x_values[15]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[14]), "+f"(acc[15])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_111));
#else
              acc[14] = __fmaf_rn(w_t[14], x_values[15], acc[14]);
              acc[15] = __fmaf_rn(w_t[15], x_values[15], acc[15]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_112;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_112) : "f"(x_values[16]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[22]), "+f"(acc[23])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_112));
#else
              acc[22] = __fmaf_rn(w_t[0], x_values[16], acc[22]);
              acc[23] = __fmaf_rn(w_t[1], x_values[16], acc[23]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_113;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_113) : "f"(x_values[17]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[22]), "+f"(acc[23])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_113));
#else
              acc[22] = __fmaf_rn(w_t[2], x_values[17], acc[22]);
              acc[23] = __fmaf_rn(w_t[3], x_values[17], acc[23]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_114;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_114) : "f"(x_values[18]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[22]), "+f"(acc[23])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_114));
#else
              acc[22] = __fmaf_rn(w_t[4], x_values[18], acc[22]);
              acc[23] = __fmaf_rn(w_t[5], x_values[18], acc[23]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_115;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_115) : "f"(x_values[19]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[22]), "+f"(acc[23])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_115));
#else
              acc[22] = __fmaf_rn(w_t[6], x_values[19], acc[22]);
              acc[23] = __fmaf_rn(w_t[7], x_values[19], acc[23]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_116;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_116) : "f"(x_values[20]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[22]), "+f"(acc[23])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_116));
#else
              acc[22] = __fmaf_rn(w_t[8], x_values[20], acc[22]);
              acc[23] = __fmaf_rn(w_t[9], x_values[20], acc[23]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_117;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_117) : "f"(x_values[21]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[22]), "+f"(acc[23])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_117));
#else
              acc[22] = __fmaf_rn(w_t[10], x_values[21], acc[22]);
              acc[23] = __fmaf_rn(w_t[11], x_values[21], acc[23]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_118;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_118) : "f"(x_values[22]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[22]), "+f"(acc[23])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_118));
#else
              acc[22] = __fmaf_rn(w_t[12], x_values[22], acc[22]);
              acc[23] = __fmaf_rn(w_t[13], x_values[22], acc[23]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_119;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_119) : "f"(x_values[23]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[22]), "+f"(acc[23])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_119));
#else
              acc[22] = __fmaf_rn(w_t[14], x_values[23], acc[22]);
              acc[23] = __fmaf_rn(w_t[15], x_values[23], acc[23]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_120;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_120) : "f"(x_values[24]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[30]), "+f"(acc[31])
                  : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_120));
#else
              acc[30] = __fmaf_rn(w_t[0], x_values[24], acc[30]);
              acc[31] = __fmaf_rn(w_t[1], x_values[24], acc[31]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_121;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_121) : "f"(x_values[25]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[30]), "+f"(acc[31])
                  : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_121));
#else
              acc[30] = __fmaf_rn(w_t[2], x_values[25], acc[30]);
              acc[31] = __fmaf_rn(w_t[3], x_values[25], acc[31]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_122;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_122) : "f"(x_values[26]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[30]), "+f"(acc[31])
                  : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_122));
#else
              acc[30] = __fmaf_rn(w_t[4], x_values[26], acc[30]);
              acc[31] = __fmaf_rn(w_t[5], x_values[26], acc[31]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_123;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_123) : "f"(x_values[27]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[30]), "+f"(acc[31])
                  : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_123));
#else
              acc[30] = __fmaf_rn(w_t[6], x_values[27], acc[30]);
              acc[31] = __fmaf_rn(w_t[7], x_values[27], acc[31]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_124;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_124) : "f"(x_values[28]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[30]), "+f"(acc[31])
                  : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_124));
#else
              acc[30] = __fmaf_rn(w_t[8], x_values[28], acc[30]);
              acc[31] = __fmaf_rn(w_t[9], x_values[28], acc[31]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_125;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_125) : "f"(x_values[29]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[30]), "+f"(acc[31])
                  : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_125));
#else
              acc[30] = __fmaf_rn(w_t[10], x_values[29], acc[30]);
              acc[31] = __fmaf_rn(w_t[11], x_values[29], acc[31]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_126;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_126) : "f"(x_values[30]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[30]), "+f"(acc[31])
                  : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_126));
#else
              acc[30] = __fmaf_rn(w_t[12], x_values[30], acc[30]);
              acc[31] = __fmaf_rn(w_t[13], x_values[30], acc[31]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              unsigned long long _fma_acc_scale2_127;
              asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_127) : "f"(x_values[31]));
              asm volatile(
                  "{\n\t"
                  ".reg .b64 _src2, _acc2, _out2;\n\t"
                  "mov.b64 _src2, {%2, %3};\n\t"
                  "mov.b64 _acc2, {%0, %1};\n\t"
                  "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                  "mov.b64 {%0, %1}, _out2;\n\t"
                  "}"
                  : "+f"(acc[30]), "+f"(acc[31])
                  : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_127));
#else
              acc[30] = __fmaf_rn(w_t[14], x_values[31], acc[30]);
              acc[31] = __fmaf_rn(w_t[15], x_values[31], acc[31]);
#endif
            }
          }
        }
#pragma unroll
        for (int i = 0; i < 16; i++) {
          float _shfl_xor_0 =
              __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) != 0) ? acc[i] : acc[i + 16]), 16);
          red_a[i] = (((lane_0 & 16) != 0) ? acc[i + 16] : acc[i]) + _shfl_xor_0;
        }
#pragma unroll
        for (int i_1 = 0; i_1 < 8; i_1++) {
          float _shfl_xor_1 =
              __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) != 0) ? red_a[i_1] : red_a[i_1 + 8]), 8);
          red_b[i_1] = (((lane_0 & 8) != 0) ? red_a[i_1 + 8] : red_a[i_1]) + _shfl_xor_1;
        }
#pragma unroll
        for (int i_2 = 0; i_2 < 4; i_2++) {
          float _shfl_xor_2 =
              __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 4) != 0) ? red_b[i_2] : red_b[i_2 + 4]), 4);
          red_c[i_2] = (((lane_0 & 4) != 0) ? red_b[i_2 + 4] : red_b[i_2]) + _shfl_xor_2;
        }
#pragma unroll
        for (int i_3 = 0; i_3 < 2; i_3++) {
          float _shfl_xor_3 =
              __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 2) != 0) ? red_c[i_3] : red_c[i_3 + 2]), 2);
          red_d[i_3] = (((lane_0 & 2) != 0) ? red_c[i_3 + 2] : red_c[i_3]) + _shfl_xor_3;
        }
#pragma unroll
        for (int i_4 = 0; i_4 < 1; i_4++) {
          float _shfl_xor_4 =
              __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 1) != 0) ? red_d[i_4] : red_d[i_4 + 1]), 1);
          red_e[i_4] = (((lane_0 & 1) != 0) ? red_d[i_4 + 1] : red_d[i_4]) + _shfl_xor_4;
        }
#pragma unroll
        for (int i_5 = 0; i_5 < 1; i_5++) {
          warp_partials[(lane_0 + i_5) * 4 + warp] = red_e[i_5];
        }
        __syncthreads();
        if (tid < 32) {
          float owned_accum = 0.0f;
#pragma unroll
          for (int source_warp = 0; source_warp < 4; source_warp++) {
            owned_accum += warp_partials[tid * 4 + source_warp];
          }
          int owner_j = tid / 8;
          int owner_rr = tid % 8;
          if (owner_j < count) {
            int owner_route = (int)reinterpret_cast<const unsigned int*>(
                workspace_raw)[off_sorted_routes + start + owner_j];
            *(reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<__nv_bfloat16*>(shrink_out_raw) +
                                               (owner_route * 32 + rank_base_it + owner_rr)) +
              (0)) = __float2bfloat16_rn(owned_accum);
          }
        }
        __syncthreads();
        rank_base_it = rank_base_it + 8;
        weight_words_it = weight_words_it + weight_words_step;
      }
    }
  }
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

}  // extern "C"

#undef BLACKWELL_INF
#undef NUM_MAIN_STAGES
#undef SMEM_TOTAL
#undef SMEM_WARP_PARTIALS_OFF
#undef SMEM_WARP_PARTIALS_STAGE_BYTES
#undef SMEM_WARP_PARTIALS_STRIDE
#undef SMEM_W_RING_OFF
#undef SMEM_W_RING_STAGE_BYTES
#undef SMEM_W_RING_STRIDE
#undef SMEM_X_RING_OFF
#undef SMEM_X_RING_STAGE_BYTES
#undef SMEM_X_RING_STRIDE
#undef THREADS

#define BLACKWELL_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_WARP_PARTIALS_OFF 0
#define SMEM_WARP_PARTIALS_STAGE_BYTES 512
#define SMEM_WARP_PARTIALS_STRIDE 512
#define SMEM_X_RING_OFF 512
#define SMEM_X_RING_STAGE_BYTES 16
#define SMEM_X_RING_STRIDE 16
#define SMEM_W_RING_OFF 512
#define SMEM_W_RING_STAGE_BYTES 32768
#define SMEM_W_RING_STRIDE 32768
#define SMEM_TOTAL 33280
#define THREADS 128

extern "C" {

__global__
__launch_bounds__(128, 6) void kernel_flashinfer_bgmv_moe_shrink_grouped_ring_mixed_bf16_r32(
    uint16_t* __restrict__ shrink_out_raw, uint16_t* __restrict__ x_raw,
    uint16_t* __restrict__ lora_a_raw, long long* __restrict__ sorted_token_ids, int num_pairs,
    int num_experts, int hidden, int num_tiles, int rt_per_cta, int rt_groups,
    unsigned int* __restrict__ workspace_raw, int off_group_offset, int off_tile_table,
    int off_sorted_routes) {
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
  float* warp_partials = reinterpret_cast<float*>(smem_raw + 0);
  const int warp_partials_addr = smem + 0;
  __nv_bfloat16* x_ring = reinterpret_cast<__nv_bfloat16*>(smem_raw + 512);
  const int x_ring_addr = smem + 512;
  __nv_bfloat16* w_ring = reinterpret_cast<__nv_bfloat16*>(smem_raw + 512);
  const int w_ring_addr = smem + 512;

  // === Task calls (dependency order) ===
  asm volatile("griddepcontrol.wait;" ::: "memory");
  int rt_group = blockIdx.x % ((0) ? 4 : rt_groups);
  int half = blockIdx.x / ((0) ? 4 : rt_groups) % 4;
  int tile = blockIdx.x / (((0) ? 4 : rt_groups) * 4);
  int n_tiles = (int)reinterpret_cast<const unsigned int*>(workspace_raw)[0];
  if (tile < n_tiles) {
    int entry = (int)reinterpret_cast<const unsigned int*>(workspace_raw)[off_tile_table + tile];
    int group = entry / 65536;
    int chunk = entry % 65536;
    int start =
        (int)reinterpret_cast<const unsigned int*>(workspace_raw)[off_group_offset + group] +
        chunk * 16 + half * 4;
    int group_end =
        (int)reinterpret_cast<const unsigned int*>(workspace_raw)[off_group_offset + group + 1];
    int count = group_end - start;
    if (count > 4) {
      count = 4;
    }
    if (count > 0) {
      int lora = group / num_experts;
      int expert = group % num_experts;
      int hidden_words = hidden / 2;
      long long weight_row_base = (long long)(lora * num_experts + expert) * 32;
      int routes[4];
      long long tokens[4];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        routes[j] = -1;
        tokens[j] = 0;
        if (count > j) {
          routes[j] = (int)reinterpret_cast<const unsigned int*>(
              workspace_raw)[off_sorted_routes + start + j];
          tokens[j] = sorted_token_ids[routes[j]];
        }
      }
      int rank_tile0 = rt_group * rt_per_cta;
      int rank_tile_end = rank_tile0 + rt_per_cta;
      int rank_base_it = rank_tile0 * 8;
      long long weight_words_it =
          (weight_row_base + (long long)rank_base_it) * (long long)hidden_words;
      long long weight_words_step = hidden_words * 8;
#pragma unroll 1
      for (int rank_tile = rank_tile0; rank_tile < rank_tile_end; rank_tile++) {
        float acc[32];
        unsigned int x_car[16];
        unsigned int xr_car[4];
        unsigned int w_car[4];
        float x_values[32];
        float w_values[8];
        float w_t[16];
        int lane_0 = lane;
        float red_a[16];
        float red_b[8];
        float red_c[4];
        float red_d[2];
        float red_e[1];
#pragma unroll
        for (int owner = 0; owner < 32; owner++) {
          acc[owner] = 0.0f;
        }
        int tid_vec = tid * 8;
        if (tid_vec < hidden) {
          int kw_x0 = tid_vec / 2;
#pragma unroll
          for (int j_1 = 0; j_1 < 4; j_1++) {
            {
              const uint4* _ivptr_0 = reinterpret_cast<const uint4*>(
                  reinterpret_cast<const unsigned int*>(x_raw) +
                  tokens[j_1] * (long long)hidden_words + (long long)kw_x0);
              uint4 _ivld_0;
              asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                           : "=r"(_ivld_0.x), "=r"(_ivld_0.y), "=r"(_ivld_0.z), "=r"(_ivld_0.w)
                           : "l"((const void*)(_ivptr_0))
                           : "memory");
              (x_car + j_1 * 4)[0 + 0] = _ivld_0.x;
              (x_car + j_1 * 4)[0 + 1] = _ivld_0.y;
              (x_car + j_1 * 4)[0 + 2] = _ivld_0.z;
              (x_car + j_1 * 4)[0 + 3] = _ivld_0.w;
            }
          }
        }
#pragma unroll
        for (int d = 0; d < 1; d++) {
          int k_p = d * 1024 + tid_vec;
          if (k_p < hidden) {
            int kw_p = k_p / 2;
#pragma unroll
            for (int r = 0; r < 8; r++) {
              asm volatile(
                  "cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                      w_ring_addr + (unsigned int)(((d * 8 + r) * 1024 + tid_vec) * 2)),
                  "l"(reinterpret_cast<const unsigned int*>(lora_a_raw) +
                      (weight_words_it + (long long)(r * hidden_words) + (long long)kw_p)));
            }
          }
          asm volatile("cp.async.commit_group;");
        }
#pragma unroll 1
        for (int local = 0; local < num_tiles; local++) {
          int stage = local % 2;
          int nxt = local + 1;
          if (nxt < num_tiles) {
            int nstage = nxt % 2;
            int k_n = nxt * 1024 + tid_vec;
            if (k_n < hidden) {
              int kw_n = k_n / 2;
#pragma unroll
              for (int r_1 = 0; r_1 < 8; r_1++) {
                asm volatile(
                    "cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                        w_ring_addr + (unsigned int)(((nstage * 8 + r_1) * 1024 + tid_vec) * 2)),
                    "l"(reinterpret_cast<const unsigned int*>(lora_a_raw) +
                        (weight_words_it + (long long)(r_1 * hidden_words) + (long long)kw_n)));
              }
            }
          }
          asm volatile("cp.async.commit_group;");
          asm volatile("cp.async.wait_group 1;");
          int k_base_r = local * 1024 + tid_vec;
          if (k_base_r < hidden) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                           "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                           "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                           "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
                         : "r"(w_ring_addr + (unsigned int)((stage * 8 * 1024 + tid_vec) * 2)));
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[0])
                  : "r"(x_car[0]), "r"(x_car[1]), "r"(x_car[2]), "r"(x_car[3]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[0] = __fmaf_rn(__uint_as_float(x_car[0] << 16), __uint_as_float(w_car[0] << 16),
                                 acc[0]);
              acc[0] = __fmaf_rn(__uint_as_float(x_car[0] & 0xFFFF0000u),
                                 __uint_as_float(w_car[0] & 0xFFFF0000u), acc[0]);
              acc[0] = __fmaf_rn(__uint_as_float(x_car[1] << 16), __uint_as_float(w_car[1] << 16),
                                 acc[0]);
              acc[0] = __fmaf_rn(__uint_as_float(x_car[1] & 0xFFFF0000u),
                                 __uint_as_float(w_car[1] & 0xFFFF0000u), acc[0]);
              acc[0] = __fmaf_rn(__uint_as_float(x_car[2] << 16), __uint_as_float(w_car[2] << 16),
                                 acc[0]);
              acc[0] = __fmaf_rn(__uint_as_float(x_car[2] & 0xFFFF0000u),
                                 __uint_as_float(w_car[2] & 0xFFFF0000u), acc[0]);
              acc[0] = __fmaf_rn(__uint_as_float(x_car[3] << 16), __uint_as_float(w_car[3] << 16),
                                 acc[0]);
              acc[0] = __fmaf_rn(__uint_as_float(x_car[3] & 0xFFFF0000u),
                                 __uint_as_float(w_car[3] & 0xFFFF0000u), acc[0]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[8])
                  : "r"(x_car[4]), "r"(x_car[5]), "r"(x_car[6]), "r"(x_car[7]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[8] = __fmaf_rn(__uint_as_float(x_car[4] << 16), __uint_as_float(w_car[0] << 16),
                                 acc[8]);
              acc[8] = __fmaf_rn(__uint_as_float(x_car[4] & 0xFFFF0000u),
                                 __uint_as_float(w_car[0] & 0xFFFF0000u), acc[8]);
              acc[8] = __fmaf_rn(__uint_as_float(x_car[5] << 16), __uint_as_float(w_car[1] << 16),
                                 acc[8]);
              acc[8] = __fmaf_rn(__uint_as_float(x_car[5] & 0xFFFF0000u),
                                 __uint_as_float(w_car[1] & 0xFFFF0000u), acc[8]);
              acc[8] = __fmaf_rn(__uint_as_float(x_car[6] << 16), __uint_as_float(w_car[2] << 16),
                                 acc[8]);
              acc[8] = __fmaf_rn(__uint_as_float(x_car[6] & 0xFFFF0000u),
                                 __uint_as_float(w_car[2] & 0xFFFF0000u), acc[8]);
              acc[8] = __fmaf_rn(__uint_as_float(x_car[7] << 16), __uint_as_float(w_car[3] << 16),
                                 acc[8]);
              acc[8] = __fmaf_rn(__uint_as_float(x_car[7] & 0xFFFF0000u),
                                 __uint_as_float(w_car[3] & 0xFFFF0000u), acc[8]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[16])
                  : "r"(x_car[8]), "r"(x_car[9]), "r"(x_car[10]), "r"(x_car[11]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[16] = __fmaf_rn(__uint_as_float(x_car[8] << 16), __uint_as_float(w_car[0] << 16),
                                  acc[16]);
              acc[16] = __fmaf_rn(__uint_as_float(x_car[8] & 0xFFFF0000u),
                                  __uint_as_float(w_car[0] & 0xFFFF0000u), acc[16]);
              acc[16] = __fmaf_rn(__uint_as_float(x_car[9] << 16), __uint_as_float(w_car[1] << 16),
                                  acc[16]);
              acc[16] = __fmaf_rn(__uint_as_float(x_car[9] & 0xFFFF0000u),
                                  __uint_as_float(w_car[1] & 0xFFFF0000u), acc[16]);
              acc[16] = __fmaf_rn(__uint_as_float(x_car[10] << 16), __uint_as_float(w_car[2] << 16),
                                  acc[16]);
              acc[16] = __fmaf_rn(__uint_as_float(x_car[10] & 0xFFFF0000u),
                                  __uint_as_float(w_car[2] & 0xFFFF0000u), acc[16]);
              acc[16] = __fmaf_rn(__uint_as_float(x_car[11] << 16), __uint_as_float(w_car[3] << 16),
                                  acc[16]);
              acc[16] = __fmaf_rn(__uint_as_float(x_car[11] & 0xFFFF0000u),
                                  __uint_as_float(w_car[3] & 0xFFFF0000u), acc[16]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[24])
                  : "r"(x_car[12]), "r"(x_car[13]), "r"(x_car[14]), "r"(x_car[15]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[24] = __fmaf_rn(__uint_as_float(x_car[12] << 16), __uint_as_float(w_car[0] << 16),
                                  acc[24]);
              acc[24] = __fmaf_rn(__uint_as_float(x_car[12] & 0xFFFF0000u),
                                  __uint_as_float(w_car[0] & 0xFFFF0000u), acc[24]);
              acc[24] = __fmaf_rn(__uint_as_float(x_car[13] << 16), __uint_as_float(w_car[1] << 16),
                                  acc[24]);
              acc[24] = __fmaf_rn(__uint_as_float(x_car[13] & 0xFFFF0000u),
                                  __uint_as_float(w_car[1] & 0xFFFF0000u), acc[24]);
              acc[24] = __fmaf_rn(__uint_as_float(x_car[14] << 16), __uint_as_float(w_car[2] << 16),
                                  acc[24]);
              acc[24] = __fmaf_rn(__uint_as_float(x_car[14] & 0xFFFF0000u),
                                  __uint_as_float(w_car[2] & 0xFFFF0000u), acc[24]);
              acc[24] = __fmaf_rn(__uint_as_float(x_car[15] << 16), __uint_as_float(w_car[3] << 16),
                                  acc[24]);
              acc[24] = __fmaf_rn(__uint_as_float(x_car[15] & 0xFFFF0000u),
                                  __uint_as_float(w_car[3] & 0xFFFF0000u), acc[24]);
#endif
            }
            asm volatile(
                "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
                : "r"(w_ring_addr + (unsigned int)(((stage * 8 + 1) * 1024 + tid_vec) * 2)));
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[1])
                  : "r"(x_car[0]), "r"(x_car[1]), "r"(x_car[2]), "r"(x_car[3]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[1] = __fmaf_rn(__uint_as_float(x_car[0] << 16), __uint_as_float(w_car[0] << 16),
                                 acc[1]);
              acc[1] = __fmaf_rn(__uint_as_float(x_car[0] & 0xFFFF0000u),
                                 __uint_as_float(w_car[0] & 0xFFFF0000u), acc[1]);
              acc[1] = __fmaf_rn(__uint_as_float(x_car[1] << 16), __uint_as_float(w_car[1] << 16),
                                 acc[1]);
              acc[1] = __fmaf_rn(__uint_as_float(x_car[1] & 0xFFFF0000u),
                                 __uint_as_float(w_car[1] & 0xFFFF0000u), acc[1]);
              acc[1] = __fmaf_rn(__uint_as_float(x_car[2] << 16), __uint_as_float(w_car[2] << 16),
                                 acc[1]);
              acc[1] = __fmaf_rn(__uint_as_float(x_car[2] & 0xFFFF0000u),
                                 __uint_as_float(w_car[2] & 0xFFFF0000u), acc[1]);
              acc[1] = __fmaf_rn(__uint_as_float(x_car[3] << 16), __uint_as_float(w_car[3] << 16),
                                 acc[1]);
              acc[1] = __fmaf_rn(__uint_as_float(x_car[3] & 0xFFFF0000u),
                                 __uint_as_float(w_car[3] & 0xFFFF0000u), acc[1]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[9])
                  : "r"(x_car[4]), "r"(x_car[5]), "r"(x_car[6]), "r"(x_car[7]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[9] = __fmaf_rn(__uint_as_float(x_car[4] << 16), __uint_as_float(w_car[0] << 16),
                                 acc[9]);
              acc[9] = __fmaf_rn(__uint_as_float(x_car[4] & 0xFFFF0000u),
                                 __uint_as_float(w_car[0] & 0xFFFF0000u), acc[9]);
              acc[9] = __fmaf_rn(__uint_as_float(x_car[5] << 16), __uint_as_float(w_car[1] << 16),
                                 acc[9]);
              acc[9] = __fmaf_rn(__uint_as_float(x_car[5] & 0xFFFF0000u),
                                 __uint_as_float(w_car[1] & 0xFFFF0000u), acc[9]);
              acc[9] = __fmaf_rn(__uint_as_float(x_car[6] << 16), __uint_as_float(w_car[2] << 16),
                                 acc[9]);
              acc[9] = __fmaf_rn(__uint_as_float(x_car[6] & 0xFFFF0000u),
                                 __uint_as_float(w_car[2] & 0xFFFF0000u), acc[9]);
              acc[9] = __fmaf_rn(__uint_as_float(x_car[7] << 16), __uint_as_float(w_car[3] << 16),
                                 acc[9]);
              acc[9] = __fmaf_rn(__uint_as_float(x_car[7] & 0xFFFF0000u),
                                 __uint_as_float(w_car[3] & 0xFFFF0000u), acc[9]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[17])
                  : "r"(x_car[8]), "r"(x_car[9]), "r"(x_car[10]), "r"(x_car[11]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[17] = __fmaf_rn(__uint_as_float(x_car[8] << 16), __uint_as_float(w_car[0] << 16),
                                  acc[17]);
              acc[17] = __fmaf_rn(__uint_as_float(x_car[8] & 0xFFFF0000u),
                                  __uint_as_float(w_car[0] & 0xFFFF0000u), acc[17]);
              acc[17] = __fmaf_rn(__uint_as_float(x_car[9] << 16), __uint_as_float(w_car[1] << 16),
                                  acc[17]);
              acc[17] = __fmaf_rn(__uint_as_float(x_car[9] & 0xFFFF0000u),
                                  __uint_as_float(w_car[1] & 0xFFFF0000u), acc[17]);
              acc[17] = __fmaf_rn(__uint_as_float(x_car[10] << 16), __uint_as_float(w_car[2] << 16),
                                  acc[17]);
              acc[17] = __fmaf_rn(__uint_as_float(x_car[10] & 0xFFFF0000u),
                                  __uint_as_float(w_car[2] & 0xFFFF0000u), acc[17]);
              acc[17] = __fmaf_rn(__uint_as_float(x_car[11] << 16), __uint_as_float(w_car[3] << 16),
                                  acc[17]);
              acc[17] = __fmaf_rn(__uint_as_float(x_car[11] & 0xFFFF0000u),
                                  __uint_as_float(w_car[3] & 0xFFFF0000u), acc[17]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[25])
                  : "r"(x_car[12]), "r"(x_car[13]), "r"(x_car[14]), "r"(x_car[15]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[25] = __fmaf_rn(__uint_as_float(x_car[12] << 16), __uint_as_float(w_car[0] << 16),
                                  acc[25]);
              acc[25] = __fmaf_rn(__uint_as_float(x_car[12] & 0xFFFF0000u),
                                  __uint_as_float(w_car[0] & 0xFFFF0000u), acc[25]);
              acc[25] = __fmaf_rn(__uint_as_float(x_car[13] << 16), __uint_as_float(w_car[1] << 16),
                                  acc[25]);
              acc[25] = __fmaf_rn(__uint_as_float(x_car[13] & 0xFFFF0000u),
                                  __uint_as_float(w_car[1] & 0xFFFF0000u), acc[25]);
              acc[25] = __fmaf_rn(__uint_as_float(x_car[14] << 16), __uint_as_float(w_car[2] << 16),
                                  acc[25]);
              acc[25] = __fmaf_rn(__uint_as_float(x_car[14] & 0xFFFF0000u),
                                  __uint_as_float(w_car[2] & 0xFFFF0000u), acc[25]);
              acc[25] = __fmaf_rn(__uint_as_float(x_car[15] << 16), __uint_as_float(w_car[3] << 16),
                                  acc[25]);
              acc[25] = __fmaf_rn(__uint_as_float(x_car[15] & 0xFFFF0000u),
                                  __uint_as_float(w_car[3] & 0xFFFF0000u), acc[25]);
#endif
            }
            asm volatile(
                "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
                : "r"(w_ring_addr + (unsigned int)(((stage * 8 + 2) * 1024 + tid_vec) * 2)));
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[2])
                  : "r"(x_car[0]), "r"(x_car[1]), "r"(x_car[2]), "r"(x_car[3]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[2] = __fmaf_rn(__uint_as_float(x_car[0] << 16), __uint_as_float(w_car[0] << 16),
                                 acc[2]);
              acc[2] = __fmaf_rn(__uint_as_float(x_car[0] & 0xFFFF0000u),
                                 __uint_as_float(w_car[0] & 0xFFFF0000u), acc[2]);
              acc[2] = __fmaf_rn(__uint_as_float(x_car[1] << 16), __uint_as_float(w_car[1] << 16),
                                 acc[2]);
              acc[2] = __fmaf_rn(__uint_as_float(x_car[1] & 0xFFFF0000u),
                                 __uint_as_float(w_car[1] & 0xFFFF0000u), acc[2]);
              acc[2] = __fmaf_rn(__uint_as_float(x_car[2] << 16), __uint_as_float(w_car[2] << 16),
                                 acc[2]);
              acc[2] = __fmaf_rn(__uint_as_float(x_car[2] & 0xFFFF0000u),
                                 __uint_as_float(w_car[2] & 0xFFFF0000u), acc[2]);
              acc[2] = __fmaf_rn(__uint_as_float(x_car[3] << 16), __uint_as_float(w_car[3] << 16),
                                 acc[2]);
              acc[2] = __fmaf_rn(__uint_as_float(x_car[3] & 0xFFFF0000u),
                                 __uint_as_float(w_car[3] & 0xFFFF0000u), acc[2]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[10])
                  : "r"(x_car[4]), "r"(x_car[5]), "r"(x_car[6]), "r"(x_car[7]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[10] = __fmaf_rn(__uint_as_float(x_car[4] << 16), __uint_as_float(w_car[0] << 16),
                                  acc[10]);
              acc[10] = __fmaf_rn(__uint_as_float(x_car[4] & 0xFFFF0000u),
                                  __uint_as_float(w_car[0] & 0xFFFF0000u), acc[10]);
              acc[10] = __fmaf_rn(__uint_as_float(x_car[5] << 16), __uint_as_float(w_car[1] << 16),
                                  acc[10]);
              acc[10] = __fmaf_rn(__uint_as_float(x_car[5] & 0xFFFF0000u),
                                  __uint_as_float(w_car[1] & 0xFFFF0000u), acc[10]);
              acc[10] = __fmaf_rn(__uint_as_float(x_car[6] << 16), __uint_as_float(w_car[2] << 16),
                                  acc[10]);
              acc[10] = __fmaf_rn(__uint_as_float(x_car[6] & 0xFFFF0000u),
                                  __uint_as_float(w_car[2] & 0xFFFF0000u), acc[10]);
              acc[10] = __fmaf_rn(__uint_as_float(x_car[7] << 16), __uint_as_float(w_car[3] << 16),
                                  acc[10]);
              acc[10] = __fmaf_rn(__uint_as_float(x_car[7] & 0xFFFF0000u),
                                  __uint_as_float(w_car[3] & 0xFFFF0000u), acc[10]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[18])
                  : "r"(x_car[8]), "r"(x_car[9]), "r"(x_car[10]), "r"(x_car[11]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[18] = __fmaf_rn(__uint_as_float(x_car[8] << 16), __uint_as_float(w_car[0] << 16),
                                  acc[18]);
              acc[18] = __fmaf_rn(__uint_as_float(x_car[8] & 0xFFFF0000u),
                                  __uint_as_float(w_car[0] & 0xFFFF0000u), acc[18]);
              acc[18] = __fmaf_rn(__uint_as_float(x_car[9] << 16), __uint_as_float(w_car[1] << 16),
                                  acc[18]);
              acc[18] = __fmaf_rn(__uint_as_float(x_car[9] & 0xFFFF0000u),
                                  __uint_as_float(w_car[1] & 0xFFFF0000u), acc[18]);
              acc[18] = __fmaf_rn(__uint_as_float(x_car[10] << 16), __uint_as_float(w_car[2] << 16),
                                  acc[18]);
              acc[18] = __fmaf_rn(__uint_as_float(x_car[10] & 0xFFFF0000u),
                                  __uint_as_float(w_car[2] & 0xFFFF0000u), acc[18]);
              acc[18] = __fmaf_rn(__uint_as_float(x_car[11] << 16), __uint_as_float(w_car[3] << 16),
                                  acc[18]);
              acc[18] = __fmaf_rn(__uint_as_float(x_car[11] & 0xFFFF0000u),
                                  __uint_as_float(w_car[3] & 0xFFFF0000u), acc[18]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[26])
                  : "r"(x_car[12]), "r"(x_car[13]), "r"(x_car[14]), "r"(x_car[15]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[26] = __fmaf_rn(__uint_as_float(x_car[12] << 16), __uint_as_float(w_car[0] << 16),
                                  acc[26]);
              acc[26] = __fmaf_rn(__uint_as_float(x_car[12] & 0xFFFF0000u),
                                  __uint_as_float(w_car[0] & 0xFFFF0000u), acc[26]);
              acc[26] = __fmaf_rn(__uint_as_float(x_car[13] << 16), __uint_as_float(w_car[1] << 16),
                                  acc[26]);
              acc[26] = __fmaf_rn(__uint_as_float(x_car[13] & 0xFFFF0000u),
                                  __uint_as_float(w_car[1] & 0xFFFF0000u), acc[26]);
              acc[26] = __fmaf_rn(__uint_as_float(x_car[14] << 16), __uint_as_float(w_car[2] << 16),
                                  acc[26]);
              acc[26] = __fmaf_rn(__uint_as_float(x_car[14] & 0xFFFF0000u),
                                  __uint_as_float(w_car[2] & 0xFFFF0000u), acc[26]);
              acc[26] = __fmaf_rn(__uint_as_float(x_car[15] << 16), __uint_as_float(w_car[3] << 16),
                                  acc[26]);
              acc[26] = __fmaf_rn(__uint_as_float(x_car[15] & 0xFFFF0000u),
                                  __uint_as_float(w_car[3] & 0xFFFF0000u), acc[26]);
#endif
            }
            asm volatile(
                "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
                : "r"(w_ring_addr + (unsigned int)(((stage * 8 + 3) * 1024 + tid_vec) * 2)));
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[3])
                  : "r"(x_car[0]), "r"(x_car[1]), "r"(x_car[2]), "r"(x_car[3]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[3] = __fmaf_rn(__uint_as_float(x_car[0] << 16), __uint_as_float(w_car[0] << 16),
                                 acc[3]);
              acc[3] = __fmaf_rn(__uint_as_float(x_car[0] & 0xFFFF0000u),
                                 __uint_as_float(w_car[0] & 0xFFFF0000u), acc[3]);
              acc[3] = __fmaf_rn(__uint_as_float(x_car[1] << 16), __uint_as_float(w_car[1] << 16),
                                 acc[3]);
              acc[3] = __fmaf_rn(__uint_as_float(x_car[1] & 0xFFFF0000u),
                                 __uint_as_float(w_car[1] & 0xFFFF0000u), acc[3]);
              acc[3] = __fmaf_rn(__uint_as_float(x_car[2] << 16), __uint_as_float(w_car[2] << 16),
                                 acc[3]);
              acc[3] = __fmaf_rn(__uint_as_float(x_car[2] & 0xFFFF0000u),
                                 __uint_as_float(w_car[2] & 0xFFFF0000u), acc[3]);
              acc[3] = __fmaf_rn(__uint_as_float(x_car[3] << 16), __uint_as_float(w_car[3] << 16),
                                 acc[3]);
              acc[3] = __fmaf_rn(__uint_as_float(x_car[3] & 0xFFFF0000u),
                                 __uint_as_float(w_car[3] & 0xFFFF0000u), acc[3]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[11])
                  : "r"(x_car[4]), "r"(x_car[5]), "r"(x_car[6]), "r"(x_car[7]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[11] = __fmaf_rn(__uint_as_float(x_car[4] << 16), __uint_as_float(w_car[0] << 16),
                                  acc[11]);
              acc[11] = __fmaf_rn(__uint_as_float(x_car[4] & 0xFFFF0000u),
                                  __uint_as_float(w_car[0] & 0xFFFF0000u), acc[11]);
              acc[11] = __fmaf_rn(__uint_as_float(x_car[5] << 16), __uint_as_float(w_car[1] << 16),
                                  acc[11]);
              acc[11] = __fmaf_rn(__uint_as_float(x_car[5] & 0xFFFF0000u),
                                  __uint_as_float(w_car[1] & 0xFFFF0000u), acc[11]);
              acc[11] = __fmaf_rn(__uint_as_float(x_car[6] << 16), __uint_as_float(w_car[2] << 16),
                                  acc[11]);
              acc[11] = __fmaf_rn(__uint_as_float(x_car[6] & 0xFFFF0000u),
                                  __uint_as_float(w_car[2] & 0xFFFF0000u), acc[11]);
              acc[11] = __fmaf_rn(__uint_as_float(x_car[7] << 16), __uint_as_float(w_car[3] << 16),
                                  acc[11]);
              acc[11] = __fmaf_rn(__uint_as_float(x_car[7] & 0xFFFF0000u),
                                  __uint_as_float(w_car[3] & 0xFFFF0000u), acc[11]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[19])
                  : "r"(x_car[8]), "r"(x_car[9]), "r"(x_car[10]), "r"(x_car[11]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[19] = __fmaf_rn(__uint_as_float(x_car[8] << 16), __uint_as_float(w_car[0] << 16),
                                  acc[19]);
              acc[19] = __fmaf_rn(__uint_as_float(x_car[8] & 0xFFFF0000u),
                                  __uint_as_float(w_car[0] & 0xFFFF0000u), acc[19]);
              acc[19] = __fmaf_rn(__uint_as_float(x_car[9] << 16), __uint_as_float(w_car[1] << 16),
                                  acc[19]);
              acc[19] = __fmaf_rn(__uint_as_float(x_car[9] & 0xFFFF0000u),
                                  __uint_as_float(w_car[1] & 0xFFFF0000u), acc[19]);
              acc[19] = __fmaf_rn(__uint_as_float(x_car[10] << 16), __uint_as_float(w_car[2] << 16),
                                  acc[19]);
              acc[19] = __fmaf_rn(__uint_as_float(x_car[10] & 0xFFFF0000u),
                                  __uint_as_float(w_car[2] & 0xFFFF0000u), acc[19]);
              acc[19] = __fmaf_rn(__uint_as_float(x_car[11] << 16), __uint_as_float(w_car[3] << 16),
                                  acc[19]);
              acc[19] = __fmaf_rn(__uint_as_float(x_car[11] & 0xFFFF0000u),
                                  __uint_as_float(w_car[3] & 0xFFFF0000u), acc[19]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[27])
                  : "r"(x_car[12]), "r"(x_car[13]), "r"(x_car[14]), "r"(x_car[15]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[27] = __fmaf_rn(__uint_as_float(x_car[12] << 16), __uint_as_float(w_car[0] << 16),
                                  acc[27]);
              acc[27] = __fmaf_rn(__uint_as_float(x_car[12] & 0xFFFF0000u),
                                  __uint_as_float(w_car[0] & 0xFFFF0000u), acc[27]);
              acc[27] = __fmaf_rn(__uint_as_float(x_car[13] << 16), __uint_as_float(w_car[1] << 16),
                                  acc[27]);
              acc[27] = __fmaf_rn(__uint_as_float(x_car[13] & 0xFFFF0000u),
                                  __uint_as_float(w_car[1] & 0xFFFF0000u), acc[27]);
              acc[27] = __fmaf_rn(__uint_as_float(x_car[14] << 16), __uint_as_float(w_car[2] << 16),
                                  acc[27]);
              acc[27] = __fmaf_rn(__uint_as_float(x_car[14] & 0xFFFF0000u),
                                  __uint_as_float(w_car[2] & 0xFFFF0000u), acc[27]);
              acc[27] = __fmaf_rn(__uint_as_float(x_car[15] << 16), __uint_as_float(w_car[3] << 16),
                                  acc[27]);
              acc[27] = __fmaf_rn(__uint_as_float(x_car[15] & 0xFFFF0000u),
                                  __uint_as_float(w_car[3] & 0xFFFF0000u), acc[27]);
#endif
            }
            asm volatile(
                "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
                : "r"(w_ring_addr + (unsigned int)(((stage * 8 + 4) * 1024 + tid_vec) * 2)));
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[4])
                  : "r"(x_car[0]), "r"(x_car[1]), "r"(x_car[2]), "r"(x_car[3]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[4] = __fmaf_rn(__uint_as_float(x_car[0] << 16), __uint_as_float(w_car[0] << 16),
                                 acc[4]);
              acc[4] = __fmaf_rn(__uint_as_float(x_car[0] & 0xFFFF0000u),
                                 __uint_as_float(w_car[0] & 0xFFFF0000u), acc[4]);
              acc[4] = __fmaf_rn(__uint_as_float(x_car[1] << 16), __uint_as_float(w_car[1] << 16),
                                 acc[4]);
              acc[4] = __fmaf_rn(__uint_as_float(x_car[1] & 0xFFFF0000u),
                                 __uint_as_float(w_car[1] & 0xFFFF0000u), acc[4]);
              acc[4] = __fmaf_rn(__uint_as_float(x_car[2] << 16), __uint_as_float(w_car[2] << 16),
                                 acc[4]);
              acc[4] = __fmaf_rn(__uint_as_float(x_car[2] & 0xFFFF0000u),
                                 __uint_as_float(w_car[2] & 0xFFFF0000u), acc[4]);
              acc[4] = __fmaf_rn(__uint_as_float(x_car[3] << 16), __uint_as_float(w_car[3] << 16),
                                 acc[4]);
              acc[4] = __fmaf_rn(__uint_as_float(x_car[3] & 0xFFFF0000u),
                                 __uint_as_float(w_car[3] & 0xFFFF0000u), acc[4]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[12])
                  : "r"(x_car[4]), "r"(x_car[5]), "r"(x_car[6]), "r"(x_car[7]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[12] = __fmaf_rn(__uint_as_float(x_car[4] << 16), __uint_as_float(w_car[0] << 16),
                                  acc[12]);
              acc[12] = __fmaf_rn(__uint_as_float(x_car[4] & 0xFFFF0000u),
                                  __uint_as_float(w_car[0] & 0xFFFF0000u), acc[12]);
              acc[12] = __fmaf_rn(__uint_as_float(x_car[5] << 16), __uint_as_float(w_car[1] << 16),
                                  acc[12]);
              acc[12] = __fmaf_rn(__uint_as_float(x_car[5] & 0xFFFF0000u),
                                  __uint_as_float(w_car[1] & 0xFFFF0000u), acc[12]);
              acc[12] = __fmaf_rn(__uint_as_float(x_car[6] << 16), __uint_as_float(w_car[2] << 16),
                                  acc[12]);
              acc[12] = __fmaf_rn(__uint_as_float(x_car[6] & 0xFFFF0000u),
                                  __uint_as_float(w_car[2] & 0xFFFF0000u), acc[12]);
              acc[12] = __fmaf_rn(__uint_as_float(x_car[7] << 16), __uint_as_float(w_car[3] << 16),
                                  acc[12]);
              acc[12] = __fmaf_rn(__uint_as_float(x_car[7] & 0xFFFF0000u),
                                  __uint_as_float(w_car[3] & 0xFFFF0000u), acc[12]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[20])
                  : "r"(x_car[8]), "r"(x_car[9]), "r"(x_car[10]), "r"(x_car[11]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[20] = __fmaf_rn(__uint_as_float(x_car[8] << 16), __uint_as_float(w_car[0] << 16),
                                  acc[20]);
              acc[20] = __fmaf_rn(__uint_as_float(x_car[8] & 0xFFFF0000u),
                                  __uint_as_float(w_car[0] & 0xFFFF0000u), acc[20]);
              acc[20] = __fmaf_rn(__uint_as_float(x_car[9] << 16), __uint_as_float(w_car[1] << 16),
                                  acc[20]);
              acc[20] = __fmaf_rn(__uint_as_float(x_car[9] & 0xFFFF0000u),
                                  __uint_as_float(w_car[1] & 0xFFFF0000u), acc[20]);
              acc[20] = __fmaf_rn(__uint_as_float(x_car[10] << 16), __uint_as_float(w_car[2] << 16),
                                  acc[20]);
              acc[20] = __fmaf_rn(__uint_as_float(x_car[10] & 0xFFFF0000u),
                                  __uint_as_float(w_car[2] & 0xFFFF0000u), acc[20]);
              acc[20] = __fmaf_rn(__uint_as_float(x_car[11] << 16), __uint_as_float(w_car[3] << 16),
                                  acc[20]);
              acc[20] = __fmaf_rn(__uint_as_float(x_car[11] & 0xFFFF0000u),
                                  __uint_as_float(w_car[3] & 0xFFFF0000u), acc[20]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[28])
                  : "r"(x_car[12]), "r"(x_car[13]), "r"(x_car[14]), "r"(x_car[15]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[28] = __fmaf_rn(__uint_as_float(x_car[12] << 16), __uint_as_float(w_car[0] << 16),
                                  acc[28]);
              acc[28] = __fmaf_rn(__uint_as_float(x_car[12] & 0xFFFF0000u),
                                  __uint_as_float(w_car[0] & 0xFFFF0000u), acc[28]);
              acc[28] = __fmaf_rn(__uint_as_float(x_car[13] << 16), __uint_as_float(w_car[1] << 16),
                                  acc[28]);
              acc[28] = __fmaf_rn(__uint_as_float(x_car[13] & 0xFFFF0000u),
                                  __uint_as_float(w_car[1] & 0xFFFF0000u), acc[28]);
              acc[28] = __fmaf_rn(__uint_as_float(x_car[14] << 16), __uint_as_float(w_car[2] << 16),
                                  acc[28]);
              acc[28] = __fmaf_rn(__uint_as_float(x_car[14] & 0xFFFF0000u),
                                  __uint_as_float(w_car[2] & 0xFFFF0000u), acc[28]);
              acc[28] = __fmaf_rn(__uint_as_float(x_car[15] << 16), __uint_as_float(w_car[3] << 16),
                                  acc[28]);
              acc[28] = __fmaf_rn(__uint_as_float(x_car[15] & 0xFFFF0000u),
                                  __uint_as_float(w_car[3] & 0xFFFF0000u), acc[28]);
#endif
            }
            asm volatile(
                "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
                : "r"(w_ring_addr + (unsigned int)(((stage * 8 + 5) * 1024 + tid_vec) * 2)));
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[5])
                  : "r"(x_car[0]), "r"(x_car[1]), "r"(x_car[2]), "r"(x_car[3]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[5] = __fmaf_rn(__uint_as_float(x_car[0] << 16), __uint_as_float(w_car[0] << 16),
                                 acc[5]);
              acc[5] = __fmaf_rn(__uint_as_float(x_car[0] & 0xFFFF0000u),
                                 __uint_as_float(w_car[0] & 0xFFFF0000u), acc[5]);
              acc[5] = __fmaf_rn(__uint_as_float(x_car[1] << 16), __uint_as_float(w_car[1] << 16),
                                 acc[5]);
              acc[5] = __fmaf_rn(__uint_as_float(x_car[1] & 0xFFFF0000u),
                                 __uint_as_float(w_car[1] & 0xFFFF0000u), acc[5]);
              acc[5] = __fmaf_rn(__uint_as_float(x_car[2] << 16), __uint_as_float(w_car[2] << 16),
                                 acc[5]);
              acc[5] = __fmaf_rn(__uint_as_float(x_car[2] & 0xFFFF0000u),
                                 __uint_as_float(w_car[2] & 0xFFFF0000u), acc[5]);
              acc[5] = __fmaf_rn(__uint_as_float(x_car[3] << 16), __uint_as_float(w_car[3] << 16),
                                 acc[5]);
              acc[5] = __fmaf_rn(__uint_as_float(x_car[3] & 0xFFFF0000u),
                                 __uint_as_float(w_car[3] & 0xFFFF0000u), acc[5]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[13])
                  : "r"(x_car[4]), "r"(x_car[5]), "r"(x_car[6]), "r"(x_car[7]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[13] = __fmaf_rn(__uint_as_float(x_car[4] << 16), __uint_as_float(w_car[0] << 16),
                                  acc[13]);
              acc[13] = __fmaf_rn(__uint_as_float(x_car[4] & 0xFFFF0000u),
                                  __uint_as_float(w_car[0] & 0xFFFF0000u), acc[13]);
              acc[13] = __fmaf_rn(__uint_as_float(x_car[5] << 16), __uint_as_float(w_car[1] << 16),
                                  acc[13]);
              acc[13] = __fmaf_rn(__uint_as_float(x_car[5] & 0xFFFF0000u),
                                  __uint_as_float(w_car[1] & 0xFFFF0000u), acc[13]);
              acc[13] = __fmaf_rn(__uint_as_float(x_car[6] << 16), __uint_as_float(w_car[2] << 16),
                                  acc[13]);
              acc[13] = __fmaf_rn(__uint_as_float(x_car[6] & 0xFFFF0000u),
                                  __uint_as_float(w_car[2] & 0xFFFF0000u), acc[13]);
              acc[13] = __fmaf_rn(__uint_as_float(x_car[7] << 16), __uint_as_float(w_car[3] << 16),
                                  acc[13]);
              acc[13] = __fmaf_rn(__uint_as_float(x_car[7] & 0xFFFF0000u),
                                  __uint_as_float(w_car[3] & 0xFFFF0000u), acc[13]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[21])
                  : "r"(x_car[8]), "r"(x_car[9]), "r"(x_car[10]), "r"(x_car[11]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[21] = __fmaf_rn(__uint_as_float(x_car[8] << 16), __uint_as_float(w_car[0] << 16),
                                  acc[21]);
              acc[21] = __fmaf_rn(__uint_as_float(x_car[8] & 0xFFFF0000u),
                                  __uint_as_float(w_car[0] & 0xFFFF0000u), acc[21]);
              acc[21] = __fmaf_rn(__uint_as_float(x_car[9] << 16), __uint_as_float(w_car[1] << 16),
                                  acc[21]);
              acc[21] = __fmaf_rn(__uint_as_float(x_car[9] & 0xFFFF0000u),
                                  __uint_as_float(w_car[1] & 0xFFFF0000u), acc[21]);
              acc[21] = __fmaf_rn(__uint_as_float(x_car[10] << 16), __uint_as_float(w_car[2] << 16),
                                  acc[21]);
              acc[21] = __fmaf_rn(__uint_as_float(x_car[10] & 0xFFFF0000u),
                                  __uint_as_float(w_car[2] & 0xFFFF0000u), acc[21]);
              acc[21] = __fmaf_rn(__uint_as_float(x_car[11] << 16), __uint_as_float(w_car[3] << 16),
                                  acc[21]);
              acc[21] = __fmaf_rn(__uint_as_float(x_car[11] & 0xFFFF0000u),
                                  __uint_as_float(w_car[3] & 0xFFFF0000u), acc[21]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[29])
                  : "r"(x_car[12]), "r"(x_car[13]), "r"(x_car[14]), "r"(x_car[15]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[29] = __fmaf_rn(__uint_as_float(x_car[12] << 16), __uint_as_float(w_car[0] << 16),
                                  acc[29]);
              acc[29] = __fmaf_rn(__uint_as_float(x_car[12] & 0xFFFF0000u),
                                  __uint_as_float(w_car[0] & 0xFFFF0000u), acc[29]);
              acc[29] = __fmaf_rn(__uint_as_float(x_car[13] << 16), __uint_as_float(w_car[1] << 16),
                                  acc[29]);
              acc[29] = __fmaf_rn(__uint_as_float(x_car[13] & 0xFFFF0000u),
                                  __uint_as_float(w_car[1] & 0xFFFF0000u), acc[29]);
              acc[29] = __fmaf_rn(__uint_as_float(x_car[14] << 16), __uint_as_float(w_car[2] << 16),
                                  acc[29]);
              acc[29] = __fmaf_rn(__uint_as_float(x_car[14] & 0xFFFF0000u),
                                  __uint_as_float(w_car[2] & 0xFFFF0000u), acc[29]);
              acc[29] = __fmaf_rn(__uint_as_float(x_car[15] << 16), __uint_as_float(w_car[3] << 16),
                                  acc[29]);
              acc[29] = __fmaf_rn(__uint_as_float(x_car[15] & 0xFFFF0000u),
                                  __uint_as_float(w_car[3] & 0xFFFF0000u), acc[29]);
#endif
            }
            asm volatile(
                "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
                : "r"(w_ring_addr + (unsigned int)(((stage * 8 + 6) * 1024 + tid_vec) * 2)));
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[6])
                  : "r"(x_car[0]), "r"(x_car[1]), "r"(x_car[2]), "r"(x_car[3]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[6] = __fmaf_rn(__uint_as_float(x_car[0] << 16), __uint_as_float(w_car[0] << 16),
                                 acc[6]);
              acc[6] = __fmaf_rn(__uint_as_float(x_car[0] & 0xFFFF0000u),
                                 __uint_as_float(w_car[0] & 0xFFFF0000u), acc[6]);
              acc[6] = __fmaf_rn(__uint_as_float(x_car[1] << 16), __uint_as_float(w_car[1] << 16),
                                 acc[6]);
              acc[6] = __fmaf_rn(__uint_as_float(x_car[1] & 0xFFFF0000u),
                                 __uint_as_float(w_car[1] & 0xFFFF0000u), acc[6]);
              acc[6] = __fmaf_rn(__uint_as_float(x_car[2] << 16), __uint_as_float(w_car[2] << 16),
                                 acc[6]);
              acc[6] = __fmaf_rn(__uint_as_float(x_car[2] & 0xFFFF0000u),
                                 __uint_as_float(w_car[2] & 0xFFFF0000u), acc[6]);
              acc[6] = __fmaf_rn(__uint_as_float(x_car[3] << 16), __uint_as_float(w_car[3] << 16),
                                 acc[6]);
              acc[6] = __fmaf_rn(__uint_as_float(x_car[3] & 0xFFFF0000u),
                                 __uint_as_float(w_car[3] & 0xFFFF0000u), acc[6]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[14])
                  : "r"(x_car[4]), "r"(x_car[5]), "r"(x_car[6]), "r"(x_car[7]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[14] = __fmaf_rn(__uint_as_float(x_car[4] << 16), __uint_as_float(w_car[0] << 16),
                                  acc[14]);
              acc[14] = __fmaf_rn(__uint_as_float(x_car[4] & 0xFFFF0000u),
                                  __uint_as_float(w_car[0] & 0xFFFF0000u), acc[14]);
              acc[14] = __fmaf_rn(__uint_as_float(x_car[5] << 16), __uint_as_float(w_car[1] << 16),
                                  acc[14]);
              acc[14] = __fmaf_rn(__uint_as_float(x_car[5] & 0xFFFF0000u),
                                  __uint_as_float(w_car[1] & 0xFFFF0000u), acc[14]);
              acc[14] = __fmaf_rn(__uint_as_float(x_car[6] << 16), __uint_as_float(w_car[2] << 16),
                                  acc[14]);
              acc[14] = __fmaf_rn(__uint_as_float(x_car[6] & 0xFFFF0000u),
                                  __uint_as_float(w_car[2] & 0xFFFF0000u), acc[14]);
              acc[14] = __fmaf_rn(__uint_as_float(x_car[7] << 16), __uint_as_float(w_car[3] << 16),
                                  acc[14]);
              acc[14] = __fmaf_rn(__uint_as_float(x_car[7] & 0xFFFF0000u),
                                  __uint_as_float(w_car[3] & 0xFFFF0000u), acc[14]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[22])
                  : "r"(x_car[8]), "r"(x_car[9]), "r"(x_car[10]), "r"(x_car[11]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[22] = __fmaf_rn(__uint_as_float(x_car[8] << 16), __uint_as_float(w_car[0] << 16),
                                  acc[22]);
              acc[22] = __fmaf_rn(__uint_as_float(x_car[8] & 0xFFFF0000u),
                                  __uint_as_float(w_car[0] & 0xFFFF0000u), acc[22]);
              acc[22] = __fmaf_rn(__uint_as_float(x_car[9] << 16), __uint_as_float(w_car[1] << 16),
                                  acc[22]);
              acc[22] = __fmaf_rn(__uint_as_float(x_car[9] & 0xFFFF0000u),
                                  __uint_as_float(w_car[1] & 0xFFFF0000u), acc[22]);
              acc[22] = __fmaf_rn(__uint_as_float(x_car[10] << 16), __uint_as_float(w_car[2] << 16),
                                  acc[22]);
              acc[22] = __fmaf_rn(__uint_as_float(x_car[10] & 0xFFFF0000u),
                                  __uint_as_float(w_car[2] & 0xFFFF0000u), acc[22]);
              acc[22] = __fmaf_rn(__uint_as_float(x_car[11] << 16), __uint_as_float(w_car[3] << 16),
                                  acc[22]);
              acc[22] = __fmaf_rn(__uint_as_float(x_car[11] & 0xFFFF0000u),
                                  __uint_as_float(w_car[3] & 0xFFFF0000u), acc[22]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[30])
                  : "r"(x_car[12]), "r"(x_car[13]), "r"(x_car[14]), "r"(x_car[15]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[30] = __fmaf_rn(__uint_as_float(x_car[12] << 16), __uint_as_float(w_car[0] << 16),
                                  acc[30]);
              acc[30] = __fmaf_rn(__uint_as_float(x_car[12] & 0xFFFF0000u),
                                  __uint_as_float(w_car[0] & 0xFFFF0000u), acc[30]);
              acc[30] = __fmaf_rn(__uint_as_float(x_car[13] << 16), __uint_as_float(w_car[1] << 16),
                                  acc[30]);
              acc[30] = __fmaf_rn(__uint_as_float(x_car[13] & 0xFFFF0000u),
                                  __uint_as_float(w_car[1] & 0xFFFF0000u), acc[30]);
              acc[30] = __fmaf_rn(__uint_as_float(x_car[14] << 16), __uint_as_float(w_car[2] << 16),
                                  acc[30]);
              acc[30] = __fmaf_rn(__uint_as_float(x_car[14] & 0xFFFF0000u),
                                  __uint_as_float(w_car[2] & 0xFFFF0000u), acc[30]);
              acc[30] = __fmaf_rn(__uint_as_float(x_car[15] << 16), __uint_as_float(w_car[3] << 16),
                                  acc[30]);
              acc[30] = __fmaf_rn(__uint_as_float(x_car[15] & 0xFFFF0000u),
                                  __uint_as_float(w_car[3] & 0xFFFF0000u), acc[30]);
#endif
            }
            asm volatile(
                "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                  "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
                : "r"(w_ring_addr + (unsigned int)(((stage * 8 + 7) * 1024 + tid_vec) * 2)));
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[7])
                  : "r"(x_car[0]), "r"(x_car[1]), "r"(x_car[2]), "r"(x_car[3]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[7] = __fmaf_rn(__uint_as_float(x_car[0] << 16), __uint_as_float(w_car[0] << 16),
                                 acc[7]);
              acc[7] = __fmaf_rn(__uint_as_float(x_car[0] & 0xFFFF0000u),
                                 __uint_as_float(w_car[0] & 0xFFFF0000u), acc[7]);
              acc[7] = __fmaf_rn(__uint_as_float(x_car[1] << 16), __uint_as_float(w_car[1] << 16),
                                 acc[7]);
              acc[7] = __fmaf_rn(__uint_as_float(x_car[1] & 0xFFFF0000u),
                                 __uint_as_float(w_car[1] & 0xFFFF0000u), acc[7]);
              acc[7] = __fmaf_rn(__uint_as_float(x_car[2] << 16), __uint_as_float(w_car[2] << 16),
                                 acc[7]);
              acc[7] = __fmaf_rn(__uint_as_float(x_car[2] & 0xFFFF0000u),
                                 __uint_as_float(w_car[2] & 0xFFFF0000u), acc[7]);
              acc[7] = __fmaf_rn(__uint_as_float(x_car[3] << 16), __uint_as_float(w_car[3] << 16),
                                 acc[7]);
              acc[7] = __fmaf_rn(__uint_as_float(x_car[3] & 0xFFFF0000u),
                                 __uint_as_float(w_car[3] & 0xFFFF0000u), acc[7]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[15])
                  : "r"(x_car[4]), "r"(x_car[5]), "r"(x_car[6]), "r"(x_car[7]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[15] = __fmaf_rn(__uint_as_float(x_car[4] << 16), __uint_as_float(w_car[0] << 16),
                                  acc[15]);
              acc[15] = __fmaf_rn(__uint_as_float(x_car[4] & 0xFFFF0000u),
                                  __uint_as_float(w_car[0] & 0xFFFF0000u), acc[15]);
              acc[15] = __fmaf_rn(__uint_as_float(x_car[5] << 16), __uint_as_float(w_car[1] << 16),
                                  acc[15]);
              acc[15] = __fmaf_rn(__uint_as_float(x_car[5] & 0xFFFF0000u),
                                  __uint_as_float(w_car[1] & 0xFFFF0000u), acc[15]);
              acc[15] = __fmaf_rn(__uint_as_float(x_car[6] << 16), __uint_as_float(w_car[2] << 16),
                                  acc[15]);
              acc[15] = __fmaf_rn(__uint_as_float(x_car[6] & 0xFFFF0000u),
                                  __uint_as_float(w_car[2] & 0xFFFF0000u), acc[15]);
              acc[15] = __fmaf_rn(__uint_as_float(x_car[7] << 16), __uint_as_float(w_car[3] << 16),
                                  acc[15]);
              acc[15] = __fmaf_rn(__uint_as_float(x_car[7] & 0xFFFF0000u),
                                  __uint_as_float(w_car[3] & 0xFFFF0000u), acc[15]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[23])
                  : "r"(x_car[8]), "r"(x_car[9]), "r"(x_car[10]), "r"(x_car[11]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[23] = __fmaf_rn(__uint_as_float(x_car[8] << 16), __uint_as_float(w_car[0] << 16),
                                  acc[23]);
              acc[23] = __fmaf_rn(__uint_as_float(x_car[8] & 0xFFFF0000u),
                                  __uint_as_float(w_car[0] & 0xFFFF0000u), acc[23]);
              acc[23] = __fmaf_rn(__uint_as_float(x_car[9] << 16), __uint_as_float(w_car[1] << 16),
                                  acc[23]);
              acc[23] = __fmaf_rn(__uint_as_float(x_car[9] & 0xFFFF0000u),
                                  __uint_as_float(w_car[1] & 0xFFFF0000u), acc[23]);
              acc[23] = __fmaf_rn(__uint_as_float(x_car[10] << 16), __uint_as_float(w_car[2] << 16),
                                  acc[23]);
              acc[23] = __fmaf_rn(__uint_as_float(x_car[10] & 0xFFFF0000u),
                                  __uint_as_float(w_car[2] & 0xFFFF0000u), acc[23]);
              acc[23] = __fmaf_rn(__uint_as_float(x_car[11] << 16), __uint_as_float(w_car[3] << 16),
                                  acc[23]);
              acc[23] = __fmaf_rn(__uint_as_float(x_car[11] & 0xFFFF0000u),
                                  __uint_as_float(w_car[3] & 0xFFFF0000u), acc[23]);
#endif
            }
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, "
                  "_b2l, _b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 "
                  "{_b0l, _b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, "
                  "%6;\n\tmov.b32 {_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, "
                  "_a3h}, %4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                  "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                  : "+f"(acc[31])
                  : "r"(x_car[12]), "r"(x_car[13]), "r"(x_car[14]), "r"(x_car[15]), "r"(w_car[0]),
                    "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
              acc[31] = __fmaf_rn(__uint_as_float(x_car[12] << 16), __uint_as_float(w_car[0] << 16),
                                  acc[31]);
              acc[31] = __fmaf_rn(__uint_as_float(x_car[12] & 0xFFFF0000u),
                                  __uint_as_float(w_car[0] & 0xFFFF0000u), acc[31]);
              acc[31] = __fmaf_rn(__uint_as_float(x_car[13] << 16), __uint_as_float(w_car[1] << 16),
                                  acc[31]);
              acc[31] = __fmaf_rn(__uint_as_float(x_car[13] & 0xFFFF0000u),
                                  __uint_as_float(w_car[1] & 0xFFFF0000u), acc[31]);
              acc[31] = __fmaf_rn(__uint_as_float(x_car[14] << 16), __uint_as_float(w_car[2] << 16),
                                  acc[31]);
              acc[31] = __fmaf_rn(__uint_as_float(x_car[14] & 0xFFFF0000u),
                                  __uint_as_float(w_car[2] & 0xFFFF0000u), acc[31]);
              acc[31] = __fmaf_rn(__uint_as_float(x_car[15] << 16), __uint_as_float(w_car[3] << 16),
                                  acc[31]);
              acc[31] = __fmaf_rn(__uint_as_float(x_car[15] & 0xFFFF0000u),
                                  __uint_as_float(w_car[3] & 0xFFFF0000u), acc[31]);
#endif
            }
            int k_xn = k_base_r + 1024;
            if (k_xn < hidden) {
              int kw_xn = k_xn / 2;
#pragma unroll
              for (int j_2 = 0; j_2 < 4; j_2++) {
                {
                  const uint4* _ivptr_1 = reinterpret_cast<const uint4*>(
                      reinterpret_cast<const unsigned int*>(x_raw) +
                      tokens[j_2] * (long long)hidden_words + (long long)kw_xn);
                  uint4 _ivld_1;
                  asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                               : "=r"(_ivld_1.x), "=r"(_ivld_1.y), "=r"(_ivld_1.z), "=r"(_ivld_1.w)
                               : "l"((const void*)(_ivptr_1))
                               : "memory");
                  (x_car + j_2 * 4)[0 + 0] = _ivld_1.x;
                  (x_car + j_2 * 4)[0 + 1] = _ivld_1.y;
                  (x_car + j_2 * 4)[0 + 2] = _ivld_1.z;
                  (x_car + j_2 * 4)[0 + 3] = _ivld_1.w;
                }
              }
            }
          }
        }
#pragma unroll
        for (int i = 0; i < 16; i++) {
          float _shfl_xor_0 =
              __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) != 0) ? acc[i] : acc[i + 16]), 16);
          red_a[i] = (((lane_0 & 16) != 0) ? acc[i + 16] : acc[i]) + _shfl_xor_0;
        }
#pragma unroll
        for (int i_1 = 0; i_1 < 8; i_1++) {
          float _shfl_xor_1 =
              __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) != 0) ? red_a[i_1] : red_a[i_1 + 8]), 8);
          red_b[i_1] = (((lane_0 & 8) != 0) ? red_a[i_1 + 8] : red_a[i_1]) + _shfl_xor_1;
        }
#pragma unroll
        for (int i_2 = 0; i_2 < 4; i_2++) {
          float _shfl_xor_2 =
              __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 4) != 0) ? red_b[i_2] : red_b[i_2 + 4]), 4);
          red_c[i_2] = (((lane_0 & 4) != 0) ? red_b[i_2 + 4] : red_b[i_2]) + _shfl_xor_2;
        }
#pragma unroll
        for (int i_3 = 0; i_3 < 2; i_3++) {
          float _shfl_xor_3 =
              __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 2) != 0) ? red_c[i_3] : red_c[i_3 + 2]), 2);
          red_d[i_3] = (((lane_0 & 2) != 0) ? red_c[i_3 + 2] : red_c[i_3]) + _shfl_xor_3;
        }
#pragma unroll
        for (int i_4 = 0; i_4 < 1; i_4++) {
          float _shfl_xor_4 =
              __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 1) != 0) ? red_d[i_4] : red_d[i_4 + 1]), 1);
          red_e[i_4] = (((lane_0 & 1) != 0) ? red_d[i_4 + 1] : red_d[i_4]) + _shfl_xor_4;
        }
#pragma unroll
        for (int i_5 = 0; i_5 < 1; i_5++) {
          warp_partials[(lane_0 + i_5) * 4 + warp] = red_e[i_5];
        }
        __syncthreads();
        if (tid < 32) {
          float owned_accum = 0.0f;
#pragma unroll
          for (int source_warp = 0; source_warp < 4; source_warp++) {
            owned_accum += warp_partials[tid * 4 + source_warp];
          }
          int owner_j = tid / 8;
          int owner_rr = tid % 8;
          if (owner_j < count) {
            int owner_route = (int)reinterpret_cast<const unsigned int*>(
                workspace_raw)[off_sorted_routes + start + owner_j];
            *(reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<__nv_bfloat16*>(shrink_out_raw) +
                                               (owner_route * 32 + rank_base_it + owner_rr)) +
              (0)) = __float2bfloat16_rn(owned_accum);
          }
        }
        __syncthreads();
        rank_base_it = rank_base_it + 8;
        weight_words_it = weight_words_it + weight_words_step;
      }
    }
  }
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

}  // extern "C"

#undef BLACKWELL_INF
#undef NUM_MAIN_STAGES
#undef SMEM_TOTAL
#undef SMEM_WARP_PARTIALS_OFF
#undef SMEM_WARP_PARTIALS_STAGE_BYTES
#undef SMEM_WARP_PARTIALS_STRIDE
#undef SMEM_W_RING_OFF
#undef SMEM_W_RING_STAGE_BYTES
#undef SMEM_W_RING_STRIDE
#undef SMEM_X_RING_OFF
#undef SMEM_X_RING_STAGE_BYTES
#undef SMEM_X_RING_STRIDE
#undef THREADS

#define BLACKWELL_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_WARP_PARTIALS_OFF 0
#define SMEM_WARP_PARTIALS_STAGE_BYTES 512
#define SMEM_WARP_PARTIALS_STRIDE 512
#define SMEM_X_RING_OFF 0
#define SMEM_X_RING_STAGE_BYTES 16
#define SMEM_X_RING_STRIDE 16
#define SMEM_W_RING_OFF 0
#define SMEM_W_RING_STAGE_BYTES 16
#define SMEM_W_RING_STRIDE 16
#define SMEM_TOTAL 512
#define THREADS 128

extern "C" {

__global__ __launch_bounds__(128, 4) void kernel_flashinfer_bgmv_moe_shrink_grouped_single_bf16_r32(
    uint16_t* __restrict__ shrink_out_raw, uint16_t* __restrict__ x_raw,
    uint16_t* __restrict__ lora_a_raw, long long* __restrict__ sorted_token_ids, int num_pairs,
    int num_experts, int hidden, int num_tiles, int rt_per_cta, int rt_groups,
    unsigned int* __restrict__ workspace_raw, int off_group_offset, int off_tile_table,
    int off_sorted_routes) {
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
  float* warp_partials = reinterpret_cast<float*>(smem_raw + 0);
  const int warp_partials_addr = smem + 0;
  __nv_bfloat16* x_ring = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
  const int x_ring_addr = smem + 0;
  __nv_bfloat16* w_ring = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
  const int w_ring_addr = smem + 0;

  // === Task calls (dependency order) ===
  asm volatile("griddepcontrol.wait;" ::: "memory");
  int rt_group = blockIdx.x % ((1) ? 4 : rt_groups);
  int half = blockIdx.x / ((1) ? 4 : rt_groups) % 4;
  int tile = blockIdx.x / (((1) ? 4 : rt_groups) * 4);
  int n_tiles = (int)reinterpret_cast<const unsigned int*>(workspace_raw)[0];
  if (tile < n_tiles) {
    int entry = (int)reinterpret_cast<const unsigned int*>(workspace_raw)[off_tile_table + tile];
    int group = entry / 65536;
    int chunk = entry % 65536;
    int start =
        (int)reinterpret_cast<const unsigned int*>(workspace_raw)[off_group_offset + group] +
        chunk * 16 + half * 4;
    int group_end =
        (int)reinterpret_cast<const unsigned int*>(workspace_raw)[off_group_offset + group + 1];
    int count = group_end - start;
    if (count > 4) {
      count = 4;
    }
    if (count > 0) {
      int lora = group / num_experts;
      int expert = group % num_experts;
      int hidden_words = hidden / 2;
      long long weight_row_base = (long long)(lora * num_experts + expert) * 32;
      int routes[4];
      long long tokens[4];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        routes[j] = -1;
        tokens[j] = 0;
        if (count > j) {
          routes[j] = (int)reinterpret_cast<const unsigned int*>(
              workspace_raw)[off_sorted_routes + start + j];
          tokens[j] = sorted_token_ids[routes[j]];
        }
      }
      int rank_base0 = rt_group * 8;
      long long weight_words0 = (weight_row_base + (long long)rank_base0) * (long long)hidden_words;
      float acc[32];
      unsigned int x_car[16];
      unsigned int xr_car[4];
      unsigned int w_car[4];
      float x_values[32];
      float w_values[8];
      float w_t[16];
      int lane_0 = lane;
      float red_a[16];
      float red_b[8];
      float red_c[4];
      float red_d[2];
      float red_e[1];
#pragma unroll
      for (int owner = 0; owner < 32; owner++) {
        acc[owner] = 0.0f;
      }
#pragma unroll 1
      for (int local = 0; local < num_tiles; local++) {
        int k_base = local * 1024 + tid * 8;
        if (k_base < hidden) {
          int k_words = k_base / 2;
#pragma unroll
          for (int j_1 = 0; j_1 < 4; j_1++) {
            {
              const uint4* _ivptr_0 = reinterpret_cast<const uint4*>(
                  reinterpret_cast<const unsigned int*>(x_raw) +
                  tokens[j_1] * (long long)hidden_words + (long long)k_words);
              uint4 _ivld_0;
              asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                           : "=r"(_ivld_0.x), "=r"(_ivld_0.y), "=r"(_ivld_0.z), "=r"(_ivld_0.w)
                           : "l"((const void*)(_ivptr_0))
                           : "memory");
              (x_car + j_1 * 4)[0 + 0] = _ivld_0.x;
              (x_car + j_1 * 4)[0 + 1] = _ivld_0.y;
              (x_car + j_1 * 4)[0 + 2] = _ivld_0.z;
              (x_car + j_1 * 4)[0 + 3] = _ivld_0.w;
            }
          }
#pragma unroll
          for (int j_2 = 0; j_2 < 4; j_2++) {
#pragma unroll
            for (int pair = 0; pair < 4; pair++) {
              x_values[j_2 * 8 + 2 * pair] = __uint_as_float(x_car[j_2 * 4 + pair] << 16);
              x_values[j_2 * 8 + 2 * pair + 1] =
                  __uint_as_float(x_car[j_2 * 4 + pair] & 4294901760u);
            }
          }
          {
            uint32_t _uv4_1_0;
            uint32_t _uv4_1_1;
            uint32_t _uv4_1_2;
            uint32_t _uv4_1_3;
            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(_uv4_1_0), "=r"(_uv4_1_1), "=r"(_uv4_1_2), "=r"(_uv4_1_3)
                         : "l"((const void*)(reinterpret_cast<const unsigned int*>(lora_a_raw) +
                                             (weight_words0 + (long long)k_words)))
                         : "memory");
            w_car[0 + 0] = _uv4_1_0;
            w_car[0 + 1] = _uv4_1_1;
            w_car[0 + 2] = _uv4_1_2;
            w_car[0 + 3] = _uv4_1_3;
          }
          w_t[0] = __uint_as_float(w_car[0] << 16);
          w_t[2] = __uint_as_float(w_car[0] & 4294901760u);
          w_t[4] = __uint_as_float(w_car[1] << 16);
          w_t[6] = __uint_as_float(w_car[1] & 4294901760u);
          w_t[8] = __uint_as_float(w_car[2] << 16);
          w_t[10] = __uint_as_float(w_car[2] & 4294901760u);
          w_t[12] = __uint_as_float(w_car[3] << 16);
          w_t[14] = __uint_as_float(w_car[3] & 4294901760u);
          {
            uint32_t _uv4_2_0;
            uint32_t _uv4_2_1;
            uint32_t _uv4_2_2;
            uint32_t _uv4_2_3;
            asm volatile(
                "ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(_uv4_2_0), "=r"(_uv4_2_1), "=r"(_uv4_2_2), "=r"(_uv4_2_3)
                : "l"((const void*)(reinterpret_cast<const unsigned int*>(lora_a_raw) +
                                    (weight_words0 + (long long)hidden_words + (long long)k_words)))
                : "memory");
            w_car[0 + 0] = _uv4_2_0;
            w_car[0 + 1] = _uv4_2_1;
            w_car[0 + 2] = _uv4_2_2;
            w_car[0 + 3] = _uv4_2_3;
          }
          w_t[1] = __uint_as_float(w_car[0] << 16);
          w_t[3] = __uint_as_float(w_car[0] & 4294901760u);
          w_t[5] = __uint_as_float(w_car[1] << 16);
          w_t[7] = __uint_as_float(w_car[1] & 4294901760u);
          w_t[9] = __uint_as_float(w_car[2] << 16);
          w_t[11] = __uint_as_float(w_car[2] & 4294901760u);
          w_t[13] = __uint_as_float(w_car[3] << 16);
          w_t[15] = __uint_as_float(w_car[3] & 4294901760u);
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_3;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_3) : "f"(x_values[0]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[0]), "+f"(acc[1])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_3));
#else
            acc[0] = __fmaf_rn(w_t[0], x_values[0], acc[0]);
            acc[1] = __fmaf_rn(w_t[1], x_values[0], acc[1]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_4;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_4) : "f"(x_values[1]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[0]), "+f"(acc[1])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_4));
#else
            acc[0] = __fmaf_rn(w_t[2], x_values[1], acc[0]);
            acc[1] = __fmaf_rn(w_t[3], x_values[1], acc[1]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_5;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_5) : "f"(x_values[2]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[0]), "+f"(acc[1])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_5));
#else
            acc[0] = __fmaf_rn(w_t[4], x_values[2], acc[0]);
            acc[1] = __fmaf_rn(w_t[5], x_values[2], acc[1]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_6;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_6) : "f"(x_values[3]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[0]), "+f"(acc[1])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_6));
#else
            acc[0] = __fmaf_rn(w_t[6], x_values[3], acc[0]);
            acc[1] = __fmaf_rn(w_t[7], x_values[3], acc[1]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_7;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_7) : "f"(x_values[4]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[0]), "+f"(acc[1])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_7));
#else
            acc[0] = __fmaf_rn(w_t[8], x_values[4], acc[0]);
            acc[1] = __fmaf_rn(w_t[9], x_values[4], acc[1]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_8;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_8) : "f"(x_values[5]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[0]), "+f"(acc[1])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_8));
#else
            acc[0] = __fmaf_rn(w_t[10], x_values[5], acc[0]);
            acc[1] = __fmaf_rn(w_t[11], x_values[5], acc[1]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_9;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_9) : "f"(x_values[6]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[0]), "+f"(acc[1])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_9));
#else
            acc[0] = __fmaf_rn(w_t[12], x_values[6], acc[0]);
            acc[1] = __fmaf_rn(w_t[13], x_values[6], acc[1]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_10;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_10) : "f"(x_values[7]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[0]), "+f"(acc[1])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_10));
#else
            acc[0] = __fmaf_rn(w_t[14], x_values[7], acc[0]);
            acc[1] = __fmaf_rn(w_t[15], x_values[7], acc[1]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_11;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_11) : "f"(x_values[8]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[8]), "+f"(acc[9])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_11));
#else
            acc[8] = __fmaf_rn(w_t[0], x_values[8], acc[8]);
            acc[9] = __fmaf_rn(w_t[1], x_values[8], acc[9]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_12;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_12) : "f"(x_values[9]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[8]), "+f"(acc[9])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_12));
#else
            acc[8] = __fmaf_rn(w_t[2], x_values[9], acc[8]);
            acc[9] = __fmaf_rn(w_t[3], x_values[9], acc[9]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_13;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_13) : "f"(x_values[10]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[8]), "+f"(acc[9])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_13));
#else
            acc[8] = __fmaf_rn(w_t[4], x_values[10], acc[8]);
            acc[9] = __fmaf_rn(w_t[5], x_values[10], acc[9]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_14;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_14) : "f"(x_values[11]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[8]), "+f"(acc[9])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_14));
#else
            acc[8] = __fmaf_rn(w_t[6], x_values[11], acc[8]);
            acc[9] = __fmaf_rn(w_t[7], x_values[11], acc[9]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_15;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_15) : "f"(x_values[12]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[8]), "+f"(acc[9])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_15));
#else
            acc[8] = __fmaf_rn(w_t[8], x_values[12], acc[8]);
            acc[9] = __fmaf_rn(w_t[9], x_values[12], acc[9]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_16;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_16) : "f"(x_values[13]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[8]), "+f"(acc[9])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_16));
#else
            acc[8] = __fmaf_rn(w_t[10], x_values[13], acc[8]);
            acc[9] = __fmaf_rn(w_t[11], x_values[13], acc[9]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_17;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_17) : "f"(x_values[14]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[8]), "+f"(acc[9])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_17));
#else
            acc[8] = __fmaf_rn(w_t[12], x_values[14], acc[8]);
            acc[9] = __fmaf_rn(w_t[13], x_values[14], acc[9]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_18;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_18) : "f"(x_values[15]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[8]), "+f"(acc[9])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_18));
#else
            acc[8] = __fmaf_rn(w_t[14], x_values[15], acc[8]);
            acc[9] = __fmaf_rn(w_t[15], x_values[15], acc[9]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_19;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_19) : "f"(x_values[16]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[16]), "+f"(acc[17])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_19));
#else
            acc[16] = __fmaf_rn(w_t[0], x_values[16], acc[16]);
            acc[17] = __fmaf_rn(w_t[1], x_values[16], acc[17]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_20;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_20) : "f"(x_values[17]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[16]), "+f"(acc[17])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_20));
#else
            acc[16] = __fmaf_rn(w_t[2], x_values[17], acc[16]);
            acc[17] = __fmaf_rn(w_t[3], x_values[17], acc[17]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_21;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_21) : "f"(x_values[18]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[16]), "+f"(acc[17])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_21));
#else
            acc[16] = __fmaf_rn(w_t[4], x_values[18], acc[16]);
            acc[17] = __fmaf_rn(w_t[5], x_values[18], acc[17]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_22;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_22) : "f"(x_values[19]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[16]), "+f"(acc[17])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_22));
#else
            acc[16] = __fmaf_rn(w_t[6], x_values[19], acc[16]);
            acc[17] = __fmaf_rn(w_t[7], x_values[19], acc[17]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_23;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_23) : "f"(x_values[20]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[16]), "+f"(acc[17])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_23));
#else
            acc[16] = __fmaf_rn(w_t[8], x_values[20], acc[16]);
            acc[17] = __fmaf_rn(w_t[9], x_values[20], acc[17]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_24;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_24) : "f"(x_values[21]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[16]), "+f"(acc[17])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_24));
#else
            acc[16] = __fmaf_rn(w_t[10], x_values[21], acc[16]);
            acc[17] = __fmaf_rn(w_t[11], x_values[21], acc[17]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_25;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_25) : "f"(x_values[22]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[16]), "+f"(acc[17])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_25));
#else
            acc[16] = __fmaf_rn(w_t[12], x_values[22], acc[16]);
            acc[17] = __fmaf_rn(w_t[13], x_values[22], acc[17]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_26;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_26) : "f"(x_values[23]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[16]), "+f"(acc[17])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_26));
#else
            acc[16] = __fmaf_rn(w_t[14], x_values[23], acc[16]);
            acc[17] = __fmaf_rn(w_t[15], x_values[23], acc[17]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_27;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_27) : "f"(x_values[24]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[24]), "+f"(acc[25])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_27));
#else
            acc[24] = __fmaf_rn(w_t[0], x_values[24], acc[24]);
            acc[25] = __fmaf_rn(w_t[1], x_values[24], acc[25]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_28;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_28) : "f"(x_values[25]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[24]), "+f"(acc[25])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_28));
#else
            acc[24] = __fmaf_rn(w_t[2], x_values[25], acc[24]);
            acc[25] = __fmaf_rn(w_t[3], x_values[25], acc[25]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_29;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_29) : "f"(x_values[26]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[24]), "+f"(acc[25])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_29));
#else
            acc[24] = __fmaf_rn(w_t[4], x_values[26], acc[24]);
            acc[25] = __fmaf_rn(w_t[5], x_values[26], acc[25]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_30;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_30) : "f"(x_values[27]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[24]), "+f"(acc[25])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_30));
#else
            acc[24] = __fmaf_rn(w_t[6], x_values[27], acc[24]);
            acc[25] = __fmaf_rn(w_t[7], x_values[27], acc[25]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_31;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_31) : "f"(x_values[28]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[24]), "+f"(acc[25])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_31));
#else
            acc[24] = __fmaf_rn(w_t[8], x_values[28], acc[24]);
            acc[25] = __fmaf_rn(w_t[9], x_values[28], acc[25]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_32;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_32) : "f"(x_values[29]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[24]), "+f"(acc[25])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_32));
#else
            acc[24] = __fmaf_rn(w_t[10], x_values[29], acc[24]);
            acc[25] = __fmaf_rn(w_t[11], x_values[29], acc[25]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_33;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_33) : "f"(x_values[30]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[24]), "+f"(acc[25])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_33));
#else
            acc[24] = __fmaf_rn(w_t[12], x_values[30], acc[24]);
            acc[25] = __fmaf_rn(w_t[13], x_values[30], acc[25]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_34;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_34) : "f"(x_values[31]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[24]), "+f"(acc[25])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_34));
#else
            acc[24] = __fmaf_rn(w_t[14], x_values[31], acc[24]);
            acc[25] = __fmaf_rn(w_t[15], x_values[31], acc[25]);
#endif
          }
          {
            uint32_t _uv4_35_0;
            uint32_t _uv4_35_1;
            uint32_t _uv4_35_2;
            uint32_t _uv4_35_3;
            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(_uv4_35_0), "=r"(_uv4_35_1), "=r"(_uv4_35_2), "=r"(_uv4_35_3)
                         : "l"((const void*)(reinterpret_cast<const unsigned int*>(lora_a_raw) +
                                             (weight_words0 + (long long)(2 * hidden_words) +
                                              (long long)k_words)))
                         : "memory");
            w_car[0 + 0] = _uv4_35_0;
            w_car[0 + 1] = _uv4_35_1;
            w_car[0 + 2] = _uv4_35_2;
            w_car[0 + 3] = _uv4_35_3;
          }
          w_t[0] = __uint_as_float(w_car[0] << 16);
          w_t[2] = __uint_as_float(w_car[0] & 4294901760u);
          w_t[4] = __uint_as_float(w_car[1] << 16);
          w_t[6] = __uint_as_float(w_car[1] & 4294901760u);
          w_t[8] = __uint_as_float(w_car[2] << 16);
          w_t[10] = __uint_as_float(w_car[2] & 4294901760u);
          w_t[12] = __uint_as_float(w_car[3] << 16);
          w_t[14] = __uint_as_float(w_car[3] & 4294901760u);
          {
            uint32_t _uv4_36_0;
            uint32_t _uv4_36_1;
            uint32_t _uv4_36_2;
            uint32_t _uv4_36_3;
            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(_uv4_36_0), "=r"(_uv4_36_1), "=r"(_uv4_36_2), "=r"(_uv4_36_3)
                         : "l"((const void*)(reinterpret_cast<const unsigned int*>(lora_a_raw) +
                                             (weight_words0 + (long long)(3 * hidden_words) +
                                              (long long)k_words)))
                         : "memory");
            w_car[0 + 0] = _uv4_36_0;
            w_car[0 + 1] = _uv4_36_1;
            w_car[0 + 2] = _uv4_36_2;
            w_car[0 + 3] = _uv4_36_3;
          }
          w_t[1] = __uint_as_float(w_car[0] << 16);
          w_t[3] = __uint_as_float(w_car[0] & 4294901760u);
          w_t[5] = __uint_as_float(w_car[1] << 16);
          w_t[7] = __uint_as_float(w_car[1] & 4294901760u);
          w_t[9] = __uint_as_float(w_car[2] << 16);
          w_t[11] = __uint_as_float(w_car[2] & 4294901760u);
          w_t[13] = __uint_as_float(w_car[3] << 16);
          w_t[15] = __uint_as_float(w_car[3] & 4294901760u);
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_37;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_37) : "f"(x_values[0]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[2]), "+f"(acc[3])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_37));
#else
            acc[2] = __fmaf_rn(w_t[0], x_values[0], acc[2]);
            acc[3] = __fmaf_rn(w_t[1], x_values[0], acc[3]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_38;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_38) : "f"(x_values[1]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[2]), "+f"(acc[3])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_38));
#else
            acc[2] = __fmaf_rn(w_t[2], x_values[1], acc[2]);
            acc[3] = __fmaf_rn(w_t[3], x_values[1], acc[3]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_39;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_39) : "f"(x_values[2]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[2]), "+f"(acc[3])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_39));
#else
            acc[2] = __fmaf_rn(w_t[4], x_values[2], acc[2]);
            acc[3] = __fmaf_rn(w_t[5], x_values[2], acc[3]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_40;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_40) : "f"(x_values[3]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[2]), "+f"(acc[3])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_40));
#else
            acc[2] = __fmaf_rn(w_t[6], x_values[3], acc[2]);
            acc[3] = __fmaf_rn(w_t[7], x_values[3], acc[3]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_41;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_41) : "f"(x_values[4]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[2]), "+f"(acc[3])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_41));
#else
            acc[2] = __fmaf_rn(w_t[8], x_values[4], acc[2]);
            acc[3] = __fmaf_rn(w_t[9], x_values[4], acc[3]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_42;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_42) : "f"(x_values[5]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[2]), "+f"(acc[3])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_42));
#else
            acc[2] = __fmaf_rn(w_t[10], x_values[5], acc[2]);
            acc[3] = __fmaf_rn(w_t[11], x_values[5], acc[3]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_43;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_43) : "f"(x_values[6]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[2]), "+f"(acc[3])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_43));
#else
            acc[2] = __fmaf_rn(w_t[12], x_values[6], acc[2]);
            acc[3] = __fmaf_rn(w_t[13], x_values[6], acc[3]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_44;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_44) : "f"(x_values[7]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[2]), "+f"(acc[3])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_44));
#else
            acc[2] = __fmaf_rn(w_t[14], x_values[7], acc[2]);
            acc[3] = __fmaf_rn(w_t[15], x_values[7], acc[3]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_45;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_45) : "f"(x_values[8]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[10]), "+f"(acc[11])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_45));
#else
            acc[10] = __fmaf_rn(w_t[0], x_values[8], acc[10]);
            acc[11] = __fmaf_rn(w_t[1], x_values[8], acc[11]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_46;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_46) : "f"(x_values[9]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[10]), "+f"(acc[11])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_46));
#else
            acc[10] = __fmaf_rn(w_t[2], x_values[9], acc[10]);
            acc[11] = __fmaf_rn(w_t[3], x_values[9], acc[11]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_47;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_47) : "f"(x_values[10]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[10]), "+f"(acc[11])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_47));
#else
            acc[10] = __fmaf_rn(w_t[4], x_values[10], acc[10]);
            acc[11] = __fmaf_rn(w_t[5], x_values[10], acc[11]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_48;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_48) : "f"(x_values[11]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[10]), "+f"(acc[11])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_48));
#else
            acc[10] = __fmaf_rn(w_t[6], x_values[11], acc[10]);
            acc[11] = __fmaf_rn(w_t[7], x_values[11], acc[11]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_49;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_49) : "f"(x_values[12]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[10]), "+f"(acc[11])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_49));
#else
            acc[10] = __fmaf_rn(w_t[8], x_values[12], acc[10]);
            acc[11] = __fmaf_rn(w_t[9], x_values[12], acc[11]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_50;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_50) : "f"(x_values[13]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[10]), "+f"(acc[11])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_50));
#else
            acc[10] = __fmaf_rn(w_t[10], x_values[13], acc[10]);
            acc[11] = __fmaf_rn(w_t[11], x_values[13], acc[11]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_51;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_51) : "f"(x_values[14]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[10]), "+f"(acc[11])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_51));
#else
            acc[10] = __fmaf_rn(w_t[12], x_values[14], acc[10]);
            acc[11] = __fmaf_rn(w_t[13], x_values[14], acc[11]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_52;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_52) : "f"(x_values[15]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[10]), "+f"(acc[11])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_52));
#else
            acc[10] = __fmaf_rn(w_t[14], x_values[15], acc[10]);
            acc[11] = __fmaf_rn(w_t[15], x_values[15], acc[11]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_53;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_53) : "f"(x_values[16]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[18]), "+f"(acc[19])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_53));
#else
            acc[18] = __fmaf_rn(w_t[0], x_values[16], acc[18]);
            acc[19] = __fmaf_rn(w_t[1], x_values[16], acc[19]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_54;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_54) : "f"(x_values[17]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[18]), "+f"(acc[19])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_54));
#else
            acc[18] = __fmaf_rn(w_t[2], x_values[17], acc[18]);
            acc[19] = __fmaf_rn(w_t[3], x_values[17], acc[19]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_55;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_55) : "f"(x_values[18]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[18]), "+f"(acc[19])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_55));
#else
            acc[18] = __fmaf_rn(w_t[4], x_values[18], acc[18]);
            acc[19] = __fmaf_rn(w_t[5], x_values[18], acc[19]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_56;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_56) : "f"(x_values[19]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[18]), "+f"(acc[19])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_56));
#else
            acc[18] = __fmaf_rn(w_t[6], x_values[19], acc[18]);
            acc[19] = __fmaf_rn(w_t[7], x_values[19], acc[19]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_57;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_57) : "f"(x_values[20]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[18]), "+f"(acc[19])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_57));
#else
            acc[18] = __fmaf_rn(w_t[8], x_values[20], acc[18]);
            acc[19] = __fmaf_rn(w_t[9], x_values[20], acc[19]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_58;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_58) : "f"(x_values[21]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[18]), "+f"(acc[19])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_58));
#else
            acc[18] = __fmaf_rn(w_t[10], x_values[21], acc[18]);
            acc[19] = __fmaf_rn(w_t[11], x_values[21], acc[19]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_59;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_59) : "f"(x_values[22]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[18]), "+f"(acc[19])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_59));
#else
            acc[18] = __fmaf_rn(w_t[12], x_values[22], acc[18]);
            acc[19] = __fmaf_rn(w_t[13], x_values[22], acc[19]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_60;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_60) : "f"(x_values[23]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[18]), "+f"(acc[19])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_60));
#else
            acc[18] = __fmaf_rn(w_t[14], x_values[23], acc[18]);
            acc[19] = __fmaf_rn(w_t[15], x_values[23], acc[19]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_61;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_61) : "f"(x_values[24]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[26]), "+f"(acc[27])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_61));
#else
            acc[26] = __fmaf_rn(w_t[0], x_values[24], acc[26]);
            acc[27] = __fmaf_rn(w_t[1], x_values[24], acc[27]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_62;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_62) : "f"(x_values[25]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[26]), "+f"(acc[27])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_62));
#else
            acc[26] = __fmaf_rn(w_t[2], x_values[25], acc[26]);
            acc[27] = __fmaf_rn(w_t[3], x_values[25], acc[27]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_63;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_63) : "f"(x_values[26]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[26]), "+f"(acc[27])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_63));
#else
            acc[26] = __fmaf_rn(w_t[4], x_values[26], acc[26]);
            acc[27] = __fmaf_rn(w_t[5], x_values[26], acc[27]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_64;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_64) : "f"(x_values[27]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[26]), "+f"(acc[27])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_64));
#else
            acc[26] = __fmaf_rn(w_t[6], x_values[27], acc[26]);
            acc[27] = __fmaf_rn(w_t[7], x_values[27], acc[27]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_65;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_65) : "f"(x_values[28]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[26]), "+f"(acc[27])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_65));
#else
            acc[26] = __fmaf_rn(w_t[8], x_values[28], acc[26]);
            acc[27] = __fmaf_rn(w_t[9], x_values[28], acc[27]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_66;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_66) : "f"(x_values[29]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[26]), "+f"(acc[27])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_66));
#else
            acc[26] = __fmaf_rn(w_t[10], x_values[29], acc[26]);
            acc[27] = __fmaf_rn(w_t[11], x_values[29], acc[27]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_67;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_67) : "f"(x_values[30]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[26]), "+f"(acc[27])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_67));
#else
            acc[26] = __fmaf_rn(w_t[12], x_values[30], acc[26]);
            acc[27] = __fmaf_rn(w_t[13], x_values[30], acc[27]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_68;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_68) : "f"(x_values[31]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[26]), "+f"(acc[27])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_68));
#else
            acc[26] = __fmaf_rn(w_t[14], x_values[31], acc[26]);
            acc[27] = __fmaf_rn(w_t[15], x_values[31], acc[27]);
#endif
          }
          {
            uint32_t _uv4_69_0;
            uint32_t _uv4_69_1;
            uint32_t _uv4_69_2;
            uint32_t _uv4_69_3;
            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(_uv4_69_0), "=r"(_uv4_69_1), "=r"(_uv4_69_2), "=r"(_uv4_69_3)
                         : "l"((const void*)(reinterpret_cast<const unsigned int*>(lora_a_raw) +
                                             (weight_words0 + (long long)(4 * hidden_words) +
                                              (long long)k_words)))
                         : "memory");
            w_car[0 + 0] = _uv4_69_0;
            w_car[0 + 1] = _uv4_69_1;
            w_car[0 + 2] = _uv4_69_2;
            w_car[0 + 3] = _uv4_69_3;
          }
          w_t[0] = __uint_as_float(w_car[0] << 16);
          w_t[2] = __uint_as_float(w_car[0] & 4294901760u);
          w_t[4] = __uint_as_float(w_car[1] << 16);
          w_t[6] = __uint_as_float(w_car[1] & 4294901760u);
          w_t[8] = __uint_as_float(w_car[2] << 16);
          w_t[10] = __uint_as_float(w_car[2] & 4294901760u);
          w_t[12] = __uint_as_float(w_car[3] << 16);
          w_t[14] = __uint_as_float(w_car[3] & 4294901760u);
          {
            uint32_t _uv4_70_0;
            uint32_t _uv4_70_1;
            uint32_t _uv4_70_2;
            uint32_t _uv4_70_3;
            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(_uv4_70_0), "=r"(_uv4_70_1), "=r"(_uv4_70_2), "=r"(_uv4_70_3)
                         : "l"((const void*)(reinterpret_cast<const unsigned int*>(lora_a_raw) +
                                             (weight_words0 + (long long)(5 * hidden_words) +
                                              (long long)k_words)))
                         : "memory");
            w_car[0 + 0] = _uv4_70_0;
            w_car[0 + 1] = _uv4_70_1;
            w_car[0 + 2] = _uv4_70_2;
            w_car[0 + 3] = _uv4_70_3;
          }
          w_t[1] = __uint_as_float(w_car[0] << 16);
          w_t[3] = __uint_as_float(w_car[0] & 4294901760u);
          w_t[5] = __uint_as_float(w_car[1] << 16);
          w_t[7] = __uint_as_float(w_car[1] & 4294901760u);
          w_t[9] = __uint_as_float(w_car[2] << 16);
          w_t[11] = __uint_as_float(w_car[2] & 4294901760u);
          w_t[13] = __uint_as_float(w_car[3] << 16);
          w_t[15] = __uint_as_float(w_car[3] & 4294901760u);
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_71;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_71) : "f"(x_values[0]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[4]), "+f"(acc[5])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_71));
#else
            acc[4] = __fmaf_rn(w_t[0], x_values[0], acc[4]);
            acc[5] = __fmaf_rn(w_t[1], x_values[0], acc[5]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_72;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_72) : "f"(x_values[1]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[4]), "+f"(acc[5])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_72));
#else
            acc[4] = __fmaf_rn(w_t[2], x_values[1], acc[4]);
            acc[5] = __fmaf_rn(w_t[3], x_values[1], acc[5]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_73;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_73) : "f"(x_values[2]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[4]), "+f"(acc[5])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_73));
#else
            acc[4] = __fmaf_rn(w_t[4], x_values[2], acc[4]);
            acc[5] = __fmaf_rn(w_t[5], x_values[2], acc[5]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_74;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_74) : "f"(x_values[3]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[4]), "+f"(acc[5])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_74));
#else
            acc[4] = __fmaf_rn(w_t[6], x_values[3], acc[4]);
            acc[5] = __fmaf_rn(w_t[7], x_values[3], acc[5]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_75;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_75) : "f"(x_values[4]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[4]), "+f"(acc[5])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_75));
#else
            acc[4] = __fmaf_rn(w_t[8], x_values[4], acc[4]);
            acc[5] = __fmaf_rn(w_t[9], x_values[4], acc[5]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_76;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_76) : "f"(x_values[5]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[4]), "+f"(acc[5])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_76));
#else
            acc[4] = __fmaf_rn(w_t[10], x_values[5], acc[4]);
            acc[5] = __fmaf_rn(w_t[11], x_values[5], acc[5]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_77;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_77) : "f"(x_values[6]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[4]), "+f"(acc[5])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_77));
#else
            acc[4] = __fmaf_rn(w_t[12], x_values[6], acc[4]);
            acc[5] = __fmaf_rn(w_t[13], x_values[6], acc[5]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_78;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_78) : "f"(x_values[7]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[4]), "+f"(acc[5])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_78));
#else
            acc[4] = __fmaf_rn(w_t[14], x_values[7], acc[4]);
            acc[5] = __fmaf_rn(w_t[15], x_values[7], acc[5]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_79;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_79) : "f"(x_values[8]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[12]), "+f"(acc[13])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_79));
#else
            acc[12] = __fmaf_rn(w_t[0], x_values[8], acc[12]);
            acc[13] = __fmaf_rn(w_t[1], x_values[8], acc[13]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_80;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_80) : "f"(x_values[9]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[12]), "+f"(acc[13])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_80));
#else
            acc[12] = __fmaf_rn(w_t[2], x_values[9], acc[12]);
            acc[13] = __fmaf_rn(w_t[3], x_values[9], acc[13]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_81;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_81) : "f"(x_values[10]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[12]), "+f"(acc[13])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_81));
#else
            acc[12] = __fmaf_rn(w_t[4], x_values[10], acc[12]);
            acc[13] = __fmaf_rn(w_t[5], x_values[10], acc[13]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_82;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_82) : "f"(x_values[11]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[12]), "+f"(acc[13])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_82));
#else
            acc[12] = __fmaf_rn(w_t[6], x_values[11], acc[12]);
            acc[13] = __fmaf_rn(w_t[7], x_values[11], acc[13]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_83;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_83) : "f"(x_values[12]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[12]), "+f"(acc[13])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_83));
#else
            acc[12] = __fmaf_rn(w_t[8], x_values[12], acc[12]);
            acc[13] = __fmaf_rn(w_t[9], x_values[12], acc[13]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_84;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_84) : "f"(x_values[13]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[12]), "+f"(acc[13])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_84));
#else
            acc[12] = __fmaf_rn(w_t[10], x_values[13], acc[12]);
            acc[13] = __fmaf_rn(w_t[11], x_values[13], acc[13]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_85;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_85) : "f"(x_values[14]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[12]), "+f"(acc[13])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_85));
#else
            acc[12] = __fmaf_rn(w_t[12], x_values[14], acc[12]);
            acc[13] = __fmaf_rn(w_t[13], x_values[14], acc[13]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_86;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_86) : "f"(x_values[15]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[12]), "+f"(acc[13])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_86));
#else
            acc[12] = __fmaf_rn(w_t[14], x_values[15], acc[12]);
            acc[13] = __fmaf_rn(w_t[15], x_values[15], acc[13]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_87;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_87) : "f"(x_values[16]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[20]), "+f"(acc[21])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_87));
#else
            acc[20] = __fmaf_rn(w_t[0], x_values[16], acc[20]);
            acc[21] = __fmaf_rn(w_t[1], x_values[16], acc[21]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_88;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_88) : "f"(x_values[17]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[20]), "+f"(acc[21])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_88));
#else
            acc[20] = __fmaf_rn(w_t[2], x_values[17], acc[20]);
            acc[21] = __fmaf_rn(w_t[3], x_values[17], acc[21]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_89;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_89) : "f"(x_values[18]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[20]), "+f"(acc[21])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_89));
#else
            acc[20] = __fmaf_rn(w_t[4], x_values[18], acc[20]);
            acc[21] = __fmaf_rn(w_t[5], x_values[18], acc[21]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_90;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_90) : "f"(x_values[19]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[20]), "+f"(acc[21])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_90));
#else
            acc[20] = __fmaf_rn(w_t[6], x_values[19], acc[20]);
            acc[21] = __fmaf_rn(w_t[7], x_values[19], acc[21]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_91;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_91) : "f"(x_values[20]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[20]), "+f"(acc[21])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_91));
#else
            acc[20] = __fmaf_rn(w_t[8], x_values[20], acc[20]);
            acc[21] = __fmaf_rn(w_t[9], x_values[20], acc[21]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_92;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_92) : "f"(x_values[21]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[20]), "+f"(acc[21])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_92));
#else
            acc[20] = __fmaf_rn(w_t[10], x_values[21], acc[20]);
            acc[21] = __fmaf_rn(w_t[11], x_values[21], acc[21]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_93;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_93) : "f"(x_values[22]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[20]), "+f"(acc[21])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_93));
#else
            acc[20] = __fmaf_rn(w_t[12], x_values[22], acc[20]);
            acc[21] = __fmaf_rn(w_t[13], x_values[22], acc[21]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_94;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_94) : "f"(x_values[23]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[20]), "+f"(acc[21])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_94));
#else
            acc[20] = __fmaf_rn(w_t[14], x_values[23], acc[20]);
            acc[21] = __fmaf_rn(w_t[15], x_values[23], acc[21]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_95;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_95) : "f"(x_values[24]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[28]), "+f"(acc[29])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_95));
#else
            acc[28] = __fmaf_rn(w_t[0], x_values[24], acc[28]);
            acc[29] = __fmaf_rn(w_t[1], x_values[24], acc[29]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_96;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_96) : "f"(x_values[25]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[28]), "+f"(acc[29])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_96));
#else
            acc[28] = __fmaf_rn(w_t[2], x_values[25], acc[28]);
            acc[29] = __fmaf_rn(w_t[3], x_values[25], acc[29]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_97;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_97) : "f"(x_values[26]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[28]), "+f"(acc[29])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_97));
#else
            acc[28] = __fmaf_rn(w_t[4], x_values[26], acc[28]);
            acc[29] = __fmaf_rn(w_t[5], x_values[26], acc[29]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_98;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_98) : "f"(x_values[27]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[28]), "+f"(acc[29])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_98));
#else
            acc[28] = __fmaf_rn(w_t[6], x_values[27], acc[28]);
            acc[29] = __fmaf_rn(w_t[7], x_values[27], acc[29]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_99;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_99) : "f"(x_values[28]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[28]), "+f"(acc[29])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_99));
#else
            acc[28] = __fmaf_rn(w_t[8], x_values[28], acc[28]);
            acc[29] = __fmaf_rn(w_t[9], x_values[28], acc[29]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_100;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_100) : "f"(x_values[29]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[28]), "+f"(acc[29])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_100));
#else
            acc[28] = __fmaf_rn(w_t[10], x_values[29], acc[28]);
            acc[29] = __fmaf_rn(w_t[11], x_values[29], acc[29]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_101;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_101) : "f"(x_values[30]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[28]), "+f"(acc[29])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_101));
#else
            acc[28] = __fmaf_rn(w_t[12], x_values[30], acc[28]);
            acc[29] = __fmaf_rn(w_t[13], x_values[30], acc[29]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_102;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_102) : "f"(x_values[31]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[28]), "+f"(acc[29])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_102));
#else
            acc[28] = __fmaf_rn(w_t[14], x_values[31], acc[28]);
            acc[29] = __fmaf_rn(w_t[15], x_values[31], acc[29]);
#endif
          }
          {
            uint32_t _uv4_103_0;
            uint32_t _uv4_103_1;
            uint32_t _uv4_103_2;
            uint32_t _uv4_103_3;
            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(_uv4_103_0), "=r"(_uv4_103_1), "=r"(_uv4_103_2), "=r"(_uv4_103_3)
                         : "l"((const void*)(reinterpret_cast<const unsigned int*>(lora_a_raw) +
                                             (weight_words0 + (long long)(6 * hidden_words) +
                                              (long long)k_words)))
                         : "memory");
            w_car[0 + 0] = _uv4_103_0;
            w_car[0 + 1] = _uv4_103_1;
            w_car[0 + 2] = _uv4_103_2;
            w_car[0 + 3] = _uv4_103_3;
          }
          w_t[0] = __uint_as_float(w_car[0] << 16);
          w_t[2] = __uint_as_float(w_car[0] & 4294901760u);
          w_t[4] = __uint_as_float(w_car[1] << 16);
          w_t[6] = __uint_as_float(w_car[1] & 4294901760u);
          w_t[8] = __uint_as_float(w_car[2] << 16);
          w_t[10] = __uint_as_float(w_car[2] & 4294901760u);
          w_t[12] = __uint_as_float(w_car[3] << 16);
          w_t[14] = __uint_as_float(w_car[3] & 4294901760u);
          {
            uint32_t _uv4_104_0;
            uint32_t _uv4_104_1;
            uint32_t _uv4_104_2;
            uint32_t _uv4_104_3;
            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(_uv4_104_0), "=r"(_uv4_104_1), "=r"(_uv4_104_2), "=r"(_uv4_104_3)
                         : "l"((const void*)(reinterpret_cast<const unsigned int*>(lora_a_raw) +
                                             (weight_words0 + (long long)(7 * hidden_words) +
                                              (long long)k_words)))
                         : "memory");
            w_car[0 + 0] = _uv4_104_0;
            w_car[0 + 1] = _uv4_104_1;
            w_car[0 + 2] = _uv4_104_2;
            w_car[0 + 3] = _uv4_104_3;
          }
          w_t[1] = __uint_as_float(w_car[0] << 16);
          w_t[3] = __uint_as_float(w_car[0] & 4294901760u);
          w_t[5] = __uint_as_float(w_car[1] << 16);
          w_t[7] = __uint_as_float(w_car[1] & 4294901760u);
          w_t[9] = __uint_as_float(w_car[2] << 16);
          w_t[11] = __uint_as_float(w_car[2] & 4294901760u);
          w_t[13] = __uint_as_float(w_car[3] << 16);
          w_t[15] = __uint_as_float(w_car[3] & 4294901760u);
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_105;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_105) : "f"(x_values[0]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[6]), "+f"(acc[7])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_105));
#else
            acc[6] = __fmaf_rn(w_t[0], x_values[0], acc[6]);
            acc[7] = __fmaf_rn(w_t[1], x_values[0], acc[7]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_106;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_106) : "f"(x_values[1]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[6]), "+f"(acc[7])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_106));
#else
            acc[6] = __fmaf_rn(w_t[2], x_values[1], acc[6]);
            acc[7] = __fmaf_rn(w_t[3], x_values[1], acc[7]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_107;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_107) : "f"(x_values[2]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[6]), "+f"(acc[7])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_107));
#else
            acc[6] = __fmaf_rn(w_t[4], x_values[2], acc[6]);
            acc[7] = __fmaf_rn(w_t[5], x_values[2], acc[7]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_108;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_108) : "f"(x_values[3]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[6]), "+f"(acc[7])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_108));
#else
            acc[6] = __fmaf_rn(w_t[6], x_values[3], acc[6]);
            acc[7] = __fmaf_rn(w_t[7], x_values[3], acc[7]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_109;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_109) : "f"(x_values[4]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[6]), "+f"(acc[7])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_109));
#else
            acc[6] = __fmaf_rn(w_t[8], x_values[4], acc[6]);
            acc[7] = __fmaf_rn(w_t[9], x_values[4], acc[7]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_110;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_110) : "f"(x_values[5]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[6]), "+f"(acc[7])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_110));
#else
            acc[6] = __fmaf_rn(w_t[10], x_values[5], acc[6]);
            acc[7] = __fmaf_rn(w_t[11], x_values[5], acc[7]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_111;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_111) : "f"(x_values[6]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[6]), "+f"(acc[7])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_111));
#else
            acc[6] = __fmaf_rn(w_t[12], x_values[6], acc[6]);
            acc[7] = __fmaf_rn(w_t[13], x_values[6], acc[7]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_112;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_112) : "f"(x_values[7]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[6]), "+f"(acc[7])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_112));
#else
            acc[6] = __fmaf_rn(w_t[14], x_values[7], acc[6]);
            acc[7] = __fmaf_rn(w_t[15], x_values[7], acc[7]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_113;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_113) : "f"(x_values[8]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[14]), "+f"(acc[15])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_113));
#else
            acc[14] = __fmaf_rn(w_t[0], x_values[8], acc[14]);
            acc[15] = __fmaf_rn(w_t[1], x_values[8], acc[15]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_114;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_114) : "f"(x_values[9]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[14]), "+f"(acc[15])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_114));
#else
            acc[14] = __fmaf_rn(w_t[2], x_values[9], acc[14]);
            acc[15] = __fmaf_rn(w_t[3], x_values[9], acc[15]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_115;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_115) : "f"(x_values[10]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[14]), "+f"(acc[15])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_115));
#else
            acc[14] = __fmaf_rn(w_t[4], x_values[10], acc[14]);
            acc[15] = __fmaf_rn(w_t[5], x_values[10], acc[15]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_116;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_116) : "f"(x_values[11]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[14]), "+f"(acc[15])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_116));
#else
            acc[14] = __fmaf_rn(w_t[6], x_values[11], acc[14]);
            acc[15] = __fmaf_rn(w_t[7], x_values[11], acc[15]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_117;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_117) : "f"(x_values[12]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[14]), "+f"(acc[15])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_117));
#else
            acc[14] = __fmaf_rn(w_t[8], x_values[12], acc[14]);
            acc[15] = __fmaf_rn(w_t[9], x_values[12], acc[15]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_118;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_118) : "f"(x_values[13]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[14]), "+f"(acc[15])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_118));
#else
            acc[14] = __fmaf_rn(w_t[10], x_values[13], acc[14]);
            acc[15] = __fmaf_rn(w_t[11], x_values[13], acc[15]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_119;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_119) : "f"(x_values[14]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[14]), "+f"(acc[15])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_119));
#else
            acc[14] = __fmaf_rn(w_t[12], x_values[14], acc[14]);
            acc[15] = __fmaf_rn(w_t[13], x_values[14], acc[15]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_120;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_120) : "f"(x_values[15]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[14]), "+f"(acc[15])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_120));
#else
            acc[14] = __fmaf_rn(w_t[14], x_values[15], acc[14]);
            acc[15] = __fmaf_rn(w_t[15], x_values[15], acc[15]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_121;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_121) : "f"(x_values[16]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[22]), "+f"(acc[23])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_121));
#else
            acc[22] = __fmaf_rn(w_t[0], x_values[16], acc[22]);
            acc[23] = __fmaf_rn(w_t[1], x_values[16], acc[23]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_122;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_122) : "f"(x_values[17]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[22]), "+f"(acc[23])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_122));
#else
            acc[22] = __fmaf_rn(w_t[2], x_values[17], acc[22]);
            acc[23] = __fmaf_rn(w_t[3], x_values[17], acc[23]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_123;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_123) : "f"(x_values[18]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[22]), "+f"(acc[23])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_123));
#else
            acc[22] = __fmaf_rn(w_t[4], x_values[18], acc[22]);
            acc[23] = __fmaf_rn(w_t[5], x_values[18], acc[23]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_124;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_124) : "f"(x_values[19]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[22]), "+f"(acc[23])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_124));
#else
            acc[22] = __fmaf_rn(w_t[6], x_values[19], acc[22]);
            acc[23] = __fmaf_rn(w_t[7], x_values[19], acc[23]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_125;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_125) : "f"(x_values[20]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[22]), "+f"(acc[23])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_125));
#else
            acc[22] = __fmaf_rn(w_t[8], x_values[20], acc[22]);
            acc[23] = __fmaf_rn(w_t[9], x_values[20], acc[23]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_126;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_126) : "f"(x_values[21]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[22]), "+f"(acc[23])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_126));
#else
            acc[22] = __fmaf_rn(w_t[10], x_values[21], acc[22]);
            acc[23] = __fmaf_rn(w_t[11], x_values[21], acc[23]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_127;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_127) : "f"(x_values[22]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[22]), "+f"(acc[23])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_127));
#else
            acc[22] = __fmaf_rn(w_t[12], x_values[22], acc[22]);
            acc[23] = __fmaf_rn(w_t[13], x_values[22], acc[23]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_128;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_128) : "f"(x_values[23]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[22]), "+f"(acc[23])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_128));
#else
            acc[22] = __fmaf_rn(w_t[14], x_values[23], acc[22]);
            acc[23] = __fmaf_rn(w_t[15], x_values[23], acc[23]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_129;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_129) : "f"(x_values[24]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[30]), "+f"(acc[31])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_129));
#else
            acc[30] = __fmaf_rn(w_t[0], x_values[24], acc[30]);
            acc[31] = __fmaf_rn(w_t[1], x_values[24], acc[31]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_130;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_130) : "f"(x_values[25]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[30]), "+f"(acc[31])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_130));
#else
            acc[30] = __fmaf_rn(w_t[2], x_values[25], acc[30]);
            acc[31] = __fmaf_rn(w_t[3], x_values[25], acc[31]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_131;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_131) : "f"(x_values[26]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[30]), "+f"(acc[31])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_131));
#else
            acc[30] = __fmaf_rn(w_t[4], x_values[26], acc[30]);
            acc[31] = __fmaf_rn(w_t[5], x_values[26], acc[31]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_132;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_132) : "f"(x_values[27]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[30]), "+f"(acc[31])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_132));
#else
            acc[30] = __fmaf_rn(w_t[6], x_values[27], acc[30]);
            acc[31] = __fmaf_rn(w_t[7], x_values[27], acc[31]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_133;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_133) : "f"(x_values[28]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[30]), "+f"(acc[31])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_133));
#else
            acc[30] = __fmaf_rn(w_t[8], x_values[28], acc[30]);
            acc[31] = __fmaf_rn(w_t[9], x_values[28], acc[31]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_134;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_134) : "f"(x_values[29]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[30]), "+f"(acc[31])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_134));
#else
            acc[30] = __fmaf_rn(w_t[10], x_values[29], acc[30]);
            acc[31] = __fmaf_rn(w_t[11], x_values[29], acc[31]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_135;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_135) : "f"(x_values[30]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[30]), "+f"(acc[31])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_135));
#else
            acc[30] = __fmaf_rn(w_t[12], x_values[30], acc[30]);
            acc[31] = __fmaf_rn(w_t[13], x_values[30], acc[31]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_136;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_136) : "f"(x_values[31]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[30]), "+f"(acc[31])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_136));
#else
            acc[30] = __fmaf_rn(w_t[14], x_values[31], acc[30]);
            acc[31] = __fmaf_rn(w_t[15], x_values[31], acc[31]);
#endif
          }
        }
      }
#pragma unroll
      for (int i = 0; i < 16; i++) {
        float _shfl_xor_0 =
            __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) != 0) ? acc[i] : acc[i + 16]), 16);
        red_a[i] = (((lane_0 & 16) != 0) ? acc[i + 16] : acc[i]) + _shfl_xor_0;
      }
#pragma unroll
      for (int i_1 = 0; i_1 < 8; i_1++) {
        float _shfl_xor_1 =
            __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) != 0) ? red_a[i_1] : red_a[i_1 + 8]), 8);
        red_b[i_1] = (((lane_0 & 8) != 0) ? red_a[i_1 + 8] : red_a[i_1]) + _shfl_xor_1;
      }
#pragma unroll
      for (int i_2 = 0; i_2 < 4; i_2++) {
        float _shfl_xor_2 =
            __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 4) != 0) ? red_b[i_2] : red_b[i_2 + 4]), 4);
        red_c[i_2] = (((lane_0 & 4) != 0) ? red_b[i_2 + 4] : red_b[i_2]) + _shfl_xor_2;
      }
#pragma unroll
      for (int i_3 = 0; i_3 < 2; i_3++) {
        float _shfl_xor_3 =
            __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 2) != 0) ? red_c[i_3] : red_c[i_3 + 2]), 2);
        red_d[i_3] = (((lane_0 & 2) != 0) ? red_c[i_3 + 2] : red_c[i_3]) + _shfl_xor_3;
      }
#pragma unroll
      for (int i_4 = 0; i_4 < 1; i_4++) {
        float _shfl_xor_4 =
            __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 1) != 0) ? red_d[i_4] : red_d[i_4 + 1]), 1);
        red_e[i_4] = (((lane_0 & 1) != 0) ? red_d[i_4 + 1] : red_d[i_4]) + _shfl_xor_4;
      }
#pragma unroll
      for (int i_5 = 0; i_5 < 1; i_5++) {
        warp_partials[(lane_0 + i_5) * 4 + warp] = red_e[i_5];
      }
      __syncthreads();
      if (tid < 32) {
        float owned_accum = 0.0f;
#pragma unroll
        for (int source_warp = 0; source_warp < 4; source_warp++) {
          owned_accum += warp_partials[tid * 4 + source_warp];
        }
        int owner_j = tid / 8;
        int owner_rr = tid % 8;
        if (owner_j < count) {
          int owner_route = (int)reinterpret_cast<const unsigned int*>(
              workspace_raw)[off_sorted_routes + start + owner_j];
          *(reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<__nv_bfloat16*>(shrink_out_raw) +
                                             (owner_route * 32 + rank_base0 + owner_rr)) +
            (0)) = __float2bfloat16_rn(owned_accum);
        }
      }
    }
  }
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

}  // extern "C"

#undef BLACKWELL_INF
#undef NUM_MAIN_STAGES
#undef SMEM_TOTAL
#undef SMEM_WARP_PARTIALS_OFF
#undef SMEM_WARP_PARTIALS_STAGE_BYTES
#undef SMEM_WARP_PARTIALS_STRIDE
#undef SMEM_W_RING_OFF
#undef SMEM_W_RING_STAGE_BYTES
#undef SMEM_W_RING_STRIDE
#undef SMEM_X_RING_OFF
#undef SMEM_X_RING_STAGE_BYTES
#undef SMEM_X_RING_STRIDE
#undef THREADS

#define BLACKWELL_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_WARP_PARTIALS_OFF 0
#define SMEM_WARP_PARTIALS_STAGE_BYTES 512
#define SMEM_WARP_PARTIALS_STRIDE 512
#define SMEM_X_RING_OFF 512
#define SMEM_X_RING_STAGE_BYTES 16384
#define SMEM_X_RING_STRIDE 16384
#define SMEM_W_RING_OFF 16896
#define SMEM_W_RING_STAGE_BYTES 32768
#define SMEM_W_RING_STRIDE 32768
#define SMEM_TOTAL 49664
#define THREADS 128

extern "C" {

__global__
__launch_bounds__(128, 4) void kernel_flashinfer_bgmv_moe_shrink_grouped_ring_single_bf16_r32(
    uint16_t* __restrict__ shrink_out_raw, uint16_t* __restrict__ x_raw,
    uint16_t* __restrict__ lora_a_raw, long long* __restrict__ sorted_token_ids, int num_pairs,
    int num_experts, int hidden, int num_tiles, int rt_per_cta, int rt_groups,
    unsigned int* __restrict__ workspace_raw, int off_group_offset, int off_tile_table,
    int off_sorted_routes) {
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
  float* warp_partials = reinterpret_cast<float*>(smem_raw + 0);
  const int warp_partials_addr = smem + 0;
  __nv_bfloat16* x_ring = reinterpret_cast<__nv_bfloat16*>(smem_raw + 512);
  const int x_ring_addr = smem + 512;
  __nv_bfloat16* w_ring = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16896);
  const int w_ring_addr = smem + 16896;

  // === Task calls (dependency order) ===
  asm volatile("griddepcontrol.wait;" ::: "memory");
  int rt_group = blockIdx.x % ((1) ? 4 : rt_groups);
  int half = blockIdx.x / ((1) ? 4 : rt_groups) % 4;
  int tile = blockIdx.x / (((1) ? 4 : rt_groups) * 4);
  int n_tiles = (int)reinterpret_cast<const unsigned int*>(workspace_raw)[0];
  if (tile < n_tiles) {
    int entry = (int)reinterpret_cast<const unsigned int*>(workspace_raw)[off_tile_table + tile];
    int group = entry / 65536;
    int chunk = entry % 65536;
    int start =
        (int)reinterpret_cast<const unsigned int*>(workspace_raw)[off_group_offset + group] +
        chunk * 16 + half * 4;
    int group_end =
        (int)reinterpret_cast<const unsigned int*>(workspace_raw)[off_group_offset + group + 1];
    int count = group_end - start;
    if (count > 4) {
      count = 4;
    }
    if (count > 0) {
      int lora = group / num_experts;
      int expert = group % num_experts;
      int hidden_words = hidden / 2;
      long long weight_row_base = (long long)(lora * num_experts + expert) * 32;
      int routes[4];
      long long tokens[4];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        routes[j] = -1;
        tokens[j] = 0;
        if (count > j) {
          routes[j] = (int)reinterpret_cast<const unsigned int*>(
              workspace_raw)[off_sorted_routes + start + j];
          tokens[j] = sorted_token_ids[routes[j]];
        }
      }
      int rank_base0 = rt_group * 8;
      long long weight_words0 = (weight_row_base + (long long)rank_base0) * (long long)hidden_words;
      float acc[32];
      unsigned int x_car[16];
      unsigned int xr_car[4];
      unsigned int w_car[4];
      float x_values[32];
      float w_values[8];
      float w_t[16];
      int lane_0 = lane;
      float red_a[16];
      float red_b[8];
      float red_c[4];
      float red_d[2];
      float red_e[1];
#pragma unroll
      for (int owner = 0; owner < 32; owner++) {
        acc[owner] = 0.0f;
      }
      int tid_vec = tid * 8;
#pragma unroll
      for (int d = 0; d < 1; d++) {
        int k_p = d * 1024 + tid_vec;
        if (k_p < hidden) {
          int kw_p = k_p / 2;
          {
#pragma unroll
            for (int j_1 = 0; j_1 < 4; j_1++) {
              asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                               x_ring_addr + (unsigned int)(((d * 4 + j_1) * 1024 + tid_vec) * 2)),
                           "l"(reinterpret_cast<const unsigned int*>(x_raw) +
                               (tokens[j_1] * (long long)hidden_words + (long long)kw_p)));
            }
          }
#pragma unroll
          for (int r = 0; r < 8; r++) {
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                             w_ring_addr + (unsigned int)(((d * 8 + r) * 1024 + tid_vec) * 2)),
                         "l"(reinterpret_cast<const unsigned int*>(lora_a_raw) +
                             (weight_words0 + (long long)(r * hidden_words) + (long long)kw_p)));
          }
        }
        asm volatile("cp.async.commit_group;");
      }
#pragma unroll 1
      for (int local = 0; local < num_tiles; local++) {
        int stage = local % 2;
        int nxt = local + 1;
        if (nxt < num_tiles) {
          int nstage = nxt % 2;
          int k_n = nxt * 1024 + tid_vec;
          if (k_n < hidden) {
            int kw_n = k_n / 2;
            {
#pragma unroll
              for (int j_2 = 0; j_2 < 4; j_2++) {
                asm volatile(
                    "cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                        x_ring_addr + (unsigned int)(((nstage * 4 + j_2) * 1024 + tid_vec) * 2)),
                    "l"(reinterpret_cast<const unsigned int*>(x_raw) +
                        (tokens[j_2] * (long long)hidden_words + (long long)kw_n)));
              }
            }
#pragma unroll
            for (int r_1 = 0; r_1 < 8; r_1++) {
              asm volatile(
                  "cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                      w_ring_addr + (unsigned int)(((nstage * 8 + r_1) * 1024 + tid_vec) * 2)),
                  "l"(reinterpret_cast<const unsigned int*>(lora_a_raw) +
                      (weight_words0 + (long long)(r_1 * hidden_words) + (long long)kw_n)));
            }
          }
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 1;");
        int k_base_r = local * 1024 + tid_vec;
        if (k_base_r < hidden) {
#pragma unroll
          for (int j_3 = 0; j_3 < 4; j_3++) {
            asm volatile(
                "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xr_car[0])),
                  "=r"(*reinterpret_cast<uint32_t*>(&xr_car[(0) + 1])),
                  "=r"(*reinterpret_cast<uint32_t*>(&xr_car[(0) + 2])),
                  "=r"(*reinterpret_cast<uint32_t*>(&xr_car[(0) + 3]))
                : "r"(x_ring_addr + (unsigned int)(((stage * 4 + j_3) * 1024 + tid_vec) * 2)));
#pragma unroll
            for (int pair = 0; pair < 4; pair++) {
              x_values[j_3 * 8 + 2 * pair] = __uint_as_float(xr_car[pair] << 16);
              x_values[j_3 * 8 + 2 * pair + 1] = __uint_as_float(xr_car[pair] & 4294901760u);
            }
          }
          asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                       : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
                       : "r"(w_ring_addr + (unsigned int)((stage * 8 * 1024 + tid_vec) * 2)));
          w_t[0] = __uint_as_float(w_car[0] << 16);
          w_t[2] = __uint_as_float(w_car[0] & 4294901760u);
          w_t[4] = __uint_as_float(w_car[1] << 16);
          w_t[6] = __uint_as_float(w_car[1] & 4294901760u);
          w_t[8] = __uint_as_float(w_car[2] << 16);
          w_t[10] = __uint_as_float(w_car[2] & 4294901760u);
          w_t[12] = __uint_as_float(w_car[3] << 16);
          w_t[14] = __uint_as_float(w_car[3] & 4294901760u);
          asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                       : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
                       : "r"(w_ring_addr + (unsigned int)(((stage * 8 + 1) * 1024 + tid_vec) * 2)));
          w_t[1] = __uint_as_float(w_car[0] << 16);
          w_t[3] = __uint_as_float(w_car[0] & 4294901760u);
          w_t[5] = __uint_as_float(w_car[1] << 16);
          w_t[7] = __uint_as_float(w_car[1] & 4294901760u);
          w_t[9] = __uint_as_float(w_car[2] << 16);
          w_t[11] = __uint_as_float(w_car[2] & 4294901760u);
          w_t[13] = __uint_as_float(w_car[3] << 16);
          w_t[15] = __uint_as_float(w_car[3] & 4294901760u);
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_0;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_0) : "f"(x_values[0]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[0]), "+f"(acc[1])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_0));
#else
            acc[0] = __fmaf_rn(w_t[0], x_values[0], acc[0]);
            acc[1] = __fmaf_rn(w_t[1], x_values[0], acc[1]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_1;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_1) : "f"(x_values[1]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[0]), "+f"(acc[1])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_1));
#else
            acc[0] = __fmaf_rn(w_t[2], x_values[1], acc[0]);
            acc[1] = __fmaf_rn(w_t[3], x_values[1], acc[1]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_2;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_2) : "f"(x_values[2]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[0]), "+f"(acc[1])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_2));
#else
            acc[0] = __fmaf_rn(w_t[4], x_values[2], acc[0]);
            acc[1] = __fmaf_rn(w_t[5], x_values[2], acc[1]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_3;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_3) : "f"(x_values[3]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[0]), "+f"(acc[1])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_3));
#else
            acc[0] = __fmaf_rn(w_t[6], x_values[3], acc[0]);
            acc[1] = __fmaf_rn(w_t[7], x_values[3], acc[1]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_4;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_4) : "f"(x_values[4]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[0]), "+f"(acc[1])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_4));
#else
            acc[0] = __fmaf_rn(w_t[8], x_values[4], acc[0]);
            acc[1] = __fmaf_rn(w_t[9], x_values[4], acc[1]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_5;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_5) : "f"(x_values[5]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[0]), "+f"(acc[1])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_5));
#else
            acc[0] = __fmaf_rn(w_t[10], x_values[5], acc[0]);
            acc[1] = __fmaf_rn(w_t[11], x_values[5], acc[1]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_6;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_6) : "f"(x_values[6]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[0]), "+f"(acc[1])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_6));
#else
            acc[0] = __fmaf_rn(w_t[12], x_values[6], acc[0]);
            acc[1] = __fmaf_rn(w_t[13], x_values[6], acc[1]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_7;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_7) : "f"(x_values[7]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[0]), "+f"(acc[1])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_7));
#else
            acc[0] = __fmaf_rn(w_t[14], x_values[7], acc[0]);
            acc[1] = __fmaf_rn(w_t[15], x_values[7], acc[1]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_8;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_8) : "f"(x_values[8]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[8]), "+f"(acc[9])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_8));
#else
            acc[8] = __fmaf_rn(w_t[0], x_values[8], acc[8]);
            acc[9] = __fmaf_rn(w_t[1], x_values[8], acc[9]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_9;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_9) : "f"(x_values[9]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[8]), "+f"(acc[9])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_9));
#else
            acc[8] = __fmaf_rn(w_t[2], x_values[9], acc[8]);
            acc[9] = __fmaf_rn(w_t[3], x_values[9], acc[9]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_10;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_10) : "f"(x_values[10]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[8]), "+f"(acc[9])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_10));
#else
            acc[8] = __fmaf_rn(w_t[4], x_values[10], acc[8]);
            acc[9] = __fmaf_rn(w_t[5], x_values[10], acc[9]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_11;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_11) : "f"(x_values[11]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[8]), "+f"(acc[9])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_11));
#else
            acc[8] = __fmaf_rn(w_t[6], x_values[11], acc[8]);
            acc[9] = __fmaf_rn(w_t[7], x_values[11], acc[9]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_12;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_12) : "f"(x_values[12]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[8]), "+f"(acc[9])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_12));
#else
            acc[8] = __fmaf_rn(w_t[8], x_values[12], acc[8]);
            acc[9] = __fmaf_rn(w_t[9], x_values[12], acc[9]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_13;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_13) : "f"(x_values[13]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[8]), "+f"(acc[9])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_13));
#else
            acc[8] = __fmaf_rn(w_t[10], x_values[13], acc[8]);
            acc[9] = __fmaf_rn(w_t[11], x_values[13], acc[9]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_14;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_14) : "f"(x_values[14]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[8]), "+f"(acc[9])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_14));
#else
            acc[8] = __fmaf_rn(w_t[12], x_values[14], acc[8]);
            acc[9] = __fmaf_rn(w_t[13], x_values[14], acc[9]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_15;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_15) : "f"(x_values[15]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[8]), "+f"(acc[9])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_15));
#else
            acc[8] = __fmaf_rn(w_t[14], x_values[15], acc[8]);
            acc[9] = __fmaf_rn(w_t[15], x_values[15], acc[9]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_16;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_16) : "f"(x_values[16]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[16]), "+f"(acc[17])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_16));
#else
            acc[16] = __fmaf_rn(w_t[0], x_values[16], acc[16]);
            acc[17] = __fmaf_rn(w_t[1], x_values[16], acc[17]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_17;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_17) : "f"(x_values[17]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[16]), "+f"(acc[17])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_17));
#else
            acc[16] = __fmaf_rn(w_t[2], x_values[17], acc[16]);
            acc[17] = __fmaf_rn(w_t[3], x_values[17], acc[17]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_18;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_18) : "f"(x_values[18]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[16]), "+f"(acc[17])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_18));
#else
            acc[16] = __fmaf_rn(w_t[4], x_values[18], acc[16]);
            acc[17] = __fmaf_rn(w_t[5], x_values[18], acc[17]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_19;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_19) : "f"(x_values[19]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[16]), "+f"(acc[17])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_19));
#else
            acc[16] = __fmaf_rn(w_t[6], x_values[19], acc[16]);
            acc[17] = __fmaf_rn(w_t[7], x_values[19], acc[17]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_20;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_20) : "f"(x_values[20]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[16]), "+f"(acc[17])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_20));
#else
            acc[16] = __fmaf_rn(w_t[8], x_values[20], acc[16]);
            acc[17] = __fmaf_rn(w_t[9], x_values[20], acc[17]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_21;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_21) : "f"(x_values[21]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[16]), "+f"(acc[17])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_21));
#else
            acc[16] = __fmaf_rn(w_t[10], x_values[21], acc[16]);
            acc[17] = __fmaf_rn(w_t[11], x_values[21], acc[17]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_22;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_22) : "f"(x_values[22]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[16]), "+f"(acc[17])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_22));
#else
            acc[16] = __fmaf_rn(w_t[12], x_values[22], acc[16]);
            acc[17] = __fmaf_rn(w_t[13], x_values[22], acc[17]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_23;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_23) : "f"(x_values[23]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[16]), "+f"(acc[17])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_23));
#else
            acc[16] = __fmaf_rn(w_t[14], x_values[23], acc[16]);
            acc[17] = __fmaf_rn(w_t[15], x_values[23], acc[17]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_24;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_24) : "f"(x_values[24]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[24]), "+f"(acc[25])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_24));
#else
            acc[24] = __fmaf_rn(w_t[0], x_values[24], acc[24]);
            acc[25] = __fmaf_rn(w_t[1], x_values[24], acc[25]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_25;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_25) : "f"(x_values[25]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[24]), "+f"(acc[25])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_25));
#else
            acc[24] = __fmaf_rn(w_t[2], x_values[25], acc[24]);
            acc[25] = __fmaf_rn(w_t[3], x_values[25], acc[25]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_26;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_26) : "f"(x_values[26]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[24]), "+f"(acc[25])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_26));
#else
            acc[24] = __fmaf_rn(w_t[4], x_values[26], acc[24]);
            acc[25] = __fmaf_rn(w_t[5], x_values[26], acc[25]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_27;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_27) : "f"(x_values[27]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[24]), "+f"(acc[25])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_27));
#else
            acc[24] = __fmaf_rn(w_t[6], x_values[27], acc[24]);
            acc[25] = __fmaf_rn(w_t[7], x_values[27], acc[25]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_28;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_28) : "f"(x_values[28]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[24]), "+f"(acc[25])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_28));
#else
            acc[24] = __fmaf_rn(w_t[8], x_values[28], acc[24]);
            acc[25] = __fmaf_rn(w_t[9], x_values[28], acc[25]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_29;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_29) : "f"(x_values[29]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[24]), "+f"(acc[25])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_29));
#else
            acc[24] = __fmaf_rn(w_t[10], x_values[29], acc[24]);
            acc[25] = __fmaf_rn(w_t[11], x_values[29], acc[25]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_30;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_30) : "f"(x_values[30]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[24]), "+f"(acc[25])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_30));
#else
            acc[24] = __fmaf_rn(w_t[12], x_values[30], acc[24]);
            acc[25] = __fmaf_rn(w_t[13], x_values[30], acc[25]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_31;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_31) : "f"(x_values[31]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[24]), "+f"(acc[25])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_31));
#else
            acc[24] = __fmaf_rn(w_t[14], x_values[31], acc[24]);
            acc[25] = __fmaf_rn(w_t[15], x_values[31], acc[25]);
#endif
          }
          asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                       : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
                       : "r"(w_ring_addr + (unsigned int)(((stage * 8 + 2) * 1024 + tid_vec) * 2)));
          w_t[0] = __uint_as_float(w_car[0] << 16);
          w_t[2] = __uint_as_float(w_car[0] & 4294901760u);
          w_t[4] = __uint_as_float(w_car[1] << 16);
          w_t[6] = __uint_as_float(w_car[1] & 4294901760u);
          w_t[8] = __uint_as_float(w_car[2] << 16);
          w_t[10] = __uint_as_float(w_car[2] & 4294901760u);
          w_t[12] = __uint_as_float(w_car[3] << 16);
          w_t[14] = __uint_as_float(w_car[3] & 4294901760u);
          asm volatile(
              "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
              : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
              : "r"(w_ring_addr + (unsigned int)(((stage * 8 + 2 + 1) * 1024 + tid_vec) * 2)));
          w_t[1] = __uint_as_float(w_car[0] << 16);
          w_t[3] = __uint_as_float(w_car[0] & 4294901760u);
          w_t[5] = __uint_as_float(w_car[1] << 16);
          w_t[7] = __uint_as_float(w_car[1] & 4294901760u);
          w_t[9] = __uint_as_float(w_car[2] << 16);
          w_t[11] = __uint_as_float(w_car[2] & 4294901760u);
          w_t[13] = __uint_as_float(w_car[3] << 16);
          w_t[15] = __uint_as_float(w_car[3] & 4294901760u);
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_32;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_32) : "f"(x_values[0]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[2]), "+f"(acc[3])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_32));
#else
            acc[2] = __fmaf_rn(w_t[0], x_values[0], acc[2]);
            acc[3] = __fmaf_rn(w_t[1], x_values[0], acc[3]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_33;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_33) : "f"(x_values[1]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[2]), "+f"(acc[3])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_33));
#else
            acc[2] = __fmaf_rn(w_t[2], x_values[1], acc[2]);
            acc[3] = __fmaf_rn(w_t[3], x_values[1], acc[3]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_34;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_34) : "f"(x_values[2]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[2]), "+f"(acc[3])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_34));
#else
            acc[2] = __fmaf_rn(w_t[4], x_values[2], acc[2]);
            acc[3] = __fmaf_rn(w_t[5], x_values[2], acc[3]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_35;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_35) : "f"(x_values[3]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[2]), "+f"(acc[3])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_35));
#else
            acc[2] = __fmaf_rn(w_t[6], x_values[3], acc[2]);
            acc[3] = __fmaf_rn(w_t[7], x_values[3], acc[3]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_36;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_36) : "f"(x_values[4]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[2]), "+f"(acc[3])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_36));
#else
            acc[2] = __fmaf_rn(w_t[8], x_values[4], acc[2]);
            acc[3] = __fmaf_rn(w_t[9], x_values[4], acc[3]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_37;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_37) : "f"(x_values[5]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[2]), "+f"(acc[3])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_37));
#else
            acc[2] = __fmaf_rn(w_t[10], x_values[5], acc[2]);
            acc[3] = __fmaf_rn(w_t[11], x_values[5], acc[3]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_38;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_38) : "f"(x_values[6]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[2]), "+f"(acc[3])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_38));
#else
            acc[2] = __fmaf_rn(w_t[12], x_values[6], acc[2]);
            acc[3] = __fmaf_rn(w_t[13], x_values[6], acc[3]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_39;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_39) : "f"(x_values[7]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[2]), "+f"(acc[3])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_39));
#else
            acc[2] = __fmaf_rn(w_t[14], x_values[7], acc[2]);
            acc[3] = __fmaf_rn(w_t[15], x_values[7], acc[3]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_40;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_40) : "f"(x_values[8]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[10]), "+f"(acc[11])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_40));
#else
            acc[10] = __fmaf_rn(w_t[0], x_values[8], acc[10]);
            acc[11] = __fmaf_rn(w_t[1], x_values[8], acc[11]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_41;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_41) : "f"(x_values[9]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[10]), "+f"(acc[11])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_41));
#else
            acc[10] = __fmaf_rn(w_t[2], x_values[9], acc[10]);
            acc[11] = __fmaf_rn(w_t[3], x_values[9], acc[11]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_42;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_42) : "f"(x_values[10]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[10]), "+f"(acc[11])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_42));
#else
            acc[10] = __fmaf_rn(w_t[4], x_values[10], acc[10]);
            acc[11] = __fmaf_rn(w_t[5], x_values[10], acc[11]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_43;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_43) : "f"(x_values[11]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[10]), "+f"(acc[11])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_43));
#else
            acc[10] = __fmaf_rn(w_t[6], x_values[11], acc[10]);
            acc[11] = __fmaf_rn(w_t[7], x_values[11], acc[11]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_44;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_44) : "f"(x_values[12]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[10]), "+f"(acc[11])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_44));
#else
            acc[10] = __fmaf_rn(w_t[8], x_values[12], acc[10]);
            acc[11] = __fmaf_rn(w_t[9], x_values[12], acc[11]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_45;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_45) : "f"(x_values[13]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[10]), "+f"(acc[11])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_45));
#else
            acc[10] = __fmaf_rn(w_t[10], x_values[13], acc[10]);
            acc[11] = __fmaf_rn(w_t[11], x_values[13], acc[11]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_46;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_46) : "f"(x_values[14]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[10]), "+f"(acc[11])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_46));
#else
            acc[10] = __fmaf_rn(w_t[12], x_values[14], acc[10]);
            acc[11] = __fmaf_rn(w_t[13], x_values[14], acc[11]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_47;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_47) : "f"(x_values[15]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[10]), "+f"(acc[11])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_47));
#else
            acc[10] = __fmaf_rn(w_t[14], x_values[15], acc[10]);
            acc[11] = __fmaf_rn(w_t[15], x_values[15], acc[11]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_48;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_48) : "f"(x_values[16]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[18]), "+f"(acc[19])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_48));
#else
            acc[18] = __fmaf_rn(w_t[0], x_values[16], acc[18]);
            acc[19] = __fmaf_rn(w_t[1], x_values[16], acc[19]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_49;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_49) : "f"(x_values[17]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[18]), "+f"(acc[19])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_49));
#else
            acc[18] = __fmaf_rn(w_t[2], x_values[17], acc[18]);
            acc[19] = __fmaf_rn(w_t[3], x_values[17], acc[19]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_50;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_50) : "f"(x_values[18]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[18]), "+f"(acc[19])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_50));
#else
            acc[18] = __fmaf_rn(w_t[4], x_values[18], acc[18]);
            acc[19] = __fmaf_rn(w_t[5], x_values[18], acc[19]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_51;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_51) : "f"(x_values[19]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[18]), "+f"(acc[19])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_51));
#else
            acc[18] = __fmaf_rn(w_t[6], x_values[19], acc[18]);
            acc[19] = __fmaf_rn(w_t[7], x_values[19], acc[19]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_52;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_52) : "f"(x_values[20]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[18]), "+f"(acc[19])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_52));
#else
            acc[18] = __fmaf_rn(w_t[8], x_values[20], acc[18]);
            acc[19] = __fmaf_rn(w_t[9], x_values[20], acc[19]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_53;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_53) : "f"(x_values[21]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[18]), "+f"(acc[19])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_53));
#else
            acc[18] = __fmaf_rn(w_t[10], x_values[21], acc[18]);
            acc[19] = __fmaf_rn(w_t[11], x_values[21], acc[19]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_54;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_54) : "f"(x_values[22]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[18]), "+f"(acc[19])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_54));
#else
            acc[18] = __fmaf_rn(w_t[12], x_values[22], acc[18]);
            acc[19] = __fmaf_rn(w_t[13], x_values[22], acc[19]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_55;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_55) : "f"(x_values[23]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[18]), "+f"(acc[19])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_55));
#else
            acc[18] = __fmaf_rn(w_t[14], x_values[23], acc[18]);
            acc[19] = __fmaf_rn(w_t[15], x_values[23], acc[19]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_56;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_56) : "f"(x_values[24]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[26]), "+f"(acc[27])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_56));
#else
            acc[26] = __fmaf_rn(w_t[0], x_values[24], acc[26]);
            acc[27] = __fmaf_rn(w_t[1], x_values[24], acc[27]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_57;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_57) : "f"(x_values[25]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[26]), "+f"(acc[27])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_57));
#else
            acc[26] = __fmaf_rn(w_t[2], x_values[25], acc[26]);
            acc[27] = __fmaf_rn(w_t[3], x_values[25], acc[27]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_58;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_58) : "f"(x_values[26]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[26]), "+f"(acc[27])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_58));
#else
            acc[26] = __fmaf_rn(w_t[4], x_values[26], acc[26]);
            acc[27] = __fmaf_rn(w_t[5], x_values[26], acc[27]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_59;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_59) : "f"(x_values[27]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[26]), "+f"(acc[27])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_59));
#else
            acc[26] = __fmaf_rn(w_t[6], x_values[27], acc[26]);
            acc[27] = __fmaf_rn(w_t[7], x_values[27], acc[27]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_60;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_60) : "f"(x_values[28]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[26]), "+f"(acc[27])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_60));
#else
            acc[26] = __fmaf_rn(w_t[8], x_values[28], acc[26]);
            acc[27] = __fmaf_rn(w_t[9], x_values[28], acc[27]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_61;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_61) : "f"(x_values[29]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[26]), "+f"(acc[27])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_61));
#else
            acc[26] = __fmaf_rn(w_t[10], x_values[29], acc[26]);
            acc[27] = __fmaf_rn(w_t[11], x_values[29], acc[27]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_62;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_62) : "f"(x_values[30]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[26]), "+f"(acc[27])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_62));
#else
            acc[26] = __fmaf_rn(w_t[12], x_values[30], acc[26]);
            acc[27] = __fmaf_rn(w_t[13], x_values[30], acc[27]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_63;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_63) : "f"(x_values[31]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[26]), "+f"(acc[27])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_63));
#else
            acc[26] = __fmaf_rn(w_t[14], x_values[31], acc[26]);
            acc[27] = __fmaf_rn(w_t[15], x_values[31], acc[27]);
#endif
          }
          asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                       : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
                       : "r"(w_ring_addr + (unsigned int)(((stage * 8 + 4) * 1024 + tid_vec) * 2)));
          w_t[0] = __uint_as_float(w_car[0] << 16);
          w_t[2] = __uint_as_float(w_car[0] & 4294901760u);
          w_t[4] = __uint_as_float(w_car[1] << 16);
          w_t[6] = __uint_as_float(w_car[1] & 4294901760u);
          w_t[8] = __uint_as_float(w_car[2] << 16);
          w_t[10] = __uint_as_float(w_car[2] & 4294901760u);
          w_t[12] = __uint_as_float(w_car[3] << 16);
          w_t[14] = __uint_as_float(w_car[3] & 4294901760u);
          asm volatile(
              "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
              : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
              : "r"(w_ring_addr + (unsigned int)(((stage * 8 + 4 + 1) * 1024 + tid_vec) * 2)));
          w_t[1] = __uint_as_float(w_car[0] << 16);
          w_t[3] = __uint_as_float(w_car[0] & 4294901760u);
          w_t[5] = __uint_as_float(w_car[1] << 16);
          w_t[7] = __uint_as_float(w_car[1] & 4294901760u);
          w_t[9] = __uint_as_float(w_car[2] << 16);
          w_t[11] = __uint_as_float(w_car[2] & 4294901760u);
          w_t[13] = __uint_as_float(w_car[3] << 16);
          w_t[15] = __uint_as_float(w_car[3] & 4294901760u);
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_64;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_64) : "f"(x_values[0]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[4]), "+f"(acc[5])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_64));
#else
            acc[4] = __fmaf_rn(w_t[0], x_values[0], acc[4]);
            acc[5] = __fmaf_rn(w_t[1], x_values[0], acc[5]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_65;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_65) : "f"(x_values[1]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[4]), "+f"(acc[5])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_65));
#else
            acc[4] = __fmaf_rn(w_t[2], x_values[1], acc[4]);
            acc[5] = __fmaf_rn(w_t[3], x_values[1], acc[5]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_66;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_66) : "f"(x_values[2]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[4]), "+f"(acc[5])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_66));
#else
            acc[4] = __fmaf_rn(w_t[4], x_values[2], acc[4]);
            acc[5] = __fmaf_rn(w_t[5], x_values[2], acc[5]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_67;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_67) : "f"(x_values[3]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[4]), "+f"(acc[5])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_67));
#else
            acc[4] = __fmaf_rn(w_t[6], x_values[3], acc[4]);
            acc[5] = __fmaf_rn(w_t[7], x_values[3], acc[5]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_68;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_68) : "f"(x_values[4]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[4]), "+f"(acc[5])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_68));
#else
            acc[4] = __fmaf_rn(w_t[8], x_values[4], acc[4]);
            acc[5] = __fmaf_rn(w_t[9], x_values[4], acc[5]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_69;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_69) : "f"(x_values[5]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[4]), "+f"(acc[5])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_69));
#else
            acc[4] = __fmaf_rn(w_t[10], x_values[5], acc[4]);
            acc[5] = __fmaf_rn(w_t[11], x_values[5], acc[5]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_70;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_70) : "f"(x_values[6]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[4]), "+f"(acc[5])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_70));
#else
            acc[4] = __fmaf_rn(w_t[12], x_values[6], acc[4]);
            acc[5] = __fmaf_rn(w_t[13], x_values[6], acc[5]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_71;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_71) : "f"(x_values[7]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[4]), "+f"(acc[5])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_71));
#else
            acc[4] = __fmaf_rn(w_t[14], x_values[7], acc[4]);
            acc[5] = __fmaf_rn(w_t[15], x_values[7], acc[5]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_72;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_72) : "f"(x_values[8]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[12]), "+f"(acc[13])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_72));
#else
            acc[12] = __fmaf_rn(w_t[0], x_values[8], acc[12]);
            acc[13] = __fmaf_rn(w_t[1], x_values[8], acc[13]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_73;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_73) : "f"(x_values[9]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[12]), "+f"(acc[13])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_73));
#else
            acc[12] = __fmaf_rn(w_t[2], x_values[9], acc[12]);
            acc[13] = __fmaf_rn(w_t[3], x_values[9], acc[13]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_74;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_74) : "f"(x_values[10]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[12]), "+f"(acc[13])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_74));
#else
            acc[12] = __fmaf_rn(w_t[4], x_values[10], acc[12]);
            acc[13] = __fmaf_rn(w_t[5], x_values[10], acc[13]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_75;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_75) : "f"(x_values[11]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[12]), "+f"(acc[13])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_75));
#else
            acc[12] = __fmaf_rn(w_t[6], x_values[11], acc[12]);
            acc[13] = __fmaf_rn(w_t[7], x_values[11], acc[13]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_76;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_76) : "f"(x_values[12]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[12]), "+f"(acc[13])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_76));
#else
            acc[12] = __fmaf_rn(w_t[8], x_values[12], acc[12]);
            acc[13] = __fmaf_rn(w_t[9], x_values[12], acc[13]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_77;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_77) : "f"(x_values[13]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[12]), "+f"(acc[13])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_77));
#else
            acc[12] = __fmaf_rn(w_t[10], x_values[13], acc[12]);
            acc[13] = __fmaf_rn(w_t[11], x_values[13], acc[13]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_78;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_78) : "f"(x_values[14]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[12]), "+f"(acc[13])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_78));
#else
            acc[12] = __fmaf_rn(w_t[12], x_values[14], acc[12]);
            acc[13] = __fmaf_rn(w_t[13], x_values[14], acc[13]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_79;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_79) : "f"(x_values[15]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[12]), "+f"(acc[13])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_79));
#else
            acc[12] = __fmaf_rn(w_t[14], x_values[15], acc[12]);
            acc[13] = __fmaf_rn(w_t[15], x_values[15], acc[13]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_80;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_80) : "f"(x_values[16]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[20]), "+f"(acc[21])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_80));
#else
            acc[20] = __fmaf_rn(w_t[0], x_values[16], acc[20]);
            acc[21] = __fmaf_rn(w_t[1], x_values[16], acc[21]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_81;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_81) : "f"(x_values[17]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[20]), "+f"(acc[21])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_81));
#else
            acc[20] = __fmaf_rn(w_t[2], x_values[17], acc[20]);
            acc[21] = __fmaf_rn(w_t[3], x_values[17], acc[21]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_82;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_82) : "f"(x_values[18]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[20]), "+f"(acc[21])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_82));
#else
            acc[20] = __fmaf_rn(w_t[4], x_values[18], acc[20]);
            acc[21] = __fmaf_rn(w_t[5], x_values[18], acc[21]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_83;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_83) : "f"(x_values[19]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[20]), "+f"(acc[21])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_83));
#else
            acc[20] = __fmaf_rn(w_t[6], x_values[19], acc[20]);
            acc[21] = __fmaf_rn(w_t[7], x_values[19], acc[21]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_84;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_84) : "f"(x_values[20]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[20]), "+f"(acc[21])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_84));
#else
            acc[20] = __fmaf_rn(w_t[8], x_values[20], acc[20]);
            acc[21] = __fmaf_rn(w_t[9], x_values[20], acc[21]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_85;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_85) : "f"(x_values[21]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[20]), "+f"(acc[21])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_85));
#else
            acc[20] = __fmaf_rn(w_t[10], x_values[21], acc[20]);
            acc[21] = __fmaf_rn(w_t[11], x_values[21], acc[21]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_86;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_86) : "f"(x_values[22]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[20]), "+f"(acc[21])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_86));
#else
            acc[20] = __fmaf_rn(w_t[12], x_values[22], acc[20]);
            acc[21] = __fmaf_rn(w_t[13], x_values[22], acc[21]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_87;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_87) : "f"(x_values[23]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[20]), "+f"(acc[21])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_87));
#else
            acc[20] = __fmaf_rn(w_t[14], x_values[23], acc[20]);
            acc[21] = __fmaf_rn(w_t[15], x_values[23], acc[21]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_88;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_88) : "f"(x_values[24]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[28]), "+f"(acc[29])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_88));
#else
            acc[28] = __fmaf_rn(w_t[0], x_values[24], acc[28]);
            acc[29] = __fmaf_rn(w_t[1], x_values[24], acc[29]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_89;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_89) : "f"(x_values[25]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[28]), "+f"(acc[29])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_89));
#else
            acc[28] = __fmaf_rn(w_t[2], x_values[25], acc[28]);
            acc[29] = __fmaf_rn(w_t[3], x_values[25], acc[29]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_90;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_90) : "f"(x_values[26]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[28]), "+f"(acc[29])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_90));
#else
            acc[28] = __fmaf_rn(w_t[4], x_values[26], acc[28]);
            acc[29] = __fmaf_rn(w_t[5], x_values[26], acc[29]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_91;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_91) : "f"(x_values[27]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[28]), "+f"(acc[29])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_91));
#else
            acc[28] = __fmaf_rn(w_t[6], x_values[27], acc[28]);
            acc[29] = __fmaf_rn(w_t[7], x_values[27], acc[29]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_92;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_92) : "f"(x_values[28]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[28]), "+f"(acc[29])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_92));
#else
            acc[28] = __fmaf_rn(w_t[8], x_values[28], acc[28]);
            acc[29] = __fmaf_rn(w_t[9], x_values[28], acc[29]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_93;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_93) : "f"(x_values[29]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[28]), "+f"(acc[29])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_93));
#else
            acc[28] = __fmaf_rn(w_t[10], x_values[29], acc[28]);
            acc[29] = __fmaf_rn(w_t[11], x_values[29], acc[29]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_94;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_94) : "f"(x_values[30]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[28]), "+f"(acc[29])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_94));
#else
            acc[28] = __fmaf_rn(w_t[12], x_values[30], acc[28]);
            acc[29] = __fmaf_rn(w_t[13], x_values[30], acc[29]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_95;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_95) : "f"(x_values[31]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[28]), "+f"(acc[29])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_95));
#else
            acc[28] = __fmaf_rn(w_t[14], x_values[31], acc[28]);
            acc[29] = __fmaf_rn(w_t[15], x_values[31], acc[29]);
#endif
          }
          asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                       : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
                       : "r"(w_ring_addr + (unsigned int)(((stage * 8 + 6) * 1024 + tid_vec) * 2)));
          w_t[0] = __uint_as_float(w_car[0] << 16);
          w_t[2] = __uint_as_float(w_car[0] & 4294901760u);
          w_t[4] = __uint_as_float(w_car[1] << 16);
          w_t[6] = __uint_as_float(w_car[1] & 4294901760u);
          w_t[8] = __uint_as_float(w_car[2] << 16);
          w_t[10] = __uint_as_float(w_car[2] & 4294901760u);
          w_t[12] = __uint_as_float(w_car[3] << 16);
          w_t[14] = __uint_as_float(w_car[3] & 4294901760u);
          asm volatile(
              "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
              : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
              : "r"(w_ring_addr + (unsigned int)(((stage * 8 + 6 + 1) * 1024 + tid_vec) * 2)));
          w_t[1] = __uint_as_float(w_car[0] << 16);
          w_t[3] = __uint_as_float(w_car[0] & 4294901760u);
          w_t[5] = __uint_as_float(w_car[1] << 16);
          w_t[7] = __uint_as_float(w_car[1] & 4294901760u);
          w_t[9] = __uint_as_float(w_car[2] << 16);
          w_t[11] = __uint_as_float(w_car[2] & 4294901760u);
          w_t[13] = __uint_as_float(w_car[3] << 16);
          w_t[15] = __uint_as_float(w_car[3] & 4294901760u);
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_96;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_96) : "f"(x_values[0]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[6]), "+f"(acc[7])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_96));
#else
            acc[6] = __fmaf_rn(w_t[0], x_values[0], acc[6]);
            acc[7] = __fmaf_rn(w_t[1], x_values[0], acc[7]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_97;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_97) : "f"(x_values[1]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[6]), "+f"(acc[7])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_97));
#else
            acc[6] = __fmaf_rn(w_t[2], x_values[1], acc[6]);
            acc[7] = __fmaf_rn(w_t[3], x_values[1], acc[7]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_98;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_98) : "f"(x_values[2]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[6]), "+f"(acc[7])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_98));
#else
            acc[6] = __fmaf_rn(w_t[4], x_values[2], acc[6]);
            acc[7] = __fmaf_rn(w_t[5], x_values[2], acc[7]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_99;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_99) : "f"(x_values[3]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[6]), "+f"(acc[7])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_99));
#else
            acc[6] = __fmaf_rn(w_t[6], x_values[3], acc[6]);
            acc[7] = __fmaf_rn(w_t[7], x_values[3], acc[7]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_100;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_100) : "f"(x_values[4]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[6]), "+f"(acc[7])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_100));
#else
            acc[6] = __fmaf_rn(w_t[8], x_values[4], acc[6]);
            acc[7] = __fmaf_rn(w_t[9], x_values[4], acc[7]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_101;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_101) : "f"(x_values[5]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[6]), "+f"(acc[7])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_101));
#else
            acc[6] = __fmaf_rn(w_t[10], x_values[5], acc[6]);
            acc[7] = __fmaf_rn(w_t[11], x_values[5], acc[7]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_102;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_102) : "f"(x_values[6]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[6]), "+f"(acc[7])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_102));
#else
            acc[6] = __fmaf_rn(w_t[12], x_values[6], acc[6]);
            acc[7] = __fmaf_rn(w_t[13], x_values[6], acc[7]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_103;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_103) : "f"(x_values[7]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[6]), "+f"(acc[7])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_103));
#else
            acc[6] = __fmaf_rn(w_t[14], x_values[7], acc[6]);
            acc[7] = __fmaf_rn(w_t[15], x_values[7], acc[7]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_104;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_104) : "f"(x_values[8]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[14]), "+f"(acc[15])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_104));
#else
            acc[14] = __fmaf_rn(w_t[0], x_values[8], acc[14]);
            acc[15] = __fmaf_rn(w_t[1], x_values[8], acc[15]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_105;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_105) : "f"(x_values[9]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[14]), "+f"(acc[15])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_105));
#else
            acc[14] = __fmaf_rn(w_t[2], x_values[9], acc[14]);
            acc[15] = __fmaf_rn(w_t[3], x_values[9], acc[15]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_106;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_106) : "f"(x_values[10]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[14]), "+f"(acc[15])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_106));
#else
            acc[14] = __fmaf_rn(w_t[4], x_values[10], acc[14]);
            acc[15] = __fmaf_rn(w_t[5], x_values[10], acc[15]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_107;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_107) : "f"(x_values[11]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[14]), "+f"(acc[15])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_107));
#else
            acc[14] = __fmaf_rn(w_t[6], x_values[11], acc[14]);
            acc[15] = __fmaf_rn(w_t[7], x_values[11], acc[15]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_108;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_108) : "f"(x_values[12]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[14]), "+f"(acc[15])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_108));
#else
            acc[14] = __fmaf_rn(w_t[8], x_values[12], acc[14]);
            acc[15] = __fmaf_rn(w_t[9], x_values[12], acc[15]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_109;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_109) : "f"(x_values[13]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[14]), "+f"(acc[15])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_109));
#else
            acc[14] = __fmaf_rn(w_t[10], x_values[13], acc[14]);
            acc[15] = __fmaf_rn(w_t[11], x_values[13], acc[15]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_110;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_110) : "f"(x_values[14]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[14]), "+f"(acc[15])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_110));
#else
            acc[14] = __fmaf_rn(w_t[12], x_values[14], acc[14]);
            acc[15] = __fmaf_rn(w_t[13], x_values[14], acc[15]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_111;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_111) : "f"(x_values[15]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[14]), "+f"(acc[15])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_111));
#else
            acc[14] = __fmaf_rn(w_t[14], x_values[15], acc[14]);
            acc[15] = __fmaf_rn(w_t[15], x_values[15], acc[15]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_112;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_112) : "f"(x_values[16]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[22]), "+f"(acc[23])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_112));
#else
            acc[22] = __fmaf_rn(w_t[0], x_values[16], acc[22]);
            acc[23] = __fmaf_rn(w_t[1], x_values[16], acc[23]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_113;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_113) : "f"(x_values[17]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[22]), "+f"(acc[23])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_113));
#else
            acc[22] = __fmaf_rn(w_t[2], x_values[17], acc[22]);
            acc[23] = __fmaf_rn(w_t[3], x_values[17], acc[23]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_114;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_114) : "f"(x_values[18]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[22]), "+f"(acc[23])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_114));
#else
            acc[22] = __fmaf_rn(w_t[4], x_values[18], acc[22]);
            acc[23] = __fmaf_rn(w_t[5], x_values[18], acc[23]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_115;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_115) : "f"(x_values[19]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[22]), "+f"(acc[23])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_115));
#else
            acc[22] = __fmaf_rn(w_t[6], x_values[19], acc[22]);
            acc[23] = __fmaf_rn(w_t[7], x_values[19], acc[23]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_116;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_116) : "f"(x_values[20]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[22]), "+f"(acc[23])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_116));
#else
            acc[22] = __fmaf_rn(w_t[8], x_values[20], acc[22]);
            acc[23] = __fmaf_rn(w_t[9], x_values[20], acc[23]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_117;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_117) : "f"(x_values[21]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[22]), "+f"(acc[23])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_117));
#else
            acc[22] = __fmaf_rn(w_t[10], x_values[21], acc[22]);
            acc[23] = __fmaf_rn(w_t[11], x_values[21], acc[23]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_118;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_118) : "f"(x_values[22]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[22]), "+f"(acc[23])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_118));
#else
            acc[22] = __fmaf_rn(w_t[12], x_values[22], acc[22]);
            acc[23] = __fmaf_rn(w_t[13], x_values[22], acc[23]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_119;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_119) : "f"(x_values[23]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[22]), "+f"(acc[23])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_119));
#else
            acc[22] = __fmaf_rn(w_t[14], x_values[23], acc[22]);
            acc[23] = __fmaf_rn(w_t[15], x_values[23], acc[23]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_120;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_120) : "f"(x_values[24]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[30]), "+f"(acc[31])
                : "f"(w_t[0]), "f"(w_t[1]), "l"(_fma_acc_scale2_120));
#else
            acc[30] = __fmaf_rn(w_t[0], x_values[24], acc[30]);
            acc[31] = __fmaf_rn(w_t[1], x_values[24], acc[31]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_121;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_121) : "f"(x_values[25]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[30]), "+f"(acc[31])
                : "f"(w_t[2]), "f"(w_t[3]), "l"(_fma_acc_scale2_121));
#else
            acc[30] = __fmaf_rn(w_t[2], x_values[25], acc[30]);
            acc[31] = __fmaf_rn(w_t[3], x_values[25], acc[31]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_122;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_122) : "f"(x_values[26]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[30]), "+f"(acc[31])
                : "f"(w_t[4]), "f"(w_t[5]), "l"(_fma_acc_scale2_122));
#else
            acc[30] = __fmaf_rn(w_t[4], x_values[26], acc[30]);
            acc[31] = __fmaf_rn(w_t[5], x_values[26], acc[31]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_123;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_123) : "f"(x_values[27]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[30]), "+f"(acc[31])
                : "f"(w_t[6]), "f"(w_t[7]), "l"(_fma_acc_scale2_123));
#else
            acc[30] = __fmaf_rn(w_t[6], x_values[27], acc[30]);
            acc[31] = __fmaf_rn(w_t[7], x_values[27], acc[31]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_124;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_124) : "f"(x_values[28]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[30]), "+f"(acc[31])
                : "f"(w_t[8]), "f"(w_t[9]), "l"(_fma_acc_scale2_124));
#else
            acc[30] = __fmaf_rn(w_t[8], x_values[28], acc[30]);
            acc[31] = __fmaf_rn(w_t[9], x_values[28], acc[31]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_125;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_125) : "f"(x_values[29]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[30]), "+f"(acc[31])
                : "f"(w_t[10]), "f"(w_t[11]), "l"(_fma_acc_scale2_125));
#else
            acc[30] = __fmaf_rn(w_t[10], x_values[29], acc[30]);
            acc[31] = __fmaf_rn(w_t[11], x_values[29], acc[31]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_126;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_126) : "f"(x_values[30]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[30]), "+f"(acc[31])
                : "f"(w_t[12]), "f"(w_t[13]), "l"(_fma_acc_scale2_126));
#else
            acc[30] = __fmaf_rn(w_t[12], x_values[30], acc[30]);
            acc[31] = __fmaf_rn(w_t[13], x_values[30], acc[31]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            unsigned long long _fma_acc_scale2_127;
            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_127) : "f"(x_values[31]));
            asm volatile(
                "{\n\t"
                ".reg .b64 _src2, _acc2, _out2;\n\t"
                "mov.b64 _src2, {%2, %3};\n\t"
                "mov.b64 _acc2, {%0, %1};\n\t"
                "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
                "mov.b64 {%0, %1}, _out2;\n\t"
                "}"
                : "+f"(acc[30]), "+f"(acc[31])
                : "f"(w_t[14]), "f"(w_t[15]), "l"(_fma_acc_scale2_127));
#else
            acc[30] = __fmaf_rn(w_t[14], x_values[31], acc[30]);
            acc[31] = __fmaf_rn(w_t[15], x_values[31], acc[31]);
#endif
          }
        }
      }
#pragma unroll
      for (int i = 0; i < 16; i++) {
        float _shfl_xor_0 =
            __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) != 0) ? acc[i] : acc[i + 16]), 16);
        red_a[i] = (((lane_0 & 16) != 0) ? acc[i + 16] : acc[i]) + _shfl_xor_0;
      }
#pragma unroll
      for (int i_1 = 0; i_1 < 8; i_1++) {
        float _shfl_xor_1 =
            __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) != 0) ? red_a[i_1] : red_a[i_1 + 8]), 8);
        red_b[i_1] = (((lane_0 & 8) != 0) ? red_a[i_1 + 8] : red_a[i_1]) + _shfl_xor_1;
      }
#pragma unroll
      for (int i_2 = 0; i_2 < 4; i_2++) {
        float _shfl_xor_2 =
            __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 4) != 0) ? red_b[i_2] : red_b[i_2 + 4]), 4);
        red_c[i_2] = (((lane_0 & 4) != 0) ? red_b[i_2 + 4] : red_b[i_2]) + _shfl_xor_2;
      }
#pragma unroll
      for (int i_3 = 0; i_3 < 2; i_3++) {
        float _shfl_xor_3 =
            __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 2) != 0) ? red_c[i_3] : red_c[i_3 + 2]), 2);
        red_d[i_3] = (((lane_0 & 2) != 0) ? red_c[i_3 + 2] : red_c[i_3]) + _shfl_xor_3;
      }
#pragma unroll
      for (int i_4 = 0; i_4 < 1; i_4++) {
        float _shfl_xor_4 =
            __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 1) != 0) ? red_d[i_4] : red_d[i_4 + 1]), 1);
        red_e[i_4] = (((lane_0 & 1) != 0) ? red_d[i_4 + 1] : red_d[i_4]) + _shfl_xor_4;
      }
#pragma unroll
      for (int i_5 = 0; i_5 < 1; i_5++) {
        warp_partials[(lane_0 + i_5) * 4 + warp] = red_e[i_5];
      }
      __syncthreads();
      if (tid < 32) {
        float owned_accum = 0.0f;
#pragma unroll
        for (int source_warp = 0; source_warp < 4; source_warp++) {
          owned_accum += warp_partials[tid * 4 + source_warp];
        }
        int owner_j = tid / 8;
        int owner_rr = tid % 8;
        if (owner_j < count) {
          int owner_route = (int)reinterpret_cast<const unsigned int*>(
              workspace_raw)[off_sorted_routes + start + owner_j];
          *(reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<__nv_bfloat16*>(shrink_out_raw) +
                                             (owner_route * 32 + rank_base0 + owner_rr)) +
            (0)) = __float2bfloat16_rn(owned_accum);
        }
      }
    }
  }
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

}  // extern "C"

#undef BLACKWELL_INF
#undef NUM_MAIN_STAGES
#undef SMEM_TOTAL
#undef SMEM_WARP_PARTIALS_OFF
#undef SMEM_WARP_PARTIALS_STAGE_BYTES
#undef SMEM_WARP_PARTIALS_STRIDE
#undef SMEM_W_RING_OFF
#undef SMEM_W_RING_STAGE_BYTES
#undef SMEM_W_RING_STRIDE
#undef SMEM_X_RING_OFF
#undef SMEM_X_RING_STAGE_BYTES
#undef SMEM_X_RING_STRIDE
#undef THREADS

#define BLACKWELL_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_WARP_PARTIALS_OFF 0
#define SMEM_WARP_PARTIALS_STAGE_BYTES 512
#define SMEM_WARP_PARTIALS_STRIDE 512
#define SMEM_X_RING_OFF 512
#define SMEM_X_RING_STAGE_BYTES 16
#define SMEM_X_RING_STRIDE 16
#define SMEM_W_RING_OFF 512
#define SMEM_W_RING_STAGE_BYTES 32768
#define SMEM_W_RING_STRIDE 32768
#define SMEM_TOTAL 33280
#define THREADS 128

extern "C" {

__global__
__launch_bounds__(128, 6) void kernel_flashinfer_bgmv_moe_shrink_grouped_ring_mixed_single_bf16_r32(
    uint16_t* __restrict__ shrink_out_raw, uint16_t* __restrict__ x_raw,
    uint16_t* __restrict__ lora_a_raw, long long* __restrict__ sorted_token_ids, int num_pairs,
    int num_experts, int hidden, int num_tiles, int rt_per_cta, int rt_groups,
    unsigned int* __restrict__ workspace_raw, int off_group_offset, int off_tile_table,
    int off_sorted_routes) {
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
  float* warp_partials = reinterpret_cast<float*>(smem_raw + 0);
  const int warp_partials_addr = smem + 0;
  __nv_bfloat16* x_ring = reinterpret_cast<__nv_bfloat16*>(smem_raw + 512);
  const int x_ring_addr = smem + 512;
  __nv_bfloat16* w_ring = reinterpret_cast<__nv_bfloat16*>(smem_raw + 512);
  const int w_ring_addr = smem + 512;

  // === Task calls (dependency order) ===
  asm volatile("griddepcontrol.wait;" ::: "memory");
  int rt_group = blockIdx.x % ((1) ? 4 : rt_groups);
  int half = blockIdx.x / ((1) ? 4 : rt_groups) % 4;
  int tile = blockIdx.x / (((1) ? 4 : rt_groups) * 4);
  int n_tiles = (int)reinterpret_cast<const unsigned int*>(workspace_raw)[0];
  if (tile < n_tiles) {
    int entry = (int)reinterpret_cast<const unsigned int*>(workspace_raw)[off_tile_table + tile];
    int group = entry / 65536;
    int chunk = entry % 65536;
    int start =
        (int)reinterpret_cast<const unsigned int*>(workspace_raw)[off_group_offset + group] +
        chunk * 16 + half * 4;
    int group_end =
        (int)reinterpret_cast<const unsigned int*>(workspace_raw)[off_group_offset + group + 1];
    int count = group_end - start;
    if (count > 4) {
      count = 4;
    }
    if (count > 0) {
      int lora = group / num_experts;
      int expert = group % num_experts;
      int hidden_words = hidden / 2;
      long long weight_row_base = (long long)(lora * num_experts + expert) * 32;
      int routes[4];
      long long tokens[4];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        routes[j] = -1;
        tokens[j] = 0;
        if (count > j) {
          routes[j] = (int)reinterpret_cast<const unsigned int*>(
              workspace_raw)[off_sorted_routes + start + j];
          tokens[j] = sorted_token_ids[routes[j]];
        }
      }
      int rank_base0 = rt_group * 8;
      long long weight_words0 = (weight_row_base + (long long)rank_base0) * (long long)hidden_words;
      float acc[32];
      unsigned int x_car[16];
      unsigned int xr_car[4];
      unsigned int w_car[4];
      float x_values[32];
      float w_values[8];
      float w_t[16];
      int lane_0 = lane;
      float red_a[16];
      float red_b[8];
      float red_c[4];
      float red_d[2];
      float red_e[1];
#pragma unroll
      for (int owner = 0; owner < 32; owner++) {
        acc[owner] = 0.0f;
      }
      int tid_vec = tid * 8;
      if (tid_vec < hidden) {
        int kw_x0 = tid_vec / 2;
#pragma unroll
        for (int j_1 = 0; j_1 < 4; j_1++) {
          {
            const uint4* _ivptr_0 = reinterpret_cast<const uint4*>(
                reinterpret_cast<const unsigned int*>(x_raw) +
                tokens[j_1] * (long long)hidden_words + (long long)kw_x0);
            uint4 _ivld_0;
            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(_ivld_0.x), "=r"(_ivld_0.y), "=r"(_ivld_0.z), "=r"(_ivld_0.w)
                         : "l"((const void*)(_ivptr_0))
                         : "memory");
            (x_car + j_1 * 4)[0 + 0] = _ivld_0.x;
            (x_car + j_1 * 4)[0 + 1] = _ivld_0.y;
            (x_car + j_1 * 4)[0 + 2] = _ivld_0.z;
            (x_car + j_1 * 4)[0 + 3] = _ivld_0.w;
          }
        }
      }
#pragma unroll
      for (int d = 0; d < 1; d++) {
        int k_p = d * 1024 + tid_vec;
        if (k_p < hidden) {
          int kw_p = k_p / 2;
#pragma unroll
          for (int r = 0; r < 8; r++) {
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                             w_ring_addr + (unsigned int)(((d * 8 + r) * 1024 + tid_vec) * 2)),
                         "l"(reinterpret_cast<const unsigned int*>(lora_a_raw) +
                             (weight_words0 + (long long)(r * hidden_words) + (long long)kw_p)));
          }
        }
        asm volatile("cp.async.commit_group;");
      }
#pragma unroll 1
      for (int local = 0; local < num_tiles; local++) {
        int stage = local % 2;
        int nxt = local + 1;
        if (nxt < num_tiles) {
          int nstage = nxt % 2;
          int k_n = nxt * 1024 + tid_vec;
          if (k_n < hidden) {
            int kw_n = k_n / 2;
#pragma unroll
            for (int r_1 = 0; r_1 < 8; r_1++) {
              asm volatile(
                  "cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                      w_ring_addr + (unsigned int)(((nstage * 8 + r_1) * 1024 + tid_vec) * 2)),
                  "l"(reinterpret_cast<const unsigned int*>(lora_a_raw) +
                      (weight_words0 + (long long)(r_1 * hidden_words) + (long long)kw_n)));
            }
          }
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 1;");
        int k_base_r = local * 1024 + tid_vec;
        if (k_base_r < hidden) {
          asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                       : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
                       : "r"(w_ring_addr + (unsigned int)((stage * 8 * 1024 + tid_vec) * 2)));
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[0])
                : "r"(x_car[0]), "r"(x_car[1]), "r"(x_car[2]), "r"(x_car[3]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[0] =
                __fmaf_rn(__uint_as_float(x_car[0] << 16), __uint_as_float(w_car[0] << 16), acc[0]);
            acc[0] = __fmaf_rn(__uint_as_float(x_car[0] & 0xFFFF0000u),
                               __uint_as_float(w_car[0] & 0xFFFF0000u), acc[0]);
            acc[0] =
                __fmaf_rn(__uint_as_float(x_car[1] << 16), __uint_as_float(w_car[1] << 16), acc[0]);
            acc[0] = __fmaf_rn(__uint_as_float(x_car[1] & 0xFFFF0000u),
                               __uint_as_float(w_car[1] & 0xFFFF0000u), acc[0]);
            acc[0] =
                __fmaf_rn(__uint_as_float(x_car[2] << 16), __uint_as_float(w_car[2] << 16), acc[0]);
            acc[0] = __fmaf_rn(__uint_as_float(x_car[2] & 0xFFFF0000u),
                               __uint_as_float(w_car[2] & 0xFFFF0000u), acc[0]);
            acc[0] =
                __fmaf_rn(__uint_as_float(x_car[3] << 16), __uint_as_float(w_car[3] << 16), acc[0]);
            acc[0] = __fmaf_rn(__uint_as_float(x_car[3] & 0xFFFF0000u),
                               __uint_as_float(w_car[3] & 0xFFFF0000u), acc[0]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[8])
                : "r"(x_car[4]), "r"(x_car[5]), "r"(x_car[6]), "r"(x_car[7]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[8] =
                __fmaf_rn(__uint_as_float(x_car[4] << 16), __uint_as_float(w_car[0] << 16), acc[8]);
            acc[8] = __fmaf_rn(__uint_as_float(x_car[4] & 0xFFFF0000u),
                               __uint_as_float(w_car[0] & 0xFFFF0000u), acc[8]);
            acc[8] =
                __fmaf_rn(__uint_as_float(x_car[5] << 16), __uint_as_float(w_car[1] << 16), acc[8]);
            acc[8] = __fmaf_rn(__uint_as_float(x_car[5] & 0xFFFF0000u),
                               __uint_as_float(w_car[1] & 0xFFFF0000u), acc[8]);
            acc[8] =
                __fmaf_rn(__uint_as_float(x_car[6] << 16), __uint_as_float(w_car[2] << 16), acc[8]);
            acc[8] = __fmaf_rn(__uint_as_float(x_car[6] & 0xFFFF0000u),
                               __uint_as_float(w_car[2] & 0xFFFF0000u), acc[8]);
            acc[8] =
                __fmaf_rn(__uint_as_float(x_car[7] << 16), __uint_as_float(w_car[3] << 16), acc[8]);
            acc[8] = __fmaf_rn(__uint_as_float(x_car[7] & 0xFFFF0000u),
                               __uint_as_float(w_car[3] & 0xFFFF0000u), acc[8]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[16])
                : "r"(x_car[8]), "r"(x_car[9]), "r"(x_car[10]), "r"(x_car[11]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[16] = __fmaf_rn(__uint_as_float(x_car[8] << 16), __uint_as_float(w_car[0] << 16),
                                acc[16]);
            acc[16] = __fmaf_rn(__uint_as_float(x_car[8] & 0xFFFF0000u),
                                __uint_as_float(w_car[0] & 0xFFFF0000u), acc[16]);
            acc[16] = __fmaf_rn(__uint_as_float(x_car[9] << 16), __uint_as_float(w_car[1] << 16),
                                acc[16]);
            acc[16] = __fmaf_rn(__uint_as_float(x_car[9] & 0xFFFF0000u),
                                __uint_as_float(w_car[1] & 0xFFFF0000u), acc[16]);
            acc[16] = __fmaf_rn(__uint_as_float(x_car[10] << 16), __uint_as_float(w_car[2] << 16),
                                acc[16]);
            acc[16] = __fmaf_rn(__uint_as_float(x_car[10] & 0xFFFF0000u),
                                __uint_as_float(w_car[2] & 0xFFFF0000u), acc[16]);
            acc[16] = __fmaf_rn(__uint_as_float(x_car[11] << 16), __uint_as_float(w_car[3] << 16),
                                acc[16]);
            acc[16] = __fmaf_rn(__uint_as_float(x_car[11] & 0xFFFF0000u),
                                __uint_as_float(w_car[3] & 0xFFFF0000u), acc[16]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[24])
                : "r"(x_car[12]), "r"(x_car[13]), "r"(x_car[14]), "r"(x_car[15]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[24] = __fmaf_rn(__uint_as_float(x_car[12] << 16), __uint_as_float(w_car[0] << 16),
                                acc[24]);
            acc[24] = __fmaf_rn(__uint_as_float(x_car[12] & 0xFFFF0000u),
                                __uint_as_float(w_car[0] & 0xFFFF0000u), acc[24]);
            acc[24] = __fmaf_rn(__uint_as_float(x_car[13] << 16), __uint_as_float(w_car[1] << 16),
                                acc[24]);
            acc[24] = __fmaf_rn(__uint_as_float(x_car[13] & 0xFFFF0000u),
                                __uint_as_float(w_car[1] & 0xFFFF0000u), acc[24]);
            acc[24] = __fmaf_rn(__uint_as_float(x_car[14] << 16), __uint_as_float(w_car[2] << 16),
                                acc[24]);
            acc[24] = __fmaf_rn(__uint_as_float(x_car[14] & 0xFFFF0000u),
                                __uint_as_float(w_car[2] & 0xFFFF0000u), acc[24]);
            acc[24] = __fmaf_rn(__uint_as_float(x_car[15] << 16), __uint_as_float(w_car[3] << 16),
                                acc[24]);
            acc[24] = __fmaf_rn(__uint_as_float(x_car[15] & 0xFFFF0000u),
                                __uint_as_float(w_car[3] & 0xFFFF0000u), acc[24]);
#endif
          }
          asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                       : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
                       : "r"(w_ring_addr + (unsigned int)(((stage * 8 + 1) * 1024 + tid_vec) * 2)));
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[1])
                : "r"(x_car[0]), "r"(x_car[1]), "r"(x_car[2]), "r"(x_car[3]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[1] =
                __fmaf_rn(__uint_as_float(x_car[0] << 16), __uint_as_float(w_car[0] << 16), acc[1]);
            acc[1] = __fmaf_rn(__uint_as_float(x_car[0] & 0xFFFF0000u),
                               __uint_as_float(w_car[0] & 0xFFFF0000u), acc[1]);
            acc[1] =
                __fmaf_rn(__uint_as_float(x_car[1] << 16), __uint_as_float(w_car[1] << 16), acc[1]);
            acc[1] = __fmaf_rn(__uint_as_float(x_car[1] & 0xFFFF0000u),
                               __uint_as_float(w_car[1] & 0xFFFF0000u), acc[1]);
            acc[1] =
                __fmaf_rn(__uint_as_float(x_car[2] << 16), __uint_as_float(w_car[2] << 16), acc[1]);
            acc[1] = __fmaf_rn(__uint_as_float(x_car[2] & 0xFFFF0000u),
                               __uint_as_float(w_car[2] & 0xFFFF0000u), acc[1]);
            acc[1] =
                __fmaf_rn(__uint_as_float(x_car[3] << 16), __uint_as_float(w_car[3] << 16), acc[1]);
            acc[1] = __fmaf_rn(__uint_as_float(x_car[3] & 0xFFFF0000u),
                               __uint_as_float(w_car[3] & 0xFFFF0000u), acc[1]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[9])
                : "r"(x_car[4]), "r"(x_car[5]), "r"(x_car[6]), "r"(x_car[7]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[9] =
                __fmaf_rn(__uint_as_float(x_car[4] << 16), __uint_as_float(w_car[0] << 16), acc[9]);
            acc[9] = __fmaf_rn(__uint_as_float(x_car[4] & 0xFFFF0000u),
                               __uint_as_float(w_car[0] & 0xFFFF0000u), acc[9]);
            acc[9] =
                __fmaf_rn(__uint_as_float(x_car[5] << 16), __uint_as_float(w_car[1] << 16), acc[9]);
            acc[9] = __fmaf_rn(__uint_as_float(x_car[5] & 0xFFFF0000u),
                               __uint_as_float(w_car[1] & 0xFFFF0000u), acc[9]);
            acc[9] =
                __fmaf_rn(__uint_as_float(x_car[6] << 16), __uint_as_float(w_car[2] << 16), acc[9]);
            acc[9] = __fmaf_rn(__uint_as_float(x_car[6] & 0xFFFF0000u),
                               __uint_as_float(w_car[2] & 0xFFFF0000u), acc[9]);
            acc[9] =
                __fmaf_rn(__uint_as_float(x_car[7] << 16), __uint_as_float(w_car[3] << 16), acc[9]);
            acc[9] = __fmaf_rn(__uint_as_float(x_car[7] & 0xFFFF0000u),
                               __uint_as_float(w_car[3] & 0xFFFF0000u), acc[9]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[17])
                : "r"(x_car[8]), "r"(x_car[9]), "r"(x_car[10]), "r"(x_car[11]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[17] = __fmaf_rn(__uint_as_float(x_car[8] << 16), __uint_as_float(w_car[0] << 16),
                                acc[17]);
            acc[17] = __fmaf_rn(__uint_as_float(x_car[8] & 0xFFFF0000u),
                                __uint_as_float(w_car[0] & 0xFFFF0000u), acc[17]);
            acc[17] = __fmaf_rn(__uint_as_float(x_car[9] << 16), __uint_as_float(w_car[1] << 16),
                                acc[17]);
            acc[17] = __fmaf_rn(__uint_as_float(x_car[9] & 0xFFFF0000u),
                                __uint_as_float(w_car[1] & 0xFFFF0000u), acc[17]);
            acc[17] = __fmaf_rn(__uint_as_float(x_car[10] << 16), __uint_as_float(w_car[2] << 16),
                                acc[17]);
            acc[17] = __fmaf_rn(__uint_as_float(x_car[10] & 0xFFFF0000u),
                                __uint_as_float(w_car[2] & 0xFFFF0000u), acc[17]);
            acc[17] = __fmaf_rn(__uint_as_float(x_car[11] << 16), __uint_as_float(w_car[3] << 16),
                                acc[17]);
            acc[17] = __fmaf_rn(__uint_as_float(x_car[11] & 0xFFFF0000u),
                                __uint_as_float(w_car[3] & 0xFFFF0000u), acc[17]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[25])
                : "r"(x_car[12]), "r"(x_car[13]), "r"(x_car[14]), "r"(x_car[15]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[25] = __fmaf_rn(__uint_as_float(x_car[12] << 16), __uint_as_float(w_car[0] << 16),
                                acc[25]);
            acc[25] = __fmaf_rn(__uint_as_float(x_car[12] & 0xFFFF0000u),
                                __uint_as_float(w_car[0] & 0xFFFF0000u), acc[25]);
            acc[25] = __fmaf_rn(__uint_as_float(x_car[13] << 16), __uint_as_float(w_car[1] << 16),
                                acc[25]);
            acc[25] = __fmaf_rn(__uint_as_float(x_car[13] & 0xFFFF0000u),
                                __uint_as_float(w_car[1] & 0xFFFF0000u), acc[25]);
            acc[25] = __fmaf_rn(__uint_as_float(x_car[14] << 16), __uint_as_float(w_car[2] << 16),
                                acc[25]);
            acc[25] = __fmaf_rn(__uint_as_float(x_car[14] & 0xFFFF0000u),
                                __uint_as_float(w_car[2] & 0xFFFF0000u), acc[25]);
            acc[25] = __fmaf_rn(__uint_as_float(x_car[15] << 16), __uint_as_float(w_car[3] << 16),
                                acc[25]);
            acc[25] = __fmaf_rn(__uint_as_float(x_car[15] & 0xFFFF0000u),
                                __uint_as_float(w_car[3] & 0xFFFF0000u), acc[25]);
#endif
          }
          asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                       : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
                       : "r"(w_ring_addr + (unsigned int)(((stage * 8 + 2) * 1024 + tid_vec) * 2)));
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[2])
                : "r"(x_car[0]), "r"(x_car[1]), "r"(x_car[2]), "r"(x_car[3]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[2] =
                __fmaf_rn(__uint_as_float(x_car[0] << 16), __uint_as_float(w_car[0] << 16), acc[2]);
            acc[2] = __fmaf_rn(__uint_as_float(x_car[0] & 0xFFFF0000u),
                               __uint_as_float(w_car[0] & 0xFFFF0000u), acc[2]);
            acc[2] =
                __fmaf_rn(__uint_as_float(x_car[1] << 16), __uint_as_float(w_car[1] << 16), acc[2]);
            acc[2] = __fmaf_rn(__uint_as_float(x_car[1] & 0xFFFF0000u),
                               __uint_as_float(w_car[1] & 0xFFFF0000u), acc[2]);
            acc[2] =
                __fmaf_rn(__uint_as_float(x_car[2] << 16), __uint_as_float(w_car[2] << 16), acc[2]);
            acc[2] = __fmaf_rn(__uint_as_float(x_car[2] & 0xFFFF0000u),
                               __uint_as_float(w_car[2] & 0xFFFF0000u), acc[2]);
            acc[2] =
                __fmaf_rn(__uint_as_float(x_car[3] << 16), __uint_as_float(w_car[3] << 16), acc[2]);
            acc[2] = __fmaf_rn(__uint_as_float(x_car[3] & 0xFFFF0000u),
                               __uint_as_float(w_car[3] & 0xFFFF0000u), acc[2]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[10])
                : "r"(x_car[4]), "r"(x_car[5]), "r"(x_car[6]), "r"(x_car[7]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[10] = __fmaf_rn(__uint_as_float(x_car[4] << 16), __uint_as_float(w_car[0] << 16),
                                acc[10]);
            acc[10] = __fmaf_rn(__uint_as_float(x_car[4] & 0xFFFF0000u),
                                __uint_as_float(w_car[0] & 0xFFFF0000u), acc[10]);
            acc[10] = __fmaf_rn(__uint_as_float(x_car[5] << 16), __uint_as_float(w_car[1] << 16),
                                acc[10]);
            acc[10] = __fmaf_rn(__uint_as_float(x_car[5] & 0xFFFF0000u),
                                __uint_as_float(w_car[1] & 0xFFFF0000u), acc[10]);
            acc[10] = __fmaf_rn(__uint_as_float(x_car[6] << 16), __uint_as_float(w_car[2] << 16),
                                acc[10]);
            acc[10] = __fmaf_rn(__uint_as_float(x_car[6] & 0xFFFF0000u),
                                __uint_as_float(w_car[2] & 0xFFFF0000u), acc[10]);
            acc[10] = __fmaf_rn(__uint_as_float(x_car[7] << 16), __uint_as_float(w_car[3] << 16),
                                acc[10]);
            acc[10] = __fmaf_rn(__uint_as_float(x_car[7] & 0xFFFF0000u),
                                __uint_as_float(w_car[3] & 0xFFFF0000u), acc[10]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[18])
                : "r"(x_car[8]), "r"(x_car[9]), "r"(x_car[10]), "r"(x_car[11]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[18] = __fmaf_rn(__uint_as_float(x_car[8] << 16), __uint_as_float(w_car[0] << 16),
                                acc[18]);
            acc[18] = __fmaf_rn(__uint_as_float(x_car[8] & 0xFFFF0000u),
                                __uint_as_float(w_car[0] & 0xFFFF0000u), acc[18]);
            acc[18] = __fmaf_rn(__uint_as_float(x_car[9] << 16), __uint_as_float(w_car[1] << 16),
                                acc[18]);
            acc[18] = __fmaf_rn(__uint_as_float(x_car[9] & 0xFFFF0000u),
                                __uint_as_float(w_car[1] & 0xFFFF0000u), acc[18]);
            acc[18] = __fmaf_rn(__uint_as_float(x_car[10] << 16), __uint_as_float(w_car[2] << 16),
                                acc[18]);
            acc[18] = __fmaf_rn(__uint_as_float(x_car[10] & 0xFFFF0000u),
                                __uint_as_float(w_car[2] & 0xFFFF0000u), acc[18]);
            acc[18] = __fmaf_rn(__uint_as_float(x_car[11] << 16), __uint_as_float(w_car[3] << 16),
                                acc[18]);
            acc[18] = __fmaf_rn(__uint_as_float(x_car[11] & 0xFFFF0000u),
                                __uint_as_float(w_car[3] & 0xFFFF0000u), acc[18]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[26])
                : "r"(x_car[12]), "r"(x_car[13]), "r"(x_car[14]), "r"(x_car[15]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[26] = __fmaf_rn(__uint_as_float(x_car[12] << 16), __uint_as_float(w_car[0] << 16),
                                acc[26]);
            acc[26] = __fmaf_rn(__uint_as_float(x_car[12] & 0xFFFF0000u),
                                __uint_as_float(w_car[0] & 0xFFFF0000u), acc[26]);
            acc[26] = __fmaf_rn(__uint_as_float(x_car[13] << 16), __uint_as_float(w_car[1] << 16),
                                acc[26]);
            acc[26] = __fmaf_rn(__uint_as_float(x_car[13] & 0xFFFF0000u),
                                __uint_as_float(w_car[1] & 0xFFFF0000u), acc[26]);
            acc[26] = __fmaf_rn(__uint_as_float(x_car[14] << 16), __uint_as_float(w_car[2] << 16),
                                acc[26]);
            acc[26] = __fmaf_rn(__uint_as_float(x_car[14] & 0xFFFF0000u),
                                __uint_as_float(w_car[2] & 0xFFFF0000u), acc[26]);
            acc[26] = __fmaf_rn(__uint_as_float(x_car[15] << 16), __uint_as_float(w_car[3] << 16),
                                acc[26]);
            acc[26] = __fmaf_rn(__uint_as_float(x_car[15] & 0xFFFF0000u),
                                __uint_as_float(w_car[3] & 0xFFFF0000u), acc[26]);
#endif
          }
          asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                       : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
                       : "r"(w_ring_addr + (unsigned int)(((stage * 8 + 3) * 1024 + tid_vec) * 2)));
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[3])
                : "r"(x_car[0]), "r"(x_car[1]), "r"(x_car[2]), "r"(x_car[3]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[3] =
                __fmaf_rn(__uint_as_float(x_car[0] << 16), __uint_as_float(w_car[0] << 16), acc[3]);
            acc[3] = __fmaf_rn(__uint_as_float(x_car[0] & 0xFFFF0000u),
                               __uint_as_float(w_car[0] & 0xFFFF0000u), acc[3]);
            acc[3] =
                __fmaf_rn(__uint_as_float(x_car[1] << 16), __uint_as_float(w_car[1] << 16), acc[3]);
            acc[3] = __fmaf_rn(__uint_as_float(x_car[1] & 0xFFFF0000u),
                               __uint_as_float(w_car[1] & 0xFFFF0000u), acc[3]);
            acc[3] =
                __fmaf_rn(__uint_as_float(x_car[2] << 16), __uint_as_float(w_car[2] << 16), acc[3]);
            acc[3] = __fmaf_rn(__uint_as_float(x_car[2] & 0xFFFF0000u),
                               __uint_as_float(w_car[2] & 0xFFFF0000u), acc[3]);
            acc[3] =
                __fmaf_rn(__uint_as_float(x_car[3] << 16), __uint_as_float(w_car[3] << 16), acc[3]);
            acc[3] = __fmaf_rn(__uint_as_float(x_car[3] & 0xFFFF0000u),
                               __uint_as_float(w_car[3] & 0xFFFF0000u), acc[3]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[11])
                : "r"(x_car[4]), "r"(x_car[5]), "r"(x_car[6]), "r"(x_car[7]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[11] = __fmaf_rn(__uint_as_float(x_car[4] << 16), __uint_as_float(w_car[0] << 16),
                                acc[11]);
            acc[11] = __fmaf_rn(__uint_as_float(x_car[4] & 0xFFFF0000u),
                                __uint_as_float(w_car[0] & 0xFFFF0000u), acc[11]);
            acc[11] = __fmaf_rn(__uint_as_float(x_car[5] << 16), __uint_as_float(w_car[1] << 16),
                                acc[11]);
            acc[11] = __fmaf_rn(__uint_as_float(x_car[5] & 0xFFFF0000u),
                                __uint_as_float(w_car[1] & 0xFFFF0000u), acc[11]);
            acc[11] = __fmaf_rn(__uint_as_float(x_car[6] << 16), __uint_as_float(w_car[2] << 16),
                                acc[11]);
            acc[11] = __fmaf_rn(__uint_as_float(x_car[6] & 0xFFFF0000u),
                                __uint_as_float(w_car[2] & 0xFFFF0000u), acc[11]);
            acc[11] = __fmaf_rn(__uint_as_float(x_car[7] << 16), __uint_as_float(w_car[3] << 16),
                                acc[11]);
            acc[11] = __fmaf_rn(__uint_as_float(x_car[7] & 0xFFFF0000u),
                                __uint_as_float(w_car[3] & 0xFFFF0000u), acc[11]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[19])
                : "r"(x_car[8]), "r"(x_car[9]), "r"(x_car[10]), "r"(x_car[11]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[19] = __fmaf_rn(__uint_as_float(x_car[8] << 16), __uint_as_float(w_car[0] << 16),
                                acc[19]);
            acc[19] = __fmaf_rn(__uint_as_float(x_car[8] & 0xFFFF0000u),
                                __uint_as_float(w_car[0] & 0xFFFF0000u), acc[19]);
            acc[19] = __fmaf_rn(__uint_as_float(x_car[9] << 16), __uint_as_float(w_car[1] << 16),
                                acc[19]);
            acc[19] = __fmaf_rn(__uint_as_float(x_car[9] & 0xFFFF0000u),
                                __uint_as_float(w_car[1] & 0xFFFF0000u), acc[19]);
            acc[19] = __fmaf_rn(__uint_as_float(x_car[10] << 16), __uint_as_float(w_car[2] << 16),
                                acc[19]);
            acc[19] = __fmaf_rn(__uint_as_float(x_car[10] & 0xFFFF0000u),
                                __uint_as_float(w_car[2] & 0xFFFF0000u), acc[19]);
            acc[19] = __fmaf_rn(__uint_as_float(x_car[11] << 16), __uint_as_float(w_car[3] << 16),
                                acc[19]);
            acc[19] = __fmaf_rn(__uint_as_float(x_car[11] & 0xFFFF0000u),
                                __uint_as_float(w_car[3] & 0xFFFF0000u), acc[19]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[27])
                : "r"(x_car[12]), "r"(x_car[13]), "r"(x_car[14]), "r"(x_car[15]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[27] = __fmaf_rn(__uint_as_float(x_car[12] << 16), __uint_as_float(w_car[0] << 16),
                                acc[27]);
            acc[27] = __fmaf_rn(__uint_as_float(x_car[12] & 0xFFFF0000u),
                                __uint_as_float(w_car[0] & 0xFFFF0000u), acc[27]);
            acc[27] = __fmaf_rn(__uint_as_float(x_car[13] << 16), __uint_as_float(w_car[1] << 16),
                                acc[27]);
            acc[27] = __fmaf_rn(__uint_as_float(x_car[13] & 0xFFFF0000u),
                                __uint_as_float(w_car[1] & 0xFFFF0000u), acc[27]);
            acc[27] = __fmaf_rn(__uint_as_float(x_car[14] << 16), __uint_as_float(w_car[2] << 16),
                                acc[27]);
            acc[27] = __fmaf_rn(__uint_as_float(x_car[14] & 0xFFFF0000u),
                                __uint_as_float(w_car[2] & 0xFFFF0000u), acc[27]);
            acc[27] = __fmaf_rn(__uint_as_float(x_car[15] << 16), __uint_as_float(w_car[3] << 16),
                                acc[27]);
            acc[27] = __fmaf_rn(__uint_as_float(x_car[15] & 0xFFFF0000u),
                                __uint_as_float(w_car[3] & 0xFFFF0000u), acc[27]);
#endif
          }
          asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                       : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
                       : "r"(w_ring_addr + (unsigned int)(((stage * 8 + 4) * 1024 + tid_vec) * 2)));
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[4])
                : "r"(x_car[0]), "r"(x_car[1]), "r"(x_car[2]), "r"(x_car[3]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[4] =
                __fmaf_rn(__uint_as_float(x_car[0] << 16), __uint_as_float(w_car[0] << 16), acc[4]);
            acc[4] = __fmaf_rn(__uint_as_float(x_car[0] & 0xFFFF0000u),
                               __uint_as_float(w_car[0] & 0xFFFF0000u), acc[4]);
            acc[4] =
                __fmaf_rn(__uint_as_float(x_car[1] << 16), __uint_as_float(w_car[1] << 16), acc[4]);
            acc[4] = __fmaf_rn(__uint_as_float(x_car[1] & 0xFFFF0000u),
                               __uint_as_float(w_car[1] & 0xFFFF0000u), acc[4]);
            acc[4] =
                __fmaf_rn(__uint_as_float(x_car[2] << 16), __uint_as_float(w_car[2] << 16), acc[4]);
            acc[4] = __fmaf_rn(__uint_as_float(x_car[2] & 0xFFFF0000u),
                               __uint_as_float(w_car[2] & 0xFFFF0000u), acc[4]);
            acc[4] =
                __fmaf_rn(__uint_as_float(x_car[3] << 16), __uint_as_float(w_car[3] << 16), acc[4]);
            acc[4] = __fmaf_rn(__uint_as_float(x_car[3] & 0xFFFF0000u),
                               __uint_as_float(w_car[3] & 0xFFFF0000u), acc[4]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[12])
                : "r"(x_car[4]), "r"(x_car[5]), "r"(x_car[6]), "r"(x_car[7]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[12] = __fmaf_rn(__uint_as_float(x_car[4] << 16), __uint_as_float(w_car[0] << 16),
                                acc[12]);
            acc[12] = __fmaf_rn(__uint_as_float(x_car[4] & 0xFFFF0000u),
                                __uint_as_float(w_car[0] & 0xFFFF0000u), acc[12]);
            acc[12] = __fmaf_rn(__uint_as_float(x_car[5] << 16), __uint_as_float(w_car[1] << 16),
                                acc[12]);
            acc[12] = __fmaf_rn(__uint_as_float(x_car[5] & 0xFFFF0000u),
                                __uint_as_float(w_car[1] & 0xFFFF0000u), acc[12]);
            acc[12] = __fmaf_rn(__uint_as_float(x_car[6] << 16), __uint_as_float(w_car[2] << 16),
                                acc[12]);
            acc[12] = __fmaf_rn(__uint_as_float(x_car[6] & 0xFFFF0000u),
                                __uint_as_float(w_car[2] & 0xFFFF0000u), acc[12]);
            acc[12] = __fmaf_rn(__uint_as_float(x_car[7] << 16), __uint_as_float(w_car[3] << 16),
                                acc[12]);
            acc[12] = __fmaf_rn(__uint_as_float(x_car[7] & 0xFFFF0000u),
                                __uint_as_float(w_car[3] & 0xFFFF0000u), acc[12]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[20])
                : "r"(x_car[8]), "r"(x_car[9]), "r"(x_car[10]), "r"(x_car[11]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[20] = __fmaf_rn(__uint_as_float(x_car[8] << 16), __uint_as_float(w_car[0] << 16),
                                acc[20]);
            acc[20] = __fmaf_rn(__uint_as_float(x_car[8] & 0xFFFF0000u),
                                __uint_as_float(w_car[0] & 0xFFFF0000u), acc[20]);
            acc[20] = __fmaf_rn(__uint_as_float(x_car[9] << 16), __uint_as_float(w_car[1] << 16),
                                acc[20]);
            acc[20] = __fmaf_rn(__uint_as_float(x_car[9] & 0xFFFF0000u),
                                __uint_as_float(w_car[1] & 0xFFFF0000u), acc[20]);
            acc[20] = __fmaf_rn(__uint_as_float(x_car[10] << 16), __uint_as_float(w_car[2] << 16),
                                acc[20]);
            acc[20] = __fmaf_rn(__uint_as_float(x_car[10] & 0xFFFF0000u),
                                __uint_as_float(w_car[2] & 0xFFFF0000u), acc[20]);
            acc[20] = __fmaf_rn(__uint_as_float(x_car[11] << 16), __uint_as_float(w_car[3] << 16),
                                acc[20]);
            acc[20] = __fmaf_rn(__uint_as_float(x_car[11] & 0xFFFF0000u),
                                __uint_as_float(w_car[3] & 0xFFFF0000u), acc[20]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[28])
                : "r"(x_car[12]), "r"(x_car[13]), "r"(x_car[14]), "r"(x_car[15]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[28] = __fmaf_rn(__uint_as_float(x_car[12] << 16), __uint_as_float(w_car[0] << 16),
                                acc[28]);
            acc[28] = __fmaf_rn(__uint_as_float(x_car[12] & 0xFFFF0000u),
                                __uint_as_float(w_car[0] & 0xFFFF0000u), acc[28]);
            acc[28] = __fmaf_rn(__uint_as_float(x_car[13] << 16), __uint_as_float(w_car[1] << 16),
                                acc[28]);
            acc[28] = __fmaf_rn(__uint_as_float(x_car[13] & 0xFFFF0000u),
                                __uint_as_float(w_car[1] & 0xFFFF0000u), acc[28]);
            acc[28] = __fmaf_rn(__uint_as_float(x_car[14] << 16), __uint_as_float(w_car[2] << 16),
                                acc[28]);
            acc[28] = __fmaf_rn(__uint_as_float(x_car[14] & 0xFFFF0000u),
                                __uint_as_float(w_car[2] & 0xFFFF0000u), acc[28]);
            acc[28] = __fmaf_rn(__uint_as_float(x_car[15] << 16), __uint_as_float(w_car[3] << 16),
                                acc[28]);
            acc[28] = __fmaf_rn(__uint_as_float(x_car[15] & 0xFFFF0000u),
                                __uint_as_float(w_car[3] & 0xFFFF0000u), acc[28]);
#endif
          }
          asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                       : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
                       : "r"(w_ring_addr + (unsigned int)(((stage * 8 + 5) * 1024 + tid_vec) * 2)));
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[5])
                : "r"(x_car[0]), "r"(x_car[1]), "r"(x_car[2]), "r"(x_car[3]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[5] =
                __fmaf_rn(__uint_as_float(x_car[0] << 16), __uint_as_float(w_car[0] << 16), acc[5]);
            acc[5] = __fmaf_rn(__uint_as_float(x_car[0] & 0xFFFF0000u),
                               __uint_as_float(w_car[0] & 0xFFFF0000u), acc[5]);
            acc[5] =
                __fmaf_rn(__uint_as_float(x_car[1] << 16), __uint_as_float(w_car[1] << 16), acc[5]);
            acc[5] = __fmaf_rn(__uint_as_float(x_car[1] & 0xFFFF0000u),
                               __uint_as_float(w_car[1] & 0xFFFF0000u), acc[5]);
            acc[5] =
                __fmaf_rn(__uint_as_float(x_car[2] << 16), __uint_as_float(w_car[2] << 16), acc[5]);
            acc[5] = __fmaf_rn(__uint_as_float(x_car[2] & 0xFFFF0000u),
                               __uint_as_float(w_car[2] & 0xFFFF0000u), acc[5]);
            acc[5] =
                __fmaf_rn(__uint_as_float(x_car[3] << 16), __uint_as_float(w_car[3] << 16), acc[5]);
            acc[5] = __fmaf_rn(__uint_as_float(x_car[3] & 0xFFFF0000u),
                               __uint_as_float(w_car[3] & 0xFFFF0000u), acc[5]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[13])
                : "r"(x_car[4]), "r"(x_car[5]), "r"(x_car[6]), "r"(x_car[7]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[13] = __fmaf_rn(__uint_as_float(x_car[4] << 16), __uint_as_float(w_car[0] << 16),
                                acc[13]);
            acc[13] = __fmaf_rn(__uint_as_float(x_car[4] & 0xFFFF0000u),
                                __uint_as_float(w_car[0] & 0xFFFF0000u), acc[13]);
            acc[13] = __fmaf_rn(__uint_as_float(x_car[5] << 16), __uint_as_float(w_car[1] << 16),
                                acc[13]);
            acc[13] = __fmaf_rn(__uint_as_float(x_car[5] & 0xFFFF0000u),
                                __uint_as_float(w_car[1] & 0xFFFF0000u), acc[13]);
            acc[13] = __fmaf_rn(__uint_as_float(x_car[6] << 16), __uint_as_float(w_car[2] << 16),
                                acc[13]);
            acc[13] = __fmaf_rn(__uint_as_float(x_car[6] & 0xFFFF0000u),
                                __uint_as_float(w_car[2] & 0xFFFF0000u), acc[13]);
            acc[13] = __fmaf_rn(__uint_as_float(x_car[7] << 16), __uint_as_float(w_car[3] << 16),
                                acc[13]);
            acc[13] = __fmaf_rn(__uint_as_float(x_car[7] & 0xFFFF0000u),
                                __uint_as_float(w_car[3] & 0xFFFF0000u), acc[13]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[21])
                : "r"(x_car[8]), "r"(x_car[9]), "r"(x_car[10]), "r"(x_car[11]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[21] = __fmaf_rn(__uint_as_float(x_car[8] << 16), __uint_as_float(w_car[0] << 16),
                                acc[21]);
            acc[21] = __fmaf_rn(__uint_as_float(x_car[8] & 0xFFFF0000u),
                                __uint_as_float(w_car[0] & 0xFFFF0000u), acc[21]);
            acc[21] = __fmaf_rn(__uint_as_float(x_car[9] << 16), __uint_as_float(w_car[1] << 16),
                                acc[21]);
            acc[21] = __fmaf_rn(__uint_as_float(x_car[9] & 0xFFFF0000u),
                                __uint_as_float(w_car[1] & 0xFFFF0000u), acc[21]);
            acc[21] = __fmaf_rn(__uint_as_float(x_car[10] << 16), __uint_as_float(w_car[2] << 16),
                                acc[21]);
            acc[21] = __fmaf_rn(__uint_as_float(x_car[10] & 0xFFFF0000u),
                                __uint_as_float(w_car[2] & 0xFFFF0000u), acc[21]);
            acc[21] = __fmaf_rn(__uint_as_float(x_car[11] << 16), __uint_as_float(w_car[3] << 16),
                                acc[21]);
            acc[21] = __fmaf_rn(__uint_as_float(x_car[11] & 0xFFFF0000u),
                                __uint_as_float(w_car[3] & 0xFFFF0000u), acc[21]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[29])
                : "r"(x_car[12]), "r"(x_car[13]), "r"(x_car[14]), "r"(x_car[15]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[29] = __fmaf_rn(__uint_as_float(x_car[12] << 16), __uint_as_float(w_car[0] << 16),
                                acc[29]);
            acc[29] = __fmaf_rn(__uint_as_float(x_car[12] & 0xFFFF0000u),
                                __uint_as_float(w_car[0] & 0xFFFF0000u), acc[29]);
            acc[29] = __fmaf_rn(__uint_as_float(x_car[13] << 16), __uint_as_float(w_car[1] << 16),
                                acc[29]);
            acc[29] = __fmaf_rn(__uint_as_float(x_car[13] & 0xFFFF0000u),
                                __uint_as_float(w_car[1] & 0xFFFF0000u), acc[29]);
            acc[29] = __fmaf_rn(__uint_as_float(x_car[14] << 16), __uint_as_float(w_car[2] << 16),
                                acc[29]);
            acc[29] = __fmaf_rn(__uint_as_float(x_car[14] & 0xFFFF0000u),
                                __uint_as_float(w_car[2] & 0xFFFF0000u), acc[29]);
            acc[29] = __fmaf_rn(__uint_as_float(x_car[15] << 16), __uint_as_float(w_car[3] << 16),
                                acc[29]);
            acc[29] = __fmaf_rn(__uint_as_float(x_car[15] & 0xFFFF0000u),
                                __uint_as_float(w_car[3] & 0xFFFF0000u), acc[29]);
#endif
          }
          asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                       : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
                       : "r"(w_ring_addr + (unsigned int)(((stage * 8 + 6) * 1024 + tid_vec) * 2)));
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[6])
                : "r"(x_car[0]), "r"(x_car[1]), "r"(x_car[2]), "r"(x_car[3]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[6] =
                __fmaf_rn(__uint_as_float(x_car[0] << 16), __uint_as_float(w_car[0] << 16), acc[6]);
            acc[6] = __fmaf_rn(__uint_as_float(x_car[0] & 0xFFFF0000u),
                               __uint_as_float(w_car[0] & 0xFFFF0000u), acc[6]);
            acc[6] =
                __fmaf_rn(__uint_as_float(x_car[1] << 16), __uint_as_float(w_car[1] << 16), acc[6]);
            acc[6] = __fmaf_rn(__uint_as_float(x_car[1] & 0xFFFF0000u),
                               __uint_as_float(w_car[1] & 0xFFFF0000u), acc[6]);
            acc[6] =
                __fmaf_rn(__uint_as_float(x_car[2] << 16), __uint_as_float(w_car[2] << 16), acc[6]);
            acc[6] = __fmaf_rn(__uint_as_float(x_car[2] & 0xFFFF0000u),
                               __uint_as_float(w_car[2] & 0xFFFF0000u), acc[6]);
            acc[6] =
                __fmaf_rn(__uint_as_float(x_car[3] << 16), __uint_as_float(w_car[3] << 16), acc[6]);
            acc[6] = __fmaf_rn(__uint_as_float(x_car[3] & 0xFFFF0000u),
                               __uint_as_float(w_car[3] & 0xFFFF0000u), acc[6]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[14])
                : "r"(x_car[4]), "r"(x_car[5]), "r"(x_car[6]), "r"(x_car[7]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[14] = __fmaf_rn(__uint_as_float(x_car[4] << 16), __uint_as_float(w_car[0] << 16),
                                acc[14]);
            acc[14] = __fmaf_rn(__uint_as_float(x_car[4] & 0xFFFF0000u),
                                __uint_as_float(w_car[0] & 0xFFFF0000u), acc[14]);
            acc[14] = __fmaf_rn(__uint_as_float(x_car[5] << 16), __uint_as_float(w_car[1] << 16),
                                acc[14]);
            acc[14] = __fmaf_rn(__uint_as_float(x_car[5] & 0xFFFF0000u),
                                __uint_as_float(w_car[1] & 0xFFFF0000u), acc[14]);
            acc[14] = __fmaf_rn(__uint_as_float(x_car[6] << 16), __uint_as_float(w_car[2] << 16),
                                acc[14]);
            acc[14] = __fmaf_rn(__uint_as_float(x_car[6] & 0xFFFF0000u),
                                __uint_as_float(w_car[2] & 0xFFFF0000u), acc[14]);
            acc[14] = __fmaf_rn(__uint_as_float(x_car[7] << 16), __uint_as_float(w_car[3] << 16),
                                acc[14]);
            acc[14] = __fmaf_rn(__uint_as_float(x_car[7] & 0xFFFF0000u),
                                __uint_as_float(w_car[3] & 0xFFFF0000u), acc[14]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[22])
                : "r"(x_car[8]), "r"(x_car[9]), "r"(x_car[10]), "r"(x_car[11]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[22] = __fmaf_rn(__uint_as_float(x_car[8] << 16), __uint_as_float(w_car[0] << 16),
                                acc[22]);
            acc[22] = __fmaf_rn(__uint_as_float(x_car[8] & 0xFFFF0000u),
                                __uint_as_float(w_car[0] & 0xFFFF0000u), acc[22]);
            acc[22] = __fmaf_rn(__uint_as_float(x_car[9] << 16), __uint_as_float(w_car[1] << 16),
                                acc[22]);
            acc[22] = __fmaf_rn(__uint_as_float(x_car[9] & 0xFFFF0000u),
                                __uint_as_float(w_car[1] & 0xFFFF0000u), acc[22]);
            acc[22] = __fmaf_rn(__uint_as_float(x_car[10] << 16), __uint_as_float(w_car[2] << 16),
                                acc[22]);
            acc[22] = __fmaf_rn(__uint_as_float(x_car[10] & 0xFFFF0000u),
                                __uint_as_float(w_car[2] & 0xFFFF0000u), acc[22]);
            acc[22] = __fmaf_rn(__uint_as_float(x_car[11] << 16), __uint_as_float(w_car[3] << 16),
                                acc[22]);
            acc[22] = __fmaf_rn(__uint_as_float(x_car[11] & 0xFFFF0000u),
                                __uint_as_float(w_car[3] & 0xFFFF0000u), acc[22]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[30])
                : "r"(x_car[12]), "r"(x_car[13]), "r"(x_car[14]), "r"(x_car[15]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[30] = __fmaf_rn(__uint_as_float(x_car[12] << 16), __uint_as_float(w_car[0] << 16),
                                acc[30]);
            acc[30] = __fmaf_rn(__uint_as_float(x_car[12] & 0xFFFF0000u),
                                __uint_as_float(w_car[0] & 0xFFFF0000u), acc[30]);
            acc[30] = __fmaf_rn(__uint_as_float(x_car[13] << 16), __uint_as_float(w_car[1] << 16),
                                acc[30]);
            acc[30] = __fmaf_rn(__uint_as_float(x_car[13] & 0xFFFF0000u),
                                __uint_as_float(w_car[1] & 0xFFFF0000u), acc[30]);
            acc[30] = __fmaf_rn(__uint_as_float(x_car[14] << 16), __uint_as_float(w_car[2] << 16),
                                acc[30]);
            acc[30] = __fmaf_rn(__uint_as_float(x_car[14] & 0xFFFF0000u),
                                __uint_as_float(w_car[2] & 0xFFFF0000u), acc[30]);
            acc[30] = __fmaf_rn(__uint_as_float(x_car[15] << 16), __uint_as_float(w_car[3] << 16),
                                acc[30]);
            acc[30] = __fmaf_rn(__uint_as_float(x_car[15] & 0xFFFF0000u),
                                __uint_as_float(w_car[3] & 0xFFFF0000u), acc[30]);
#endif
          }
          asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                       : "=r"(*reinterpret_cast<uint32_t*>(&w_car[0])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 1])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 2])),
                         "=r"(*reinterpret_cast<uint32_t*>(&w_car[(0) + 3]))
                       : "r"(w_ring_addr + (unsigned int)(((stage * 8 + 7) * 1024 + tid_vec) * 2)));
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[7])
                : "r"(x_car[0]), "r"(x_car[1]), "r"(x_car[2]), "r"(x_car[3]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[7] =
                __fmaf_rn(__uint_as_float(x_car[0] << 16), __uint_as_float(w_car[0] << 16), acc[7]);
            acc[7] = __fmaf_rn(__uint_as_float(x_car[0] & 0xFFFF0000u),
                               __uint_as_float(w_car[0] & 0xFFFF0000u), acc[7]);
            acc[7] =
                __fmaf_rn(__uint_as_float(x_car[1] << 16), __uint_as_float(w_car[1] << 16), acc[7]);
            acc[7] = __fmaf_rn(__uint_as_float(x_car[1] & 0xFFFF0000u),
                               __uint_as_float(w_car[1] & 0xFFFF0000u), acc[7]);
            acc[7] =
                __fmaf_rn(__uint_as_float(x_car[2] << 16), __uint_as_float(w_car[2] << 16), acc[7]);
            acc[7] = __fmaf_rn(__uint_as_float(x_car[2] & 0xFFFF0000u),
                               __uint_as_float(w_car[2] & 0xFFFF0000u), acc[7]);
            acc[7] =
                __fmaf_rn(__uint_as_float(x_car[3] << 16), __uint_as_float(w_car[3] << 16), acc[7]);
            acc[7] = __fmaf_rn(__uint_as_float(x_car[3] & 0xFFFF0000u),
                               __uint_as_float(w_car[3] & 0xFFFF0000u), acc[7]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[15])
                : "r"(x_car[4]), "r"(x_car[5]), "r"(x_car[6]), "r"(x_car[7]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[15] = __fmaf_rn(__uint_as_float(x_car[4] << 16), __uint_as_float(w_car[0] << 16),
                                acc[15]);
            acc[15] = __fmaf_rn(__uint_as_float(x_car[4] & 0xFFFF0000u),
                                __uint_as_float(w_car[0] & 0xFFFF0000u), acc[15]);
            acc[15] = __fmaf_rn(__uint_as_float(x_car[5] << 16), __uint_as_float(w_car[1] << 16),
                                acc[15]);
            acc[15] = __fmaf_rn(__uint_as_float(x_car[5] & 0xFFFF0000u),
                                __uint_as_float(w_car[1] & 0xFFFF0000u), acc[15]);
            acc[15] = __fmaf_rn(__uint_as_float(x_car[6] << 16), __uint_as_float(w_car[2] << 16),
                                acc[15]);
            acc[15] = __fmaf_rn(__uint_as_float(x_car[6] & 0xFFFF0000u),
                                __uint_as_float(w_car[2] & 0xFFFF0000u), acc[15]);
            acc[15] = __fmaf_rn(__uint_as_float(x_car[7] << 16), __uint_as_float(w_car[3] << 16),
                                acc[15]);
            acc[15] = __fmaf_rn(__uint_as_float(x_car[7] & 0xFFFF0000u),
                                __uint_as_float(w_car[3] & 0xFFFF0000u), acc[15]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[23])
                : "r"(x_car[8]), "r"(x_car[9]), "r"(x_car[10]), "r"(x_car[11]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[23] = __fmaf_rn(__uint_as_float(x_car[8] << 16), __uint_as_float(w_car[0] << 16),
                                acc[23]);
            acc[23] = __fmaf_rn(__uint_as_float(x_car[8] & 0xFFFF0000u),
                                __uint_as_float(w_car[0] & 0xFFFF0000u), acc[23]);
            acc[23] = __fmaf_rn(__uint_as_float(x_car[9] << 16), __uint_as_float(w_car[1] << 16),
                                acc[23]);
            acc[23] = __fmaf_rn(__uint_as_float(x_car[9] & 0xFFFF0000u),
                                __uint_as_float(w_car[1] & 0xFFFF0000u), acc[23]);
            acc[23] = __fmaf_rn(__uint_as_float(x_car[10] << 16), __uint_as_float(w_car[2] << 16),
                                acc[23]);
            acc[23] = __fmaf_rn(__uint_as_float(x_car[10] & 0xFFFF0000u),
                                __uint_as_float(w_car[2] & 0xFFFF0000u), acc[23]);
            acc[23] = __fmaf_rn(__uint_as_float(x_car[11] << 16), __uint_as_float(w_car[3] << 16),
                                acc[23]);
            acc[23] = __fmaf_rn(__uint_as_float(x_car[11] & 0xFFFF0000u),
                                __uint_as_float(w_car[3] & 0xFFFF0000u), acc[23]);
#endif
          }
          {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
                "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
                "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
                "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
                "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
                "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
                : "+f"(acc[31])
                : "r"(x_car[12]), "r"(x_car[13]), "r"(x_car[14]), "r"(x_car[15]), "r"(w_car[0]),
                  "r"(w_car[1]), "r"(w_car[2]), "r"(w_car[3]));
#else
            acc[31] = __fmaf_rn(__uint_as_float(x_car[12] << 16), __uint_as_float(w_car[0] << 16),
                                acc[31]);
            acc[31] = __fmaf_rn(__uint_as_float(x_car[12] & 0xFFFF0000u),
                                __uint_as_float(w_car[0] & 0xFFFF0000u), acc[31]);
            acc[31] = __fmaf_rn(__uint_as_float(x_car[13] << 16), __uint_as_float(w_car[1] << 16),
                                acc[31]);
            acc[31] = __fmaf_rn(__uint_as_float(x_car[13] & 0xFFFF0000u),
                                __uint_as_float(w_car[1] & 0xFFFF0000u), acc[31]);
            acc[31] = __fmaf_rn(__uint_as_float(x_car[14] << 16), __uint_as_float(w_car[2] << 16),
                                acc[31]);
            acc[31] = __fmaf_rn(__uint_as_float(x_car[14] & 0xFFFF0000u),
                                __uint_as_float(w_car[2] & 0xFFFF0000u), acc[31]);
            acc[31] = __fmaf_rn(__uint_as_float(x_car[15] << 16), __uint_as_float(w_car[3] << 16),
                                acc[31]);
            acc[31] = __fmaf_rn(__uint_as_float(x_car[15] & 0xFFFF0000u),
                                __uint_as_float(w_car[3] & 0xFFFF0000u), acc[31]);
#endif
          }
          int k_xn = k_base_r + 1024;
          if (k_xn < hidden) {
            int kw_xn = k_xn / 2;
#pragma unroll
            for (int j_2 = 0; j_2 < 4; j_2++) {
              {
                const uint4* _ivptr_1 = reinterpret_cast<const uint4*>(
                    reinterpret_cast<const unsigned int*>(x_raw) +
                    tokens[j_2] * (long long)hidden_words + (long long)kw_xn);
                uint4 _ivld_1;
                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                             : "=r"(_ivld_1.x), "=r"(_ivld_1.y), "=r"(_ivld_1.z), "=r"(_ivld_1.w)
                             : "l"((const void*)(_ivptr_1))
                             : "memory");
                (x_car + j_2 * 4)[0 + 0] = _ivld_1.x;
                (x_car + j_2 * 4)[0 + 1] = _ivld_1.y;
                (x_car + j_2 * 4)[0 + 2] = _ivld_1.z;
                (x_car + j_2 * 4)[0 + 3] = _ivld_1.w;
              }
            }
          }
        }
      }
#pragma unroll
      for (int i = 0; i < 16; i++) {
        float _shfl_xor_0 =
            __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) != 0) ? acc[i] : acc[i + 16]), 16);
        red_a[i] = (((lane_0 & 16) != 0) ? acc[i + 16] : acc[i]) + _shfl_xor_0;
      }
#pragma unroll
      for (int i_1 = 0; i_1 < 8; i_1++) {
        float _shfl_xor_1 =
            __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) != 0) ? red_a[i_1] : red_a[i_1 + 8]), 8);
        red_b[i_1] = (((lane_0 & 8) != 0) ? red_a[i_1 + 8] : red_a[i_1]) + _shfl_xor_1;
      }
#pragma unroll
      for (int i_2 = 0; i_2 < 4; i_2++) {
        float _shfl_xor_2 =
            __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 4) != 0) ? red_b[i_2] : red_b[i_2 + 4]), 4);
        red_c[i_2] = (((lane_0 & 4) != 0) ? red_b[i_2 + 4] : red_b[i_2]) + _shfl_xor_2;
      }
#pragma unroll
      for (int i_3 = 0; i_3 < 2; i_3++) {
        float _shfl_xor_3 =
            __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 2) != 0) ? red_c[i_3] : red_c[i_3 + 2]), 2);
        red_d[i_3] = (((lane_0 & 2) != 0) ? red_c[i_3 + 2] : red_c[i_3]) + _shfl_xor_3;
      }
#pragma unroll
      for (int i_4 = 0; i_4 < 1; i_4++) {
        float _shfl_xor_4 =
            __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 1) != 0) ? red_d[i_4] : red_d[i_4 + 1]), 1);
        red_e[i_4] = (((lane_0 & 1) != 0) ? red_d[i_4 + 1] : red_d[i_4]) + _shfl_xor_4;
      }
#pragma unroll
      for (int i_5 = 0; i_5 < 1; i_5++) {
        warp_partials[(lane_0 + i_5) * 4 + warp] = red_e[i_5];
      }
      __syncthreads();
      if (tid < 32) {
        float owned_accum = 0.0f;
#pragma unroll
        for (int source_warp = 0; source_warp < 4; source_warp++) {
          owned_accum += warp_partials[tid * 4 + source_warp];
        }
        int owner_j = tid / 8;
        int owner_rr = tid % 8;
        if (owner_j < count) {
          int owner_route = (int)reinterpret_cast<const unsigned int*>(
              workspace_raw)[off_sorted_routes + start + owner_j];
          *(reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<__nv_bfloat16*>(shrink_out_raw) +
                                             (owner_route * 32 + rank_base0 + owner_rr)) +
            (0)) = __float2bfloat16_rn(owned_accum);
        }
      }
    }
  }
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

}  // extern "C"

#undef BLACKWELL_INF
#undef NUM_MAIN_STAGES
#undef SMEM_TOTAL
#undef SMEM_WARP_PARTIALS_OFF
#undef SMEM_WARP_PARTIALS_STAGE_BYTES
#undef SMEM_WARP_PARTIALS_STRIDE
#undef SMEM_W_RING_OFF
#undef SMEM_W_RING_STAGE_BYTES
#undef SMEM_W_RING_STRIDE
#undef SMEM_X_RING_OFF
#undef SMEM_X_RING_STAGE_BYTES
#undef SMEM_X_RING_STRIDE
#undef THREADS

// Dynamic shared memory per launch, in bytes.
#define CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_DECODE 221824
#define CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_PREFILL 37120
#define CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_DECODE_PDL 221824
#define CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_PREFILL_PDL 37120
#define CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_PREFILL_S3 55552
#define CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_PREFILL_S3_PDL 55552
#define CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_PREFILL_REMAP 37120
#define CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_PREFILL_REMAP_PDL 37120
#define CAKE_BGMV_MOE_GENERIC_SMEM_EXPAND_T64 128
#define CAKE_BGMV_MOE_GENERIC_SMEM_EXPAND_T128 128
#define CAKE_BGMV_MOE_GENERIC_SMEM_EXPAND_T64_PF 128
#define CAKE_BGMV_MOE_GENERIC_SMEM_EXPAND_T128_PF 128
#define CAKE_BGMV_MOE_GENERIC_SMEM_GROUP_HIST 16384
#define CAKE_BGMV_MOE_GENERIC_SMEM_GROUP_SCAN 384
#define CAKE_BGMV_MOE_GENERIC_SMEM_GROUP_SCATTER 16384
#define CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_GROUPED 512
#define CAKE_BGMV_MOE_GENERIC_SMEM_EXPAND_GROUPED 1152
#define CAKE_BGMV_MOE_GENERIC_SMEM_COMBINE_GROUPED 128
#define CAKE_BGMV_MOE_GENERIC_SMEM_ORDER_BUILD 16640
#define CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_GROUPED_RING 49664
#define CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_GROUPED_RING_MIXED 33280
#define CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_GROUPED_SINGLE 512
#define CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_GROUPED_RING_SINGLE 49664
#define CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_GROUPED_RING_MIXED_SINGLE 33280
