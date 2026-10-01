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
// Bundle: Cake BGMV MoE generic-shape shrink and deterministic expand, f16 rank 16.
// Target: sm_90a, sm_100a, sm_103a (cp.async, shuffles, atomics, FMA only); compile flags: none.
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
template <int N>
struct __align__(128) BlackwellTensorMapPack {
  BlackwellTensorMap maps[N];
};

typedef struct __align__(64) {
  uint64_t opaque[16];
} CUtensorMap;

#include <cuda_bf16.h>

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
#define SMEM_TOTAL 221696
#define THREADS 128

extern "C" {

__global__ __launch_bounds__(128, 1) void kernel_flashinfer_bgmv_moe_shrink_generic_f16_r16_p4_s3(
    uint16_t* __restrict__ shrink_out_raw, uint16_t* __restrict__ x_raw,
    uint16_t* __restrict__ lora_a_raw, long long* __restrict__ sorted_token_ids,
    long long* __restrict__ expert_ids, long long* __restrict__ lora_indices, int num_pairs,
    int num_experts, int num_tokens, unsigned int* __restrict__ route_index_raw, int route_build,
    int hidden, int num_tiles) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
  asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }"
               : "=r"(smem)
               : "l"(smem_raw));
  smem = make_warp_uniform(smem);

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  const int cta_rank = 0;

  // Kernel setup ops
  __half* x_smem = reinterpret_cast<__half*>(smem_raw + 0);
  const int x_smem_addr = smem + 0;
  __half* w_smem = reinterpret_cast<__half*>(smem_raw + 24576);
  const int w_smem_addr = smem + 24576;
  float* warp_partials = reinterpret_cast<float*>(smem_raw + 221184);
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
  for (int tile = 0; tile < 3; tile++) {
    if (tile < num_tiles) {
      int k_base = tile * 1024 + tid * 8;
#pragma unroll
      for (int pp_1 = 0; pp_1 < 4; pp_1++) {
        if (valid[pp_1] != 0) {
          if (k_base < hidden) {
            asm volatile(
                "cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                    x_smem_addr + (unsigned int)((tile * 4 * 1024 + pp_1 * 1024 + tid * 8) * 2)),
                "l"(reinterpret_cast<const __half*>(x_raw) +
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
                           "l"(reinterpret_cast<const __half*>(lora_a_raw) + weight_index));
            }
          }
        }
      }
    }
    asm volatile("cp.async.commit_group;");
  }
  if (route_build != 0) {
    if (blockIdx.y == 0) {
      if (tid == 0) {
        int route_parity = (int)(route_index_raw[0] & 1);
        if (blockIdx.x == 0) {
          *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(route_index_raw) + 1) +
            (0)) = (unsigned int)route_parity;
        }
#pragma unroll
        for (int pp_2 = 0; pp_2 < 4; pp_2++) {
          if (valid[pp_2] != 0) {
            int route_token = (int)tokens[pp_2];
            unsigned int route_base =
                route_index_raw[4 + num_tokens + route_parity * num_tokens + route_token];
            unsigned int _atomic_old_0 =
                atomicAdd(&reinterpret_cast<unsigned int*>(route_index_raw)[4 + route_token], 1);
            unsigned int route_old = _atomic_old_0;
            unsigned int route_slot = route_old - route_base;
            if (route_slot < 16) {
              *(reinterpret_cast<unsigned int*>(
                    reinterpret_cast<unsigned int*>(route_index_raw) +
                    (4 + 3 * num_tokens + route_token * 16 + (int)route_slot)) +
                (0)) = (unsigned int)(pair_block * 4 + pp_2);
            }
          }
        }
      }
    }
  }
  float owned_accum = 0.0f;
  unsigned int x_carriers[4];
  unsigned int w_carriers[4];
  float x_values[8];
  float w_values[8];
#pragma unroll 1
  for (int tile_1 = 0; tile_1 < num_tiles; tile_1++) {
    asm volatile("cp.async.wait_group 2;");
    __syncthreads();
    int stage = tile_1 % 3;
    int k_base_1 = tile_1 * 1024 + tid * 8;
#pragma unroll
    for (int pp_3 = 0; pp_3 < 4; pp_3++) {
      int x_thread_base = stage * 4 * 1024 + pp_3 * 1024 + tid * 8;
      if (valid[pp_3] != 0) {
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
                  ".reg .b16 h_lo, h_hi;\n\t"
                  ".reg .b32 f_lo, f_hi;\n\t"
                  "mov.b32 {h_lo, h_hi}, %1;\n\t"
                  "cvt.f32.f16 f_lo, h_lo;\n\t"
                  "cvt.f32.f16 f_hi, h_hi;\n\t"
                  "mov.b64 %0, {f_lo, f_hi};\n\t"
                  "}\n"
                  : "=l"(*reinterpret_cast<unsigned long long*>(&x_values[_pair * 2]))
                  : "r"(x_carriers[_pair]));
            }
          }
        }
      }
#pragma unroll
      for (int rr_1 = 0; rr_1 < 8; rr_1++) {
        float partial = 0.0f;
        if (valid[pp_3] != 0) {
          if (k_base_1 < hidden) {
            int w_thread_base = stage * 4 * 8 * 1024 + pp_3 * 8 * 1024 + rr_1 * 1024 + tid * 8;
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
                    ".reg .b16 h_lo, h_hi;\n\t"
                    ".reg .b32 f_lo, f_hi;\n\t"
                    "mov.b32 {h_lo, h_hi}, %1;\n\t"
                    "cvt.f32.f16 f_lo, h_lo;\n\t"
                    "cvt.f32.f16 f_hi, h_hi;\n\t"
                    "mov.b64 %0, {f_lo, f_hi};\n\t"
                    "}\n"
                    : "=l"(*reinterpret_cast<unsigned long long*>(&w_values[_pair * 2]))
                    : "r"(w_carriers[_pair]));
              }
            }
#pragma unroll
            for (int element = 0; element < 8; element++) {
              float _fma_0 = __fmaf_rn(x_values[element], w_values[element], partial);
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
          warp_partials[(pp_3 * 8 + rr_1) * 4 + warp] = partial;
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
    int refill_tile = tile_1 + 3;
    if (refill_tile < num_tiles) {
      int refill_k = refill_tile * 1024 + tid * 8;
#pragma unroll
      for (int pp_4 = 0; pp_4 < 4; pp_4++) {
        if (valid[pp_4] != 0) {
          if (refill_k < hidden) {
            asm volatile(
                "cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                    x_smem_addr + (unsigned int)((stage * 4 * 1024 + pp_4 * 1024 + tid * 8) * 2)),
                "l"(reinterpret_cast<const __half*>(x_raw) +
                    (tokens[pp_4] * (long long)hidden + (long long)refill_k)));
#pragma unroll
            for (int rr_2 = 0; rr_2 < 8; rr_2++) {
              int rank_row_1 = rank_base + rr_2;
              long long weight_index_1 =
                  ((loras[pp_4] * (long long)num_experts + experts[pp_4]) * 16 +
                   (long long)rank_row_1) *
                      (long long)hidden +
                  (long long)refill_k;
              asm volatile(
                  "cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                      w_smem_addr + (unsigned int)((stage * 4 * 8 * 1024 + pp_4 * 8 * 1024 +
                                                    rr_2 * 1024 + tid * 8) *
                                                   2)),
                  "l"(reinterpret_cast<const __half*>(lora_a_raw) + weight_index_1));
            }
          }
        }
      }
    }
    asm volatile("cp.async.commit_group;");
  }
  if (warp == 0) {
    if (lane < 32) {
      int owner_pp = lane / 8;
      int owner_rr = lane % 8;
      int pair_1 = pair_block * 4 + owner_pp;
      if (pair_1 < num_pairs) {
        *(reinterpret_cast<__half*>(reinterpret_cast<__half*>(shrink_out_raw) +
                                    (pair_1 * 16 + rank_base + owner_rr)) +
          (0)) = __float2half_rn(owned_accum);
      }
    }
  }
}

}  // extern "C"

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

__global__ __launch_bounds__(128, 1) void kernel_flashinfer_bgmv_moe_shrink_generic_f16_r16_p1_s2(
    uint16_t* __restrict__ shrink_out_raw, uint16_t* __restrict__ x_raw,
    uint16_t* __restrict__ lora_a_raw, long long* __restrict__ sorted_token_ids,
    long long* __restrict__ expert_ids, long long* __restrict__ lora_indices, int num_pairs,
    int num_experts, int num_tokens, unsigned int* __restrict__ route_index_raw, int route_build,
    int hidden, int num_tiles) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
  asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }"
               : "=r"(smem)
               : "l"(smem_raw));
  smem = make_warp_uniform(smem);

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  const int cta_rank = 0;

  // Kernel setup ops
  __half* x_smem = reinterpret_cast<__half*>(smem_raw + 0);
  const int x_smem_addr = smem + 0;
  __half* w_smem = reinterpret_cast<__half*>(smem_raw + 4096);
  const int w_smem_addr = smem + 4096;
  float* warp_partials = reinterpret_cast<float*>(smem_raw + 36864);
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
  for (int tile = 0; tile < 2; tile++) {
    if (tile < num_tiles) {
      int k_base = tile * 1024 + tid * 8;
#pragma unroll
      for (int pp_1 = 0; pp_1 < 1; pp_1++) {
        if (valid[pp_1] != 0) {
          if (k_base < hidden) {
            asm volatile(
                "cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                    x_smem_addr + (unsigned int)((tile * 1024 + pp_1 * 1024 + tid * 8) * 2)),
                "l"(reinterpret_cast<const __half*>(x_raw) +
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
                           "l"(reinterpret_cast<const __half*>(lora_a_raw) + weight_index));
            }
          }
        }
      }
    }
    asm volatile("cp.async.commit_group;");
  }
  if (route_build != 0) {
    if (blockIdx.y == 0) {
      if (tid == 0) {
        int route_parity = (int)(route_index_raw[0] & 1);
        if (blockIdx.x == 0) {
          *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(route_index_raw) + 1) +
            (0)) = (unsigned int)route_parity;
        }
#pragma unroll
        for (int pp_2 = 0; pp_2 < 1; pp_2++) {
          if (valid[pp_2] != 0) {
            int route_token = (int)tokens[pp_2];
            unsigned int route_base =
                route_index_raw[4 + num_tokens + route_parity * num_tokens + route_token];
            unsigned int _atomic_old_0 =
                atomicAdd(&reinterpret_cast<unsigned int*>(route_index_raw)[4 + route_token], 1);
            unsigned int route_old = _atomic_old_0;
            unsigned int route_slot = route_old - route_base;
            if (route_slot < 16) {
              *(reinterpret_cast<unsigned int*>(
                    reinterpret_cast<unsigned int*>(route_index_raw) +
                    (4 + 3 * num_tokens + route_token * 16 + (int)route_slot)) +
                (0)) = (unsigned int)(pair_block + pp_2);
            }
          }
        }
      }
    }
  }
  float owned_accum = 0.0f;
  unsigned int x_carriers[4];
  unsigned int w_carriers[4];
  float x_values[8];
  float w_values[8];
#pragma unroll 1
  for (int tile_1 = 0; tile_1 < num_tiles; tile_1++) {
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();
    int stage = tile_1 % 2;
    int k_base_1 = tile_1 * 1024 + tid * 8;
#pragma unroll
    for (int pp_3 = 0; pp_3 < 1; pp_3++) {
      int x_thread_base = stage * 1024 + pp_3 * 1024 + tid * 8;
      if (valid[pp_3] != 0) {
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
                  ".reg .b16 h_lo, h_hi;\n\t"
                  ".reg .b32 f_lo, f_hi;\n\t"
                  "mov.b32 {h_lo, h_hi}, %1;\n\t"
                  "cvt.f32.f16 f_lo, h_lo;\n\t"
                  "cvt.f32.f16 f_hi, h_hi;\n\t"
                  "mov.b64 %0, {f_lo, f_hi};\n\t"
                  "}\n"
                  : "=l"(*reinterpret_cast<unsigned long long*>(&x_values[_pair * 2]))
                  : "r"(x_carriers[_pair]));
            }
          }
        }
      }
#pragma unroll
      for (int rr_1 = 0; rr_1 < 8; rr_1++) {
        float partial = 0.0f;
        if (valid[pp_3] != 0) {
          if (k_base_1 < hidden) {
            int w_thread_base = stage * 8 * 1024 + pp_3 * 8 * 1024 + rr_1 * 1024 + tid * 8;
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
                    ".reg .b16 h_lo, h_hi;\n\t"
                    ".reg .b32 f_lo, f_hi;\n\t"
                    "mov.b32 {h_lo, h_hi}, %1;\n\t"
                    "cvt.f32.f16 f_lo, h_lo;\n\t"
                    "cvt.f32.f16 f_hi, h_hi;\n\t"
                    "mov.b64 %0, {f_lo, f_hi};\n\t"
                    "}\n"
                    : "=l"(*reinterpret_cast<unsigned long long*>(&w_values[_pair * 2]))
                    : "r"(w_carriers[_pair]));
              }
            }
#pragma unroll
            for (int element = 0; element < 8; element++) {
              float _fma_0 = __fmaf_rn(x_values[element], w_values[element], partial);
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
          warp_partials[(pp_3 * 8 + rr_1) * 4 + warp] = partial;
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
    int refill_tile = tile_1 + 2;
    if (refill_tile < num_tiles) {
      int refill_k = refill_tile * 1024 + tid * 8;
#pragma unroll
      for (int pp_4 = 0; pp_4 < 1; pp_4++) {
        if (valid[pp_4] != 0) {
          if (refill_k < hidden) {
            asm volatile(
                "cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                    x_smem_addr + (unsigned int)((stage * 1024 + pp_4 * 1024 + tid * 8) * 2)),
                "l"(reinterpret_cast<const __half*>(x_raw) +
                    (tokens[pp_4] * (long long)hidden + (long long)refill_k)));
#pragma unroll
            for (int rr_2 = 0; rr_2 < 8; rr_2++) {
              int rank_row_1 = rank_base + rr_2;
              long long weight_index_1 =
                  ((loras[pp_4] * (long long)num_experts + experts[pp_4]) * 16 +
                   (long long)rank_row_1) *
                      (long long)hidden +
                  (long long)refill_k;
              asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                               w_smem_addr + (unsigned int)((stage * 8 * 1024 + pp_4 * 8 * 1024 +
                                                             rr_2 * 1024 + tid * 8) *
                                                            2)),
                           "l"(reinterpret_cast<const __half*>(lora_a_raw) + weight_index_1));
            }
          }
        }
      }
    }
    asm volatile("cp.async.commit_group;");
  }
  if (warp == 0) {
    if (lane < 8) {
      int owner_pp = lane / 8;
      int owner_rr = lane % 8;
      int pair_1 = pair_block + owner_pp;
      if (pair_1 < num_pairs) {
        *(reinterpret_cast<__half*>(reinterpret_cast<__half*>(shrink_out_raw) +
                                    (pair_1 * 16 + rank_base + owner_rr)) +
          (0)) = __float2half_rn(owned_accum);
      }
    }
  }
}

}  // extern "C"

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

#define BLACKWELL_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SHRINK_STAGE_OFF 0
#define SMEM_SHRINK_STAGE_STAGE_BYTES 64
#define SMEM_SHRINK_STAGE_STRIDE 64
#define SMEM_ROUTE_LIST_OFF 64
#define SMEM_ROUTE_LIST_STAGE_BYTES 64
#define SMEM_ROUTE_LIST_STRIDE 64
#define SMEM_TOTAL 128
#define THREADS 64

extern "C" {

__global__
__launch_bounds__(64, 1) void kernel_flashinfer_bgmv_moe_expand_generic_token_t64_f16_r16(
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
  asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }"
               : "=r"(smem)
               : "l"(smem_raw));
  smem = make_warp_uniform(smem);

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  const int cta_rank = 0;

  // Kernel setup ops
  __half* shrink_stage = reinterpret_cast<__half*>(smem_raw + 0);
  const int shrink_stage_addr = smem + 0;
  int* route_list = reinterpret_cast<int*>(smem_raw + 64);
  const int route_list_addr = smem + 64;

  // === Task calls (dependency order) ===
  int token = blockIdx.x;
  int output_col = blockIdx.y * 64 + tid;
  unsigned int activation_carriers[4];
  float activation_values[8];
  if (token < num_tokens) {
    if (route_advance != 0) {
      if (blockIdx.y == 0) {
        if (tid == 0) {
          int advance_parity = (int)route_index_raw[1];
          *(reinterpret_cast<unsigned int*>(
                reinterpret_cast<unsigned int*>(route_index_raw) +
                (4 + num_tokens + (1 - advance_parity) * num_tokens + token)) +
            (0)) = route_index_raw[4 + token];
          if (token == 0) {
            *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(route_index_raw)) +
              (0)) = route_index_raw[0] + 1;
          }
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
      float total = 0.0f;
      if (contiguous != 0) {
        if (tid < 4) {
          int stage_route = tid / 2;
          int stage_rank_block = tid % 2;
          asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                           shrink_stage_addr +
                           (unsigned int)((stage_route * 16 + stage_rank_block * 8) * 2)),
                       "l"(reinterpret_cast<const __half*>(shrink_raw) +
                           ((pair_base + stage_route) * 16 + stage_rank_block * 8)));
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        if (output_col < hidden) {
#pragma unroll
          for (int route = 0; route < 2; route++) {
            int pair = pair_base + route;
            long long expert = expert_ids[pair];
            float route_partial = 0.0f;
#pragma unroll
            for (int rank_block = 0; rank_block < 2; rank_block++) {
              int rank_col = rank_block * 8;
              asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                           : "=r"(*reinterpret_cast<uint32_t*>(&activation_carriers[0])),
                             "=r"(*reinterpret_cast<uint32_t*>(&activation_carriers[(0) + 1])),
                             "=r"(*reinterpret_cast<uint32_t*>(&activation_carriers[(0) + 2])),
                             "=r"(*reinterpret_cast<uint32_t*>(&activation_carriers[(0) + 3]))
                           : "r"(shrink_stage_addr + (unsigned int)((route * 16 + rank_col) * 2)));
              {
#pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                  asm volatile(
                      "{\n\t"
                      ".reg .b16 h_lo, h_hi;\n\t"
                      ".reg .b32 f_lo, f_hi;\n\t"
                      "mov.b32 {h_lo, h_hi}, %1;\n\t"
                      "cvt.f32.f16 f_lo, h_lo;\n\t"
                      "cvt.f32.f16 f_hi, h_hi;\n\t"
                      "mov.b64 %0, {f_lo, f_hi};\n\t"
                      "}\n"
                      : "=l"(*reinterpret_cast<unsigned long long*>(&activation_values[_pair * 2]))
                      : "r"(activation_carriers[_pair]));
                }
              }
              long long weight_index =
                  ((lora_id * (long long)num_experts + expert) * (long long)hidden +
                   (long long)output_col) *
                      16 +
                  (long long)rank_col;
              float _vec_load_0[8];
              {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(
                    reinterpret_cast<const __half*>(lora_b_raw) + weight_index + 0);
                uint4 _vld_0[1];
#pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                  _vld_0[_blk] = _vptr_0[_blk];
                  uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
#pragma unroll
                  for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 h_lo, h_hi;\n\t"
                        ".reg .b32 f_lo, f_hi;\n\t"
                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(
                            &_vec_load_0[0 + _blk * 8 + _pair * 2]))
                        : "r"(_vpairs_0[_pair]));
                  }
                }
              }
#pragma unroll
              for (int element = 0; element < 8; element++) {
                float _fma_0 =
                    __fmaf_rn(activation_values[element], _vec_load_0[element], route_partial);
                route_partial = _fma_0;
              }
            }
            float _fma_1 = __fmaf_rn(route_partial, topk_weights[pair], total);
            total = _fma_1;
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
            route_count = indexed_count;
            int table_base = 4 + 3 * num_tokens + token * 16;
            if (indexed_count > tid) {
              int own_pair = (int)route_index_raw[table_base + tid];
              int own_order = 0;
#pragma unroll 1
              for (int probe = 0; probe < indexed_count; probe++) {
                if (own_pair > (int)route_index_raw[table_base + probe]) {
                  own_order += 1;
                }
              }
              route_list[own_order] = own_pair;
            }
            __syncthreads();
          }
        }
        if (output_col < hidden) {
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
              float route_partial_1 = 0.0f;
#pragma unroll
              for (int rank_block_1 = 0; rank_block_1 < 2; rank_block_1++) {
                int rank_col_1 = rank_block_1 * 8;
                float _vec_load_1[8];
                {
                  const uint4* _vptr_1 = reinterpret_cast<const uint4*>(
                      reinterpret_cast<const __half*>(shrink_raw) + (pair_1 * 16 + rank_col_1) + 0);
                  uint4 _vld_1[1];
#pragma unroll
                  for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_1[_blk] = _vptr_1[_blk];
                    uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
#pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                      asm volatile(
                          "{\n\t"
                          ".reg .b16 h_lo, h_hi;\n\t"
                          ".reg .b32 f_lo, f_hi;\n\t"
                          "mov.b32 {h_lo, h_hi}, %1;\n\t"
                          "cvt.f32.f16 f_lo, h_lo;\n\t"
                          "cvt.f32.f16 f_hi, h_hi;\n\t"
                          "mov.b64 %0, {f_lo, f_hi};\n\t"
                          "}\n"
                          : "=l"(*reinterpret_cast<unsigned long long*>(
                              &_vec_load_1[0 + _blk * 8 + _pair * 2]))
                          : "r"(_vpairs_1[_pair]));
                    }
                  }
                }
                long long weight_index_1 =
                    ((lora_id * (long long)num_experts + expert_1) * (long long)hidden +
                     (long long)output_col) *
                        16 +
                    (long long)rank_col_1;
                float _vec_load_2[8];
                {
                  const uint4* _vptr_2 = reinterpret_cast<const uint4*>(
                      reinterpret_cast<const __half*>(lora_b_raw) + weight_index_1 + 0);
                  uint4 _vld_2[1];
#pragma unroll
                  for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_2[_blk] = _vptr_2[_blk];
                    uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
#pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                      asm volatile(
                          "{\n\t"
                          ".reg .b16 h_lo, h_hi;\n\t"
                          ".reg .b32 f_lo, f_hi;\n\t"
                          "mov.b32 {h_lo, h_hi}, %1;\n\t"
                          "cvt.f32.f16 f_lo, h_lo;\n\t"
                          "cvt.f32.f16 f_hi, h_hi;\n\t"
                          "mov.b64 %0, {f_lo, f_hi};\n\t"
                          "}\n"
                          : "=l"(*reinterpret_cast<unsigned long long*>(
                              &_vec_load_2[0 + _blk * 8 + _pair * 2]))
                          : "r"(_vpairs_2[_pair]));
                    }
                  }
                }
#pragma unroll
                for (int element_1 = 0; element_1 < 8; element_1++) {
                  float _fma_2 =
                      __fmaf_rn(_vec_load_1[element_1], _vec_load_2[element_1], route_partial_1);
                  route_partial_1 = _fma_2;
                }
              }
              float _fma_3 = __fmaf_rn(route_partial_1, topk_weights[pair_1], total);
              total = _fma_3;
            }
          }
        }
      }
      if (output_col < hidden) {
        *(reinterpret_cast<float*>(y_accum + (token * output_stride + output_offset + output_col)) +
          (0)) = total;
      }
    } else if (output_col < hidden) {
      *(reinterpret_cast<float*>(y_accum + (token * output_stride + output_offset + output_col)) +
        (0)) = 0.0f;
    }
  }
}

}  // extern "C"

#undef BLACKWELL_INF
#undef NUM_MAIN_STAGES
#undef SMEM_ROUTE_LIST_OFF
#undef SMEM_ROUTE_LIST_STAGE_BYTES
#undef SMEM_ROUTE_LIST_STRIDE
#undef SMEM_SHRINK_STAGE_OFF
#undef SMEM_SHRINK_STAGE_STAGE_BYTES
#undef SMEM_SHRINK_STAGE_STRIDE
#undef SMEM_TOTAL
#undef THREADS

#define BLACKWELL_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SHRINK_STAGE_OFF 0
#define SMEM_SHRINK_STAGE_STAGE_BYTES 64
#define SMEM_SHRINK_STAGE_STRIDE 64
#define SMEM_ROUTE_LIST_OFF 64
#define SMEM_ROUTE_LIST_STAGE_BYTES 64
#define SMEM_ROUTE_LIST_STRIDE 64
#define SMEM_TOTAL 128
#define THREADS 128

extern "C" {

__global__
__launch_bounds__(128, 1) void kernel_flashinfer_bgmv_moe_expand_generic_token_t128_f16_r16(
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
  asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }"
               : "=r"(smem)
               : "l"(smem_raw));
  smem = make_warp_uniform(smem);

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  const int cta_rank = 0;

  // Kernel setup ops
  __half* shrink_stage = reinterpret_cast<__half*>(smem_raw + 0);
  const int shrink_stage_addr = smem + 0;
  int* route_list = reinterpret_cast<int*>(smem_raw + 64);
  const int route_list_addr = smem + 64;

  // === Task calls (dependency order) ===
  int token = blockIdx.x;
  int output_col = blockIdx.y * 128 + tid;
  unsigned int activation_carriers[4];
  float activation_values[8];
  if (token < num_tokens) {
    if (route_advance != 0) {
      if (blockIdx.y == 0) {
        if (tid == 0) {
          int advance_parity = (int)route_index_raw[1];
          *(reinterpret_cast<unsigned int*>(
                reinterpret_cast<unsigned int*>(route_index_raw) +
                (4 + num_tokens + (1 - advance_parity) * num_tokens + token)) +
            (0)) = route_index_raw[4 + token];
          if (token == 0) {
            *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(route_index_raw)) +
              (0)) = route_index_raw[0] + 1;
          }
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
      float total = 0.0f;
      if (contiguous != 0) {
        if (tid < 4) {
          int stage_route = tid / 2;
          int stage_rank_block = tid % 2;
          asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(
                           shrink_stage_addr +
                           (unsigned int)((stage_route * 16 + stage_rank_block * 8) * 2)),
                       "l"(reinterpret_cast<const __half*>(shrink_raw) +
                           ((pair_base + stage_route) * 16 + stage_rank_block * 8)));
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        if (output_col < hidden) {
#pragma unroll
          for (int route = 0; route < 2; route++) {
            int pair = pair_base + route;
            long long expert = expert_ids[pair];
            float route_partial = 0.0f;
#pragma unroll
            for (int rank_block = 0; rank_block < 2; rank_block++) {
              int rank_col = rank_block * 8;
              asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                           : "=r"(*reinterpret_cast<uint32_t*>(&activation_carriers[0])),
                             "=r"(*reinterpret_cast<uint32_t*>(&activation_carriers[(0) + 1])),
                             "=r"(*reinterpret_cast<uint32_t*>(&activation_carriers[(0) + 2])),
                             "=r"(*reinterpret_cast<uint32_t*>(&activation_carriers[(0) + 3]))
                           : "r"(shrink_stage_addr + (unsigned int)((route * 16 + rank_col) * 2)));
              {
#pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                  asm volatile(
                      "{\n\t"
                      ".reg .b16 h_lo, h_hi;\n\t"
                      ".reg .b32 f_lo, f_hi;\n\t"
                      "mov.b32 {h_lo, h_hi}, %1;\n\t"
                      "cvt.f32.f16 f_lo, h_lo;\n\t"
                      "cvt.f32.f16 f_hi, h_hi;\n\t"
                      "mov.b64 %0, {f_lo, f_hi};\n\t"
                      "}\n"
                      : "=l"(*reinterpret_cast<unsigned long long*>(&activation_values[_pair * 2]))
                      : "r"(activation_carriers[_pair]));
                }
              }
              long long weight_index =
                  ((lora_id * (long long)num_experts + expert) * (long long)hidden +
                   (long long)output_col) *
                      16 +
                  (long long)rank_col;
              float _vec_load_0[8];
              {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(
                    reinterpret_cast<const __half*>(lora_b_raw) + weight_index + 0);
                uint4 _vld_0[1];
#pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                  _vld_0[_blk] = _vptr_0[_blk];
                  uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
#pragma unroll
                  for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 h_lo, h_hi;\n\t"
                        ".reg .b32 f_lo, f_hi;\n\t"
                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(
                            &_vec_load_0[0 + _blk * 8 + _pair * 2]))
                        : "r"(_vpairs_0[_pair]));
                  }
                }
              }
#pragma unroll
              for (int element = 0; element < 8; element++) {
                float _fma_0 =
                    __fmaf_rn(activation_values[element], _vec_load_0[element], route_partial);
                route_partial = _fma_0;
              }
            }
            float _fma_1 = __fmaf_rn(route_partial, topk_weights[pair], total);
            total = _fma_1;
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
            route_count = indexed_count;
            int table_base = 4 + 3 * num_tokens + token * 16;
            if (indexed_count > tid) {
              int own_pair = (int)route_index_raw[table_base + tid];
              int own_order = 0;
#pragma unroll 1
              for (int probe = 0; probe < indexed_count; probe++) {
                if (own_pair > (int)route_index_raw[table_base + probe]) {
                  own_order += 1;
                }
              }
              route_list[own_order] = own_pair;
            }
            __syncthreads();
          }
        }
        if (output_col < hidden) {
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
              float route_partial_1 = 0.0f;
#pragma unroll
              for (int rank_block_1 = 0; rank_block_1 < 2; rank_block_1++) {
                int rank_col_1 = rank_block_1 * 8;
                float _vec_load_1[8];
                {
                  const uint4* _vptr_1 = reinterpret_cast<const uint4*>(
                      reinterpret_cast<const __half*>(shrink_raw) + (pair_1 * 16 + rank_col_1) + 0);
                  uint4 _vld_1[1];
#pragma unroll
                  for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_1[_blk] = _vptr_1[_blk];
                    uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
#pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                      asm volatile(
                          "{\n\t"
                          ".reg .b16 h_lo, h_hi;\n\t"
                          ".reg .b32 f_lo, f_hi;\n\t"
                          "mov.b32 {h_lo, h_hi}, %1;\n\t"
                          "cvt.f32.f16 f_lo, h_lo;\n\t"
                          "cvt.f32.f16 f_hi, h_hi;\n\t"
                          "mov.b64 %0, {f_lo, f_hi};\n\t"
                          "}\n"
                          : "=l"(*reinterpret_cast<unsigned long long*>(
                              &_vec_load_1[0 + _blk * 8 + _pair * 2]))
                          : "r"(_vpairs_1[_pair]));
                    }
                  }
                }
                long long weight_index_1 =
                    ((lora_id * (long long)num_experts + expert_1) * (long long)hidden +
                     (long long)output_col) *
                        16 +
                    (long long)rank_col_1;
                float _vec_load_2[8];
                {
                  const uint4* _vptr_2 = reinterpret_cast<const uint4*>(
                      reinterpret_cast<const __half*>(lora_b_raw) + weight_index_1 + 0);
                  uint4 _vld_2[1];
#pragma unroll
                  for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_2[_blk] = _vptr_2[_blk];
                    uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
#pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                      asm volatile(
                          "{\n\t"
                          ".reg .b16 h_lo, h_hi;\n\t"
                          ".reg .b32 f_lo, f_hi;\n\t"
                          "mov.b32 {h_lo, h_hi}, %1;\n\t"
                          "cvt.f32.f16 f_lo, h_lo;\n\t"
                          "cvt.f32.f16 f_hi, h_hi;\n\t"
                          "mov.b64 %0, {f_lo, f_hi};\n\t"
                          "}\n"
                          : "=l"(*reinterpret_cast<unsigned long long*>(
                              &_vec_load_2[0 + _blk * 8 + _pair * 2]))
                          : "r"(_vpairs_2[_pair]));
                    }
                  }
                }
#pragma unroll
                for (int element_1 = 0; element_1 < 8; element_1++) {
                  float _fma_2 =
                      __fmaf_rn(_vec_load_1[element_1], _vec_load_2[element_1], route_partial_1);
                  route_partial_1 = _fma_2;
                }
              }
              float _fma_3 = __fmaf_rn(route_partial_1, topk_weights[pair_1], total);
              total = _fma_3;
            }
          }
        }
      }
      if (output_col < hidden) {
        *(reinterpret_cast<float*>(y_accum + (token * output_stride + output_offset + output_col)) +
          (0)) = total;
      }
    } else if (output_col < hidden) {
      *(reinterpret_cast<float*>(y_accum + (token * output_stride + output_offset + output_col)) +
        (0)) = 0.0f;
    }
  }
}

}  // extern "C"

#undef BLACKWELL_INF
#undef NUM_MAIN_STAGES
#undef SMEM_ROUTE_LIST_OFF
#undef SMEM_ROUTE_LIST_STAGE_BYTES
#undef SMEM_ROUTE_LIST_STRIDE
#undef SMEM_SHRINK_STAGE_OFF
#undef SMEM_SHRINK_STAGE_STAGE_BYTES
#undef SMEM_SHRINK_STAGE_STRIDE
#undef SMEM_TOTAL
#undef THREADS

// Dynamic shared memory per launch, in bytes.
#define CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_DECODE 221696
#define CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_PREFILL 36992
#define CAKE_BGMV_MOE_GENERIC_SMEM_EXPAND_T64 128
#define CAKE_BGMV_MOE_GENERIC_SMEM_EXPAND_T128 128
