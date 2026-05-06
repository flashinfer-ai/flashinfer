#pragma once

/*
 * Multi-LoRA MoE BGMV CUDA Kernel Implementation.
 *
 * Two kernels:
 *   1. Shrink: y[slice, pair, rank] += x[token] @ lora_a[expert, lora_id]
 *      - Compute-bound, uses async pipeline, RANK_TILE tiling, multi-pair blocking
 *   2. Expand: y[token, feat] += shrink_out[pair, rank] @ lora_b[expert, lora_id] * topk_weight
 *      - Memory-bound, uses warp-level reduction
 *
 * Grid (shrink): (ceil(num_pairs/PPB), ceil(feat_out/RANK_TILE), num_slices)
 * Grid (expand): (num_pairs, feat_out/(ty*tz), num_slices)
 *
 * Copyright (c) 2025 by FlashInfer team.
 * Licensed under the Apache License, Version 2.0.
 */

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cuda/pipeline>

// Get the current CUDA stream. In TVM-FFI context, the stream is set by the
// framework before kernel dispatch. We use the default stream (0) which maps
// to the current stream in per-thread default stream mode.
#define BGMV_MOE_GET_STREAM() 0

#include <flashinfer/vec_dtypes.cuh>

#include "kernel_config.h"

using namespace flashinfer;

namespace cg = cooperative_groups;

// ============================================================
// MoE BGMV Shrink Sliced Kernel
//
// Optimizations:
//   1. RANK_TILE tiling — reuse X tile across RANK_TILE weight rows
//   2. Multi-pair — PPB pairs per block (PPB=4 decode, PPB=1 prefill)
//   3. Deep pipeline — NUM_STAGES async pipeline stages (3 decode, 2 prefill)
//
// Uses dynamic shared memory so that large configurations compile for all archs.
// The host wrapper calls cudaFuncSetAttribute on sm_80+ to raise the limit.
// ============================================================
template <int feat_in, int feat_out, int RANK_TILE, int PAIRS_PER_BLOCK, int NUM_STAGES,
          size_t vec_size, size_t X_copy_size, size_t W_copy_size, int tx, int ty, typename in_T,
          typename out_T, typename W_T>
__global__ void moe_bgmv_shrink_sliced_kernel(
    out_T* __restrict__ Y, const in_T* __restrict__ X, W_T** __restrict__ w_ptr,
    const int64_t* __restrict__ sorted_token_ids, const int64_t* __restrict__ expert_ids,
    const int64_t* __restrict__ lora_indices, int64_t num_pairs, int64_t num_experts,
    int64_t num_tokens, int64_t lora_stride, float scale) {
  const int slice_id = blockIdx.z;
  const int pair_block_idx = blockIdx.x;
  const int rank_tile_idx = blockIdx.y;
  const int j0 = rank_tile_idx * RANK_TILE;
  const int p0 = pair_block_idx * PAIRS_PER_BLOCK;

  auto block = cg::this_thread_block();
  constexpr size_t tile_size = tx * ty * vec_size;
  constexpr size_t num_tiles = (feat_in + tile_size - 1) / tile_size;

  // Per-pair metadata
  const in_T* X_tok[PAIRS_PER_BLOCK];
  const W_T* W_base[PAIRS_PER_BLOCK];
  bool pair_valid[PAIRS_PER_BLOCK];

#pragma unroll
  for (int pp = 0; pp < PAIRS_PER_BLOCK; ++pp) {
    const int pair_idx = p0 + pp;
    if (pair_idx < num_pairs) {
      const int64_t token_idx = sorted_token_ids[pair_idx];
      if (token_idx >= 0 && token_idx < num_tokens) {
        const int64_t eid = expert_ids[pair_idx];
        const int64_t lid = lora_indices[token_idx];
        if (lid >= 0) {
          X_tok[pp] = X + token_idx * feat_in;
          W_base[pp] = w_ptr[slice_id * num_experts + eid] + lid * lora_stride + j0 * feat_in;
          pair_valid[pp] = true;
          continue;
        }
      }
    }
    X_tok[pp] = nullptr;
    W_base[pp] = nullptr;
    pair_valid[pp] = false;
  }

  bool any_valid = false;
#pragma unroll
  for (int pp = 0; pp < PAIRS_PER_BLOCK; ++pp) any_valid |= pair_valid[pp];
  if (!any_valid) return;

  // Dynamic shared memory layout
  extern __shared__ char smem[];
  constexpr size_t x_elems = NUM_STAGES * PAIRS_PER_BLOCK * tile_size;
  constexpr size_t w_elems = NUM_STAGES * PAIRS_PER_BLOCK * RANK_TILE * tile_size;
  in_T* X_shared = reinterpret_cast<in_T*>(smem);
  W_T* W_shared = reinterpret_cast<W_T*>(smem + x_elems * sizeof(in_T));
  float* y_warpwise =
      reinterpret_cast<float*>(smem + x_elems * sizeof(in_T) + w_elems * sizeof(W_T));

  auto pipe = cuda::make_pipeline();
  const size_t toff = (threadIdx.y * tx + threadIdx.x) * vec_size;

  float y_acc[PAIRS_PER_BLOCK][RANK_TILE];
#pragma unroll
  for (int pp = 0; pp < PAIRS_PER_BLOCK; ++pp)
#pragma unroll
    for (int r = 0; r < RANK_TILE; ++r) y_acc[pp][r] = 0.f;

  vec_t<in_T, vec_size> x_vec;
  vec_t<W_T, vec_size> w_vec;

  // Prologue: fill pipeline
  constexpr size_t pro = (num_tiles < NUM_STAGES) ? num_tiles : NUM_STAGES;
#pragma unroll
  for (size_t t = 0; t < pro; ++t) {
    const size_t s = t % NUM_STAGES;
    const size_t tb = t * tile_size;
    pipe.producer_acquire();
#pragma unroll
    for (int pp = 0; pp < PAIRS_PER_BLOCK; ++pp) {
      if (pair_valid[pp] && tb + toff < feat_in) {
        cuda::memcpy_async(X_shared + (s * PAIRS_PER_BLOCK + pp) * tile_size + toff,
                           X_tok[pp] + tb + toff, cuda::aligned_size_t<X_copy_size>(X_copy_size),
                           pipe);
#pragma unroll
        for (int r = 0; r < RANK_TILE; ++r)
          if (j0 + r < feat_out)
            cuda::memcpy_async(
                W_shared + ((s * PAIRS_PER_BLOCK + pp) * RANK_TILE + r) * tile_size + toff,
                W_base[pp] + r * feat_in + tb + toff,
                cuda::aligned_size_t<W_copy_size>(W_copy_size), pipe);
      }
    }
    pipe.producer_commit();
  }

  // Main loop
  for (size_t t = pro; t < num_tiles; ++t) {
    const size_t cs = (t - pro) % NUM_STAGES;
    pipe.consumer_wait();
    block.sync();
#pragma unroll
    for (int pp = 0; pp < PAIRS_PER_BLOCK; ++pp) {
      if (!pair_valid[pp]) continue;
      x_vec.load(X_shared + (cs * PAIRS_PER_BLOCK + pp) * tile_size + toff);
#pragma unroll
      for (int r = 0; r < RANK_TILE; ++r) {
        if (j0 + r < feat_out) {
          w_vec.load(W_shared + ((cs * PAIRS_PER_BLOCK + pp) * RANK_TILE + r) * tile_size + toff);
          float sum = 0.f;
#pragma unroll
          for (size_t i = 0; i < vec_size; ++i) sum += float(w_vec[i]) * float(x_vec[i]) * scale;
#pragma unroll
          for (size_t off = tx / 2; off > 0; off /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, off);
          if (threadIdx.x == 0) y_warpwise[pp * RANK_TILE * ty + r * ty + threadIdx.y] = sum;
        }
      }
    }
    block.sync();
#pragma unroll
    for (int pp = 0; pp < PAIRS_PER_BLOCK; ++pp) {
      if (!pair_valid[pp]) continue;
#pragma unroll
      for (int r = 0; r < RANK_TILE; ++r)
        if (j0 + r < feat_out) {
          float v = 0.f;
          for (int w = 0; w < ty; ++w) v += y_warpwise[pp * RANK_TILE * ty + r * ty + w];
          y_acc[pp][r] += v;
        }
    }
    block.sync();
    pipe.consumer_release();

    // Load next tile
    const size_t ls = t % NUM_STAGES;
    const size_t tb = t * tile_size;
    pipe.producer_acquire();
#pragma unroll
    for (int pp = 0; pp < PAIRS_PER_BLOCK; ++pp) {
      if (pair_valid[pp] && tb + toff < feat_in) {
        cuda::memcpy_async(X_shared + (ls * PAIRS_PER_BLOCK + pp) * tile_size + toff,
                           X_tok[pp] + tb + toff, cuda::aligned_size_t<X_copy_size>(X_copy_size),
                           pipe);
#pragma unroll
        for (int r = 0; r < RANK_TILE; ++r)
          if (j0 + r < feat_out)
            cuda::memcpy_async(
                W_shared + ((ls * PAIRS_PER_BLOCK + pp) * RANK_TILE + r) * tile_size + toff,
                W_base[pp] + r * feat_in + tb + toff,
                cuda::aligned_size_t<W_copy_size>(W_copy_size), pipe);
      }
    }
    pipe.producer_commit();
  }

  // Epilogue: drain remaining pipeline stages
  for (size_t t = (num_tiles > pro ? num_tiles - pro : 0); t < num_tiles; ++t) {
    const size_t cs = t % NUM_STAGES;
    const size_t ts = t * tile_size;
    pipe.consumer_wait();
    block.sync();
#pragma unroll
    for (int pp = 0; pp < PAIRS_PER_BLOCK; ++pp) {
      if (!pair_valid[pp]) continue;
      x_vec.load(X_shared + (cs * PAIRS_PER_BLOCK + pp) * tile_size + toff);
#pragma unroll
      for (int r = 0; r < RANK_TILE; ++r) {
        if (j0 + r < feat_out) {
          w_vec.load(W_shared + ((cs * PAIRS_PER_BLOCK + pp) * RANK_TILE + r) * tile_size + toff);
          float sum = 0.f;
#pragma unroll
          for (size_t i = 0; i < vec_size; ++i) sum += float(w_vec[i]) * float(x_vec[i]) * scale;
#pragma unroll
          for (size_t off = tx / 2; off > 0; off /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, off);
          if (threadIdx.x == 0) {
            if (t == num_tiles - 1) sum = (ts + threadIdx.y * tx * vec_size < feat_in) ? sum : 0.f;
            y_warpwise[pp * RANK_TILE * ty + r * ty + threadIdx.y] = sum;
          }
        }
      }
    }
    block.sync();
#pragma unroll
    for (int pp = 0; pp < PAIRS_PER_BLOCK; ++pp) {
      if (!pair_valid[pp]) continue;
#pragma unroll
      for (int r = 0; r < RANK_TILE; ++r)
        if (j0 + r < feat_out) {
          float v = 0.f;
          for (int w = 0; w < ty; ++w) v += y_warpwise[pp * RANK_TILE * ty + r * ty + w];
          y_acc[pp][r] += v;
        }
    }
    block.sync();
    pipe.consumer_release();
  }

  // Write results
  if (block.thread_rank() == 0) {
#pragma unroll
    for (int pp = 0; pp < PAIRS_PER_BLOCK; ++pp) {
      if (!pair_valid[pp]) continue;
#pragma unroll
      for (int r = 0; r < RANK_TILE; ++r)
        if (j0 + r < feat_out)
          Y[slice_id * num_pairs * feat_out + (p0 + pp) * feat_out + j0 + r] +=
              static_cast<out_T>(y_acc[pp][r]);
    }
  }
}

// ============================================================
// MoE BGMV Expand Sliced Kernel
// ============================================================
template <int feat_in, int feat_out, size_t vec_size, int tx, int ty, int tz, typename in_T,
          typename W_T>
__global__ void moe_bgmv_expand_sliced_kernel(
    float* __restrict__ Y, const in_T* __restrict__ X, W_T** __restrict__ w_ptr,
    const int64_t* __restrict__ sorted_token_ids, const int64_t* __restrict__ expert_ids,
    const int64_t* __restrict__ lora_indices, const float* __restrict__ topk_weights,
    const int64_t* __restrict__ slice_start_loc, int64_t num_pairs, int64_t num_experts,
    int64_t total_feat_out, int32_t current_feat_out, int64_t num_tokens, int64_t lora_stride,
    float scale) {
  size_t pair_idx = blockIdx.x;
  size_t tile_idx = blockIdx.y;
  int64_t token_idx = sorted_token_ids[pair_idx];
  if (token_idx < 0 || token_idx >= num_tokens) return;
  int64_t lora_id = lora_indices[token_idx];
  if (lora_id < 0) return;
  int slice_id = blockIdx.z;
  int64_t expert_id = expert_ids[pair_idx];
  float topk_w = topk_weights[pair_idx];
  int64_t col_offset = slice_start_loc[slice_id];
  const W_T* W = w_ptr[slice_id * num_experts + expert_id] + lora_id * lora_stride;
  auto block = cg::this_thread_block();
  vec_t<in_T, vec_size> x_vec;
  x_vec.load(X + slice_id * num_pairs * feat_in + pair_idx * feat_in + threadIdx.x * vec_size);
  vec_t<W_T, vec_size> w_vec;
  w_vec.load(W + (tile_idx * tz * ty) * feat_in + block.thread_rank() * vec_size);
  float sum = 0.f;
#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) sum += float(w_vec[i]) * float(x_vec[i]) * scale;
  cg::thread_block_tile<tx> g = cg::tiled_partition<tx>(block);
#pragma unroll
  for (size_t offset = tx / 2; offset > 0; offset /= 2) sum += g.shfl_down(sum, offset);
  sum = g.shfl(sum, 0);
  if (threadIdx.x == 0) {
    int out_col = col_offset + tile_idx * (tz * ty) + threadIdx.z * ty + threadIdx.y;
    atomicAdd(Y + token_idx * total_feat_out + out_col, sum * topk_w);
  }
}

// ============================================================
// Host-side dispatch: Shrink
// ============================================================

template <int feat_in, int feat_out, typename in_T, typename out_T, typename W_T>
void moe_bgmv_shrink_sliced(out_T* __restrict__ Y, const in_T* __restrict__ X,
                            W_T** __restrict__ w_ptr, const int64_t* sorted_token_ids,
                            const int64_t* expert_ids, const int64_t* lora_indices,
                            int64_t num_pairs, int64_t num_slices, int64_t num_experts,
                            int64_t num_tokens, int64_t lora_stride, float scale) {
  // Use the current CUDA stream
  const cudaStream_t stream = BGMV_MOE_GET_STREAM();

  constexpr int cfg_tx = MoeShrinkKernelConfig::tx;
  constexpr int cfg_ty = MoeShrinkKernelConfig::ty;
  constexpr int RT = MoeShrinkKernelConfig::rank_tile;
  constexpr int gy = (feat_out + RT - 1) / RT;
  constexpr size_t fvs = MoeShrinkKernelConfig::vec_size;

  // Runtime: detect sm_80+ for extended shared memory
  int dev;
  cudaGetDevice(&dev);
  int sm_major = 0;
  cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, dev);
  const bool extended = (sm_major >= 9);
  const bool decode = (num_pairs <= MoeShrinkKernelConfig::decode_threshold);

  const int ppb = (extended && decode) ? MoeShrinkKernelConfig::pairs_per_block_decode
                                       : MoeShrinkKernelConfig::pairs_per_block_prefill;
  const int nstg = (extended && decode) ? MoeShrinkKernelConfig::num_stages_extended
                                        : MoeShrinkKernelConfig::num_stages_default;

#define LAUNCH(PPB, NSTG, VS)                                                                  \
  do {                                                                                         \
    constexpr size_t ts = cfg_tx * cfg_ty * (VS);                                              \
    constexpr size_t shmem = (NSTG) * (PPB) * ts * sizeof(in_T) +                              \
                             (NSTG) * (PPB) * RT * ts * sizeof(W_T) +                          \
                             (PPB) * RT * cfg_ty * sizeof(float);                              \
    auto kfn = &moe_bgmv_shrink_sliced_kernel<feat_in, feat_out, RT, (PPB), (NSTG), (VS),      \
                                              (VS) * sizeof(in_T), (VS) * sizeof(W_T), cfg_tx, \
                                              cfg_ty, in_T, out_T, W_T>;                       \
    if constexpr (shmem > 48 * 1024)                                                           \
      cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);      \
    dim3 g((int)((num_pairs + (PPB) - 1) / (PPB)), gy, num_slices);                            \
    kfn<<<g, dim3(cfg_tx, cfg_ty), shmem, stream>>>(Y, X, w_ptr, sorted_token_ids, expert_ids, \
                                                    lora_indices, num_pairs, num_experts,      \
                                                    num_tokens, lora_stride, scale);           \
  } while (0)

#define DISPATCH(VS)             \
  do {                           \
    if (ppb == 4 && nstg == 3) { \
      LAUNCH(4, 3, VS);          \
    } else {                     \
      LAUNCH(1, 2, VS);          \
    }                            \
  } while (0)

  if constexpr (feat_in % (fvs * cfg_tx) == 0) {
    DISPATCH(fvs);
  } else if constexpr (feat_in % (fvs / 2 * cfg_tx) == 0) {
    DISPATCH(fvs / 2);
  } else if constexpr (feat_in % (fvs / 4 * cfg_tx) == 0) {
    DISPATCH(fvs / 4);
  } else if constexpr (feat_in % cfg_tx == 0) {
    DISPATCH(1);
  }

#undef DISPATCH
#undef LAUNCH
}

// ============================================================
// Host-side dispatch: Expand
// ============================================================

template <int feat_in, int feat_out, typename in_T, typename W_T>
void moe_bgmv_expand_sliced(float* __restrict__ Y, const in_T* __restrict__ X,
                            W_T** __restrict__ w_ptr, const int64_t* sorted_token_ids,
                            const int64_t* expert_ids, const int64_t* lora_indices,
                            const float* topk_weights, const int64_t* slice_start_loc,
                            int64_t num_pairs, int64_t num_slices, int64_t num_experts,
                            int64_t total_feat_out, int32_t current_feat_out, int64_t num_tokens,
                            int64_t lora_stride, float scale) {
  const cudaStream_t stream = BGMV_MOE_GET_STREAM();  // current CUDA stream

  constexpr size_t vec_size = MoeExpandKernelConfig::vec_size;
  constexpr int tz = MoeExpandKernelConfig::tz;
  static_assert(feat_in % vec_size == 0);
  constexpr int tx = feat_in / vec_size;

  if constexpr (32 % tx == 0 && feat_out % (32 / tx * tz) == 0) {
    constexpr int ty = 32 / tx;
    moe_bgmv_expand_sliced_kernel<feat_in, feat_out, vec_size, tx, ty, tz, in_T, W_T>
        <<<dim3(num_pairs, feat_out / (ty * tz), num_slices), dim3(tx, ty, tz), 0, stream>>>(
            Y, X, w_ptr, sorted_token_ids, expert_ids, lora_indices, topk_weights, slice_start_loc,
            num_pairs, num_experts, total_feat_out, current_feat_out, num_tokens, lora_stride,
            scale);
  } else if constexpr (16 % tx == 0 && feat_out % (16 / tx * tz) == 0) {
    constexpr int ty = 16 / tx;
    moe_bgmv_expand_sliced_kernel<feat_in, feat_out, vec_size, tx, ty, tz, in_T, W_T>
        <<<dim3(num_pairs, feat_out / (ty * tz), num_slices), dim3(tx, ty, tz), 0, stream>>>(
            Y, X, w_ptr, sorted_token_ids, expert_ids, lora_indices, topk_weights, slice_start_loc,
            num_pairs, num_experts, total_feat_out, current_feat_out, num_tokens, lora_stride,
            scale);
  } else if constexpr (8 % tx == 0 && feat_out % (8 / tx * tz) == 0) {
    constexpr int ty = 8 / tx;
    moe_bgmv_expand_sliced_kernel<feat_in, feat_out, vec_size, tx, ty, tz, in_T, W_T>
        <<<dim3(num_pairs, feat_out / (ty * tz), num_slices), dim3(tx, ty, tz), 0, stream>>>(
            Y, X, w_ptr, sorted_token_ids, expert_ids, lora_indices, topk_weights, slice_start_loc,
            num_pairs, num_experts, total_feat_out, current_feat_out, num_tokens, lora_stride,
            scale);
  }
}

// ============================================================
// Instantiation macros
// ============================================================
#define INST_MOE_BGMV_SHRINK_SLICED(feat_in, feat_out, in_T, out_T, W_T)                   \
  template void moe_bgmv_shrink_sliced<feat_in, feat_out, in_T, out_T, W_T>(               \
      out_T*, const in_T*, W_T**, const int64_t*, const int64_t*, const int64_t*, int64_t, \
      int64_t, int64_t, int64_t, int64_t, float);

#define INST_MOE_BGMV_EXPAND_SLICED(feat_in, feat_out, in_T, W_T)                               \
  template void moe_bgmv_expand_sliced<feat_in, feat_out, in_T, W_T>(                           \
      float*, const in_T*, W_T**, const int64_t*, const int64_t*, const int64_t*, const float*, \
      const int64_t*, int64_t, int64_t, int64_t, int64_t, int32_t, int64_t, int64_t, float);

#define INST_MOE_BGMV_TWOSIDE(in_T, out_T, W_T, narrow, wide) \
  INST_MOE_BGMV_SHRINK_SLICED(wide, narrow, in_T, out_T, W_T) \
  INST_MOE_BGMV_EXPAND_SLICED(narrow, wide, in_T, W_T)
