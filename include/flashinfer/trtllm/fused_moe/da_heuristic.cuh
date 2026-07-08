/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cub/block/block_histogram.cuh>
#include <cub/block/block_radix_sort.cuh>

#include "flashinfer/trtllm/fused_moe/da_heuristic_constants.cuh"

namespace flashinfer {
namespace da_heuristic {

// ===========================================================================
// DAKNNv2 tile selector.
//
// Each uploaded state stores a small table of sorted-descending, L2-normalized
// per-expert count vectors. At runtime the selector builds the current expert
// histogram, normalizes it, picks the nearest exemplar by cosine similarity,
// and activates that exemplar's SWITCH body.
// ===========================================================================

struct __align__(16) DAKnnParams {
  // How many exemplars are populated (1..kMaxExemplars).
  int num_exemplars;

  // Number of candidate tile sizes (1..kMaxTiles), and the tile sizes
  // themselves in ascending order.
  int num_tiles;
  int tile_sizes[kMaxTiles];

  int num_local_experts;  // <= kMaxKnnExperts
  int local_expert_offset;

  // Per-exemplar SWITCH body index: best_body_idx[e] is the SWITCH conditional
  // value to activate when exemplar e wins the k-NN vote. In the v2 bundle
  // format this is the exemplar's ordinal position (0..num_exemplars-1).
  int best_body_idx[kMaxExemplars];
  // Per-exemplar tile shape (tile_n tokens per tile). Looked up by exemplar
  // index, NOT by body index — each exemplar carries its own tile shape so
  // the kernel can write *selected_tile_n without an extra indirection.
  int exemplar_tile_shape[kMaxExemplars];
  // Per-exemplar kernel config ID. Stored here for host readback after
  // the decision kernel selects an exemplar; the MoE body kernel doesn't
  // read this field.
  int exemplar_kernel_id[kMaxExemplars];
  // Per-exemplar sorted-descending L2-normalized count vector.
  // Stored row-major: exemplar e is exemplar_norm[e*kMaxKnnExperts +
  // 0..num_local_experts). Trailing entries past num_local_experts are 0.
  float exemplar_norm[kMaxExemplars * kMaxKnnExperts];
};

__device__ __forceinline__ unsigned int get_power_of_2_value(unsigned int n) {
  if (n == 0) return 1;
  if ((n & (n - 1)) == 0) return n;
  return 1U << (32 - __clz(n - 1));
}

__device__ __forceinline__ int da_knn_decode_local_id(int32_t raw, int local_expert_offset,
                                                      int max_local_expert, int num_local_experts) {
  int expert_id = raw;
  if (expert_id < local_expert_offset || expert_id >= max_local_expert) {
    int hi = (raw >> 16) & 0xFFFF;
    if (hi >= local_expert_offset && hi < max_local_expert) {
      expert_id = hi;
    } else {
      expert_id = raw & 0xFFFF;
    }
  }
  int const local_id = expert_id - local_expert_offset;
  return (local_id >= 0 && local_id < num_local_experts) ? local_id : -1;
}

// Register-resident bitonic. Each thread owns one element in a register;
// intra-warp stages (j < 32) use __shfl_xor_sync (no shared mem traffic and
// no __syncthreads), inter-warp stages (j >= 32) write back to shared, sync,
// read partner. For sort_len = blockDim.x = 256 this saves ~30 shared-memory
// round-trips per stage versus the all-shared bitonic, which on B200 is
// faster than CUB BlockRadixSort by ~0.5 µs at the same sort_len.
__device__ __forceinline__ void da_knn_sort_counts_register_bitonic_desc(int* counts_int,
                                                                         int sort_len) {
  // Caller guarantees blockDim.x >= sort_len and sort_len is a power of 2.
  int value = (threadIdx.x < sort_len) ? counts_int[threadIdx.x] : INT_MIN;

  for (int k = 2; k <= sort_len; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      int partner;
      if (j >= 32) {
        counts_int[threadIdx.x] = value;
        __syncthreads();
        partner = counts_int[threadIdx.x ^ j];
        __syncthreads();
      } else {
        partner = __shfl_xor_sync(0xFFFFFFFF, value, j);
      }
      // For descending sort:
      //   ascending segment ((idx & k) == 0): lower index keeps max, higher keeps min
      //   descending segment:                 lower index keeps min, higher keeps max
      bool ascending = ((threadIdx.x & k) == 0);
      bool low = ((threadIdx.x & j) == 0);
      bool keep_max = (ascending == low);
      value = keep_max ? max(value, partner) : min(value, partner);
    }
  }
  if (threadIdx.x < sort_len) counts_int[threadIdx.x] = value;
  __syncthreads();
}

__device__ __forceinline__ void da_knn_sort_counts_warp_bitonic_desc(int* counts_int,
                                                                     int sort_len) {
  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    for (int k = 2; k <= sort_len; k <<= 1) {
      for (int j = k >> 1; j > 0; j >>= 1) {
        __syncwarp();
        for (int i = lane; i < sort_len; i += 32) {
          int ix = i ^ j;
          if (ix > i) {
            bool ascending = ((i & k) == 0);
            int a = counts_int[i];
            int b = counts_int[ix];
            bool swap = ascending ? (a < b) : (a > b);
            if (swap) {
              counts_int[i] = b;
              counts_int[ix] = a;
            }
          }
        }
      }
    }
  }
  __syncthreads();
}

template <int ItemsPerThread>
__device__ __forceinline__ void da_knn_sort_counts_cub_desc(int* counts_int, int num_local_experts,
                                                            int count_upper_bound) {
  using BlockSort = cub::BlockRadixSort<int, kKnnSelectorBlockThreads, ItemsPerThread>;
  __shared__ typename BlockSort::TempStorage sort_storage;

  int items[ItemsPerThread];
#pragma unroll
  for (int item = 0; item < ItemsPerThread; ++item) {
    // CUB BlockRadixSort uses blocked per-thread item layout:
    // thread t owns logical indices [t * ItemsPerThread, ...].
    int idx = threadIdx.x * ItemsPerThread + item;
    items[item] = idx < num_local_experts ? counts_int[idx] : 0;
  }

  // Counts are non-negative and bounded by the number of expanded routing
  // entries this graph bucket can contain. Restricting radix sort to that
  // bit range trims passes from 32 bits to ~13-16 bits for decode buckets.
  int end_bit = 32;
  if (count_upper_bound > 0) {
    unsigned int ub = static_cast<unsigned int>(count_upper_bound);
    end_bit = 32 - __clz(ub);
    if (end_bit <= 0) end_bit = 1;
  }
  BlockSort(sort_storage).SortDescendingBlockedToStriped(items, 0, end_bit);

#pragma unroll
  for (int item = 0; item < ItemsPerThread; ++item) {
    int idx = threadIdx.x + item * kKnnSelectorBlockThreads;
    if (idx < num_local_experts) {
      counts_int[idx] = items[item];
    }
  }
  __syncthreads();
}

template <typename CountT>
__device__ __forceinline__ int da_knn_compact_counts_for_sort(int* counts_int,
                                                              const CountT* __restrict__ raw_counts,
                                                              int num_local_experts) {
  if (num_local_experts <= kKnnSelectorBlockThreads) {
    __shared__ int warp_nz[8];
    __shared__ int warp_min[8];
    __shared__ int warp_max[8];
    __shared__ int total_nz;
    __shared__ int min_count;
    __shared__ int max_count;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int v = threadIdx.x < num_local_experts ? static_cast<int>(raw_counts[threadIdx.x]) : 0;
    const bool active = threadIdx.x < num_local_experts && v > 0;
    const unsigned mask = __ballot_sync(0xFFFFFFFF, active);
    const int rank = __popc(mask & ((1u << lane) - 1u));
    const int nz = __popc(mask);

    int local_min = active ? v : 0x7fffffff;
    int local_max = active ? v : 0;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      local_min = min(local_min, __shfl_down_sync(0xFFFFFFFF, local_min, off));
      local_max = max(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, off));
    }
    if (lane == 0) {
      warp_nz[warp] = nz;
      warp_min[warp] = local_min;
      warp_max[warp] = local_max;
    }
    __syncthreads();

    int prefix = 0;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      if (w < warp) prefix += warp_nz[w];
    }
    if (active) {
      counts_int[prefix + rank] = v;
    }
    if (threadIdx.x == 0) {
      int sum = 0;
      int mn = 0x7fffffff;
      int mx = 0;
#pragma unroll
      for (int w = 0; w < 8; ++w) {
        sum += warp_nz[w];
        mn = min(mn, warp_min[w]);
        mx = max(mx, warp_max[w]);
      }
      total_nz = sum;
      min_count = mn;
      max_count = mx;
    }
    __syncthreads();

    const int nz_total = total_nz;
    for (int i = threadIdx.x + nz_total; i < num_local_experts; i += blockDim.x) {
      counts_int[i] = 0;
    }
    __syncthreads();

    if (nz_total <= 1 || min_count == max_count) {
      return 1;
    }
    return nz_total;
  }

  __shared__ int nonzero_count;
  __shared__ int min_count;
  __shared__ int max_count;

  if (threadIdx.x == 0) {
    nonzero_count = 0;
    min_count = 0x7fffffff;
    max_count = 0;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < num_local_experts; i += blockDim.x) {
    int v = static_cast<int>(raw_counts[i]);
    if (v > 0) {
      int pos = atomicAdd(&nonzero_count, 1);
      counts_int[pos] = v;
      atomicMin(&min_count, v);
      atomicMax(&max_count, v);
    }
  }
  __syncthreads();

  int const nz = nonzero_count;
  for (int i = threadIdx.x + nz; i < num_local_experts; i += blockDim.x) {
    counts_int[i] = 0;
  }
  __syncthreads();

  // With <=1 populated expert the compacted vector is already sorted. Any
  // equal-count nonzero prefix is also sorted after compaction because zeros
  // have already been moved behind it; this catches synthetic uniform buckets
  // without weakening the permutation-invariant kNN definition.
  if (nz <= 1 || min_count == max_count) {
    return 1;
  }
  return nz;
}

// Variant that re-zeros the global counts array as it reads it. Lets the
// histogram pass for the next replay assume counts[] is already zero, so the
// dedicated zero kernel can be removed from the graph. The caller must
// cudaMemset counts[] to zero exactly once at allocation; from there each
// invocation's select kernel maintains the invariant for the next.
__device__ __forceinline__ int da_knn_compact_counts_for_sort_and_zero(
    int* counts_int, int32_t* __restrict__ raw_counts, int num_local_experts) {
  if (num_local_experts <= kKnnSelectorBlockThreads) {
    __shared__ int warp_nz[8];
    __shared__ int warp_min[8];
    __shared__ int warp_max[8];
    __shared__ int total_nz;
    __shared__ int min_count;
    __shared__ int max_count;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int v = threadIdx.x < num_local_experts ? raw_counts[threadIdx.x] : 0;
    if (threadIdx.x < num_local_experts) {
      raw_counts[threadIdx.x] = 0;
    }
    const bool active = threadIdx.x < num_local_experts && v > 0;
    const unsigned mask = __ballot_sync(0xFFFFFFFF, active);
    const int rank = __popc(mask & ((1u << lane) - 1u));
    const int nz = __popc(mask);

    int local_min = active ? v : 0x7fffffff;
    int local_max = active ? v : 0;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      local_min = min(local_min, __shfl_down_sync(0xFFFFFFFF, local_min, off));
      local_max = max(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, off));
    }
    if (lane == 0) {
      warp_nz[warp] = nz;
      warp_min[warp] = local_min;
      warp_max[warp] = local_max;
    }
    __syncthreads();

    int prefix = 0;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      if (w < warp) prefix += warp_nz[w];
    }
    if (active) {
      counts_int[prefix + rank] = v;
    }
    if (threadIdx.x == 0) {
      int sum = 0;
      int mn = 0x7fffffff;
      int mx = 0;
#pragma unroll
      for (int w = 0; w < 8; ++w) {
        sum += warp_nz[w];
        mn = min(mn, warp_min[w]);
        mx = max(mx, warp_max[w]);
      }
      total_nz = sum;
      min_count = mn;
      max_count = mx;
    }
    __syncthreads();

    const int nz_total = total_nz;
    for (int i = threadIdx.x + nz_total; i < num_local_experts; i += blockDim.x) {
      counts_int[i] = 0;
    }
    __syncthreads();

    if (nz_total <= 1 || min_count == max_count) {
      return 1;
    }
    return nz_total;
  }

  __shared__ int nonzero_count;
  __shared__ int min_count;
  __shared__ int max_count;

  if (threadIdx.x == 0) {
    nonzero_count = 0;
    min_count = 0x7fffffff;
    max_count = 0;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < num_local_experts; i += blockDim.x) {
    int v = raw_counts[i];
    if (v > 0) {
      int pos = atomicAdd(&nonzero_count, 1);
      counts_int[pos] = v;
      atomicMin(&min_count, v);
      atomicMax(&max_count, v);
    }
    raw_counts[i] = 0;  // Self-zero; eliminates the per-replay zero kernel.
  }
  __syncthreads();

  int const nz = nonzero_count;
  for (int i = threadIdx.x + nz; i < num_local_experts; i += blockDim.x) {
    counts_int[i] = 0;
  }
  __syncthreads();

  if (nz <= 1 || min_count == max_count) {
    return 1;
  }
  return nz;
}

// One warp per exemplar dot product. Computes the un-normalized inner product
// (counts . exemplar_e) for each exemplar; ||counts|| is identical across all
// exemplars so it factors out of the argmax. `exemplar_base` points to either
// shared-memory or global-memory exemplar rows (row stride kMaxKnnExperts).
__device__ __forceinline__ void da_knn_compute_sims(const int* __restrict__ counts_int,
                                                    const float* __restrict__ exemplar_base,
                                                    float* sims, int num_local_experts,
                                                    int num_exemplars) {
  const int warp_id = threadIdx.x >> 5;
  const int lane_id = threadIdx.x & 31;
  if (warp_id < num_exemplars) {
    const float* row = &exemplar_base[warp_id * kMaxKnnExperts];
    float partial = 0.0f;
    for (int i = lane_id; i < num_local_experts; i += 256) {
#pragma unroll
      for (int delta = 0; delta < 256; delta += 32) {
        int idx = i + delta;
        if (idx < num_local_experts) {
          partial += static_cast<float>(counts_int[idx]) * row[idx];
        }
      }
    }
    for (int off = 16; off > 0; off >>= 1) {
      partial += __shfl_down_sync(0xFFFFFFFF, partial, off);
    }
    if (lane_id == 0) sims[warp_id] = partial;
  }
}

// Compact → sort → cosine-sim → argmax → cudaGraphSetConditional.
// `exemplar_base` is the base pointer to exemplar rows (row stride
// kMaxKnnExperts); callers pass either a shared-memory copy or
// `params->exemplar_norm` directly.
__device__ __forceinline__ void da_knn_finish_selection_from_counts(
    int* counts_int, float* sims, const float* __restrict__ exemplar_base,
    const DAKnnParams* __restrict__ params, int32_t* __restrict__ selected_tile_idx,
    int32_t* __restrict__ selected_tile_n, cudaGraphConditionalHandle switch_handle,
    int count_upper_bound, int sort_items) {
  const int num_local_experts = params->num_local_experts;
  const int num_exemplars = params->num_exemplars;

  int sort_len = static_cast<int>(get_power_of_2_value(static_cast<unsigned int>(sort_items)));
  if (sort_len > kMaxKnnExperts) sort_len = kMaxKnnExperts;
  if (sort_len > 1) {
    if (sort_len <= 32) {
      da_knn_sort_counts_warp_bitonic_desc(counts_int, sort_len);
    } else if (sort_len <= kKnnSelectorBlockThreads) {
      da_knn_sort_counts_register_bitonic_desc(counts_int, sort_len);
    } else {
      da_knn_sort_counts_cub_desc<2>(counts_int, num_local_experts, count_upper_bound);
    }
  }

  da_knn_compute_sims(counts_int, exemplar_base, sims, num_local_experts, num_exemplars);
  __syncthreads();

  if (threadIdx.x == 0) {
    int best_e = 0;
    float best_sim = sims[0];
    for (int e = 1; e < num_exemplars; e++) {
      if (sims[e] > best_sim) {
        best_sim = sims[e];
        best_e = e;
      }
    }
    int best_body = params->best_body_idx[best_e];
    *selected_tile_idx = best_body;
    *selected_tile_n = params->exemplar_tile_shape[best_e];
    cudaGraphSetConditional(switch_handle, static_cast<unsigned int>(best_body));
  }
}

template <bool UsePdl>
__global__ void da_knn_histogram_counts_graph_kernel(const int32_t* __restrict__ topk_ids,
                                                     int num_elements,
                                                     const DAKnnParams* __restrict__ params,
                                                     int32_t* __restrict__ counts) {
  __shared__ unsigned int cub_counts[kKnnSelectorHistogramBins];

  using BlockHistogram = cub::BlockHistogram<unsigned short, kKnnSplitHistogramBlockThreads,
                                             kKnnSplitHistogramItemsPerThread,
                                             kKnnSelectorHistogramBins, cub::BLOCK_HISTO_ATOMIC>;
  __shared__ typename BlockHistogram::TempStorage hist_storage;

  static_assert(kKnnSplitHistogramItemsPerThread == 8,
                "Vectorised histogram path assumes 8 items/thread (two int4 loads).");

  const int num_local_experts = params->num_local_experts;
  const int local_expert_offset = params->local_expert_offset;
  const int max_local_expert = local_expert_offset + num_local_experts;

  BlockHistogram(hist_storage).InitHistogram(cub_counts);
  __syncthreads();

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  // Wait for the upstream routing kernel to finish writing topk_ids[] before
  // we read it. Under PDL (LaunchCompletion port from the routing edge in the
  // graph), this kernel may have started running while routing's tail is
  // still in flight; the InitHistogram + params reads above are
  // parent-independent and overlap that window.
  if constexpr (UsePdl) cudaGridDependencySynchronize();
#endif

  // Vectorized histogram: each thread reads 8 int32s as two int4 loads.
  // Falls back to scalar loads when topk_ids is not 16-byte aligned.
  bool const aligned = ((reinterpret_cast<uintptr_t>(topk_ids) & 0xF) == 0);
  constexpr int kItemsPerBlockIter =
      kKnnSplitHistogramBlockThreads * kKnnSplitHistogramItemsPerThread;
  int const tile_stride = gridDim.x * kItemsPerBlockIter;
  if (aligned) {
    for (int base = blockIdx.x * kItemsPerBlockIter; base < num_elements; base += tile_stride) {
      unsigned short items[kKnnSplitHistogramItemsPerThread];
      int const block_lane = threadIdx.x;
      int const lane_base0 = base + block_lane * 4;
      int const lane_base1 = base + (block_lane + kKnnSplitHistogramBlockThreads) * 4;
      int4 v0, v1;
      if (lane_base0 + 3 < num_elements) {
        v0 = *reinterpret_cast<const int4*>(&topk_ids[lane_base0]);
      } else {
        v0.x = (lane_base0 + 0 < num_elements) ? topk_ids[lane_base0 + 0] : -1;
        v0.y = (lane_base0 + 1 < num_elements) ? topk_ids[lane_base0 + 1] : -1;
        v0.z = (lane_base0 + 2 < num_elements) ? topk_ids[lane_base0 + 2] : -1;
        v0.w = (lane_base0 + 3 < num_elements) ? topk_ids[lane_base0 + 3] : -1;
      }
      if (lane_base1 + 3 < num_elements) {
        v1 = *reinterpret_cast<const int4*>(&topk_ids[lane_base1]);
      } else {
        v1.x = (lane_base1 + 0 < num_elements) ? topk_ids[lane_base1 + 0] : -1;
        v1.y = (lane_base1 + 1 < num_elements) ? topk_ids[lane_base1 + 1] : -1;
        v1.z = (lane_base1 + 2 < num_elements) ? topk_ids[lane_base1 + 2] : -1;
        v1.w = (lane_base1 + 3 < num_elements) ? topk_ids[lane_base1 + 3] : -1;
      }
      auto enc = [&](int32_t raw) {
        int local_id =
            da_knn_decode_local_id(raw, local_expert_offset, max_local_expert, num_local_experts);
        return static_cast<unsigned short>(local_id >= 0 ? local_id : kMaxKnnExperts);
      };
      items[0] = enc(v0.x);
      items[1] = enc(v0.y);
      items[2] = enc(v0.z);
      items[3] = enc(v0.w);
      items[4] = enc(v1.x);
      items[5] = enc(v1.y);
      items[6] = enc(v1.z);
      items[7] = enc(v1.w);
      BlockHistogram(hist_storage).Composite(items, cub_counts);
      __syncthreads();
    }
  } else {
    // Scalar fallback for unaligned input.
    for (int base = blockIdx.x * kItemsPerBlockIter; base < num_elements; base += tile_stride) {
      unsigned short items[kKnnSplitHistogramItemsPerThread];
#pragma unroll
      for (int item = 0; item < kKnnSplitHistogramItemsPerThread; ++item) {
        int const idx = base + threadIdx.x + item * blockDim.x;
        int local_id = -1;
        if (idx < num_elements) {
          local_id = da_knn_decode_local_id(topk_ids[idx], local_expert_offset, max_local_expert,
                                            num_local_experts);
        }
        items[item] = static_cast<unsigned short>(local_id >= 0 ? local_id : kMaxKnnExperts);
      }
      BlockHistogram(hist_storage).Composite(items, cub_counts);
      __syncthreads();
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  // The downstream selector cannot read `counts` until
  // cudaGridDependencySynchronize() completes, but it can be scheduled while
  // this kernel is still reducing block-private histograms into global memory.
  if constexpr (UsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif

  for (int i = threadIdx.x; i < num_local_experts; i += blockDim.x) {
    atomicAdd(&counts[i], static_cast<int32_t>(cub_counts[i]));
  }
}

template <bool UsePdl>
__global__ void da_knn_select_tile_from_counts_graph_kernel(
    int32_t* __restrict__ counts, int count_upper_bound, const DAKnnParams* __restrict__ params,
    int32_t* __restrict__ selected_tile_idx, int32_t* __restrict__ selected_tile_n,
    cudaGraphConditionalHandle switch_handle) {
  __shared__ int counts_int[kMaxKnnExperts];
  __shared__ float sims[kMaxExemplars];
  __shared__ float exemplar_smem[kMaxExemplars * kMaxKnnExperts];

  // Read params and run fast-path checks BEFORE the gridDep sync. `params`
  // is host-uploaded read-only and not touched by the histogram kernel, so
  // these reads don't depend on the parent.
  const int num_local_experts = params->num_local_experts;
  const int num_exemplars = params->num_exemplars;

  if (num_exemplars == 1) {
    if (threadIdx.x == 0) {
      int best_body = params->best_body_idx[0];
      *selected_tile_idx = best_body;
      *selected_tile_n = params->exemplar_tile_shape[0];
      cudaGraphSetConditional(switch_handle, static_cast<unsigned int>(best_body));
    }
    return;
  }

  // Issue the exemplar global → shared copy as ASYNC (cp.async on
  // sm_80+, TMA bulk async on sm_90+). The copy is fire-and-forget and
  // doesn't block the warp. Then we wait for the histogram kernel via PDL
  // (cudaGridDependencySynchronize). Finally we wait on the async copy
  // before reading exemplar_smem in the cosine-sim. Result: the global
  // load runs concurrently with the histogram kernel's tail (atomicAdd
  // reduction) and is fully hidden under PDL.
  //
  // Load only the cols actually consumed by the cosine sim
  // (num_local_experts per row, not the full kMaxKnnExperts row stride —
  // the trailing cols are zero-padded at upload time and never read).
  namespace cg = cooperative_groups;
  auto block = cg::this_thread_block();
  for (int e = 0; e < num_exemplars; ++e) {
    cg::memcpy_async(block, &exemplar_smem[e * kMaxKnnExperts],
                     &params->exemplar_norm[e * kMaxKnnExperts], sizeof(float) * num_local_experts);
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (UsePdl) cudaGridDependencySynchronize();
#endif

  // Self-zeroing compact (reads counts[], writes back zeros). The compact
  // pass touches only counts_int (shared) — exemplar_smem isn't needed
  // until the cosine sim, so the async memcpy can keep landing in the
  // background while the compact runs.
  int const sort_items =
      da_knn_compact_counts_for_sort_and_zero(counts_int, counts, num_local_experts);

  // Wait for exemplar_smem to be fully landed before cosine sim reads it.
  cg::wait(block);

  da_knn_finish_selection_from_counts(counts_int, sims, exemplar_smem, params, selected_tile_idx,
                                      selected_tile_n, switch_handle, count_upper_bound,
                                      sort_items);
}

__global__ void da_knn_select_tile_from_global_counts_graph_kernel(
    const int32_t* __restrict__ counts, int count_upper_bound,
    const DAKnnParams* __restrict__ params, int32_t* __restrict__ selected_tile_idx,
    int32_t* __restrict__ selected_tile_n, cudaGraphConditionalHandle switch_handle) {
  __shared__ int counts_int[kMaxKnnExperts];
  __shared__ float sims[kMaxExemplars];
  __shared__ float exemplar_smem[kMaxExemplars * kMaxKnnExperts];

  const int num_local_experts = params->num_local_experts;
  const int num_exemplars = params->num_exemplars;

  if (num_exemplars == 1) {
    if (threadIdx.x == 0) {
      int best_body = params->best_body_idx[0];
      *selected_tile_idx = best_body;
      *selected_tile_n = params->exemplar_tile_shape[0];
      cudaGraphSetConditional(switch_handle, static_cast<unsigned int>(best_body));
    }
    return;
  }

  namespace cg = cooperative_groups;
  auto block = cg::this_thread_block();
  for (int e = 0; e < num_exemplars; ++e) {
    cg::memcpy_async(block, &exemplar_smem[e * kMaxKnnExperts],
                     &params->exemplar_norm[e * kMaxKnnExperts], sizeof(float) * num_local_experts);
  }

  const int32_t* local_counts = counts + params->local_expert_offset;
  int const sort_items =
      da_knn_compact_counts_for_sort(counts_int, local_counts, num_local_experts);

  cg::wait(block);

  da_knn_finish_selection_from_counts(counts_int, sims, exemplar_smem, params, selected_tile_idx,
                                      selected_tile_n, switch_handle, count_upper_bound,
                                      sort_items);
}

// ===========================================================================
// Decision policies for the fused histogram + decision kernel. Each policy
// struct provides a static `decide_last_block` method that runs in the
// last block after all histogram contributions are globally visible.
// ===========================================================================

// k-NN policy: compact → sort → cosine-sim → argmax → cudaGraphSetConditional.
struct KnnDecisionPolicy {
  // Returns true if the kernel should skip the histogram phase entirely
  // (single-exemplar fast path — decision is independent of the distribution).
  __device__ static bool fast_path_skip(const DAKnnParams* params, int32_t* selected_tile_idx,
                                        int32_t* selected_tile_n,
                                        cudaGraphConditionalHandle switch_handle) {
    if (params->num_exemplars == 1) {
      if (threadIdx.x == 0 && blockIdx.x == 0) {
        int best_body = params->best_body_idx[0];
        *selected_tile_idx = best_body;
        *selected_tile_n = params->exemplar_tile_shape[0];
        cudaGraphSetConditional(switch_handle, static_cast<unsigned int>(best_body));
      }
      return true;
    }
    return false;
  }

  __device__ static void decide_last_block(const DAKnnParams* params, int32_t* counts,
                                           int num_local_experts, int num_elements,
                                           int32_t* selected_tile_idx, int32_t* selected_tile_n,
                                           cudaGraphConditionalHandle switch_handle) {
    __shared__ int counts_int[kMaxKnnExperts];
    __shared__ float sims[kMaxExemplars];
    __shared__ float exemplar_smem[kMaxExemplars * kMaxKnnExperts];

    const int num_exemplars = params->num_exemplars;

    // Preload exemplars global → shared; overlap with compact below.
    namespace cg = cooperative_groups;
    auto block = cg::this_thread_block();
    for (int e = 0; e < num_exemplars; ++e) {
      cg::memcpy_async(block, &exemplar_smem[e * kMaxKnnExperts],
                       &params->exemplar_norm[e * kMaxKnnExperts],
                       sizeof(float) * num_local_experts);
    }

    int const sort_items =
        da_knn_compact_counts_for_sort_and_zero(counts_int, counts, num_local_experts);

    cg::wait(block);

    da_knn_finish_selection_from_counts(counts_int, sims, exemplar_smem, params, selected_tile_idx,
                                        selected_tile_n, switch_handle, num_elements, sort_items);
  }
};

// ---------------------------------------------------------------------------
// Fused histogram + decision kernel (CUDA graph path, last-block).
//
// All blocks build a multi-block histogram via CUB BlockHistogram +
// atomicAdd into `counts[]`. The last block (detected via atomicInc on
// `d_block_done`) runs DecisionPolicy::decide_last_block to select the
// tile and drive cudaGraphSetConditional.
//
// Memory ordering: every thread fences after its atomicAdd(counts), then the
// CTA synchronizes before thread 0 increments d_block_done. This guarantees
// the last-block reader sees all per-thread histogram contributions.
// Self-zeroing counts for next replay.
//
// `UsePdl` controls whether the histogram phase waits for an upstream
// routing kernel via cudaGridDependencySynchronize().
// ---------------------------------------------------------------------------
template <bool UsePdl, typename DecisionPolicy>
__global__ void da_fused_hist_decision_graph_kernel(
    const int32_t* __restrict__ topk_ids, int num_elements, const DAKnnParams* __restrict__ params,
    int32_t* __restrict__ counts, unsigned int* __restrict__ d_block_done,
    int32_t* __restrict__ selected_tile_idx, int32_t* __restrict__ selected_tile_n,
    cudaGraphConditionalHandle switch_handle) {
  __shared__ unsigned int cub_counts[kKnnSelectorHistogramBins];
  using BlockHistogram = cub::BlockHistogram<unsigned short, kKnnSplitHistogramBlockThreads,
                                             kKnnSplitHistogramItemsPerThread,
                                             kKnnSelectorHistogramBins, cub::BLOCK_HISTO_ATOMIC>;
  __shared__ typename BlockHistogram::TempStorage hist_storage;

  static_assert(kKnnSplitHistogramItemsPerThread == 8,
                "Vectorised histogram path assumes 8 items/thread (two int4 loads).");

  const int num_local_experts = params->num_local_experts;
  const int local_expert_offset = params->local_expert_offset;
  const int max_local_expert = local_expert_offset + num_local_experts;
  // Fast path: single exemplar — decision is trivial regardless of policy.
  if (params->num_exemplars == 1) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      int best_body = params->best_body_idx[0];
      *selected_tile_idx = best_body;
      *selected_tile_n = params->exemplar_tile_shape[0];
      cudaGraphSetConditional(switch_handle, static_cast<unsigned int>(best_body));
    }
    return;
  }

  // Policy-specific fast path (e.g. kNN single-exemplar shortcut).
  if (DecisionPolicy::fast_path_skip(params, selected_tile_idx, selected_tile_n, switch_handle)) {
    return;
  }

  BlockHistogram(hist_storage).InitHistogram(cub_counts);
  __syncthreads();

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (UsePdl) cudaGridDependencySynchronize();
#endif

  // Vectorized histogram: each thread reads 8 int32s as two int4 loads.
  // Falls back to scalar loads when topk_ids is not 16-byte aligned.
  constexpr int kItemsPerBlockIter =
      kKnnSplitHistogramBlockThreads * kKnnSplitHistogramItemsPerThread;
  int const tile_stride = gridDim.x * kItemsPerBlockIter;
  bool const aligned = ((reinterpret_cast<uintptr_t>(topk_ids) & 0xF) == 0);
  if (aligned) {
    for (int base = blockIdx.x * kItemsPerBlockIter; base < num_elements; base += tile_stride) {
      unsigned short items[kKnnSplitHistogramItemsPerThread];
      int const block_lane = threadIdx.x;
      int const lane_base0 = base + block_lane * 4;
      int const lane_base1 = base + (block_lane + kKnnSplitHistogramBlockThreads) * 4;
      int4 v0, v1;
      if (lane_base0 + 3 < num_elements) {
        v0 = *reinterpret_cast<const int4*>(&topk_ids[lane_base0]);
      } else {
        v0.x = (lane_base0 + 0 < num_elements) ? topk_ids[lane_base0 + 0] : -1;
        v0.y = (lane_base0 + 1 < num_elements) ? topk_ids[lane_base0 + 1] : -1;
        v0.z = (lane_base0 + 2 < num_elements) ? topk_ids[lane_base0 + 2] : -1;
        v0.w = (lane_base0 + 3 < num_elements) ? topk_ids[lane_base0 + 3] : -1;
      }
      if (lane_base1 + 3 < num_elements) {
        v1 = *reinterpret_cast<const int4*>(&topk_ids[lane_base1]);
      } else {
        v1.x = (lane_base1 + 0 < num_elements) ? topk_ids[lane_base1 + 0] : -1;
        v1.y = (lane_base1 + 1 < num_elements) ? topk_ids[lane_base1 + 1] : -1;
        v1.z = (lane_base1 + 2 < num_elements) ? topk_ids[lane_base1 + 2] : -1;
        v1.w = (lane_base1 + 3 < num_elements) ? topk_ids[lane_base1 + 3] : -1;
      }
      auto enc = [&](int32_t raw) {
        int local_id =
            da_knn_decode_local_id(raw, local_expert_offset, max_local_expert, num_local_experts);
        return static_cast<unsigned short>(local_id >= 0 ? local_id : kMaxKnnExperts);
      };
      items[0] = enc(v0.x);
      items[1] = enc(v0.y);
      items[2] = enc(v0.z);
      items[3] = enc(v0.w);
      items[4] = enc(v1.x);
      items[5] = enc(v1.y);
      items[6] = enc(v1.z);
      items[7] = enc(v1.w);
      BlockHistogram(hist_storage).Composite(items, cub_counts);
      __syncthreads();
    }
  } else {
    // Scalar fallback for unaligned input.
    for (int base = blockIdx.x * kItemsPerBlockIter; base < num_elements; base += tile_stride) {
      unsigned short items[kKnnSplitHistogramItemsPerThread];
#pragma unroll
      for (int item = 0; item < kKnnSplitHistogramItemsPerThread; ++item) {
        int const idx = base + threadIdx.x + item * blockDim.x;
        int local_id = -1;
        if (idx < num_elements) {
          local_id = da_knn_decode_local_id(topk_ids[idx], local_expert_offset, max_local_expert,
                                            num_local_experts);
        }
        items[item] = static_cast<unsigned short>(local_id >= 0 ? local_id : kMaxKnnExperts);
      }
      BlockHistogram(hist_storage).Composite(items, cub_counts);
      __syncthreads();
    }
  }

  for (int i = threadIdx.x; i < num_local_experts; i += blockDim.x) {
    atomicAdd(&counts[i], static_cast<int32_t>(cub_counts[i]));
  }

  __syncthreads();

  // atomicInc wraps to 0 at gridDim.x-1, self-resetting for next replay.
  __shared__ bool is_last_block;
  if (threadIdx.x == 0) {
    unsigned int ticket = atomicInc(d_block_done, gridDim.x - 1);
    is_last_block = (ticket == gridDim.x - 1);
  }
  __syncthreads();

  if (is_last_block) {
    DecisionPolicy::decide_last_block(params, counts, num_local_experts, num_elements,
                                      selected_tile_idx, selected_tile_n, switch_handle);
  }
}

// ---------------------------------------------------------------------------
// k-NN decision kernel (graph variant, single-block). Builds an in-block
// histogram, then compact+sort+cosine-sim+argmax to select a tile.
// Exemplars are pre-normalized; the argmax uses raw inner products since
// ||counts|| is constant across exemplars. One warp per exemplar dot product.
// Launch: 1 block, 256 threads.
// ---------------------------------------------------------------------------
__global__ void da_knn_select_tile_graph_kernel(const int32_t* __restrict__ topk_ids,
                                                int num_elements,
                                                const DAKnnParams* __restrict__ params,
                                                int32_t* __restrict__ selected_tile_idx,
                                                int32_t* __restrict__ selected_tile_n,
                                                cudaGraphConditionalHandle switch_handle) {
  __shared__ int counts_int[kMaxKnnExperts];
  __shared__ unsigned int cub_counts[kKnnSelectorHistogramBins];
  __shared__ float sims[kMaxExemplars];

  const int num_local_experts = params->num_local_experts;
  const int local_expert_offset = params->local_expert_offset;
  const int max_local_expert = local_expert_offset + num_local_experts;
  const int num_exemplars = params->num_exemplars;

  if (num_exemplars == 1) {
    if (threadIdx.x == 0) {
      int best_body = params->best_body_idx[0];
      *selected_tile_idx = best_body;
      *selected_tile_n = params->exemplar_tile_shape[0];
      cudaGraphSetConditional(switch_handle, static_cast<unsigned int>(best_body));
    }
    return;
  }

  using BlockHistogram = cub::BlockHistogram<unsigned short, kKnnSelectorBlockThreads,
                                             kKnnSelectorHistogramItemsPerThread,
                                             kKnnSelectorHistogramBins, cub::BLOCK_HISTO_ATOMIC>;
  __shared__ typename BlockHistogram::TempStorage hist_storage;
  BlockHistogram(hist_storage).InitHistogram(cub_counts);
  __syncthreads();

  for (int base = 0; base < num_elements;
       base += blockDim.x * kKnnSelectorHistogramItemsPerThread) {
    unsigned short items[kKnnSelectorHistogramItemsPerThread];
#pragma unroll
    for (int item = 0; item < kKnnSelectorHistogramItemsPerThread; ++item) {
      int const idx = base + threadIdx.x + item * blockDim.x;
      int local_id = -1;
      if (idx < num_elements) {
        local_id = da_knn_decode_local_id(topk_ids[idx], local_expert_offset, max_local_expert,
                                          num_local_experts);
      }
      items[item] = static_cast<unsigned short>(local_id >= 0 ? local_id : kMaxKnnExperts);
    }
    BlockHistogram(hist_storage).Composite(items, cub_counts);
    __syncthreads();
  }

  int const sort_items = da_knn_compact_counts_for_sort(counts_int, cub_counts, num_local_experts);

  da_knn_finish_selection_from_counts(counts_int, sims, params->exemplar_norm, params,
                                      selected_tile_idx, selected_tile_n, switch_handle,
                                      num_elements, sort_items);
}

}  // namespace da_heuristic
}  // namespace flashinfer
