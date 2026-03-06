/*
 * Copyright (c) 2025 by FlashInfer team.
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
#pragma once
#include <cuda_runtime.h>

#include <cub/device/device_scan.cuh>

namespace flashinfer {
namespace mamba {

struct SumOp {
  __device__ __host__ __forceinline__ int32_t operator()(int32_t a, int32_t b) const {
    return a + b;
  }
};

/*!
 * \brief Binary search for lower_bound on the virtual sorted array
 *        seq_id[i] = seq_idx[chunk_indices[i] * chunk_size + chunk_offsets[i]].
 *
 * Returns the smallest i in [0, num_logical_chunks) such that seq_id[i] >= target,
 * or num_logical_chunks if no such i exists.
 */
template <typename SeqIdxT>
__device__ __forceinline__ int lower_bound_seq_id(const SeqIdxT* __restrict__ seq_idx,
                                                  const int32_t* __restrict__ chunk_indices,
                                                  const int32_t* __restrict__ chunk_offsets,
                                                  int chunk_size, int num_logical_chunks,
                                                  int target) {
  int lo = 0, hi = num_logical_chunks;
  while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    int seq_id = static_cast<int>(seq_idx[chunk_indices[mid] * chunk_size + chunk_offsets[mid]]);
    if (seq_id < target) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// ---------------------------------------------------------------------------
// Single-block kernel: each thread computes one element independently via
// binary search.  Covers all practical cases (num_seqs <= 1024).
// ---------------------------------------------------------------------------

template <typename SeqIdxT>
__global__ void SeqChunkCumsumKernel(const SeqIdxT* __restrict__ seq_idx,
                                     const int32_t* __restrict__ chunk_indices,
                                     const int32_t* __restrict__ chunk_offsets,
                                     int32_t* __restrict__ output, int chunk_size,
                                     int num_logical_chunks, int num_seqs) {
  int s = threadIdx.x;
  if (s <= num_seqs) {
    if (s == num_seqs) {
      output[s] = num_logical_chunks;
    } else {
      output[s] = lower_bound_seq_id(seq_idx, chunk_indices, chunk_offsets, chunk_size,
                                     num_logical_chunks, s);
    }
  }
}

// ---------------------------------------------------------------------------
// Multi-block kernel with decoupled lookback.
// Uses cub::ScanTileState for inter-block communication.
// Each block handles TILE_SIZE sequences; each thread computes its count
// via binary search, then a block-level exclusive scan + lookback produces
// the global prefix.
// ---------------------------------------------------------------------------

static constexpr int TILE_SIZE = 256;

template <typename SeqIdxT>
__global__ void SeqChunkCumsumKernelMultiBlock(
    const SeqIdxT* __restrict__ seq_idx, const int32_t* __restrict__ chunk_indices,
    const int32_t* __restrict__ chunk_offsets, int32_t* __restrict__ output,
    cub::ScanTileState<int32_t> tile_state, int chunk_size, int num_logical_chunks, int num_seqs) {
  using ScanOp = SumOp;
  using TileState = cub::ScanTileState<int32_t>;
  using PrefixOp = cub::TilePrefixCallbackOp<int32_t, ScanOp, TileState>;
  using PrefixTempStorage = typename PrefixOp::TempStorage;

  __shared__ PrefixTempStorage prefix_storage;
  // +1 for the inclusive-to-exclusive conversion
  __shared__ int32_t smem[TILE_SIZE + 1];

  ScanOp scan_op{};
  PrefixOp prefix_op(tile_state, prefix_storage, scan_op);
  int tile_idx = prefix_op.GetTileIdx();
  int tid = threadIdx.x;
  int s = tile_idx * TILE_SIZE + tid;

  // Compute per-sequence chunk count via binary search
  int count = 0;
  if (s < num_seqs) {
    int lb = lower_bound_seq_id(seq_idx, chunk_indices, chunk_offsets, chunk_size,
                                num_logical_chunks, s);
    int ub = lower_bound_seq_id(seq_idx, chunk_indices, chunk_offsets, chunk_size,
                                num_logical_chunks, s + 1);
    count = ub - lb;
  }

  // Block-level inclusive scan (Hillis-Steele in shared memory)
  smem[tid] = count;
  __syncthreads();
  for (int stride = 1; stride < TILE_SIZE; stride <<= 1) {
    int val = (tid >= stride) ? smem[tid - stride] : 0;
    __syncthreads();
    smem[tid] += val;
    __syncthreads();
  }
  int inclusive = smem[tid];  // inclusive prefix sum within tile
  int tile_aggregate = smem[TILE_SIZE - 1];

  // Decoupled lookback
  int global_prefix;
  if (tile_idx == 0) {
    // First tile: no lookback needed
    if (tid == 0) {
      tile_state.SetInclusive(tile_idx, tile_aggregate);
    }
    global_prefix = 0;
  } else {
    // Only first warp performs the lookback
    int warp_id = tid / 32;
    if (warp_id == 0) {
      int exclusive_prefix = prefix_op(tile_aggregate);
      // Store in smem[TILE_SIZE] so all threads can read it
      if (tid == 0) {
        smem[TILE_SIZE] = exclusive_prefix;
      }
    }
    __syncthreads();
    global_prefix = smem[TILE_SIZE];
  }

  // Write output: exclusive prefix = inclusive - count + global_prefix
  if (s < num_seqs) {
    output[s] = global_prefix + inclusive - count;
  }
  // Last active thread writes the sentinel
  if (s == num_seqs - 1 || (s < num_seqs && s == num_seqs - 1)) {
    output[num_seqs] = global_prefix + inclusive;
  }
  // Handle case where num_seqs isn't a multiple of TILE_SIZE:
  // the last block's last active thread writes the final element.
  int last_seq_in_tile = min((tile_idx + 1) * TILE_SIZE, num_seqs) - 1;
  if (s == last_seq_in_tile) {
    output[num_seqs] = global_prefix + inclusive;
  }
}

template <class ScanTileStateT>
__global__ void InitTileStateKernel(ScanTileStateT tile_state, int num_tiles) {
  tile_state.InitializeStatus(num_tiles);
}

/*!
 * \brief Host-side launcher.  Picks single-block or multi-block path.
 */
template <typename SeqIdxT>
inline cudaError_t SeqChunkCumsumLauncher(const SeqIdxT* seq_idx, const int32_t* chunk_indices,
                                          const int32_t* chunk_offsets, int32_t* output,
                                          int chunk_size, int num_logical_chunks, int num_seqs,
                                          cudaStream_t stream = nullptr) {
  if (num_seqs + 1 <= 1024) {
    // Single-block path: each thread does an independent binary search
    int threads = ((num_seqs + 1) + 31) / 32 * 32;  // round up to warp
    SeqChunkCumsumKernel<<<1, threads, 0, stream>>>(seq_idx, chunk_indices, chunk_offsets, output,
                                                    chunk_size, num_logical_chunks, num_seqs);
  } else {
    // Multi-block path with decoupled lookback
    using TileState = cub::ScanTileState<int32_t>;

    int num_tiles = (num_seqs + TILE_SIZE - 1) / TILE_SIZE;

    // Allocate tile state
    std::size_t temp_bytes = 0;
    TileState::AllocationSize(num_tiles, temp_bytes);
    void* d_temp = nullptr;
    cudaError_t err = cudaMallocAsync(&d_temp, temp_bytes, stream);
    if (err != cudaSuccess) return err;

    TileState tile_state;
    tile_state.Init(num_tiles, d_temp, temp_bytes);

    // Initialize tile state
    int init_threads = 256;
    int init_blocks = (num_tiles + init_threads - 1) / init_threads;
    InitTileStateKernel<<<init_blocks, init_threads, 0, stream>>>(tile_state, num_tiles);

    // Launch scan kernel
    SeqChunkCumsumKernelMultiBlock<<<num_tiles, TILE_SIZE, 0, stream>>>(
        seq_idx, chunk_indices, chunk_offsets, output, tile_state, chunk_size, num_logical_chunks,
        num_seqs);

    cudaFreeAsync(d_temp, stream);
  }

  return cudaGetLastError();
}

}  // namespace mamba
}  // namespace flashinfer
