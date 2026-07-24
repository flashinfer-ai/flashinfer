/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Adapted from ThunderKittens' NVLink all-to-all kernel:
 * https://github.com/HazyResearch/ThunderKittens/blob/main/kernels/parallel/all_to_all/all_to_all.cu
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

// Fused-transpose Ulysses all-to-all over NVLink P2P (CUDA IPC).
//
// Implements the head-scatter / sequence-gather collective used by Ulysses
// context parallelism, with the layout permutation folded directly into the
// cross-GPU write addresses. It reuses the Signal / multi_gpu_barrier machinery
// from the (vLLM-derived) custom all-reduce kernel for inter-GPU sync.
//
// Push model: each rank reads from its *local* input tensor (no registration
// required) and writes the head block destined for each peer directly into that
// peer's IPC-registered output staging buffer. Only the output staging buffers
// and the signal buffers need to be IPC-shared.
//
// head_dim == 2 layout, uniform sequence splits. With
//   W        = ulysses world size
//   H_local  = H / W
//   S_global = S_local * W
//
//   mode == 0 (input  a2a): [B, S_local, H,       D] -> [B, S_global, H_local, D]
//       y_r[b, j*S_local + s, hl, d] = x_j[b, s, r*H_local + hl, d]
//   mode == 1 (output a2a): [B, S_global, H_local, D] -> [B, S_local, H,       D]
//       out_j[b, s, r*H_local + hl, d] = u_r[b, j*S_local + s, hl, d]
//
// In both modes the unit of transfer is a contiguous (H_local * D) block, so
// every cross-GPU store is fully coalesced.

#ifndef FLASHINFER_COMM_ULYSSES_ALL_TO_ALL_CUH_
#define FLASHINFER_COMM_ULYSSES_ALL_TO_ALL_CUH_

#include <cstdint>

#include "flashinfer/comm/vllm_custom_all_reduce.cuh"

namespace flashinfer {
namespace comm {
namespace ulysses {

// Reuse the (vLLM-derived) IPC signal / barrier machinery from custom all-reduce.
using vllm::kMaxBlocks;
using vllm::multi_gpu_barrier;
using vllm::RankData;
using vllm::RankSignals;
using vllm::Signal;

constexpr int kUlyssesThreads = 512;

// Handle holding the IPC-shared output staging buffers and signals for one
// Ulysses group. It does not own any device memory; buffers are passed in from
// Python. Passed by pointer (as an int64 handle) across the FFI boundary.
class UlyssesA2A {
 public:
  int rank_;
  int world_size_;
  bool full_nvlink_;

  RankSignals sg_;
  Signal* self_sg_;
  // ptrs[r] points to rank r's output staging buffer (device pointer, opened
  // via IPC for remote ranks). Passed to the kernel by value.
  RankData out_ptrs_;
  // Convenience copy of this rank's own output staging buffer base pointer.
  void* local_out_buf_;

  UlyssesA2A(Signal** signals, void** out_bufs, int rank, int world_size, bool full_nvlink)
      : rank_(rank), world_size_(world_size), full_nvlink_(full_nvlink), self_sg_(signals[rank]) {
    for (int i = 0; i < world_size_; i++) {
      sg_.signals[i] = signals[i];
      out_ptrs_.ptrs[i] = out_bufs[i];
    }
    local_out_buf_ = out_bufs[rank];
  }
};

// Shared movement body for the fused-transpose all-to-all (no barriers).
//
// Rows are ordered as ((b * W + peer) * S_local + s), so consecutive rows share
// the same (batch, peer) and are therefore contiguous on the "gather" side of
// the transpose. Each block is assigned a *contiguous* slab of rows (rather than
// an interleaved grid-stride), and threads are flattened over all 16B vector
// units in that slab. This makes consecutive lanes/iterations issue back-to-back
// addresses to a single peer buffer, so the remote NVLink writes coalesce into
// large bursts instead of the tiny (H_local*D) scattered writes the naive
// mapping produced (which collapsed badly at large world sizes where H_local is
// small).
template <typename T, int NGPUS, int MODE>
__device__ __forceinline__ void ulysses_a2a_move(const T* __restrict__ local_in, RankData out_ptrs,
                                                 int rank, int B, int S_local, int H_local, int D) {
  static_assert(MODE == 0 || MODE == 1, "MODE must be 0 or 1");
  const int W = NGPUS;
  const int64_t H = static_cast<int64_t>(H_local) * W;
  const int64_t S_global = static_cast<int64_t>(S_local) * W;
  const int64_t block_len = static_cast<int64_t>(H_local) * D;  // elements/row
  const int64_t num_rows = static_cast<int64_t>(B) * W * S_local;

  // 16B-vectorized fast path when every row is 16B aligned (the common case:
  // contiguous bf16/fp16/fp32 tensors with block_len * sizeof(T) % 16 == 0).
  using Vec = int4;
  constexpr int kVecBytes = sizeof(Vec);
  const int64_t row_bytes = block_len * static_cast<int64_t>(sizeof(T));
  const bool vec_ok =
      (row_bytes % kVecBytes) == 0 && (reinterpret_cast<uintptr_t>(local_in) % kVecBytes) == 0;

  // Contiguous slab of rows for this block.
  const int64_t rows_per_block = (num_rows + gridDim.x - 1) / gridDim.x;
  const int64_t row_lo = static_cast<int64_t>(blockIdx.x) * rows_per_block;
  int64_t row_hi = row_lo + rows_per_block;
  if (row_hi > num_rows) row_hi = num_rows;
  if (row_lo >= row_hi) return;

  const int tid = threadIdx.x;
  const int nthr = blockDim.x;

  // Decode (b, peer, s) and compute src/dst element offsets for a given row.
  auto offsets = [&](int64_t row, int64_t& src_off, int64_t& dst_off) {
    const int64_t s = row % S_local;
    const int64_t tmp = row / S_local;
    const int64_t peer = tmp % W;
    const int64_t b = tmp / W;
    if constexpr (MODE == 0) {
      src_off = ((b * S_local + s) * H + peer * H_local) * D;
      dst_off = (b * S_global + static_cast<int64_t>(rank) * S_local + s) * block_len;
    } else {
      src_off = (b * S_global + peer * S_local + s) * block_len;
      dst_off = ((b * S_local + s) * H + static_cast<int64_t>(rank) * H_local) * D;
    }
    return peer;
  };

  if (vec_ok) {
    const int64_t units_per_row = row_bytes / kVecBytes;
    const int64_t total_units = (row_hi - row_lo) * units_per_row;
    for (int64_t u = tid; u < total_units; u += nthr) {
      const int64_t local_row = u / units_per_row;
      const int64_t unit = u - local_row * units_per_row;
      const int64_t row = row_lo + local_row;
      int64_t src_off, dst_off;
      const int64_t peer = offsets(row, src_off, dst_off);
      const Vec* s4 = reinterpret_cast<const Vec*>(local_in + src_off);
      Vec* d4 = reinterpret_cast<Vec*>((T*)out_ptrs.ptrs[peer] + dst_off);
      d4[unit] = s4[unit];
    }
  } else {
    // Scalar fallback (unaligned / odd shapes).
    for (int64_t row = row_lo; row < row_hi; ++row) {
      int64_t src_off, dst_off;
      const int64_t peer = offsets(row, src_off, dst_off);
      const T* s_ptr = local_in + src_off;
      T* d_ptr = (T*)out_ptrs.ptrs[peer] + dst_off;
      for (int64_t i = tid; i < block_len; i += nthr) {
        d_ptr[i] = s_ptr[i];
      }
    }
  }
}

// The transfer mode is a compile-time template parameter so the address math
// specializes and the coalesced slab decomposition is fully unrolled per mode.
template <typename T, int NGPUS, int MODE>
__global__ void __launch_bounds__(kUlyssesThreads, 1)
    ulysses_a2a_kernel(const T* __restrict__ local_in, RankData out_ptrs, RankSignals sg,
                       Signal* self_sg, int rank, int B, int S_local, int H_local, int D) {
  // Ensure every rank has entered before we start writing into peer buffers.
  multi_gpu_barrier<NGPUS, true>(sg, self_sg, rank);
  ulysses_a2a_move<T, NGPUS, MODE>(local_in, out_ptrs, rank, B, S_local, H_local, D);
  // Release-acquire barrier so all peer writes are visible before any rank
  // reads its own (now complete) output staging buffer.
  multi_gpu_barrier<NGPUS, false, true>(sg, self_sg, rank);
}

}  // namespace ulysses
}  // namespace comm
}  // namespace flashinfer

#endif  // FLASHINFER_COMM_ULYSSES_ALL_TO_ALL_CUH_
