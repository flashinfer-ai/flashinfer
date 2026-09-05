/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Fused decode context-parallel all-to-all + LSE-weighted reduce.
//
// Under decode context parallelism the KV cache is sharded across a team. Every
// rank attends the query heads against its own KV shard and produces a partial
// output plus the log-sum-exp of the softmax denominator over that shard. The
// merge is a standard softmax-weighted reduction: gathering the per-rank LSEs
// for one (token, head) into a vector l,
//
//   out = sum_r softmax(l)_r * o_r
//   lse = logsumexp(l)
//
// i.e. the LSEs act as the logits and the softmax is over the *rank* axis, not
// the key axis. It is evaluated in the max-shifted stable form, which is
// identical to softmax(l) because softmax is shift invariant. With base-2 LSE
// (FlashInfer MLA) the exponentials are base 2, giving softmax(l*ln2).
//
// The one deviation from softmax proper: if every shard reports -inf for a head
// (an empty KV range everywhere) softmax would be 0/0, so that case is defined
// to produce a zero output rather than NaN. NaN and +inf inputs fold to -inf
// before the softmax.
//
// The reduction is associative and commutative, so the order is free. This is
// the same merge flash-attention uses to combine its splits, with the splits
// being ranks rather than key blocks.
//
// Data movement uses the NCCL device API: each rank stores its per-destination
// slice straight into the destination's registered symmetric window
// (ncclGetLsaPointer), then publishes a system-scope release flag. Receivers
// acquire-wait on every source flag and use an unconditional cooperative-grid
// sync before blocks are reassigned to merge locally. That requires a single
// load/store-accessible (NVLink) domain; a team spanning several LSA domains
// needs a hierarchical algorithm which is not implemented here.

#ifndef FLASHINFER_COMM_DCP_LSE_REDUCE_CUH_
#define FLASHINFER_COMM_DCP_LSE_REDUCE_CUH_

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <nccl_device.h>

namespace cg = cooperative_groups;

namespace flashinfer {
namespace comm {
namespace dcp {

// Two workspace slots are used alternately. A call cannot enter its receive
// phase until every rank has completed its sends, so by the time slot i is
// reused all ranks have completed the receive phase from two calls earlier.
//
// Each slot also has one readiness word per source rank. The sender publishes
// epoch + 1 with a system-scope release store after its payload is visible; the
// receiver waits with system-scope acquire loads before entering the merge.
constexpr int kNumSlots = 2;

// Merging more than this many ranks per block would blow the shared-memory LSE
// staging area; workspace creation rejects larger teams.
constexpr int kMaxRanks = 64;

constexpr int kFusedBlockSize = 128;
constexpr size_t kMetadataBytes = 16;
constexpr size_t kWorkspaceAlignment = 16;

inline constexpr size_t SignalRegionBytes(int nranks) {
  return kNumSlots * static_cast<size_t>(nranks) * sizeof(uint32_t);
}

inline constexpr size_t PayloadOffset(int nranks) {
  return (kMetadataBytes + SignalRegionBytes(nranks) + kWorkspaceAlignment - 1) /
         kWorkspaceAlignment * kWorkspaceAlignment;
}

__device__ __forceinline__ void store_release_sys(uint32_t* ptr, uint32_t value) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" : : "l"(ptr), "r"(value) : "memory");
}

__device__ __forceinline__ uint32_t load_acquire_sys(const uint32_t* ptr) {
  uint32_t value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(ptr) : "memory");
  return value;
}

// ---------------------------------------------------------------------------
// Element conversion. Accumulation is always fp32 regardless of storage type;
// the partials are pre-normalisation, so reducing in fp16 would lose more than
// the final rounding.
// ---------------------------------------------------------------------------
template <typename T>
struct Conv;

template <>
struct Conv<float> {
  __device__ static inline float to_f32(float v) { return v; }
  __device__ static inline float from_f32(float v) { return v; }
};

template <>
struct Conv<__half> {
  __device__ static inline float to_f32(__half v) { return __half2float(v); }
  __device__ static inline __half from_f32(float v) { return __float2half(v); }
};

template <>
struct Conv<__nv_bfloat16> {
  __device__ static inline float to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }
  __device__ static inline __nv_bfloat16 from_f32(float v) { return __float2bfloat16(v); }
};

// Sanitise an incoming LSE. A shard that contributed nothing reports -inf; some
// backends emit NaN or +inf instead, and both must fold to "no weight" rather
// than poisoning the whole merge.
__device__ static inline float sanitize_lse(float l) {
  return (isnan(l) || l == INFINITY) ? -INFINITY : l;
}

// Cooperative fused send, receive synchronization, and LSE merge.
//
// The source is already sliced per destination, which is the layout the DCP
// all-to-all convention uses:
//   partial_o   [num_tokens, local_heads, nranks, head_dim]
//   partial_lse [num_tokens, local_heads, nranks]
// The destination layout inside the peer's slot is
//   [src_rank][token][local_head][*]
//
// The grid is persistent and cooperatively launched:
//   1. block 0 selects the graph-safe device epoch; grid sync;
//   2. blocks publish complete destination payloads and release-store flags;
//   3. block 0 acquire-waits for every source flag; unconditional grid sync;
//   4. blocks are reassigned to output rows and perform the LSE merge.
template <typename T, bool BASE_E>
__global__ void FusedKernel(const T* __restrict__ partial_o, const float* __restrict__ partial_lse,
                            unsigned char* __restrict__ workspace, ncclWindow_t window,
                            size_t signal_window_offset, size_t out_region_window_offset,
                            size_t out_region_local_offset, size_t slot_out_bytes,
                            size_t lse_region_window_offset, size_t lse_region_local_offset,
                            size_t slot_lse_bytes, T* __restrict__ combined_out, int rank,
                            int nranks, int num_tokens, int max_tokens, int local_heads,
                            int head_dim) {
  extern __shared__ float smem[];  // [nranks] merge weights
  cg::grid_group grid = cg::this_grid();
  auto* state = reinterpret_cast<uint32_t*>(workspace);
  auto* ready = reinterpret_cast<uint32_t*>(workspace + kMetadataBytes);

  // Keep epoch selection on device so CUDA graph replay advances the slot.
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const uint32_t epoch = state[0];
    state[1] = epoch;
    state[0] = epoch + 1;
  }
  grid.sync();

  const uint32_t epoch = state[1];
  const uint32_t signal_value = epoch + 1;
  const uint32_t slot = epoch % kNumSlots;
  const int num_entries = num_tokens * local_heads;
  const size_t slot_out_window_offset = out_region_window_offset + slot * slot_out_bytes;
  const size_t slot_lse_window_offset = lse_region_window_offset + slot * slot_lse_bytes;

  // Each destination is owned by one block. That block writes every row before
  // publishing this source rank's readiness flag to the destination.
  for (int dst = blockIdx.x; dst < nranks; dst += gridDim.x) {
    for (int tile = 0; tile < num_entries; ++tile) {
      const int token = tile / local_heads;
      const int local_head = tile - token * local_heads;
      const size_t src_base = (((static_cast<size_t>(tile) * nranks + dst) * head_dim));
      const size_t dst_row =
          ((static_cast<size_t>(rank) * max_tokens + token) * local_heads + local_head);
      T* dst_out = reinterpret_cast<T*>(
          ncclGetLsaPointer(window, slot_out_window_offset + dst_row * head_dim * sizeof(T), dst));
      float* dst_lse = reinterpret_cast<float*>(
          ncclGetLsaPointer(window, slot_lse_window_offset + dst_row * sizeof(float), dst));

      constexpr int kVec = static_cast<int>(sizeof(int4) / sizeof(T));
      const int4* src4 = reinterpret_cast<const int4*>(partial_o + src_base);
      int4* dst4 = reinterpret_cast<int4*>(dst_out);
      const int n4 = head_dim / kVec;
      for (int d = threadIdx.x; d < n4; d += blockDim.x) {
        dst4[d] = src4[d];
      }
      if (threadIdx.x == 0) {
        *dst_lse = partial_lse[static_cast<size_t>(tile) * nranks + dst];
      }
      __syncthreads();
    }

    // Conservatively flush every writer before the signaling thread publishes
    // readiness. This can be relaxed after compute-sanitizer and perf testing.
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
      auto* dst_ready = reinterpret_cast<uint32_t*>(ncclGetLsaPointer(
          window, signal_window_offset + (slot * nranks + rank) * sizeof(uint32_t), dst));
      store_release_sys(dst_ready, signal_value);
    }
    __syncthreads();
  }

  // Different threads wait for different source ranks. The grid sync must be
  // unconditional: after it, blocks are reassigned to merge work, so every
  // block must observe that all source payloads are ready. This is the same
  // invariant as NCCL EP's staged LL combine receive path.
  if (blockIdx.x == 0) {
    for (int src = threadIdx.x; src < nranks; src += blockDim.x) {
      const uint32_t* src_ready = ready + slot * nranks + src;
      while (load_acquire_sys(src_ready) != signal_value) {
        __nanosleep(64);
      }
    }
  }
  grid.sync();

  const T* recv_out =
      reinterpret_cast<const T*>(workspace + out_region_local_offset + slot * slot_out_bytes);
  const float* recv_lse =
      reinterpret_cast<const float*>(workspace + lse_region_local_offset + slot * slot_lse_bytes);

  __shared__ float s_denom;
  for (int entry = blockIdx.x; entry < num_entries; entry += gridDim.x) {
    const int token = entry / local_heads;
    const int head = entry - token * local_heads;
    if (threadIdx.x < nranks) {
      const int r = threadIdx.x;
      const size_t idx = ((static_cast<size_t>(r) * max_tokens + token) * local_heads + head);
      smem[r] = sanitize_lse(recv_lse[idx]);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      float m = -INFINITY;
      for (int r = 0; r < nranks; ++r) m = fmaxf(m, smem[r]);
      if (m == -INFINITY) m = 0.0f;
      float denom = 0.0f;
      for (int r = 0; r < nranks; ++r) {
        const float e = BASE_E ? __expf(smem[r] - m) : exp2f(smem[r] - m);
        smem[r] = e;
        denom += e;
      }
      s_denom = denom;
    }
    __syncthreads();

    const float inv = (s_denom == 0.0f) ? 0.0f : (1.0f / s_denom);
    const size_t out_base = static_cast<size_t>(entry) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      float acc = 0.0f;
      for (int r = 0; r < nranks; ++r) {
        const size_t idx =
            (((static_cast<size_t>(r) * max_tokens + token) * local_heads + head) * head_dim + d);
        acc += Conv<T>::to_f32(recv_out[idx]) * (smem[r] * inv);
      }
      combined_out[out_base + d] = Conv<T>::from_f32(acc);
    }
    __syncthreads();
  }
}

}  // namespace dcp
}  // namespace comm
}  // namespace flashinfer

#endif  // FLASHINFER_COMM_DCP_LSE_REDUCE_CUH_
