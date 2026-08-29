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
// to produce a zero output and a -inf combined LSE rather than NaN. NaN and
// +inf inputs fold to -inf before the softmax.
//
// The reduction is associative and commutative, so the order is free. This is
// the same merge flash-attention uses to combine its splits, with the splits
// being ranks rather than key blocks.
//
// Data movement uses the NCCL device API: each rank stores its per-destination
// slice straight into the destination's registered symmetric window
// (ncclGetLsaPointer), an LSA barrier makes those stores visible, then every
// rank merges locally. That requires a single load/store-accessible (NVLink)
// domain; a team spanning several LSA domains needs a hierarchical algorithm
// which is not implemented here.

#ifndef FLASHINFER_COMM_DCP_LSE_REDUCE_CUH_
#define FLASHINFER_COMM_DCP_LSE_REDUCE_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <nccl_device.h>

namespace flashinfer {
namespace comm {
namespace dcp {

// Two workspace slots, used alternately, so one barrier per call is enough.
//
// With a single slot: rank r issues its put for call i+1 as soon as it clears
// barrier i, but peer p only issues its merge for call i *after* that same
// barrier -- r would overwrite data p has not read. Alternating puts the
// previous reader two calls back, and on p's stream that read is ordered
// before the barrier r had to clear to reach call i+2.
//
// The alternative, two barriers per call, costs about as much as the fused
// path saves.
constexpr int kNumSlots = 2;

// Merging more than this many ranks per block would blow the shared-memory LSE
// staging area; workspace creation rejects larger teams.
constexpr int kMaxRanks = 64;

constexpr int kPutBlockSize = 128;
constexpr int kMergeBlockSize = 128;
constexpr size_t kMetadataBytes = 16;

// Select the alternating workspace slot on device so CUDA graph replay
// advances the epoch rather than baking a host-side slot value into the graph.
// state[0] is the monotonically increasing epoch and state[1] is the slot used
// by the kernels in this invocation.
__global__ void SelectSlotKernel(uint32_t* state) {
  if (threadIdx.x == 0) {
    const uint32_t epoch = state[0];
    state[1] = epoch % kNumSlots;
    state[0] = epoch + 1;
  }
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

// ---------------------------------------------------------------------------
// Put: publish this rank's partial for every destination directly into that
// destination's workspace slot.
//
// The source is already sliced per destination, which is the layout the DCP
// all-to-all convention uses:
//   partial_o   [num_tokens, local_heads, nranks, head_dim]
//   partial_lse [num_tokens, local_heads, nranks]
// The destination layout inside the peer's slot is
//   [src_rank][token][local_head][*]
//
// Grid  (num_tokens * local_heads, nranks)
// Block 128 threads, each striding over head_dim.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void PutPackedKernel(const T* __restrict__ partial_o,
                                const float* __restrict__ partial_lse, ncclWindow_t window,
                                size_t out_region_byte_offset, size_t slot_out_bytes,
                                size_t lse_region_byte_offset, size_t slot_lse_bytes,
                                const uint32_t* state, int rank, int nranks, int num_tokens,
                                int max_tokens, int local_heads, int head_dim) {
  const int tile = blockIdx.x;  // token * local_heads + local head
  const int dst = blockIdx.y;
  const int token = tile / local_heads;
  const int local_head = tile - token * local_heads;
  if (token >= num_tokens || dst >= nranks) return;

  const size_t src_base =
      ((((size_t)token * local_heads + local_head) * nranks + dst) * head_dim);
  const unsigned int slot = state[1];
  const size_t slot_out_byte_offset = out_region_byte_offset + slot * slot_out_bytes;
  const size_t slot_lse_byte_offset = lse_region_byte_offset + slot * slot_lse_bytes;
  const size_t dst_row = (((size_t)rank * max_tokens + token) * local_heads + local_head);

  T* dst_out =
      (T*)ncclGetLsaPointer(window, slot_out_byte_offset + dst_row * head_dim * sizeof(T), dst);
  float* dst_lse =
      (float*)ncclGetLsaPointer(window, slot_lse_byte_offset + dst_row * sizeof(float), dst);

  // Widen to 16-byte stores when the row allows it. These go over NVLink, so
  // the difference between 2-byte and 16-byte transactions is large. Rows are
  // head_dim*sizeof(T) bytes and both bases are at least 16-byte aligned, so
  // divisibility of the row is the only condition to check.
  constexpr int kVec = (int)(sizeof(int4) / sizeof(T));
  if (head_dim % kVec == 0) {
    const int4* src4 = reinterpret_cast<const int4*>(partial_o + src_base);
    int4* dst4 = reinterpret_cast<int4*>(dst_out);
    const int n4 = head_dim / kVec;
    for (int d = threadIdx.x; d < n4; d += blockDim.x) {
      dst4[d] = src4[d];
    }
  } else {
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      dst_out[d] = partial_o[src_base + d];
    }
  }
  if (threadIdx.x == 0) {
    *dst_lse = partial_lse[((size_t)token * local_heads + local_head) * nranks + dst];
  }
}

// ---------------------------------------------------------------------------
// Barrier: make every peer's puts visible before anyone merges.
//
// A single CTA. Kernel boundaries give the intra-rank ordering (all put CTAs
// have retired), and the LSA barrier with acq_rel gives the inter-rank
// ordering. Keeping it separate rather than fusing a signal into the put and a
// spin into the merge costs one launch; it is also the version whose
// correctness does not depend on getting a hand-rolled protocol right.
// ---------------------------------------------------------------------------
__global__ void BarrierKernel(ncclDevComm dev_comm) {
  ncclLsaBarrierSession<ncclCoopCta> bar(ncclCoopCta(), dev_comm, ncclTeamLsa(dev_comm),
                                         dev_comm.lsaBarrier, /*index=*/0);
  bar.sync(ncclCoopCta(), cuda::memory_order_acq_rel);
}

// ---------------------------------------------------------------------------
// Merge: softmax-weighted reduction of the nranks partials this rank received.
//
// Grid  (num_tokens, local_heads)
// Block 128 threads.
//
// One pass over the payload. The LSEs are staged in shared memory and the
// weights derived once per block, rather than re-read per pass.
// ---------------------------------------------------------------------------
template <typename T, bool BASE_E>
__global__ void MergeKernel(const unsigned char* __restrict__ workspace,
                            size_t out_region_byte_offset, size_t slot_out_bytes,
                            size_t lse_region_byte_offset, size_t slot_lse_bytes,
                            const uint32_t* state, T* __restrict__ combined_out, int nranks,
                            int num_tokens, int max_tokens, int local_heads, int head_dim) {
  extern __shared__ float smem[];  // [nranks] merge weights

  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const unsigned int slot = state[1];
  const T* recv_out = reinterpret_cast<const T*>(
      workspace + out_region_byte_offset + slot * slot_out_bytes);
  const float* recv_lse = reinterpret_cast<const float*>(
      workspace + lse_region_byte_offset + slot * slot_lse_bytes);

  // Stage the LSEs, then let one thread reduce them. nranks is small (<= 64),
  // so a single-threaded reduction over shared memory is cheaper than a tree
  // and keeps the ordering deterministic across launches.
  for (int r = threadIdx.x; r < nranks; r += blockDim.x) {
    const size_t idx = (((size_t)r * max_tokens + token) * local_heads + head);
    smem[r] = sanitize_lse(recv_lse[idx]);
  }
  __syncthreads();

  __shared__ float s_denom;
  if (threadIdx.x == 0) {
    float m = -INFINITY;
    for (int r = 0; r < nranks; ++r) m = fmaxf(m, smem[r]);
    // Every shard was empty: emit zeros and a -inf LSE, matching what a single
    // shard with no contribution would produce.
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

  const float denom = s_denom;
  const float inv = (denom == 0.0f) ? 0.0f : (1.0f / denom);

  const size_t out_base = ((size_t)token * local_heads + head) * head_dim;
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int r = 0; r < nranks; ++r) {
      const size_t idx =
          ((((size_t)r * max_tokens + token) * local_heads + head) * head_dim + d);
      acc += Conv<T>::to_f32(recv_out[idx]) * (smem[r] * inv);
    }
    combined_out[out_base + d] = Conv<T>::from_f32(acc);
  }

}

}  // namespace dcp
}  // namespace comm
}  // namespace flashinfer

#endif  // FLASHINFER_COMM_DCP_LSE_REDUCE_CUH_
