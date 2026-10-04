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

// Fused combine tail for the native BF16 SM90 push MegaMoE
// (sm90_bf16_bf16_bf16_push_cake).  One launch replaces the vendored
// `wait_combine` -> `combine_reduce` -> `ack` sequence on the receiving rank:
//
//   1. every block acquire-waits on this rank's per-source cdone cells for the
//      current round tag (same cell, tag, abort and timeout semantics as
//      `wait_combine_kernel`);
//   2. each thread reduces 8 consecutive columns of one token over the top-k
//      bf16 combine slots in fp32, in ascending k, skipping masked routes
//      (`topk_ids < 0`) instead of reading a zeroed slot, and rounds once to the
//      output dtype.  This is the arithmetic of `combine_reduce_kernel` (fp32
//      sum in k order, round-to-nearest store), so the output is bit-identical
//      while the per-round fill of the whole combine inbox becomes unnecessary:
//      every unmasked slot of a token below `num_tokens` is rewritten by the
//      senders' `combine_publish` each round, and a masked slot contributed an
//      exact +0 before;
//   3. the last block to finish resets the key scratch and this rank's pool
//      head and releases the ack cells, exactly as `ack_kernel`.
//
// The wire format (combine inbox layout, cells, tags, SlotMeta) is the vendored
// one and the sending rank keeps running the vendored `combine_publish`, so a
// peer on the unfused path interoperates with this kernel.

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "sm90_push_a2a.cuh"
#include "tvm_ffi_utils.h"

using tvm::ffi::TensorView;
using namespace flashinfer::sm90_push;

namespace {

constexpr int kMaxEpSize = 32;
constexpr int kThreads = 128;
constexpr int kColsPerThread = 8;  // one 16-byte bf16 vector per thread and slot
constexpr int kColsPerBlock = kThreads * kColsPerThread;

PushLayout build_layout(int64_t peer_bases, int64_t ep_size, int64_t rank,
                        int64_t num_local_experts, int64_t pool_rows, int64_t meta_rows,
                        int64_t bytes_per_row, int64_t hidden, int64_t top_k, int64_t t_cap,
                        int64_t pool_offset, int64_t pool_sc_offset, int64_t pool_meta_offset,
                        int64_t pool_head_offset, int64_t base_cells_offset,
                        int64_t count_cells_offset, int64_t cdone_cells_offset,
                        int64_t ack_cells_offset, int64_t combine_offset, int64_t cfp8_offset,
                        int64_t csc_offset) {
  PushLayout L;
  L.peer_bases = reinterpret_cast<uint64_t const*>(peer_bases);
  L.ep_size = static_cast<int>(ep_size);
  L.rank = static_cast<int>(rank);
  L.num_local_experts = static_cast<int>(num_local_experts);
  L.pool_rows = static_cast<int>(pool_rows);
  L.meta_rows = static_cast<int>(meta_rows);
  L.bytes_per_row = static_cast<int>(bytes_per_row);
  L.hidden = static_cast<int>(hidden);
  L.top_k = static_cast<int>(top_k);
  L.t_cap = static_cast<int>(t_cap);
  L.pool_offset = static_cast<uint64_t>(pool_offset);
  L.pool_sc_offset = static_cast<uint64_t>(pool_sc_offset);
  L.pool_meta_offset = static_cast<uint64_t>(pool_meta_offset);
  L.pool_head_offset = static_cast<uint64_t>(pool_head_offset);
  L.base_cells_offset = static_cast<uint64_t>(base_cells_offset);
  L.count_cells_offset = static_cast<uint64_t>(count_cells_offset);
  L.cdone_cells_offset = static_cast<uint64_t>(cdone_cells_offset);
  L.ack_cells_offset = static_cast<uint64_t>(ack_cells_offset);
  L.combine_offset = static_cast<uint64_t>(combine_offset);
  L.cfp8_offset = static_cast<uint64_t>(cfp8_offset);
  L.csc_offset = static_cast<uint64_t>(csc_offset);
  return L;
}

#define LAYOUT_PARAMS                                                                              \
  int64_t peer_bases, int64_t ep_size, int64_t rank, int64_t num_local_experts, int64_t pool_rows, \
      int64_t meta_rows, int64_t bytes_per_row, int64_t hidden, int64_t top_k, int64_t t_cap,      \
      int64_t pool_offset, int64_t pool_sc_offset, int64_t pool_meta_offset,                       \
      int64_t pool_head_offset, int64_t base_cells_offset, int64_t count_cells_offset,             \
      int64_t cdone_cells_offset, int64_t ack_cells_offset, int64_t combine_offset,                \
      int64_t cfp8_offset, int64_t csc_offset

#define LAYOUT_ARGS                                                                                \
  peer_bases, ep_size, rank, num_local_experts, pool_rows, meta_rows, bytes_per_row, hidden,       \
      top_k, t_cap, pool_offset, pool_sc_offset, pool_meta_offset, pool_head_offset,               \
      base_cells_offset, count_cells_offset, cdone_cells_offset, ack_cells_offset, combine_offset, \
      cfp8_offset, csc_offset

void check_layout(LAYOUT_PARAMS) {
  TVM_FFI_ICHECK(peer_bases != 0) << "sm90_cake_bf16: null peer_bases";
  TVM_FFI_ICHECK(ep_size >= 1 && ep_size <= kMaxEpSize)
      << "sm90_cake_bf16: ep_size " << ep_size << " out of [1, " << kMaxEpSize << "]";
  TVM_FFI_ICHECK(rank >= 0 && rank < ep_size) << "sm90_cake_bf16: bad rank " << rank;
  TVM_FFI_ICHECK(num_local_experts >= 1) << "sm90_cake_bf16: bad num_local_experts";
  TVM_FFI_ICHECK(hidden >= 128 && hidden % 128 == 0)
      << "sm90_cake_bf16: hidden must be a positive multiple of 128";
  TVM_FFI_ICHECK(top_k >= 1 && t_cap >= 1 && pool_rows >= 1 && meta_rows >= pool_rows)
      << "sm90_cake_bf16: bad capacity args";
  // bf16 combine wire: the bf16 inbox is present and the fp8 combine regions are empty
  TVM_FFI_ICHECK(combine_offset < cfp8_offset && cfp8_offset == csc_offset)
      << "sm90_cake_bf16: the fused combine tail requires a bf16 combine wire";
}

__device__ __forceinline__ void accumulate_bf16x8(const uint4& packed, float* acc) {
  __nv_bfloat162 h2[4];
  memcpy(h2, &packed, sizeof(packed));
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float2 f = __bfloat1622float2(h2[j]);
    acc[2 * j] += f.x;  // fp32 sum in slot order, as combine_reduce_kernel
    acc[2 * j + 1] += f.y;
  }
}

// RN, matching the per-element __float2bfloat16 store of combine_reduce_kernel
__device__ __forceinline__ void store_out8(__nv_bfloat16* out, const float* acc) {
  __nv_bfloat162 h2[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) h2[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
  uint4 packed;
  memcpy(&packed, h2, sizeof(packed));
  *reinterpret_cast<uint4*>(out) = packed;
}

__device__ __forceinline__ void store_out8(float* out, const float* acc) {
  float4* out4 = reinterpret_cast<float4*>(out);
  out4[0] = make_float4(acc[0], acc[1], acc[2], acc[3]);
  out4[1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
}

// grid (ceil(H / kColsPerBlock), max(num_tokens, 1)); block kThreads.
template <typename TOut>
__global__ void __launch_bounds__(kThreads)
    combine_tail_bf16_kernel(PushLayout L, const int32_t* __restrict__ round_ctr,
                             const int32_t* __restrict__ topk_ids, TOut* __restrict__ out,
                             int num_tokens, int32_t* __restrict__ lc, int32_t* __restrict__ done,
                             int nreset_lc, int nkeys, int32_t* __restrict__ blocks_done) {
  __shared__ int s_last;
  uint32_t const tag = static_cast<uint32_t>(*round_ctr);

  // 1. wait: lane s acquires source s's cdone cell; the CTA barrier extends the
  //    acquired visibility of the peers' inbox rows to the whole block.
  if (threadIdx.x < L.ep_size) {
    wait_tag_u64(L, L.cdone_cell(L.rank, threadIdx.x), tag);
  }
  __syncthreads();

  // 2. reduce: 8 columns of token t over the top-k slots, masked routes skipped
  int const t = blockIdx.y;
  int const col = (blockIdx.x * kThreads + threadIdx.x) * kColsPerThread;
  if (t < num_tokens && col < L.hidden) {  // H % 128 == 0, so col + 8 <= H
    float acc[kColsPerThread];
#pragma unroll
    for (int j = 0; j < kColsPerThread; ++j) acc[j] = 0.0f;
    const int32_t* ids = topk_ids + static_cast<int64_t>(t) * L.top_k;
    for (int k = 0; k < L.top_k; ++k) {
      if (ids[k] < 0) continue;  // masked route: never written this round, exact +0 before
      const uint4* src = reinterpret_cast<const uint4*>(L.combine_row(L.rank, t, k) + col);
      accumulate_bf16x8(__ldcg(src), acc);  // L2-only load: inbox rows arrive by P2P writes
    }
    store_out8(out + static_cast<int64_t>(t) * L.hidden + col, acc);
  }

  // 3. the last block of the grid acks the round (ack_kernel body)
  __syncthreads();  // every thread's inbox reads precede this block's completion count
  if (threadIdx.x == 0) {
    int const total = static_cast<int>(gridDim.x * gridDim.y);
    int const prev = atomicAdd(blocks_done, 1);
    s_last = (prev + 1 == total) ? 1 : 0;
    if (s_last) *blocks_done = 0;  // all blocks have counted: reset for the next round
  }
  __syncthreads();
  if (!s_last) return;
  for (int i = threadIdx.x; i < nreset_lc; i += blockDim.x) lc[i] = 0;
  for (int i = threadIdx.x; i < nkeys; i += blockDim.x) done[i] = 0;
  if (threadIdx.x == 0) *L.pool_head64(L.rank) = 0ull;
  __syncthreads();
  if (threadIdx.x < L.ep_size) {
    __threadfence_system();  // order the resets before the ack becomes visible
    st_release_sys_u64(L.ack_cell(threadIdx.x, L.rank), pack_count_tag(0, tag));
  }
}

}  // namespace

void sm90_cake_combine_tail_bf16(TensorView out, TensorView topk_ids, LAYOUT_PARAMS,
                                 TensorView round_ctr, TensorView lc, TensorView done,
                                 TensorView blocks_done, int64_t num_tokens) {
  check_layout(LAYOUT_ARGS);
  auto L = build_layout(LAYOUT_ARGS);
  int const nkeys = L.num_local_experts * L.ep_size;
  CHECK_INPUT(out);
  CHECK_INPUT_AND_TYPE(topk_ids, dl_int32);
  CHECK_INPUT_AND_TYPE(round_ctr, dl_int32);
  CHECK_INPUT_AND_TYPE(lc, dl_int32);
  CHECK_INPUT_AND_TYPE(done, dl_int32);
  CHECK_INPUT_AND_TYPE(blocks_done, dl_int32);
  CHECK_DIM(2, out);
  CHECK_DIM(2, topk_ids);
  TVM_FFI_ICHECK(out.size(1) == L.hidden) << "combine_tail: out hidden mismatch";
  TVM_FFI_ICHECK(num_tokens >= 0 && num_tokens <= out.size(0) && num_tokens <= L.t_cap)
      << "combine_tail: bad num_tokens " << num_tokens;
  TVM_FFI_ICHECK(topk_ids.size(0) >= num_tokens && topk_ids.size(1) == L.top_k)
      << "combine_tail: topk_ids must be (>= num_tokens, top_k)";
  TVM_FFI_ICHECK(lc.numel() >= nkeys && done.numel() >= nkeys)
      << "combine_tail: key scratch too small";
  TVM_FFI_ICHECK(blocks_done.numel() >= 1) << "combine_tail: blocks_done scratch too small";
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0)
      << "combine_tail: out must be 16-byte aligned";
  int const nreset_lc =
      static_cast<int>(std::min<int64_t>(lc.numel(), static_cast<int64_t>(nkeys) + L.ep_size));
  dim3 const grid((L.hidden + kColsPerBlock - 1) / kColsPerBlock,
                  static_cast<unsigned>(std::max<int64_t>(num_tokens, 1)));
  auto stream = get_stream(out.device());
  auto* rc = static_cast<const int32_t*>(round_ctr.data_ptr());
  auto* ids = static_cast<const int32_t*>(topk_ids.data_ptr());
  auto* lcp = static_cast<int32_t*>(lc.data_ptr());
  auto* donep = static_cast<int32_t*>(done.data_ptr());
  auto* bd = static_cast<int32_t*>(blocks_done.data_ptr());
  int const nt = static_cast<int>(num_tokens);
  if (out.dtype() == dl_float32) {
    combine_tail_bf16_kernel<float><<<grid, kThreads, 0, stream>>>(
        L, rc, ids, static_cast<float*>(out.data_ptr()), nt, lcp, donep, nreset_lc, nkeys, bd);
  } else if (out.dtype() == dl_bfloat16) {
    combine_tail_bf16_kernel<__nv_bfloat16>
        <<<grid, kThreads, 0, stream>>>(L, rc, ids, static_cast<__nv_bfloat16*>(out.data_ptr()), nt,
                                        lcp, donep, nreset_lc, nkeys, bd);
  } else {
    TVM_FFI_ICHECK(false) << "combine_tail: out must be float32 or bfloat16";
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm90_cake_combine_tail_bf16, sm90_cake_combine_tail_bf16);
