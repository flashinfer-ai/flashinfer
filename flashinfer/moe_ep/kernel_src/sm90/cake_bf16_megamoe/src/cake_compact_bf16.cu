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

// Native BF16 compaction for the SM90 push MegaMoE protocol
// (sm90_bf16_bf16_bf16_push_cake).  After `sm90_push_wait_prefix` has
// published the expert-major row order, this kernel gathers every received
// BF16 payload row from this rank's symmetric inbox into a dense, expert-
// contiguous activation matrix `a_bf16[m, H]` and the packed per-row combine
// meta `(src_rank, src_token, route_k, weight_bits)`.  No quantisation and no
// scale planes: the payload bytes are copied verbatim (bytes_per_row == 2H).
//
// The window layout and the SlotMeta record come from the vendored push
// protocol header; only `PushLayout` construction and the segment search
// are reproduced here so this translation unit stays independent of the
// FP8 ops file.

#include <cuda_bf16.h>

#include <cstdint>

#include "sm90_push_a2a.cuh"
#include "tvm_ffi_utils.h"

using tvm::ffi::TensorView;
using namespace flashinfer::sm90_push;

namespace {

constexpr int kMaxEpSize = 32;

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
  TVM_FFI_ICHECK(bytes_per_row == 2 * hidden)
      << "sm90_cake_bf16: the native BF16 path requires a bf16 dispatch payload "
         "(bytes_per_row == 2 * hidden), got bytes_per_row="
      << bytes_per_row;
}

__device__ __forceinline__ int find_segment(const int32_t* __restrict__ seg_out_base, int nkeys,
                                            int row) {
  int lo = 0, hi = nkeys;
  while (hi - lo > 1) {
    int mid = (lo + hi) >> 1;
    if (seg_out_base[mid] <= row) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// One block per compacted row (persistent work queue through `next_row`, which
// `sm90_push_wait_prefix` resets to zero every round).  Rows are expert-major:
// segment (e, s) holds the rows received from source rank s for local expert e.
__global__ void compact_bf16_persistent_kernel(
    PushLayout L, const int32_t* __restrict__ seg_src_base,
    const int32_t* __restrict__ seg_out_base, const int32_t* __restrict__ m_dev,
    int32_t* __restrict__ next_row, __nv_bfloat16* __restrict__ a_bf16,
    int32_t* __restrict__ meta_out, int32_t* __restrict__ row_expert, int H) {
  __shared__ int s_row;
  int eps = L.ep_size;
  int nkeys = L.num_local_experts * eps;
  int nv = H >> 3;  // 16-byte vectors per bf16 row
  for (;;) {
    __syncthreads();  // protect s_row from the previous iteration's readers
    if (threadIdx.x == 0) s_row = atomicAdd(next_row, 1);
    __syncthreads();
    int row = s_row;
    if (row >= *m_dev) return;
    int seg = find_segment(seg_out_base, nkeys, row);
    int e = seg / eps;
    int rec = seg_src_base[seg] + (row - seg_out_base[seg]);
    const SlotMeta* mi = L.pool_meta(L.rank, rec);
    int pslot = mi->payload_slot;
    const uint4* src4 = reinterpret_cast<const uint4*>(L.pool_row(L.rank, pslot));
    uint4* out4 = reinterpret_cast<uint4*>(a_bf16 + static_cast<int64_t>(row) * H);
    for (int v = threadIdx.x; v < nv; v += blockDim.x) out4[v] = src4[v];
    if (threadIdx.x == 0) {
      int32_t rk = mi->src_rank_k;
      int32_t* mo = meta_out + static_cast<int64_t>(row) * 4;
      mo[0] = unpack_src_rank(rk);
      mo[1] = mi->src_token;
      mo[2] = unpack_k(rk);
      mo[3] = __float_as_int(mi->weight);
      row_expert[row] = e;
    }
  }
}

int compact_grid_blocks(DLDevice device) {
  static int cache[64] = {0};
  int id = device.device_id;
  if (id >= 0 && id < 64 && cache[id] > 0) return cache[id];
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, id);
  int blocks = sms > 0 ? sms * 4 : 432;
  if (id >= 0 && id < 64) cache[id] = blocks;
  return blocks;
}

}  // namespace

void sm90_cake_compact_bf16(TensorView a_bf16, TensorView meta_out, TensorView row_expert,
                            LAYOUT_PARAMS, TensorView seg_src_base, TensorView seg_out_base,
                            TensorView m_dev, TensorView next_row) {
  check_layout(LAYOUT_ARGS);
  auto L = build_layout(LAYOUT_ARGS);
  int H = L.hidden;
  CHECK_INPUT_AND_TYPE(a_bf16, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(meta_out, dl_int32);
  CHECK_INPUT_AND_TYPE(row_expert, dl_int32);
  CHECK_INPUT_AND_TYPE(seg_src_base, dl_int32);
  CHECK_INPUT_AND_TYPE(seg_out_base, dl_int32);
  CHECK_INPUT_AND_TYPE(m_dev, dl_int32);
  CHECK_INPUT_AND_TYPE(next_row, dl_int32);
  int64_t m_capacity = a_bf16.size(0);
  TVM_FFI_ICHECK(a_bf16.ndim() == 2 && a_bf16.size(1) == H)
      << "compact_bf16: a_bf16 must be (Mcap, H)";
  TVM_FFI_ICHECK(m_capacity >= meta_rows)
      << "compact_bf16: a_bf16 rows " << m_capacity << " below meta_rows " << meta_rows;
  TVM_FFI_ICHECK(meta_out.numel() >= m_capacity * 4) << "compact_bf16: meta_out too small";
  TVM_FFI_ICHECK(row_expert.numel() >= m_capacity) << "compact_bf16: row_expert too small";
  TVM_FFI_ICHECK(seg_src_base.numel() >= num_local_experts * ep_size &&
                 seg_out_base.numel() >= num_local_experts * ep_size + 1)
      << "compact_bf16: segment tables too small";
  auto stream = get_stream(a_bf16.device());
  int blocks = compact_grid_blocks(a_bf16.device());
  compact_bf16_persistent_kernel<<<blocks, 256, 0, stream>>>(
      L, static_cast<const int32_t*>(seg_src_base.data_ptr()),
      static_cast<const int32_t*>(seg_out_base.data_ptr()),
      static_cast<const int32_t*>(m_dev.data_ptr()), static_cast<int32_t*>(next_row.data_ptr()),
      static_cast<__nv_bfloat16*>(a_bf16.data_ptr()), static_cast<int32_t*>(meta_out.data_ptr()),
      static_cast<int32_t*>(row_expert.data_ptr()), H);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm90_cake_compact_bf16, sm90_cake_compact_bf16);
