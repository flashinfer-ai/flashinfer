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

// Per-route BF16 combine publish for the native BF16 SM90 push MegaMoE
// (sm90_bf16_bf16_bf16_push_cake, combine_wire="per_route").  Cake-owned twin
// of the vendored `combine_publish_kernel` (same wire, same arithmetic: one
// bf16(fp32(y_k) * w_k) row per received route, 16-byte P2P stores, the same
// cdone cell / tag release once every row bound for a source is out), with two
// launch-level differences that leave every stored byte unchanged:
//
//   * no per-round memset of the per-source row counters: the last block that
//     published a row resets them (grid-completion counter, as the pre-reduced
//     publish), and the runner zeroes them once at construction;
//   * programmatic dependent launch: the kernel waits for the preceding grid
//     before touching global memory, so it may be launched with the PDL
//     attribute right behind FC2 (use_pdl).

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

// Programmatic dependent launch (PDL).  Every kernel of the round waits for the
// preceding grid before it touches global memory and then, unless told
// otherwise, lets the next PDL-launched grid be scheduled at once; without the
// early trigger the next grid is scheduled when this grid's blocks exit.  Both
// instructions are no-ops when the launch carries no PDL attribute.  use_pdl:
// bit 0 = launch with the PDL attribute, bit 1 = no early trigger.
__device__ __forceinline__ void pdl_wait_and_trigger(int trigger) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (trigger) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

// Launch on the caller's stream, optionally with the PDL attribute (the kernel
// may then start while the preceding kernel finishes; see pdl_wait_and_trigger).
template <typename... Params, typename... Args>
void launch_kernel(void (*kernel)(Params...), dim3 grid, dim3 block, size_t smem,
                   cudaStream_t stream, bool use_pdl, Args... args) {
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = smem;
  config.stream = stream;
  config.attrs = &attr;
  config.numAttrs = use_pdl ? 1 : 0;
  cudaError_t const status = cudaLaunchKernelEx(&config, kernel, args...);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "sm90_cake_bf16: kernel launch failed: " << cudaGetErrorString(status);
}

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
      << "sm90_cake_bf16: the per-route publish requires a bf16 dispatch payload "
         "(bytes_per_row == 2 * hidden), got bytes_per_row="
      << bytes_per_row;
}

__global__ void combine_publish_per_route_bf16_kernel(
    PushLayout L, const __nv_bfloat16* __restrict__ y, const int32_t* __restrict__ meta,
    const int32_t* __restrict__ m_dev, const int32_t* __restrict__ rows_per_src,
    int32_t* __restrict__ cdone_local, const int32_t* __restrict__ round_ctr,
    int32_t* __restrict__ blocks_done, int pdl_trigger) {
  pdl_wait_and_trigger(pdl_trigger);
  __shared__ int s_last;
  int const m = *m_dev;
  int const grid = static_cast<int>(gridDim.x);
  // only blocks that own at least one row take part in the grid completion below
  int const participants = min(grid, m);
  if (static_cast<int>(blockIdx.x) >= participants) return;
  for (int r = blockIdx.x; r < m; r += grid) {
    const int32_t* mrow = meta + static_cast<int64_t>(r) * 4;
    int dst = meta_src_rank(mrow);
    float w = meta_weight(mrow);
    const __nv_bfloat16* row = y + static_cast<uint64_t>(r) * L.hidden;
    __nv_bfloat16* out = L.combine_row(dst, meta_src_token(mrow), meta_route_k(mrow));
    // 16B vectorized P2P stores; the same fp32 mul + RN cast per element as the
    // vendored kernel, so results are bit-identical
    const uint4* src4 = reinterpret_cast<const uint4*>(row);
    uint4* dst4 = reinterpret_cast<uint4*>(out);
    int nv = L.hidden >> 3;
    for (int v = threadIdx.x; v < nv; v += blockDim.x) {
      uint4 pk = src4[v];
      __nv_bfloat162 h2[4];
      memcpy(h2, &pk, sizeof(pk));
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float2 f = __bfloat1622float2(h2[j]);
        h2[j] = __floats2bfloat162_rn(f.x * w, f.y * w);
      }
      memcpy(&pk, h2, sizeof(pk));
      dst4[v] = pk;
    }
    __syncthreads();  // this row's remote stores all issued before the publish tail
    if (threadIdx.x == 0) {
      __threadfence_system();
      int prev = atomicAdd(&cdone_local[dst], 1);
      if (prev + 1 == rows_per_src[dst]) {
        __threadfence_system();
        st_release_sys_u64(L.cdone_cell(dst, L.rank),
                           pack_count_tag(rows_per_src[dst], static_cast<uint32_t>(*round_ctr)));
      }
    }
  }
  // grid completion: the last participating block zeroes the per-source row
  // counters for the next round (every other participant has finished its
  // atomics; non-participants never touch them)
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    int const prev = atomicAdd(blocks_done, 1);
    s_last = (prev + 1 == participants) ? 1 : 0;
    if (s_last) {
      __threadfence();
      *blocks_done = 0;
      for (int s = 0; s < L.ep_size; ++s) cdone_local[s] = 0;
    }
  }
}

// persistent-grid sizing, cached per device (as the vendored publish kernel)
int publish_grid_blocks(DLDevice device) {
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

void sm90_cake_combine_per_route_bf16(TensorView y, TensorView meta, LAYOUT_PARAMS,
                                      TensorView m_dev, TensorView rows_per_src,
                                      TensorView cdone_local, TensorView round_ctr,
                                      TensorView blocks_done, int64_t use_pdl) {
  check_layout(LAYOUT_ARGS);
  auto L = build_layout(LAYOUT_ARGS);
  CHECK_INPUT_AND_TYPE(y, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(meta, dl_int32);
  CHECK_INPUT_AND_TYPE(m_dev, dl_int32);
  CHECK_INPUT_AND_TYPE(rows_per_src, dl_int32);
  CHECK_INPUT_AND_TYPE(cdone_local, dl_int32);
  CHECK_INPUT_AND_TYPE(round_ctr, dl_int32);
  CHECK_INPUT_AND_TYPE(blocks_done, dl_int32);
  int64_t const Mcap = y.size(0);
  int64_t const H = y.size(1);
  TVM_FFI_ICHECK(H == L.hidden) << "combine_per_route: y hidden mismatch";
  TVM_FFI_ICHECK(meta.numel() >= Mcap * 4) << "combine_per_route: packed meta too small";
  TVM_FFI_ICHECK(rows_per_src.numel() >= L.ep_size && cdone_local.numel() >= L.ep_size)
      << "combine_per_route: per-source scratch too small";
  TVM_FFI_ICHECK(blocks_done.numel() >= 1) << "combine_per_route: blocks_done too small";
  auto stream = get_stream(y.device());
  // no memset: the counters were zeroed at construction and self-reset every round
  if (Mcap == 0) return;  // zero-row destinations were published by wait_prefix
  int const blocks = publish_grid_blocks(y.device());
  launch_kernel(
      combine_publish_per_route_bf16_kernel, dim3(blocks), dim3(256), 0, stream, (use_pdl & 1) != 0,
      L, static_cast<const __nv_bfloat16*>(y.data_ptr()),
      static_cast<const int32_t*>(meta.data_ptr()), static_cast<const int32_t*>(m_dev.data_ptr()),
      static_cast<const int32_t*>(rows_per_src.data_ptr()),
      static_cast<int32_t*>(cdone_local.data_ptr()),
      static_cast<const int32_t*>(round_ctr.data_ptr()),
      static_cast<int32_t*>(blocks_done.data_ptr()), static_cast<int>((use_pdl & 2) == 0));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm90_cake_combine_per_route_bf16, sm90_cake_combine_per_route_bf16);
