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

// Pre-reduced BF16 combine publish for the native BF16 SM90 push MegaMoE
// (sm90_bf16_bf16_bf16_push_cake, combine_wire="prereduced").
//
// The per-route wire (`combine_publish_kernel`, combine_wire="per_route") sends
// one bf16 row `bf16(fp32(y_k) * w_k)` per received route back to the token's
// owner, which sums the top-k rows in fp32.  This kernel instead groups the
// received rows by (owner rank, token), pre-reduces every group on the expert
// rank in fp32 (`acc = fmaf(w_k, fp32(y_k), acc)` in ascending route index k),
// rounds ONCE to bf16 and pushes ONE row per (token, source rank) into the
// owner's inbox.  The owner (`cake_combine_tail_prereduced_bf16.cu`) sums the
// <= ep_size group rows in ascending source-rank order and rounds once.
//
// Wire format: the vendored bf16 combine inbox `[t_cap][top_k][hidden]` is
// reused unchanged.  A group is stored in the slot of its smallest route index
// `k_min` (`L.combine_row(owner, token, k_min)`); distinct source ranks of one
// token have distinct k_min, so the top_k slots always suffice and the owner
// recomputes k_min per source rank from its own `topk_ids`.  The cdone cell of
// each owner is released once every group bound for it has been pushed
// (`groups_per_src`), with the same cell / tag / abort semantics as the
// per-route publish.
//
// Worklist: built by `sm90_cake_compact_bf16` (build_groups != 0) in the same
// pass that writes the compacted meta, so no extra kernel precedes the publish.
// Scratch lifecycle: no per-round memsets.  The publish kernel resets every
// group counter it consumes (grp_cnt[g] = 0) and its last block (grid-completion
// counter blocks_done, as the fused tail) zeroes n_groups, groups_per_src and
// cdone_local for the next round; the runner zeroes them once at construction.
//
// Determinism: the group worklist is built with integer atomics (order of
// groups is irrelevant: every group is reduced independently in a fixed route
// order), no floating-point atomics anywhere.  Every element of the output is
// rounded exactly twice (group partial sum, final sum) instead of
// (top_k + 1) times on the per-route wire.
//
// split_partials (combine_wire="prereduced_hilo"): a group of >= 2 routes
// carries its fp32 partial as TWO bf16 rows, hi = bf16(p) in slot k_min and
// lo = bf16(p - hi) in the slot of its second-smallest route index (p - hi is
// exact in fp32, so hi + lo recovers p to ~16 significant bits); the owner adds
// hi and lo in fp32.  Single-route groups are bf16(w * y) exactly as the
// per-route wire, so this variant is never less precise than the R0 wire
// while still sending fewer rows than one per route for groups of >= 3.

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
constexpr int kMaxTopK = 8;  // matches the pipe's top_k validation; sizes the in-register row lists
constexpr int kThreads = 256;

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
  TVM_FFI_ICHECK(top_k >= 1 && top_k <= kMaxTopK)
      << "sm90_cake_bf16: top_k " << top_k << " out of [1, " << kMaxTopK << "]";
  TVM_FFI_ICHECK(t_cap >= 1 && pool_rows >= 1 && meta_rows >= pool_rows)
      << "sm90_cake_bf16: bad capacity args";
  // bf16 combine wire: the bf16 inbox is present and the fp8 combine regions are empty
  TVM_FFI_ICHECK(combine_offset < cfp8_offset && cfp8_offset == csc_offset)
      << "sm90_cake_bf16: the pre-reduced combine requires a bf16 combine wire";
}

// Worklist of (owner rank, token) groups over the received rows.  One group
// record per (src, token) pair: grp_cnt[g] rows, their row indices in
// grp_rows[g * top_k + i] (any order; the publish sorts them by route index).
__global__ void __launch_bounds__(kThreads) combine_publish_prereduced_bf16_kernel(
    PushLayout L, const __nv_bfloat16* __restrict__ y, const int32_t* __restrict__ meta,
    int32_t* __restrict__ grp_cnt, const int32_t* __restrict__ grp_rows,
    const int32_t* __restrict__ grp_list, int32_t* __restrict__ n_groups,
    int32_t* __restrict__ groups_per_src, int32_t* __restrict__ cdone_local,
    const int32_t* __restrict__ round_ctr, int32_t* __restrict__ blocks_done, int split_partials) {
  int const H = L.hidden;
  int const nv = H >> 3;               // 16-byte vectors per row
  int const total_groups = *n_groups;  // read once: the last block zeroes it below
  // S blocks per group when the grid has room (decode sizes: the kernel is latency-bound,
  // so each block publishes a contiguous slice of the row); S == 1 at prefill sizes.
  // Uniform device-side decision, hence deterministic for a given routing.
  int const grid = static_cast<int>(gridDim.x);
  int const S = (total_groups * 4 <= grid) ? 4 : ((total_groups * 2 <= grid) ? 2 : 1);
  int const items = total_groups * S;
  // only blocks that own at least one (group, slice) item take part in the grid completion
  // below; the others have nothing to publish and leave before touching any scratch
  int const participants = min(grid, items);
  if (static_cast<int>(blockIdx.x) >= participants) return;
  __shared__ int s_last;
  for (int item = blockIdx.x; item < items; item += grid) {
    int const idx = item / S;
    int const part = item - idx * S;
    int const g = grp_list[idx];  // uniform across the block
    int const cnt = grp_cnt[g];   // >= 1 by worklist construction, <= top_k
    int const dst = g / L.t_cap;
    int const tok = g - dst * L.t_cap;
    int rows[kMaxTopK];
    int ks[kMaxTopK];
    float ws[kMaxTopK];
    const int32_t* glist = grp_rows + static_cast<int64_t>(g) * L.top_k;
    for (int i = 0; i < cnt; ++i) {
      rows[i] = glist[i];
      ks[i] = meta_route_k(meta + static_cast<int64_t>(rows[i]) * 4);
    }
    for (int i = 1; i < cnt; ++i) {  // insertion sort by route index, cnt <= top_k <= 8
      int kk = ks[i], rr = rows[i];
      int j = i - 1;
      while (j >= 0 && ks[j] > kk) {
        ks[j + 1] = ks[j];
        rows[j + 1] = rows[j];
        --j;
      }
      ks[j + 1] = kk;
      rows[j + 1] = rr;
    }
    for (int i = 0; i < cnt; ++i) ws[i] = meta_weight(meta + static_cast<int64_t>(rows[i]) * 4);
    // the group lives in the slot of its smallest route index (see the file comment);
    // with split_partials a multi-route group also writes its bf16 residual into
    // the slot of its second-smallest route index
    uint4* out4 = reinterpret_cast<uint4*>(L.combine_row(dst, tok, ks[0]));
    bool const split = split_partials != 0 && cnt >= 2;
    uint4* lo4 = split ? reinterpret_cast<uint4*>(L.combine_row(dst, tok, ks[1])) : nullptr;
    int const v_begin = (nv * part) / S;
    int const v_end = (nv * (part + 1)) / S;
    for (int v = v_begin + threadIdx.x; v < v_end; v += blockDim.x) {
      float acc[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[j] = 0.0f;
      for (int i = 0; i < cnt; ++i) {  // ascending k: fixed fp32 fma order
        uint4 pk = *reinterpret_cast<const uint4*>(y + static_cast<uint64_t>(rows[i]) * H +
                                                   static_cast<uint64_t>(v) * 8);
        __nv_bfloat162 h2[4];
        memcpy(h2, &pk, sizeof(pk));
        float const w = ws[i];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          float2 f = __bfloat1622float2(h2[j]);
          acc[2 * j] = fmaf(w, f.x, acc[2 * j]);
          acc[2 * j + 1] = fmaf(w, f.y, acc[2 * j + 1]);
        }
      }
      __nv_bfloat162 o2[4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        o2[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);  // one RN
      uint4 pk;
      memcpy(&pk, o2, sizeof(pk));
      out4[v] = pk;  // 16-byte P2P store into the owner's inbox
      if (split) {   // residual row: lo = bf16(p - hi), p - hi exact in fp32
        __nv_bfloat162 r2[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          float2 hi = __bfloat1622float2(o2[j]);
          r2[j] = __floats2bfloat162_rn(acc[2 * j] - hi.x, acc[2 * j + 1] - hi.y);
        }
        uint4 pl;
        memcpy(&pl, r2, sizeof(pl));
        lo4[v] = pl;
      }
    }
    __syncthreads();  // this slice's remote stores all issued before the publish tail
    if (threadIdx.x == 0) {
      if (S == 1) grp_cnt[g] = 0;  // single owner block: the slot is clean for the next round
      __threadfence_system();
      int prev = atomicAdd(&cdone_local[dst], 1);
      if (prev + 1 == groups_per_src[dst] * S) {  // every slice of every group for dst is out
        __threadfence_system();
        st_release_sys_u64(L.cdone_cell(dst, L.rank),
                           pack_count_tag(groups_per_src[dst], static_cast<uint32_t>(*round_ctr)));
      }
    }
  }
  // grid completion: the last participating block resets the per-round worklist
  // scratch (every other participant has finished reading n_groups / groups_per_src /
  // cdone_local; non-participants never read them after the early return above)
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    int const prev = atomicAdd(blocks_done, 1);
    s_last = (prev + 1 == participants) ? 1 : 0;
    if (s_last) {
      __threadfence();
      *blocks_done = 0;
      *n_groups = 0;
      for (int s = 0; s < L.ep_size; ++s) {
        groups_per_src[s] = 0;
        cdone_local[s] = 0;
      }
    }
  }
  __syncthreads();
  if (s_last && S > 1) {  // sliced groups: no single owner block, reset the listed counters here
    for (int i = threadIdx.x; i < total_groups; i += blockDim.x) grp_cnt[grp_list[i]] = 0;
  }
}

// persistent-grid sizing, cached per device (as the vendored compact / publish kernels)
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

void sm90_cake_combine_prereduced_bf16(TensorView y, TensorView meta, LAYOUT_PARAMS,
                                       TensorView m_dev, TensorView grp_cnt, TensorView grp_rows,
                                       TensorView grp_list, TensorView n_groups,
                                       TensorView groups_per_src, TensorView cdone_local,
                                       TensorView round_ctr, TensorView blocks_done,
                                       int64_t split_partials) {
  check_layout(LAYOUT_ARGS);
  auto L = build_layout(LAYOUT_ARGS);
  CHECK_INPUT_AND_TYPE(y, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(meta, dl_int32);
  CHECK_INPUT_AND_TYPE(m_dev, dl_int32);
  CHECK_INPUT_AND_TYPE(grp_cnt, dl_int32);
  CHECK_INPUT_AND_TYPE(grp_rows, dl_int32);
  CHECK_INPUT_AND_TYPE(grp_list, dl_int32);
  CHECK_INPUT_AND_TYPE(n_groups, dl_int32);
  CHECK_INPUT_AND_TYPE(groups_per_src, dl_int32);
  CHECK_INPUT_AND_TYPE(cdone_local, dl_int32);
  CHECK_INPUT_AND_TYPE(round_ctr, dl_int32);
  CHECK_INPUT_AND_TYPE(blocks_done, dl_int32);
  CHECK_DIM(2, y);
  int64_t const Mcap = y.size(0);
  int64_t const H = y.size(1);
  int const eps = L.ep_size;
  int64_t const nslots = static_cast<int64_t>(eps) * L.t_cap;
  TVM_FFI_ICHECK(H == L.hidden) << "combine_prereduced: y hidden mismatch";
  TVM_FFI_ICHECK(meta.numel() >= Mcap * 4) << "combine_prereduced: packed meta too small";
  TVM_FFI_ICHECK(grp_cnt.numel() >= nslots) << "combine_prereduced: grp_cnt too small";
  TVM_FFI_ICHECK(grp_rows.numel() >= nslots * L.top_k) << "combine_prereduced: grp_rows too small";
  TVM_FFI_ICHECK(grp_list.numel() >= nslots) << "combine_prereduced: grp_list too small";
  TVM_FFI_ICHECK(n_groups.numel() >= 1) << "combine_prereduced: n_groups too small";
  TVM_FFI_ICHECK(blocks_done.numel() >= 1) << "combine_prereduced: blocks_done too small";
  TVM_FFI_ICHECK(groups_per_src.numel() >= eps && cdone_local.numel() >= eps)
      << "combine_prereduced: per-source scratch too small";
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(y.data_ptr()) % 16 == 0)
      << "combine_prereduced: y must be 16-byte aligned";
  auto stream = get_stream(y.device());
  auto* cdl = static_cast<int32_t*>(cdone_local.data_ptr());
  auto* cnt = static_cast<int32_t*>(grp_cnt.data_ptr());
  auto* gps = static_cast<int32_t*>(groups_per_src.data_ptr());
  auto* glist = static_cast<int32_t*>(grp_list.data_ptr());
  auto* ng = static_cast<int32_t*>(n_groups.data_ptr());
  auto* bd = static_cast<int32_t*>(blocks_done.data_ptr());
  // no memsets: the scratch was zeroed at construction and self-resets every round
  if (Mcap == 0) return;  // zero-row destinations were published by wait_prefix
  int const blocks = publish_grid_blocks(y.device());
  combine_publish_prereduced_bf16_kernel<<<blocks, kThreads, 0, stream>>>(
      L, static_cast<const __nv_bfloat16*>(y.data_ptr()),
      static_cast<const int32_t*>(meta.data_ptr()), cnt,
      static_cast<const int32_t*>(grp_rows.data_ptr()), glist, ng, gps, cdl,
      static_cast<const int32_t*>(round_ctr.data_ptr()), bd, split_partials != 0 ? 1 : 0);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm90_cake_combine_prereduced_bf16, sm90_cake_combine_prereduced_bf16);
