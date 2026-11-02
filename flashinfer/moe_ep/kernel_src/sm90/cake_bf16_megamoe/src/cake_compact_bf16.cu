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
// With build_groups != 0 (pre-reduced combine wires) the same pass also builds
// the (owner rank, token) group worklist consumed by the pre-reduced publish:
// grp_cnt[g] / grp_rows[g][pos] / grp_list / n_groups / groups_per_src, g =
// src_rank * t_cap + src_token, with integer atomics only (the worklist order is
// irrelevant: every group is reduced independently in route order).  The publish
// kernel resets this scratch for the next round.  Corrupt meta (out-of-range
// source rank / token / route, or more than top_k rows for one token) aborts
// every peer's round and traps, as the former stand-alone group-build kernel.
//
// The window layout and the SlotMeta record come from the vendored push
// protocol header; only `PushLayout` construction and the segment search
// are reproduced here so this translation unit stays independent of the
// FP8 ops file.
//
// Direct placement (`sm90_cake_compact_direct_bf16`, rounds dispatched by the grid
// dispatch's direct entry): a1 is this rank's symmetric activation matrix and the
// senders may have stored a row there already.  SlotMeta.payload_slot then reads
//   bit 31 = 0       an inbox payload slot: copy that inbox row (as above);
//   0b10 | row'      a carrier route the sender stored at a1 row row' == row: no copy
//                    (row' != row traps: the placement rule was broken);
//   0b11 | crow      a later route of the same token to this rank: copy a1 row crow
//                    (the carrier's) to row.
// meta, row_expert and the worklist are written exactly as above.

#include <cuda_bf16.h>

#include <cstdint>

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

// One block per compacted row, assigned by static grid stride (row = blockIdx.x
// + k * gridDim.x): every row costs the same (one segment search + one row copy),
// so no work queue is needed and blocks beyond the row count exit at once.
// `next_row` (reset by `sm90_push_wait_prefix` every round) is no longer read;
// the argument stays for the binding.  Rows are expert-major: segment (e, s)
// holds the rows received from source rank s for local expert e.
__global__ void compact_bf16_persistent_kernel(
    PushLayout L, const int32_t* __restrict__ seg_src_base,
    const int32_t* __restrict__ seg_out_base, const int32_t* __restrict__ m_dev,
    int32_t* __restrict__ next_row, __nv_bfloat16* __restrict__ a_bf16,
    int32_t* __restrict__ meta_out, int32_t* __restrict__ row_expert, int H,
    int32_t* __restrict__ grp_cnt, int32_t* __restrict__ grp_rows, int32_t* __restrict__ grp_list,
    int32_t* __restrict__ n_groups, int32_t* __restrict__ groups_per_src,
    const int32_t* __restrict__ round_ctr, int build_groups, int pdl_trigger) {
  (void)next_row;
  pdl_wait_and_trigger(pdl_trigger);
  int eps = L.ep_size;
  uint32_t const tag = build_groups ? static_cast<uint32_t>(*round_ctr) : 0u;
  int nkeys = L.num_local_experts * eps;
  int nv = H >> 3;       // 16-byte vectors per bf16 row
  int const m = *m_dev;  // uniform across the grid for this round
  for (int row = blockIdx.x; row < m; row += gridDim.x) {
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
      int const src = unpack_src_rank(rk);
      int const tok = mi->src_token;
      int const k = unpack_k(rk);
      mo[0] = src;
      mo[1] = tok;
      mo[2] = k;
      mo[3] = __float_as_int(mi->weight);
      row_expert[row] = e;
      if (build_groups) {  // pre-reduced combine worklist (see the file comment)
        if (src < 0 || src >= eps || tok < 0 || tok >= L.t_cap || k < 0 || k >= L.top_k) {
          printf("sm90_cake_bf16: corrupt combine meta at row %d (src %d, token %d, k %d)\n", row,
                 src, tok, k);
          publish_abort_all(L, tag);
          asm volatile("trap;");
        }
        int const g = src * L.t_cap + tok;
        int const pos = atomicAdd(&grp_cnt[g], 1);
        if (pos >= L.top_k) {  // a token has at most top_k routes in total
          printf("sm90_cake_bf16: group (src %d, token %d) exceeds top_k %d\n", src, tok, L.top_k);
          publish_abort_all(L, tag);
          asm volatile("trap;");
        }
        grp_rows[static_cast<int64_t>(g) * L.top_k + pos] = row;
        if (pos == 0) {  // first row opens the group
          grp_list[atomicAdd(n_groups, 1)] = g;
          atomicAdd(&groups_per_src[src], 1);
        }
      }
    }
  }
}

// Latency-pipelined form (compact_form 1): the same products, written by the same
// arithmetic, with more bytes in flight.  The shipped kernel above runs one dependent
// chain per row (segment search, SlotMeta load, then 3.5 load -> store iterations with
// one 16-byte load in flight per thread, then thread 0's meta and worklist atomics before
// warp 0 moves on).  This form
//   * copies the segment tables into shared memory once per block,
//   * loads each SlotMeta as one 16-byte record,
//   * issues all (up to kPipeLoads) 16-byte loads of a thread's row slice before its stores,
//   * resolves the next row's segment and SlotMeta while those loads are in flight,
//   * rotates the meta / worklist duty over the row's warps, and
//   * assigns R rows to a 256-thread block at a time: R = 1 when every block owns at most
//     2 rows (decode: one load round trip per row), R = 4 up to 6 rows per block, R = 8
//     (one warp per row) above.  m is uniform across the grid, so the choice is uniform
//     and depends only on this rank's row count.
// a1 rows [0, m) are copied verbatim to the same positions; meta, row_expert and the
// worklist contents are identical (grp_rows positions and the grp_list order come from
// integer atomics in both forms; the publish sorts every group by route index and treats
// groups independently).
constexpr int kPipeThreads = 256;
constexpr int kPipeLoads = 4;
constexpr int kPipeMaxKeys = 1024;

// One compacted row's meta, row_expert and worklist record (the shipped thread-0 block;
// mt = SlotMeta {src_token, src_rank_k, weight bits, payload_slot}).
__device__ __forceinline__ void compact_row_record(
    PushLayout const& L, int row, int e, int4 mt, int32_t* __restrict__ meta_out,
    int32_t* __restrict__ row_expert, int32_t* __restrict__ grp_cnt, int32_t* __restrict__ grp_rows,
    int32_t* __restrict__ grp_list, int32_t* __restrict__ n_groups,
    int32_t* __restrict__ groups_per_src, uint32_t tag, int build_groups) {
  int const eps = L.ep_size;
  int const src = unpack_src_rank(mt.y);
  int const tok = mt.x;
  int const k = unpack_k(mt.y);
  reinterpret_cast<int4*>(meta_out)[row] = make_int4(src, tok, k, mt.z);
  row_expert[row] = e;
  if (build_groups) {  // pre-reduced combine worklist (see the file comment)
    if (src < 0 || src >= eps || tok < 0 || tok >= L.t_cap || k < 0 || k >= L.top_k) {
      printf("sm90_cake_bf16: corrupt combine meta at row %d (src %d, token %d, k %d)\n", row, src,
             tok, k);
      publish_abort_all(L, tag);
      asm volatile("trap;");
    }
    int const g = src * L.t_cap + tok;
    int const pos = atomicAdd(&grp_cnt[g], 1);
    if (pos >= L.top_k) {  // a token has at most top_k routes in total
      printf("sm90_cake_bf16: group (src %d, token %d) exceeds top_k %d\n", src, tok, L.top_k);
      publish_abort_all(L, tag);
      asm volatile("trap;");
    }
    grp_rows[static_cast<int64_t>(g) * L.top_k + pos] = row;
    if (pos == 0) {  // first row opens the group
      grp_list[atomicAdd(n_groups, 1)] = g;
      atomicAdd(&groups_per_src[src], 1);
    }
  }
}

// Segment of `row` (find_segment over the shared-memory table) and its SlotMeta record.
__device__ __forceinline__ void compact_row_lookup(const int4* __restrict__ metas, int eps,
                                                   const int32_t* s_out, const int32_t* s_src,
                                                   int nkeys, int row, int& e, int4& mt) {
  int lo = 0, hi = nkeys;
  while (hi - lo > 1) {
    int mid = (lo + hi) >> 1;
    if (s_out[mid] <= row) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  e = lo / eps;
  mt = __ldg(metas + (s_src[lo] + (row - s_out[lo])));
}

// DIRECT: a1_src = a_bf16 (the carriers' rows; a separate pointer because a_bf16 is
// __restrict__ and written by the same kernel: a row is either read or written, never both).
template <int R, bool DIRECT>
__device__ __forceinline__ void compact_rows_pipelined(
    PushLayout const& L, const int32_t* s_out, const int32_t* s_src, int nkeys, int m, int H,
    __nv_bfloat16* __restrict__ a_bf16, int32_t* __restrict__ meta_out,
    int32_t* __restrict__ row_expert, int32_t* __restrict__ grp_cnt, int32_t* __restrict__ grp_rows,
    int32_t* __restrict__ grp_list, int32_t* __restrict__ n_groups,
    int32_t* __restrict__ groups_per_src, uint32_t tag, int build_groups, const uint8_t* a1_src) {
  constexpr int kTpr = kPipeThreads / R;  // threads per row
  constexpr int kWpr = kTpr / 32;         // warps per row
  int const nv = H >> 3;                  // 16-byte vectors per bf16 row
  int const eps = L.ep_size;
  // this rank's inbox payload rows and SlotMeta records (window base read once)
  const uint8_t* pool = L.pool_row(L.rank, 0);
  const int4* metas = reinterpret_cast<const int4*>(L.pool_meta(L.rank, 0));
  int const row_bytes = L.bytes_per_row;
  int const sub = threadIdx.x / kTpr;
  int const lt = threadIdx.x - sub * kTpr;
  int const stride = gridDim.x * R;
  int row = blockIdx.x * R + sub;
  int e_cur = 0;
  int4 mt_cur = make_int4(0, 0, 0, 0);
  if (row < m) compact_row_lookup(metas, eps, s_out, s_src, nkeys, row, e_cur, mt_cur);
  for (int it = 0; row < m; row += stride, ++it) {
    const uint4* src4 =
        reinterpret_cast<const uint4*>(pool + static_cast<uint64_t>(mt_cur.w) * row_bytes);
    int nv_row = nv;
    if constexpr (DIRECT) {
      uint32_t const w = static_cast<uint32_t>(mt_cur.w);
      if (w & 0x80000000u) {
        int const crow = static_cast<int>(w & 0x3fffffffu);
        if (w & 0x40000000u) {  // later route of a token: copy its carrier's row
          if (crow >= m || crow == row) {
            printf("sm90_cake_bf16: duplicate route at row %d names carrier row %d (m %d)\n", row,
                   crow, m);
            publish_abort_all(L, tag);
            asm volatile("trap;");
          }
          src4 = reinterpret_cast<const uint4*>(a1_src + static_cast<uint64_t>(crow) * row_bytes);
        } else {  // carrier: already stored at this row by the sender
          if (crow != row) {
            printf("sm90_cake_bf16: carrier placement mismatch at row %d (sender row %d)\n", row,
                   crow);
            publish_abort_all(L, tag);
            asm volatile("trap;");
          }
          nv_row = 0;
        }
      }
    }
    uint4* out4 = reinterpret_cast<uint4*>(a_bf16 + static_cast<int64_t>(row) * H);
    int const next = row + stride;
    int e_next = 0;
    int4 mt_next = make_int4(0, 0, 0, 0);
    bool looked_up = false;
    for (int v0 = lt; v0 < nv_row; v0 += kTpr * kPipeLoads) {
      uint4 buf[kPipeLoads];
#pragma unroll
      for (int j = 0; j < kPipeLoads; ++j) {
        int const v = v0 + j * kTpr;
        if constexpr (DIRECT) {  // rows written by peers: L2 loads (no non-coherent path)
          if (v < nv) buf[j] = __ldcg(src4 + v);
        } else {
          if (v < nv) buf[j] = __ldg(src4 + v);
        }
      }
      if (!looked_up) {  // the next row's chain overlaps the loads in flight
        looked_up = true;
        if (next < m) compact_row_lookup(metas, eps, s_out, s_src, nkeys, next, e_next, mt_next);
      }
#pragma unroll
      for (int j = 0; j < kPipeLoads; ++j) {
        int const v = v0 + j * kTpr;
        if (v < nv) out4[v] = buf[j];
      }
    }
    if (lt == (it % kWpr) * 32) {
      compact_row_record(L, row, e_cur, mt_cur, meta_out, row_expert, grp_cnt, grp_rows, grp_list,
                         n_groups, groups_per_src, tag, build_groups);
    }
    if (!looked_up && next < m) {
      compact_row_lookup(metas, eps, s_out, s_src, nkeys, next, e_next, mt_next);
    }
    e_cur = e_next;
    mt_cur = mt_next;
  }
}

template <bool DIRECT>
__global__ void __launch_bounds__(kPipeThreads) compact_bf16_pipelined_kernel(
    PushLayout L, const int32_t* __restrict__ seg_src_base,
    const int32_t* __restrict__ seg_out_base, const int32_t* __restrict__ m_dev,
    __nv_bfloat16* __restrict__ a_bf16, int32_t* __restrict__ meta_out,
    int32_t* __restrict__ row_expert, int H, int32_t* __restrict__ grp_cnt,
    int32_t* __restrict__ grp_rows, int32_t* __restrict__ grp_list, int32_t* __restrict__ n_groups,
    int32_t* __restrict__ groups_per_src, const int32_t* __restrict__ round_ctr, int build_groups,
    int pdl_trigger, const uint8_t* a1_src) {
  __shared__ int32_t s_out[kPipeMaxKeys + 1];
  __shared__ int32_t s_src[kPipeMaxKeys];
  pdl_wait_and_trigger(pdl_trigger);
  int const nkeys = L.num_local_experts * L.ep_size;  // <= kPipeMaxKeys (host check)
  for (int i = threadIdx.x; i <= nkeys; i += blockDim.x) {
    s_out[i] = seg_out_base[i];
    if (i < nkeys) s_src[i] = seg_src_base[i];
  }
  // the trap paths of the direct form publish the round's abort tag
  uint32_t const tag = (DIRECT || build_groups) ? static_cast<uint32_t>(*round_ctr) : 0u;
  int const m = *m_dev;  // uniform across the grid for this round
  __syncthreads();
  int const grid = static_cast<int>(gridDim.x);
  if (m <= 2 * grid) {
    compact_rows_pipelined<1, DIRECT>(L, s_out, s_src, nkeys, m, H, a_bf16, meta_out, row_expert,
                                      grp_cnt, grp_rows, grp_list, n_groups, groups_per_src, tag,
                                      build_groups, a1_src);
  } else if (m <= 6 * grid) {
    compact_rows_pipelined<4, DIRECT>(L, s_out, s_src, nkeys, m, H, a_bf16, meta_out, row_expert,
                                      grp_cnt, grp_rows, grp_list, n_groups, groups_per_src, tag,
                                      build_groups, a1_src);
  } else {
    compact_rows_pipelined<8, DIRECT>(L, s_out, s_src, nkeys, m, H, a_bf16, meta_out, row_expert,
                                      grp_cnt, grp_rows, grp_list, n_groups, groups_per_src, tag,
                                      build_groups, a1_src);
  }
  if constexpr (DIRECT) {
    // generic -> async proxy: the a1 rows FC1 reads with TMA were written by the peers (ordered
    // before this kernel by the count-cell acquire of the prefix) and by this kernel
    asm volatile("fence.proxy.async.global;" ::: "memory");
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

#define COMPACT_OUT_PARAMS                                                                       \
  TensorView a_bf16, TensorView meta_out, TensorView row_expert, TensorView seg_src_base,        \
      TensorView seg_out_base, TensorView m_dev, TensorView grp_cnt, TensorView grp_rows,        \
      TensorView grp_list, TensorView n_groups, TensorView groups_per_src, TensorView round_ctr, \
      int64_t build_groups

#define COMPACT_OUT_ARGS                                                                        \
  a_bf16, meta_out, row_expert, seg_src_base, seg_out_base, m_dev, grp_cnt, grp_rows, grp_list, \
      n_groups, groups_per_src, round_ctr, build_groups

void check_compact_args(const PushLayout& L, COMPACT_OUT_PARAMS) {
  int64_t const nkeys = static_cast<int64_t>(L.num_local_experts) * L.ep_size;
  CHECK_INPUT_AND_TYPE(a_bf16, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(meta_out, dl_int32);
  CHECK_INPUT_AND_TYPE(row_expert, dl_int32);
  CHECK_INPUT_AND_TYPE(seg_src_base, dl_int32);
  CHECK_INPUT_AND_TYPE(seg_out_base, dl_int32);
  CHECK_INPUT_AND_TYPE(m_dev, dl_int32);
  int64_t m_capacity = a_bf16.size(0);
  TVM_FFI_ICHECK(a_bf16.ndim() == 2 && a_bf16.size(1) == L.hidden)
      << "compact_bf16: a_bf16 must be (Mcap, H)";
  TVM_FFI_ICHECK(m_capacity >= L.meta_rows)
      << "compact_bf16: a_bf16 rows " << m_capacity << " below meta_rows " << L.meta_rows;
  TVM_FFI_ICHECK(meta_out.numel() >= m_capacity * 4) << "compact_bf16: meta_out too small";
  TVM_FFI_ICHECK(row_expert.numel() >= m_capacity) << "compact_bf16: row_expert too small";
  TVM_FFI_ICHECK(seg_src_base.numel() >= nkeys && seg_out_base.numel() >= nkeys + 1)
      << "compact_bf16: segment tables too small";
  if (build_groups != 0) {
    int64_t const nslots = static_cast<int64_t>(L.ep_size) * L.t_cap;
    CHECK_INPUT_AND_TYPE(grp_cnt, dl_int32);
    CHECK_INPUT_AND_TYPE(grp_rows, dl_int32);
    CHECK_INPUT_AND_TYPE(grp_list, dl_int32);
    CHECK_INPUT_AND_TYPE(n_groups, dl_int32);
    CHECK_INPUT_AND_TYPE(groups_per_src, dl_int32);
    CHECK_INPUT_AND_TYPE(round_ctr, dl_int32);
    TVM_FFI_ICHECK(grp_cnt.numel() >= nslots && grp_list.numel() >= nslots &&
                   grp_rows.numel() >= nslots * L.top_k && n_groups.numel() >= 1 &&
                   groups_per_src.numel() >= L.ep_size && round_ctr.numel() >= 1)
        << "compact_bf16: pre-reduced group scratch too small";
  }
}

template <bool DIRECT>
void launch_pipelined(const PushLayout& L, COMPACT_OUT_PARAMS, int64_t use_pdl) {
  launch_kernel(
      compact_bf16_pipelined_kernel<DIRECT>, dim3(compact_grid_blocks(a_bf16.device())),
      dim3(kPipeThreads), 0, get_stream(a_bf16.device()), (use_pdl & 1) != 0, L,
      static_cast<const int32_t*>(seg_src_base.data_ptr()),
      static_cast<const int32_t*>(seg_out_base.data_ptr()),
      static_cast<const int32_t*>(m_dev.data_ptr()), static_cast<__nv_bfloat16*>(a_bf16.data_ptr()),
      static_cast<int32_t*>(meta_out.data_ptr()), static_cast<int32_t*>(row_expert.data_ptr()),
      L.hidden, static_cast<int32_t*>(grp_cnt.data_ptr()),
      static_cast<int32_t*>(grp_rows.data_ptr()), static_cast<int32_t*>(grp_list.data_ptr()),
      static_cast<int32_t*>(n_groups.data_ptr()), static_cast<int32_t*>(groups_per_src.data_ptr()),
      static_cast<const int32_t*>(round_ctr.data_ptr()), build_groups != 0 ? 1 : 0,
      static_cast<int>((use_pdl & 2) == 0),
      DIRECT ? static_cast<const uint8_t*>(a_bf16.data_ptr()) : nullptr);
}

}  // namespace

void sm90_cake_compact_bf16(TensorView a_bf16, TensorView meta_out, TensorView row_expert,
                            LAYOUT_PARAMS, TensorView seg_src_base, TensorView seg_out_base,
                            TensorView m_dev, TensorView next_row, TensorView grp_cnt,
                            TensorView grp_rows, TensorView grp_list, TensorView n_groups,
                            TensorView groups_per_src, TensorView round_ctr, int64_t build_groups,
                            int64_t use_pdl, int64_t form) {
  check_layout(LAYOUT_ARGS);
  auto L = build_layout(LAYOUT_ARGS);
  int H = L.hidden;
  check_compact_args(L, COMPACT_OUT_ARGS);
  CHECK_INPUT_AND_TYPE(next_row, dl_int32);
  TVM_FFI_ICHECK(form == 0 || form == 1) << "compact_bf16: form must be 0 or 1, got " << form;
  if (form == 1 && num_local_experts * ep_size <= kPipeMaxKeys) {  // latency-pipelined form
    launch_pipelined<false>(L, COMPACT_OUT_ARGS, use_pdl);
    return;
  }
  auto stream = get_stream(a_bf16.device());
  int blocks = compact_grid_blocks(a_bf16.device());
  launch_kernel(
      compact_bf16_persistent_kernel, dim3(blocks), dim3(256), 0, stream, (use_pdl & 1) != 0, L,
      static_cast<const int32_t*>(seg_src_base.data_ptr()),
      static_cast<const int32_t*>(seg_out_base.data_ptr()),
      static_cast<const int32_t*>(m_dev.data_ptr()), static_cast<int32_t*>(next_row.data_ptr()),
      static_cast<__nv_bfloat16*>(a_bf16.data_ptr()), static_cast<int32_t*>(meta_out.data_ptr()),
      static_cast<int32_t*>(row_expert.data_ptr()), H, static_cast<int32_t*>(grp_cnt.data_ptr()),
      static_cast<int32_t*>(grp_rows.data_ptr()), static_cast<int32_t*>(grp_list.data_ptr()),
      static_cast<int32_t*>(n_groups.data_ptr()), static_cast<int32_t*>(groups_per_src.data_ptr()),
      static_cast<const int32_t*>(round_ctr.data_ptr()), build_groups != 0 ? 1 : 0,
      static_cast<int>((use_pdl & 2) == 0));
}

// Direct placement (see the file comment): the latency-pipelined form over a symmetric a1 whose
// carrier rows the senders stored already; a_bf16 rows must fit the 30-bit row field.
void sm90_cake_compact_direct_bf16(TensorView a_bf16, TensorView meta_out, TensorView row_expert,
                                   LAYOUT_PARAMS, TensorView seg_src_base, TensorView seg_out_base,
                                   TensorView m_dev, TensorView grp_cnt, TensorView grp_rows,
                                   TensorView grp_list, TensorView n_groups,
                                   TensorView groups_per_src, TensorView round_ctr,
                                   int64_t build_groups, int64_t use_pdl) {
  check_layout(LAYOUT_ARGS);
  auto L = build_layout(LAYOUT_ARGS);
  check_compact_args(L, COMPACT_OUT_ARGS);
  CHECK_INPUT_AND_TYPE(round_ctr, dl_int32);
  TVM_FFI_ICHECK(round_ctr.numel() >= 1) << "compact_direct_bf16: round counter missing";
  TVM_FFI_ICHECK(num_local_experts * ep_size <= kPipeMaxKeys)
      << "compact_direct_bf16: E*ep = " << num_local_experts * ep_size << " above " << kPipeMaxKeys;
  TVM_FFI_ICHECK(a_bf16.size(0) < (int64_t{1} << 30))
      << "compact_direct_bf16: a_bf16 rows " << a_bf16.size(0) << " exceed the 30-bit row field";
  launch_pipelined<true>(L, COMPACT_OUT_ARGS, use_pdl);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm90_cake_compact_bf16, sm90_cake_compact_bf16);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm90_cake_compact_direct_bf16, sm90_cake_compact_direct_bf16);
