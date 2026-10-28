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

// Grid-strided single-launch dispatch for the native BF16 SM90 push MegaMoE
// (sm90_bf16_bf16_bf16_push_cake), dedup payload mode, for rounds above the
// bound of the fused small-T dispatch (cake_dispatch_fused_bf16.cu).  One
// cooperative launch over the co-resident blocks replaces the vendored
// `count_dedup` -> `reserve_publish_dedup` -> `store_publish_dedup` sequence
// on the sending rank and, optionally, the round opening (`bump_tag` +
// `wait_acks`) and the receiver-side `wait_prefix`:
//
//   1. count: every block takes a contiguous chunk of tokens.  Per route it
//      takes, with shared-memory atomics, its rank among the block's routes of
//      the same key and (carrier routes) among the block's carriers to the same
//      destination; then ONE global atomic per touched key / destination
//      returns the block's base inside it (the vendored count kernel takes one
//      global atomic per route).  Carrier routes are appended to a dense item
//      list (one global atomic per block).  With fuse_begin, lane d of block 0
//      then waits for destination d's ack of the previous round (the
//      `wait_acks_kernel` cell and tag);  grid barrier;
//   2. reserve: block 0 stores the new round tag, reserves the meta/payload
//      ranges on every destination with the packed 64-bit pool-head atomic,
//      releases the base cells and writes the per-key meta bases;  grid barrier;
//   3. store: the SlotMeta records (grid-stride over routes, one 16-byte store
//      each) and the carrier payload rows (block-stride over the item list),
//      then one system-scope fence per block;  grid barrier;
//   4. publish: block 0 releases every count cell of this source (zero counts
//      included) in parallel;
//   5. fuse_prefix: block 0 runs the vendored `wait_prefix` body (prefix form
//      0) or the same integer prefix sums computed by the whole block (form 1,
//      wait_prefix_parallel).
//
// Slot rule (as the vendored kernels): meta slot = key base + rank inside the
// key, payload slot = destination payload base + carrier rank, a duplicate
// destination reuses its carrier's row.  The ranks inside a key / destination
// come from integer atomics, so their order is not fixed -- exactly as in the
// vendored count kernel, whose `atomicAdd(&lc[eid])` / `atomicAdd(&pc[dst])`
// order is unordered too.  No consumer depends on that order: compact copies
// segment row i with its own meta, every GEMM row depends only on its own
// input row, the combine publish sorts each (owner, token) group by route
// index, and the tail sums the sources in ascending rank.  Cells, tags,
// SlotMeta fields, pool-head packing and the overflow / invalid-expert traps
// are the vendored ones, so peers running any dispatch variant interoperate.
//
// Scratch: `lc` = the pipe's per-key counters followed by its per-destination
// carrier counters (zero at entry; the round's combine tail resets them), the
// pipe's loff / ploff / keybase, and the runner's item list, item-local index
// and item counter (zero at entry; reset by this kernel at exit).
//
// Direct placement (DIRECT instance, `sm90_cake_dispatch_grid_direct_bf16`): every
// rank owns a peer-writable activation matrix a1 (symmetric memory) followed by a
// count all-gather area (one flag per source, then one row of per-key counts per
// source).  In phase 2 block 0 also stores its per-key totals into every peer's
// all-gather row and, per destination d, releases its flag on d before it reserves
// on d (and so before its base cell on d).  After the reserve it waits, per source,
// for that source's flag or else its base cell, whichever carries the round's tag
// first; a base cell without the flag means the source runs another instance.  The
// round is DIRECT only if every source released its flag this round (only this
// instance writes flags), so every rank of the round reaches the same decision;
// otherwise the whole round takes the inbox path.  In a
// DIRECT round the payload row of a carrier route r (token t, destination d) goes
// straight to row(r) of d's a1, the row the compact would have copied it to:
//   row(r) = X[key] - X[d * E] + P[key] + loff[r],   key = expert id of r,
//   X = exclusive scan over every key of the summed counts of all sources,
//   P[key] = the counts of the sources below this rank,
// and the SlotMeta payload_slot field (bit 31 = 0: inbox slot, as before) carries
//   0x80000000 | row(r)        for the carrier itself (nothing left to copy) and
//   0xC0000000 | row(carrier)  for a later route of the token to d (the compact
//                              copies the carrier's a1 row to the route's row).
// The count cells stay the only data-ready signal: their release follows the a1
// stores exactly as it follows the inbox stores.

#include <cooperative_groups.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "sm90_push_a2a.cuh"
#include "tvm_ffi_utils.h"

using tvm::ffi::TensorView;
using namespace flashinfer::sm90_push;
namespace cg = cooperative_groups;

namespace {

constexpr int kMaxEpSize = 32;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kMaxTopK = 8;
// Register-bounded to kMinBlocks blocks per SM (<= 64 registers at 256 threads);
// the grid is kMinBlocks x SMs co-resident blocks (cooperative launch).
constexpr int kMinBlocks = 4;
constexpr int kUnroll = 4;  // 16-byte vectors in flight per thread per payload row

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
  TVM_FFI_ICHECK(top_k >= 1 && top_k <= kMaxTopK && t_cap >= 1 && pool_rows >= 1 &&
                 meta_rows >= pool_rows)
      << "sm90_cake_bf16: bad capacity args";
  TVM_FFI_ICHECK(bytes_per_row == 2 * hidden)
      << "sm90_cake_bf16: the grid dispatch requires a bf16 dispatch payload "
         "(bytes_per_row == 2 * hidden), got bytes_per_row="
      << bytes_per_row;
}

// Receiver-side prefix outputs (the arguments of the vendored `wait_prefix`).
struct PrefixArgs {
  int32_t* rows_per_src;
  int64_t* offsets;
  int32_t* seg_src_base;
  int32_t* seg_out_base;
  int32_t* pad_base;
  int32_t* m_dev;
  int32_t* p_dev;
  int32_t* next_row;
  int m_cap;
  int max_rows_per_key;
};

// Receiver prefix, form 1: the outputs of the vendored `wait_prefix` serial loop
// computed by the whole block.  Every output is an integer prefix sum of the
// same counts (integer addition is associative), so every stored value equals
// the serial loop's; only the order of the block's global stores changes, and
// every reader runs after this kernel.  The cells, tags, checks, the m_cap trap
// and the zero-row cdone releases are those of the serial form (the same
// outputs are stored before the m_cap check); the base cells, which every
// source releases before its count cells, are waited for first.
//   a. exclusive int64 scan of cnt in key order idx = e * eps + s (a contiguous
//      chunk per thread, warp shuffle scan, per-warp partials): seg_out_base,
//      offsets[e] = scan[e * eps], pad_base[e] and the total;
//   b. one warp per source s: exclusive scan of cnt[e * eps + s] over e:
//      seg_src_base = sbase[s] + scan, column total = rows_per_src[s].
// cnt [nkeys], sbase [eps], srows [eps]: shared memory of the caller.
__device__ __forceinline__ void wait_prefix_parallel(const PushLayout& L, const PrefixArgs& pa,
                                                     uint32_t tag, int32_t* cnt, int32_t* sbase,
                                                     int32_t* srows) {
  __shared__ long long s_part[kWarps + 1];  // per-warp partials, then the total
  int const E = L.num_local_experts;
  int const eps = L.ep_size;
  int const nkeys = E * eps;
  int const tid = threadIdx.x;
  int const lane = tid & 31;
  int const warp = tid >> 5;
  uint64_t* cells = reinterpret_cast<uint64_t*>(L.window(L.rank) + L.count_cells_offset);
  for (int s = tid; s < eps; s += kThreads) {
    sbase[s] = wait_tag_u64(L, L.base_cell(L.rank, s), tag);
  }
  for (int i = tid; i < nkeys; i += kThreads) {
    int c = wait_tag_u64(L, cells + i, tag);
    if (c < 0 || c > pa.max_rows_per_key) {  // protocol corruption, not overflow
      printf("sm90_push: bad count %d for (expert %d, src %d), max %d\n", c, i / eps, i % eps,
             pa.max_rows_per_key);
      publish_abort_all(L, tag);
      asm volatile("trap;");
    }
    cnt[i] = c;
  }
  __syncthreads();
  // a. per-thread chunk sums and their warp-inclusive scan
  int const ipt = (nkeys + kThreads - 1) / kThreads;
  int const begin = min(tid * ipt, nkeys);
  int const end = min(begin + ipt, nkeys);
  long long local = 0;
  for (int i = begin; i < end; ++i) local += cnt[i];
  long long incl = local;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    long long const v = __shfl_up_sync(0xffffffffu, incl, off);
    if (lane >= off) incl += v;
  }
  if (lane == 31) s_part[warp] = incl;
  // b. per-source column scans (warp-uniform loop)
  int const epl = (E + 31) / 32;
  int const e0 = min(lane * epl, E);
  int const e1 = min(e0 + epl, E);
  for (int s = warp; s < eps; s += kWarps) {
    int csum = 0;
    for (int e = e0; e < e1; ++e) csum += cnt[e * eps + s];
    int cincl = csum;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      int const v = __shfl_up_sync(0xffffffffu, cincl, off);
      if (lane >= off) cincl += v;
    }
    int src_row = sbase[s] + (cincl - csum);
    for (int e = e0; e < e1; ++e) {
      int const idx = e * eps + s;
      pa.seg_src_base[idx] = src_row;
      src_row += cnt[idx];
    }
    if (lane == 31) srows[s] = cincl;
  }
  __syncthreads();
  long long acc = incl - local;
  for (int w = 0; w < warp; ++w) acc += s_part[w];
  int e = begin / eps;
  int s = begin - e * eps;
  for (int i = begin; i < end; ++i) {
    if (s == 0) {
      pa.offsets[e] = acc;
      // deep_gemm::compute_padded_offset(acc, e): 32-align with headroom
      pa.pad_base[e] = static_cast<int32_t>((acc + static_cast<int64_t>(e) * 31) / 32 * 32);
    }
    pa.seg_out_base[i] = static_cast<int32_t>(acc);
    acc += cnt[i];
    if (++s == eps) {
      s = 0;
      ++e;
    }
  }
  if (tid == kThreads - 1) s_part[kWarps] = acc;  // grand total (the last chunk ends at nkeys)
  __syncthreads();
  if (tid == 0) {
    int64_t const total = s_part[kWarps];
    if (total > static_cast<int64_t>(pa.m_cap)) {
      printf("sm90_push: accumulated row count %lld exceeds m_cap %d\n",
             static_cast<long long>(total), pa.m_cap);
      publish_abort_all(L, tag);
      asm volatile("trap;");
      return;
    }
    pa.offsets[E] = total;
    pa.seg_out_base[nkeys] = static_cast<int32_t>(total);
    *pa.m_dev = static_cast<int32_t>(total);
    int64_t p = (static_cast<int64_t>(pa.m_cap) + static_cast<int64_t>(E) * 31) / 32 * 32;
    *pa.p_dev = static_cast<int32_t>(p < 1 ? 1 : p);
    *pa.next_row = 0;  // reset the persistent-compact work queue
  }
  __syncthreads();
  for (int src = tid; src < eps; src += kThreads) {
    int const rows = srows[src];
    pa.rows_per_src[src] = rows;
    if (rows == 0) {  // no combine rows will flow back to src: publish now
      st_release_sys_u64(L.cdone_cell(src, L.rank), pack_count_tag(0, tag));
    }
  }
}

// The form-1 receiver prefix of the DIRECT instances, as a call: inlined, it makes those
// 64-register instances spill.  It ends the kernel, so no state is live across the call.
__device__ __noinline__ void wait_prefix_parallel_call(PushLayout L, PrefixArgs pa, uint32_t tag,
                                                       int32_t* cnt, int32_t* sbase,
                                                       int32_t* srows) {
  wait_prefix_parallel(L, pa, tag, cnt, sbase, srows);
}

// Per-round scratch (see the file comment).
struct GridScratch {
  int32_t* lc;        // [nkeys + eps]: records per key, then carriers per destination
  int32_t* loff;      // [T * top_k]: meta rank inside the key (-1 = masked route)
  int32_t* ploff;     // [T * top_k]: carrier rank inside the destination (-1 = not a carrier)
  int32_t* keybase;   // [nkeys]: meta slot of the key's first record on its destination
  int32_t* bases;     // [2 * eps]: meta base, payload base per destination
  int32_t* items;     // [T * top_k]: dense list of carrier route ids
  int32_t* iloc;      // [T * top_k]: block-local item index of a carrier route
  int32_t* item_ctr;  // [1]: items appended so far (zero at entry, reset at exit)
};

// Direct placement (see the file comment); unused by the inbox-only instance.
struct DirectArgs {
  const uint64_t* a1_peers;  // device uint64[eps]: every rank's symmetric a1 base, self included
  uint64_t ag_offset;        // byte offset of the count all-gather area inside each a1 buffer
  int32_t* a1base;           // [nkeys + 1]: a1 row base of each key on its destination, then
                             // this round's direct flag
  int ag_stride;             // ints per all-gather row (nkeys rounded up to a multiple of 4)
  int m_cap;                 // row capacity of every receiver (m_cap of the pipe)
  int max_rows_per_key;      // one source's routes for one expert (t_cap * top_k)
};

__device__ __forceinline__ uint64_t* ag_flag(const DirectArgs& D, int p, int s) {
  return reinterpret_cast<uint64_t*>(D.a1_peers[p] + D.ag_offset) + s;
}

__device__ __forceinline__ int32_t* ag_row(const DirectArgs& D, int eps, int p, int s) {
  uint64_t const flags_bytes = (static_cast<uint64_t>(eps) * 8 + 255) / 256 * 256;
  return reinterpret_cast<int32_t*>(D.a1_peers[p] + D.ag_offset + flags_bytes) +
         static_cast<uint64_t>(s) * D.ag_stride;
}

// One source's placement vote: 1 once its flag carries the round's tag, 0 once its base
// cell carries the tag without the flag (a source releases its flag, if any, before its
// base cell).  Abort and timeout handling of wait_tag_u64.
__device__ __forceinline__ int wait_flag_or_base(PushLayout const& L, const uint64_t* flag,
                                                 const uint64_t* base, uint32_t tag) {
  auto start = globaltimer_ns();
  while (true) {
    if (static_cast<uint32_t>(ld_acquire_sys_u64(flag) >> 32) == tag) return 1;
    if (static_cast<uint32_t>(ld_acquire_sys_u64(base) >> 32) == tag) {
      return static_cast<uint32_t>(ld_acquire_sys_u64(flag) >> 32) == tag ? 1 : 0;
    }
    uint64_t const abort = ld_acquire_sys_u64(L.abort_cell(L.rank));
    uint32_t const abort_rank_plus_one = static_cast<uint32_t>(abort);
    if (abort_rank_plus_one != 0) {
      uint32_t const abort_tag = static_cast<uint32_t>(abort >> 32);
      publish_abort_all(L, abort_tag);
      printf("sm90_cake_bf16: rank %d aborted round %u while waiting for a placement vote\n",
             static_cast<int>(abort_rank_plus_one) - 1, abort_tag);
      __trap();
    }
    __nanosleep(64);
    if (FLASHINFER_SM90_PUSH_CHECK_TIMEOUT(start)) {
      publish_abort_all(L, tag);
      printf("sm90_cake_bf16: placement vote wait timed out (round %u)\n", tag);
      __trap();
    }
  }
}

// Dynamic smem (ints): A [nkeys] | B [nkeys + 1] | C [eps] | D [eps].
//   phase 1: A = per-key counts -> bases, C = per-destination carrier counts -> bases
//   phase 2 (block 0): A = per-key totals (kept for phase 4), B = exclusive scan, C = meta bases
//   phase 5 (block 0): A = received counts, C = source bases, D = per-source accumulators
//   (form 1: per-source row totals)
//   DIRECT, phase 2 after the key bases: B = summed per-key counts of every source -> their
//   exclusive scan
template <int TOP_K, int PREFIX_FORM, bool DIRECT>
__global__ void __launch_bounds__(kThreads, kMinBlocks)
    dispatch_grid_dedup_kernel(PushLayout L, const uint8_t* __restrict__ x_bytes,
                               int bytes_per_token, const int32_t* __restrict__ topk_ids,
                               const float* __restrict__ topk_w, int32_t* round_ctr, GridScratch s,
                               int local_num_tokens, int fuse_begin, int fuse_prefix, PrefixArgs pa,
                               DirectArgs D) {
  extern __shared__ int32_t smem[];
  int const E = L.num_local_experts;
  int const eps = L.ep_size;
  int const nkeys = E * eps;
  int32_t* sA = smem;
  int32_t* sB = sA + nkeys;
  int32_t* sC = sB + nkeys + 1;
  int32_t* sD = sC + eps;
  __shared__ int s_items;  // phase 1: carriers of this block, then its base in the item list
  __shared__ int s_wsum[kWarps];
  cg::grid_group grid = cg::this_grid();
  int const tid = threadIdx.x;
  int const lane = tid & 31;
  int const warp = tid >> 5;
  int const G = static_cast<int>(gridDim.x);
  int const T = local_num_tokens;
  // read once at entry by every block; with fuse_begin this round's tag is the
  // counter + 1 (block 0 stores it after the first grid barrier)
  uint32_t const tag = static_cast<uint32_t>(*reinterpret_cast<volatile int32_t*>(round_ctr)) +
                       (fuse_begin ? 1u : 0u);

  // ---- 1. count: this block's contiguous token chunk [t0, t1)
  int const chunk = (T + G - 1) / G;
  int const t0 = min(static_cast<int>(blockIdx.x) * chunk, T);
  int const t1 = min(t0 + chunk, T);
  int const r0 = t0 * TOP_K;
  int const r1 = t1 * TOP_K;
  for (int i = tid; i < nkeys; i += kThreads) sA[i] = 0;
  for (int i = tid; i < eps; i += kThreads) sC[i] = 0;
  if (tid == 0) s_items = 0;
  __syncthreads();
  for (int r = r0 + tid; r < r1; r += kThreads) {
    int const eid = topk_ids[r];
    if (eid >= nkeys) {
      printf("sm90_cake_bf16: invalid expert id %d (num_experts=%d) at route %d\n", eid, nkeys, r);
      publish_abort_all(L, tag);
      asm volatile("trap;");
    }
    if (eid < 0) {  // masked route: not counted, never a payload carrier
      s.loff[r] = -1;
      s.ploff[r] = -1;
      continue;
    }
    int const t = r / TOP_K;
    int const k = r - t * TOP_K;
    int const d = expert_owner_rank(eid, E);
    bool carrier = true;  // first route of the token to destination d
#pragma unroll
    for (int j = 0; j < TOP_K; ++j) {
      if (j < k) {
        int const ej = topk_ids[t * TOP_K + j];
        if (ej >= 0 && expert_owner_rank(ej, E) == d) carrier = false;
      }
    }
    s.loff[r] = atomicAdd(&sA[eid], 1);  // block-local rank, rebased below
    if (carrier) {
      s.ploff[r] = atomicAdd(&sC[d], 1);
      s.iloc[r] = atomicAdd(&s_items, 1);
    } else {
      s.ploff[r] = -1;
    }
  }
  __syncthreads();
  // one global atomic per touched key / destination: the block's base inside it
  for (int i = tid; i < nkeys; i += kThreads) {
    int const c = sA[i];
    if (c != 0) sA[i] = atomicAdd(&s.lc[i], c);
  }
  for (int i = tid; i < eps; i += kThreads) {
    int const c = sC[i];
    if (c != 0) sC[i] = atomicAdd(&s.lc[nkeys + i], c);
  }
  if (tid == 0 && s_items != 0) s_items = atomicAdd(s.item_ctr, s_items);
  __syncthreads();
  for (int r = r0 + tid; r < r1; r += kThreads) {  // same route -> thread map as above
    int const eid = topk_ids[r];
    if (eid < 0) continue;
    s.loff[r] += sA[eid];
    int const p = s.ploff[r];
    if (p >= 0) {
      s.ploff[r] = p + sC[expert_owner_rank(eid, E)];
      s.items[s_items + s.iloc[r]] = r;
    }
  }
  // wait_acks_kernel: destination d has drained the previous round (its pool
  // head is reset and its pool rows are free) before block 0 reserves on d and
  // before any block stores into d (both follow the grid barrier below)
  if (fuse_begin && blockIdx.x == 0 && tid < eps) {
    wait_tag_u64(L, L.ack_cell(L.rank, tid), tag - 1u);
  }
  grid.sync();

  // ---- 2. reserve (block 0)
  if (blockIdx.x == 0) {
    if (fuse_begin && tid == 0) {  // every block has read the old counter
      *reinterpret_cast<volatile int32_t*>(round_ctr) = static_cast<int32_t>(tag);
    }
    for (int i = tid; i < nkeys; i += kThreads) sA[i] = __ldcg(s.lc + i);
    __syncthreads();
    // exclusive scan of sA into sB (keys of destination d are [d*E, (d+1)*E))
    int const ipt = (nkeys + kThreads - 1) / kThreads;
    int const begin = min(tid * ipt, nkeys);
    int const end = min(begin + ipt, nkeys);
    int local = 0;
    for (int i = begin; i < end; ++i) local += sA[i];
    int incl = local;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      int const v = __shfl_up_sync(0xffffffffu, incl, off);
      if (lane >= off) incl += v;
    }
    if (lane == 31) s_wsum[warp] = incl;
    __syncthreads();
    int woff = 0;
    for (int w = 0; w < warp; ++w) woff += s_wsum[w];
    int excl = woff + incl - local;
    for (int i = begin; i < end; ++i) {
      sB[i] = excl;
      excl += sA[i];
    }
    if (tid == kThreads - 1) sB[nkeys] = excl;  // grand total
    if constexpr (DIRECT) {  // this source's per-key totals into every peer's all-gather row
      int const n4 = D.ag_stride >> 2;
      for (int i = tid; i < eps * n4; i += kThreads) {
        int const p = i / n4;
        int const k0 = (i - p * n4) * 4;
        int4 const v = make_int4(sA[k0], k0 + 1 < nkeys ? sA[k0 + 1] : 0,
                                 k0 + 2 < nkeys ? sA[k0 + 2] : 0, k0 + 3 < nkeys ? sA[k0 + 3] : 0);
        reinterpret_cast<int4*>(ag_row(D, eps, p, L.rank))[i - p * n4] = v;
      }
    }
    __syncthreads();
    if (tid < eps) {
      int const d = tid;
      if constexpr (DIRECT) {  // the block's all-gather row stores (barrier above), then the
                               // flag, before the reserve round trip on d
        __threadfence_system();
        st_release_sys_u64(ag_flag(D, d, L.rank), pack_count_tag(1, tag));
      }
      int const total_meta = sB[(d + 1) * E] - sB[d * E];
      int const total_payload = __ldcg(s.lc + nkeys + d);
      int meta_base = 0, payload_base = 0;
      if (total_meta > 0) {  // >= 1 route implies >= 1 carrier and vice versa
        unsigned long long const add = (static_cast<unsigned long long>(total_payload) << 32) |
                                       static_cast<unsigned long long>(total_meta);
        unsigned long long const old = atomicAdd_system(L.pool_head64(d), add);
        meta_base = static_cast<int>(old & 0xffffffffull);
        payload_base = static_cast<int>(old >> 32);
        if (meta_base + total_meta > L.meta_rows || payload_base + total_payload > L.pool_rows) {
          printf(
              "sm90_cake_bf16: dedup pool overflow on dst %d (meta %d+%d > %d or payload %d+%d > "
              "%d); raise capacity_factor\n",
              d, meta_base, total_meta, L.meta_rows, payload_base, total_payload, L.pool_rows);
          publish_abort_all(L, tag);
          asm volatile("trap;");
        }
      }
      st_release_sys_u64(L.base_cell(d, L.rank), pack_count_tag(meta_base, tag));
      s.bases[d] = meta_base;
      s.bases[eps + d] = payload_base;
      sC[d] = meta_base;
    }
    __syncthreads();
    for (int key = tid; key < nkeys; key += kThreads) {
      int const d = key / E;
      s.keybase[key] = sC[d] + sB[key] - sB[d * E];
    }
    if constexpr (DIRECT) {  // DIRECT round? then the a1 row base of every key
      bool ok = true;
      if (tid < eps) {  // source tid: its flag (released before its reserve) or its base cell
        ok = wait_flag_or_base(L, ag_flag(D, L.rank, tid), L.base_cell(L.rank, tid), tag) != 0;
      }
      int const direct = __syncthreads_and(ok);  // also: sB (key scan) is consumed
      if (direct) {
        // sB = counts summed over the sources, a1base = counts of the sources below this rank
        for (int key = tid; key < nkeys; key += kThreads) {
          int tot = 0, below = 0;
          bool bad = false;
          for (int src = 0; src < eps; ++src) {
            int const c = __ldcg(ag_row(D, eps, L.rank, src) + key);
            bad |= c < 0 || c > D.max_rows_per_key;  // protocol corruption
            tot += c;
            if (src < L.rank) below += c;
          }
          if (bad) {
            printf("sm90_cake_bf16: bad all-gathered count for key %d (max %d per source)\n", key,
                   D.max_rows_per_key);
            publish_abort_all(L, tag);
            __trap();
          }
          sB[key] = tot;
          D.a1base[key] = below;
        }
        __syncthreads();
        // exclusive scan of sB in place (the chunked scan above)
        int chunk = 0;
        for (int i = begin; i < end; ++i) chunk += sB[i];
        int chunk_incl = chunk;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
          int const v = __shfl_up_sync(0xffffffffu, chunk_incl, off);
          if (lane >= off) chunk_incl += v;
        }
        if (lane == 31) s_wsum[warp] = chunk_incl;
        __syncthreads();
        int run = chunk_incl - chunk;
        for (int w = 0; w < warp; ++w) run += s_wsum[w];
        for (int i = begin; i < end; ++i) {
          int const c = sB[i];
          sB[i] = run;
          run += c;
        }
        if (tid == kThreads - 1) sB[nkeys] = run;
        __syncthreads();
        if (tid < eps) {  // every sender sees the same counts: all of them stop before any a1 store
          int const m_d = sB[(tid + 1) * E] - sB[tid * E];
          if (m_d > D.m_cap) {
            printf("sm90_cake_bf16: %d rows for rank %d exceed m_cap %d\n", m_d, tid, D.m_cap);
            publish_abort_all(L, tag);
            __trap();
          }
        }
        for (int key = tid; key < nkeys; key += kThreads) {
          D.a1base[key] += sB[key] - sB[key / E * E];
        }
      }
      if (tid == 0) D.a1base[nkeys] = direct;
    }
  }
  grid.sync();

  // ---- 3a. SlotMeta records: one 16-byte store per unmasked route
  int const n_routes = T * TOP_K;
  int const direct = DIRECT ? __ldcg(D.a1base + nkeys) : 0;  // this round's placement (phase 2)
  for (int r = static_cast<int>(blockIdx.x) * kThreads + tid; r < n_routes; r += G * kThreads) {
    int const lo = __ldcg(s.loff + r);
    if (lo < 0) continue;  // masked route
    int const eid = topk_ids[r];
    int const t = r / TOP_K;
    int const k = r - t * TOP_K;
    int const d = expert_owner_rank(eid, E);
    int p = __ldcg(s.ploff + r);
    int rc = r;   // DIRECT: the carrier route
    if (p < 0) {  // duplicate destination: the carrier's row (first route k' < k to d)
#pragma unroll
      for (int j = TOP_K - 1; j >= 0; --j) {
        if (j < k) {
          int const ej = topk_ids[t * TOP_K + j];
          if (ej >= 0 && expert_owner_rank(ej, E) == d) {
            p = __ldcg(s.ploff + t * TOP_K + j);
            if constexpr (DIRECT) rc = t * TOP_K + j;
          }
        }
      }
    }
    int const pslot = __ldcg(s.bases + eps + d) + p;
    int const slot = __ldcg(s.keybase + eid) + lo;
    uint4 rec;
    rec.x = static_cast<uint32_t>(t);
    rec.y = static_cast<uint32_t>(pack_rank_k(L.rank, k));
    rec.z = __float_as_uint(topk_w[r]);
    rec.w = static_cast<uint32_t>(pslot);
    if constexpr (DIRECT) {
      if (direct) {  // the carrier's row in d's a1, tagged carrier (10) or duplicate (11)
        int const row = __ldcg(D.a1base + topk_ids[rc]) + (rc == r ? lo : __ldcg(s.loff + rc));
        rec.w = (rc == r ? 0x80000000u : 0xC0000000u) | static_cast<uint32_t>(row);
      }
    }
    *reinterpret_cast<uint4*>(L.pool_meta(d, slot)) =
        rec;  // SlotMeta {src_token, rank|k, weight, payload_slot}
  }

  // ---- 3b. carrier payload rows: block-stride over the dense item list
  int const n_items = __ldcg(s.item_ctr);
  int const nv = bytes_per_token >> 4;
  for (int i = static_cast<int>(blockIdx.x); i < n_items; i += G) {
    int const r = __ldcg(s.items + i);
    int const t = r / TOP_K;
    int const d = expert_owner_rank(topk_ids[r], E);
    int const pslot = __ldcg(s.bases + eps + d) + __ldcg(s.ploff + r);
    const uint4* src =
        reinterpret_cast<const uint4*>(x_bytes + static_cast<uint64_t>(t) * bytes_per_token);
    uint4* dst = reinterpret_cast<uint4*>(L.pool_row(d, pslot));
    if constexpr (DIRECT) {
      if (direct) {  // straight into d's a1 at the route's row
        int const row = __ldcg(D.a1base + topk_ids[r]) + __ldcg(s.loff + r);
        dst =
            reinterpret_cast<uint4*>(D.a1_peers[d] + static_cast<uint64_t>(row) * bytes_per_token);
      }
    }
    for (int v0 = tid; v0 < nv; v0 += kUnroll * kThreads) {
      uint4 buf[kUnroll];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        int const v = v0 + u * kThreads;
        if (v < nv) buf[u] = src[v];
      }
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        int const v = v0 + u * kThreads;
        if (v < nv) dst[v] = buf[u];
      }
    }
  }
  __syncthreads();                       // the block's remote stores are issued
  if (tid == 0) __threadfence_system();  // and ordered at system scope
  grid.sync();                           // before block 0 publishes for everyone

  // ---- 4. block 0: every count cell of this source, zero counts included
  if (blockIdx.x != 0) return;
  __threadfence_system();
  for (int key = tid; key < nkeys; key += kThreads) {
    st_release_sys_u64(L.count_cell(key / E, key % E, L.rank), pack_count_tag(sA[key], tag));
  }
  if (tid == 0) *s.item_ctr = 0;  // every block has read it (before the last grid barrier)
  if (!fuse_prefix) return;

  // ---- 5. block 0 (fuse_prefix): the vendored wait_prefix body
  int32_t* cnt = sA;    // [nkeys], indexed e * eps + s
  int32_t* sbase = sC;  // [eps]
  int32_t* sacc = sD;   // [eps]
  __syncthreads();      // sA (per-key totals) has been published above
  if constexpr (PREFIX_FORM == 1) {
    if constexpr (DIRECT) {
      wait_prefix_parallel_call(L, pa, tag, cnt, sbase, sacc);
    } else {
      wait_prefix_parallel(L, pa, tag, cnt, sbase, sacc);
    }
    return;
  }
  uint64_t* cells = reinterpret_cast<uint64_t*>(L.window(L.rank) + L.count_cells_offset);
  for (int i = tid; i < nkeys; i += kThreads) {
    int c = wait_tag_u64(L, cells + i, tag);
    if (c < 0 || c > pa.max_rows_per_key) {  // protocol corruption, not overflow
      printf("sm90_push: bad count %d for (expert %d, src %d), max %d\n", c, i / eps, i % eps,
             pa.max_rows_per_key);
      publish_abort_all(L, tag);
      asm volatile("trap;");
    }
    cnt[i] = c;
  }
  for (int src = tid; src < eps; src += kThreads) {
    sbase[src] = wait_tag_u64(L, L.base_cell(L.rank, src), tag);
  }
  __syncthreads();
  if (tid == 0) {
    int64_t acc = 0;
    for (int src = 0; src < eps; ++src) sacc[src] = 0;
    for (int e = 0; e < E; ++e) {
      pa.offsets[e] = acc;
      // deep_gemm::compute_padded_offset(acc, e): 32-align with headroom
      pa.pad_base[e] = static_cast<int32_t>((acc + static_cast<int64_t>(e) * 31) / 32 * 32);
      for (int src = 0; src < eps; ++src) {
        int idx = e * eps + src;
        pa.seg_out_base[idx] = static_cast<int32_t>(acc);
        pa.seg_src_base[idx] = sbase[src] + sacc[src];
        sacc[src] += cnt[idx];
        acc += cnt[idx];
      }
    }
    if (acc > static_cast<int64_t>(pa.m_cap)) {
      printf("sm90_push: accumulated row count %lld exceeds m_cap %d\n",
             static_cast<long long>(acc), pa.m_cap);
      publish_abort_all(L, tag);
      asm volatile("trap;");
      return;
    }
    pa.offsets[E] = acc;
    pa.seg_out_base[nkeys] = static_cast<int32_t>(acc);
    *pa.m_dev = static_cast<int32_t>(acc);
    int64_t p = (static_cast<int64_t>(pa.m_cap) + static_cast<int64_t>(E) * 31) / 32 * 32;
    *pa.p_dev = static_cast<int32_t>(p < 1 ? 1 : p);
    *pa.next_row = 0;  // reset the persistent-compact work queue
  }
  __syncthreads();
  for (int src = tid; src < eps; src += kThreads) {
    int rows = 0;
    for (int e = 0; e < E; ++e) rows += cnt[e * eps + src];
    pa.rows_per_src[src] = rows;
    if (rows == 0) {  // no combine rows will flow back to src: publish now
      st_release_sys_u64(L.cdone_cell(src, L.rank), pack_count_tag(0, tag));
    }
  }
}

template <int TOP_K, int PREFIX_FORM, bool DIRECT>
void launch_grid_dispatch(const PushLayout& L, const uint8_t* x_bytes, int bytes_per_token,
                          const int32_t* ids, const float* w, int32_t* rc, GridScratch s, int lt,
                          int device_id, cudaStream_t stream, int fuse_begin, int fuse_prefix,
                          PrefixArgs pa, DirectArgs D) {
  auto kernel = dispatch_grid_dedup_kernel<TOP_K, PREFIX_FORM, DIRECT>;
  int const nkeys = L.num_local_experts * L.ep_size;
  size_t const smem = static_cast<size_t>(2 * nkeys + 1 + 2 * L.ep_size) * sizeof(int32_t);
  TVM_FFI_ICHECK(smem <= 48 * 1024) << "dispatch_grid: E*ep = " << nkeys << " needs " << smem
                                    << "B smem, over the 48KB static limit";
  int sms = 0, per_sm = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device_id);
  cudaError_t occ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kThreads,
                                                                  static_cast<int>(smem));
  TVM_FFI_ICHECK(occ == cudaSuccess)
      << "dispatch_grid: occupancy query failed: " << cudaGetErrorString(occ);
  int const grid = std::min(per_sm, kMinBlocks) * sms;
  TVM_FFI_ICHECK(grid >= 1) << "dispatch_grid: no co-resident block (occupancy " << per_sm << ")";
  PushLayout layout = L;
  void* args[] = {&layout,
                  const_cast<uint8_t**>(&x_bytes),
                  &bytes_per_token,
                  const_cast<int32_t**>(&ids),
                  const_cast<float**>(&w),
                  &rc,
                  &s,
                  &lt,
                  &fuse_begin,
                  &fuse_prefix,
                  &pa,
                  &D};
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned>(grid));
  config.blockDim = dim3(kThreads);
  config.dynamicSmemBytes = smem;
  config.stream = stream;
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeCooperative;
  attr.val.cooperative = 1;
  config.attrs = &attr;
  config.numAttrs = 1;
  cudaError_t const status =
      cudaLaunchKernelExC(&config, reinterpret_cast<const void*>(kernel), args);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "dispatch_grid: cooperative launch failed: " << cudaGetErrorString(status);
}

#define GRID_DISPATCH_PARAMS                                                                     \
  TensorView x, TensorView topk_ids, TensorView topk_w, LAYOUT_PARAMS, TensorView round_ctr,     \
      TensorView lc, TensorView loff, TensorView ploff, TensorView keybase, TensorView bases,    \
      TensorView items, TensorView iloc, TensorView item_ctr, int64_t fuse_begin,                \
      int64_t fuse_prefix, TensorView rows_per_src, TensorView offsets, TensorView seg_src_base, \
      TensorView seg_out_base, TensorView pad_base, TensorView m_dev, TensorView p_dev,          \
      TensorView next_row, int64_t m_cap, int64_t prefix_form

#define GRID_DISPATCH_ARGS                                                                   \
  x, topk_ids, topk_w, LAYOUT_ARGS, round_ctr, lc, loff, ploff, keybase, bases, items, iloc, \
      item_ctr, fuse_begin, fuse_prefix, rows_per_src, offsets, seg_src_base, seg_out_base,  \
      pad_base, m_dev, p_dev, next_row, m_cap, prefix_form

template <bool DIRECT>
void dispatch_grid(GRID_DISPATCH_PARAMS, DirectArgs D) {
  check_layout(LAYOUT_ARGS);
  auto L = build_layout(LAYOUT_ARGS);
  int64_t const T = x.size(0);
  int64_t const H = x.size(1);
  int64_t const K = topk_ids.size(1);
  int const eps = L.ep_size;
  int const nkeys = L.num_local_experts * eps;
  CHECK_INPUT_AND_TYPE(x, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(topk_ids, dl_int32);
  CHECK_INPUT_AND_TYPE(topk_w, dl_float32);
  CHECK_INPUT_AND_TYPE(round_ctr, dl_int32);
  CHECK_INPUT_AND_TYPE(lc, dl_int32);
  CHECK_INPUT_AND_TYPE(loff, dl_int32);
  CHECK_INPUT_AND_TYPE(ploff, dl_int32);
  CHECK_INPUT_AND_TYPE(keybase, dl_int32);
  CHECK_INPUT_AND_TYPE(bases, dl_int32);
  CHECK_INPUT_AND_TYPE(items, dl_int32);
  CHECK_INPUT_AND_TYPE(iloc, dl_int32);
  CHECK_INPUT_AND_TYPE(item_ctr, dl_int32);
  CHECK_DIM(2, x);
  CHECK_DIM(2, topk_ids);
  CHECK_DIM(2, topk_w);
  TVM_FFI_ICHECK(H == L.hidden) << "dispatch_grid: x hidden " << H << " != layout " << L.hidden;
  TVM_FFI_ICHECK(T <= L.t_cap) << "dispatch_grid: T " << T << " exceeds t_cap " << L.t_cap;
  TVM_FFI_ICHECK(K == L.top_k) << "dispatch_grid: top_k mismatch";
  TVM_FFI_ICHECK(topk_ids.size(0) == T && topk_w.size(0) == T && topk_w.size(1) == K)
      << "dispatch_grid: routing shape mismatch";
  TVM_FFI_ICHECK(lc.numel() >= nkeys + eps) << "dispatch_grid: key/destination counters too small";
  TVM_FFI_ICHECK(keybase.numel() >= nkeys) << "dispatch_grid: keybase too small";
  TVM_FFI_ICHECK(bases.numel() >= 2 * eps) << "dispatch_grid: bases scratch too small";
  TVM_FFI_ICHECK(loff.numel() >= T * K && ploff.numel() >= T * K && items.numel() >= T * K &&
                 iloc.numel() >= T * K)
      << "dispatch_grid: route scratch too small";
  TVM_FFI_ICHECK(item_ctr.numel() >= 1) << "dispatch_grid: item counter too small";
  PrefixArgs pa{};
  if (fuse_prefix != 0) {
    TVM_FFI_ICHECK(fuse_begin != 0) << "dispatch_grid: fuse_prefix requires fuse_begin";
    CHECK_INPUT_AND_TYPE(rows_per_src, dl_int32);
    CHECK_INPUT_AND_TYPE(offsets, dl_int64);
    CHECK_INPUT_AND_TYPE(seg_src_base, dl_int32);
    CHECK_INPUT_AND_TYPE(seg_out_base, dl_int32);
    CHECK_INPUT_AND_TYPE(pad_base, dl_int32);
    CHECK_INPUT_AND_TYPE(m_dev, dl_int32);
    CHECK_INPUT_AND_TYPE(p_dev, dl_int32);
    CHECK_INPUT_AND_TYPE(next_row, dl_int32);
    TVM_FFI_ICHECK(rows_per_src.numel() >= eps) << "wait_prefix: rows_per_src too small";
    TVM_FFI_ICHECK(offsets.numel() >= L.num_local_experts + 1) << "wait_prefix: offsets too small";
    TVM_FFI_ICHECK(seg_src_base.numel() >= nkeys) << "wait_prefix: seg_src_base too small";
    TVM_FFI_ICHECK(seg_out_base.numel() >= nkeys + 1) << "wait_prefix: seg_out_base too small";
    TVM_FFI_ICHECK(pad_base.numel() >= L.num_local_experts) << "wait_prefix: pad_base too small";
    TVM_FFI_ICHECK(m_cap >= 1) << "wait_prefix: bad m_cap";
    pa.rows_per_src = static_cast<int32_t*>(rows_per_src.data_ptr());
    pa.offsets = static_cast<int64_t*>(offsets.data_ptr());
    pa.seg_src_base = static_cast<int32_t*>(seg_src_base.data_ptr());
    pa.seg_out_base = static_cast<int32_t*>(seg_out_base.data_ptr());
    pa.pad_base = static_cast<int32_t*>(pad_base.data_ptr());
    pa.m_dev = static_cast<int32_t*>(m_dev.data_ptr());
    pa.p_dev = static_cast<int32_t*>(p_dev.data_ptr());
    pa.next_row = static_cast<int32_t*>(next_row.data_ptr());
    pa.m_cap = static_cast<int>(m_cap);
    pa.max_rows_per_key = L.t_cap * L.top_k;  // one source's routes for one expert
  }
  GridScratch s{};
  s.lc = static_cast<int32_t*>(lc.data_ptr());
  s.loff = static_cast<int32_t*>(loff.data_ptr());
  s.ploff = static_cast<int32_t*>(ploff.data_ptr());
  s.keybase = static_cast<int32_t*>(keybase.data_ptr());
  s.bases = static_cast<int32_t*>(bases.data_ptr());
  s.items = static_cast<int32_t*>(items.data_ptr());
  s.iloc = static_cast<int32_t*>(iloc.data_ptr());
  s.item_ctr = static_cast<int32_t*>(item_ctr.data_ptr());
  auto stream = get_stream(x.device());
  auto* xb = static_cast<const uint8_t*>(x.data_ptr());
  auto* ids = static_cast<const int32_t*>(topk_ids.data_ptr());
  auto* w = static_cast<const float*>(topk_w.data_ptr());
  auto* rc = static_cast<int32_t*>(round_ctr.data_ptr());
  TVM_FFI_ICHECK(prefix_form == 0 || prefix_form == 1)
      << "dispatch_grid: prefix_form must be 0 or 1, got " << prefix_form;
  int const fb = fuse_begin != 0 ? 1 : 0;
  int const fp = fuse_prefix != 0 ? 1 : 0;
  int const lt = static_cast<int>(T);
  int const bpt = static_cast<int>(H) * 2;
  int const dev = x.device().device_id;
#define CAKE_GRID_DISPATCH_CASE(KV)                                                               \
  case KV:                                                                                        \
    if (prefix_form == 1) {                                                                       \
      launch_grid_dispatch<KV, 1, DIRECT>(L, xb, bpt, ids, w, rc, s, lt, dev, stream, fb, fp, pa, \
                                          D);                                                     \
    } else {                                                                                      \
      launch_grid_dispatch<KV, 0, DIRECT>(L, xb, bpt, ids, w, rc, s, lt, dev, stream, fb, fp, pa, \
                                          D);                                                     \
    }                                                                                             \
    break;
  switch (K) {
    CAKE_GRID_DISPATCH_CASE(1)
    CAKE_GRID_DISPATCH_CASE(2)
    CAKE_GRID_DISPATCH_CASE(4)
    CAKE_GRID_DISPATCH_CASE(6)
    CAKE_GRID_DISPATCH_CASE(8)
    default:
      TVM_FFI_ICHECK(false) << "dispatch_grid: unsupported top_k " << K
                            << " (supported: 1, 2, 4, 6, 8)";
  }
#undef CAKE_GRID_DISPATCH_CASE
}

}  // namespace

void sm90_cake_dispatch_grid_bf16(GRID_DISPATCH_PARAMS) {
  dispatch_grid<false>(GRID_DISPATCH_ARGS, DirectArgs{});
}

// Direct placement (see the file comment): a1base = int32 [nkeys + 1] scratch, a1_peer_bases =
// device uint64 [ep] of the symmetric a1 buffers, ag_offset = byte offset of the count all-gather
// area in each of them.
void sm90_cake_dispatch_grid_direct_bf16(GRID_DISPATCH_PARAMS, TensorView a1base,
                                         int64_t a1_peer_bases, int64_t ag_offset) {
  int64_t const nkeys = num_local_experts * ep_size;
  CHECK_INPUT_AND_TYPE(a1base, dl_int32);
  TVM_FFI_ICHECK(a1base.numel() >= nkeys + 1) << "dispatch_grid_direct: a1base too small";
  TVM_FFI_ICHECK(a1_peer_bases != 0) << "dispatch_grid_direct: null a1 peer bases";
  TVM_FFI_ICHECK(ag_offset > 0 && ag_offset % 256 == 0)
      << "dispatch_grid_direct: ag_offset must be a positive multiple of 256, got " << ag_offset;
  TVM_FFI_ICHECK(m_cap >= 1) << "dispatch_grid_direct: bad m_cap " << m_cap;
  DirectArgs D;
  D.a1_peers = reinterpret_cast<const uint64_t*>(a1_peer_bases);
  D.ag_offset = static_cast<uint64_t>(ag_offset);
  D.a1base = static_cast<int32_t*>(a1base.data_ptr());
  D.ag_stride = static_cast<int>((nkeys + 3) / 4 * 4);
  D.m_cap = static_cast<int>(m_cap);
  D.max_rows_per_key = static_cast<int>(t_cap * top_k);
  dispatch_grid<true>(GRID_DISPATCH_ARGS, D);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm90_cake_dispatch_grid_bf16, sm90_cake_dispatch_grid_bf16);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm90_cake_dispatch_grid_direct_bf16,
                              sm90_cake_dispatch_grid_direct_bf16);
