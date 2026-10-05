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

// Fused small-T dispatch for the native BF16 SM90 push MegaMoE
// (sm90_bf16_bf16_bf16_push_cake), dedup payload mode.  One cooperative launch
// replaces the vendored `count_dedup` -> `reserve_publish_dedup` ->
// `store_publish_dedup` sequence on the sending rank:
//
//   1. every block loads all T*top_k routes into shared memory and derives,
//      deterministically and without global atomics, the per-key record
//      counts, the per-destination carrier counts, and for its own token the
//      rank of each route among the earlier routes of the same key / the
//      earlier carriers to the same destination (the slot assignment rule of
//      the vendored kernels: meta slot = key base + rank, payload slot =
//      destination base + carrier rank, duplicates reuse the carrier row);
//   2. block 0 reserves the meta/payload ranges on every destination with the
//      packed 64-bit pool-head atomic and releases the base cells -- the single
//      blocking cross-rank round trip of the dispatch;  grid barrier;
//   3. every block writes its token's SlotMeta records and carrier payload rows
//      into the destinations' pools, then fences at system scope;  grid barrier;
//   4. block 0 releases every count cell (including the zero counts) in parallel.
//
// Cells, tags, SlotMeta fields, pool-head packing and the overflow / invalid
// expert traps are those of the vendored protocol, so a peer running the
// vendored `wait_prefix` / compact kernels consumes this dispatch unchanged.
// The host keeps this path for T <= kMaxTokens (T * top_k <= kMaxRoutes) and
// falls back to the vendored kernels above that.

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
constexpr int kMaxRoutes = 1024;  // T * top_k bound of the fused path (route ids in smem)

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
      << "sm90_cake_bf16: the fused dispatch requires a bf16 dispatch payload "
         "(bytes_per_row == 2 * hidden), got bytes_per_row="
      << bytes_per_row;
}

__device__ __forceinline__ int warp_sum(int v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

// grid = max(T, 1) blocks (cooperative), block = kThreads.  Dynamic smem:
// route ids [kMaxRoutes], lc [nkeys], exclusive scan [nkeys + 1], pc [eps].
template <int TOP_K>
__global__ void __launch_bounds__(kThreads)
    dispatch_fused_dedup_kernel(PushLayout L, const uint8_t* __restrict__ x_bytes,
                                int bytes_per_token, const int32_t* __restrict__ topk_ids,
                                const float* __restrict__ topk_w,
                                const int32_t* __restrict__ round_ctr, int32_t* __restrict__ bases,
                                int local_num_tokens) {
  extern __shared__ int32_t smem[];
  int const E = L.num_local_experts;
  int const eps = L.ep_size;
  int const nkeys = E * eps;
  int const n_routes = local_num_tokens * TOP_K;
  int32_t* s_ids = smem;               // [kMaxRoutes]
  int32_t* s_lc = s_ids + kMaxRoutes;  // [nkeys]  records per (expert, this source)
  int32_t* s_scan = s_lc + nkeys;      // [nkeys + 1]  exclusive prefix of s_lc
  int32_t* s_pc = s_scan + nkeys + 1;  // [eps]  carrier (payload) rows per destination
  __shared__ int s_eid[TOP_K], s_dst[TOP_K], s_carrier[TOP_K], s_loff[TOP_K], s_ploff[TOP_K];
  __shared__ int s_pslot[TOP_K];
  __shared__ int s_red[kWarps][2 * TOP_K];
  __shared__ int s_wsum[kWarps];
  cg::grid_group grid = cg::this_grid();
  uint32_t const tag = static_cast<uint32_t>(*round_ctr);
  int const tid = threadIdx.x;
  int const lane = tid & 31;
  int const warp = tid >> 5;
  int const t = blockIdx.x;
  bool const has_token = t < local_num_tokens;

  // 1a. route ids (validated as count_dedup_kernel does) and zeroed counters
  for (int i = tid; i < nkeys; i += kThreads) s_lc[i] = 0;
  for (int i = tid; i < eps; i += kThreads) s_pc[i] = 0;
  for (int r = tid; r < n_routes; r += kThreads) {
    int const eid = topk_ids[r];
    if (eid >= nkeys) {
      printf("sm90_cake_bf16: invalid expert id %d (num_experts=%d) at route %d\n", eid, nkeys, r);
      publish_abort_all(L, tag);
      asm volatile("trap;");
    }
    s_ids[r] = eid;
  }
  __syncthreads();

  // 1b. this block's token: destination and carrier flag per route
  if (tid == 0) {
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
      int const eid = has_token ? s_ids[t * TOP_K + k] : -1;
      if (eid < 0) {  // masked route: not counted, never a payload carrier
        s_eid[k] = -1;
        s_dst[k] = -1;
        s_carrier[k] = 0;
        continue;
      }
      int const d = expert_owner_rank(eid, E);
      int carrier = 1;
#pragma unroll
      for (int j = 0; j < TOP_K; ++j) {
        if (j < k && s_dst[j] == d) carrier = 0;
      }
      s_eid[k] = eid;
      s_dst[k] = d;
      s_carrier[k] = carrier;
    }
  }
  __syncthreads();

  // 1c. one pass over every route: key / carrier totals (smem atomics, order
  //     independent) and, for this token's routes, the number of earlier routes
  //     of the same key and of earlier carriers to the same destination.
  int cl[TOP_K], cp[TOP_K];
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) cl[k] = cp[k] = 0;
  for (int r = tid; r < n_routes; r += kThreads) {
    int const eid = s_ids[r];
    if (eid < 0) continue;
    int const tok = r / TOP_K;
    int const kk = r - tok * TOP_K;
    int const d = expert_owner_rank(eid, E);
    bool carrier = true;
#pragma unroll
    for (int j = 0; j < TOP_K; ++j) {
      if (j < kk) {
        int const ej = s_ids[tok * TOP_K + j];
        if (ej >= 0 && expert_owner_rank(ej, E) == d) carrier = false;
      }
    }
    atomicAdd(&s_lc[eid], 1);
    if (carrier) atomicAdd(&s_pc[d], 1);
    if (has_token) {
#pragma unroll
      for (int k = 0; k < TOP_K; ++k) {
        if (s_eid[k] >= 0 && r < t * TOP_K + k) {
          if (eid == s_eid[k]) ++cl[k];
          if (carrier && s_carrier[k] && d == s_dst[k]) ++cp[k];
        }
      }
    }
  }
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) {
    int const a = warp_sum(cl[k]);
    int const b = warp_sum(cp[k]);
    if (lane == 0) {
      s_red[warp][k] = a;
      s_red[warp][TOP_K + k] = b;
    }
  }
  __syncthreads();
  if (tid < 2 * TOP_K) {
    int sum = 0;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) sum += s_red[w][tid];
    if (tid < TOP_K) {
      s_loff[tid] = sum;
    } else {
      s_ploff[tid - TOP_K] = sum;
    }
  }

  // 1d. exclusive scan of s_lc (keys of destination d are [d*E, (d+1)*E))
  int const ipt = (nkeys + kThreads - 1) / kThreads;
  int const begin = min(tid * ipt, nkeys);
  int const end = min(begin + ipt, nkeys);
  int local = 0;
  for (int i = begin; i < end; ++i) local += s_lc[i];
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
    s_scan[i] = excl;
    excl += s_lc[i];
  }
  if (tid == kThreads - 1) s_scan[nkeys] = excl;  // grand total
  __syncthreads();

  // 2. block 0: reserve the meta/payload ranges on every destination
  if (t == 0 && tid < eps) {
    int const d = tid;
    int const total_meta = s_scan[(d + 1) * E] - s_scan[d * E];
    int const total_payload = s_pc[d];
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
    bases[d] = meta_base;
    bases[eps + d] = payload_base;
  }
  grid.sync();

  // 3. this token's SlotMeta records and carrier payload rows
  if (has_token) {
    if (tid == 0) {
#pragma unroll
      for (int k = 0; k < TOP_K; ++k) {
        if (s_eid[k] < 0) {
          s_pslot[k] = 0;
          continue;
        }
        int const d = s_dst[k];
        int const meta_base = __ldcg(bases + d);
        int const payload_base = __ldcg(bases + eps + d);
        int const slot = meta_base + (s_scan[s_eid[k]] - s_scan[d * E]) + s_loff[k];
        int pslot;
        if (s_carrier[k]) {
          pslot = payload_base + s_ploff[k];
        } else {  // duplicate destination: reuse the carrier's row (k' < k)
          pslot = 0;
#pragma unroll
          for (int j = 0; j < TOP_K; ++j) {
            if (j < k && s_dst[j] == d) pslot = s_pslot[j];
          }
        }
        s_pslot[k] = pslot;
        SlotMeta* m = L.pool_meta(d, slot);
        m->src_token = t;
        m->src_rank_k = pack_rank_k(L.rank, k);
        m->weight = topk_w[t * TOP_K + k];
        m->payload_slot = pslot;
      }
    }
    __syncthreads();
    uint4* dst[TOP_K];
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
      dst[k] = (s_eid[k] < 0 || !s_carrier[k])
                   ? nullptr
                   : reinterpret_cast<uint4*>(L.pool_row(s_dst[k], s_pslot[k]));
    }
    const uint4* src =
        reinterpret_cast<const uint4*>(x_bytes + static_cast<uint64_t>(t) * bytes_per_token);
    int const nv = bytes_per_token >> 4;
    for (int v = tid; v < nv; v += kThreads) {
      uint4 const pk = src[v];
#pragma unroll
      for (int k = 0; k < TOP_K; ++k) {
        if (dst[k] != nullptr) dst[k][v] = pk;
      }
    }
  }
  __syncthreads();                       // the block's remote stores are issued
  if (tid == 0) __threadfence_system();  // and ordered at system scope
  grid.sync();                           // before block 0 publishes for everyone

  // 4. block 0: every count cell of this source, zero counts included
  if (t == 0) {
    __threadfence_system();
    for (int key = tid; key < nkeys; key += kThreads) {
      st_release_sys_u64(L.count_cell(key / E, key % E, L.rank), pack_count_tag(s_lc[key], tag));
    }
  }
}

template <int TOP_K>
void launch_fused_dispatch(const PushLayout& L, const uint8_t* x_bytes, int bytes_per_token,
                           const int32_t* ids, const float* w, const int32_t* rc, int32_t* bases,
                           int lt, int device_id, cudaStream_t stream) {
  auto kernel = dispatch_fused_dedup_kernel<TOP_K>;
  int const nkeys = L.num_local_experts * L.ep_size;
  size_t const smem = static_cast<size_t>(kMaxRoutes + 2 * nkeys + 1 + L.ep_size) * sizeof(int32_t);
  TVM_FFI_ICHECK(smem <= 48 * 1024) << "dispatch_fused: E*ep = " << nkeys << " needs " << smem
                                    << "B smem, over the 48KB static limit";
  int const grid = std::max(lt, 1);
  int sms = 0, per_sm = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device_id);
  cudaError_t occ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kThreads,
                                                                  static_cast<int>(smem));
  TVM_FFI_ICHECK(occ == cudaSuccess)
      << "dispatch_fused: occupancy query failed: " << cudaGetErrorString(occ);
  TVM_FFI_ICHECK(grid <= per_sm * sms)
      << "dispatch_fused: grid " << grid << " exceeds the co-resident capacity " << per_sm * sms;
  PushLayout layout = L;
  void* args[] = {&layout,
                  const_cast<uint8_t**>(&x_bytes),
                  &bytes_per_token,
                  const_cast<int32_t**>(&ids),
                  const_cast<float**>(&w),
                  const_cast<int32_t**>(&rc),
                  &bases,
                  &lt};
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
      << "dispatch_fused: cooperative launch failed: " << cudaGetErrorString(status);
}

}  // namespace

void sm90_cake_dispatch_fused_bf16(TensorView x, TensorView topk_ids, TensorView topk_w,
                                   LAYOUT_PARAMS, TensorView round_ctr, TensorView bases) {
  check_layout(LAYOUT_ARGS);
  auto L = build_layout(LAYOUT_ARGS);
  int64_t const T = x.size(0);
  int64_t const H = x.size(1);
  int64_t const K = topk_ids.size(1);
  CHECK_INPUT_AND_TYPE(x, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(topk_ids, dl_int32);
  CHECK_INPUT_AND_TYPE(topk_w, dl_float32);
  CHECK_INPUT_AND_TYPE(round_ctr, dl_int32);
  CHECK_INPUT_AND_TYPE(bases, dl_int32);
  CHECK_DIM(2, x);
  CHECK_DIM(2, topk_ids);
  CHECK_DIM(2, topk_w);
  TVM_FFI_ICHECK(H == L.hidden) << "dispatch_fused: x hidden " << H << " != layout " << L.hidden;
  TVM_FFI_ICHECK(T <= L.t_cap) << "dispatch_fused: T " << T << " exceeds t_cap " << L.t_cap;
  TVM_FFI_ICHECK(K == L.top_k) << "dispatch_fused: top_k mismatch";
  TVM_FFI_ICHECK(topk_ids.size(0) == T && topk_w.size(0) == T && topk_w.size(1) == K)
      << "dispatch_fused: routing shape mismatch";
  TVM_FFI_ICHECK(T * K <= kMaxRoutes)
      << "dispatch_fused: T * top_k = " << T * K << " exceeds " << kMaxRoutes;
  TVM_FFI_ICHECK(bases.numel() >= 2 * L.ep_size) << "dispatch_fused: bases scratch too small";
  auto stream = get_stream(x.device());
  auto* xb = static_cast<const uint8_t*>(x.data_ptr());
  auto* ids = static_cast<const int32_t*>(topk_ids.data_ptr());
  auto* w = static_cast<const float*>(topk_w.data_ptr());
  auto* rc = static_cast<const int32_t*>(round_ctr.data_ptr());
  auto* bp = static_cast<int32_t*>(bases.data_ptr());
  int const lt = static_cast<int>(T);
  int const bpt = static_cast<int>(H) * 2;
  int const dev = x.device().device_id;
  switch (K) {
    case 1:
      launch_fused_dispatch<1>(L, xb, bpt, ids, w, rc, bp, lt, dev, stream);
      break;
    case 2:
      launch_fused_dispatch<2>(L, xb, bpt, ids, w, rc, bp, lt, dev, stream);
      break;
    case 4:
      launch_fused_dispatch<4>(L, xb, bpt, ids, w, rc, bp, lt, dev, stream);
      break;
    case 6:
      launch_fused_dispatch<6>(L, xb, bpt, ids, w, rc, bp, lt, dev, stream);
      break;
    case 8:
      launch_fused_dispatch<8>(L, xb, bpt, ids, w, rc, bp, lt, dev, stream);
      break;
    default:
      TVM_FFI_ICHECK(false) << "dispatch_fused: unsupported top_k " << K
                            << " (supported: 1, 2, 4, 6, 8)";
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm90_cake_dispatch_fused_bf16, sm90_cake_dispatch_fused_bf16);
