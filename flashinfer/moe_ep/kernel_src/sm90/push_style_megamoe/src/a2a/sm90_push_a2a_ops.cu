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

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <tvm/ffi/extra/module.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "sm90_push_a2a.cuh"
#include "tvm_ffi_utils.h"

using tvm::ffi::TensorView;
using namespace flashinfer::sm90_push;

namespace {

constexpr int kMaxEpSize = 32;  // single-node NVLink domain; wait kernels use one warp
constexpr int kMaxTopK = 8;     // matches the pipe's top_k validation; sizes in-register row lists

__device__ __forceinline__ float atomic_max_nonneg(float* addr, float v) {
  return __int_as_float(atomicMax(reinterpret_cast<int*>(addr), __float_as_int(v)));
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
  TVM_FFI_ICHECK(peer_bases != 0) << "sm90_push: null peer_bases";
  TVM_FFI_ICHECK(ep_size >= 1 && ep_size <= kMaxEpSize)
      << "sm90_push: ep_size " << ep_size << " out of [1, " << kMaxEpSize << "]";
  TVM_FFI_ICHECK(rank >= 0 && rank < ep_size) << "sm90_push: bad rank " << rank;
  TVM_FFI_ICHECK(num_local_experts >= 1) << "sm90_push: bad num_local_experts";
  TVM_FFI_ICHECK(hidden >= 128 && hidden % 128 == 0)
      << "sm90_push: hidden must be a positive multiple of 128";
  TVM_FFI_ICHECK(top_k >= 1 && t_cap >= 1 && pool_rows >= 1 && meta_rows >= pool_rows)
      << "sm90_push: bad capacity args";
  TVM_FFI_ICHECK(meta_rows <= INT32_MAX) << "sm90_push: meta_rows overflows the packed head";
  TVM_FFI_ICHECK(base_cells_offset >=
                 pool_head_offset + kPoolHeadStorageBytes + kAbortCellStorageBytes)
      << "sm90_push: pool-head padding does not contain the abort cell";
}

// unsigned increment: int32 overflow at 2^31 rounds would be UB; uint32
// wrap is well-defined and every tag check is equality-based
__global__ void bump_tag_kernel(int32_t* round_ctr) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    auto* u = reinterpret_cast<uint32_t*>(round_ctr);
    *u = *u + 1u;
  }
}

__global__ void publish_abort_kernel(PushLayout L) {
  int peer = threadIdx.x;
  if (peer < L.ep_size) {
    st_release_sys_u64(L.abort_cell(peer), pack_count_tag(L.rank + 1, 0));
  }
}

__global__ void wait_acks_kernel(PushLayout L, const int32_t* __restrict__ round_ctr) {
  int d = threadIdx.x;
  if (d >= L.ep_size) return;
  uint32_t current = static_cast<uint32_t>(*round_ctr);
  uint32_t want = current - 1u;
  wait_tag_u64(L, L.ack_cell(L.rank, d), want);
}

template <int TOP_K>
__global__ void count_kernel(PushLayout L, const int32_t* __restrict__ topk_ids,
                             int num_experts_total, int32_t* __restrict__ lc,
                             int32_t* __restrict__ loff, const int32_t* __restrict__ round_ctr,
                             int local_num_tokens) {
  int route = blockIdx.x * blockDim.x + threadIdx.x;
  if (route >= local_num_tokens * TOP_K) return;
  int eid = topk_ids[route];
  if (eid < 0) {
    loff[route] = -1;
    return;
  }
  if (eid >= num_experts_total) {
    printf("sm90_push: invalid expert id %d (num_experts=%d) at route %d\n", eid, num_experts_total,
           route);
    publish_abort_all(L, static_cast<uint32_t>(*round_ctr));
    asm volatile("trap;");
  }
  loff[route] = atomicAdd(&lc[eid], 1);
}

template <int TOP_K>
__global__ void count_dedup_kernel(PushLayout L, const int32_t* __restrict__ topk_ids,
                                   int num_local_experts, int num_experts_total,
                                   int32_t* __restrict__ lc, int32_t* __restrict__ loff,
                                   int32_t* __restrict__ pc, int32_t* __restrict__ ploff,
                                   const int32_t* __restrict__ round_ctr, int local_num_tokens) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= local_num_tokens) return;
  int dst[TOP_K];
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) {
    int route = t * TOP_K + k;
    int eid = topk_ids[route];
    if (eid < 0) {  // masked route: not counted, never a payload carrier
      loff[route] = -1;
      ploff[route] = -1;
      dst[k] = -1;
      continue;
    }
    if (eid >= num_experts_total) {
      printf("sm90_push: invalid expert id %d (num_experts=%d) at route %d\n", eid,
             num_experts_total, route);
      publish_abort_all(L, static_cast<uint32_t>(*round_ctr));
      asm volatile("trap;");
    }
    dst[k] = expert_owner_rank(eid, num_local_experts);
    loff[route] = atomicAdd(&lc[eid], 1);
    bool carrier = true;
#pragma unroll
    for (int j = 0; j < TOP_K; ++j) {
      if (j < k && dst[j] == dst[k]) carrier = false;
    }
    ploff[route] = carrier ? atomicAdd(&pc[dst[k]], 1) : -1;
  }
}

__global__ void reserve_publish_kernel(PushLayout L, const int32_t* __restrict__ lc,
                                       int32_t* __restrict__ keybase,
                                       const int32_t* __restrict__ round_ctr) {
  int E = L.num_local_experts;
  int eps = L.ep_size;
  int nkeys = E * eps;
  uint32_t tag = static_cast<uint32_t>(*round_ctr);
  int tid = threadIdx.x;
  if (tid < eps) {
    int total = 0;
    for (int e = 0; e < E; ++e) total += lc[tid * E + e];
    int base = 0;
    if (total > 0) {
      base = static_cast<int>(atomicAdd_system(L.pool_head(tid), static_cast<unsigned int>(total)));
      if (base + total > L.pool_rows) {
        printf(
            "sm90_push: pool overflow on dst %d (base %d + rows %d > pool_rows %d); "
            "raise capacity_factor\n",
            tid, base, total, L.pool_rows);
        publish_abort_all(L, tag);
        asm volatile("trap;");
      }
    }
    st_release_sys_u64(L.base_cell(tid, L.rank), pack_count_tag(base, tag));
    int acc = base;
    for (int e = 0; e < E; ++e) {
      keybase[tid * E + e] = acc;
      acc += lc[tid * E + e];
    }
  }
  for (int key = tid; key < nkeys; key += blockDim.x) {
    if (lc[key] == 0) {
      st_release_sys_u64(L.count_cell(key / E, key % E, L.rank), pack_count_tag(0, tag));
    }
  }
}

__global__ void reserve_publish_dedup_kernel(PushLayout L, const int32_t* __restrict__ lc,
                                             const int32_t* __restrict__ pc,
                                             int32_t* __restrict__ keybase,
                                             int32_t* __restrict__ pkeybase,
                                             const int32_t* __restrict__ round_ctr) {
  int E = L.num_local_experts;
  int eps = L.ep_size;
  int nkeys = E * eps;
  uint32_t tag = static_cast<uint32_t>(*round_ctr);
  int tid = threadIdx.x;
  if (tid < eps) {
    int total_meta = 0;
    for (int e = 0; e < E; ++e) total_meta += lc[tid * E + e];
    int total_payload = pc[tid];
    int meta_base = 0, payload_base = 0;
    if (total_meta > 0) {  // >= 1 route implies >= 1 carrier and vice versa
      unsigned long long add = (static_cast<unsigned long long>(total_payload) << 32) |
                               static_cast<unsigned long long>(total_meta);
      unsigned long long old = atomicAdd_system(L.pool_head64(tid), add);
      meta_base = static_cast<int>(old & 0xffffffffull);
      payload_base = static_cast<int>(old >> 32);
      if (meta_base + total_meta > L.meta_rows || payload_base + total_payload > L.pool_rows) {
        printf(
            "sm90_push: dedup pool overflow on dst %d (meta %d+%d > %d or payload %d+%d > %d); "
            "raise capacity_factor\n",
            tid, meta_base, total_meta, L.meta_rows, payload_base, total_payload, L.pool_rows);
        publish_abort_all(L, tag);
        asm volatile("trap;");
      }
    }
    st_release_sys_u64(L.base_cell(tid, L.rank), pack_count_tag(meta_base, tag));
    pkeybase[tid] = payload_base;
    int acc = meta_base;
    for (int e = 0; e < E; ++e) {
      keybase[tid * E + e] = acc;
      acc += lc[tid * E + e];
    }
  }
  for (int key = tid; key < nkeys; key += blockDim.x) {
    if (lc[key] == 0) {
      st_release_sys_u64(L.count_cell(key / E, key % E, L.rank), pack_count_tag(0, tag));
    }
  }
}

template <int TOP_K>
__global__ void store_publish_kernel(PushLayout L, const uint8_t* __restrict__ x_bytes,
                                     int bytes_per_token, const int32_t* __restrict__ topk_ids,
                                     const float* __restrict__ topk_w,
                                     const int32_t* __restrict__ loff,
                                     const int32_t* __restrict__ lc, int32_t* __restrict__ done,
                                     const int32_t* __restrict__ keybase,
                                     const int32_t* __restrict__ round_ctr, int local_num_tokens) {
  int t = blockIdx.x;
  if (t >= local_num_tokens) return;
  int E = L.num_local_experts;
  __shared__ int s_rank[TOP_K];
  __shared__ int s_slot[TOP_K];
  __shared__ int s_key[TOP_K];
  if (threadIdx.x == 0) {
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
      int route = t * TOP_K + k;
      int eid = topk_ids[route];
      int lo = loff[route];
      if (eid < 0 || lo < 0) {  // masked route: not counted, not stored
        s_rank[k] = -1;
        s_slot[k] = 0;
        s_key[k] = -1;
        continue;
      }
      int r = expert_owner_rank(eid, E);
      int slot = keybase[eid] + lo;
      s_rank[k] = r;
      s_slot[k] = slot;
      s_key[k] = eid;
      SlotMeta* m = L.pool_meta(r, slot);
      m->src_token = t;
      m->src_rank_k = pack_rank_k(L.rank, k);
      m->weight = topk_w[route];
      m->payload_slot = slot;  // non-dedup: payload lives at the record's own row
    }
  }
  __syncthreads();
  uint4* dst[TOP_K];
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) {
    dst[k] = (s_rank[k] < 0) ? nullptr : reinterpret_cast<uint4*>(L.pool_row(s_rank[k], s_slot[k]));
  }
  const uint4* src =
      reinterpret_cast<const uint4*>(x_bytes + static_cast<uint64_t>(t) * bytes_per_token);
  int nv = bytes_per_token >> 4;
  for (int v = threadIdx.x; v < nv; v += blockDim.x) {
    uint4 pk = src[v];
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
      if (dst[k] != nullptr) dst[k][v] = pk;
    }
  }
  __syncthreads();  // whole block's remote stores issued before the publish tail
  if (threadIdx.x == 0) {
    __threadfence_system();  // release: order this block's remote payload+meta stores
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
      int key = s_key[k];
      if (key < 0) continue;
      int prev = atomicAdd(&done[key], 1);
      if (prev + 1 == lc[key]) {  // last route for this key: publish
        __threadfence_system();   // order the observed done increments
        st_release_sys_u64(L.count_cell(key / E, key % E, L.rank),
                           pack_count_tag(lc[key], static_cast<uint32_t>(*round_ctr)));
      }
    }
  }
}

template <int TOP_K>
__global__ void store_publish_fp8_kernel(
    PushLayout L, const __nv_bfloat16* __restrict__ x, int H, const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_w, const int32_t* __restrict__ loff,
    const int32_t* __restrict__ lc, int32_t* __restrict__ done, const int32_t* __restrict__ keybase,
    const int32_t* __restrict__ round_ctr, int local_num_tokens) {
  int t = blockIdx.x;
  if (t >= local_num_tokens) return;
  int E = L.num_local_experts;
  int nkb = H >> 7, nv = H >> 4, tid = threadIdx.x;
  __shared__ int s_rank[TOP_K];
  __shared__ int s_slot[TOP_K];
  __shared__ int s_key[TOP_K];
  extern __shared__ __align__(16) float ssc[];  // nkb per-128-block absmax
  if (tid == 0) {
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
      int route = t * TOP_K + k;
      int eid = topk_ids[route];
      int lo = loff[route];
      if (eid < 0 || lo < 0) {
        s_rank[k] = -1;
        s_slot[k] = 0;
        s_key[k] = -1;
        continue;
      }
      int r = expert_owner_rank(eid, E);
      int slot = keybase[eid] + lo;
      s_rank[k] = r;
      s_slot[k] = slot;
      s_key[k] = eid;
      SlotMeta* m = L.pool_meta(r, slot);
      m->src_token = t;
      m->src_rank_k = pack_rank_k(L.rank, k);
      m->weight = topk_w[route];
      m->payload_slot = slot;  // non-dedup: payload lives at the record's own row
    }
  }
  for (int b = tid; b < nkb; b += blockDim.x) ssc[b] = 0.0f;
  __syncthreads();
  const __nv_bfloat16* xt = x + static_cast<uint64_t>(t) * H;
  for (int v = tid; v < nv; v += blockDim.x) {  // pass 1: absmax
    const __nv_bfloat16* p = xt + v * 16;
    float a = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) a = fmaxf(a, fabsf(__bfloat162float(p[j])));
    atomic_max_nonneg(&ssc[(v * 16) >> 7], a);
  }
  __syncthreads();
  uint8_t* dst[TOP_K];
  float* sc_dst[TOP_K];
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) {
    if (s_rank[k] < 0) {
      dst[k] = nullptr;
      sc_dst[k] = nullptr;
      continue;
    }
    dst[k] = L.pool_row(s_rank[k], s_slot[k]);  // bytes_per_row == H
    sc_dst[k] = L.pool_sc_row(s_rank[k], s_slot[k]);
  }
  for (int v = tid; v < nv; v += blockDim.x) {  // pass 2: quant + 16B stores
    float m = ssc[(v * 16) >> 7];
    float sc = (m > 0.0f) ? (m / 448.0f) : 1.0f;
    const __nv_bfloat16* p = xt + v * 16;
    alignas(16) __nv_fp8_e4m3 o[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      float f = __bfloat162float(p[j]) / sc;
      o[j] = __nv_fp8_e4m3(fminf(fmaxf(f, -448.0f), 448.0f));
    }
    uint4 pk;
    memcpy(&pk, o, sizeof(pk));
#pragma unroll
    for (int k = 0; k < TOP_K; ++k)
      if (dst[k] != nullptr) *reinterpret_cast<uint4*>(dst[k] + v * 16) = pk;
  }
  for (int b = tid; b < nkb; b += blockDim.x) {
    float m = ssc[b];
    float sc = (m > 0.0f) ? (m / 448.0f) : 1.0f;
#pragma unroll
    for (int k = 0; k < TOP_K; ++k)
      if (sc_dst[k] != nullptr) sc_dst[k][b] = sc;
  }
  __syncthreads();
  if (tid == 0) {  // fused last-block publish (same causality chain)
    __threadfence_system();
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
      int key = s_key[k];
      if (key < 0) continue;
      int prev = atomicAdd(&done[key], 1);
      if (prev + 1 == lc[key]) {
        __threadfence_system();
        st_release_sys_u64(L.count_cell(key / E, key % E, L.rank),
                           pack_count_tag(lc[key], static_cast<uint32_t>(*round_ctr)));
      }
    }
  }
}

template <int TOP_K>
__global__ void store_publish_dedup_kernel(
    PushLayout L, const uint8_t* __restrict__ x_bytes, int bytes_per_token,
    const int32_t* __restrict__ topk_ids, const float* __restrict__ topk_w,
    const int32_t* __restrict__ loff, const int32_t* __restrict__ ploff,
    const int32_t* __restrict__ lc, int32_t* __restrict__ done, const int32_t* __restrict__ keybase,
    const int32_t* __restrict__ pkeybase, const int32_t* __restrict__ round_ctr,
    int local_num_tokens) {
  int t = blockIdx.x;
  if (t >= local_num_tokens) return;
  int E = L.num_local_experts;
  __shared__ int s_rank[TOP_K];   // destination rank per route (-1 = masked)
  __shared__ int s_pslot[TOP_K];  // the token's payload row on that rank
  __shared__ int s_key[TOP_K];
  __shared__ int s_carrier[TOP_K];  // 1 = this route stores the payload
  if (threadIdx.x == 0) {
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
      int route = t * TOP_K + k;
      int eid = topk_ids[route];
      int lo = loff[route];
      if (eid < 0 || lo < 0) {
        s_rank[k] = -1;
        s_pslot[k] = 0;
        s_key[k] = -1;
        s_carrier[k] = 0;
        continue;
      }
      int r = expert_owner_rank(eid, E);
      int plo = ploff[route];
      int pslot;
      if (plo >= 0) {  // carrier: first valid route of this token to rank r
        pslot = pkeybase[r] + plo;
      } else {  // duplicate destination: reuse the carrier's row (k' < k)
        pslot = 0;
#pragma unroll
        for (int j = 0; j < TOP_K; ++j) {
          if (j < k && s_rank[j] == r) pslot = s_pslot[j];
        }
      }
      s_rank[k] = r;
      s_pslot[k] = pslot;
      s_key[k] = eid;
      s_carrier[k] = (plo >= 0) ? 1 : 0;
      SlotMeta* m = L.pool_meta(r, keybase[eid] + lo);
      m->src_token = t;
      m->src_rank_k = pack_rank_k(L.rank, k);
      m->weight = topk_w[route];
      m->payload_slot = pslot;
    }
  }
  __syncthreads();
  uint4* dst[TOP_K];
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) {
    dst[k] = (s_rank[k] < 0 || !s_carrier[k])
                 ? nullptr
                 : reinterpret_cast<uint4*>(L.pool_row(s_rank[k], s_pslot[k]));
  }
  const uint4* src =
      reinterpret_cast<const uint4*>(x_bytes + static_cast<uint64_t>(t) * bytes_per_token);
  int nv = bytes_per_token >> 4;
  for (int v = threadIdx.x; v < nv; v += blockDim.x) {
    uint4 pk = src[v];
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
      if (dst[k] != nullptr) dst[k][v] = pk;
    }
  }
  __syncthreads();  // whole block's remote stores issued before the publish tail
  if (threadIdx.x == 0) {
    __threadfence_system();
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
      int key = s_key[k];
      if (key < 0) continue;
      int prev = atomicAdd(&done[key], 1);
      if (prev + 1 == lc[key]) {
        __threadfence_system();
        st_release_sys_u64(L.count_cell(key / E, key % E, L.rank),
                           pack_count_tag(lc[key], static_cast<uint32_t>(*round_ctr)));
      }
    }
  }
}

template <int TOP_K>
__global__ void store_publish_dedup_fp8_kernel(
    PushLayout L, const __nv_bfloat16* __restrict__ x, int H, const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_w, const int32_t* __restrict__ loff,
    const int32_t* __restrict__ ploff, const int32_t* __restrict__ lc, int32_t* __restrict__ done,
    const int32_t* __restrict__ keybase, const int32_t* __restrict__ pkeybase,
    const int32_t* __restrict__ round_ctr, int local_num_tokens) {
  int t = blockIdx.x;
  if (t >= local_num_tokens) return;
  int E = L.num_local_experts;
  int nkb = H >> 7, nv = H >> 4, tid = threadIdx.x;
  __shared__ int s_rank[TOP_K];
  __shared__ int s_pslot[TOP_K];
  __shared__ int s_key[TOP_K];
  __shared__ int s_carrier[TOP_K];
  extern __shared__ __align__(16) float ssc[];  // nkb per-128-block absmax
  if (tid == 0) {
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
      int route = t * TOP_K + k;
      int eid = topk_ids[route];
      int lo = loff[route];
      if (eid < 0 || lo < 0) {
        s_rank[k] = -1;
        s_pslot[k] = 0;
        s_key[k] = -1;
        s_carrier[k] = 0;
        continue;
      }
      int r = expert_owner_rank(eid, E);
      int plo = ploff[route];
      int pslot;
      if (plo >= 0) {
        pslot = pkeybase[r] + plo;
      } else {
        pslot = 0;
#pragma unroll
        for (int j = 0; j < TOP_K; ++j) {
          if (j < k && s_rank[j] == r) pslot = s_pslot[j];
        }
      }
      s_rank[k] = r;
      s_pslot[k] = pslot;
      s_key[k] = eid;
      s_carrier[k] = (plo >= 0) ? 1 : 0;
      SlotMeta* m = L.pool_meta(r, keybase[eid] + lo);
      m->src_token = t;
      m->src_rank_k = pack_rank_k(L.rank, k);
      m->weight = topk_w[route];
      m->payload_slot = pslot;
    }
  }
  for (int b = tid; b < nkb; b += blockDim.x) ssc[b] = 0.0f;
  __syncthreads();
  const __nv_bfloat16* xt = x + static_cast<uint64_t>(t) * H;
  for (int v = tid; v < nv; v += blockDim.x) {  // pass 1: absmax
    const __nv_bfloat16* p = xt + v * 16;
    float a = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) a = fmaxf(a, fabsf(__bfloat162float(p[j])));
    atomic_max_nonneg(&ssc[(v * 16) >> 7], a);
  }
  __syncthreads();
  uint8_t* dst[TOP_K];
  float* sc_dst[TOP_K];
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) {
    if (s_rank[k] < 0 || !s_carrier[k]) {
      dst[k] = nullptr;
      sc_dst[k] = nullptr;
      continue;
    }
    dst[k] = L.pool_row(s_rank[k], s_pslot[k]);  // bytes_per_row == H
    sc_dst[k] = L.pool_sc_row(s_rank[k], s_pslot[k]);
  }
  for (int v = tid; v < nv; v += blockDim.x) {  // pass 2: quant + 16B stores
    float m = ssc[(v * 16) >> 7];
    float sc = (m > 0.0f) ? (m / 448.0f) : 1.0f;
    const __nv_bfloat16* p = xt + v * 16;
    alignas(16) __nv_fp8_e4m3 o[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      float f = __bfloat162float(p[j]) / sc;
      o[j] = __nv_fp8_e4m3(fminf(fmaxf(f, -448.0f), 448.0f));
    }
    uint4 pk;
    memcpy(&pk, o, sizeof(pk));
#pragma unroll
    for (int k = 0; k < TOP_K; ++k)
      if (dst[k] != nullptr) *reinterpret_cast<uint4*>(dst[k] + v * 16) = pk;
  }
  for (int b = tid; b < nkb; b += blockDim.x) {
    float m = ssc[b];
    float sc = (m > 0.0f) ? (m / 448.0f) : 1.0f;
#pragma unroll
    for (int k = 0; k < TOP_K; ++k)
      if (sc_dst[k] != nullptr) sc_dst[k][b] = sc;
  }
  __syncthreads();
  if (tid == 0) {  // fused last-block publish (same causality chain)
    __threadfence_system();
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
      int key = s_key[k];
      if (key < 0) continue;
      int prev = atomicAdd(&done[key], 1);
      if (prev + 1 == lc[key]) {
        __threadfence_system();
        st_release_sys_u64(L.count_cell(key / E, key % E, L.rank),
                           pack_count_tag(lc[key], static_cast<uint32_t>(*round_ctr)));
      }
    }
  }
}

__global__ void wait_prefix_kernel(
    PushLayout L, const int32_t* __restrict__ round_ctr, int32_t* __restrict__ rows_per_src,
    int64_t* __restrict__ offsets, int32_t* __restrict__ seg_src_base,
    int32_t* __restrict__ seg_out_base, int32_t* __restrict__ pad_base, int32_t* __restrict__ m_dev,
    int32_t* __restrict__ p_dev, int32_t* __restrict__ next_row, int m_cap, int max_rows_per_key) {
  extern __shared__ int32_t smem[];
  int E = L.num_local_experts;
  int eps = L.ep_size;
  int nkeys = E * eps;
  int32_t* cnt = smem;            // [nkeys], indexed e * eps + s
  int32_t* sbase = smem + nkeys;  // [eps]
  int32_t* sacc = sbase + eps;    // [eps]
  uint32_t tag = static_cast<uint32_t>(*round_ctr);
  uint64_t* cells = reinterpret_cast<uint64_t*>(L.window(L.rank) + L.count_cells_offset);
  for (int i = threadIdx.x; i < nkeys; i += blockDim.x) {
    int c = wait_tag_u64(L, cells + i, tag);
    if (c < 0 || c > max_rows_per_key) {  // protocol corruption, not overflow
      printf("sm90_push: bad count %d for (expert %d, src %d), max %d\n", c, i / eps, i % eps,
             max_rows_per_key);
      publish_abort_all(L, tag);
      asm volatile("trap;");
    }
    cnt[i] = c;
  }
  for (int s = threadIdx.x; s < eps; s += blockDim.x) {
    sbase[s] = wait_tag_u64(L, L.base_cell(L.rank, s), tag);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int64_t acc = 0;
    for (int s = 0; s < eps; ++s) sacc[s] = 0;
    for (int e = 0; e < E; ++e) {
      offsets[e] = acc;
      // deep_gemm::compute_padded_offset(acc, e): 32-align with headroom
      pad_base[e] = static_cast<int32_t>((acc + static_cast<int64_t>(e) * 31) / 32 * 32);
      for (int s = 0; s < eps; ++s) {
        int idx = e * eps + s;
        seg_out_base[idx] = static_cast<int32_t>(acc);
        seg_src_base[idx] = sbase[s] + sacc[s];
        sacc[s] += cnt[idx];
        acc += cnt[idx];
      }
    }
    if (acc > static_cast<int64_t>(m_cap)) {
      printf("sm90_push: accumulated row count %lld exceeds m_cap %d\n",
             static_cast<long long>(acc), m_cap);
      publish_abort_all(L, tag);
      asm volatile("trap;");
      return;
    }
    offsets[E] = acc;
    seg_out_base[nkeys] = static_cast<int32_t>(acc);
    *m_dev = static_cast<int32_t>(acc);
    int64_t p = (static_cast<int64_t>(m_cap) + static_cast<int64_t>(E) * 31) / 32 * 32;
    *p_dev = static_cast<int32_t>(p < 1 ? 1 : p);
    *next_row = 0;  // reset the persistent-compact work queue
  }
  __syncthreads();
  for (int s = threadIdx.x; s < eps; s += blockDim.x) {
    int rows = 0;
    for (int e = 0; e < E; ++e) rows += cnt[e * eps + s];
    rows_per_src[s] = rows;
    if (rows == 0) {  // no combine rows will flow back to s: publish now
      st_release_sys_u64(L.cdone_cell(s, L.rank), pack_count_tag(0, tag));
    }
  }
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

__device__ __forceinline__ void quant_row_1x128(const __nv_bfloat16* __restrict__ src,
                                                uint8_t* __restrict__ dst_row, float* ssc, int K,
                                                float* __restrict__ sfa, int64_t sfa_col,
                                                int64_t P) {
  int nkb = K >> 7, nv = K >> 4, tid = threadIdx.x;
  for (int b = tid; b < nkb; b += blockDim.x) ssc[b] = 0.0f;
  __syncthreads();
  for (int v = tid; v < nv; v += blockDim.x) {
    const __nv_bfloat16* p = src + v * 16;
    float a = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) a = fmaxf(a, fabsf(__bfloat162float(p[j])));
    atomic_max_nonneg(&ssc[(v * 16) >> 7], a);
  }
  __syncthreads();
  for (int v = tid; v < nv; v += blockDim.x) {
    float m = ssc[(v * 16) >> 7];
    float sc = (m > 0.0f) ? (m / 448.0f) : 1.0f;
    const __nv_bfloat16* p = src + v * 16;
    alignas(16) __nv_fp8_e4m3 o[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      float f = __bfloat162float(p[j]) / sc;
      o[j] = __nv_fp8_e4m3(fminf(fmaxf(f, -448.0f), 448.0f));
    }
    uint4 pk;
    memcpy(&pk, o, sizeof(pk));
    *reinterpret_cast<uint4*>(dst_row + v * 16) = pk;
  }
  for (int b = tid; b < nkb; b += blockDim.x) {
    float m = ssc[b];
    sfa[static_cast<int64_t>(b) * P + sfa_col] = (m > 0.0f) ? (m / 448.0f) : 1.0f;
  }
}

__global__ void compact_fp8_persistent_kernel(
    PushLayout L, const int32_t* __restrict__ seg_src_base,
    const int32_t* __restrict__ seg_out_base, const int64_t* __restrict__ offsets,
    const int32_t* __restrict__ pad_base, const int32_t* __restrict__ m_dev,
    const int32_t* __restrict__ p_dev, int32_t* __restrict__ next_row, uint8_t* __restrict__ a_fp8,
    float* __restrict__ sfa, int32_t* __restrict__ meta_out, int32_t* __restrict__ row_expert,
    int K) {
  __shared__ int s_row;
  int eps = L.ep_size;
  int nkeys = L.num_local_experts * eps;
  int nkb = K >> 7;
  for (;;) {
    __syncthreads();  // protect s_row from the previous iteration's readers
    if (threadIdx.x == 0) s_row = atomicAdd(next_row, 1);
    __syncthreads();
    int row = s_row;
    if (row >= *m_dev) return;
    int seg = find_segment(seg_out_base, nkeys, row);
    int e = seg / eps;
    int rec = seg_src_base[seg] + (row - seg_out_base[seg]);
    // typed SlotMeta reads (float weight through an int lvalue would be UB)
    const SlotMeta* mi = L.pool_meta(L.rank, rec);
    int pslot = mi->payload_slot;
    const uint4* src4 = reinterpret_cast<const uint4*>(L.pool_row(L.rank, pslot));
    uint4* out4 = reinterpret_cast<uint4*>(a_fp8 + static_cast<int64_t>(row) * K);
    for (int v = threadIdx.x; v < (K >> 4); v += blockDim.x) out4[v] = src4[v];
    const float* scs = L.pool_sc_row(L.rank, pslot);
    int64_t col = static_cast<int64_t>(pad_base[e]) + (row - static_cast<int>(offsets[e]));
    int64_t P = static_cast<int64_t>(*p_dev);
    for (int b = threadIdx.x; b < nkb; b += blockDim.x) {
      sfa[static_cast<int64_t>(b) * P + col] = scs[b];
    }
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

__global__ void compact_quant_persistent_kernel(
    PushLayout L, const int32_t* __restrict__ seg_src_base,
    const int32_t* __restrict__ seg_out_base, const int64_t* __restrict__ offsets,
    const int32_t* __restrict__ pad_base, const int32_t* __restrict__ m_dev,
    const int32_t* __restrict__ p_dev, int32_t* __restrict__ next_row, uint8_t* __restrict__ a_fp8,
    float* __restrict__ sfa, int32_t* __restrict__ meta_out, int32_t* __restrict__ row_expert,
    int K) {
  __shared__ int s_row;
  extern __shared__ __align__(16) float ssc[];
  int eps = L.ep_size;
  int nkeys = L.num_local_experts * eps;
  for (;;) {
    __syncthreads();
    if (threadIdx.x == 0) s_row = atomicAdd(next_row, 1);
    __syncthreads();
    int row = s_row;
    if (row >= *m_dev) return;
    int seg = find_segment(seg_out_base, nkeys, row);
    int e = seg / eps;
    int rec = seg_src_base[seg] + (row - seg_out_base[seg]);
    const SlotMeta* mi = L.pool_meta(L.rank, rec);  // typed reads; see fp8 compact
    int pslot = mi->payload_slot;
    const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(L.pool_row(L.rank, pslot));
    quant_row_1x128(src, a_fp8 + static_cast<int64_t>(row) * K, ssc, K, sfa,
                    static_cast<int64_t>(pad_base[e]) + (row - static_cast<int>(offsets[e])),
                    static_cast<int64_t>(*p_dev));
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

__global__ void silu_mul_quant_grouped_kernel(
    const __nv_bfloat16* __restrict__ h, const int64_t* __restrict__ offsets,
    const int32_t* __restrict__ pad_base, const int32_t* __restrict__ m_dev,
    const int32_t* __restrict__ p_dev, const int32_t* __restrict__ row_expert,
    uint8_t* __restrict__ a_fp8, float* __restrict__ sfa, int I) {
  int nkb = I >> 7, nv = I >> 4, tid = threadIdx.x;
  extern __shared__ __align__(16) float ssc[];  // nkb floats (padded to 16B), then I bf16
  __nv_bfloat16* sg = reinterpret_cast<__nv_bfloat16*>(ssc + ((nkb + 3) & ~3));
  for (int r = blockIdx.x; r < *m_dev; r += gridDim.x) {
    __syncthreads();  // ssc/sg may still have readers from the previous row
    int e = row_expert[r];
    const __nv_bfloat16* ha = h + static_cast<int64_t>(r) * 2 * I;
    const __nv_bfloat16* hb = ha + I;
    for (int b = tid; b < nkb; b += blockDim.x) ssc[b] = 0.0f;
    __syncthreads();
    for (int v = tid; v < nv; v += blockDim.x) {
      const __nv_bfloat16* pa = ha + v * 16;
      const __nv_bfloat16* pb = hb + v * 16;
      alignas(16) __nv_bfloat16 o[16];
      float amax = 0.0f;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        float a = __bfloat162float(pa[j]);
        float b = __bfloat162float(pb[j]);
        o[j] = __float2bfloat16((a / (1.0f + __expf(-a))) * b);  // == silu_mul_gated
        amax = fmaxf(amax, fabsf(__bfloat162float(o[j])));
      }
      // 16 bf16 = 32 bytes = TWO uint4 stores
      uint4 pk2[2];
      memcpy(pk2, o, sizeof(pk2));
      reinterpret_cast<uint4*>(sg + v * 16)[0] = pk2[0];
      reinterpret_cast<uint4*>(sg + v * 16)[1] = pk2[1];
      atomic_max_nonneg(&ssc[(v * 16) >> 7], amax);
    }
    __syncthreads();
    uint8_t* dst_row = a_fp8 + static_cast<int64_t>(r) * I;
    int64_t sfa_col = static_cast<int64_t>(pad_base[e]) + (r - static_cast<int>(offsets[e]));
    int64_t P = static_cast<int64_t>(*p_dev);
    for (int v = tid; v < nv; v += blockDim.x) {
      float m = ssc[(v * 16) >> 7];
      float sc = (m > 0.0f) ? (m / 448.0f) : 1.0f;
      const __nv_bfloat16* p = sg + v * 16;
      alignas(16) __nv_fp8_e4m3 o[16];
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        float f = __bfloat162float(p[j]) / sc;
        o[j] = __nv_fp8_e4m3(fminf(fmaxf(f, -448.0f), 448.0f));
      }
      uint4 pk;
      memcpy(&pk, o, sizeof(pk));
      *reinterpret_cast<uint4*>(dst_row + v * 16) = pk;
    }
    for (int b = tid; b < nkb; b += blockDim.x) {
      float m = ssc[b];
      sfa[static_cast<int64_t>(b) * P + sfa_col] = (m > 0.0f) ? (m / 448.0f) : 1.0f;
    }
  }
}

__global__ void silu_mul_gated_kernel(__nv_bfloat16* __restrict__ g,
                                      const __nv_bfloat16* __restrict__ h,
                                      const int32_t* __restrict__ m_dev, int I) {
  // No smem, no cross-iteration state: plain block-stride over the real rows.
  for (int r = blockIdx.x; r < *m_dev; r += gridDim.x) {
    const __nv_bfloat16* hr = h + static_cast<int64_t>(r) * 2 * I;
    __nv_bfloat16* gr = g + static_cast<int64_t>(r) * I;
    for (int i = threadIdx.x; i < I; i += blockDim.x) {
      float a = __bfloat162float(hr[i]);
      float b = __bfloat162float(hr[I + i]);
      gr[i] = __float2bfloat16((a / (1.0f + __expf(-a))) * b);
    }
  }
}

// contiguous rows (already expert-grouped) -> fp8 + grouped sfa
__global__ void quant_grouped_kernel(const __nv_bfloat16* __restrict__ x,
                                     const int64_t* __restrict__ offsets,
                                     const int32_t* __restrict__ pad_base,
                                     const int32_t* __restrict__ m_dev,
                                     const int32_t* __restrict__ p_dev,
                                     const int32_t* __restrict__ row_expert,
                                     uint8_t* __restrict__ a_fp8, float* __restrict__ sfa, int K) {
  extern __shared__ __align__(16) float ssc[];
  for (int r = blockIdx.x; r < *m_dev; r += gridDim.x) {
    __syncthreads();  // ssc still has readers from the previous row's quant
    int e = row_expert[r];
    quant_row_1x128(x + static_cast<int64_t>(r) * K, a_fp8 + static_cast<int64_t>(r) * K, ssc, K,
                    sfa, static_cast<int64_t>(pad_base[e]) + (r - static_cast<int>(offsets[e])),
                    static_cast<int64_t>(*p_dev));
  }
}

__global__ void combine_publish_kernel(PushLayout L, const __nv_bfloat16* __restrict__ y,
                                       const int32_t* __restrict__ meta,
                                       const int32_t* __restrict__ m_dev,
                                       const int32_t* __restrict__ rows_per_src,
                                       int32_t* __restrict__ cdone_local,
                                       const int32_t* __restrict__ round_ctr) {
  for (int r = blockIdx.x; r < *m_dev; r += gridDim.x) {
    const int32_t* mrow = meta + static_cast<int64_t>(r) * 4;
    int dst = meta_src_rank(mrow);
    float w = meta_weight(mrow);
    const __nv_bfloat16* row = y + static_cast<uint64_t>(r) * L.hidden;
    __nv_bfloat16* out = L.combine_row(dst, meta_src_token(mrow), meta_route_k(mrow));
    // 16B vectorized P2P stores (2B stores waste NVLink); same fp32 mul +
    // RN cast per element as a scalar path, so results are bit-identical
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
}

__global__ void combine_publish_fp8_kernel(PushLayout L, const __nv_bfloat16* __restrict__ y,
                                           const int32_t* __restrict__ meta,
                                           const int32_t* __restrict__ m_dev,
                                           const int32_t* __restrict__ rows_per_src,
                                           int32_t* __restrict__ cdone_local,
                                           const int32_t* __restrict__ round_ctr) {
  int H = L.hidden, nkb = H >> 7, nv = H >> 4, tid = threadIdx.x;
  extern __shared__ __align__(16) float ssc[];
  for (int r = blockIdx.x; r < *m_dev; r += gridDim.x) {
    __syncthreads();  // ssc may still have readers from the previous row
    const int32_t* mrow = meta + static_cast<int64_t>(r) * 4;
    int dst = meta_src_rank(mrow);
    float w = meta_weight(mrow);
    const __nv_bfloat16* row = y + static_cast<uint64_t>(r) * H;
    for (int b = tid; b < nkb; b += blockDim.x) ssc[b] = 0.0f;
    __syncthreads();
    for (int v = tid; v < nv; v += blockDim.x) {  // pass 1: absmax of y*w
      const __nv_bfloat16* p = row + v * 16;
      float a = 0.0f;
#pragma unroll
      for (int j = 0; j < 16; ++j) a = fmaxf(a, fabsf(__bfloat162float(p[j]) * w));
      atomic_max_nonneg(&ssc[(v * 16) >> 7], a);
    }
    __syncthreads();
    uint8_t* q = L.cfp8_row(dst, meta_src_token(mrow), meta_route_k(mrow));
    float* scd = L.csc_row(dst, meta_src_token(mrow), meta_route_k(mrow));
    for (int v = tid; v < nv; v += blockDim.x) {  // pass 2: quant + 16B stores
      float m = ssc[(v * 16) >> 7];
      float sc = (m > 0.0f) ? (m / 448.0f) : 1.0f;
      const __nv_bfloat16* p = row + v * 16;
      alignas(16) __nv_fp8_e4m3 o[16];
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        float f = (__bfloat162float(p[j]) * w) / sc;
        o[j] = __nv_fp8_e4m3(fminf(fmaxf(f, -448.0f), 448.0f));
      }
      uint4 pk;
      memcpy(&pk, o, sizeof(pk));
      *reinterpret_cast<uint4*>(q + v * 16) = pk;
    }
    for (int b = tid; b < nkb; b += blockDim.x) {
      float m = ssc[b];
      scd[b] = (m > 0.0f) ? (m / 448.0f) : 1.0f;
    }
    __syncthreads();  // this row's remote stores all issued before the publish tail
    if (tid == 0) {
      __threadfence_system();
      int prev = atomicAdd(&cdone_local[dst], 1);
      if (prev + 1 == rows_per_src[dst]) {
        __threadfence_system();
        st_release_sys_u64(L.cdone_cell(dst, L.rank),
                           pack_count_tag(rows_per_src[dst], static_cast<uint32_t>(*round_ctr)));
      }
    }
  }
}

__global__ void wait_combine_kernel(PushLayout L, const int32_t* __restrict__ round_ctr) {
  int s = threadIdx.x;
  if (s >= L.ep_size) return;
  uint32_t tag = static_cast<uint32_t>(*round_ctr);
  wait_tag_u64(L, L.cdone_cell(L.rank, s), tag);
}

__device__ __forceinline__ void store_reduce_out(float* out, uint64_t idx, float v) {
  out[idx] = v;
}
__device__ __forceinline__ void store_reduce_out(__nv_bfloat16* out, uint64_t idx, float v) {
  out[idx] = __float2bfloat16(v);  // RN, matching every other f32->bf16 cast here
}

template <typename TOut>
__global__ void combine_reduce_kernel(TOut* __restrict__ out, PushLayout L, int num_tokens) {
  int t = blockIdx.x;
  if (t >= num_tokens) return;
  for (int i = threadIdx.x; i < L.hidden; i += blockDim.x) {
    float acc = 0.0f;
    for (int k = 0; k < L.top_k; ++k) {
      acc += __bfloat162float(L.combine_row(L.rank, t, k)[i]);
    }
    store_reduce_out(out, static_cast<uint64_t>(t) * L.hidden + i, acc);
  }
}

template <typename TOut>
__global__ void combine_reduce_fp8_kernel(TOut* __restrict__ out, PushLayout L, int num_tokens) {
  int t = blockIdx.x;
  if (t >= num_tokens) return;
  int H = L.hidden;
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    float acc = 0.0f;
    for (int k = 0; k < L.top_k; ++k) {
      const __nv_fp8_e4m3* q = reinterpret_cast<const __nv_fp8_e4m3*>(L.cfp8_row(L.rank, t, k));
      // sc == 0 == unwritten slot (begin_round zeroes scales only): skip, or
      // stale payload bytes (fp8 NaN patterns) pollute output via 0 * NaN
      float sc = L.csc_row(L.rank, t, k)[i >> 7];
      if (sc != 0.0f) acc += static_cast<float>(static_cast<__half>(q[i])) * sc;
    }
    store_reduce_out(out, static_cast<uint64_t>(t) * H + i, acc);
  }
}

__global__ void combine_group_build_kernel(
    PushLayout L, const int32_t* __restrict__ meta, const int32_t* __restrict__ m_dev,
    int32_t* __restrict__ grp_cnt, int32_t* __restrict__ grp_rows, int32_t* __restrict__ grp_list,
    int32_t* __restrict__ n_groups, int32_t* __restrict__ groups_per_src,
    const int32_t* __restrict__ round_ctr) {
  uint32_t const tag = static_cast<uint32_t>(*round_ctr);
  for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < *m_dev; r += gridDim.x * blockDim.x) {
    const int32_t* mrow = meta + static_cast<int64_t>(r) * 4;
    int src = meta_src_rank(mrow);
    int tok = meta_src_token(mrow);
    if (src < 0 || src >= L.ep_size || tok < 0 || tok >= L.t_cap) {
      printf("sm90_push: corrupt combine meta at row %d (src %d, token %d)\n", r, src, tok);
      publish_abort_all(L, tag);
      asm volatile("trap;");
    }
    int g = src * L.t_cap + tok;
    int pos = atomicAdd(&grp_cnt[g], 1);
    if (pos >= L.top_k) {  // a token has at most top_k routes in total
      printf("sm90_push: group (src %d, token %d) exceeds top_k %d\n", src, tok, L.top_k);
      publish_abort_all(L, tag);
      asm volatile("trap;");
    }
    grp_rows[static_cast<int64_t>(g) * L.top_k + pos] = r;
    if (pos == 0) {  // first row opens the group
      grp_list[atomicAdd(n_groups, 1)] = g;
      atomicAdd(&groups_per_src[src], 1);
    }
  }
}

__global__ void combine_publish_fp8_grouped_kernel(
    PushLayout L, const __nv_bfloat16* __restrict__ y, const int32_t* __restrict__ meta,
    const int32_t* __restrict__ grp_cnt, const int32_t* __restrict__ grp_rows,
    const int32_t* __restrict__ grp_list, const int32_t* __restrict__ n_groups,
    const int32_t* __restrict__ groups_per_src, int32_t* __restrict__ cdone_local,
    const int32_t* __restrict__ round_ctr, int col_tile) {
  int H = L.hidden, tid = threadIdx.x;
  extern __shared__ float smem_g[];
  float* ssc = smem_g;  // [pad4(col_tile/128)] per-128-block absmax of the SUM
  float* srow = smem_g + (((col_tile >> 7) + 3) & ~3);  // [col_tile] fp32 pre-reduced columns
  for (int idx = blockIdx.x; idx < *n_groups; idx += gridDim.x) {
    int g = grp_list[idx];  // same for the whole block -> uniform control flow
    int cnt = grp_cnt[g];   // >= 1 by worklist construction
    int dst = g / L.t_cap;
    int tok = g - dst * L.t_cap;
    int rows[kMaxTopK];
    int ks[kMaxTopK];
    float ws[kMaxTopK];
    const int32_t* glist = grp_rows + static_cast<int64_t>(g) * L.top_k;
    for (int i = 0; i < cnt; ++i) {
      rows[i] = glist[i];
      ks[i] = meta_route_k(meta + static_cast<int64_t>(rows[i]) * 4);
    }
    for (int i = 1; i < cnt; ++i) {  // insertion sort, cnt <= top_k <= 8
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
    for (int i = 0; i < cnt; ++i) {
      ws[i] = meta_weight(meta + static_cast<int64_t>(rows[i]) * 4);
    }
    uint8_t* q = L.cfp8_row_grouped(dst, tok, L.rank);
    float* scd = L.csc_row_grouped(dst, tok, L.rank);
    for (int t0 = 0; t0 < H; t0 += col_tile) {  // column tiles (see kernel comment)
      int tlen = min(col_tile, H - t0);         // H, col_tile both multiples of 128
      int tnkb = tlen >> 7, tnv = tlen >> 4;
      for (int b = tid; b < tnkb; b += blockDim.x) ssc[b] = 0.0f;
      __syncthreads();                               // ssc zeros visible before the atomic max
      for (int v = tid; v < tnv; v += blockDim.x) {  // pass 1: pre-reduce + absmax
        float acc[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) acc[j] = 0.0f;
        for (int i = 0; i < cnt; ++i) {
          const __nv_bfloat16* p = y + static_cast<uint64_t>(rows[i]) * H + t0 + v * 16;
          float w = ws[i];
#pragma unroll
          for (int j = 0; j < 16; ++j) acc[j] = fmaf(w, __bfloat162float(p[j]), acc[j]);
        }
        float amax = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          srow[v * 16 + j] = acc[j];
          amax = fmaxf(amax, fabsf(acc[j]));
        }
        atomic_max_nonneg(&ssc[(v * 16) >> 7], amax);
      }
      __syncthreads();  // srow/ssc complete before quantization reads them
      for (int v = tid; v < tnv; v += blockDim.x) {  // pass 2: quant + 16B P2P stores
        float m = ssc[(v * 16) >> 7];
        float sc = (m > 0.0f) ? (m / 448.0f) : 1.0f;
        alignas(16) __nv_fp8_e4m3 o[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          float f = srow[v * 16 + j] / sc;
          o[j] = __nv_fp8_e4m3(fminf(fmaxf(f, -448.0f), 448.0f));
        }
        uint4 pk;
        memcpy(&pk, o, sizeof(pk));
        *reinterpret_cast<uint4*>(q + t0 + v * 16) = pk;
      }
      for (int b = tid; b < tnkb; b += blockDim.x) {
        float m = ssc[b];
        scd[(t0 >> 7) + b] = (m > 0.0f) ? (m / 448.0f) : 1.0f;
      }
      __syncthreads();  // tile's smem readers + stores joined before the next tile zeroes
    }
    // All tiles' remote stores issued and barrier-joined above; publish ONCE.
    if (tid == 0) {
      __threadfence_system();
      int prev = atomicAdd(&cdone_local[dst], 1);
      if (prev + 1 == groups_per_src[dst]) {
        __threadfence_system();
        st_release_sys_u64(L.cdone_cell(dst, L.rank),
                           pack_count_tag(groups_per_src[dst], static_cast<uint32_t>(*round_ctr)));
      }
    }
  }
}

template <typename TOut>
__global__ void combine_reduce_fp8_grouped_kernel(TOut* __restrict__ out, PushLayout L,
                                                  int num_tokens) {
  int t = blockIdx.x;
  if (t >= num_tokens) return;
  int H = L.hidden;
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    float acc = 0.0f;
    for (int s = 0; s < L.ep_size; ++s) {
      const __nv_fp8_e4m3* q =
          reinterpret_cast<const __nv_fp8_e4m3*>(L.cfp8_row_grouped(L.rank, t, s));
      // sc == 0 marks an unwritten slot; skip (see per-route reduce)
      float sc = L.csc_row_grouped(L.rank, t, s)[i >> 7];
      if (sc != 0.0f) acc += static_cast<float>(static_cast<__half>(q[i])) * sc;
    }
    store_reduce_out(out, static_cast<uint64_t>(t) * H + i, acc);
  }
}

__global__ void ack_kernel(PushLayout L, const int32_t* __restrict__ round_ctr,
                           int32_t* __restrict__ lc, int32_t* __restrict__ done, int nreset_lc,
                           int nkeys) {
  for (int i = threadIdx.x; i < nreset_lc; i += blockDim.x) lc[i] = 0;
  for (int i = threadIdx.x; i < nkeys; i += blockDim.x) done[i] = 0;
  if (threadIdx.x == 0) *L.pool_head64(L.rank) = 0ull;
  __syncthreads();
  int d = threadIdx.x;
  if (d < L.ep_size) {
    __threadfence_system();  // order the resets before the ack becomes visible
    st_release_sys_u64(L.ack_cell(d, L.rank), pack_count_tag(0, static_cast<uint32_t>(*round_ctr)));
  }
}

// persistent-grid sizing, cached per device
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

bool sm90_push_p2p_native_atomics(int64_t device, int64_t peer) {
  if (device == peer) return true;
  int v = 0;
  cudaError_t e = cudaDeviceGetP2PAttribute(&v, cudaDevP2PAttrNativeAtomicSupported,
                                            static_cast<int>(device), static_cast<int>(peer));
  TVM_FFI_ICHECK(e == cudaSuccess) << "cudaDeviceGetP2PAttribute(NativeAtomicSupported, " << device
                                   << ", " << peer << ") failed: " << cudaGetErrorString(e);
  return v != 0;
}

void sm90_push_bump_tag(TensorView round_ctr) {
  CHECK_INPUT_AND_TYPE(round_ctr, dl_int32);
  auto stream = get_stream(round_ctr.device());
  bump_tag_kernel<<<1, 32, 0, stream>>>(static_cast<int32_t*>(round_ctr.data_ptr()));
}

void sm90_push_publish_abort(LAYOUT_PARAMS, TensorView device_anchor) {
  check_layout(LAYOUT_ARGS);
  CHECK_INPUT_AND_TYPE(device_anchor, dl_int32);
  auto L = build_layout(LAYOUT_ARGS);
  auto stream = get_stream(device_anchor.device());
  publish_abort_kernel<<<1, kMaxEpSize, 0, stream>>>(L);
}

void sm90_push_wait_acks(LAYOUT_PARAMS, TensorView round_ctr) {
  check_layout(LAYOUT_ARGS);
  CHECK_INPUT_AND_TYPE(round_ctr, dl_int32);
  auto L = build_layout(LAYOUT_ARGS);
  auto stream = get_stream(round_ctr.device());
  wait_acks_kernel<<<1, kMaxEpSize, 0, stream>>>(L,
                                                 static_cast<const int32_t*>(round_ctr.data_ptr()));
}

static void dispatch_common(TensorView x, TensorView topk_ids, TensorView topk_w, bool fp8_payload,
                            const PushLayout& L, TensorView lc, TensorView loff, TensorView done,
                            TensorView keybase, TensorView round_ctr) {
  int64_t T = x.size(0);
  int64_t H = x.size(1);
  int64_t top_k = topk_ids.size(1);
  int E = L.num_local_experts;
  int eps = L.ep_size;
  int nkeys = E * eps;
  int E_total = nkeys;

  CHECK_INPUT_AND_TYPE(x, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(topk_ids, dl_int32);
  CHECK_INPUT_AND_TYPE(topk_w, dl_float32);
  CHECK_INPUT_AND_TYPE(lc, dl_int32);
  CHECK_INPUT_AND_TYPE(loff, dl_int32);
  CHECK_INPUT_AND_TYPE(done, dl_int32);
  CHECK_INPUT_AND_TYPE(keybase, dl_int32);
  CHECK_INPUT_AND_TYPE(round_ctr, dl_int32);
  CHECK_DIM(2, x);
  CHECK_DIM(2, topk_ids);
  CHECK_DIM(2, topk_w);
  TVM_FFI_ICHECK(H == L.hidden) << "dispatch: x hidden " << H << " != layout " << L.hidden;
  TVM_FFI_ICHECK(T <= L.t_cap) << "dispatch: T " << T << " exceeds t_cap " << L.t_cap;
  TVM_FFI_ICHECK(top_k == L.top_k) << "dispatch: top_k mismatch";
  TVM_FFI_ICHECK(topk_ids.size(0) == T && topk_w.size(0) == T && topk_w.size(1) == top_k)
      << "dispatch: routing shape mismatch";
  TVM_FFI_ICHECK(lc.numel() >= nkeys && done.numel() >= nkeys && keybase.numel() >= nkeys)
      << "dispatch: key scratch too small";
  TVM_FFI_ICHECK(loff.numel() >= T * top_k) << "dispatch: loff scratch too small";
  int expected_row = fp8_payload ? static_cast<int>(H) : static_cast<int>(H) * 2;
  TVM_FFI_ICHECK(L.bytes_per_row == expected_row)
      << "dispatch: payload dtype does not match the pipe's payload config";

  auto stream = get_stream(x.device());
  auto* ids = static_cast<const int32_t*>(topk_ids.data_ptr());
  auto* w = static_cast<const float*>(topk_w.data_ptr());
  auto* lcp = static_cast<int32_t*>(lc.data_ptr());
  auto* loffp = static_cast<int32_t*>(loff.data_ptr());
  auto* donep = static_cast<int32_t*>(done.data_ptr());
  auto* kbp = static_cast<int32_t*>(keybase.data_ptr());
  auto* rc = static_cast<const int32_t*>(round_ctr.data_ptr());
  int lt = static_cast<int>(T);
  int nroutes = lt * static_cast<int>(top_k);

  dim3 blk(256);
  int smem = fp8_payload ? static_cast<int>(H / 128) * static_cast<int>(sizeof(float)) : 0;

#define DISPATCH_CASE(K)                                                                          \
  case K: {                                                                                       \
    if (nroutes > 0) {                                                                            \
      count_kernel<K>                                                                             \
          <<<(nroutes + 255) / 256, blk, 0, stream>>>(L, ids, E_total, lcp, loffp, rc, lt);       \
    }                                                                                             \
    reserve_publish_kernel<<<1, 256, 0, stream>>>(L, lcp, kbp, rc);                               \
    if (lt > 0) {                                                                                 \
      if (fp8_payload) {                                                                          \
        store_publish_fp8_kernel<K><<<lt, blk, smem, stream>>>(                                   \
            L, static_cast<const __nv_bfloat16*>(x.data_ptr()), static_cast<int>(H), ids, w,      \
            loffp, lcp, donep, kbp, rc, lt);                                                      \
      } else {                                                                                    \
        store_publish_kernel<K><<<lt, blk, 0, stream>>>(                                          \
            L, static_cast<const uint8_t*>(x.data_ptr()), static_cast<int>(H) * 2, ids, w, loffp, \
            lcp, donep, kbp, rc, lt);                                                             \
      }                                                                                           \
    }                                                                                             \
    break;                                                                                        \
  }
  switch (top_k) {
    DISPATCH_CASE(1)
    DISPATCH_CASE(2)
    DISPATCH_CASE(4)
    DISPATCH_CASE(6)
    DISPATCH_CASE(8)
    default:
      TVM_FFI_ICHECK(false) << "dispatch: unsupported top_k " << top_k
                            << " (supported: 1, 2, 4, 6, 8)";
  }
#undef DISPATCH_CASE
}

void sm90_push_dispatch(TensorView x, TensorView topk_ids, TensorView topk_w, LAYOUT_PARAMS,
                        TensorView lc, TensorView loff, TensorView done, TensorView keybase,
                        TensorView round_ctr) {
  check_layout(LAYOUT_ARGS);
  auto L = build_layout(LAYOUT_ARGS);
  dispatch_common(x, topk_ids, topk_w, /*fp8_payload=*/false, L, lc, loff, done, keybase,
                  round_ctr);
}

void sm90_push_dispatch_fp8(TensorView x, TensorView topk_ids, TensorView topk_w, LAYOUT_PARAMS,
                            TensorView lc, TensorView loff, TensorView done, TensorView keybase,
                            TensorView round_ctr) {
  check_layout(LAYOUT_ARGS);
  auto L = build_layout(LAYOUT_ARGS);
  dispatch_common(x, topk_ids, topk_w, /*fp8_payload=*/true, L, lc, loff, done, keybase, round_ctr);
}

static void dispatch_dedup_common(TensorView x, TensorView topk_ids, TensorView topk_w,
                                  bool fp8_payload, const PushLayout& L, TensorView lc,
                                  TensorView pc, TensorView loff, TensorView ploff, TensorView done,
                                  TensorView keybase, TensorView pkeybase, TensorView round_ctr) {
  int64_t T = x.size(0);
  int64_t H = x.size(1);
  int64_t top_k = topk_ids.size(1);
  int E = L.num_local_experts;
  int eps = L.ep_size;
  int nkeys = E * eps;
  int E_total = nkeys;

  CHECK_INPUT_AND_TYPE(x, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(topk_ids, dl_int32);
  CHECK_INPUT_AND_TYPE(topk_w, dl_float32);
  CHECK_INPUT_AND_TYPE(lc, dl_int32);
  CHECK_INPUT_AND_TYPE(pc, dl_int32);
  CHECK_INPUT_AND_TYPE(loff, dl_int32);
  CHECK_INPUT_AND_TYPE(ploff, dl_int32);
  CHECK_INPUT_AND_TYPE(done, dl_int32);
  CHECK_INPUT_AND_TYPE(keybase, dl_int32);
  CHECK_INPUT_AND_TYPE(pkeybase, dl_int32);
  CHECK_INPUT_AND_TYPE(round_ctr, dl_int32);
  CHECK_DIM(2, x);
  CHECK_DIM(2, topk_ids);
  CHECK_DIM(2, topk_w);
  TVM_FFI_ICHECK(H == L.hidden) << "dispatch_dedup: x hidden " << H << " != layout " << L.hidden;
  TVM_FFI_ICHECK(T <= L.t_cap) << "dispatch_dedup: T " << T << " exceeds t_cap " << L.t_cap;
  TVM_FFI_ICHECK(top_k == L.top_k) << "dispatch_dedup: top_k mismatch";
  TVM_FFI_ICHECK(topk_ids.size(0) == T && topk_w.size(0) == T && topk_w.size(1) == top_k)
      << "dispatch_dedup: routing shape mismatch";
  TVM_FFI_ICHECK(lc.numel() >= nkeys && done.numel() >= nkeys && keybase.numel() >= nkeys)
      << "dispatch_dedup: key scratch too small";
  TVM_FFI_ICHECK(pc.numel() >= eps && pkeybase.numel() >= eps)
      << "dispatch_dedup: per-destination scratch too small";
  TVM_FFI_ICHECK(loff.numel() >= T * top_k && ploff.numel() >= T * top_k)
      << "dispatch_dedup: route scratch too small";
  int expected_row = fp8_payload ? static_cast<int>(H) : static_cast<int>(H) * 2;
  TVM_FFI_ICHECK(L.bytes_per_row == expected_row)
      << "dispatch_dedup: payload dtype does not match the pipe's payload config";

  auto stream = get_stream(x.device());
  auto* ids = static_cast<const int32_t*>(topk_ids.data_ptr());
  auto* w = static_cast<const float*>(topk_w.data_ptr());
  auto* lcp = static_cast<int32_t*>(lc.data_ptr());
  auto* pcp = static_cast<int32_t*>(pc.data_ptr());
  auto* loffp = static_cast<int32_t*>(loff.data_ptr());
  auto* ploffp = static_cast<int32_t*>(ploff.data_ptr());
  auto* donep = static_cast<int32_t*>(done.data_ptr());
  auto* kbp = static_cast<int32_t*>(keybase.data_ptr());
  auto* pkbp = static_cast<int32_t*>(pkeybase.data_ptr());
  auto* rc = static_cast<const int32_t*>(round_ctr.data_ptr());
  int lt = static_cast<int>(T);

  dim3 blk(256);
  int smem = fp8_payload ? static_cast<int>(H / 128) * static_cast<int>(sizeof(float)) : 0;

#define DISPATCH_DEDUP_CASE(K)                                                                    \
  case K: {                                                                                       \
    if (lt > 0) {                                                                                 \
      count_dedup_kernel<K><<<(lt + 255) / 256, blk, 0, stream>>>(L, ids, E, E_total, lcp, loffp, \
                                                                  pcp, ploffp, rc, lt);           \
    }                                                                                             \
    reserve_publish_dedup_kernel<<<1, 256, 0, stream>>>(L, lcp, pcp, kbp, pkbp, rc);              \
    if (lt > 0) {                                                                                 \
      if (fp8_payload) {                                                                          \
        store_publish_dedup_fp8_kernel<K><<<lt, blk, smem, stream>>>(                             \
            L, static_cast<const __nv_bfloat16*>(x.data_ptr()), static_cast<int>(H), ids, w,      \
            loffp, ploffp, lcp, donep, kbp, pkbp, rc, lt);                                        \
      } else {                                                                                    \
        store_publish_dedup_kernel<K><<<lt, blk, 0, stream>>>(                                    \
            L, static_cast<const uint8_t*>(x.data_ptr()), static_cast<int>(H) * 2, ids, w, loffp, \
            ploffp, lcp, donep, kbp, pkbp, rc, lt);                                               \
      }                                                                                           \
    }                                                                                             \
    break;                                                                                        \
  }
  switch (top_k) {
    DISPATCH_DEDUP_CASE(1)
    DISPATCH_DEDUP_CASE(2)
    DISPATCH_DEDUP_CASE(4)
    DISPATCH_DEDUP_CASE(6)
    DISPATCH_DEDUP_CASE(8)
    default:
      TVM_FFI_ICHECK(false) << "dispatch_dedup: unsupported top_k " << top_k
                            << " (supported: 1, 2, 4, 6, 8)";
  }
#undef DISPATCH_DEDUP_CASE
}

void sm90_push_dispatch_dedup(TensorView x, TensorView topk_ids, TensorView topk_w, LAYOUT_PARAMS,
                              TensorView lc, TensorView pc, TensorView loff, TensorView ploff,
                              TensorView done, TensorView keybase, TensorView pkeybase,
                              TensorView round_ctr) {
  check_layout(LAYOUT_ARGS);
  auto L = build_layout(LAYOUT_ARGS);
  dispatch_dedup_common(x, topk_ids, topk_w, /*fp8_payload=*/false, L, lc, pc, loff, ploff, done,
                        keybase, pkeybase, round_ctr);
}

void sm90_push_dispatch_dedup_fp8(TensorView x, TensorView topk_ids, TensorView topk_w,
                                  LAYOUT_PARAMS, TensorView lc, TensorView pc, TensorView loff,
                                  TensorView ploff, TensorView done, TensorView keybase,
                                  TensorView pkeybase, TensorView round_ctr) {
  check_layout(LAYOUT_ARGS);
  auto L = build_layout(LAYOUT_ARGS);
  dispatch_dedup_common(x, topk_ids, topk_w, /*fp8_payload=*/true, L, lc, pc, loff, ploff, done,
                        keybase, pkeybase, round_ctr);
}

void sm90_push_wait_prefix(LAYOUT_PARAMS, TensorView round_ctr, TensorView rows_per_src,
                           TensorView offsets, TensorView seg_src_base, TensorView seg_out_base,
                           TensorView pad_base, TensorView m_dev, TensorView p_dev,
                           TensorView next_row, int64_t m_cap) {
  check_layout(LAYOUT_ARGS);
  auto L = build_layout(LAYOUT_ARGS);
  int E = L.num_local_experts;
  int eps = L.ep_size;
  int nkeys = E * eps;
  CHECK_INPUT_AND_TYPE(round_ctr, dl_int32);
  CHECK_INPUT_AND_TYPE(rows_per_src, dl_int32);
  CHECK_INPUT_AND_TYPE(offsets, dl_int64);
  CHECK_INPUT_AND_TYPE(seg_src_base, dl_int32);
  CHECK_INPUT_AND_TYPE(seg_out_base, dl_int32);
  CHECK_INPUT_AND_TYPE(pad_base, dl_int32);
  CHECK_INPUT_AND_TYPE(m_dev, dl_int32);
  CHECK_INPUT_AND_TYPE(p_dev, dl_int32);
  CHECK_INPUT_AND_TYPE(next_row, dl_int32);
  TVM_FFI_ICHECK(rows_per_src.numel() >= eps) << "wait_prefix: rows_per_src too small";
  TVM_FFI_ICHECK(offsets.numel() >= E + 1) << "wait_prefix: offsets too small";
  TVM_FFI_ICHECK(seg_src_base.numel() >= nkeys) << "wait_prefix: seg_src_base too small";
  TVM_FFI_ICHECK(seg_out_base.numel() >= nkeys + 1) << "wait_prefix: seg_out_base too small";
  TVM_FFI_ICHECK(pad_base.numel() >= E) << "wait_prefix: pad_base too small";
  TVM_FFI_ICHECK(m_cap >= 1) << "wait_prefix: bad m_cap";
  auto stream = get_stream(round_ctr.device());
  int smem = (nkeys + 2 * eps) * static_cast<int>(sizeof(int32_t));
  TVM_FFI_ICHECK(smem <= 48 * 1024) << "wait_prefix: E*ep = " << nkeys << " needs " << smem
                                    << "B smem, over the 48KB static limit";
  int max_rows_per_key = L.t_cap * L.top_k;  // one source's routes for one expert
  wait_prefix_kernel<<<1, 256, smem, stream>>>(
      L, static_cast<const int32_t*>(round_ctr.data_ptr()),
      static_cast<int32_t*>(rows_per_src.data_ptr()), static_cast<int64_t*>(offsets.data_ptr()),
      static_cast<int32_t*>(seg_src_base.data_ptr()),
      static_cast<int32_t*>(seg_out_base.data_ptr()), static_cast<int32_t*>(pad_base.data_ptr()),
      static_cast<int32_t*>(m_dev.data_ptr()), static_cast<int32_t*>(p_dev.data_ptr()),
      static_cast<int32_t*>(next_row.data_ptr()), static_cast<int>(m_cap), max_rows_per_key);
}

static void compact_common(TensorView a_fp8, TensorView sfa, TensorView meta_out,
                           TensorView row_expert, const PushLayout& L, TensorView offsets,
                           TensorView seg_src_base, TensorView seg_out_base, TensorView pad_base,
                           TensorView m_dev, TensorView p_dev, TensorView next_row,
                           bool fp8_payload) {
  int K = L.hidden;
  CHECK_INPUT_AND_TYPE(a_fp8, dl_uint8);
  CHECK_INPUT_AND_TYPE(sfa, dl_float32);
  CHECK_INPUT_AND_TYPE(meta_out, dl_int32);
  CHECK_INPUT_AND_TYPE(row_expert, dl_int32);
  int64_t m_capacity = a_fp8.size(0);
  TVM_FFI_ICHECK(a_fp8.ndim() == 2 && a_fp8.size(1) == K) << "compact: a_fp8 must be (Mcap, H)";
  TVM_FFI_ICHECK(meta_out.numel() >= m_capacity * 4) << "compact: meta_out too small";
  TVM_FFI_ICHECK(row_expert.numel() >= m_capacity) << "compact: row_expert too small";
  auto stream = get_stream(a_fp8.device());
  int blocks = compact_grid_blocks(a_fp8.device());
  if (fp8_payload) {
    compact_fp8_persistent_kernel<<<blocks, 256, 0, stream>>>(
        L, static_cast<const int32_t*>(seg_src_base.data_ptr()),
        static_cast<const int32_t*>(seg_out_base.data_ptr()),
        static_cast<const int64_t*>(offsets.data_ptr()),
        static_cast<const int32_t*>(pad_base.data_ptr()),
        static_cast<const int32_t*>(m_dev.data_ptr()),
        static_cast<const int32_t*>(p_dev.data_ptr()), static_cast<int32_t*>(next_row.data_ptr()),
        static_cast<uint8_t*>(a_fp8.data_ptr()), static_cast<float*>(sfa.data_ptr()),
        static_cast<int32_t*>(meta_out.data_ptr()), static_cast<int32_t*>(row_expert.data_ptr()),
        K);
  } else {
    int smem = (K / 128) * static_cast<int>(sizeof(float));
    compact_quant_persistent_kernel<<<blocks, 256, smem, stream>>>(
        L, static_cast<const int32_t*>(seg_src_base.data_ptr()),
        static_cast<const int32_t*>(seg_out_base.data_ptr()),
        static_cast<const int64_t*>(offsets.data_ptr()),
        static_cast<const int32_t*>(pad_base.data_ptr()),
        static_cast<const int32_t*>(m_dev.data_ptr()),
        static_cast<const int32_t*>(p_dev.data_ptr()), static_cast<int32_t*>(next_row.data_ptr()),
        static_cast<uint8_t*>(a_fp8.data_ptr()), static_cast<float*>(sfa.data_ptr()),
        static_cast<int32_t*>(meta_out.data_ptr()), static_cast<int32_t*>(row_expert.data_ptr()),
        K);
  }
}

void sm90_push_compact(TensorView a_fp8, TensorView sfa, TensorView meta_out, TensorView row_expert,
                       LAYOUT_PARAMS, TensorView offsets, TensorView seg_src_base,
                       TensorView seg_out_base, TensorView pad_base, TensorView m_dev,
                       TensorView p_dev, TensorView next_row) {
  check_layout(LAYOUT_ARGS);
  auto L = build_layout(LAYOUT_ARGS);
  bool fp8_payload = (L.bytes_per_row == L.hidden);
  TVM_FFI_ICHECK(fp8_payload || L.bytes_per_row == 2 * L.hidden)
      << "compact: bytes_per_row must be H (fp8 payload) or 2H (bf16 payload)";
  compact_common(a_fp8, sfa, meta_out, row_expert, L, offsets, seg_src_base, seg_out_base, pad_base,
                 m_dev, p_dev, next_row, fp8_payload);
}

void sm90_silu_mul_quant_grouped(TensorView a_fp8, TensorView sfa, TensorView h, TensorView offsets,
                                 TensorView pad_base, TensorView m_dev, TensorView p_dev,
                                 TensorView row_expert, int64_t m_cap) {
  CHECK_INPUT_AND_TYPE(a_fp8, dl_uint8);
  CHECK_INPUT_AND_TYPE(sfa, dl_float32);
  CHECK_INPUT_AND_TYPE(h, dl_bfloat16);
  int64_t I = a_fp8.size(1);
  TVM_FFI_ICHECK(h.size(1) == 2 * I) << "silu_mul_quant: h must be (rows, 2*I)";
  TVM_FFI_ICHECK(I % 128 == 0) << "silu_mul_quant: I must be a multiple of 128";
  // smem row staging: default 48KB static limit covers I up to ~16K
  TVM_FFI_ICHECK(I <= 16384) << "silu_mul_quant: I too large for smem staging";
  TVM_FFI_ICHECK(m_cap > 0) << "silu_mul_quant: m_cap must be positive";
  auto stream = get_stream(h.device());
  int nkb = static_cast<int>(I >> 7);
  int smem = ((nkb + 3) & ~3) * static_cast<int>(sizeof(float)) + static_cast<int>(I) * 2;
  silu_mul_quant_grouped_kernel<<<compact_grid_blocks(h.device()), 256, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(h.data_ptr()),
      static_cast<const int64_t*>(offsets.data_ptr()),
      static_cast<const int32_t*>(pad_base.data_ptr()),
      static_cast<const int32_t*>(m_dev.data_ptr()), static_cast<const int32_t*>(p_dev.data_ptr()),
      static_cast<const int32_t*>(row_expert.data_ptr()), static_cast<uint8_t*>(a_fp8.data_ptr()),
      static_cast<float*>(sfa.data_ptr()), static_cast<int>(I));
}

void sm90_silu_mul_gated(TensorView g, TensorView h, TensorView m_dev, int64_t m_cap) {
  CHECK_INPUT_AND_TYPE(g, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(h, dl_bfloat16);
  int64_t I = g.size(1);
  TVM_FFI_ICHECK(h.size(1) == 2 * I) << "silu_mul_gated: h must be (rows, 2*I)";
  TVM_FFI_ICHECK(m_cap > 0) << "silu_mul_gated: m_cap must be positive";
  auto stream = get_stream(g.device());
  silu_mul_gated_kernel<<<compact_grid_blocks(g.device()), 256, 0, stream>>>(
      static_cast<__nv_bfloat16*>(g.data_ptr()), static_cast<const __nv_bfloat16*>(h.data_ptr()),
      static_cast<const int32_t*>(m_dev.data_ptr()), static_cast<int>(I));
}

void sm90_quant_grouped(TensorView a_fp8, TensorView sfa, TensorView x, TensorView offsets,
                        TensorView pad_base, TensorView m_dev, TensorView p_dev,
                        TensorView row_expert, int64_t m_cap) {
  CHECK_INPUT_AND_TYPE(a_fp8, dl_uint8);
  CHECK_INPUT_AND_TYPE(sfa, dl_float32);
  CHECK_INPUT_AND_TYPE(x, dl_bfloat16);
  int64_t K = x.size(1);
  TVM_FFI_ICHECK(K % 128 == 0) << "quant_grouped: K must be a multiple of 128";
  TVM_FFI_ICHECK(m_cap > 0) << "quant_grouped: m_cap must be positive";
  auto stream = get_stream(x.device());
  int smem = static_cast<int>(K / 128) * static_cast<int>(sizeof(float));
  quant_grouped_kernel<<<compact_grid_blocks(x.device()), 256, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(x.data_ptr()),
      static_cast<const int64_t*>(offsets.data_ptr()),
      static_cast<const int32_t*>(pad_base.data_ptr()),
      static_cast<const int32_t*>(m_dev.data_ptr()), static_cast<const int32_t*>(p_dev.data_ptr()),
      static_cast<const int32_t*>(row_expert.data_ptr()), static_cast<uint8_t*>(a_fp8.data_ptr()),
      static_cast<float*>(sfa.data_ptr()), static_cast<int>(K));
}

static void combine_common(TensorView y, TensorView meta, const PushLayout& L, TensorView m_dev,
                           TensorView rows_per_src, TensorView cdone_local, TensorView round_ctr,
                           bool fp8_combine) {
  CHECK_INPUT_AND_TYPE(y, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(meta, dl_int32);
  CHECK_INPUT_AND_TYPE(m_dev, dl_int32);
  CHECK_INPUT_AND_TYPE(rows_per_src, dl_int32);
  CHECK_INPUT_AND_TYPE(cdone_local, dl_int32);
  CHECK_INPUT_AND_TYPE(round_ctr, dl_int32);
  int64_t Mcap = y.size(0);
  int64_t H = y.size(1);
  TVM_FFI_ICHECK(H == L.hidden) << "combine: y hidden mismatch";
  TVM_FFI_ICHECK(meta.numel() >= Mcap * 4) << "combine: packed meta too small";
  TVM_FFI_ICHECK(rows_per_src.numel() >= L.ep_size && cdone_local.numel() >= L.ep_size)
      << "combine: per-source scratch too small";
  auto stream = get_stream(y.device());
  auto* cdl = static_cast<int32_t*>(cdone_local.data_ptr());
  cudaMemsetAsync(cdl, 0, static_cast<size_t>(L.ep_size) * sizeof(int32_t), stream);
  if (Mcap == 0) return;  // zero-row destinations were published by wait_prefix
  int blocks = compact_grid_blocks(y.device());
  if (fp8_combine) {
    int smem = static_cast<int>(H / 128) * static_cast<int>(sizeof(float));
    combine_publish_fp8_kernel<<<blocks, 256, smem, stream>>>(
        L, static_cast<const __nv_bfloat16*>(y.data_ptr()),
        static_cast<const int32_t*>(meta.data_ptr()), static_cast<const int32_t*>(m_dev.data_ptr()),
        static_cast<const int32_t*>(rows_per_src.data_ptr()), cdl,
        static_cast<const int32_t*>(round_ctr.data_ptr()));
  } else {
    combine_publish_kernel<<<blocks, 256, 0, stream>>>(
        L, static_cast<const __nv_bfloat16*>(y.data_ptr()),
        static_cast<const int32_t*>(meta.data_ptr()), static_cast<const int32_t*>(m_dev.data_ptr()),
        static_cast<const int32_t*>(rows_per_src.data_ptr()), cdl,
        static_cast<const int32_t*>(round_ctr.data_ptr()));
  }
}

void sm90_push_combine(TensorView y, TensorView meta, LAYOUT_PARAMS, TensorView m_dev,
                       TensorView rows_per_src, TensorView cdone_local, TensorView round_ctr) {
  check_layout(LAYOUT_ARGS);
  auto L = build_layout(LAYOUT_ARGS);
  combine_common(y, meta, L, m_dev, rows_per_src, cdone_local, round_ctr, /*fp8_combine=*/false);
}

void sm90_push_combine_fp8(TensorView y, TensorView meta, LAYOUT_PARAMS, TensorView m_dev,
                           TensorView rows_per_src, TensorView cdone_local, TensorView round_ctr) {
  check_layout(LAYOUT_ARGS);
  auto L = build_layout(LAYOUT_ARGS);
  combine_common(y, meta, L, m_dev, rows_per_src, cdone_local, round_ctr, /*fp8_combine=*/true);
}

void sm90_push_combine_fp8_grouped(TensorView y, TensorView meta, LAYOUT_PARAMS, TensorView m_dev,
                                   TensorView grp_cnt, TensorView grp_rows, TensorView grp_list,
                                   TensorView n_groups, TensorView groups_per_src,
                                   TensorView cdone_local, TensorView round_ctr) {
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
  int64_t Mcap = y.size(0);
  int64_t H = y.size(1);
  int eps = L.ep_size;
  int64_t nslots = static_cast<int64_t>(eps) * L.t_cap;
  TVM_FFI_ICHECK(H == L.hidden) << "grouped_combine: y hidden mismatch";
  TVM_FFI_ICHECK(L.top_k <= kMaxTopK) << "grouped_combine: top_k > " << kMaxTopK;
  TVM_FFI_ICHECK(meta.numel() >= Mcap * 4) << "grouped_combine: packed meta too small";
  TVM_FFI_ICHECK(grp_cnt.numel() >= nslots) << "grouped_combine: grp_cnt too small";
  TVM_FFI_ICHECK(grp_rows.numel() >= nslots * L.top_k) << "grouped_combine: grp_rows too small";
  TVM_FFI_ICHECK(grp_list.numel() >= nslots) << "grouped_combine: grp_list too small";
  TVM_FFI_ICHECK(n_groups.numel() >= 1) << "grouped_combine: n_groups too small";
  TVM_FFI_ICHECK(groups_per_src.numel() >= eps && cdone_local.numel() >= eps)
      << "grouped_combine: per-source scratch too small";
  int col_tile = static_cast<int>(H) < 4096 ? static_cast<int>(H) : 4096;
  int smem = ((((col_tile >> 7) + 3) & ~3) + col_tile) * static_cast<int>(sizeof(float));
  auto stream = get_stream(y.device());
  auto* cdl = static_cast<int32_t*>(cdone_local.data_ptr());
  auto* cnt = static_cast<int32_t*>(grp_cnt.data_ptr());
  auto* gps = static_cast<int32_t*>(groups_per_src.data_ptr());
  auto* glist = static_cast<int32_t*>(grp_list.data_ptr());
  auto* ng = static_cast<int32_t*>(n_groups.data_ptr());
  cudaMemsetAsync(cdl, 0, static_cast<size_t>(eps) * sizeof(int32_t), stream);
  cudaMemsetAsync(cnt, 0, static_cast<size_t>(nslots) * sizeof(int32_t), stream);
  cudaMemsetAsync(gps, 0, static_cast<size_t>(eps) * sizeof(int32_t), stream);
  cudaMemsetAsync(ng, 0, sizeof(int32_t), stream);
  if (Mcap == 0) return;
  int blocks = compact_grid_blocks(y.device());
  combine_group_build_kernel<<<blocks, 256, 0, stream>>>(
      L, static_cast<const int32_t*>(meta.data_ptr()),
      static_cast<const int32_t*>(m_dev.data_ptr()), cnt,
      static_cast<int32_t*>(grp_rows.data_ptr()), glist, ng, gps,
      static_cast<const int32_t*>(round_ctr.data_ptr()));
  combine_publish_fp8_grouped_kernel<<<blocks, 256, smem, stream>>>(
      L, static_cast<const __nv_bfloat16*>(y.data_ptr()),
      static_cast<const int32_t*>(meta.data_ptr()), cnt,
      static_cast<const int32_t*>(grp_rows.data_ptr()), glist, ng, gps, cdl,
      static_cast<const int32_t*>(round_ctr.data_ptr()), col_tile);
}

void sm90_push_wait_combine(LAYOUT_PARAMS, TensorView round_ctr) {
  check_layout(LAYOUT_ARGS);
  CHECK_INPUT_AND_TYPE(round_ctr, dl_int32);
  auto L = build_layout(LAYOUT_ARGS);
  auto stream = get_stream(round_ctr.device());
  wait_combine_kernel<<<1, kMaxEpSize, 0, stream>>>(
      L, static_cast<const int32_t*>(round_ctr.data_ptr()));
}

template <typename LaunchF32, typename LaunchBf16>
void dispatch_reduce_out(TensorView out, const PushLayout& L, int64_t num_tokens, const char* what,
                         LaunchF32&& launch_f32, LaunchBf16&& launch_bf16) {
  CHECK_INPUT(out);
  TVM_FFI_ICHECK(out.size(1) == L.hidden) << what << ": out hidden mismatch";
  TVM_FFI_ICHECK(num_tokens <= out.size(0) && num_tokens <= L.t_cap) << what << ": bad num_tokens";
  if (num_tokens == 0) return;
  if (out.dtype() == dl_float32) {
    launch_f32();
  } else if (out.dtype() == dl_bfloat16) {
    launch_bf16();
  } else {
    TVM_FFI_ICHECK(false) << what << ": out must be float32 or bfloat16";
  }
}

void sm90_combine_reduce(TensorView out, LAYOUT_PARAMS, int64_t num_tokens) {
  check_layout(LAYOUT_ARGS);
  auto L = build_layout(LAYOUT_ARGS);
  auto stream = get_stream(out.device());
  int nt = static_cast<int>(num_tokens);
  dispatch_reduce_out(
      out, L, num_tokens, "reduce",
      [&] {
        combine_reduce_kernel<float><<<static_cast<unsigned>(nt), 256, 0, stream>>>(
            static_cast<float*>(out.data_ptr()), L, nt);
      },
      [&] {
        combine_reduce_kernel<__nv_bfloat16><<<static_cast<unsigned>(nt), 256, 0, stream>>>(
            static_cast<__nv_bfloat16*>(out.data_ptr()), L, nt);
      });
}

void sm90_combine_reduce_fp8(TensorView out, LAYOUT_PARAMS, int64_t num_tokens) {
  check_layout(LAYOUT_ARGS);
  auto L = build_layout(LAYOUT_ARGS);
  auto stream = get_stream(out.device());
  int nt = static_cast<int>(num_tokens);
  dispatch_reduce_out(
      out, L, num_tokens, "reduce_fp8",
      [&] {
        combine_reduce_fp8_kernel<float><<<static_cast<unsigned>(nt), 256, 0, stream>>>(
            static_cast<float*>(out.data_ptr()), L, nt);
      },
      [&] {
        combine_reduce_fp8_kernel<__nv_bfloat16><<<static_cast<unsigned>(nt), 256, 0, stream>>>(
            static_cast<__nv_bfloat16*>(out.data_ptr()), L, nt);
      });
}

void sm90_combine_reduce_fp8_grouped(TensorView out, LAYOUT_PARAMS, int64_t num_tokens) {
  check_layout(LAYOUT_ARGS);
  auto L = build_layout(LAYOUT_ARGS);
  auto stream = get_stream(out.device());
  int nt = static_cast<int>(num_tokens);
  dispatch_reduce_out(
      out, L, num_tokens, "reduce_fp8_grouped",
      [&] {
        combine_reduce_fp8_grouped_kernel<float><<<static_cast<unsigned>(nt), 256, 0, stream>>>(
            static_cast<float*>(out.data_ptr()), L, nt);
      },
      [&] {
        combine_reduce_fp8_grouped_kernel<__nv_bfloat16>
            <<<static_cast<unsigned>(nt), 256, 0, stream>>>(
                static_cast<__nv_bfloat16*>(out.data_ptr()), L, nt);
      });
}

void sm90_push_ack(LAYOUT_PARAMS, TensorView round_ctr, TensorView lc, TensorView done) {
  check_layout(LAYOUT_ARGS);
  CHECK_INPUT_AND_TYPE(round_ctr, dl_int32);
  CHECK_INPUT_AND_TYPE(lc, dl_int32);
  CHECK_INPUT_AND_TYPE(done, dl_int32);
  auto L = build_layout(LAYOUT_ARGS);
  int nkeys = L.num_local_experts * L.ep_size;
  TVM_FFI_ICHECK(lc.numel() >= nkeys && done.numel() >= nkeys) << "ack: key scratch too small";
  int nreset_lc =
      static_cast<int>(std::min<int64_t>(lc.numel(), static_cast<int64_t>(nkeys) + L.ep_size));
  auto stream = get_stream(round_ctr.device());
  ack_kernel<<<1, 256, 0, stream>>>(L, static_cast<const int32_t*>(round_ctr.data_ptr()),
                                    static_cast<int32_t*>(lc.data_ptr()),
                                    static_cast<int32_t*>(done.data_ptr()), nreset_lc, nkeys);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm90_push_p2p_native_atomics, sm90_push_p2p_native_atomics);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm90_push_bump_tag, sm90_push_bump_tag);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm90_push_publish_abort, sm90_push_publish_abort);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm90_push_wait_acks, sm90_push_wait_acks);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm90_push_dispatch, sm90_push_dispatch);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm90_push_dispatch_fp8, sm90_push_dispatch_fp8);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm90_push_dispatch_dedup, sm90_push_dispatch_dedup);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm90_push_dispatch_dedup_fp8, sm90_push_dispatch_dedup_fp8);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm90_push_wait_prefix, sm90_push_wait_prefix);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm90_push_compact, sm90_push_compact);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm90_silu_mul_quant_grouped, sm90_silu_mul_quant_grouped);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm90_silu_mul_gated, sm90_silu_mul_gated);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm90_quant_grouped, sm90_quant_grouped);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm90_push_combine, sm90_push_combine);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm90_push_combine_fp8, sm90_push_combine_fp8);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm90_push_combine_fp8_grouped, sm90_push_combine_fp8_grouped);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm90_push_wait_combine, sm90_push_wait_combine);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm90_combine_reduce, sm90_combine_reduce);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm90_combine_reduce_fp8, sm90_combine_reduce_fp8);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm90_combine_reduce_fp8_grouped, sm90_combine_reduce_fp8_grouped);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm90_push_ack, sm90_push_ack);
