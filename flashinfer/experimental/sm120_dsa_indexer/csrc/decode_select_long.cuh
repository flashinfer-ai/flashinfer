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

// Decode selection of rows longer than kRowSelKeys, several CTAs a row, and
// the selection kernel.
//
// A row's CTAs take contiguous slices (at least kLongSlice keys each). The
// scan built the row's histogram of rowsel_bin, so every CTA finds the same
// cut bin without a pass of its own, emits its keys above the bin and appends
// the bin's keys to the row's list; after a barrier over the row's CTAs, each
// ranks its share of the list. A crowded cut bin (rare) takes further rounds
// of 4096-bin histograms over the order keys in its range, each 4096x
// narrower, and a single remaining key value is settled by id across the
// CTAs, whose slices are id ranges.
//
// Between calls the row's barrier resets itself, the histogram, list and
// output counters are double-buffered by the epoch's parity, and the rounds'
// histograms are cleared by the rounds that used them.
#pragma once

#include "decode_select_row.cuh"

namespace flashinfer::sm120_dsa_indexer {

constexpr int kLongSlice = 4096;

struct LongShared : RowSelShared {
  int row, part, ctas;  // this CTA's row and part of its ctas
  int more;             // the CTA takes another row
  uint32_t epoch, synced;
  int want;  // the rounds' input (longsel_rounds)
  uint32_t in_cut;
};

// A CTA's part of a row.
struct LongCtx {
  const float* src;
  int s0, s1, G, part, top_k;
  uint32_t* state;
  uint32_t* ctr;
  uint32_t* out_n;
  uint32_t* list_n;
  uint32_t* list_k;
  uint32_t* list_i;
  int32_t* out_i;
  float* out_s;
};

__device__ __forceinline__ void longsel_slice(int len, int part, int G, int& s0, int& s1) {
  s0 = static_cast<int>(static_cast<int64_t>(len) * part / G);
  s1 = static_cast<int>(static_cast<int64_t>(len) * (part + 1) / G);
}

// A sense-reversing barrier over the row's G CTAs. They are co-resident: the
// long rows take at most one wave of CTAs, and the short rows' CTAs never
// wait. Returns whether this CTA arrived last.
__device__ __forceinline__ bool longsel_sync(uint32_t* ctr, int G, LongShared& sh) {
  __syncthreads();
  if (threadIdx.x == 0) {
    // The generation cannot move before this CTA arrives.
    const uint32_t g = ld_relaxed_gpu(ctr + kCtrGen);
    __threadfence();
    const bool last = atom_acq_rel_add_gpu(reinterpret_cast<int32_t*>(ctr + kCtrCount), 1) == G - 1;
    if (last) {
      atomicExch(ctr + kCtrCount, 0u);
      red_release_add_gpu(reinterpret_cast<int32_t*>(ctr + kCtrGen), 1);
    } else {
      while (ld_acquire_gpu(ctr + kCtrGen) == g) __nanosleep(32);
    }
    sh.synced = last;
  }
  __syncthreads();
  return sh.synced;
}

// Emits the slice's keys of class cls(x) (0: output, 1: list, 2: neither),
// reading them again.
template <typename Cls>
__device__ __forceinline__ void longsel_reload_emit(const LongCtx& c, Cls cls) {
  const int tid = threadIdx.x, lane = tid & 31;
  uint32_t n0 = 0u, n1 = 0u;
  for (int i = c.s0 + tid; i < c.s1; i += kSelThreads) {
    const int k = cls(__ldcg(c.src + i));
    n0 += k == 0;
    n1 += k == 1;
  }
  uint32_t i0 = n0, i1 = n1;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const uint32_t y0 = __shfl_up_sync(0xffffffffu, i0, d);
    const uint32_t y1 = __shfl_up_sync(0xffffffffu, i1, d);
    if (lane >= d) i0 += y0, i1 += y1;
  }
  uint32_t b0 = 0u, b1 = 0u;
  if (lane == 31) {
    if (i0) b0 = atomicAdd(c.out_n, i0);
    if (i1) b1 = atomicAdd(c.list_n, i1);
  }
  uint32_t a0 = __shfl_sync(0xffffffffu, b0, 31) + i0 - n0;
  uint32_t a1 = __shfl_sync(0xffffffffu, b1, 31) + i1 - n1;
  for (int i = c.s0 + tid; i < c.s1; i += kSelThreads) {
    const float x = __ldcg(c.src + i);
    const int k = cls(x);
    if (k == 0) {
      c.out_i[a0] = i;
      if (c.out_s) c.out_s[a0] = x;
      ++a0;
    } else if (k == 1) {
      if (a1 < static_cast<uint32_t>(kSelList)) {
        c.list_k[a1] = score_order(x);
        c.list_i[a1] = static_cast<uint32_t>(i);
      }
      ++a1;
    }
  }
}

// After the list is complete: this CTA's share of it (entries part, part + G,
// ...) ranked against the whole list and written from out position `above`.
__device__ __forceinline__ void longsel_rank_list(LongShared& sh, const LongCtx& c, int n,
                                                  uint32_t above, int want) {
  const int tid = threadIdx.x;
  for (int e = tid; e < n; e += kSelThreads) {
    sh.list_k[e] = __ldcg(c.list_k + e);
    sh.list_i[e] = static_cast<int32_t>(__ldcg(c.list_i + e));
  }
  __syncthreads();
  const int lane = tid & 31, warp = tid >> 5;
  for (int e = c.part + warp * c.G; e < n; e += (kSelThreads / 32) * c.G) {
    const uint32_t k = sh.list_k[e];
    const int id = sh.list_i[e];
    int r = 0;
    for (int j = lane; j < n; j += 32) {
      const uint32_t kj = sh.list_k[j];
      r += kj > k || (kj == k && sh.list_i[j] > id);
    }
    r = __reduce_add_sync(0xffffffffu, r);
    if (lane == 0 && r < want) {
      c.out_i[above + r] = id;
      if (c.out_s) c.out_s[above + r] = order_value(k);
    }
  }
}

// The rounds over a crowded cut bin's order keys [klo, khi], then a single
// key value settled by id. Every CTA of the row takes the same steps.
__device__ __noinline__ void longsel_rounds(LongShared& sh, const LongCtx c, uint32_t klo,
                                            uint32_t khi) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  int want = sh.want;
  uint32_t in_cut = sh.in_cut;
  bool all = false, listed = false, ties = false;
  int rounds = 1;
  while (true) {
    const uint32_t span = khi - klo;
    if (span == 0u) {
      ties = true;
      break;
    }
    const int sh_bits = max(0, 32 - __clz(span) - 12);
    __syncthreads();
    for (int i = tid; i < 4096; i += kSelThreads) sh.hist[i] = 0u;
    __syncthreads();
    for (int i = c.s0 + tid; i < c.s1; i += kSelThreads) {
      const uint32_t k = score_order(__ldcg(c.src + i));
      if (k >= klo && k <= khi) atomicAdd(&sh.hist[(k - klo) >> sh_bits], 1u);
    }
    __syncthreads();
    uint32_t* gh = c.state + kLongHistR + (rounds - 1) * 4096;
    for (int i = tid; i < 4096; i += kSelThreads)
      if (sh.hist[i]) red_add_gpu(gh + i, sh.hist[i]);
    longsel_sync(c.ctr, c.G, sh);
    reinterpret_cast<uint4*>(sh.hist)[tid] = __ldcg(reinterpret_cast<const uint4*>(gh) + tid);
    __syncthreads();
    rowsel_cut(&sh, 4096, want);
    const uint32_t cb = sh.cut[0];
    want -= static_cast<int>(sh.cut[1]);
    in_cut = sh.cut[2];
    const uint32_t blo = klo + (cb << sh_bits);
    const uint32_t bhi = min(khi, blo + ((1u << sh_bits) - 1u));
    all = in_cut == static_cast<uint32_t>(want);
    listed = !all && in_cut <= static_cast<uint32_t>(kSelList);
    longsel_reload_emit(c, [&](float x) {
      const uint32_t k = score_order(x);
      if (k > bhi && k <= khi) return 0;
      if (k >= blo && k <= bhi) return all ? 0 : listed ? 1 : 2;
      return 2;
    });
    klo = blo;
    khi = bhi;
    ++rounds;
    if (all || listed) break;
    if (rounds >= kLongRounds) {
      ties = khi == klo;
      break;
    }
  }
  if (ties) {
    // The `want` largest ids holding key klo: a CTA's quota follows from the
    // counts of the CTAs after it.
    uint32_t n = 0u;
    for (int i = c.s0 + tid; i < c.s1; i += kSelThreads) n += score_order(__ldcg(c.src + i)) == klo;
    uint32_t tot;
    rowsel_excl_sum(n, sh.wsum, tot);
    if (tid == 0) c.state[kLongTie + c.part] = tot;
    longsel_sync(c.ctr, c.G, sh);
    uint32_t later = 0u;
    for (int q = c.part + 1 + tid; q < c.G; q += kSelThreads)
      later += __ldcg(c.state + kLongTie + q);
    uint32_t lt;
    rowsel_excl_sum(later, sh.wsum, lt);
    uint32_t r = lt;
    const uint32_t uw = static_cast<uint32_t>(want);
    const uint32_t base_out = static_cast<uint32_t>(c.top_k - want);
    for (int t1 = c.s1; t1 > c.s0 && r < uw; t1 -= kSelThreads) {
      const int i = t1 - kSelThreads + tid;
      const bool eq = i >= c.s0 && score_order(__ldcg(c.src + i)) == klo;
      const uint32_t m = __ballot_sync(0xffffffffu, eq);
      if (lane == 0) sh.wsum[warp] = __popc(m);
      __syncthreads();
      uint32_t after = 0u, all_t = 0u;
      for (int w = 0; w < kSelThreads / 32; ++w) {
        after += w > warp ? sh.wsum[w] : 0u;
        all_t += sh.wsum[w];
      }
      const uint32_t rank = r + after + __popc(m & ~((2u << lane) - 1u));
      if (eq && rank < uw) {
        c.out_i[base_out + rank] = i;
        if (c.out_s) c.out_s[base_out + rank] = order_value(klo);
      }
      __syncthreads();
      r += all_t;
    }
  }
  if (listed) {
    longsel_sync(c.ctr, c.G, sh);
    longsel_rank_list(sh, c, static_cast<int>(in_cut), static_cast<uint32_t>(c.top_k - want), want);
  }
  // Every CTA is done with the rounds' histograms and tie counts: clear them.
  longsel_sync(c.ctr, c.G, sh);
  for (int i = c.part * kSelThreads + tid; i < (rounds - 1) * 4096; i += c.G * kSelThreads)
    c.state[kLongHistR + i] = 0u;
  if (c.part == 0)
    for (int i = tid; i < c.G; i += kSelThreads) c.state[kLongTie + i] = 0u;
}

// CTA `slot_want` of the nslots long-row CTAs, in round `round`. Rounds take
// the long rows nslots at a time; a round of n rows gives each nslots / n
// CTAs. (With more long rows than CTAs, the last round's few rows thus spread
// over all CTAs.) A row's CTAs are all in one round, and every CTA takes
// every round, so the rows' barriers cannot deadlock.
__device__ __forceinline__ void longsel_body(const SelParams& p, LongShared& sh, int slot_want,
                                             int nslots, int round) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  // The plan reads only the lengths (inputs): before the dependency wait.
  if (warp == 0) {
    int nlong = 0, mine = -1;
    uint32_t longm[8];  // the first 256 rows' long-row masks (their loads in flight together)
#pragma unroll
    for (int b = 0; b < 8; ++b) {
      const int r = b * 32 + lane;
      longm[b] = r < p.num_rows ? __ldg(p.row_len + r) : 0;
    }
#pragma unroll
    for (int b = 0; b < 8; ++b) {
      longm[b] = __ballot_sync(0xffffffffu, static_cast<int>(longm[b]) > kRowSelKeys);
      nlong += __popc(longm[b]);
    }
    for (int r0 = 256; r0 < p.num_rows; r0 += 32) {
      const int r = r0 + lane;
      const bool lg = r < p.num_rows && __ldg(p.row_len + r) > kRowSelKeys;
      nlong += __popc(__ballot_sync(0xffffffffu, lg));
    }
    int G = 0, part = -1;
    bool more = false;
    if (nlong > 0) {
      const int base = round * nslots, left = min(nlong - base, nslots);
      const int gs = max(1, nslots / left);
      const int slot = slot_want / gs;
      part = slot_want - slot * gs;
      const int target = slot < left ? base + slot : nlong;
      more = base + nslots < nlong;
      int seen = 0;
      for (int r0 = 0; r0 < p.num_rows && mine < 0; r0 += 32) {
        const int r = r0 + lane;
        uint32_t m;
        if (r0 < 256) {
          m = longm[0];
#pragma unroll
          for (int b = 1; b < 8; ++b)
            if (r0 == b * 32) m = longm[b];
        } else {
          m = __ballot_sync(0xffffffffu, r < p.num_rows && __ldg(p.row_len + r) > kRowSelKeys);
        }
        if (target < seen + __popc(m) && target < nlong) {
          uint32_t mm = m;  // the (target - seen)-th set bit
          for (int i = 0; i < target - seen; ++i) mm &= mm - 1;
          mine = r0 + __ffs(mm) - 1;
        }
        seen += __popc(m);
      }
      if (mine >= 0) {
        const int mlen = __ldg(p.row_len + mine);
        G = min(gs, min((mlen + kLongSlice - 1) / kLongSlice, kLongMaxCtas));
      }
    }
    if (lane == 0) {
      sh.row = (mine >= 0 && part < G) ? mine : -1;
      sh.part = part;
      sh.ctas = G;
      sh.more = more;
    }
  }
  __syncthreads();
  const int row = sh.row;
  if (row < 0) return;
  const int G = sh.ctas, part = sh.part;
  const int top_k = p.top_k;
  uint32_t* state = p.long_ws + static_cast<int64_t>(row) * kLongWords;
  uint32_t* ctr = state + kLongCtr;
  const int len = __ldg(p.row_len + row);
  int s0, s1;
  longsel_slice(len, part, G, s0, s1);
  const float* src = p.logits + static_cast<int64_t>(row) * p.stride;
  int32_t* out_i = p.indices + static_cast<int64_t>(row) * top_k;
  float* out_s = p.scores ? p.scores + static_cast<int64_t>(row) * top_k : nullptr;
  constexpr int kPer = kLongSlice / kSelThreads;
  const bool single = s1 - s0 <= kLongSlice;
  asm volatile("griddepcontrol.wait;" ::: "memory");
  // Every CTA reads the row's epoch, then arrives; the last to arrive advances
  // the epoch after its emit. The arrival depends on the read only through a
  // register, so the epoch, both parities' histograms and the slice's first
  // keys are all in flight together.
  uint32_t arrived = 0u;
  if (tid == 0) {
    const uint32_t e = __ldcg(ctr + kCtrEpoch);
    uint32_t one = 1u;
    asm volatile("" : "+r"(one) : "r"(e));
    arrived = atomicAdd(ctr + kCtrArrive, one);
    sh.epoch = e;
  }
  auto u4 = [](float4 f) {
    return make_uint4(static_cast<uint32_t>(f.x), static_cast<uint32_t>(f.y),
                      static_cast<uint32_t>(f.z), static_cast<uint32_t>(f.w));
  };
  reinterpret_cast<uint4*>(sh.hist)[tid] =
      u4(__ldcg(reinterpret_cast<const float4*>(state + kLongHist0) + tid));
  const uint4 h1 = u4(__ldcg(reinterpret_cast<const float4*>(state + kLongHist0 + 4096) + tid));
  float xr[kPer];
  if (single) {
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      const int i = s0 + tid + j * kSelThreads;
      xr[j] = i < s1 ? __ldcg(src + i) : 0.0f;
    }
  }
  __syncthreads();
  const int par = static_cast<int>(sh.epoch & 1u);
  if (par == 1) reinterpret_cast<uint4*>(sh.hist)[tid] = h1;
  // The slice kLongSlice keys at a time from registers, the next chunk's
  // loads in flight: fn(x[kPer], c0, full), full when the chunk is all in the
  // slice.
  auto for_chunks = [&](auto fn) {
    if (single) {
      fn(xr, s0, std::false_type{});
      return;
    }
    float x[kPer], nx[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      const int i = s0 + tid + j * kSelThreads;
      nx[j] = i < s1 ? __ldcg(src + i) : 0.0f;
    }
    for (int c0 = s0; c0 < s1; c0 += kLongSlice) {
#pragma unroll
      for (int j = 0; j < kPer; ++j) x[j] = nx[j];
#pragma unroll
      for (int j = 0; j < kPer; ++j) {
        const int i = c0 + kLongSlice + tid + j * kSelThreads;
        if (i < s1) nx[j] = __ldcg(src + i);
      }
      if (c0 + kLongSlice <= s1)
        fn(x, c0, std::true_type{});
      else
        fn(x, c0, std::false_type{});
    }
  };
  // A warp's positions for keys of two classes counted in one word (1 and
  // 0x10000 a key): one add per class by the last lane. False when the warp
  // has none.
  auto place = [&](uint32_t c, uint32_t* n0, uint32_t* n1, uint32_t& a0, uint32_t& a1) {
    if (!__any_sync(0xffffffffu, c != 0u)) return false;
    uint32_t inc = c;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const uint32_t y = __shfl_up_sync(0xffffffffu, inc, d);
      if (lane >= d) inc += y;
    }
    uint32_t b0 = 0u, b1 = 0u;
    if (lane == 31) {
      if (inc & 0xffffu) b0 = atomicAdd(n0, inc & 0xffffu);
      if (inc >> 16) b1 = atomicAdd(n1, inc >> 16);
    }
    const uint32_t ex = inc - c;
    a0 = __shfl_sync(0xffffffffu, b0, 31) + (ex & 0xffffu);
    a1 = __shfl_sync(0xffffffffu, b1, 31) + (ex >> 16);
    return true;
  };
  // The other parity's buffers, cleared for the row's next call: their last
  // reader was the previous call, and the next scan runs after this grid.
  for (int i = part * kSelThreads + tid; i < 4096; i += G * kSelThreads)
    state[kLongHist0 + (par ^ 1) * 4096 + i] = 0u;
  if (part == 0 && tid < 2) ctr[(tid == 0 ? kCtrOut : kCtrList) + (par ^ 1)] = 0u;
  uint32_t* list = state + kLongList0 + par * 2 * kSelList;
  const LongCtx cx{src,
                   s0,
                   s1,
                   G,
                   part,
                   top_k,
                   state,
                   ctr,
                   ctr + kCtrOut + par,
                   ctr + kCtrList + par,
                   list,
                   list + kSelList,
                   out_i,
                   out_s};
  __syncthreads();
  rowsel_cut(&sh, 4096, top_k);
  const int want = top_k - static_cast<int>(sh.cut[1]);
  const uint32_t in_cut = sh.cut[2];
  const float lo = rowsel_bin_lo(sh.cut[0]), hi = rowsel_bin_lo(sh.cut[0] + 1u);
  const bool all = in_cut == static_cast<uint32_t>(want);
  const bool listed = !all && in_cut <= static_cast<uint32_t>(kSelList);

  // Emit: the keys above the cut bin out, the bin's keys to the list.
  {
    const float top = all ? lo : hi;
    for_chunks([&](const float(&x)[kPer], int c0, auto full) {
      constexpr bool kFull = decltype(full)::value;
      uint32_t c = 0u;
#pragma unroll
      for (int j = 0; j < kPer; ++j) {
        const bool in = kFull || c0 + tid + j * kSelThreads < s1;
        c += in && x[j] >= top ? 1u : in && listed && x[j] >= lo ? 0x10000u : 0u;
      }
      uint32_t a0, a1;
      if (!place(c, cx.out_n, cx.list_n, a0, a1) || c == 0u) return;
#pragma unroll
      for (int j = 0; j < kPer; ++j) {
        const int i = c0 + tid + j * kSelThreads;
        if (!kFull && i >= s1) continue;
        if (x[j] >= top) {
          out_i[a0] = i;
          if (out_s) out_s[a0] = x[j];
          ++a0;
        } else if (listed && x[j] >= lo) {
          cx.list_k[a1] = score_order(x[j]);
          cx.list_i[a1] = static_cast<uint32_t>(i);
          ++a1;
        }
      }
    });
  }
  if (tid == 0 && arrived == static_cast<uint32_t>(G - 1)) {
    atomicExch(ctr + kCtrArrive, 0u);
    atomicAdd(ctr + kCtrEpoch, 1u);
  }
  if (all) return;
  if (listed) {
    longsel_sync(ctr, G, sh);
    longsel_rank_list(sh, cx, static_cast<int>(in_cut), static_cast<uint32_t>(top_k - want), want);
    return;
  }
  if (tid == 0) {
    sh.want = want;
    sh.in_cut = in_cut;
  }
  __syncthreads();
  longsel_rounds(sh, cx, score_order(lo), score_order(hi) - 1u);
}

// The selection after the logits scan, in one launch. CTA b < num_rows selects
// row b if it has at most kRowSelKeys keys; the remaining CTAs are the long
// rows' (p.long_ws null: no row may be long). The long rows' CTAs wait for one
// another, so the kernel never triggers its dependents early: their CTAs
// could take the SMs the long rows' CTAs still need.
// kDrop: each CTA drops the logits lines it read from L2 (the scan's kKeep).
template <bool kDrop>
__global__ void __launch_bounds__(kSelThreads, 1)
    select_kernel(const __grid_constant__ SelParams p) {
  __shared__ __align__(16) LongShared sh;
  const int b = static_cast<int>(blockIdx.x);
  if (b < p.num_rows) {
    rowsel_body<kDrop>(p, sh, b);
    return;
  }
  if (p.long_ws == nullptr) return;
  const int slot = b - p.num_rows, nslots = static_cast<int>(gridDim.x) - p.num_rows;
  for (int round = 0;; ++round) {
    longsel_body(p, sh, slot, nslots, round);
    __syncthreads();
    const bool more = sh.more;
    if constexpr (kDrop) {
      if (sh.row >= 0) {
        int s0, s1;
        longsel_slice(__ldg(p.row_len + sh.row), sh.part, sh.ctas, s0, s1);
        const float* src = p.logits + static_cast<int64_t>(sh.row) * p.stride;
        discard_lines(src + s0, src + s1, threadIdx.x, kSelThreads);
      }
    }
    if (!more) break;
    __syncthreads();
  }
}

}  // namespace flashinfer::sm120_dsa_indexer
