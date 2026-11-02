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

// Decode selection of a row of at most kRowSelKeys keys: one CTA, the row in
// registers.
//
// After SGLang's DeepSeek-V4 top-k (TopKRegister in
// sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh, Apache-2.0;
// see NOTICE):
// one histogram of rowsel_bin gives the cut bin, one pass sends the keys above
// it to the output and the cut bin's keys to a shared list, and the list is
// ranked by ballots when short, else by radix passes over the full order key.
// Unlike SGLang's bounded tie buffer, an overflowing list falls back to radix
// passes over the registers, so the selection stays exact. Ties at the k-th
// score go to the larger key ids.
#pragma once

#include <cstdint>

#include "decode_common.cuh"

namespace flashinfer::sm120_dsa_indexer {

constexpr int kRowSelRank = 256;  // the longest list ranked by ballots

struct SelParams {
  const float* logits;  // [num_rows, stride]
  int64_t stride;
  const int32_t* row_len;
  int num_rows;
  int top_k;
  int32_t* indices;   // [num_rows, top_k]
  float* scores;      // [num_rows, top_k] or null
  uint32_t* long_ws;  // [num_rows][kLongWords]; null when no row may be long
};

struct RowSelShared {
  uint32_t hist[4096];
  uint32_t list_k[kSelList];
  int32_t list_i[kSelList];
  uint32_t wsum[32];
  uint32_t cut[3];
  uint32_t list_n, out_n;
};

// The smallest fp32 score whose rowsel_bin is at least `bin` (SGLang's
// coarse_bin_lower_bound for 12 bits), so a pass classifies keys by fp32
// compares instead of a conversion per key. fp16 rounds to nearest even: a
// boundary is the midpoint of the fp16 values at the bin's low edge and
// below, one ulp up where the edge's significand is odd.
__device__ __forceinline__ float rowsel_bin_lo(uint32_t bin) {
  const uint32_t key = bin << 4;
  auto val = [](uint32_t okey) -> float {
    if (okey < 0x03ffu) return -__int_as_float(0x7f800000);
    if (okey == 0x03ffu) return -65536.0f;
    if (okey == 0xfc00u) return 65536.0f;
    const uint16_t hb =
        (okey & 0x8000u) ? static_cast<uint16_t>(okey ^ 0x8000u) : static_cast<uint16_t>(~okey);
    return __half2float(__ushort_as_half(hb));
  };
  auto up = [](float v) {
    const int32_t b = __float_as_int(v);
    return __int_as_float(b >= 0 ? b + 1 : b - 1);
  };
  if (bin == 0) return -__int_as_float(0x7f800000);
  if (bin > (0xfc00u >> 4)) return __int_as_float(0x7f800000);
  const float mid = 0.5f * (val(key) + val(key - 1));
  if (key - 0x0401u <= 0xfbffu - 0x0401u) {
    const uint32_t ob = (key & 0x8000u) ? (key ^ 0x8000u) : (~key & 0xffffu);
    return (ob & 1u) ? up(mid) : mid;
  }
  return rowsel_bin(mid) < bin ? up(mid) : mid;
}

// Block-wide exclusive prefix sum (a warp scan, then the warps' totals by one
// redux).
__device__ __forceinline__ uint32_t rowsel_excl_sum(uint32_t v, uint32_t* wsum, uint32_t& total) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  uint32_t x = v;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const uint32_t y = __shfl_up_sync(0xffffffffu, x, d);
    if (lane >= d) x += y;
  }
  if (lane == 31) wsum[warp] = x;
  __syncthreads();
  const uint32_t w = wsum[lane];
  const uint32_t before = __reduce_add_sync(0xffffffffu, lane < warp ? w : 0u);
  total = __reduce_add_sync(0xffffffffu, w);
  __syncthreads();
  return before + x - v;
}

// The bin of hist[nb] holding the want-th key from the top:
// cut = (bin, keys above it, keys in it). Thread t owns bins [4t, 4t + 4).
__device__ __forceinline__ void rowsel_cut(RowSelShared* sh, int nb, int want) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  uint4 hv = make_uint4(0u, 0u, 0u, 0u);
  if (tid * 4 < nb) hv = reinterpret_cast<const uint4*>(sh->hist)[tid];
  const uint32_t own = hv.x + hv.y + hv.z + hv.w;
  uint32_t x = own;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const uint32_t y = __shfl_up_sync(0xffffffffu, x, d);
    if (lane >= d) x += y;
  }
  if (lane == 31) sh->wsum[warp] = x;
  __syncthreads();
  const uint32_t w = sh->wsum[lane];
  const uint32_t before = __reduce_add_sync(0xffffffffu, lane < warp ? w : 0u) + x - own;
  const uint32_t total = __reduce_add_sync(0xffffffffu, w);
  uint32_t above = total - before - own;
  const uint32_t uw = static_cast<uint32_t>(want);
  if (above < uw && above + own >= uw) {
    const uint32_t h[4] = {hv.x, hv.y, hv.z, hv.w};
#pragma unroll
    for (int b = 3; b >= 0; --b) {
      if (above < uw && above + h[b] >= uw) {
        sh->cut[0] = static_cast<uint32_t>(tid * 4 + b);
        sh->cut[1] = above;
        sh->cut[2] = h[b];
      }
      above += h[b];
    }
  }
  __syncthreads();
}

// Exact selection of the `rest` largest active keys by radix passes of 12, 12
// and 8 bits, written from out position `base`. Returns true with the key in
// `tie` when the passes leave ties; kIdKeys: the keys are distinct ids.
template <int N, bool kIdKeys>
__device__ __forceinline__ bool rowsel_radix(RowSelShared* sh, const uint32_t (&k)[N],
                                             const int32_t (&id)[N], const uint32_t (&val)[N],
                                             bool (&act)[N], uint32_t& base, int& rest,
                                             uint32_t& tie, int32_t* out_i, float* out_s) {
  const int tid = threadIdx.x;
  uint32_t prefix = 0u, pmask = 0u;
#pragma unroll 1
  for (int pass = 0; pass < 3; ++pass) {
    const int shift = pass == 0 ? 20 : pass == 1 ? 8 : 0;
    const int nb = pass == 2 ? 256 : 4096;
    __syncthreads();
    for (int i = tid; i < nb; i += kSelThreads) sh->hist[i] = 0u;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < N; ++i)
      if (act[i] && (k[i] & pmask) == prefix) atomicAdd(&sh->hist[(k[i] >> shift) & (nb - 1)], 1u);
    __syncthreads();
    rowsel_cut(sh, nb, rest);
    const uint32_t c = sh->cut[0];
    const int left = rest - static_cast<int>(sh->cut[1]);
    const bool done = sh->cut[2] == static_cast<uint32_t>(left);
    uint32_t m = 0u;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      const uint32_t b = (k[i] >> shift) & (nb - 1);
      m += act[i] && (k[i] & pmask) == prefix && (b > c || (done && b == c));
    }
    uint32_t t;
    uint32_t at = base + rowsel_excl_sum(m, sh->wsum, t);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      const uint32_t b = (k[i] >> shift) & (nb - 1);
      if (act[i] && (k[i] & pmask) == prefix && (b > c || (done && b == c))) {
        out_i[at] = id[i];
        if (out_s) out_s[at] = order_value(val[i]);
        ++at;
      }
    }
    base += t;
    rest -= static_cast<int>(t);
    if (done) return false;
    prefix |= c << shift;
    pmask |= static_cast<uint32_t>(nb - 1) << shift;
  }
  tie = prefix;
  return true;
}

// rowsel_radix over the order keys, then over the ids of the tied keys.
template <int N>
__device__ __forceinline__ void rowsel_finish(RowSelShared* sh, const uint32_t (&k)[N],
                                              const int32_t (&id)[N], bool (&act)[N], uint32_t base,
                                              int rest, int32_t* out_i, float* out_s) {
  uint32_t tie;
  if (!rowsel_radix<N, false>(sh, k, id, k, act, base, rest, tie, out_i, out_s)) return;
  uint32_t ids[N], vals[N];
#pragma unroll
  for (int i = 0; i < N; ++i) {
    act[i] = act[i] && k[i] == tie;
    ids[i] = static_cast<uint32_t>(id[i]);
    vals[i] = tie;
  }
  rowsel_radix<N, true>(sh, ids, id, vals, act, base, rest, tie, out_i, out_s);
}

// Drops the 128-byte lines wholly inside [p0, p1) from L2 without writing them
// back (the scan's logits, read for the last time).
__device__ __forceinline__ void discard_lines(const float* p0, const float* p1, int tid,
                                              int threads) {
  const uintptr_t a = (reinterpret_cast<uintptr_t>(p0) + 127u) & ~uintptr_t{127};
  const uintptr_t b = reinterpret_cast<uintptr_t>(p1) & ~uintptr_t{127};
  for (uintptr_t x = a + static_cast<uintptr_t>(tid) * 128u; x < b;
       x += static_cast<uintptr_t>(threads) * 128u)
    asm volatile("discard.global.L2 [%0], 128;" ::"l"(x) : "memory");
}

// The rare paths stay out of line: a selection runs on cold instruction
// caches, so the common path's code is kept compact.
// A list longer than kRowSelRank: radix passes over it.
__device__ __noinline__ void rowsel_list_radix(RowSelShared* sh, int c, uint32_t above, int want,
                                               int32_t* out_i, float* out_s) {
  const int tid = threadIdx.x;
  uint32_t k[2];
  int32_t ids[2];
  bool act[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int e = tid + i * kSelThreads;
    act[i] = e < c;
    k[i] = act[i] ? sh->list_k[e] : 0u;
    ids[i] = act[i] ? sh->list_i[e] : -1;
  }
  rowsel_finish<2>(sh, k, ids, act, above, want, out_i, out_s);
}

// A cut bin with more keys than the list holds: radix passes over its keys,
// read again.
template <int kVecs>
__device__ __noinline__ void rowsel_overflow(RowSelShared* sh, const float* src, int len, float lo,
                                             float hi, uint32_t above, int want, int32_t* out_i,
                                             float* out_s) {
  const int tid = threadIdx.x;
  uint32_t k[kVecs * 4];
  int32_t ids[kVecs * 4];
  bool act[kVecs * 4];
#pragma unroll
  for (int v = 0; v < kVecs; ++v) {
    const int base = (tid + v * kSelThreads) * 4;
    float4 f = make_float4(0.f, 0.f, 0.f, 0.f);
    if (base < len) f = __ldcg(reinterpret_cast<const float4*>(src + base));
    const float y[4] = {f.x, f.y, f.z, f.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      act[v * 4 + j] = base + j < len && y[j] >= lo && y[j] < hi;
      k[v * 4 + j] = score_order(y[j]);
      ids[v * 4 + j] = base + j;
    }
  }
  rowsel_finish<kVecs * 4>(sh, k, ids, act, above, want, out_i, out_s);
}

// A row of len (top_k, kVecs * 4096] keys, kVecs float4 a thread.
template <int kVecs>
__device__ __forceinline__ void rowsel_row(RowSelShared& sh, const float* src, int len, int top_k,
                                           int32_t* out_i, float* out_s) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  // Keys past the row read as -inf, below every cut (the row has more than
  // top_k keys), so no loop masks them.
  float x[kVecs][4];
#pragma unroll
  for (int v = 0; v < kVecs; ++v) {
    const int base = (tid + v * kSelThreads) * 4;
    float4 f = make_float4(0.f, 0.f, 0.f, 0.f);
    if (base < len) f = __ldcg(reinterpret_cast<const float4*>(src + base));
    const float ninf = -__int_as_float(0x7f800000);
    x[v][0] = base + 0 < len ? f.x : ninf;
    x[v][1] = base + 1 < len ? f.y : ninf;
    x[v][2] = base + 2 < len ? f.z : ninf;
    x[v][3] = base + 3 < len ? f.w : ninf;
  }
  __syncthreads();
#pragma unroll
  for (int v = 0; v < kVecs; ++v)
#pragma unroll
    for (int j = 0; j < 4; ++j) atomicAdd(&sh.hist[rowsel_bin(x[v][j])], 1u);
  __syncthreads();
  rowsel_cut(&sh, 4096, top_k);
  const uint32_t cut = sh.cut[0], above = sh.cut[1], in_cut = sh.cut[2];
  const int want = top_k - static_cast<int>(above);
  const float lo = rowsel_bin_lo(cut), hi = rowsel_bin_lo(cut + 1u);
  const bool all_cut = in_cut == static_cast<uint32_t>(want);
  const bool listed = !all_cut && in_cut <= static_cast<uint32_t>(kSelList);
  const float top = all_cut ? lo : hi;
  if constexpr (kVecs < 4) {
    // Few keys a lane: per-key shared adds (aggregated per warp by the
    // hardware) cost less than a warp scan.
#pragma unroll
    for (int v = 0; v < kVecs; ++v)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float xv = x[v][j];
        const int id = (tid + v * kSelThreads) * 4 + j;
        if (xv >= top) {
          const uint32_t at = atomicAdd(&sh.out_n, 1u);
          out_i[at] = id;
          if (out_s) out_s[at] = xv;
        } else if (listed && xv >= lo) {
          const uint32_t s = atomicAdd(&sh.list_n, 1u);
          sh.list_k[s] = score_order(xv);
          sh.list_i[s] = id;
        }
      }
  } else {
    // Both classes counted in one word (1: out, 0x10000: list), a warp scan,
    // one shared add per warp and class.
    uint32_t cls = 0u;
#pragma unroll
    for (int v = 0; v < kVecs; ++v)
#pragma unroll
      for (int j = 0; j < 4; ++j)
        cls += x[v][j] >= top ? 1u : listed && x[v][j] >= lo ? 0x10000u : 0u;
    uint32_t inc = cls;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const uint32_t y = __shfl_up_sync(0xffffffffu, inc, d);
      if (lane >= d) inc += y;
    }
    uint32_t b0 = 0u, b1 = 0u;
    if (lane == 31) {
      if (inc & 0xffffu) b0 = atomicAdd(&sh.out_n, inc & 0xffffu);
      if (inc >> 16) b1 = atomicAdd(&sh.list_n, inc >> 16);
    }
    const uint32_t ex = inc - cls;
    const uint32_t wb = __shfl_sync(0xffffffffu, b0, 31);
    const uint32_t wn = __shfl_sync(0xffffffffu, inc & 0xffffu, 31);
    uint32_t a0 = wb + (ex & 0xffffu);
    uint32_t a1 = __shfl_sync(0xffffffffu, b1, 31) + (ex >> 16);
    // The output is staged in the (now free) histogram and written by the
    // warp in order: written directly, each store would span the warp's whole
    // range, and the emit is store-bound.
    const bool stage = top_k <= 2048;
    int32_t* st_i = reinterpret_cast<int32_t*>(sh.hist);
    float* st_s = reinterpret_cast<float*>(sh.hist + 2048);
    if (cls != 0u) {
#pragma unroll
      for (int v = 0; v < kVecs; ++v)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float xv = x[v][j];
          const int id = (tid + v * kSelThreads) * 4 + j;
          if (xv >= top) {
            if (stage) {
              st_i[a0] = id;
              st_s[a0] = xv;
            } else {
              out_i[a0] = id;
              if (out_s) out_s[a0] = xv;
            }
            ++a0;
          } else if (listed && xv >= lo) {
            sh.list_k[a1] = score_order(xv);
            sh.list_i[a1] = id;
            ++a1;
          }
        }
    }
    if (stage) {
      __syncwarp();
      for (uint32_t i = wb + lane; i < wb + wn; i += 32) {
        out_i[i] = st_i[i];
        if (out_s) out_s[i] = st_s[i];
      }
    }
  }
  if (all_cut) return;
  const int c = static_cast<int>(in_cut);
  if (listed && c <= kRowSelRank) {
    __syncthreads();
    if (c <= 64) {
      // Warp t ranks list keys t and t + 32 by ballots over the list held by
      // its lanes (keys above, or equal with a larger id).
      const uint32_t k0 = lane < c ? sh.list_k[lane] : 0u;
      const uint32_t k1 = lane + 32 < c ? sh.list_k[lane + 32] : 0u;
      const int i0 = lane < c ? sh.list_i[lane] : -1;
      const int i1 = lane + 32 < c ? sh.list_i[lane + 32] : -1;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int t = warp + 32 * h;
        if (t < c) {
          const uint32_t tk = sh.list_k[t];
          const int ti = sh.list_i[t];
          const bool g0 = lane < c && (k0 > tk || (k0 == tk && i0 > ti));
          const bool g1 = lane + 32 < c && (k1 > tk || (k1 == tk && i1 > ti));
          const int r =
              __popc(__ballot_sync(0xffffffffu, g0)) + __popc(__ballot_sync(0xffffffffu, g1));
          if (lane == 0 && r < want) {
            out_i[above + r] = ti;
            if (out_s) out_s[above + r] = order_value(tk);
          }
        }
      }
      return;
    }
    const int nm = (c + 31) / 32;
    uint32_t lk[kRowSelRank / 32];
    int li[kRowSelRank / 32];
#pragma unroll
    for (int m = 0; m < kRowSelRank / 32; ++m) {
      const int e = lane + 32 * m;
      lk[m] = e < c ? sh.list_k[e] : 0u;
      li[m] = e < c ? sh.list_i[e] : -1;
    }
    for (int t = warp; t < c; t += 32) {
      const uint32_t tk = sh.list_k[t];
      const int ti = sh.list_i[t];
      int r = 0;
#pragma unroll
      for (int m = 0; m < kRowSelRank / 32; ++m) {
        if (m >= nm) break;
        const bool gt = lane + 32 * m < c && (lk[m] > tk || (lk[m] == tk && li[m] > ti));
        r += __popc(__ballot_sync(0xffffffffu, gt));
      }
      if (lane == 0 && r < want) {
        out_i[above + r] = ti;
        if (out_s) out_s[above + r] = order_value(tk);
      }
    }
    return;
  }
  if (listed) {
    __syncthreads();
    rowsel_list_radix(&sh, c, above, want, out_i, out_s);
    return;
  }
  __syncthreads();  // the staged writes share the histogram's space
  rowsel_overflow<kVecs>(&sh, src, len, lo, hi, above, want, out_i, out_s);
}

// Row `row` if it has at most kRowSelKeys keys; a longer row's CTA exits
// before the dependency wait, freeing its SM for the long rows' CTAs.
template <bool kDrop>
__device__ __forceinline__ void rowsel_body(const SelParams& p, RowSelShared& sh, int row) {
  const int tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < 4; ++i) sh.hist[tid + i * kSelThreads] = 0u;
  if (tid == 0) sh.list_n = sh.out_n = 0u;
  const int len = __ldg(p.row_len + row);
  if (len > kRowSelKeys) return;
  const int top_k = p.top_k;
  int32_t* out_i = p.indices + static_cast<int64_t>(row) * top_k;
  float* out_s = p.scores ? p.scores + static_cast<int64_t>(row) * top_k : nullptr;
  const float* src = p.logits + static_cast<int64_t>(row) * p.stride;
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (len <= top_k) {
    for (int i = tid; i < top_k; i += kSelThreads) {
      out_i[i] = i < len ? i : -1;
      if (out_s) out_s[i] = i < len ? src[i] : -__int_as_float(0x7f800000);
    }
  } else if (len <= 4 * kSelThreads) {
    rowsel_row<1>(sh, src, len, top_k, out_i, out_s);
  } else if (len <= 8 * kSelThreads) {
    rowsel_row<2>(sh, src, len, top_k, out_i, out_s);
  } else {
    rowsel_row<4>(sh, src, len, top_k, out_i, out_s);
  }
  if constexpr (kDrop) {
    __syncthreads();
    discard_lines(src, src + len, tid, kSelThreads);
  }
}

}  // namespace flashinfer::sm120_dsa_indexer
