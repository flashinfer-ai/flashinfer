/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

// Exact top-2048 selection for full q4 indexing rows above 202760 scored tokens.
//
// The scoring stage publishes, for every 32-token causal group, the full 32-bit ordering key of
// the group's FP32 maximum, the lowest lane attaining it and an upper bound on the group's
// second-highest key.  Group IDs are implicit, so lower group indices resolve equal maxima.
// The selector keeps a bounded superset of groups that contains every global winner: a coarse
// filter on maxima of four consecutive group keys, then adaptive radix refinement.  If the bound
// does not hold, selection continues over the complete group-key row.  Groups whose bound shows
// that only the maximum can qualify are reconstructed from the metadata; the others are gathered
// from their stored FP32 scores.  Final selection compares complete 64-bit score/ID keys, ranks a
// target radix bucket of at most 384 keys directly, and falls back to an exact scan of the full
// row if candidates exceed their storage, so candidates are never truncated.  The ordering
// matches short rows (signed zero included, smaller IDs win ties); slot order is unspecified.
//
// A long row never enters the short-row selector (18-bit epoch-tagged keys); the scoring stage
// publishes complete scores and summaries through C4 before this scratch is read.
namespace flashinfer::glm5_tp8::indexer_q4::long_topk {
constexpr int kTopK = 2048;
constexpr int kGroup = 32;
constexpr int kPool = (1048576 + 4096) / kGroup / 2 + 8;
constexpr int kThreads = 384;
constexpr int kWarps = kThreads / 32;
constexpr int kLogicalMax = 1048576 + 4096;

__host__ __device__ constexpr int group_count(int length) { return (length + kGroup - 1) / kGroup; }
// A row holds the group maxima at uint32 [0, G) and one auxiliary word per
// causal group from the 16-byte aligned uint32 offset (G_causal + 4) & ~3,
// which lies past every maximum because G <= G_causal + 1. The stride keeps
// both regions in the row: 2 * stride >= 2 * G + 8.
__host__ __device__ constexpr int summary_stride(int length) {
  return (group_count(length) + 5) & ~1;
}
__device__ __forceinline__ uint32_t score_key(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}
__device__ __forceinline__ uint64_t composite(float value, uint32_t id) {
  // Preserve signed-zero ordering and choose smaller IDs on score ties.
  return (uint64_t(score_key(value)) << 32) | (0xffffffffu - id);
}
__device__ __forceinline__ uint64_t load_cg(const uint64_t* p) {
  uint64_t x;
  asm volatile("ld.global.cg.b64 %0, [%1];" : "=l"(x) : "l"(p) : "memory");
  return x;
}
__device__ __forceinline__ float load_cg(const float* p) {
  float x;
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(x) : "l"(p) : "memory");
  return x;
}

// Every lane participates, including the final partial 128-row tile. A group
// is one warp (lane 0 sits on a multiple of 32 rows) and has exactly one
// writer; release/acquire at C4 makes atomic publication tags unnecessary.
// The summary row holds one full 32-bit score key per group: for disjoint
// contiguous groups, (max key, smaller group index) orders groups exactly as
// their maximum composites do. The auxiliary word holds the lowest lane with
// the maximum key (bits 31..27) and the second-largest key of the group
// (largest key of every other lane) truncated to bits 31..5, so the true
// second key is below ((word & 0x07ffffff) + 1) << 5. Invalid rows have key 0,
// below every finite score key. Never preload these summaries before C4.
__device__ __forceinline__ void publish_group(float score, int position, int valid_rows,
                                              int physical_rows, uint64_t* summary) {
  const uint32_t key = position < valid_rows ? score_key(score) : 0u;
  uint32_t top = key;
#pragma unroll
  for (int offset = kGroup / 2; offset > 0; offset >>= 1)
    top = max(top, __shfl_xor_sync(0xffffffffu, top, offset));
  const uint32_t lane = threadIdx.x & (kGroup - 1);
  const uint32_t arg = uint32_t(__ffs(__ballot_sync(0xffffffffu, key == top)) - 1);
  uint32_t second = lane == arg ? 0u : key;
#pragma unroll
  for (int offset = kGroup / 2; offset > 0; offset >>= 1)
    second = max(second, __shfl_xor_sync(0xffffffffu, second, offset));
  if (lane == 0u && position < valid_rows && position < physical_rows) {
    uint32_t* const row = reinterpret_cast<uint32_t*>(summary);
    const int group = position / kGroup;
    __stcg(row + group, top);
    __stcg(row + ((group_count(valid_rows) + 4) & ~3) + group, (arg << 27) | (second >> 5));
  }
}

struct alignas(16) Scratch {
  uint64_t candidates[kPool];
  uint32_t superkeys[kLogicalMax / kGroup / 4];
  uint32_t hist[2048];
  uint32_t groups[kTopK + kTopK / 2];
  uint32_t group_keys[kTopK + kTopK / 2];
  uint32_t multi_groups[kTopK + kTopK / 2];
  uint32_t partial[kWarps];
  uint32_t count;
  uint32_t remaining;
  uint32_t bucket_count;
  uint32_t multi;
  uint32_t digit;
  uint32_t key_max;
  uint32_t key_min;
  uint32_t key_floor;
  uint32_t pool;
  uint32_t kept;
  uint32_t primed;
  uint32_t small;
  uint64_t bucket[kThreads];
  uint64_t prefix;
  uint64_t mask;
  uint64_t cutoff;
};
static_assert(sizeof(Scratch) + 0x3c00 <= 0x38000,
              "long selector fits the indexer's shared buffer");

struct ScoreRow {
  const float* data;
  __device__ __forceinline__ uint64_t operator()(int i) const {
    return composite(load_cg(data + i), uint32_t(i));
  }
};
struct SummaryRow {
  const uint64_t* data;
  __device__ __forceinline__ uint64_t operator()(int i) const { return load_cg(data + i); }
};
struct CandidateRow {
  const uint64_t* data;
  __device__ __forceinline__ uint64_t operator()(int i) const { return data[i]; }
};

// Exact descending radix rank. Histogram population is aggregated within a
// warp before a shared atomic, so constant score digits do not serialize 32
// identical updates. One CTA owns a query; no inter-CTA barrier is added.
// Digits are 11 bits wide and start right below the leading bits fixed by the
// caller in s.prefix/s.mask (a leading-ones mask, zero when unknown); the 2048
// bins are scanned in parallel by 256 threads. Once the target bucket holds
// exactly the remaining count, every key in it is selected, so the prefix with
// a zero suffix is already the exact cutoff for emit (value >= cutoff).
// Callers synchronize after setting s.prefix/s.mask/s.primed. When s.primed is
// set, the caller already counted every row item's first digit into s.hist.
template <class Row>
__device__ __forceinline__ uint64_t kth(Row row, int count, Scratch& s) {
  const int tid = int(threadIdx.x), lane = tid & 31, warp = tid >> 5;
  const unsigned below = (1u << lane) - 1u;
  if (tid == 0) s.small = 0u;
  uint64_t prefix = s.prefix, mask = s.mask;
  uint32_t remaining = kTopK;
  bool primed = s.primed != 0u;
  int shift = 64 - __popcll(mask);
  while (shift > 0) {
    const int width = shift < 11 ? shift : 11;
    shift -= width;
    if (!primed) {
      for (int i = tid; i < 2048; i += kThreads) s.hist[i] = 0u;
      __syncthreads();
      for (int base = warp * 32; base < count; base += kThreads) {
        const int i = base + lane;
        const uint64_t value = i < count ? row(i) : 0;
        const bool match = i < count && (value & mask) == prefix;
        const unsigned active = __ballot_sync(0xffffffffu, match);
        if (match) {
          const unsigned digit = unsigned(value >> shift) & ((1u << width) - 1u);
          const unsigned peers = __match_any_sync(active, digit);
          if (lane == __ffs(peers) - 1) atomicAdd(&s.hist[digit], unsigned(__popc(peers)));
        }
      }
      __syncthreads();
    }
    primed = false;
    // Thread t < 256 owns digits 2047 - 8t - j, j < 8 (two 16-byte words).
    uint32_t h[8];
    if (tid < 256) {
      const uint4 a = *reinterpret_cast<const uint4*>(s.hist + (2044 - 8 * tid));
      const uint4 b = *reinterpret_cast<const uint4*>(s.hist + (2040 - 8 * tid));
      h[0] = a.w;
      h[1] = a.z;
      h[2] = a.y;
      h[3] = a.x;
      h[4] = b.w;
      h[5] = b.z;
      h[6] = b.y;
      h[7] = b.x;
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j) h[j] = 0u;
    }
    uint32_t local = 0u;
#pragma unroll
    for (int j = 0; j < 8; ++j) local += h[j];
    uint32_t incl = local;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const uint32_t other = __shfl_up_sync(0xffffffffu, incl, offset);
      if (lane >= offset) incl += other;
    }
    if (lane == 31) s.partial[warp] = incl;
    __syncthreads();
    uint32_t before = incl - local;
    for (int w = 0; w < warp; ++w) before += s.partial[w];
    const bool hit = before < remaining && before + local >= remaining;
    if (hit) {
      uint32_t left = remaining - before;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (left > h[j]) {
          left -= h[j];
        } else {
          s.digit = uint32_t(2047 - 8 * tid - j);
          s.remaining = left;
          s.bucket_count = h[j];
          break;
        }
      }
    }
    if (__syncthreads_count(hit) != 1) __trap();
    prefix |= uint64_t(s.digit) << shift;
    mask |= uint64_t((1u << width) - 1u) << shift;
    remaining = s.remaining;
    const uint32_t bucket = s.bucket_count;
    if (bucket == remaining) return prefix;
    if (shift > 0 && bucket <= uint32_t(kThreads)) {
      // Few keys share the target bucket: gather them and rank them directly.
      // Composites are distinct (unique IDs), so exactly one bucket key has
      // remaining - 1 larger bucket keys; it is the exact K-th composite.
      for (int base = warp * 32; base < count; base += kThreads) {
        const int i = base + lane;
        const uint64_t value = i < count ? row(i) : 0;
        const bool member = i < count && (value & mask) == prefix;
        const unsigned ballot = __ballot_sync(0xffffffffu, member);
        if (ballot != 0u) {
          uint32_t slot = 0u;
          if (lane == 0) slot = atomicAdd(&s.small, unsigned(__popc(ballot)));
          slot = __shfl_sync(0xffffffffu, slot, 0);
          const uint32_t at = slot + unsigned(__popc(ballot & below));
          if (member && at < uint32_t(kThreads)) s.bucket[at] = value;
        }
      }
      __syncthreads();
      if (s.small != bucket) __trap();
      if (tid < int(bucket)) {
        const uint64_t mine = s.bucket[tid];
        uint32_t above = 0u;
        for (uint32_t u = 0; u < bucket; ++u) above += s.bucket[u] > mine ? 1u : 0u;
        if (above == remaining - 1u) s.cutoff = mine;
      }
      __syncthreads();
      return s.cutoff;
    }
  }
  return prefix;
}

template <class Row>
__device__ __forceinline__ void emit(Row row, int count, uint64_t cutoff, int32_t* out,
                                     Scratch& s) {
  const int tid = int(threadIdx.x), lane = tid & 31, warp = tid >> 5;
  const unsigned below = (1u << lane) - 1u;
  // Callers zero s.count before the barrier that precedes kth/emit.
  for (int base = warp * 128; base < count; base += kThreads * 4) {
    uint64_t value[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int i = base + j * 32 + lane;
      value[j] = i < count ? row(i) : 0;
    }
    unsigned ballot[4];
    uint32_t total = 0u;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int i = base + j * 32 + lane;
      ballot[j] = __ballot_sync(0xffffffffu, i < count && value[j] >= cutoff);
      total += unsigned(__popc(ballot[j]));
    }
    if (total != 0u) {
      uint32_t slot = 0;
      if (lane == 0) slot = atomicAdd(&s.count, total);
      slot = __shfl_sync(0xffffffffu, slot, 0);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if ((ballot[j] >> lane) & 1u) {
          const uint32_t at = slot + unsigned(__popc(ballot[j] & below));
          if (at < kTopK) out[at] = int32_t(0xffffffffu - uint32_t(value[j]));
        }
        slot += unsigned(__popc(ballot[j]));
      }
    }
  }
  __syncthreads();
  if (s.count != kTopK) __trap();
}

// Keep the large radix body outside the persistent kernel's short-row path.
// Grouped path: stage all 32-row group keys in shared memory once, then rank
// them with adaptive 11-bit digits only until the groups at or above the
// current bucket fit kCap. Those groups (at least kTopK of them) are a superset
// of the top-kTopK groups; their smallest key is the floor. Only when a shared
// full 32-bit key still overflows kCap are exactly kTopK groups kept, smaller
// group index first on equal keys. Every top-K item has key >= floor and lies
// in a kept group. A kept group whose second key is below the floor
// contributes exactly its maximum, taken from the summary; only the other kept
// groups are gathered. The bounded pool is then ranked. A pool that overflows
// the bounded storage falls back to the exact full-row rank.
static __device__ __noinline__ void select(const float* scores, int rows, const uint64_t* maxima,
                                           int32_t* out, void* scratch, bool grouped) {
  Scratch& s = *reinterpret_cast<Scratch*>(scratch);
  const int tid = int(threadIdx.x), lane = tid & 31, warp = tid >> 5;
  constexpr uint32_t kCap = kTopK + kTopK / 2;
  const unsigned below = (1u << lane) - 1u;
  if (rows <= kTopK) {
    for (int i = tid; i < kTopK; i += kThreads) out[i] = i < rows ? i : -1;
    __syncthreads();
    return;
  }
  if (!grouped || group_count(rows) < kTopK) {
    if (tid == 0) {
      s.prefix = 0;
      s.mask = 0;
      s.count = 0u;
      s.primed = 0u;
    }
    __syncthreads();
    const uint64_t cutoff = kth(ScoreRow{scores}, rows, s);
    emit(ScoreRow{scores}, rows, cutoff, out, s);
    return;
  }
  const int ngroups = group_count(rows);
  uint32_t* const keys = reinterpret_cast<uint32_t*>(s.candidates);
  const uint32_t* const summary = reinterpret_cast<const uint32_t*>(maxima);
  if (tid == 0) {
    s.key_floor = 0xffffffffu;
    s.count = 0u;
    s.pool = 0u;
    s.multi = 0u;
    s.kept = 0u;
  }
  // Long rows also form superkeys, the maxima of four consecutive group keys.
  // Superkey j belongs to groups 4j..4j+3 (the last one may be partial).
  const int nsuper = (ngroups + 3) >> 2;
  const bool coarse = ngroups >= 4 * 4096 && nsuper <= kLogicalMax / kGroup / 4;
  uint32_t kmax = 0u, kmin = 0xffffffffu;
  if (coarse) {
    // Keys are staged as 16-byte words; kmin tracks the smallest superkey.
    const int nfull = ngroups >> 2;
    uint32_t tail[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const int i = nfull * 4 + j;
      tail[j] = tid == kThreads - 1 && i < ngroups ? __ldcg(summary + i) : 0u;
    }
    for (int base = tid; base < nfull; base += kThreads * 12) {
      uint4 v[12];
#pragma unroll
      for (int j = 0; j < 12; ++j) {
        const int i = base + j * kThreads;
        uint4 w;
        w.x = 0u;
        w.y = 0u;
        w.z = 0u;
        w.w = 0u;
        if (i < nfull) w = __ldcg(reinterpret_cast<const uint4*>(summary) + i);
        v[j] = w;
      }
#pragma unroll
      for (int j = 0; j < 12; ++j) {
        const int i = base + j * kThreads;
        if (i < nfull) {
          reinterpret_cast<uint4*>(keys)[i] = v[j];
          const uint32_t sk = max(max(v[j].x, v[j].y), max(v[j].z, v[j].w));
          s.superkeys[i] = sk;
          kmax = max(kmax, sk);
          kmin = min(kmin, sk);
        }
      }
    }
    if (tid == kThreads - 1 && nfull < nsuper) {
      uint32_t sk = 0u;
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        const int i = nfull * 4 + j;
        if (i < ngroups) {
          keys[i] = tail[j];
          sk = max(sk, tail[j]);
        }
      }
      s.superkeys[nfull] = sk;
      kmax = max(kmax, sk);
      kmin = min(kmin, sk);
    }
  } else {
    for (int base = tid; base < ngroups; base += kThreads * 48) {
      uint32_t v[48];
#pragma unroll
      for (int j = 0; j < 48; ++j) {
        const int i = base + j * kThreads;
        v[j] = i < ngroups ? __ldcg(summary + i) : 0u;
      }
#pragma unroll
      for (int j = 0; j < 48; ++j) {
        const int i = base + j * kThreads;
        if (i < ngroups) {
          keys[i] = v[j];
          kmax = max(kmax, v[j]);
          kmin = min(kmin, v[j]);
        }
      }
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    kmax = max(kmax, __shfl_xor_sync(0xffffffffu, kmax, offset));
    kmin = min(kmin, __shfl_xor_sync(0xffffffffu, kmin, offset));
  }
  // Per-warp bounds go to s.bucket (unused until the pool rank), and the first
  // radix histogram is cleared, under the one barrier that ends staging.
  if (lane == 0) s.bucket[warp] = (uint64_t(kmax) << 32) | kmin;
  for (int i = tid; i < 2048; i += kThreads) s.hist[i] = 0u;
  __syncthreads();
  // key_min is the smallest superkey on the coarse path. At least kTopK keys
  // (every superkey's maximal member) lie in [key_min, key_max], so the leading
  // bits shared by both bounds still enclose every top-kTopK group key.
  uint32_t key_max = 0u, key_min = 0xffffffffu;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) {
    const uint64_t b = s.bucket[w];
    key_max = max(key_max, uint32_t(b >> 32));
    key_min = min(key_min, uint32_t(b));
  }

  uint32_t floor_key = 0xffffffffu, kept = 0u;
  bool settled = false;
  if (coarse) {
    // Rank superkeys until those at or above the current bucket number at most
    // kTopK + kTopK / 4. Their lower bucket edge T has at least kTopK superkeys,
    // hence at least kTopK distinct groups, at or above it, so the kTopK-th
    // group key is >= T: every top-kTopK group lies in a superkey >= T.
    uint32_t prefix = key_max, mask = 0xffffffffu, remaining = kTopK, bucket = uint32_t(nsuper);
    int shift = 0;
    if (key_max != key_min) {
      shift = 32 - __clz(int(key_max ^ key_min));
      mask = uint32_t(0xffffffffull << shift);
      prefix = key_max & mask;
    }
    while (shift > 0 && (kTopK - remaining) + bucket > kTopK + kTopK / 4) {
      const int width = shift < 11 ? shift : 11;
      shift -= width;
      // s.hist is clear: zeroed before the staging barrier or after the last scan.
      for (int base = tid; base < nsuper; base += kThreads * 8) {
        uint32_t v[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int i = base + j * kThreads;
          v[j] = i < nsuper ? s.superkeys[i] : 0u;
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int i = base + j * kThreads;
          if (i < nsuper && (v[j] & mask) == prefix)
            atomicAdd(&s.hist[(v[j] >> shift) & ((1u << width) - 1u)], 1u);
        }
      }
      __syncthreads();
      uint32_t h[8];
      if (tid < 256) {
        const uint4 a = *reinterpret_cast<const uint4*>(s.hist + (2044 - 8 * tid));
        const uint4 b = *reinterpret_cast<const uint4*>(s.hist + (2040 - 8 * tid));
        h[0] = a.w;
        h[1] = a.z;
        h[2] = a.y;
        h[3] = a.x;
        h[4] = b.w;
        h[5] = b.z;
        h[6] = b.y;
        h[7] = b.x;
      } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) h[j] = 0u;
      }
      uint32_t local = 0u;
#pragma unroll
      for (int j = 0; j < 8; ++j) local += h[j];
      uint32_t incl = local;
#pragma unroll
      for (int offset = 1; offset < 32; offset <<= 1) {
        const uint32_t other = __shfl_up_sync(0xffffffffu, incl, offset);
        if (lane >= offset) incl += other;
      }
      if (lane == 31) s.partial[warp] = incl;
      __syncthreads();
      // Every bin was read into h[] before the barrier above.
      for (int i = tid; i < 2048; i += kThreads) s.hist[i] = 0u;
      uint32_t before = incl - local;
      for (int w = 0; w < warp; ++w) before += s.partial[w];
      const bool hit = before < remaining && before + local >= remaining;
      if (hit) {
        uint32_t left = remaining - before;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          if (left > h[j]) {
            left -= h[j];
          } else {
            s.digit = uint32_t(2047 - 8 * tid - j);
            s.remaining = left;
            s.bucket_count = h[j];
            break;
          }
        }
      }
      if (__syncthreads_count(hit) != 1) __trap();
      prefix |= s.digit << shift;
      mask |= ((1u << width) - 1u) << shift;
      remaining = s.remaining;
      bucket = s.bucket_count;
    }
    // Superkeys >= prefix (the lower edge of the current bucket).
    const uint32_t nlist = (kTopK - remaining) + bucket;
    // Each thread classifies at most 7 listed superkeys (register budget); a
    // longer list (only when every key bit is fixed) takes the group radix.
    if (nlist <= uint32_t(7 * kThreads)) {
      const uint32_t threshold = prefix;
      // List them in multi_groups (free until the kept groups are classified).
      for (int base = warp * 256; base < nsuper; base += kThreads * 8) {
        uint32_t v[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int i = base + j * 32 + lane;
          v[j] = i < nsuper ? s.superkeys[i] : 0u;
        }
        unsigned ballot[8];
        uint32_t total = 0u;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int i = base + j * 32 + lane;
          ballot[j] = __ballot_sync(0xffffffffu, i < nsuper && v[j] >= threshold);
          total += unsigned(__popc(ballot[j]));
        }
        if (total != 0u) {
          uint32_t slot = 0u;
          if (lane == 0) slot = atomicAdd(&s.count, total);
          slot = __shfl_sync(0xffffffffu, slot, 0);
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            if ((ballot[j] >> lane) & 1u) {
              const uint32_t at = slot + unsigned(__popc(ballot[j] & below));
              if (at < kCap) s.multi_groups[at] = uint32_t(base + j * 32 + lane);
            }
            slot += unsigned(__popc(ballot[j]));
          }
        }
      }
      __syncthreads();
      if (s.count != nlist) __trap();
      // Every member group whose own key is >= T is kept: exactly the groups
      // with key >= T, a superset of the top-kTopK groups, and T is a valid
      // (conservative) floor. Each listed superkey's member keys and auxiliary
      // words are read together (thread t owns list entries t + j * kThreads).
      floor_key = threshold;
      settled = true;
      const int nl = int(nlist);
      const int nfull = ngroups >> 2;
      const uint32_t* const aux = summary + ((ngroups + 4) & ~3);
      uint32_t kv[7][4], av[7][4];
      int first[7];
#pragma unroll
      for (int j = 0; j < 7; ++j) {
        const int l = tid + j * kThreads;
        uint4 k4, a4;
        k4.x = 0u;
        k4.y = 0u;
        k4.z = 0u;
        k4.w = 0u;
        a4 = k4;
        first[j] = ngroups;
        if (l < nl) {
          const int sidx = int(s.multi_groups[l]);
          first[j] = sidx * 4;
          k4 = reinterpret_cast<const uint4*>(keys)[sidx];
          if (sidx < nfull) {
            a4 = __ldcg(reinterpret_cast<const uint4*>(aux) + sidx);
          } else {
            // Partial last superkey: only its published members are read.
            a4.x = __ldcg(aux + sidx * 4);
            if (sidx * 4 + 1 < ngroups) a4.y = __ldcg(aux + sidx * 4 + 1);
            if (sidx * 4 + 2 < ngroups) a4.z = __ldcg(aux + sidx * 4 + 2);
          }
        }
        kv[j][0] = k4.x;
        kv[j][1] = k4.y;
        kv[j][2] = k4.z;
        kv[j][3] = k4.w;
        av[j][0] = a4.x;
        av[j][1] = a4.y;
        av[j][2] = a4.z;
        av[j][3] = a4.w;
      }
      // The staged keys and the superkey list are dead from here on; the pool
      // reuses their storage. s.hist is clear (zeroed after the last scan or
      // before the staging barrier).
      __syncthreads();
      const uint32_t lead_diff = floor_key ^ key_max;
      const int pool_top = 64 - (lead_diff == 0u ? 32 : __clz(int(lead_diff)));
      const int pool_width = pool_top < 11 ? pool_top : 11;
      const int pool_shift = pool_top - pool_width;
      const uint32_t pool_digits = (1u << pool_width) - 1u;
      uint32_t takes = 0u, ones = 0u;
#pragma unroll
      for (int j = 0; j < 7; ++j) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const bool take = first[j] + c < ngroups && kv[j][c] >= threshold;
          const bool single = (uint64_t((av[j][c] & 0x07ffffffu) + 1u) << 5) <= uint64_t(floor_key);
          takes |= (take ? 1u : 0u) << (j * 4 + c);
          ones |= (take && single ? 1u : 0u) << (j * 4 + c);
        }
      }
      // Warp-aggregated slots: singles count in the low half, others in the high half.
      const uint32_t mine = uint32_t(__popc(ones)) | (uint32_t(__popc(takes & ~ones)) << 16);
      uint32_t incl = mine;
#pragma unroll
      for (int offset = 1; offset < 32; offset <<= 1) {
        const uint32_t other = __shfl_up_sync(0xffffffffu, incl, offset);
        if (lane >= offset) incl += other;
      }
      uint32_t pbase = 0u, mbase = 0u;
      if (lane == 31) {
        if ((incl & 0xffffu) != 0u) pbase = atomicAdd(&s.pool, incl & 0xffffu);
        if ((incl >> 16) != 0u) mbase = atomicAdd(&s.multi, incl >> 16);
      }
      pbase = __shfl_sync(0xffffffffu, pbase, 31);
      mbase = __shfl_sync(0xffffffffu, mbase, 31);
      const uint32_t excl = incl - mine;
      uint32_t pslot = pbase + (excl & 0xffffu), mslot = mbase + (excl >> 16);
#pragma unroll
      for (int j = 0; j < 7; ++j) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          if ((takes >> (j * 4 + c)) & 1u) {
            const uint32_t g = uint32_t(first[j] + c);
            if ((ones >> (j * 4 + c)) & 1u) {
              // A group whose second key is below the floor adds only its maximum:
              // the summary key and the ID of its lowest maximal lane.
              const uint64_t value = (uint64_t(kv[j][c]) << 32) |
                                     (0xffffffffu - (g * uint32_t(kGroup) + (av[j][c] >> 27)));
              atomicAdd(&s.hist[uint32_t(value >> pool_shift) & pool_digits], 1u);
              if (pslot < kPool) s.candidates[pslot] = value;
              ++pslot;
            } else {
              if (mslot < kCap) s.multi_groups[mslot] = g;
              ++mslot;
            }
          }
        }
      }
    }
  }

  if (!settled) {
    // Bits above the highest bit where keys differ are common to every key.
    uint32_t prefix = key_max, mask = 0xffffffffu, remaining = kTopK, bucket = uint32_t(ngroups);
    int shift = 0;
    if (key_max != key_min) {
      shift = 32 - __clz(int(key_max ^ key_min));
      mask = uint32_t(0xffffffffull << shift);
      prefix = key_max & mask;
    }
    // (kTopK - remaining) keys lie above the current bucket of `bucket` keys.
    while (shift > 0 && (kTopK - remaining) + bucket > kCap) {
      const int width = shift < 11 ? shift : 11;
      shift -= width;
      for (int i = tid; i < 2048; i += kThreads) s.hist[i] = 0u;
      __syncthreads();
      for (int base = tid; base < ngroups; base += kThreads * 8) {
        uint32_t v[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int i = base + j * kThreads;
          v[j] = i < ngroups ? keys[i] : 0u;
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int i = base + j * kThreads;
          if (i < ngroups && (v[j] & mask) == prefix)
            atomicAdd(&s.hist[(v[j] >> shift) & ((1u << width) - 1u)], 1u);
        }
      }
      __syncthreads();
      // Parallel descending scan: thread t < 256 owns digits 2047 - 8t - j, j < 8,
      // read as two conflict-free 16-byte words.
      uint32_t h[8];
      if (tid < 256) {
        const uint4 a = *reinterpret_cast<const uint4*>(s.hist + (2044 - 8 * tid));
        const uint4 b = *reinterpret_cast<const uint4*>(s.hist + (2040 - 8 * tid));
        h[0] = a.w;
        h[1] = a.z;
        h[2] = a.y;
        h[3] = a.x;
        h[4] = b.w;
        h[5] = b.z;
        h[6] = b.y;
        h[7] = b.x;
      } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) h[j] = 0u;
      }
      uint32_t local = 0u;
#pragma unroll
      for (int j = 0; j < 8; ++j) local += h[j];
      uint32_t incl = local;
#pragma unroll
      for (int offset = 1; offset < 32; offset <<= 1) {
        const uint32_t other = __shfl_up_sync(0xffffffffu, incl, offset);
        if (lane >= offset) incl += other;
      }
      if (lane == 31) s.partial[warp] = incl;
      __syncthreads();
      uint32_t before = incl - local;
      for (int w = 0; w < warp; ++w) before += s.partial[w];
      const bool hit = before < remaining && before + local >= remaining;
      if (hit) {
        uint32_t left = remaining - before;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          if (left > h[j]) {
            left -= h[j];
          } else {
            s.digit = uint32_t(2047 - 8 * tid - j);
            s.remaining = left;
            s.bucket_count = h[j];
            break;
          }
        }
      }
      if (__syncthreads_count(hit) != 1) __trap();
      prefix |= s.digit << shift;
      mask |= ((1u << width) - 1u) << shift;
      remaining = s.remaining;
      bucket = s.bucket_count;
    }
    // Unresolved only when all 32 key bits are fixed and the shared key's bucket
    // still exceeds kCap: then exactly kTopK groups are kept.
    const bool ties = (kTopK - remaining) + bucket > kCap;
    kept = ties ? uint32_t(kTopK) : (kTopK - remaining) + bucket;

    // Each warp compacts 256 contiguous keys per step with one slot reservation.
    for (int base = warp * 256; base < ngroups; base += kThreads * 8) {
      uint32_t v[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int i = base + j * 32 + lane;
        v[j] = i < ngroups ? keys[i] : 0u;
      }
      unsigned ballot[8];
      uint32_t total = 0u;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int i = base + j * 32 + lane;
        const bool take = i < ngroups && (ties ? v[j] > prefix : v[j] >= prefix);
        ballot[j] = __ballot_sync(0xffffffffu, take);
        total += unsigned(__popc(ballot[j]));
      }
      if (total != 0u) {
        uint32_t slot = 0u;
        if (lane == 0) slot = atomicAdd(&s.count, total);
        slot = __shfl_sync(0xffffffffu, slot, 0);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          if ((ballot[j] >> lane) & 1u) {
            const uint32_t at = slot + unsigned(__popc(ballot[j] & below));
            if (at < kCap) {
              s.groups[at] = uint32_t(base + j * 32 + lane);
              s.group_keys[at] = v[j];
            }
            floor_key = min(floor_key, v[j]);
          }
          slot += unsigned(__popc(ballot[j]));
        }
      }
    }
    if (ties) {
      // Equal keys: the lowest group indices win. Rank contiguous chunks.
      const int chunk = (ngroups + kThreads - 1) / kThreads;
      const int begin = min(tid * chunk, ngroups), end = min(begin + chunk, ngroups);
      uint32_t mine = 0u;
      for (int i = begin; i < end; ++i) mine += keys[i] == prefix ? 1u : 0u;
      uint32_t incl = mine;
#pragma unroll
      for (int offset = 1; offset < 32; offset <<= 1) {
        const uint32_t other = __shfl_up_sync(0xffffffffu, incl, offset);
        if (lane >= offset) incl += other;
      }
      if (lane == 31) s.partial[warp] = incl;
      __syncthreads();
      uint32_t rank = incl - mine;
      for (int w = 0; w < warp; ++w) rank += s.partial[w];
      for (int i = begin; i < end && rank < remaining; ++i) {
        if (keys[i] == prefix) {
          const uint32_t slot = atomicAdd(&s.count, 1u);
          if (slot < kCap) {
            s.groups[slot] = uint32_t(i);
            s.group_keys[slot] = prefix;
          }
          ++rank;
        }
      }
      floor_key = min(floor_key, prefix);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      floor_key = min(floor_key, __shfl_xor_sync(0xffffffffu, floor_key, offset));
    if (lane == 0) atomicMin(&s.key_floor, floor_key);
    for (int i = tid; i < 2048; i += kThreads) s.hist[i] = 0u;
    __syncthreads();
    if (s.count != kept) __trap();
    floor_key = s.key_floor;

    // At least kTopK kept maxima have key >= floor_key, so the K-th item does
    // too, and every top-K item lies in a kept group. The staged keys are dead
    // from here on; the pool reuses their storage. A group whose second key is
    // below the floor adds only its maximum: the summary key and the ID of its
    // lowest maximal lane form that item's exact composite.
    const uint32_t lead_diff = floor_key ^ key_max;
    const int pool_top = 64 - (lead_diff == 0u ? 32 : __clz(int(lead_diff)));
    const int pool_width = pool_top < 11 ? pool_top : 11;
    const int pool_shift = pool_top - pool_width;
    const uint32_t pool_digits = (1u << pool_width) - 1u;
    const int nkept = int(kept);
    const uint32_t* const aux = summary + ((ngroups + 4) & ~3);
    uint32_t word[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int i = (warp + j * kWarps) * 32 + lane;
      word[j] = i < nkept ? __ldcg(aux + s.groups[i]) : 0u;
    }
    unsigned ones[8], rest[8];
    uint32_t nones = 0u, nrest = 0u;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int i = (warp + j * kWarps) * 32 + lane;
      const bool single =
          i < nkept && (uint64_t((word[j] & 0x07ffffffu) + 1u) << 5) <= uint64_t(floor_key);
      ones[j] = __ballot_sync(0xffffffffu, single);
      rest[j] = __ballot_sync(0xffffffffu, i < nkept && !single);
      nones += unsigned(__popc(ones[j]));
      nrest += unsigned(__popc(rest[j]));
    }
    uint32_t pslot = 0u, mslot = 0u;
    if (lane == 0) {
      if (nones != 0u) pslot = atomicAdd(&s.pool, nones);
      if (nrest != 0u) mslot = atomicAdd(&s.multi, nrest);
    }
    pslot = __shfl_sync(0xffffffffu, pslot, 0);
    mslot = __shfl_sync(0xffffffffu, mslot, 0);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int i = (warp + j * kWarps) * 32 + lane;
      if ((ones[j] >> lane) & 1u) {
        const uint32_t slot = pslot + unsigned(__popc(ones[j] & below));
        const uint32_t g = s.groups[i];
        const uint64_t value = (uint64_t(s.group_keys[i]) << 32) |
                               (0xffffffffu - (g * uint32_t(kGroup) + (word[j] >> 27)));
        atomicAdd(&s.hist[uint32_t(value >> pool_shift) & pool_digits], 1u);
        if (slot < kPool) s.candidates[slot] = value;
      } else if ((rest[j] >> lane) & 1u) {
        const uint32_t slot = mslot + unsigned(__popc(rest[j] & below));
        if (slot < kCap) s.multi_groups[slot] = s.groups[i];
      }
      pslot += unsigned(__popc(ones[j]));
      mslot += unsigned(__popc(rest[j]));
    }
  }
  __syncthreads();

  // Every pool key lies in [floor_key, key_max]: leading bits common to both
  // bounds are common to every pool composite, and the pool rank's first digit
  // is the 11 bits right below them (counted into s.hist as items were added).
  const uint32_t diff = floor_key ^ key_max;
  const int common = diff == 0u ? 32 : __clz(int(diff));
  const uint64_t lead = common == 0 ? uint64_t(0) : ~uint64_t(0) << (64 - common);
  const uint64_t lead_prefix = (uint64_t(key_max) << 32) & lead;
  const int pool_width = 64 - common < 11 ? 64 - common : 11;
  const int pool_shift = 64 - common - pool_width;
  const uint32_t pool_digits = (1u << pool_width) - 1u;
  if (tid == 0) {
    // Read by the generic pool rank; s.count/s.small were last read before an
    // earlier barrier.
    s.mask = lead;
    s.prefix = lead_prefix;
    s.count = 0u;
    s.small = 0u;
    s.primed = 1u;
  }
  const int nmulti = int(s.multi);
  // More than kCap other groups overflow the list: take the exact fallback.
  const int ngather = nmulti > int(kCap) ? 0 : nmulti;
  // Rebuild every other kept group's items from the FP32 scores.
  for (int g0 = warp; g0 < ngather; g0 += kWarps * 16) {
    int ids[16];
    float x[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int g = g0 + j * kWarps;
      ids[j] = g < ngather ? int(s.multi_groups[g]) * kGroup + lane : rows;
      x[j] = ids[j] < rows ? __ldcg(scores + ids[j]) : 0.0f;
    }
    unsigned ballot[16];
    uint32_t total = 0u;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const uint64_t value = composite(x[j], uint32_t(ids[j]));
      const bool keep = ids[j] < rows && uint32_t(value >> 32) >= floor_key;
      ballot[j] = __ballot_sync(0xffffffffu, keep);
      total += unsigned(__popc(ballot[j]));
    }
    if (total != 0u) {
      uint32_t slot = 0u;
      if (lane == 0) slot = atomicAdd(&s.pool, total);
      slot = __shfl_sync(0xffffffffu, slot, 0);
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        if ((ballot[j] >> lane) & 1u) {
          const uint32_t at = slot + unsigned(__popc(ballot[j] & below));
          const uint64_t value = composite(x[j], uint32_t(ids[j]));
          atomicAdd(&s.hist[uint32_t(value >> pool_shift) & pool_digits], 1u);
          if (at < kPool) s.candidates[at] = value;
        }
        slot += unsigned(__popc(ballot[j]));
      }
    }
  }
  __syncthreads();
  const int pool = int(s.pool);
  if (pool > kPool || nmulti > int(kCap)) {
    if (tid == 0) {
      s.prefix = 0;
      s.mask = 0;
      s.count = 0u;
      s.primed = 0u;
    }
    __syncthreads();
    const uint64_t cutoff = kth(ScoreRow{scores}, rows, s);
    emit(ScoreRow{scores}, rows, cutoff, out, s);
    return;
  }
  if (pool < kTopK) __trap();
  if (pool == kTopK) {
    emit(CandidateRow{s.candidates}, pool, uint64_t(0), out, s);
    return;
  }
  // First digit of the pool rank, from the histogram filled as the pool was
  // built. Thread t < 256 owns digits 2047 - 8t - j, j < 8.
  {
    uint32_t h[8];
    if (tid < 256) {
      const uint4 a = *reinterpret_cast<const uint4*>(s.hist + (2044 - 8 * tid));
      const uint4 b = *reinterpret_cast<const uint4*>(s.hist + (2040 - 8 * tid));
      h[0] = a.w;
      h[1] = a.z;
      h[2] = a.y;
      h[3] = a.x;
      h[4] = b.w;
      h[5] = b.z;
      h[6] = b.y;
      h[7] = b.x;
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j) h[j] = 0u;
    }
    uint32_t local = 0u;
#pragma unroll
    for (int j = 0; j < 8; ++j) local += h[j];
    uint32_t incl = local;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const uint32_t other = __shfl_up_sync(0xffffffffu, incl, offset);
      if (lane >= offset) incl += other;
    }
    if (lane == 31) s.partial[warp] = incl;
    __syncthreads();
    uint32_t before = incl - local;
    for (int w = 0; w < warp; ++w) before += s.partial[w];
    const bool hit = before < uint32_t(kTopK) && before + local >= uint32_t(kTopK);
    if (hit) {
      uint32_t left = uint32_t(kTopK) - before;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (left > h[j]) {
          left -= h[j];
        } else {
          s.digit = uint32_t(2047 - 8 * tid - j);
          s.remaining = left;
          s.bucket_count = h[j];
          break;
        }
      }
    }
    if (__syncthreads_count(hit) != 1) __trap();
  }
  const uint64_t top_mask = lead | (uint64_t(pool_digits) << pool_shift);
  const uint64_t top_prefix = lead_prefix | (uint64_t(s.digit) << pool_shift);
  const uint32_t left = s.remaining, nbucket = s.bucket_count;
  if (nbucket == left) {
    // The whole target bucket is selected: its lower edge is the exact cutoff.
    emit(CandidateRow{s.candidates}, pool, top_prefix, out, s);
    return;
  }
  if (nbucket <= uint32_t(kThreads)) {
    // Few keys share the target bucket. One pass emits the kTopK - left items
    // above it and gathers the bucket; composites are distinct (unique IDs), so
    // the bucket key with r larger bucket keys, r < left, takes output slot
    // kTopK - left + r.
    for (int base = warp * 32; base < pool; base += kThreads) {
      const int i = base + lane;
      const uint64_t value = i < pool ? s.candidates[i] : 0;
      const uint64_t top = value & top_mask;
      const bool above = i < pool && top > top_prefix;
      const bool member = i < pool && top == top_prefix;
      const unsigned upper = __ballot_sync(0xffffffffu, above);
      const unsigned inner = __ballot_sync(0xffffffffu, member);
      uint32_t uslot = 0u, islot = 0u;
      if (lane == 0) {
        if (upper != 0u) uslot = atomicAdd(&s.count, unsigned(__popc(upper)));
        if (inner != 0u) islot = atomicAdd(&s.small, unsigned(__popc(inner)));
      }
      uslot = __shfl_sync(0xffffffffu, uslot, 0);
      islot = __shfl_sync(0xffffffffu, islot, 0);
      if (above) {
        const uint32_t at = uslot + unsigned(__popc(upper & below));
        if (at < uint32_t(kTopK) - left) out[at] = int32_t(0xffffffffu - uint32_t(value));
      }
      if (member) {
        const uint32_t at = islot + unsigned(__popc(inner & below));
        if (at < uint32_t(kThreads)) s.bucket[at] = value;
      }
    }
    __syncthreads();
    if (s.count != uint32_t(kTopK) - left || s.small != nbucket) __trap();
    if (tid < int(nbucket)) {
      const uint64_t mine = s.bucket[tid];
      uint32_t larger = 0u;
      for (uint32_t u = 0; u < nbucket; ++u) larger += s.bucket[u] > mine ? 1u : 0u;
      if (larger < left)
        out[(uint32_t(kTopK) - left) + larger] = int32_t(0xffffffffu - uint32_t(mine));
    }
    __syncthreads();
    return;
  }
  // A larger bucket continues the generic rank, which rescans the same primed
  // first histogram.
  const uint64_t cutoff = kth(CandidateRow{s.candidates}, pool, s);
  emit(CandidateRow{s.candidates}, pool, cutoff, out, s);
}
}  // namespace flashinfer::glm5_tp8::indexer_q4::long_topk
