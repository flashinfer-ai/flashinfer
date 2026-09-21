// SPDX-License-Identifier: MIT
// Copyright (c) 2025 DeepSeek
// Derived from DeepGEMM commit 891d57b4db1071624b5c8fa0d1e51cb317fa709f;
// see LICENSE.deepseek-deepgemm.

#pragma once

#include "dsa_indexer_scan.cuh"

namespace dsa_litetopk {

// One CTA per row: derive the bucket transform and initial gate from the prefix logits, export the
// prefix histogram as the refresher's base and emit the passing prefix positions as candidates.
template <int kRetainedHead, int BT>
__global__ void seed_prep_kernel(const float* __restrict__ slog, const int64_t slog_stride,
                                 const int head, const int NB, const int K,
                                 float* __restrict__ origin, float* __restrict__ inv_delta,
                                 int32_t* __restrict__ th_bucket,
                                 CandidateValue* __restrict__ cand_val,
                                 int32_t* __restrict__ cand_idx, int32_t* __restrict__ cand_cnt,
                                 const int cand_cap, const int physical_index_base,
                                 int32_t* __restrict__ bcount_out) {
  constexpr int NSUB = 4;  // sub-histograms to spread smem atomic conflicts
  static_assert(kRetainedHead == 8192 || kRetainedHead == 12288,
                "production seed supports only the qualified 8K/12K layouts");
  constexpr int kRetainVecs = kRetainedHead / (BT * 4);
  const int row = gridDim.x - 1 - blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const float* srow = slog + (size_t)row * slog_stride;
  extern __shared__ int s_hist[];  // NSUB * NB ints

  // Pass 1: min/max of the finite scores (-inf padding must not poison the range).
  __shared__ float s_mx[BT / 32];
  __shared__ float s_mn[BT / 32];
  float mx = -INFINITY, mn = INFINITY;
  const auto acc = [&](const float s) {
    if (isfinite(s)) {
      mx = fmaxf(mx, s);
      mn = fminf(mn, s);
    }
  };
  // Keep the prefix scores in registers for the histogram and emit passes; tail lanes carry -inf.
  static_assert(BT == 256 || BT == 384 || BT == 512,
                "retained HOT seed requires a qualified CTA size");
  static_assert(BT % (NSUB * 32) == 0, "each seed sub-histogram must own whole warps");
  float4 retained[kRetainVecs];
  if (head == kRetainedHead) {
#pragma unroll
    for (int it = 0; it < kRetainVecs; ++it) {
      const int j = tid * 4 + it * BT * 4;
      const float4 s4 = *reinterpret_cast<const float4*>(srow + j);
      retained[it] = s4;
      acc(s4.x);
      acc(s4.y);
      acc(s4.z);
      acc(s4.w);
    }
  } else {
#pragma unroll
    for (int it = 0; it < kRetainVecs; ++it) {
      const int j = tid * 4 + it * BT * 4;
      float4 s4 = make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
      if (j + 3 < head) {
        s4 = *reinterpret_cast<const float4*>(srow + j);
      } else {
        if (j < head) s4.x = srow[j];
        if (j + 1 < head) s4.y = srow[j + 1];
        if (j + 2 < head) s4.z = srow[j + 2];
      }
      retained[it] = s4;
      acc(s4.x);
      acc(s4.y);
      acc(s4.z);
      acc(s4.w);
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, off));
  }
  if (lane == 0) {
    s_mx[tid >> 5] = mx;
    s_mn[tid >> 5] = mn;
  }
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (int wgi = 1; wgi < BT / 32; ++wgi) {
      s_mx[0] = fmaxf(s_mx[0], s_mx[wgi]);
      s_mn[0] = fminf(s_mn[0], s_mn[wgi]);
    }
  }
  __syncthreads();
  const float o = -s_mx[0];   // min over x = -score
  const float hi = -s_mn[0];  // max over x
  const float span = fmaxf(hi - o, 1e-20f);
  const float inv = (NB - 1) / span;
  const float vth = -o * inv;

  // Pass 2: bucket histogram, split into NSUB copies to cut shared-atomic conflicts.
  for (int b = tid; b < NSUB * NB; b += BT) s_hist[b] = 0;
  __syncthreads();
  int* my_hist = s_hist + (tid / (BT / NSUB)) * NB;
  const auto bucket_of = [&](const float s) -> int {
    // The same FMA as the emitter and the scan: two roundings could move a boundary score one
    // bucket, so the histogram would count a record the emitter rejects (a silent underfill).
    const float bq = fmaf(-s, inv, vth);
    int b = static_cast<int>(bq);
    return b < 0 ? 0 : (b > NB - 1 ? NB - 1 : b);
  };
#pragma unroll
  for (int it = 0; it < kRetainVecs; ++it) {
    const float4 s4 = retained[it];
    if (isfinite(s4.x)) atomicAdd(&my_hist[bucket_of(s4.x)], 1);
    if (isfinite(s4.y)) atomicAdd(&my_hist[bucket_of(s4.y)], 1);
    if (isfinite(s4.z)) atomicAdd(&my_hist[bucket_of(s4.z)], 1);
    if (isfinite(s4.w)) atomicAdd(&my_hist[bucket_of(s4.w)], 1);
  }
  __syncthreads();
  // merge sub-histograms into s_hist[0..NB)
  for (int b = tid; b < NB; b += BT) {
    int c = s_hist[b];
#pragma unroll
    for (int g = 1; g < NSUB; ++g) c += s_hist[g * NB + b];
    s_hist[b] = c;
  }
  __syncthreads();
  // The refresher's base histogram; the scan starts after the prefix, so nothing counts twice.
  for (int b = tid; b < NB; b += BT) bcount_out[(size_t)row * NB + b] = s_hist[b];
  // Initial gate: the first bucket whose histogram prefix reaches K (one owner thread per bin).
  __shared__ int s_th;
  __shared__ int s_wsum[BT / 32];
  if (tid == 0) s_th = NB - 1;
  const int h = (tid < NB) ? s_hist[tid] : 0;
  int x = h;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, off);
    if ((tid & 31) >= off) x += y;
  }
  if ((tid & 31) == 31) s_wsum[tid >> 5] = x;
  __syncthreads();
  int base = 0;
#pragma unroll
  for (int w = 0; w < BT / 32; ++w)
    if (w < (tid >> 5)) base += s_wsum[w];
  const int incl = base + x;
  const int excl = incl - h;
  if (tid < NB && excl < K && K <= incl) s_th = tid;
  __syncthreads();
  if (tid == 0) {
    th_bucket[row] = s_th;
    origin[row] = o;
    inv_delta[row] = inv;
  }
  __syncthreads();
  // Emit the passing prefix records at deterministic CTA-prefix offsets (no global atomics); the
  // row's true, possibly over-cap, count is published once.
  int emitted_before = 0;
  const float gate_edge = static_cast<float>(s_th + 1);
  const uint64_t row_base = static_cast<uint64_t>(row) * cand_cap;
#pragma unroll
  for (int it = 0; it < kRetainVecs; ++it) {
    const int j0 = tid * 4 + it * BT * 4;
    const float4 s4 = retained[it];
    const float score[4] = {s4.x, s4.y, s4.z, s4.w};
    float bq[4];
    bool pass[4];
    int local_count = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      bq[k] = fmaf(-score[k], inv, vth);
      pass[k] =
          j0 + k < head && isfinite(score[k]) && __float_as_int(bq[k]) < __float_as_int(gate_edge);
      local_count += pass[k] ? 1 : 0;
    }

    int warp_inclusive = local_count;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const int other = __shfl_up_sync(0xffffffffu, warp_inclusive, off);
      if (lane >= off) warp_inclusive += other;
    }
    if (lane == 31) s_wsum[tid >> 5] = warp_inclusive;
    __syncthreads();

    int warp_before = 0;
#pragma unroll
    for (int w = 0; w < BT / 32; ++w) {
      if (w < (tid >> 5)) warp_before += s_wsum[w];
    }
    const int thread_base = emitted_before + warp_before + warp_inclusive - local_count;
    int local_rank = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      if (pass[k]) {
        const int out = thread_base + local_rank++;
        if (out < cand_cap) {
          const uint32_t physical_idx = static_cast<uint32_t>(physical_index_base + j0 + k);
          dsa_litetopk::store_candidate(cand_val + row_base + out, cand_idx + row_base + out, bq[k],
                                        physical_idx);
        }
      }
    }

    int block_total = 0;
#pragma unroll
    for (int w = 0; w < BT / 32; ++w) block_total += s_wsum[w];
    emitted_before += block_total;
    // A fast warp must not overwrite s_wsum while a slow one still reads it.
    __syncthreads();
  }
  if (tid == 0) cand_cnt[row] = emitted_before;
}

}  // namespace dsa_litetopk

// Exact top-2048 per row: 12-bit radix passes over the high and then the low half of the 24-bit
// score code, then a stable compaction that emits KV positions. Rows with a count outside
// [top_k, cap], a non-finite score or an out-of-range position get a nonzero status and -1s.
namespace dsa_litetopk {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kRadixBits = 12;
constexpr int kRadixBins = 1 << kRadixBits;
constexpr int kBinsPerThread = kRadixBins / kThreads;
constexpr int kTopK = 2048;
constexpr uint32_t kPhysicalMask = (1u << 20) - 1u;

enum StatusBits : uint32_t {
  kBadCount = 1u << 0,
  kNonFinite = 1u << 1,
  kBadPhysical = 1u << 2,
  kHistogramFailure = 1u << 4,
  kCompactFailure = 1u << 6,
};

__device__ __forceinline__ uint32_t candidate_score_code(uint16_t value, int32_t packed_index) {
  return ((static_cast<uint32_t>(packed_index) >> 20) << 16) | static_cast<uint32_t>(value);
}

__device__ __forceinline__ float decode_candidate_score(uint32_t code) {
  const uint32_t ordered = code << 8;
  const uint32_t bits = (ordered & 0x80000000u) ? (ordered ^ 0x80000000u) : ~ordered;
  return __uint_as_float(bits);
}

__device__ __forceinline__ int block_exclusive_sum(int value, int* warp_prefix) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  int inclusive = value;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int other = __shfl_up_sync(0xffffffffu, inclusive, offset);
    if (lane >= offset) inclusive += other;
  }
  if (lane == 31) warp_prefix[warp] = inclusive;
  __syncthreads();
  if (warp == 0) {
    const int original = lane < kWarps ? warp_prefix[lane] : 0;
    int warp_inclusive = original;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const int other = __shfl_up_sync(0xffffffffu, warp_inclusive, offset);
      if (lane >= offset) warp_inclusive += other;
    }
    if (lane < kWarps) warp_prefix[lane] = warp_inclusive - original;
  }
  __syncthreads();
  return warp_prefix[warp] + inclusive - value;
}

__device__ __forceinline__ void select_histogram_bin(const int* histogram, int target,
                                                     int* warp_prefix, int* selected_bin,
                                                     int* selected_count_lt) {
  const int begin = static_cast<int>(threadIdx.x) * kBinsPerThread;
  int segment_sum = 0;
#pragma unroll
  for (int i = 0; i < kBinsPerThread; ++i) {
    segment_sum += histogram[begin + i];
  }
  const int segment_lt = block_exclusive_sum(segment_sum, warp_prefix);
  if (target > segment_lt && target <= segment_lt + segment_sum) {
    int local_lt = 0;
#pragma unroll
    for (int i = 0; i < kBinsPerThread; ++i) {
      const int count = histogram[begin + i];
      if (target <= segment_lt + local_lt + count) {
        *selected_bin = begin + i;
        *selected_count_lt = segment_lt + local_lt;
        break;
      }
      local_lt += count;
    }
  }
  __syncthreads();
}

__global__ __launch_bounds__(kThreads) void exact_topk_kernel(
    const uint16_t* __restrict__ values, const int32_t* __restrict__ packed_indices,
    const int32_t* __restrict__ counts, int32_t* __restrict__ output, int32_t* __restrict__ status,
    int rows, int cap, int sequence_length, int topk_arg = kTopK) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows) return;

  __shared__ int histogram[kRadixBins];
  __shared__ int warp_scratch[kWarps];
  __shared__ int selected_bin;
  __shared__ int selected_count_lt;
  __shared__ int first_count_lt;
  __shared__ uint32_t threshold_code;
  __shared__ int warp_lt[kWarps];
  __shared__ int warp_eq[kWarps];
  __shared__ int tile_lt;
  __shared__ int tile_eq;
  __shared__ int base_lt;
  __shared__ int base_eq;
  __shared__ uint32_t block_status;

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const unsigned lane_before = lane == 0 ? 0u : ((1u << static_cast<uint32_t>(lane)) - 1u);
  const int64_t candidate_row = static_cast<int64_t>(row) * cap;
  const int64_t output_row = static_cast<int64_t>(row) * topk_arg;
  const int count = counts[row];

  if (tid == 0) {
    block_status = 0u;
    selected_bin = -1;
    selected_count_lt = -1;
  }
  __syncthreads();

  if (count < topk_arg || count > cap) {
    if (tid == 0) block_status |= kBadCount;
    for (int col = tid; col < topk_arg; col += kThreads) {
      output[output_row + col] = -1;
    }
    __syncthreads();
    if (tid == 0) status[row] = static_cast<int32_t>(block_status);
    return;
  }

  for (int bin = tid; bin < kRadixBins; bin += kThreads) {
    histogram[bin] = 0;
  }
  __syncthreads();

  // Pass 1: histogram of the high 12 bits, validating every record.
  for (int col = tid; col < count; col += kThreads) {
    const int64_t offset = candidate_row + col;
    const int32_t packed = packed_indices[offset];
    const uint32_t physical = static_cast<uint32_t>(packed) & kPhysicalMask;
    const uint32_t code = candidate_score_code(values[offset], packed);
    if (!isfinite(decode_candidate_score(code))) {
      atomicOr(&block_status, static_cast<uint32_t>(kNonFinite));
    }
    if (physical >= static_cast<uint32_t>(sequence_length)) {
      atomicOr(&block_status, static_cast<uint32_t>(kBadPhysical));
    }
    atomicAdd(histogram + (code >> kRadixBits), 1);
  }
  __syncthreads();

  if (block_status != 0u) {
    for (int col = tid; col < topk_arg; col += kThreads) {
      output[output_row + col] = -1;
    }
    __syncthreads();
    if (tid == 0) status[row] = static_cast<int32_t>(block_status);
    return;
  }

  select_histogram_bin(histogram, topk_arg, warp_scratch, &selected_bin, &selected_count_lt);
  if (tid == 0) {
    if (selected_bin < 0 || selected_count_lt < 0 || selected_count_lt >= topk_arg) {
      block_status |= kHistogramFailure;
    } else {
      first_count_lt = selected_count_lt;
    }
  }
  __syncthreads();
  if (block_status != 0u) {
    for (int col = tid; col < topk_arg; col += kThreads) {
      output[output_row + col] = -1;
    }
    __syncthreads();
    if (tid == 0) status[row] = static_cast<int32_t>(block_status);
    return;
  }
  const int high_bin = selected_bin;
  const int remaining_rank = topk_arg - first_count_lt;
  __syncthreads();  // all reads of selected_bin precede its reset below

  for (int bin = tid; bin < kRadixBins; bin += kThreads) {
    histogram[bin] = 0;
  }
  if (tid == 0) {
    selected_bin = -1;
    selected_count_lt = -1;
  }
  __syncthreads();

  // Pass 2: low 12 bits inside the winning high bucket.
  for (int col = tid; col < count; col += kThreads) {
    const int64_t offset = candidate_row + col;
    const int32_t packed = packed_indices[offset];
    const uint32_t code = candidate_score_code(values[offset], packed);
    if (static_cast<int>(code >> kRadixBits) == high_bin) {
      atomicAdd(histogram + (code & (kRadixBins - 1)), 1);
    }
  }
  __syncthreads();
  select_histogram_bin(histogram, remaining_rank, warp_scratch, &selected_bin, &selected_count_lt);
  if (tid == 0) {
    if (selected_bin < 0 || selected_count_lt < 0 || selected_count_lt >= remaining_rank) {
      block_status |= kHistogramFailure;
    } else {
      threshold_code =
          (static_cast<uint32_t>(high_bin) << kRadixBits) | static_cast<uint32_t>(selected_bin);
      first_count_lt += selected_count_lt;
      base_lt = 0;
      base_eq = first_count_lt;
    }
  }
  __syncthreads();
  if (block_status != 0u) {
    for (int col = tid; col < topk_arg; col += kThreads) {
      output[output_row + col] = -1;
    }
    __syncthreads();
    if (tid == 0) status[row] = static_cast<int32_t>(block_status);
    return;
  }

  // Pass 3: stable exact compact and fused winner mapping.
  for (int tile = 0; tile < count; tile += kThreads) {
    const int col = tile + tid;
    uint32_t code = 0xffffffffu;
    int32_t packed = 0;
    if (col < count) {
      packed = packed_indices[candidate_row + col];
      code = candidate_score_code(values[candidate_row + col], packed);
    }
    const bool is_lt = col < count && code < threshold_code;
    const bool is_eq = col < count && code == threshold_code;
    const unsigned lt_mask = __ballot_sync(0xffffffffu, is_lt);
    const unsigned eq_mask = __ballot_sync(0xffffffffu, is_eq);
    const int lane_lt = __popc(lt_mask & lane_before);
    const int lane_eq = __popc(eq_mask & lane_before);
    if (lane == 0) {
      warp_lt[warp] = __popc(lt_mask);
      warp_eq[warp] = __popc(eq_mask);
    }
    __syncthreads();
    if (tid == 0) {
      int prefix_lt = 0;
      int prefix_eq = 0;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) {
        const int count_lt = warp_lt[w];
        const int count_eq = warp_eq[w];
        warp_lt[w] = prefix_lt;
        warp_eq[w] = prefix_eq;
        prefix_lt += count_lt;
        prefix_eq += count_eq;
      }
      tile_lt = base_lt;
      tile_eq = base_eq;
      base_lt += prefix_lt;
      base_eq += prefix_eq;
    }
    __syncthreads();

    int output_col = -1;
    if (is_lt) {
      output_col = tile_lt + warp_lt[warp] + lane_lt;
    } else if (is_eq) {
      output_col = tile_eq + warp_eq[warp] + lane_eq;
      if (output_col >= topk_arg) output_col = -1;
    }
    if (output_col >= 0 && output_col < topk_arg) {
      const uint32_t physical = static_cast<uint32_t>(packed) & kPhysicalMask;
      output[output_row + output_col] = static_cast<int32_t>(physical);
    }
    __syncthreads();
  }

  if (tid == 0) {
    if (base_lt != first_count_lt || base_lt > topk_arg || base_eq < topk_arg) {
      block_status |= kCompactFailure;
    }
  }
  __syncthreads();
  if (block_status != 0u) {
    for (int col = tid; col < topk_arg; col += kThreads) {
      output[output_row + col] = -1;
    }
  }
  __syncthreads();
  if (tid == 0) {
    status[row] = static_cast<int32_t>(block_status);
  }
}

}  // namespace dsa_litetopk
