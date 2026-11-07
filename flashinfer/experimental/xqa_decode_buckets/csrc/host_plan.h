// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "task_geometry.h"

// The host and device planners write this same compact descriptor ABI.
struct XQABucketTask {
  int request, begin, end, stride;
};
struct XQABucketRange {
  int begin, count;
};
struct XQABucketSpan {
  int begin, end;
};

namespace flashinfer::xqa_buckets {
constexpr int MaxBatch = 32;
constexpr int Tile = 256;
/// Limit table width so converting page capacity to a rounded tile count fits int32.
constexpr int max_table_width(int page_size) { return (0x7fffffff - (Tile - 1)) / page_size; }

/// Size the shared task/request/span/header backing in int32 words, including unused slots.
inline int64_t descriptor_words(int batch, int ctas) {
  return int64_t(6) * (int64_t(ctas) + batch);
}

// Caller validates storage size/type. No CUDA calls or allocation occur here.
// Keep the ownership arithmetic identical to assign; only its execution moves
// to the CPU. The GPU planner retains its parallel task/span fill.
/**
 * Build descriptors from caller-owned CPU lengths for a subsequent ordered H2D copy.
 * Storage contains (ctas + batch - 1) four-word tasks, batch two-word request ranges,
 * ctas two-word owner spans, and a four-word header. Persistent ownership intersects
 * global tile buckets with requests; aligned ownership assigns request-local splits.
 * Unused storage is cleared so copying the whole backing is well-defined. This routine
 * validates lengths but neither allocates buffers nor performs or synchronizes CUDA work.
 */
inline void plan_host(const int* lengths, int* storage, int batch, int ctas, int table_width,
                      bool aligned, int page_size = 32, int q_len_per_req = 1) {
  if (batch < 1 || batch > MaxBatch || ctas < batch || ctas > INT32_MAX - MaxBatch ||
      (page_size != 16 && page_size != 32 && page_size != 64) || table_width < 1 ||
      table_width > max_table_width(page_size) || q_len_per_req < 1 || q_len_per_req > 5)
    throw std::invalid_argument("Invalid host bucket plan geometry");
  int prefix[MaxBatch + 1] = {}, base[MaxBatch + 1] = {}, first[MaxBatch];
  for (int r = 0; r < batch; ++r) {
    if (lengths[r] < 0 || int64_t(lengths[r]) > int64_t(table_width) * page_size)
      throw std::invalid_argument("Host sequence lengths must fit the page table capacity");
    if (lengths[r] != 0 && lengths[r] < q_len_per_req)
      throw std::invalid_argument("Active KV lengths must include the full query suffix");
    prefix[r + 1] = prefix[r] + direct_tile_count(lengths[r]);
  }
  auto ceil_div = [](int a, int b) { return a / b + (a % b != 0); };
  int budget = std::max(1, ceil_div(prefix[batch], ctas));
  if (aligned) {
    int lo = 1, hi = 1;
    for (int r = 0; r < batch; ++r) hi = std::max(hi, prefix[r + 1] - prefix[r]);
    while (lo < hi) {
      int mid = (lo + hi) / 2, count = 0;
      for (int r = 0; r < batch; ++r) count += ceil_div(prefix[r + 1] - prefix[r], mid);
      if (count <= ctas)
        hi = mid;
      else
        lo = mid + 1;
    }
    budget = lo;
  }
  // Clear unused slots too, so a single whole-backing H2D never copies garbage.
  std::fill(storage, storage + descriptor_words(batch, ctas), 0);
  int capacity = ctas + batch - 1;
  auto* tasks = reinterpret_cast<XQABucketTask*>(storage);
  auto* requests = reinterpret_cast<XQABucketRange*>(storage + 4 * int64_t(capacity));
  auto* spans = reinterpret_cast<XQABucketSpan*>(storage + 4 * int64_t(capacity) + 2 * batch);
  int* header = storage + descriptor_words(batch, ctas) - 4;
  int max_parts = 0;
  for (int r = 0; r < batch; ++r) {
    int n = prefix[r + 1] - prefix[r];
    first[r] = prefix[r] / budget;
    int count =
        aligned ? ceil_div(n, budget) : (n ? (prefix[r + 1] - 1) / budget - first[r] + 1 : 0);
    requests[r] = {base[r], count};
    base[r + 1] = base[r] + count;
    max_parts = std::max(max_parts, count);
    for (int ordinal = 0; ordinal < count; ++ordinal) {
      if (aligned) {
        tasks[base[r] + ordinal] = {r, aligned_boundary(ordinal, n, count),
                                    aligned_boundary(ordinal + 1, n, count), 1};
      } else {
        int bucket = first[r] + ordinal;
        tasks[base[r] + ordinal] = {r, std::max(prefix[r], bucket * budget) - prefix[r],
                                    std::min(prefix[r + 1], (bucket + 1) * budget) - prefix[r], 1};
      }
    }
  }
  int slots = base[batch];
  header[0] = budget;
  header[1] = slots;
  header[2] = aligned ? slots : ceil_div(prefix[batch], budget);
  header[3] = max_parts;
  for (int bucket = 0; bucket < ctas; ++bucket) {
    if (aligned) {
      spans[bucket] = bucket < slots ? XQABucketSpan{bucket, bucket + 1} : XQABucketSpan{0, 0};
      continue;
    }
    int64_t lo = int64_t(bucket) * budget;
    int64_t hi = std::min(int64_t(prefix[batch]), lo + budget);
    if (lo >= hi) continue;
    int first_req = 0;
    while (first_req + 1 < batch && prefix[first_req + 1] <= lo) ++first_req;
    int last_req = first_req;
    while (last_req + 1 < batch && prefix[last_req + 1] < hi) ++last_req;
    spans[bucket] = {base[first_req] + bucket - first[first_req],
                     base[last_req] + bucket - first[last_req] + 1};
  }
}
}  // namespace flashinfer::xqa_buckets
