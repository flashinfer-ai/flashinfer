// Copyright (c) 2026 FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "decode.cuh"

namespace flashinfer {
namespace sm90_nvfp4_rs {

constexpr int kBlockM = kTileN;
constexpr int kBlockK = kTileK;
constexpr int kThreads = 128;

struct GroupedTask {
  int32_t group;
  int32_t output_tile;
  int64_t row_begin;
  int64_t row_end;
  bool valid;
};

__host__ __device__ constexpr int64_t ceil_div_nonnegative(int64_t value, int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

template <int TokenTileN>
__device__ __forceinline__ GroupedTask map_grouped_task(uint64_t task_index, const int64_t* offsets,
                                                        int32_t num_groups, int32_t output_tiles,
                                                        int64_t row_capacity) {
  GroupedTask result{-1, 0, 0, 0, false};
  uint64_t task_base = 0;
  for (int32_t group = 0; group < num_groups; ++group) {
    const int64_t group_begin = offsets[group];
    const int64_t group_end = offsets[group + 1];
    if (group_begin < 0 || group_end < group_begin || group_end > row_capacity) {
      return result;
    }
    const uint64_t row_tiles =
        static_cast<uint64_t>(ceil_div_nonnegative(group_end - group_begin, TokenTileN));
    const uint64_t group_tasks = row_tiles * static_cast<uint64_t>(output_tiles);
    if (task_index < task_base + group_tasks) {
      const uint64_t local_task = task_index - task_base;
      const uint64_t output_tile = local_task % static_cast<uint64_t>(output_tiles);
      const uint64_t row_tile = local_task / static_cast<uint64_t>(output_tiles);
      result.group = group;
      result.output_tile = static_cast<int32_t>(output_tile);
      result.row_begin = group_begin + static_cast<int64_t>(row_tile) * TokenTileN;
      result.row_end = group_end;
      result.valid = true;
      return result;
    }
    task_base += group_tasks;
  }
  return result;
}

template <int TokenTileN>
__device__ __forceinline__ GroupedTask
map_grouped_task_prefix(uint64_t task_index, const int64_t* offsets, const int64_t* tile_prefix,
                        int32_t num_groups, int32_t output_tiles, int64_t row_capacity) {
  GroupedTask result{-1, 0, 0, 0, false};
  uint64_t row_task = task_index / static_cast<uint64_t>(output_tiles);
  if (row_task >= static_cast<uint64_t>(tile_prefix[num_groups])) return result;
  int32_t low = 0;
  int32_t high = num_groups;
  while (low + 1 < high) {
    int32_t middle = (low + high) / 2;
    if (static_cast<uint64_t>(tile_prefix[middle]) <= row_task) {
      low = middle;
    } else {
      high = middle;
    }
  }
  int64_t group_begin = offsets[low];
  int64_t group_end = offsets[low + 1];
  if (group_begin < 0 || group_end < group_begin || group_end > row_capacity) return result;
  uint64_t local_row_tile = row_task - static_cast<uint64_t>(tile_prefix[low]);
  result.group = low;
  result.output_tile = static_cast<int32_t>(task_index % static_cast<uint64_t>(output_tiles));
  result.row_begin = group_begin + static_cast<int64_t>(local_row_tile) * TokenTileN;
  result.row_end = group_end;
  result.valid = true;
  return result;
}

}  // namespace sm90_nvfp4_rs
}  // namespace flashinfer
