/*
 * Copyright (c) 2025 by SageAttention team.
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

#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"

namespace nvfp4_attention {

using namespace cute;

struct WarpGroupConstants {
  static constexpr int kThreadsPerWarp = 32;
  static constexpr int kWarpsPerGroup = 4;
  static constexpr int kThreadsPerGroup = kThreadsPerWarp * kWarpsPerGroup;
};

struct WarpGroupIndex {
  __device__ __forceinline__ static int lane_id() {
    return threadIdx.x % WarpGroupConstants::kThreadsPerWarp;
  }

  __device__ __forceinline__ static int warp_id() {
    return threadIdx.x / WarpGroupConstants::kThreadsPerWarp;
  }

  __device__ __forceinline__ static int warp_id_in_group() {
    return warp_id() % WarpGroupConstants::kWarpsPerGroup;
  }

  __device__ __forceinline__ static int warp_group_id() {
    return warp_id() / WarpGroupConstants::kWarpsPerGroup;
  }

  __device__ __forceinline__ static int thread_id_in_group() {
    return warp_id_in_group() * WarpGroupConstants::kThreadsPerWarp + lane_id();
  }

  __device__ __forceinline__ static bool is_first_lane() { return lane_id() == 0; }

  __device__ __forceinline__ static bool is_first_thread_in_group() {
    return thread_id_in_group() == 0;
  }
};

struct WarpGroupSync {
  __device__ __forceinline__ static void sync() { __syncwarp(); }

  template <typename T>
  __device__ __forceinline__ static T shuffle(T var, int src_lane) {
    return __shfl_sync(0xffffffff, var, src_lane);
  }

  template <typename T>
  __device__ __forceinline__ static T shuffle_xor(T var, int lane_mask) {
    return __shfl_xor_sync(0xffffffff, var, lane_mask);
  }

  __device__ __forceinline__ static float reduce_max(float val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      val = fmaxf(val, shuffle_xor(val, mask));
    }
    return val;
  }

  __device__ __forceinline__ static float reduce_sum(float val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      val += shuffle_xor(val, mask);
    }
    return val;
  }
};

struct WarpGroupElect {
  __device__ __forceinline__ static bool elect_one_in_warp() { return cute::elect_one_sync(); }

  __device__ __forceinline__ static bool elect_one_in_group() {
    bool is_first = WarpGroupIndex::is_first_lane();

    return is_first && (WarpGroupIndex::warp_id_in_group() == 0);
  }
};

template <typename TiledMMA>
struct WarpGroupLayout {
  __device__ __forceinline__ static auto get_thread_slice(TiledMMA const& tiled_mma,
                                                          int thread_idx) {
    return tiled_mma.get_thread_slice(thread_idx);
  }

  template <typename ThreadMMA, typename CoordTensor>
  __device__ __forceinline__ static auto partition_accumulator(ThreadMMA const& thread_mma,
                                                               CoordTensor const& coord_tensor) {
    return thread_mma.partition_C(coord_tensor);
  }
};

template <typename TiledMMA>
struct WarpGroupMMA {
  using Traits = TiledMMA;

  __device__ __forceinline__ static TiledMMA get_tiled_mma() { return TiledMMA{}; }

  __device__ __forceinline__ static auto get_thread_slice(int thread_idx) {
    return get_tiled_mma().get_thread_slice(thread_idx);
  }

  template <typename TensorA, typename TensorB, typename TensorC>
  __device__ __forceinline__ static void gemm(TensorA const& A, TensorB const& B, TensorC& C) {
    cute::gemm(get_tiled_mma(), A, B, C);
  }
};

__device__ __forceinline__ int canonical_warp_idx() { return cutlass::canonical_warp_idx_sync(); }

}  // namespace nvfp4_attention
