/*
 * Copyright (c) 2026 by the PatchShift Conv3d contributors.
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

#include <cstddef>
#include <flashinfer/conv3d/patchshift/common.cuh>

namespace flashinfer::conv3d::patchshift {

constexpr int kK16 = 16;
constexpr int kK16GroupsPerPackedStage = 2;
constexpr int kTemporalExtent = 3;

__host__ __device__ constexpr int C32Groups(int c) { return round_up(c, 2 * kK16) / (2 * kK16); }

__host__ __device__ constexpr int Supergroups(int c) { return kTemporalExtent * C32Groups(c); }

__host__ __device__ constexpr size_t PackedWeightNumel(int c, int k, int tile_m) {
  int tiles = round_up(k, tile_m) / tile_m;
  int tasks = tiles * Supergroups(c) * 3;
  return size_t(tasks) * 3 * kK16GroupsPerPackedStage * 2 * tile_m * 8;
}

__host__ __device__ constexpr size_t PackedWeightOffset(int k_tile, int supergroup, int filter_row,
                                                        int kw, int kg, int m, int kk,
                                                        int supergroups, int tile_m) {
  int task = (k_tile * supergroups + supergroup) * 3 + filter_row;
  int kw_group = kw * kK16GroupsPerPackedStage + kg;
  int plane = kk >> 3;
  int k8 = kk & 7;
  return (
      ((((size_t(task) * (3 * kK16GroupsPerPackedStage) + size_t(kw_group)) * 2 + size_t(plane)) *
            tile_m +
        size_t(m)) *
           8 +
       size_t(k8)));
}

}  // namespace flashinfer::conv3d::patchshift
