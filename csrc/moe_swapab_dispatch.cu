/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Dual-width work-list dispatch for the swap-AB grouped GEMMs.
//
// ``moe_sort`` groups the permuted rows of every local expert into
// ``group_rows``-row groups (``tile_idx_to_mn_limit`` is the exclusive valid
// row bound of each group).  This single-CTA kernel splits those groups into
// two compact, order-preserving work lists:
//   * wide:   groups with more than ``wide_min_rows`` valid rows, one work
//             item per group (row group index = g);
//   * narrow: the remaining groups, one work item per ``narrow_tile``-row
//             sub-tile that still holds rows (row group index in
//             ``narrow_tile`` units = g * (group_rows / narrow_tile) + s).
// Each list feeds one swap-AB kernel instance (n_tile = group_rows and
// n_tile = narrow_tile) through its ``tile_idx_to_row_group`` indirection.
// Optionally a third list (``all``) holds every occupied ``narrow_tile``-row
// sub-tile of every group, for a narrow-tile GEMM2 that follows a mixed
// dense/narrow GEMM1.

#include <cuda_runtime.h>

#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

#include "tvm_ffi_utils.h"

namespace {

constexpr int kThreads = 1024;

struct Counts {
  int wide;
  int narrow;
  int all;
};

struct CountsAdd {
  __device__ __forceinline__ Counts operator()(const Counts& a, const Counts& b) const {
    return Counts{a.wide + b.wide, a.narrow + b.narrow, a.all + b.all};
  }
};

template <bool kPdl>
__global__ void __launch_bounds__(kThreads)
    swapab_dispatch_kernel(const int32_t* __restrict__ mn_limit,
                           const int32_t* __restrict__ num_groups_ptr, int32_t group_rows,
                           int32_t narrow_tile, int32_t wide_min_rows, int32_t wide_min_permille,
                           int32_t* __restrict__ wide_list, int32_t* __restrict__ wide_count,
                           int32_t* __restrict__ narrow_list, int32_t* __restrict__ narrow_count,
                           int32_t* __restrict__ all_list, int32_t* __restrict__ all_count) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (kPdl) {
    cudaGridDependencySynchronize();
  }
#endif
  using Scan = cub::BlockScan<Counts, kThreads>;
  __shared__ typename Scan::TempStorage temp;
  const int num_groups = *num_groups_ptr;
  const int sub = group_rows / narrow_tile;
  // Optional global rule: the wide (dense-tile) list is only worth a launch
  // when the candidate groups hold at least ``wide_min_permille`` of all
  // valid rows (a routing concentrated on few experts); otherwise every
  // group stays narrow. One counting pass over the groups decides it.
  if (wide_min_permille > 0) {
    using Sum = cub::BlockReduce<long long, kThreads>;
    __shared__ typename Sum::TempStorage sum_temp;
    __shared__ int use_wide;
    long long wide_rows = 0, all_rows = 0;
    for (int g = static_cast<int>(threadIdx.x); g < num_groups; g += kThreads) {
      int rows = min(group_rows, mn_limit[g] - g * group_rows);
      rows = max(rows, 0);
      all_rows += rows;
      if (rows > wide_min_rows) wide_rows += rows;
    }
    const long long packed = Sum(sum_temp).Sum((wide_rows << 32) | all_rows);
    if (threadIdx.x == 0) {
      const long long w = packed >> 32, a = packed & 0xffffffffLL;
      use_wide = (w * 1000 >= a * static_cast<long long>(wide_min_permille)) ? 1 : 0;
    }
    __syncthreads();
    if (!use_wide) wide_min_rows = group_rows;
  }
  Counts carry{0, 0, 0};
  for (int base = 0; base < num_groups; base += kThreads) {
    const int g = base + static_cast<int>(threadIdx.x);
    int rows = 0;
    if (g < num_groups) {
      rows = min(group_rows, mn_limit[g] - g * group_rows);
      rows = max(rows, 0);
    }
    Counts mine{0, 0, 0};
    if (rows > 0) {
      mine.all = (rows + narrow_tile - 1) / narrow_tile;
    }
    if (rows > wide_min_rows) {
      mine.wide = 1;
    } else {
      mine.narrow = mine.all;
    }
    Counts excl{0, 0, 0}, total{0, 0, 0};
    Scan(temp).ExclusiveScan(mine, excl, Counts{0, 0, 0}, CountsAdd(), total);
    if (mine.wide) {
      wide_list[carry.wide + excl.wide] = g;
    }
    for (int s = 0; s < mine.narrow; ++s) {
      narrow_list[carry.narrow + excl.narrow + s] = g * sub + s;
    }
    if (all_list != nullptr) {
      for (int s = 0; s < mine.all; ++s) {
        all_list[carry.all + excl.all + s] = g * sub + s;
      }
    }
    carry = CountsAdd()(carry, total);
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *wide_count = carry.wide;
    *narrow_count = carry.narrow;
    if (all_count != nullptr) {
      *all_count = carry.all;
    }
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (kPdl) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
}

}  // namespace

void moe_swapab_dispatch(int64_t mn_limit_ptr, int64_t num_groups_ptr, int64_t group_rows,
                         int64_t narrow_tile, int64_t wide_min_rows, int64_t wide_min_permille,
                         int64_t wide_list_ptr, int64_t wide_count_ptr, int64_t narrow_list_ptr,
                         int64_t narrow_count_ptr, int64_t all_list_ptr, int64_t all_count_ptr,
                         bool use_pdl, int64_t cuda_stream_ptr) {
  TVM_FFI_ICHECK((all_list_ptr == 0) == (all_count_ptr == 0))
      << "all_list and all_count must be given together";
  TVM_FFI_ICHECK(group_rows > 0 && narrow_tile > 0 && group_rows % narrow_tile == 0)
      << "group_rows must be a positive multiple of narrow_tile";
  cudaStream_t stream =
      cuda_stream_ptr != 0 ? reinterpret_cast<cudaStream_t>(cuda_stream_ptr) : get_current_stream();
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(1);
  config.blockDim = dim3(kThreads);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = use_pdl ? 1 : 0;
  config.attrs = attrs;
  config.numAttrs = 1;
  auto* kernel = use_pdl ? swapab_dispatch_kernel<true> : swapab_dispatch_kernel<false>;
  cudaError_t err = cudaLaunchKernelEx(
      &config, kernel, reinterpret_cast<const int32_t*>(mn_limit_ptr),
      reinterpret_cast<const int32_t*>(num_groups_ptr), static_cast<int32_t>(group_rows),
      static_cast<int32_t>(narrow_tile), static_cast<int32_t>(wide_min_rows),
      static_cast<int32_t>(wide_min_permille), reinterpret_cast<int32_t*>(wide_list_ptr),
      reinterpret_cast<int32_t*>(wide_count_ptr), reinterpret_cast<int32_t*>(narrow_list_ptr),
      reinterpret_cast<int32_t*>(narrow_count_ptr), reinterpret_cast<int32_t*>(all_list_ptr),
      reinterpret_cast<int32_t*>(all_count_ptr));
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "moe_swapab_dispatch launch failed: " << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(flashinfer_moe_swapab_dispatch, moe_swapab_dispatch);
