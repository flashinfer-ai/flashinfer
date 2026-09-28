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

#include <algorithm>
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

// Mixed-width work lists for the wide (2-CTA, ``narrow_tile``-row) swap-AB
// kernel plus dense ``group_rows``-row tiles, both over one ``moe_sort``
// permutation with ``group_rows``-row groups (expert bases are
// ``group_rows``-aligned; ``mn_limit[g]`` is the group's own exclusive row
// bound, ``min((g + 1) * group_rows, expert end)``, so the expert's end is
// the bound of its last group).  Per expert with ``c`` valid rows the kernel
// picks ``nwide`` dense tiles followed by ``a`` ``narrow_tile``-row windows
// covering the fewest rows (``group_rows * nwide + narrow_tile * a >= c``,
// ties to fewer windows) and emits the dense
// tiles as sort-group indices (``wide_list``) and the windows as row
// offsets in ``row_unit`` rows (``narrow_list``), both in permutation
// order.  The windows start ``group_rows``-aligned plus a multiple of
// ``narrow_tile`` and never leave the expert's groups: the minimal cover is
// at most ``group_rows * ceil(c / group_rows)`` (a windows-only rule would
// not be: 192 rows over a 37-row expert's single 128-row group).
//
// Dual-tile routing (``alt_expert_idx != nullptr``): the routing padded every
// expert to ``group_rows`` or ``alt_group_rows`` rows at run time and wrote
// the coarser list only when it chose it (``base_active == 0`` and
// ``alt_num_groups > 0``).  The kernel then covers each expert with
// ``alt_group_rows``-row dense tiles (listed as alternate-group indices in
// ``alt_wide_list``) and windows; the list of the tile the routing did not
// choose gets count 0.  The windows are ``row_unit`` offsets of the shared
// permutation either way (the base list is always valid over the chosen
// padding, so the swap kernel's ``group_rows``-granular lookups hold).
//
// The expert / limit arrays of both lists are staged in dynamic shared memory
// up to their buffer lengths (``stage_base`` / ``stage_alt`` groups, issued
// together with the three count loads so the kernel pays one memory latency
// before the scan); a list longer than its staged length reads global memory.
// ``narrow_count_base`` (optional) receives the window count under the base
// padding and 0 under the alternate padding, for a GEMM2 that runs the swap
// finalize over the windows only when the routing chose the base tile.
constexpr int kMixedItems = 4;

__device__ __forceinline__ uint64_t globaltimer_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// Staged loads per thread issued before their shared-memory stores, so the
// staging loop overlaps its global latencies instead of serializing them.
constexpr int kStageUnroll = 4;

template <int kUnroll>
__device__ __forceinline__ void stage_pair(const int32_t* __restrict__ a,
                                           const int32_t* __restrict__ b, int32_t n,
                                           int32_t* __restrict__ sa, int32_t* __restrict__ sb) {
  for (int i = static_cast<int>(threadIdx.x); i < n; i += kThreads * kUnroll) {
    int32_t va[kUnroll], vb[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int idx = i + u * kThreads;
      if (idx < n) {
        va[u] = a[idx];
        vb[u] = b[idx];
      }
    }
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int idx = i + u * kThreads;
      if (idx < n) {
        sa[idx] = va[u];
        sb[idx] = vb[u];
      }
    }
  }
}

template <bool kPdl>
__global__ void __launch_bounds__(kThreads) swapab_dispatch_mixed_kernel(
    const int32_t* __restrict__ expert_idx, const int32_t* __restrict__ mn_limit,
    const int32_t* __restrict__ num_groups_ptr, const int32_t* __restrict__ alt_expert_idx,
    const int32_t* __restrict__ alt_mn_limit, const int32_t* __restrict__ alt_num_groups_ptr,
    const int32_t* __restrict__ base_active_ptr, int32_t group_rows, int32_t alt_group_rows,
    int32_t narrow_tile, int32_t row_unit, int32_t stage_base, int32_t stage_alt,
    int32_t* __restrict__ wide_list, int32_t* __restrict__ wide_count,
    int32_t* __restrict__ alt_wide_list, int32_t* __restrict__ alt_wide_count,
    int32_t* __restrict__ narrow_list, int32_t* __restrict__ narrow_count,
    int32_t* __restrict__ narrow_count_base, int64_t* __restrict__ trace) {
  // Optional phase trace (globaltimer ns, thread 0): [0] start, [1] lists
  // staged, [2 + k] after scan pass k (k < 6), [8] end.
  if (trace != nullptr && threadIdx.x == 0) {
    trace[0] = static_cast<int64_t>(globaltimer_ns());
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (kPdl) {
    cudaGridDependencySynchronize();
    // Dependents wait for this grid's completion before reading the lists
    // (every consumer runs griddepcontrol.wait first); releasing them here
    // hides their launch latency under this kernel.
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
  // Warp scans: the default raking scan serializes 32 struct adds per lane
  // of one warp twice per pass (about 3 us for 1024 threads).
  using Scan = cub::BlockScan<Counts, kThreads, cub::BLOCK_SCAN_WARP_SCANS>;
  __shared__ typename Scan::TempStorage temp;
  __shared__ int s_counts[3];
  extern __shared__ int32_t staged[];
  // One latency: the counts and both lists' arrays (up to their staged
  // lengths, buffer contents past the routing's groups are never read).
  if (threadIdx.x < 3) {
    const int32_t* p = threadIdx.x == 0   ? num_groups_ptr
                       : threadIdx.x == 1 ? alt_num_groups_ptr
                                          : base_active_ptr;
    s_counts[threadIdx.x] = p != nullptr ? *p : 0;
  }
  int32_t* const s_base_e = staged;
  int32_t* const s_base_l = staged + stage_base;
  int32_t* const s_alt_e = staged + 2 * stage_base;
  int32_t* const s_alt_l = staged + 2 * stage_base + stage_alt;
  stage_pair<kStageUnroll>(expert_idx, mn_limit, stage_base, s_base_e, s_base_l);
  if (stage_alt > 0) {
    stage_pair<kStageUnroll>(alt_expert_idx, alt_mn_limit, stage_alt, s_alt_e, s_alt_l);
  }
  __syncthreads();
  if (trace != nullptr && threadIdx.x == 0) {
    trace[1] = static_cast<int64_t>(globaltimer_ns());
  }
  const bool alt = (alt_expert_idx != nullptr) && (s_counts[2] == 0) && (s_counts[1] > 0);
  const int num_groups = alt ? s_counts[1] : s_counts[0];
  const int staged_len = alt ? stage_alt : stage_base;
  const int32_t* e = alt ? alt_expert_idx : expert_idx;
  const int32_t* l = alt ? alt_mn_limit : mn_limit;
  if (num_groups <= staged_len) {
    e = alt ? s_alt_e : s_base_e;
    l = alt ? s_alt_l : s_base_l;
  }
  const int rows = alt ? alt_group_rows : group_rows;
  int32_t* const wl = alt ? alt_wide_list : wide_list;
  const int gu = rows / row_unit;
  Counts carry{0, 0, 0};
  for (int base_g = 0; base_g < num_groups; base_g += kThreads * kMixedItems) {
    Counts mine[kMixedItems];
#pragma unroll
    for (int j = 0; j < kMixedItems; ++j) {
      const int g = base_g + static_cast<int>(threadIdx.x) * kMixedItems + j;
      int nwide = 0, nnarrow = 0;
      if (g < num_groups) {
        // The first group of an expert owns the expert's whole row range,
        // bounded by the last group of the run of equal expert indices.  The
        // sort writes the groups in expert order, so the run end is found by
        // a binary search over the non-decreasing expert indices.
        const int ex = e[g];
        const bool first = (g == 0) || (e[g - 1] != ex);
        int c = 0;
        if (first) {
          int lo = g + 1, hi = num_groups;  // first index with a different expert in [lo, hi]
          while (lo < hi) {
            const int mid = lo + ((hi - lo) >> 1);
            if (e[mid] == ex) {
              lo = mid + 1;
            } else {
              hi = mid;
            }
          }
          c = max(l[lo - 1] - g * rows, 0);
        }
        if (c > 0) {
          // Minimal cover by wide (rows) tiles and narrow windows, ties to
          // fewer windows.  cover(a + gu) == cover(a) for gu = rows /
          // row_unit, so a in [0, gu) suffices: with 128-row groups and
          // 192-row windows the cover is ceil(c / 64) * 64 (at least one wide
          // tile) and at most one window per expert; with 256-row groups it
          // is at most three windows per expert.
          int best_cover = 0x7fffffff;
          for (int a = 0; a < gu; ++a) {
            const int rem = c - a * narrow_tile;
            const int w = rem > 0 ? (rem + rows - 1) / rows : 0;
            const int cover = w * rows + a * narrow_tile;
            if (cover < best_cover) {
              best_cover = cover;
              nwide = w;
              nnarrow = a;
            }
            if (rem <= 0) break;
          }
        }
      }
      mine[j] = Counts{nwide, nnarrow, 0};
    }
    Counts excl[kMixedItems];
    Counts total{0, 0, 0};
    Scan(temp).ExclusiveScan(mine, excl, Counts{0, 0, 0}, CountsAdd(), total);
#pragma unroll
    for (int j = 0; j < kMixedItems; ++j) {
      const int g = base_g + static_cast<int>(threadIdx.x) * kMixedItems + j;
      for (int t = 0; t < mine[j].wide; ++t) {
        wl[carry.wide + excl[j].wide + t] = g + t;
      }
      const int narrow_row0 = (g + mine[j].wide) * rows;
      for (int i = 0; i < mine[j].narrow; ++i) {
        narrow_list[carry.narrow + excl[j].narrow + i] = (narrow_row0 + i * narrow_tile) / row_unit;
      }
    }
    carry = CountsAdd()(carry, total);
    __syncthreads();
    if (trace != nullptr && threadIdx.x == 0) {
      const int k = base_g / (kThreads * kMixedItems);
      if (k < 6) trace[2 + k] = static_cast<int64_t>(globaltimer_ns());
    }
  }
  if (threadIdx.x == 0) {
    if (trace != nullptr) {
      trace[8] = static_cast<int64_t>(globaltimer_ns());
    }
    *wide_count = alt ? 0 : carry.wide;
    if (alt_wide_count != nullptr) {
      *alt_wide_count = alt ? carry.wide : 0;
    }
    *narrow_count = carry.narrow;
    if (narrow_count_base != nullptr) {
      *narrow_count_base = alt ? 0 : carry.narrow;
    }
  }
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

void moe_swapab_dispatch_mixed(int64_t expert_idx_ptr, int64_t mn_limit_ptr, int64_t num_groups_ptr,
                               int64_t alt_expert_idx_ptr, int64_t alt_mn_limit_ptr,
                               int64_t alt_num_groups_ptr, int64_t base_active_ptr,
                               int64_t group_rows, int64_t alt_group_rows, int64_t narrow_tile,
                               int64_t row_unit, int64_t stage_base, int64_t stage_alt,
                               int64_t wide_list_ptr, int64_t wide_count_ptr,
                               int64_t alt_wide_list_ptr, int64_t alt_wide_count_ptr,
                               int64_t narrow_list_ptr, int64_t narrow_count_ptr,
                               int64_t narrow_count_base_ptr, int64_t trace_ptr, bool use_pdl,
                               int64_t cuda_stream_ptr) {
  TVM_FFI_ICHECK(row_unit > 0 && group_rows > 0 && narrow_tile > 0 && group_rows % row_unit == 0 &&
                 narrow_tile % row_unit == 0)
      << "group_rows and narrow_tile must be positive multiples of row_unit";
  const bool dual = alt_expert_idx_ptr != 0;
  TVM_FFI_ICHECK((alt_mn_limit_ptr != 0) == dual && (alt_num_groups_ptr != 0) == dual &&
                 (base_active_ptr != 0) == dual && (alt_wide_list_ptr != 0) == dual &&
                 (alt_wide_count_ptr != 0) == dual)
      << "the dual-tile arrays, counts and alternate wide list must be given together";
  TVM_FFI_ICHECK(!dual || (alt_group_rows > group_rows && alt_group_rows % row_unit == 0))
      << "alt_group_rows must be a multiple of row_unit above group_rows";
  TVM_FFI_ICHECK(stage_base >= 0 && stage_alt >= 0 && (dual || stage_alt == 0) &&
                 stage_base + stage_alt <= (1 << 20))
      << "staged lengths out of range";
  cudaStream_t stream =
      cuda_stream_ptr != 0 ? reinterpret_cast<cudaStream_t>(cuda_stream_ptr) : get_current_stream();
  const size_t smem_bytes = 2 * sizeof(int32_t) * static_cast<size_t>(stage_base + stage_alt);
  auto* kernel = use_pdl ? swapab_dispatch_mixed_kernel<true> : swapab_dispatch_mixed_kernel<false>;
  // Static (block scan) plus dynamic (staging) shared memory exceeds the 48
  // KB default without the opt-in; the largest dynamic size seen is recorded
  // per kernel variant so replays (graph capture) skip the call.
  static size_t configured_bytes[2] = {0, 0};
  size_t& configured = configured_bytes[use_pdl ? 1 : 0];
  if (smem_bytes > configured || configured == 0) {
    const size_t request = std::max(smem_bytes, static_cast<size_t>(1));
    cudaError_t aerr = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(request));
    TVM_FFI_ICHECK(aerr == cudaSuccess)
        << "moe_swapab_dispatch_mixed: cannot reserve " << request
        << " B of dynamic shared memory: " << cudaGetErrorString(aerr);
    configured = request;
  }
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(1);
  config.blockDim = dim3(kThreads);
  config.dynamicSmemBytes = smem_bytes;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = use_pdl ? 1 : 0;
  config.attrs = attrs;
  config.numAttrs = 1;
  cudaError_t err = cudaLaunchKernelEx(
      &config, kernel, reinterpret_cast<const int32_t*>(expert_idx_ptr),
      reinterpret_cast<const int32_t*>(mn_limit_ptr),
      reinterpret_cast<const int32_t*>(num_groups_ptr),
      reinterpret_cast<const int32_t*>(alt_expert_idx_ptr),
      reinterpret_cast<const int32_t*>(alt_mn_limit_ptr),
      reinterpret_cast<const int32_t*>(alt_num_groups_ptr),
      reinterpret_cast<const int32_t*>(base_active_ptr), static_cast<int32_t>(group_rows),
      static_cast<int32_t>(alt_group_rows), static_cast<int32_t>(narrow_tile),
      static_cast<int32_t>(row_unit), static_cast<int32_t>(stage_base),
      static_cast<int32_t>(stage_alt), reinterpret_cast<int32_t*>(wide_list_ptr),
      reinterpret_cast<int32_t*>(wide_count_ptr), reinterpret_cast<int32_t*>(alt_wide_list_ptr),
      reinterpret_cast<int32_t*>(alt_wide_count_ptr), reinterpret_cast<int32_t*>(narrow_list_ptr),
      reinterpret_cast<int32_t*>(narrow_count_ptr),
      reinterpret_cast<int32_t*>(narrow_count_base_ptr), reinterpret_cast<int64_t*>(trace_ptr));
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "moe_swapab_dispatch_mixed launch failed: " << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(flashinfer_moe_swapab_dispatch, moe_swapab_dispatch);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(flashinfer_moe_swapab_dispatch_mixed, moe_swapab_dispatch_mixed);
