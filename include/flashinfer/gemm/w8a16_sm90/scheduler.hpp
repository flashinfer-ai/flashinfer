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
#ifndef FLASHINFER_GEMM_W8A16_SM90_SCHEDULER_HPP_
#define FLASHINFER_GEMM_W8A16_SM90_SCHEDULER_HPP_

// A persistent tile scheduler that groups tiles for L2 without padding the tile grid.
// CUTLASS's sm90 persistent scheduler implements its L2 swizzle by rounding both tile
// counts up to a multiple of the swizzle size and running the padding tiles (see
// tile_order.hpp), which would force the large-M rows to run ungrouped: the 132 CTAs in
// flight walk ~132 weight tiles of ONE activation tile, and every activation tile re-streams
// the weight from HBM once the weight is larger than what fits next to it in the 50 MB L2.
// This scheduler walks the same linear work index as CUTLASS's (swizzle 1, so its grid and
// work count are exact), but maps it differently: groups of G tiles of the
// slow dimension (activation tiles for raster "m"), the fast dimension walked within a group,
// and a last, narrower group for the remainder instead of padding.  G = 1 is exactly
// CUTLASS's order.  Cluster size 1 only; the stream-K variants keep CUTLASS's scheduler.

#include <cstdint>

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.hpp"

namespace flashinfer::gemm {

struct W8A16Sm90GroupedTileSchedulerTag {};

struct W8A16Sm90GroupedParams : cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params {
  uint32_t group_ = 1;       // tiles of the slow dimension per group
  uint32_t tiles_fast_ = 0;  // tiles along the raster direction
  uint32_t tiles_slow_ = 0;
  uint32_t full_groups_ = 0;             // tiles_slow_ / group_
  uint32_t full_tiles_ = 0;              // full_groups_ * group_ * tiles_fast_
  cutlass::FastDivmod div_group_tiles_;  // / (group_ * tiles_fast_)
  cutlass::FastDivmod div_group_;        // / group_
  cutlass::FastDivmod div_rem_;          // / (tiles_slow_ - full_groups_ * group_), or 1
  cutlass::FastDivmod div_batch_tiles_;  // / (tiles_slow_ * tiles_fast_): tiles per batch
};

// Index map: FastDivmod (multiply-high + shift, precomputed on the host), as CUTLASS's own
// scheduler does.  Measured equal to the stock scheduler and to a plain-division build
// (same time within noise, same instruction count under ncu).
class W8A16Sm90GroupedTileScheduler
    : public cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90 {
  using Base = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90;

 public:
  using Params = W8A16Sm90GroupedParams;
  using Arguments = Base::Arguments;  // max_swizzle_size carries the group size
  using WorkTileInfo = Base::WorkTileInfo;
  using RasterOrder = Base::RasterOrder;
  using RasterOrderOptions = Base::RasterOrderOptions;

  CUTLASS_HOST_DEVICE W8A16Sm90GroupedTileScheduler() {}

  CUTLASS_DEVICE explicit W8A16Sm90GroupedTileScheduler(Params const& p)
      : Base(p), p_(&p), along_m_(p.raster_order_ == RasterOrder::AlongM) {
#if defined(__CUDA_ARCH__)
    // Same start index as CUTLASS (one of gridDim.x / gridDim.y is 1 at cluster size 1).
    linear_ = uint64_t(blockIdx.x) * uint64_t(gridDim.y) + uint64_t(blockIdx.y);
    stride_ = uint64_t(gridDim.x) * uint64_t(gridDim.y) * uint64_t(gridDim.z);
#endif
  }

  template <class ProblemShapeMNKL, class TileShape, class ClusterShape>
  static Params to_underlying_arguments(ProblemShapeMNKL problem_shape_mnkl, TileShape tile_shape,
                                        ClusterShape cluster_shape,
                                        cutlass::KernelHardwareInfo const& hw_info,
                                        Arguments const& arguments, void* workspace = nullptr,
                                        const uint32_t = 1, uint32_t = 1u) {
    static_assert(cute::size(ClusterShape{}) == 1, "grouped scheduler: cluster size 1 only");
    Arguments a = arguments;
    // max_swizzle_size controls output-tile traversal for L2 reuse; it is independent
    // of the TMA shared-memory swizzle chosen by the CUTLASS mainloop.
    // Disable CUTLASS's tile-order grouping here so the base scheduler does not pad
    // either tile count to the group size. Preserve the requested size in p.group_
    // below: work_for_linear_idx() applies our grouping to the unpadded tile grid,
    // using a smaller final group for the remainder.
    a.max_swizzle_size = 1;
    Params p;
    static_cast<cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params&>(p) =
        Base::to_underlying_arguments(problem_shape_mnkl, tile_shape, cluster_shape, hw_info, a,
                                      workspace);
    p.group_ =
        static_cast<uint32_t>(arguments.max_swizzle_size > 1 ? arguments.max_swizzle_size : 1);
    const bool along_m = p.raster_order_ == RasterOrder::AlongM;
    p.tiles_fast_ = along_m ? p.problem_tiles_m_ : p.problem_tiles_n_;
    p.tiles_slow_ = along_m ? p.problem_tiles_n_ : p.problem_tiles_m_;
    if (p.group_ > p.tiles_slow_) p.group_ = p.tiles_slow_ > 0 ? p.tiles_slow_ : 1;
    p.full_groups_ = p.tiles_slow_ / p.group_;
    p.full_tiles_ = p.full_groups_ * p.group_ * p.tiles_fast_;
    const uint32_t rem = p.tiles_slow_ - p.full_groups_ * p.group_;
    p.div_group_tiles_ = cutlass::FastDivmod(static_cast<int>(p.group_ * p.tiles_fast_));
    p.div_group_ = cutlass::FastDivmod(static_cast<int>(p.group_));
    p.div_rem_ = cutlass::FastDivmod(static_cast<int>(rem > 0 ? rem : 1));
    p.div_batch_tiles_ = cutlass::FastDivmod(static_cast<int>(p.tiles_slow_ * p.tiles_fast_));
    return p;
  }

  template <class ClusterShape>
  CUTLASS_DEVICE WorkTileInfo initial_work_tile_info(ClusterShape) {
    return get_current_work();
  }

  CUTLASS_DEVICE WorkTileInfo get_current_work() const { return work_for_linear_idx(linear_); }

  CUTLASS_DEVICE WorkTileInfo work_for_linear_idx(uint64_t u64) const {
    Params const& p = *p_;
    if (u64 >= p.blocks_per_problem_) return WorkTileInfo::invalid_work_tile();
    // Tile counts here fit in int (CUTLASS's own FastDivmod paths assume the same).  The
    // batch (split-K slice) is the slowest index, as in CUTLASS's own scheduler
    // (blocks_per_problem_ counts all batches); the group map runs inside one batch.
    int l, u;
    p.div_batch_tiles_.fast_divmod(l, u, static_cast<int>(u64));
    int slow, fast;
    if (static_cast<uint32_t>(u) < p.full_tiles_) {
      int gi, in, q, r;
      p.div_group_tiles_.fast_divmod(gi, in, u);
      p.div_group_.fast_divmod(q, r, in);
      slow = gi * static_cast<int>(p.group_) + r;
      fast = q;
    } else {  // the remainder group, narrower, no padding
      int q, r;
      p.div_rem_.fast_divmod(q, r, u - static_cast<int>(p.full_tiles_));
      slow = static_cast<int>(p.full_groups_ * p.group_) + r;
      fast = q;
    }
    return {along_m_ ? fast : slow, along_m_ ? slow : fast, l, true};
  }

  CUTLASS_DEVICE void advance_to_next_work(uint32_t advance_count = 1) {
    linear_ += stride_ * uint64_t(advance_count);
  }

  // Only validity is asked here (the consumer's per-tile last-tile test): no index map.
  CUTLASS_DEVICE bool is_last_tile(WorkTileInfo&, uint32_t advance_count = 1) const {
    return linear_ + stride_ * uint64_t(advance_count) >= p_->blocks_per_problem_;
  }

  CUTLASS_DEVICE auto fetch_next_work(WorkTileInfo) {
    advance_to_next_work();
    return cute::make_tuple(get_current_work(), true);
  }

  template <class TileSchedulerPipeline, class TileSchedulerPipelineState>
  CUTLASS_DEVICE auto fetch_next_work(WorkTileInfo w, TileSchedulerPipeline&,
                                      TileSchedulerPipelineState) {
    return fetch_next_work(w);
  }

 private:
  Params const* p_ = nullptr;  // the kernel's __grid_constant__ params
  bool along_m_ = true;
  uint64_t linear_ = 0;
  uint64_t stride_ = 1;
};

}  // namespace flashinfer::gemm

namespace cutlass::gemm::kernel::detail {
template <class ArchTag, class TileShape, class ClusterShape, uint32_t SchedulerPipelineStageCount>
struct TileSchedulerSelector<flashinfer::gemm::W8A16Sm90GroupedTileSchedulerTag, ArchTag, TileShape,
                             ClusterShape, SchedulerPipelineStageCount> {
  using Scheduler = flashinfer::gemm::W8A16Sm90GroupedTileScheduler;
};
}  // namespace cutlass::gemm::kernel::detail

#endif  // FLASHINFER_GEMM_W8A16_SM90_SCHEDULER_HPP_
