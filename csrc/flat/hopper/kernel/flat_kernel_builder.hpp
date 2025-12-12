#pragma once

#include "../collective/flat_collective_tma_warpspecialized.hpp"
#include "../kernel/flat_tile_scheduler.hpp"
#include "../kernel/flat_kernel_tma_warpspecialized.hpp"
#include "../kernel/flat_options.hpp"

#include "flat/type_traits.hpp"

namespace flat::kernel {

template <
    class Element_,
    class ElementAccumulatorQK_,
    class ElementAccumulatorPV_,
    class TileShape_,  // BlkSeqQO, BlkSeqKV, HeadSize
    class LayoutQ_,
    class LayoutK_,
    class LayoutV_,
    class LayoutO_,
    class DispatchPolicy,
    class Options = DefaultOptions>
struct FlatBuilder;

template <
    class Element,
    class ElementAccumulatorQK,
    class ElementAccumulatorPV,
    class TileShape,  // BlkSeqQO, BlkSeqKV, HeadSize
    class LayoutQ,
    class LayoutK,
    class LayoutV,
    class LayoutO,
    class Options>
struct FlatBuilder<
    Element,
    ElementAccumulatorQK,
    ElementAccumulatorPV,
    TileShape,
    LayoutQ,
    LayoutK,
    LayoutV,
    LayoutO,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    Options> {
  using CollectiveMainloop = flat::collective::FlatMainloopTmaWarpSpecialized<
      Element, ElementAccumulatorQK, ElementAccumulatorPV,
      TileShape, LayoutQ, LayoutK, LayoutV, LayoutO,
      Options>;

  static constexpr bool kIsPersistent = find_option_t<Tag::kIsPersistent, false_type, Options>::value;
  static_assert(!kIsPersistent, "not implemented");

  using GroupingTag   = GQATag;
  using TileScheduler = flat::kernel::IndividualTileScheduler<GroupingTag>;
  // using TileScheduler = std::conditional_t<kIsPersistent, flat::kernel::PersistentTileScheduler, flat::kernel::IndividualTileScheduler>;

  using Kernel = flat::kernel::FlatKernelTmaWarpSpecialized<CollectiveMainloop, TileScheduler, Options>;
};

}  // namespace flat::kernel
