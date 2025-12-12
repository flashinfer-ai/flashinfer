#pragma once

#include "cute/tensor.hpp"
#include "cutlass/detail/layout.hpp"

namespace flat {

using namespace cute;

template <int... Is, typename Layout>
__forceinline__ __host__ __device__ constexpr auto select_layout(Layout&& l) {
  if constexpr (is_composed_layout<Layout>::value) {
    return make_composed_layout(l.layout_a(), l.offset(), select<Is...>(l.layout_b()));
  } else {
    return select<Is...>(l);
  }
}

template <int... Is, typename Tensor>
__forceinline__ __host__ __device__ constexpr auto select_tensor(Tensor&& t) {
  if constexpr (is_composed_layout<decltype(t.layout())>::value) {
    return make_tensor(
        std::forward<Tensor>(t).data(),
        make_composed_layout(std::forward<Tensor>(t).layout().layout_a(),
                             std::forward<Tensor>(t).layout().offset(),
                             select<Is...>(std::forward<Tensor>(t).layout().layout_b())));
  } else {
    return make_tensor(std::forward<Tensor>(t).data(), select<Is...>(t.layout()));
  }
}

template <class Layout>
CUTE_DEVICE constexpr size_t alignment_for_swizzle(Layout&& layout) {
  return cutlass::detail::alignment_for_swizzle(std::forward<Layout>(layout));
}

}  // namespace flat
