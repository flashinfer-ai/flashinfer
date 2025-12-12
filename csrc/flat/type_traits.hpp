#pragma once

#include <type_traits>

#include "cutlass/numeric_types.h"

namespace flat {

// clang-format off
template <typename T> struct map_to_cutlass;
template<> struct map_to_cutlass<cutlass::half_t>             { using type = cutlass::half_t;                    };
template<> struct map_to_cutlass<cutlass::bfloat16_t>         { using type = cutlass::bfloat16_t;                };
template<> struct map_to_cutlass<half>                        { using type = cutlass::half_t;                    };
template<> struct map_to_cutlass<nv_bfloat16>                 { using type = cutlass::bfloat16_t;                };

template <typename T> using map_to_cutlass_t = typename map_to_cutlass<T>::type;
// clang-format on

template <typename... Ts>
struct first_non_void {
  static_assert(sizeof...(Ts) > 0, "all voids is not allowed");
  using type = void;
};

template <typename T, typename... Ts>
struct first_non_void<T, Ts...> {
  using type = T;
};

template <typename... Ts>
struct first_non_void<void, Ts...> : first_non_void<Ts...> {};

template <typename... Ts>
using first_non_void_t = typename first_non_void<Ts...>::type;

}  // namespace flat
