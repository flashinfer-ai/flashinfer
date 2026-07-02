/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <cute/util/type_traits.hpp>

namespace flashinfer::gemm::mxfp8_cute_sm120::sm120_blockscaled {
namespace math {

CUTE_HOST_DEVICE
static auto ceil_div(const int& x, const int& y) { return (x + y - 1) / y; }

CUTE_HOST_DEVICE
static auto align(const int& x, const int& alignment) { return ceil_div(x, alignment) * alignment; }

template <typename T_offset, typename T_index>
CUTE_HOST_DEVICE T_offset compute_padded_offset(T_offset offset, T_index problem_idx) {
  constexpr T_offset alignment = 4;
  return (offset + problem_idx * (alignment - 1)) / alignment * alignment;
}

}  // namespace math
}  // namespace flashinfer::gemm::mxfp8_cute_sm120::sm120_blockscaled
