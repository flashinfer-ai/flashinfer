/*
 * Copyright (c) 2024 by FlashInfer team.
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
#ifndef FLASHINFER_ATTENTION_WARP_LAYOUT_CUH_
#define FLASHINFER_ATTENTION_WARP_LAYOUT_CUH_
#include <cstdint>
#include <sstream>
#include <stdexcept>

namespace flashinfer {

// NOTE(Zihao): this file is ugly and should be refactored

enum class WarpLayout {
  k4x1x2 = 0U,
  k4x1x1 = 1U,
  k1x4x1 = 2U,
};

template <WarpLayout warp_layout>
constexpr uint32_t get_num_warps_x() {
  return 4;
}

template <>
constexpr uint32_t get_num_warps_x<WarpLayout::k4x1x2>() {
  return 4;
}

template <>
constexpr uint32_t get_num_warps_x<WarpLayout::k4x1x1>() {
  return 4;
}

template <>
constexpr uint32_t get_num_warps_x<WarpLayout::k1x4x1>() {
  return 1;
}

template <WarpLayout warp_layout>
constexpr uint32_t get_num_warps_z() {
  return 1;
}

template <>
constexpr uint32_t get_num_warps_z<WarpLayout::k4x1x2>() {
  return 1;
}

template <>
constexpr uint32_t get_num_warps_z<WarpLayout::k4x1x1>() {
  return 1;
}

template <>
constexpr uint32_t get_num_warps_z<WarpLayout::k1x4x1>() {
  return 4;
}

template <WarpLayout warp_layout>
constexpr uint32_t get_num_frags_x() {
  return 2;
}

template <>
constexpr uint32_t get_num_frags_x<WarpLayout::k4x1x2>() {
  return 2;
}

template <>
constexpr uint32_t get_num_frags_x<WarpLayout::k4x1x1>() {
  return 1;
}

template <>
constexpr uint32_t get_num_frags_x<WarpLayout::k1x4x1>() {
  return 1;
}

#define DISPATCH_WARP_LAYOUT(warp_layout, WARP_LAYOUT, ...)     \
  if (warp_layout == WarpLayout::k4x1x2) {                      \
    constexpr WarpLayout WARP_LAYOUT = WarpLayout::k4x1x2;      \
    __VA_ARGS__                                                 \
  } else if (warp_layout == WarpLayout::k4x1x1) {               \
    constexpr WarpLayout WARP_LAYOUT = WarpLayout::k4x1x1;      \
    __VA_ARGS__                                                 \
  } else if (warp_layout == WarpLayout::k1x4x1) {               \
    constexpr WarpLayout WARP_LAYOUT = WarpLayout::k1x4x1;      \
    __VA_ARGS__                                                 \
  } else {                                                      \
    std::ostringstream err_msg;                                 \
    err_msg << "Unsupported warp layout: " << int(warp_layout); \
    throw std::invalid_argument(err_msg.str());                 \
  }

inline uint32_t get_num_rows_per_cta(WarpLayout warp_layout) {
  DISPATCH_WARP_LAYOUT(warp_layout, WARP_LAYOUT, {
    return get_num_warps_x<WARP_LAYOUT>() * get_num_frags_x<WARP_LAYOUT>() * 16;
  });
}

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_WARP_LAYOUT_CUH_
