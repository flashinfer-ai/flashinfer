// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "model_type.h"

struct E2m1Encoding {
  static constexpr int BITS = 4;
};
struct E4m3Encoding {
  static constexpr int BITS = 8;
};
struct E4m3G16ScaleSpec {
  using Encoding = E4m3Encoding;
  static constexpr int GROUP = 16;
  static constexpr int BYTES_PER_SCALE = 1;
};

// ScaleSpec: the scale configuration of a packed KV cache row, decoupled from
// model geometry. Three axes — numeric FORMAT, quantization GROUP size, and
// placement INLINE (between nope and rope inside the row) vs footer (a padded
// footer after the block's data rows). KVCacheTraits composes one spec per
// model and forwards the derived constants, so a new (geometry, scale)
// combination is one traits row rather than a traits rewrite.
//
// All helpers take the geometry as arguments so the spec stays usable for any
// D_NOPE / D_ROPE.
template <ScaleFormat F_, int GROUP_, bool INLINE_>
struct ScaleSpec {
  static constexpr ScaleFormat FORMAT = F_;
  static constexpr int GROUP = GROUP_;
  static constexpr bool INLINE = INLINE_;
  static constexpr int BYTES_PER_SCALE = (F_ == ScaleFormat::UE8M0_BYTE) ? 1 : 4;

  static constexpr int count(int d_nope) { return d_nope / GROUP; }
  // Footer layouts round the per-token scale bytes up to 8B (one uint64
  // gather); inline layouts are exact FP32 arrays.
  static constexpr int bytes_per_token(int d_nope) {
    return INLINE ? count(d_nope) * BYTES_PER_SCALE : (count(d_nope) * BYTES_PER_SCALE + 7) / 8 * 8;
  }
  // Data bytes preceding scales per token: an inline offset, but only the
  // data-row width for footer storage. Layout::scale_offset owns addressing.
  static constexpr int data_prefix_bytes(int d_nope, int rope_bytes) {
    return INLINE ? d_nope : d_nope + rope_bytes;
  }
};

template <int DataBytes, int ScaleBytes>
struct FooterScaleLayout {
  static constexpr int DATA_BYTES = DataBytes;
  static constexpr int SCALE_BYTES = ScaleBytes;
  static constexpr int BYTES_PER_TOKEN = DataBytes + ScaleBytes;
  template <typename Index>
  __host__ __device__ static constexpr auto data_offset(Index row) {
    return row * DataBytes;
  }
  template <typename PageSize, typename Index>
  __host__ __device__ static constexpr auto scale_offset(PageSize page_size, Index row) {
    return page_size * DataBytes + row * ScaleBytes;
  }
  template <typename Other, typename Index>
  __host__ __device__ static constexpr auto selected_data_offset(bool selected, Index row) {
    return row * (selected ? DataBytes : Other::DATA_BYTES);
  }
  template <typename Other, typename PageSize, typename Index>
  __host__ __device__ static constexpr auto selected_scale_offset(bool selected, PageSize page_size,
                                                                  Index row) {
    return page_size * (selected ? DataBytes : Other::DATA_BYTES) +
           row * (selected ? ScaleBytes : Other::SCALE_BYTES);
  }
};
