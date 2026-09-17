// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "model_type.h"
#include "scale_convert.cuh"

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
  // Offset of the scale payload within the packed token row (inline) or block
  // (footer, after all data rows).
  static constexpr int gmem_offset(int d_nope, int rope_bytes) {
    return INLINE ? d_nope : d_nope + rope_bytes;
  }
};

// Per-format scale -> UE8M0 conversion for block-scaled MMA. UE8M0 caches
// need no conversion; both FP32 formats reduce to the exponent byte.
template <ScaleFormat F>
struct ScaleConvert {
  static_assert(F == ScaleFormat::POW2_FP32 || F == ScaleFormat::ARBITRARY_FP32,
                "add a ScaleConvert specialization for this scale format");
  __device__ static __forceinline__ uint8_t to_ue8m0(float scale) { return fp32_to_ue8m0(scale); }
};

template <>
struct ScaleConvert<ScaleFormat::UE8M0_BYTE> {
  __device__ static __forceinline__ uint8_t to_ue8m0(uint8_t scale) { return scale; }
};
