// Copyright (c) 2026 by FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>

#include "dsv4_geometry.cuh"
#include "kv_storage.cuh"

using bf16 = __nv_bfloat16;

namespace flashinfer::sparse_mla_sm120 {

struct Dsv4Nvfp4Layout
    : FooterScaleLayout<
          Dsv4Geometry::D_NOPE * E2m1Encoding::BITS / 8 + Dsv4Geometry::D_ROPE * sizeof(bf16),
          (Dsv4Geometry::D_NOPE / E4m3G16ScaleSpec::GROUP * E4m3G16ScaleSpec::BYTES_PER_SCALE +
           15) /
              16 * 16> {
  using Geometry = Dsv4Geometry;
  using Encoding = E2m1Encoding;
  using Scales = E4m3G16ScaleSpec;
  static constexpr int D_NOPE = Geometry::D_NOPE;
  static constexpr int D_ROPE = Geometry::D_ROPE;
  static constexpr int D_QK = Geometry::D_QK;
  static constexpr int D_V = Geometry::D_V;
  static constexpr int SCALE_GROUP_SIZE = Scales::GROUP;
  static constexpr int NUM_SCALES = D_NOPE / SCALE_GROUP_SIZE;
  static constexpr int PACKED_NOPE_BYTES = D_NOPE * Encoding::BITS / 8;
  static constexpr int ROPE_BYTES = D_ROPE * sizeof(bf16);
  static constexpr int DATA_BYTES_PER_TOKEN = DATA_BYTES;
  static constexpr int SCALE_BYTES_PER_TOKEN = SCALE_BYTES;
};

static_assert(Dsv4Nvfp4Layout::NUM_SCALES == 28);
static_assert(Dsv4Nvfp4Layout::DATA_BYTES_PER_TOKEN == 352);
static_assert(Dsv4Nvfp4Layout::BYTES_PER_TOKEN == 384);

}  // namespace flashinfer::sparse_mla_sm120
