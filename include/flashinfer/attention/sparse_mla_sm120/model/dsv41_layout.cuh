// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "kv_storage.cuh"

struct Dsv41Geometry {
  static constexpr int D_NOPE = 512;
  static constexpr int D_ROPE = 0;
  static constexpr int D_QK = D_NOPE;
  static constexpr int D_V = 512;
  static constexpr bool V_HAS_ROPE = false;
};

using Dsv41Fp8Scales = ScaleSpec<ScaleFormat::UE8M0_BYTE, 32, false>;
struct Dsv41Fp8Layout : Dsv41Geometry,
                        FooterScaleLayout<Dsv41Geometry::D_NOPE,
                                          Dsv41Fp8Scales::bytes_per_token(Dsv41Geometry::D_NOPE)> {
  using Geometry = Dsv41Geometry;
  using Encoding = E4m3Encoding;
  using Scales = Dsv41Fp8Scales;
  static constexpr int QUANT_TILE = Scales::GROUP;
  static constexpr int NUM_SCALES = Scales::count(D_NOPE);
  static constexpr ScaleFormat SCALE_FORMAT = Scales::FORMAT;
  static constexpr bool SCALE_INLINE = Scales::INLINE;
  static constexpr int SCALE_BYTES_PER_TOKEN = SCALE_BYTES;
  static constexpr int KV_ROPE_GMEM_OFFSET = D_NOPE;
  static constexpr int SCALE_DATA_PREFIX_BYTES = Scales::data_prefix_bytes(D_NOPE, 0);
  static constexpr bool V_HAS_ROPE = false;
};

// Copyright (c) 2026 by FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
namespace flashinfer::sparse_mla_sm120 {

struct Dsv41Fp4Layout : FooterScaleLayout<Dsv41Geometry::D_QK * E2m1Encoding::BITS / 8,
                                          Dsv41Geometry::D_QK / E4m3G16ScaleSpec::GROUP *
                                              E4m3G16ScaleSpec::BYTES_PER_SCALE> {
  using Geometry = Dsv41Geometry;
  using Encoding = E2m1Encoding;
  using Scales = E4m3G16ScaleSpec;
  static constexpr int DIMS = Geometry::D_QK;
  static constexpr int SCALE_GROUP = Scales::GROUP;
  static constexpr int NUM_SCALES = DIMS / SCALE_GROUP;
};

}  // namespace flashinfer::sparse_mla_sm120
