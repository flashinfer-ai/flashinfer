// Copyright (c) 2026 by FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../model/dsv4_nvfp4_layout.cuh"

namespace flashinfer::sparse_mla_sm120::nvfp4 {

using DSV4NVFP4Cache = Dsv4Nvfp4Layout;

constexpr int NVFP4_VT_CANDIDATES = 64;
constexpr int NVFP4_VT_PACKED_K_BYTES = NVFP4_VT_CANDIDATES / 2;
constexpr int NVFP4_VT_SCALE_GROUPS = NVFP4_VT_CANDIDATES / DSV4NVFP4Cache::Scales::GROUP;
constexpr int NVFP4_VT_DATA_BYTES = DSV4NVFP4Cache::D_NOPE * NVFP4_VT_PACKED_K_BYTES;
constexpr int NVFP4_VT_SCALE_BYTES = DSV4NVFP4Cache::D_NOPE * NVFP4_VT_SCALE_GROUPS;

}  // namespace flashinfer::sparse_mla_sm120::nvfp4
