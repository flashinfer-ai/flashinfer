/*
 * Copyright (c) 2026 by FlashInfer team.
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

#pragma once

#include "kv_cache_traits.cuh"

namespace flashinfer::sparse_mla_sm120::nvfp4 {

template <ModelType MT>
struct NVFP4CacheTraits;

template <>
struct NVFP4CacheTraits<ModelType::DSV4> {
  using Model = KVCacheTraits<ModelType::DSV4>;

  static constexpr int D_NOPE = Model::D_NOPE;
  static constexpr int D_ROPE = Model::D_ROPE;
  static constexpr int D_QK = Model::D_QK;
  static constexpr int D_V = Model::D_V;

  // Per-token paged-cache ABI:
  //   [page_size * 352B data][page_size * 32B E4M3 scale footer]
  // Each data row contains 224B packed E2M1 NoPE followed by 128B BF16 RoPE.
  static constexpr int SCALE_GROUP_SIZE = 16;
  static constexpr int NUM_SCALES = D_NOPE / SCALE_GROUP_SIZE;
  static constexpr int PACKED_NOPE_BYTES = D_NOPE / 2;
  static constexpr int ROPE_BYTES = D_ROPE * sizeof(bf16);
  static constexpr int DATA_BYTES_PER_TOKEN = PACKED_NOPE_BYTES + ROPE_BYTES;
  static constexpr int SCALE_BYTES_PER_TOKEN = (NUM_SCALES + 15) / 16 * 16;
  static constexpr int BYTES_PER_TOKEN = DATA_BYTES_PER_TOKEN + SCALE_BYTES_PER_TOKEN;

  // Shared-memory Q/K rows retain padding needed by ldmatrix and bulk copies.
  static constexpr int KV_SMEM_STRIDE = PACKED_NOPE_BYTES + 16;
  static constexpr int Q_PACKED_STRIDE = PACKED_NOPE_BYTES + 16;
  static constexpr int Q_SCALE_STRIDE = SCALE_BYTES_PER_TOKEN;
};

static_assert(NVFP4CacheTraits<ModelType::DSV4>::NUM_SCALES == 28);
static_assert(NVFP4CacheTraits<ModelType::DSV4>::DATA_BYTES_PER_TOKEN == 352);
static_assert(NVFP4CacheTraits<ModelType::DSV4>::BYTES_PER_TOKEN == 384);

}  // namespace flashinfer::sparse_mla_sm120::nvfp4
