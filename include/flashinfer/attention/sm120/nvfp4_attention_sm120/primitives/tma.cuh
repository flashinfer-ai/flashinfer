/*
 * Copyright (c) 2025 by SageAttention team.
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

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"

namespace nvfp4_attention {

using namespace cute;

template <typename TMADesc, typename TensorSrc, typename TensorDst>
__device__ __forceinline__ void tma_load(TMADesc const& tma_desc, void* barrier,
                                         uint16_t mcast_mask, TensorSrc const& src,
                                         TensorDst& dst) {
  cute::copy(tma_desc.with(barrier, mcast_mask), src, dst);
}

template <typename TMADesc, typename TensorSrc, typename TensorDst>
__device__ __forceinline__ void tma_store(TMADesc const& tma_desc, TensorSrc const& src,
                                          TensorDst& dst) {
  cute::copy(tma_desc, src, dst);
}

__device__ __forceinline__ void tma_store_arrive() {
  asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template <int N = 0>
__device__ __forceinline__ void tma_store_wait() {
  asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(N) : "memory");
}

template <typename TMADesc>
__device__ __forceinline__ void prefetch_tma_descriptor(TMADesc const& tma_desc) {
  cute::prefetch_tma_descriptor(tma_desc.get_tma_descriptor());
}

template <typename TMADesc>
struct TMALoader {
  TMADesc tma_desc_;

  __device__ __forceinline__ TMALoader(TMADesc const& tma_desc) : tma_desc_(tma_desc) {}

  __device__ __forceinline__ void prefetch() const {
    nvfp4_attention::prefetch_tma_descriptor(tma_desc_);
  }

  template <typename TensorSrc, typename TensorDst>
  __device__ __forceinline__ void load(void* barrier, uint16_t mcast_mask, TensorSrc const& src,
                                       TensorDst& dst) const {
    nvfp4_attention::tma_load(tma_desc_, barrier, mcast_mask, src, dst);
  }

  template <typename TensorSrc, typename TensorDst>
  __device__ __forceinline__ void load(void* barrier, TensorSrc const& src, TensorDst& dst) const {
    nvfp4_attention::tma_load(tma_desc_, barrier, 0, src, dst);
  }
};

template <typename TMADesc>
struct TMAStorer {
  TMADesc tma_desc_;

  __device__ __forceinline__ TMAStorer(TMADesc const& tma_desc) : tma_desc_(tma_desc) {}

  __device__ __forceinline__ void prefetch() const {
    nvfp4_attention::prefetch_tma_descriptor(tma_desc_);
  }

  template <typename TensorSrc, typename TensorDst>
  __device__ __forceinline__ void store(TensorSrc const& src, TensorDst& dst) const {
    nvfp4_attention::tma_store(tma_desc_, src, dst);
  }

  __device__ __forceinline__ void arrive() const { nvfp4_attention::tma_store_arrive(); }

  template <int N = 0>
  __device__ __forceinline__ void wait() const {
    nvfp4_attention::tma_store_wait<N>();
  }
};

template <typename TMADesc>
__device__ __forceinline__ auto make_tma_loader(TMADesc const& tma_desc) {
  return TMALoader<TMADesc>(tma_desc);
}

template <typename TMADesc>
__device__ __forceinline__ auto make_tma_storer(TMADesc const& tma_desc) {
  return TMAStorer<TMADesc>(tma_desc);
}

}  // namespace nvfp4_attention
