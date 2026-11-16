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
#ifndef FLASHINFER_FP8_KV_WIDEN_CUH_
#define FLASHINFER_FP8_KV_WIDEN_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

#include "vec_dtypes.cuh"

namespace flashinfer {
namespace fp8_kv_widen {

// One page is a contiguous block of page_elems values in both the NHD and HND
// layouts, so the gather only needs the page table and the page stride.
template <typename DTypeKV, typename DTypeO>
__device__ __forceinline__ void Widen16(const DTypeKV* src, DTypeO* dst) {
  const uint4 packed = *reinterpret_cast<const uint4*>(src);
  alignas(16) DTypeO out[16];
  vec_cast<DTypeO, DTypeKV>::template cast<16>(out, reinterpret_cast<const DTypeKV*>(&packed));
  *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(&out[0]);
  *reinterpret_cast<uint4*>(dst + 8) = *reinterpret_cast<const uint4*>(&out[8]);
}

template <typename DTypeKV, typename DTypeO>
__global__ void GatherWidenKernel(const DTypeKV* __restrict__ src, DTypeO* __restrict__ dst,
                                  const int32_t* __restrict__ page_indices, int64_t page_elems) {
  const int64_t page = blockIdx.x;
  const int64_t src_base = static_cast<int64_t>(page_indices[page]) * page_elems;
  const int64_t dst_base = page * page_elems;
  const int64_t num_vecs = page_elems / 16;
  for (int64_t i = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x; i < num_vecs;
       i += static_cast<int64_t>(gridDim.y) * blockDim.x) {
    Widen16(src + src_base + i * 16, dst + dst_base + i * 16);
  }
}

}  // namespace fp8_kv_widen
}  // namespace flashinfer

#endif  // FLASHINFER_FP8_KV_WIDEN_CUH_
