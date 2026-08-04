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
#ifndef FLASHINFER_FASTDIV_CUH_
#define FLASHINFER_FASTDIV_CUH_
#include <cstdint>

// CUDA 12 and earlier: cuda::fast_mod_div is available in <cuda/cmath>.
// CUDA 13+ renamed it to cub::detail::fast_div_mod (internal header).
// We use the public CUDA 12 API when available and fall back to plain division otherwise.
#if (__CUDACC_VER_MAJOR__ < 13)
#include <cuda/cmath>
#endif

namespace flashinfer {

#if (__CUDACC_VER_MAJOR__ < 13)
// CUDA 12 path: use cuda::fast_mod_div for hardware-accelerated division.
struct uint_fastdiv {
  __host__ __device__ uint_fastdiv() : impl_(1), d_(0) {}

  __host__ uint_fastdiv(uint32_t d) : impl_(d ? d : 1), d_(d) {}

  __host__ __device__ __forceinline__ operator unsigned int() const { return d_; }

  __host__ __device__ __forceinline__ void divmod(uint32_t n, uint32_t& q, uint32_t& r) const {
    q = n / impl_;
    r = n - q * d_;
  }

 private:
  cuda::fast_mod_div<uint32_t> impl_;
  uint32_t d_;
};
#else
// CUDA 13+ path: cuda::fast_mod_div was removed. Fall back to plain division.
struct uint_fastdiv {
  __host__ __device__ uint_fastdiv() : impl_(1), d_(0) {}

  __host__ __device__ uint_fastdiv(uint32_t d) : impl_(d ? d : 1), d_(d) {}

  __host__ __device__ __forceinline__ operator unsigned int() const { return d_; }

  __host__ __device__ __forceinline__ void divmod(uint32_t n, uint32_t& q, uint32_t& r) const {
    q = n / impl_;
    r = n - q * d_;
  }

 private:
  uint32_t impl_;
  uint32_t d_;
};
#endif

__host__ __device__ __forceinline__ uint32_t operator/(const uint32_t n,
                                                       const uint_fastdiv& divisor) {
  uint32_t q, r;
  divisor.divmod(n, q, r);
  return q;
}

__host__ __device__ __forceinline__ uint32_t operator%(const uint32_t n,
                                                       const uint_fastdiv& divisor) {
  uint32_t q, r;
  divisor.divmod(n, q, r);
  return r;
}

}  // namespace flashinfer

#endif  // FLASHINFER_FASTDIV_CUH_
