// Copyright (c) 2026 by FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace wrapper_detail {
// Fresh status, never a sticky error. Integer comparisons intentionally avoid
// FTZ: all positive IEEE encodings through +inf compare equal iff bit-equal.
static __global__ void Check(const float* weights, int64_t t, const float* scale, int64_t n,
                             int* status) {
  bool bad_scale = false, bad_weights = false;
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x)
    bad_scale |= __float_as_uint(scale[i]) != 0x3f800000u;
  for (int64_t i = threadIdx.x; i < 4 * t; i += blockDim.x) {
    unsigned v = __float_as_uint(weights[i]);
    bad_weights |= v == 0 || v > 0x7f800000u || v != __float_as_uint(weights[i - i % 4]);
  }
  int s = __syncthreads_or(bad_scale);
  int w = __syncthreads_or(bad_weights);
  if (!threadIdx.x) *status = (s ? 1 : 0) | (w ? 2 : 0);
}

template <int Width, int K>
static __global__ void Unpack(const int* packed, const int* visible, const int* status,
                              int* repair_error, int* out, int64_t t, int64_t rows) {
  bool valid = *status == 0;
  if (!valid && !blockIdx.x && !threadIdx.x) atomicOr(repair_error, 16);
  for (int64_t r = blockIdx.x; r < rows; r += gridDim.x) {
    int count = valid && r < t ? visible[r] : 0;
    for (int j = threadIdx.x; j < K; j += blockDim.x)
      out[r * K + j] = j < count ? packed[r * Width + 4 * j] >> 2 : -1;
  }
}
}  // namespace wrapper_detail
