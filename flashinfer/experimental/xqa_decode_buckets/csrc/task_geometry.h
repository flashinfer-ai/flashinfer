// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

#ifdef __CUDACC__
#define XQA_BUCKET_HOST_DEVICE __host__ __device__
#else
#define XQA_BUCKET_HOST_DEVICE
#endif

namespace flashinfer::xqa_buckets {
/// Count 256-token tiles for a nonnegative live length using wide rounding arithmetic.
XQA_BUCKET_HOST_DEVICE constexpr int direct_tile_count(int length) {
  return int((int64_t(length) + 255) / 256);
}
/// Bound active single-request owners by both live tiles and the reserved CTA count.
XQA_BUCKET_HOST_DEVICE constexpr int direct_split_count(int length, int ctas) {
  int tiles = direct_tile_count(length);
  return tiles < ctas ? tiles : ctas;
}
/// Return a request-local split boundary; count is positive and ordinal is in [0, count].
XQA_BUCKET_HOST_DEVICE constexpr int aligned_boundary(int ordinal, int tiles, int count) {
  // Table capacity is independent of live length; ordinal*tiles can exceed int32.
  return int(int64_t(ordinal) * tiles / count);
}
/// Use one tile per direct owner when possible, otherwise evenly divide its live tiles.
XQA_BUCKET_HOST_DEVICE constexpr int direct_boundary(int ordinal, int tiles, int count) {
  // One tile per owner needs no runtime int64 division.
  return tiles == count ? ordinal : aligned_boundary(ordinal, tiles, count);
}
}  // namespace flashinfer::xqa_buckets
#undef XQA_BUCKET_HOST_DEVICE
