#pragma once

#include <cstdint>

namespace flashinfer::gemm::mxfp8_cute_sm120::sm120_moe_select {

inline int64_t ceil_div(int64_t x, int64_t y) { return x <= 0 ? 0 : (x + y - 1) / y; }

inline int balanced_max_rows(int total_rows, int num_experts) {
  return num_experts > 0 ? int(ceil_div(total_rows, num_experts)) : 0;
}

inline int64_t balanced_m_tiles(int total_rows, int num_experts, int tile_m) {
  if (num_experts <= 0) {
    return 0;
  }
  int q = total_rows / num_experts;
  int r = total_rows % num_experts;
  return int64_t(r) * ceil_div(q + 1, tile_m) + int64_t(num_experts - r) * ceil_div(q, tile_m);
}

inline int64_t balanced_tile_count(int total_rows, int shape_n, int num_experts, int tile_m,
                                   int tile_n) {
  return balanced_m_tiles(total_rows, num_experts, tile_m) * ceil_div(shape_n, tile_n);
}

inline int64_t wave_count(int64_t tile_count, int num_sms) { return ceil_div(tile_count, num_sms); }

inline int select_plain_m64_or_m128(int total_rows, int shape_n, int num_experts, int num_sms,
                                    int tile_n_m64 = 128, int tile_n_m128 = 128) {
  constexpr int kPlainTileOverhead = 48;
  auto cost = [&](int tile_m, int tile_n) {
    int64_t tiles = balanced_tile_count(total_rows, shape_n, num_experts, tile_m, tile_n);
    return wave_count(tiles, num_sms) * int64_t(tile_m + kPlainTileOverhead);
  };
  return (cost(64, tile_n_m64) < cost(128, tile_n_m128)) ? 64 : 128;
}

}  // namespace flashinfer::gemm::mxfp8_cute_sm120::sm120_moe_select
