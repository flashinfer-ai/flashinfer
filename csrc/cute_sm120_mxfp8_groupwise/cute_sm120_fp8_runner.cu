/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

#include "cute_sm120_mxfp8_groupwise/cute_sm120_fp8_runner.h"
#include "cute_sm120_mxfp8_groupwise/sm120_blockscaling/builder.cuh"
#include "cute_sm120_mxfp8_groupwise/sm120_blockscaling/launch.cuh"
// clang-format on

namespace flashinfer::gemm::mxfp8_cute_sm120 {

template <typename ElementType, typename OutElementType, typename AccumElementType,
          typename BlockScaleElementType>
CuteSm120Fp8GemmRunner<ElementType, OutElementType, AccumElementType,
                       BlockScaleElementType>::CuteSm120Fp8GemmRunner() {}

template <typename ElementType, typename OutElementType, typename AccumElementType,
          typename BlockScaleElementType>
CuteSm120Fp8GemmRunner<ElementType, OutElementType, AccumElementType,
                       BlockScaleElementType>::~CuteSm120Fp8GemmRunner() {}

static void check_scale_granularity_mnk(int scale_granularity_m, int scale_granularity_n,
                                        int scale_granularity_k) {
  if (scale_granularity_m != 1 || scale_granularity_n != 128 || scale_granularity_k != 128) {
    throw std::runtime_error("unsupported FP8 scale granularity");
  }
}

static int select_fp8_flat_tile_m(int shape_m, int shape_n, int num_groups, int num_sms) {
  auto tile_count = [&](int tile_m) {
    int64_t num_m = (int64_t(shape_m) + tile_m - 1) / tile_m;
    int64_t num_n = (int64_t(shape_n) + 128 - 1) / 128;
    return int64_t(num_groups) * num_m * num_n;
  };
  auto wave_count = [&](int tile_m) { return (tile_count(tile_m) + num_sms - 1) / num_sms; };
  auto tail_is_amortized = [&](int tile_m, int min_tail_tiles) {
    int64_t tiles = tile_count(tile_m);
    int64_t last_wave_tiles = (tiles - 1) % num_sms + 1;
    return tiles >= int64_t(2) * num_sms || last_wave_tiles >= min_tail_tiles;
  };

  auto reduces_waves = [&](int tile_m, int smaller_tile_m) {
    return wave_count(tile_m) < wave_count(smaller_tile_m);
  };
  auto critical_m_work = [&](int tile_m) { return int64_t(wave_count(tile_m)) * tile_m; };
  auto exact_boundary_dominates = [&](int tile_m) {
    return shape_m == tile_m && critical_m_work(tile_m) <= critical_m_work(tile_m / 2);
  };

  int64_t swapab_tiles = tile_count(8);
  int64_t m32_tiles = tile_count(32);
  bool underfilled_flat_work =
      int64_t(num_groups) * shape_m > 32 && m32_tiles < num_sms / 2 && swapab_tiles <= num_sms;
  bool bounded_small_m_work = shape_m <= 16 && int64_t(num_groups) * shape_m <= 64 &&
                              wave_count(8) < int64_t(2) * wave_count(32);
  if (shape_m <= 8 || underfilled_flat_work || bounded_small_m_work) {
    return 8;
  }

  int64_t m128_tiles = tile_count(128);
  if (m128_tiles >= num_sms / 2 && (shape_m >= 128 || m128_tiles >= int64_t(2) * num_sms) &&
      tail_is_amortized(128, num_sms / 2) && reduces_waves(128, 64) &&
      (shape_m != 128 || exact_boundary_dominates(128))) {
    return 128;
  }
  if (tile_count(64) >= num_sms / 2 && tail_is_amortized(64, num_sms / 2) &&
      reduces_waves(64, 32) && (shape_m != 64 || exact_boundary_dominates(64))) {
    return 64;
  }
  return 32;
}

template <typename ElementType, typename OutElementType, typename AccumElementType,
          typename BlockScaleElementType>
void CuteSm120Fp8GemmRunner<ElementType, OutElementType, AccumElementType, BlockScaleElementType>::
    gemm_fp8_nt_groupwise(void* D, void const* A, void const* B, int shape_m, int shape_n,
                          int shape_k, float const* SFA, float const* SFB, cudaStream_t stream,
                          int scale_granularity_m, int scale_granularity_n,
                          int scale_granularity_k) {
  check_scale_granularity_mnk(scale_granularity_m, scale_granularity_n, scale_granularity_k);
  gemm_fp8_nt_groupwise_impl(D, A, B, shape_m, shape_n, shape_k, SFA, SFB, stream);
}

template <typename ElementType, typename OutElementType, typename AccumElementType,
          typename BlockScaleElementType>
void CuteSm120Fp8GemmRunner<ElementType, OutElementType, AccumElementType, BlockScaleElementType>::
    gemm_fp8_nt_groupwise_impl(void* D, void const* A, void const* B, int shape_m, int shape_n,
                               int shape_k, float const* SFA, float const* SFB,
                               cudaStream_t stream) {
  constexpr auto kGT = sm120_common::GemmType::Normal;
  using KT_M128 = sm120_blockscaling::SM120BlockScalingBuilder<128, 128, 128, 2, 1, 128, 128, kGT>;
  using KT_M64 = sm120_blockscaling::SM120BlockScalingBuilder<64, 128, 128, 2, 1, 128, 128, kGT>;
  using KT_M32 = sm120_blockscaling::SM120BlockScalingBuilder<32, 128, 128, 2, 1, 128, 128, kGT>;
  using KT_SWAPAB_N8 =
      sm120_blockscaling::SM120BlockScalingBuilder<128, 8, 128, 2, 128, 1, 128, kGT, true>;

  auto ptr_A = reinterpret_cast<typename KT_M128::ElementA const*>(A);
  auto ptr_B = reinterpret_cast<typename KT_M128::ElementB const*>(B);
  auto ptr_SFA = reinterpret_cast<typename KT_M128::ElementScale const*>(SFA);
  auto ptr_SFB = reinterpret_cast<typename KT_M128::ElementScale const*>(SFB);
  auto ptr_D = reinterpret_cast<typename KT_M128::ElementD*>(D);

  int num_sms = sm120_blockscaling::get_num_sms();
  int tile_m = select_fp8_flat_tile_m(shape_m, shape_n, 1, num_sms);

  if (tile_m == KT_SWAPAB_N8::kTileN) {
    sm120_blockscaling::launch_gemm<KT_SWAPAB_N8>(ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_D, shape_m,
                                                  shape_n, shape_k, num_sms, stream);
    return;
  }

  if (tile_m == KT_M128::kTileM) {
    sm120_blockscaling::launch_gemm<KT_M128>(ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_D, shape_m,
                                             shape_n, shape_k, num_sms, stream);
  } else if (tile_m == KT_M64::kTileM) {
    sm120_blockscaling::launch_gemm<KT_M64>(ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_D, shape_m, shape_n,
                                            shape_k, num_sms, stream);
  } else {
    sm120_blockscaling::launch_gemm<KT_M32>(ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_D, shape_m, shape_n,
                                            shape_k, num_sms, stream);
  }
}

template <typename ElementType, typename OutElementType, typename AccumElementType,
          typename BlockScaleElementType>
void CuteSm120Fp8GemmRunner<ElementType, OutElementType, AccumElementType, BlockScaleElementType>::
    batch_gemm_fp8_nt_groupwise(void* D, void const* A, void const* B, int num_groups, int shape_m,
                                int shape_n, int shape_k, float const* SFA, float const* SFB,
                                cudaStream_t stream, int scale_granularity_m,
                                int scale_granularity_n, int scale_granularity_k) {
  check_scale_granularity_mnk(scale_granularity_m, scale_granularity_n, scale_granularity_k);
  batch_gemm_fp8_nt_groupwise_impl(D, A, B, num_groups, shape_m, shape_n, shape_k, SFA, SFB,
                                   stream);
}

template <typename ElementType, typename OutElementType, typename AccumElementType,
          typename BlockScaleElementType>
void CuteSm120Fp8GemmRunner<ElementType, OutElementType, AccumElementType, BlockScaleElementType>::
    batch_gemm_fp8_nt_groupwise_impl(void* D, void const* A, void const* B, int num_groups,
                                     int shape_m, int shape_n, int shape_k, float const* SFA,
                                     float const* SFB, cudaStream_t stream) {
  constexpr auto kGT = sm120_common::GemmType::Batched;
  using KT_M128 = sm120_blockscaling::SM120BlockScalingBuilder<128, 128, 128, 2, 1, 128, 128, kGT>;
  using KT_M64 = sm120_blockscaling::SM120BlockScalingBuilder<64, 128, 128, 2, 1, 128, 128, kGT>;
  using KT_M32 = sm120_blockscaling::SM120BlockScalingBuilder<32, 128, 128, 2, 1, 128, 128, kGT>;
  using KT_SWAPAB_N8 =
      sm120_blockscaling::SM120BlockScalingBuilder<128, 8, 128, 2, 128, 1, 128, kGT, true>;

  auto ptr_A = reinterpret_cast<typename KT_M128::ElementA const*>(A);
  auto ptr_B = reinterpret_cast<typename KT_M128::ElementB const*>(B);
  auto ptr_SFA = reinterpret_cast<typename KT_M128::ElementScale const*>(SFA);
  auto ptr_SFB = reinterpret_cast<typename KT_M128::ElementScale const*>(SFB);
  auto ptr_D = reinterpret_cast<typename KT_M128::ElementD*>(D);

  int num_sms = sm120_blockscaling::get_num_sms();
  int tile_m = select_fp8_flat_tile_m(shape_m, shape_n, num_groups, num_sms);

  if (tile_m == KT_SWAPAB_N8::kTileN) {
    sm120_blockscaling::launch_bmm<KT_SWAPAB_N8>(ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_D, shape_m,
                                                 shape_n, shape_k, num_groups, num_sms, stream);
    return;
  }

  if (tile_m == KT_M128::kTileM) {
    sm120_blockscaling::launch_bmm<KT_M128>(ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_D, shape_m, shape_n,
                                            shape_k, num_groups, num_sms, stream);
  } else if (tile_m == KT_M64::kTileM) {
    sm120_blockscaling::launch_bmm<KT_M64>(ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_D, shape_m, shape_n,
                                           shape_k, num_groups, num_sms, stream);
  } else {
    sm120_blockscaling::launch_bmm<KT_M32>(ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_D, shape_m, shape_n,
                                           shape_k, num_groups, num_sms, stream);
  }
}

template <typename ElementType, typename OutElementType, typename AccumElementType,
          typename BlockScaleElementType>
void CuteSm120Fp8GemmRunner<ElementType, OutElementType, AccumElementType, BlockScaleElementType>::
    group_gemm_fp8_nt_groupwise_masked(void* D, void const* A, void const* B,
                                       int32_t const* masked_m, int num_groups, int max_m, int n,
                                       int k, cudaStream_t stream, float const* SFA,
                                       float const* SFB, int scale_granularity_m,
                                       int scale_granularity_n, int scale_granularity_k) {
  check_scale_granularity_mnk(scale_granularity_m, scale_granularity_n, scale_granularity_k);
  group_gemm_fp8_nt_groupwise_masked_impl(D, A, B, masked_m, num_groups, max_m, n, k, stream, SFA,
                                          SFB);
}

template <typename ElementType, typename OutElementType, typename AccumElementType,
          typename BlockScaleElementType>
void CuteSm120Fp8GemmRunner<ElementType, OutElementType, AccumElementType, BlockScaleElementType>::
    group_gemm_fp8_nt_groupwise_masked_impl(void* D, void const* A, void const* B,
                                            int32_t const* masked_m, int num_groups, int max_m,
                                            int n, int k, cudaStream_t stream, float const* SFA,
                                            float const* SFB) {
  constexpr auto kGT = sm120_common::GemmType::MGroupedMasked;
  using KT_M128 = sm120_blockscaling::SM120BlockScalingBuilder<128, 128, 128, 2, 1, 128, 128, kGT>;
  using KT_M64 = sm120_blockscaling::SM120BlockScalingBuilder<64, 128, 128, 2, 1, 128, 128, kGT>;
  using KT_M32 = sm120_blockscaling::SM120BlockScalingBuilder<32, 128, 128, 2, 1, 128, 128, kGT>;
  using KT_SWAPAB_N8 =
      sm120_blockscaling::SM120BlockScalingBuilder<128, 8, 128, 2, 128, 1, 128, kGT, true>;

  auto ptr_A = reinterpret_cast<typename KT_M128::ElementA const*>(A);
  auto ptr_B = reinterpret_cast<typename KT_M128::ElementB const*>(B);
  auto ptr_SFA = reinterpret_cast<typename KT_M128::ElementScale const*>(SFA);
  auto ptr_SFB = reinterpret_cast<typename KT_M128::ElementScale const*>(SFB);
  auto ptr_D = reinterpret_cast<typename KT_M128::ElementD*>(D);

  int num_sms = sm120_blockscaling::get_num_sms();
  int tile_m = select_fp8_flat_tile_m(max_m, n, num_groups, num_sms);

  if (tile_m == KT_SWAPAB_N8::kTileN) {
    sm120_blockscaling::launch_masked_gemm<KT_SWAPAB_N8>(
        ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_D, max_m, n, k, num_groups, masked_m, num_sms, stream);
    return;
  }

  if (tile_m == KT_M128::kTileM) {
    sm120_blockscaling::launch_masked_gemm<KT_M128>(ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_D, max_m, n,
                                                    k, num_groups, masked_m, num_sms, stream);
  } else if (tile_m == KT_M64::kTileM) {
    sm120_blockscaling::launch_masked_gemm<KT_M64>(ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_D, max_m, n,
                                                   k, num_groups, masked_m, num_sms, stream);
  } else {
    sm120_blockscaling::launch_masked_gemm<KT_M32>(ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_D, max_m, n,
                                                   k, num_groups, masked_m, num_sms, stream);
  }
}

template <typename ElementType, typename OutElementType, typename AccumElementType,
          typename BlockScaleElementType>
void CuteSm120Fp8GemmRunner<ElementType, OutElementType, AccumElementType, BlockScaleElementType>::
    moe_gemm_fp8_nt_groupwise(void* D, void const* A, void const* B, int32_t const* token_offset,
                              int num_experts, int total_rows, int shape_n, int shape_k,
                              cudaStream_t stream, float const* SFA, float const* SFB,
                              int scale_granularity_m, int scale_granularity_n,
                              int scale_granularity_k) {
  check_scale_granularity_mnk(scale_granularity_m, scale_granularity_n, scale_granularity_k);
  moe_gemm_fp8_nt_groupwise_impl(D, A, B, token_offset, num_experts, total_rows, shape_n, shape_k,
                                 stream, SFA, SFB);
}

template <typename ElementType, typename OutElementType, typename AccumElementType,
          typename BlockScaleElementType>
void CuteSm120Fp8GemmRunner<ElementType, OutElementType, AccumElementType, BlockScaleElementType>::
    moe_gemm_fp8_nt_groupwise_impl(void* D, void const* A, void const* B,
                                   int32_t const* token_offset, int num_experts, int total_rows,
                                   int shape_n, int shape_k, cudaStream_t stream, float const* SFA,
                                   float const* SFB) {
  constexpr auto kGT = sm120_common::GemmType::MGroupedContiguousWithZeroPadding;
  using KT_M32 = sm120_blockscaling::SM120BlockScalingBuilder<32, 128, 128, 2, 1, 128, 128, kGT>;
  using KT_M64 = sm120_blockscaling::SM120BlockScalingBuilder<64, 128, 128, 2, 1, 128, 128, kGT>;
  using KT_M128 = sm120_blockscaling::SM120BlockScalingBuilder<128, 128, 128, 2, 1, 128, 128, kGT>;
  using KT_SWAPAB_N8 =
      sm120_blockscaling::SM120BlockScalingBuilder<128, 8, 128, 2, 128, 1, 128, kGT, true>;

  auto ptr_A = reinterpret_cast<typename KT_M128::ElementA const*>(A);
  auto ptr_B = reinterpret_cast<typename KT_M128::ElementB const*>(B);
  auto ptr_SFA = reinterpret_cast<typename KT_M128::ElementScale const*>(SFA);
  auto ptr_SFB = reinterpret_cast<typename KT_M128::ElementScale const*>(SFB);
  auto ptr_D = reinterpret_cast<typename KT_M128::ElementD*>(D);

  int num_sms = sm120_blockscaling::get_num_sms();
  int m_per_expert = num_experts > 0 ? (total_rows / num_experts) : 0;

  if (m_per_expert <= 8) {
    sm120_blockscaling::launch_moe_gemm<KT_SWAPAB_N8>(ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_D,
                                                      total_rows, shape_n, shape_k, num_experts,
                                                      token_offset, num_sms, stream);
  } else if (m_per_expert <= 32) {
    sm120_blockscaling::launch_moe_gemm<KT_M32>(ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_D, total_rows,
                                                shape_n, shape_k, num_experts, token_offset,
                                                num_sms, stream);
  } else if (m_per_expert <= 64) {
    sm120_blockscaling::launch_moe_gemm<KT_M64>(ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_D, total_rows,
                                                shape_n, shape_k, num_experts, token_offset,
                                                num_sms, stream);
  } else {
    sm120_blockscaling::launch_moe_gemm<KT_M128>(ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_D, total_rows,
                                                 shape_n, shape_k, num_experts, token_offset,
                                                 num_sms, stream);
  }
}

template <typename ElementType, typename OutElementType, typename AccumElementType,
          typename BlockScaleElementType>
void CuteSm120Fp8GemmRunner<ElementType, OutElementType, AccumElementType, BlockScaleElementType>::
    group_gemm_fp8_nt_groupwise_contiguous(void* D, void const* A, void const* B,
                                           int32_t const* m_indices, int num_groups, int m,
                                           int shape_n, int shape_k, cudaStream_t stream,
                                           float const* SFA, float const* SFB,
                                           int scale_granularity_m, int scale_granularity_n,
                                           int scale_granularity_k, bool use_psum_layout) {
  check_scale_granularity_mnk(scale_granularity_m, scale_granularity_n, scale_granularity_k);
  group_gemm_fp8_nt_groupwise_contiguous_impl(D, A, B, m_indices, num_groups, m, shape_n, shape_k,
                                              stream, SFA, SFB, use_psum_layout);
}

template <typename ElementType, typename OutElementType, typename AccumElementType,
          typename BlockScaleElementType>
void CuteSm120Fp8GemmRunner<ElementType, OutElementType, AccumElementType, BlockScaleElementType>::
    group_gemm_fp8_nt_groupwise_contiguous_impl(void* D, void const* A, void const* B,
                                                int32_t const* m_indices, int num_groups, int m,
                                                int shape_n, int shape_k, cudaStream_t stream,
                                                float const* SFA, float const* SFB,
                                                bool use_psum_layout) {
  int num_sms = sm120_blockscaling::get_num_sms();

  if (use_psum_layout) {
    constexpr auto kGT = sm120_common::GemmType::MGroupedContiguousWithPsumLayout;
    using KT_M128 =
        sm120_blockscaling::SM120BlockScalingBuilder<128, 128, 128, 2, 1, 128, 128, kGT>;

    auto ptr_A = reinterpret_cast<typename KT_M128::ElementA const*>(A);
    auto ptr_B = reinterpret_cast<typename KT_M128::ElementB const*>(B);
    auto ptr_SFA = reinterpret_cast<typename KT_M128::ElementScale const*>(SFA);
    auto ptr_SFB = reinterpret_cast<typename KT_M128::ElementScale const*>(SFB);
    auto ptr_D = reinterpret_cast<typename KT_M128::ElementD*>(D);

    sm120_blockscaling::launch_moe_gemm<KT_M128>(ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_D, m, shape_n,
                                                 shape_k, num_groups, m_indices, num_sms, stream);
  } else {
    constexpr auto kGT = sm120_common::GemmType::MGroupedContiguous;
    using KT_M128 =
        sm120_blockscaling::SM120BlockScalingBuilder<128, 128, 128, 2, 1, 128, 128, kGT>;

    auto ptr_A = reinterpret_cast<typename KT_M128::ElementA const*>(A);
    auto ptr_B = reinterpret_cast<typename KT_M128::ElementB const*>(B);
    auto ptr_SFA = reinterpret_cast<typename KT_M128::ElementScale const*>(SFA);
    auto ptr_SFB = reinterpret_cast<typename KT_M128::ElementScale const*>(SFB);
    auto ptr_D = reinterpret_cast<typename KT_M128::ElementD*>(D);

    sm120_blockscaling::launch_moe_gemm<KT_M128>(ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_D, m, shape_n,
                                                 shape_k, num_groups, m_indices, num_sms, stream);
  }
}

template class CuteSm120Fp8GemmRunner<cute::float_e4m3_t, cute::bfloat16_t, float, float>;

}  // namespace flashinfer::gemm::mxfp8_cute_sm120
