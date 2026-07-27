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
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include "cute_sm120_mxfp8_groupwise/cute_sm120_mxfp8_runner.h"
#include "cutlass/gemm_coord.h"
#include "tvm_ffi_utils.h"
#include "cute_sm120_mxfp8_groupwise/sm120_blockscaled/builder.cuh"
#include "cute_sm120_mxfp8_groupwise/sm120_blockscaled/kernel_impl.cuh"
#include "cute_sm120_mxfp8_groupwise/sm120_blockscaled/launch.cuh"
// clang-format on

// 2-value granK dispatch. Validates input and instantiates the templated
// callsite for each supported `granK` value. The else branch defends against
// bypasses of the caller-side validators.
#define DISPATCH_GRAN_K(granK, GRAN_K, ...)             \
  if ((granK) == 32) {                                  \
    constexpr int GRAN_K = 32;                          \
    __VA_ARGS__;                                        \
  } else if ((granK) == 128) {                          \
    constexpr int GRAN_K = 128;                         \
    __VA_ARGS__;                                        \
  } else {                                              \
    TVM_FFI_ICHECK(false) << "granK must be 32 or 128"; \
  }

namespace flashinfer::gemm::mxfp8_cute_sm120 {

template <typename ElementType, typename OutElementType, typename AccumElementType,
          typename BlockScaleElementType>
CuteSm120Mxfp8GemmRunner<ElementType, OutElementType, AccumElementType,
                         BlockScaleElementType>::CuteSm120Mxfp8GemmRunner() {}

template <typename ElementType, typename OutElementType, typename AccumElementType,
          typename BlockScaleElementType>
CuteSm120Mxfp8GemmRunner<ElementType, OutElementType, AccumElementType,
                         BlockScaleElementType>::~CuteSm120Mxfp8GemmRunner() {}

template <typename ElementType, typename OutElementType, typename AccumElementType,
          typename BlockScaleElementType>
void CuteSm120Mxfp8GemmRunner<ElementType, OutElementType, AccumElementType,
                              BlockScaleElementType>::gemm_mxfp8_nt_groupwise(void* D,
                                                                              void const* A,
                                                                              void const* B,
                                                                              int shape_m,
                                                                              int shape_n,
                                                                              int shape_k,
                                                                              int32_t const* SFA,
                                                                              int32_t const* SFB,
                                                                              cudaStream_t stream,
                                                                              int granK) {
  DISPATCH_GRAN_K(granK, GRAN_K, {
    gemm_mxfp8_nt_groupwise_impl<GRAN_K>(D, A, B, shape_m, shape_n, shape_k, SFA, SFB, stream);
  })
}

template <typename ElementType, typename OutElementType, typename AccumElementType,
          typename BlockScaleElementType>
template <int GranK>
void CuteSm120Mxfp8GemmRunner<
    ElementType, OutElementType, AccumElementType,
    BlockScaleElementType>::gemm_mxfp8_nt_groupwise_impl(void* D, void const* A, void const* B,
                                                         int shape_m, int shape_n, int shape_k,
                                                         int32_t const* SFA, int32_t const* SFB,
                                                         cudaStream_t stream) {
  using KT_M128 = sm120_blockscaled::SM120BlockScaledBuilder<128, 128, 64, 4, GranK,
                                                             sm120_common::GemmType::Normal>;
  using KT_M64 = sm120_blockscaled::SM120BlockScaledBuilder<64, 128, 64, 4, GranK,
                                                            sm120_common::GemmType::Normal>;
  using KT_M32 = sm120_blockscaled::SM120BlockScaledBuilder<32, 128, 128, 4, GranK,
                                                            sm120_common::GemmType::Normal>;
  using KT_SWAPAB_N8 =
      sm120_blockscaled::SM120BlockScaledBuilder<128, 8, 128, 4, GranK,
                                                 sm120_common::GemmType::Normal, true>;

  auto ptr_A_in = reinterpret_cast<typename KT_M128::ElementA*>(const_cast<void*>(A));
  auto ptr_B_in = reinterpret_cast<typename KT_M128::ElementB*>(const_cast<void*>(B));
  auto ptr_SFA_in =
      reinterpret_cast<typename KT_M128::SFConfig::ElementSFLoad*>(const_cast<int32_t*>(SFA));
  auto ptr_SFB_in =
      reinterpret_cast<typename KT_M128::SFConfig::ElementSFLoad*>(const_cast<int32_t*>(SFB));
  auto ptr_D_in = reinterpret_cast<typename KT_M128::ElementD*>(D);

  int num_sms = sm120_blockscaled::get_num_sms();

  if (shape_m <= 8) {
    sm120_blockscaled::launch_gemm<KT_SWAPAB_N8>(ptr_A_in, ptr_B_in, ptr_SFA_in, ptr_SFB_in,
                                                 ptr_D_in, shape_m, shape_n, shape_k, num_sms,
                                                 stream);
    return;
  }

  int min_tiles = num_sms / 2;

  auto can_use = [&](int tile_m, int tile_n) {
    int num_m = (shape_m + tile_m - 1) / tile_m;
    int num_n = (shape_n + tile_n - 1) / tile_n;
    int tiles = num_m * num_n;
    return shape_m >= tile_m && num_m >= 2 && tiles >= min_tiles;
  };

  if (can_use(KT_M128::kTileM, KT_M128::kTileN)) {
    sm120_blockscaled::launch_gemm<KT_M128>(ptr_A_in, ptr_B_in, ptr_SFA_in, ptr_SFB_in, ptr_D_in,
                                            shape_m, shape_n, shape_k, num_sms, stream);
  } else if (can_use(KT_M64::kTileM, KT_M64::kTileN)) {
    sm120_blockscaled::launch_gemm<KT_M64>(ptr_A_in, ptr_B_in, ptr_SFA_in, ptr_SFB_in, ptr_D_in,
                                           shape_m, shape_n, shape_k, num_sms, stream);
  } else {
    sm120_blockscaled::launch_gemm<KT_M32>(ptr_A_in, ptr_B_in, ptr_SFA_in, ptr_SFB_in, ptr_D_in,
                                           shape_m, shape_n, shape_k, num_sms, stream);
  }
}

template <typename ElementType, typename OutElementType, typename AccumElementType,
          typename BlockScaleElementType>
void CuteSm120Mxfp8GemmRunner<
    ElementType, OutElementType, AccumElementType,
    BlockScaleElementType>::batch_gemm_mxfp8_nt_groupwise(void* A, int ld_a, int stride_a, void* B,
                                                          int ld_b, int stride_b, void* D, int ld_d,
                                                          int stride_d, int32_t* SFA, int32_t* SFB,
                                                          int num_groups, int shape_m, int shape_n,
                                                          int shape_k, cudaStream_t stream,
                                                          int granK) {
  DISPATCH_GRAN_K(granK, GRAN_K, {
    batch_gemm_mxfp8_nt_groupwise_impl<GRAN_K>(A, ld_a, stride_a, B, ld_b, stride_b, D, ld_d,
                                               stride_d, SFA, SFB, num_groups, shape_m, shape_n,
                                               shape_k, stream);
  })
}

template <typename ElementType, typename OutElementType, typename AccumElementType,
          typename BlockScaleElementType>
template <int GranK>
void CuteSm120Mxfp8GemmRunner<
    ElementType, OutElementType, AccumElementType,
    BlockScaleElementType>::batch_gemm_mxfp8_nt_groupwise_impl(void* A, int ld_a, int stride_a,
                                                               void* B, int ld_b, int stride_b,
                                                               void* D, int ld_d, int stride_d,
                                                               int32_t* SFA, int32_t* SFB,
                                                               int num_groups, int shape_m,
                                                               int shape_n, int shape_k,
                                                               cudaStream_t stream) {
  using KT_M128 = sm120_blockscaled::SM120BlockScaledBuilder<128, 128, 64, 4, GranK,
                                                             sm120_common::GemmType::Batched>;
  using KT_M64 = sm120_blockscaled::SM120BlockScaledBuilder<64, 128, 64, 4, GranK,
                                                            sm120_common::GemmType::Batched>;
  using KT_M32 = sm120_blockscaled::SM120BlockScaledBuilder<32, 128, 128, 4, GranK,
                                                            sm120_common::GemmType::Batched>;
  using KT_SWAPAB_N8 =
      sm120_blockscaled::SM120BlockScaledBuilder<128, 8, 128, 4, GranK,
                                                 sm120_common::GemmType::Batched, true>;

  auto ptr_A_in = reinterpret_cast<typename KT_M128::ElementA*>(A);
  auto ptr_B_in = reinterpret_cast<typename KT_M128::ElementB*>(B);
  auto ptr_SFA_in = reinterpret_cast<typename KT_M128::SFConfig::ElementSFLoad*>(SFA);
  auto ptr_SFB_in = reinterpret_cast<typename KT_M128::SFConfig::ElementSFLoad*>(SFB);
  auto ptr_D_in = reinterpret_cast<typename KT_M128::ElementD*>(D);

  int num_sms = sm120_blockscaled::get_num_sms();

  if (shape_m <= 8) {
    sm120_blockscaled::launch_bmm<KT_SWAPAB_N8>(ptr_A_in, ptr_B_in, ptr_SFA_in, ptr_SFB_in,
                                                ptr_D_in, shape_m, shape_n, shape_k, num_groups,
                                                num_sms, stream);
    return;
  }

  int min_tiles = num_sms / 2;

  auto can_use = [&](int tile_m, int tile_n) {
    int num_m = (shape_m + tile_m - 1) / tile_m;
    int num_n = (shape_n + tile_n - 1) / tile_n;
    int tiles = num_m * num_n;
    return shape_m >= tile_m && num_m >= 2 && tiles >= min_tiles;
  };

  if (can_use(KT_M128::kTileM, KT_M128::kTileN)) {
    sm120_blockscaled::launch_bmm<KT_M128>(ptr_A_in, ptr_B_in, ptr_SFA_in, ptr_SFB_in, ptr_D_in,
                                           shape_m, shape_n, shape_k, num_groups, num_sms, stream);
  } else if (can_use(KT_M64::kTileM, KT_M64::kTileN)) {
    sm120_blockscaled::launch_bmm<KT_M64>(ptr_A_in, ptr_B_in, ptr_SFA_in, ptr_SFB_in, ptr_D_in,
                                          shape_m, shape_n, shape_k, num_groups, num_sms, stream);
  } else {
    sm120_blockscaled::launch_bmm<KT_M32>(ptr_A_in, ptr_B_in, ptr_SFA_in, ptr_SFB_in, ptr_D_in,
                                          shape_m, shape_n, shape_k, num_groups, num_sms, stream);
  }
}

template <typename ElementType, typename OutElementType, typename AccumElementType,
          typename BlockScaleElementType>
void CuteSm120Mxfp8GemmRunner<
    ElementType, OutElementType, AccumElementType,
    BlockScaleElementType>::group_gemm_mxfp8_nt_groupwise_masked(void* D, void const* A,
                                                                 void const* B, int const* masked_m,
                                                                 int num_groups, int max_m, int n,
                                                                 int k, cudaStream_t stream,
                                                                 int32_t const* SFA,
                                                                 int32_t const* SFB, int granK) {
  DISPATCH_GRAN_K(granK, GRAN_K, {
    group_gemm_mxfp8_nt_groupwise_masked_impl<GRAN_K>(D, A, B, masked_m, num_groups, max_m, n, k,
                                                      stream, SFA, SFB);
  })
}

template <typename ElementType, typename OutElementType, typename AccumElementType,
          typename BlockScaleElementType>
template <int GranK>
void CuteSm120Mxfp8GemmRunner<ElementType, OutElementType, AccumElementType,
                              BlockScaleElementType>::
    group_gemm_mxfp8_nt_groupwise_masked_impl(void* D, void const* A, void const* B,
                                              int const* masked_m, int num_groups, int max_m, int n,
                                              int k, cudaStream_t stream, int32_t const* SFA,
                                              int32_t const* SFB) {
  using KT_MASKED =
      sm120_blockscaled::SM120BlockScaledBuilder<64, 128, 64, 4, GranK,
                                                 sm120_common::GemmType::MGroupedMasked>;

  auto ptr_A_in = reinterpret_cast<typename KT_MASKED::ElementA*>(const_cast<void*>(A));
  auto ptr_B_in = reinterpret_cast<typename KT_MASKED::ElementB*>(const_cast<void*>(B));
  auto ptr_SFA_in =
      reinterpret_cast<typename KT_MASKED::SFConfig::ElementSFLoad*>(const_cast<int32_t*>(SFA));
  auto ptr_SFB_in =
      reinterpret_cast<typename KT_MASKED::SFConfig::ElementSFLoad*>(const_cast<int32_t*>(SFB));
  auto ptr_D_in = reinterpret_cast<typename KT_MASKED::ElementD*>(D);

  int num_sms = sm120_blockscaled::get_num_sms();

  sm120_blockscaled::launch_masked_gemm<KT_MASKED>(ptr_A_in, ptr_B_in, ptr_SFA_in, ptr_SFB_in,
                                                   ptr_D_in, max_m, n, k, num_groups, masked_m,
                                                   num_sms, stream);
}

template <typename ElementType, typename OutElementType, typename AccumElementType,
          typename BlockScaleElementType>
void CuteSm120Mxfp8GemmRunner<
    ElementType, OutElementType, AccumElementType,
    BlockScaleElementType>::moe_gemm_mxfp8_nt_groupwise(void* D, void const* A, void const* B,
                                                        int32_t const* token_offset,
                                                        int num_experts, int total_rows,
                                                        int shape_n, int shape_k,
                                                        cudaStream_t stream, int32_t const* SFA,
                                                        int32_t const* SFB, int granK) {
  DISPATCH_GRAN_K(granK, GRAN_K, {
    moe_gemm_mxfp8_nt_groupwise_impl<GRAN_K>(D, A, B, token_offset, num_experts, total_rows,
                                             shape_n, shape_k, stream, SFA, SFB);
  })
}

template <typename ElementType, typename OutElementType, typename AccumElementType,
          typename BlockScaleElementType>
template <int GranK>
void CuteSm120Mxfp8GemmRunner<
    ElementType, OutElementType, AccumElementType,
    BlockScaleElementType>::moe_gemm_mxfp8_nt_groupwise_impl(void* D, void const* A, void const* B,
                                                             int32_t const* token_offset,
                                                             int num_experts, int total_rows,
                                                             int shape_n, int shape_k,
                                                             cudaStream_t stream,
                                                             int32_t const* SFA,
                                                             int32_t const* SFB) {
  constexpr auto kGT = sm120_common::GemmType::MGroupedContiguousWithZeroPadding;
  constexpr int kTileK_M64 = (GranK == 32) ? 64 : 128;
  using KT_M32 = sm120_blockscaled::SM120BlockScaledBuilder<32, 128, 128, 4, GranK, kGT>;
  using KT_M64 = sm120_blockscaled::SM120BlockScaledBuilder<64, 128, kTileK_M64, 4, GranK, kGT>;
  using KT_M128 = sm120_blockscaled::SM120BlockScaledBuilder<128, 128, 64, 4, GranK, kGT>;
  using KT_SWAPAB_N8 =
      sm120_blockscaled::SM120BlockScaledBuilder<128, 8, 128, 4, GranK, kGT, /*SwapAB=*/true>;

  auto ptr_A_in = reinterpret_cast<typename KT_M128::ElementA*>(const_cast<void*>(A));
  auto ptr_B_in = reinterpret_cast<typename KT_M128::ElementB*>(const_cast<void*>(B));
  auto ptr_SFA_in =
      reinterpret_cast<typename KT_M128::SFConfig::ElementSFLoad*>(const_cast<int32_t*>(SFA));
  auto ptr_SFB_in =
      reinterpret_cast<typename KT_M128::SFConfig::ElementSFLoad*>(const_cast<int32_t*>(SFB));
  auto ptr_D_in = reinterpret_cast<typename KT_M128::ElementD*>(D);

  int num_sms = sm120_blockscaled::get_num_sms();
  int m_per_expert = num_experts > 0 ? (total_rows / num_experts) : 0;

  if (m_per_expert <= 12) {
    sm120_blockscaled::launch_moe_gemm<KT_SWAPAB_N8>(ptr_A_in, ptr_B_in, ptr_SFA_in, ptr_SFB_in,
                                                     ptr_D_in, total_rows, shape_n, shape_k,
                                                     num_experts, token_offset, num_sms, stream);
  } else if (m_per_expert <= 32) {
    sm120_blockscaled::launch_moe_gemm<KT_M32>(ptr_A_in, ptr_B_in, ptr_SFA_in, ptr_SFB_in, ptr_D_in,
                                               total_rows, shape_n, shape_k, num_experts,
                                               token_offset, num_sms, stream);
  } else if (m_per_expert < 96 || (m_per_expert < 192 && shape_k <= 2048)) {
    sm120_blockscaled::launch_moe_gemm<KT_M64>(ptr_A_in, ptr_B_in, ptr_SFA_in, ptr_SFB_in, ptr_D_in,
                                               total_rows, shape_n, shape_k, num_experts,
                                               token_offset, num_sms, stream);
  } else {
    sm120_blockscaled::launch_moe_gemm<KT_M128>(ptr_A_in, ptr_B_in, ptr_SFA_in, ptr_SFB_in,
                                                ptr_D_in, total_rows, shape_n, shape_k, num_experts,
                                                token_offset, num_sms, stream);
  }
}

template <typename ElementType, typename OutElementType, typename AccumElementType,
          typename BlockScaleElementType>
void CuteSm120Mxfp8GemmRunner<ElementType, OutElementType, AccumElementType,
                              BlockScaleElementType>::
    group_gemm_mxfp8_nt_groupwise_contiguous(void* D, void const* A, void const* B,
                                             int32_t const* m_indices, int num_groups, int m,
                                             int shape_n, int shape_k, cudaStream_t stream,
                                             int32_t const* SFA, int32_t const* SFB, int granK,
                                             bool use_psum_layout) {
  DISPATCH_GRAN_K(granK, GRAN_K, {
    group_gemm_mxfp8_nt_groupwise_contiguous_impl<GRAN_K>(
        D, A, B, m_indices, num_groups, m, shape_n, shape_k, stream, SFA, SFB, use_psum_layout);
  })
}

template <typename ElementType, typename OutElementType, typename AccumElementType,
          typename BlockScaleElementType>
template <int GranK>
void CuteSm120Mxfp8GemmRunner<ElementType, OutElementType, AccumElementType,
                              BlockScaleElementType>::
    group_gemm_mxfp8_nt_groupwise_contiguous_impl(void* D, void const* A, void const* B,
                                                  int32_t const* m_indices, int num_groups, int m,
                                                  int shape_n, int shape_k, cudaStream_t stream,
                                                  int32_t const* SFA, int32_t const* SFB,
                                                  bool use_psum_layout) {
  constexpr int kTileK_M64 = (GranK == 32) ? 64 : 128;
  int num_sms = sm120_blockscaled::get_num_sms();
  int m_per_expert = num_groups > 0 ? (m / num_groups) : 0;

  if (use_psum_layout) {
    constexpr auto kGT = sm120_common::GemmType::MGroupedContiguousWithPsumLayout;
    using KT_M32 = sm120_blockscaled::SM120BlockScaledBuilder<32, 128, 128, 4, GranK, kGT>;
    using KT_M64 = sm120_blockscaled::SM120BlockScaledBuilder<64, 128, kTileK_M64, 4, GranK, kGT>;
    using KT_M128 = sm120_blockscaled::SM120BlockScaledBuilder<128, 128, 64, 4, GranK, kGT>;
    using KT_SWAPAB_N8 =
        sm120_blockscaled::SM120BlockScaledBuilder<128, 8, 128, 4, GranK, kGT, /*SwapAB=*/true>;

    auto ptr_A_in = reinterpret_cast<typename KT_M128::ElementA*>(const_cast<void*>(A));
    auto ptr_B_in = reinterpret_cast<typename KT_M128::ElementB*>(const_cast<void*>(B));
    auto ptr_SFA_in =
        reinterpret_cast<typename KT_M128::SFConfig::ElementSFLoad*>(const_cast<int32_t*>(SFA));
    auto ptr_SFB_in =
        reinterpret_cast<typename KT_M128::SFConfig::ElementSFLoad*>(const_cast<int32_t*>(SFB));
    auto ptr_D_in = reinterpret_cast<typename KT_M128::ElementD*>(D);

    if (m_per_expert <= 12) {
      sm120_blockscaled::launch_moe_gemm<KT_SWAPAB_N8>(ptr_A_in, ptr_B_in, ptr_SFA_in, ptr_SFB_in,
                                                       ptr_D_in, m, shape_n, shape_k, num_groups,
                                                       m_indices, num_sms, stream);
    } else if (m_per_expert <= 32) {
      sm120_blockscaled::launch_moe_gemm<KT_M32>(ptr_A_in, ptr_B_in, ptr_SFA_in, ptr_SFB_in,
                                                 ptr_D_in, m, shape_n, shape_k, num_groups,
                                                 m_indices, num_sms, stream);
    } else if (m_per_expert < 96 || (m_per_expert < 192 && shape_k <= 2048)) {
      sm120_blockscaled::launch_moe_gemm<KT_M64>(ptr_A_in, ptr_B_in, ptr_SFA_in, ptr_SFB_in,
                                                 ptr_D_in, m, shape_n, shape_k, num_groups,
                                                 m_indices, num_sms, stream);
    } else {
      sm120_blockscaled::launch_moe_gemm<KT_M128>(ptr_A_in, ptr_B_in, ptr_SFA_in, ptr_SFB_in,
                                                  ptr_D_in, m, shape_n, shape_k, num_groups,
                                                  m_indices, num_sms, stream);
    }
  } else {
    constexpr auto kGT = sm120_common::GemmType::MGroupedContiguous;
    using KT_M32 = sm120_blockscaled::SM120BlockScaledBuilder<32, 128, 128, 4, GranK, kGT>;
    using KT_M64 = sm120_blockscaled::SM120BlockScaledBuilder<64, 128, kTileK_M64, 4, GranK, kGT>;
    using KT_M128 = sm120_blockscaled::SM120BlockScaledBuilder<128, 128, 64, 4, GranK, kGT>;
    using KT_SWAPAB_N8 =
        sm120_blockscaled::SM120BlockScaledBuilder<128, 8, 128, 4, GranK, kGT, /*SwapAB=*/true>;

    auto ptr_A_in = reinterpret_cast<typename KT_M128::ElementA*>(const_cast<void*>(A));
    auto ptr_B_in = reinterpret_cast<typename KT_M128::ElementB*>(const_cast<void*>(B));
    auto ptr_SFA_in =
        reinterpret_cast<typename KT_M128::SFConfig::ElementSFLoad*>(const_cast<int32_t*>(SFA));
    auto ptr_SFB_in =
        reinterpret_cast<typename KT_M128::SFConfig::ElementSFLoad*>(const_cast<int32_t*>(SFB));
    auto ptr_D_in = reinterpret_cast<typename KT_M128::ElementD*>(D);

    if (m_per_expert <= 12) {
      sm120_blockscaled::launch_moe_gemm<KT_SWAPAB_N8>(ptr_A_in, ptr_B_in, ptr_SFA_in, ptr_SFB_in,
                                                       ptr_D_in, m, shape_n, shape_k, num_groups,
                                                       m_indices, num_sms, stream);
    } else if (m_per_expert <= 32) {
      sm120_blockscaled::launch_moe_gemm<KT_M32>(ptr_A_in, ptr_B_in, ptr_SFA_in, ptr_SFB_in,
                                                 ptr_D_in, m, shape_n, shape_k, num_groups,
                                                 m_indices, num_sms, stream);
    } else if (m_per_expert < 96 || (m_per_expert < 192 && shape_k <= 2048)) {
      sm120_blockscaled::launch_moe_gemm<KT_M64>(ptr_A_in, ptr_B_in, ptr_SFA_in, ptr_SFB_in,
                                                 ptr_D_in, m, shape_n, shape_k, num_groups,
                                                 m_indices, num_sms, stream);
    } else {
      sm120_blockscaled::launch_moe_gemm<KT_M128>(ptr_A_in, ptr_B_in, ptr_SFA_in, ptr_SFB_in,
                                                  ptr_D_in, m, shape_n, shape_k, num_groups,
                                                  m_indices, num_sms, stream);
    }
  }
}

template class CuteSm120Mxfp8GemmRunner<cute::float_e4m3_t, cute::bfloat16_t, float,
                                        cute::float_ue8m0_t>;

}  // namespace flashinfer::gemm::mxfp8_cute_sm120
