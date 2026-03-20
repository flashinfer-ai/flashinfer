/*
# SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
# ==============================================================================
*/
#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>

namespace batchedGemm {

namespace trtllm {
namespace dev {

////////////////////////////////////////////////////////////////////////////////////////////////////

// Conversion Utility to convert RMEM from one type to another. Used for conversion from AccumType
// to input/output type.
template <typename To_type, typename From_type, typename Fragment>
inline __device__ auto convert_type(Fragment const& tensor) {
  // The number of the elements in the source.
  constexpr int numel = decltype(size(tensor))::value;
  // The converter.
  cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
  // The data of the input.
  auto const* data = reinterpret_cast<const cutlass::Array<From_type, numel>*>(tensor.data());
  // Create the destination tensor (at least the array in registers). The src must be contiguous.
  auto dst = convert_op(*data);
  // Reconstruct the tensor.
  return cute::make_tensor(cute::make_rmem_ptr<To_type>(&dst), tensor.layout());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace dev
} // namespace trtllm

} // namespace batchedGemm
