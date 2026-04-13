/***************************************************************************************************
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>

#include <cute/tensor.hpp>

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

}  // namespace dev
}  // namespace trtllm
