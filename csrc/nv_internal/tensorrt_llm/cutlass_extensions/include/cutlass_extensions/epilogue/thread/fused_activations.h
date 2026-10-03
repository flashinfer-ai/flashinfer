/*
 * Copyright (c) 2017-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

/*! \file
  \brief Functor performing linear combination with a maximum operation used by epilogues.
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_generic.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/functional.h"
#include "cutlass/half.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

__forceinline__ __device__ float copysignf_pos(float a, float b) {
  float r;
  r = __int_as_float(__float_as_int(a) | (__float_as_int(b) & 0x80000000));
  return r;
}

__forceinline__ __device__ float tanh_opt(float x) {
#if (__CUDACC_VER_MAJOR__ < 11) || (__CUDA_ARCH__ < 750)
  float const exp_val = -1.f * fabs(2 * x);
  return copysignf_pos((1.0f - __expf(exp_val)) / (__expf(exp_val) + 1.0f), x);
#else
  return fast_tanh(x);
#endif
}

template <typename T>
struct Relu2 {
  static const bool kIsHeavy = false;

  CUTLASS_HOST_DEVICE
  T operator()(T threshold, T value) const {
    ReLu<T> relu_op;
    multiplies<T> mul;
    T val = relu_op(threshold, value);
    return mul(val, val);
  }

  CUTLASS_HOST_DEVICE
  T operator()(T value) const {
    ReLu<T> relu_op;
    multiplies<T> mul;
    T val = relu_op(value);
    return mul(val, val);
  }
};

// Tanh-soft-clamped squared-ReLU: out = [limit * tanh(relu(x) / limit)]^2.
//
// Implements megatron-core's clamped_squared_relu(x, clamp_scale) (see
// megatron/core/fusions/fused_weighted_squared_relu.py in NVIDIA/Megatron-LM).
// `limit` is the clamp scale and must be a finite, strictly positive value
// supplied as a model-wide scalar via ActivationParams::swiglu_limit (the
// field name is SwiGLU-specific in name only).
struct ClampedRelu2Arguments {
  float const* limit_ptr = nullptr;
};

template <typename T>
struct ClampedRelu2 {
  static const bool kIsHeavy = true;

  // Sm90Compute instantiates the activation on an Array<T, FragmentSize>, but
  // obtains Arguments from the scalar activation type. Keep the argument type
  // independent of T so both instantiations have the same function signature.
  using Arguments = ClampedRelu2Arguments;

  CUTLASS_HOST_DEVICE
  T operator()(T value, float limit) const {
    ReLu<T> relu_op;
    Tanh<T> tanh_op;
    // cutlass::Array<T, N> has no single-scalar constructor; multiply/divide
    // against a bare float directly via the native operator*, the same
    // mixed Array-times-float idiom already used for `gate_act * quant_scale`
    // in doActivationKernel below.
    T r = relu_op(value);
    T t = tanh_op(r * (1.0f / limit));
    T clamped = t * limit;
    return clamped * clamped;
  }

  CUTLASS_DEVICE
  T operator()(T value, Arguments const& args) const { return (*this)(value, args.limit_ptr[0]); }
};

}  // namespace thread
}  // namespace epilogue
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
