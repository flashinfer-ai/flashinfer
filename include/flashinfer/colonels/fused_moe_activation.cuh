#pragma once

#include "fused_moe_activation.h"
#include "fused_moe_helpers.cuh"

namespace moe_activation {

template <typename Activation>
__device__ __forceinline__ float apply(float x, float value, typename Activation::Params) {
  static_assert(Activation::KIND == Kind::SWIGLU);
  return swiglu_mul(x, value);
}

template <typename Activation>
__device__ __forceinline__ float apply(float x, typename Activation::Params params) {
  static_assert(!Activation::GATED);
  if constexpr (Activation::KIND == Kind::RELU_SQ) {
    return relu_sq(x);
  } else {
    static_assert(Activation::KIND == Kind::RELU_CLAMP_SQ);
    const float clamped = params.clamp * tanhf(fmaxf(x, 0.0f) / params.clamp);
    return clamped * clamped;
  }
}

}  // namespace moe_activation
