/*
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Modified by the FlashInfer team.
 */
#ifndef FLASHINFER_ATTENTION_HOPPER_BLOCK_SPARSE_GATHER_CUH
#define FLASHINFER_ATTENTION_HOPPER_BLOCK_SPARSE_GATHER_CUH

#include <cstdint>

#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"
#include "cutlass/fast_math.h"

namespace flashinfer {

using namespace cute;

template <class IdType>
struct BlockSparseIndexedGather {
  CUTE_HOST_DEVICE constexpr BlockSparseIndexedGather(IdType const* indices) : indices_(indices) {}

  template <typename I>
  CUTE_HOST_DEVICE constexpr IdType operator()(I i) const {
    // NOTE(Zihao): there is a risk of out-of-bound access, adding boundary check here
    // would degrade performance significantly. It is the user's responsibility to ensure
    // that (indptr[-2] + TILE_KV) is less than the size of the indices tensor.
    return indices_[i];
  }

  CUTE_HOST_DEVICE friend void print(BlockSparseIndexedGather const& s) {
    cute::print("BlockSparseIndexedGather");
  }

  IdType const* indices_;
};

/// Custom stride object that applies a function followed by a stride
template <class Func>
struct CustomStride {
  CUTE_HOST_DEVICE constexpr CustomStride(Func const& func, int stride_n)
      : func_(func), stride_n_(stride_n) {}

  template <class I>
  CUTE_HOST_DEVICE friend auto operator*(I i, CustomStride const& s) {
    //     uint64_t ret;
    // #if defined(__CUDA_ARCH__)
    //     asm("{\n\t"
    //         "mul.wide.u32 %0, %1, %2;\n\t"
    //         "}" : "=l"(ret) : "r"(s.func_(i)), "r"(s.stride_n_));
    // #else
    //     ret = uint64_t(s.func_(i)) * uint64_t(s.stride_n_);
    // #endif
    //     return ret;

    // NOTE(Zihao): if the tensor is larger than 64GB ((2 ** 32) * 16byte), we use
    // 64-bit multiplication to avoid overflow. Otherwise, 32-bit multiplication is
    // sufficient.
    // There is a 20+ TFLOPs/s gap between 32-bit and 64-bit multiplication on H100.
    return uint32_t(s.func_(i)) * s.stride_n_;
  }

  template <class I>
  CUTE_HOST_DEVICE friend auto operator*(CustomStride const& s, I i) {
    //     uint64_t ret;
    // #if defined(__CUDA_ARCH__)
    //     asm("{\n\t"
    //         "mul.wide.u32 %0, %1, %2;\n\t"
    //         "}" : "=l"(ret) : "r"(s.func_(i)), "r"(s.stride_n_));
    // #else
    //     ret = uint64_t(s.func_(i)) * uint64_t(s.stride_n_);
    // #endif
    //     return ret;

    // NOTE(Zihao): if the tensor is larger than 64GB = (2 ** 32) * 16byte (16byte is the
    // element size after upcasting), we use 64-bit multiplication to avoid overflow. Otherwise,
    // 32-bit multiplication is sufficient.
    // There is a 20+ TFLOPs/s gap between 32-bit and 64-bit multiplication on H100.
    return uint32_t(s.func_(i)) * s.stride_n_;
  }

  CUTE_HOST_DEVICE friend void print(CustomStride const& s) {
    cute::print("BlockSparseStride{");
    print(s.func_);
    cute::print(",");
    print(s.stride_n_);
    cute::print("}");
  }

  template <class Div>
  CUTE_HOST_DEVICE constexpr friend auto safe_div(CustomStride const& s, Div const& div) {
    return CustomStride<Func>(s.func_, safe_div(s.stride_n_, div));
  }

  // Circumvent the requirement on make_layout that shape and stride are integral
  template <class Shape>
  CUTE_HOST_DEVICE constexpr friend auto make_layout(Shape const& shape,
                                                     CustomStride const& stride) {
    return Layout<Shape, CustomStride>(shape, stride);
  }

  Func func_;
  uint32_t stride_n_;
};

template <class Func>
CUTLASS_HOST_DEVICE auto make_custom_stride_layout(int stride_n, Func&& func) {
  return make_layout(make_shape(_1{}, _1{}),
                     make_stride(CustomStride(static_cast<Func&&>(func), stride_n), _1{}));
}

/// Helper function to optionally create a block sparse gather tensor
template <class Iterator, class Shape, class Func>
CUTLASS_HOST_DEVICE auto make_block_sparse_tensor(Iterator iter, Shape const& shape, int stride_n,
                                                  Func&& func) {
  Layout matrix_layout = make_identity_layout(shape);
  auto offset = as_arithmetic_tuple(repeat_like(shape, _0{}));
  Layout gather_layout = make_custom_stride_layout(stride_n, static_cast<Func&&>(func));

  return make_tensor(iter, ComposedLayout{gather_layout, offset, matrix_layout});
}

}  // namespace flashinfer

namespace cute {

template <int N, int I, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto upcast(Shape const& shape, Stride const& stride) {
  if constexpr (is_tuple<Shape>::value) {
    return transform_layout(shape, stride,
                            [](auto const& s, auto const& d) { return upcast<N, I>(s, d); });
  } else if constexpr (is_scaled_basis<Stride>::value) {
    if constexpr (Stride::mode() == I) {
      return make_layout(ceil_div(shape, Int<N>{}), ceil_div(stride, Int<N>{}));
    } else {
      return make_layout(shape, stride);
    }
  } else {
    return upcast<N>(shape, stride);
  }

  CUTE_GCC_UNREACHABLE;
}

template <int N, class OuterShape, class OuterStride, class Offset, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto upcast(
    ComposedLayout<Layout<OuterShape, OuterStride>, Offset, Layout<Shape, Stride>> const& layout) {
  // Find index of the stride-1 mode - that is the only one that requires updating inner shape and
  // offset
  auto idx =
      find_if(layout.layout_a().stride(), [](auto x) { return is_constant<1, decltype(x)>{}; });
  constexpr int I = decltype(idx)::value;

  // Upcast the outer layout (works as expected)
  auto outer = upcast<N>(layout.layout_a());

  // Upcast the accumulated offset along stride-1 mode
  auto offset =
      as_arithmetic_tuple(replace<I>(layout.offset(), upcast<N>(get<I>(layout.offset()))));

  // Upcast the inner layout's shape along stride-1 mode
  auto inner = upcast<N, I>(layout.layout_b().shape(), layout.layout_b().stride());

  return composition(outer, offset, inner);
}

}  // namespace cute

#endif  // FLASHINFER_ATTENTION_HOPPER_BLOCK_SPARSE_GATHER_CUH
