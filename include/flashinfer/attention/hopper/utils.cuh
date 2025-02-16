/*
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 * Dao. Licensed under the BSD 3-Clause.
 *
 * Modified by the FlashInfer team.
 */
#ifndef FLASHINFER_ATTENTION_HOPPER_UTILS_CUH_
#define FLASHINFER_ATTENTION_HOPPER_UTILS_CUH_

#include <assert.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <cuda_runtime.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <cmath>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>

#include "../../math.cuh"
#include "../../utils.cuh"
#include "cutlass/fast_math.h"

namespace flashinfer {

using namespace cute;

template <int CTA_Q, int CTA_KV>
CUTLASS_DEVICE int get_swa_begin_kv_tile_idx(int window_left, int q_tile_idx, const int qo_len,
                                             const int kv_len) {
  return std::max((q_tile_idx * CTA_Q + kv_len - qo_len - window_left) / CTA_KV - 1, 0);
}

template <int CTA_Q, int CTA_KV>
CUTLASS_DEVICE int get_swa_end_kv_tile_idx(int window_left, int q_tile_idx, const int qo_len,
                                           const int kv_len) {
  return std::max(((q_tile_idx + 1) * CTA_Q + kv_len - qo_len - window_left) / CTA_KV, -1);
}

template <typename TensorT>
CUTLASS_HOST_DEVICE auto flatten_1(TensorT tensor) {
  Tensor tensor_flatten = cute::flatten(tensor);
  return cute::group_modes<1, rank(tensor_flatten)>(tensor_flatten);
}

CUTLASS_HOST_DEVICE auto get_gmem_layout(int nnz, int num_heads, int head_dim, int64_t n_stride,
                                         int64_t h_stride) {
  return make_layout(make_shape(nnz, head_dim, num_heads),
                     make_stride(n_stride, cute::_1{}, h_stride));
}

CUTLASS_HOST_DEVICE auto get_lse_gmem_layout(int nnz, int num_heads) {
  return make_layout(make_shape(num_heads, nnz), make_stride(cute::_1{}, int64_t(num_heads)));
}

template <typename MTensor, typename Shape>
CUTLASS_DEVICE auto get_local_tile_tensor(const MTensor& m_tensor, const Shape& tile_shape,
                                          int head_idx, int offset, int seq_len) {
  auto g_offset = local_tile(m_tensor(_, _, head_idx), cute::make_shape(1, get<1>(tile_shape)),
                             make_coord(offset, _0{}));
  auto g_sequence =
      make_tensor(g_offset.data(),
                  make_layout(cute::make_shape(seq_len, get<1>(tile_shape)), g_offset.stride()));
  auto g_tensor = local_tile(g_sequence, tile_shape, make_coord(_, _0{}));
  return g_tensor;
}

template <typename MTensor, typename Shape>
CUTLASS_DEVICE auto get_lse_local_tile_tensor(const MTensor& m_tensor, const Shape& tile_shape,
                                              int head_idx, int offset, int seq_len) {
  auto g_offset = local_tile(m_tensor(head_idx, _), cute::make_shape(_1{}), make_coord(offset));

  auto g_sequence = make_tensor(g_offset.data(), make_layout(cute::make_shape(seq_len),
                                                             cute::make_shape(shape<0>(m_tensor))));
  auto g_tensor = local_tile(g_sequence, tile_shape, make_coord(_));
  return g_tensor;
}

// For SM90, convert acc_layout from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V,
// MMA_N))
template <typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
  static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
  static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
  static_assert(decltype(rank(acc_layout))::value == 3);
  auto l = acc_layout;
  return make_layout(make_layout(get<0, 1>(l), get<1>(l)),
                     make_layout(get<0, 0>(l), get<0, 2>(l), get<2>(l)));
};

// For SM90, convert acc_layout from ((2, 2, N / 8), MMA_M, MMA_N) to ((2, 2, 2), MMA_M, (N / 16,
// MMA_N))
template <typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
  using X = Underscore;
  static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
  static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
  static_assert(decltype(rank(acc_layout))::value == 3);
  static_assert(decltype(rank(get<0>(acc_layout)))::value == 3);
  auto l = logical_divide(get<0>(acc_layout), Shape<X, X, _2>{});  // (2, 2, (2, N / 16)))
  return make_layout(make_layout(get<0>(l), get<1>(l), get<2, 0>(l)), get<1>(acc_layout),
                     make_layout(get<2, 1>(l), get<2>(acc_layout)));
};

// Convert acc_layout from ((2, 2, N / 8), MMA_M, MMA_N) to ((4, 2, 2), MMA_M,
// (N / 32, MMA_N))
template <typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs_fp8(Layout acc_layout) {
  using X = Underscore;
  static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
  static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
  static_assert(decltype(rank(acc_layout))::value == 3);
  static_assert(decltype(rank(get<0>(acc_layout)))::value == 3);
  auto l = logical_divide(get<0>(acc_layout), Shape<X, X, _4>{});  // (2, 2, (2, N / 32)))
  return make_layout(make_layout(Shape<_4, _2, _2>{}), get<1>(acc_layout),
                     make_layout(get<2, 1>(l), get<2>(acc_layout)));
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Byte permute for fp8 kernel
template <typename Fragment>
CUTLASS_DEVICE void permute_regs_A_to_C(Fragment& accum) {
  auto data = accum.data();
#pragma unroll
  for (int n = 0; n < size(accum); n += 8) {
    uint32_t* data_32bit = reinterpret_cast<uint32_t*>(&data[n]);
    auto upper = data_32bit[0];
    auto lower = data_32bit[1];
    data_32bit[0] = __byte_perm(upper, lower, 0x5410);
    data_32bit[1] = __byte_perm(upper, lower, 0x7632);
  }
}

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const& tensor) {
  using From_type = typename Engine::value_type;
  constexpr int numel = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To_type, From_type, numel,
                                 cutlass::FloatRoundStyle::round_to_nearest>
      convert_op;
  // HACK: this requires tensor to be "contiguous"
  auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel>*>(tensor.data()));
  return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

template <bool init = false, int wg_wait = 0, typename TensorA, typename TensorB, typename TensorC,
          typename TiledMma>
__forceinline__ __device__ void gemm(TiledMma& tiled_mma, TensorA const& tCrA, TensorB const& tCrB,
                                     TensorC& tCrC) {
  constexpr bool Is_RS =
      !cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value;
  // Need to cast away const on tCrA since warpgroup_fence_operand doesn't take const
  if constexpr (Is_RS) {
    warpgroup_fence_operand(const_cast<TensorA&>(tCrA));
  }
  warpgroup_fence_operand(tCrC);
  warpgroup_arrive();
  if constexpr (init) {
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    // Unroll the K mode manually to set scale D to 1
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
      cute::gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
  } else {
    // cute::gemm(tiled_mma, tCrA, tCrB, tCrC);
    // Unroll the K mode manually to set scale D to 1
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
      cute::gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
  }
  warpgroup_commit_batch();
  if constexpr (wg_wait >= 0) {
    warpgroup_wait<wg_wait>();
  }
  warpgroup_fence_operand(tCrC);
  if constexpr (Is_RS) {
    warpgroup_fence_operand(const_cast<TensorA&>(tCrA));
  }
}

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_UTILS_CUH_
