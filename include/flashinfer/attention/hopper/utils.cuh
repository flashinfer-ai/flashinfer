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

struct VariableWindowKvTileBounds {
  int begin;  // inclusive first KV tile to load
  int end;    // inclusive last KV tile to load
};

DEFINE_HAS_MEMBER(maybe_variable_window_token_starts)
DEFINE_HAS_MEMBER(maybe_variable_window_token_ends)

// Min start / max end over valid Q rows in this CTA tile. packed_qo_offset is
// qo_indptr[batch] into the packed [nnz_qo] start/end arrays.
template <int CTA_Q, int CTA_KV>
CUTLASS_DEVICE VariableWindowKvTileBounds
get_variable_window_kv_tile_bounds(int32_t const* starts, int32_t const* ends, int packed_qo_offset,
                                   int q_tile_idx, int qo_len, int kv_len) {
  int q_start = q_tile_idx * CTA_Q;
  int q_end = std::min(q_start + CTA_Q, qo_len);
  int min_start = kv_len;
  int max_end = -1;
#pragma unroll 1
  for (int q = q_start; q < q_end; ++q) {
    min_start = std::min(min_start, __ldg(starts + packed_qo_offset + q));
    max_end = std::max(max_end, __ldg(ends + packed_qo_offset + q));
  }
  int num_kv_tiles = cute::ceil_div(kv_len, CTA_KV);
  int first = std::max(min_start / CTA_KV - 1, 0);
  int last = std::min(num_kv_tiles - 1, std::max(max_end, 0) / CTA_KV);
  if (last < first) {
    last = first;
  }
  return {first, last};
}

template <int CTA_Q, int CTA_KV, bool LEFT_SLIDING_WINDOW, bool LEFT_VARIABLE_WINDOW,
          typename Params>
CUTLASS_DEVICE void apply_window_kv_tile_skip(Params const& mainloop_params, int packed_qo_offset,
                                              int q_tile_idx, int qo_len, int kv_len,
                                              int& kv_tile_idx, int& swa_begin_kv_tile_idx) {
  static_assert(!(LEFT_SLIDING_WINDOW && LEFT_VARIABLE_WINDOW),
                "VariableWindow cannot be combined with sliding window");
  if constexpr (LEFT_SLIDING_WINDOW) {
    swa_begin_kv_tile_idx = get_swa_begin_kv_tile_idx<CTA_Q, CTA_KV>(mainloop_params.window_left,
                                                                     q_tile_idx, qo_len, kv_len);
  } else if constexpr (LEFT_VARIABLE_WINDOW) {
    using AdditionalParamsT = decltype(mainloop_params.additional_params);
    if constexpr (has_maybe_variable_window_token_starts_v<AdditionalParamsT> &&
                  has_maybe_variable_window_token_ends_v<AdditionalParamsT>) {
      auto bounds = get_variable_window_kv_tile_bounds<CTA_Q, CTA_KV>(
          mainloop_params.additional_params.maybe_variable_window_token_starts,
          mainloop_params.additional_params.maybe_variable_window_token_ends, packed_qo_offset,
          q_tile_idx, qo_len, kv_len);
      swa_begin_kv_tile_idx = bounds.begin;
      kv_tile_idx = bounds.end;
    }
  }
}

template <int CTA_Q, int CTA_KV, bool LEFT_SLIDING_WINDOW, bool LEFT_VARIABLE_WINDOW,
          typename Params>
CUTLASS_DEVICE void apply_window_kv_tile_skip_consumer(Params const& mainloop_params,
                                                       int packed_qo_offset, int q_tile_idx,
                                                       int qo_len, int kv_len, int& num_kv_tiles,
                                                       int& swa_begin_kv_tile_idx,
                                                       int& swa_end_kv_tile_idx) {
  static_assert(!(LEFT_SLIDING_WINDOW && LEFT_VARIABLE_WINDOW),
                "VariableWindow cannot be combined with sliding window");
  if constexpr (LEFT_SLIDING_WINDOW) {
    swa_begin_kv_tile_idx = get_swa_begin_kv_tile_idx<CTA_Q, CTA_KV>(mainloop_params.window_left,
                                                                     q_tile_idx, qo_len, kv_len);
    swa_end_kv_tile_idx = get_swa_end_kv_tile_idx<CTA_Q, CTA_KV>(mainloop_params.window_left,
                                                                 q_tile_idx, qo_len, kv_len);
  } else if constexpr (LEFT_VARIABLE_WINDOW) {
    using AdditionalParamsT = decltype(mainloop_params.additional_params);
    if constexpr (has_maybe_variable_window_token_starts_v<AdditionalParamsT> &&
                  has_maybe_variable_window_token_ends_v<AdditionalParamsT>) {
      auto bounds = get_variable_window_kv_tile_bounds<CTA_Q, CTA_KV>(
          mainloop_params.additional_params.maybe_variable_window_token_starts,
          mainloop_params.additional_params.maybe_variable_window_token_ends, packed_qo_offset,
          q_tile_idx, qo_len, kv_len);
      // Drain is unused: middle loop consumes (first, last], init consumes last.
      swa_begin_kv_tile_idx = bounds.begin;
      swa_end_kv_tile_idx = bounds.begin - 1;
      num_kv_tiles = bounds.end + 1;
    }
  }
}

template <typename Acc>
CUTLASS_DEVICE void mask_variable_window_score(int32_t const* starts, int32_t const* ends,
                                               int packed_qo_offset, int qo_idx, int qo_len,
                                               int kv_idx, int kv_len, Acc& score, Acc fill_value) {
  if (qo_idx >= qo_len || kv_idx >= kv_len) {
    score = fill_value;
    return;
  }
  int32_t window_start = __ldg(starts + packed_qo_offset + qo_idx);
  int32_t window_end = __ldg(ends + packed_qo_offset + qo_idx);
  if (kv_idx < window_start || kv_idx > window_end) {
    score = fill_value;
  }
}

// Field access is gated on AdditionalParams so FA3 modules compiled with
// LEFT_VARIABLE_WINDOW=false (no start/end pointers) stay well-formed under NVCC.
template <bool LEFT_VARIABLE_WINDOW, typename Params, typename Acc>
CUTLASS_DEVICE void apply_variable_window_score_mask(Params const& mainloop_params,
                                                     int packed_qo_offset, int qo_idx, int qo_len,
                                                     int kv_idx, int kv_len, Acc& score,
                                                     Acc fill_value) {
  if constexpr (LEFT_VARIABLE_WINDOW) {
    using AdditionalParamsT = decltype(mainloop_params.additional_params);
    if constexpr (has_maybe_variable_window_token_starts_v<AdditionalParamsT> &&
                  has_maybe_variable_window_token_ends_v<AdditionalParamsT>) {
      mask_variable_window_score(
          mainloop_params.additional_params.maybe_variable_window_token_starts,
          mainloop_params.additional_params.maybe_variable_window_token_ends, packed_qo_offset,
          qo_idx, qo_len, kv_idx, kv_len, score, fill_value);
    }
  }
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
