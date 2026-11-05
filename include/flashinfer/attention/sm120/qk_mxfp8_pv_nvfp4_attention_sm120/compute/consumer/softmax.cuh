/*
 * Copyright (c) 2025 by SageAttention team.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include <cmath>

#include "../../quantization/fp4_convert.cuh"
#include "../../utils/layout.cuh"
#include "../../utils/math.cuh"
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"

namespace qk_mxfp8_pv_nvfp4_attention {

using namespace cute;

/**
 * Fused Online Softmax with Quantization
 *
 *
 *
 */
template <int Rows>
struct SoftmaxFused {
  using TensorT = decltype(make_fragment_like<float>(Shape<Int<Rows>>{}));
  TensorT row_sum;
  TensorT row_max;
  TensorT scores_scale;

  static constexpr float fp8_scalexfp4_scale = 1.f / (448 * 6);
  static constexpr float fp8_scalexfp4_scale_log2 =
      -11.392317422778762f;  // log2(fp8_scalexfp4_scale)
  static constexpr float fp4_scale_log2 = -2.584962500721156f;
  static constexpr float n64_fp8_scalexfp4_scale_log2 = fp8_scalexfp4_scale_log2;
  static constexpr int RowReductionThr = 4;

  /**
   */
  CUTLASS_DEVICE SoftmaxFused() {};

  CUTLASS_DEVICE static float reduce_row_max_from_pairs(float value) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 2; i < RowReductionThr; i <<= 1) {
      value = fmaxf(value, __shfl_xor_sync(int32_t(-1), value, i));
    }
    return value;
  }

  // Find the row maximum for the complete N128 score tile before either
  // N64 half is converted to P and reused for the following QK tile.
  template <bool FirstTile, bool InfCheck = false, typename TensorAcc, typename TensorMax>
  CUTLASS_DEVICE void prepare_online_softmax_n128(TensorAcc& acc, TensorMax& AbsMaxP,
                                                  const float softmax_scale_log2) {
    Tensor acc_reduction_view = make_tensor(
        acc.data(), qk_mxfp8_pv_nvfp4_attention::convert_to_reduction_layout(acc.layout()));

    static_assert(decltype(size<1, 1>(acc_reduction_view))::value == 4,
                  "An N128 score tile must contain four N32 MMA repeats");

    if constexpr (FirstTile) {
      fill(row_max, -INFINITY);
      clear(row_sum);
      fill(scores_scale, 1.f);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < size<0>(acc_reduction_view); ++mi) {
      float const scores_max_prev = row_max(mi);
      auto find_chunk_max = [&](auto ni) {
        float local_max = -INFINITY;
        CUTLASS_PRAGMA_UNROLL
        for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ++ei) {
          local_max = fmaxf(local_max, acc_reduction_view(mi, make_coord(ei, ni)));
        }
        float const max_recv = __shfl_xor_sync(int32_t(-1), local_max, 1);
        AbsMaxP(mi, ni) = fmaxf(local_max, max_recv);
        row_max(mi) = fmaxf(row_max(mi), AbsMaxP(mi, ni));
      };
      find_chunk_max(_0{});
      find_chunk_max(_1{});
      find_chunk_max(_2{});
      find_chunk_max(_3{});
      row_max(mi) = reduce_row_max_from_pairs(row_max(mi));

      if constexpr (!FirstTile) {
        float const scores_max_cur =
            !InfCheck ? row_max(mi) : (row_max(mi) == -INFINITY ? 0.f : row_max(mi));
        scores_scale(mi) = ptx_exp2((scores_max_prev - scores_max_cur) * softmax_scale_log2);
        row_sum(mi) *= scores_scale(mi);
      }
    }
  }

  // Convert one N64 score half to normalized FP32 values ready for the
  // existing packed-FP4 quantizer. The other N64 half remains untouched.
  template <int ScoreSlot, bool InfCheck = false, int FirstRow = 0, int RowCount = Rows,
            typename TensorAcc, typename TensorMax, typename TensorP>
  CUTLASS_DEVICE void softmax_quantize_n64(TensorAcc& acc, TensorMax& AbsMaxP,
                                           const float softmax_scale_log2, TensorP& packed_p) {
    Tensor acc_reduction_view = make_tensor(
        acc.data(), qk_mxfp8_pv_nvfp4_attention::convert_to_reduction_layout(acc.layout()));
    static_assert(ScoreSlot == 0 || ScoreSlot == 1, "N64 score slot must be 0 or 1");
    static_assert(FirstRow >= 0 && RowCount > 0 && FirstRow + RowCount <= Rows,
                  "N64 softmax row slice must be in range");
    static_assert(decltype(size<1, 1>(acc_reduction_view))::value == 4,
                  "An N128 score tile must contain four N32 MMA repeats");

    CUTLASS_PRAGMA_UNROLL
    for (int mi = FirstRow; mi < FirstRow + RowCount; ++mi) {
      float const max_scaled =
          InfCheck ? (row_max(mi) == -INFINITY
                          ? 0.f
                          : row_max(mi) * softmax_scale_log2 + n64_fp8_scalexfp4_scale_log2)
                   : row_max(mi) * softmax_scale_log2 + n64_fp8_scalexfp4_scale_log2;

      auto exp2_sum = [&](auto ni) {
        // Compute P directly in the FP4-normalized domain. For a
        // chunk maximum c and online row maximum m:
        //   P_norm = exp2((s-c)*scale) * 6
        //   P_sf   = exp2((c-m)*scale) * 448
        // Their product is the same 2688-scaled probability used by
        // the original path. Accumulate the normalized chunk first,
        // then apply P_sf once instead of normalizing every element
        // with a reciprocal and a multiply in a second pass.
        float const chunk_max = AbsMaxP(mi, ni);
        bool const empty_chunk = chunk_max == -INFINITY;
        float const chunk_max_safe = empty_chunk ? 0.0f : chunk_max;
        float const norm_base = chunk_max_safe * softmax_scale_log2 + fp4_scale_log2;
        float chunk_sum = 0.0f;
        CUTLASS_PRAGMA_UNROLL
        for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ++ei) {
          float const value =
              ptx_exp2(acc_reduction_view(mi, make_coord(ei, ni)) * softmax_scale_log2 - norm_base);
          acc_reduction_view(mi, make_coord(ei, ni)) = value;
          chunk_sum += value;
        }
        float const p_scale =
            empty_chunk ? 0.0f
                        : pscale_exp2(chunk_max * softmax_scale_log2 - max_scaled + fp4_scale_log2);
        AbsMaxP(mi, ni) = p_scale;
        row_sum(mi) += chunk_sum * p_scale;
      };

      if constexpr (ScoreSlot == 0) {
        exp2_sum(_0{});
        exp2_sum(_1{});
      } else {
        exp2_sum(_2{});
        exp2_sum(_3{});
      }
    }
  }

  /**
   *
   *
   *
   *
   */
  template <bool FirstTile, bool InfCheck = false, typename TensorAcc, typename TensorMax>
  CUTLASS_DEVICE auto online_softmax_with_quant(TensorAcc& acc, TensorMax& AbsMaxP,
                                                const float softmax_scale_log2) {
    Tensor acc_reduction_view = make_tensor(
        acc.data(), qk_mxfp8_pv_nvfp4_attention::convert_to_reduction_layout(acc.layout()));

    Tensor acc_conversion_view = make_tensor(
        acc.data(), qk_mxfp8_pv_nvfp4_attention::convert_to_conversion_layout(acc.layout()));
    auto temp1 = flatten(acc_conversion_view);
    auto temp2 = group_modes<0, 2>(temp1);
    auto acc_conversion_flatten = group_modes<1, 5>(temp2);

    if constexpr (FirstTile) {
      clear(row_sum);

      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
        float row_max_cur = -INFINITY;
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1, 1>(acc_reduction_view); ni++) {
          float local_max = -INFINITY;
          CUTLASS_PRAGMA_UNROLL
          for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
            local_max = fmaxf(local_max, acc_reduction_view(mi, make_coord(ei, ni)));
          }
          float max_recv = __shfl_xor_sync(int32_t(-1), local_max, 1);
          AbsMaxP(mi, ni) = fmaxf(local_max, max_recv);
          row_max_cur = fmaxf(row_max_cur, AbsMaxP(mi, ni));
        }

        float max_recv = __shfl_xor_sync(int32_t(-1), row_max_cur, 2);
        row_max_cur = fmaxf(row_max_cur, max_recv);
        row_max(mi) = row_max_cur;

        const float max_scaled =
            InfCheck ? (row_max_cur == -INFINITY
                            ? 0.f
                            : (row_max_cur * softmax_scale_log2 + fp8_scalexfp4_scale_log2))
                     : (row_max_cur * softmax_scale_log2 + fp8_scalexfp4_scale_log2);

        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
          acc_reduction_view(mi, ni) =
              ptx_exp2(acc_reduction_view(mi, ni) * softmax_scale_log2 - max_scaled);
        }

        CUTLASS_PRAGMA_UNROLL
        for (int sfi = 0; sfi < size<1>(AbsMaxP); sfi++) {
          AbsMaxP(mi, sfi) =
              ptx_exp2(AbsMaxP(mi, sfi) * softmax_scale_log2 - max_scaled + fp4_scale_log2);
        }
      }

      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
          row_sum(mi) += acc_reduction_view(mi, ni);
        }
      }
    } else {
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
        float scores_max_prev_mi = row_max(mi);
        float row_max_cur = scores_max_prev_mi;
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1, 1>(acc_reduction_view); ni++) {
          float local_max = -INFINITY;
          CUTLASS_PRAGMA_UNROLL
          for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
            local_max = fmaxf(local_max, acc_reduction_view(mi, make_coord(ei, ni)));
          }
          float max_recv = __shfl_xor_sync(int32_t(-1), local_max, 1);
          AbsMaxP(mi, ni) = fmaxf(local_max, max_recv);
          row_max_cur = fmaxf(row_max_cur, AbsMaxP(mi, ni));
        }

        float max_recv = __shfl_xor_sync(int32_t(-1), row_max_cur, 2);
        row_max_cur = fmaxf(row_max_cur, max_recv);
        row_max(mi) = row_max_cur;

        float scores_max_cur =
            !InfCheck ? row_max_cur : (row_max_cur == -INFINITY ? 0.0f : row_max_cur);
        scores_scale(mi) = ptx_exp2((scores_max_prev_mi - scores_max_cur) * softmax_scale_log2);

        const float max_scaled =
            InfCheck ? (row_max_cur == -INFINITY
                            ? 0.f
                            : (row_max_cur * softmax_scale_log2 + fp8_scalexfp4_scale_log2))
                     : (row_max_cur * softmax_scale_log2 + fp8_scalexfp4_scale_log2);

        row_sum(mi) = row_sum(mi) * scores_scale(mi);

        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
          float val = ptx_exp2(acc_reduction_view(mi, ni) * softmax_scale_log2 - max_scaled);
          acc_reduction_view(mi, ni) = val;
          row_sum(mi) += val;
        }

        CUTLASS_PRAGMA_UNROLL
        for (int sfi = 0; sfi < size<1>(AbsMaxP); sfi++) {
          AbsMaxP(mi, sfi) =
              ptx_exp2(AbsMaxP(mi, sfi) * softmax_scale_log2 - max_scaled + fp4_scale_log2);
        }
      }
    }

    // Quantize in-place with a scalar reciprocal to avoid keeping a second
    // AbsMaxP-shaped register tensor live across the conversion loop.
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(AbsMaxP); ++i) {
      float inv_absmax = AbsMaxP(i) == 0.0f ? 0.0f : 1.0f / AbsMaxP(i);
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<0>(acc_conversion_flatten); ++j) {
        acc_conversion_flatten(j, i) *= inv_absmax;
      }
    }
  }

  template <bool InfCheck = false, typename TensorAcc, typename TensorMax>
  CUTLASS_DEVICE auto online_softmax_with_quant_direct_norm_nonfirst(
      TensorAcc& acc, TensorMax& AbsMaxP, const float softmax_scale_log2) {
    Tensor acc_reduction_view = make_tensor(
        acc.data(), qk_mxfp8_pv_nvfp4_attention::convert_to_reduction_layout(acc.layout()));

    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
      float scores_max_prev_mi = row_max(mi);
      float row_max_cur = scores_max_prev_mi;

      CUTLASS_PRAGMA_UNROLL
      for (int ni = 0; ni < size<1, 1>(acc_reduction_view); ni++) {
        float local_max = -INFINITY;
        CUTLASS_PRAGMA_UNROLL
        for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
          local_max = fmaxf(local_max, acc_reduction_view(mi, make_coord(ei, ni)));
        }
        float max_recv = __shfl_xor_sync(int32_t(-1), local_max, 1);
        AbsMaxP(mi, ni) = fmaxf(local_max, max_recv);
        row_max_cur = fmaxf(row_max_cur, AbsMaxP(mi, ni));
      }

      float max_recv = __shfl_xor_sync(int32_t(-1), row_max_cur, 2);
      row_max_cur = fmaxf(row_max_cur, max_recv);
      row_max(mi) = row_max_cur;

      float const scores_max_cur =
          !InfCheck ? row_max_cur : (row_max_cur == -INFINITY ? 0.0f : row_max_cur);
      scores_scale(mi) = InfCheck && row_max_cur == -INFINITY
                             ? 1.0f
                             : ptx_exp2((scores_max_prev_mi - scores_max_cur) * softmax_scale_log2);
      row_sum(mi) *= scores_scale(mi);

      float const max_scaled = scores_max_cur * softmax_scale_log2 + fp8_scalexfp4_scale_log2;

      CUTLASS_PRAGMA_UNROLL
      for (int ni = 0; ni < size<1, 1>(acc_reduction_view); ni++) {
        float const chunk_max = AbsMaxP(mi, ni);
        bool const empty_chunk = InfCheck && chunk_max == -INFINITY;
        float const sf =
            empty_chunk ? 0.0f
                        : ptx_exp2(chunk_max * softmax_scale_log2 - max_scaled + fp4_scale_log2);
        AbsMaxP(mi, ni) = sf;

        float const norm_base =
            empty_chunk ? 0.0f : chunk_max * softmax_scale_log2 + fp4_scale_log2;
        CUTLASS_PRAGMA_UNROLL
        for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
          float const norm =
              empty_chunk
                  ? 0.0f
                  : ptx_exp2(acc_reduction_view(mi, make_coord(ei, ni)) * softmax_scale_log2 -
                             norm_base);
          acc_reduction_view(mi, make_coord(ei, ni)) = norm;
          row_sum(mi) += norm * sf;
        }
      }
    }
  }

  /**
   *
   *
   */
  template <typename TensorAcc>
  CUTLASS_DEVICE void finalize(TensorAcc& o_store) {
    Tensor o_store_reduction_view =
        make_tensor(o_store.data(), convert_to_reduction_layout(o_store.layout()));

    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < size(row_max); ++mi) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 1; i < RowReductionThr; i <<= 1) {
        float sum_recv = __shfl_xor_sync(int32_t(-1), row_sum(mi), i);
        row_sum(mi) += sum_recv;
      }

      float sum = row_sum(mi);
      float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;

      CUTLASS_PRAGMA_UNROLL
      for (int ni = 0; ni < size<1>(o_store_reduction_view); ++ni) {
        o_store_reduction_view(mi, ni) *= inv_sum;
      }
    }
  }

  /**
   *
   * O_new = O_old * scale + O_current
   *
   */
  template <typename TensorAcc>
  CUTLASS_DEVICE void rescale_o(TensorAcc& o_store, TensorAcc const& o_tmp) {
    Tensor o_store_reduction_view = make_tensor(
        o_store.data(), qk_mxfp8_pv_nvfp4_attention::convert_to_reduction_layout(o_store.layout()));
    Tensor o_tmp_reduction_view = make_tensor(
        o_tmp.data(), qk_mxfp8_pv_nvfp4_attention::convert_to_reduction_layout(o_tmp.layout()));

    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < size(row_max); ++mi) {
      CUTLASS_PRAGMA_UNROLL
      for (int ni = 0; ni < size<1>(o_store_reduction_view); ++ni) {
        o_store_reduction_view(mi, ni) =
            o_store_reduction_view(mi, ni) * scores_scale(mi) + o_tmp_reduction_view(mi, ni);
      }
    }
  }

  // The N64 slot-reuse path accumulates PV directly into o_store. Once the
  // next tile establishes a new online-softmax maximum, only the existing
  // accumulator needs to be rescaled before that tile's PV contributions.
  template <typename TensorAcc>
  CUTLASS_DEVICE void rescale_o_inplace(TensorAcc& o_store) {
    Tensor o_store_reduction_view = make_tensor(
        o_store.data(), qk_mxfp8_pv_nvfp4_attention::convert_to_reduction_layout(o_store.layout()));

    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < size(row_max); ++mi) {
      CUTLASS_PRAGMA_UNROLL
      for (int ni = 0; ni < size<1>(o_store_reduction_view); ++ni) {
        o_store_reduction_view(mi, ni) *= scores_scale(mi);
      }
    }
  }

 private:
  __device__ __forceinline__ static float pscale_exp2(float x) { return ptx_exp2(x); }

  /**
   */
  __device__ __forceinline__ static float ptx_exp2(float x) {
    float result;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
  }
};

}  // namespace qk_mxfp8_pv_nvfp4_attention
