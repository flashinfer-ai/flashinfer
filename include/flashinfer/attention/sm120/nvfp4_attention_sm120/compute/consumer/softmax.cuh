/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cmath>

#include "../../utils/layout.cuh"
#include "../../utils/math.cuh"
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"

namespace nvfp4_attention {

using cute::clear;
using cute::copy;
using cute::fill;
using cute::flatten;
using cute::group_modes;
using cute::Int;
using cute::make_coord;
using cute::make_fragment_like;
using cute::make_tensor;
using cute::make_tensor_like;
using cute::Shape;
using cute::size;
using cute::Tensor;

template <int Rows>
struct SoftmaxFused {
  using TensorT = decltype(make_fragment_like<float>(Shape<Int<Rows>>{}));
  TensorT row_sum;
  TensorT row_max;
  TensorT scores_scale;

  static constexpr float fp8_scalexfp4_scale = 1.f / (448 * 6);
  static constexpr float fp8_scalexfp4_scale_log2 = -11.392317422778762f;
  static constexpr float fp4_scale = 1.f / 6.f;
  static constexpr float fp4_scale_log2 = -2.584962500721156f;
  static constexpr float AbsMaxPEps = 1.0e-8f;
  // One accumulator row is spread across 4 threads (m16n8 acc: 2 columns
  // per thread per 8-column atom); reducing across 8 threads would fold
  // the neighboring row's sum into row_sum and halve the output.
  static constexpr int RowReductionThr = 4;

  CUTLASS_DEVICE SoftmaxFused() {};

  CUTLASS_DEVICE static float reduce_row_max_from_pairs(float value) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 2; i < RowReductionThr; i <<= 1) {
      value = fmaxf(value, __shfl_xor_sync(int32_t(-1), value, i));
    }
    return value;
  }

#if defined(FAST_RCP_ABSMAXP)
  CUTLASS_DEVICE static float fast_rcp_approx(float x) {
    float y;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
  }
#endif

  CUTLASS_DEVICE static float safe_inv_absmax(float x) {
    float denom = fmaxf(x, AbsMaxPEps);
#if defined(FAST_RCP_ABSMAXP)
    return fast_rcp_approx(denom);
#else
    return 1.0f / denom;
#endif
  }

  template <bool FirstTile, bool InfCheck = false, typename TensorAcc, typename TensorMax>
  CUTLASS_DEVICE auto online_softmax_with_quant(TensorAcc& acc, TensorMax& AbsMaxP,
                                                const float softmax_scale_log2) {
    Tensor acc_reduction_view =
        make_tensor(acc.data(), nvfp4_attention::convert_to_reduction_layout(acc.layout()));

    Tensor acc_conversion_view =
        make_tensor(acc.data(), nvfp4_attention::convert_to_conversion_layout(acc.layout()));

    auto temp1 = flatten(acc_conversion_view);
    auto temp2 = group_modes<0, 2>(temp1);
    auto acc_conversion_flatten = group_modes<1, 5>(temp2);

    if constexpr (FirstTile) {
      fill(row_max, -INFINITY);
      clear(row_sum);
      fill(scores_scale, 1.f);

      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1, 1>(acc_reduction_view); ni++) {
          float local_max = -INFINITY;
          CUTLASS_PRAGMA_UNROLL
          for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
            local_max = fmaxf(local_max, acc_reduction_view(mi, make_coord(ei, ni)));
          }

          float max_recv = __shfl_xor_sync(int32_t(-1), local_max, 1);
          AbsMaxP(mi, ni) = fmaxf(local_max, max_recv);
          row_max(mi) = fmaxf(row_max(mi), AbsMaxP(mi, ni));
        }

        row_max(mi) = reduce_row_max_from_pairs(row_max(mi));

        const float max_scaled =
            InfCheck ? (row_max(mi) == -INFINITY
                            ? 0.f
                            : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2))
                     : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2);

#if defined(DIRECT_P_QUANT_SOFTMAX)

        CUTLASS_PRAGMA_UNROLL
        for (int sfi = 0; sfi < size<1>(AbsMaxP); sfi++) {
          float chunk_max = AbsMaxP(mi, sfi);
          float sfp = 0.0f;
          if constexpr (InfCheck) {
            if (chunk_max == -INFINITY) {
              CUTLASS_PRAGMA_UNROLL
              for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
                acc_reduction_view(mi, make_coord(ei, sfi)) = 0.0f;
              }
              AbsMaxP(mi, sfi) = 0.0f;
              continue;
            }
          }
          float chunk_scaled = chunk_max * softmax_scale_log2;
          sfp = softmax_exp2<InfCheck>(chunk_scaled - max_scaled + fp4_scale_log2);
          AbsMaxP(mi, sfi) = sfp;
          CUTLASS_PRAGMA_UNROLL
          for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
            float p = softmax_exp2<InfCheck>(acc_reduction_view(mi, make_coord(ei, sfi)) *
                                                 softmax_scale_log2 -
                                             chunk_scaled - fp4_scale_log2);
            acc_reduction_view(mi, make_coord(ei, sfi)) = p;
            row_sum(mi) += p * sfp;
          }
        }
#else

        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
          float exp_val =
              softmax_exp2<InfCheck>(acc_reduction_view(mi, ni) * softmax_scale_log2 - max_scaled);
          acc_reduction_view(mi, ni) = exp_val;
#if defined(FIRST_TILE_SUM_IN_EXP)
          row_sum(mi) += exp_val;
#endif
        }

        CUTLASS_PRAGMA_UNROLL
        for (int sfi = 0; sfi < size<1>(AbsMaxP); sfi++) {
#if defined(SFP_FROM_EXP_MAX)
          float local_exp_max = 0.0f;
          CUTLASS_PRAGMA_UNROLL
          for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
            local_exp_max = fmaxf(local_exp_max, acc_reduction_view(mi, make_coord(ei, sfi)));
          }
          float peer_exp_max = __shfl_xor_sync(int32_t(-1), local_exp_max, 1);
          AbsMaxP(mi, sfi) = fmaxf(local_exp_max, peer_exp_max) * fp4_scale;
#else
          AbsMaxP(mi, sfi) = softmax_exp2<InfCheck>(AbsMaxP(mi, sfi) * softmax_scale_log2 -
                                                    max_scaled + fp4_scale_log2);
#endif
        }
#endif
      }

#if !defined(DIRECT_P_QUANT_SOFTMAX) && !defined(FIRST_TILE_SUM_IN_EXP)

      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
          row_sum(mi) += acc_reduction_view(mi, ni);
        }
      }
#endif
    } else {
      Tensor scores_max_prev = make_fragment_like(row_max);
      cute::copy(row_max, scores_max_prev);

      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1, 1>(acc_reduction_view); ni++) {
          float local_max = -INFINITY;
          CUTLASS_PRAGMA_UNROLL
          for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
            local_max = fmaxf(local_max, acc_reduction_view(mi, make_coord(ei, ni)));
          }
          float max_recv = __shfl_xor_sync(int32_t(-1), local_max, 1);
          AbsMaxP(mi, ni) = fmaxf(local_max, max_recv);
          row_max(mi) = fmaxf(row_max(mi), AbsMaxP(mi, ni));
        }

        row_max(mi) = reduce_row_max_from_pairs(row_max(mi));

        float scores_max_cur =
            !InfCheck ? row_max(mi) : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
        scores_scale(mi) =
            softmax_exp2<InfCheck>((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);

        const float max_scaled =
            InfCheck ? (row_max(mi) == -INFINITY
                            ? 0.f
                            : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2))
                     : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2);

        row_sum(mi) = row_sum(mi) * scores_scale(mi);

#if defined(DIRECT_P_QUANT_SOFTMAX)

        CUTLASS_PRAGMA_UNROLL
        for (int sfi = 0; sfi < size<1>(AbsMaxP); sfi++) {
          float chunk_max = AbsMaxP(mi, sfi);
          float sfp = 0.0f;
          if constexpr (InfCheck) {
            if (chunk_max == -INFINITY) {
              CUTLASS_PRAGMA_UNROLL
              for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
                acc_reduction_view(mi, make_coord(ei, sfi)) = 0.0f;
              }
              AbsMaxP(mi, sfi) = 0.0f;
              continue;
            }
          }
          float chunk_scaled = chunk_max * softmax_scale_log2;
          sfp = softmax_exp2<InfCheck>(chunk_scaled - max_scaled + fp4_scale_log2);
          AbsMaxP(mi, sfi) = sfp;
          CUTLASS_PRAGMA_UNROLL
          for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
            float p = softmax_exp2<InfCheck>(acc_reduction_view(mi, make_coord(ei, sfi)) *
                                                 softmax_scale_log2 -
                                             chunk_scaled - fp4_scale_log2);
            acc_reduction_view(mi, make_coord(ei, sfi)) = p;
            row_sum(mi) += p * sfp;
          }
        }
#else

        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
          acc_reduction_view(mi, ni) =
              softmax_exp2<InfCheck>(acc_reduction_view(mi, ni) * softmax_scale_log2 - max_scaled);
          row_sum(mi) += acc_reduction_view(mi, ni);
        }

        CUTLASS_PRAGMA_UNROLL
        for (int sfi = 0; sfi < size<1>(AbsMaxP); sfi++) {
#if defined(SFP_FROM_EXP_MAX)
          float local_exp_max = 0.0f;
          CUTLASS_PRAGMA_UNROLL
          for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
            local_exp_max = fmaxf(local_exp_max, acc_reduction_view(mi, make_coord(ei, sfi)));
          }
          float peer_exp_max = __shfl_xor_sync(int32_t(-1), local_exp_max, 1);
          AbsMaxP(mi, sfi) = fmaxf(local_exp_max, peer_exp_max) * fp4_scale;
#else
          AbsMaxP(mi, sfi) = softmax_exp2<InfCheck>(AbsMaxP(mi, sfi) * softmax_scale_log2 -
                                                    max_scaled + fp4_scale_log2);
#endif
        }
#endif
      }
    }

#if !defined(DIRECT_P_QUANT_SOFTMAX)

#if defined(SCALAR_INV_ABSMAXP)
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(AbsMaxP); ++i) {
      const float inv_absmax = safe_inv_absmax(AbsMaxP(i));
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<0>(acc_conversion_flatten); ++j) {
        acc_conversion_flatten(j, i) *= inv_absmax;
      }
    }
#else
    Tensor inv_AbsMaxP = make_tensor_like<float>(AbsMaxP.layout());
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(inv_AbsMaxP); ++i) {
      inv_AbsMaxP(i) = safe_inv_absmax(AbsMaxP(i));
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(inv_AbsMaxP); ++i) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<0>(acc_conversion_flatten); ++j) {
        acc_conversion_flatten(j, i) *= inv_AbsMaxP(i);
      }
    }
#endif
#endif
  }

#if defined(FP16_SOFTMAX)

  template <bool FirstTile, bool InfCheck = false, typename TensorAcc, typename TensorMax>
  CUTLASS_DEVICE auto online_softmax_with_quant_fp16(TensorAcc& acc, TensorMax& AbsMaxP,
                                                     const float softmax_scale_log2) {
    Tensor acc_reduction_view =
        make_tensor(acc.data(), nvfp4_attention::convert_to_reduction_layout(acc.layout()));
    Tensor acc_conversion_view =
        make_tensor(acc.data(), nvfp4_attention::convert_to_conversion_layout(acc.layout()));
    auto acc_conversion_flatten =
        group_modes<1, 5>(group_modes<0, 2>(flatten(acc_conversion_view)));

    if constexpr (FirstTile) {
      fill(row_max, -INFINITY);
      clear(row_sum);
      fill(scores_scale, 1.f);

      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1, 1>(acc_reduction_view); ni++) {
          float local_max = -INFINITY;
          CUTLASS_PRAGMA_UNROLL
          for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
            local_max = fmaxf(local_max, acc_reduction_view(mi, make_coord(ei, ni)));
          }
          float max_recv = __shfl_xor_sync(int32_t(-1), local_max, 1);
          AbsMaxP(mi, ni) = fmaxf(local_max, max_recv);
          row_max(mi) = fmaxf(row_max(mi), AbsMaxP(mi, ni));
        }

        row_max(mi) = reduce_row_max_from_pairs(row_max(mi));

        const float max_scaled =
            InfCheck ? (row_max(mi) == -INFINITY
                            ? 0.f
                            : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2))
                     : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2);

        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
          acc_reduction_view(mi, ni) =
              softmax_exp2<InfCheck>(acc_reduction_view(mi, ni) * softmax_scale_log2 - max_scaled);
        }

        CUTLASS_PRAGMA_UNROLL
        for (int sfi = 0; sfi < size<1>(AbsMaxP); sfi++) {
          AbsMaxP(mi, sfi) = softmax_exp2<InfCheck>(AbsMaxP(mi, sfi) * softmax_scale_log2 -
                                                    max_scaled + fp4_scale_log2);
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
      Tensor scores_max_prev = make_fragment_like(row_max);
      cute::copy(row_max, scores_max_prev);

      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1, 1>(acc_reduction_view); ni++) {
          float local_max = -INFINITY;
          CUTLASS_PRAGMA_UNROLL
          for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
            local_max = fmaxf(local_max, acc_reduction_view(mi, make_coord(ei, ni)));
          }
          float max_recv = __shfl_xor_sync(int32_t(-1), local_max, 1);
          AbsMaxP(mi, ni) = fmaxf(local_max, max_recv);
          row_max(mi) = fmaxf(row_max(mi), AbsMaxP(mi, ni));
        }

        row_max(mi) = reduce_row_max_from_pairs(row_max(mi));

        float scores_max_cur =
            !InfCheck ? row_max(mi) : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
        scores_scale(mi) =
            softmax_exp2<InfCheck>((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);

        const float max_scaled =
            InfCheck ? (row_max(mi) == -INFINITY
                            ? 0.f
                            : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2))
                     : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2);

        row_sum(mi) = row_sum(mi) * scores_scale(mi);

        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
          acc_reduction_view(mi, ni) =
              softmax_exp2<InfCheck>(acc_reduction_view(mi, ni) * softmax_scale_log2 - max_scaled);
          row_sum(mi) += acc_reduction_view(mi, ni);
        }

        CUTLASS_PRAGMA_UNROLL
        for (int sfi = 0; sfi < size<1>(AbsMaxP); sfi++) {
          AbsMaxP(mi, sfi) = softmax_exp2<InfCheck>(AbsMaxP(mi, sfi) * softmax_scale_log2 -
                                                    max_scaled + fp4_scale_log2);
        }
      }
    }

#if defined(SCALAR_INV_ABSMAXP)
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(AbsMaxP); ++i) {
      const float inv_absmax = safe_inv_absmax(AbsMaxP(i));
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<0>(acc_conversion_flatten); ++j) {
        acc_conversion_flatten(j, i) *= inv_absmax;
      }
    }
#else
    Tensor inv_AbsMaxP = make_tensor_like<float>(AbsMaxP.layout());
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(inv_AbsMaxP); ++i) {
      inv_AbsMaxP(i) = safe_inv_absmax(AbsMaxP(i));
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(inv_AbsMaxP); ++i) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<0>(acc_conversion_flatten); ++j) {
        acc_conversion_flatten(j, i) *= inv_AbsMaxP(i);
      }
    }
#endif
  }
#endif

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

  template <typename TensorAcc>
  CUTLASS_DEVICE void rescale_o(TensorAcc& o_store, TensorAcc const& o_tmp) {
    Tensor o_store_reduction_view =
        make_tensor(o_store.data(), nvfp4_attention::convert_to_reduction_layout(o_store.layout()));
    Tensor o_tmp_reduction_view =
        make_tensor(o_tmp.data(), nvfp4_attention::convert_to_reduction_layout(o_tmp.layout()));

    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < size(row_max); ++mi) {
      CUTLASS_PRAGMA_UNROLL
      for (int ni = 0; ni < size<1>(o_store_reduction_view); ++ni) {
        o_store_reduction_view(mi, ni) =
            o_store_reduction_view(mi, ni) * scores_scale(mi) + o_tmp_reduction_view(mi, ni);
      }
    }
  }

  template <bool InfCheck = false, typename TensorAcc, typename TensorMax>
  CUTLASS_DEVICE void find_max_chunk(TensorAcc& acc, TensorMax& AbsMaxP, int ni) {
    Tensor acc_rv =
        make_tensor(acc.data(), nvfp4_attention::convert_to_reduction_layout(acc.layout()));

    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < size<0>(acc_rv); mi++) {
      float chunk_max = -INFINITY;
      CUTLASS_PRAGMA_UNROLL
      for (int ei = 0; ei < size<1, 0>(acc_rv); ei++) {
        chunk_max = fmaxf(chunk_max, acc_rv(mi, make_coord(ei, ni)));
      }
      float max_recv = __shfl_xor_sync(int32_t(-1), chunk_max, 1);
      chunk_max = fmaxf(chunk_max, max_recv);
      AbsMaxP(mi, ni) = chunk_max;

      row_max(mi) = fmaxf(row_max(mi), chunk_max);
      row_max(mi) = reduce_row_max_from_pairs(row_max(mi));
    }
  }

  template <bool InfCheck = false, typename TensorAcc, typename TensorMax>
  CUTLASS_DEVICE void exp2_sum_chunk(TensorAcc& acc, TensorMax& AbsMaxP, int ni,
                                     const float softmax_scale_log2) {
    Tensor acc_rv =
        make_tensor(acc.data(), nvfp4_attention::convert_to_reduction_layout(acc.layout()));

    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < size<0>(acc_rv); mi++) {
      const float max_scaled =
          InfCheck ? (row_max(mi) == -INFINITY
                          ? 0.f
                          : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2))
                   : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2);

      CUTLASS_PRAGMA_UNROLL
      for (int ei = 0; ei < size<1, 0>(acc_rv); ei++) {
        float val = softmax_exp2<InfCheck>(acc_rv(mi, make_coord(ei, ni)) * softmax_scale_log2 -
                                           max_scaled);
        acc_rv(mi, make_coord(ei, ni)) = val;
        row_sum(mi) += val;
      }

      AbsMaxP(mi, ni) = softmax_exp2<InfCheck>(AbsMaxP(mi, ni) * softmax_scale_log2 - max_scaled +
                                               fp4_scale_log2);
    }
  }

  template <typename TensorAcc, typename TensorMax>
  CUTLASS_DEVICE void quantize_after_partial_softmax(TensorAcc& acc, TensorMax& AbsMaxP) {
    Tensor acc_cv =
        make_tensor(acc.data(), nvfp4_attention::convert_to_conversion_layout(acc.layout()));
    auto temp1 = flatten(acc_cv);
    auto temp2 = group_modes<0, 2>(temp1);
    auto acc_flat = group_modes<1, 5>(temp2);

    Tensor inv_AbsMaxP = make_tensor_like<float>(AbsMaxP.layout());
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(inv_AbsMaxP); ++i) {
      inv_AbsMaxP(i) = safe_inv_absmax(AbsMaxP(i));
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(inv_AbsMaxP); ++i) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<0>(acc_flat); ++j) {
        acc_flat(j, i) *= inv_AbsMaxP(i);
      }
    }
  }

  template <bool InfCheck = false, typename TensorAcc, typename TensorMax, typename TensorPrev>
  CUTLASS_DEVICE void chunked_softmax_fixed(TensorAcc& acc, TensorMax& AbsMaxP, bool is_first,
                                            const float softmax_scale_log2,
                                            TensorPrev const& scores_max_prev) {
    Tensor acc_rv =
        make_tensor(acc.data(), nvfp4_attention::convert_to_reduction_layout(acc.layout()));

    Tensor acc_cv =
        make_tensor(acc.data(), nvfp4_attention::convert_to_conversion_layout(acc.layout()));
    auto acc_cv_flat = group_modes<1, 5>(group_modes<0, 2>(flatten(acc_cv)));

    constexpr int MmaN = decltype(size<1, 1>(acc_rv))::value;

    if (is_first) {
      fill(row_max, -INFINITY);
      clear(row_sum);
      fill(scores_scale, 1.f);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < size<0>(acc_rv); mi++) {
      CUTLASS_PRAGMA_UNROLL
      for (int ni = 0; ni < MmaN; ni++) {
        float local_max = -INFINITY;
        CUTLASS_PRAGMA_UNROLL
        for (int ei = 0; ei < size<1, 0>(acc_rv); ei++) {
          local_max = fmaxf(local_max, acc_rv(mi, make_coord(ei, ni)));
        }
        float max_recv = __shfl_xor_sync(int32_t(-1), local_max, 1);
        AbsMaxP(mi, ni) = fmaxf(local_max, max_recv);
        row_max(mi) = fmaxf(row_max(mi), AbsMaxP(mi, ni));
      }
      row_max(mi) = reduce_row_max_from_pairs(row_max(mi));
    }

    if (!is_first) {
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < size<0>(acc_rv); mi++) {
        float scores_max_cur =
            !InfCheck ? row_max(mi) : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
        scores_scale(mi) =
            softmax_exp2<InfCheck>((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
        row_sum(mi) *= scores_scale(mi);
      }
    }

    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < size<0>(acc_rv); mi++) {
      const float max_scaled =
          InfCheck ? (row_max(mi) == -INFINITY
                          ? 0.f
                          : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2))
                   : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2);

      CUTLASS_PRAGMA_UNROLL
      for (int ni = 0; ni < size<1>(acc_rv); ni++) {
        float val = softmax_exp2<InfCheck>(acc_rv(mi, ni) * softmax_scale_log2 - max_scaled);
        acc_rv(mi, ni) = val;
        row_sum(mi) += val;
      }

      CUTLASS_PRAGMA_UNROLL
      for (int sfi = 0; sfi < size<1>(AbsMaxP); sfi++) {
        AbsMaxP(mi, sfi) = softmax_exp2<InfCheck>(AbsMaxP(mi, sfi) * softmax_scale_log2 -
                                                  max_scaled + fp4_scale_log2);
      }
    }

    Tensor inv_AbsMaxP = make_tensor_like<float>(AbsMaxP.layout());
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(inv_AbsMaxP); ++i) {
      inv_AbsMaxP(i) = safe_inv_absmax(AbsMaxP(i));
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(inv_AbsMaxP); ++i) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<0>(acc_cv_flat); ++j) {
        acc_cv_flat(j, i) *= inv_AbsMaxP(i);
      }
    }
  }

  template <bool InfCheck = false, typename TensorAcc, typename TensorMax, typename TensorPrev>
  CUTLASS_DEVICE void exp2_sum_and_quantize(TensorAcc& acc, TensorMax& AbsMaxP, bool is_first,
                                            const float softmax_scale_log2,
                                            TensorPrev const& scores_max_prev) {
    Tensor acc_rv =
        make_tensor(acc.data(), nvfp4_attention::convert_to_reduction_layout(acc.layout()));
    Tensor acc_cv =
        make_tensor(acc.data(), nvfp4_attention::convert_to_conversion_layout(acc.layout()));
    auto acc_cv_flat = group_modes<1, 5>(group_modes<0, 2>(flatten(acc_cv)));

    if (!is_first) {
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < size<0>(acc_rv); mi++) {
        float scores_max_cur =
            !InfCheck ? row_max(mi) : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
        scores_scale(mi) =
            softmax_exp2<InfCheck>((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
        row_sum(mi) *= scores_scale(mi);
      }
    }

    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < size<0>(acc_rv); mi++) {
      const float max_scaled =
          InfCheck ? (row_max(mi) == -INFINITY
                          ? 0.f
                          : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2))
                   : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2);

      CUTLASS_PRAGMA_UNROLL
      for (int ni = 0; ni < size<1>(acc_rv); ni++) {
        float val = softmax_exp2<InfCheck>(acc_rv(mi, ni) * softmax_scale_log2 - max_scaled);
        acc_rv(mi, ni) = val;
        row_sum(mi) += val;
      }

      CUTLASS_PRAGMA_UNROLL
      for (int sfi = 0; sfi < size<1>(AbsMaxP); sfi++) {
        AbsMaxP(mi, sfi) = softmax_exp2<InfCheck>(AbsMaxP(mi, sfi) * softmax_scale_log2 -
                                                  max_scaled + fp4_scale_log2);
      }
    }

    Tensor inv_AbsMaxP = make_tensor_like<float>(AbsMaxP.layout());
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(inv_AbsMaxP); ++i) {
      inv_AbsMaxP(i) = safe_inv_absmax(AbsMaxP(i));
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(inv_AbsMaxP); ++i) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<0>(acc_cv_flat); ++j) {
        acc_cv_flat(j, i) *= inv_AbsMaxP(i);
      }
    }
  }

  template <bool IsInit, bool InfCheck = false, typename TensorAcc, typename TensorMax>
  CUTLASS_DEVICE void online_softmax_chunk(TensorAcc& acc, TensorMax& AbsMaxP, int ni,
                                           const float softmax_scale_log2) {
    Tensor acc_rv =
        make_tensor(acc.data(), nvfp4_attention::convert_to_reduction_layout(acc.layout()));

    if constexpr (IsInit) {
      fill(row_max, -INFINITY);
      clear(row_sum);
      fill(scores_scale, 1.f);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < size<0>(acc_rv); mi++) {
      float chunk_max = -INFINITY;
      CUTLASS_PRAGMA_UNROLL
      for (int ei = 0; ei < size<1, 0>(acc_rv); ei++) {
        chunk_max = fmaxf(chunk_max, acc_rv(mi, make_coord(ei, ni)));
      }
      float max_recv = __shfl_xor_sync(int32_t(-1), chunk_max, 1);
      chunk_max = fmaxf(chunk_max, max_recv);
      AbsMaxP(mi, ni) = chunk_max;

      float prev_max = row_max(mi);
      row_max(mi) = fmaxf(row_max(mi), chunk_max);
      row_max(mi) = reduce_row_max_from_pairs(row_max(mi));

      const float max_scaled =
          InfCheck ? (row_max(mi) == -INFINITY
                          ? 0.f
                          : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2))
                   : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2);

      if constexpr (!IsInit) {
        if (prev_max != row_max(mi)) {
          scores_scale(mi) = softmax_exp2<InfCheck>((prev_max - row_max(mi)) * softmax_scale_log2);
          row_sum(mi) *= scores_scale(mi);
          CUTLASS_PRAGMA_UNROLL
          for (int prev_ni = 0; prev_ni < ni; prev_ni++) {
            CUTLASS_PRAGMA_UNROLL
            for (int ei = 0; ei < size<1, 0>(acc_rv); ei++) {
              acc_rv(mi, make_coord(ei, prev_ni)) *= scores_scale(mi);
            }
            AbsMaxP(mi, prev_ni) *= scores_scale(mi);
          }
        }
      }

      CUTLASS_PRAGMA_UNROLL
      for (int ei = 0; ei < size<1, 0>(acc_rv); ei++) {
        float val = softmax_exp2<InfCheck>(acc_rv(mi, make_coord(ei, ni)) * softmax_scale_log2 -
                                           max_scaled);
        acc_rv(mi, make_coord(ei, ni)) = val;
        row_sum(mi) += val;
      }

      AbsMaxP(mi, ni) = softmax_exp2<InfCheck>(AbsMaxP(mi, ni) * softmax_scale_log2 - max_scaled +
                                               fp4_scale_log2);
    }
  }

 private:
  __device__ __forceinline__ static float ptx_exp2(float x) {
    float result;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
  }

  template <bool InfCheck>
  __device__ __forceinline__ static float softmax_exp2(float x) {
    if (x <= -126.0f) {
      return 0.0f;
    }
#if defined(SOFTMAX_FMA_EXP2)
    if constexpr (!InfCheck) {
      return nvfp4_attention::exp2_fma_poly(x);
    }
#endif
    return ptx_exp2(x);
  }
};

}  // namespace nvfp4_attention
