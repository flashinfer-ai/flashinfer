/*
 * Copyright (c) 2025 by FlashInfer team.
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
// NOTE(Zihao): this is the include file for AOT mode
#pragma once
#include <flashinfer/attention/hopper/default_params.cuh>
#include <flashinfer/attention/hopper/variants.cuh>
#include <flashinfer/cutlass_utils.cuh>

#include "aot_default_additional_params.h"
#include "aot_extension_utils.h"

using IdType = int32_t;

#define ADDITIONAL_FUNC_PARAMS SINGLE_PREFILL_SM90_ADDITIONAL_FUNC_PARAMS
#define ADDITIONAL_PARAMS_SETTER SINGLE_PREFILL_SM90_ADDITIONAL_PARAMS_SETTER

#define DISPATCH_context(DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO,   \
                         USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, AttentionVariant, Params, ...) \
  {                                                                                              \
    DISPATCH_mask_mode(mask_mode, MASK_MODE, [&] {                                               \
      if (q_scalar_type != kv_scalar_type) {                                                     \
        return false;                                                                            \
      }                                                                                          \
      return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q_scalar_type, dtype_q, [&] {                  \
        using DTypeQ = cutlass_dtype_t<dtype_q>;                                                 \
        using DTypeKV = DTypeQ;                                                                  \
        using DTypeO = DTypeQ;                                                                   \
        using Params = SinglePrefillParams<DTypeQ, DTypeKV, DTypeO>;                             \
        return DISPATCH_head_dim_sm90(head_dim_qk, head_dim_vo, HEAD_DIM_QK, HEAD_DIM_VO, [&] {  \
          return DISPATCH_BOOL(window_left > -1, USE_SLIDING_WINDOW, [&] {                       \
            return DISPATCH_BOOL(logits_soft_cap > 0.f, USE_LOGITS_SOFT_CAP, [&] {               \
              using AttentionVariant = DefaultAttention<USE_LOGITS_SOFT_CAP>;                    \
              __VA_ARGS__();                                                                     \
              return true;                                                                       \
            });                                                                                  \
          });                                                                                    \
        });                                                                                      \
      });                                                                                        \
    });                                                                                          \
  }
