/*
 * Copyright (c) 2024 by FlashInfer team.
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
#include "generated/dispatch.inc"
#include "pytorch_extension_utils.h"

#define DISPATCH_head_dim(expr, const_expr, ...) \
  _DISPATCH_SWITCH("head_dim", expr, _DISPATCH_CASES_head_dim(const_expr, __VA_ARGS__))

#define DISPATCH_head_dim_sm90(expr, const_expr, ...) \
  _DISPATCH_SWITCH("head_dim", expr, _DISPATCH_CASES_head_dim_sm90(const_expr, __VA_ARGS__))

#define DISPATCH_pos_encoding_mode(expr, const_expr, ...) \
  _DISPATCH_SWITCH("positional encoding mode", expr,      \
                   _DISPATCH_CASES_pos_encoding_mode(const_expr, __VA_ARGS__))

#define DISPATCH_use_fp16_qk_reduction(expr, const_expr, ...) \
  _DISPATCH_SWITCH("use_fp16_qk_reduction", expr,             \
                   _DISPATCH_CASES_use_fp16_qk_reduction(const_expr, __VA_ARGS__))

#define DISPATCH_mask_mode(expr, const_expr, ...) \
  _DISPATCH_SWITCH("mask_mode", expr, _DISPATCH_CASES_mask_mode(const_expr, __VA_ARGS__))

#define DISPATCH_PYTORCH_QKV_DTYPE_TO_CTYPE(q_dtype, kv_dtype, c_type_q, c_type_kv, ...) \
  [&]() -> bool {                                                                        \
    if (kv_dtype == q_dtype) {                                                           \
      return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q_dtype, c_type_q, [&] {               \
        using c_type_kv = c_type_q;                                                      \
        return __VA_ARGS__();                                                            \
      });                                                                                \
    } else {                                                                             \
      return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q_dtype, c_type_q, [&] {               \
        return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8(kv_dtype, c_type_kv,                  \
                                                   [&] { return __VA_ARGS__(); });       \
      });                                                                                \
    }                                                                                    \
  }()
