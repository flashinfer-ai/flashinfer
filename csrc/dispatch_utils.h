/*
 * Copyright (c) 2023 by FlashInfer team.
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
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <flashinfer/exception.h>

#include <flashinfer/layout.cuh>
#include <flashinfer/pos_enc.cuh>

#include "dispatch_type_code.h"
#include "generated/dispatch.inc"

using namespace flashinfer;

#define _DISPATCH_SWITCH(var_name, cond, ...)                                           \
  [&]() -> bool {                                                                       \
    switch (cond) {                                                                     \
      __VA_ARGS__                                                                       \
      default:                                                                          \
        std::ostringstream oss;                                                         \
        oss << __PRETTY_FUNCTION__ << " failed to dispatch " var_name " " << int(cond); \
        FLASHINFER_ERROR(oss.str());                                                    \
        return false;                                                                   \
    }                                                                                   \
  }()

#define _DISPATCH_CASE(case_expr, case_var, ...) \
  case case_expr: {                              \
    constexpr auto case_var = case_expr;         \
    return __VA_ARGS__();                        \
  }

#define DISPATCH_head_dim(expr, const_expr, ...) \
  _DISPATCH_SWITCH("head_dim", expr, _DISPATCH_CASES_head_dim(const_expr, __VA_ARGS__))

#define DISPATCH_pos_encoding_mode(expr, const_expr, ...) \
  _DISPATCH_SWITCH("positional encoding mode", expr,      \
                   _DISPATCH_CASES_pos_encoding_mode(const_expr, __VA_ARGS__))

#define DISPATCH_allow_fp16_qk_reduction(expr, const_expr, ...) \
  _DISPATCH_SWITCH("allow_fp16_qk_reduction", expr,             \
                   _DISPATCH_CASES_allow_fp16_qk_reduction(const_expr, __VA_ARGS__))

#define DISPATCH_mask_mode(expr, const_expr, ...) \
  _DISPATCH_SWITCH("mask_mode", expr, _DISPATCH_CASES_mask_mode(const_expr, __VA_ARGS__))

#define DISPATCH_BOOL(use_logits_soft_cap, USE_LOGITS_SOFT_CAP, ...) \
  [&]() -> bool {                                                    \
    if (use_logits_soft_cap) {                                       \
      constexpr bool USE_LOGITS_SOFT_CAP = true;                     \
      return __VA_ARGS__();                                          \
    } else {                                                         \
      constexpr bool USE_LOGITS_SOFT_CAP = false;                    \
      return __VA_ARGS__();                                          \
    }                                                                \
  }()
