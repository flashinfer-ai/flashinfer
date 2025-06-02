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
#pragma once

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/int_tuple.h>
#include <tvm/runtime/ndarray.h>

using IdType = int32_t;
using tvm::ffi::Array;
using tvm::runtime::DataType;
using tvm::runtime::IntTuple;
using tvm::runtime::NDArray;

#define DISPATCH_BOOL(expr, const_expr, ...) \
  [&]() -> bool {                            \
    if (expr) {                              \
      constexpr bool const_expr = true;      \
      return __VA_ARGS__();                  \
    } else {                                 \
      constexpr bool const_expr = false;     \
      return __VA_ARGS__();                  \
    }                                        \
  }()
