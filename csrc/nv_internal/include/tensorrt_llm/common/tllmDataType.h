/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>

namespace tensorrt_llm {
namespace common {

// Tensor element types, numbered as in TensorRT-LLM's tensorrt_llm::DataType.
enum class DataType : int32_t {
  kFLOAT = 0,
  kHALF = 1,
  kINT8 = 2,
  kINT32 = 3,
  kBOOL = 4,
  kUINT8 = 5,
  kFP8 = 6,
  kBF16 = 7,
  kINT64 = 8,
  kINT4 = 9,
  kFP4 = 10,
  kE8M0 = 11,
};

}  // namespace common

using common::DataType;

}  // namespace tensorrt_llm
