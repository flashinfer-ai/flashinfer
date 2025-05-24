/*
 * Copyright (c) 1993-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "flashinfer/comm/trtllm/common/logger.h"
// #include <NvInferRuntime.h>
#include <map>

#include "flashinfer/comm/trtllm/types.h"

namespace tensorrt_llm::common {

constexpr static size_t getDTypeSize(DataType type) {
  switch (type) {
    case DataType::kINT64:
      return 8;
    case DataType::kINT32:
      [[fallthrough]];
    case DataType::kFLOAT:
      return 4;
    case DataType::kBF16:
      [[fallthrough]];
    case DataType::kHALF:
      return 2;
    case DataType::kBOOL:
      [[fallthrough]];
    case DataType::kUINT8:
      [[fallthrough]];
    case DataType::kINT8:
      [[fallthrough]];
    case DataType::kFP8:
      return 1;
    case DataType::kINT4:
      TLLM_THROW("Cannot determine size of INT4 data type");
    case DataType::kFP4:
      TLLM_THROW("Cannot determine size of FP4 data type");
    default:
      TLLM_THROW("Unknown dtype %d", static_cast<int>(type));
  }
  return 0;
}

constexpr static size_t getDTypeSizeInBits(DataType type) {
  switch (type) {
    case DataType::kINT64:
      return 64;
    case DataType::kINT32:
      [[fallthrough]];
    case DataType::kFLOAT:
      return 32;
    case DataType::kBF16:
      [[fallthrough]];
    case DataType::kHALF:
      return 16;
    case DataType::kBOOL:
      [[fallthrough]];
    case DataType::kUINT8:
      [[fallthrough]];
    case DataType::kINT8:
      [[fallthrough]];
    case DataType::kFP8:
      return 8;
    case DataType::kINT4:
      [[fallthrough]];
    case DataType::kFP4:
      return 4;
    default:
      TLLM_THROW("Unknown dtype %d", static_cast<int>(type));
  }
  return 0;
}

[[maybe_unused]] static std::string getDtypeString(DataType type) {
  switch (type) {
    case DataType::kFP32:
      return "fp32";
      break;
    case DataType::kFP16:
      return "fp16";
      break;
    case DataType::kINT8:
      return "int8";
      break;
    case DataType::kINT32:
      return "int32";
      break;
    case DataType::kBOOL:
      return "bool";
      break;
    case DataType::kUINT8:
      return "uint8";
      break;
    case DataType::kFP8:
      return "fp8";
      break;
    case DataType::kBF16:
      return "bf16";
      break;
    case DataType::kINT64:
      return "int64";
      break;
    case DataType::kINT4:
      return "int4";
      break;
    case DataType::kFP4:
      return "fp4";
      break;
    default:
      throw std::runtime_error("Unsupported data type");
      break;
  }

  return "";
}

}  // namespace tensorrt_llm::common
