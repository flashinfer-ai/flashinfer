/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/DeviceType.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include <algorithm>
#include <initializer_list>
#include <type_traits>
#include <vector>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

namespace tensorrt_llm::runtime {

class TorchUtils {
 public:
  static nvinfer1::DataType dataType(at::ScalarType scalarType) {
    switch (scalarType) {
      case at::ScalarType::Float:
        return nvinfer1::DataType::kFLOAT;
      case at::ScalarType::Half:
        return nvinfer1::DataType::kHALF;
      case torch::kInt8:
        return nvinfer1::DataType::kINT8;
      case torch::kUInt8:
        return nvinfer1::DataType::kUINT8;
      case torch::kInt32:
        return nvinfer1::DataType::kINT32;
      case torch::kInt64:
        return nvinfer1::DataType::kINT64;
      case at::ScalarType::Bool:
        return nvinfer1::DataType::kBOOL;
      case at::ScalarType::Float8_e4m3fn:
        return nvinfer1::DataType::kFP8;
      case at::ScalarType::BFloat16:
        return nvinfer1::DataType::kBF16;
      case at::ScalarType::QUInt4x2:
        return nvinfer1::DataType::kINT4;
      default:
        TLLM_THROW("unsupported data type");
    }
  }

 private:
  TorchUtils() = default;
};

}  // namespace tensorrt_llm::runtime
