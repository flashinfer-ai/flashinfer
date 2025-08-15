/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "pytorch_extension_utils.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/thop/thUtils.h"
// #include "tensorrt_llm/runtime/torchUtils.h"

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/cuda/EmptyTensor.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cstdint>
// namespace th = torch;

namespace torch_ext {

static int getExp(float v) {
  int vIntRepr;
  memcpy(&vIntRepr, &v, sizeof(vIntRepr));
  int expBits = (vIntRepr >> 23) & 0xff;
  return expBits - 127;
}

static int getMantissaBits(float v) {
  int vIntRepr;
  memcpy(&vIntRepr, &v, sizeof(vIntRepr));
  int mantissaBits = vIntRepr & 0x7fffff;
  return mantissaBits;
}

static bool getSign(float v) {
  int vIntRepr;
  memcpy(&vIntRepr, &v, sizeof(vIntRepr));
  return vIntRepr >> 31;
}

static float makeExpFloat(int expValue) {
  expValue += 127;
  expValue <<= 23;
  float vFloat;
  memcpy(&vFloat, &expValue, sizeof(vFloat));
  return vFloat;
}

/*
 * E2M1 to float
 * 0111 -> 6
 * 0110 -> 4
 * 0101 -> 3
 * 0100 -> 2
 * 0011 -> 1.5
 * 0010 -> 1
 * 0001 -> 0.5
 * 0000 -> 0
 */
static float const kE2M1ToFloatArray[] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
static float const kE2M1Array[] = {0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5};
static int const kE2M1Count = 8;

uint8_t floatToE2M1(float value) {
  float absValue = fabs(value);
  TORCH_CHECK_LT(absValue, 8.0f);
  uint8_t result = getSign(value) ? 8 : 0;
  int fp4AbsValue = kE2M1Count - 1;
  for (; fp4AbsValue > 0; --fp4AbsValue) {
    if (kE2M1Array[fp4AbsValue] < absValue) break;
    // Tie to even.
    if (kE2M1Array[fp4AbsValue] == absValue && !(fp4AbsValue & 1)) break;
  }
  result |= fp4AbsValue;
  return result;
}

float e2M1ToFloat(uint8_t value) {
  bool signBit = value & 8;
  auto absValue = value & 7;
  float result = kE2M1ToFloatArray[absValue];
  if (signBit) result = -result;
  return result;
}

// Given the rowIdx and colIdx in the unswizzled SFMatrix, compute the 1D offset in the swizzled
// SFMatrix. colIdx and totalCloumn should be in SFMatrix, not activation Matrix, so no sfVecSize
// needed.
int computeSFIndex(int rowIdx, int colIdx, int totalRow, int totalColumn,
                   tensorrt_llm::QuantizationSFLayout layout) {
  constexpr int kColumnGroup0Size = 4;
  constexpr int kRowGroup0Size = 32;
  constexpr int kRowGroup1Size = kRowGroup0Size * 4;

  // Swizzled layout is used as default layout.
  if (layout == tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4 ||
      layout == tensorrt_llm::QuantizationSFLayout::SWIZZLED_8x4) {
    // int paddedRow = PadUpFn(totalRow, 128);
    int paddedColumn = PadUpFn(totalColumn, 4);

    int columnIdxInGroup0 = colIdx % kColumnGroup0Size;
    int columnGroupIdx = colIdx / kColumnGroup0Size;
    constexpr int columnGroupStride = kColumnGroup0Size * kRowGroup1Size;

    int rowIdxInGroup0 = rowIdx % kRowGroup0Size;
    int rowIdxInGroup1 = rowIdx % kRowGroup1Size / kRowGroup0Size;
    int rowGroupIdx = rowIdx / kRowGroup1Size;
    constexpr int rowGroup1Stride = kColumnGroup0Size;
    constexpr int rowGroup0Stride = kColumnGroup0Size * rowGroup1Stride;
    int rowGroupStride = kRowGroup1Size * paddedColumn;

    return columnIdxInGroup0 + columnGroupIdx * columnGroupStride +
           rowIdxInGroup0 * rowGroup0Stride + rowIdxInGroup1 * rowGroup1Stride +
           rowGroupIdx * rowGroupStride;
  }
  // Linear layout is only used in E2M1AndUFP8SFScaleToFloatV2.
  else if (layout == tensorrt_llm::QuantizationSFLayout::LINEAR) {
    // no padding needed. totalColumn is multiple of kVecSize.
    return rowIdx * totalColumn + colIdx;
  } else {
    TLLM_THROW("Other layout not implemented yet.");
  }
}

// Interleave (and possibly pad) the weights block scaling factor.
// blockScale: [num_experts, rows, cols] or [rows, cols]
// Return: num_experts * pad_up(rows, 128) * pad_up(cols, 4)
at::Tensor BlockScaleInterleave(at::Tensor const& blockScale) {
  bool is_cuda = blockScale.device().is_cuda();
  if (is_cuda) {
    CHECK_INPUT_TYPE(blockScale, SF_DTYPE);
  } else {
    CHECK_CPU_INPUT(blockScale, SF_DTYPE);
  }
  auto blockScaleShape = blockScale.sizes();
  TORCH_CHECK(blockScaleShape.size() == 2 || blockScaleShape.size() == 3,
              "Block Scale should be 2D or 3D tensor.");
  auto num_experts = blockScaleShape.size() == 3 ? blockScaleShape[0] : 1;
  auto rows = blockScaleShape.size() == 3 ? blockScaleShape[1] : blockScaleShape[0];
  auto cols = blockScaleShape.size() == 3 ? blockScaleShape[2] : blockScaleShape[1];

  auto expert_out_size = tensorrt_llm::computeSwizzledLayoutSFSize(rows, cols);
  auto rows_padded = PadUpFn(rows, 128);
  auto cols_padded = PadUpFn(cols, 4);
  TORCH_CHECK(expert_out_size == rows_padded * cols_padded,
              "expert_out_size should be equal to rows_padded * cols_padded.");
  at::Tensor interleavedBlockScale =
      at::empty({expert_out_size * num_experts},
                at::dtype(SF_DTYPE).device(blockScale.device()).requires_grad(false));

  if (is_cuda) {
    const thread_local int smCount = tensorrt_llm::common::getMultiProcessorCount();
    auto stream = at::cuda::getCurrentCUDAStream(blockScale.get_device());
    tensorrt_llm::kernels::invokeBlockScaleInterleave(
        num_experts, rows, rows_padded, cols, cols_padded, blockScale.data_ptr<uint8_t>(),
        static_cast<uint8_t*>(interleavedBlockScale.data_ptr()), smCount, stream);
  } else {
    for (int eIdx = 0; eIdx < static_cast<int>(num_experts); eIdx++) {
      uint8_t* interleavedBlockScalePtr =
          static_cast<uint8_t*>(interleavedBlockScale.data_ptr()) + eIdx * expert_out_size;
      for (int rIdx = 0; rIdx < static_cast<int>(rows_padded); ++rIdx) {
        auto globalRowIdx = eIdx * rows + rIdx;
        uint8_t* blockScalePtr = blockScale.data_ptr<uint8_t>() + globalRowIdx * cols;
        for (int cIdx = 0; cIdx < static_cast<int>(cols_padded); ++cIdx) {
          uint8_t sf_ori = 0;
          if (rIdx < static_cast<int>(rows) && cIdx < static_cast<int>(cols)) {
            sf_ori = blockScalePtr[cIdx];
          }
          int sf_index = computeSFIndex(rIdx, cIdx, rows, cols,
                                        tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4);
          interleavedBlockScalePtr[sf_index] = sf_ori;
        }
      }
    }
  }

  return interleavedBlockScale;
}

// Reverse interleave the weights block scaling factor.
// blockScale: [num_experts, rows, cols] or [rows, cols]
// Note: rows and cols are the dimensions of the original unswizzled SFMatrix, so reshape input
// before passing into this function! Return: The same shape as blockScale
at::Tensor BlockScaleInterleaveReverse(at::Tensor const& blockScale) {
  bool is_cuda = blockScale.device().is_cuda();
  if (is_cuda) {
    CHECK_INPUT_TYPE(blockScale, SF_DTYPE);
  } else {
    CHECK_CPU_INPUT(blockScale, SF_DTYPE);
  }
  auto blockScaleShape = blockScale.sizes();
  TORCH_CHECK(blockScaleShape.size() == 2 || blockScaleShape.size() == 3,
              "Block Scale should be 2D or 3D tensor.");
  auto num_experts = blockScaleShape.size() == 3 ? blockScaleShape[0] : 1;
  auto rows = blockScaleShape.size() == 3 ? blockScaleShape[1] : blockScaleShape[0];
  auto cols = blockScaleShape.size() == 3 ? blockScaleShape[2] : blockScaleShape[1];
  TORCH_CHECK(rows % 128 == 0, "rows of Interleaved block scales should be multiple of 128.");
  TORCH_CHECK(cols % 4 == 0, "cols of Interleaved block scales should be multiple of 4.");
  auto expert_out_size = rows * cols;
  at::Tensor reversedBlockScale = at::empty(
      blockScaleShape, at::dtype(SF_DTYPE).device(blockScale.device()).requires_grad(false));

  if (is_cuda) {
    const thread_local int smCount = tensorrt_llm::common::getMultiProcessorCount();
    auto stream = at::cuda::getCurrentCUDAStream(blockScale.get_device());
    tensorrt_llm::kernels::invokeBlockScaleInterleaveReverse(
        num_experts, rows, cols, blockScale.data_ptr<uint8_t>(),
        static_cast<uint8_t*>(reversedBlockScale.data_ptr()), smCount, stream);
  } else {
    // index in the swizzled SFMatrix -> (eIdx, rIdx, cIdx) in the unswizzled SFMatrix
    std::map<int, std::array<int, 3>> identity;
    for (int eIdx = 0; eIdx < num_experts; eIdx++) {
      for (int rIdx = 0; rIdx < rows; ++rIdx) {
        for (int cIdx = 0; cIdx < cols; ++cIdx) {
          int sf_index = computeSFIndex(rIdx, cIdx, rows, cols,
                                        tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4);
          identity[eIdx * expert_out_size + sf_index] = std::array<int, 3>{eIdx, rIdx, cIdx};
        }
      }
    }
    uint8_t* blockScalePtr = static_cast<uint8_t*>(blockScale.data_ptr());
    for (int i = 0; i < expert_out_size * num_experts; i++) {
      auto loc_2d = identity[i];
      if (loc_2d[1] < rows && loc_2d[2] < cols) {
        uint8_t* reversedBlockScalePtr = reversedBlockScale.data_ptr<uint8_t>() +
                                         (loc_2d[0] * rows + loc_2d[1]) * cols + loc_2d[2];
        *reversedBlockScalePtr = blockScalePtr[i];
      }
    }
  }

  return reversedBlockScale;
}

at::Tensor E2M1AndUFP8SFScaleToFloat(at::Tensor valueE2M1, at::Tensor scaleFP8SF, int64_t sfVecSize,
                                     int64_t sfType) {
  CHECK_CPU_INPUT(valueE2M1, FLOAT4_E2M1X2);
  CHECK_CPU_INPUT(scaleFP8SF, SF_DTYPE);
  auto packedShape = valueE2M1.sizes();
  auto scaleShape = scaleFP8SF.sizes();
  TORCH_CHECK(packedShape.size() == 2, "valueE2M1 should be 2D tensor.");
  TORCH_CHECK(scaleShape.size() == 1, "scaleFP8SF should be 1D tensor.");
  at::Tensor floatTensor = at::zeros({packedShape[0], packedShape[1] * 2},
                                     at::dtype(at::ScalarType::Float).requires_grad(false));

  int hiddenDim = packedShape[1] * 2;
  int packedFp4HiddenDim = hiddenDim / 2;
  int groupsPerHiddenDim = hiddenDim / sfVecSize;

  for (size_t vIdx = 0; vIdx < static_cast<size_t>(packedShape[0]); ++vIdx) {
    for (int group = 0; group < groupsPerHiddenDim; ++group) {
      float* floatPtr = floatTensor.data_ptr<float>() + vIdx * hiddenDim + group * sfVecSize;
      uint8_t* packedFp4Ptr =
          valueE2M1.data_ptr<uint8_t>() + vIdx * packedFp4HiddenDim + group * sfVecSize / 2;
      uint8_t* scaleFP8SFPtr = scaleFP8SF.data_ptr<uint8_t>();
      uint8_t fp8Scale =
          scaleFP8SFPtr[computeSFIndex(vIdx, group, packedShape[0], groupsPerHiddenDim,
                                       tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4)];
      int scale = fp8Scale;
      if (sfType == 0) {
        scale -= 127;
      } else {
        scale >>= 3;
        scale -= 7;
      }
      float scaleFloat = makeExpFloat(scale);
      for (int i = 0; i < sfVecSize; ++i) {
        uint8_t packedFp4 = packedFp4Ptr[i / 2];
        if (i % 2 == 1) {
          packedFp4 >>= 4;
        }
        packedFp4 &= 0xf;
        float value = e2M1ToFloat(packedFp4) * scaleFloat;
        floatPtr[i] = value;
      }
    }
  }
  return floatTensor;
}

// Used by the (fp16 -> int4) quant layer + int4 gemm network.
at::Tensor E2M1AndUFP8SFScaleToFloatV2(at::Tensor valueE2M1, at::Tensor scaleFP8SF,
                                       std::optional<at::Tensor> globalScale, int64_t sfVecSize,
                                       int64_t sfType, bool isSfSwizzledLayout = true) {
  CHECK_CPU_INPUT(valueE2M1, FLOAT4_E2M1X2);
  CHECK_CPU_INPUT(scaleFP8SF, SF_DTYPE);
  auto packedShape = valueE2M1.sizes();
  auto scaleShape = scaleFP8SF.sizes();
  TORCH_CHECK(packedShape.size() == 2, "valueE2M1 should be 2D tensor.");
  TORCH_CHECK(scaleShape.size() == 1, "scaleFP8SF should be 1D tensor.");
  at::Tensor floatTensor = at::zeros({packedShape[0], packedShape[1] * 2},
                                     at::dtype(at::ScalarType::Float).requires_grad(false));

  // CHECK_CPU_INPUT(globalScale, at::ScalarType::Float);
  float globalScaleVal{1.0f};
  if (sfType == 1) {
    TORCH_CHECK(globalScale.has_value(), "globalScale is required when sfType is 1.");
    // CHECK_CPU_INPUT(globalScale.value(), at::kFloat32);
    globalScaleVal = globalScale->data_ptr<float>()[0];
  }

  int hiddenDim = packedShape[1] * 2;
  int packedFp4HiddenDim = hiddenDim / 2;
  int groupsPerHiddenDim = hiddenDim / sfVecSize;

  tensorrt_llm::QuantizationSFLayout layout =
      isSfSwizzledLayout ? tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4
                         : tensorrt_llm::QuantizationSFLayout::LINEAR;

  for (size_t vIdx = 0; vIdx < static_cast<size_t>(packedShape[0]); ++vIdx) {
    for (int group = 0; group < groupsPerHiddenDim; ++group) {
      float* floatPtr = floatTensor.data_ptr<float>() + vIdx * hiddenDim + group * sfVecSize;
      uint8_t* packedFp4Ptr =
          valueE2M1.data_ptr<uint8_t>() + vIdx * packedFp4HiddenDim + group * sfVecSize / 2;
      uint8_t* scaleFP8SFPtr = scaleFP8SF.data_ptr<uint8_t>();
      uint8_t fp8Scale =
          scaleFP8SFPtr[computeSFIndex(vIdx, group, packedShape[0], groupsPerHiddenDim, layout)];
      float scaleFloat;
      if (sfType == 0) {
        uint32_t tmp = uint32_t(fp8Scale) << 23;
        memcpy(&scaleFloat, &tmp, sizeof(scaleFloat));
      } else {
        scaleFloat = float(reinterpret_cast<__nv_fp8_e4m3&>(fp8Scale));
      }
      scaleFloat *= globalScaleVal;
      for (int i = 0; i < sfVecSize; ++i) {
        uint8_t packedFp4 = packedFp4Ptr[i / 2];
        if (i % 2 == 1) {
          packedFp4 >>= 4;
        }
        packedFp4 &= 0xf;
        float value = e2M1ToFloat(packedFp4) * scaleFloat;
        floatPtr[i] = value;
      }
    }
  }
  return floatTensor;
}

at::Tensor mxfp4_dequantize_host(at::Tensor weight, at::Tensor scale, int64_t group_size) {
  // weight (n, k / 2)
  // scale (n, k / group_size)

  CHECK_CPU_INPUT(weight, FLOAT4_E2M1X2);
  CHECK_CPU_INPUT(scale, SF_DTYPE);
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(scale.is_contiguous(), "scale must be contiguous");
  TORCH_CHECK(weight.numel() != 0, "weight should not be empty tensor");
  TORCH_CHECK(weight.dtype() == at::ScalarType::Byte, "Weight must be a packed int8 tensor");
  TORCH_CHECK(scale.dtype() == at::ScalarType::Byte, "Scale must be a int8 tensor");

  TORCH_CHECK(weight.size(0) == scale.size(0),
              "weight and scale must have the same number of rows");
  TORCH_CHECK(weight.size(1) * 2 == scale.size(1) * group_size,
              "weight and scale must have the same number of columns");

  uint8_t* weight_packed_ptr = weight.data_ptr<uint8_t>();
  __nv_fp8_e8m0* scale_ptr = reinterpret_cast<__nv_fp8_e8m0*>(scale.data_ptr<uint8_t>());

  int const n = weight.size(0);
  int const k = weight.size(1) * 2;

  at::Tensor dequant_weight =
      at::empty({n, k}, at::dtype(at::ScalarType::Float).device(at::kCPU).requires_grad(false));
  float* dequant_weight_ptr = dequant_weight.data_ptr<float>();

  float fp4_lut[] = {0.0, 0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
                     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};

  for (int packed_idx = 0; packed_idx < weight.numel(); ++packed_idx) {
    int8_t weight_packed_data = weight_packed_ptr[packed_idx];

    uint8_t weight_low_ = weight_packed_data & 0xF;
    uint8_t weight_high_ = (weight_packed_data & 0xF0) >> 4;

    float weight_low = fp4_lut[weight_low_];
    float weight_high = fp4_lut[weight_high_];

    int scale_n_idx = packed_idx / (k / 2);
    int scale_k_idx = ((packed_idx * 2) % k) / group_size;

    float scale_ = static_cast<float>(scale_ptr[scale_n_idx * scale.size(1) + scale_k_idx]);

    dequant_weight_ptr[2 * packed_idx] = weight_low * scale_;
    dequant_weight_ptr[2 * packed_idx + 1] = weight_high * scale_;
  }

  return dequant_weight;
}

}  // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("block_scale_interleave_sm100", &torch_ext::BlockScaleInterleave);
  m.def("block_scale_interleave_reverse_sm100", &torch_ext::BlockScaleInterleaveReverse);
  m.def("e2m1_and_ufp8sf_scale_to_float_sm100", &torch_ext::E2M1AndUFP8SFScaleToFloatV2);
  m.def("mxfp4_dequantize_host", &torch_ext::mxfp4_dequantize_host);
}
