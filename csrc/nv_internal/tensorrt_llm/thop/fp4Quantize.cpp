/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/thop/fp4Quantize.h"

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/cuda/EmptyTensor.h>
#include <cuda_fp16.h>

#include <cstdint>

#include "pytorch_extension_utils.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/thop/thUtils.h"

namespace torch_ext {
// self: [M, K], fp16/bf16/fp8_quantized
// globalScale: [1] float, = (448 * 6) / self.abs().max()
// nvfp4: sfVecSize = 16, sfUseUE8M0 = false
// mxfp4: sfVecSize = 32 (not supported yet), sfUseUE8M0 = true
// alignment: sfVecSize
// isSfSwizzledLayout: bool, if true, the scale factors are stored in swizzled layout, otherwise in linear layout.
// See FP4QuantizationSFLayout enum for more details about the two layouts.
// returns self_fp4, self_block_scale_factors
// self_fp4: [M, K / 2], FLOAT4_E2M1X2
// self_block_scale_factors: ceil(M / 128) * 128 * ceil(K / sfVecSize / 4) * 4, SF_DTYPE (UE4M3 or UE8M0)
std::tuple<at::Tensor, at::Tensor> fp4_quantize(
    at::Tensor const& self, at::Tensor const& globalScale, int64_t sfVecSize, bool sfUseUE8M0, bool isSfSwizzledLayout)
{
    CHECK_TH_CUDA(self);
    CHECK_CONTIGUOUS(self);
    CHECK_INPUT_TYPE(globalScale, c10::ScalarType::Float);
    TORCH_CHECK(sfVecSize == 16, "sfVecSize can only be 16");

    auto const& inputShape = self.sizes();
    auto const& rank = inputShape.size();

    TORCH_CHECK(rank >= 2, "Input should be >=2D tensor.");
    int64_t m = 1;
    for (size_t i = 0; i < rank - 1; i++)
    {
        m *= inputShape[i];
    }
    auto const k = inputShape[rank - 1];
    TORCH_CHECK(k % sfVecSize == 0);

    std::vector<int64_t> outputShape(inputShape.begin(), inputShape.end());
    outputShape[rank - 1] = k / 2;

    at::Tensor valueE2M1 = at::detail::empty_cuda(outputShape, FLOAT4_E2M1X2, self.device(), /* stride */ std::nullopt);

    int64_t SFSize = isSfSwizzledLayout ? tensorrt_llm::computeFP4SwizzledLayoutSFSize(m, k / sfVecSize)
                                        : tensorrt_llm::computeFP4LinearLayoutSFSize(m, k / sfVecSize);

    at::Tensor scaleFP8SF
        = at::detail::empty_cuda({SFSize}, SF_DTYPE, self.device(), /* stride */ std::nullopt); // 1D tensor

    const thread_local int mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();

    auto const layout = isSfSwizzledLayout ? tensorrt_llm::FP4QuantizationSFLayout::SWIZZLED
                                           : tensorrt_llm::FP4QuantizationSFLayout::LINEAR;

#define LAUNCH_FP4_QUANTIZE_KERNEL(T)                                                                                  \
    tensorrt_llm::kernels::invokeFP4Quantization(m, k, reinterpret_cast<T*>(self.data_ptr()),                          \
        globalScale.data_ptr<float>(), reinterpret_cast<int64_t*>(valueE2M1.data_ptr()),                               \
        reinterpret_cast<int32_t*>(scaleFP8SF.data_ptr()), sfUseUE8M0, layout, mMultiProcessorCount,                   \
        at::cuda::getCurrentCUDAStream(self.get_device()));

    if (self.scalar_type() == at::ScalarType::Half)
    {
        LAUNCH_FP4_QUANTIZE_KERNEL(half)
    }
    else if (self.scalar_type() == at::ScalarType::BFloat16)
    {
#ifdef ENABLE_BF16
        LAUNCH_FP4_QUANTIZE_KERNEL(__nv_bfloat16)
#else
        C10_THROW_ERROR(NotImplementedError, "BFloat16 must be enabled to quantize an bf16 tensor to fp4.");
#endif
    }
    else if (self.scalar_type() == at::ScalarType::Float8_e4m3fn)
    {
#ifdef ENABLE_FP8
        LAUNCH_FP4_QUANTIZE_KERNEL(__nv_fp8_e4m3)
#else
        C10_THROW_ERROR(NotImplementedError, "FP8 must be enabled to quantize an fp8 tensor to fp4.");
#endif
    }
    else
    {
        C10_THROW_ERROR(NotImplementedError, "fp4_quantize only supports input tensor with dtypes fp16/bf16/e4m3.");
    }

#undef LAUNCH_FP4_QUANTIZE_KERNEL

    return {valueE2M1, scaleFP8SF};
}

// at::Tensor block_scale_interleave(at::Tensor const& blockScale) {
//   CHECK_INPUT_TYPE(blockScale, c10::ScalarType::Byte);
//   auto blockScaleShape = blockScale.sizes();
//   TORCH_CHECK(blockScaleShape.size() == 2 || blockScaleShape.size() == 3,
//               "Block Scale should be 2D or 3D tensor.");
//   auto num_experts = blockScaleShape.size() == 3 ? blockScaleShape[0] : 1;
//   auto rows = blockScaleShape.size() == 3 ? blockScaleShape[1] : blockScaleShape[0];
//   auto cols = blockScaleShape.size() == 3 ? blockScaleShape[2] : blockScaleShape[1];

//   auto expert_out_size = tensorrt_llm::computeSwizzledLayoutSFSize(rows, cols);
//   auto rows_padded = PadUpFn(rows, 128);
//   auto cols_padded = PadUpFn(cols, 4);
//   TORCH_CHECK(expert_out_size == rows_padded * cols_padded,
//               "expert_out_size should be equal to rows_padded * cols_padded.");
//   at::Tensor interleavedBlockScale =
//       at::empty({expert_out_size * num_experts},
//                 at::dtype(c10::ScalarType::Byte).device(blockScale.device()).requires_grad(false));

//   const thread_local int smCount = tensorrt_llm::common::getMultiProcessorCount();
//   auto stream = at::cuda::getCurrentCUDAStream(blockScale.get_device());
//   tensorrt_llm::kernels::invokeBlockScaleInterleave(
//       num_experts, rows, rows_padded, cols, cols_padded, blockScale.data_ptr<uint8_t>(),
//       static_cast<uint8_t*>(interleavedBlockScale.data_ptr()), smCount, stream);

//   return interleavedBlockScale;
// }

// static float const kE2M1ToFloatArray[] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};

// float e2M1ToFloat(uint8_t value) {
//   bool signBit = value & 8;
//   auto absValue = value & 7;
//   float result = kE2M1ToFloatArray[absValue];
//   if (signBit) result = -result;
//   return result;
// }

// int computeSFIndex(int rowIdx, int colIdx, int totalRow, int totalColumn,
//                    tensorrt_llm::QuantizationSFLayout layout) {
//   constexpr int kColumnGroup0Size = 4;
//   constexpr int kRowGroup0Size = 32;
//   constexpr int kRowGroup1Size = kRowGroup0Size * 4;

//   // Swizzled layout is used as default layout.
//   if (layout == tensorrt_llm::QuantizationSFLayout::SWIZZLED) {
//     // int paddedRow = PadUpFn(totalRow, 128);
//     int paddedColumn = PadUpFn(totalColumn, 4);

//     int columnIdxInGroup0 = colIdx % kColumnGroup0Size;
//     int columnGroupIdx = colIdx / kColumnGroup0Size;
//     constexpr int columnGroupStride = kColumnGroup0Size * kRowGroup1Size;

//     int rowIdxInGroup0 = rowIdx % kRowGroup0Size;
//     int rowIdxInGroup1 = rowIdx % kRowGroup1Size / kRowGroup0Size;
//     int rowGroupIdx = rowIdx / kRowGroup1Size;
//     constexpr int rowGroup1Stride = kColumnGroup0Size;
//     constexpr int rowGroup0Stride = kColumnGroup0Size * rowGroup1Stride;
//     int rowGroupStride = kRowGroup1Size * paddedColumn;

//     return columnIdxInGroup0 + columnGroupIdx * columnGroupStride +
//            rowIdxInGroup0 * rowGroup0Stride + rowIdxInGroup1 * rowGroup1Stride +
//            rowGroupIdx * rowGroupStride;
//   }
//   // Linear layout is only used in E2M1AndUFP8SFScaleToFloatV2.
//   else if (layout == tensorrt_llm::QuantizationSFLayout::LINEAR) {
//     // no padding needed. totalColumn is multiple of kVecSize.
//     return rowIdx * totalColumn + colIdx;
//   } else {
//     TORCH_CHECK(false, "Other layout not implemented yet.");
//   }
// }

// at::Tensor e2m1_and_ufp8sf_scale_to_float(at::Tensor valueE2M1, at::Tensor scaleFP8SF,
//                                           std::optional<at::Tensor> globalScale, int64_t sfVecSize,
//                                           int64_t sfType, bool isSfSwizzledLayout) {
//   CHECK_CONTIGUOUS(valueE2M1);
//   CHECK_CONTIGUOUS(scaleFP8SF);
//   CHECK_INPUT_TYPE(valueE2M1, c10::ScalarType::Byte);
//   CHECK_INPUT_TYPE(scaleFP8SF, c10::ScalarType::Byte);
//   if (sfType == 1) {
//     TORCH_CHECK(globalScale.has_value(), "globalScale is required for NvFp4");
//     CHECK_INPUT_TYPE(globalScale.value(), c10::ScalarType::Float);
//     TORCH_CHECK(globalScale.value().device().is_cpu(), "globalScale must be CPU tensor");
//   }
//   TORCH_CHECK(valueE2M1.device().is_cpu(), "valueE2M1 must be CPU tensor");
//   TORCH_CHECK(scaleFP8SF.device().is_cpu(), "scaleFP8SF must be CPU tensor");

//   auto packedShape = valueE2M1.sizes();
//   auto scaleShape = scaleFP8SF.sizes();
//   TORCH_CHECK(packedShape.size() == 2, "valueE2M1 should be 2D tensor.");
//   TORCH_CHECK(scaleShape.size() == 1, "scaleFP8SF should be 1D tensor.");
//   at::Tensor floatTensor = at::zeros({packedShape[0], packedShape[1] * 2},
//                                      at::dtype(at::ScalarType::Float).requires_grad(false));
//   float globalScaleVal = sfType == 1 ? globalScale.value().data_ptr<float>()[0] : 1.0f;

//   int hiddenDim = packedShape[1] * 2;
//   int packedFp4HiddenDim = hiddenDim / 2;
//   int groupsPerHiddenDim = hiddenDim / sfVecSize;

//   auto layout = isSfSwizzledLayout ? tensorrt_llm::QuantizationSFLayout::SWIZZLED
//                                    : tensorrt_llm::QuantizationSFLayout::LINEAR;

//   for (size_t vIdx = 0; vIdx < static_cast<size_t>(packedShape[0]); ++vIdx) {
//     for (int group = 0; group < groupsPerHiddenDim; ++group) {
//       float* floatPtr = floatTensor.data_ptr<float>() + vIdx * hiddenDim + group * sfVecSize;
//       uint8_t* packedFp4Ptr =
//           valueE2M1.data_ptr<uint8_t>() + vIdx * packedFp4HiddenDim + group * sfVecSize / 2;
//       uint8_t* scaleFP8SFPtr = scaleFP8SF.data_ptr<uint8_t>();

//       int sf_index = computeSFIndex(vIdx, group, packedShape[0], groupsPerHiddenDim, layout);
//       uint8_t fp8Scale = scaleFP8SFPtr[sf_index];
//       float scaleFloat;
//       if (sfType == 0) {
//         uint32_t tmp = uint32_t(fp8Scale) << 23;
//         memcpy(&scaleFloat, &tmp, sizeof(scaleFloat));
//       } else {
//         scaleFloat = static_cast<float>(reinterpret_cast<__nv_fp8_e4m3&>(fp8Scale));
//       }

//       scaleFloat *= globalScaleVal;
//       for (int i = 0; i < sfVecSize; ++i) {
//         uint8_t packedFp4 = packedFp4Ptr[i / 2];
//         if (i % 2 == 1) {
//           packedFp4 >>= 4;
//         }
//         packedFp4 &= 0xf;
//         float value = e2M1ToFloat(packedFp4) * scaleFloat;
//         floatPtr[i] = value;
//       }
//     }
//   }
//   return floatTensor;
// }
}  // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("fp4_quantize", &torch_ext::fp4_quantize);
  // m.def("block_scale_interleave", &torch_ext::block_scale_interleave);
  // m.def("e2m1_and_ufp8sf_scale_to_float", &torch_ext::e2m1_and_ufp8sf_scale_to_float);
}
