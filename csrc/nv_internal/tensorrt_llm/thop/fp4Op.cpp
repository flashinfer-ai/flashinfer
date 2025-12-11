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
#include <cuda_fp16.h>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/thop/utils.h"
#include "tvm/ffi/container/array.h"

#if CUDA_VERSION >= 12080
#include <cuda_fp4.h>
#endif
#include <cuda_fp8.h>
#include <tvm/ffi/container/map.h>

#include <cmath>
#include <cstdint>

using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::Optional;

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
  TVM_FFI_ICHECK_LT(absValue, 8.0f);
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

template <typename T>
void blockScaleInterleaveHost(TensorView blockScale, TensorView interleavedBlockScale) {
  auto blockScaleShape = blockScale.sizes();
  auto num_experts = blockScaleShape.size() == 3 ? blockScaleShape[0] : 1;
  auto rows = blockScaleShape.size() == 3 ? blockScaleShape[1] : blockScaleShape[0];
  auto cols = blockScaleShape.size() == 3 ? blockScaleShape[2] : blockScaleShape[1];

  auto expert_out_size = tensorrt_llm::computeSwizzledLayoutSFSize(rows, cols);
  auto rows_padded = PadUpFn(rows, 128);
  auto cols_padded = PadUpFn(cols, 4);

  for (int eIdx = 0; eIdx < static_cast<int>(num_experts); eIdx++) {
    T* interleavedBlockScalePtr =
        static_cast<T*>(interleavedBlockScale.data_ptr()) + eIdx * expert_out_size;
    for (int rIdx = 0; rIdx < static_cast<int>(rows_padded); ++rIdx) {
      auto globalRowIdx = eIdx * rows + rIdx;
      T* blockScalePtr = static_cast<T*>(blockScale.data_ptr()) + globalRowIdx * cols;
      for (int cIdx = 0; cIdx < static_cast<int>(cols_padded); ++cIdx) {
        T sf_ori = 0;
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

template void blockScaleInterleaveHost<uint8_t>(TensorView blockScale,
                                                TensorView interleavedBlockScale);
template void blockScaleInterleaveHost<__nv_bfloat16>(TensorView blockScale,
                                                      TensorView interleavedBlockScale);

// Interleave (and possibly pad) the weights block scaling factor.
// blockScale: [num_experts, rows, cols] or [rows, cols]
// Return: num_experts * pad_up(rows, 128) * pad_up(cols, 4)
void BlockScaleInterleave(TensorView blockScale, TensorView interleavedBlockScale) {
  bool is_cuda = (blockScale.device().device_type == kDLCUDA);
  if (is_cuda) {
    CHECK_CUDA(blockScale);
  } else {
    CHECK_CPU(blockScale);
  }
  CHECK_CONTIGUOUS(blockScale);
  TVM_FFI_ICHECK(blockScale.dtype() == dl_uint8 || blockScale.dtype() == dl_bfloat16)
      << "Block Scale must be uint8 or bfloat16.";
  auto blockScaleShape = blockScale.sizes();
  TVM_FFI_ICHECK(blockScaleShape.size() == 2 || blockScaleShape.size() == 3)
      << "Block Scale should be 2D or 3D tensor.";
  auto num_experts = blockScaleShape.size() == 3 ? blockScaleShape[0] : 1;
  auto rows = blockScaleShape.size() == 3 ? blockScaleShape[1] : blockScaleShape[0];
  auto cols = blockScaleShape.size() == 3 ? blockScaleShape[2] : blockScaleShape[1];

  auto expert_out_size = tensorrt_llm::computeSwizzledLayoutSFSize(rows, cols);
  auto rows_padded = PadUpFn(rows, 128);
  auto cols_padded = PadUpFn(cols, 4);
  TVM_FFI_ICHECK(expert_out_size == rows_padded * cols_padded)
      << "expert_out_size should be equal to rows_padded * cols_padded.";

  if (is_cuda) {
    const thread_local int smCount = tensorrt_llm::common::getMultiProcessorCount();
    const cudaStream_t stream = get_stream(blockScale.device());

    if (blockScale.dtype() == dl_uint8) {
      tensorrt_llm::kernels::invokeBlockScaleInterleave(
          num_experts, rows, rows_padded, cols, cols_padded,
          static_cast<uint8_t*>(blockScale.data_ptr()),
          static_cast<uint8_t*>(interleavedBlockScale.data_ptr()), smCount, stream);
    } else if (blockScale.dtype() == dl_bfloat16) {
      tensorrt_llm::kernels::invokeBlockScaleInterleave(
          num_experts, rows, rows_padded, cols, cols_padded,
          static_cast<__nv_bfloat16*>(blockScale.data_ptr()),
          static_cast<__nv_bfloat16*>(interleavedBlockScale.data_ptr()), smCount, stream);
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError)
          << "block_scale_interleave only supports uint8 and bfloat16.";
    }
  } else {
    if (blockScale.dtype() == dl_uint8) {
      blockScaleInterleaveHost<uint8_t>(blockScale, interleavedBlockScale);
    } else if (blockScale.dtype() == dl_bfloat16) {
      blockScaleInterleaveHost<__nv_bfloat16>(blockScale, interleavedBlockScale);
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError)
          << "blockScaleInterleaveHost only supports uint8 and bfloat16.";
    }
  }
}

// Reverse interleave the weights block scaling factor.
// blockScale: [num_experts, rows, cols] or [rows, cols]
// Note: rows and cols are the dimensions of the original unswizzled SFMatrix, so reshape input
// before passing into this function! Return: The same shape as blockScale
void BlockScaleInterleaveReverse(TensorView const& blockScale, TensorView reversedBlockScale) {
  bool is_cuda = (blockScale.device().device_type == kDLCUDA);
  if (is_cuda) {
    CHECK_CUDA(blockScale);
  } else {
    CHECK_CPU(blockScale);
  }
  CHECK_CONTIGUOUS(blockScale);
  CHECK_INPUT_TYPE(blockScale, dl_uint8);
  auto blockScaleShape = blockScale.sizes();
  TVM_FFI_ICHECK(blockScaleShape.size() == 2 || blockScaleShape.size() == 3)
      << "Block Scale should be 2D or 3D tensor.";
  auto num_experts = blockScaleShape.size() == 3 ? blockScaleShape[0] : 1;
  auto rows = blockScaleShape.size() == 3 ? blockScaleShape[1] : blockScaleShape[0];
  auto cols = blockScaleShape.size() == 3 ? blockScaleShape[2] : blockScaleShape[1];
  TVM_FFI_ICHECK(rows % 128 == 0) << "rows of Interleaved block scales should be multiple of 128.";
  TVM_FFI_ICHECK(cols % 4 == 0) << "cols of Interleaved block scales should be multiple of 4.";
  auto expert_out_size = rows * cols;

  if (is_cuda) {
    const thread_local int smCount = tensorrt_llm::common::getMultiProcessorCount();
    const cudaStream_t stream = get_stream(blockScale.device());
    tensorrt_llm::kernels::invokeBlockScaleInterleaveReverse(
        num_experts, rows, cols, static_cast<uint8_t*>(blockScale.data_ptr()),
        static_cast<uint8_t*>(reversedBlockScale.data_ptr()), smCount, stream);
  } else {
    // index in the swizzled SFMatrix -> (eIdx, rIdx, cIdx) in the unswizzled SFMatrix
    Map<int, Array<int>> identity;
    for (int eIdx = 0; eIdx < num_experts; eIdx++) {
      for (int rIdx = 0; rIdx < rows; ++rIdx) {
        for (int cIdx = 0; cIdx < cols; ++cIdx) {
          int sf_index = computeSFIndex(rIdx, cIdx, rows, cols,
                                        tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4);
          identity.Set(eIdx * expert_out_size + sf_index, Array({eIdx, rIdx, cIdx}));
        }
      }
    }
    uint8_t* blockScalePtr = static_cast<uint8_t*>(blockScale.data_ptr());
    for (int i = 0; i < expert_out_size * num_experts; i++) {
      auto loc_2d = identity[i];
      if (loc_2d[1] < rows && loc_2d[2] < cols) {
        uint8_t* reversedBlockScalePtr = static_cast<uint8_t*>(reversedBlockScale.data_ptr()) +
                                         (loc_2d[0] * rows + loc_2d[1]) * cols + loc_2d[2];
        *reversedBlockScalePtr = blockScalePtr[i];
      }
    }
  }
}

// Used by the (fp16 -> int4) quant layer + int4 gemm network.
void E2M1AndUFP8SFScaleToFloatV2(TensorView valueE2M1, TensorView scaleFP8SF,
                                 Optional<TensorView> globalScale, TensorView floatTensorView,
                                 int64_t sfVecSize, int64_t sfType,
                                 bool isSfSwizzledLayout = true) {
  CHECK_CPU_INPUT(valueE2M1, dl_uint8);
  CHECK_CPU_INPUT(scaleFP8SF, dl_uint8);
  auto packedShape = valueE2M1.sizes();
  auto scaleShape = scaleFP8SF.sizes();
  TVM_FFI_ICHECK_EQ(packedShape.size(), 2) << "valueE2M1 should be 2D tensor.";
  TVM_FFI_ICHECK_EQ(scaleShape.size(), 1) << "scaleFP8SF should be 1D tensor.";

  float globalScaleVal{1.0f};
  if (sfType == 1) {
    TVM_FFI_ICHECK(globalScale.has_value()) << "globalScale is required when sfType is 1.";
    globalScaleVal = static_cast<float*>(globalScale.value().data_ptr())[0];
  }

  int hiddenDim = packedShape[1] * 2;
  int packedFp4HiddenDim = hiddenDim / 2;
  int groupsPerHiddenDim = hiddenDim / sfVecSize;
  tensorrt_llm::QuantizationSFLayout layout =
      isSfSwizzledLayout ? tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4
                         : tensorrt_llm::QuantizationSFLayout::LINEAR;

  for (size_t vIdx = 0; vIdx < static_cast<size_t>(packedShape[0]); ++vIdx) {
    for (int group = 0; group < groupsPerHiddenDim; ++group) {
      float* floatPtr =
          static_cast<float*>(floatTensorView.data_ptr()) + vIdx * hiddenDim + group * sfVecSize;
      uint8_t* packedFp4Ptr = static_cast<uint8_t*>(valueE2M1.data_ptr()) +
                              vIdx * packedFp4HiddenDim + group * sfVecSize / 2;
      uint8_t* scaleFP8SFPtr = static_cast<uint8_t*>(scaleFP8SF.data_ptr());
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
}

void mxfp4_dequantize_host(TensorView weight, TensorView scale, TensorView dequant_weight,
                           int64_t group_size) {
  // weight (n, k / 2)
  // scale (n, k / group_size)

  CHECK_CPU_INPUT(weight, dl_uint8);
  CHECK_CPU_INPUT(scale, dl_uint8);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(scale);
  TVM_FFI_ICHECK_NE(weight.numel(), 0) << "weight should not be empty tensor";
  CHECK_INPUT_TYPE(weight, dl_uint8);
  CHECK_INPUT_TYPE(scale, dl_uint8);

  int const n = weight.size(0);
  int const k = weight.size(1) * 2;

  TVM_FFI_ICHECK_EQ(n, scale.size(0)) << "weight and scale must have the same number of rows";
  TVM_FFI_ICHECK_EQ(k, scale.size(1) * group_size)
      << "weight and scale must have the same number of columns";

#if CUDA_VERSION >= 12080
  uint8_t* weight_packed_ptr = static_cast<uint8_t*>(weight.data_ptr());
  __nv_fp8_e8m0* scale_ptr =
      reinterpret_cast<__nv_fp8_e8m0*>(static_cast<uint8_t*>(scale.data_ptr()));
  float* dequant_weight_ptr = static_cast<float*>(dequant_weight.data_ptr());
  float fp4_lut[] = {0.0, 0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
                     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};

  const auto num_packed_elements = weight.numel();
  for (int packed_idx = 0; packed_idx < num_packed_elements; ++packed_idx) {
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
#else
  // NOTE(Zihao): __nv_fp8_e8m0 is not supported for CUDA < 12.8
  TLLM_THROW("mxfp4_dequantize_host is not supported for CUDA < 12.8");
#endif
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(block_scale_interleave_sm100, BlockScaleInterleave);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(block_scale_interleave_reverse_sm100, BlockScaleInterleaveReverse);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(e2m1_and_ufp8sf_scale_to_float_sm100, E2M1AndUFP8SFScaleToFloatV2);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mxfp4_dequantize_host, mxfp4_dequantize_host);
