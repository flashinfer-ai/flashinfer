/*
 * Copyright (c) 2026 by FlashInfer team.
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

#include <flashinfer/conv3d_nvfp4_activation.cuh>

#include "tvm_ffi_utils.h"

namespace {

constexpr DLDataType kBFloat16 = DLDataType{kDLBfloat, 16, 1};
constexpr DLDataType kFloat32 = DLDataType{kDLFloat, 32, 1};
constexpr DLDataType kUInt8 = DLDataType{kDLUInt, 8, 1};

}  // namespace

void nvfp4_conv3d_quantize_activation(TensorView input, TensorView global_scale,
                                      TensorView packed_output, TensorView scale_output,
                                      int64_t pad_height, int64_t pad_width, int64_t tile_variant) {
  CHECK_INPUT_AND_TYPE(input, kBFloat16);
  CHECK_INPUT_AND_TYPE(global_scale, kFloat32);
  CHECK_INPUT_AND_TYPE(packed_output, kUInt8);
  CHECK_INPUT_AND_TYPE(scale_output, kUInt8);
  CHECK_DIM(5, input);
  CHECK_DIM(1, global_scale);
  CHECK_DIM(5, packed_output);
  CHECK_DIM(5, scale_output);
  CHECK_DEVICE(input, global_scale);
  CHECK_DEVICE(input, packed_output);
  CHECK_DEVICE(input, scale_output);

  TVM_FFI_ICHECK_EQ(global_scale.size(0), 1) << "global_scale must have shape (1,)";
  TVM_FFI_ICHECK(pad_height == 0 || pad_height == 1) << "pad_height must be zero or one";
  TVM_FFI_ICHECK(pad_width == 0 || pad_width == 1) << "pad_width must be zero or one";
  TVM_FFI_ICHECK_GE(tile_variant, 0);
  TVM_FFI_ICHECK_LE(tile_variant, 4);

  const int batch = static_cast<int>(input.size(0));
  const int channels = static_cast<int>(input.size(1));
  const int depth = static_cast<int>(input.size(2));
  const int height = static_cast<int>(input.size(3));
  const int width = static_cast<int>(input.size(4));
  TVM_FFI_ICHECK_GT(batch, 0);
  TVM_FFI_ICHECK_GT(depth, 0);
  TVM_FFI_ICHECK_GT(height, 0);
  TVM_FFI_ICHECK_GT(width, 0);

  const int channel_tile =
      flashinfer::conv3d_nvfp4::activation_channel_tile(static_cast<int>(tile_variant));
  TVM_FFI_ICHECK_GT(channel_tile, 0);
  TVM_FFI_ICHECK_EQ(channels % channel_tile, 0)
      << "channels must be divisible by the selected activation channel tile " << channel_tile;

  const int physical_height = height + 2 * static_cast<int>(pad_height);
  const int physical_width = width + 2 * static_cast<int>(pad_width);
  const int scale_groups = channels / flashinfer::conv3d_nvfp4::kScaleVectorSize;

  TVM_FFI_ICHECK_EQ(packed_output.size(0), batch);
  TVM_FFI_ICHECK_EQ(packed_output.size(1), depth);
  TVM_FFI_ICHECK_EQ(packed_output.size(2), physical_height);
  TVM_FFI_ICHECK_EQ(packed_output.size(3), physical_width);
  TVM_FFI_ICHECK_EQ(packed_output.size(4), channels / 2);
  TVM_FFI_ICHECK_EQ(scale_output.size(0), batch);
  TVM_FFI_ICHECK_EQ(scale_output.size(1), depth);
  TVM_FFI_ICHECK_EQ(scale_output.size(2), physical_height);
  TVM_FFI_ICHECK_EQ(scale_output.size(3), physical_width);
  TVM_FFI_ICHECK_EQ(scale_output.size(4), scale_groups);

  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  cudaStream_t stream = get_stream(input.device());
  cudaError_t status = flashinfer::conv3d_nvfp4::launch_activation_quantization(
      static_cast<const __nv_bfloat16*>(input.data_ptr()),
      static_cast<const float*>(global_scale.data_ptr()),
      static_cast<uint8_t*>(packed_output.data_ptr()),
      static_cast<uint8_t*>(scale_output.data_ptr()), batch, channels, depth, height, width, 0,
      static_cast<int>(pad_height), static_cast<int>(pad_width), static_cast<int>(tile_variant),
      stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "NVFP4 Conv3d activation quantization failed: " << cudaGetErrorString(status);
}
