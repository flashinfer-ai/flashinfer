/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <initializer_list>

#include "tvm_ffi_utils.h"

// The frozen device program owns these standalone typedef names. Isolate them
// from the host headers while keeping the generated file as the sole device
// implementation in this translation unit.
#define uint8_t wan_hybrid_generated_uint8_t
#define uint16_t wan_hybrid_generated_uint16_t
#define uint32_t wan_hybrid_generated_uint32_t
#define uint64_t wan_hybrid_generated_uint64_t
#define int32_t wan_hybrid_generated_int32_t
#define int16_t wan_hybrid_generated_int16_t
#define WanHybridTensorMap wan_hybrid_generated_TensorMap
#define WanHybridTensorMapPack wan_hybrid_generated_TensorMapPack
#define CUtensorMap wan_hybrid_generated_CUtensorMap
#if FLASHINFER_WAN_HYBRID_TARGET_MINOR == 0
#include "device/wan_hybrid_quantize_value_sm100.cu"
#elif FLASHINFER_WAN_HYBRID_TARGET_MINOR == 3
#include "device/wan_hybrid_quantize_value_sm103.cu"
#else
#error "Wan hybrid quantization requires target minor 0 or 3"
#endif
#undef CUtensorMap
#undef WanHybridTensorMapPack
#undef WanHybridTensorMap
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t

namespace flashinfer {
namespace wan_hybrid {

constexpr int64_t kBatch = 1;
constexpr int64_t kSequence = 4800;
constexpr int64_t kHeads = 40;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kPaddedSequence = 5120;
constexpr int64_t kLogicalBlocks = 38;
constexpr int64_t kPhysicalBlocks = 40;
constexpr int64_t kValueRows = kBatch * kHeads * kHeadDim;
constexpr int64_t kPackedColumns = kPaddedSequence / 2;
constexpr int64_t kScaleRows = kBatch * kHeads * kPhysicalBlocks * 16;
constexpr int64_t kScaleColumns = 32;
constexpr int64_t kGridX = kBatch * kHeads * kLogicalBlocks;

static_assert(THREADS == 256);
static_assert(SMEM_TOTAL == 33280);

void CheckExactTensor(TensorView tensor, const char* name, DLDataType dtype,
                      std::initializer_list<int64_t> shape, int32_t device_id) {
  CHECK_INPUT(tensor);
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(tensor.dtype()), encode_dlpack_dtype(dtype))
      << name << " has the wrong dtype";
  TVM_FFI_ICHECK_EQ(tensor.ndim(), static_cast<int32_t>(shape.size()))
      << name << " has the wrong rank";
  int32_t axis = 0;
  for (int64_t extent : shape) {
    TVM_FFI_ICHECK_EQ(tensor.size(axis), extent)
        << name << " has the wrong extent at axis " << axis;
    ++axis;
  }
  TVM_FFI_ICHECK_EQ(tensor.device().device_id, device_id)
      << name << " must be on the same CUDA device as value";
}

void QuantizeValue(TensorView value, TensorView base, TensorView residual,
                   TensorView base_scale_lo, TensorView base_scale_hi,
                   TensorView residual_scale_lo, TensorView residual_scale_hi) {
  CHECK_INPUT_AND_TYPE(value, dl_bfloat16);
  const int32_t device_id = value.device().device_id;
  CheckExactTensor(value, "value", dl_bfloat16,
                   {kBatch, kSequence, kHeads, kHeadDim}, device_id);
  CheckExactTensor(base, "base", dl_uint8, {kValueRows, kPackedColumns}, device_id);
  CheckExactTensor(residual, "residual", dl_uint8,
                   {kValueRows, kPackedColumns}, device_id);
  CheckExactTensor(base_scale_lo, "base_scale_lo", dl_uint8,
                   {kScaleRows, kScaleColumns}, device_id);
  CheckExactTensor(base_scale_hi, "base_scale_hi", dl_uint8,
                   {kScaleRows, kScaleColumns}, device_id);
  CheckExactTensor(residual_scale_lo, "residual_scale_lo", dl_uint8,
                   {kScaleRows, kScaleColumns}, device_id);
  CheckExactTensor(residual_scale_hi, "residual_scale_hi", dl_uint8,
                   {kScaleRows, kScaleColumns}, device_id);

  ffi::CUDADeviceGuard device_guard(device_id);
  const cudaStream_t stream = get_stream(value.device());
  kernel_wan_hybrid_quantize_value<<<dim3(kGridX, 1, 1), dim3(THREADS, 1, 1),
                                     SMEM_TOTAL, stream>>>(
      static_cast<__nv_bfloat16*>(value.data_ptr()),
      static_cast<wan_hybrid_generated_uint8_t*>(base.data_ptr()),
      static_cast<wan_hybrid_generated_uint8_t*>(residual.data_ptr()),
      static_cast<wan_hybrid_generated_uint8_t*>(base_scale_lo.data_ptr()),
      static_cast<wan_hybrid_generated_uint8_t*>(base_scale_hi.data_ptr()),
      static_cast<wan_hybrid_generated_uint8_t*>(residual_scale_lo.data_ptr()),
      static_cast<wan_hybrid_generated_uint8_t*>(residual_scale_hi.data_ptr()),
      static_cast<wan_hybrid_generated_int32_t>(kHeads),
      static_cast<wan_hybrid_generated_int32_t>(kSequence),
      static_cast<wan_hybrid_generated_int32_t>(kPaddedSequence),
      static_cast<wan_hybrid_generated_int32_t>(kLogicalBlocks),
      static_cast<wan_hybrid_generated_int32_t>(kPhysicalBlocks));
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess)
      << "wan_hybrid_quantize_value launch failed";
}

}  // namespace wan_hybrid
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(wan_hybrid_quantize_value,
                              flashinfer::wan_hybrid::QuantizeValue);
