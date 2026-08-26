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

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
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
#include "device/wan_hybrid_attention_sm100.cu"
#elif FLASHINFER_WAN_HYBRID_TARGET_MINOR == 3
#include "device/wan_hybrid_attention_sm103.cu"
#else
#error "Wan hybrid attention requires target minor 0 or 3"
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
constexpr int64_t kValueRows = kHeads * kHeadDim;
constexpr int64_t kPackedValueRows = 2 * kValueRows;
constexpr int64_t kPackedValueColumns = kPaddedSequence / 2;
constexpr int64_t kScaleRows = 25'600;
constexpr int64_t kScaleColumns = 32;
constexpr int64_t kValueScaleRows = 2 * kScaleRows;
constexpr int64_t kTensorMapCount = 6;
constexpr int64_t kTensorMapBytes = 128;
constexpr int64_t kThreads = 512;
constexpr int64_t kMaximumTiles = 147;
constexpr size_t kDynamicSmemBytes = 231'424;

static_assert(SMEM_TOTAL == kDynamicSmemBytes);
static_assert(sizeof(CUtensorMap) == kTensorMapBytes);
static_assert(sizeof(wan_hybrid_generated_TensorMap) == kTensorMapBytes);

void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK_EQ(status, cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

void CheckDriver(CUresult status, const char* operation) {
  TVM_FFI_ICHECK_EQ(status, CUDA_SUCCESS)
      << operation << " failed with CUresult=" << static_cast<int>(status);
}

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
      << name << " must be on the same CUDA device as q";
}

void CheckTarget(int32_t device_id) {
  int major = 0;
  int minor = 0;
  CheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
            "cudaDeviceGetAttribute(major)");
  CheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
            "cudaDeviceGetAttribute(minor)");
  TVM_FFI_ICHECK_EQ(major, 10) << "wan_hybrid attention requires compute capability 10.x";
  TVM_FFI_ICHECK_EQ(minor, FLASHINFER_WAN_HYBRID_TARGET_MINOR)
      << "wan_hybrid attention module target does not match the CUDA device";
}

CUtensorMap EncodeTensorMap(const void* address, CUtensorMapDataType data_type, uint32_t rank,
                            const uint64_t* global_dims, const uint64_t* global_strides,
                            const uint32_t* box_dims, CUtensorMapSwizzle swizzle,
                            const char* name) {
  std::array<uint32_t, 5> element_strides{1, 1, 1, 1, 1};
  CUtensorMap tensor_map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &tensor_map, data_type, rank, const_cast<void*>(address), global_dims, global_strides,
      box_dims, element_strides.data(), CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  CheckDriver(result, name);
  return tensor_map;
}

CUtensorMap EncodeNHD(const TensorView& tensor, uint32_t rows_per_box, const char* name) {
  constexpr uint64_t global_dims[5] = {64, kSequence, kHeads, kBatch, 2};
  constexpr uint64_t global_strides[4] = {
      kHeads * kHeadDim * sizeof(__nv_bfloat16),
      kHeadDim * sizeof(__nv_bfloat16),
      kSequence * kHeads * kHeadDim * sizeof(__nv_bfloat16),
      64 * sizeof(__nv_bfloat16),
  };
  const uint32_t box_dims[5] = {64, rows_per_box, 1, 1, 2};
  return EncodeTensorMap(tensor.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, global_dims,
                         global_strides, box_dims, CU_TENSOR_MAP_SWIZZLE_128B, name);
}

CUtensorMap Encode2D(const TensorView& tensor, uint64_t columns, uint64_t rows,
                     uint32_t box_columns, uint32_t box_rows, CUtensorMapSwizzle swizzle,
                     const char* name) {
  const uint64_t global_dims[2] = {columns, rows};
  const uint64_t global_strides[1] = {columns};
  const uint32_t box_dims[2] = {box_columns, box_rows};
  return EncodeTensorMap(tensor.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, global_dims,
                         global_strides, box_dims, swizzle, name);
}

void PrepareTensorMaps(const TensorView& q, const TensorView& k, const TensorView& vt,
                       const TensorView& sfvt_lo, const TensorView& sfvt_hi, const TensorView& out,
                       const TensorView& descriptor_storage) {
  std::array<CUtensorMap, kTensorMapCount> maps{
      EncodeNHD(q, 128, "cuTensorMapEncodeTiled(q)"),
      EncodeNHD(k, 128, "cuTensorMapEncodeTiled(k)"),
      Encode2D(vt, kPackedValueColumns, kPackedValueRows, 64, 128, CU_TENSOR_MAP_SWIZZLE_64B,
               "cuTensorMapEncodeTiled(vt)"),
      Encode2D(sfvt_lo, kScaleColumns, kValueScaleRows, 32, 16, CU_TENSOR_MAP_SWIZZLE_NONE,
               "cuTensorMapEncodeTiled(sfvt_lo)"),
      Encode2D(sfvt_hi, kScaleColumns, kValueScaleRows, 32, 16, CU_TENSOR_MAP_SWIZZLE_NONE,
               "cuTensorMapEncodeTiled(sfvt_hi)"),
      EncodeNHD(out, 128, "cuTensorMapEncodeTiled(out)"),
  };
  CheckCuda(
      cudaMemcpy(descriptor_storage.data_ptr(), maps.data(), sizeof(maps), cudaMemcpyHostToDevice),
      "cudaMemcpy(wan_hybrid tensor maps)");
}

void Attention(TensorView q, TensorView k, TensorView vt, TensorView sfvt_lo, TensorView sfvt_hi,
               TensorView out, TensorView descriptor_storage, bool prepare_descriptors,
               double sm_scale) {
  CHECK_INPUT_AND_TYPE(q, dl_bfloat16);
  const int32_t device_id = q.device().device_id;
  CheckExactTensor(q, "q", dl_bfloat16, {kBatch, kSequence, kHeads, kHeadDim}, device_id);
  CheckExactTensor(k, "k", dl_bfloat16, {kBatch, kSequence, kHeads, kHeadDim}, device_id);
  CheckExactTensor(vt, "vt", dl_uint8, {kPackedValueRows, kPackedValueColumns}, device_id);
  CheckExactTensor(sfvt_lo, "sfvt_lo", dl_uint8, {kValueScaleRows, kScaleColumns}, device_id);
  CheckExactTensor(sfvt_hi, "sfvt_hi", dl_uint8, {kValueScaleRows, kScaleColumns}, device_id);
  CheckExactTensor(out, "out", dl_bfloat16, {kBatch, kSequence, kHeads, kHeadDim}, device_id);
  CheckExactTensor(descriptor_storage, "descriptor_storage", dl_uint8,
                   {kTensorMapCount, kTensorMapBytes}, device_id);
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(descriptor_storage.data_ptr()) % 128, 0)
      << "descriptor_storage must be 128-byte aligned";
  TVM_FFI_ICHECK(std::isfinite(sm_scale)) << "sm_scale must be finite";

  ffi::CUDADeviceGuard device_guard(device_id);
  CheckTarget(device_id);
  if (prepare_descriptors) {
    PrepareTensorMaps(q, k, vt, sfvt_lo, sfvt_hi, out, descriptor_storage);
    CheckCuda(cudaFuncSetAttribute(kernel_wan_hybrid_attention,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(kDynamicSmemBytes)),
              "cudaFuncSetAttribute(MaxDynamicSharedMemorySize)");
  }

  int multiprocessor_count = 0;
  CheckCuda(
      cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device_id),
      "cudaDeviceGetAttribute(multiProcessorCount)");
  constexpr int kTotalTiles = ((kSequence + 255) / 256) * kBatch * kHeads;
  const int grid_x = std::min({multiprocessor_count, kTotalTiles, static_cast<int>(kMaximumTiles)});
  TVM_FFI_ICHECK_GT(grid_x, 0) << "wan_hybrid attention requires at least one SM";

  auto* descriptor_bytes = static_cast<uint8_t*>(descriptor_storage.data_ptr());
  auto tensor_map = [descriptor_bytes](int index) {
    return reinterpret_cast<const wan_hybrid_generated_TensorMap*>(descriptor_bytes +
                                                                   index * kTensorMapBytes);
  };
  const auto* q_map = tensor_map(0);
  const auto* k_map = tensor_map(1);
  const auto* vt_map = tensor_map(2);
  const auto* sfvt_lo_map = tensor_map(3);
  const auto* sfvt_hi_map = tensor_map(4);
  const auto* out_map = tensor_map(5);
  constexpr int seqlen_q = kSequence;
  constexpr int seqlen_kv = kSequence;
  constexpr int heads = kHeads;
  constexpr int total_bh = kBatch * kHeads;
  constexpr int physical_num_blocks = kPaddedSequence / 128;
  const float softmax_scale_log2 = static_cast<float>(sm_scale / std::log(2.0));

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<uint32_t>(grid_x), 1, 1);
  config.blockDim = dim3(kThreads, 1, 1);
  config.dynamicSmemBytes = kDynamicSmemBytes;
  config.stream = get_stream(q.device());
  CheckCuda(cudaLaunchKernelEx(&config, kernel_wan_hybrid_attention, q_map, k_map, vt_map,
                               sfvt_lo_map, sfvt_hi_map, out_map, seqlen_q, seqlen_kv,
                               softmax_scale_log2, heads, total_bh, physical_num_blocks),
            "wan_hybrid_attention launch");
}

}  // namespace wan_hybrid
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(wan_hybrid_attention, flashinfer::wan_hybrid::Attention);
