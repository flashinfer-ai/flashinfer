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
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "gated_act_mxfp8_launch.cuh"
#include "tvm_ffi_utils.h"

namespace flashinfer::gated_act_mxfp8 {

namespace {

constexpr int64_t kSm100ForwardBothNoAllocateElements = int64_t{16384} * 7168;
constexpr int64_t kSm103ForwardNoAllocateElements = int64_t{131072} * 8192;

void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK(status == cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

int CheckArchitecture(int device_id) {
  int major = 0;
  int minor = 0;
  CheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
            "reading the CUDA compute-capability major version");
  CheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
            "reading the CUDA compute-capability minor version");
  TVM_FFI_ICHECK(major == 10 && (minor == 0 || minor == 3))
      << "fused gated MXFP8 quantization requires SM100 or SM103";
  return minor;
}

void CheckDevice(const TensorView& tensor, const TensorView& input, const char* name) {
  TVM_FFI_ICHECK(tensor.device().device_type == kDLCUDA) << name << " must be a CUDA tensor";
  TVM_FFI_ICHECK(tensor.device().device_id == input.device().device_id)
      << name << " must be on the input device";
}

void CheckEmpty(const TensorView& tensor, const TensorView& input, const char* name) {
  CheckDevice(tensor, input, name);
  TVM_FFI_ICHECK(tensor.numel() == 0) << name << " must be empty when its route is disabled";
}

void CheckQData(const TensorView& tensor, const TensorView& input, int64_t m, int64_t n,
                bool row_major, const char* name) {
  CheckDevice(tensor, input, name);
  TVM_FFI_ICHECK(tensor.dtype() == dl_float8_e4m3fn) << name << " must have float8_e4m3fn dtype";
  TVM_FFI_ICHECK(tensor.ndim() == 2 && tensor.size(0) == m && tensor.size(1) == n)
      << name << " has an invalid shape";
  if (row_major) {
    TVM_FFI_ICHECK(tensor.stride(0) == n && tensor.stride(1) == 1) << name << " must be row-major";
  } else {
    TVM_FFI_ICHECK(tensor.stride(0) == 1 && tensor.stride(1) == m)
        << name << " must be column-major";
  }
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 32 == 0)
      << name << " must be 32-byte aligned";
}

void CheckScales(const TensorView& tensor, const TensorView& input, int64_t elements,
                 const char* name) {
  CheckDevice(tensor, input, name);
  TVM_FFI_ICHECK(tensor.dtype() == dl_uint8) << name << " must use uint8 storage";
  TVM_FFI_ICHECK(tensor.numel() == elements && tensor.IsContiguous())
      << name << " has an invalid shape or layout";
}

void CheckCommon(const TensorView& gated_input, const TensorView* grad_output, int64_t& m,
                 int64_t& k) {
  CHECK_INPUT_AND_TYPE(gated_input, dl_bfloat16);
  CHECK_DIM(2, gated_input);
  m = gated_input.size(0);
  const int64_t doubled_k = gated_input.size(1);
  TVM_FFI_ICHECK(doubled_k % 2 == 0) << "gated_input.shape[1] must be even";
  k = doubled_k / 2;
  TVM_FFI_ICHECK(m > 0 && k > 0 && m % 128 == 0 && k % 128 == 0)
      << "M and K must be positive multiples of 128";
  TVM_FFI_ICHECK(m * doubled_k - k - 1 <= INT32_MAX)
      << "gated activation shape exceeds signed int32 indexing";
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(gated_input.data_ptr()) % 32 == 0)
      << "gated_input must be 32-byte aligned";
  if (grad_output != nullptr) {
    CHECK_INPUT_AND_TYPE((*grad_output), dl_bfloat16);
    CHECK_DIM(2, (*grad_output));
    CHECK_DEVICE((*grad_output), gated_input);
    TVM_FFI_ICHECK(grad_output->size(0) == m && grad_output->size(1) == k)
        << "grad_output must have shape [M, K]";
    TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(grad_output->data_ptr()) % 32 == 0)
        << "grad_output must be 32-byte aligned";
  }
}

CUtensorMap MakeTensorMap(void* base, CUtensorMapDataType data_type, uint64_t inner, uint64_t outer,
                          uint64_t row_stride_bytes, uint32_t box_inner, uint32_t box_outer,
                          const char* name) {
  CUtensorMap descriptor{};
  constexpr uint32_t kRank = 2;
  uint64_t global_dim[kRank] = {inner, outer};
  uint64_t global_strides[kRank - 1] = {row_stride_bytes};
  uint32_t box_dim[kRank] = {box_inner, box_outer};
  uint32_t element_strides[kRank] = {1, 1};
  const CUresult result = cuTensorMapEncodeTiled(
      &descriptor, data_type, kRank, base, global_dim, global_strides, box_dim, element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for " << name << ": CUresult=" << static_cast<int>(result);
  return descriptor;
}

CUtensorMap MakeInputMap(void* base, int64_t m, int64_t k, int64_t row_elements, const char* name) {
  return MakeTensorMap(base, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, k, m,
                       row_elements * sizeof(__nv_bfloat16), 64, 32, name);
}

CUtensorMap MakeRowOutputMap(void* base, int64_t m, int64_t width, int64_t row_elements,
                             uint32_t box_width, const char* name) {
  return MakeTensorMap(base, CU_TENSOR_MAP_DATA_TYPE_UINT8, width, m, row_elements, box_width, 32,
                       name);
}

CUtensorMap MakeColOutputMap(void* base, int64_t m, int64_t width, const char* name) {
  return MakeTensorMap(base, CU_TENSOR_MAP_DATA_TYPE_UINT8, m, width, m, 32, 64, name);
}

}  // namespace

void Forward(TensorView gated_input, TensorView row_output, TensorView col_output,
             TensorView row_scales, TensorView col_scales, bool rowwise, bool colwise) {
  TVM_FFI_ICHECK(rowwise || colwise) << "at least one quantization route must be enabled";
  int64_t m = 0;
  int64_t k = 0;
  CheckCommon(gated_input, nullptr, m, k);
  if (rowwise) {
    CheckQData(row_output, gated_input, m, k, true, "row_output");
    CheckScales(row_scales, gated_input, m * (k / 32), "row_scales");
  } else {
    CheckEmpty(row_output, gated_input, "row_output");
    CheckEmpty(row_scales, gated_input, "row_scales");
  }
  if (colwise) {
    CheckQData(col_output, gated_input, m, k, false, "col_output");
    CheckScales(col_scales, gated_input, k * (m / 32), "col_scales");
  } else {
    CheckEmpty(col_output, gated_input, "col_output");
    CheckEmpty(col_scales, gated_input, "col_scales");
  }

  const int device_id = gated_input.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  const int architecture_minor = CheckArchitecture(device_id);
  auto* input = static_cast<__nv_bfloat16*>(gated_input.data_ptr());
  auto* row_q = static_cast<uint8_t*>(row_output.data_ptr());
  auto* col_q = static_cast<uint8_t*>(col_output.data_ptr());
  auto* row_sf = static_cast<uint8_t*>(row_scales.data_ptr());
  auto* col_sf = static_cast<uint8_t*>(col_scales.data_ptr());
  const cudaStream_t stream = get_stream(gated_input.device());
  const int m32 = static_cast<int>(m);
  const int k32 = static_cast<int>(k);
  const int64_t elements = m * k;
  cudaError_t status = cudaSuccess;

  if (rowwise && colwise) {
    const CUtensorMap row_map = MakeRowOutputMap(row_q, m, k, k, 64, "row_output");
    const CUtensorMap col_map = MakeColOutputMap(col_q, m, k, "col_output");
    const bool use_no_allocate =
        (architecture_minor == 3 && elements >= kSm103ForwardNoAllocateElements) ||
        (architecture_minor == 0 && elements >= kSm100ForwardBothNoAllocateElements);
    status =
        use_no_allocate
            ? LaunchForwardBothNoAllocate(input, row_map, col_map, row_sf, col_sf, m32, k32, stream)
            : LaunchForwardBoth(input, row_map, col_map, row_sf, col_sf, m32, k32, stream);
  } else if (rowwise) {
    const CUtensorMap row_map = MakeRowOutputMap(row_q, m, k, k, 128, "row_output");
    status = architecture_minor == 3 && elements >= kSm103ForwardNoAllocateElements
                 ? LaunchForwardRowNoAllocate(input, row_map, row_sf, m32, k32, stream)
                 : LaunchForwardRow(input, row_map, row_sf, m32, k32, stream);
  } else {
    const CUtensorMap gate_map = MakeInputMap(input, m, k, 2 * k, "gate input");
    const CUtensorMap up_map = MakeInputMap(input + k, m, k, 2 * k, "up input");
    const CUtensorMap col_map = MakeColOutputMap(col_q, m, k, "col_output");
    status = LaunchForwardCol(gate_map, up_map, col_map, col_sf, m32, k32, stream);
  }
  CheckCuda(status, "launching fused gated MXFP8 forward");
}

void Backward(TensorView gated_input, TensorView grad_output, TensorView row_output,
              TensorView col_output, TensorView row_scales, TensorView col_scales, bool rowwise,
              bool colwise) {
  TVM_FFI_ICHECK(rowwise || colwise) << "at least one quantization route must be enabled";
  int64_t m = 0;
  int64_t k = 0;
  CheckCommon(gated_input, &grad_output, m, k);
  const int64_t output_k = 2 * k;
  if (rowwise) {
    CheckQData(row_output, gated_input, m, output_k, true, "row_output");
    CheckScales(row_scales, gated_input, m * (output_k / 32), "row_scales");
  } else {
    CheckEmpty(row_output, gated_input, "row_output");
    CheckEmpty(row_scales, gated_input, "row_scales");
  }
  if (colwise) {
    CheckQData(col_output, gated_input, m, output_k, false, "col_output");
    CheckScales(col_scales, gated_input, output_k * (m / 32), "col_scales");
  } else {
    CheckEmpty(col_output, gated_input, "col_output");
    CheckEmpty(col_scales, gated_input, "col_scales");
  }

  const int device_id = gated_input.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  const int architecture_minor = CheckArchitecture(device_id);
  auto* input = static_cast<__nv_bfloat16*>(gated_input.data_ptr());
  auto* grad = static_cast<__nv_bfloat16*>(grad_output.data_ptr());
  auto* row_q = static_cast<uint8_t*>(row_output.data_ptr());
  auto* col_q = static_cast<uint8_t*>(col_output.data_ptr());
  auto* row_sf = static_cast<uint8_t*>(row_scales.data_ptr());
  auto* col_sf = static_cast<uint8_t*>(col_scales.data_ptr());
  const cudaStream_t stream = get_stream(gated_input.device());
  const int m32 = static_cast<int>(m);
  const int k32 = static_cast<int>(k);
  cudaError_t status = cudaSuccess;

  if (rowwise && colwise) {
    const CUtensorMap row_act =
        MakeRowOutputMap(row_q, m, k, output_k, 64, "row activation gradient");
    const CUtensorMap row_gate =
        MakeRowOutputMap(row_q + k, m, k, output_k, 64, "row gate gradient");
    const CUtensorMap col_act = MakeColOutputMap(col_q, m, k, "col activation gradient");
    const CUtensorMap col_gate = MakeColOutputMap(col_q + k * m, m, k, "col gate gradient");
    status = architecture_minor == 3
                 ? LaunchBackwardBothSm103(input, grad, row_act, row_gate, col_act, col_gate,
                                           row_sf, col_sf, m32, k32, stream)
                 : LaunchBackwardBoth(input, grad, row_act, row_gate, col_act, col_gate, row_sf,
                                      col_sf, m32, k32, stream);
  } else if (rowwise) {
    const CUtensorMap row_act =
        MakeRowOutputMap(row_q, m, k, output_k, 64, "row activation gradient");
    const CUtensorMap row_gate =
        MakeRowOutputMap(row_q + k, m, k, output_k, 64, "row gate gradient");
    status = architecture_minor == 3
                 ? LaunchBackwardRowSm103(input, grad, row_act, row_gate, row_sf, m32, k32, stream)
                 : LaunchBackwardRow(input, grad, row_act, row_gate, row_sf, m32, k32, stream);
  } else {
    const CUtensorMap gate_map = MakeInputMap(input, m, k, output_k, "gate input");
    const CUtensorMap up_map = MakeInputMap(input + k, m, k, output_k, "up input");
    const CUtensorMap grad_map = MakeInputMap(grad, m, k, k, "output gradient");
    const CUtensorMap col_act = MakeColOutputMap(col_q, m, k, "col activation gradient");
    const CUtensorMap col_gate = MakeColOutputMap(col_q + k * m, m, k, "col gate gradient");
    status =
        LaunchBackwardCol(gate_map, up_map, grad_map, col_act, col_gate, col_sf, m32, k32, stream);
  }
  CheckCuda(status, "launching fused gated MXFP8 backward");
}

}  // namespace flashinfer::gated_act_mxfp8
