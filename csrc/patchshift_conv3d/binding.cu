/*
 * Copyright (c) 2026 by the PatchShift Conv3d contributors.
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

#include <flashinfer/conv3d/patchshift/weight_layout.cuh>

#include "launcher.cuh"
#include "tvm_ffi_utils.h"

namespace flashinfer::conv3d::patchshift::binding {

using host::DescriptorWorkspace;
using host::Status;
using host::StatusDomain;

namespace {

void CheckStatus(Status status, const char* operation) {
  if (status.ok()) return;
  if (status.domain == StatusDomain::kCudaRuntime) {
    auto error = static_cast<cudaError_t>(status.code);
    TVM_FFI_THROW(RuntimeError) << operation << " failed: " << cudaGetErrorString(error);
  }
  auto error = static_cast<CUresult>(status.code);
  const char* name = nullptr;
  const char* message = nullptr;
  cuGetErrorName(error, &name);
  cuGetErrorString(error, &message);
  TVM_FFI_THROW(RuntimeError) << operation << " failed: " << (name == nullptr ? "CUDA_ERROR" : name)
                              << " (" << (message == nullptr ? "unknown driver error" : message)
                              << ")";
}

void CheckSm100a(TensorView tensor) {
  cudaDeviceProp properties{};
  cudaError_t error = cudaGetDeviceProperties(&properties, tensor.device().device_id);
  TVM_FFI_ICHECK_EQ(error, cudaSuccess)
      << "Unable to query CUDA device: " << cudaGetErrorString(error);
  TVM_FFI_ICHECK(properties.major == 10 && properties.minor == 0)
      << "patchshift_conv3d currently requires SM100a/B200, got compute " << properties.major << "."
      << properties.minor;
}

int64_t SegmentNumel(int c, int k, int tile_m) {
  return static_cast<int64_t>(patchshift::PackedWeightNumel(c, k, tile_m));
}

int64_t TotalPackedWeightNumel(int c, int k) {
  return SegmentNumel(c, k, 128) + SegmentNumel(c, k, 64) + SegmentNumel(c, k, 32);
}

void CheckPackedWeight(TensorView packed_weight, TensorView reference, int c, int k) {
  CHECK_INPUT_AND_TYPE(packed_weight, dl_bfloat16);
  CHECK_DEVICE(packed_weight, reference);
  CHECK_DIM(1, packed_weight);
  TVM_FFI_ICHECK_EQ(packed_weight.numel(), TotalPackedWeightNumel(c, k))
      << "packed_weight has the wrong size for C=" << c << ", K=" << k;
}

void SplitPackedWeight(TensorView packed_weight, int c, int k, patchshift::Element** packed_m128,
                       patchshift::Element** packed_m64, patchshift::Element** packed_m32) {
  auto* base = static_cast<patchshift::Element*>(packed_weight.data_ptr());
  int64_t size_m128 = SegmentNumel(c, k, 128);
  int64_t size_m64 = SegmentNumel(c, k, 64);
  *packed_m128 = base;
  *packed_m64 = base + size_m128;
  *packed_m32 = base + size_m128 + size_m64;
}

void CheckWorkspace(TensorView workspace, TensorView reference) {
  CHECK_INPUT_AND_TYPE(workspace, dl_uint8);
  CHECK_DEVICE(workspace, reference);
  CHECK_DIM(1, workspace);
  TVM_FFI_ICHECK_GE(workspace.numel(), sizeof(DescriptorWorkspace));
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 128, 0)
      << "descriptor workspace must be 128-byte aligned";
}

}  // namespace

int64_t packed_weight_numel(int64_t c, int64_t k) {
  TVM_FFI_ICHECK(c > 0 && c % 8 == 0) << "C must be positive and divisible by 8";
  TVM_FFI_ICHECK_GT(k, 0) << "K must be positive";
  return TotalPackedWeightNumel(static_cast<int>(c), static_cast<int>(k));
}

int64_t descriptor_workspace_size() { return sizeof(DescriptorWorkspace); }

void pack_weight(TensorView weight, TensorView packed_weight) {
  CHECK_CUDA(weight);
  CHECK_INPUT_TYPE(weight, dl_bfloat16);
  CHECK_DIM(5, weight);
  TVM_FFI_ICHECK_EQ(weight.size(2), 3);
  TVM_FFI_ICHECK_EQ(weight.size(3), 3);
  TVM_FFI_ICHECK_EQ(weight.size(4), 3) << "weight must have shape [K, C, 3, 3, 3]";
  int k = static_cast<int>(weight.size(0));
  int c = static_cast<int>(weight.size(1));
  TVM_FFI_ICHECK(c > 0 && c % 8 == 0) << "weight input channels must be divisible by 8";
  TVM_FFI_ICHECK_GT(k, 0);
  CheckPackedWeight(packed_weight, weight, c, k);
  CheckSm100a(weight);

  ffi::CUDADeviceGuard device_guard(weight.device().device_id);
  patchshift::Element* packed_m128 = nullptr;
  patchshift::Element* packed_m64 = nullptr;
  patchshift::Element* packed_m32 = nullptr;
  SplitPackedWeight(packed_weight, c, k, &packed_m128, &packed_m64, &packed_m32);
  patchshift::Conv3dProblem problem{1, 1, 1, 1, c, k};
  CheckStatus(host::PackWeights(static_cast<const patchshift::Element*>(weight.data_ptr()),
                                packed_m128, packed_m64, packed_m32, problem, weight.stride(0),
                                weight.stride(1), weight.stride(2), weight.stride(3),
                                weight.stride(4), get_stream(weight.device())),
              "patchshift_conv3d weight packing");
}

void prepare(TensorView workspace, TensorView input, TensorView packed_weight, int64_t k) {
  CHECK_INPUT_AND_TYPE(input, dl_bfloat16);
  CHECK_DIM(5, input);
  int n = static_cast<int>(input.size(0));
  int d = static_cast<int>(input.size(1));
  int h = static_cast<int>(input.size(2));
  int w = static_cast<int>(input.size(3));
  int c = static_cast<int>(input.size(4));
  patchshift::Conv3dProblem problem{n, d, h, w, c, static_cast<int>(k)};
  TVM_FFI_ICHECK(patchshift::IsSupportedProblem(problem))
      << "input must be positive BF16 NDHWC with C divisible by 8";
  CheckWorkspace(workspace, input);
  CheckPackedWeight(packed_weight, input, c, static_cast<int>(k));
  CheckSm100a(input);

  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  cudaDeviceProp properties{};
  TVM_FFI_ICHECK_EQ(cudaGetDeviceProperties(&properties, input.device().device_id), cudaSuccess);
  patchshift::Element* packed_m128 = nullptr;
  patchshift::Element* packed_m64 = nullptr;
  patchshift::Element* packed_m32 = nullptr;
  SplitPackedWeight(packed_weight, c, static_cast<int>(k), &packed_m128, &packed_m64, &packed_m32);
  CheckStatus(host::PrepareDescriptors(static_cast<DescriptorWorkspace*>(workspace.data_ptr()),
                                       static_cast<patchshift::Element*>(input.data_ptr()),
                                       packed_m128, packed_m64, packed_m32, problem,
                                       properties.multiProcessorCount, get_stream(input.device())),
              "patchshift_conv3d descriptor preparation");
}

void run(TensorView workspace, TensorView input, TensorView packed_weight, TensorView output) {
  CHECK_INPUT_AND_TYPE(input, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(output, dl_bfloat16);
  CHECK_DIM(5, input);
  CHECK_DIM(5, output);
  CHECK_DEVICE(input, output);
  TVM_FFI_ICHECK_EQ(input.size(0), output.size(0));
  TVM_FFI_ICHECK_EQ(input.size(1), output.size(1));
  TVM_FFI_ICHECK_EQ(input.size(2), output.size(2));
  TVM_FFI_ICHECK_EQ(input.size(3), output.size(3));
  TVM_FFI_ICHECK_NE(input.data_ptr(), output.data_ptr()) << "output must not alias input";
  int n = static_cast<int>(input.size(0));
  int d = static_cast<int>(input.size(1));
  int h = static_cast<int>(input.size(2));
  int w = static_cast<int>(input.size(3));
  int c = static_cast<int>(input.size(4));
  int k = static_cast<int>(output.size(4));
  patchshift::Conv3dProblem problem{n, d, h, w, c, k};
  TVM_FFI_ICHECK(patchshift::IsSupportedProblem(problem));
  CheckWorkspace(workspace, input);
  CheckPackedWeight(packed_weight, input, c, k);
  CheckSm100a(input);

  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  cudaDeviceProp properties{};
  TVM_FFI_ICHECK_EQ(cudaGetDeviceProperties(&properties, input.device().device_id), cudaSuccess);
  CheckStatus(host::Launch(static_cast<DescriptorWorkspace*>(workspace.data_ptr()),
                           static_cast<patchshift::Element*>(input.data_ptr()),
                           static_cast<patchshift::Element*>(output.data_ptr()), problem,
                           properties.multiProcessorCount, get_stream(input.device())),
              "patchshift_conv3d launch");
}

}  // namespace flashinfer::conv3d::patchshift::binding

TVM_FFI_DLL_EXPORT_TYPED_FUNC(packed_weight_numel,
                              flashinfer::conv3d::patchshift::binding::packed_weight_numel);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(descriptor_workspace_size,
                              flashinfer::conv3d::patchshift::binding::descriptor_workspace_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(pack_weight, flashinfer::conv3d::patchshift::binding::pack_weight);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare, flashinfer::conv3d::patchshift::binding::prepare);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::conv3d::patchshift::binding::run);
