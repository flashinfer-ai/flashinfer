/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include <initializer_list>
#include <limits>
#include "tvm_ffi_utils.h"

namespace {
using OptionalTensor = tvm::ffi::Optional<TensorView>;

void CheckCuda(cudaError_t status) {
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError) << cudaGetErrorString(status);
}

class DeviceGuard {
 public:
  explicit DeviceGuard(DLDevice device) {
    CheckCuda(cudaGetDevice(&previous_));
    changed_ = previous_ != device.device_id;
    if (changed_) CheckCuda(cudaSetDevice(device.device_id));
  }
  ~DeviceGuard() { if (changed_) (void)cudaSetDevice(previous_); }
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;
 private:
  int previous_;
  bool changed_;
};

void CheckArch(DLDevice device) {
  int major, minor;
  CheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device.device_id));
  CheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device.device_id));
  TVM_FFI_CHECK(major == 11 && minor == 0, ValueError) << "SM110 XQA requires exact SM110";
}

int64_t U32(int64_t value, const char* name, bool zero = false) {
  TVM_FFI_CHECK(value >= (zero ? 0 : 1) &&
                static_cast<uint64_t>(value) <= std::numeric_limits<uint32_t>::max(),
                ValueError) << name << " is outside the uint32 launch contract";
  return value;
}

int64_t Product(std::initializer_list<int64_t> values) {
  uint64_t result = 1;
  for (int64_t value : values) {
    U32(value, "tensor extent", true);
    TVM_FFI_CHECK(value == 0 || result <= std::numeric_limits<uint32_t>::max() /
                    static_cast<uint64_t>(value), ValueError)
        << "tensor address exceeds uint32 indexing";
    result *= static_cast<uint64_t>(value);
  }
  return static_cast<int64_t>(result);
}

int64_t Elements(TensorView tensor) {
  int64_t result = 1;
  for (int i = 0; i < tensor.ndim(); ++i) result = Product({result, tensor.size(i)});
  return result;
}

void CheckTensor(TensorView tensor, const char* name, DLDevice device,
            DLDataType dtype, size_t alignment = 1) {
  TVM_FFI_CHECK(tensor.device().device_type == kDLCUDA &&
                tensor.device().device_id == device.device_id &&
                tensor.IsContiguous(), ValueError)
      << name << " must be contiguous CUDA storage on the query device";
  TVM_FFI_CHECK(encode_dlpack_dtype(tensor.dtype()) == encode_dlpack_dtype(dtype),
                TypeError) << name << " has an incorrect storage dtype";
  TVM_FFI_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignment == 0,
                ValueError) << name << " has insufficient address alignment";
  Elements(tensor);
}

void Shape(TensorView tensor, std::initializer_list<int64_t> shape, const char* name) {
  TVM_FFI_CHECK(tensor.ndim() == static_cast<int>(shape.size()), ValueError)
      << name << " has an incorrect rank";
  int axis = 0;
  for (int64_t extent : shape) {
    TVM_FFI_CHECK(tensor.size(axis++) == extent, ValueError)
        << name << " has an incorrect extent";
  }
}

void Like(TensorView output, TensorView input) {
  TVM_FFI_CHECK(output.ndim() == input.ndim(), ValueError) << "output rank differs from Q";
  for (int i = 0; i < input.ndim(); ++i)
    TVM_FFI_CHECK(output.size(i) == input.size(i), ValueError) << "output shape differs from Q";
}

void Disjoint(TensorView output, std::initializer_list<TensorView> inputs) {
  const uintptr_t begin = reinterpret_cast<uintptr_t>(output.data_ptr());
  const uint64_t bytes = static_cast<uint64_t>(Elements(output)) * output.dtype().bits / 8;
  TVM_FFI_CHECK(bytes <= std::numeric_limits<uintptr_t>::max() - begin, ValueError)
      << "invalid output extent";
  for (TensorView input : inputs) {
    const uintptr_t other = reinterpret_cast<uintptr_t>(input.data_ptr());
    const uint64_t other_bytes = static_cast<uint64_t>(Elements(input)) * input.dtype().bits / 8;
    TVM_FFI_CHECK(other_bytes <= std::numeric_limits<uintptr_t>::max() - other, ValueError)
        << "invalid input extent";
    TVM_FFI_CHECK(bytes == 0 || other_bytes == 0 || begin + bytes <= other ||
                  other + other_bytes <= begin, ValueError)
        << "mutable output/workspace overlaps another tensor";
  }
}

float Scale(double value) {
  TVM_FFI_CHECK(std::isfinite(value) && std::isfinite(static_cast<float>(value)),
                ValueError) << "scale must be representable as a finite float32";
  return static_cast<float>(value);
}

void Grid(int64_t x, int64_t y, int64_t z,
          int64_t expected_x, int64_t expected_y, int64_t expected_z) {
  U32(x, "grid_x"); U32(y, "grid_y"); U32(z, "grid_z");
  TVM_FFI_CHECK(x <= 2147483647 && y <= 65535 && z <= 65535 &&
                x == expected_x && y == expected_y && z == expected_z, ValueError)
      << "grid differs from the frozen attention launch geometry";
}
}  // namespace

extern "C" __global__ void kernel_sm110_xqa_decode_merge(const float* partial, const float* statistics, __half* output, uint32_t partitions);

void run_decode_merge(
    TensorView partial,
    TensorView statistics,
    TensorView output,
    int64_t partitions,
    int64_t grid_x,
    int64_t grid_y,
    int64_t grid_z) {

  const DLDevice device = output.device();
  CheckTensor(output, "output", device, dl_float16, 2);
  CheckTensor(partial, "partial", device, dl_float32, 4);
  CheckTensor(statistics, "statistics", device, dl_float32, 4);
  U32(partitions, "partitions");
  TVM_FFI_CHECK(partitions > 1 && output.ndim() == 3 && output.size(2) == 128,
                ValueError) << "merge requires rank-three D128 output and multiple partitions";
  const int64_t rows = Product({output.size(0), output.size(1)});
  U32(rows, "output rows");
  Shape(partial, {rows, partitions, 128}, "partial");
  Shape(statistics, {rows, partitions, 2}, "statistics");
  Disjoint(output, {partial, statistics});
  Grid(grid_x, grid_y, grid_z, rows, 1, 1);

  DeviceGuard guard(device);
  CheckArch(device);
  cudaStream_t stream = get_stream(device);
  const float* p_partial = reinterpret_cast<const float*>(partial.data_ptr());
  const float* p_statistics = reinterpret_cast<const float*>(statistics.data_ptr());
  __half* p_output = reinterpret_cast<__half*>(output.data_ptr());
  uint32_t p_partitions = static_cast<uint32_t>(partitions);
  void* arguments[] = {&p_partial, &p_statistics, &p_output, &p_partitions};
  CheckCuda(cudaLaunchKernel(reinterpret_cast<const void*>(kernel_sm110_xqa_decode_merge),
      dim3(static_cast<uint32_t>(grid_x), static_cast<uint32_t>(grid_y), static_cast<uint32_t>(grid_z)),
      dim3(128, 1, 1), arguments, 0, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_decode_merge, run_decode_merge);
