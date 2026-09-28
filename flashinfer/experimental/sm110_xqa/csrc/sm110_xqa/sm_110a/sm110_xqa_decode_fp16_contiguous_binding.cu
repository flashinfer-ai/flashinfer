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

extern "C" __global__ void kernel_sm110_xqa_decode_fp16_contiguous(const __half* q, const __half* kv, const int* sequence_lengths, __half* output, float* partial, float* statistics, uint32_t* counters, uint32_t heads, uint32_t ratio, uint32_t capacity, uint32_t partitions, uint32_t partition_tokens, float attention_scale);

void run_decode(
    TensorView q,
    TensorView kv,
    TensorView lengths,
    TensorView output,
    TensorView partial,
    TensorView statistics,
    TensorView counters,
    int64_t heads,
    int64_t ratio,
    int64_t capacity,
    int64_t partitions,
    int64_t partition_tokens,
    double sm_scale,
    int64_t grid_x,
    int64_t grid_y,
    int64_t grid_z) {

  const DLDevice device = q.device();
  CheckTensor(q, "q", device, dl_float16, 16);
  CheckTensor(kv, "kv", device, dl_float16, 16);
  CheckTensor(lengths, "lengths", device, dl_int32, 4);
  CheckTensor(output, "output", device, dl_float16, 16);
  CheckTensor(partial, "partial", device, dl_float32, 16);
  CheckTensor(statistics, "statistics", device, dl_float32, 8);
  CheckTensor(counters, "counters", device, dl_int32, 4);
  U32(heads, "heads"); U32(capacity, "capacity"); U32(partitions, "partitions");
  U32(partition_tokens, "partition_tokens");
  TVM_FFI_CHECK(partition_tokens % 64 == 0, ValueError)
      << "partition_tokens must be a positive multiple of 64";
  TVM_FFI_CHECK(ratio == 4 || ratio == 8 || ratio == 16, ValueError)
      << "decode GQA ratio must be 4, 8 or 16";
  TVM_FFI_CHECK(lengths.ndim() == 1, ValueError) << "lengths must have rank one";
  const int64_t batch = U32(lengths.size(0), "batch");
  const int64_t q_heads = Product({heads, ratio});
  TVM_FFI_CHECK(partitions == (capacity + partition_tokens - 1) / partition_tokens, ValueError)
      << "partition count differs from the runtime token partition";
  Shape(q, {batch, q_heads, 128}, "q");
  Shape(kv, {batch, 2, heads, capacity, 128}, "kv");
  Like(output, q);
  Shape(partial, {Product({batch, q_heads}), partitions, 128}, "partial");
  Shape(statistics, {Product({batch, q_heads}), partitions, 2}, "statistics");
  Shape(counters, {batch, heads}, "counters");
  Disjoint(output, {q, kv, lengths, partial, statistics, counters});
  Disjoint(partial, {q, kv, lengths, statistics, counters});
  Disjoint(statistics, {q, kv, lengths, counters});
  Disjoint(counters, {q, kv, lengths});
  Scale(sm_scale);
  Grid(grid_x, grid_y, grid_z, partitions, heads, batch);

  TVM_FFI_CHECK(partitions > 1, ValueError) << "stats-cache route requires multiple partitions";

  DeviceGuard guard(device);
  CheckArch(device);
  cudaStream_t stream = get_stream(device);
  CheckCuda(cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel_sm110_xqa_decode_fp16_contiguous),
      cudaFuncAttributeMaxDynamicSharedMemorySize, 50176));
  const __half* p_q = reinterpret_cast<const __half*>(q.data_ptr());
  const __half* p_kv = reinterpret_cast<const __half*>(kv.data_ptr());
  const int* p_sequence_lengths = reinterpret_cast<const int*>(lengths.data_ptr());
  __half* p_output = reinterpret_cast<__half*>(output.data_ptr());
  float* p_partial = reinterpret_cast<float*>(partial.data_ptr());
  float* p_statistics = reinterpret_cast<float*>(statistics.data_ptr());
  uint32_t* p_counters = reinterpret_cast<uint32_t*>(counters.data_ptr());
  uint32_t p_heads = static_cast<uint32_t>(heads);
  uint32_t p_ratio = static_cast<uint32_t>(ratio);
  uint32_t p_capacity = static_cast<uint32_t>(capacity);
  uint32_t p_partitions = static_cast<uint32_t>(partitions);
  uint32_t p_partition_tokens = static_cast<uint32_t>(partition_tokens);
  float p_attention_scale = static_cast<float>(Scale(sm_scale));
  void* arguments[] = {&p_q, &p_kv, &p_sequence_lengths, &p_output, &p_partial, &p_statistics, &p_counters, &p_heads, &p_ratio, &p_capacity, &p_partitions, &p_partition_tokens, &p_attention_scale};
  CheckCuda(cudaLaunchKernel(reinterpret_cast<const void*>(kernel_sm110_xqa_decode_fp16_contiguous),
      dim3(static_cast<uint32_t>(grid_x), static_cast<uint32_t>(grid_y), static_cast<uint32_t>(grid_z)),
      dim3(128, 1, 1), arguments, 50176, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_decode, run_decode);
