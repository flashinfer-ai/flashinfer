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

extern "C" __global__ void kernel_sm110_xqa_tree_fp8_contiguous(uint32_t q_seq_len, uint32_t num_kv_heads, uint32_t head_group_size, const uint32_t* q_cu_seq_lens, float attention_scale, __half* output, const __half* q, const uint32_t* mask, const uint8_t* cache, const int* sequence_lengths, const int* page_list, uint32_t capacity, uint32_t max_pages, float k_cache_scale, float v_cache_scale);

void run_tree(
    TensorView q,
    TensorView kv,
    TensorView lengths,
    TensorView mask,
    TensorView output,
    OptionalTensor q_offsets,
    OptionalTensor pages,
    int64_t q_len,
    int64_t heads,
    int64_t ratio,
    int64_t capacity,
    int64_t max_pages,
    double sm_scale,
    double k_scale,
    double v_scale,
    int64_t grid_x,
    int64_t grid_y,
    int64_t grid_z) {

  const DLDevice device = q.device();
  CheckTensor(q, "q", device, dl_float16, 16);
  CheckTensor(kv, "kv", device, dl_float8_e4m3fn, 16);
  CheckTensor(lengths, "lengths", device, dl_int32, 4);
  CheckTensor(mask, "mask", device, dl_int32, 4);
  CheckTensor(output, "output", device, dl_float16, 2);
  U32(q_len, "q_len"); U32(heads, "heads"); U32(capacity, "capacity");
  TVM_FFI_CHECK(ratio == 2 || ratio == 4 || ratio == 8 || ratio == 16, ValueError)
      << "tree GQA ratio must be 2, 4, 8 or 16";
  TVM_FFI_CHECK(lengths.ndim() == 1, ValueError) << "lengths must have rank one";
  const int64_t batch = U32(lengths.size(0), "batch");
  const int64_t q_heads = Product({heads, ratio});
  const int64_t mask_words = (q_len + 31) / 32;
  TVM_FFI_CHECK(q_len <= capacity, ValueError) << "query length exceeds KV capacity";
  if (q_offsets.has_value()) {
    CheckTensor(q_offsets.value(), "q_offsets", device, dl_int32, 4);
    Shape(q_offsets.value(), {batch + 1}, "q_offsets");
    TVM_FFI_CHECK(q.ndim() == 3, ValueError) << "packed Q must have rank three";
    Shape(q, {q.size(0), q_heads, 512}, "q");
    Shape(mask, {q.size(0), mask_words}, "mask");
    Disjoint(output, {q_offsets.value()});
  } else {
    Shape(q, {batch, q_len, q_heads, 512}, "q");
    Shape(mask, {batch, q_len, mask_words}, "mask");
  }
  Like(output, q);
  Disjoint(output, {q, kv, lengths, mask});
  Grid(grid_x, grid_y, grid_z, 2,
       Product({heads, (Product({q_len, ratio}) + 63) / 64}), batch);
  Scale(sm_scale); Scale(k_scale); Scale(v_scale);

  TVM_FFI_CHECK(!pages.has_value() && max_pages == 0, ValueError)
      << "contiguous route does not accept pages";
  Shape(kv, {batch, 2, heads, capacity, 512}, "kv");

  DeviceGuard guard(device);
  CheckArch(device);
  cudaStream_t stream = get_stream(device);
  CheckCuda(cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel_sm110_xqa_tree_fp8_contiguous),
      cudaFuncAttributeMaxDynamicSharedMemorySize, 214016));
  uint32_t p_q_seq_len = static_cast<uint32_t>(q_len);
  uint32_t p_num_kv_heads = static_cast<uint32_t>(heads);
  uint32_t p_head_group_size = static_cast<uint32_t>(ratio);
  const uint32_t* p_q_cu_seq_lens = reinterpret_cast<const uint32_t*>(q_offsets.has_value() ? q_offsets.value().data_ptr() : nullptr);
  float p_attention_scale = static_cast<float>(Scale(sm_scale));
  __half* p_output = reinterpret_cast<__half*>(output.data_ptr());
  const __half* p_q = reinterpret_cast<const __half*>(q.data_ptr());
  const uint32_t* p_mask = reinterpret_cast<const uint32_t*>(mask.data_ptr());
  const uint8_t* p_cache = reinterpret_cast<const uint8_t*>(kv.data_ptr());
  const int* p_sequence_lengths = reinterpret_cast<const int*>(lengths.data_ptr());
  const int* p_page_list = reinterpret_cast<const int*>(pages.has_value() ? pages.value().data_ptr() : nullptr);
  uint32_t p_capacity = static_cast<uint32_t>(capacity);
  uint32_t p_max_pages = static_cast<uint32_t>(max_pages);
  float p_k_cache_scale = static_cast<float>(Scale(k_scale));
  float p_v_cache_scale = static_cast<float>(Scale(v_scale));
  void* arguments[] = {&p_q_seq_len, &p_num_kv_heads, &p_head_group_size, &p_q_cu_seq_lens, &p_attention_scale, &p_output, &p_q, &p_mask, &p_cache, &p_sequence_lengths, &p_page_list, &p_capacity, &p_max_pages, &p_k_cache_scale, &p_v_cache_scale};
  CheckCuda(cudaLaunchKernel(reinterpret_cast<const void*>(kernel_sm110_xqa_tree_fp8_contiguous),
      dim3(static_cast<uint32_t>(grid_x), static_cast<uint32_t>(grid_y), static_cast<uint32_t>(grid_z)),
      dim3(256, 1, 1), arguments, 214016, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_tree, run_tree);
