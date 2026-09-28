/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda.h>
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

// Leading dimensions above the descriptor's trailing dims must form one dense
// row-major chain; the descriptor reads its adjacent physical step separately.
inline void CheckDenseLeadingFold(const TensorView& t, int trailing, const char* name) {
  TVM_FFI_CHECK(trailing > 0 && t.ndim() >= trailing, ValueError)
      << name << " cannot fold leading dimensions above " << trailing
      << " trailing dims from ndim=" << t.ndim();
  int outer_last = t.ndim() - trailing - 1;
  if (outer_last <= 0) {
    return;
  }
  int64_t step = t.stride(outer_last);
  TVM_FFI_CHECK(step > 0, ValueError)
      << name << " physical strides must be positive";
  int64_t expected = step;
  for (int axis = outer_last - 1; axis >= 0; --axis) {
    expected *= t.size(axis + 1);
    if (t.size(axis) > 1) {
      TVM_FFI_CHECK(t.stride(axis) == expected, ValueError)
          << name << " leading dims are not physically foldable above " << trailing
          << " trailing dims: stride(" << axis << ")=" << t.stride(axis)
          << ", expected " << expected;
    }
  }
}

// 3D TMA descriptor for buffer 'Q' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_Q(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << "TMA source 'Q' must have at least 3 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'Q' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'Q' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  CheckDenseLeadingFold(t, 2, "Q");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'Q' physical strides must be positive";
  int64_t s3 = t.stride(t.ndim() - 3) * 1;
  TVM_FFI_CHECK(s3 > 0, ValueError)
      << "TMA source 'Q' physical strides must be positive";
  uint64_t global_dim[3] = {(uint64_t)(d1), (uint64_t)(d2), (uint64_t)(outer2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'Q' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 8u <= global_dim[1], ValueError)
      << "TMA box (64, 8, 16) exceeds resolved global dims for 'Q'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'Q' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'Q' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'Q' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = s3;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'Q' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'Q' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'Q' resolved global stride 2 to a non-whole-byte offset";
  uint64_t global_strides[2] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
  };
  uint32_t box_dim[3] = {64u, 8u, 16u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'Q') failed: CUresult=" << (int)r;
  return tm;
}

// 3D TMA descriptor for buffer 'KV' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_KV(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << "TMA source 'KV' must have at least 3 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'KV' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'KV' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  CheckDenseLeadingFold(t, 2, "KV");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'KV' physical strides must be positive";
  int64_t s3 = t.stride(t.ndim() - 3) * 1;
  TVM_FFI_CHECK(s3 > 0, ValueError)
      << "TMA source 'KV' physical strides must be positive";
  uint64_t global_dim[3] = {(uint64_t)(d1), (uint64_t)(d2), (uint64_t)(outer2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'KV' resolved a non-positive global dim";
  TVM_FFI_CHECK(128u <= global_dim[0], ValueError)
      << "TMA box (128, 64, 1) exceeds resolved global dims for 'KV'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'KV' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'KV' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 8) % 8 == 0, ValueError)
      << "TMA descriptor for 'KV' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = s3;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'KV' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'KV' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 8) % 8 == 0, ValueError)
      << "TMA descriptor for 'KV' resolved global stride 2 to a non-whole-byte offset";
  uint64_t global_strides[2] = {
      (uint64_t)((carrier_stride_0 * 8) / 8),
      (uint64_t)((carrier_stride_1 * 8) / 8),
  };
  uint32_t box_dim[3] = {128u, 64u, 1u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'KV') failed: CUresult=" << (int)r;
  return tm;
}
}  // namespace

// 64-byte aligned by-value tensor-map kernel parameter (matches the kernel translation units).
struct __align__(64) Sm110XqaTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(Sm110XqaTensorMap64) == 128 && alignof(Sm110XqaTensorMap64) == 64,
              "tensor-map kernel parameter ABI");
static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");

struct __align__(8) KVCacheList {
  void* data;
  const int* sequence_lengths;
  unsigned int capacity;
};

extern "C" __global__ void kernel_sm110_xqa_tree_fp8_contiguous_tmem(const __grid_constant__ Sm110XqaTensorMap64 Q, const __grid_constant__ Sm110XqaTensorMap64 KV, unsigned int q_seq_len, unsigned int num_kv_heads, unsigned int head_group_size, const unsigned int* __restrict__ q_cu_seq_lens, float attention_scale, __half* __restrict__ output, const unsigned int* __restrict__ mask, KVCacheList kv_cache_list, unsigned int batch_size, float k_cache_scale, float v_cache_scale);
extern "C" __global__ void kernel_sm110_xqa_tree_fp8_contiguous_tmem_q(const __grid_constant__ Sm110XqaTensorMap64 Q, const __grid_constant__ Sm110XqaTensorMap64 KV, unsigned int q_seq_len, unsigned int num_kv_heads, unsigned int head_group_size, const unsigned int* __restrict__ q_cu_seq_lens, float attention_scale, __half* __restrict__ output, const unsigned int* __restrict__ mask, KVCacheList kv_cache_list, unsigned int batch_size, float k_cache_scale, float v_cache_scale);

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
  CheckTensor(output, "output", device, dl_float16, 16);
  U32(q_len, "q_len"); U32(heads, "heads"); U32(capacity, "capacity");
  // The frozen trace is specialised for GQA ratio 8 (Q tile = 8 heads x 16 tokens).
  TVM_FFI_CHECK(ratio == 8, ValueError) << "tmem tree routes are frozen for GQA ratio 8";
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
  // One 128-row Q tile x one 256-column output half per CTA:
  // grid (2, heads * ceil(q_len * ratio / 128), batch).
  const int64_t q_tiles = (Product({q_len, ratio}) + 127) / 128;
  Grid(grid_x, grid_y, grid_z, 2, Product({heads, q_tiles}), batch);
  Scale(sm_scale); Scale(k_scale); Scale(v_scale);

  TVM_FFI_CHECK(!pages.has_value() && max_pages == 0, ValueError) << "contiguous route takes no page table";
  Shape(kv, {batch, 2, heads, capacity, 512}, "kv");

  DeviceGuard guard(device);
  CheckArch(device);
  cudaStream_t stream = get_stream(device);
  // The (2, 2, 1) cluster form pairs the two Q tiles of one KV head for K/V multicast and needs an
  // even Q-tile count per head; otherwise the (2, 1, 1) form multicasts Q only. Cluster dimensions
  // are compiled into each kernel.
  const void* kernel = q_tiles % 2 == 0 ? reinterpret_cast<const void*>(kernel_sm110_xqa_tree_fp8_contiguous_tmem)
                                        : reinterpret_cast<const void*>(kernel_sm110_xqa_tree_fp8_contiguous_tmem_q);
  CheckCuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 230400));
  CUtensorMap p_Q = EncodeTma_Q(q);
  CUtensorMap p_KV = EncodeTma_KV(kv);
  uint32_t p_q_seq_len = static_cast<uint32_t>(q_len);
  uint32_t p_num_kv_heads = static_cast<uint32_t>(heads);
  uint32_t p_head_group_size = static_cast<uint32_t>(ratio);
  const uint32_t* p_q_cu_seq_lens = reinterpret_cast<const uint32_t*>(q_offsets.has_value() ? q_offsets.value().data_ptr() : nullptr);
  float p_attention_scale = static_cast<float>(Scale(sm_scale));
  __half* p_output = reinterpret_cast<__half*>(output.data_ptr());
  const uint32_t* p_mask = reinterpret_cast<const uint32_t*>(mask.data_ptr());
  KVCacheList p_kv_cache_list;
  p_kv_cache_list.data = kv.data_ptr();
  p_kv_cache_list.sequence_lengths = reinterpret_cast<const int*>(lengths.data_ptr());
  p_kv_cache_list.capacity = static_cast<unsigned int>(capacity);
  uint32_t p_batch_size = static_cast<uint32_t>(batch);
  float p_k_cache_scale = static_cast<float>(Scale(k_scale));
  float p_v_cache_scale = static_cast<float>(Scale(v_scale));
  void* arguments[] = {&p_Q, &p_KV, &p_q_seq_len, &p_num_kv_heads, &p_head_group_size, &p_q_cu_seq_lens, &p_attention_scale, &p_output, &p_mask, &p_kv_cache_list, &p_batch_size, &p_k_cache_scale, &p_v_cache_scale};
  CheckCuda(cudaLaunchKernel(kernel,
      dim3(static_cast<uint32_t>(grid_x), static_cast<uint32_t>(grid_y), static_cast<uint32_t>(grid_z)),
      dim3(512, 1, 1), arguments, 230400, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_tree, run_tree);
