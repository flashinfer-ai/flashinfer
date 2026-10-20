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

// Exporter output, hand-folded here pending exporter support: the argument checks of Run
// use the shared helpers of tvm_ffi_utils.h instead of per-file copies.
// tvm-ffi direct-source launcher for Cake kernel 'kernel_cake_sage_block_sparse_attention_f63b7fc8674de9925f7e'.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "tvm_ffi_utils.h"

#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>

struct CakeTensorMap;
extern "C" __global__ void kernel_cake_sage_block_sparse_attention_f63b7fc8674de9925f7e(CakeTensorMap const* q, CakeTensorMap const* k, CakeTensorMap const* v, __nv_bfloat16* __restrict__ out, float* __restrict__ lse, float* __restrict__ q_scale, float* __restrict__ k_scale, float* __restrict__ v_scale, int* __restrict__ block_index, int* __restrict__ block_nums, int* __restrict__ block_sizes, int seqlen_q, int seqlen_k, int num_heads, int num_kv_heads, int q_blocks, int k_blocks, int k_scale_groups, int capacity, int block_sparse_num, int has_block_nums, int block_sizes_mode, int v_scale_batch_stride, int total_tiles, float softmax_scale_log2, int return_lse);

namespace cake_host_shim_c802b60dd9d50557 {

using tvm::ffi::Optional;
using tvm::ffi::TensorView;

class ScopedCudaDevice {
 public:
  explicit ScopedCudaDevice(int device_id) {
    cudaError_t error = cudaGetDevice(&previous_device_);
    TVM_FFI_CHECK(error == cudaSuccess, RuntimeError)
        << "cudaGetDevice failed before host-shim launch: cudaError="
        << static_cast<int>(error);
    if (previous_device_ != device_id) {
      error = cudaSetDevice(device_id);
      TVM_FFI_CHECK(error == cudaSuccess, RuntimeError)
          << "cudaSetDevice failed before host-shim launch for cuda:"
          << device_id << ": cudaError=" << static_cast<int>(error);
      restore_ = true;
    }
  }

  ScopedCudaDevice(const ScopedCudaDevice&) = delete;
  ScopedCudaDevice& operator=(const ScopedCudaDevice&) = delete;

  ~ScopedCudaDevice() noexcept {
    if (restore_) {
      (void)cudaSetDevice(previous_device_);
    }
  }

 private:
  int previous_device_ = -1;
  bool restore_ = false;
};

// Caller-owned storage is mutable scratch, including when its address and
// underlying CUDA allocation stay unchanged. Refresh all descriptors on the
// launch stream every time; allocation identity cannot establish their contents.
// By-value kernel parameters make the upload independent of host-buffer lifetime
// and record the descriptor bytes in CUDA Graphs for every replay.
template <size_t N>
struct CallerTmaDescriptorBatch {
  uint4 words[N * sizeof(CUtensorMap) / sizeof(uint4)];
};

template <size_t N>
__global__ void UploadCallerTmaDescriptors(
    uint4* workspace, CallerTmaDescriptorBatch<N> descriptors) {
  for (size_t i = threadIdx.x;
       i < N * sizeof(CUtensorMap) / sizeof(uint4); i += blockDim.x) {
    workspace[i] = descriptors.words[i];
  }
  // Generated pointer-ABI consumers acquire the tensor-map proxy before use.
  asm volatile("fence.proxy.tensormap::generic.release.sys;" ::: "memory");
}

template <size_t N>
static inline void UploadCallerTmaWorkspace(
    void* workspace, const CUtensorMap (&maps)[N], cudaStream_t stream) {
  CallerTmaDescriptorBatch<N> descriptors;
  static_assert(sizeof(descriptors) == sizeof(maps), "tensor-map batch ABI");
  std::memcpy(&descriptors, maps, sizeof(maps));
  void* args[] = {&workspace, &descriptors};
  cudaError_t result = cudaLaunchKernel(
      reinterpret_cast<const void*>(UploadCallerTmaDescriptors<N>),
      dim3(1), dim3(32), args, 0, stream);
  TVM_FFI_CHECK(result == cudaSuccess, RuntimeError)
      << "uploading caller-owned TMA descriptors failed: "
      << cudaGetErrorString(result);
}

static inline void* CallerTmaWorkspaceSlot(
    const tvm::ffi::TensorView& workspace,
    size_t slot) {
  const uintptr_t base = reinterpret_cast<uintptr_t>(workspace.data_ptr());
  const uintptr_t address = base + slot * sizeof(CUtensorMap);
  return reinterpret_cast<void*>(address);
}

// 4D TMA descriptor for buffer 'q' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_q(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 4, ValueError)
      << "TMA source 'q' must have at least 4 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'q' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  int64_t d3 = t.size(t.ndim() - 3);
  int64_t d4 = t.size(t.ndim() - 4);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0 && d3 > 0 && d4 > 0, ValueError)
      << "TMA source 'q' trailing dims must be positive";
  uint64_t global_dim[4] = {(uint64_t)(128), (uint64_t)(d2), (uint64_t)(d3), (uint64_t)(d4)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'q' resolved a non-positive global dim";
  TVM_FFI_CHECK(128u <= global_dim[0] && 1u <= global_dim[1] && 1u <= global_dim[3], ValueError)
      << "TMA box (128, 1, 64, 1) exceeds resolved global dims for 'q'";
  int64_t carrier_stride_0 = d1;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'q' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'q' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 8) % 8 == 0, ValueError)
      << "TMA descriptor for 'q' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'q' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'q' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 8) % 8 == 0, ValueError)
      << "TMA descriptor for 'q' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = ((d3 * d2) * d1);
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'q' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'q' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 8) % 8 == 0, ValueError)
      << "TMA descriptor for 'q' resolved global stride 3 to a non-whole-byte offset";
  uint64_t global_strides[3] = {
      (uint64_t)((carrier_stride_0 * 8) / 8),
      (uint64_t)((carrier_stride_1 * 8) / 8),
      (uint64_t)((carrier_stride_2 * 8) / 8),
  };
  uint32_t box_dim[4] = {128u, 1u, 64u, 1u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'q') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'k' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_k(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 4, ValueError)
      << "TMA source 'k' must have at least 4 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'k' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  int64_t d3 = t.size(t.ndim() - 3);
  int64_t d4 = t.size(t.ndim() - 4);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0 && d3 > 0 && d4 > 0, ValueError)
      << "TMA source 'k' trailing dims must be positive";
  uint64_t global_dim[4] = {(uint64_t)(128), (uint64_t)(d3), (uint64_t)(d2), (uint64_t)(d4)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'k' resolved a non-positive global dim";
  TVM_FFI_CHECK(128u <= global_dim[0] && 1u <= global_dim[2] && 1u <= global_dim[3], ValueError)
      << "TMA box (128, 64, 1, 1) exceeds resolved global dims for 'k'";
  int64_t carrier_stride_0 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'k' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'k' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 8) % 8 == 0, ValueError)
      << "TMA descriptor for 'k' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = d1;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'k' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'k' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 8) % 8 == 0, ValueError)
      << "TMA descriptor for 'k' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = ((d3 * d2) * d1);
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'k' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'k' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 8) % 8 == 0, ValueError)
      << "TMA descriptor for 'k' resolved global stride 3 to a non-whole-byte offset";
  uint64_t global_strides[3] = {
      (uint64_t)((carrier_stride_0 * 8) / 8),
      (uint64_t)((carrier_stride_1 * 8) / 8),
      (uint64_t)((carrier_stride_2 * 8) / 8),
  };
  uint32_t box_dim[4] = {128u, 64u, 1u, 1u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'k') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'v' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_v(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 4, ValueError)
      << "TMA source 'v' must have at least 4 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'v' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  int64_t d3 = t.size(t.ndim() - 3);
  int64_t d4 = t.size(t.ndim() - 4);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0 && d3 > 0 && d4 > 0, ValueError)
      << "TMA source 'v' trailing dims must be positive";
  uint64_t global_dim[4] = {(uint64_t)(128), (uint64_t)(d3), (uint64_t)(d2), (uint64_t)(d4)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'v' resolved a non-positive global dim";
  TVM_FFI_CHECK(128u <= global_dim[0] && 1u <= global_dim[2] && 1u <= global_dim[3], ValueError)
      << "TMA box (128, 64, 1, 1) exceeds resolved global dims for 'v'";
  int64_t carrier_stride_0 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'v' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'v' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 8) % 8 == 0, ValueError)
      << "TMA descriptor for 'v' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = d1;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'v' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'v' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 8) % 8 == 0, ValueError)
      << "TMA descriptor for 'v' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = ((d3 * d2) * d1);
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'v' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'v' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 8) % 8 == 0, ValueError)
      << "TMA descriptor for 'v' resolved global stride 3 to a non-whole-byte offset";
  uint64_t global_strides[3] = {
      (uint64_t)((carrier_stride_0 * 8) / 8),
      (uint64_t)((carrier_stride_1 * 8) / 8),
      (uint64_t)((carrier_stride_2 * 8) / 8),
  };
  uint32_t box_dim[4] = {128u, 64u, 1u, 1u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'v') failed: CUresult=" << (int)r;
  return tm;
}

void Run(TensorView arg_q, TensorView arg_k, TensorView arg_v, TensorView arg_out, TensorView arg_lse, TensorView arg_q_scale, TensorView arg_k_scale, TensorView arg_v_scale, TensorView arg_block_index, TensorView arg_block_nums, TensorView arg_block_sizes, int64_t arg_seqlen_q, int64_t arg_seqlen_k, int64_t arg_num_heads, int64_t arg_num_kv_heads, int64_t arg_q_blocks, int64_t arg_k_blocks, int64_t arg_k_scale_groups, int64_t arg_capacity, int64_t arg_block_sparse_num, int64_t arg_has_block_nums, int64_t arg_block_sizes_mode, int64_t arg_v_scale_batch_stride, int64_t arg_total_tiles, double arg_softmax_scale_log2, int64_t arg_return_lse, TensorView arg_tma_descriptor_workspace, int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  DLDevice dev = arg_q.device();
  ScopedCudaDevice device_guard(dev.device_id);
  check_cuda_tensor(arg_q, "q");
  check_dtype(arg_q, dl_uint8, "q");
  check_contiguous(arg_q, "q");
  check_cuda_tensor(arg_k, "k");
  check_dtype(arg_k, dl_uint8, "k");
  check_contiguous(arg_k, "k");
  check_cuda_tensor(arg_v, "v");
  check_dtype(arg_v, dl_uint8, "v");
  check_contiguous(arg_v, "v");
  check_cuda_tensor(arg_out, "out");
  check_dtype(arg_out, dl_bfloat16, "out");
  check_contiguous(arg_out, "out");
  check_cuda_tensor(arg_lse, "lse");
  check_dtype(arg_lse, dl_float32, "lse");
  check_contiguous(arg_lse, "lse");
  check_cuda_tensor(arg_q_scale, "q_scale");
  check_dtype(arg_q_scale, dl_float32, "q_scale");
  check_contiguous(arg_q_scale, "q_scale");
  check_cuda_tensor(arg_k_scale, "k_scale");
  check_dtype(arg_k_scale, dl_float32, "k_scale");
  check_contiguous(arg_k_scale, "k_scale");
  check_cuda_tensor(arg_v_scale, "v_scale");
  check_dtype(arg_v_scale, dl_float32, "v_scale");
  check_contiguous(arg_v_scale, "v_scale");
  check_cuda_tensor(arg_block_index, "block_index");
  check_dtype(arg_block_index, dl_int32, "block_index");
  check_contiguous(arg_block_index, "block_index");
  check_cuda_tensor(arg_block_nums, "block_nums");
  check_dtype(arg_block_nums, dl_int32, "block_nums");
  check_contiguous(arg_block_nums, "block_nums");
  check_cuda_tensor(arg_block_sizes, "block_sizes");
  check_dtype(arg_block_sizes, dl_int32, "block_sizes");
  check_contiguous(arg_block_sizes, "block_sizes");
  TVM_FFI_CHECK(arg_seqlen_q >= -2147483648LL && arg_seqlen_q <= 2147483647LL, ValueError)
      << "scalar 'seqlen_q' value " << arg_seqlen_q
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_seqlen_k >= -2147483648LL && arg_seqlen_k <= 2147483647LL, ValueError)
      << "scalar 'seqlen_k' value " << arg_seqlen_k
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_num_heads >= -2147483648LL && arg_num_heads <= 2147483647LL, ValueError)
      << "scalar 'num_heads' value " << arg_num_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_num_kv_heads >= -2147483648LL && arg_num_kv_heads <= 2147483647LL, ValueError)
      << "scalar 'num_kv_heads' value " << arg_num_kv_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_q_blocks >= -2147483648LL && arg_q_blocks <= 2147483647LL, ValueError)
      << "scalar 'q_blocks' value " << arg_q_blocks
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_k_blocks >= -2147483648LL && arg_k_blocks <= 2147483647LL, ValueError)
      << "scalar 'k_blocks' value " << arg_k_blocks
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_k_scale_groups >= -2147483648LL && arg_k_scale_groups <= 2147483647LL, ValueError)
      << "scalar 'k_scale_groups' value " << arg_k_scale_groups
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_capacity >= -2147483648LL && arg_capacity <= 2147483647LL, ValueError)
      << "scalar 'capacity' value " << arg_capacity
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_block_sparse_num >= -2147483648LL && arg_block_sparse_num <= 2147483647LL, ValueError)
      << "scalar 'block_sparse_num' value " << arg_block_sparse_num
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_has_block_nums >= -2147483648LL && arg_has_block_nums <= 2147483647LL, ValueError)
      << "scalar 'has_block_nums' value " << arg_has_block_nums
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_block_sizes_mode >= -2147483648LL && arg_block_sizes_mode <= 2147483647LL, ValueError)
      << "scalar 'block_sizes_mode' value " << arg_block_sizes_mode
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_v_scale_batch_stride >= -2147483648LL && arg_v_scale_batch_stride <= 2147483647LL, ValueError)
      << "scalar 'v_scale_batch_stride' value " << arg_v_scale_batch_stride
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_total_tiles >= -2147483648LL && arg_total_tiles <= 2147483647LL, ValueError)
      << "scalar 'total_tiles' value " << arg_total_tiles
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_return_lse >= -2147483648LL && arg_return_lse <= 2147483647LL, ValueError)
      << "scalar 'return_lse' value " << arg_return_lse
      << " is outside i32 range [-2147483648, 2147483647]";
  check_same_device(arg_k, arg_q, "k", "q");
  check_same_device(arg_v, arg_q, "v", "q");
  check_same_device(arg_out, arg_q, "out", "q");
  check_same_device(arg_lse, arg_q, "lse", "q");
  check_same_device(arg_q_scale, arg_q, "q_scale", "q");
  check_same_device(arg_k_scale, arg_q, "k_scale", "q");
  check_same_device(arg_v_scale, arg_q, "v_scale", "q");
  check_same_device(arg_block_index, arg_q, "block_index", "q");
  check_same_device(arg_block_nums, arg_q, "block_nums", "q");
  check_same_device(arg_block_sizes, arg_q, "block_sizes", "q");
  check_cuda_tensor(arg_tma_descriptor_workspace, "tma_descriptor_workspace");
  check_same_device(arg_tma_descriptor_workspace, arg_q, "tma_descriptor_workspace", "q");
  check_contiguous(arg_tma_descriptor_workspace, "tma_descriptor_workspace");
  check_dtype(arg_tma_descriptor_workspace, dl_uint8, "tma_descriptor_workspace");
  TVM_FFI_CHECK(arg_tma_descriptor_workspace.numel() >= 384, ValueError)
      << "tma_descriptor_workspace requires at least 384 bytes";
  TVM_FFI_CHECK(
      reinterpret_cast<uintptr_t>(arg_tma_descriptor_workspace.data_ptr()) % 128 == 0,
      ValueError)
      << "tma_descriptor_workspace must be 128-byte aligned";
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";

  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(dev.device_type, dev.device_id);
  CUtensorMap h_q = EncodeTma_q(arg_q);
  CakeTensorMap const* p_q = static_cast<CakeTensorMap const*>(CallerTmaWorkspaceSlot(arg_tma_descriptor_workspace, 0u));
  CUtensorMap h_k = EncodeTma_k(arg_k);
  CakeTensorMap const* p_k = static_cast<CakeTensorMap const*>(CallerTmaWorkspaceSlot(arg_tma_descriptor_workspace, 1u));
  CUtensorMap h_v = EncodeTma_v(arg_v);
  CakeTensorMap const* p_v = static_cast<CakeTensorMap const*>(CallerTmaWorkspaceSlot(arg_tma_descriptor_workspace, 2u));
  __nv_bfloat16* p_out = static_cast<__nv_bfloat16*>(arg_out.data_ptr());
  float* p_lse = static_cast<float*>(arg_lse.data_ptr());
  float* p_q_scale = static_cast<float*>(arg_q_scale.data_ptr());
  float* p_k_scale = static_cast<float*>(arg_k_scale.data_ptr());
  float* p_v_scale = static_cast<float*>(arg_v_scale.data_ptr());
  int* p_block_index = static_cast<int*>(arg_block_index.data_ptr());
  int* p_block_nums = static_cast<int*>(arg_block_nums.data_ptr());
  int* p_block_sizes = static_cast<int*>(arg_block_sizes.data_ptr());
  int32_t v_seqlen_q = (int32_t)arg_seqlen_q;
  int32_t v_seqlen_k = (int32_t)arg_seqlen_k;
  int32_t v_num_heads = (int32_t)arg_num_heads;
  int32_t v_num_kv_heads = (int32_t)arg_num_kv_heads;
  int32_t v_q_blocks = (int32_t)arg_q_blocks;
  int32_t v_k_blocks = (int32_t)arg_k_blocks;
  int32_t v_k_scale_groups = (int32_t)arg_k_scale_groups;
  int32_t v_capacity = (int32_t)arg_capacity;
  int32_t v_block_sparse_num = (int32_t)arg_block_sparse_num;
  int32_t v_has_block_nums = (int32_t)arg_has_block_nums;
  int32_t v_block_sizes_mode = (int32_t)arg_block_sizes_mode;
  int32_t v_v_scale_batch_stride = (int32_t)arg_v_scale_batch_stride;
  int32_t v_total_tiles = (int32_t)arg_total_tiles;
  float v_softmax_scale_log2 = (float)arg_softmax_scale_log2;
  int32_t v_return_lse = (int32_t)arg_return_lse;
  void* kargs[] = {&p_q, &p_k, &p_v, &p_out, &p_lse, &p_q_scale, &p_k_scale, &p_v_scale, &p_block_index, &p_block_nums, &p_block_sizes, &v_seqlen_q, &v_seqlen_k, &v_num_heads, &v_num_kv_heads, &v_q_blocks, &v_k_blocks, &v_k_scale_groups, &v_capacity, &v_block_sparse_num, &v_has_block_nums, &v_block_sizes_mode, &v_v_scale_batch_stride, &v_total_tiles, &v_softmax_scale_log2, &v_return_lse};

  static std::mutex smem_mu;
  static auto* smem_status_by_device =
      new std::unordered_map<int, cudaError_t>();
  cudaError_t smem_status = cudaSuccess;
  {
    std::lock_guard<std::mutex> lock(smem_mu);
    auto it = smem_status_by_device->find(dev.device_id);
    if (it == smem_status_by_device->end()) {
      int current_device = -1;
      smem_status = cudaGetDevice(&current_device);
      if (smem_status == cudaSuccess && current_device != dev.device_id) {
        smem_status = cudaErrorInvalidDevice;
      }
      if (smem_status == cudaSuccess) {
        smem_status = cudaFuncSetAttribute(
            reinterpret_cast<const void*>(kernel_cake_sage_block_sparse_attention_f63b7fc8674de9925f7e),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            77824);
      }
      smem_status_by_device->emplace(dev.device_id, smem_status);
    } else {
      smem_status = it->second;
    }
  }
  TVM_FFI_CHECK(smem_status == cudaSuccess, RuntimeError)
      << "cudaFuncSetAttribute for kernel_cake_sage_block_sparse_attention_f63b7fc8674de9925f7e failed: "
      << cudaGetErrorString(smem_status);
  dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  dim3 block(512u, 1u, 1u);
  const CUtensorMap workspace_maps[] = {h_q, h_k, h_v};
  UploadCallerTmaWorkspace(arg_tma_descriptor_workspace.data_ptr(), workspace_maps, stream);

  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 77824u;
  config.stream = stream;
  cudaError_t launch_status = cudaLaunchKernelExC(
      &config, reinterpret_cast<const void*>(kernel_cake_sage_block_sparse_attention_f63b7fc8674de9925f7e), kargs);
  TVM_FFI_CHECK(launch_status == cudaSuccess, RuntimeError)
      << "cudaLaunchKernelExC for kernel_cake_sage_block_sparse_attention_f63b7fc8674de9925f7e failed: "
      << cudaGetErrorString(launch_status);
}

}  // namespace cake_host_shim_c802b60dd9d50557

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, cake_host_shim_c802b60dd9d50557::Run);
