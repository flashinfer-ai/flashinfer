/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "tvm_ffi_utils.h"

// Keep the source-level generated declarations private to this translation
// unit and separate from the CUDA Driver API's CUtensorMap declaration.
#define uint8_t cake_trtllm_mla_generated_uint8_t
#define uint16_t cake_trtllm_mla_generated_uint16_t
#define uint32_t cake_trtllm_mla_generated_uint32_t
#define uint64_t cake_trtllm_mla_generated_uint64_t
#define int32_t cake_trtllm_mla_generated_int32_t
#define int16_t cake_trtllm_mla_generated_int16_t
#define CakeTensorMap cake_trtllm_mla_generated_CakeTensorMap
#define CakeTensorMapPack cake_trtllm_mla_generated_CakeTensorMapPack
#define CUtensorMap cake_trtllm_mla_generated_CUtensorMap
#include "cake_trtllm_mla_bf16_low_batch_single_launch.cu"
#undef CUtensorMap
#undef CakeTensorMapPack
#undef CakeTensorMap
#undef int16_t
#undef int32_t
#undef uint64_t
#undef uint32_t
#undef uint16_t
#undef uint8_t

namespace flashinfer {
namespace cake_trtllm_mla {

constexpr int32_t kHeads = 128;
constexpr int32_t kHeadDim = 576;
constexpr int32_t kValueDim = 512;
constexpr int32_t kPageSize = 32;
constexpr int32_t kMaxSequenceLength = 1024;
constexpr int32_t kClusterSize = 2;
constexpr int32_t kThreads = 384;
constexpr int32_t kDynamicSmemBytes = 227840;

static_assert(THREADS == kThreads);
static_assert(SMEM_TOTAL == kDynamicSmemBytes);
static_assert(sizeof(CUtensorMap) == 128);
static_assert(sizeof(cake_trtllm_mla_generated_CakeTensorMap) == sizeof(CUtensorMap));

inline void CakeCheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK(status == cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

inline void CakeCheckDriver(CUresult status, const char* operation) {
  TVM_FFI_ICHECK(status == CUDA_SUCCESS)
      << operation << " failed with CUresult=" << static_cast<int>(status);
}

inline void CakeCheckCudaTensor(const TensorView& tensor, const char* name) {
  TVM_FFI_ICHECK(tensor.device().device_type == kDLCUDA) << name << " must be a CUDA tensor";
}

inline void CakeCheckSameDevice(const TensorView& tensor, const TensorView& reference,
                                const char* name) {
  TVM_FFI_ICHECK(tensor.device().device_id == reference.device().device_id)
      << name << " must be on the same CUDA device as query";
}

inline void CakeCheckBFloat16(const TensorView& tensor, const char* name) {
  const DLDataType dtype = tensor.dtype();
  TVM_FFI_ICHECK(dtype.code == kDLBfloat && dtype.bits == 16 && dtype.lanes == 1)
      << name << " must have bfloat16 dtype";
}

inline void CakeCheckInt32(const TensorView& tensor, const char* name) {
  const DLDataType dtype = tensor.dtype();
  TVM_FFI_ICHECK(dtype.code == kDLInt && dtype.bits == 32 && dtype.lanes == 1)
      << name << " must have int32 dtype";
}

inline CUtensorMap CakeEncodeQueryTma(const TensorView& query) {
  const uint64_t outer = static_cast<uint64_t>(query.numel() / kHeadDim);
  const uint64_t global_dims[3] = {64, outer, kHeadDim / 64};
  const uint64_t global_strides[2] = {
      static_cast<uint64_t>(query.stride(query.ndim() - 2) * sizeof(__nv_bfloat16)),
      64 * sizeof(__nv_bfloat16),
  };
  const uint32_t box_dims[3] = {64, 64, 1};
  const uint32_t element_strides[3] = {1, 1, 1};
  CUtensorMap tensor_map{};
  CakeCheckDriver(
      cuTensorMapEncodeTiled(&tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, query.data_ptr(),
                             global_dims, global_strides, box_dims, element_strides,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                             CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled(query)");
  return tensor_map;
}

inline CUtensorMap CakeEncodeKvTma(const TensorView& kv_cache) {
  const uint64_t outer = static_cast<uint64_t>(kv_cache.numel() / kHeadDim);
  const uint64_t global_dims[2] = {kHeadDim, outer};
  const uint64_t global_strides[1] = {
      static_cast<uint64_t>(kv_cache.stride(kv_cache.ndim() - 2) * sizeof(__nv_bfloat16)),
  };
  const uint32_t box_dims[2] = {64, 1};
  const uint32_t element_strides[2] = {1, 1};
  CUtensorMap tensor_map{};
  CakeCheckDriver(
      cuTensorMapEncodeTiled(&tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, kv_cache.data_ptr(),
                             global_dims, global_strides, box_dims, element_strides,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                             CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled(kv_cache)");
  return tensor_map;
}

struct CakeTmaArena {
  static constexpr size_t kSlotsPerChunk = 256;
  static constexpr size_t kMaxSlots = 4096;
  std::vector<CUdeviceptr> chunks;
  size_t used = 0;
};

inline void* CakeTmaDeviceSlot(const CUtensorMap& tensor_map, int32_t device_id,
                               cudaStream_t stream) {
  static std::mutex mutex;
  static auto* slots = new std::unordered_map<std::string, void*>();
  static auto* arenas = new std::unordered_map<CUcontext, CakeTmaArena>();

  CUcontext context = nullptr;
  CakeCheckDriver(cuCtxGetCurrent(&context), "cuCtxGetCurrent");
  TVM_FFI_ICHECK(context != nullptr) << "Cake TMA pointer ABI requires an active CUDA context";
  CUdevice context_device = -1;
  CakeCheckDriver(cuCtxGetDevice(&context_device), "cuCtxGetDevice");
  TVM_FFI_ICHECK(context_device == device_id) << "Cake TMA descriptor device mismatch";

  std::string key = std::to_string(reinterpret_cast<uintptr_t>(context));
  key.push_back(':');
  key.append(reinterpret_cast<const char*>(&tensor_map), sizeof(tensor_map));
  std::lock_guard<std::mutex> lock(mutex);
  const auto existing = slots->find(key);
  if (existing != slots->end()) {
    return existing->second;
  }

  CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
  CakeCheckDriver(cuStreamIsCapturing(reinterpret_cast<CUstream>(stream), &capture_status),
                  "cuStreamIsCapturing(Cake TMA descriptor)");
  TVM_FFI_ICHECK(capture_status == CU_STREAM_CAPTURE_STATUS_NONE)
      << "Cake pointer TMA ABI cannot allocate a descriptor during CUDA Graph "
         "capture; prewarm this exact tensor binding before capture";

  CakeTmaArena& arena = (*arenas)[context];
  TVM_FFI_ICHECK(arena.used < CakeTmaArena::kMaxSlots) << "Cake TMA descriptor arena exhausted";
  if (arena.used % CakeTmaArena::kSlotsPerChunk == 0) {
    CUdeviceptr chunk = 0;
    CakeCheckDriver(cuMemAlloc(&chunk, CakeTmaArena::kSlotsPerChunk * sizeof(CUtensorMap)),
                    "cuMemAlloc(Cake TMA arena)");
    arena.chunks.push_back(chunk);
  }
  const size_t chunk_index = arena.used / CakeTmaArena::kSlotsPerChunk;
  const size_t slot_index = arena.used % CakeTmaArena::kSlotsPerChunk;
  const CUdeviceptr device_pointer = arena.chunks[chunk_index] + slot_index * sizeof(CUtensorMap);
  CakeCheckDriver(cuMemcpyHtoD(device_pointer, &tensor_map, sizeof(tensor_map)),
                  "cuMemcpyHtoD(Cake TMA descriptor)");
  ++arena.used;
  void* pointer = reinterpret_cast<void*>(static_cast<uintptr_t>(device_pointer));
  (*slots)[key] = pointer;
  return pointer;
}

void CakeRun(TensorView query, TensorView kv_cache, TensorView output, TensorView block_tables,
             TensorView seq_lens, int64_t q_len, int64_t source_table_width,
             int64_t max_sequence_length, double softmax_scale_log2, int64_t total_work_items,
             int64_t grid_x, int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0) << "cuda_stream must be non-negative";
  CakeCheckCudaTensor(query, "query");
  const int32_t device_id = query.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);

  int32_t major = 0;
  int32_t minor = 0;
  CakeCheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
                "cudaDeviceGetAttribute(major)");
  CakeCheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
                "cudaDeviceGetAttribute(minor)");
  TVM_FFI_ICHECK(major == 10 && minor == 3)
      << "Cake TRT-LLM MLA requires compute capability 10.3, got " << major << "." << minor;

  for (const auto& named : std::array<std::pair<const TensorView*, const char*>, 4>{
           std::pair{&kv_cache, "kv_cache"}, std::pair{&output, "output"},
           std::pair{&block_tables, "block_tables"}, std::pair{&seq_lens, "seq_lens"}}) {
    CakeCheckCudaTensor(*named.first, named.second);
    CakeCheckSameDevice(*named.first, query, named.second);
  }
  CakeCheckBFloat16(query, "query");
  CakeCheckBFloat16(kv_cache, "kv_cache");
  CakeCheckBFloat16(output, "output");
  CakeCheckInt32(block_tables, "block_tables");
  CakeCheckInt32(seq_lens, "seq_lens");
  TVM_FFI_ICHECK(query.IsContiguous()) << "query must be contiguous";
  TVM_FFI_ICHECK(kv_cache.IsContiguous()) << "kv_cache must be contiguous";
  TVM_FFI_ICHECK(output.IsContiguous()) << "output must be contiguous";
  TVM_FFI_ICHECK(block_tables.IsContiguous()) << "block_tables must be contiguous";
  TVM_FFI_ICHECK(seq_lens.IsContiguous()) << "seq_lens must be contiguous";

  TVM_FFI_ICHECK(query.ndim() == 4 && query.size(0) > 0 && query.size(1) == q_len &&
                 query.size(2) == kHeads && query.size(3) == kHeadDim)
      << "query must have shape [B, q_len, 128, 576]";
  const int64_t batch = query.size(0);
  TVM_FFI_ICHECK((kv_cache.ndim() == 3 || kv_cache.ndim() == 4) &&
                 kv_cache.size(kv_cache.ndim() - 2) == kPageSize &&
                 kv_cache.size(kv_cache.ndim() - 1) == kHeadDim)
      << "kv_cache must have page size 32 and head dimension 576";
  if (kv_cache.ndim() == 4) {
    TVM_FFI_ICHECK(kv_cache.size(1) == 1) << "4D kv_cache must have singleton head axis";
  }
  TVM_FFI_ICHECK(output.ndim() == 4 && output.size(0) == batch && output.size(1) == q_len &&
                 output.size(2) == kHeads && output.size(3) == kValueDim)
      << "output must have shape [B, q_len, 128, 512]";
  TVM_FFI_ICHECK(block_tables.ndim() == 2 && block_tables.size(0) == batch &&
                 block_tables.size(1) == source_table_width &&
                 source_table_width >= kMaxSequenceLength / kPageSize)
      << "block_tables must have shape [B, width] with width >= 32";
  TVM_FFI_ICHECK(seq_lens.ndim() == 1 && seq_lens.size(0) == batch)
      << "seq_lens must have shape [B]";
  TVM_FFI_ICHECK(q_len > 0 && q_len <= 16) << "q_len must be in [1, 16]";
  TVM_FFI_ICHECK(max_sequence_length == kMaxSequenceLength)
      << "Cake TRT-LLM MLA physical sequence extent must be 1024";
  TVM_FFI_ICHECK(total_work_items == batch * q_len * 2)
      << "total_work_items must equal B * q_len * 2";
  TVM_FFI_ICHECK(grid_x > 0 && grid_x % kClusterSize == 0 &&
                 grid_x <= std::numeric_limits<uint32_t>::max())
      << "grid_x must be positive, even, and fit uint32";
  TVM_FFI_ICHECK(std::isfinite(softmax_scale_log2) && softmax_scale_log2 > 0.0)
      << "softmax_scale_log2 must be finite and positive";

  int32_t max_dynamic_smem = 0;
  CakeCheckCuda(
      cudaDeviceGetAttribute(&max_dynamic_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id),
      "cudaDeviceGetAttribute(max dynamic shared memory)");
  TVM_FFI_ICHECK(max_dynamic_smem >= kDynamicSmemBytes)
      << "device dynamic shared-memory capacity is insufficient";
  CakeCheckCuda(
      cudaFuncSetAttribute(kernel_cake_trtllm_mla_bf16_low_batch_single_launch,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSmemBytes),
      "cudaFuncSetAttribute(Cake TRT-LLM MLA)");

  const CUtensorMap query_map = CakeEncodeQueryTma(query);
  const CUtensorMap kv_map = CakeEncodeKvTma(kv_cache);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  auto* query_map_device = reinterpret_cast<cake_trtllm_mla_generated_CakeTensorMap const*>(
      CakeTmaDeviceSlot(query_map, device_id, stream));
  auto* kv_map_device = reinterpret_cast<cake_trtllm_mla_generated_CakeTensorMap const*>(
      CakeTmaDeviceSlot(kv_map, device_id, stream));

  cudaLaunchAttribute attributes[2]{};
  attributes[0].id = cudaLaunchAttributeClusterDimension;
  attributes[0].val.clusterDim.x = kClusterSize;
  attributes[0].val.clusterDim.y = 1;
  attributes[0].val.clusterDim.z = 1;
  attributes[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
  attributes[1].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicySpread;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<uint32_t>(grid_x), 1, 1);
  config.blockDim = dim3(kThreads, 1, 1);
  config.dynamicSmemBytes = kDynamicSmemBytes;
  config.stream = stream;
  config.attrs = attributes;
  config.numAttrs = 2;

  CakeCheckCuda(
      cudaLaunchKernelEx(
          &config, kernel_cake_trtllm_mla_bf16_low_batch_single_launch, query_map_device,
          kv_map_device, reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
          reinterpret_cast<int32_t*>(block_tables.data_ptr()),
          reinterpret_cast<int32_t*>(seq_lens.data_ptr()), static_cast<int32_t>(q_len),
          static_cast<int32_t>(source_table_width), static_cast<int32_t>(max_sequence_length),
          static_cast<float>(softmax_scale_log2), static_cast<int32_t>(total_work_items)),
      "cudaLaunchKernelEx(Cake TRT-LLM MLA)");
}

}  // namespace cake_trtllm_mla
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::cake_trtllm_mla::CakeRun);
