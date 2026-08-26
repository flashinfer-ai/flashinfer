/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 */

#include "flashkda_binding_common.cuh"

extern "C" __global__ void kernel_flashkda_forward_checkpoint_c16(
    unsigned int*, const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap,
    const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap, __nv_bfloat16*,
    const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap, __nv_bfloat16*,
    __nv_bfloat16*, float*, float*, long long*, long long*, int*, float*, float*, int, int,
    int, int, int, int, float, float);

namespace flashinfer {
namespace flash_kda_training {

using flash_kda::CheckCuda;
using flash_kda::CheckCudaTensor;
using flash_kda::CheckDtype;
using flash_kda::CheckDynamicSmemCapacity;
using flash_kda::CheckFlashKDATarget;

constexpr int64_t kTokens = 8192;
constexpr int64_t kSequences = 8;
constexpr int64_t kHeads = 96;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kChunks = 512;
constexpr int64_t kWorkItems = kSequences * kHeads;
constexpr size_t kTensorMapCount = 6;
constexpr size_t kDescriptorBytes = kTensorMapCount * sizeof(CUtensorMap);
constexpr int32_t kSmemBytes = 230016;

inline void CheckTensor(const TensorView& tensor, const char* name, int32_t device_id,
                        DLDataType dtype) {
  CheckCudaTensor(tensor, name, device_id);
  CheckDtype(tensor, name, dtype);
}

inline void CheckNumel(const TensorView& tensor, const char* name, int64_t numel) {
  TVM_FFI_ICHECK(tensor.numel() == numel) << name << " must contain " << numel << " elements";
}

inline CUtensorMap EncodeTokenTensor(const TensorView& tensor, const char* name) {
  uint64_t global_dim[3] = {kHeadDim, kHeads, kTokens};
  uint64_t global_strides[2] = {kHeadDim * sizeof(__nv_bfloat16),
                                kHeadDim * kHeads * sizeof(__nv_bfloat16)};
  uint32_t box_dim[3] = {64, 1, 16};
  uint32_t elem_strides[3] = {1, 1, 1};
  CUtensorMap map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, tensor.data_ptr(), global_dim, global_strides,
      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for " << name << " with CUresult=" << int(result);
  return map;
}

inline CUtensorMap EncodeCheckpointTensor(const TensorView& tensor) {
  uint64_t global_dim[4] = {kHeadDim, kHeadDim, kHeads, kChunks};
  uint64_t global_strides[3] = {kHeadDim * sizeof(__nv_bfloat16),
                                kHeadDim * kHeadDim * sizeof(__nv_bfloat16),
                                kHeadDim * kHeadDim * kHeads * sizeof(__nv_bfloat16)};
  uint32_t box_dim[4] = {64, 128, 1, 1};
  uint32_t elem_strides[4] = {1, 1, 1, 1};
  CUtensorMap map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, tensor.data_ptr(), global_dim, global_strides,
      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for checkpoints with CUresult=" << int(result);
  return map;
}

struct TensorMapWords {
  uint64_t words[kDescriptorBytes / sizeof(uint64_t)];
};

static __global__ void PublishTensorMaps(uint64_t* destination, TensorMapWords source) {
  if (threadIdx.x < kDescriptorBytes / sizeof(uint64_t)) {
    destination[threadIdx.x] = source.words[threadIdx.x];
  }
}

void RunForward(TensorView q, TensorView k, TensorView v, TensorView g, TensorView beta,
                TensorView A_log, TensorView dt_bias, TensorView initial_state,
                TensorView cu_seqlens, TensorView checkpoint_cu_starts, TensorView work_items,
                TensorView descriptor_storage, TensorView out, TensorView final_state_bf16,
                TensorView final_state, TensorView state_checkpoints, TensorView beta_active,
                TensorView counter, int64_t prepare_descriptors, int64_t num_sequences,
                int64_t num_heads, double scale, double lower_bound, int64_t cuda_stream) {
  TVM_FFI_ICHECK(q.device().device_type == kDLCUDA) << "q must be a CUDA tensor";
  const int32_t device_id = q.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckFlashKDATarget(device_id);
  TVM_FFI_ICHECK(num_sequences == kSequences && num_heads == kHeads)
      << "the training forward requires eight sequences and 96 heads";
  TVM_FFI_ICHECK(prepare_descriptors == 0 || prepare_descriptors == 1);
  TVM_FFI_ICHECK(std::abs(scale - 1.0 / std::sqrt(128.0)) <= 1e-15)
      << "the training forward fixes scale=1/sqrt(128)";
  TVM_FFI_ICHECK(lower_bound == -5.0) << "the training forward fixes lower_bound=-5.0";

  for (const auto& named : std::initializer_list<std::pair<TensorView*, const char*>>{
           {&q, "q"},
           {&k, "k"},
           {&v, "v"},
           {&g, "g"},
           {&beta, "beta"},
           {&out, "out"},
           {&final_state_bf16, "final_state_bf16"},
           {&state_checkpoints, "state_checkpoints"},
           {&beta_active, "beta_active"}}) {
    CheckTensor(*named.first, named.second, device_id, dl_bfloat16);
  }
  for (const auto& named :
       std::initializer_list<std::pair<TensorView*, const char*>>{{&A_log, "A_log"},
                                                                  {&dt_bias, "dt_bias"},
                                                                  {&initial_state, "initial_state"},
                                                                  {&final_state, "final_state"}}) {
    CheckTensor(*named.first, named.second, device_id, dl_float32);
  }
  CheckTensor(cu_seqlens, "cu_seqlens", device_id, dl_int64);
  CheckTensor(checkpoint_cu_starts, "checkpoint_cu_starts", device_id, dl_int64);
  CheckTensor(work_items, "work_items", device_id, dl_int32);
  CheckTensor(descriptor_storage, "descriptor_storage", device_id, dl_uint8);
  CheckTensor(counter, "counter", device_id, dl_uint32);
  CheckNumel(q, "q", kTokens * kHeads * kHeadDim);
  CheckNumel(beta, "beta", kTokens * kHeads);
  CheckNumel(initial_state, "initial_state", kSequences * kHeads * kHeadDim * kHeadDim);
  CheckNumel(final_state_bf16, "final_state_bf16", kSequences * kHeads * kHeadDim * kHeadDim);
  CheckNumel(final_state, "final_state", kSequences * kHeads * kHeadDim * kHeadDim);
  CheckNumel(state_checkpoints, "state_checkpoints", kChunks * kHeads * kHeadDim * kHeadDim);
  CheckNumel(beta_active, "beta_active", kTokens * kHeads);
  CheckNumel(work_items, "work_items", kWorkItems * 8);
  TVM_FFI_ICHECK(descriptor_storage.numel() >= static_cast<int64_t>(kDescriptorBytes));
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(descriptor_storage.data_ptr()) % 64 == 0);

  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  if (prepare_descriptors != 0) {
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    CheckCuda(cudaStreamIsCapturing(stream, &capture_status), "cudaStreamIsCapturing");
    TVM_FFI_ICHECK(capture_status == cudaStreamCaptureStatusNone)
        << "training-forward descriptors must be warmed before CUDA graph "
           "capture";
    const std::array<CUtensorMap, kTensorMapCount> maps = {
        EncodeTokenTensor(q, "q"),     EncodeTokenTensor(k, "k"),
        EncodeTokenTensor(v, "v"),     EncodeTokenTensor(g, "g"),
        EncodeTokenTensor(out, "out"), EncodeCheckpointTensor(state_checkpoints),
    };
    TensorMapWords words{};
    std::memcpy(words.words, maps.data(), sizeof(maps));
    PublishTensorMaps<<<1, kDescriptorBytes / sizeof(uint64_t), 0, stream>>>(
        reinterpret_cast<uint64_t*>(descriptor_storage.data_ptr()), words);
    CheckCuda(cudaGetLastError(), "PublishTensorMaps launch");
  }
  auto* descriptor_bytes = static_cast<unsigned char*>(descriptor_storage.data_ptr());
  constexpr size_t stride = sizeof(CUtensorMap);
  int resident_ctas = 0;
  CheckCuda(cudaDeviceGetAttribute(&resident_ctas, cudaDevAttrMultiProcessorCount, device_id),
            "cudaDeviceGetAttribute(multiProcessorCount)");
  const dim3 grid(std::min<int64_t>(kWorkItems, resident_ctas), 1, 1);
  CheckDynamicSmemCapacity(device_id, kSmemBytes);
  CheckCuda(cudaFuncSetAttribute(kernel_flashkda_forward_checkpoint_c16,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes),
            "cudaFuncSetAttribute(training forward)");
  kernel_flashkda_forward_checkpoint_c16<<<grid, 512, kSmemBytes, stream>>>(
      reinterpret_cast<unsigned int*>(counter.data_ptr()),
      *reinterpret_cast<CUtensorMap const*>(descriptor_bytes + 0 * stride),
      *reinterpret_cast<CUtensorMap const*>(descriptor_bytes + 1 * stride),
      *reinterpret_cast<CUtensorMap const*>(descriptor_bytes + 2 * stride),
      *reinterpret_cast<CUtensorMap const*>(descriptor_bytes + 3 * stride),
      reinterpret_cast<__nv_bfloat16*>(g.data_ptr()),
      *reinterpret_cast<CUtensorMap const*>(descriptor_bytes + 4 * stride),
      *reinterpret_cast<CUtensorMap const*>(descriptor_bytes + 5 * stride),
      reinterpret_cast<__nv_bfloat16*>(beta.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(beta_active.data_ptr()),
      reinterpret_cast<float*>(A_log.data_ptr()), reinterpret_cast<float*>(dt_bias.data_ptr()),
      reinterpret_cast<long long*>(cu_seqlens.data_ptr()),
      reinterpret_cast<long long*>(checkpoint_cu_starts.data_ptr()),
      reinterpret_cast<int*>(work_items.data_ptr()),
      reinterpret_cast<float*>(initial_state.data_ptr()),
      reinterpret_cast<float*>(final_state.data_ptr()), kWorkItems, 1, kHeads, kHeads, kHeads, 16,
      static_cast<float>(scale), static_cast<float>(lower_bound));
  CheckCuda(cudaGetLastError(), "training forward launch");
}

}  // namespace flash_kda_training
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_forward, flashinfer::flash_kda_training::RunForward);
