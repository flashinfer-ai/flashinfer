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

struct alignas(128) FlashKDATrainingTensorMap {
  uint64_t opaque[16];
};

extern "C" {

__global__ void kernel_flashkda_forward_checkpoint_c16(
    unsigned int*, const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap,
    const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap, __nv_bfloat16*,
    const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap, __nv_bfloat16*,
    __nv_bfloat16*, float*, float*, long long*, long long*, int*, float*, __nv_bfloat16*, int, int,
    int, int, int, int, float, float);

__global__ void kernel_flashkda_backward_persistent_c16(
    unsigned int*, const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap,
    const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap,
    const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap, float*,
    const __grid_constant__ CUtensorMap, __nv_bfloat16*, __nv_bfloat16*, float*, float*, float*,
    float*, float*, const __grid_constant__ CUtensorMap, float*, long long*, long long*, int*,
    unsigned int*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*,
    float*, float*, float*, float*, int, int, int, int, int, int, int, float, float);

__global__ void kernel_flashkda_refine_forgetting_horizons(__nv_bfloat16*, float*, float*, int*,
                                                           int*, unsigned int*, int, float, float);

__global__ void kernel_flashkda_backward_param_reduce_c16_partial(
    __nv_bfloat16*, __nv_bfloat16*, float*, float*, float*, float*, float*, __nv_bfloat16*,
    __nv_bfloat16*, float*, float*, unsigned int*, float*, float*, int, int, int, int, float);

__global__ void kernel_flashkda_grouped_qk_reduce(__nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
                                                  __nv_bfloat16*, int, int, int);

__global__ void kernel_flashkda_blackwell_prefill_fp32_state_initial(
    const FlashKDATrainingTensorMap*, const FlashKDATrainingTensorMap*,
    const FlashKDATrainingTensorMap*, const FlashKDATrainingTensorMap*,
    const FlashKDATrainingTensorMap*, __nv_bfloat16*, float*, float*, long long*, float*, float*,
    float*, int*, uint8_t*, int, float, int, int, int, int, float);

}  // extern "C"

namespace flashinfer {
namespace flash_kda_training_paired {

using flash_kda::CheckCuda;
using flash_kda::CheckCudaTensor;
using flash_kda::CheckDtype;
using flash_kda::CheckDynamicSmemCapacity;
using flash_kda::CheckFlashKDATarget;

constexpr int64_t kHeadDim = 128;
constexpr int64_t kChunk = 16;
constexpr int32_t kForwardSmemBytes = 230016;
constexpr int32_t kBackwardSmemBytes = 230400;
constexpr int32_t kRefineSmemBytes = 128;
constexpr int32_t kReduceSmemBytes = 4224;
constexpr int32_t kFinalSmemBytes = 226048;
constexpr float kDefaultLog2Threshold = -14.426950408889635f;
constexpr int64_t kFinalDescriptorCount = 5;
constexpr int64_t kFinalDescriptorBytes = kFinalDescriptorCount * sizeof(CUtensorMap);
constexpr int64_t kFinalWorkspaceBytesPerCta = 10 * 128;

inline void CheckTensor(const TensorView& tensor, const char* name, int32_t device_id,
                        DLDataType dtype) {
  CheckCudaTensor(tensor, name, device_id);
  CheckDtype(tensor, name, dtype);
}

inline void CheckElements(const TensorView& tensor, const char* name, int64_t elements) {
  TVM_FFI_ICHECK(tensor.numel() == elements)
      << name << " must contain " << elements << " elements, got " << tensor.numel();
}

inline CUtensorMap EncodeTokenMap(const TensorView& tensor, const char* name, int64_t total_tokens,
                                  int64_t num_heads) {
  uint64_t global_dim[3] = {kHeadDim, static_cast<uint64_t>(num_heads),
                            static_cast<uint64_t>(total_tokens)};
  uint64_t global_strides[2] = {
      kHeadDim * sizeof(__nv_bfloat16),
      kHeadDim * static_cast<uint64_t>(num_heads) * sizeof(__nv_bfloat16)};
  uint32_t box_dim[3] = {64, 1, kChunk};
  uint32_t element_strides[3] = {1, 1, 1};
  CUtensorMap map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, tensor.data_ptr(), global_dim, global_strides,
      box_dim, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for " << name << " with CUresult=" << int(result);
  return map;
}

inline CUtensorMap EncodeCheckpointMap(const TensorView& tensor, int64_t total_chunks,
                                       int64_t num_heads) {
  uint64_t global_dim[4] = {kHeadDim, kHeadDim, static_cast<uint64_t>(num_heads),
                            static_cast<uint64_t>(total_chunks)};
  uint64_t global_strides[3] = {
      kHeadDim * sizeof(__nv_bfloat16), kHeadDim * kHeadDim * sizeof(__nv_bfloat16),
      kHeadDim * kHeadDim * static_cast<uint64_t>(num_heads) * sizeof(__nv_bfloat16)};
  uint32_t box_dim[4] = {64, kHeadDim, 1, 1};
  uint32_t element_strides[4] = {1, 1, 1, 1};
  CUtensorMap map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, tensor.data_ptr(), global_dim, global_strides,
      box_dim, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for state checkpoints with CUresult=" << int(result);
  return map;
}

inline CUtensorMap EncodeBetaMap(const TensorView& tensor, int64_t total_tokens,
                                 int64_t beta_stride) {
  uint64_t global_dim[2] = {static_cast<uint64_t>(beta_stride),
                            static_cast<uint64_t>(total_tokens)};
  uint64_t global_strides[1] = {static_cast<uint64_t>(beta_stride) * sizeof(__nv_bfloat16)};
  uint32_t box_dim[2] = {8, kChunk};
  uint32_t element_strides[2] = {1, 1};
  CUtensorMap map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, tensor.data_ptr(), global_dim, global_strides,
      box_dim, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for beta_active with CUresult=" << int(result);
  return map;
}

inline CUtensorMap EncodeFinalMap(const TensorView& tensor, const char* name, int64_t total_tokens,
                                  int64_t num_heads) {
  uint64_t global_dim[3] = {kHeadDim, static_cast<uint64_t>(total_tokens),
                            static_cast<uint64_t>(num_heads)};
  uint64_t global_strides[2] = {kHeadDim * static_cast<uint64_t>(num_heads) * sizeof(__nv_bfloat16),
                                kHeadDim * sizeof(__nv_bfloat16)};
  uint32_t box_dim[3] = {64, 64, 1};
  uint32_t element_strides[3] = {1, 1, 1};
  CUtensorMap map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, tensor.data_ptr(), global_dim, global_strides,
      box_dim, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for " << name << " with CUresult=" << int(result);
  return map;
}

struct FinalTensorMapWords {
  uint64_t words[kFinalDescriptorBytes / sizeof(uint64_t)];
};

static __global__ void PublishFinalTensorMaps(uint64_t* destination, FinalTensorMapWords source) {
  if (threadIdx.x < kFinalDescriptorBytes / sizeof(uint64_t)) {
    destination[threadIdx.x] = source.words[threadIdx.x];
  }
}

inline void PrepareFinalTensorMaps(const TensorView& q, const TensorView& k, const TensorView& v,
                                   const TensorView& g, const TensorView& out,
                                   const TensorView& descriptor_storage, int64_t total_tokens,
                                   int64_t num_qk_heads, int64_t num_v_heads,
                                   int64_t prepare_descriptors, cudaStream_t stream) {
  if (prepare_descriptors == 0) {
    return;
  }
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  CheckCuda(cudaStreamIsCapturing(stream, &capture_status), "cudaStreamIsCapturing");
  TVM_FFI_ICHECK(capture_status == cudaStreamCaptureStatusNone)
      << "final-state descriptors must be prepared outside CUDA graph capture";
  const std::array<CUtensorMap, kFinalDescriptorCount> maps = {
      EncodeFinalMap(q, "q", total_tokens, num_qk_heads),
      EncodeFinalMap(k, "k", total_tokens, num_qk_heads),
      EncodeFinalMap(v, "v", total_tokens, num_v_heads),
      EncodeFinalMap(g, "g", total_tokens, num_v_heads),
      EncodeFinalMap(out, "final output scratch", total_tokens, num_v_heads),
  };
  FinalTensorMapWords words{};
  std::memcpy(words.words, maps.data(), sizeof(maps));
  PublishFinalTensorMaps<<<1, kFinalDescriptorBytes / sizeof(uint64_t), 0, stream>>>(
      reinterpret_cast<uint64_t*>(descriptor_storage.data_ptr()), words);
  CheckCuda(cudaGetLastError(), "PublishFinalTensorMaps launch");
}

template <typename Kernel>
inline void ConfigureDynamicSmem(Kernel kernel, int32_t bytes, int32_t device_id,
                                 const char* name) {
  CheckDynamicSmemCapacity(device_id, bytes);
  CheckCuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes), name);
}

inline int32_t ResidentCtas(int32_t device_id) {
  int32_t resident_ctas = 0;
  CheckCuda(cudaDeviceGetAttribute(&resident_ctas, cudaDevAttrMultiProcessorCount, device_id),
            "cudaDeviceGetAttribute(multiProcessorCount)");
  return resident_ctas;
}

void RunTrainingForward(
    TensorView q, TensorView k, TensorView v, TensorView g, TensorView beta, TensorView A_log,
    TensorView dt_bias, TensorView initial_state, TensorView cu_seqlens,
    TensorView checkpoint_cu_starts, TensorView base_work_items, TensorView work_items,
    TensorView boundaries, TensorView counters, TensorView out, TensorView final_state,
    TensorView state_checkpoints, TensorView beta_active, TensorView final_output_scratch,
    TensorView final_descriptor_storage, TensorView final_tensormap_workspace, TensorView dummy_f32,
    TensorView dummy_i32, int64_t boundary_count, int64_t total_work_items, int64_t total_tokens,
    int64_t num_sequences, int64_t num_qk_heads, int64_t num_v_heads, int64_t total_chunks,
    int64_t beta_active_stride, int64_t uniform_work_items, int64_t final_grid_ctas,
    int64_t prepare_final_descriptors, double scale, double lower_bound, int64_t cuda_stream) {
  TVM_FFI_ICHECK(q.device().device_type == kDLCUDA) << "q must be a CUDA tensor";
  const int32_t device_id = q.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckFlashKDATarget(device_id);
  TVM_FFI_ICHECK(total_tokens > 0 && num_sequences > 0 && num_qk_heads > 0 &&
                 num_v_heads >= num_qk_heads && num_v_heads % num_qk_heads == 0);
  TVM_FFI_ICHECK(total_chunks > 0 && total_work_items > 0 && final_grid_ctas > 0);
  TVM_FFI_ICHECK(boundary_count >= 0 && beta_active_stride >= num_v_heads);
  TVM_FFI_ICHECK(uniform_work_items == 0 || uniform_work_items == 1);
  TVM_FFI_ICHECK(prepare_final_descriptors == 0 || prepare_final_descriptors == 1);

  for (const auto& named : std::initializer_list<std::pair<TensorView*, const char*>>{
           {&q, "q"},
           {&k, "k"},
           {&v, "v"},
           {&g, "g"},
           {&beta, "beta"},
           {&out, "out"},
           {&state_checkpoints, "state_checkpoints"},
           {&beta_active, "beta_active"},
           {&final_output_scratch, "final_output_scratch"}}) {
    CheckTensor(*named.first, named.second, device_id, dl_bfloat16);
  }
  for (const auto& named :
       std::initializer_list<std::pair<TensorView*, const char*>>{{&A_log, "A_log"},
                                                                  {&dt_bias, "dt_bias"},
                                                                  {&initial_state, "initial_state"},
                                                                  {&final_state, "final_state"},
                                                                  {&dummy_f32, "dummy_f32"}}) {
    CheckTensor(*named.first, named.second, device_id, dl_float32);
  }
  CheckTensor(cu_seqlens, "cu_seqlens", device_id, dl_int64);
  CheckTensor(checkpoint_cu_starts, "checkpoint_cu_starts", device_id, dl_int64);
  CheckTensor(base_work_items, "base_work_items", device_id, dl_int32);
  CheckTensor(work_items, "work_items", device_id, dl_int32);
  CheckTensor(boundaries, "boundaries", device_id, dl_int32);
  CheckTensor(counters, "counters", device_id, dl_uint32);
  CheckTensor(final_descriptor_storage, "final_descriptor_storage", device_id, dl_uint8);
  CheckTensor(final_tensormap_workspace, "final_tensormap_workspace", device_id, dl_uint8);
  CheckTensor(dummy_i32, "dummy_i32", device_id, dl_int32);

  CheckElements(q, "q", total_tokens * num_qk_heads * kHeadDim);
  CheckElements(k, "k", q.numel());
  for (const auto& named : std::initializer_list<std::pair<TensorView*, const char*>>{
           {&v, "v"}, {&g, "g"}, {&out, "out"}, {&final_output_scratch, "final_output_scratch"}}) {
    CheckElements(*named.first, named.second, total_tokens * num_v_heads * kHeadDim);
  }
  CheckElements(beta, "beta", total_tokens * num_v_heads);
  CheckElements(A_log, "A_log", num_v_heads);
  CheckElements(dt_bias, "dt_bias", num_v_heads * kHeadDim);
  CheckElements(initial_state, "initial_state", num_sequences * num_v_heads * kHeadDim * kHeadDim);
  CheckElements(final_state, "final_state", initial_state.numel());
  CheckElements(cu_seqlens, "cu_seqlens", num_sequences + 1);
  CheckElements(checkpoint_cu_starts, "checkpoint_cu_starts", num_sequences + 1);
  CheckElements(base_work_items, "base_work_items", total_work_items * 8);
  CheckElements(work_items, "work_items", total_work_items * 8);
  TVM_FFI_ICHECK(boundaries.numel() >= std::max<int64_t>(2, boundary_count * 2));
  CheckElements(counters, "counters", num_v_heads + 2);
  CheckElements(state_checkpoints, "state_checkpoints",
                total_chunks * num_v_heads * kHeadDim * kHeadDim);
  CheckElements(beta_active, "beta_active", total_tokens * beta_active_stride);
  TVM_FFI_ICHECK(final_descriptor_storage.numel() >= kFinalDescriptorBytes);
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(final_descriptor_storage.data_ptr()) % 64 == 0);
  TVM_FFI_ICHECK(final_tensormap_workspace.numel() >= final_grid_ctas * kFinalWorkspaceBytesPerCta);

  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  CheckCuda(
      cudaMemcpyAsync(work_items.data_ptr(), base_work_items.data_ptr(),
                      total_work_items * 8 * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream),
      "copy training work items");
  CheckCuda(cudaMemsetAsync(counters.data_ptr(), 0, counters.numel() * sizeof(uint32_t), stream),
            "reset training counters");
  if (boundary_count != 0) {
    ConfigureDynamicSmem(kernel_flashkda_refine_forgetting_horizons, kRefineSmemBytes, device_id,
                         "cudaFuncSetAttribute(forgetting-horizon refinement)");
    kernel_flashkda_refine_forgetting_horizons<<<dim3(boundary_count, 1, 1), 128, kRefineSmemBytes,
                                                 stream>>>(
        reinterpret_cast<__nv_bfloat16*>(g.data_ptr()), reinterpret_cast<float*>(A_log.data_ptr()),
        reinterpret_cast<float*>(dt_bias.data_ptr()), reinterpret_cast<int*>(work_items.data_ptr()),
        reinterpret_cast<int*>(boundaries.data_ptr()),
        reinterpret_cast<unsigned int*>(counters.data_ptr()), static_cast<int>(num_v_heads),
        static_cast<float>(lower_bound), kDefaultLog2Threshold);
    CheckCuda(cudaGetLastError(), "forgetting-horizon refinement launch");
  }

  const CUtensorMap q_map = EncodeTokenMap(q, "q", total_tokens, num_qk_heads);
  const CUtensorMap k_map = EncodeTokenMap(k, "k", total_tokens, num_qk_heads);
  const CUtensorMap v_map = EncodeTokenMap(v, "v", total_tokens, num_v_heads);
  const CUtensorMap g_map = EncodeTokenMap(g, "g", total_tokens, num_v_heads);
  const CUtensorMap out_map = EncodeTokenMap(out, "out", total_tokens, num_v_heads);
  const CUtensorMap checkpoint_map =
      EncodeCheckpointMap(state_checkpoints, total_chunks, num_v_heads);
  const dim3 grid(std::min<int64_t>(total_work_items, ResidentCtas(device_id)), 1, 1);
  ConfigureDynamicSmem(kernel_flashkda_forward_checkpoint_c16, kForwardSmemBytes, device_id,
                       "cudaFuncSetAttribute(training forward)");
  kernel_flashkda_forward_checkpoint_c16<<<grid, 512, kForwardSmemBytes, stream>>>(
      reinterpret_cast<unsigned int*>(counters.data_ptr()), q_map, k_map, v_map, g_map,
      reinterpret_cast<__nv_bfloat16*>(g.data_ptr()), out_map, checkpoint_map,
      reinterpret_cast<__nv_bfloat16*>(beta.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(beta_active.data_ptr()),
      reinterpret_cast<float*>(A_log.data_ptr()), reinterpret_cast<float*>(dt_bias.data_ptr()),
      reinterpret_cast<long long*>(cu_seqlens.data_ptr()),
      reinterpret_cast<long long*>(checkpoint_cu_starts.data_ptr()),
      reinterpret_cast<int*>(work_items.data_ptr()),
      reinterpret_cast<float*>(initial_state.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(beta_active.data_ptr()), static_cast<int>(total_work_items),
      static_cast<int>(uniform_work_items), static_cast<int>(num_qk_heads),
      static_cast<int>(num_v_heads), static_cast<int>(beta_active_stride), kChunk,
      static_cast<float>(scale), static_cast<float>(lower_bound));
  CheckCuda(cudaGetLastError(), "training forward launch");

  PrepareFinalTensorMaps(q, k, v, g, final_output_scratch, final_descriptor_storage, total_tokens,
                         num_qk_heads, num_v_heads, prepare_final_descriptors, stream);
  ConfigureDynamicSmem(kernel_flashkda_blackwell_prefill_fp32_state_initial, kFinalSmemBytes,
                       device_id, "cudaFuncSetAttribute(training final state)");
  auto* final_maps =
      reinterpret_cast<const FlashKDATrainingTensorMap*>(final_descriptor_storage.data_ptr());
  kernel_flashkda_blackwell_prefill_fp32_state_initial<<<dim3(final_grid_ctas, 1, 1), 384,
                                                         kFinalSmemBytes, stream>>>(
      final_maps + 0, final_maps + 1, final_maps + 2, final_maps + 3, final_maps + 4,
      reinterpret_cast<__nv_bfloat16*>(beta.data_ptr()), reinterpret_cast<float*>(A_log.data_ptr()),
      reinterpret_cast<float*>(dt_bias.data_ptr()),
      reinterpret_cast<long long*>(cu_seqlens.data_ptr()),
      reinterpret_cast<float*>(initial_state.data_ptr()),
      reinterpret_cast<float*>(final_state.data_ptr()),
      reinterpret_cast<float*>(dummy_f32.data_ptr()), reinterpret_cast<int*>(dummy_i32.data_ptr()),
      reinterpret_cast<uint8_t*>(final_tensormap_workspace.data_ptr()), 0,
      static_cast<float>(scale), static_cast<int>(num_sequences), static_cast<int>(num_qk_heads),
      static_cast<int>(num_v_heads), static_cast<int>(num_sequences * num_v_heads),
      static_cast<float>(lower_bound));
  CheckCuda(cudaGetLastError(), "training final-state launch");
}

void RunTrainingBackward(TensorView q, TensorView k, TensorView v, TensorView g, TensorView A_log,
                         TensorView dt_bias, TensorView do_tensor, TensorView dfinal_state,
                         TensorView cu_seqlens, TensorView checkpoint_cu_starts,
                         TensorView work_items, TensorView counters, TensorView state_checkpoints,
                         TensorView beta_active, TensorView dlog_decay, TensorView dlog_boundary,
                         TensorView dbeta_active, TensorView gate_part_a, TensorView gate_part_dt,
                         TensorView dummy_u32, TensorView dummy_f32, TensorView dq_value_heads,
                         TensorView dk_value_heads, TensorView dv, TensorView dg, TensorView dbeta,
                         TensorView dA_log, TensorView ddt_bias, TensorView dinitial_state,
                         TensorView dq, TensorView dk, int64_t total_work_items,
                         int64_t total_tokens, int64_t num_sequences, int64_t num_qk_heads,
                         int64_t num_v_heads, int64_t total_chunks, int64_t beta_active_stride,
                         int64_t uniform_work_items, int64_t grouped, double scale,
                         double lower_bound, int64_t cuda_stream) {
  TVM_FFI_ICHECK(q.device().device_type == kDLCUDA) << "q must be a CUDA tensor";
  const int32_t device_id = q.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckFlashKDATarget(device_id);
  TVM_FFI_ICHECK(grouped == 0 || grouped == 1);
  TVM_FFI_ICHECK(grouped == int64_t(num_qk_heads != num_v_heads));
  TVM_FFI_ICHECK(total_work_items > 0 && total_tokens > 0 && num_sequences > 0 &&
                 num_qk_heads > 0 && num_v_heads >= num_qk_heads &&
                 num_v_heads % num_qk_heads == 0 && total_chunks > 0);

  for (const auto& named : std::initializer_list<std::pair<TensorView*, const char*>>{
           {&q, "q"},
           {&k, "k"},
           {&v, "v"},
           {&g, "g"},
           {&do_tensor, "do"},
           {&state_checkpoints, "state_checkpoints"},
           {&beta_active, "beta_active"},
           {&dq_value_heads, "dq_value_heads"},
           {&dk_value_heads, "dk_value_heads"},
           {&dv, "dv"},
           {&dg, "dg"},
           {&dbeta, "dbeta"},
           {&dq, "dq"},
           {&dk, "dk"}}) {
    CheckTensor(*named.first, named.second, device_id, dl_bfloat16);
  }
  for (const auto& named : std::initializer_list<std::pair<TensorView*, const char*>>{
           {&A_log, "A_log"},
           {&dt_bias, "dt_bias"},
           {&dfinal_state, "dfinal_state"},
           {&dlog_decay, "dlog_decay"},
           {&dlog_boundary, "dlog_boundary"},
           {&dbeta_active, "dbeta_active"},
           {&gate_part_a, "gate_part_a"},
           {&gate_part_dt, "gate_part_dt"},
           {&dummy_f32, "dummy_f32"},
           {&dA_log, "dA_log"},
           {&ddt_bias, "ddt_bias"},
           {&dinitial_state, "dinitial_state"}}) {
    CheckTensor(*named.first, named.second, device_id, dl_float32);
  }
  CheckTensor(cu_seqlens, "cu_seqlens", device_id, dl_int64);
  CheckTensor(checkpoint_cu_starts, "checkpoint_cu_starts", device_id, dl_int64);
  CheckTensor(work_items, "work_items", device_id, dl_int32);
  CheckTensor(counters, "counters", device_id, dl_uint32);
  CheckTensor(dummy_u32, "dummy_u32", device_id, dl_uint32);

  CheckElements(q, "q", total_tokens * num_qk_heads * kHeadDim);
  CheckElements(k, "k", q.numel());
  for (const auto& named : std::initializer_list<std::pair<TensorView*, const char*>>{
           {&v, "v"},
           {&g, "g"},
           {&do_tensor, "do"},
           {&dq_value_heads, "dq_value_heads"},
           {&dk_value_heads, "dk_value_heads"},
           {&dv, "dv"},
           {&dg, "dg"}}) {
    CheckElements(*named.first, named.second, total_tokens * num_v_heads * kHeadDim);
  }
  CheckElements(dbeta, "dbeta", total_tokens * num_v_heads);
  CheckElements(dq, "dq", q.numel());
  CheckElements(dk, "dk", k.numel());
  CheckElements(dfinal_state, "dfinal_state", num_sequences * num_v_heads * kHeadDim * kHeadDim);
  CheckElements(state_checkpoints, "state_checkpoints",
                total_chunks * num_v_heads * kHeadDim * kHeadDim);
  CheckElements(beta_active, "beta_active", total_tokens * beta_active_stride);
  CheckElements(dlog_decay, "dlog_decay", total_tokens * num_v_heads * kHeadDim);
  CheckElements(dlog_boundary, "dlog_boundary", total_chunks * num_v_heads * kHeadDim);
  CheckElements(dbeta_active, "dbeta_active", total_tokens * num_v_heads);
  CheckElements(gate_part_a, "gate_part_a", 128 * num_v_heads * kHeadDim);
  CheckElements(gate_part_dt, "gate_part_dt", gate_part_a.numel());
  CheckElements(dA_log, "dA_log", num_v_heads);
  CheckElements(ddt_bias, "ddt_bias", num_v_heads * kHeadDim);
  CheckElements(dinitial_state, "dinitial_state", dfinal_state.numel());
  CheckElements(counters, "counters", num_v_heads + 2);

  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  CheckCuda(cudaMemsetAsync(static_cast<unsigned int*>(counters.data_ptr()) + 1, 0,
                            (num_v_heads + 1) * sizeof(uint32_t), stream),
            "reset backward counters");
  CUtensorMap q_map = EncodeTokenMap(q, "q", total_tokens, num_qk_heads);
  CUtensorMap k_map = EncodeTokenMap(k, "k", total_tokens, num_qk_heads);
  CUtensorMap g_map = EncodeTokenMap(g, "g", total_tokens, num_v_heads);
  CUtensorMap do_map = EncodeTokenMap(do_tensor, "do", total_tokens, num_v_heads);
  CUtensorMap v_map = EncodeTokenMap(v, "v", total_tokens, num_v_heads);
  CUtensorMap state_map = EncodeCheckpointMap(state_checkpoints, total_chunks, num_v_heads);
  CUtensorMap dv_map = EncodeTokenMap(dv, "dv", total_tokens, num_v_heads);
  CUtensorMap beta_map = EncodeBetaMap(beta_active, total_tokens, beta_active_stride);
  const dim3 grid(std::min<int64_t>(total_work_items, ResidentCtas(device_id)), 1, 1);
  ConfigureDynamicSmem(kernel_flashkda_backward_persistent_c16, kBackwardSmemBytes, device_id,
                       "cudaFuncSetAttribute(training backward)");
  auto* dynamic_counter = reinterpret_cast<unsigned int*>(counters.data_ptr()) + 1;
  auto* dfinal_state_ptr = reinterpret_cast<float*>(dfinal_state.data_ptr());
  auto* dq_value_heads_ptr = reinterpret_cast<__nv_bfloat16*>(dq_value_heads.data_ptr());
  auto* dk_value_heads_ptr = reinterpret_cast<__nv_bfloat16*>(dk_value_heads.data_ptr());
  auto* dlog_decay_ptr = reinterpret_cast<float*>(dlog_decay.data_ptr());
  auto* dlog_boundary_ptr = reinterpret_cast<float*>(dlog_boundary.data_ptr());
  auto* dinitial_state_ptr = reinterpret_cast<float*>(dinitial_state.data_ptr());
  auto* A_log_ptr = reinterpret_cast<float*>(A_log.data_ptr());
  auto* dt_bias_ptr = reinterpret_cast<float*>(dt_bias.data_ptr());
  auto* dbeta_active_ptr = reinterpret_cast<float*>(dbeta_active.data_ptr());
  auto* cu_seqlens_ptr = reinterpret_cast<long long*>(cu_seqlens.data_ptr());
  auto* checkpoint_cu_starts_ptr = reinterpret_cast<long long*>(checkpoint_cu_starts.data_ptr());
  auto* work_items_ptr = reinterpret_cast<int*>(work_items.data_ptr());
  auto* visits_ptr = reinterpret_cast<unsigned int*>(dummy_u32.data_ptr());
  auto* diagnostic = reinterpret_cast<float*>(dummy_f32.data_ptr());
  int total_work_items_arg = static_cast<int>(total_work_items);
  int uniform_work_items_arg = static_cast<int>(uniform_work_items);
  int total_chunks_arg = static_cast<int>(total_chunks);
  int num_qk_heads_arg = static_cast<int>(num_qk_heads);
  int num_v_heads_arg = static_cast<int>(num_v_heads);
  int enable_kk = 1;
  int enable_tinv = 1;
  float scale_arg = static_cast<float>(scale);
  float lower_bound_arg = static_cast<float>(lower_bound);
  void* kernel_args[] = {
      &dynamic_counter,
      &q_map,
      &k_map,
      &g_map,
      &do_map,
      &v_map,
      &state_map,
      &dfinal_state_ptr,
      &dv_map,
      &dq_value_heads_ptr,
      &dk_value_heads_ptr,
      &dlog_decay_ptr,
      &dlog_boundary_ptr,
      &dinitial_state_ptr,
      &A_log_ptr,
      &dt_bias_ptr,
      &beta_map,
      &dbeta_active_ptr,
      &cu_seqlens_ptr,
      &checkpoint_cu_starts_ptr,
      &work_items_ptr,
      &visits_ptr,
      &diagnostic,
      &diagnostic,
      &diagnostic,
      &diagnostic,
      &diagnostic,
      &diagnostic,
      &diagnostic,
      &diagnostic,
      &diagnostic,
      &diagnostic,
      &diagnostic,
      &diagnostic,
      &diagnostic,
      &diagnostic,
      &total_work_items_arg,
      &uniform_work_items_arg,
      &total_chunks_arg,
      &num_qk_heads_arg,
      &num_v_heads_arg,
      &enable_kk,
      &enable_tinv,
      &scale_arg,
      &lower_bound_arg,
  };
  CheckCuda(cudaLaunchKernel(reinterpret_cast<const void*>(kernel_flashkda_backward_persistent_c16),
                             grid, dim3(512, 1, 1), kernel_args, kBackwardSmemBytes, stream),
            "training backward launch");

  ConfigureDynamicSmem(kernel_flashkda_backward_param_reduce_c16_partial, kReduceSmemBytes,
                       device_id, "cudaFuncSetAttribute(training parameter reduction)");
  kernel_flashkda_backward_param_reduce_c16_partial<<<dim3(128, num_v_heads, 1), 128,
                                                      kReduceSmemBytes, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(g.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(beta_active.data_ptr()),
      reinterpret_cast<float*>(A_log.data_ptr()), reinterpret_cast<float*>(dt_bias.data_ptr()),
      reinterpret_cast<float*>(dlog_decay.data_ptr()),
      reinterpret_cast<float*>(dlog_boundary.data_ptr()),
      reinterpret_cast<float*>(dbeta_active.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(dg.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(dbeta.data_ptr()),
      reinterpret_cast<float*>(gate_part_a.data_ptr()),
      reinterpret_cast<float*>(gate_part_dt.data_ptr()),
      reinterpret_cast<unsigned int*>(counters.data_ptr()) + 2,
      reinterpret_cast<float*>(dA_log.data_ptr()), reinterpret_cast<float*>(ddt_bias.data_ptr()),
      static_cast<int>(total_tokens), static_cast<int>(num_v_heads),
      static_cast<int>(beta_active_stride), static_cast<int>((total_tokens + 127) / 128),
      static_cast<float>(lower_bound));
  CheckCuda(cudaGetLastError(), "training parameter-reduction launch");

  if (grouped != 0) {
    kernel_flashkda_grouped_qk_reduce<<<dim3((total_tokens + 15) / 16, num_qk_heads, 1), 128, 0,
                                        stream>>>(
        reinterpret_cast<__nv_bfloat16*>(dq_value_heads.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dk_value_heads.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dq.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dk.data_ptr()), static_cast<int>(total_tokens),
        static_cast<int>(num_qk_heads), static_cast<int>(num_v_heads));
    CheckCuda(cudaGetLastError(), "grouped Q/K gradient reduction launch");
  }
}

}  // namespace flash_kda_training_paired
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_training_forward,
                              flashinfer::flash_kda_training_paired::RunTrainingForward);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_training_backward,
                              flashinfer::flash_kda_training_paired::RunTrainingBackward);
