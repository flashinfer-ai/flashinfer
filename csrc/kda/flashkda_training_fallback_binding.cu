/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include "flashkda_binding_common.cuh"

struct alignas(128) FlashKDATrainingFallbackTensorMap {
  uint64_t opaque[16];
};

extern "C" {

__global__ void kernel_flashkda_backward_preprocess(__nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
                                                    __nv_bfloat16*, float*, float*, float*, float*,
                                                    float*, float*, int, int, float);
__global__ void kernel_flashkda_backward_checkpoint(float*, float*, float*, __nv_bfloat16*, float*,
                                                    long long*, float*, int, int);
__global__ void kernel_flashkda_backward_reverse_rows(float*, float*, float*, float*,
                                                      __nv_bfloat16*, __nv_bfloat16*, float*,
                                                      float*, long long*, float*, float*, float*,
                                                      float*, float*, __nv_bfloat16*, float*, int,
                                                      int, float);
__global__ void kernel_flashkda_backward_finalize_tokens(__nv_bfloat16*, __nv_bfloat16*,
                                                         __nv_bfloat16*, float*, float*, float*,
                                                         float*, float*, float*, float*, float*,
                                                         float*, __nv_bfloat16*, __nv_bfloat16*,
                                                         __nv_bfloat16*, __nv_bfloat16*, int,
                                                         float);
__global__ void kernel_flashkda_backward_gate_reduce_split(float*, __nv_bfloat16*, float*, float*,
                                                           float*, int, int, int);

__global__ void kernel_flashkda_grouped_qk_expand(__nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
                                                  __nv_bfloat16*, int, int, int);
__global__ void kernel_flashkda_grouped_qk_reduce(__nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
                                                  __nv_bfloat16*, int, int, int);

__global__ void kernel_flashkda_bf16_fused_m128(
    __nv_bfloat16*, const FlashKDATrainingFallbackTensorMap*, __nv_bfloat16*,
    const FlashKDATrainingFallbackTensorMap*, __nv_bfloat16*,
    const FlashKDATrainingFallbackTensorMap*, __nv_bfloat16*,
    const FlashKDATrainingFallbackTensorMap*, __nv_bfloat16*,
    const FlashKDATrainingFallbackTensorMap*, float*, float*, long long*, int*, __nv_bfloat16*,
    __nv_bfloat16*, const FlashKDATrainingFallbackTensorMap*, __nv_bfloat16*, int, int, int, float,
    float, unsigned long long, unsigned long long, unsigned long long, long long, long long, int,
    int, long long*, __nv_bfloat16*, unsigned int*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
    __nv_bfloat16*, float*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, float*, __nv_bfloat16*,
    float*, float*, unsigned int*, int, int, const FlashKDATrainingFallbackTensorMap*);
__global__ void kernel_flashkda_bf16_fused_m128_unsplit(
    __nv_bfloat16*, const FlashKDATrainingFallbackTensorMap*, __nv_bfloat16*,
    const FlashKDATrainingFallbackTensorMap*, __nv_bfloat16*,
    const FlashKDATrainingFallbackTensorMap*, __nv_bfloat16*,
    const FlashKDATrainingFallbackTensorMap*, __nv_bfloat16*,
    const FlashKDATrainingFallbackTensorMap*, float*, float*, long long*, int*, __nv_bfloat16*,
    __nv_bfloat16*, const FlashKDATrainingFallbackTensorMap*, __nv_bfloat16*, int, int, int, float,
    float, unsigned long long, unsigned long long, unsigned long long, long long, long long, int,
    int, long long*, __nv_bfloat16*, unsigned int*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
    __nv_bfloat16*, float*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, float*, __nv_bfloat16*,
    float*, float*, unsigned int*, int, int, const FlashKDATrainingFallbackTensorMap*);
__global__ void kernel_flashkda_backward_state_checkpoint_fallback_c32(
    long long*, long long*, unsigned int*, float*, __nv_bfloat16*, float*, __nv_bfloat16*,
    __nv_bfloat16*, int, int, int, float, int);
__global__ void kernel_flashkda_backward_boundary_c32_tcgen_m64(
    __nv_bfloat16*, float*, float*, long long*, long long*, int*, __nv_bfloat16*, __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, float*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, float*,
    unsigned int*, int, int, float);
__global__ void kernel_flashkda_backward_boundary_c32_tcgen(
    __nv_bfloat16*, float*, float*, long long*, long long*, int*, __nv_bfloat16*, __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, float*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
    unsigned int*, float*, int, int, float);
__global__ void kernel_flashkda_backward_local_c32_tcgen(
    __nv_bfloat16*, float*, long long*, int*, int*, int*, __nv_bfloat16*, unsigned int*, float*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, float*, __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, unsigned int*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, float*, float*, __nv_bfloat16*, int, int,
    float);
__global__ void kernel_flashkda_backward_map_finalize_c32(
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, float*, __nv_bfloat16*, float*, float*, float*,
    long long*, int*, int*, int*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, float*, float*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, float*, float*, int, int, float,
    float);

__global__ void kernel_flashkda_blackwell_prefill_fp32_state_initial(
    const FlashKDATrainingFallbackTensorMap*, const FlashKDATrainingFallbackTensorMap*,
    const FlashKDATrainingFallbackTensorMap*, const FlashKDATrainingFallbackTensorMap*,
    const FlashKDATrainingFallbackTensorMap*, __nv_bfloat16*, float*, float*, long long*, float*,
    float*, float*, int*, uint8_t*, int, float, int, int, int, int, float);

}  // extern "C"

namespace flashinfer {
namespace flash_kda_training_fallback {

using flash_kda::CheckCuda;
using flash_kda::CheckCudaTensor;
using flash_kda::CheckDtype;
using flash_kda::CheckDynamicSmemCapacity;
using flash_kda::CheckFlashKDATarget;
using flash_kda::EncodeTmaPointers;
using flash_kda::PackBetaForTmaIfNeeded;
using flash_kda::TmaPointers;

constexpr int64_t kHeadDim = 128;
constexpr int32_t kFinalSmemBytes = 226048;
constexpr int64_t kFinalDescriptorCount = 5;
constexpr int64_t kFinalDescriptorBytes = kFinalDescriptorCount * sizeof(CUtensorMap);
constexpr int64_t kFinalWorkspaceBytesPerCta = 10 * 128;
constexpr int64_t kLowGateTokensPerSplit = 128;
constexpr int32_t kTapeThreads = 1024;
constexpr int32_t kTapeSmemBytes = 227968;
constexpr int32_t kFallbackThreads = 256;
constexpr int32_t kFallbackSmemBytes = 128;
constexpr int32_t kBoundaryM64Threads = 288;
constexpr int32_t kBoundaryM64SmemBytes = 38016;
constexpr int32_t kBoundaryM128Threads = 512;
constexpr int32_t kBoundaryM128SmemBytes = 74880;
constexpr int32_t kLocalThreads = 384;
constexpr int32_t kLocalSmemBytes = 155136;
constexpr int32_t kMapFinalizeThreads = 128;

inline void CheckTensor(const TensorView& tensor, const char* name, int32_t device_id,
                        DLDataType dtype) {
  CheckCudaTensor(tensor, name, device_id);
  CheckDtype(tensor, name, dtype);
}

inline void CheckElements(const TensorView& tensor, const char* name, int64_t elements) {
  TVM_FFI_ICHECK(tensor.numel() == elements)
      << name << " must contain " << elements << " elements, got " << tensor.numel();
}

inline uint32_t CheckedGridX(int64_t blocks, const char* name) {
  TVM_FFI_ICHECK(blocks > 0 && blocks <= std::numeric_limits<uint32_t>::max())
      << name << " grid.x is out of range: " << blocks;
  return static_cast<uint32_t>(blocks);
}

template <typename Kernel>
inline void ConfigureDynamicSmem(Kernel kernel, int32_t bytes, int32_t device_id,
                                 const char* name) {
  CheckDynamicSmemCapacity(device_id, bytes);
  CheckCuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes), name);
}

inline CUtensorMap EncodeFinalTokenMap(const TensorView& tensor, const char* name,
                                       int64_t total_tokens, int64_t num_heads) {
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
  const uint32_t index = threadIdx.x;
  if (index < kFinalDescriptorBytes / sizeof(uint64_t)) {
    destination[index] = source.words[index];
  }
}

struct CheckpointTensorMapWords {
  uint64_t words[sizeof(CUtensorMap) / sizeof(uint64_t)];
};

static __global__ void PublishCheckpointTensorMap(uint64_t* destination,
                                                  CheckpointTensorMapWords source) {
  if (threadIdx.x < sizeof(CUtensorMap) / sizeof(uint64_t)) {
    destination[threadIdx.x] = source.words[threadIdx.x];
  }
}

inline CUtensorMap EncodeCheckpointTensorMap(const TensorView& checkpoint, int64_t total_chunks,
                                             int64_t num_heads) {
  uint64_t global_dim[4] = {kHeadDim, kHeadDim, static_cast<uint64_t>(num_heads),
                            static_cast<uint64_t>(total_chunks)};
  uint64_t global_strides[3] = {
      kHeadDim * sizeof(__nv_bfloat16), kHeadDim * kHeadDim * sizeof(__nv_bfloat16),
      kHeadDim * kHeadDim * static_cast<uint64_t>(num_heads) * sizeof(__nv_bfloat16)};
  uint32_t box_dim[4] = {64, 128, 1, 1};
  uint32_t element_strides[4] = {1, 1, 1, 1};
  CUtensorMap map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, checkpoint.data_ptr(), global_dim, global_strides,
      box_dim, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for C32 checkpoint with CUresult=" << int(result);
  return map;
}

inline TmaPointers PrepareC32TensorMaps(const TensorView& q, const TensorView& k,
                                        const TensorView& v, const TensorView& g,
                                        const TensorView& beta_tma, const TensorView& out,
                                        const TensorView& checkpoint,
                                        const TensorView& descriptor_storage, int64_t total_chunks,
                                        int64_t num_heads, int64_t prepare_descriptors,
                                        cudaStream_t stream) {
  const TmaPointers pointers = EncodeTmaPointers<128, 32>(
      q, k, v, g, beta_tma, out, descriptor_storage, prepare_descriptors, stream);
  if (prepare_descriptors != 0) {
    const CUtensorMap map = EncodeCheckpointTensorMap(checkpoint, total_chunks, num_heads);
    CheckpointTensorMapWords words{};
    std::memcpy(words.words, &map, sizeof(map));
    auto* bytes = static_cast<unsigned char*>(descriptor_storage.data_ptr());
    PublishCheckpointTensorMap<<<1, 32, 0, stream>>>(
        reinterpret_cast<uint64_t*>(bytes + 6 * sizeof(CUtensorMap)), words);
    CheckCuda(cudaGetLastError(), "PublishCheckpointTensorMap launch");
  }
  return pointers;
}

inline void PrepareFinalTensorMaps(const TensorView& q, const TensorView& k, const TensorView& v,
                                   const TensorView& g, const TensorView& out,
                                   const TensorView& descriptor_storage, int64_t total_tokens,
                                   int64_t num_qk_heads, int64_t num_v_heads,
                                   int64_t prepare_descriptors, cudaStream_t stream) {
  if (prepare_descriptors == 0) {
    return;
  }
  std::array<CUtensorMap, kFinalDescriptorCount> host_maps = {
      EncodeFinalTokenMap(q, "q", total_tokens, num_qk_heads),
      EncodeFinalTokenMap(k, "k", total_tokens, num_qk_heads),
      EncodeFinalTokenMap(v, "v", total_tokens, num_v_heads),
      EncodeFinalTokenMap(g, "g", total_tokens, num_v_heads),
      EncodeFinalTokenMap(out, "out", total_tokens, num_v_heads),
  };
  FinalTensorMapWords words{};
  std::memcpy(words.words, host_maps.data(), sizeof(host_maps));
  PublishFinalTensorMaps<<<1, 128, 0, stream>>>(
      reinterpret_cast<uint64_t*>(descriptor_storage.data_ptr()), words);
  CheckCuda(cudaGetLastError(), "PublishFinalTensorMaps launch");
}

inline void LaunchAccurateForward(
    TensorView q, TensorView k, TensorView v, TensorView g, TensorView beta, TensorView A_log,
    TensorView dt_bias, TensorView initial_state, TensorView cu_seqlens, TensorView out,
    TensorView final_state, TensorView descriptor_storage, TensorView tensormap_workspace,
    TensorView dummy_f32, TensorView dummy_i32, int64_t total_tokens, int64_t num_sequences,
    int64_t num_qk_heads, int64_t num_v_heads, int64_t final_grid_ctas, int64_t prepare_descriptors,
    double scale, double lower_bound, int32_t device_id, cudaStream_t stream) {
  TVM_FFI_ICHECK(descriptor_storage.numel() >= kFinalDescriptorBytes);
  TVM_FFI_ICHECK(tensormap_workspace.numel() >= final_grid_ctas * kFinalWorkspaceBytesPerCta);
  PrepareFinalTensorMaps(q, k, v, g, out, descriptor_storage, total_tokens, num_qk_heads,
                         num_v_heads, prepare_descriptors, stream);
  ConfigureDynamicSmem(kernel_flashkda_blackwell_prefill_fp32_state_initial, kFinalSmemBytes,
                       device_id, "cudaFuncSetAttribute(training accurate forward)");
  auto* maps =
      reinterpret_cast<const FlashKDATrainingFallbackTensorMap*>(descriptor_storage.data_ptr());
  kernel_flashkda_blackwell_prefill_fp32_state_initial<<<dim3(final_grid_ctas, 1, 1), 384,
                                                         kFinalSmemBytes, stream>>>(
      maps + 0, maps + 1, maps + 2, maps + 3, maps + 4,
      reinterpret_cast<__nv_bfloat16*>(beta.data_ptr()), reinterpret_cast<float*>(A_log.data_ptr()),
      reinterpret_cast<float*>(dt_bias.data_ptr()),
      reinterpret_cast<long long*>(cu_seqlens.data_ptr()),
      reinterpret_cast<float*>(initial_state.data_ptr()),
      reinterpret_cast<float*>(final_state.data_ptr()),
      reinterpret_cast<float*>(dummy_f32.data_ptr()), reinterpret_cast<int*>(dummy_i32.data_ptr()),
      reinterpret_cast<uint8_t*>(tensormap_workspace.data_ptr()), 0, static_cast<float>(scale),
      static_cast<int>(num_sequences), static_cast<int>(num_qk_heads),
      static_cast<int>(num_v_heads), static_cast<int>(num_sequences * num_v_heads),
      static_cast<float>(lower_bound));
  CheckCuda(cudaGetLastError(), "training accurate forward launch");
}

void RunTrainingRowForward(
    TensorView q, TensorView k, TensorView v, TensorView g, TensorView beta, TensorView A_log,
    TensorView dt_bias, TensorView initial_state, TensorView cu_seqlens, TensorView out,
    TensorView final_state, TensorView q_norm, TensorView k_norm, TensorView decay,
    TensorView beta_active, TensorView checkpoint, TensorView final_descriptor_storage,
    TensorView final_tensormap_workspace, TensorView dummy_f32, TensorView dummy_i32,
    int64_t total_tokens, int64_t num_sequences, int64_t num_heads, int64_t final_grid_ctas,
    int64_t prepare_final_descriptors, double scale, double lower_bound, int64_t cuda_stream) {
  const int32_t device_id = q.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckFlashKDATarget(device_id);
  TVM_FFI_ICHECK(total_tokens > 0 && num_sequences > 0 && num_heads > 0);
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  const dim3 preprocess_grid(total_tokens, num_heads, 1);
  kernel_flashkda_backward_preprocess<<<preprocess_grid, 32, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(k.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(g.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(beta.data_ptr()), reinterpret_cast<float*>(A_log.data_ptr()),
      reinterpret_cast<float*>(dt_bias.data_ptr()), reinterpret_cast<float*>(q_norm.data_ptr()),
      reinterpret_cast<float*>(k_norm.data_ptr()), reinterpret_cast<float*>(decay.data_ptr()),
      reinterpret_cast<float*>(beta_active.data_ptr()), static_cast<int>(total_tokens),
      static_cast<int>(num_heads), static_cast<float>(lower_bound));
  CheckCuda(cudaGetLastError(), "row training preprocess launch");
  const int64_t row_grid = num_sequences * num_heads * kHeadDim;
  kernel_flashkda_backward_checkpoint<<<static_cast<uint32_t>(row_grid), 32, 0, stream>>>(
      reinterpret_cast<float*>(k_norm.data_ptr()), reinterpret_cast<float*>(decay.data_ptr()),
      reinterpret_cast<float*>(beta_active.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(v.data_ptr()),
      reinterpret_cast<float*>(initial_state.data_ptr()),
      reinterpret_cast<long long*>(cu_seqlens.data_ptr()),
      reinterpret_cast<float*>(checkpoint.data_ptr()), static_cast<int>(num_sequences),
      static_cast<int>(num_heads));
  CheckCuda(cudaGetLastError(), "row training checkpoint launch");
  LaunchAccurateForward(q, k, v, g, beta, A_log, dt_bias, initial_state, cu_seqlens, out,
                        final_state, final_descriptor_storage, final_tensormap_workspace, dummy_f32,
                        dummy_i32, total_tokens, num_sequences, num_heads, num_heads,
                        final_grid_ctas, prepare_final_descriptors, scale, lower_bound, device_id,
                        stream);
}

void RunTrainingRowBackward(TensorView q, TensorView k, TensorView v, TensorView g,
                            TensorView A_log, TensorView dt_bias, TensorView initial_state,
                            TensorView do_tensor, TensorView dfinal_state, TensorView cu_seqlens,
                            TensorView q_norm, TensorView k_norm, TensorView decay,
                            TensorView beta_active, TensorView checkpoint, TensorView dq_normalized,
                            TensorView dk_normalized, TensorView dlog_decay,
                            TensorView dbeta_active, TensorView dq, TensorView dk, TensorView dv,
                            TensorView dg, TensorView dbeta, TensorView dA_log, TensorView ddt_bias,
                            TensorView dinitial_state, int64_t total_tokens, int64_t num_sequences,
                            int64_t num_heads, double scale, double lower_bound,
                            int64_t cuda_stream) {
  const int32_t device_id = q.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckFlashKDATarget(device_id);
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  CheckCuda(cudaMemsetAsync(dA_log.data_ptr(), 0, dA_log.numel() * sizeof(float), stream),
            "clear row dA_log");
  CheckCuda(cudaMemsetAsync(ddt_bias.data_ptr(), 0, ddt_bias.numel() * sizeof(float), stream),
            "clear row ddt_bias");
  CheckCuda(
      cudaMemsetAsync(dq_normalized.data_ptr(), 0, dq_normalized.numel() * sizeof(float), stream),
      "clear row dq_normalized");
  CheckCuda(
      cudaMemsetAsync(dk_normalized.data_ptr(), 0, dk_normalized.numel() * sizeof(float), stream),
      "clear row dk_normalized");
  CheckCuda(cudaMemsetAsync(dlog_decay.data_ptr(), 0, dlog_decay.numel() * sizeof(float), stream),
            "clear row dlog_decay");
  CheckCuda(
      cudaMemsetAsync(dbeta_active.data_ptr(), 0, dbeta_active.numel() * sizeof(float), stream),
      "clear row dbeta_active");
  const int64_t row_grid = num_sequences * num_heads * kHeadDim;
  kernel_flashkda_backward_reverse_rows<<<static_cast<uint32_t>(row_grid), 32, 0, stream>>>(
      reinterpret_cast<float*>(q_norm.data_ptr()), reinterpret_cast<float*>(k_norm.data_ptr()),
      reinterpret_cast<float*>(decay.data_ptr()), reinterpret_cast<float*>(beta_active.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(v.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(do_tensor.data_ptr()),
      reinterpret_cast<float*>(initial_state.data_ptr()),
      reinterpret_cast<float*>(dfinal_state.data_ptr()),
      reinterpret_cast<long long*>(cu_seqlens.data_ptr()),
      reinterpret_cast<float*>(checkpoint.data_ptr()),
      reinterpret_cast<float*>(dq_normalized.data_ptr()),
      reinterpret_cast<float*>(dk_normalized.data_ptr()),
      reinterpret_cast<float*>(dlog_decay.data_ptr()),
      reinterpret_cast<float*>(dbeta_active.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(dv.data_ptr()),
      reinterpret_cast<float*>(dinitial_state.data_ptr()), static_cast<int>(num_sequences),
      static_cast<int>(num_heads), static_cast<float>(scale));
  CheckCuda(cudaGetLastError(), "row training reverse launch");
  const dim3 token_grid(total_tokens, num_heads, 1);
  kernel_flashkda_backward_finalize_tokens<<<token_grid, 32, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(k.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(g.data_ptr()),
      reinterpret_cast<float*>(beta_active.data_ptr()), reinterpret_cast<float*>(A_log.data_ptr()),
      reinterpret_cast<float*>(dt_bias.data_ptr()), reinterpret_cast<float*>(q_norm.data_ptr()),
      reinterpret_cast<float*>(k_norm.data_ptr()),
      reinterpret_cast<float*>(dq_normalized.data_ptr()),
      reinterpret_cast<float*>(dk_normalized.data_ptr()),
      reinterpret_cast<float*>(dlog_decay.data_ptr()),
      reinterpret_cast<float*>(dbeta_active.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(dq.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(dk.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(dg.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(dbeta.data_ptr()), static_cast<int>(num_heads),
      static_cast<float>(lower_bound));
  CheckCuda(cudaGetLastError(), "row training finalize launch");
  ConfigureDynamicSmem(kernel_flashkda_backward_gate_reduce_split, 128, device_id,
                       "cudaFuncSetAttribute(row gate reduce)");
  const int64_t splits = (total_tokens + kLowGateTokensPerSplit - 1) / kLowGateTokensPerSplit;
  kernel_flashkda_backward_gate_reduce_split<<<dim3(num_heads, splits, 1), 128, 128, stream>>>(
      reinterpret_cast<float*>(dlog_decay.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(g.data_ptr()), reinterpret_cast<float*>(dt_bias.data_ptr()),
      reinterpret_cast<float*>(dA_log.data_ptr()), reinterpret_cast<float*>(ddt_bias.data_ptr()),
      static_cast<int>(total_tokens), static_cast<int>(num_heads),
      static_cast<int>(kLowGateTokensPerSplit));
  CheckCuda(cudaGetLastError(), "row training gate reduce launch");
}

void RunTrainingC32Forward(
    TensorView q_native, TensorView k_native, TensorView q, TensorView k, TensorView v,
    TensorView g, TensorView beta, TensorView beta_tma, TensorView A_log, TensorView dt_bias,
    TensorView initial_state, TensorView cu_seqlens, TensorView work_items, TensorView seq_order,
    TensorView cu_chunk_offsets, TensorView descriptor_storage, TensorView out,
    TensorView final_state, TensorView chunk_state, TensorView state_checkpoint_needed,
    TensorView tape_qd, TensorView tape_kd, TensorView tape_kr, TensorView tape_j,
    TensorView tape_restore_factor, TensorView tape_e, TensorView tape_x, TensorView tape_r,
    TensorView norm_inv, TensorView decay, TensorView beta_active, TensorView zero_workspace,
    TensorView final_output_scratch, TensorView final_descriptor_storage,
    TensorView final_tensormap_workspace, TensorView dummy_f32, TensorView dummy_i32,
    int64_t total_tokens, int64_t num_sequences, int64_t num_qk_heads, int64_t num_heads,
    int64_t total_chunks, int64_t num_work_items, int64_t use_split_work_items, int64_t grouped,
    int64_t prepare_descriptors, int64_t prepare_final_descriptors, int64_t final_grid_ctas,
    double scale, double lower_bound, int64_t cuda_stream) {
  const int32_t device_id = q_native.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckFlashKDATarget(device_id);
  TVM_FFI_ICHECK(total_tokens > 0 && num_sequences > 0 && num_qk_heads > 0 && num_heads > 0);
  TVM_FFI_ICHECK(total_chunks > 0 && num_work_items > 0);
  TVM_FFI_ICHECK(use_split_work_items == 0 || use_split_work_items == 1);
  TVM_FFI_ICHECK(grouped == 0 || grouped == 1);
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));

  if (grouped != 0) {
    kernel_flashkda_grouped_qk_expand<<<
        dim3(CheckedGridX((total_tokens + 31) / 32, "grouped expand"), num_heads, 1), 128, 0,
        stream>>>(reinterpret_cast<__nv_bfloat16*>(q_native.data_ptr()),
                  reinterpret_cast<__nv_bfloat16*>(k_native.data_ptr()),
                  reinterpret_cast<__nv_bfloat16*>(q.data_ptr()),
                  reinterpret_cast<__nv_bfloat16*>(k.data_ptr()), static_cast<int>(total_tokens),
                  static_cast<int>(num_qk_heads), static_cast<int>(num_heads));
    CheckCuda(cudaGetLastError(), "grouped C32 Q/K expand launch");
  }

  PackBetaForTmaIfNeeded(beta, beta_tma, num_heads, num_heads, stream);
  const TmaPointers tma =
      PrepareC32TensorMaps(q, k, v, g, beta_tma, out, chunk_state, descriptor_storage, total_chunks,
                           num_heads, prepare_descriptors, stream);
  auto* descriptor_bytes = static_cast<unsigned char*>(descriptor_storage.data_ptr());
  auto* checkpoint_map = reinterpret_cast<const FlashKDATrainingFallbackTensorMap*>(
      descriptor_bytes + 6 * sizeof(CUtensorMap));

#define FLASHINFER_LAUNCH_C32_TAPE(KERNEL, GRID, SCHEDULE)                                       \
  KERNEL<<<CheckedGridX(GRID, "training C32 tape"), kTapeThreads, kTapeSmemBytes, stream>>>(     \
      reinterpret_cast<__nv_bfloat16*>(q.data_ptr()),                                            \
      reinterpret_cast<const FlashKDATrainingFallbackTensorMap*>(tma.q),                         \
      reinterpret_cast<__nv_bfloat16*>(k.data_ptr()),                                            \
      reinterpret_cast<const FlashKDATrainingFallbackTensorMap*>(tma.k),                         \
      reinterpret_cast<__nv_bfloat16*>(v.data_ptr()),                                            \
      reinterpret_cast<const FlashKDATrainingFallbackTensorMap*>(tma.v),                         \
      reinterpret_cast<__nv_bfloat16*>(g.data_ptr()),                                            \
      reinterpret_cast<const FlashKDATrainingFallbackTensorMap*>(tma.g),                         \
      reinterpret_cast<__nv_bfloat16*>(beta.data_ptr()),                                         \
      reinterpret_cast<const FlashKDATrainingFallbackTensorMap*>(tma.beta),                      \
      reinterpret_cast<float*>(A_log.data_ptr()), reinterpret_cast<float*>(dt_bias.data_ptr()),  \
      reinterpret_cast<long long*>(cu_seqlens.data_ptr()), reinterpret_cast<int*>(SCHEDULE),     \
      reinterpret_cast<__nv_bfloat16*>(q.data_ptr()),                                            \
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),                                          \
      reinterpret_cast<const FlashKDATrainingFallbackTensorMap*>(tma.out),                       \
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(num_heads), 1, 0,       \
      static_cast<float>(scale), static_cast<float>(lower_bound), 0ULL, 0ULL, 0ULL, 0LL, 0LL, 0, \
      0, reinterpret_cast<long long*>(cu_chunk_offsets.data_ptr()),                              \
      reinterpret_cast<__nv_bfloat16*>(chunk_state.data_ptr()),                                  \
      reinterpret_cast<unsigned int*>(state_checkpoint_needed.data_ptr()),                       \
      reinterpret_cast<__nv_bfloat16*>(tape_qd.data_ptr()),                                      \
      reinterpret_cast<__nv_bfloat16*>(tape_kd.data_ptr()),                                      \
      reinterpret_cast<__nv_bfloat16*>(tape_kr.data_ptr()),                                      \
      reinterpret_cast<__nv_bfloat16*>(tape_j.data_ptr()),                                       \
      reinterpret_cast<float*>(tape_restore_factor.data_ptr()),                                  \
      reinterpret_cast<__nv_bfloat16*>(tape_e.data_ptr()),                                       \
      reinterpret_cast<__nv_bfloat16*>(tape_x.data_ptr()),                                       \
      reinterpret_cast<__nv_bfloat16*>(tape_r.data_ptr()),                                       \
      reinterpret_cast<float*>(norm_inv.data_ptr()),                                             \
      reinterpret_cast<__nv_bfloat16*>(decay.data_ptr()),                                        \
      reinterpret_cast<float*>(beta_active.data_ptr()),                                          \
      reinterpret_cast<float*>(initial_state.data_ptr()),                                        \
      reinterpret_cast<unsigned int*>(zero_workspace.data_ptr()),                                \
      static_cast<int>(zero_workspace.numel()), static_cast<int>(num_sequences), checkpoint_map)

  if (use_split_work_items != 0) {
    ConfigureDynamicSmem(kernel_flashkda_bf16_fused_m128, kTapeSmemBytes, device_id,
                         "cudaFuncSetAttribute(training split C32 tape)");
    FLASHINFER_LAUNCH_C32_TAPE(kernel_flashkda_bf16_fused_m128, num_work_items,
                               work_items.data_ptr());
  } else {
    ConfigureDynamicSmem(kernel_flashkda_bf16_fused_m128_unsplit, kTapeSmemBytes, device_id,
                         "cudaFuncSetAttribute(training unsplit C32 tape)");
    FLASHINFER_LAUNCH_C32_TAPE(kernel_flashkda_bf16_fused_m128_unsplit, num_sequences * num_heads,
                               seq_order.data_ptr());
  }
#undef FLASHINFER_LAUNCH_C32_TAPE
  CheckCuda(cudaGetLastError(), "training C32 tape launch");

  ConfigureDynamicSmem(kernel_flashkda_backward_state_checkpoint_fallback_c32, kFallbackSmemBytes,
                       device_id, "cudaFuncSetAttribute(training C32 state fallback)");
  kernel_flashkda_backward_state_checkpoint_fallback_c32<<<1, kFallbackThreads, kFallbackSmemBytes,
                                                           stream>>>(
      reinterpret_cast<long long*>(cu_seqlens.data_ptr()),
      reinterpret_cast<long long*>(cu_chunk_offsets.data_ptr()),
      reinterpret_cast<unsigned int*>(state_checkpoint_needed.data_ptr()),
      reinterpret_cast<float*>(initial_state.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(tape_kr.data_ptr()),
      reinterpret_cast<float*>(tape_restore_factor.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(tape_r.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(chunk_state.data_ptr()), static_cast<int>(num_sequences),
      static_cast<int>(num_heads), static_cast<int>(use_split_work_items != 0 ? num_work_items : 0),
      static_cast<float>(lower_bound), static_cast<int>(use_split_work_items));
  CheckCuda(cudaGetLastError(), "training C32 state fallback launch");

  LaunchAccurateForward(q_native, k_native, v, g, beta, A_log, dt_bias, initial_state, cu_seqlens,
                        final_output_scratch, final_state, final_descriptor_storage,
                        final_tensormap_workspace, dummy_f32, dummy_i32, total_tokens,
                        num_sequences, num_qk_heads, num_heads, final_grid_ctas,
                        prepare_final_descriptors, scale, lower_bound, device_id, stream);
}

void RunTrainingC32Backward(
    TensorView q, TensorView k, TensorView v, TensorView g, TensorView A_log, TensorView dt_bias,
    TensorView initial_state, TensorView do_tensor, TensorView dfinal_state, TensorView cu_seqlens,
    TensorView cu_chunk_offsets, TensorView boundary_work_items, TensorView consumer_chunk_order,
    TensorView chunk_sequence, TensorView chunk_index, TensorView chunk_pair_start,
    TensorView chunk_state, TensorView state_checkpoint_needed, TensorView tape_qd,
    TensorView tape_kd, TensorView tape_kr, TensorView tape_j, TensorView tape_restore_factor,
    TensorView tape_e, TensorView tape_x, TensorView tape_r, TensorView norm_inv, TensorView decay,
    TensorView beta_active, TensorView chunk_dh, TensorView chunk_dr, TensorView chunk_dx,
    TensorView boundary_ready, TensorView grad_qd, TensorView grad_kd, TensorView grad_ki,
    TensorView dlog_decay, TensorView dbeta_active, TensorView dq_value, TensorView dk_value,
    TensorView dv, TensorView dg, TensorView dbeta, TensorView dA_log, TensorView ddt_bias,
    TensorView dinitial_state, TensorView dq_native, TensorView dk_native, int64_t total_tokens,
    int64_t num_sequences, int64_t num_qk_heads, int64_t num_heads, int64_t total_chunks,
    int64_t total_pairs, int64_t boundary_count, int64_t split_boundary, int64_t grouped,
    double scale, double lower_bound, int64_t cuda_stream) {
  const int32_t device_id = q.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckFlashKDATarget(device_id);
  TVM_FFI_ICHECK(total_tokens > 0 && num_sequences > 0 && num_heads > 0 && total_chunks > 0);
  TVM_FFI_ICHECK(total_pairs > 0 && boundary_count > 0);
  TVM_FFI_ICHECK(split_boundary == 0 || split_boundary == 1);
  TVM_FFI_ICHECK(grouped == 0 || grouped == 1);
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  CheckCuda(cudaMemsetAsync(dA_log.data_ptr(), 0, dA_log.numel() * sizeof(float), stream),
            "clear training C32 dA_log");
  CheckCuda(cudaMemsetAsync(ddt_bias.data_ptr(), 0, ddt_bias.numel() * sizeof(float), stream),
            "clear training C32 ddt_bias");
  CheckCuda(cudaMemsetAsync(boundary_ready.data_ptr(), 0,
                            boundary_ready.numel() * sizeof(unsigned int), stream),
            "clear training C32 boundary_ready");

  if (split_boundary != 0) {
    ConfigureDynamicSmem(kernel_flashkda_backward_boundary_c32_tcgen_m64, kBoundaryM64SmemBytes,
                         device_id, "cudaFuncSetAttribute(training C32 boundary M64)");
    kernel_flashkda_backward_boundary_c32_tcgen_m64<<<
        CheckedGridX(boundary_count * 2, "training C32 boundary M64"), kBoundaryM64Threads,
        kBoundaryM64SmemBytes, stream>>>(reinterpret_cast<__nv_bfloat16*>(do_tensor.data_ptr()),
                                         reinterpret_cast<float*>(dfinal_state.data_ptr()),
                                         reinterpret_cast<float*>(beta_active.data_ptr()),
                                         reinterpret_cast<long long*>(cu_seqlens.data_ptr()),
                                         reinterpret_cast<long long*>(cu_chunk_offsets.data_ptr()),
                                         reinterpret_cast<int*>(boundary_work_items.data_ptr()),
                                         reinterpret_cast<__nv_bfloat16*>(tape_qd.data_ptr()),
                                         reinterpret_cast<__nv_bfloat16*>(tape_kd.data_ptr()),
                                         reinterpret_cast<__nv_bfloat16*>(tape_kr.data_ptr()),
                                         reinterpret_cast<__nv_bfloat16*>(tape_j.data_ptr()),
                                         reinterpret_cast<float*>(tape_restore_factor.data_ptr()),
                                         reinterpret_cast<__nv_bfloat16*>(chunk_dh.data_ptr()),
                                         reinterpret_cast<__nv_bfloat16*>(chunk_dr.data_ptr()),
                                         reinterpret_cast<__nv_bfloat16*>(chunk_dx.data_ptr()),
                                         reinterpret_cast<float*>(dinitial_state.data_ptr()),
                                         reinterpret_cast<unsigned int*>(boundary_ready.data_ptr()),
                                         static_cast<int>(num_heads), 0,
                                         static_cast<float>(lower_bound));
    CheckCuda(cudaGetLastError(), "training C32 boundary M64 launch");
  } else {
    ConfigureDynamicSmem(kernel_flashkda_backward_boundary_c32_tcgen, kBoundaryM128SmemBytes,
                         device_id, "cudaFuncSetAttribute(training C32 boundary M128)");
    kernel_flashkda_backward_boundary_c32_tcgen<<<
        CheckedGridX(boundary_count, "training C32 boundary M128"), kBoundaryM128Threads,
        kBoundaryM128SmemBytes, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(do_tensor.data_ptr()),
        reinterpret_cast<float*>(dfinal_state.data_ptr()),
        reinterpret_cast<float*>(beta_active.data_ptr()),
        reinterpret_cast<long long*>(cu_seqlens.data_ptr()),
        reinterpret_cast<long long*>(cu_chunk_offsets.data_ptr()),
        reinterpret_cast<int*>(boundary_work_items.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(tape_qd.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(tape_kd.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(tape_kr.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(tape_j.data_ptr()),
        reinterpret_cast<float*>(tape_restore_factor.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(chunk_dh.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(chunk_dr.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(chunk_dx.data_ptr()),
        reinterpret_cast<unsigned int*>(boundary_ready.data_ptr()),
        reinterpret_cast<float*>(dinitial_state.data_ptr()), static_cast<int>(num_heads), 0,
        static_cast<float>(lower_bound));
    CheckCuda(cudaGetLastError(), "training C32 boundary M128 launch");
  }

  ConfigureDynamicSmem(kernel_flashkda_backward_local_c32_tcgen, kLocalSmemBytes, device_id,
                       "cudaFuncSetAttribute(training C32 local)");
  kernel_flashkda_backward_local_c32_tcgen<<<CheckedGridX(total_chunks * num_heads,
                                                          "training C32 local"),
                                             kLocalThreads, kLocalSmemBytes, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(do_tensor.data_ptr()),
      reinterpret_cast<float*>(beta_active.data_ptr()),
      reinterpret_cast<long long*>(cu_seqlens.data_ptr()),
      reinterpret_cast<int*>(consumer_chunk_order.data_ptr()),
      reinterpret_cast<int*>(chunk_sequence.data_ptr()),
      reinterpret_cast<int*>(chunk_index.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(chunk_state.data_ptr()),
      reinterpret_cast<unsigned int*>(state_checkpoint_needed.data_ptr()),
      reinterpret_cast<float*>(initial_state.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(tape_qd.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(tape_kd.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(tape_kr.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(tape_j.data_ptr()),
      reinterpret_cast<float*>(tape_restore_factor.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(tape_e.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(tape_x.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(tape_r.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(chunk_dh.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(chunk_dr.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(chunk_dx.data_ptr()),
      reinterpret_cast<unsigned int*>(boundary_ready.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(grad_qd.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(grad_kd.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(grad_ki.data_ptr()),
      reinterpret_cast<float*>(dlog_decay.data_ptr()),
      reinterpret_cast<float*>(dbeta_active.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(dv.data_ptr()), static_cast<int>(num_heads), 0,
      static_cast<float>(lower_bound));
  CheckCuda(cudaGetLastError(), "training C32 local launch");

  const int64_t num_pair_heads = total_pairs * num_heads;
  kernel_flashkda_backward_map_finalize_c32<<<CheckedGridX(((num_pair_heads + 3) / 4) * 4,
                                                           "training C32 map finalize"),
                                              kMapFinalizeThreads, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(decay.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(k.data_ptr()), reinterpret_cast<float*>(norm_inv.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(g.data_ptr()),
      reinterpret_cast<float*>(beta_active.data_ptr()), reinterpret_cast<float*>(A_log.data_ptr()),
      reinterpret_cast<float*>(dt_bias.data_ptr()),
      reinterpret_cast<long long*>(cu_seqlens.data_ptr()),
      reinterpret_cast<int*>(chunk_sequence.data_ptr()),
      reinterpret_cast<int*>(chunk_index.data_ptr()),
      reinterpret_cast<int*>(chunk_pair_start.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(grad_qd.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(grad_kd.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(grad_ki.data_ptr()),
      reinterpret_cast<float*>(dlog_decay.data_ptr()),
      reinterpret_cast<float*>(dbeta_active.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(dq_value.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(dk_value.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(dg.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(dbeta.data_ptr()),
      reinterpret_cast<float*>(dA_log.data_ptr()), reinterpret_cast<float*>(ddt_bias.data_ptr()),
      static_cast<int>(num_pair_heads), static_cast<int>(num_heads), static_cast<float>(scale),
      static_cast<float>(lower_bound));
  CheckCuda(cudaGetLastError(), "training C32 map finalize launch");

  if (grouped != 0) {
    kernel_flashkda_grouped_qk_reduce<<<
        dim3(CheckedGridX((total_tokens + 15) / 16, "grouped reduce"), num_qk_heads, 1), 128, 0,
        stream>>>(reinterpret_cast<__nv_bfloat16*>(dq_value.data_ptr()),
                  reinterpret_cast<__nv_bfloat16*>(dk_value.data_ptr()),
                  reinterpret_cast<__nv_bfloat16*>(dq_native.data_ptr()),
                  reinterpret_cast<__nv_bfloat16*>(dk_native.data_ptr()),
                  static_cast<int>(total_tokens), static_cast<int>(num_qk_heads),
                  static_cast<int>(num_heads));
    CheckCuda(cudaGetLastError(), "grouped C32 Q/K reduce launch");
  }
}

}  // namespace flash_kda_training_fallback
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_training_row_forward,
                              flashinfer::flash_kda_training_fallback::RunTrainingRowForward);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_training_row_backward,
                              flashinfer::flash_kda_training_fallback::RunTrainingRowBackward);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_training_c32_forward,
                              flashinfer::flash_kda_training_fallback::RunTrainingC32Forward);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_training_c32_backward,
                              flashinfer::flash_kda_training_fallback::RunTrainingC32Backward);
