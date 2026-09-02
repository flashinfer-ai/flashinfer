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

#define uint8_t flashkda_checkpoint_generated_uint8_t
#define uint16_t flashkda_checkpoint_generated_uint16_t
#define uint32_t flashkda_checkpoint_generated_uint32_t
#define uint64_t flashkda_checkpoint_generated_uint64_t
#define int32_t flashkda_checkpoint_generated_int32_t
#define int16_t flashkda_checkpoint_generated_int16_t
#define LoomTensorMap flashkda_checkpoint_generated_LoomTensorMap
#define LoomTensorMapPack flashkda_checkpoint_generated_LoomTensorMapPack
#define CUtensorMap flashkda_checkpoint_generated_CUtensorMap
#include "flashkda_bf16_fused_m128_n16_checkpoint.cu"
#undef CUtensorMap
#undef LoomTensorMapPack
#undef LoomTensorMap
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t

namespace flashinfer {
namespace flash_kda {

constexpr int32_t kCheckpointThreads = 1024;
constexpr int32_t kCheckpointSmemBytes = 183296;

struct CheckpointMapWords {
  uint64_t words[sizeof(CUtensorMap) / sizeof(uint64_t)];
};

static __global__ void PublishCheckpointMap(uint64_t* destination, CheckpointMapWords source) {
  if (threadIdx.x < sizeof(CUtensorMap) / sizeof(uint64_t)) {
    destination[threadIdx.x] = source.words[threadIdx.x];
  }
}

inline CUtensorMap EncodeCheckpointValueTma(const TensorView& tensor) {
  TVM_FFI_ICHECK(tensor.ndim() >= 2) << "v must have at least two dimensions";
  TVM_FFI_ICHECK(tensor.stride(tensor.ndim() - 1) == 1) << "v must have unit innermost stride";
  const int64_t d1 = tensor.size(tensor.ndim() - 1);
  const int64_t d2 = tensor.size(tensor.ndim() - 2);
  TVM_FFI_ICHECK(d1 > 0 && d2 > 0 && d1 % 64 == 0)
      << "v trailing dimensions cannot encode the checkpoint N16 TMA box";
  const int64_t outer2 = tensor.numel() / (d1 * d2);
  uint64_t global_dim[4] = {64, static_cast<uint64_t>(d2), static_cast<uint64_t>(outer2),
                            static_cast<uint64_t>(d1 / 64)};
  uint64_t global_strides[3] = {static_cast<uint64_t>(d1 * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(d1 * d2 * sizeof(__nv_bfloat16)),
                                64 * sizeof(__nv_bfloat16)};
  uint32_t box_dim[4] = {64, 1, 16, 1};
  uint32_t elem_strides[4] = {1, 1, 1, 1};
  CUtensorMap map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, tensor.data_ptr(), global_dim, global_strides,
      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for checkpoint N16 v with CUresult=" << int(result);
  return map;
}

inline CUtensorMap EncodeCheckpointTma(const TensorView& tensor) {
  const int64_t checkpoints = tensor.size(0);
  const int64_t heads = tensor.size(1);
  uint64_t global_dim[4] = {128, 128, static_cast<uint64_t>(heads),
                            static_cast<uint64_t>(checkpoints)};
  uint64_t global_strides[3] = {128 * sizeof(__nv_bfloat16), 128 * 128 * sizeof(__nv_bfloat16),
                                static_cast<uint64_t>(heads) * 128 * 128 * sizeof(__nv_bfloat16)};
  uint32_t box_dim[4] = {64, 128, 1, 1};
  uint32_t elem_strides[4] = {1, 1, 1, 1};
  CUtensorMap map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, tensor.data_ptr(), global_dim, global_strides,
      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for state checkpoints with CUresult=" << int(result);
  return map;
}

void RunM128N16Checkpoint(TensorView q, TensorView k, TensorView v, TensorView g, TensorView beta,
                          TensorView beta_tma, TensorView A_log, TensorView dt_bias,
                          TensorView cu_seqlens, TensorView seq_order, TensorView state_indices,
                          TensorView initial_state, TensorView out, TensorView final_state,
                          TensorView state_checkpoints, TensorView checkpoint_cu_starts,
                          TensorView descriptor_storage, int64_t prepare_descriptors,
                          int64_t num_heads, int64_t beta_token_stride, int64_t state_slot_stride,
                          int64_t use_state_indices, int64_t use_initial_state,
                          int64_t store_final_state, int64_t checkpoint_every_n_tokens,
                          double scale, double lower_bound, int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0) << "cuda_stream must be a non-negative stream handle";
  TVM_FFI_ICHECK(checkpoint_every_n_tokens > 0 && checkpoint_every_n_tokens % 16 == 0)
      << "checkpoint_every_n_tokens must be a positive multiple of 16";
  TVM_FFI_ICHECK(q.device().device_type == kDLCUDA) << "q must be a CUDA tensor";
  const int32_t device_id = q.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckFlashKDATarget(device_id);

  const int64_t unchecked_num_seqs = cu_seqlens.numel() - 1;
  const int64_t state_pool_slots = ResolveAndCheckServingStatePool(
      state_indices, initial_state, final_state, device_id, unchecked_num_seqs, num_heads,
      state_slot_stride, use_state_indices, use_initial_state, store_final_state);
  const int64_t num_seqs = CheckCommonInputs(
      q, k, v, g, beta, beta_tma, A_log, dt_bias, cu_seqlens, seq_order, initial_state, out,
      final_state, descriptor_storage, prepare_descriptors, num_heads, use_initial_state,
      store_final_state, scale, lower_bound, true, state_pool_slots);
  TVM_FFI_ICHECK(descriptor_storage.numel() >= 7 * static_cast<int64_t>(sizeof(CUtensorMap)))
      << "checkpoint N16 descriptor_storage must provide at least 896 bytes";
  TVM_FFI_ICHECK(beta_token_stride == beta.stride(beta.ndim() - 2))
      << "beta_token_stride must match beta's physical token stride";
  CheckServingCheckpointInputs(state_checkpoints, checkpoint_cu_starts, device_id, num_seqs,
                               num_heads, checkpoint_every_n_tokens, 16);
  CheckServingAuxiliaryNoOverlap(state_indices, state_checkpoints, checkpoint_cu_starts, q, k, v, g,
                                 beta, beta_tma, A_log, dt_bias, cu_seqlens, seq_order,
                                 initial_state, out, final_state, descriptor_storage,
                                 use_state_indices, checkpoint_every_n_tokens);

  constexpr int32_t kSmemBytes = kCheckpointSmemBytes;
  CheckDynamicSmemCapacity(device_id, kSmemBytes);
  CheckCuda(cudaFuncSetAttribute(kernel_flashkda_bf16_fused_m128,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes),
            "cudaFuncSetAttribute(checkpoint N16)");
  const int64_t grid_x_i64 = num_seqs * num_heads;
  TVM_FFI_ICHECK(grid_x_i64 > 0 && grid_x_i64 <= std::numeric_limits<uint32_t>::max())
      << "checkpoint N16 FlashKDA grid.x is out of range: " << grid_x_i64;
  const dim3 grid(static_cast<uint32_t>(grid_x_i64), 1, 1);
  const dim3 block(kCheckpointThreads, 1, 1);
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  const TmaPointers tma = EncodeTmaPointers<128, 16>(q, k, v, g, beta_tma, out, descriptor_storage,
                                                     prepare_descriptors, stream);
  auto* descriptor_bytes = static_cast<unsigned char*>(descriptor_storage.data_ptr());
  if (prepare_descriptors != 0) {
    const CUtensorMap value_map = EncodeCheckpointValueTma(v);
    CheckpointMapWords value_words{};
    std::memcpy(value_words.words, &value_map, sizeof(value_map));
    PublishCheckpointMap<<<1, sizeof(CUtensorMap) / sizeof(uint64_t), 0, stream>>>(
        reinterpret_cast<uint64_t*>(descriptor_bytes + 2 * sizeof(CUtensorMap)), value_words);
    CheckCuda(cudaGetLastError(), "PublishCheckpointValueMap launch");
    const CUtensorMap checkpoint_map = EncodeCheckpointTma(state_checkpoints);
    CheckpointMapWords words{};
    std::memcpy(words.words, &checkpoint_map, sizeof(checkpoint_map));
    PublishCheckpointMap<<<1, sizeof(CUtensorMap) / sizeof(uint64_t), 0, stream>>>(
        reinterpret_cast<uint64_t*>(descriptor_bytes + 6 * sizeof(CUtensorMap)), words);
    CheckCuda(cudaGetLastError(), "PublishCheckpointMap launch");
  }
  PackBetaForTmaIfNeeded(beta, beta_tma, num_heads, beta_token_stride, stream);

  kernel_flashkda_bf16_fused_m128<<<grid, block, kSmemBytes, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<flashkda_checkpoint_generated_LoomTensorMap const*>(tma.q),
      reinterpret_cast<__nv_bfloat16*>(k.data_ptr()),
      reinterpret_cast<flashkda_checkpoint_generated_LoomTensorMap const*>(tma.k),
      reinterpret_cast<__nv_bfloat16*>(v.data_ptr()),
      reinterpret_cast<flashkda_checkpoint_generated_LoomTensorMap const*>(tma.v),
      reinterpret_cast<__nv_bfloat16*>(g.data_ptr()),
      reinterpret_cast<flashkda_checkpoint_generated_LoomTensorMap const*>(tma.g),
      reinterpret_cast<__nv_bfloat16*>(beta.data_ptr()),
      reinterpret_cast<flashkda_checkpoint_generated_LoomTensorMap const*>(tma.beta),
      reinterpret_cast<float*>(A_log.data_ptr()), reinterpret_cast<float*>(dt_bias.data_ptr()),
      reinterpret_cast<long long*>(cu_seqlens.data_ptr()),
      reinterpret_cast<int*>(seq_order.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(initial_state.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      reinterpret_cast<flashkda_checkpoint_generated_LoomTensorMap const*>(tma.out),
      reinterpret_cast<__nv_bfloat16*>(final_state.data_ptr()), static_cast<int32_t>(num_heads),
      static_cast<int32_t>(use_initial_state), static_cast<int32_t>(store_final_state),
      static_cast<float>(scale), static_cast<float>(lower_bound),
      static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(state_indices.data_ptr())),
      static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(state_checkpoints.data_ptr())),
      static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(checkpoint_cu_starts.data_ptr())),
      static_cast<int64_t>(beta_token_stride), static_cast<int64_t>(state_slot_stride),
      static_cast<int32_t>(use_state_indices), static_cast<int32_t>(checkpoint_every_n_tokens),
      reinterpret_cast<long long*>(checkpoint_cu_starts.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(state_checkpoints.data_ptr()),
      reinterpret_cast<unsigned int*>(descriptor_storage.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(state_checkpoints.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(state_checkpoints.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(state_checkpoints.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(state_checkpoints.data_ptr()),
      reinterpret_cast<float*>(A_log.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(state_checkpoints.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(state_checkpoints.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(state_checkpoints.data_ptr()),
      reinterpret_cast<float*>(A_log.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(state_checkpoints.data_ptr()),
      reinterpret_cast<float*>(A_log.data_ptr()), reinterpret_cast<float*>(A_log.data_ptr()),
      reinterpret_cast<unsigned int*>(descriptor_storage.data_ptr()), 0,
      static_cast<int32_t>(num_seqs),
      reinterpret_cast<flashkda_checkpoint_generated_LoomTensorMap const*>(
          descriptor_bytes + 6 * sizeof(CUtensorMap)));
  CheckCuda(cudaGetLastError(), "checkpoint N16 FlashKDA launch");
}

}  // namespace flash_kda
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::flash_kda::RunM128N16Checkpoint);
