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

#define uint8_t flashkda_v483_generated_uint8_t
#define uint16_t flashkda_v483_generated_uint16_t
#define uint32_t flashkda_v483_generated_uint32_t
#define uint64_t flashkda_v483_generated_uint64_t
#define int32_t flashkda_v483_generated_int32_t
#define int16_t flashkda_v483_generated_int16_t
#define FlashKDATensorMap flashkda_v483_generated_FlashKDATensorMap
#define FlashKDATensorMapPack flashkda_v483_generated_FlashKDATensorMapPack
#define CUtensorMap flashkda_v483_generated_CUtensorMap
#include "flashkda_backward_v483.cu"
#undef CUtensorMap
#undef FlashKDATensorMapPack
#undef FlashKDATensorMap
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t

namespace flashinfer {
namespace flash_kda_backward_v483 {

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
constexpr size_t kTensorMapCount = 8;
constexpr size_t kDescriptorBytes = kTensorMapCount * sizeof(CUtensorMap);
constexpr int32_t kBackwardSmemBytes = 230400;

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

inline CUtensorMap EncodeStateTensor(const TensorView& tensor) {
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
      << "cuTensorMapEncodeTiled failed for state checkpoints with CUresult=" << int(result);
  return map;
}

inline CUtensorMap EncodeBetaTensor(const TensorView& tensor) {
  uint64_t global_dim[2] = {kHeads, kTokens};
  uint64_t global_strides[1] = {kHeads * sizeof(__nv_bfloat16)};
  uint32_t box_dim[2] = {8, 16};
  uint32_t elem_strides[2] = {1, 1};
  CUtensorMap map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, tensor.data_ptr(), global_dim, global_strides,
      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for beta with CUresult=" << int(result);
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

struct TmaPointers {
  void* q;
  void* k;
  void* v;
  void* g;
  void* do_;
  void* state;
  void* dv;
  void* beta;
};

inline TmaPointers PrepareTensorMaps(const TensorView& q, const TensorView& k, const TensorView& v,
                                     const TensorView& g, const TensorView& do_,
                                     const TensorView& state, const TensorView& dv,
                                     const TensorView& beta, const TensorView& descriptor_storage,
                                     int64_t prepare_descriptors, cudaStream_t stream) {
  if (prepare_descriptors != 0) {
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    CheckCuda(cudaStreamIsCapturing(stream, &capture_status), "cudaStreamIsCapturing");
    TVM_FFI_ICHECK(capture_status == cudaStreamCaptureStatusNone)
        << "C16 descriptors must be warmed before CUDA graph capture";
    const std::array<CUtensorMap, kTensorMapCount> maps = {
        EncodeTokenTensor(q, "q"),   EncodeTokenTensor(k, "k"),    EncodeTokenTensor(v, "v"),
        EncodeTokenTensor(g, "g"),   EncodeTokenTensor(do_, "do"), EncodeStateTensor(state),
        EncodeTokenTensor(dv, "dv"), EncodeBetaTensor(beta),
    };
    TensorMapWords words{};
    std::memcpy(words.words, maps.data(), sizeof(maps));
    PublishTensorMaps<<<1, kDescriptorBytes / sizeof(uint64_t), 0, stream>>>(
        reinterpret_cast<uint64_t*>(descriptor_storage.data_ptr()), words);
    CheckCuda(cudaGetLastError(), "PublishTensorMaps launch");
  }
  auto* bytes = static_cast<unsigned char*>(descriptor_storage.data_ptr());
  constexpr size_t stride = sizeof(CUtensorMap);
  return {bytes + 0 * stride, bytes + 1 * stride, bytes + 2 * stride, bytes + 3 * stride,
          bytes + 4 * stride, bytes + 5 * stride, bytes + 6 * stride, bytes + 7 * stride};
}

template <typename Kernel>
inline void ConfigureDynamicSmem(Kernel kernel, int32_t bytes, int32_t device_id,
                                 const char* name) {
  CheckDynamicSmemCapacity(device_id, bytes);
  CheckCuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes), name);
}

void RunC16Backward(TensorView q, TensorView k, TensorView v, TensorView g, TensorView A_log,
                    TensorView dt_bias, TensorView do_, TensorView dfinal_state,
                    TensorView cu_seqlens, TensorView backward_work_items,
                    TensorView descriptor_storage, TensorView state_checkpoints,
                    TensorView beta_active, TensorView dlog_decay, TensorView dlog_boundary,
                    TensorView dbeta_active, TensorView gate_part_a, TensorView gate_part_dt,
                    TensorView counter, TensorView dummy_u32, TensorView dummy_f32, TensorView dq,
                    TensorView dk, TensorView dv, TensorView dg, TensorView dbeta,
                    TensorView dA_log, TensorView ddt_bias, TensorView dinitial_state,
                    int64_t prepare_descriptors, int64_t num_sequences, int64_t num_heads,
                    double scale, double lower_bound, int64_t cuda_stream) {
  TVM_FFI_ICHECK(q.device().device_type == kDLCUDA) << "q must be a CUDA tensor";
  const int32_t device_id = q.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckFlashKDATarget(device_id);
  TVM_FFI_ICHECK(num_sequences == kSequences && num_heads == kHeads)
      << "the C16 route requires eight sequences and 96 heads";
  TVM_FFI_ICHECK(prepare_descriptors == 0 || prepare_descriptors == 1);
  TVM_FFI_ICHECK(std::abs(scale - 1.0 / std::sqrt(128.0)) <= 1e-15)
      << "the C16 route fixes scale=1/sqrt(128)";
  TVM_FFI_ICHECK(lower_bound == -5.0) << "the C16 route fixes lower_bound=-5.0";

  for (const auto& named : std::initializer_list<std::pair<TensorView*, const char*>>{
           {&q, "q"},
           {&k, "k"},
           {&v, "v"},
           {&g, "g"},
           {&do_, "do"},
           {&state_checkpoints, "state_checkpoints"},
           {&beta_active, "beta_active"},
           {&dq, "dq"},
           {&dk, "dk"},
           {&dv, "dv"},
           {&dg, "dg"},
           {&dbeta, "dbeta"}}) {
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
  CheckTensor(backward_work_items, "backward_work_items", device_id, dl_int32);
  CheckTensor(counter, "counter", device_id, dl_uint32);
  CheckTensor(dummy_u32, "dummy_u32", device_id, dl_uint32);
  CheckTensor(descriptor_storage, "descriptor_storage", device_id, dl_uint8);
  CheckNumel(q, "q", kTokens * kHeads * kHeadDim);
  CheckNumel(state_checkpoints, "state_checkpoints", kChunks * kHeads * kHeadDim * kHeadDim);
  CheckNumel(backward_work_items, "backward_work_items", kWorkItems * 5);
  TVM_FFI_ICHECK(descriptor_storage.numel() >= static_cast<int64_t>(kDescriptorBytes));
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(descriptor_storage.data_ptr()) % 64 == 0);

  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  const TmaPointers tma = PrepareTensorMaps(q, k, v, g, do_, state_checkpoints, dv, beta_active,
                                            descriptor_storage, prepare_descriptors, stream);
  int resident_ctas = 0;
  CheckCuda(cudaDeviceGetAttribute(&resident_ctas, cudaDevAttrMultiProcessorCount, device_id),
            "cudaDeviceGetAttribute(multiProcessorCount)");
  const dim3 persistent_grid(std::min<int64_t>(kWorkItems, resident_ctas), 1, 1);

  ConfigureDynamicSmem(kernel_flashkda_backward_persistent_c16, kBackwardSmemBytes, device_id,
                       "cudaFuncSetAttribute(C16 backward)");
  kernel_flashkda_backward_persistent_c16<<<persistent_grid, 512, kBackwardSmemBytes, stream>>>(
      reinterpret_cast<unsigned int*>(counter.data_ptr()),
      reinterpret_cast<flashkda_v483_generated_FlashKDATensorMap const*>(tma.q),
      reinterpret_cast<flashkda_v483_generated_FlashKDATensorMap const*>(tma.k),
      reinterpret_cast<flashkda_v483_generated_FlashKDATensorMap const*>(tma.g),
      reinterpret_cast<flashkda_v483_generated_FlashKDATensorMap const*>(tma.do_),
      reinterpret_cast<flashkda_v483_generated_FlashKDATensorMap const*>(tma.v),
      reinterpret_cast<flashkda_v483_generated_FlashKDATensorMap const*>(tma.state),
      reinterpret_cast<float*>(dfinal_state.data_ptr()),
      reinterpret_cast<flashkda_v483_generated_FlashKDATensorMap const*>(tma.dv),
      reinterpret_cast<__nv_bfloat16*>(dq.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(dk.data_ptr()),
      reinterpret_cast<float*>(dlog_decay.data_ptr()),
      reinterpret_cast<float*>(dlog_boundary.data_ptr()),
      reinterpret_cast<float*>(dinitial_state.data_ptr()),
      reinterpret_cast<float*>(A_log.data_ptr()), reinterpret_cast<float*>(dt_bias.data_ptr()),
      reinterpret_cast<flashkda_v483_generated_FlashKDATensorMap const*>(tma.beta),
      reinterpret_cast<float*>(dbeta_active.data_ptr()),
      reinterpret_cast<long long*>(cu_seqlens.data_ptr()),
      reinterpret_cast<int*>(backward_work_items.data_ptr()),
      reinterpret_cast<unsigned int*>(dummy_u32.data_ptr()),
      reinterpret_cast<float*>(dummy_f32.data_ptr()),
      reinterpret_cast<float*>(dummy_f32.data_ptr()),
      reinterpret_cast<float*>(dummy_f32.data_ptr()),
      reinterpret_cast<float*>(dummy_f32.data_ptr()),
      reinterpret_cast<float*>(dummy_f32.data_ptr()),
      reinterpret_cast<float*>(dummy_f32.data_ptr()),
      reinterpret_cast<float*>(dummy_f32.data_ptr()),
      reinterpret_cast<float*>(dummy_f32.data_ptr()),
      reinterpret_cast<float*>(dummy_f32.data_ptr()),
      reinterpret_cast<float*>(dummy_f32.data_ptr()),
      reinterpret_cast<float*>(dummy_f32.data_ptr()),
      reinterpret_cast<float*>(dummy_f32.data_ptr()),
      reinterpret_cast<float*>(dummy_f32.data_ptr()),
      reinterpret_cast<float*>(dummy_f32.data_ptr()), kWorkItems, 1, 64, kHeads, 1, 1,
      static_cast<float>(scale), static_cast<float>(lower_bound));
  CheckCuda(cudaGetLastError(), "C16 backward launch");

  kernel_flashkda_backward_param_reduce_c16_partial<<<dim3(128, kHeads, 1), 128, 4096, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(g.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(beta_active.data_ptr()),
      reinterpret_cast<float*>(A_log.data_ptr()), reinterpret_cast<float*>(dt_bias.data_ptr()),
      reinterpret_cast<float*>(dlog_decay.data_ptr()),
      reinterpret_cast<float*>(dlog_boundary.data_ptr()),
      reinterpret_cast<float*>(dbeta_active.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(dg.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(dbeta.data_ptr()),
      reinterpret_cast<float*>(gate_part_a.data_ptr()),
      reinterpret_cast<float*>(gate_part_dt.data_ptr()), kTokens, kHeads, 64,
      static_cast<float>(lower_bound));
  CheckCuda(cudaGetLastError(), "C16 parameter partial launch");
  kernel_flashkda_backward_param_reduce_c16_finish<<<kHeads, 128, 16, stream>>>(
      reinterpret_cast<float*>(gate_part_a.data_ptr()),
      reinterpret_cast<float*>(gate_part_dt.data_ptr()),
      reinterpret_cast<float*>(dA_log.data_ptr()), reinterpret_cast<float*>(ddt_bias.data_ptr()),
      kHeads);
  CheckCuda(cudaGetLastError(), "C16 parameter finish launch");
}

}  // namespace flash_kda_backward_v483
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_c16_backward,
                              flashinfer::flash_kda_backward_v483::RunC16Backward);
