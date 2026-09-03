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

#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>

#include "flashkda_binding_common.cuh"

#if defined(FLASHKDA_GENERATED_EMBEDDED_CUBIN)
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#ifndef FLASHKDA_GENERATED_CUBIN_IDENT
#error "FLASHKDA_GENERATED_CUBIN_IDENT must name the embedded cubin"
#endif
#define FLASHKDA_FP32_COMPAT_EMBED_IMPL(name) TVM_FFI_EMBED_CUBIN(name)
#define FLASHKDA_FP32_COMPAT_EMBED(name) \
  FLASHKDA_FP32_COMPAT_EMBED_IMPL(name)
FLASHKDA_FP32_COMPAT_EMBED(FLASHKDA_GENERATED_CUBIN_IDENT);
#define FLASHKDA_FP32_COMPAT_GET_KERNEL_IMPL(name, kernel_name) \
  TVM_FFI_EMBED_CUBIN_GET_KERNEL(name, kernel_name)
#define FLASHKDA_FP32_COMPAT_GET_KERNEL(name, kernel_name) \
  FLASHKDA_FP32_COMPAT_GET_KERNEL_IMPL(name, kernel_name)
#define FLASHKDA_FP32_COMPAT_STRINGIFY_IMPL(value) #value
#define FLASHKDA_FP32_COMPAT_STRINGIFY(value) \
  FLASHKDA_FP32_COMPAT_STRINGIFY_IMPL(value)
#define FLASHKDA_FP32_COMPAT_KERNEL_ARGUMENT nullptr
#else
#define FLASHKDA_FP32_COMPAT_KERNEL_ARGUMENT \
  reinterpret_cast<const void*>(FLASHKDA_GENERATED_KERNEL)
#endif

#ifndef FLASHKDA_GENERATED_BODY_FILE
#error "compact-FP32 selector must name one audited generated body"
#endif
#ifndef FLASHKDA_GENERATED_KERNEL
#error "compact-FP32 selector must name the audited kernel symbol"
#endif
#ifndef FLASHKDA_GENERATED_THREADS
#error "compact-FP32 selector must identify the audited thread count"
#endif
#ifndef FLASHKDA_GENERATED_SMEM_BYTES
#error "compact-FP32 selector must identify the audited dynamic shared memory"
#endif
#ifndef FLASHKDA_GENERATED_USE_PDL
#error "compact-FP32 selector must identify the audited PDL mode"
#endif

#ifndef FLASHKDA_FP32_COMPAT_USE_INITIAL_STATE
#error "compact-FP32 selector must identify USE_INITIAL_STATE"
#endif
#ifndef FLASHKDA_FP32_COMPAT_STORE_FINAL_STATE
#error "compact-FP32 selector must identify STORE_FINAL_STATE"
#endif

static_assert(FLASHKDA_FP32_COMPAT_USE_INITIAL_STATE == 1,
              "the public compact-FP32 route requires initial state");
static_assert(FLASHKDA_FP32_COMPAT_STORE_FINAL_STATE == 1,
              "the public compact-FP32 route stores final state");
static_assert(FLASHKDA_GENERATED_THREADS == 384,
              "compact-FP32 source launch uses 384 threads");
static_assert(FLASHKDA_GENERATED_SMEM_BYTES == 226048,
              "compact-FP32 source launch uses 226048 dynamic shared-memory bytes");
static_assert(FLASHKDA_GENERATED_USE_PDL == 0,
              "compact-FP32 source launch does not use PDL");

#if !defined(FLASHKDA_GENERATED_EMBEDDED_CUBIN)
#define int8_t flashkda_fp32_compat_private_int8_t
#define uint8_t flashkda_fp32_compat_private_uint8_t
#define uint16_t flashkda_fp32_compat_private_uint16_t
#define uint32_t flashkda_fp32_compat_private_uint32_t
#define uint64_t flashkda_fp32_compat_private_uint64_t
#define int32_t flashkda_fp32_compat_private_int32_t
#define int16_t flashkda_fp32_compat_private_int16_t
#define CUtensorMap flashkda_fp32_compat_private_CUtensorMap
#include FLASHKDA_GENERATED_BODY_FILE
#undef CUtensorMap
#undef int8_t
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t
#ifdef THREADS
static_assert(THREADS == FLASHKDA_GENERATED_THREADS,
              "compact-FP32 body and selector thread counts disagree");
#endif
#ifdef SMEM_TOTAL
static_assert(SMEM_TOTAL == FLASHKDA_GENERATED_SMEM_BYTES,
              "compact-FP32 body and selector shared-memory sizes disagree");
#endif
#endif

namespace flashinfer::flash_kda_generated_fp32_compat {

using flash_kda::CheckCuda;
using flash_kda::CheckCudaTensor;
using flash_kda::CheckCudaTensorDevice;
using flash_kda::CheckDtype;

constexpr int64_t kHeadDim = 128;
constexpr int64_t kDescriptorCount = 5;
constexpr int64_t kDescriptorBytes = kDescriptorCount * sizeof(CUtensorMap);
constexpr int64_t kTensorMapWorkspaceBytesPerCta = 10 * 128;
constexpr int64_t kMaxActiveClusters = 148;

inline dim3 CheckedGrid(int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  for (const auto& named : {std::pair<int64_t, const char*>{grid_x, "grid_x"},
                            std::pair<int64_t, const char*>{grid_y, "grid_y"},
                            std::pair<int64_t, const char*>{grid_z, "grid_z"}}) {
    TVM_FFI_ICHECK(named.first > 0 &&
                   named.first <= std::numeric_limits<uint32_t>::max())
        << named.second << " must be in the positive uint32 range";
  }
  return dim3(static_cast<uint32_t>(grid_x), static_cast<uint32_t>(grid_y),
              static_cast<uint32_t>(grid_z));
}

inline int32_t CheckedInt32(int64_t value, const char* name) {
  TVM_FFI_ICHECK(value >= std::numeric_limits<int32_t>::min() &&
                 value <= std::numeric_limits<int32_t>::max())
      << name << " must fit int32";
  return static_cast<int32_t>(value);
}

inline cudaStream_t CheckedStream(int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0)
      << "cuda_stream must be a non-negative stream handle";
  return reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
}

template <size_t Expected, size_t Actual>
constexpr void CheckArgumentCount(void* (&)[Actual]) {
  static_assert(Expected == Actual,
                "compact-FP32 generated kernel ABI argument count changed");
}

static inline void ConfigureAndLaunch(const void* kernel, dim3 grid,
                                      cudaStream_t stream, void** args,
                                      const char* launch_name) {
  int32_t device_id = 0;
  CheckCuda(cudaGetDevice(&device_id), "cudaGetDevice");
  flash_kda::CheckDynamicSmemCapacity(device_id,
                                      FLASHKDA_GENERATED_SMEM_BYTES);
#if defined(FLASHKDA_GENERATED_EMBEDDED_CUBIN)
  (void)kernel;
  static auto embedded_kernel = FLASHKDA_FP32_COMPAT_GET_KERNEL(
      FLASHKDA_GENERATED_CUBIN_IDENT,
      FLASHKDA_FP32_COMPAT_STRINGIFY(FLASHKDA_GENERATED_KERNEL));
  namespace cuda_api = tvm::ffi::cuda_api;
  auto device = cuda_api::GetDeviceHandle(device_id);
  TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(cuda_api::SetKernelMaxDynamicSharedMem(
      embedded_kernel.GetHandle(), FLASHKDA_GENERATED_SMEM_BYTES, device));
  TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(embedded_kernel.Launch(
      args, tvm::ffi::dim3(grid.x, grid.y, grid.z),
      tvm::ffi::dim3(FLASHKDA_GENERATED_THREADS, 1, 1), stream,
      FLASHKDA_GENERATED_SMEM_BYTES));
#else
  CheckCuda(cudaFuncSetAttribute(kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 FLASHKDA_GENERATED_SMEM_BYTES),
            "cudaFuncSetAttribute(compact-FP32 FlashKDA kernel)");
  CheckCuda(cudaLaunchKernel(kernel, grid,
                             dim3(FLASHKDA_GENERATED_THREADS, 1, 1), args,
                             FLASHKDA_GENERATED_SMEM_BYTES, stream),
            launch_name);
#endif
}

inline void CheckExactBf16Tensor(const TensorView& tensor, const char* name,
                                 int32_t device_id) {
  CheckCudaTensor(tensor, name, device_id);
  CheckDtype(tensor, name, dl_bfloat16);
}

inline void CheckExactF32Tensor(const TensorView& tensor, const char* name,
                                int32_t device_id) {
  CheckCudaTensor(tensor, name, device_id);
  CheckDtype(tensor, name, dl_float32);
}

inline CUtensorMap EncodeCompatTensorMap(const TensorView& tensor,
                                         const char* name,
                                         int64_t total_tokens,
                                         int64_t num_heads) {
  uint64_t global_dim[3] = {kHeadDim, static_cast<uint64_t>(total_tokens),
                            static_cast<uint64_t>(num_heads)};
  uint64_t global_strides[2] = {
      static_cast<uint64_t>(num_heads) * kHeadDim * sizeof(__nv_bfloat16),
      kHeadDim * sizeof(__nv_bfloat16)};
  uint32_t box_dim[3] = {64, 64, 1};
  uint32_t element_strides[3] = {1, 1, 1};
  CUtensorMap map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, tensor.data_ptr(),
      global_dim, global_strides, box_dim, element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for compact-FP32 " << name
      << " with CUresult=" << int(result);
  return map;
}

struct CompatTensorMapWords {
  uint64_t words[kDescriptorBytes / sizeof(uint64_t)];
};

static __global__ void PublishCompatTensorMaps(uint64_t* destination,
                                                CompatTensorMapWords source) {
  if (threadIdx.x == 0) {
    for (uint32_t index = 0;
         index < kDescriptorBytes / sizeof(uint64_t); ++index) {
      destination[index] = source.words[index];
    }
    asm volatile("fence.proxy.tensormap::generic.release.sys;" ::: "memory");
  }
}

inline void PrepareCompatTensorMaps(
    const TensorView& q, const TensorView& k, const TensorView& v,
    const TensorView& g, const TensorView& out,
    const TensorView& descriptor_storage, int64_t total_tokens,
    int64_t num_heads, int64_t prepare_descriptors, cudaStream_t stream) {
  TVM_FFI_ICHECK(prepare_descriptors == 0 || prepare_descriptors == 1)
      << "prepare_descriptors must be 0 or 1";
  if (prepare_descriptors == 0) {
    return;
  }
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  CheckCuda(cudaStreamIsCapturing(stream, &capture_status),
            "cudaStreamIsCapturing");
  TVM_FFI_ICHECK(capture_status == cudaStreamCaptureStatusNone)
      << "compact-FP32 descriptors must be prepared outside CUDA graph capture";
  const std::array<CUtensorMap, kDescriptorCount> maps = {
      EncodeCompatTensorMap(q, "q", total_tokens, num_heads),
      EncodeCompatTensorMap(k, "k", total_tokens, num_heads),
      EncodeCompatTensorMap(v, "v", total_tokens, num_heads),
      EncodeCompatTensorMap(g, "g", total_tokens, num_heads),
      EncodeCompatTensorMap(out, "out", total_tokens, num_heads),
  };
  CompatTensorMapWords words{};
  std::memcpy(words.words, maps.data(), sizeof(maps));
  PublishCompatTensorMaps<<<1, 1, 0, stream>>>(
      reinterpret_cast<uint64_t*>(descriptor_storage.data_ptr()), words);
  CheckCuda(cudaGetLastError(), "PublishCompatTensorMaps launch");
}

inline void CheckCompactState(const TensorView& state, const char* name,
                              int32_t device_id, int64_t num_sequences,
                              int64_t num_heads, int64_t state_slot_stride) {
  CheckCudaTensorDevice(state, name, device_id);
  CheckDtype(state, name, dl_float32);
  TVM_FFI_ICHECK(state.ndim() == 4 && state.size(0) == num_sequences &&
                 state.size(1) == num_heads && state.size(2) == kHeadDim &&
                 state.size(3) == kHeadDim)
      << name << " must have shape [num_sequences,H,128,128]";
  TVM_FFI_ICHECK(state.stride(3) == 1 && state.stride(2) == kHeadDim &&
                 state.stride(1) == kHeadDim * kHeadDim)
      << name << " must be contiguous inside each compact state slot";
  TVM_FFI_ICHECK(state.stride(0) == state_slot_stride &&
                 state_slot_stride >= num_heads * kHeadDim * kHeadDim)
      << name << " has an invalid compact state slot stride";
}

void RunCompactFP32Compat(
    TensorView q, TensorView k, TensorView v, TensorView g, TensorView beta,
    TensorView a_log, TensorView dt_bias, TensorView cu_seqlens,
    TensorView state_indices_dummy, TensorView initial_state, TensorView out,
    TensorView final_state, TensorView checkpoint_state_dummy,
    TensorView cu_checkpoints_dummy, TensorView tensormap_workspace,
    TensorView descriptor_storage, int64_t prepare_descriptors,
    int64_t state_slot_stride, int64_t use_state_indices,
    int64_t use_initial_state, int64_t store_final_state, double scale,
    double lower_bound, int64_t cuda_stream) {
  TVM_FFI_ICHECK(q.device().device_type == kDLCUDA)
      << "q must be a CUDA tensor";
  const int32_t device_id = q.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  flash_kda::CheckFlashKDATarget(device_id);
  TVM_FFI_ICHECK(use_state_indices == 0)
      << "compact-FP32 route does not accept state indices";
  TVM_FFI_ICHECK(use_initial_state == 1 && store_final_state == 1)
      << "compact-FP32 selector requires initial and final state";
  TVM_FFI_ICHECK(std::isfinite(scale) &&
                 std::isfinite(static_cast<float>(scale)))
      << "scale must be finite and representable as float32";
  TVM_FFI_ICHECK(std::isfinite(lower_bound) && lower_bound >= -5.0 &&
                 lower_bound <= 0.0 &&
                 std::isfinite(static_cast<float>(lower_bound)))
      << "compact-FP32 lower_bound must be in [-5.0, 0.0]";

  for (const auto& named :
       std::initializer_list<std::pair<TensorView*, const char*>>{
           {&q, "q"}, {&k, "k"}, {&v, "v"}, {&g, "g"},
           {&out, "out"}, {&beta, "beta"}}) {
    CheckExactBf16Tensor(*named.first, named.second, device_id);
  }
  CheckExactF32Tensor(a_log, "A_log", device_id);
  CheckExactF32Tensor(dt_bias, "dt_bias", device_id);
  CheckCudaTensor(cu_seqlens, "cu_seqlens", device_id);
  CheckDtype(cu_seqlens, "cu_seqlens", dl_int64);
  CheckCudaTensor(state_indices_dummy, "state_indices_dummy", device_id);
  CheckDtype(state_indices_dummy, "state_indices_dummy", dl_int32);
  CheckCudaTensor(checkpoint_state_dummy, "checkpoint_state_dummy", device_id);
  CheckDtype(checkpoint_state_dummy, "checkpoint_state_dummy", dl_float32);
  CheckCudaTensor(cu_checkpoints_dummy, "cu_checkpoints_dummy", device_id);
  CheckDtype(cu_checkpoints_dummy, "cu_checkpoints_dummy", dl_int32);
  CheckCudaTensor(tensormap_workspace, "tensormap_workspace", device_id);
  CheckDtype(tensormap_workspace, "tensormap_workspace", dl_uint8);
  CheckCudaTensor(descriptor_storage, "descriptor_storage", device_id);
  CheckDtype(descriptor_storage, "descriptor_storage", dl_uint8);

  TVM_FFI_ICHECK(q.ndim() == 4 && q.size(0) > 0 && q.size(1) > 1 &&
                 q.size(2) > 0 && q.size(3) == kHeadDim)
      << "q must have non-empty shape [B,T,H,128] with T > 1";
  for (const auto& named :
       std::initializer_list<std::pair<TensorView*, const char*>>{
           {&k, "k"}, {&v, "v"}, {&g, "g"}, {&out, "out"}}) {
    TVM_FFI_ICHECK(named.first->ndim() == 4 &&
                   named.first->numel() == q.numel())
        << named.second << " must match q's [B,T,H,128] shape";
    for (int32_t dim = 0; dim < 4; ++dim) {
      TVM_FFI_ICHECK(named.first->size(dim) == q.size(dim))
          << named.second << " must match q's [B,T,H,128] shape";
    }
  }
  TVM_FFI_ICHECK(beta.ndim() == 3 && beta.size(0) == q.size(0) &&
                 beta.size(1) == q.size(1) && beta.size(2) == q.size(2))
      << "beta must have shape [B,T,H] matching q";
  const int64_t total_tokens = q.size(0) * q.size(1);
  const int64_t num_heads = q.size(2);
  TVM_FFI_ICHECK(a_log.ndim() == 1 && a_log.size(0) == num_heads)
      << "A_log must have shape [H]";
  TVM_FFI_ICHECK(dt_bias.ndim() == 2 && dt_bias.size(0) == num_heads &&
                 dt_bias.size(1) == kHeadDim)
      << "dt_bias must have shape [H,128]";
  const int64_t num_sequences = cu_seqlens.numel() - 1;
  TVM_FFI_ICHECK(num_sequences > 0 && num_sequences <= total_tokens)
      << "cu_seqlens must contain at least two entries";
  CheckCompactState(initial_state, "initial_state", device_id, num_sequences,
                    num_heads, state_slot_stride);
  CheckCompactState(final_state, "final_state", device_id, num_sequences,
                    num_heads, state_slot_stride);

  TVM_FFI_ICHECK(descriptor_storage.numel() >= kDescriptorBytes)
      << "descriptor_storage must hold five CUtensorMap objects";
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(descriptor_storage.data_ptr()) %
                     flash_kda::kTensorMapAlignment ==
                 0)
      << "descriptor_storage must be CUtensorMap aligned";
  const int64_t total_tiles = num_sequences * num_heads;
  TVM_FFI_ICHECK(total_tiles > 0 &&
                 total_tiles <= std::numeric_limits<int32_t>::max())
      << "compact-FP32 total tile count must fit int32";
  const int64_t grid_x = std::min(kMaxActiveClusters, total_tiles);
  TVM_FFI_ICHECK(tensormap_workspace.numel() >=
                 grid_x * kTensorMapWorkspaceBytesPerCta)
      << "tensormap_workspace is too small for compact-FP32 grid";

  const cudaStream_t stream = CheckedStream(cuda_stream);
  PrepareCompatTensorMaps(q, k, v, g, out, descriptor_storage, total_tokens,
                          num_heads, prepare_descriptors, stream);
  auto* descriptor_base =
      reinterpret_cast<uint8_t*>(descriptor_storage.data_ptr());
  void* q_map = descriptor_base + 0 * sizeof(CUtensorMap);
  void* k_map = descriptor_base + 1 * sizeof(CUtensorMap);
  void* v_map = descriptor_base + 2 * sizeof(CUtensorMap);
  void* g_map = descriptor_base + 3 * sizeof(CUtensorMap);
  void* out_map = descriptor_base + 4 * sizeof(CUtensorMap);
  void* beta_ptr = beta.data_ptr();
  void* a_log_ptr = a_log.data_ptr();
  void* dt_bias_ptr = dt_bias.data_ptr();
  void* cu_seqlens_ptr = cu_seqlens.data_ptr();
  void* state_indices_ptr = state_indices_dummy.data_ptr();
  int64_t state_slot_stride_arg = state_slot_stride;
  int32_t use_state_indices_arg = 0;
  void* initial_state_ptr = initial_state.data_ptr();
  void* output_state_ptr = final_state.data_ptr();
  void* checkpoint_state_ptr = checkpoint_state_dummy.data_ptr();
  void* cu_checkpoints_ptr = cu_checkpoints_dummy.data_ptr();
  void* tensormap_workspace_ptr = tensormap_workspace.data_ptr();
  int32_t checkpoint_every_n_tokens_arg = 0;
  float scale_arg = static_cast<float>(scale);
  int32_t num_sequences_arg = CheckedInt32(num_sequences, "num_sequences");
  int32_t num_heads_arg = CheckedInt32(num_heads, "num_heads");
  int32_t total_tiles_arg = CheckedInt32(total_tiles, "total_tiles");
  float lower_bound_arg = static_cast<float>(lower_bound);
  void* kernel_args[] = {
      &q_map, &k_map, &v_map, &g_map, &out_map, &beta_ptr, &a_log_ptr,
      &dt_bias_ptr, &cu_seqlens_ptr, &state_indices_ptr,
      &state_slot_stride_arg, &use_state_indices_arg, &initial_state_ptr,
      &output_state_ptr, &checkpoint_state_ptr, &cu_checkpoints_ptr,
      &tensormap_workspace_ptr, &checkpoint_every_n_tokens_arg, &scale_arg,
      &num_sequences_arg, &num_heads_arg, &num_heads_arg, &total_tiles_arg,
      &lower_bound_arg};
  CheckArgumentCount<24>(kernel_args);
  ConfigureAndLaunch(FLASHKDA_FP32_COMPAT_KERNEL_ARGUMENT,
                     CheckedGrid(grid_x, 1, 1), stream, kernel_args,
                     "generated compact-FP32 compatibility launch");
}

}  // namespace flashinfer::flash_kda_generated_fp32_compat

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    run, flashinfer::flash_kda_generated_fp32_compat::RunCompactFP32Compat);
