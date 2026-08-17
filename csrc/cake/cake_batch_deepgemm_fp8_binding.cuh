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

#ifndef FLASHINFER_CAKE_BATCH_DEEPGEMM_BODY_FILE
#error "FLASHINFER_CAKE_BATCH_DEEPGEMM_BODY_FILE must name one exported Cake source"
#endif
#ifndef FLASHINFER_CAKE_BATCH_DEEPGEMM_KERNEL
#error "FLASHINFER_CAKE_BATCH_DEEPGEMM_KERNEL must name the exported kernel symbol"
#endif
#ifndef FLASHINFER_CAKE_BATCH_DEEPGEMM_VARIANT
#error "FLASHINFER_CAKE_BATCH_DEEPGEMM_VARIANT must identify the shape bucket"
#endif
#ifndef FLASHINFER_CAKE_BATCH_DEEPGEMM_N
#error "FLASHINFER_CAKE_BATCH_DEEPGEMM_N must identify the output width"
#endif
#ifndef FLASHINFER_CAKE_BATCH_DEEPGEMM_K
#error "FLASHINFER_CAKE_BATCH_DEEPGEMM_K must identify the reduction width"
#endif
#ifndef FLASHINFER_CAKE_BATCH_DEEPGEMM_SMEM_BYTES
#error "FLASHINFER_CAKE_BATCH_DEEPGEMM_SMEM_BYTES must identify dynamic shared memory"
#endif
#ifndef FLASHINFER_CAKE_BATCH_DEEPGEMM_TARGET_KIND
#error "FLASHINFER_CAKE_BATCH_DEEPGEMM_TARGET_KIND must identify the exact target"
#endif

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "tvm_ffi_utils.h"

// The exported source is frozen and self-contained. Rename its private
// fixed-width aliases and tensor-map stand-ins at the include boundary so
// they do not collide with CUDA or libc declarations already in this TU.
#define uint8_t cake_batch_generated_uint8_t
#define uint16_t cake_batch_generated_uint16_t
#define uint32_t cake_batch_generated_uint32_t
#define uint64_t cake_batch_generated_uint64_t
#define int32_t cake_batch_generated_int32_t
#define int16_t cake_batch_generated_int16_t
#define CakeTensorMap cake_batch_generated_CakeTensorMap
#define CakeTensorMapPack cake_batch_generated_CakeTensorMapPack
#define CUtensorMap cake_batch_generated_CUtensorMap
#include FLASHINFER_CAKE_BATCH_DEEPGEMM_BODY_FILE
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t
#undef CakeTensorMap
#undef CakeTensorMapPack
#undef CUtensorMap

namespace flashinfer::cake_batch_deepgemm_fp8 {

constexpr int32_t kVariantN128K512 = 0;
constexpr int32_t kVariantN512K128 = 1;
constexpr int32_t kVariantN4096K7168 = 2;
constexpr int32_t kVariantN7168K2048 = 3;
constexpr int32_t kTargetSM100a = 1000;
constexpr int32_t kTargetSM103a = 1003;

static_assert(FLASHINFER_CAKE_BATCH_DEEPGEMM_TARGET_KIND == kTargetSM100a ||
                  FLASHINFER_CAKE_BATCH_DEEPGEMM_TARGET_KIND == kTargetSM103a,
              "Cake batch DeepGEMM FP8 requires exact SM100a or SM103a");
static_assert(FLASHINFER_CAKE_BATCH_DEEPGEMM_VARIANT >= kVariantN128K512 &&
                  FLASHINFER_CAKE_BATCH_DEEPGEMM_VARIANT <= kVariantN7168K2048,
              "unknown Cake batch DeepGEMM FP8 variant");

inline void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK(status == cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

inline void CheckTarget(int32_t device_id) {
  int major = 0;
  int minor = 0;
  CheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
            "cudaDeviceGetAttribute(major)");
  CheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
            "cudaDeviceGetAttribute(minor)");
  constexpr int32_t expected_minor =
      FLASHINFER_CAKE_BATCH_DEEPGEMM_TARGET_KIND == kTargetSM103a ? 3 : 0;
  TVM_FFI_ICHECK(major == 10 && minor == expected_minor)
      << "this Cake batch DeepGEMM FP8 module requires exact compute capability 10."
      << expected_minor << ", got " << major << "." << minor;
}

inline bool IsSupportedBatch(int64_t batch) {
  return batch == 1 || batch == 4 || batch == 8 || batch == 64 || batch == 128 || batch == 256;
}

inline bool IsSupportedM(int64_t batch, int64_t m) {
  if (m == 128 || m == 256 || m == 512 || m == 1024) return true;
  return (m == 8192 || m == 16384) && batch * m <= 16384;
}

inline void CheckDescriptor(const TensorView& desc, const TensorView& reference,
                            const char* name) {
  CHECK_CUDA(desc);
  CHECK_DEVICE(reference, desc);
  CHECK_INPUT_TYPE(desc, dl_uint8);
  CHECK_CONTIGUOUS(desc);
  TVM_FFI_ICHECK(desc.ndim() == 1 && desc.numel() == 128)
      << name << " must be one 128-byte device CUtensorMap";
}

void Run(TensorView a, TensorView b, TensorView a_scale, TensorView b_scale, TensorView masked_m,
         TensorView out, TensorView a_desc, TensorView b_desc, TensorView c_desc,
         int64_t expected_m) {
  CHECK_CUDA(a);
  const int32_t device_id = a.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckTarget(device_id);

  CHECK_CUDA(b);
  CHECK_CUDA(a_scale);
  CHECK_CUDA(b_scale);
  CHECK_CUDA(masked_m);
  CHECK_CUDA(out);
  CHECK_DEVICE(a, b);
  CHECK_DEVICE(a, a_scale);
  CHECK_DEVICE(a, b_scale);
  CHECK_DEVICE(a, masked_m);
  CHECK_DEVICE(a, out);
  CHECK_INPUT_TYPE(a, dl_float8_e4m3fn);
  CHECK_INPUT_TYPE(b, dl_float8_e4m3fn);
  CHECK_INPUT_TYPE(a_scale, dl_float32);
  CHECK_INPUT_TYPE(b_scale, dl_float32);
  CHECK_INPUT_TYPE(masked_m, dl_int32);
  CHECK_INPUT_TYPE(out, dl_bfloat16);
  CHECK_CONTIGUOUS(a);
  CHECK_CONTIGUOUS(b);
  CHECK_CONTIGUOUS(a_scale);
  CHECK_CONTIGUOUS(b_scale);
  CHECK_CONTIGUOUS(masked_m);
  CHECK_CONTIGUOUS(out);

  TVM_FFI_ICHECK(a.ndim() == 3 && b.ndim() == 3) << "a and b must be rank-3 tensors";
  const int64_t batch = a.size(0);
  const int64_t m = a.size(1);
  constexpr int64_t n = FLASHINFER_CAKE_BATCH_DEEPGEMM_N;
  constexpr int64_t k = FLASHINFER_CAKE_BATCH_DEEPGEMM_K;
  TVM_FFI_ICHECK(IsSupportedBatch(batch)) << "unsupported Cake batch size " << batch;
  TVM_FFI_ICHECK(IsSupportedM(batch, m)) << "unsupported Cake M shape " << m;
  TVM_FFI_ICHECK(a.size(2) == k) << "a must have shape [B,M," << k << "]";
  TVM_FFI_ICHECK(b.size(0) == batch && b.size(1) == n && b.size(2) == k)
      << "b must have shape [B," << n << "," << k << "]";
  TVM_FFI_ICHECK(a_scale.ndim() == 3 && a_scale.size(0) == batch &&
                 a_scale.size(1) == m && a_scale.size(2) == k / 128)
      << "a_scale must have shape [B,M,K/128]";
  TVM_FFI_ICHECK(b_scale.ndim() == 3 && b_scale.size(0) == batch &&
                 b_scale.size(1) == n / 128 && b_scale.size(2) == k / 128)
      << "b_scale must have shape [B,N/128,K/128]";
  TVM_FFI_ICHECK(masked_m.ndim() == 1 && masked_m.numel() == batch)
      << "masked_m must have shape [B]";
  TVM_FFI_ICHECK(out.ndim() == 3 && out.size(0) == batch && out.size(1) == m &&
                 out.size(2) == n)
      << "out must have shape [B,M,N]";
  TVM_FFI_ICHECK(expected_m >= 0 && expected_m <= m)
      << "expected_m must be in [0,M], got " << expected_m;

  CheckDescriptor(a_desc, a, "a_desc");
  CheckDescriptor(b_desc, a, "b_desc");
#if FLASHINFER_CAKE_BATCH_DEEPGEMM_VARIANT == 1
  CheckDescriptor(c_desc, a, "c_desc");
#else
  (void)c_desc;
#endif

  int32_t num_sms = 0;
  CheckCuda(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id),
            "cudaDeviceGetAttribute(multiProcessorCount)");
  TVM_FFI_ICHECK(num_sms > 0) << "device reports no streaming multiprocessors";
  const auto stream = get_stream(a.device());
  CheckCuda(cudaFuncSetAttribute(FLASHINFER_CAKE_BATCH_DEEPGEMM_KERNEL,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 FLASHINFER_CAKE_BATCH_DEEPGEMM_SMEM_BYTES),
            "cudaFuncSetAttribute(max dynamic shared memory)");

  cudaLaunchConfig_t config{};
  config.blockDim = dim3(192, 1, 1);
  config.dynamicSmemBytes = FLASHINFER_CAKE_BATCH_DEEPGEMM_SMEM_BYTES;
  config.stream = stream;
  cudaLaunchAttribute attrs[2]{};

  auto* a_tma = reinterpret_cast<cake_batch_generated_CakeTensorMap const*>(a_desc.data_ptr());
  auto* b_tma = reinterpret_cast<cake_batch_generated_CakeTensorMap const*>(b_desc.data_ptr());
  auto* a_sf = reinterpret_cast<float*>(a_scale.data_ptr());
  auto* b_sf = reinterpret_cast<float*>(b_scale.data_ptr());
  auto* mask = reinterpret_cast<int*>(masked_m.data_ptr());
  auto* output = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());

#if FLASHINFER_CAKE_BATCH_DEEPGEMM_VARIANT == 0
  const uint32_t max_chunks = static_cast<uint32_t>(batch * ((m + 255) / 256));
  const uint32_t clusters = std::max<uint32_t>(1, std::min<uint32_t>(num_sms / 2, max_chunks));
  config.gridDim = dim3(clusters * 2, 1, 1);
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim = dim3(2, 1, 1);
  attrs[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
  attrs[1].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicySpread;
  config.attrs = attrs;
  config.numAttrs = 2;
  CheckCuda(cudaLaunchKernelEx(&config, FLASHINFER_CAKE_BATCH_DEEPGEMM_KERNEL, a_tma, b_tma,
                               a_sf, b_sf, mask, output, static_cast<int32_t>(batch),
                               static_cast<int32_t>(m)),
            "Cake N128/K512 launch");
#elif FLASHINFER_CAKE_BATCH_DEEPGEMM_VARIANT == 1
  const uint32_t m_tile_bound = static_cast<uint32_t>((expected_m + 127) / 128 + 1);
  config.gridDim = dim3(static_cast<uint32_t>(batch) * m_tile_bound * 4, 1, 1);
  auto* c_tma = reinterpret_cast<cake_batch_generated_CakeTensorMap const*>(c_desc.data_ptr());
  CheckCuda(cudaLaunchKernelEx(&config, FLASHINFER_CAKE_BATCH_DEEPGEMM_KERNEL, a_tma, b_tma,
                               c_tma, a_sf, b_sf, mask, static_cast<uint32_t>(batch),
                               static_cast<uint32_t>(m), 4u),
            "Cake N512/K128 launch");
#elif FLASHINFER_CAKE_BATCH_DEEPGEMM_VARIANT == 2
  config.gridDim = dim3(static_cast<uint32_t>(num_sms), 1, 1);
  CheckCuda(cudaLaunchKernelEx(&config, FLASHINFER_CAKE_BATCH_DEEPGEMM_KERNEL, a_tma, b_tma,
                               reinterpret_cast<int*>(a_scale.data_ptr()),
                               reinterpret_cast<int*>(b_scale.data_ptr()), output, mask,
                               static_cast<uint32_t>(batch), static_cast<uint32_t>(m)),
            "Cake N4096/K7168 launch");
#elif FLASHINFER_CAKE_BATCH_DEEPGEMM_VARIANT == 3
  constexpr uint32_t grid_n = 56;
  constexpr uint32_t k_tiles = 8;
  constexpr uint32_t sf_cols = 16;
  const uint32_t scheduled_pair_blocks = static_cast<uint32_t>((expected_m + 255) / 256 + 1);
  const uint64_t scheduled_tiles =
      static_cast<uint64_t>(batch) * scheduled_pair_blocks * grid_n;
  const uint32_t clusters = std::max<uint32_t>(
      1, std::min<uint32_t>(num_sms / 2, static_cast<uint32_t>(scheduled_tiles)));
  config.gridDim = dim3(clusters * 2, 1, 1);
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim = dim3(2, 1, 1);
  attrs[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
  attrs[1].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicySpread;
  config.attrs = attrs;
  config.numAttrs = 2;
  CheckCuda(cudaLaunchKernelEx(
                &config, FLASHINFER_CAKE_BATCH_DEEPGEMM_KERNEL, a_tma, b_tma,
                reinterpret_cast<int*>(a_scale.data_ptr()),
                reinterpret_cast<int*>(b_scale.data_ptr()), mask, output,
                static_cast<uint32_t>(batch), static_cast<uint32_t>(m), grid_n, k_tiles,
                sf_cols, scheduled_pair_blocks),
            "Cake N7168/K2048 launch");
#endif
}

}  // namespace flashinfer::cake_batch_deepgemm_fp8

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::cake_batch_deepgemm_fp8::Run);
