/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#pragma once

#ifndef CAKE_GROUPED_MXFP8_BODY_FILE
#error "CAKE_GROUPED_MXFP8_BODY_FILE must name one generated device body"
#endif
#ifndef CAKE_GROUPED_MXFP8_KERNEL
#error "CAKE_GROUPED_MXFP8_KERNEL must name the generated kernel symbol"
#endif
#ifndef CAKE_GROUPED_MXFP8_INPUT_DLTYPE
#error "CAKE_GROUPED_MXFP8_INPUT_DLTYPE must name the input DLDataType"
#endif
#ifndef FLASHINFER_CAKE_GROUPED_MXFP8_TARGET_MINOR
#error "FLASHINFER_CAKE_GROUPED_MXFP8_TARGET_MINOR must be 0 or 3"
#endif

#include <cuda.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <limits>

#include "tvm_ffi_utils.h"

// Generated programs carry private fixed-width aliases. Rename them at the
// include boundary so they cannot collide with CUDA or TVM FFI headers.
#define uint8_t cake_grouped_mxfp8_generated_uint8_t
#define uint16_t cake_grouped_mxfp8_generated_uint16_t
#define uint32_t cake_grouped_mxfp8_generated_uint32_t
#define uint64_t cake_grouped_mxfp8_generated_uint64_t
#define int16_t cake_grouped_mxfp8_generated_int16_t
#define int32_t cake_grouped_mxfp8_generated_int32_t
#include CAKE_GROUPED_MXFP8_BODY_FILE
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int16_t
#undef int32_t

#include "cake_grouped_mxfp8_quantize_launch.cuh"

namespace flashinfer::cake_grouped_mxfp8_quantize {

inline void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK(status == cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

inline int64_t RoundUp(int64_t value, int64_t alignment) {
  TVM_FFI_ICHECK(value >= 0 && value <= std::numeric_limits<int64_t>::max() - alignment + 1)
      << "shape is too large to pad safely";
  return (value + alignment - 1) / alignment * alignment;
}

inline uint64_t CheckedProduct(uint64_t lhs, uint64_t rhs, const char* name) {
  TVM_FFI_ICHECK(rhs == 0 || lhs <= std::numeric_limits<uint64_t>::max() / rhs)
      << name << " overflows uint64";
  return lhs * rhs;
}

inline void CheckTarget(int32_t device_id) {
  static_assert(FLASHINFER_CAKE_GROUPED_MXFP8_TARGET_MINOR == 0 ||
                    FLASHINFER_CAKE_GROUPED_MXFP8_TARGET_MINOR == 3,
                "Cake grouped MXFP8 target minor must be 0 or 3");
  int major = 0;
  int minor = 0;
  CheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
            "cudaDeviceGetAttribute(major)");
  CheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
            "cudaDeviceGetAttribute(minor)");
  TVM_FFI_ICHECK(major == 10 && minor == FLASHINFER_CAKE_GROUPED_MXFP8_TARGET_MINOR)
      << "this Cake grouped MXFP8 module requires exact compute capability 10."
      << FLASHINFER_CAKE_GROUPED_MXFP8_TARGET_MINOR << ", got " << major << "." << minor;
}

void Run(TensorView input, TensorView mask, TensorView quantized, TensorView scales,
         int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0) << "cuda_stream must be a non-negative stream handle";
  CHECK_CUDA(input);
  CHECK_CUDA(mask);
  CHECK_CUDA(quantized);
  CHECK_CUDA(scales);
  CHECK_DEVICE(input, mask);
  CHECK_DEVICE(input, quantized);
  CHECK_DEVICE(input, scales);
  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  CheckTarget(input.device().device_id);

  CHECK_INPUT_TYPE(input, CAKE_GROUPED_MXFP8_INPUT_DLTYPE);
  CHECK_INPUT_TYPE(mask, dl_int32);
  CHECK_INPUT_TYPE(quantized, dl_float8_e4m3fn);
  CHECK_INPUT_TYPE(scales, dl_uint8);
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(mask);
  CHECK_CONTIGUOUS(quantized);
  CHECK_CONTIGUOUS(scales);

  TVM_FFI_ICHECK(input.ndim() == 3) << "input must have shape [B,M,K]";
  const int64_t batch64 = input.size(0);
  const int64_t rows64 = input.size(1);
  const int64_t columns64 = input.size(2);
  TVM_FFI_ICHECK(batch64 >= 0 && rows64 >= 0 && columns64 > 0 && columns64 % kQuantBlock == 0)
      << "input must have non-negative B/M and positive K divisible by 32";
  TVM_FFI_ICHECK(mask.ndim() == 1 && mask.size(0) == batch64)
      << "mask must be contiguous int32 [B]";

  const int64_t padded_rows64 = RoundUp(rows64, kScaleTileM);
  const int64_t padded_columns64 = RoundUp(columns64, kScaleTileK);
  TVM_FFI_ICHECK(quantized.ndim() == 3 && quantized.size(0) == batch64 &&
                 quantized.size(1) == rows64 && quantized.size(2) == padded_columns64)
      << "quantized must have physical shape [B,M,padded_K]";
  TVM_FFI_ICHECK(scales.ndim() == 3 && scales.size(0) == batch64 &&
                 scales.size(1) == padded_rows64 && scales.size(2) == padded_columns64 / kQuantBlock)
      << "scales must have physical shape [B,padded_M,padded_K/32]";

  TVM_FFI_ICHECK(batch64 <= INT_MAX && rows64 <= INT_MAX && columns64 <= INT_MAX &&
                 padded_columns64 <= INT_MAX)
      << "generated grouped MXFP8 shape parameters must fit int32";
  int max_grid_y = 0;
  int max_grid_z = 0;
  CheckCuda(cudaDeviceGetAttribute(&max_grid_y, cudaDevAttrMaxGridDimY, input.device().device_id),
            "cudaDeviceGetAttribute(maxGridDimY)");
  CheckCuda(cudaDeviceGetAttribute(&max_grid_z, cudaDevAttrMaxGridDimZ, input.device().device_id),
            "cudaDeviceGetAttribute(maxGridDimZ)");
  TVM_FFI_ICHECK(rows64 <= max_grid_y && batch64 <= max_grid_z)
      << "B/M exceed the generated row2d CUDA grid limits";

  const int32_t batch = static_cast<int32_t>(batch64);
  const int32_t rows = static_cast<int32_t>(rows64);
  const int32_t columns = static_cast<int32_t>(columns64);
  const int32_t padded_columns = static_cast<int32_t>(padded_columns64);
  const int32_t padded_m_tiles = static_cast<int32_t>(padded_rows64 / kScaleTileM);
  const int32_t padded_k_tiles = static_cast<int32_t>(padded_columns64 / kScaleTileK);
  const int32_t blocks_per_row = padded_columns / kQuantBlock;
  uint64_t total_tasks = CheckedProduct(static_cast<uint64_t>(batch64),
                                        static_cast<uint64_t>(rows64), "B*M");
  total_tasks = CheckedProduct(total_tasks, static_cast<uint64_t>(blocks_per_row),
                               "B*M*blocks_per_row");

  CheckCuda(Launch(input.data_ptr(), reinterpret_cast<const int32_t*>(mask.data_ptr()),
                   quantized.data_ptr(), reinterpret_cast<uint8_t*>(scales.data_ptr()), batch,
                   rows, columns, padded_columns, padded_m_tiles, padded_k_tiles,
                   blocks_per_row, total_tasks, reinterpret_cast<cudaStream_t>(cuda_stream)),
            "Cake grouped MXFP8 launch");
}

}  // namespace flashinfer::cake_grouped_mxfp8_quantize

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::cake_grouped_mxfp8_quantize::Run);
