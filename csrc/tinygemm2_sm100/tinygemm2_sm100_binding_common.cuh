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

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>
#include <cstring>
#include <limits>

#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace tinygemm2_sm100 {

using tvm::ffi::TensorView;

// Fixed tile geometry shared by every generated variant. These mirror the
// TensorRT-LLM tinygemm2 template constants (WARP_TILE_M=16, TILE_N=8,
// TILE_K=64) that the generated schedules were ported from.
constexpr int kTileM = 16;  // output-features tile
constexpr int kTileN = 8;   // batch tile
constexpr int kTileK = 64;  // reduction tile (one TMA box)
constexpr int kThreads = 384;

struct ProblemDims {
  int batch;
  int in_features;
  int out_features;
};

inline void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK(status == cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

inline void CheckSm100Family(int device_id) {
  int major = 0;
  int minor = 0;
  CheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
            "cudaDeviceGetAttribute(compute capability major)");
  CheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
            "cudaDeviceGetAttribute(compute capability minor)");
  TVM_FFI_ICHECK(major == 10 && (minor == 0 || minor == 3))
      << "tinygemm2_sm100 requires an SM100/SM103 (B200/B300 class) device, got sm_" << major
      << minor;
}

inline void CheckBf16(const TensorView& t, const char* name) {
  const DLDataType d = t.dtype();
  TVM_FFI_ICHECK(d.code == kDLBfloat && d.bits == 16 && d.lanes == 1)
      << name << " must be bfloat16, got (code=" << int(d.code) << ", bits=" << int(d.bits)
      << ", lanes=" << int(d.lanes) << ")";
}

inline void CheckCudaBf16Contiguous(const TensorView& t, int ndim, const char* name) {
  TVM_FFI_ICHECK(t.device().device_type == kDLCUDA) << name << " must be a CUDA tensor";
  TVM_FFI_ICHECK(t.ndim() == ndim) << name << " must be " << ndim << "D, got ndim=" << t.ndim();
  TVM_FFI_ICHECK(t.IsContiguous()) << name << " must be contiguous";
  CheckBf16(t, name);
}

// Validate the public `out = input @ weight.T + bias` contract plus the
// coverage guards of the generated kernels (mirroring the Loom host shim):
// in_features must fit one TMA box; out_features must be a positive multiple
// of the kTileM output tile. The batch axis has NO lower guard — the
// activation descriptor deliberately allows an out-of-bounds box on that axis
// and TMA zero-fills rows past the end, so batch 1..7 inputs are valid.
inline ProblemDims CheckInputs(const TensorView& input, const TensorView& weight,
                               const TensorView& bias, const TensorView& out) {
  CheckCudaBf16Contiguous(input, 2, "input");
  CheckCudaBf16Contiguous(weight, 2, "weight");
  CheckCudaBf16Contiguous(bias, 1, "bias");
  CheckCudaBf16Contiguous(out, 2, "out");
  const int device_id = input.device().device_id;
  TVM_FFI_ICHECK(weight.device().device_id == device_id && bias.device().device_id == device_id &&
                 out.device().device_id == device_id)
      << "input/weight/bias/out must live on the same CUDA device";
  CheckSm100Family(device_id);

  const int64_t batch = input.size(0);
  const int64_t in_features = input.size(1);
  const int64_t out_features = weight.size(0);
  TVM_FFI_ICHECK(weight.size(1) == in_features)
      << "weight.shape[1] (" << weight.size(1) << ") must equal input.shape[1] (" << in_features
      << ")";
  TVM_FFI_ICHECK(bias.size(0) == out_features)
      << "bias.shape[0] (" << bias.size(0) << ") must equal weight.shape[0] (" << out_features
      << ")";
  TVM_FFI_ICHECK(out.size(0) == batch && out.size(1) == out_features)
      << "out must have shape (" << batch << ", " << out_features << "), got (" << out.size(0)
      << ", " << out.size(1) << ")";

  TVM_FFI_ICHECK(batch > 0) << "batch must be positive, got " << batch;
  TVM_FFI_ICHECK(in_features >= kTileK)
      << "in_features (" << in_features << ") must be at least " << kTileK << " (one TMA box)";
  TVM_FFI_ICHECK(out_features >= kTileM && out_features % kTileM == 0)
      << "out_features (" << out_features << ") must be a positive multiple of " << kTileM;
  TVM_FFI_ICHECK(batch <= std::numeric_limits<int>::max() &&
                 in_features <= std::numeric_limits<int>::max() &&
                 out_features <= std::numeric_limits<int>::max())
      << "problem dimensions exceed the kernel's i32 scalar range";

  return ProblemDims{static_cast<int>(batch), static_cast<int>(in_features),
                     static_cast<int>(out_features)};
}

// 2D TMA descriptor for the weight matrix — field-for-field the descriptor
// the Loom host shim encodes for 'tmap_wt': box (kTileK, kTileM), 128B
// swizzle, no L2 promotion, no OOB fill. Both boxed axes stay in bounds
// (CheckInputs guarantees in_features >= kTileK and out_features >= kTileM).
inline CUtensorMap EncodeWeightTma(const TensorView& weight) {
  const uint64_t global_dim[2] = {static_cast<uint64_t>(weight.size(1)),
                                  static_cast<uint64_t>(weight.size(0))};
  const uint64_t global_strides[1] = {static_cast<uint64_t>(weight.stride(0)) *
                                      sizeof(__nv_bfloat16)};
  const uint32_t box_dim[2] = {static_cast<uint32_t>(kTileK), static_cast<uint32_t>(kTileM)};
  const uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm;
  const CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, weight.data_ptr(), global_dim, global_strides,
      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(r == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for the weight descriptor: CUresult=" << int(r);
  return tm;
}

// 2D TMA descriptor for the activation matrix — the Loom host shim's
// 'tmap_act' descriptor. The batch axis opts into an out-of-bounds box
// (box kTileN may exceed batch for batch 1..7); TMA zero-fills those rows.
inline CUtensorMap EncodeActivationTma(const TensorView& input) {
  const uint64_t global_dim[2] = {static_cast<uint64_t>(input.size(1)),
                                  static_cast<uint64_t>(input.size(0))};
  const uint64_t global_strides[1] = {static_cast<uint64_t>(input.stride(0)) *
                                      sizeof(__nv_bfloat16)};
  const uint32_t box_dim[2] = {static_cast<uint32_t>(kTileK), static_cast<uint32_t>(kTileN)};
  const uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm;
  const CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, input.data_ptr(), global_dim, global_strides,
      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(r == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for the activation descriptor: CUresult=" << int(r);
  return tm;
}

// Launch one generated variant. TMap is each generated TU's by-value
// __grid_constant__ tensor-map parameter type (LoomTensorMap); it is
// layout-compatible with CUtensorMap. PDL variants launch through
// cudaLaunchKernelEx with programmatic stream serialization, matching the
// in-kernel griddepcontrol pair compiled into those TUs.
template <typename TMap>
inline void LaunchVariant(void (*kernel)(TMap, TMap, __nv_bfloat16*, __nv_bfloat16*, int, int, int),
                          int smem_bytes, bool pdl, const CUtensorMap& weight_map,
                          const CUtensorMap& activation_map, __nv_bfloat16* out,
                          __nv_bfloat16* bias, const ProblemDims& dims, cudaStream_t stream) {
  static_assert(sizeof(TMap) == sizeof(CUtensorMap),
                "generated tensor-map parameter must be layout-compatible with CUtensorMap");
  TMap wt_param, act_param;
  std::memcpy(&wt_param, &weight_map, sizeof(TMap));
  std::memcpy(&act_param, &activation_map, sizeof(TMap));

  CheckCuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes),
            "cudaFuncSetAttribute(tinygemm2_sm100 dynamic smem)");

  const dim3 grid((dims.out_features + kTileM - 1) / kTileM, (dims.batch + kTileN - 1) / kTileN);
  const dim3 block(kThreads);

  if (pdl) {
    cudaLaunchConfig_t config;
    cudaLaunchAttribute attrs[1];
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_bytes;
    config.stream = stream;
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    CheckCuda(cudaLaunchKernelEx(&config, kernel, wt_param, act_param, out, bias, dims.out_features,
                                 dims.batch, dims.in_features),
              "cudaLaunchKernelEx(tinygemm2_sm100)");
  } else {
    kernel<<<grid, block, smem_bytes, stream>>>(wt_param, act_param, out, bias, dims.out_features,
                                                dims.batch, dims.in_features);
    CheckCuda(cudaGetLastError(), "tinygemm2_sm100 kernel launch");
  }
}

}  // namespace tinygemm2_sm100
}  // namespace flashinfer
