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

#include <initializer_list>

#include "flashkda_binding_common.cuh"

// The frozen bundle is standalone CUDA and intentionally declares its own
// fixed-width and tensor-map carrier types.  Keep those declarations isolated
// from CUDA's host headers while retaining one translation unit for all eleven
// kernels and the TVM FFI binding.
#define uint8_t flashkda_backward_generated_uint8_t
#define uint16_t flashkda_backward_generated_uint16_t
#define uint32_t flashkda_backward_generated_uint32_t
#define uint64_t flashkda_backward_generated_uint64_t
#define int32_t flashkda_backward_generated_int32_t
#define int16_t flashkda_backward_generated_int16_t
#define FlashKDATensorMap flashkda_backward_generated_FlashKDATensorMap
#define FlashKDATensorMapPack flashkda_backward_generated_FlashKDATensorMapPack
#define CUtensorMap flashkda_backward_generated_CUtensorMap
#include "flashkda_backward.cu"
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
namespace flash_kda_backward {

using flash_kda::CheckCuda;
using flash_kda::CheckCudaTensor;
using flash_kda::CheckDtype;
using flash_kda::CheckDynamicSmemCapacity;
using flash_kda::CheckFlashKDATarget;
using flash_kda::CheckNoOverlap;
using flash_kda::EncodeTmaPointers;
using flash_kda::TmaPointers;

constexpr int64_t kHeadDim = 128;
constexpr int64_t kChunkTokens = 32;
constexpr int64_t kLowGateTokensPerSplit = 128;
constexpr int32_t kTapeThreads = 1024;
constexpr int32_t kTapeSmemBytes = 227456;
constexpr int32_t kFallbackThreads = 256;
constexpr int32_t kFallbackSmemBytes = 128;
constexpr int32_t kBoundaryM64Threads = 288;
constexpr int32_t kBoundaryM64SmemBytes = 37888;
constexpr int32_t kBoundaryM128Threads = 512;
constexpr int32_t kBoundaryM128SmemBytes = 74752;
constexpr int32_t kLocalThreads = 384;
constexpr int32_t kLocalSmemBytes = 155136;
constexpr int32_t kMapFinalizeThreads = 128;

struct NamedTensor {
  const TensorView* tensor;
  const char* name;
};

inline void CheckTensor(const TensorView& tensor, const char* name, int32_t device_id,
                        DLDataType dtype) {
  CheckCudaTensor(tensor, name, device_id);
  CheckDtype(tensor, name, dtype);
}

inline void CheckShape(const TensorView& actual, const TensorView& expected, const char* name,
                       const char* expected_name) {
  TVM_FFI_ICHECK(actual.ndim() == expected.ndim())
      << name << " must have the same rank as " << expected_name;
  for (int32_t dim = 0; dim < actual.ndim(); ++dim) {
    TVM_FFI_ICHECK(actual.size(dim) == expected.size(dim))
        << name << " must have the same shape as " << expected_name;
  }
}

inline void Check1D(const TensorView& tensor, const char* name, int64_t elements) {
  TVM_FFI_ICHECK(tensor.ndim() == 1 && tensor.numel() == elements)
      << name << " must be one-dimensional with " << elements << " elements";
}

inline void Check2D(const TensorView& tensor, const char* name, int64_t dim0, int64_t dim1) {
  TVM_FFI_ICHECK(tensor.ndim() == 2 && tensor.size(0) == dim0 && tensor.size(1) == dim1)
      << name << " must have shape [" << dim0 << ", " << dim1 << "]";
}

inline void Check3D(const TensorView& tensor, const char* name, int64_t dim0, int64_t dim1,
                    int64_t dim2) {
  TVM_FFI_ICHECK(tensor.ndim() == 3 && tensor.size(0) == dim0 && tensor.size(1) == dim1 &&
                 tensor.size(2) == dim2)
      << name << " must have shape [" << dim0 << ", " << dim1 << ", " << dim2 << "]";
}

inline void Check4D(const TensorView& tensor, const char* name, int64_t dim0, int64_t dim1,
                    int64_t dim2, int64_t dim3) {
  TVM_FFI_ICHECK(tensor.ndim() == 4 && tensor.size(0) == dim0 && tensor.size(1) == dim1 &&
                 tensor.size(2) == dim2 && tensor.size(3) == dim3)
      << name << " must have shape [" << dim0 << ", " << dim1 << ", " << dim2 << ", " << dim3
      << "]";
}

inline void CheckPairwiseDisjoint(std::initializer_list<NamedTensor> tensors) {
  for (auto lhs = tensors.begin(); lhs != tensors.end(); ++lhs) {
    for (auto rhs = lhs + 1; rhs != tensors.end(); ++rhs) {
      TVM_FFI_ICHECK(lhs->tensor->data_ptr() != rhs->tensor->data_ptr())
          << lhs->name << " and " << rhs->name << " must not alias";
      CheckNoOverlap(*lhs->tensor, lhs->name, *rhs->tensor, rhs->name);
    }
  }
}

inline void CheckDisjointFrom(const TensorView& tensor, const char* name,
                              std::initializer_list<NamedTensor> others) {
  for (const NamedTensor& other : others) {
    TVM_FFI_ICHECK(tensor.data_ptr() != other.tensor->data_ptr())
        << name << " and " << other.name << " must not alias";
    CheckNoOverlap(tensor, name, *other.tensor, other.name);
  }
}

enum class CanonicalShape : int32_t {
  kInvalid = -1,
  kFixedT17H1 = 0,
  kPackedT115N3H4 = 1,
  kFixedT17H16 = 2,
  kFixedT1024H4 = 3,
  kFixedT4096H32 = 4,
  kFixedT8192H96 = 5,
  kPackedT8192N6H96 = 6,
  kPackedT8192N8H96 = 7,
};

inline CanonicalShape ResolveCanonicalShape(const TensorView& q, int64_t num_sequences,
                                            int64_t num_heads) {
  if (q.ndim() != 4 || q.size(0) != 1 || q.size(2) != num_heads || q.size(3) != kHeadDim) {
    return CanonicalShape::kInvalid;
  }
  const int64_t tokens = q.size(1);
  if (tokens == 17 && num_sequences == 1 && num_heads == 1) {
    return CanonicalShape::kFixedT17H1;
  }
  if (tokens == 115 && num_sequences == 3 && num_heads == 4) {
    return CanonicalShape::kPackedT115N3H4;
  }
  if (tokens == 17 && num_sequences == 1 && num_heads == 16) {
    return CanonicalShape::kFixedT17H16;
  }
  if (tokens == 1024 && num_sequences == 1 && num_heads == 4) {
    return CanonicalShape::kFixedT1024H4;
  }
  if (tokens == 4096 && num_sequences == 1 && num_heads == 32) {
    return CanonicalShape::kFixedT4096H32;
  }
  if (tokens == 8192 && num_sequences == 1 && num_heads == 96) {
    return CanonicalShape::kFixedT8192H96;
  }
  if (tokens == 8192 && num_sequences == 6 && num_heads == 96) {
    return CanonicalShape::kPackedT8192N6H96;
  }
  if (tokens == 8192 && num_sequences == 8 && num_heads == 96) {
    return CanonicalShape::kPackedT8192N8H96;
  }
  return CanonicalShape::kInvalid;
}

__global__ void ValidateCanonicalCuSeqlens(const long long* cu_seqlens, CanonicalShape shape) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  bool valid = false;
  switch (shape) {
    case CanonicalShape::kFixedT17H1:
    case CanonicalShape::kFixedT17H16:
      valid = cu_seqlens[0] == 0 && cu_seqlens[1] == 17;
      break;
    case CanonicalShape::kPackedT115N3H4:
      valid =
          cu_seqlens[0] == 0 && cu_seqlens[1] == 17 && cu_seqlens[2] == 50 && cu_seqlens[3] == 115;
      break;
    case CanonicalShape::kFixedT1024H4:
      valid = cu_seqlens[0] == 0 && cu_seqlens[1] == 1024;
      break;
    case CanonicalShape::kFixedT4096H32:
      valid = cu_seqlens[0] == 0 && cu_seqlens[1] == 4096;
      break;
    case CanonicalShape::kFixedT8192H96:
      valid = cu_seqlens[0] == 0 && cu_seqlens[1] == 8192;
      break;
    case CanonicalShape::kPackedT8192N6H96:
      valid = cu_seqlens[0] == 0 && cu_seqlens[1] == 1300 && cu_seqlens[2] == 1847 &&
              cu_seqlens[3] == 3895 && cu_seqlens[4] == 4858 && cu_seqlens[5] == 5129 &&
              cu_seqlens[6] == 8192;
      break;
    case CanonicalShape::kPackedT8192N8H96:
      valid = true;
#pragma unroll
      for (int32_t index = 0; index <= 8; ++index) {
        valid = valid && cu_seqlens[index] == static_cast<long long>(index) * 1024;
      }
      break;
    case CanonicalShape::kInvalid:
      break;
  }
  if (!valid) {
    asm volatile("trap;");
  }
}

inline void LaunchCuSeqlensValidation(const TensorView& cu_seqlens, CanonicalShape shape,
                                      cudaStream_t stream) {
  ValidateCanonicalCuSeqlens<<<1, 1, 0, stream>>>(
      reinterpret_cast<const long long*>(cu_seqlens.data_ptr()), shape);
  CheckCuda(cudaGetLastError(), "ValidateCanonicalCuSeqlens launch");
}

inline int64_t CheckCommonContractInputs(const TensorView& q, const TensorView& k,
                                         const TensorView& v, const TensorView& g,
                                         const TensorView& beta, const TensorView& A_log,
                                         const TensorView& dt_bias, const TensorView& initial_state,
                                         const TensorView& do_, const TensorView& dfinal_state,
                                         const TensorView& cu_seqlens, int64_t num_sequences,
                                         int64_t num_heads, double scale, double lower_bound,
                                         bool high_path) {
  TVM_FFI_ICHECK(num_sequences > 0 && num_sequences <= std::numeric_limits<int32_t>::max())
      << "num_sequences must be in the positive int32 range";
  const bool supported_heads = high_path ? (num_heads == 16 || num_heads == 32 || num_heads == 96)
                                         : (num_heads == 1 || num_heads == 4);
  TVM_FFI_ICHECK(supported_heads) << (high_path ? "run_high requires H in {16, 32, 96}"
                                                : "run_low requires H in {1, 4}");
  constexpr double kRequiredScale = 0.08838834764831843;
  TVM_FFI_ICHECK(std::isfinite(scale) && scale == kRequiredScale)
      << "FlashKDA backward fixes scale=1/sqrt(128)";
  TVM_FFI_ICHECK(std::isfinite(lower_bound) && lower_bound == -5.0)
      << "FlashKDA backward fixes lower_bound=-5.0";

  const int32_t device_id = q.device().device_id;
  for (const auto& named :
       {NamedTensor{&q, "q"}, NamedTensor{&k, "k"}, NamedTensor{&v, "v"}, NamedTensor{&g, "g"},
        NamedTensor{&beta, "beta"}, NamedTensor{&do_, "do"}}) {
    CheckTensor(*named.tensor, named.name, device_id, dl_bfloat16);
  }
  for (const auto& named :
       {NamedTensor{&A_log, "A_log"}, NamedTensor{&dt_bias, "dt_bias"},
        NamedTensor{&initial_state, "initial_state"}, NamedTensor{&dfinal_state, "dfinal_state"}}) {
    CheckTensor(*named.tensor, named.name, device_id, dl_float32);
  }
  CheckTensor(cu_seqlens, "cu_seqlens", device_id, dl_int64);

  TVM_FFI_ICHECK(ResolveCanonicalShape(q, num_sequences, num_heads) != CanonicalShape::kInvalid)
      << "q is outside the eight canonical FlashKDA backward shapes";
  CheckShape(k, q, "k", "q");
  CheckShape(v, q, "v", "q");
  CheckShape(g, q, "g", "q");
  CheckShape(do_, q, "do", "q");
  TVM_FFI_ICHECK(beta.ndim() == q.ndim() - 1) << "beta must remove only q's trailing K dimension";
  for (int32_t dim = 0; dim < beta.ndim(); ++dim) {
    TVM_FFI_ICHECK(beta.size(dim) == q.size(dim))
        << "beta must match q's leading [batch, tokens, heads] shape";
  }
  TVM_FFI_ICHECK(A_log.ndim() == 1 && A_log.numel() == num_heads) << "A_log must have shape [H]";
  TVM_FFI_ICHECK(dt_bias.ndim() == 2 && dt_bias.size(0) == num_heads && dt_bias.size(1) == kHeadDim)
      << "dt_bias must have shape [H, 128]";
  Check4D(initial_state, "initial_state", num_sequences, num_heads, kHeadDim, kHeadDim);
  CheckShape(dfinal_state, initial_state, "dfinal_state", "initial_state");
  Check1D(cu_seqlens, "cu_seqlens", num_sequences + 1);

  const int64_t total_tokens = q.numel() / (num_heads * kHeadDim);
  TVM_FFI_ICHECK(total_tokens > 0 && total_tokens <= std::numeric_limits<int32_t>::max())
      << "total_tokens must be in the positive int32 range";
  return total_tokens;
}

inline void CheckExactBlackwellTarget(int32_t device_id) {
  CheckFlashKDATarget(device_id);
  int major = 0;
  int minor = 0;
  CheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
            "cudaDeviceGetAttribute(major)");
  CheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
            "cudaDeviceGetAttribute(minor)");
  TVM_FFI_ICHECK(major == 10 && minor == FLASHINFER_FLASH_KDA_TARGET_MINOR)
      << "FlashKDA backward module targets exact compute capability 10."
      << FLASHINFER_FLASH_KDA_TARGET_MINOR << ", got " << major << "." << minor;
}

inline uint32_t CheckedGridX(int64_t grid_x, const char* name) {
  TVM_FFI_ICHECK(grid_x > 0 && grid_x <= std::numeric_limits<uint32_t>::max())
      << name << " grid.x is out of range: " << grid_x;
  return static_cast<uint32_t>(grid_x);
}

template <typename Kernel>
inline void ConfigureDynamicSmem(Kernel kernel, int32_t smem_bytes, int32_t device_id,
                                 const char* name) {
  CheckDynamicSmemCapacity(device_id, smem_bytes);
  CheckCuda(cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes),
            name);
}

void RunLow(TensorView q, TensorView k, TensorView v, TensorView g, TensorView beta,
            TensorView A_log, TensorView dt_bias, TensorView initial_state, TensorView do_,
            TensorView dfinal_state, TensorView cu_seqlens, TensorView q_norm, TensorView k_norm,
            TensorView decay, TensorView beta_active, TensorView checkpoint,
            TensorView dq_normalized, TensorView dk_normalized, TensorView dlog_decay,
            TensorView dbeta_active, TensorView dq, TensorView dk, TensorView dv, TensorView dg,
            TensorView dbeta, TensorView dA_log, TensorView ddt_bias, TensorView dinitial_state,
            int64_t num_sequences, int64_t num_heads, double scale, double lower_bound,
            int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0) << "cuda_stream must be a non-negative stream handle";
  TVM_FFI_ICHECK(q.device().device_type == kDLCUDA) << "q must be a CUDA tensor";
  const int32_t device_id = q.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckExactBlackwellTarget(device_id);
  const int64_t total_tokens =
      CheckCommonContractInputs(q, k, v, g, beta, A_log, dt_bias, initial_state, do_, dfinal_state,
                                cu_seqlens, num_sequences, num_heads, scale, lower_bound, false);

  for (const auto& named :
       {NamedTensor{&q_norm, "q_norm"}, NamedTensor{&k_norm, "k_norm"},
        NamedTensor{&decay, "decay"}, NamedTensor{&beta_active, "beta_active"},
        NamedTensor{&checkpoint, "checkpoint"}, NamedTensor{&dq_normalized, "dq_normalized"},
        NamedTensor{&dk_normalized, "dk_normalized"}, NamedTensor{&dlog_decay, "dlog_decay"},
        NamedTensor{&dbeta_active, "dbeta_active"}, NamedTensor{&dA_log, "dA_log"},
        NamedTensor{&ddt_bias, "ddt_bias"}, NamedTensor{&dinitial_state, "dinitial_state"}}) {
    CheckTensor(*named.tensor, named.name, device_id, dl_float32);
  }
  for (const auto& named : {NamedTensor{&dq, "dq"}, NamedTensor{&dk, "dk"}, NamedTensor{&dv, "dv"},
                            NamedTensor{&dg, "dg"}, NamedTensor{&dbeta, "dbeta"}}) {
    CheckTensor(*named.tensor, named.name, device_id, dl_bfloat16);
  }

  for (const auto& named :
       {NamedTensor{&q_norm, "q_norm"}, NamedTensor{&k_norm, "k_norm"},
        NamedTensor{&decay, "decay"}, NamedTensor{&dq_normalized, "dq_normalized"},
        NamedTensor{&dk_normalized, "dk_normalized"}, NamedTensor{&dlog_decay, "dlog_decay"}}) {
    Check3D(*named.tensor, named.name, total_tokens, num_heads, kHeadDim);
  }
  Check2D(beta_active, "beta_active", total_tokens, num_heads);
  Check2D(dbeta_active, "dbeta_active", total_tokens, num_heads);
  Check4D(checkpoint, "checkpoint", total_tokens, num_heads, kHeadDim, kHeadDim);
  for (const auto& named : {NamedTensor{&dq, "dq"}, NamedTensor{&dk, "dk"}, NamedTensor{&dv, "dv"},
                            NamedTensor{&dg, "dg"}}) {
    CheckShape(*named.tensor, q, named.name, "q");
  }
  CheckShape(dbeta, beta, "dbeta", "beta");
  CheckShape(dA_log, A_log, "dA_log", "A_log");
  CheckShape(ddt_bias, dt_bias, "ddt_bias", "dt_bias");
  CheckShape(dinitial_state, initial_state, "dinitial_state", "initial_state");

  CheckPairwiseDisjoint({
      {&q, "q"},
      {&k, "k"},
      {&v, "v"},
      {&g, "g"},
      {&beta, "beta"},
      {&A_log, "A_log"},
      {&dt_bias, "dt_bias"},
      {&initial_state, "initial_state"},
      {&do_, "do"},
      {&dfinal_state, "dfinal_state"},
      {&cu_seqlens, "cu_seqlens"},
      {&q_norm, "q_norm"},
      {&k_norm, "k_norm"},
      {&decay, "decay"},
      {&beta_active, "beta_active"},
      {&checkpoint, "checkpoint"},
      {&dq_normalized, "dq_normalized"},
      {&dk_normalized, "dk_normalized"},
      {&dlog_decay, "dlog_decay"},
      {&dbeta_active, "dbeta_active"},
      {&dq, "dq"},
      {&dk, "dk"},
      {&dv, "dv"},
      {&dg, "dg"},
      {&dbeta, "dbeta"},
      {&dA_log, "dA_log"},
      {&ddt_bias, "ddt_bias"},
      {&dinitial_state, "dinitial_state"},
  });

  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  LaunchCuSeqlensValidation(cu_seqlens, ResolveCanonicalShape(q, num_sequences, num_heads), stream);
  for (const NamedTensor& named :
       {NamedTensor{&dq_normalized, "dq_normalized"}, NamedTensor{&dk_normalized, "dk_normalized"},
        NamedTensor{&dlog_decay, "dlog_decay"}, NamedTensor{&dbeta_active, "dbeta_active"},
        NamedTensor{&dA_log, "dA_log"}, NamedTensor{&ddt_bias, "ddt_bias"}}) {
    CheckCuda(
        cudaMemsetAsync(named.tensor->data_ptr(), 0, named.tensor->numel() * sizeof(float), stream),
        named.name);
  }

  const dim3 preprocess_grid(CheckedGridX(total_tokens, "preprocess"),
                             CheckedGridX(num_heads, "preprocess"), 1);
  kernel_flashkda_backward_preprocess<<<preprocess_grid, 32, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(k.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(g.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(beta.data_ptr()), reinterpret_cast<float*>(A_log.data_ptr()),
      reinterpret_cast<float*>(dt_bias.data_ptr()), reinterpret_cast<float*>(q_norm.data_ptr()),
      reinterpret_cast<float*>(k_norm.data_ptr()), reinterpret_cast<float*>(decay.data_ptr()),
      reinterpret_cast<float*>(beta_active.data_ptr()), static_cast<int32_t>(total_tokens),
      static_cast<int32_t>(num_heads), static_cast<float>(lower_bound));
  CheckCuda(cudaGetLastError(), "kernel_flashkda_backward_preprocess launch");

  const int64_t row_grid_x = num_sequences * num_heads * kHeadDim;
  kernel_flashkda_backward_checkpoint<<<CheckedGridX(row_grid_x, "checkpoint"), 32, 0, stream>>>(
      reinterpret_cast<float*>(k_norm.data_ptr()), reinterpret_cast<float*>(decay.data_ptr()),
      reinterpret_cast<float*>(beta_active.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(v.data_ptr()),
      reinterpret_cast<float*>(initial_state.data_ptr()),
      reinterpret_cast<long long*>(cu_seqlens.data_ptr()),
      reinterpret_cast<float*>(checkpoint.data_ptr()), static_cast<int32_t>(num_sequences),
      static_cast<int32_t>(num_heads));
  CheckCuda(cudaGetLastError(), "kernel_flashkda_backward_checkpoint launch");

  kernel_flashkda_backward_reverse_rows<<<CheckedGridX(row_grid_x, "reverse_rows"), 32, 0,
                                          stream>>>(
      reinterpret_cast<float*>(q_norm.data_ptr()), reinterpret_cast<float*>(k_norm.data_ptr()),
      reinterpret_cast<float*>(decay.data_ptr()), reinterpret_cast<float*>(beta_active.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(v.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(do_.data_ptr()),
      reinterpret_cast<float*>(initial_state.data_ptr()),
      reinterpret_cast<float*>(dfinal_state.data_ptr()),
      reinterpret_cast<long long*>(cu_seqlens.data_ptr()),
      reinterpret_cast<float*>(checkpoint.data_ptr()),
      reinterpret_cast<float*>(dq_normalized.data_ptr()),
      reinterpret_cast<float*>(dk_normalized.data_ptr()),
      reinterpret_cast<float*>(dlog_decay.data_ptr()),
      reinterpret_cast<float*>(dbeta_active.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(dv.data_ptr()),
      reinterpret_cast<float*>(dinitial_state.data_ptr()), static_cast<int32_t>(num_sequences),
      static_cast<int32_t>(num_heads), static_cast<float>(scale));
  CheckCuda(cudaGetLastError(), "kernel_flashkda_backward_reverse_rows launch");

  kernel_flashkda_backward_finalize_tokens<<<preprocess_grid, 32, 0, stream>>>(
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
      reinterpret_cast<__nv_bfloat16*>(dbeta.data_ptr()), static_cast<int32_t>(num_heads),
      static_cast<float>(lower_bound));
  CheckCuda(cudaGetLastError(), "kernel_flashkda_backward_finalize_tokens launch");

  ConfigureDynamicSmem(kernel_flashkda_backward_gate_reduce_split, 128, device_id,
                       "cudaFuncSetAttribute(gate_reduce_split)");
  const int64_t gate_splits = (total_tokens + kLowGateTokensPerSplit - 1) / kLowGateTokensPerSplit;
  const dim3 gate_grid(CheckedGridX(num_heads, "gate_reduce_split"),
                       CheckedGridX(gate_splits, "gate_reduce_split"), 1);
  kernel_flashkda_backward_gate_reduce_split<<<gate_grid, 128, 128, stream>>>(
      reinterpret_cast<float*>(dlog_decay.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(g.data_ptr()), reinterpret_cast<float*>(dt_bias.data_ptr()),
      reinterpret_cast<float*>(dA_log.data_ptr()), reinterpret_cast<float*>(ddt_bias.data_ptr()),
      static_cast<int32_t>(total_tokens), static_cast<int32_t>(num_heads),
      static_cast<int32_t>(kLowGateTokensPerSplit));
  CheckCuda(cudaGetLastError(), "kernel_flashkda_backward_gate_reduce_split launch");
}

void RunHigh(TensorView q, TensorView k, TensorView v, TensorView g, TensorView beta,
             TensorView beta_tma, TensorView A_log, TensorView dt_bias, TensorView initial_state,
             TensorView do_, TensorView dfinal_state, TensorView cu_seqlens, TensorView seq_order,
             TensorView cu_chunk_offsets, TensorView consumer_chunk_order,
             TensorView chunk_sequence, TensorView chunk_index, TensorView chunk_pair_start,
             TensorView descriptor_storage, TensorView forward_out, TensorView forward_final,
             TensorView chunk_state, TensorView state_checkpoint_needed, TensorView tape_qd,
             TensorView tape_kd, TensorView tape_kr, TensorView tape_j,
             TensorView tape_restore_factor, TensorView tape_e, TensorView tape_x,
             TensorView tape_r, TensorView norm_inv, TensorView decay, TensorView beta_active,
             TensorView zero_workspace, TensorView chunk_dh, TensorView chunk_dr,
             TensorView chunk_dx, TensorView grad_qd, TensorView grad_kd, TensorView grad_ki,
             TensorView dlog_decay, TensorView dbeta_active, TensorView dq, TensorView dk,
             TensorView dv, TensorView dg, TensorView dbeta, TensorView dA_log, TensorView ddt_bias,
             TensorView dinitial_state, int64_t prepare_descriptors, int64_t num_sequences,
             int64_t num_heads, double scale, double lower_bound, int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0) << "cuda_stream must be a non-negative stream handle";
  TVM_FFI_ICHECK(prepare_descriptors == 0 || prepare_descriptors == 1)
      << "prepare_descriptors must be 0 or 1";
  TVM_FFI_ICHECK(q.device().device_type == kDLCUDA) << "q must be a CUDA tensor";
  const int32_t device_id = q.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckExactBlackwellTarget(device_id);
  const int64_t total_tokens =
      CheckCommonContractInputs(q, k, v, g, beta, A_log, dt_bias, initial_state, do_, dfinal_state,
                                cu_seqlens, num_sequences, num_heads, scale, lower_bound, true);
  const int64_t total_chunks = chunk_sequence.numel();
  const int64_t total_pairs = chunk_pair_start.numel();
  TVM_FFI_ICHECK(total_chunks > 0 && total_chunks <= std::numeric_limits<int32_t>::max())
      << "total_chunks must be in the positive int32 range";
  TVM_FFI_ICHECK(total_pairs > 0 && total_pairs <= std::numeric_limits<int32_t>::max())
      << "total_pairs must be in the positive int32 range";

  for (const auto& named : {NamedTensor{&beta_tma, "beta_tma"},
                            NamedTensor{&forward_out, "forward_out"},
                            NamedTensor{&forward_final, "forward_final"},
                            NamedTensor{&chunk_state, "chunk_state"},
                            NamedTensor{&tape_qd, "tape_qd"},
                            NamedTensor{&tape_kd, "tape_kd"},
                            NamedTensor{&tape_kr, "tape_kr"},
                            NamedTensor{&tape_j, "tape_j"},
                            NamedTensor{&tape_e, "tape_e"},
                            NamedTensor{&tape_x, "tape_x"},
                            NamedTensor{&tape_r, "tape_r"},
                            NamedTensor{&decay, "decay"},
                            NamedTensor{&chunk_dh, "chunk_dh"},
                            NamedTensor{&chunk_dr, "chunk_dr"},
                            NamedTensor{&chunk_dx, "chunk_dx"},
                            NamedTensor{&grad_qd, "grad_qd"},
                            NamedTensor{&grad_kd, "grad_kd"},
                            NamedTensor{&grad_ki, "grad_ki"},
                            NamedTensor{&dq, "dq"},
                            NamedTensor{&dk, "dk"},
                            NamedTensor{&dv, "dv"},
                            NamedTensor{&dg, "dg"},
                            NamedTensor{&dbeta, "dbeta"}}) {
    CheckTensor(*named.tensor, named.name, device_id, dl_bfloat16);
  }
  for (const auto& named :
       {NamedTensor{&tape_restore_factor, "tape_restore_factor"},
        NamedTensor{&norm_inv, "norm_inv"}, NamedTensor{&beta_active, "beta_active"},
        NamedTensor{&dlog_decay, "dlog_decay"}, NamedTensor{&dbeta_active, "dbeta_active"},
        NamedTensor{&dA_log, "dA_log"}, NamedTensor{&ddt_bias, "ddt_bias"},
        NamedTensor{&dinitial_state, "dinitial_state"}}) {
    CheckTensor(*named.tensor, named.name, device_id, dl_float32);
  }
  for (const auto& named :
       {NamedTensor{&seq_order, "seq_order"},
        NamedTensor{&consumer_chunk_order, "consumer_chunk_order"},
        NamedTensor{&chunk_sequence, "chunk_sequence"}, NamedTensor{&chunk_index, "chunk_index"},
        NamedTensor{&chunk_pair_start, "chunk_pair_start"}}) {
    CheckTensor(*named.tensor, named.name, device_id, dl_int32);
  }
  CheckTensor(cu_chunk_offsets, "cu_chunk_offsets", device_id, dl_int64);
  CheckTensor(descriptor_storage, "descriptor_storage", device_id, dl_uint8);
  CheckTensor(state_checkpoint_needed, "state_checkpoint_needed", device_id, dl_uint32);
  CheckTensor(zero_workspace, "zero_workspace", device_id, dl_uint32);

  Check1D(seq_order, "seq_order", num_sequences);
  Check1D(cu_chunk_offsets, "cu_chunk_offsets", num_sequences + 1);
  Check1D(consumer_chunk_order, "consumer_chunk_order", total_chunks);
  Check1D(chunk_sequence, "chunk_sequence", total_chunks);
  Check1D(chunk_index, "chunk_index", total_chunks);
  Check1D(chunk_pair_start, "chunk_pair_start", total_pairs);
  Check1D(state_checkpoint_needed, "state_checkpoint_needed",
          (total_chunks + num_sequences) * num_heads);
  TVM_FFI_ICHECK(descriptor_storage.ndim() == 1 &&
                 descriptor_storage.numel() >=
                     static_cast<int64_t>(flash_kda::kDescriptorStorageBytes))
      << "descriptor_storage must provide at least 768 bytes";
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(descriptor_storage.data_ptr()) %
                     flash_kda::kTensorMapAlignment ==
                 0)
      << "descriptor_storage must be 64-byte aligned";

  Check3D(forward_out, "forward_out", total_tokens, num_heads, kHeadDim);
  Check4D(forward_final, "forward_final", num_sequences, num_heads, kHeadDim, kHeadDim);
  Check4D(chunk_state, "chunk_state", total_chunks, num_heads, kHeadDim, kHeadDim);
  for (const auto& named : {NamedTensor{&tape_qd, "tape_qd"}, NamedTensor{&tape_kd, "tape_kd"},
                            NamedTensor{&tape_kr, "tape_kr"}, NamedTensor{&grad_qd, "grad_qd"},
                            NamedTensor{&grad_kd, "grad_kd"}, NamedTensor{&grad_ki, "grad_ki"}}) {
    Check4D(*named.tensor, named.name, total_chunks, num_heads, kChunkTokens, kHeadDim);
  }
  Check4D(tape_j, "tape_j", total_chunks, num_heads, kChunkTokens, kChunkTokens);
  Check3D(tape_restore_factor, "tape_restore_factor", total_chunks, num_heads, kHeadDim);
  for (const auto& named : {NamedTensor{&tape_e, "tape_e"}, NamedTensor{&tape_x, "tape_x"},
                            NamedTensor{&tape_r, "tape_r"}, NamedTensor{&chunk_dr, "chunk_dr"},
                            NamedTensor{&chunk_dx, "chunk_dx"}}) {
    Check4D(*named.tensor, named.name, total_chunks, num_heads, kHeadDim, kChunkTokens);
  }
  Check4D(chunk_dh, "chunk_dh", total_chunks, num_heads, kHeadDim, kHeadDim);
  Check3D(norm_inv, "norm_inv", total_tokens, num_heads, 2);
  Check3D(decay, "decay", total_tokens, num_heads, kHeadDim);
  Check2D(beta_active, "beta_active", total_tokens, num_heads);
  Check3D(dlog_decay, "dlog_decay", total_tokens, num_heads, kHeadDim);
  Check2D(dbeta_active, "dbeta_active", total_tokens, num_heads);
  for (const auto& named : {NamedTensor{&dq, "dq"}, NamedTensor{&dk, "dk"}, NamedTensor{&dv, "dv"},
                            NamedTensor{&dg, "dg"}}) {
    CheckShape(*named.tensor, q, named.name, "q");
  }
  CheckShape(dbeta, beta, "dbeta", "beta");
  CheckShape(dA_log, A_log, "dA_log", "A_log");
  CheckShape(ddt_bias, dt_bias, "ddt_bias", "dt_bias");
  CheckShape(dinitial_state, initial_state, "dinitial_state", "initial_state");

  const int64_t beta_tma_rows = std::max<int64_t>(total_tokens, kChunkTokens);
  TVM_FFI_ICHECK(beta_tma.ndim() == 2 && beta_tma.size(0) == beta_tma_rows &&
                 beta_tma.size(1) == num_heads)
      << "beta_tma must have shape [max(total_tokens, 32), H]";
  const bool beta_tma_aliases_beta = beta_tma.data_ptr() == beta.data_ptr();
  TVM_FFI_ICHECK((total_tokens >= kChunkTokens && beta_tma_aliases_beta) ||
                 (total_tokens < kChunkTokens && !beta_tma_aliases_beta))
      << "beta_tma must exactly alias beta for T>=32 and use padded disjoint storage for T<32";

  TVM_FFI_ICHECK(tape_e.data_ptr() == tape_x.data_ptr() && tape_e.numel() == tape_x.numel())
      << "the tcgen local route requires tape_e to exactly alias tape_x";
  const int64_t zero_words = total_chunks * num_heads;
  Check1D(zero_workspace, "zero_workspace", zero_words);

  const std::initializer_list<NamedTensor> disjoint = {
      {&q, "q"},
      {&k, "k"},
      {&v, "v"},
      {&g, "g"},
      {&beta, "beta"},
      {&A_log, "A_log"},
      {&dt_bias, "dt_bias"},
      {&initial_state, "initial_state"},
      {&do_, "do"},
      {&dfinal_state, "dfinal_state"},
      {&cu_seqlens, "cu_seqlens"},
      {&seq_order, "seq_order"},
      {&cu_chunk_offsets, "cu_chunk_offsets"},
      {&consumer_chunk_order, "consumer_chunk_order"},
      {&chunk_sequence, "chunk_sequence"},
      {&chunk_index, "chunk_index"},
      {&chunk_pair_start, "chunk_pair_start"},
      {&descriptor_storage, "descriptor_storage"},
      {&forward_out, "forward_out"},
      {&forward_final, "forward_final"},
      {&chunk_state, "chunk_state"},
      {&state_checkpoint_needed, "state_checkpoint_needed"},
      {&tape_qd, "tape_qd"},
      {&tape_kd, "tape_kd"},
      {&tape_kr, "tape_kr"},
      {&tape_j, "tape_j"},
      {&tape_restore_factor, "tape_restore_factor"},
      {&tape_x, "tape_x_and_e"},
      {&tape_r, "tape_r"},
      {&norm_inv, "norm_inv"},
      {&decay, "decay"},
      {&beta_active, "beta_active"},
      {&zero_workspace, "zero_workspace"},
      {&chunk_dh, "chunk_dh"},
      {&chunk_dr, "chunk_dr"},
      {&chunk_dx, "chunk_dx"},
      {&grad_qd, "grad_qd"},
      {&grad_kd, "grad_kd"},
      {&grad_ki, "grad_ki"},
      {&dlog_decay, "dlog_decay"},
      {&dbeta_active, "dbeta_active"},
      {&dq, "dq"},
      {&dk, "dk"},
      {&dv, "dv"},
      {&dg, "dg"},
      {&dbeta, "dbeta"},
      {&dA_log, "dA_log"},
      {&ddt_bias, "ddt_bias"},
      {&dinitial_state, "dinitial_state"},
  };
  CheckPairwiseDisjoint(disjoint);
  if (!beta_tma_aliases_beta) {
    CheckDisjointFrom(beta_tma, "beta_tma", disjoint);
  }

  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  LaunchCuSeqlensValidation(cu_seqlens, ResolveCanonicalShape(q, num_sequences, num_heads), stream);
  CheckCuda(cudaMemsetAsync(dA_log.data_ptr(), 0, dA_log.numel() * sizeof(float), stream),
            "clear dA_log");
  CheckCuda(cudaMemsetAsync(ddt_bias.data_ptr(), 0, ddt_bias.numel() * sizeof(float), stream),
            "clear ddt_bias");
  if (!beta_tma_aliases_beta) {
    CheckCuda(
        cudaMemcpyAsync(beta_tma.data_ptr(), beta.data_ptr(), beta.numel() * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream),
        "copy beta into padded beta_tma");
  }
  const TmaPointers tma = EncodeTmaPointers<128, 32>(
      q, k, v, g, beta_tma, forward_out, descriptor_storage, prepare_descriptors, stream);

  ConfigureDynamicSmem(kernel_flashkda_bf16_fused_m128, kTapeSmemBytes, device_id,
                       "cudaFuncSetAttribute(backward tape)");
  const dim3 tape_grid(CheckedGridX(num_sequences * num_heads, "backward tape"), 1, 1);
  kernel_flashkda_bf16_fused_m128<<<tape_grid, kTapeThreads, kTapeSmemBytes, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<flashkda_backward_generated_FlashKDATensorMap const*>(tma.q),
      reinterpret_cast<__nv_bfloat16*>(k.data_ptr()),
      reinterpret_cast<flashkda_backward_generated_FlashKDATensorMap const*>(tma.k),
      reinterpret_cast<__nv_bfloat16*>(v.data_ptr()),
      reinterpret_cast<flashkda_backward_generated_FlashKDATensorMap const*>(tma.v),
      reinterpret_cast<__nv_bfloat16*>(g.data_ptr()),
      reinterpret_cast<flashkda_backward_generated_FlashKDATensorMap const*>(tma.g),
      reinterpret_cast<__nv_bfloat16*>(beta.data_ptr()),
      reinterpret_cast<flashkda_backward_generated_FlashKDATensorMap const*>(tma.beta),
      reinterpret_cast<float*>(A_log.data_ptr()), reinterpret_cast<float*>(dt_bias.data_ptr()),
      reinterpret_cast<long long*>(cu_seqlens.data_ptr()),
      reinterpret_cast<int*>(seq_order.data_ptr()), reinterpret_cast<__nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(forward_out.data_ptr()),
      reinterpret_cast<flashkda_backward_generated_FlashKDATensorMap const*>(tma.out),
      reinterpret_cast<__nv_bfloat16*>(forward_final.data_ptr()), static_cast<int32_t>(num_heads),
      1, 0, static_cast<float>(scale), static_cast<float>(lower_bound), 0ULL, 0ULL, 0ULL, 0LL, 0LL,
      0, 0, reinterpret_cast<long long*>(cu_chunk_offsets.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(chunk_state.data_ptr()),
      reinterpret_cast<unsigned int*>(state_checkpoint_needed.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(tape_qd.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(tape_kd.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(tape_kr.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(tape_j.data_ptr()),
      reinterpret_cast<float*>(tape_restore_factor.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(tape_e.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(tape_x.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(tape_r.data_ptr()),
      reinterpret_cast<float*>(norm_inv.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(decay.data_ptr()),
      reinterpret_cast<float*>(beta_active.data_ptr()),
      reinterpret_cast<float*>(initial_state.data_ptr()),
      reinterpret_cast<unsigned int*>(zero_workspace.data_ptr()), static_cast<int32_t>(zero_words));
  CheckCuda(cudaGetLastError(), "kernel_flashkda_bf16_fused_m128 backward tape launch");

  ConfigureDynamicSmem(kernel_flashkda_backward_state_checkpoint_fallback_c32, kFallbackSmemBytes,
                       device_id, "cudaFuncSetAttribute(state checkpoint fallback)");
  kernel_flashkda_backward_state_checkpoint_fallback_c32<<<1, kFallbackThreads, kFallbackSmemBytes,
                                                           stream>>>(
      reinterpret_cast<long long*>(cu_seqlens.data_ptr()),
      reinterpret_cast<long long*>(cu_chunk_offsets.data_ptr()),
      reinterpret_cast<unsigned int*>(state_checkpoint_needed.data_ptr()),
      reinterpret_cast<float*>(initial_state.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(tape_kr.data_ptr()),
      reinterpret_cast<float*>(tape_restore_factor.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(tape_r.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(chunk_state.data_ptr()), static_cast<int32_t>(num_sequences),
      static_cast<int32_t>(num_heads), static_cast<float>(lower_bound));
  CheckCuda(cudaGetLastError(), "kernel_flashkda_backward_state_checkpoint_fallback_c32 launch");

  const bool split_boundary = num_heads <= 64;
  const int32_t boundary_ready_target = split_boundary ? 2 : 3;
  const uint32_t boundary_grid_x =
      CheckedGridX(num_sequences * num_heads * (split_boundary ? 2 : 1), "boundary");
  if (split_boundary) {
    ConfigureDynamicSmem(kernel_flashkda_backward_boundary_c32_tcgen_m64, kBoundaryM64SmemBytes,
                         device_id, "cudaFuncSetAttribute(boundary M64)");
    kernel_flashkda_backward_boundary_c32_tcgen_m64<<<boundary_grid_x, kBoundaryM64Threads,
                                                      kBoundaryM64SmemBytes, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(do_.data_ptr()),
        reinterpret_cast<float*>(dfinal_state.data_ptr()),
        reinterpret_cast<float*>(beta_active.data_ptr()),
        reinterpret_cast<long long*>(cu_seqlens.data_ptr()),
        reinterpret_cast<long long*>(cu_chunk_offsets.data_ptr()),
        reinterpret_cast<int*>(seq_order.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(tape_qd.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(tape_kd.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(tape_kr.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(tape_j.data_ptr()),
        reinterpret_cast<float*>(tape_restore_factor.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(chunk_dh.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(chunk_dr.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(chunk_dx.data_ptr()),
        reinterpret_cast<float*>(dinitial_state.data_ptr()),
        reinterpret_cast<unsigned int*>(zero_workspace.data_ptr()), static_cast<int32_t>(num_heads),
        1, static_cast<float>(lower_bound));
    CheckCuda(cudaGetLastError(), "kernel_flashkda_backward_boundary_c32_tcgen_m64 launch");
  } else {
    ConfigureDynamicSmem(kernel_flashkda_backward_boundary_c32_tcgen, kBoundaryM128SmemBytes,
                         device_id, "cudaFuncSetAttribute(boundary M128)");
    kernel_flashkda_backward_boundary_c32_tcgen<<<boundary_grid_x, kBoundaryM128Threads,
                                                  kBoundaryM128SmemBytes, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(do_.data_ptr()),
        reinterpret_cast<float*>(dfinal_state.data_ptr()),
        reinterpret_cast<float*>(beta_active.data_ptr()),
        reinterpret_cast<long long*>(cu_seqlens.data_ptr()),
        reinterpret_cast<long long*>(cu_chunk_offsets.data_ptr()),
        reinterpret_cast<int*>(seq_order.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(tape_qd.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(tape_kd.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(tape_kr.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(tape_j.data_ptr()),
        reinterpret_cast<float*>(tape_restore_factor.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(chunk_dh.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(chunk_dr.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(chunk_dx.data_ptr()),
        reinterpret_cast<unsigned int*>(zero_workspace.data_ptr()),
        reinterpret_cast<float*>(dinitial_state.data_ptr()), static_cast<int32_t>(num_heads), 1,
        static_cast<float>(lower_bound));
    CheckCuda(cudaGetLastError(), "kernel_flashkda_backward_boundary_c32_tcgen launch");
  }

  ConfigureDynamicSmem(kernel_flashkda_backward_local_c32_tcgen, kLocalSmemBytes, device_id,
                       "cudaFuncSetAttribute(local tcgen)");
  cudaLaunchAttribute local_attr[1];
  local_attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  local_attr[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t local_config{};
  local_config.gridDim = dim3(CheckedGridX(total_chunks * num_heads, "local"), 1, 1);
  local_config.blockDim = dim3(kLocalThreads, 1, 1);
  local_config.dynamicSmemBytes = kLocalSmemBytes;
  local_config.stream = stream;
  local_config.attrs = local_attr;
  local_config.numAttrs = 1;
  CheckCuda(cudaLaunchKernelEx(&local_config, kernel_flashkda_backward_local_c32_tcgen,
                               reinterpret_cast<__nv_bfloat16*>(do_.data_ptr()),
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
                               reinterpret_cast<unsigned int*>(zero_workspace.data_ptr()),
                               reinterpret_cast<__nv_bfloat16*>(grad_qd.data_ptr()),
                               reinterpret_cast<__nv_bfloat16*>(grad_kd.data_ptr()),
                               reinterpret_cast<__nv_bfloat16*>(grad_ki.data_ptr()),
                               reinterpret_cast<float*>(dlog_decay.data_ptr()),
                               reinterpret_cast<float*>(dbeta_active.data_ptr()),
                               reinterpret_cast<__nv_bfloat16*>(dv.data_ptr()),
                               static_cast<int32_t>(num_heads), boundary_ready_target,
                               static_cast<float>(lower_bound)),
            "cudaLaunchKernelEx(kernel_flashkda_backward_local_c32_tcgen)");

  const int64_t num_pair_heads = total_pairs * num_heads;
  const uint32_t finalize_grid = CheckedGridX(((num_pair_heads + 3) / 4) * 4, "map_finalize");
  kernel_flashkda_backward_map_finalize_c32<<<finalize_grid, kMapFinalizeThreads, 0, stream>>>(
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
      reinterpret_cast<__nv_bfloat16*>(dq.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(dk.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(dg.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(dbeta.data_ptr()),
      reinterpret_cast<float*>(dA_log.data_ptr()), reinterpret_cast<float*>(ddt_bias.data_ptr()),
      static_cast<int32_t>(num_pair_heads), static_cast<int32_t>(num_heads),
      static_cast<float>(scale), static_cast<float>(lower_bound));
  CheckCuda(cudaGetLastError(), "kernel_flashkda_backward_map_finalize_c32 launch");
}

}  // namespace flash_kda_backward
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_low, flashinfer::flash_kda_backward::RunLow);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_high, flashinfer::flash_kda_backward::RunHigh);
