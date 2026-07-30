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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <utility>

#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace flash_kda_decode {

constexpr int32_t kHeadDim = 128;
constexpr int32_t kTokens = 5;

struct LaunchContext {
  int32_t device_id;
  int32_t num_sequences;
  int32_t num_heads;
  int32_t num_value_heads;
  int32_t head_ratio;
  int32_t gate_token_stride;
  int32_t state_slot_stride;
  cudaStream_t stream;
};

inline void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK(status == cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

inline void CheckExactSm100a(int32_t device_id) {
  int major = 0;
  int minor = 0;
  CheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
            "cudaDeviceGetAttribute(major)");
  CheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
            "cudaDeviceGetAttribute(minor)");
  TVM_FFI_ICHECK(major == 10 && minor == 0)
      << "frozen FlashKDA decode kernels require exact compute capability 10.0 "
         "(sm_100a), got "
      << major << "." << minor;
}

inline void CheckCudaTensor(const TensorView& tensor, const char* name, int32_t device_id) {
  TVM_FFI_ICHECK(tensor.device().device_type == kDLCUDA) << name << " must be a CUDA tensor";
  TVM_FFI_ICHECK(tensor.device().device_id == device_id)
      << name << " must be on CUDA device " << device_id << ", got " << tensor.device().device_id;
}

inline void CheckDtype(const TensorView& tensor, const char* name, DLDataType expected) {
  const DLDataType actual = tensor.dtype();
  TVM_FFI_ICHECK(actual.code == expected.code && actual.bits == expected.bits &&
                 actual.lanes == expected.lanes)
      << name << " has the wrong dtype";
}

inline int64_t CheckedInt32(int64_t value, const char* name, bool positive = false) {
  TVM_FFI_ICHECK(value >= (positive ? 1 : 0) && value <= std::numeric_limits<int32_t>::max())
      << name << " is outside the supported int32 range: " << value;
  return value;
}

inline int64_t CheckedInt32Product(std::initializer_list<int64_t> factors, const char* name) {
  int64_t product = 1;
  for (const int64_t factor : factors) {
    TVM_FFI_ICHECK(factor > 0 && product <= std::numeric_limits<int32_t>::max() / factor)
        << name << " is outside the supported positive int32 range";
    product *= factor;
  }
  return product;
}

inline void CheckContiguous(const TensorView& tensor, const char* name) {
  TVM_FFI_ICHECK(tensor.IsContiguous()) << name << " must be contiguous";
}

inline void CheckAlignment(const TensorView& tensor, const char* name, uintptr_t alignment) {
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignment == 0)
      << name << " data pointer must be aligned to " << alignment << " bytes";
}

inline void CheckCompactGate(const TensorView& gate, int64_t num_value_heads, int64_t head_dim) {
  TVM_FFI_ICHECK(gate.ndim() == 4) << "g must be rank 4";
  TVM_FFI_ICHECK(gate.stride(3) == 1 && gate.stride(2) == head_dim)
      << "g must be compact in its [HV, K] trailing dimensions";
  CheckedInt32(gate.stride(1), "g token stride", true);
  TVM_FFI_ICHECK(gate.stride(1) >= num_value_heads * head_dim)
      << "g token stride must not overlap adjacent tokens";
  TVM_FFI_ICHECK(gate.stride(1) % 4 == 0)
      << "g token stride must preserve eight-byte vector alignment";
}

inline void CheckStateLayout(const TensorView& state, int64_t num_value_heads, int64_t head_dim) {
  TVM_FFI_ICHECK(state.ndim() == 4) << "initial_state must be rank 4";
  TVM_FFI_ICHECK(state.size(0) > 0 && state.size(1) == num_value_heads &&
                 state.size(2) == head_dim && state.size(3) == head_dim)
      << "initial_state must have shape [slots, HV, 128, 128]";
  TVM_FFI_ICHECK(state.stride(3) == 1 && state.stride(2) == head_dim &&
                 state.stride(1) == head_dim * head_dim &&
                 state.stride(0) >= num_value_heads * head_dim * head_dim)
      << "initial_state must have compact [HV, 128, 128] blocks and may "
         "only pad the outer slot stride";
  CheckedInt32(state.stride(0), "initial_state slot stride", true);
  TVM_FFI_ICHECK(state.stride(0) % 8 == 0)
      << "initial_state slot stride must preserve 16-byte vector alignment";
}

inline std::pair<uintptr_t, uintptr_t> TensorByteRange(const TensorView& tensor, const char* name) {
  const DLDataType dtype = tensor.dtype();
  const uint64_t bits = static_cast<uint64_t>(dtype.bits) * dtype.lanes;
  TVM_FFI_ICHECK(bits > 0 && bits % 8 == 0) << name << " has a non-byte dtype";

  uint64_t last_element = 0;
  for (int32_t i = 0; i < tensor.ndim(); ++i) {
    TVM_FFI_ICHECK(tensor.size(i) >= 0 && tensor.stride(i) >= 0)
        << name << " must not have negative shapes or strides";
    if (tensor.size(i) > 0) {
      last_element +=
          static_cast<uint64_t>(tensor.size(i) - 1) * static_cast<uint64_t>(tensor.stride(i));
    }
  }

  const uint64_t bytes = tensor.numel() == 0 ? 0 : (last_element + 1) * (bits / 8);
  const uintptr_t begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  TVM_FFI_ICHECK(bytes <= std::numeric_limits<uintptr_t>::max() - begin)
      << name << " byte range overflows uintptr_t";
  return {begin, begin + static_cast<uintptr_t>(bytes)};
}

inline void CheckNoOverlap(const TensorView& lhs, const char* lhs_name, const TensorView& rhs,
                           const char* rhs_name) {
  const auto lhs_range = TensorByteRange(lhs, lhs_name);
  const auto rhs_range = TensorByteRange(rhs, rhs_name);
  TVM_FFI_ICHECK(lhs_range.first >= rhs_range.second || rhs_range.first >= lhs_range.second)
      << lhs_name << " must not overlap " << rhs_name
      << ": the frozen kernel uses __restrict__ pointers";
}

template <int kValueSplit>
LaunchContext CheckInputs(const TensorView& q, const TensorView& k, const TensorView& v,
                          const TensorView& g, const TensorView& beta, const TensorView& A_log,
                          const TensorView& dt_bias, const TensorView& state, const TensorView& out,
                          const TensorView& cu_seqlens, const TensorView& ssm_state_indices,
                          const TensorView& num_accepted_tokens, double scale, double lower_bound,
                          int64_t cuda_stream) {
  static_assert(kValueSplit == 1 || kValueSplit == 2 || kValueSplit == 4 || kValueSplit == 8);
  static_assert(kHeadDim % (16 * kValueSplit) == 0);

  TVM_FFI_ICHECK(cuda_stream >= 0) << "cuda_stream must be a non-negative stream handle";
  TVM_FFI_ICHECK(q.device().device_type == kDLCUDA) << "q must be a CUDA tensor";
  const int32_t device_id = q.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckExactSm100a(device_id);

  for (const auto& named :
       {std::pair<const TensorView*, const char*>(&q, "q"),
        std::pair<const TensorView*, const char*>(&k, "k"),
        std::pair<const TensorView*, const char*>(&v, "v"),
        std::pair<const TensorView*, const char*>(&g, "g"),
        std::pair<const TensorView*, const char*>(&beta, "beta"),
        std::pair<const TensorView*, const char*>(&A_log, "A_log"),
        std::pair<const TensorView*, const char*>(&dt_bias, "dt_bias"),
        std::pair<const TensorView*, const char*>(&state, "initial_state"),
        std::pair<const TensorView*, const char*>(&out, "output"),
        std::pair<const TensorView*, const char*>(&cu_seqlens, "cu_seqlens"),
        std::pair<const TensorView*, const char*>(&ssm_state_indices, "ssm_state_indices"),
        std::pair<const TensorView*, const char*>(&num_accepted_tokens, "num_accepted_tokens")}) {
    CheckCudaTensor(*named.first, named.second, device_id);
  }

  for (const auto& named : {std::pair<const TensorView*, const char*>(&q, "q"),
                            std::pair<const TensorView*, const char*>(&k, "k"),
                            std::pair<const TensorView*, const char*>(&v, "v"),
                            std::pair<const TensorView*, const char*>(&g, "g"),
                            std::pair<const TensorView*, const char*>(&beta, "beta"),
                            std::pair<const TensorView*, const char*>(&state, "initial_state"),
                            std::pair<const TensorView*, const char*>(&out, "output")}) {
    CheckDtype(*named.first, named.second, dl_bfloat16);
  }
  CheckDtype(A_log, "A_log", dl_float32);
  CheckDtype(dt_bias, "dt_bias", dl_float32);
  CheckDtype(cu_seqlens, "cu_seqlens", dl_int32);
  CheckDtype(ssm_state_indices, "ssm_state_indices", dl_int32);
  CheckDtype(num_accepted_tokens, "num_accepted_tokens", dl_int32);

  TVM_FFI_ICHECK(q.ndim() == 4 && q.size(0) == 1 && q.size(3) == kHeadDim)
      << "q must have shape [1, total_tokens, H, 128]";
  const int64_t total_tokens = q.size(1);
  const int64_t num_heads = q.size(2);
  TVM_FFI_ICHECK(total_tokens > 0 && num_heads > 0)
      << "q must contain at least one token and one head";
  TVM_FFI_ICHECK(k.ndim() == 4 && k.size(0) == 1 && k.size(1) == total_tokens &&
                 k.size(2) == num_heads && k.size(3) == kHeadDim)
      << "k must match q";

  const int64_t num_value_heads = v.ndim() == 4 ? v.size(2) : 0;
  TVM_FFI_ICHECK(v.ndim() == 4 && v.size(0) == 1 && v.size(1) == total_tokens &&
                 v.size(3) == kHeadDim)
      << "v must have shape [1, total_tokens, HV, 128]";
  TVM_FFI_ICHECK(num_value_heads >= num_heads && num_value_heads % num_heads == 0)
      << "HV must be a positive multiple of H";
  TVM_FFI_ICHECK(g.ndim() == 4 && g.size(0) == 1 && g.size(1) == total_tokens &&
                 g.size(2) == num_value_heads && g.size(3) == kHeadDim)
      << "g must match v";
  TVM_FFI_ICHECK(beta.ndim() == 3 && beta.size(0) == 1 && beta.size(1) == total_tokens &&
                 beta.size(2) == num_value_heads)
      << "beta must have shape [1, total_tokens, HV]";
  TVM_FFI_ICHECK(out.ndim() == 4 && out.size(0) == 1 && out.size(1) == total_tokens &&
                 out.size(2) == num_value_heads && out.size(3) == kHeadDim)
      << "output must match v";

  CheckContiguous(q, "q");
  CheckContiguous(k, "k");
  CheckContiguous(v, "v");
  CheckCompactGate(g, num_value_heads, kHeadDim);
  CheckContiguous(beta, "beta");
  CheckContiguous(out, "output");
  CheckStateLayout(state, num_value_heads, kHeadDim);
  CheckContiguous(cu_seqlens, "cu_seqlens");
  CheckContiguous(ssm_state_indices, "ssm_state_indices");
  CheckContiguous(num_accepted_tokens, "num_accepted_tokens");
  CheckAlignment(q, "q", 8);
  CheckAlignment(k, "k", 8);
  CheckAlignment(g, "g", 8);
  CheckAlignment(state, "initial_state", 16);

  TVM_FFI_ICHECK(cu_seqlens.ndim() == 1 && cu_seqlens.numel() >= 2)
      << "cu_seqlens must be a one-dimensional [N+1] tensor";
  const int64_t num_sequences = cu_seqlens.numel() - 1;
  TVM_FFI_ICHECK(num_sequences <= 65535) << "number of sequences exceeds CUDA grid.y limit";
  TVM_FFI_ICHECK(total_tokens == num_sequences * kTokens)
      << "packed speculative input must contain exactly N * 5 tokens";
  TVM_FFI_ICHECK(ssm_state_indices.numel() == num_sequences * kTokens)
      << "ssm_state_indices must contain N * 5 elements";
  TVM_FFI_ICHECK(num_accepted_tokens.ndim() == 1 && num_accepted_tokens.numel() == num_sequences)
      << "num_accepted_tokens must contain N elements";

  TVM_FFI_ICHECK(A_log.numel() >= 1 && dt_bias.numel() >= 1)
      << "precomputed-gate variants require non-empty dummy A_log/dt_bias";
  TVM_FFI_ICHECK(lower_bound == 0.0) << "precomputed-gate variants require lower_bound=0";
  TVM_FFI_ICHECK(std::isfinite(scale) && std::isfinite(static_cast<float>(scale)))
      << "scale must be finite and representable as float32";

  CheckedInt32Product({total_tokens, num_heads, kHeadDim}, "q/k linear index extent");
  CheckedInt32Product({total_tokens, num_value_heads, kHeadDim}, "v/g/output linear index extent");
  CheckedInt32Product({total_tokens, g.stride(1)}, "g strided linear index extent");
  CheckedInt32Product({num_sequences, kTokens}, "metadata linear index extent");
  CheckedInt32Product({state.size(0), state.stride(0)}, "state linear index extent");
  CheckedInt32Product({num_value_heads, kValueSplit}, "decode grid.x extent");

  for (const auto& named :
       {std::pair<const TensorView*, const char*>(&q, "q"),
        std::pair<const TensorView*, const char*>(&k, "k"),
        std::pair<const TensorView*, const char*>(&v, "v"),
        std::pair<const TensorView*, const char*>(&g, "g"),
        std::pair<const TensorView*, const char*>(&beta, "beta"),
        std::pair<const TensorView*, const char*>(&A_log, "A_log"),
        std::pair<const TensorView*, const char*>(&dt_bias, "dt_bias"),
        std::pair<const TensorView*, const char*>(&cu_seqlens, "cu_seqlens"),
        std::pair<const TensorView*, const char*>(&ssm_state_indices, "ssm_state_indices"),
        std::pair<const TensorView*, const char*>(&num_accepted_tokens, "num_accepted_tokens")}) {
    CheckNoOverlap(out, "output", *named.first, named.second);
    CheckNoOverlap(state, "initial_state", *named.first, named.second);
  }
  CheckNoOverlap(out, "output", state, "initial_state");

  return {
      device_id,
      static_cast<int32_t>(CheckedInt32(num_sequences, "number of sequences", true)),
      static_cast<int32_t>(CheckedInt32(num_heads, "H", true)),
      static_cast<int32_t>(CheckedInt32(num_value_heads, "HV", true)),
      static_cast<int32_t>(num_value_heads / num_heads),
      static_cast<int32_t>(CheckedInt32(g.stride(1), "g token stride", true)),
      static_cast<int32_t>(CheckedInt32(state.stride(0), "state slot stride", true)),
      reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream)),
  };
}

inline void CheckDynamicSmemCapacity(int32_t device_id, int32_t required) {
  int capacity = 0;
  CheckCuda(cudaDeviceGetAttribute(&capacity, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id),
            "cudaDeviceGetAttribute(max opt-in shared memory)");
  TVM_FFI_ICHECK(required <= capacity)
      << "frozen FlashKDA decode kernel requires " << required
      << " bytes of dynamic shared memory, but the device supports " << capacity;
}

}  // namespace flash_kda_decode
}  // namespace flashinfer
