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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace flash_kda {

constexpr int64_t kHeadDim = 128;
constexpr size_t kTensorMapCount = 6;
constexpr size_t kTensorMapAlignment = 64;
static_assert(sizeof(CUtensorMap) == 128);
constexpr size_t kDescriptorStorageBytes = kTensorMapCount * sizeof(CUtensorMap);
constexpr int64_t kBetaTmaHeadsPerBox = 8;

inline int64_t RoundUpBetaTmaHeads(int64_t num_heads) {
  return (num_heads / kBetaTmaHeadsPerBox +
          static_cast<int64_t>(num_heads % kBetaTmaHeadsPerBox != 0)) *
         kBetaTmaHeadsPerBox;
}

static __global__ void PackBetaForTmaKernel(const __nv_bfloat16* beta, __nv_bfloat16* beta_tma,
                                            int64_t token_count, int64_t padded_elements,
                                            int64_t num_heads, int64_t padded_num_heads,
                                            int64_t beta_token_stride) {
  const int64_t linear_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear_index >= padded_elements) {
    return;
  }
  const int64_t token_index = linear_index / padded_num_heads;
  const int64_t head_index = linear_index % padded_num_heads;
  __nv_bfloat16 value = __float2bfloat16(0.0f);
  if (token_index < token_count && head_index < num_heads) {
    value = beta[token_index * beta_token_stride + head_index];
  }
  beta_tma[linear_index] = value;
}

inline void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK(status == cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

inline void CheckCudaTensorDevice(const TensorView& tensor, const char* name, int32_t device_id) {
  TVM_FFI_ICHECK(tensor.device().device_type == kDLCUDA) << name << " must be a CUDA tensor";
  TVM_FFI_ICHECK(tensor.device().device_id == device_id)
      << name << " must be on CUDA device " << device_id << ", got " << tensor.device().device_id;
}

inline void CheckCudaTensor(const TensorView& tensor, const char* name, int32_t device_id) {
  CheckCudaTensorDevice(tensor, name, device_id);
  TVM_FFI_ICHECK(tensor.IsContiguous()) << name << " must be contiguous";
}

inline void CheckDtype(const TensorView& tensor, const char* name, DLDataType expected) {
  const DLDataType actual = tensor.dtype();
  TVM_FFI_ICHECK(actual.code == expected.code && actual.bits == expected.bits &&
                 actual.lanes == expected.lanes)
      << name << " has wrong dtype: expected (code=" << int(expected.code)
      << ", bits=" << int(expected.bits) << ", lanes=" << int(expected.lanes)
      << "), got (code=" << int(actual.code) << ", bits=" << int(actual.bits)
      << ", lanes=" << int(actual.lanes) << ")";
}

struct TensorByteRange {
  uintptr_t begin;
  uintptr_t end;
};

inline TensorByteRange GetTensorByteRange(const TensorView& tensor, const char* name) {
  const DLDataType dtype = tensor.dtype();
  const uint64_t bits = static_cast<uint64_t>(dtype.bits) * static_cast<uint64_t>(dtype.lanes);
  TVM_FFI_ICHECK(bits > 0 && bits % 8 == 0) << name << " has a non-byte-addressable dtype";
  uint64_t max_element_offset = 0;
  for (int32_t dim = 0; dim < tensor.ndim(); ++dim) {
    TVM_FFI_ICHECK(tensor.stride(dim) >= 0) << name << " must have non-negative strides";
    if (tensor.size(dim) > 0) {
      const uint64_t extent = static_cast<uint64_t>(tensor.size(dim) - 1);
      const uint64_t stride = static_cast<uint64_t>(tensor.stride(dim));
      TVM_FFI_ICHECK(stride == 0 ||
                     extent <= (std::numeric_limits<uint64_t>::max() - max_element_offset) / stride)
          << name << " strided byte range overflows uint64";
      max_element_offset += extent * stride;
    }
  }
  const uint64_t element_bytes = bits / 8;
  TVM_FFI_ICHECK(max_element_offset < std::numeric_limits<uint64_t>::max() / element_bytes)
      << name << " strided byte range overflows uint64";
  const uint64_t bytes = tensor.numel() == 0 ? 0 : (max_element_offset + 1) * element_bytes;
  const uintptr_t begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  TVM_FFI_ICHECK(bytes <= std::numeric_limits<uintptr_t>::max() - begin)
      << name << " byte range overflows uintptr_t";
  return {begin, begin + static_cast<uintptr_t>(bytes)};
}

inline void CheckNoOverlap(const TensorView& lhs, const char* lhs_name, const TensorView& rhs,
                           const char* rhs_name) {
  const TensorByteRange lhs_range = GetTensorByteRange(lhs, lhs_name);
  const TensorByteRange rhs_range = GetTensorByteRange(rhs, rhs_name);
  const bool overlaps = lhs_range.begin < rhs_range.end && rhs_range.begin < lhs_range.end;
  TVM_FFI_ICHECK(!overlaps) << lhs_name << " must not overlap " << rhs_name
                            << ": the frozen kernel uses __restrict__ pointers";
}

inline void CheckNoPartialOverlapOrExactAlias(const TensorView& lhs, const char* lhs_name,
                                              const TensorView& rhs, const char* rhs_name) {
  const TensorByteRange lhs_range = GetTensorByteRange(lhs, lhs_name);
  const TensorByteRange rhs_range = GetTensorByteRange(rhs, rhs_name);
  const bool overlaps = lhs_range.begin < rhs_range.end && rhs_range.begin < lhs_range.end;
  const bool exact_alias = lhs_range.begin == rhs_range.begin && lhs_range.end == rhs_range.end;
  TVM_FFI_ICHECK(!overlaps || exact_alias)
      << lhs_name << " and " << rhs_name
      << " must either be disjoint or exactly alias the same storage";
}

#if defined(FLASHINFER_FLASH_KDA_TARGET_MINOR) == defined(FLASHINFER_FLASH_KDA_TARGET_FAMILY)
#error "exactly one FlashKDA target kind must be defined by the JIT/AOT spec"
#endif

#if defined(FLASHINFER_FLASH_KDA_TARGET_MINOR)
constexpr int kFlashKDATargetMinor = FLASHINFER_FLASH_KDA_TARGET_MINOR;
static_assert(kFlashKDATargetMinor == 0 || kFlashKDATargetMinor == 3,
              "exact FlashKDA target must be SM100a or SM103a");
#else
constexpr int kFlashKDATargetFamily = FLASHINFER_FLASH_KDA_TARGET_FAMILY;
static_assert(kFlashKDATargetFamily == 100, "FlashKDA family target must be SM100f");
#endif

inline void CheckFlashKDATarget(int32_t device_id) {
  int major = 0;
  int minor = 0;
  CheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
            "cudaDeviceGetAttribute(major)");
  CheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
            "cudaDeviceGetAttribute(minor)");
#if defined(FLASHINFER_FLASH_KDA_TARGET_FAMILY)
  TVM_FFI_ICHECK(major == 10 && (minor == 0 || minor == 3))
      << "this FlashKDA module was compiled for the SM100 family "
         "(compute capability 10.0 or 10.3), got "
      << major << "." << minor;
#else
  TVM_FFI_ICHECK(major == 10 && minor == kFlashKDATargetMinor)
      << "this FlashKDA module was compiled for exact compute capability 10."
      << kFlashKDATargetMinor << ", got " << major << "." << minor;
#endif
}

inline void CheckFlashKDAPersistentDevice(int32_t device_id) {
  int major = 0;
  int minor = 0;
  CheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
            "cudaDeviceGetAttribute(major)");
  CheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
            "cudaDeviceGetAttribute(minor)");
  TVM_FFI_ICHECK(major == 10 && minor == 0)
      << "persistent FlashKDA is validated only on compute capability 10.0, got " << major << "."
      << minor;
}

inline int64_t CheckCommonInputs(const TensorView& q, const TensorView& k, const TensorView& v,
                                 const TensorView& g, const TensorView& beta,
                                 const TensorView& beta_tma, const TensorView& A_log,
                                 const TensorView& dt_bias, const TensorView& cu_seqlens,
                                 const TensorView& seq_order, const TensorView& initial_state,
                                 const TensorView& out, const TensorView& final_state,
                                 const TensorView& descriptor_storage, int64_t prepare_descriptors,
                                 int64_t num_heads, int64_t use_initial_state,
                                 int64_t store_final_state, double scale, double lower_bound,
                                 bool allow_serving_layouts = false, int64_t state_pool_slots = 0,
                                 bool allow_pair_packed_beta_tma = false) {
  TVM_FFI_ICHECK(prepare_descriptors == 0 || prepare_descriptors == 1)
      << "prepare_descriptors must be 0 or 1, got " << prepare_descriptors;
  TVM_FFI_ICHECK(num_heads > 0 && num_heads <= std::numeric_limits<int32_t>::max())
      << "num_heads must be in the positive int32 range, got " << num_heads;
  TVM_FFI_ICHECK(use_initial_state == 0 || use_initial_state == 1)
      << "use_initial_state must be 0 or 1, got " << use_initial_state;
  TVM_FFI_ICHECK(store_final_state == 0 || store_final_state == 1)
      << "store_final_state must be 0 or 1, got " << store_final_state;
  TVM_FFI_ICHECK(std::isfinite(scale) && std::isfinite(static_cast<float>(scale)))
      << "scale must be finite and representable as float32, got " << scale;
  TVM_FFI_ICHECK(std::isfinite(lower_bound) && lower_bound < 0.0 &&
                 std::isfinite(static_cast<float>(lower_bound)))
      << "lower_bound must be finite, negative, and representable as "
         "float32, got "
      << lower_bound;

  const int32_t device_id = q.device().device_id;
  CheckCudaTensor(q, "q", device_id);
  CheckCudaTensor(k, "k", device_id);
  CheckCudaTensor(v, "v", device_id);
  CheckCudaTensor(g, "g", device_id);
  CheckCudaTensorDevice(beta, "beta", device_id);
  CheckCudaTensorDevice(beta_tma, "beta_tma", device_id);
  CheckCudaTensor(A_log, "A_log", device_id);
  CheckCudaTensor(dt_bias, "dt_bias", device_id);
  CheckCudaTensor(cu_seqlens, "cu_seqlens", device_id);
  CheckCudaTensor(seq_order, "seq_order", device_id);
  CheckCudaTensorDevice(initial_state, "initial_state", device_id);
  CheckCudaTensor(out, "out", device_id);
  CheckCudaTensorDevice(final_state, "final_state", device_id);
  CheckCudaTensor(descriptor_storage, "descriptor_storage", device_id);

  CheckDtype(q, "q", dl_bfloat16);
  CheckDtype(k, "k", dl_bfloat16);
  CheckDtype(v, "v", dl_bfloat16);
  CheckDtype(g, "g", dl_bfloat16);
  CheckDtype(beta, "beta", dl_bfloat16);
  CheckDtype(beta_tma, "beta_tma", dl_bfloat16);
  CheckDtype(A_log, "A_log", dl_float32);
  CheckDtype(dt_bias, "dt_bias", dl_float32);
  CheckDtype(cu_seqlens, "cu_seqlens", dl_int64);
  CheckDtype(seq_order, "seq_order", dl_int32);
  CheckDtype(initial_state, "initial_state", dl_bfloat16);
  CheckDtype(out, "out", dl_bfloat16);
  CheckDtype(final_state, "final_state", dl_bfloat16);
  CheckDtype(descriptor_storage, "descriptor_storage", dl_uint8);

  if (!allow_serving_layouts) {
    TVM_FFI_ICHECK(beta.IsContiguous()) << "beta must be contiguous";
    TVM_FFI_ICHECK(beta_tma.IsContiguous()) << "beta_tma must be contiguous";
    TVM_FFI_ICHECK(initial_state.IsContiguous()) << "initial_state must be contiguous";
    TVM_FFI_ICHECK(final_state.IsContiguous()) << "final_state must be contiguous";
  } else {
    TVM_FFI_ICHECK(beta.ndim() >= 2 && beta.stride(beta.ndim() - 1) == 1 &&
                   beta.stride(beta.ndim() - 2) >= beta.size(beta.ndim() - 1))
        << "beta must have unit head stride and non-overlapping token rows";
    TVM_FFI_ICHECK(beta_tma.ndim() >= 2 && beta_tma.stride(beta_tma.ndim() - 1) == 1 &&
                   beta_tma.stride(beta_tma.ndim() - 2) >= beta_tma.size(beta_tma.ndim() - 1))
        << "beta_tma must have unit head stride and non-overlapping token rows";
  }

  TVM_FFI_ICHECK(descriptor_storage.numel() >= static_cast<int64_t>(kDescriptorStorageBytes))
      << "descriptor_storage must contain at least " << kDescriptorStorageBytes << " bytes";
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(descriptor_storage.data_ptr()) % kTensorMapAlignment ==
                 0)
      << "descriptor_storage must be aligned to " << kTensorMapAlignment << " bytes";

  TVM_FFI_ICHECK(q.ndim() >= 3) << "q must have trailing [H, 128] dimensions";
  TVM_FFI_ICHECK(q.size(q.ndim() - 1) == kHeadDim && q.size(q.ndim() - 2) == num_heads)
      << "q must have trailing shape [" << num_heads << ", 128]";
  const int64_t token_count = q.numel() / (num_heads * kHeadDim);
  TVM_FFI_ICHECK(token_count > 0) << "q must contain at least one token";

  for (const auto& named : {std::pair<const TensorView*, const char*>(&k, "k"),
                            std::pair<const TensorView*, const char*>(&v, "v"),
                            std::pair<const TensorView*, const char*>(&g, "g"),
                            std::pair<const TensorView*, const char*>(&out, "out")}) {
    const TensorView& tensor = *named.first;
    TVM_FFI_ICHECK(tensor.ndim() >= 3 && tensor.size(tensor.ndim() - 1) == kHeadDim &&
                   tensor.size(tensor.ndim() - 2) == num_heads && tensor.numel() == q.numel())
        << named.second << " must match q's flattened [tokens, H, 128] shape";
  }

  TVM_FFI_ICHECK(beta.ndim() >= 2 && beta.size(beta.ndim() - 1) == num_heads &&
                 beta.numel() == token_count * num_heads)
      << "beta must match flattened [tokens, H]";
  TVM_FFI_ICHECK(beta_tma.ndim() >= 2) << "beta_tma must have at least two dimensions";
  const int64_t padded_beta_tma_heads = RoundUpBetaTmaHeads(num_heads);
  const int64_t beta_tma_heads = beta_tma.size(beta_tma.ndim() - 1);
  const bool direct_beta_tma = allow_serving_layouts && beta_tma_heads == num_heads;
  const bool pair_packed_beta_tma = allow_pair_packed_beta_tma && num_heads == 12 &&
                                    beta_tma_heads == 24 && token_count % 2 == 0 &&
                                    beta_tma.numel() == token_count * num_heads;
  TVM_FFI_ICHECK(beta_tma.ndim() >= 2 &&
                 (((beta_tma_heads == padded_beta_tma_heads || direct_beta_tma) &&
                   beta_tma.numel() % beta_tma_heads == 0 &&
                   beta_tma.numel() / beta_tma_heads >= token_count) ||
                  pair_packed_beta_tma))
      << "beta_tma must have at least [tokens, H] direct storage or "
         "[tokens, round_up(H, 8)] padded storage, or the enabled H12 "
         "[tokens / 2, 24] pair-packed alias";
  if (pair_packed_beta_tma) {
    const TensorByteRange beta_range = GetTensorByteRange(beta, "beta");
    const TensorByteRange beta_tma_range = GetTensorByteRange(beta_tma, "beta_tma");
    TVM_FFI_ICHECK(beta_range.begin == beta_tma_range.begin && beta_range.end == beta_tma_range.end)
        << "pair-packed beta_tma must exactly alias beta storage";
  } else {
    CheckNoPartialOverlapOrExactAlias(beta, "beta", beta_tma, "beta_tma");
  }
  if (!direct_beta_tma) {
    CheckNoOverlap(beta_tma, "beta_tma", q, "q");
    CheckNoOverlap(beta_tma, "beta_tma", k, "k");
    CheckNoOverlap(beta_tma, "beta_tma", v, "v");
    CheckNoOverlap(beta_tma, "beta_tma", g, "g");
    CheckNoOverlap(beta_tma, "beta_tma", A_log, "A_log");
    CheckNoOverlap(beta_tma, "beta_tma", dt_bias, "dt_bias");
    CheckNoOverlap(beta_tma, "beta_tma", cu_seqlens, "cu_seqlens");
    CheckNoOverlap(beta_tma, "beta_tma", seq_order, "seq_order");
  }
  TVM_FFI_ICHECK(A_log.numel() == num_heads) << "A_log must contain H elements";
  TVM_FFI_ICHECK(dt_bias.numel() == num_heads * kHeadDim)
      << "dt_bias must contain H * 128 elements";
  TVM_FFI_ICHECK(cu_seqlens.ndim() == 1 && cu_seqlens.numel() >= 2)
      << "cu_seqlens must be a one-dimensional tensor with at least two "
         "elements";

  const int64_t num_seqs = cu_seqlens.numel() - 1;
  TVM_FFI_ICHECK(seq_order.ndim() == 1 && seq_order.numel() == num_seqs)
      << "seq_order must contain one int32 entry per sequence";
  const int64_t active_state_slots =
      allow_serving_layouts && state_pool_slots > 0 ? state_pool_slots : num_seqs;
  const int64_t state_numel = active_state_slots * num_heads * kHeadDim * kHeadDim;
  if (use_initial_state != 0) {
    TVM_FFI_ICHECK(initial_state.numel() == state_numel)
        << "initial_state must have flattened [N, H, 128, 128] size";
  }
  if (store_final_state != 0) {
    TVM_FFI_ICHECK(final_state.numel() == state_numel)
        << "final_state must have flattened [N, H, 128, 128] size";
  }
  CheckNoOverlap(out, "out", q, "q");
  CheckNoOverlap(out, "out", k, "k");
  CheckNoOverlap(out, "out", v, "v");
  CheckNoOverlap(out, "out", g, "g");
  CheckNoOverlap(out, "out", beta, "beta");
  CheckNoOverlap(out, "out", beta_tma, "beta_tma");
  CheckNoOverlap(out, "out", A_log, "A_log");
  CheckNoOverlap(out, "out", dt_bias, "dt_bias");
  CheckNoOverlap(out, "out", cu_seqlens, "cu_seqlens");
  CheckNoOverlap(out, "out", seq_order, "seq_order");
  CheckNoOverlap(descriptor_storage, "descriptor_storage", q, "q");
  CheckNoOverlap(descriptor_storage, "descriptor_storage", k, "k");
  CheckNoOverlap(descriptor_storage, "descriptor_storage", v, "v");
  CheckNoOverlap(descriptor_storage, "descriptor_storage", g, "g");
  CheckNoOverlap(descriptor_storage, "descriptor_storage", beta, "beta");
  CheckNoOverlap(descriptor_storage, "descriptor_storage", beta_tma, "beta_tma");
  CheckNoOverlap(descriptor_storage, "descriptor_storage", A_log, "A_log");
  CheckNoOverlap(descriptor_storage, "descriptor_storage", dt_bias, "dt_bias");
  CheckNoOverlap(descriptor_storage, "descriptor_storage", cu_seqlens, "cu_seqlens");
  CheckNoOverlap(descriptor_storage, "descriptor_storage", seq_order, "seq_order");
  CheckNoOverlap(descriptor_storage, "descriptor_storage", out, "out");
  if (use_initial_state != 0) {
    CheckNoOverlap(initial_state, "initial_state", q, "q");
    CheckNoOverlap(initial_state, "initial_state", k, "k");
    CheckNoOverlap(initial_state, "initial_state", v, "v");
    CheckNoOverlap(initial_state, "initial_state", g, "g");
    CheckNoOverlap(initial_state, "initial_state", beta, "beta");
    CheckNoOverlap(initial_state, "initial_state", beta_tma, "beta_tma");
    CheckNoOverlap(initial_state, "initial_state", A_log, "A_log");
    CheckNoOverlap(initial_state, "initial_state", dt_bias, "dt_bias");
    CheckNoOverlap(initial_state, "initial_state", cu_seqlens, "cu_seqlens");
    CheckNoOverlap(initial_state, "initial_state", seq_order, "seq_order");
    CheckNoOverlap(out, "out", initial_state, "initial_state");
    CheckNoOverlap(descriptor_storage, "descriptor_storage", initial_state, "initial_state");
  }
  if (store_final_state != 0) {
    CheckNoOverlap(out, "out", final_state, "final_state");
    CheckNoOverlap(descriptor_storage, "descriptor_storage", final_state, "final_state");
    CheckNoOverlap(final_state, "final_state", q, "q");
    CheckNoOverlap(final_state, "final_state", k, "k");
    CheckNoOverlap(final_state, "final_state", v, "v");
    CheckNoOverlap(final_state, "final_state", g, "g");
    CheckNoOverlap(final_state, "final_state", beta, "beta");
    CheckNoOverlap(final_state, "final_state", beta_tma, "beta_tma");
    CheckNoOverlap(final_state, "final_state", A_log, "A_log");
    CheckNoOverlap(final_state, "final_state", dt_bias, "dt_bias");
    CheckNoOverlap(final_state, "final_state", cu_seqlens, "cu_seqlens");
    CheckNoOverlap(final_state, "final_state", seq_order, "seq_order");
    if (use_initial_state != 0) {
      CheckNoPartialOverlapOrExactAlias(initial_state, "initial_state", final_state, "final_state");
    }
  }
  return num_seqs;
}

inline int64_t ResolveAndCheckServingStatePool(
    const TensorView& state_indices, const TensorView& initial_state, const TensorView& final_state,
    int32_t device_id, int64_t num_seqs, int64_t num_heads, int64_t state_slot_stride,
    int64_t use_state_indices, int64_t use_initial_state, int64_t store_final_state) {
  TVM_FFI_ICHECK(use_state_indices == 0 || use_state_indices == 1)
      << "use_state_indices must be 0 or 1, got " << use_state_indices;
  const int64_t compact_slot_stride = num_heads * kHeadDim * kHeadDim;
  TVM_FFI_ICHECK(state_slot_stride >= compact_slot_stride)
      << "state_slot_stride must cover one [H, 128, 128] state";
  if (use_state_indices == 0) {
    TVM_FFI_ICHECK(state_slot_stride == compact_slot_stride)
        << "packed sequence-ordered state requires its compact slot stride";
    return 0;
  }

  CheckCudaTensor(state_indices, "state_indices", device_id);
  CheckDtype(state_indices, "state_indices", dl_int32);
  TVM_FFI_ICHECK(state_indices.ndim() == 1 && state_indices.numel() == num_seqs)
      << "state_indices must contain one int32 slot per sequence";

  int64_t pool_slots = 0;
  struct NamedState {
    const TensorView* tensor;
    const char* name;
    bool active;
  };
  for (const NamedState& named :
       {NamedState{&initial_state, "initial_state", use_initial_state != 0},
        NamedState{&final_state, "final_state", store_final_state != 0}}) {
    if (!named.active) {
      continue;
    }
    const TensorView& state = *named.tensor;
    TVM_FFI_ICHECK(state.ndim() == 4 && state.size(1) == num_heads && state.size(2) == kHeadDim &&
                   state.size(3) == kHeadDim)
        << named.name << " pool must have shape [N_pool, H, 128, 128]";
    TVM_FFI_ICHECK(state.stride(3) == 1 && state.stride(2) == kHeadDim &&
                   state.stride(1) == kHeadDim * kHeadDim && state.stride(0) == state_slot_stride)
        << named.name << " pool must be contiguous inside each slot and match state_slot_stride";
    TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(state.data_ptr()) % 16 == 0 &&
                   state_slot_stride * sizeof(__nv_bfloat16) % 16 == 0)
        << named.name << " pool base and slot stride must be 16-byte aligned";
    if (pool_slots == 0) {
      pool_slots = state.size(0);
    } else {
      TVM_FFI_ICHECK(state.size(0) == pool_slots)
          << "initial_state and final_state pools must have the same slot count";
    }
  }
  TVM_FFI_ICHECK(pool_slots > 0)
      << "indexed state requires an active initial_state or final_state pool";
  return pool_slots;
}

inline void CheckServingCheckpointInputs(const TensorView& state_checkpoints,
                                         const TensorView& checkpoint_cu_starts, int32_t device_id,
                                         int64_t num_seqs, int64_t num_heads,
                                         int64_t checkpoint_every_n_tokens,
                                         int64_t checkpoint_token_granularity = 32) {
  TVM_FFI_ICHECK(checkpoint_token_granularity > 0)
      << "checkpoint_token_granularity must be positive";
  TVM_FFI_ICHECK(checkpoint_every_n_tokens >= 0 &&
                 checkpoint_every_n_tokens <= std::numeric_limits<int32_t>::max() &&
                 checkpoint_every_n_tokens % checkpoint_token_granularity == 0)
      << "checkpoint_every_n_tokens must be zero or a multiple of " << checkpoint_token_granularity;
  if (checkpoint_every_n_tokens == 0) {
    return;
  }
  CheckCudaTensor(state_checkpoints, "state_checkpoints", device_id);
  CheckCudaTensor(checkpoint_cu_starts, "checkpoint_cu_starts", device_id);
  CheckDtype(state_checkpoints, "state_checkpoints", dl_bfloat16);
  CheckDtype(checkpoint_cu_starts, "checkpoint_cu_starts", dl_int64);
  TVM_FFI_ICHECK(state_checkpoints.ndim() == 4 && state_checkpoints.size(0) > 0 &&
                 state_checkpoints.size(1) == num_heads && state_checkpoints.size(2) == kHeadDim &&
                 state_checkpoints.size(3) == kHeadDim)
      << "state_checkpoints must have shape [C, H, 128, 128]";
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(state_checkpoints.data_ptr()) % 16 == 0)
      << "state_checkpoints base address must be 16-byte aligned";
  TVM_FFI_ICHECK(checkpoint_cu_starts.ndim() == 1 && checkpoint_cu_starts.numel() == num_seqs + 1)
      << "checkpoint_cu_starts must have shape [N+1]";
}

inline void CheckServingAuxiliaryNoOverlap(
    const TensorView& state_indices, const TensorView& state_checkpoints,
    const TensorView& checkpoint_cu_starts, const TensorView& q, const TensorView& k,
    const TensorView& v, const TensorView& g, const TensorView& beta, const TensorView& beta_tma,
    const TensorView& A_log, const TensorView& dt_bias, const TensorView& cu_seqlens,
    const TensorView& seq_order, const TensorView& initial_state, const TensorView& out,
    const TensorView& final_state, const TensorView& descriptor_storage, int64_t use_state_indices,
    int64_t checkpoint_every_n_tokens) {
  const std::array<std::pair<const TensorView*, const char*>, 14> common = {{
      {&q, "q"},
      {&k, "k"},
      {&v, "v"},
      {&g, "g"},
      {&beta, "beta"},
      {&beta_tma, "beta_tma"},
      {&A_log, "A_log"},
      {&dt_bias, "dt_bias"},
      {&cu_seqlens, "cu_seqlens"},
      {&seq_order, "seq_order"},
      {&initial_state, "initial_state"},
      {&out, "out"},
      {&final_state, "final_state"},
      {&descriptor_storage, "descriptor_storage"},
  }};
  const auto check_against_common = [&](const TensorView& auxiliary, const char* auxiliary_name) {
    for (const auto& named : common) {
      CheckNoOverlap(auxiliary, auxiliary_name, *named.first, named.second);
    }
  };
  if (use_state_indices != 0) {
    check_against_common(state_indices, "state_indices");
  }
  if (checkpoint_every_n_tokens != 0) {
    check_against_common(state_checkpoints, "state_checkpoints");
    check_against_common(checkpoint_cu_starts, "checkpoint_cu_starts");
    CheckNoOverlap(state_checkpoints, "state_checkpoints", checkpoint_cu_starts,
                   "checkpoint_cu_starts");
    if (use_state_indices != 0) {
      CheckNoOverlap(state_indices, "state_indices", state_checkpoints, "state_checkpoints");
      CheckNoOverlap(state_indices, "state_indices", checkpoint_cu_starts, "checkpoint_cu_starts");
    }
  }
}

inline void PackBetaForTmaIfNeeded(const TensorView& beta, const TensorView& beta_tma,
                                   int64_t num_heads, int64_t beta_token_stride,
                                   cudaStream_t stream) {
  // TMA may require separate storage for either token or head padding. Head
  // counts can therefore match even though beta_tma is an uninitialized
  // workspace; only an exact pointer alias is already populated.
  const int64_t padded_num_heads = beta_tma.size(beta_tma.ndim() - 1);
  if (beta_tma.data_ptr() == beta.data_ptr()) {
    return;
  }
  const int64_t token_count = beta.numel() / num_heads;
  const int64_t padded_elements = beta_tma.numel();
  constexpr int32_t kThreads = 256;
  const int64_t blocks_i64 = (padded_elements - 1) / kThreads + 1;
  TVM_FFI_ICHECK(blocks_i64 > 0 && blocks_i64 <= std::numeric_limits<uint32_t>::max())
      << "beta TMA pack grid.x is out of range: " << blocks_i64;
  PackBetaForTmaKernel<<<static_cast<uint32_t>(blocks_i64), kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(beta.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(beta_tma.data_ptr()), token_count, padded_elements,
      num_heads, padded_num_heads, beta_token_stride);
  CheckCuda(cudaGetLastError(), "PackBetaForTmaKernel launch");
}

template <int ChunkTokens>
inline CUtensorMap EncodeQkTma(const TensorView& tensor, const char* name) {
  static_assert(ChunkTokens == 16 || ChunkTokens == 32);
  TVM_FFI_ICHECK(tensor.ndim() >= 2) << name << " must have at least two dimensions";
  const int64_t d1 = tensor.size(tensor.ndim() - 1);
  const int64_t d2 = tensor.size(tensor.ndim() - 2);
  TVM_FFI_ICHECK(d1 > 0 && d2 > 0 && d1 % 64 == 0) << name << " has invalid trailing dimensions";
  const int64_t outer2 = tensor.numel() / (d1 * d2);
  uint64_t global_dim[4] = {64, static_cast<uint64_t>(outer2), static_cast<uint64_t>(d2),
                            static_cast<uint64_t>(d1 / 64)};
  TVM_FFI_ICHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] >= 1 && global_dim[3] >= 2)
      << name << " cannot encode the (64, " << ChunkTokens << ", 1, 2) TMA box";
  uint64_t global_strides[3] = {static_cast<uint64_t>(d2 * d1 * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(d1 * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(64 * sizeof(__nv_bfloat16))};
  uint32_t box_dim[4] = {64, ChunkTokens, 1, 2};
  uint32_t elem_strides[4] = {1, 1, 1, 1};
  CUtensorMap tensor_map{};
  const CUresult result =
      cuTensorMapEncodeTiled(&tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, tensor.data_ptr(),
                             global_dim, global_strides, box_dim, elem_strides,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                             CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for " << name << " with CUresult=" << int(result);
  return tensor_map;
}

template <int ValueRows, int ChunkTokens>
inline CUtensorMap EncodeValueTma(const TensorView& tensor) {
  static_assert(ValueRows == 64 || ValueRows == 128);
  static_assert(ChunkTokens == 16 || ChunkTokens == 32);
  const int64_t d1 = tensor.size(tensor.ndim() - 1);
  const int64_t d2 = tensor.size(tensor.ndim() - 2);
  const int64_t outer2 = tensor.numel() / (d1 * d2);
  if constexpr (ValueRows == 128 && ChunkTokens == 16) {
    TVM_FFI_ICHECK(d1 >= ValueRows && d1 % 64 == 0 && d2 >= 1 && outer2 > 0)
        << "v cannot encode the (64, 1, 16, 1) split-panel TMA box";
    uint64_t global_dim[4] = {64, static_cast<uint64_t>(d2), static_cast<uint64_t>(outer2),
                              static_cast<uint64_t>(d1 / 64)};
    uint64_t global_strides[3] = {static_cast<uint64_t>(d1 * sizeof(__nv_bfloat16)),
                                  static_cast<uint64_t>(d1 * d2 * sizeof(__nv_bfloat16)),
                                  static_cast<uint64_t>(64 * sizeof(__nv_bfloat16))};
    uint32_t box_dim[4] = {64, 1, ChunkTokens, 1};
    uint32_t elem_strides[4] = {1, 1, 1, 1};
    CUtensorMap tensor_map{};
    const CUresult result =
        cuTensorMapEncodeTiled(&tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, tensor.data_ptr(),
                               global_dim, global_strides, box_dim, elem_strides,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TVM_FFI_ICHECK(result == CUDA_SUCCESS)
        << "cuTensorMapEncodeTiled failed for split-panel v with CUresult=" << int(result);
    return tensor_map;
  }
  uint64_t global_dim[3] = {static_cast<uint64_t>(d1), static_cast<uint64_t>(d2),
                            static_cast<uint64_t>(outer2)};
  TVM_FFI_ICHECK(global_dim[0] >= ValueRows && global_dim[1] >= 1 && global_dim[2] > 0)
      << "v cannot encode the (" << ValueRows << ", 1, " << ChunkTokens << ") TMA box";
  uint64_t global_strides[2] = {static_cast<uint64_t>(d1 * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(d1 * d2 * sizeof(__nv_bfloat16))};
  uint32_t box_dim[3] = {ValueRows, 1, ChunkTokens};
  uint32_t elem_strides[3] = {1, 1, 1};
  CUtensorMap tensor_map{};
  constexpr CUtensorMapSwizzle swizzle =
      ValueRows == 64 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE;
  const CUresult result = cuTensorMapEncodeTiled(
      &tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, tensor.data_ptr(), global_dim,
      global_strides, box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for v with CUresult=" << int(result);
  return tensor_map;
}

template <int ChunkTokens>
inline CUtensorMap EncodeGateTma(const TensorView& tensor) {
  static_assert(ChunkTokens == 16 || ChunkTokens == 32);
  const int64_t d1 = tensor.size(tensor.ndim() - 1);
  const int64_t d2 = tensor.size(tensor.ndim() - 2);
  const int64_t outer2 = tensor.numel() / (d1 * d2);
  uint64_t global_dim[3] = {static_cast<uint64_t>(d1), static_cast<uint64_t>(d2),
                            static_cast<uint64_t>(outer2)};
  TVM_FFI_ICHECK(global_dim[0] >= 128 && global_dim[1] >= 1 && global_dim[2] > 0)
      << "g cannot encode the (128, 1, " << ChunkTokens << ") TMA box";
  uint64_t global_strides[2] = {static_cast<uint64_t>(d1 * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(d1 * d2 * sizeof(__nv_bfloat16))};
  uint32_t box_dim[3] = {128, 1, ChunkTokens};
  uint32_t elem_strides[3] = {1, 1, 1};
  CUtensorMap tensor_map{};
  const CUresult result =
      cuTensorMapEncodeTiled(&tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, tensor.data_ptr(),
                             global_dim, global_strides, box_dim, elem_strides,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                             CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for g with CUresult=" << int(result);
  return tensor_map;
}

template <int ChunkTokens, bool PairPacked = false>
inline CUtensorMap EncodeBetaTma(const TensorView& tensor) {
  static_assert(ChunkTokens == 16 || ChunkTokens == 32);
  static_assert(!PairPacked || ChunkTokens == 32);
  constexpr uint32_t kBoxHeads = PairPacked ? 24 : 8;
  constexpr uint32_t kBoxTokens = PairPacked ? ChunkTokens / 2 + 1 : ChunkTokens;
  const int64_t d1 = tensor.size(tensor.ndim() - 1);
  const int64_t outer1 = tensor.numel() / d1;
  uint64_t global_dim[2] = {static_cast<uint64_t>(d1), static_cast<uint64_t>(outer1)};
  TVM_FFI_ICHECK(global_dim[0] >= kBoxHeads && global_dim[1] >= kBoxTokens)
      << "beta_tma cannot encode the (" << kBoxHeads << ", " << kBoxTokens << ") TMA box";
  TVM_FFI_ICHECK(tensor.stride(tensor.ndim() - 1) == 1 && tensor.stride(tensor.ndim() - 2) >= d1)
      << "beta_tma must have unit head stride and non-overlapping token rows";
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0)
      << "beta_tma base address must be 16-byte aligned";
  TVM_FFI_ICHECK(tensor.stride(tensor.ndim() - 2) * sizeof(__nv_bfloat16) % 16 == 0)
      << "beta_tma token stride must be a multiple of 16 bytes";
  uint64_t global_strides[1] = {
      static_cast<uint64_t>(tensor.stride(tensor.ndim() - 2) * sizeof(__nv_bfloat16))};
  uint32_t box_dim[2] = {kBoxHeads, kBoxTokens};
  uint32_t elem_strides[2] = {1, 1};
  CUtensorMap tensor_map{};
  const CUresult result =
      cuTensorMapEncodeTiled(&tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, tensor.data_ptr(),
                             global_dim, global_strides, box_dim, elem_strides,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                             CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for beta_tma with CUresult=" << int(result);
  return tensor_map;
}

template <int ValueRows, int ChunkTokens>
inline CUtensorMap EncodeOutputTma(const TensorView& tensor) {
  static_assert(ValueRows == 64 || ValueRows == 128);
  static_assert(ChunkTokens == 16 || ChunkTokens == 32);
  const int64_t d1 = tensor.size(tensor.ndim() - 1);
  const int64_t d2 = tensor.size(tensor.ndim() - 2);
  const int64_t outer2 = tensor.numel() / (d1 * d2);
  TVM_FFI_ICHECK(d1 > 0 && d2 > 0 && d1 % 64 == 0) << "out has invalid trailing dimensions";
  uint64_t global_dim[4] = {64, static_cast<uint64_t>(outer2), static_cast<uint64_t>(d2),
                            static_cast<uint64_t>(d1 / 64)};
  constexpr uint32_t value_splits = ValueRows / 64;
  TVM_FFI_ICHECK(global_dim[0] >= 64 && global_dim[1] > 0 && global_dim[2] >= 1 &&
                 global_dim[3] >= value_splits)
      << "out cannot encode the (64, " << ChunkTokens << ", 1, " << value_splits << ") TMA box";
  uint64_t global_strides[3] = {static_cast<uint64_t>(d2 * d1 * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(d1 * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(64 * sizeof(__nv_bfloat16))};
  uint32_t box_dim[4] = {64, ChunkTokens, 1, value_splits};
  uint32_t elem_strides[4] = {1, 1, 1, 1};
  CUtensorMap tensor_map{};
  const CUresult result =
      cuTensorMapEncodeTiled(&tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, tensor.data_ptr(),
                             global_dim, global_strides, box_dim, elem_strides,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                             CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for out with CUresult=" << int(result);
  return tensor_map;
}

struct TmaPointers {
  void* q;
  void* k;
  void* v;
  void* g;
  void* beta;
  void* out;
};

struct TensorMapWords {
  static constexpr size_t kWordCount = kDescriptorStorageBytes / sizeof(uint64_t);
  uint64_t words[kWordCount];
};

static __global__ void PublishTensorMaps(uint64_t* destination, TensorMapWords source) {
  const uint32_t index = threadIdx.x;
  if (index < TensorMapWords::kWordCount) {
    destination[index] = source.words[index];
  }
}

template <int ValueRows, int ChunkTokens = 32, bool PairPackedBeta = false>
inline TmaPointers EncodeTmaPointers(const TensorView& q, const TensorView& k, const TensorView& v,
                                     const TensorView& g, const TensorView& beta_tma,
                                     const TensorView& out, const TensorView& descriptor_storage,
                                     int64_t prepare_descriptors, cudaStream_t stream) {
  if (prepare_descriptors != 0) {
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    CheckCuda(cudaStreamIsCapturing(stream, &capture_status),
              "cudaStreamIsCapturing(TMA descriptor preparation)");
    TVM_FFI_ICHECK(capture_status == cudaStreamCaptureStatusNone)
        << "prepare_descriptors must be 0 during CUDA graph capture; warm "
           "this exact workspace and tensor signature before capture";

    const std::array<CUtensorMap, kTensorMapCount> host_maps = {
        EncodeQkTma<ChunkTokens>(q, "q"),
        EncodeQkTma<ChunkTokens>(k, "k"),
        EncodeValueTma<ValueRows, ChunkTokens>(v),
        EncodeGateTma<ChunkTokens>(g),
        EncodeBetaTma<ChunkTokens, PairPackedBeta>(beta_tma),
        EncodeOutputTma<ValueRows, ChunkTokens>(out),
    };
    static_assert(sizeof(host_maps) == kDescriptorStorageBytes);
    TensorMapWords words{};
    std::memcpy(words.words, host_maps.data(), sizeof(host_maps));
    PublishTensorMaps<<<1, 128, 0, stream>>>(
        reinterpret_cast<uint64_t*>(descriptor_storage.data_ptr()), words);
    CheckCuda(cudaGetLastError(), "PublishTensorMaps launch");
  }

  auto* bytes = static_cast<unsigned char*>(descriptor_storage.data_ptr());
  constexpr size_t stride = sizeof(CUtensorMap);
  return {
      bytes + 0 * stride, bytes + 1 * stride, bytes + 2 * stride,
      bytes + 3 * stride, bytes + 4 * stride, bytes + 5 * stride,
  };
}

inline void CheckDynamicSmemCapacity(int32_t device_id, int32_t smem_bytes) {
  int max_optin = 0;
  CheckCuda(cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id),
            "cudaDeviceGetAttribute(max dynamic shared memory)");
  TVM_FFI_ICHECK(max_optin >= smem_bytes)
      << "device exposes only " << max_optin << " bytes of opt-in shared memory; FlashKDA requires "
      << smem_bytes;
}

}  // namespace flash_kda
}  // namespace flashinfer
