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

#include "flashkda_binding_common.cuh"

namespace flashinfer {
namespace flash_kda {

constexpr int64_t kBt16ChunkTokens = 16;
constexpr int64_t kBt16ValueSplits = 2;
constexpr size_t kBt16TensorMapCount = 7;
constexpr size_t kBt16DescriptorStorageBytes = kBt16TensorMapCount * sizeof(CUtensorMap);

inline void CheckBt16DescriptorStorage(const TensorView& storage, int32_t device_id,
                                       int64_t prepare_descriptors) {
  TVM_FFI_ICHECK(prepare_descriptors == 0 || prepare_descriptors == 1)
      << "prepare_descriptors must be 0 or 1, got " << prepare_descriptors;
  CheckCudaTensor(storage, "descriptor_storage", device_id);
  CheckDtype(storage, "descriptor_storage", dl_uint8);
  TVM_FFI_ICHECK(storage.numel() >= static_cast<int64_t>(kBt16DescriptorStorageBytes))
      << "descriptor_storage must contain at least " << kBt16DescriptorStorageBytes << " bytes";
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(storage.data_ptr()) % kTensorMapAlignment == 0)
      << "descriptor_storage must be aligned to " << kTensorMapAlignment << " bytes";
}

inline void CheckBt16DenseTensor(const TensorView& tensor, const char* name, int32_t device_id,
                                 DLDataType dtype) {
  CheckCudaTensor(tensor, name, device_id);
  CheckDtype(tensor, name, dtype);
}

inline void CheckBt16TokenTensor(const TensorView& tensor, const char* name, int32_t device_id,
                                 int64_t token_count, int64_t num_heads) {
  CheckBt16DenseTensor(tensor, name, device_id, dl_bfloat16);
  TVM_FFI_ICHECK(tensor.ndim() >= 3 && tensor.size(tensor.ndim() - 2) == num_heads &&
                 tensor.size(tensor.ndim() - 1) == kHeadDim &&
                 tensor.numel() == token_count * num_heads * kHeadDim)
      << name << " must match flattened [tokens, H, 128] storage";
}

inline int64_t CheckBt16PrepareInputs(
    const TensorView& q, const TensorView& k, const TensorView& raw_gate,
    const TensorView& beta_logits, const TensorView& a_log, const TensorView& dt_bias,
    const TensorView& cu_seqlens, const TensorView& cu_chunks, const TensorView& chunk_to_seq,
    const TensorView& ws_qd, const TensorView& ws_kd, const TensorView& ws_w,
    const TensorView& ws_qk_t, const TensorView& ws_diag, const TensorView& descriptor_storage,
    int64_t prepare_descriptors, int64_t prepare_total_ctas, int64_t total_chunks,
    int64_t num_heads, double gate_lower_bound) {
  TVM_FFI_ICHECK(q.device().device_type == kDLCUDA) << "q must be a CUDA tensor";
  const int32_t device_id = q.device().device_id;
  CheckFlashKDATarget(device_id);
  TVM_FFI_ICHECK(num_heads > 0 && num_heads <= std::numeric_limits<int32_t>::max())
      << "num_heads must be in the positive int32 range";
  TVM_FFI_ICHECK(total_chunks > 0 && total_chunks <= std::numeric_limits<int32_t>::max())
      << "total_chunks must be in the positive int32 range";
  TVM_FFI_ICHECK(prepare_total_ctas >= num_heads &&
                 prepare_total_ctas <= num_heads * total_chunks &&
                 prepare_total_ctas <= std::numeric_limits<uint32_t>::max())
      << "prepare_total_ctas must be in [H, H * total_chunks]";
  TVM_FFI_ICHECK(std::isfinite(gate_lower_bound) && gate_lower_bound < 0.0 &&
                 std::isfinite(static_cast<float>(gate_lower_bound)))
      << "gate_lower_bound must be finite, negative, and representable as float32";

  CheckBt16DenseTensor(q, "q", device_id, dl_bfloat16);
  TVM_FFI_ICHECK(q.ndim() >= 3 && q.size(q.ndim() - 2) == num_heads &&
                 q.size(q.ndim() - 1) == kHeadDim)
      << "q must have trailing [H, 128] dimensions";
  const int64_t token_count = q.numel() / (num_heads * kHeadDim);
  TVM_FFI_ICHECK(token_count > 0) << "q must contain at least one token";
  CheckBt16TokenTensor(k, "k", device_id, token_count, num_heads);
  CheckBt16TokenTensor(raw_gate, "raw_gate", device_id, token_count, num_heads);

  CheckBt16DenseTensor(beta_logits, "beta_logits", device_id, dl_bfloat16);
  TVM_FFI_ICHECK(beta_logits.ndim() >= 2 && beta_logits.size(beta_logits.ndim() - 1) == num_heads &&
                 beta_logits.numel() == token_count * num_heads)
      << "beta_logits must match flattened [tokens, H] storage";
  CheckBt16DenseTensor(a_log, "a_log", device_id, dl_float32);
  CheckBt16DenseTensor(dt_bias, "dt_bias", device_id, dl_float32);
  TVM_FFI_ICHECK(a_log.numel() == num_heads) << "a_log must contain H elements";
  TVM_FFI_ICHECK(dt_bias.numel() == num_heads * kHeadDim)
      << "dt_bias must contain H * 128 elements";
  CheckBt16DenseTensor(cu_seqlens, "cu_seqlens", device_id, dl_int64);
  CheckBt16DenseTensor(cu_chunks, "cu_chunks", device_id, dl_int32);
  CheckBt16DenseTensor(chunk_to_seq, "chunk_to_seq", device_id, dl_int32);
  TVM_FFI_ICHECK(cu_seqlens.ndim() == 1 && cu_seqlens.numel() >= 2)
      << "cu_seqlens must contain N + 1 entries";
  TVM_FFI_ICHECK(cu_chunks.ndim() == 1 && cu_chunks.numel() == cu_seqlens.numel())
      << "cu_chunks must contain N + 1 entries";
  TVM_FFI_ICHECK(chunk_to_seq.ndim() == 1 && chunk_to_seq.numel() == total_chunks)
      << "chunk_to_seq must contain total_chunks entries";

  const int64_t padded_tokens = total_chunks * kBt16ChunkTokens;
  for (const auto& named : {std::pair<const TensorView*, const char*>(&ws_qd, "ws_qd"),
                            std::pair<const TensorView*, const char*>(&ws_kd, "ws_kd"),
                            std::pair<const TensorView*, const char*>(&ws_w, "ws_w")}) {
    CheckBt16DenseTensor(*named.first, named.second, device_id, dl_bfloat16);
    TVM_FFI_ICHECK(named.first->ndim() == 4 && named.first->size(0) == 1 &&
                   named.first->size(1) == num_heads && named.first->size(2) == padded_tokens &&
                   named.first->size(3) == kHeadDim)
        << named.second << " must have shape [1, H, total_chunks * 16, 128]";
  }
  CheckBt16DenseTensor(ws_qk_t, "ws_qk_t", device_id, dl_bfloat16);
  TVM_FFI_ICHECK(ws_qk_t.ndim() == 5 && ws_qk_t.size(0) == 1 && ws_qk_t.size(1) == num_heads &&
                 ws_qk_t.size(2) == total_chunks && ws_qk_t.size(3) == kBt16ChunkTokens &&
                 ws_qk_t.size(4) == kBt16ChunkTokens)
      << "ws_qk_t must have shape [1, H, total_chunks, 16, 16]";
  CheckBt16DenseTensor(ws_diag, "ws_diag", device_id, dl_float32);
  TVM_FFI_ICHECK(ws_diag.ndim() == 4 && ws_diag.size(0) == 1 && ws_diag.size(1) == num_heads &&
                 ws_diag.size(2) == total_chunks && ws_diag.size(3) == kHeadDim)
      << "ws_diag must have shape [1, H, total_chunks, 128]";
  CheckBt16DescriptorStorage(descriptor_storage, device_id, prepare_descriptors);
  return token_count;
}

inline int64_t CheckBt16ChainInputs(
    const TensorView& ws_qd, const TensorView& ws_kd, const TensorView& ws_w,
    const TensorView& ws_qk, const TensorView& ws_diag, const TensorView& v,
    const TensorView& cu_seqlens, const TensorView& cu_chunks, const TensorView& seq_order,
    const TensorView& initial_state, const TensorView& out, const TensorView& final_state,
    const TensorView& descriptor_storage, int64_t prepare_descriptors, int64_t num_heads,
    int64_t use_initial_state, int64_t store_final_state, double scale) {
  TVM_FFI_ICHECK(v.device().device_type == kDLCUDA) << "v must be a CUDA tensor";
  const int32_t device_id = v.device().device_id;
  CheckFlashKDATarget(device_id);
  TVM_FFI_ICHECK(num_heads > 0 && num_heads <= std::numeric_limits<int32_t>::max())
      << "num_heads must be in the positive int32 range";
  TVM_FFI_ICHECK(use_initial_state == 0 || use_initial_state == 1)
      << "use_initial_state must be 0 or 1";
  TVM_FFI_ICHECK(store_final_state == 0 || store_final_state == 1)
      << "store_final_state must be 0 or 1";
  TVM_FFI_ICHECK(std::isfinite(scale) && std::isfinite(static_cast<float>(scale)))
      << "scale must be finite and representable as float32";

  CheckBt16DenseTensor(cu_seqlens, "cu_seqlens", device_id, dl_int64);
  CheckBt16DenseTensor(cu_chunks, "cu_chunks", device_id, dl_int32);
  CheckBt16DenseTensor(seq_order, "seq_order", device_id, dl_int32);
  TVM_FFI_ICHECK(cu_seqlens.ndim() == 1 && cu_seqlens.numel() >= 2)
      << "cu_seqlens must contain N + 1 entries";
  const int64_t num_seqs = cu_seqlens.numel() - 1;
  TVM_FFI_ICHECK(cu_chunks.ndim() == 1 && cu_chunks.numel() == num_seqs + 1)
      << "cu_chunks must contain N + 1 entries";
  TVM_FFI_ICHECK(seq_order.ndim() == 1 && seq_order.numel() == num_seqs)
      << "seq_order must contain N entries";

  TVM_FFI_ICHECK(ws_qd.ndim() == 4 && ws_qd.size(0) == 1 && ws_qd.size(1) == num_heads &&
                 ws_qd.size(3) == kHeadDim && ws_qd.size(2) % kBt16ChunkTokens == 0)
      << "ws_qd must have shape [1, H, total_chunks * 16, 128]";
  const int64_t total_chunks = ws_qd.size(2) / kBt16ChunkTokens;
  TVM_FFI_ICHECK(total_chunks > 0) << "BT16 workspaces must contain at least one chunk";
  const int64_t padded_tokens = total_chunks * kBt16ChunkTokens;
  for (const auto& named : {std::pair<const TensorView*, const char*>(&ws_qd, "ws_qd"),
                            std::pair<const TensorView*, const char*>(&ws_kd, "ws_kd"),
                            std::pair<const TensorView*, const char*>(&ws_w, "ws_w")}) {
    CheckBt16DenseTensor(*named.first, named.second, device_id, dl_bfloat16);
    TVM_FFI_ICHECK(named.first->ndim() == 4 && named.first->size(0) == 1 &&
                   named.first->size(1) == num_heads && named.first->size(2) == padded_tokens &&
                   named.first->size(3) == kHeadDim)
        << named.second << " has an incompatible factor-workspace shape";
  }
  CheckBt16DenseTensor(ws_qk, "ws_qk", device_id, dl_bfloat16);
  TVM_FFI_ICHECK(ws_qk.ndim() == 5 && ws_qk.size(0) == 1 && ws_qk.size(1) == num_heads &&
                 ws_qk.size(2) == total_chunks && ws_qk.size(3) == kBt16ChunkTokens &&
                 ws_qk.size(4) == kBt16ChunkTokens)
      << "ws_qk must have shape [1, H, total_chunks, 16, 16]";
  CheckBt16DenseTensor(ws_diag, "ws_diag", device_id, dl_float32);
  TVM_FFI_ICHECK(ws_diag.ndim() == 4 && ws_diag.size(0) == 1 && ws_diag.size(1) == num_heads &&
                 ws_diag.size(2) == total_chunks && ws_diag.size(3) == kHeadDim)
      << "ws_diag must have shape [1, H, total_chunks, 128]";

  CheckBt16DenseTensor(v, "v", device_id, dl_bfloat16);
  TVM_FFI_ICHECK(v.ndim() >= 3 && v.size(v.ndim() - 2) == num_heads &&
                 v.size(v.ndim() - 1) == kHeadDim)
      << "v must have trailing [H, 128] dimensions";
  const int64_t token_count = v.numel() / (num_heads * kHeadDim);
  CheckBt16TokenTensor(out, "out", device_id, token_count, num_heads);
  CheckBt16DenseTensor(initial_state, "initial_state", device_id, dl_bfloat16);
  CheckBt16DenseTensor(final_state, "final_state", device_id, dl_bfloat16);
  const int64_t state_numel = num_seqs * num_heads * kHeadDim * kHeadDim;
  if (use_initial_state != 0) {
    TVM_FFI_ICHECK(initial_state.numel() == state_numel)
        << "initial_state must have flattened [N, H, 128, 128] size";
  }
  if (store_final_state != 0) {
    TVM_FFI_ICHECK(final_state.numel() == state_numel)
        << "final_state must have flattened [N, H, 128, 128] size";
  }
  CheckBt16DescriptorStorage(descriptor_storage, device_id, prepare_descriptors);
  return num_seqs;
}

template <int BoxValueSplits>
inline CUtensorMap EncodeBt16FactorTma(const TensorView& tensor, const char* name) {
  static_assert(BoxValueSplits == 1 || BoxValueSplits == 2);
  const int64_t d1 = tensor.size(tensor.ndim() - 1);
  const int64_t d2 = tensor.size(tensor.ndim() - 2);
  const int64_t outer2 = tensor.numel() / (d1 * d2);
  uint64_t global_dim[4] = {64, static_cast<uint64_t>(d2), static_cast<uint64_t>(outer2),
                            static_cast<uint64_t>(d1 / 64)};
  uint64_t global_strides[3] = {static_cast<uint64_t>(d1 * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(d2 * d1 * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(64 * sizeof(__nv_bfloat16))};
  uint32_t box_dim[4] = {64, 16, 1, BoxValueSplits};
  uint32_t elem_strides[4] = {1, 1, 1, 1};
  CUtensorMap map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, tensor.data_ptr(), global_dim, global_strides,
      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for " << name << " with CUresult=" << int(result);
  return map;
}

inline CUtensorMap EncodeBt16QkWorkspaceTma(const TensorView& tensor) {
  uint64_t global_dim[5] = {
      static_cast<uint64_t>(tensor.size(4)), static_cast<uint64_t>(tensor.size(3)),
      static_cast<uint64_t>(tensor.size(2)), static_cast<uint64_t>(tensor.size(1)),
      static_cast<uint64_t>(tensor.size(0))};
  uint64_t global_strides[4] = {static_cast<uint64_t>(tensor.stride(3) * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(tensor.stride(2) * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(tensor.stride(1) * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(tensor.stride(0) * sizeof(__nv_bfloat16))};
  uint32_t box_dim[5] = {16, 16, 1, 1, 1};
  uint32_t elem_strides[5] = {1, 1, 1, 1, 1};
  CUtensorMap map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, tensor.data_ptr(), global_dim, global_strides,
      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_32B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for ws_qk with CUresult=" << int(result);
  return map;
}

inline CUtensorMap EncodeBt16DiagWorkspaceTma(const TensorView& tensor) {
  uint64_t global_dim[4] = {
      static_cast<uint64_t>(tensor.size(3)), static_cast<uint64_t>(tensor.size(2)),
      static_cast<uint64_t>(tensor.size(1)), static_cast<uint64_t>(tensor.size(0))};
  uint64_t global_strides[3] = {static_cast<uint64_t>(tensor.stride(2) * sizeof(float)),
                                static_cast<uint64_t>(tensor.stride(1) * sizeof(float)),
                                static_cast<uint64_t>(tensor.stride(0) * sizeof(float))};
  uint32_t box_dim[4] = {128, 1, 1, 1};
  uint32_t elem_strides[4] = {1, 1, 1, 1};
  CUtensorMap map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, tensor.data_ptr(), global_dim, global_strides,
      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for ws_diag with CUresult=" << int(result);
  return map;
}

struct Bt16TensorMapWords {
  static constexpr size_t kWordCount = kBt16DescriptorStorageBytes / sizeof(uint64_t);
  uint64_t words[kWordCount];
};

static __global__ void PublishBt16TensorMaps(uint64_t* destination, Bt16TensorMapWords source) {
  const uint32_t index = threadIdx.x;
  if (index < Bt16TensorMapWords::kWordCount) {
    destination[index] = source.words[index];
  }
}

inline void PublishBt16Maps(const std::array<CUtensorMap, kBt16TensorMapCount>& host_maps,
                            const TensorView& descriptor_storage, int64_t prepare_descriptors,
                            cudaStream_t stream) {
  if (prepare_descriptors == 0) {
    return;
  }
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  CheckCuda(cudaStreamIsCapturing(stream, &capture_status),
            "cudaStreamIsCapturing(BT16 TMA descriptor preparation)");
  TVM_FFI_ICHECK(capture_status == cudaStreamCaptureStatusNone)
      << "prepare_descriptors must be 0 during CUDA graph capture; warm this exact workspace "
         "and tensor signature before capture";
  Bt16TensorMapWords words{};
  static_assert(sizeof(words) == sizeof(host_maps));
  std::memcpy(words.words, host_maps.data(), sizeof(host_maps));
  PublishBt16TensorMaps<<<1, 128, 0, stream>>>(
      reinterpret_cast<uint64_t*>(descriptor_storage.data_ptr()), words);
  CheckCuda(cudaGetLastError(), "PublishBt16TensorMaps launch");
}

struct Bt16PrepareTmaPointers {
  void* q;
  void* k;
  void* raw_gate;
  void* beta_logits;
  void* ws_qd;
  void* ws_kd;
  void* ws_w;
};

template <bool UsesBetaTma>
inline Bt16PrepareTmaPointers EncodeBt16PrepareTma(
    const TensorView& q, const TensorView& k, const TensorView& raw_gate,
    const TensorView& beta_logits, const TensorView& ws_qd, const TensorView& ws_kd,
    const TensorView& ws_w, const TensorView& descriptor_storage, int64_t prepare_descriptors,
    cudaStream_t stream) {
  if (prepare_descriptors != 0) {
    const CUtensorMap raw_gate_map = EncodeGateTma<16>(raw_gate);
    CUtensorMap beta_map{};
    if constexpr (UsesBetaTma) {
      beta_map = EncodeBetaTma<16>(beta_logits);
    } else {
      // The scalar-beta frozen kernel only acquires the descriptor address; it
      // reads beta_logits directly and never consumes this tensor map. Keep the
      // seven-slot ABI stable with a valid descriptor so H < 8 does not need to
      // satisfy the beta-TMA box constraints.
      beta_map = raw_gate_map;
    }
    PublishBt16Maps({EncodeQkTma<16>(q, "q"), EncodeQkTma<16>(k, "k"), raw_gate_map, beta_map,
                     EncodeBt16FactorTma<2>(ws_qd, "ws_qd"), EncodeBt16FactorTma<2>(ws_kd, "ws_kd"),
                     EncodeBt16FactorTma<1>(ws_w, "ws_w")},
                    descriptor_storage, prepare_descriptors, stream);
  }
  auto* bytes = static_cast<unsigned char*>(descriptor_storage.data_ptr());
  constexpr size_t stride = sizeof(CUtensorMap);
  return {bytes + 0 * stride, bytes + 1 * stride, bytes + 2 * stride, bytes + 3 * stride,
          bytes + 4 * stride, bytes + 5 * stride, bytes + 6 * stride};
}

struct Bt16ChainTmaPointers {
  void* ws_qd;
  void* ws_kd;
  void* ws_w;
  void* ws_qk;
  void* ws_diag;
  void* v;
  void* out;
};

inline Bt16ChainTmaPointers EncodeBt16ChainTma(const TensorView& ws_qd, const TensorView& ws_kd,
                                               const TensorView& ws_w, const TensorView& ws_qk,
                                               const TensorView& ws_diag, const TensorView& v,
                                               const TensorView& out,
                                               const TensorView& descriptor_storage,
                                               int64_t prepare_descriptors, cudaStream_t stream) {
  if (prepare_descriptors != 0) {
    PublishBt16Maps({EncodeBt16FactorTma<2>(ws_qd, "ws_qd"), EncodeBt16FactorTma<2>(ws_kd, "ws_kd"),
                     EncodeBt16FactorTma<2>(ws_w, "ws_w"), EncodeBt16QkWorkspaceTma(ws_qk),
                     EncodeBt16DiagWorkspaceTma(ws_diag), EncodeValueTma<64, 16>(v),
                     EncodeOutputTma<64, 16>(out)},
                    descriptor_storage, prepare_descriptors, stream);
  }
  auto* bytes = static_cast<unsigned char*>(descriptor_storage.data_ptr());
  constexpr size_t stride = sizeof(CUtensorMap);
  return {bytes + 0 * stride, bytes + 1 * stride, bytes + 2 * stride, bytes + 3 * stride,
          bytes + 4 * stride, bytes + 5 * stride, bytes + 6 * stride};
}

}  // namespace flash_kda
}  // namespace flashinfer
