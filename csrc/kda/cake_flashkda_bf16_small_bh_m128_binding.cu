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

#include "flashkda_binding_common.cuh"

#define uint8_t flashkda_generated_uint8_t
#define uint16_t flashkda_generated_uint16_t
#define uint32_t flashkda_generated_uint32_t
#define uint64_t flashkda_generated_uint64_t
#define int32_t flashkda_generated_int32_t
#define int16_t flashkda_generated_int16_t
#define FlashKDATensorMap flashkda_generated_FlashKDATensorMap
#define FlashKDATensorMapPack flashkda_generated_FlashKDATensorMapPack
#define CUtensorMap flashkda_generated_CUtensorMap
#include "cake_flashkda_bf16_small_bh_m128.cu"
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
namespace flash_kda {

constexpr int64_t kSmallBHGroupSize = 8;
constexpr int64_t kSmallBHMaxTasks = 8;
constexpr int64_t kSmallBHRingStages = 35;
constexpr int64_t kSmallBHPacketRows = 123;
constexpr int64_t kSmallBHPacketElements = 128;
constexpr int64_t kSmallBHMinSequenceLength = 2048;
constexpr size_t kSmallBHTensorMapCount = 7;
constexpr size_t kSmallBHDescriptorStorageBytes = kSmallBHTensorMapCount * sizeof(CUtensorMap);

static_assert(THREADS == 1024);
static_assert(SMEM_TOTAL == 227328);

inline CUtensorMap EncodeSmallBHPacketTma(const TensorView& tensor) {
  TVM_FFI_ICHECK(tensor.ndim() == 2 && tensor.size(1) == kSmallBHPacketElements)
      << "packet_workspace must have shape [rows, 128]";
  TVM_FFI_ICHECK(tensor.stride(1) == 1 && tensor.stride(0) == kSmallBHPacketElements)
      << "packet_workspace must be contiguous";
  TVM_FFI_ICHECK(tensor.size(0) >= kSmallBHPacketRows)
      << "packet_workspace must contain at least one 123-row packet";
  uint64_t global_dim[2] = {static_cast<uint64_t>(kSmallBHPacketElements),
                            static_cast<uint64_t>(tensor.size(0))};
  uint64_t global_strides[1] = {static_cast<uint64_t>(tensor.stride(0) * sizeof(__nv_bfloat16))};
  uint32_t box_dim[2] = {static_cast<uint32_t>(kSmallBHPacketElements),
                         static_cast<uint32_t>(kSmallBHPacketRows)};
  uint32_t elem_strides[2] = {1, 1};
  CUtensorMap tensor_map{};
  const CUresult result =
      cuTensorMapEncodeTiled(&tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, tensor.data_ptr(),
                             global_dim, global_strides, box_dim, elem_strides,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                             CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for packet_workspace with CUresult=" << int(result);
  return tensor_map;
}

struct SmallBHPacketTensorMapWords {
  uint64_t words[sizeof(CUtensorMap) / sizeof(uint64_t)];
};

static __global__ void PublishSmallBHPacketTensorMap(uint64_t* destination,
                                                     SmallBHPacketTensorMapWords source) {
  const uint32_t index = threadIdx.x;
  if (index < sizeof(source.words) / sizeof(source.words[0])) {
    destination[index] = source.words[index];
  }
}

void RunSmallBHM128(TensorView q, TensorView k, TensorView v, TensorView g, TensorView beta,
                    TensorView beta_tma, TensorView A_log, TensorView dt_bias,
                    TensorView cu_seqlens, TensorView seq_order, TensorView initial_state,
                    TensorView out, TensorView final_state, TensorView descriptor_storage,
                    TensorView packet_workspace, TensorView packet_ready,
                    TensorView packet_consumed, TensorView helper_done, int64_t prepare_descriptors,
                    int64_t num_heads, int64_t use_initial_state, int64_t store_final_state,
                    double scale, double lower_bound, int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0) << "cuda_stream must be a non-negative stream handle";
  TVM_FFI_ICHECK(q.device().device_type == kDLCUDA) << "q must be a CUDA tensor";
  const int32_t device_id = q.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckFlashKDATarget(device_id);

  const int64_t num_seqs =
      CheckCommonInputs(q, k, v, g, beta, beta_tma, A_log, dt_bias, cu_seqlens, seq_order,
                        initial_state, out, final_state, descriptor_storage, prepare_descriptors,
                        num_heads, use_initial_state, store_final_state, scale, lower_bound);
  TVM_FFI_ICHECK(q.ndim() == 4 && q.size(0) == num_seqs)
      << "small-BH FlashKDA requires fixed [B, T, H, 128] layout";
  TVM_FFI_ICHECK(q.size(1) >= kSmallBHMinSequenceLength)
      << "small-BH FlashKDA requires at least 2048 tokens per fixed sequence";
  const int64_t total_tasks = num_seqs * num_heads;
  TVM_FFI_ICHECK(total_tasks > 0 && total_tasks <= kSmallBHMaxTasks &&
                 num_heads <= kSmallBHMaxTasks)
      << "small-BH FlashKDA requires 1..8 sequence/head tasks and at most 8 heads";

  int sm_count = 0;
  CheckCuda(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id),
            "cudaDeviceGetAttribute(multiprocessor count)");
  const int64_t grid_x_i64 = kSmallBHGroupSize * total_tasks;
  TVM_FFI_ICHECK(grid_x_i64 <= sm_count)
      << "small-BH FlashKDA requires all eight-CTA task groups to reside concurrently";

  CheckCudaTensor(packet_workspace, "packet_workspace", device_id);
  CheckDtype(packet_workspace, "packet_workspace", dl_bfloat16);
  const int64_t packet_slots = total_tasks * kSmallBHRingStages;
  TVM_FFI_ICHECK(packet_workspace.ndim() == 2 &&
                 packet_workspace.size(0) == packet_slots * kSmallBHPacketRows &&
                 packet_workspace.size(1) == kSmallBHPacketElements)
      << "packet_workspace has the wrong compact-ring shape";
  for (const auto& named :
       {std::pair<const TensorView*, const char*>(&packet_ready, "packet_ready"),
        std::pair<const TensorView*, const char*>(&packet_consumed, "packet_consumed"),
        std::pair<const TensorView*, const char*>(&helper_done, "helper_done")}) {
    CheckCudaTensor(*named.first, named.second, device_id);
    CheckDtype(*named.first, named.second, dl_uint32);
    TVM_FFI_ICHECK(named.first->ndim() == 1) << named.second << " must be one-dimensional";
  }
  TVM_FFI_ICHECK(packet_ready.numel() == packet_slots && packet_consumed.numel() == packet_slots)
      << "packet generation counters must contain one entry per ring slot";
  TVM_FFI_ICHECK(helper_done.numel() == total_tasks)
      << "helper_done must contain one entry per sequence/head task";
  TVM_FFI_ICHECK(descriptor_storage.numel() >= static_cast<int64_t>(kSmallBHDescriptorStorageBytes))
      << "small-BH descriptor_storage must contain at least " << kSmallBHDescriptorStorageBytes
      << " bytes";

  for (const auto& workspace_named :
       {std::pair<const TensorView*, const char*>(&packet_workspace, "packet_workspace"),
        std::pair<const TensorView*, const char*>(&packet_ready, "packet_ready"),
        std::pair<const TensorView*, const char*>(&packet_consumed, "packet_consumed"),
        std::pair<const TensorView*, const char*>(&helper_done, "helper_done")}) {
    for (const auto& input_named :
         {std::pair<const TensorView*, const char*>(&q, "q"),
          std::pair<const TensorView*, const char*>(&k, "k"),
          std::pair<const TensorView*, const char*>(&v, "v"),
          std::pair<const TensorView*, const char*>(&g, "g"),
          std::pair<const TensorView*, const char*>(&beta, "beta"),
          std::pair<const TensorView*, const char*>(&beta_tma, "beta_tma"),
          std::pair<const TensorView*, const char*>(&A_log, "A_log"),
          std::pair<const TensorView*, const char*>(&dt_bias, "dt_bias"),
          std::pair<const TensorView*, const char*>(&cu_seqlens, "cu_seqlens"),
          std::pair<const TensorView*, const char*>(&seq_order, "seq_order"),
          std::pair<const TensorView*, const char*>(&initial_state, "initial_state"),
          std::pair<const TensorView*, const char*>(&out, "out"),
          std::pair<const TensorView*, const char*>(&final_state, "final_state"),
          std::pair<const TensorView*, const char*>(&descriptor_storage, "descriptor_storage")}) {
      CheckNoOverlap(*workspace_named.first, workspace_named.second, *input_named.first,
                     input_named.second);
    }
  }
  CheckNoOverlap(packet_workspace, "packet_workspace", packet_ready, "packet_ready");
  CheckNoOverlap(packet_workspace, "packet_workspace", packet_consumed, "packet_consumed");
  CheckNoOverlap(packet_workspace, "packet_workspace", helper_done, "helper_done");
  CheckNoOverlap(packet_ready, "packet_ready", packet_consumed, "packet_consumed");
  CheckNoOverlap(packet_ready, "packet_ready", helper_done, "helper_done");
  CheckNoOverlap(packet_consumed, "packet_consumed", helper_done, "helper_done");

  constexpr int32_t kSmemBytes = SMEM_TOTAL;
  CheckDynamicSmemCapacity(device_id, kSmemBytes);
  CheckCuda(cudaFuncSetAttribute(kernel_flashkda_bf16_small_bh_m128,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes),
            "cudaFuncSetAttribute(kernel_flashkda_bf16_small_bh_m128)");

  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  const TmaPointers tma = EncodeTmaPointers<128, 32>(q, k, v, g, beta_tma, out, descriptor_storage,
                                                     prepare_descriptors, stream);
  auto* descriptor_bytes = static_cast<unsigned char*>(descriptor_storage.data_ptr());
  void* packet_tma = descriptor_bytes + kTensorMapCount * sizeof(CUtensorMap);
  if (prepare_descriptors != 0) {
    const CUtensorMap packet_map = EncodeSmallBHPacketTma(packet_workspace);
    SmallBHPacketTensorMapWords words{};
    std::memcpy(words.words, &packet_map, sizeof(packet_map));
    PublishSmallBHPacketTensorMap<<<1, 32, 0, stream>>>(reinterpret_cast<uint64_t*>(packet_tma),
                                                        words);
    CheckCuda(cudaGetLastError(), "PublishSmallBHPacketTensorMap launch");
  }
  const int64_t beta_token_stride = beta.stride(beta.ndim() - 2);
  PackBetaForTmaIfNeeded(beta, beta_tma, num_heads, beta_token_stride, stream);

  const dim3 grid(static_cast<uint32_t>(grid_x_i64), 1, 1);
  const dim3 block(THREADS, 1, 1);
  kernel_flashkda_bf16_small_bh_m128<<<grid, block, kSmemBytes, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<flashkda_generated_FlashKDATensorMap const*>(tma.q),
      reinterpret_cast<__nv_bfloat16*>(k.data_ptr()),
      reinterpret_cast<flashkda_generated_FlashKDATensorMap const*>(tma.k),
      reinterpret_cast<__nv_bfloat16*>(v.data_ptr()),
      reinterpret_cast<flashkda_generated_FlashKDATensorMap const*>(tma.v),
      reinterpret_cast<__nv_bfloat16*>(g.data_ptr()),
      reinterpret_cast<flashkda_generated_FlashKDATensorMap const*>(tma.g),
      reinterpret_cast<__nv_bfloat16*>(beta.data_ptr()),
      reinterpret_cast<flashkda_generated_FlashKDATensorMap const*>(tma.beta),
      reinterpret_cast<float*>(A_log.data_ptr()), reinterpret_cast<float*>(dt_bias.data_ptr()),
      reinterpret_cast<long long*>(cu_seqlens.data_ptr()),
      reinterpret_cast<int*>(seq_order.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(initial_state.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      reinterpret_cast<flashkda_generated_FlashKDATensorMap const*>(tma.out),
      reinterpret_cast<__nv_bfloat16*>(final_state.data_ptr()), static_cast<int32_t>(num_heads),
      static_cast<int32_t>(use_initial_state), static_cast<int32_t>(store_final_state),
      static_cast<float>(scale), static_cast<float>(lower_bound), 0, 0, 0,
      static_cast<int64_t>(beta_token_stride),
      static_cast<int64_t>(num_heads * kHeadDim * kHeadDim), 0, 0,
      reinterpret_cast<flashkda_generated_FlashKDATensorMap const*>(packet_tma),
      reinterpret_cast<unsigned int*>(packet_ready.data_ptr()),
      reinterpret_cast<unsigned int*>(packet_consumed.data_ptr()),
      reinterpret_cast<unsigned int*>(helper_done.data_ptr()));
  CheckCuda(cudaGetLastError(), "kernel_flashkda_bf16_small_bh_m128 launch");
}

}  // namespace flash_kda
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::flash_kda::RunSmallBHM128);
