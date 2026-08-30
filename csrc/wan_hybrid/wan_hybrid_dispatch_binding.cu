/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

#include "tvm_ffi_utils.h"
#include "wan_hybrid_common.cuh"

// Keep both frozen device programs unchanged while isolating their standalone
// typedefs and the two helper names that they share.
#define uint8_t wan_hybrid_quant_generated_uint8_t
#define uint16_t wan_hybrid_quant_generated_uint16_t
#define uint32_t wan_hybrid_quant_generated_uint32_t
#define uint64_t wan_hybrid_quant_generated_uint64_t
#define int32_t wan_hybrid_quant_generated_int32_t
#define int16_t wan_hybrid_quant_generated_int16_t
#define WanHybridTensorMap wan_hybrid_quant_generated_TensorMap
#define WanHybridTensorMapPack wan_hybrid_quant_generated_TensorMapPack
#define CUtensorMap wan_hybrid_quant_generated_CUtensorMap
#define make_warp_uniform wan_hybrid_quant_make_warp_uniform
#define max_noftz wan_hybrid_quant_max_noftz
#if FLASHINFER_WAN_HYBRID_TARGET_MINOR == 0
#include "device/wan_hybrid_quantize_value_sm100.cu"
#elif FLASHINFER_WAN_HYBRID_TARGET_MINOR == 3
#include "device/wan_hybrid_quantize_value_sm103.cu"
#else
#error "Wan hybrid dispatch requires target minor 0 or 3"
#endif
constexpr int kWanHybridQuantThreads = THREADS;
constexpr size_t kWanHybridQuantDynamicSmemBytes = SMEM_TOTAL;
#undef THREADS
#undef SMEM_TOTAL
#undef SMEM_V_SMEM_STRIDE
#undef SMEM_V_SMEM_STAGE_BYTES
#undef SMEM_V_SMEM_OFF
#undef NUM_MAIN_STAGES
#undef WAN_HYBRID_INF
#undef max_noftz
#undef make_warp_uniform
#undef CUtensorMap
#undef WanHybridTensorMapPack
#undef WanHybridTensorMap
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t

#define uint8_t wan_hybrid_attention_generated_uint8_t
#define uint16_t wan_hybrid_attention_generated_uint16_t
#define uint32_t wan_hybrid_attention_generated_uint32_t
#define uint64_t wan_hybrid_attention_generated_uint64_t
#define int32_t wan_hybrid_attention_generated_int32_t
#define int16_t wan_hybrid_attention_generated_int16_t
#define WanHybridTensorMap wan_hybrid_attention_generated_TensorMap
#define WanHybridTensorMapPack wan_hybrid_attention_generated_TensorMapPack
#define CUtensorMap wan_hybrid_attention_generated_CUtensorMap
#if FLASHINFER_WAN_HYBRID_TARGET_MINOR == 0
#include "device/wan_hybrid_attention_sm100.cu"
#elif FLASHINFER_WAN_HYBRID_TARGET_MINOR == 3
#include "device/wan_hybrid_attention_sm103.cu"
#else
#error "Wan hybrid dispatch requires target minor 0 or 3"
#endif
#undef CUtensorMap
#undef WanHybridTensorMapPack
#undef WanHybridTensorMap
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t

namespace flashinfer {
namespace wan_hybrid {

static_assert(kWanHybridQuantThreads == 256);
static_assert(kWanHybridQuantDynamicSmemBytes == 32896);
static_assert(SMEM_TOTAL == kAttentionDynamicSmemBytes);
static_assert(sizeof(CUtensorMap) == kTensorMapBytes);
static_assert(sizeof(wan_hybrid_attention_generated_TensorMap) == kTensorMapBytes);

void Dispatch(TensorView q, TensorView k, TensorView value, TensorView vt, TensorView sfvt_lo,
              TensorView sfvt_hi, TensorView out, TensorView descriptor_storage,
              bool prepare_descriptors, double sm_scale) {
  CHECK_INPUT_AND_TYPE(q, dl_bfloat16);
  const int32_t device_id = q.device().device_id;
  CheckExactTensor(q, "q", dl_bfloat16, {kBatch, kSequence, kHeads, kHeadDim}, device_id);
  CheckExactTensor(k, "k", dl_bfloat16, {kBatch, kSequence, kHeads, kHeadDim}, device_id);
  CheckExactTensor(value, "value", dl_bfloat16, {kBatch, kSequence, kHeads, kHeadDim}, device_id);
  CheckExactTensor(vt, "vt", dl_uint8, {kPackedValueRows, kPackedValueColumns}, device_id);
  CheckExactTensor(sfvt_lo, "sfvt_lo", dl_uint8, {kValueScaleRows, kScaleColumns}, device_id);
  CheckExactTensor(sfvt_hi, "sfvt_hi", dl_uint8, {kValueScaleRows, kScaleColumns}, device_id);
  CheckExactTensor(out, "out", dl_bfloat16, {kBatch, kSequence, kHeads, kHeadDim}, device_id);
  CheckExactTensor(descriptor_storage, "descriptor_storage", dl_uint8,
                   {kTensorMapCount, kTensorMapBytes}, device_id);
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(descriptor_storage.data_ptr()) % 128, 0)
      << "descriptor_storage must be 128-byte aligned";
  TVM_FFI_ICHECK(std::isfinite(sm_scale)) << "sm_scale must be finite";

  ffi::CUDADeviceGuard device_guard(device_id);
  CheckTarget(device_id, "dispatch");
  if (prepare_descriptors) {
    PrepareTensorMaps(q, k, vt, sfvt_lo, sfvt_hi, out, descriptor_storage);
    CheckCuda(cudaFuncSetAttribute(kernel_wan_hybrid_attention,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(kAttentionDynamicSmemBytes)),
              "cudaFuncSetAttribute(MaxDynamicSharedMemorySize)");
  }

  int multiprocessor_count = 0;
  CheckCuda(
      cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device_id),
      "cudaDeviceGetAttribute(multiProcessorCount)");
  constexpr int kTotalTiles = ((kSequence + 255) / 256) * kBatch * kHeads;
  const int grid_x = std::min({multiprocessor_count, kTotalTiles, static_cast<int>(kMaximumTiles)});
  TVM_FFI_ICHECK_GT(grid_x, 0) << "wan_hybrid attention requires at least one SM";

  auto* descriptor_bytes = static_cast<uint8_t*>(descriptor_storage.data_ptr());
  auto tensor_map = [descriptor_bytes](int index) {
    return reinterpret_cast<const wan_hybrid_attention_generated_TensorMap*>(
        descriptor_bytes + index * kTensorMapBytes);
  };
  const auto* q_map = tensor_map(0);
  const auto* k_map = tensor_map(1);
  const auto* vt_map = tensor_map(2);
  const auto* sfvt_lo_map = tensor_map(3);
  const auto* sfvt_hi_map = tensor_map(4);
  const auto* out_map = tensor_map(5);
  constexpr int seqlen_q = kSequence;
  constexpr int seqlen_kv = kSequence;
  constexpr int heads = kHeads;
  constexpr int total_bh = kBatch * kHeads;
  constexpr int physical_num_blocks = kPaddedSequence / 128;
  const float softmax_scale_log2 = static_cast<float>(sm_scale / std::log(2.0));
  const cudaStream_t stream = get_stream(q.device());

  auto* value_bytes = static_cast<wan_hybrid_quant_generated_uint8_t*>(vt.data_ptr());
  auto* scale_lo_bytes = static_cast<wan_hybrid_quant_generated_uint8_t*>(sfvt_lo.data_ptr());
  auto* scale_hi_bytes = static_cast<wan_hybrid_quant_generated_uint8_t*>(sfvt_hi.data_ptr());
  kernel_wan_hybrid_quantize_value<<<dim3(kHeads * kLogicalBlocks, 1, 1),
                                     dim3(kWanHybridQuantThreads, 1, 1),
                                     kWanHybridQuantDynamicSmemBytes, stream>>>(
      static_cast<__nv_bfloat16*>(value.data_ptr()), value_bytes,
      value_bytes + kValueRows * kPackedValueColumns, scale_lo_bytes, scale_hi_bytes,
      scale_lo_bytes + kScaleRows * kScaleColumns, scale_hi_bytes + kScaleRows * kScaleColumns,
      static_cast<wan_hybrid_quant_generated_int32_t>(kHeads),
      static_cast<wan_hybrid_quant_generated_int32_t>(kSequence),
      static_cast<wan_hybrid_quant_generated_int32_t>(kPaddedSequence),
      static_cast<wan_hybrid_quant_generated_int32_t>(kLogicalBlocks),
      static_cast<wan_hybrid_quant_generated_int32_t>(kPhysicalBlocks));
  CheckCuda(cudaGetLastError(), "wan_hybrid_quantize_value launch");

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<uint32_t>(grid_x), 1, 1);
  config.blockDim = dim3(kAttentionThreads, 1, 1);
  config.dynamicSmemBytes = kAttentionDynamicSmemBytes;
  config.stream = stream;
  CheckCuda(cudaLaunchKernelEx(&config, kernel_wan_hybrid_attention, q_map, k_map, vt_map,
                               sfvt_lo_map, sfvt_hi_map, out_map, seqlen_q, seqlen_kv,
                               softmax_scale_log2, heads, total_bh, physical_num_blocks),
            "wan_hybrid_attention launch");
}

}  // namespace wan_hybrid
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(wan_hybrid_dispatch, flashinfer::wan_hybrid::Dispatch);
