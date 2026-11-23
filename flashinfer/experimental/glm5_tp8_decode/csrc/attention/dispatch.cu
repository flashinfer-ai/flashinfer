/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>

#include "../attention_api.h"

namespace flashinfer::glm5_tp8 {
KernelResources initialize_mla_q4(int);
KernelResources initialize_indexer_q4(int);
KernelResources initialize_mla_q1(int);
KernelResources initialize_indexer_q1(int);
cudaError_t launch_mla_q4(const AttentionArguments&, cudaStream_t, bool, bool);
cudaError_t launch_indexer_q4(const AttentionArguments&, cudaStream_t, bool, bool);
cudaError_t launch_mla_q1(const AttentionArguments&, cudaStream_t, bool, bool);
cudaError_t launch_indexer_q1(const AttentionArguments&, cudaStream_t, bool, bool);

KernelResources initialize_attention(int q_len, int rank, int device) {
  if ((q_len != 1 && q_len != 4) || rank < 0 || rank >= 8)
    throw std::invalid_argument("invalid q_len or logical rank");
  if (q_len == 4) return rank == 0 ? initialize_indexer_q4(device) : initialize_mla_q4(device);
  return rank == 0 ? initialize_indexer_q1(device) : initialize_mla_q1(device);
}

cudaError_t launch_attention(const AttentionArguments& p, cudaStream_t stream, bool pdl,
                             bool cooperative) {
  if (p.rank < 0 || p.rank >= 8 || (p.q_len != 1 && p.q_len != 4)) return cudaErrorInvalidValue;
  if (p.q_len == 1) {
    if (!p.context || !p.q1_state) return cudaErrorInvalidValue;
    for (auto peer : p.peer_records)
      if (!peer || uintptr_t(peer) % 16) return cudaErrorInvalidValue;
    return p.rank == 0 ? launch_indexer_q1(p, stream, pdl, cooperative)
                       : launch_mla_q1(p, stream, pdl, cooperative);
  }
  if (!p.epoch_ctr) return cudaErrorInvalidValue;
  for (auto peer : p.peer_arenas)
    if (!peer || uintptr_t(peer) % 16) return cudaErrorInvalidValue;
  if (p.rank == 0) {
    if (!p.context) return cudaErrorInvalidValue;
    return launch_indexer_q4(p, stream, pdl, cooperative);
  }
  return launch_mla_q4(p, stream, pdl, cooperative);
}

cudaError_t encode_attention_tmap(const void* base, int64_t rows, int64_t cols, int32_t box_rows,
                                  int32_t box_cols, void* out128) {
  if (!base || !out128 || rows < 1 || cols < 1 || box_rows < 1 || box_rows > 256 || box_cols < 1 ||
      box_cols > 64 || uintptr_t(base) % 16)
    return cudaErrorInvalidValue;
  using Encode =
      CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                   const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                   CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
  void* entry = nullptr;
  cudaDriverEntryPointQueryResult query = cudaDriverEntryPointSymbolNotFound;
  auto status = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &entry, 12000,
                                                 cudaEnableDefault, &query);
  if (status != cudaSuccess) return status;
  if (!entry || query != cudaDriverEntryPointSuccess) return cudaErrorNotSupported;
  alignas(64) CUtensorMap map{};
  const cuuint64_t dims[] = {cuuint64_t(cols), cuuint64_t(rows)};
  const cuuint64_t strides[] = {cuuint64_t(cols) * 2};
  const cuuint32_t box[] = {cuuint32_t(box_cols), cuuint32_t(box_rows)};
  const cuuint32_t elem[] = {1, 1};
  auto result = reinterpret_cast<Encode>(entry)(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), dims, strides, box, elem,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (result != CUDA_SUCCESS) return cudaErrorInvalidValue;
  static_assert(sizeof(map) == 128);
  std::memcpy(out128, &map, sizeof(map));
  return cudaSuccess;
}
}  // namespace flashinfer::glm5_tp8
