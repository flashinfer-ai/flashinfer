// Round-149 (supervisor directive): session-local host-side TMA descriptor
// helpers for the vendored Cake generated two-stage BT=16 prepare/chain
// kernels (cakebt16_*_gen.cu). TVM-FFI plumbing replaced by plain ATen
// tensors; every descriptor recipe below is copied field-for-field from
// FlashInfer's csrc/kda flashkda_binding_common.cuh (RunFlashKDAPrefill
// frozen kernels) and cake_flashkda_bt16_binding_common.cuh so the kernels
// receive byte-identical CUtensorMap layouts.
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#define KDA_BT16_CHECK(cond, msg)                                            \
  do {                                                                        \
    if (!(cond)) {                                                            \
      throw std::runtime_error(std::string("kda_bt16: ") + (msg));            \
    }                                                                         \
  } while (0)

namespace kda_bt16 {

constexpr int64_t kHeadDim = 128;
constexpr int64_t kChunkTokens = 16;
constexpr size_t kTensorMapCount = 7;
constexpr size_t kDescriptorStorageBytes = kTensorMapCount * sizeof(CUtensorMap);

inline void CheckCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("kda_bt16: ") + operation +
                             " failed: " + cudaGetErrorString(status));
  }
}

inline void PublishMaps(const std::array<CUtensorMap, kTensorMapCount>& host_maps,
                        void* descriptor_storage, cudaStream_t stream,
                        int64_t prepare_descriptors) {
  if (prepare_descriptors == 0) {
    return;
  }
  CheckCuda(cudaMemcpyAsync(descriptor_storage, host_maps.data(),
                            kDescriptorStorageBytes, cudaMemcpyHostToDevice, stream),
            "PublishMaps memcpy");
}

inline void CheckEncode(CUresult result, const char* name) {
  KDA_BT16_CHECK(result == CUDA_SUCCESS,
                 std::string("cuTensorMapEncodeTiled failed for ") + name);
}

// EncodeQkTma<16> recipe for q/k: flattened [tokens, H, 128] bf16.
inline CUtensorMap EncodeQk16(const void* base, int64_t tokens, int64_t heads,
                              const char* name) {
  const int64_t d1 = kHeadDim;
  const int64_t d2 = heads;
  KDA_BT16_CHECK(d2 >= 1 && tokens >= 1, "q/k dims invalid for BT16 TMA box");
  uint64_t global_dim[4] = {64, static_cast<uint64_t>(tokens),
                            static_cast<uint64_t>(d2), 2};
  KDA_BT16_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] >= 1,
                 "cannot encode the (64, 16, 1, 2) q/k TMA box");
  uint64_t global_strides[3] = {static_cast<uint64_t>(d2 * d1 * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(d1 * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(64 * sizeof(__nv_bfloat16))};
  uint32_t box_dim[4] = {64, 16, 1, 2};
  uint32_t elem_strides[4] = {1, 1, 1, 1};
  CUtensorMap map{};
  CheckEncode(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
                                     const_cast<void*>(base), global_dim, global_strides,
                                     box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                     CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
              name);
  return map;
}

// EncodeGateTma<16> recipe for raw_gate: flattened [tokens, H, 128] bf16.
inline CUtensorMap EncodeGate16(const void* base, int64_t tokens, int64_t heads) {
  const int64_t d1 = kHeadDim;
  const int64_t d2 = heads;
  uint64_t global_dim[3] = {static_cast<uint64_t>(d1), static_cast<uint64_t>(d2),
                            static_cast<uint64_t>(tokens)};
  KDA_BT16_CHECK(global_dim[0] >= 128 && global_dim[1] >= 1 && global_dim[2] > 0,
                 "g cannot encode the (128, 1, 16) TMA box");
  uint64_t global_strides[2] = {static_cast<uint64_t>(d1 * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(d1 * d2 * sizeof(__nv_bfloat16))};
  uint32_t box_dim[3] = {128, 1, 16};
  uint32_t elem_strides[3] = {1, 1, 1};
  CUtensorMap map{};
  CheckEncode(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3,
                                     const_cast<void*>(base), global_dim, global_strides,
                                     box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                     CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
              "g");
  return map;
}

// EncodeValueTma<64, 16> recipe for v: flattened [tokens, H, 128] bf16.
inline CUtensorMap EncodeValue64x16(const void* base, int64_t tokens, int64_t heads) {
  const int64_t d1 = kHeadDim;
  const int64_t d2 = heads;
  uint64_t global_dim[3] = {static_cast<uint64_t>(d1), static_cast<uint64_t>(d2),
                            static_cast<uint64_t>(tokens)};
  KDA_BT16_CHECK(global_dim[0] >= 64 && global_dim[1] >= 1 && global_dim[2] > 0,
                 "v cannot encode the (64, 1, 16) TMA box");
  uint64_t global_strides[2] = {static_cast<uint64_t>(d1 * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(d1 * d2 * sizeof(__nv_bfloat16))};
  uint32_t box_dim[3] = {64, 1, 16};
  uint32_t elem_strides[3] = {1, 1, 1};
  CUtensorMap map{};
  CheckEncode(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3,
                                     const_cast<void*>(base), global_dim, global_strides,
                                     box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                     CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
              "v");
  return map;
}

// EncodeOutputTma<64, 16> recipe for out: flattened [tokens, H, 128] bf16.
inline CUtensorMap EncodeOut64x16(const void* base, int64_t tokens, int64_t heads) {
  const int64_t d1 = kHeadDim;
  const int64_t d2 = heads;
  uint64_t global_dim[4] = {64, static_cast<uint64_t>(tokens),
                            static_cast<uint64_t>(d2), 2};
  KDA_BT16_CHECK(global_dim[0] >= 64 && global_dim[1] > 0 && global_dim[2] >= 1,
                 "out cannot encode the (64, 16, 1, 1) TMA box");
  uint64_t global_strides[3] = {static_cast<uint64_t>(d2 * d1 * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(d1 * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(64 * sizeof(__nv_bfloat16))};
  uint32_t box_dim[4] = {64, 16, 1, 1};
  uint32_t elem_strides[4] = {1, 1, 1, 1};
  CUtensorMap map{};
  CheckEncode(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
                                     const_cast<void*>(base), global_dim, global_strides,
                                     box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                     CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
              "out");
  return map;
}

// EncodeBt16FactorTma<BoxValueSplits> for [1, H, padded_tokens, 128] bf16.
inline CUtensorMap EncodeFactor16(const void* base, int64_t heads,
                                  int64_t padded_tokens, int box_value_splits,
                                  const char* name) {
  const int64_t d1 = kHeadDim;
  uint64_t global_dim[4] = {64, static_cast<uint64_t>(padded_tokens),
                            static_cast<uint64_t>(heads), 2};
  uint64_t global_strides[3] = {static_cast<uint64_t>(d1 * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(padded_tokens * d1 * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(64 * sizeof(__nv_bfloat16))};
  uint32_t box_dim[4] = {64, 16, 1, static_cast<uint32_t>(box_value_splits)};
  uint32_t elem_strides[4] = {1, 1, 1, 1};
  CUtensorMap map{};
  CheckEncode(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
                                     const_cast<void*>(base), global_dim, global_strides,
                                     box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                     CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
              name);
  return map;
}

// EncodeBt16QkWorkspaceTma for [1, H, total_chunks, 16, 16] bf16.
inline CUtensorMap EncodeQkWs(const void* base, int64_t heads, int64_t total_chunks) {
  uint64_t global_dim[5] = {16, 16, static_cast<uint64_t>(total_chunks),
                            static_cast<uint64_t>(heads), 1};
  uint64_t global_strides[4] = {16 * sizeof(__nv_bfloat16),
                                16 * 16 * sizeof(__nv_bfloat16),
                                static_cast<uint64_t>(total_chunks) * 16 * 16 *
                                    sizeof(__nv_bfloat16),
                                static_cast<uint64_t>(heads) * static_cast<uint64_t>(total_chunks) *
                                    16 * 16 * sizeof(__nv_bfloat16)};
  uint32_t box_dim[5] = {16, 16, 1, 1, 1};
  uint32_t elem_strides[5] = {1, 1, 1, 1, 1};
  CUtensorMap map{};
  CheckEncode(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5,
                                     const_cast<void*>(base), global_dim, global_strides,
                                     box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                     CU_TENSOR_MAP_SWIZZLE_32B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
              "ws_qk");
  return map;
}

// EncodeBt16DiagWorkspaceTma for [1, H, total_chunks, 128] fp32.
inline CUtensorMap EncodeDiagWs(const void* base, int64_t heads, int64_t total_chunks) {
  uint64_t global_dim[4] = {128, static_cast<uint64_t>(total_chunks),
                            static_cast<uint64_t>(heads), 1};
  uint64_t global_strides[3] = {128 * sizeof(float),
                                static_cast<uint64_t>(total_chunks) * 128 * sizeof(float),
                                static_cast<uint64_t>(heads) * static_cast<uint64_t>(total_chunks) *
                                    128 * sizeof(float)};
  uint32_t box_dim[4] = {128, 1, 1, 1};
  uint32_t elem_strides[4] = {1, 1, 1, 1};
  CUtensorMap map{};
  CheckEncode(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4,
                                     const_cast<void*>(base), global_dim, global_strides,
                                     box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                     CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
              "ws_diag");
  return map;
}

}  // namespace kda_bt16
