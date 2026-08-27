// Host TMA descriptor helpers for the VibeCUDA FlashKDA prefill kernel
// family. Raw-pointer C++ (the TVM-FFI tensor shell lives in
// vibecuda_flashkda_binding.cu); the TMA descriptor shapes/swizzles match
// the frozen FlashKDA prefill path so the kernels receive byte-identical
// descriptor layouts.
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#define KDA_FFI_CHECK(cond, msg)                                    \
  do {                                                              \
    if (!(cond)) {                                                  \
      throw std::runtime_error(std::string("kda_flash: ") + (msg)); \
    }                                                               \
  } while (0)

namespace kda_flash {

constexpr int64_t kHeadDim = 128;
constexpr size_t kTensorMapCount = 7;
constexpr size_t kDescriptorStorageBytes = kTensorMapCount * sizeof(CUtensorMap);

inline void CheckCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("kda_flash: ") + operation +
                             " failed: " + cudaGetErrorString(status));
  }
}

template <int ChunkTokens = 32>
inline CUtensorMap EncodeQkTma(const void* base, int64_t numel, int64_t d2, int64_t d1,
                               const char* name) {
  static_assert(ChunkTokens == 16 || ChunkTokens == 32 || ChunkTokens == 64);
  KDA_FFI_CHECK(d1 > 0 && d2 > 0 && d1 % 64 == 0, "invalid trailing dims for q/k TMA");
  const int64_t outer2 = numel / (d1 * d2);
  uint64_t global_dim[4] = {64, static_cast<uint64_t>(outer2), static_cast<uint64_t>(d2),
                            static_cast<uint64_t>(d1 / 64)};
  KDA_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] >= 1 && global_dim[3] >= 2,
                std::string(name) + " cannot encode the q/k TMA box");
  uint64_t global_strides[3] = {static_cast<uint64_t>(d2 * d1 * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(d1 * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(64 * sizeof(__nv_bfloat16))};
  uint32_t box_dim[4] = {64, ChunkTokens, 1, 2};
  uint32_t elem_strides[4] = {1, 1, 1, 1};
  CUtensorMap tensor_map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(base), global_dim,
      global_strides, box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  KDA_FFI_CHECK(result == CUDA_SUCCESS, std::string("cuTensorMapEncodeTiled failed for ") + name);
  return tensor_map;
}

template <int ValueRows, int ChunkTokens = 32>
inline CUtensorMap EncodeValueTma(const void* base, int64_t numel, int64_t d2, int64_t d1) {
  static_assert(ValueRows == 64 || ValueRows == 128);
  static_assert(ChunkTokens == 16 || ChunkTokens == 32 || ChunkTokens == 64);
  const int64_t outer2 = numel / (d1 * d2);
  uint64_t global_dim[3] = {static_cast<uint64_t>(d1), static_cast<uint64_t>(d2),
                            static_cast<uint64_t>(outer2)};
  KDA_FFI_CHECK(global_dim[0] >= ValueRows && global_dim[1] >= 1 && global_dim[2] > 0,
                "v cannot encode its TMA box");
  uint64_t global_strides[2] = {static_cast<uint64_t>(d1 * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(d1 * d2 * sizeof(__nv_bfloat16))};
  uint32_t box_dim[3] = {ValueRows, 1, ChunkTokens};
  uint32_t elem_strides[3] = {1, 1, 1};
  CUtensorMap tensor_map{};
  constexpr CUtensorMapSwizzle swizzle =
      ValueRows == 64 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE;
  const CUresult result = cuTensorMapEncodeTiled(
      &tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(base), global_dim,
      global_strides, box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  KDA_FFI_CHECK(result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed for v");
  return tensor_map;
}

template <int ChunkTokens = 32>
inline CUtensorMap EncodeGateTma(const void* base, int64_t numel, int64_t d2, int64_t d1) {
  static_assert(ChunkTokens == 16 || ChunkTokens == 32 || ChunkTokens == 64);
  const int64_t outer2 = numel / (d1 * d2);
  uint64_t global_dim[3] = {static_cast<uint64_t>(d1), static_cast<uint64_t>(d2),
                            static_cast<uint64_t>(outer2)};
  KDA_FFI_CHECK(global_dim[0] >= 128 && global_dim[1] >= 1 && global_dim[2] > 0,
                "g cannot encode its TMA box");
  uint64_t global_strides[2] = {static_cast<uint64_t>(d1 * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(d1 * d2 * sizeof(__nv_bfloat16))};
  uint32_t box_dim[3] = {128, 1, ChunkTokens};
  uint32_t elem_strides[3] = {1, 1, 1};
  CUtensorMap tensor_map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(base), global_dim,
      global_strides, box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  KDA_FFI_CHECK(result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed for g");
  return tensor_map;
}

template <int ChunkTokens = 32>
inline CUtensorMap EncodeBetaTma(const void* base, int64_t numel, int64_t d1) {
  static_assert(ChunkTokens == 16 || ChunkTokens == 32 || ChunkTokens == 64);
  const int64_t outer1 = numel / d1;
  uint64_t global_dim[2] = {static_cast<uint64_t>(d1), static_cast<uint64_t>(outer1)};
  KDA_FFI_CHECK(global_dim[0] >= 8 && global_dim[1] >= ChunkTokens,
                "beta_tma cannot encode its TMA box");
  uint64_t global_strides[1] = {static_cast<uint64_t>(d1 * sizeof(__nv_bfloat16))};
  uint32_t box_dim[2] = {8, ChunkTokens};
  uint32_t elem_strides[2] = {1, 1};
  CUtensorMap tensor_map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), global_dim,
      global_strides, box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  KDA_FFI_CHECK(result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed for beta_tma");
  return tensor_map;
}

template <int ValueRows, int ChunkTokens = 32>
inline CUtensorMap EncodeOutputTma(const void* base, int64_t numel, int64_t d2, int64_t d1) {
  static_assert(ValueRows == 64 || ValueRows == 128);
  static_assert(ChunkTokens == 16 || ChunkTokens == 32 || ChunkTokens == 64);
  KDA_FFI_CHECK(d1 > 0 && d2 > 0 && d1 % 64 == 0, "out has invalid trailing dims");
  const int64_t outer2 = numel / (d1 * d2);
  uint64_t global_dim[4] = {64, static_cast<uint64_t>(outer2), static_cast<uint64_t>(d2),
                            static_cast<uint64_t>(d1 / 64)};
  constexpr uint32_t value_splits = ValueRows / 64;
  KDA_FFI_CHECK(global_dim[0] >= 64 && global_dim[1] > 0 && global_dim[2] >= 1 &&
                    global_dim[3] >= value_splits,
                "out cannot encode its TMA box");
  uint64_t global_strides[3] = {static_cast<uint64_t>(d2 * d1 * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(d1 * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(64 * sizeof(__nv_bfloat16))};
  uint32_t box_dim[4] = {64, ChunkTokens, 1, value_splits};
  uint32_t elem_strides[4] = {1, 1, 1, 1};
  CUtensorMap tensor_map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(base), global_dim,
      global_strides, box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  KDA_FFI_CHECK(result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed for out");
  return tensor_map;
}

struct TmaPointers {
  void* q;
  void* k;
  void* v;
  void* g;
  void* beta;
  void* out;
  void* scratch_out;
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

template <int ValueRows, int ChunkTokens = 32>
inline TmaPointers EncodeTmaPointersAll(const void* q, const void* k, const void* v, const void* g,
                                        const void* beta_tma, const void* out,
                                        const void* scratch_out, int64_t token_count,
                                        int64_t num_heads, int64_t beta_tma_numel,
                                        int64_t beta_tma_dim1, void* descriptor_storage_ptr,
                                        int64_t prepare_descriptors, cudaStream_t stream) {
  if (prepare_descriptors != 0) {
    const void* scratch_base = (scratch_out != nullptr) ? scratch_out : out;
    const std::array<CUtensorMap, kTensorMapCount> host_maps = {
        EncodeQkTma<ChunkTokens>(q, token_count * num_heads * kHeadDim, num_heads, kHeadDim, "q"),
        EncodeQkTma<ChunkTokens>(k, token_count * num_heads * kHeadDim, num_heads, kHeadDim, "k"),
        EncodeValueTma<ValueRows, ChunkTokens>(v, token_count * num_heads * kHeadDim, num_heads,
                                               kHeadDim),
        EncodeGateTma<ChunkTokens>(g, token_count * num_heads * kHeadDim, num_heads, kHeadDim),
        EncodeBetaTma<ChunkTokens>(beta_tma, beta_tma_numel, beta_tma_dim1),
        EncodeOutputTma<ValueRows, ChunkTokens>(out, token_count * num_heads * kHeadDim, num_heads,
                                                kHeadDim),
        EncodeOutputTma<ValueRows, ChunkTokens>(scratch_base, token_count * num_heads * kHeadDim,
                                                num_heads, kHeadDim),
    };
    static_assert(sizeof(host_maps) == kDescriptorStorageBytes);
    TensorMapWords words{};
    std::memcpy(words.words, host_maps.data(), sizeof(host_maps));
    PublishTensorMaps<<<1, 128, 0, stream>>>(reinterpret_cast<uint64_t*>(descriptor_storage_ptr),
                                             words);
    CheckCuda(cudaGetLastError(), "PublishTensorMaps launch");
  }

  auto* bytes = static_cast<unsigned char*>(descriptor_storage_ptr);
  constexpr size_t stride = sizeof(CUtensorMap);
  return {bytes + 0 * stride, bytes + 1 * stride, bytes + 2 * stride, bytes + 3 * stride,
          bytes + 4 * stride, bytes + 5 * stride, bytes + 6 * stride};
}

}  // namespace kda_flash
