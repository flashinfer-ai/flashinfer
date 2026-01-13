#pragma once
#include <cuda.h>
#include <cudaTypedefs.h>  // PFN_cuTensorMapEncodeTiled, CUtensorMap
#include <cuda_fp16.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <type_traits>

#ifndef gpuErrchk
#define gpuErrchk(ans)                    \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }
#endif

static inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    std::cout << "GPU assert failed" << std::endl;
    if (abort) exit(code);
  }
}
namespace flashinfer::mamba::tma {

// namespace cde = cuda::device::experimental;

static inline PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
  // Get pointer to cuTensorMapEncodeTiled
  cudaDriverEntryPointQueryResult driver_status;
  void* cuTensorMapEncodeTiled_ptr = nullptr;
  gpuErrchk(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr,
                                             12000, cudaEnableDefault, &driver_status));

  if (driver_status != cudaDriverEntryPointSuccess) {
    std::cerr << "Could not get cuTensorMapEncodeTiled driver entry point" << std::endl;
    abort();
  }

  return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
}

template <typename Dtype>
inline CUtensorMap createTensorMap(void* matrix_ptr, uint32_t matrix_height, uint32_t matrix_width,
                                   uint32_t tile_height, uint32_t tile_width) {
  CUtensorMap tensor_map{};
  constexpr uint32_t rank = 2;

  std::array<uint64_t, rank> matrix_dim = {matrix_width, matrix_height};
  std::array<uint64_t, rank - 1> stride = {matrix_width * sizeof(Dtype)};
  std::array<uint32_t, rank> box_size = {tile_width, tile_height};
  std::array<uint32_t, rank> elem_stride = {1, 1};

  // CUtensorMapDataType dtype_format = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  CUtensorMapDataType dtype_format;
  if constexpr (std::is_same_v<Dtype, half>) {
    dtype_format = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  } else if constexpr (std::is_same_v<Dtype, float>) {
    dtype_format = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  } else if constexpr (std::is_same_v<Dtype, __nv_bfloat16>) {
    dtype_format = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  } else {
    static_assert([]() { return false; }(), "Unsupported data type for TMA tensor map");
    return tensor_map;  // shut the compiler up
  }

  auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
  CUresult res = cuTensorMapEncodeTiled(
      &tensor_map, dtype_format, rank, matrix_ptr, matrix_dim.data(), stride.data(),
      box_size.data(), elem_stride.data(), CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  if (res != CUDA_SUCCESS) {
    const char* err_name = nullptr;
    const char* err_str = nullptr;
    cuGetErrorName(res, &err_name);
    cuGetErrorString(res, &err_str);
    std::cerr << "Could not create a tensor map" << std::endl;
    std::cerr << "Error is: " << err_name << ": " << err_str << std::endl;
    abort();
  }

  return tensor_map;
}

}  // namespace flashinfer::mamba::tma
