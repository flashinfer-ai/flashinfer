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

inline CUtensorMap buildNdDescriptor(std::type_info const& dtype,
                                     std::vector<uint64_t> const& shapes,
                                     std::vector<uint64_t> const& strides,
                                     std::vector<int32_t> const& tileShapes, void* gmemAddr) {
  // The multiplication factor of the data padding in SMEM.
  CUtensorMap desc{};
  CUtensorMapDataType tmaDataFormat;
  int dtype_size{};
  if (dtype == typeid(float)) {
    tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    dtype_size = sizeof(float);
  } else if (dtype == typeid(half)) {
    tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    dtype_size = sizeof(half);
  } else if (dtype == typeid(__nv_bfloat16)) {
    tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    dtype_size = sizeof(__nv_bfloat16);
  } else {
    throw std::invalid_argument("buildNdDescriptor: unsupported dtype");
  }

  // The swizzle type.
  CUtensorMapSwizzle swizzleType{CU_TENSOR_MAP_SWIZZLE_NONE};

  // Check gmem address must be 16B-aligned
  FLASHINFER_CHECK((reinterpret_cast<uint64_t>(gmemAddr) & 0b1111) == 0,
                   "Tensor must be 16B-aligned");

  // Check shape must be in range [1, 2^32]
  int32_t dim = shapes.size();
  // dimensions for batched gemm with blocked layout.
  // Check shape range.
  for (int32_t ii = 0; ii < dim; ++ii) {
    FLASHINFER_CHECK(shapes[ii] >= (uint64_t(1)));        // Size must be min 1
    FLASHINFER_CHECK(shapes[ii] <= (uint64_t(1) << 32));  // Size must be max 2^32
  }

  // TMA descriptor does not store the zeroth stride and assumes it is 1.
  FLASHINFER_CHECK(static_cast<int32_t>(strides.size()) == dim);
  FLASHINFER_CHECK(strides[0] == 1);

  // Build strides in bytes.
  // cuTensorMapEncodeTiled ignores the stride of the first dimension (implicitly 1).
  std::vector<uint64_t> stridesInBytes(dim - 1);
  for (int32_t ii = 0; ii < dim - 1; ++ii) {
    stridesInBytes[ii] = strides[ii + 1] * dtype_size;
  }

  // Build box dim array. If tileShapes is smaller than dim, just fill with 1s.
  FLASHINFER_CHECK(static_cast<int32_t>(tileShapes.size()) <= dim);
  std::vector<uint32_t> boxDim(dim, 1);
  boxDim[0] = tileShapes[0];
  for (size_t ii = 1; ii < tileShapes.size(); ++ii) {
    if (tileShapes[ii] > 256) {
      std::cerr << "buildNdTmaDescriptor: boxDim too large " << tileShapes[ii] << std::endl;
      FLASHINFER_CHECK(false);
    } else {
      boxDim[ii] = tileShapes[ii];
    }
  }

  // Set tile strides to 1;
  std::vector<uint32_t> tileStrides(dim, 1);

  // Build the descriptor.
  CUresult result =
      cuTensorMapEncodeTiled(&desc, tmaDataFormat,
                             /*tensorRank=*/dim, gmemAddr, shapes.data(), stridesInBytes.data(),
                             boxDim.data(), tileStrides.data(),
                             /*interleave=*/CU_TENSOR_MAP_INTERLEAVE_NONE, swizzleType,
                             /*l2Promotion=*/CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                             /*oobFill=*/CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  if (result != CUDA_SUCCESS) {
    char const* errorString;
    cuGetErrorString(result, &errorString);
    std::stringstream ss;
    ss << "Error: Failed to initialize the TMA descriptor " << result << std::endl;

    ss << "tmaFormat: " << static_cast<int>(tmaDataFormat) << " dim: " << dim
       << " gmem: " << gmemAddr << std::endl;

    ss << "Shape: ";
    for (int ii = 0; ii < dim; ++ii) {
      ss << shapes[ii] << " ";
    }
    ss << std::endl;

    ss << "Stride: ";
    for (int ii = 0; ii < dim - 1; ++ii) {
      ss << stridesInBytes[ii] << " ";
    }
    ss << std::endl;

    ss << "tileShapes: ";
    for (int ii = 0; ii < dim; ++ii) {
      ss << boxDim[ii] << " ";
    }
    ss << std::endl;

    ss << "tileStrides: ";
    for (int ii = 0; ii < dim; ++ii) {
      ss << tileStrides[ii] << " ";
    }
    ss << std::endl;
    ss << "swizzleType: " << int(swizzleType) << std::endl;
    ss << "(in " << __FILE__ << ":" << __LINE__ << ")" << std::endl;
    throw std::runtime_error(ss.str());
  }

  return desc;
}

}  // namespace flashinfer::mamba::tma
