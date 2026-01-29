#pragma once
#include <cuda.h>
#include <cudaTypedefs.h>  // PFN_cuTensorMapEncodeTiled, CUtensorMap
#include <cuda_fp16.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <type_traits>

namespace flashinfer::mamba::tma {

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
