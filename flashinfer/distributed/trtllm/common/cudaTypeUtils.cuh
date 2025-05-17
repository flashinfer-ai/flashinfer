#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace tensorrt_llm {
namespace common {

template <typename T>
struct CudaDataType {
    static constexpr cudaDataType_t value = cudaDataType::CUDA_R_32F;
};

template <>
struct CudaDataType<half> {
    static constexpr cudaDataType_t value = cudaDataType::CUDA_R_16F;
};

template <>
struct CudaDataType<__nv_bfloat16> {
    static constexpr cudaDataType_t value = cudaDataType::CUDA_R_16BF;
};

template <typename T>
struct UpperType;

template <>
struct UpperType<int8_t> {
    using Type = int;
};

template <>
struct UpperType<uint32_t> {
    using Type = uint32_t;
};

template <>
struct UpperType<int> {
    using Type = int;
};

template <>
struct UpperType<__nv_bfloat16> {
    using Type = double;
};

template <>
struct UpperType<half> {
    using Type = double;
};

template <>
struct UpperType<float> {
    using Type = double;
};

} // namespace common
} // namespace tensorrt_llm 