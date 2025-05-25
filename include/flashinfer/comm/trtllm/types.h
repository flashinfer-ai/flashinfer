#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

enum class DataType { kBOOL, kUINT8, kINT8, kINT32, kINT64, kBF16, kFP8, kFP16, kFP32, kINT4, kFP4, kUNKNOWN };

//! \brief For converting a C++ data type to a `DataType`.
template <typename T, bool = false>
struct TypeTraits {};

template <>
struct TypeTraits<float> {
  static constexpr auto value = DataType::kFP32;
};

template <>
struct TypeTraits<half> {
  static constexpr auto value = DataType::kFP16;
};

template <>
struct TypeTraits<int8_t> {
  static constexpr auto value = DataType::kINT8;
};

template <>
struct TypeTraits<int32_t> {
  static constexpr auto value = DataType::kINT32;
};

template <>
struct TypeTraits<int64_t> {
  static constexpr auto value = DataType::kINT64;
};

template <>
struct TypeTraits<bool> {
  static constexpr auto value = DataType::kBOOL;
};

template <>
struct TypeTraits<uint8_t> {
  static constexpr auto value = DataType::kUINT8;
};

template <>
struct TypeTraits<__nv_bfloat16> {
  static constexpr auto value = DataType::kBF16;
};

template <>
struct TypeTraits<__nv_fp8_e4m3> {
  static constexpr auto value = DataType::kFP8;
};

template <typename T>
struct TypeTraits<T*> {
  // Pointers are stored as int64_t.
  static constexpr auto value = DataType::kINT64;
};
