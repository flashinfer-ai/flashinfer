// Copyright (c) 2026 FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace flashinfer {
namespace sm90_nvfp4_rs {

constexpr int kTileN = 64;
constexpr int kTileK = 16;
constexpr int kRsThreads = 128;
constexpr int kRsBytesPerThread = 4;
constexpr int kValuesPerThread = kRsBytesPerThread * 2;

__device__ __forceinline__ uint16_t e2m1_bf16_bits(uint32_t code) {
  const uint32_t magnitude = code & 7U;
  const uint16_t value =
      magnitude == 0 ? 0 : (magnitude == 1 ? 0x3f00U : 0x3f80U + (magnitude - 2U) * 0x40U);
  return static_cast<uint16_t>(value | ((code & 8U) << 12));
}

__device__ __forceinline__ float decode_e2m1(uint8_t code) {
  const uint16_t bits = e2m1_bf16_bits(code);
  return __bfloat162float(__ushort_as_bfloat16(bits));
}

__device__ __forceinline__ float decode_e4m3(uint8_t raw) {
  __nv_fp8_e4m3 value;
  value.__x = static_cast<__nv_fp8_storage_t>(raw);
  return static_cast<float>(value);
}

__device__ __forceinline__ uint32_t decode_pair_scaled_bf16(uint8_t packed, uint16_t scale) {
  const uint32_t bits = static_cast<uint32_t>(packed);
  const uint32_t low_code = bits & 7U;
  const uint32_t high_code = (bits >> 4) & 7U;
  const uint32_t low_selector = low_code | (high_code << 8);
  const uint32_t high_selector = (low_code << 4) | (high_code << 12);
  uint32_t low_bytes;
  uint32_t high_bytes;
  asm volatile("prmt.b32 %0, 0xc0800000, 0xc0804000, %1;\n" : "=r"(low_bytes) : "r"(low_selector));
  asm volatile("prmt.b32 %0, 0x3f3f3f00, 0x40404040, %1;\n"
               : "=r"(high_bytes)
               : "r"(high_selector));
  const uint32_t signs = ((bits & 0x08U) << 12) | ((bits & 0x80U) << 24);
  const uint32_t values = (low_bytes & 0x00ff00ffU) | (high_bytes & 0xff00ff00U) | signs;
  uint32_t result;
  asm volatile(
      "{\n"
      ".reg .b32 scale2;\n"
      "mov.b32 scale2, {%2, %2};\n"
      "mul.rn.bf16x2 %0, %1, scale2;\n"
      "}\n"
      : "=r"(result)
      : "r"(values), "h"(scale));
  return result;
}

template <int N>
struct WgmmaRsBf16;

template <>
struct WgmmaRsBf16<16> {
  static constexpr int kAccumulatorCount = 8;

  __device__ __forceinline__ static void mma(const uint32_t (&a)[4], uint64_t desc_b,
                                             float (&d)[kAccumulatorCount], bool accumulate) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %13, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, p, 1, 1, 0;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
          "+f"(d[7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b),
          "r"(static_cast<int32_t>(accumulate)));
  }
};

template <>
struct WgmmaRsBf16<32> {
  static constexpr int kAccumulatorCount = 16;

  __device__ __forceinline__ static void mma(const uint32_t (&a)[4], uint64_t desc_b,
                                             float (&d)[kAccumulatorCount], bool accumulate) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %21, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, "
        "{%16, %17, %18, %19}, %20, p, 1, 1, 0;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
          "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
          "+f"(d[14]), "+f"(d[15])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b),
          "r"(static_cast<int32_t>(accumulate)));
  }
};

template <>
struct WgmmaRsBf16<64> {
  static constexpr int kAccumulatorCount = 32;

  __device__ __forceinline__ static void mma(const uint32_t (&a)[4], uint64_t desc_b,
                                             float (&d)[kAccumulatorCount], bool accumulate) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, "
        "%24, %25, %26, %27, %28, %29, %30, %31}, "
        "{%32, %33, %34, %35}, %36, p, 1, 1, 0;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
          "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
          "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
          "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b),
          "r"(static_cast<int32_t>(accumulate)));
  }
};

template <>
struct WgmmaRsBf16<96> {
  static constexpr int kAccumulatorCount = 48;

  __device__ __forceinline__ static void mma(const uint32_t (&a)[4], uint64_t desc_b,
                                             float (&d)[kAccumulatorCount], bool accumulate) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %53, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, "
        "%24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, "
        "%40, %41, %42, %43, %44, %45, %46, %47}, "
        "{%48, %49, %50, %51}, %52, p, 1, 1, 0;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
          "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
          "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
          "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]),
          "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]),
          "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b),
          "r"(static_cast<int32_t>(accumulate)));
  }
};

template <>
struct WgmmaRsBf16<128> {
  static constexpr int kAccumulatorCount = 64;

  __device__ __forceinline__ static void mma(const uint32_t (&a)[4], uint64_t desc_b,
                                             float (&d)[kAccumulatorCount], bool accumulate) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %69, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, "
        "%24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, "
        "%40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, "
        "%56, %57, %58, %59, %60, %61, %62, %63}, "
        "{%64, %65, %66, %67}, %68, p, 1, 1, 0;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
          "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
          "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
          "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]),
          "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]),
          "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]),
          "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
          "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]),
          "+f"(d[63])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b),
          "r"(static_cast<int32_t>(accumulate)));
  }
};

__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int Pending>
__device__ __forceinline__ void wgmma_wait() {
  static_assert(Pending >= 0 && Pending <= 7);
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" : : "n"(Pending) : "memory");
}

__device__ __forceinline__ uint32_t float2_to_bf16x2(float low, float high) {
  uint32_t result;
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(result) : "f"(high), "f"(low));
  return result;
}

__device__ __forceinline__ void stsm_t_x4(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3,
                                          void* destination) {
  asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16.trans [%0], {%1, %2, %3, %4};\n"
               :
               : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(destination))), "r"(x0),
                 "r"(x1), "r"(x2), "r"(x3)
               : "memory");
}

}  // namespace sm90_nvfp4_rs
}  // namespace flashinfer
