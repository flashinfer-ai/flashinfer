#pragma once

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cassert>
#include <cstdint>
#include <cuda/ptx>
#include <cuda/std/limits>
#include <stdexcept>
#include <string>

// Hopper max dynamic shared memory per block (opt-in).
// This is the hardware limit reported by cudaDevAttrMaxSharedMemoryPerBlockOptin.
static constexpr size_t SM90_MAX_SMEM = 232448;

#define gpuErrchk(ans)                    \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }
#define ASSERT(cond, msg, args...) assert((cond) || !fprintf(stderr, (msg "\n"), args))

inline void gpuAssert(cudaError_t code, const char* file, int line) {
  if (code != cudaSuccess) {
    throw std::runtime_error(std::string("Colonels CUDA error: ") + cudaGetErrorString(code) +
                             " at " + file + ":" + std::to_string(line));
  }
}
// not gonna type all that
using fp8 = __nv_fp8_e4m3;
using bf16 = __nv_bfloat16;

#define IS_SM90A (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000)

__forceinline__ PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
  // Get pointer to cuTensorMapEncodeTiled
  cudaDriverEntryPointQueryResult driver_status;
  void* cuTensorMapEncodeTiled_ptr = nullptr;
  gpuErrchk(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr,
                                             12000, cudaEnableDefault, &driver_status));
  if (driver_status != cudaDriverEntryPointSuccess || cuTensorMapEncodeTiled_ptr == nullptr) {
    throw std::runtime_error("CUDA driver does not expose cuTensorMapEncodeTiled");
  }

  return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
}

constexpr __device__ __forceinline__ int32_t const_ceil(float num) {
  return (static_cast<float>(static_cast<int32_t>(num)) == num)
             ? static_cast<int32_t>(num)
             : static_cast<int32_t>(num) + ((num > 0) ? 1 : 0);
}

__device__ __forceinline__ float swiglu_mul(float x, float w) { return (x / (1 + __expf(-x))) * w; }

__device__ __forceinline__ float relu_sq(float x) { return x > 0 ? x * x : 0; }

__device__ __forceinline__ uint64_t matrix_descriptor_encode(uint64_t x) {
  return (((x) & 0x3FFFF) >> 4);
}

template <int BITS, int BASE = 4, int SHIFT = 3>
__device__ __forceinline__ int32_t swizzle(const int32_t i) {
  if constexpr (BITS == 0) return i;
  constexpr uint32_t S_MASK = ((1 << BITS) - 1) << (BASE + SHIFT);
  return i ^ ((i & S_MASK) >> SHIFT);
}

// Descriptor for a shared memory matrix.
// Implementation is derived from PTX guide:
// https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-descriptor-format
template <int SDO, int LDO, uint64_t S_MODE>
__device__ __forceinline__ uint64_t make_smem_descriptor(void* ptr) {
  // Convert shared memory pointer to integer
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  uint64_t desc = matrix_descriptor_encode(addr);
  desc |= ((uint64_t)LDO) << 16;
  desc |= ((uint64_t)SDO) << 32;
  // I don't think we need this anymore
  // desc |= (((uint64_t)addr >> 0x7) & 0x7) << 49;
  desc |= S_MODE << 62;
  return desc;
}

__device__ __forceinline__ void warpgroup_arrive() {
#if IS_SM90A
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void warpgroup_commit_batch() {
#if IS_SM90A
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif
}

template <int N = 0>
__device__ __forceinline__ void warpgroup_wait() {
#if IS_SM90A
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
#endif
}

template <int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma8(float d[1][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 "
      "{%0,   %1,   %2,   %3},   "
      " %4,"
      " %5,"
      " %6, %7, %8;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma16(float d[2][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7},   "
      " %8,"
      " %9,"
      " %10, %11, %12;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma24(float d[3][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n24k32.f32.e4m3.e4m3 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11},  "
      " %12,"
      " %13,"
      " %14, %15, %16;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma32(float d[4][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  "
      "%13,  %14,  %15},  "
      " %16,"
      " %17,"
      " %18, %19, %20;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma40(float d[5][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n40k32.f32.e4m3.e4m3 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  "
      "%13,  %14,  %15,  %16,  %17,  %18,  %19},  "
      " %20,"
      " %21,"
      " %22, %23, %24;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]),
        "+f"(d[4][2]), "+f"(d[4][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma48(float d[6][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n48k32.f32.e4m3.e4m3 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  "
      "%13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23},  "
      " %24,"
      " %25,"
      " %26, %27, %28;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]),
        "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma56(float d[7][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n56k32.f32.e4m3.e4m3 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  "
      "%13,  %14,  "
      "%15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  %26,  %27},  "
      " %28,"
      " %29,"
      " %30, %31, %32;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]),
        "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma64(float d[8][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  "
      "%13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  "
      "%26, "
      " %27,  %28,  %29,  %30,  %31},  "
      " %32,"
      " %33,"
      " %34, %35, %36;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]),
        "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]),
        "+f"(d[7][2]), "+f"(d[7][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma72(float d[9][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  "
      "%13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  "
      "%26, "
      " %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  %35},  "
      " %36,"
      " %37,"
      " %38, %39, %40;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]),
        "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]),
        "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma80(float d[10][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n80k32.f32.e4m3.e4m3 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  "
      "%13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  "
      "%26, "
      " %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39},  "
      " %40,"
      " %41,"
      " %42, %43, %44;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]),
        "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]),
        "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
        "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma88(float d[11][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n88k32.f32.e4m3.e4m3 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14, "
      " "
      "%15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  %26,  %27,  %28,  %29,  "
      "%30,  %31,  %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  %40,  %41,  %42,  %43},  "
      " %44,"
      " %45,"
      " %46, %47, %48;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]),
        "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]),
        "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
        "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[10][0]), "+f"(d[10][1]),
        "+f"(d[10][2]), "+f"(d[10][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma96(float d[12][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n96k32.f32.e4m3.e4m3 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  "
      "%13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  "
      "%26, "
      " %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
      "%40,  %41,  %42,  %43,  %44,  %45,  %46,  %47},  "
      " %48,"
      " %49,"
      " %50, %51, %52;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]),
        "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]),
        "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
        "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[10][0]), "+f"(d[10][1]),
        "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]),
        "+f"(d[11][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma104(float d[13][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n104k32.f32.e4m3.e4m3 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  "
      "%13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  "
      "%26, "
      " %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
      "%40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  %48,  %49,  %50,  %51},  "
      " %52,"
      " %53,"
      " %54, %55, %56;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]),
        "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]),
        "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
        "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[10][0]), "+f"(d[10][1]),
        "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]),
        "+f"(d[11][3]), "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma112(float d[14][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n112k32.f32.e4m3.e4m3 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14, "
      " "
      "%15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  %26,  %27,  %28,  %29,  "
      "%30,  %31,  %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  %40,  %41,  %42,  %43,  %44,  "
      "%45,  %46,  %47,  %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55},  "
      " %56,"
      " %57,"
      " %58, %59, %60;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]),
        "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]),
        "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
        "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[10][0]), "+f"(d[10][1]),
        "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]),
        "+f"(d[11][3]), "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]),
        "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma120(float d[15][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n120k32.f32.e4m3.e4m3 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14, "
      " "
      "%15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  %26,  %27,  %28,  %29,  "
      "%30,  %31,  %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  %40,  %41,  %42,  %43,  %44,  "
      "%45,  %46,  %47,  %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  %56,  %57,  %58,  %59}, "
      " "
      " %60,"
      " %61,"
      " %62, %63, %64;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]),
        "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]),
        "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
        "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[10][0]), "+f"(d[10][1]),
        "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]),
        "+f"(d[11][3]), "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]),
        "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]), "+f"(d[14][0]),
        "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma128(float d[16][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  "
      "%13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  %26, "
      " %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
      "%40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  %48,  %49,  %50,  %51,  %52,  %53, "
      " %54,  %55,  %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},  "
      " %64,"
      " %65,"
      " %66, %67, %68;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]),
        "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]),
        "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
        "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[10][0]), "+f"(d[10][1]),
        "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]),
        "+f"(d[11][3]), "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]),
        "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]), "+f"(d[14][0]),
        "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]), "+f"(d[15][0]), "+f"(d[15][1]),
        "+f"(d[15][2]), "+f"(d[15][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB, int BM>
__device__ __forceinline__ void wgmma(float d[BM / 8][4], uint64_t desc_a, uint64_t desc_b) {
  if constexpr (BM == 8)
    wgmma8<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
  else if constexpr (BM == 16)
    wgmma16<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
  else if constexpr (BM == 24)
    wgmma24<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
  else if constexpr (BM == 32)
    wgmma32<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
  else if constexpr (BM == 40)
    wgmma40<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
  else if constexpr (BM == 48)
    wgmma48<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
  else if constexpr (BM == 56)
    wgmma56<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
  else if constexpr (BM == 64)
    wgmma64<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
  else if constexpr (BM == 72)
    wgmma72<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
  else if constexpr (BM == 80)
    wgmma80<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
  else if constexpr (BM == 88)
    wgmma88<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
  else if constexpr (BM == 96)
    wgmma96<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
  else if constexpr (BM == 104)
    wgmma104<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
  else if constexpr (BM == 112)
    wgmma112<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
  else if constexpr (BM == 120)
    wgmma120<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
  else if constexpr (BM == 128)
    wgmma128<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA = 0, int TransB = 0>
__device__ __forceinline__ void wgmma8_bf16(float d[1][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
      "{%0,   %1,   %2,   %3},   "
      " %4,"
      " %5,"
      " %6, %7, %8, %9, %10;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
        "n"(int32_t(TransA)), "n"(int32_t(TransB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA = 0, int TransB = 0>
__device__ __forceinline__ void wgmma16_bf16(float d[2][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7},   "
      " %8,"
      " %9,"
      " %10, %11, %12, %13, %14;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
        "n"(int32_t(TransA)), "n"(int32_t(TransB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA = 0, int TransB = 0>
__device__ __forceinline__ void wgmma24_bf16(float d[3][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11},  "
      " %12,"
      " %13,"
      " %14, %15, %16, %17, %18;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
        "n"(int32_t(TransA)), "n"(int32_t(TransB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA = 0, int TransB = 0>
__device__ __forceinline__ void wgmma32_bf16(float d[4][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  "
      "%13,  %14,  %15},  "
      " %16,"
      " %17,"
      " %18, %19, %20, %21, %22;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
        "n"(int32_t(TransA)), "n"(int32_t(TransB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA = 0, int TransB = 0>
__device__ __forceinline__ void wgmma40_bf16(float d[5][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  "
      "%13,  %14,  %15,  %16,  %17,  %18,  %19},  "
      " %20,"
      " %21,"
      " %22, %23, %24, %25, %26;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]),
        "+f"(d[4][2]), "+f"(d[4][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
        "n"(int32_t(TransA)), "n"(int32_t(TransB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA = 0, int TransB = 0>
__device__ __forceinline__ void wgmma48_bf16(float d[6][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  "
      "%13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23},  "
      " %24,"
      " %25,"
      " %26, %27, %28, %29, %30;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]),
        "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
        "n"(int32_t(TransA)), "n"(int32_t(TransB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA = 0, int TransB = 0>
__device__ __forceinline__ void wgmma56_bf16(float d[7][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  "
      "%13,  %14,  "
      "%15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  %26,  %27},  "
      " %28,"
      " %29,"
      " %30, %31, %32, %33, %34;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]),
        "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
        "n"(int32_t(TransA)), "n"(int32_t(TransB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA = 0, int TransB = 0>
__device__ __forceinline__ void wgmma64_bf16(float d[8][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  "
      "%13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  "
      "%26, "
      " %27,  %28,  %29,  %30,  %31},  "
      " %32,"
      " %33,"
      " %34, %35, %36, %37, %38;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]),
        "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]),
        "+f"(d[7][2]), "+f"(d[7][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
        "n"(int32_t(TransA)), "n"(int32_t(TransB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA = 0, int TransB = 0>
__device__ __forceinline__ void wgmma72_bf16(float d[9][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n72k16.f32.bf16.bf16 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  "
      "%13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  "
      "%26, "
      " %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  %35},  "
      " %36,"
      " %37,"
      " %38, %39, %40, %41, %42;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]),
        "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]),
        "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
        "n"(int32_t(TransA)), "n"(int32_t(TransB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA = 0, int TransB = 0>
__device__ __forceinline__ void wgmma80_bf16(float d[10][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n80k16.f32.bf16.bf16 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  "
      "%13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  "
      "%26, "
      " %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39},  "
      " %40,"
      " %41,"
      " %42, %43, %44, %45, %46;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]),
        "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]),
        "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
        "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
        "n"(int32_t(TransA)), "n"(int32_t(TransB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA = 0, int TransB = 0>
__device__ __forceinline__ void wgmma88_bf16(float d[11][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n88k16.f32.bf16.bf16 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14, "
      " "
      "%15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  %26,  %27,  %28,  %29,  "
      "%30,  %31,  %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  %40,  %41,  %42,  %43},  "
      " %44,"
      " %45,"
      " %46, %47, %48, %49, %50;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]),
        "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]),
        "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
        "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[10][0]), "+f"(d[10][1]),
        "+f"(d[10][2]), "+f"(d[10][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
        "n"(int32_t(TransA)), "n"(int32_t(TransB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA = 0, int TransB = 0>
__device__ __forceinline__ void wgmma96_bf16(float d[12][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  "
      "%13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  "
      "%26, "
      " %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
      "%40,  %41,  %42,  %43,  %44,  %45,  %46,  %47},  "
      " %48,"
      " %49,"
      " %50, %51, %52, %53, %54;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]),
        "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]),
        "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
        "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[10][0]), "+f"(d[10][1]),
        "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]),
        "+f"(d[11][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
        "n"(int32_t(TransA)), "n"(int32_t(TransB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA = 0, int TransB = 0>
__device__ __forceinline__ void wgmma104_bf16(float d[13][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n104k16.f32.bf16.bf16 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  "
      "%13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  "
      "%26, "
      " %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
      "%40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  %48,  %49,  %50,  %51},  "
      " %52,"
      " %53,"
      " %54, %55, %56, %57, %58;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]),
        "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]),
        "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
        "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[10][0]), "+f"(d[10][1]),
        "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]),
        "+f"(d[11][3]), "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
        "n"(int32_t(TransA)), "n"(int32_t(TransB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA = 0, int TransB = 0>
__device__ __forceinline__ void wgmma112_bf16(float d[14][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n112k16.f32.bf16.bf16 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14, "
      " "
      "%15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  %26,  %27,  %28,  %29,  "
      "%30,  %31,  %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  %40,  %41,  %42,  %43,  %44,  "
      "%45,  %46,  %47,  %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55},  "
      " %56,"
      " %57,"
      " %58, %59, %60, %61, %62;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]),
        "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]),
        "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
        "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[10][0]), "+f"(d[10][1]),
        "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]),
        "+f"(d[11][3]), "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]),
        "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
        "n"(int32_t(TransA)), "n"(int32_t(TransB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA = 0, int TransB = 0>
__device__ __forceinline__ void wgmma120_bf16(float d[15][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n120k16.f32.bf16.bf16 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14, "
      " "
      "%15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  %26,  %27,  %28,  %29,  "
      "%30,  %31,  %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  %40,  %41,  %42,  %43,  %44,  "
      "%45,  %46,  %47,  %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  %56,  %57,  %58,  %59}, "
      " "
      " %60,"
      " %61,"
      " %62, %63, %64, %65, %66;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]),
        "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]),
        "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
        "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[10][0]), "+f"(d[10][1]),
        "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]),
        "+f"(d[11][3]), "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]),
        "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]), "+f"(d[14][0]),
        "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
        "n"(int32_t(TransA)), "n"(int32_t(TransB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA = 0, int TransB = 0>
__device__ __forceinline__ void wgmma128_bf16(float d[16][4], uint64_t desc_a, uint64_t desc_b) {
#if IS_SM90A
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  "
      "%13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  %26, "
      " %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
      "%40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  %48,  %49,  %50,  %51,  %52,  %53, "
      " %54,  %55,  %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},  "
      " %64,"
      " %65,"
      " %66, %67, %68, %69, %70;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]),
        "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]),
        "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
        "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[10][0]), "+f"(d[10][1]),
        "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]),
        "+f"(d[11][3]), "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]),
        "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]), "+f"(d[14][0]),
        "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]), "+f"(d[15][0]), "+f"(d[15][1]),
        "+f"(d[15][2]), "+f"(d[15][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
        "n"(int32_t(TransA)), "n"(int32_t(TransB)));
#endif
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB, int BM>
__device__ __forceinline__ void wgmma_bf16(float d[BM / 8][4], uint64_t desc_a, uint64_t desc_b) {
  if constexpr (BM == 8)
    wgmma8_bf16<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, desc_a, desc_b);
  else if constexpr (BM == 16)
    wgmma16_bf16<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, desc_a, desc_b);
  else if constexpr (BM == 24)
    wgmma24_bf16<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, desc_a, desc_b);
  else if constexpr (BM == 32)
    wgmma32_bf16<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, desc_a, desc_b);
  else if constexpr (BM == 40)
    wgmma40_bf16<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, desc_a, desc_b);
  else if constexpr (BM == 48)
    wgmma48_bf16<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, desc_a, desc_b);
  else if constexpr (BM == 56)
    wgmma56_bf16<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, desc_a, desc_b);
  else if constexpr (BM == 64)
    wgmma64_bf16<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, desc_a, desc_b);
  else if constexpr (BM == 72)
    wgmma72_bf16<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, desc_a, desc_b);
  else if constexpr (BM == 80)
    wgmma80_bf16<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, desc_a, desc_b);
  else if constexpr (BM == 88)
    wgmma88_bf16<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, desc_a, desc_b);
  else if constexpr (BM == 96)
    wgmma96_bf16<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, desc_a, desc_b);
  else if constexpr (BM == 104)
    wgmma104_bf16<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, desc_a, desc_b);
  else if constexpr (BM == 112)
    wgmma112_bf16<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, desc_a, desc_b);
  else if constexpr (BM == 120)
    wgmma120_bf16<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, desc_a, desc_b);
  else if constexpr (BM == 128)
    wgmma128_bf16<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, desc_a, desc_b);
}

template <int CTA_GROUP = 1>
__device__ __forceinline__ uint32_t tma_barrier_address(uint64_t* bar) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  // Group-2 transfers from either CTA complete the even CTA's barrier.
  if constexpr (CTA_GROUP == 2) addr &= ~(1u << 24);
  return addr;
}

template <int CLUSTERS = 1, int CTA_GROUP = 1>
__device__ inline void load_async(void* dst, void const* const src_tma_map, uint64_t* bar,
                                  int global_col_idx, int global_row_idx,
                                  uint16_t multicast_mask = uint16_t((1 << CLUSTERS) - 1)) {
  uint64_t tma_ptr = reinterpret_cast<uint64_t>(src_tma_map);
  uint32_t mbar_ptr = tma_barrier_address<CTA_GROUP>(bar);
  uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

  if constexpr (CTA_GROUP == 2) {
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global."
        "mbarrier::complete_tx::bytes.multicast::cluster "
        "[%0], [%1, {%3, %4}], [%2], %5;" ::"r"(dst_ptr),
        "l"(tma_ptr), "r"(mbar_ptr), "r"(global_row_idx), "r"(global_col_idx), "h"(multicast_mask)
        : "memory");
  } else if constexpr (CLUSTERS == 1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];"
        :
        : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr), "r"(global_row_idx), "r"(global_col_idx)
        : "memory");
  } else {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes.multicast::cluster "
        "[%0], [%1, {%3, %4}], [%2], %5;"
        :
        : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr), "r"(global_row_idx), "r"(global_col_idx),
          "h"(multicast_mask)
        : "memory");
  }
}

template <int CTA_GROUP = 1, bool EVICT_FIRST = false>
__device__ inline void load_async_3d(void* dst, void const* const src_tma_map, uint64_t* bar,
                                     int global_dim0_idx, int global_dim1_idx,
                                     int global_dim2_idx) {
  uint64_t tma_ptr = reinterpret_cast<uint64_t>(src_tma_map);
  uint32_t mbar_ptr = tma_barrier_address<CTA_GROUP>(bar);
  uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

  if constexpr (EVICT_FIRST) {
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
    if constexpr (CTA_GROUP == 2) {
      asm volatile(
          "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global."
          "mbarrier::complete_tx::bytes.L2::cache_hint "
          "[%0], [%1, {%3, %4, %5}], [%2], %6;" ::"r"(dst_ptr),
          "l"(tma_ptr), "r"(mbar_ptr), "r"(global_dim0_idx), "r"(global_dim1_idx),
          "r"(global_dim2_idx), "l"(policy)
          : "memory");
    } else {
      asm volatile(
          "cp.async.bulk.tensor.3d.shared::cta.global.tile."
          "mbarrier::complete_tx::bytes.L2::cache_hint "
          "[%0], [%1, {%3, %4, %5}], [%2], %6;" ::"r"(dst_ptr),
          "l"(tma_ptr), "r"(mbar_ptr), "r"(global_dim0_idx), "r"(global_dim1_idx),
          "r"(global_dim2_idx), "l"(policy)
          : "memory");
    }
  } else if constexpr (CTA_GROUP == 2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global."
        "mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];" ::"r"(dst_ptr),
        "l"(tma_ptr), "r"(mbar_ptr), "r"(global_dim0_idx), "r"(global_dim1_idx),
        "r"(global_dim2_idx)
        : "memory");
  } else {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4, %5}], [%2];"
        :
        : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr), "r"(global_dim0_idx), "r"(global_dim1_idx),
          "r"(global_dim2_idx)
        : "memory");
  }
}

__device__ inline void load_async_3d_evict_first(void* dst, const void* tensor_map, uint64_t* bar,
                                                 int dim0, int dim1, int dim2) {
  uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.tile."
      "mbarrier::complete_tx::bytes.L2::cache_hint "
      "[%0], [%1, {%3, %4, %5}], [%2], %6;" ::"r"(
          static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
      "l"(tensor_map), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(bar))), "r"(dim0),
      "r"(dim1), "r"(dim2), "l"(policy)
      : "memory");
}

#define CP_ASYNC_CG(dst, src, Bytes)                                                        \
  asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), \
               "n"(Bytes))

#define CP_ASYNC_CG4(dst, src, Bytes)                                                      \
  asm volatile("cp.async.ca.shared.global.L2::64B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), \
               "n"(Bytes))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))

__device__ __forceinline__ void init_barrier(uint64_t* bar, int thread_count,
                                             int transaction_count) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr),
               "r"(thread_count + transaction_count));
}

__device__ __forceinline__ void cp_async_mbarrier_arrive(uint64_t* bar) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"r"(bar_ptr));
}

// CTA_RANK selects one peer; -1 publishes to every CTA in the cluster.
template <int CLUSTERS = 1, int CTA_RANK = -1>
__device__ __forceinline__ void arrive(uint64_t* bar, uint32_t count = 1) {
  static_assert(CTA_RANK >= -1 && CTA_RANK < CLUSTERS);
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  if constexpr (CLUSTERS == 1) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0], %1;\n"
                 :
                 : "r"(mbar_ptr), "r"(count)
                 : "memory");
  } else {
    constexpr int first_rank = CTA_RANK < 0 ? 0 : CTA_RANK;
    constexpr int end_rank = CTA_RANK < 0 ? CLUSTERS : CTA_RANK + 1;
    for (int rank = first_rank; rank < end_rank; ++rank) {
      uint32_t peer_ptr;
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                   : "=r"(peer_ptr)
                   : "r"(mbar_ptr), "r"(rank));
      asm volatile(
          "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0], %1;" ::"r"(peer_ptr),
          "r"(count)
          : "memory");
    }
  }
}

template <int CTA_GROUP>
__device__ __forceinline__ void arrive_mma_leader(uint64_t* bar, int cta_rank) {
  if constexpr (CTA_GROUP == 1) {
    arrive(bar);
  } else {
    uint32_t peer = __cvta_generic_to_shared(bar) & 0xFEFFFFFF;
    asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];" ::"r"(peer) : "memory");
  }
}

__device__ __forceinline__ void expect_bytes(uint64_t* bar, uint32_t bytes) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar_ptr),
      "r"(bytes));
}

template <int CLUSTERS = 1>
__device__ __forceinline__ void wait(uint64_t* bar, int kPhaseBit) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  if constexpr (CLUSTERS > 1) {
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "WAIT:\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\n"
        "@!P1 bra.uni WAIT;\n"
        "}\n" ::"r"(mbar_ptr),
        "r"(kPhaseBit)
        : "memory");
  } else {
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "WAIT:\n"
        "mbarrier.try_wait.parity.shared.b64 P1, [%0], %1;\n"
        "@!P1 bra.uni WAIT;\n"
        "}\n" ::"r"(mbar_ptr),
        "r"(kPhaseBit)
        : "memory");
  }
}

__device__ __forceinline__ void ld_matrix_x2(uint32_t* tile, uint32_t mat) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
               : "=r"(tile[0]), "=r"(tile[1])
               : "r"(mat));
}

__device__ __forceinline__ void ld_matrix_x4(uint32_t* tile, uint32_t mat) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(tile[0]), "=r"(tile[1]), "=r"(tile[2]), "=r"(tile[3])
               : "r"(mat));
}

__device__ __forceinline__ void st_matrix_x4(uint32_t* tile, uint32_t mat) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};"
               :
               : "r"(mat), "r"(tile[0]), "r"(tile[1]), "r"(tile[2]), "r"(tile[3])
               : "memory");
}

__device__ __forceinline__ void st_matrix_x4_trans(uint32_t* tile, uint32_t mat) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};"
               :
               : "r"(mat), "r"(tile[0]), "r"(tile[1]), "r"(tile[2]), "r"(tile[3])
               : "memory");
}

// Same as above but takes the four 8x8 matrices as explicit registers, for
// callers whose tiles are not contiguous in the register array (e.g. tiles
// paired along N instead of along M).
__device__ __forceinline__ void st_matrix_x4_trans(uint32_t r0, uint32_t r1, uint32_t r2,
                                                   uint32_t r3, uint32_t mat) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};"
               :
               : "r"(mat), "r"(r0), "r"(r1), "r"(r2), "r"(r3)
               : "memory");
}

__device__ __forceinline__ void st_matrix_x2_trans(uint32_t* tile, uint32_t mat) {
  asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};"
               :
               : "r"(mat), "r"(tile[0]), "r"(tile[1])
               : "memory");
}

__device__ __forceinline__ void st_matrix_x1_trans(uint32_t* tile, uint32_t mat) {
  asm volatile("stmatrix.sync.aligned.m8n8.x1.trans.shared.b16 [%0], {%1};"
               :
               : "r"(mat), "r"(tile[0])
               : "memory");
}

template <uint32_t RegCount>
__device__ void reg_alloc() {
#if IS_SM90A
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
#endif
}

template <uint32_t RegCount>
__device__ void reg_dealloc() {
#if IS_SM90A
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
#endif
}

__device__ __forceinline__ void red_add_bf16x8(__nv_bfloat16* gptr, float4 v) {
  const uint32_t* u = reinterpret_cast<const uint32_t*>(&v);
  asm volatile("red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4};" ::"l"(gptr), "r"(u[0]),
               "r"(u[1]), "r"(u[2]), "r"(u[3])
               : "memory");
}

template <int CLUSTERS = 1, int CTA_GROUP = 1>
__device__ inline void load_async_2d_gather4(void* dst, const void* tensor_map, uint64_t* bar,
                                             int col, int4 rows,
                                             uint16_t multicast_mask = uint16_t((1 << CLUSTERS) -
                                                                                1)) {
  if constexpr (CTA_GROUP == 2) {
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile::gather4."
        "mbarrier::complete_tx::bytes.multicast::cluster "
        "[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;" ::"r"(
            static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
        "l"(tensor_map), "r"(col), "r"(rows.x), "r"(rows.y), "r"(rows.z), "r"(rows.w),
        "r"(tma_barrier_address<CTA_GROUP>(bar)), "h"(multicast_mask)
        : "memory");
  } else if constexpr (CLUSTERS == 1) {
    const int32_t coords[5] = {col, rows.x, rows.y, rows.z, rows.w};
    cuda::ptx::cp_async_bulk_tensor_tile_gather4(cuda::ptx::space_shared, cuda::ptx::space_global,
                                                 dst, tensor_map, coords, bar);
  } else {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4."
        "mbarrier::complete_tx::bytes.multicast::cluster "
        "[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;" ::"r"(
            static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
        "l"(tensor_map), "r"(col), "r"(rows.x), "r"(rows.y), "r"(rows.z), "r"(rows.w),
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(bar))), "h"(multicast_mask)
        : "memory");
  }
}

template <int CTA_GROUP = 1>
__device__ __forceinline__ void tmem_alloc(uint32_t addr, int columns) {
  asm volatile("tcgen05.alloc.cta_group::%2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(addr),
               "r"(columns), "n"(CTA_GROUP));
}

template <int CTA_GROUP = 1>
__device__ __forceinline__ void tmem_dealloc(const uint32_t addr, int columns) {
  asm volatile("tcgen05.dealloc.cta_group::%2.sync.aligned.b32 %0, %1;" ::"r"(addr), "r"(columns),
               "n"(CTA_GROUP));
}

template <int M, int N, int CTA_GROUP = 1>
constexpr uint32_t build_idesc() {
  static_assert(N % 16 == 0 && N >= 16 && N <= 256, "invalid N shape for MMA");
  static_assert(M == 64 || M == 128 || (M == 256 && CTA_GROUP == 2), "invalid M shape for MMA");
  return (1U << 4U)                    // dtype=FP32
         | (1U << 7U)                  // atype=BF16
         | (1U << 10U)                 // btype=BF16
         | ((uint32_t)N >> 3U << 17U)  // MMA_N
         | ((uint32_t)M >> 4U << 24U)  // MMA_M
      ;
}

__device__ inline uint32_t elect_sync(uint32_t mask = 0xFFFFFFFF) {
  uint32_t pred = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred %%px;\n\t"
      "elect.sync _|%%px, %1;\n\t"
      "@%%px mov.s32 %0, 1;\n\t"
      "}"
      : "+r"(pred)
      : "r"(mask));
  return pred;
}

// https://github.com/NVIDIA/cutlass/blob/v4.3.1/include/cute/arch/mma_sm100_umma.hpp#L86
template <int CTA_GROUP = 1>
__device__ inline void tcgen05_mma_f16(int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
                                       int enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"  // predicate register enable-input-d
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::%5.kind::f16 [%0], %1, %2, %3, p;\n\t"
      "}" ::"r"(taddr),
      "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(enable_input_d), "n"(CTA_GROUP));
}

template <int CTA_GROUP = 1>
__device__ inline void tcgen05_mma_f16_tmem(int taddr, int a_tmem, uint64_t b_desc, uint32_t i_desc,
                                            int enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"  // predicate register enable-input-d
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::%5.kind::f16 [%0], [%1], %2, %3, p;\n\t"
      "}" ::"r"(taddr),
      "r"(a_tmem), "l"(b_desc), "r"(i_desc), "r"(enable_input_d), "n"(CTA_GROUP));
}

template <int CTA_GROUP = 1, int CLUSTERS = CTA_GROUP>
__device__ inline void tcgen05_commit(int mbar_addr,
                                      uint16_t multicast_mask = uint16_t((1 << CLUSTERS) - 1)) {
  if constexpr (CLUSTERS > 1) {
    asm volatile(
        "tcgen05.commit.cta_group::%1.mbarrier::arrive::one.shared::cluster."
        "multicast::cluster.b64 [%0], %2;" ::"r"(mbar_addr),
        "n"(CTA_GROUP), "h"(multicast_mask)
        : "memory");
  } else {
    asm volatile("tcgen05.commit.cta_group::%1.mbarrier::arrive::one.b64 [%0];" ::"r"(mbar_addr),
                 "n"(CTA_GROUP)
                 : "memory");
  }
}

template <bool MULTICAST = true>
__device__ __forceinline__ void clc_try_cancel(int4* response, uint64_t* bar) {
  const uint32_t response_addr = static_cast<uint32_t>(__cvta_generic_to_shared(response));
  const uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  if constexpr (MULTICAST) {
    asm volatile(
        "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::"
        "bytes."
        "multicast::cluster::all.b128 [%0], [%1];" ::"r"(response_addr),
        "r"(mbar_addr)
        : "memory");
  } else {
    asm volatile(
        "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::"
        "bytes."
        "b128 [%0], [%1];" ::"r"(response_addr),
        "r"(mbar_addr)
        : "memory");
  }
}

__device__ __forceinline__ void clc_query_response(const int4* response, uint32_t& is_valid,
                                                   uint32_t& new_ctaid) {
  const uint32_t response_addr = static_cast<uint32_t>(__cvta_generic_to_shared(response));
  asm volatile(
      "{\n\t"
      ".reg .b128 handle;\n\t"
      ".reg .pred p;\n\t"
      "ld.shared.b128 handle, [%2];\n\t"
      "mov.s32 %0, 0;\n\t"
      "mov.s32 %1, 0;\n\t"
      "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, handle;\n\t"
      "@p mov.s32 %0, 1;\n\t"
      "@p clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %1, handle;\n\t"
      "}"
      : "=r"(is_valid), "=r"(new_ctaid)
      : "r"(response_addr)
      : "memory");
  // Finish generic-proxy reads before the scheduler reuses this response via the async proxy.
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
