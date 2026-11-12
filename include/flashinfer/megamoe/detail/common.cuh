#pragma once

#include <cuda/std/cstdint>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/int_tuple.hpp>
#include <cutlass/detail/helper_macros.hpp>

// ============ compile.cuh ============
#if defined(__NVCC__) or (defined(__clang__) and defined(__CUDA__)) or defined(__CUDACC_RTC__) or \
    defined(__CLION_IDE__)
#define FI_MEGAMOE_IN_CUDA_COMPILATION
#endif

#if defined(__NVCC__) || (defined(__clang__) and defined(__CUDA__))
#define CUTLASS_HOST_DEVICE_NOINLINE __device__ __host__
#define CUTLASS_DEVICE_NOINLINE __device__
#elif defined(__CUDACC_RTC__)
#define CUTLASS_HOST_DEVICE_NOINLINE __device__
#define CUTLASS_DEVICE_NOINLINE __device__
#else
#define CUTLASS_HOST_DEVICE_NOINLINE
#define CUTLASS_DEVICE_NOINLINE
#endif

// ============ exception.cuh ============
// Assertions live in CUTLASS_HOST_DEVICE helpers (workspace sizing is shared by
// the launcher and the kernel), so the failure path must also be legal in a host
// pass -- a "trap;" inline asm is not.
#ifndef FI_MEGAMOE_TRAP
#if defined(__CUDA_ARCH__)
#define FI_MEGAMOE_TRAP() asm("trap;")
#else
#define FI_MEGAMOE_TRAP() __builtin_trap()
#endif
#endif

#ifdef __CLION_IDE__

CUTLASS_HOST_DEVICE void host_device_printf(const char* format, ...) { FI_MEGAMOE_TRAP(); }

#define printf host_device_printf
#endif

#ifndef FI_MEGAMOE_DEVICE_ASSERT
#define FI_MEGAMOE_DEVICE_ASSERT(cond)                                               \
  do {                                                                               \
    if (not(cond)) {                                                                 \
      printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond); \
      FI_MEGAMOE_TRAP();                                                             \
    }                                                                                \
  } while (0)
#endif

#ifndef FI_MEGAMOE_TRAP_ONLY_DEVICE_ASSERT
#define FI_MEGAMOE_TRAP_ONLY_DEVICE_ASSERT(cond) \
  do {                                           \
    if (not(cond)) FI_MEGAMOE_TRAP();            \
  } while (0)
#endif

#ifndef FI_MEGAMOE_STATIC_ASSERT
#define FI_MEGAMOE_STATIC_ASSERT(cond, ...) static_assert(cond, __VA_ARGS__)
#endif

#ifndef FI_MEGAMOE_UNIFIED_ASSERT
#ifdef FI_MEGAMOE_IN_CUDA_COMPILATION
#define FI_MEGAMOE_UNIFIED_ASSERT(cond) FI_MEGAMOE_DEVICE_ASSERT(cond)
#else
#define FI_MEGAMOE_UNIFIED_ASSERT(cond) FI_MEGAMOE_HOST_ASSERT(cond)
#endif
#endif

// ============ cute_tie.cuh ============
namespace cute {

struct ignore_t {
  template <typename T>
  constexpr const ignore_t& operator=(T&&) const noexcept {
    return *this;
  }
};

inline constexpr ignore_t ignore{};

}  // namespace cute

#define CUTE_TIE_CONCAT_IMPL(A, B) A##B
#define CUTE_TIE_CONCAT(A, B) CUTE_TIE_CONCAT_IMPL(A, B)

#define CUTE_TIE_GET_NTH_ARG(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N
#define CUTE_TIE_COUNT_ARGS(...) CUTE_TIE_GET_NTH_ARG(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)

#define CUTE_TIE_OP_DECL(I, TUPLE, VAR) auto VAR = ::cute::get<I>(TUPLE)
#define CUTE_TIE_OP_ASSIGN(I, TUPLE, VAR) VAR = ::cute::get<I>(TUPLE)

#define CUTE_TIE_APPLY_OP_1(OP, T, V1) OP(0, T, V1);
#define CUTE_TIE_APPLY_OP_2(OP, T, V1, V2) \
  OP(0, T, V1);                            \
  OP(1, T, V2);
#define CUTE_TIE_APPLY_OP_3(OP, T, V1, V2, V3) \
  OP(0, T, V1);                                \
  OP(1, T, V2);                                \
  OP(2, T, V3);
#define CUTE_TIE_APPLY_OP_4(OP, T, V1, V2, V3, V4) \
  OP(0, T, V1);                                    \
  OP(1, T, V2);                                    \
  OP(2, T, V3);                                    \
  OP(3, T, V4);
#define CUTE_TIE_APPLY_OP_5(OP, T, V1, V2, V3, V4, V5) \
  OP(0, T, V1);                                        \
  OP(1, T, V2);                                        \
  OP(2, T, V3);                                        \
  OP(3, T, V4);                                        \
  OP(4, T, V5);

#define CUTE_TIE_DECL(TUPLE_EXPR, ...)                                    \
  auto&& CUTE_TIE_CONCAT(cute_tie__temp_tuple_, __LINE__) = (TUPLE_EXPR); \
  CUTE_TIE_CONCAT(CUTE_TIE_APPLY_OP_, CUTE_TIE_COUNT_ARGS(__VA_ARGS__))(  \
      CUTE_TIE_OP_DECL, CUTE_TIE_CONCAT(cute_tie__temp_tuple_, __LINE__), __VA_ARGS__)

#define CUTE_TIE(TUPLE_EXPR, ...)                                                           \
  do {                                                                                      \
    auto&& CUTE_TIE_CONCAT(cute_tie__temp_tuple_, __LINE__) = (TUPLE_EXPR);                 \
    CUTE_TIE_CONCAT(CUTE_TIE_APPLY_OP_, CUTE_TIE_COUNT_ARGS(__VA_ARGS__))(                  \
        CUTE_TIE_OP_ASSIGN, CUTE_TIE_CONCAT(cute_tie__temp_tuple_, __LINE__), __VA_ARGS__); \
  } while (0)

// ============ types.cuh ============
namespace flashinfer::megamoe {

enum class MmaKind {
  BF16 = 0,
  MXFP8FP4 = 1,
};

enum class GemmType {
  Normal = 0,
  MGroupedContiguous = 1,
  MGroupedMasked = 2,
  KGroupedContiguous = 3,
  Batched = 4,
  MGroupedContiguousWithPsumLayout = 5,
};

enum class KernelType { Kernel1D1D = 0, Kernel1D2D = 1, KernelNoSF = 2 };

}  // namespace flashinfer::megamoe

// ============ utils.cuh ============
namespace flashinfer::megamoe::utils {

template <typename FuncT>
struct PatternVisitor {
  FuncT func;

  CUTLASS_HOST_DEVICE
  explicit PatternVisitor(FuncT&& func) : func(std::forward<FuncT>(func)) {}

  CUTLASS_HOST_DEVICE
  auto operator[](const uint32_t& i) const { return func(i); }
};

template <uint32_t kNumBytes>
struct Vectorized {
  static auto zeros() {
    // TODO: add `ulonglong4` for SM100 once `__ldg` support this
    if constexpr (kNumBytes > 0 and kNumBytes % 16 == 0) {
      return make_uint4(0, 0, 0, 0);
    } else if constexpr (kNumBytes > 0 and kNumBytes % 8 == 0) {
      return make_uint2(0, 0);
    } else if constexpr (kNumBytes > 0 and kNumBytes % 4 == 0) {
      return 0;
    } else {
      FI_MEGAMOE_STATIC_ASSERT(kNumBytes > 0 and kNumBytes % 4 == 0, "Invalid vectorization");
    }
  }

  using vec_t = decltype(zeros());
};

}  // namespace flashinfer::megamoe::utils

// ============ math.cuh ============
namespace flashinfer::megamoe::math {

/// Pointer operations
template <typename dtype_t = void>
CUTLASS_HOST_DEVICE dtype_t* advance_ptr(void* ptr, const uint64_t num_bytes) {
  return reinterpret_cast<dtype_t*>(static_cast<uint8_t*>(ptr) + num_bytes);
}

/// Math functions
template <typename T>
CUTLASS_HOST_DEVICE T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T, bool kDoCeilAlignment = true>
CUTLASS_HOST_DEVICE T align(T a, T b) {
  return (kDoCeilAlignment ? ceil_div(a, b) : (a / b)) * b;
}

template <typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_align(T a, T b) {
  return constexpr_ceil_div(a, b) * b;
}

template <typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_min(T a, T b) {
  return a < b ? a : b;
}

template <typename T>
CUTLASS_DEVICE void swap(T& a, T& b) {
  T temp = a;
  a = b;
  b = temp;
}

#ifdef FI_MEGAMOE_IN_CUDA_COMPILATION

CUTLASS_DEVICE float2 mul2(const float2& a, const float2& b) {
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)
  return __fmul2_rn(a, b);
#else
  return make_float2(__fmul_rn(a.x, b.x), __fmul_rn(a.y, b.y));
#endif
}

CUTLASS_HOST_DEVICE float fast_rcp(const float& x) {
  float ret;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(ret) : "f"(x));
  return ret;
}

/// Casting
template <typename old_t>
CUTLASS_DEVICE int cast_into_bf16_and_pack(old_t& x, old_t& y) {
  auto bf16x2 =
      __float22bfloat162_rn({*reinterpret_cast<float*>(&x), *reinterpret_cast<float*>(&y)});
  return *reinterpret_cast<int*>(&bf16x2);
}

CUTLASS_DEVICE float fast_pow2(const int& x) {
  uint32_t bits_x = (x + 127) << 23;
  return *reinterpret_cast<float*>(&bits_x);
}

CUTLASS_DEVICE int fast_log2_ceil(float x) {
  const auto bits = *reinterpret_cast<uint32_t*>(&x);
  const auto exp = bits >> 23;
  const auto man = bits & ((1 << 23) - 1);
  return exp - 127 + (man != 0);
}

template <bool kUseUE8M0 = true>
CUTLASS_DEVICE void get_e4m3_sf_and_sf_inv(const float2& amax, float2& sf, float2& sf_inv) {
  FI_MEGAMOE_STATIC_ASSERT(kUseUE8M0, "Must use UE8M0");
  const float2 finfo_factor = {1.0 / 448.0, 1.0 / 448.0};
  const auto scaled = mul2(amax, finfo_factor);
  const auto exp_x = fast_log2_ceil(scaled.x);
  const auto exp_y = fast_log2_ceil(scaled.y);
  sf.x = fast_pow2(exp_x), sf_inv.x = fast_pow2(-exp_x);
  sf.y = fast_pow2(exp_y), sf_inv.y = fast_pow2(-exp_y);
}

/// Reduction
CUTLASS_DEVICE uint32_t warp_inclusive_sum(uint32_t value, const uint32_t& lane_idx) {
#pragma unroll
  for (uint32_t offset = 1; offset < 32; offset <<= 1) {
    const uint32_t synced = __shfl_up_sync(0xffffffff, value, offset);
    if (lane_idx >= offset) value += synced;
  }
  return value;
}

// Operation functors
template <typename T>
struct ReduceSum {
  CUTLASS_DEVICE T operator()(T a, T b) const { return a + b; }
};
template <typename T>
struct ReduceMax {
  CUTLASS_DEVICE T operator()(T a, T b) const { return a > b ? a : b; }
};
template <typename T>
struct ReduceMin {
  CUTLASS_DEVICE T operator()(T a, T b) const { return a < b ? a : b; }
};
template <typename T>
struct ReduceAnd {
  CUTLASS_DEVICE T operator()(T a, T b) const { return a & b; }
};
template <typename T>
struct ReduceOr {
  CUTLASS_DEVICE T operator()(T a, T b) const { return a | b; }
};

// Unified reduction function
template <uint32_t kNumLanesPerGroup, bool kIntergroupReduce, typename T, typename Op>
CUTLASS_DEVICE T warp_reduce(T value, Op op) {
  FI_MEGAMOE_STATIC_ASSERT(kNumLanesPerGroup == 32 or kNumLanesPerGroup == 16 or
                               kNumLanesPerGroup == 8 or kNumLanesPerGroup == 4 or
                               kNumLanesPerGroup == 2 or kNumLanesPerGroup == 1,
                           "Invalid number of lanes");
  constexpr uint32_t mask = 0xffffffff;
  if constexpr (kIntergroupReduce) {
    if constexpr (kNumLanesPerGroup <= 1) value = op(value, __shfl_xor_sync(mask, value, 1));
    if constexpr (kNumLanesPerGroup <= 2) value = op(value, __shfl_xor_sync(mask, value, 2));
    if constexpr (kNumLanesPerGroup <= 4) value = op(value, __shfl_xor_sync(mask, value, 4));
    if constexpr (kNumLanesPerGroup <= 8) value = op(value, __shfl_xor_sync(mask, value, 8));
    if constexpr (kNumLanesPerGroup <= 16) value = op(value, __shfl_xor_sync(mask, value, 16));
  } else {
    if constexpr (kNumLanesPerGroup >= 32) value = op(value, __shfl_xor_sync(mask, value, 16));
    if constexpr (kNumLanesPerGroup >= 16) value = op(value, __shfl_xor_sync(mask, value, 8));
    if constexpr (kNumLanesPerGroup >= 8) value = op(value, __shfl_xor_sync(mask, value, 4));
    if constexpr (kNumLanesPerGroup >= 4) value = op(value, __shfl_xor_sync(mask, value, 2));
    if constexpr (kNumLanesPerGroup >= 2) value = op(value, __shfl_xor_sync(mask, value, 1));
  }
  return value;
}

// Convenience aliases

#endif

}  // namespace flashinfer::megamoe::math
