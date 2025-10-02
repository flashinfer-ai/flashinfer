// SPDX - FileCopyrightText : 2023-2035 FlashInfer team.
// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache 2.0

#ifndef FLASHINFER_FRAG_LAYOUT_SWIZZLE_CUH_
#define FLASHINFER_FRAG_LAYOUT_SWIZZLE_CUH_

#include <cstdint>

#include "gpu_iface/platform.hpp"

// Define platform-specific full mask for warp/wavefront operations
#if defined(PLATFORM_CUDA_DEVICE)
constexpr uint32_t WARP_FULL_MASK = 0xffffffff;  // 32-bit mask for CUDA
#elif defined(PLATFORM_HIP_DEVICE)
constexpr uint64_t WARP_FULL_MASK = 0xffffffffffffffffULL;  // 64-bit mask for HIP
#endif

__device__ __forceinline__ uint32_t frag_layout_swizzle_16b_to_8b(uint32_t x) {
  uint32_t tmp = __shfl_xor_sync(WARP_FULL_MASK, x, 0x1);
  x = __byte_perm(x, tmp, ((threadIdx.x & 0x1) == 0) ? 0x5410 : 0x3276);
  tmp = __shfl_xor_sync(WARP_FULL_MASK, x, 0x2);
  x = __byte_perm(x, tmp, ((threadIdx.x & 0x2) == 0) ? 0x5410 : 0x3276);
  return x;
}

__device__ __forceinline__ uint32_t frag_layout_swizzle_16b_to_8b_trans(uint32_t x) {
  uint32_t tmp = __shfl_xor_sync(WARP_FULL_MASK, x, 0x4);
  x = __byte_perm(x, tmp, ((threadIdx.x & 0x4) == 0) ? 0x6420 : 0x3175);
  tmp = __shfl_xor_sync(WARP_FULL_MASK, x, 0x8);
  x = __byte_perm(x, tmp, ((threadIdx.x & 0x8) == 0) ? 0x5410 : 0x3276);
  tmp = __shfl_xor_sync(WARP_FULL_MASK, x, 0x10);
  x = __byte_perm(x, tmp, ((threadIdx.x & 0x10) == 0) ? 0x5410 : 0x3276);
  return x;
}

#endif  // FLASHINFER_FRAG_LAYOUT_SWIZZLE_CUH_
