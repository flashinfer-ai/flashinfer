// SPDX - FileCopyrightText : 2023-2035 FlashInfer team.
// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache 2.0

#ifndef FLASHINFER_FRAG_LAYOUT_SWIZZLE_CUH_
#define FLASHINFER_FRAG_LAYOUT_SWIZZLE_CUH_

#include "gpu_iface/platform.hpp"

#include <cstdint>

__device__ __forceinline__ uint32_t frag_layout_swizzle_16b_to_8b(uint32_t x)
{
    uint32_t tmp = __shfl_xor_sync(0xffffffff, x, 0x1);
    x = __byte_perm(x, tmp, ((threadIdx.x & 0x1) == 0) ? 0x5410 : 0x3276);
    tmp = __shfl_xor_sync(0xffffffff, x, 0x2);
    x = __byte_perm(x, tmp, ((threadIdx.x & 0x2) == 0) ? 0x5410 : 0x3276);
    return x;
}

__device__ __forceinline__ uint32_t
frag_layout_swizzle_16b_to_8b_trans(uint32_t x)
{
    uint32_t tmp = __shfl_xor_sync(0xffffffff, x, 0x4);
    x = __byte_perm(x, tmp, ((threadIdx.x & 0x4) == 0) ? 0x6420 : 0x3175);
    tmp = __shfl_xor_sync(0xffffffff, x, 0x8);
    x = __byte_perm(x, tmp, ((threadIdx.x & 0x8) == 0) ? 0x5410 : 0x3276);
    tmp = __shfl_xor_sync(0xffffffff, x, 0x10);
    x = __byte_perm(x, tmp, ((threadIdx.x & 0x10) == 0) ? 0x5410 : 0x3276);
    return x;
}

#endif // FLASHINFER_FRAG_LAYOUT_SWIZZLE_CUH_
