/*
# SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
# ==============================================================================
*/
#pragma once

#include "cuda_runtime_api.h"
#include <cstdint>

namespace batchedGemm {

namespace trtllm {
namespace dev {

////////////////////////////////////////////////////////////////////////////////////////////////////
// *************************************************************************************************
// Helper function to manipulate the Smem descriptors for MMAs.
// *************************************************************************************************
////////////////////////////////////////////////////////////////////////////////////////////////////

union SmemDesc {
  uint64_t u64;
  uint32_t u32[2];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline __device__ uint64_t createSmemDesc(T* smemPtr, uint32_t lo, uint32_t hi) {
  // Convert the SMEM address to uint32_t.
  uint32_t mask = 0x3ffffu;
  uint32_t smemAddr = (static_cast<uint32_t>(__cvta_generic_to_shared(smemPtr)) & mask) >> 4;
  // Force the compiler to go down the URF path.
  // In some rare cases, the compiler does not think smemAddr is uniform, and generates lots of
  // conversion between URF and RF.
  smemAddr = __shfl_sync(0xffffffff, smemAddr, 0);
  // Pack the values into an uint64_t.
  SmemDesc tmp;
  tmp.u32[0] = smemAddr | lo;
  tmp.u32[1] = hi;

  // Return the uint64_t.
  return tmp.u64;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline __device__ uint64_t
createSmemDesc(T* smemPtr, int32_t nextBufferOffsetInBytes, uint32_t lo, uint32_t hi) {
  uint32_t maskFull = 0x3ffffu;
  uint32_t maskNoLsb = 0x3fff0u;
  uint32_t smemAddr = (static_cast<uint32_t>(__cvta_generic_to_shared(smemPtr)) & maskFull) >> 4;
  T* smemNextBufferPtr =
    reinterpret_cast<T*>(reinterpret_cast<char*>(smemPtr) + nextBufferOffsetInBytes);
  uint32_t smemNextBufferAddr =
    (static_cast<uint32_t>(__cvta_generic_to_shared(smemNextBufferPtr)) & maskNoLsb) << (16 - 4);

  // Pack the values into an uint64_t.
  SmemDesc tmp;
  tmp.u32[0] = smemAddr | smemNextBufferAddr;
  tmp.u32[1] = hi;

  // Return the uint64_t.
  return tmp.u64;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void decrSmemAddr(uint64_t& smemDesc, uint32_t offset) {
  SmemDesc tmp;
  tmp.u64 = smemDesc;
  tmp.u32[0] -= offset;
  smemDesc = tmp.u64;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void incrSmemAddr(uint64_t& smemDesc, uint32_t offset) {
  SmemDesc tmp;
  tmp.u64 = smemDesc;
  tmp.u32[0] += offset;
  smemDesc = tmp.u64;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void decrSmemNextBufferAddr(uint64_t& smemDesc, uint32_t offset) {
  SmemDesc tmp;
  tmp.u64 = smemDesc;
  tmp.u32[0] -= (offset << 16);
  smemDesc = tmp.u64;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void incrSmemNextBufferAddr(uint64_t& smemDesc, uint32_t offset) {
  SmemDesc tmp;
  tmp.u64 = smemDesc;
  tmp.u32[0] += (offset << 16);
  smemDesc = tmp.u64;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace dev
} // namespace trtllm

} // namespace batchedGemm
