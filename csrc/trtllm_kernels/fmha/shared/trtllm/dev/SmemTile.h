/***************************************************************************************************
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <cstdint>

#include "cuda_runtime_api.h"

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
inline __device__ uint64_t createSmemDesc(T* smemPtr, T* smemPtrNextBuffer, uint32_t lo,
                                          uint32_t hi) {
  uint32_t maskFull = 0x3ffffu;
  uint32_t maskNoLsb = 0x3fff0u;
  maskFull = 0x7ffffu;
  maskNoLsb = 0x7fff0u;
  uint32_t smemAddr = (static_cast<uint32_t>(__cvta_generic_to_shared(smemPtr)) & maskFull) >> 4;
  uint32_t smemNextBufferAddr =
      (static_cast<uint32_t>(__cvta_generic_to_shared(smemPtrNextBuffer)) & maskNoLsb) << (16 - 4);

  // Pack the values into an uint64_t.
  SmemDesc tmp;
  tmp.u32[0] = smemAddr | smemNextBufferAddr;
  tmp.u32[1] = hi;

  // Return the uint64_t.
  return tmp.u64;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline __device__ uint64_t createSmemDesc(T* smemPtr, int32_t nextBufferOffsetInBytes, uint32_t lo,
                                          uint32_t hi) {
  // Get the pointer to the next buffer
  T* smemPtrNextBuffer =
      reinterpret_cast<T*>(reinterpret_cast<char*>(smemPtr) + nextBufferOffsetInBytes);
  // Get the descriptor
  return createSmemDesc(smemPtr, smemPtrNextBuffer, lo, hi);
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

}  // namespace dev
}  // namespace trtllm
