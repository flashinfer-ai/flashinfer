// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "common.cuh"

__device__ __forceinline__ void stmatrix_x4_trans_b8(void* smem_ptr, uint32_t r0, uint32_t r1,
                                                     uint32_t r2, uint32_t r3) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0], {%1, %2, %3, %4};\n"
               :
               : "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3)
               : "memory");
}

template <int HEADS_PER_WARP, int N_CAND>
struct StMatrixTransB8Tile {
  static constexpr int BYTES = HEADS_PER_WARP * N_CAND;
  static_assert(HEADS_PER_WARP == 8, "load_b's 16 * gid + 4 * tid addressing assumes 8 heads");
  static_assert(BYTES % 128 == 0, "load_b steps 128B rows; BYTES must be a whole number of them");
  __device__ static __forceinline__ void store(uint8_t* dst, const uint32_t* regs, int lane) {
    stmatrix_x4_trans_b8(dst + 16 * lane, regs[0], regs[1], regs[2], regs[3]);
  }
  __device__ static __forceinline__ void load_b(uint32_t* b, const uint8_t* src, int lane) {
    const int gid = lane >> 2, tid = lane & 3;
#pragma unroll
    for (int i = 0; i < BYTES / 128; i++)
      b[i] = *reinterpret_cast<const uint32_t*>(src + 128 * i + 16 * gid + 4 * tid);
  }
};
