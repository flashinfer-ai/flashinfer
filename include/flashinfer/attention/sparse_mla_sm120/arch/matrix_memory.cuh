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

// ldmatrix: load matrices from shared memory for MMA operands.
//
// FP8 m16n8k32 treats 2 FP8 bytes as 1 b16 element:
//   A (16×32 FP8) = 4 × (8×8 b16) → ldmatrix.x4
//   B (8×32 FP8)  = 2 × (8×8 b16) → ldmatrix.x2
//
// BF16 m16n8k16:
//   A (16×16 BF16) = 4 × (8×8 b16) → ldmatrix.x4
//   B (8×16 BF16)  = 2 × (8×8 b16) → ldmatrix.x2
//   B transposed   = 2 × (8×8 b16) → ldmatrix.x2.trans

__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                                            const void* smem_ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t& r0, uint32_t& r1, const void* smem_ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(r0), "=r"(r1)
               : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                                  uint32_t& r3, const void* smem_ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

// Packed 16-row x 32-byte A tile: E4M3 bytes or E2M1 nibble pairs, unchanged.
__device__ __forceinline__ void ldmatrix_load_a_packed_16x32_bytes(uint32_t& a0, uint32_t& a1,
                                                                   uint32_t& a2, uint32_t& a3,
                                                                   const uint8_t* smem_base,
                                                                   int stride_bytes, int lane) {
  int row = (lane & 7) + ((lane >> 3) & 1) * 8;
  int col = (lane >> 4) * 16;
  ldmatrix_x4(a0, a1, a2, a3, smem_base + row * stride_bytes + col);
}

__device__ __forceinline__ int wfp8_row_xor(int row) { return row ^ (row >> 3); }

template <bool ROW_XOR>
__device__ __forceinline__ void ldmatrix_load_a_packed_16x32_bytes_layout(
    uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, const uint8_t* smem_base,
    int stride_bytes, int lane) {
  int row = (lane & 7) + ((lane >> 3) & 1) * 8;
  if constexpr (ROW_XOR) row = wfp8_row_xor(row);
  int col = (lane >> 4) * 16;
  ldmatrix_x4(a0, a1, a2, a3, smem_base + row * stride_bytes + col);
}

// Packed 8-row x 32-byte B tile, without numerical conversion.
__device__ __forceinline__ void ldmatrix_load_b_packed_8x32_bytes(uint32_t& b0, uint32_t& b1,
                                                                  const uint8_t* smem_base,
                                                                  int stride_bytes, int lane) {
  int row = lane & 7;
  int col = ((lane >> 3) & 1) * 16;
  ldmatrix_x2(b0, b1, smem_base + row * stride_bytes + col);
}

// BF16 A operand [16×16]
__device__ __forceinline__ void ldmatrix_load_A_bf16(uint32_t& a0, uint32_t& a1, uint32_t& a2,
                                                     uint32_t& a3, const bf16* smem_base,
                                                     int stride_elems, int lane) {
  int row = (lane & 7) + ((lane >> 3) & 1) * 8;
  int col = (lane >> 4) * 8;
  ldmatrix_x4(a0, a1, a2, a3, smem_base + row * stride_elems + col);
}

// FP8 [32×16] → the [16×32] A operand: lane l addresses row l, 16B aligned
__device__ __forceinline__ void ldmatrix_x2_trans_b8(uint32_t& d0, uint32_t& d1, uint32_t& d2,
                                                     uint32_t& d3, const void* smem_ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
               : "r"(addr));
}

// FP8 A operand [16×32] transposed, straight out of a [candidate, dim] tile
template <int KV_STRIDE_BYTES>
__device__ __forceinline__ void ldmatrix_load_a_packed_16x32_bytes_trans(uint32_t& a0, uint32_t& a1,
                                                                         uint32_t& a2, uint32_t& a3,
                                                                         const uint8_t* smem_base,
                                                                         int k_start, int dim,
                                                                         int lane) {
  ldmatrix_x2_trans_b8(a0, a1, a2, a3,
                       smem_base + (size_t)(k_start + lane) * KV_STRIDE_BYTES + dim);
}

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
