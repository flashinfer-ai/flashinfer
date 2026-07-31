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

// FP8 A operand [16×32]: map 32 threads to 4 sub-matrices
__device__ __forceinline__ void ldmatrix_load_A_fp8(uint32_t& a0, uint32_t& a1, uint32_t& a2,
                                                    uint32_t& a3, const uint8_t* smem_base,
                                                    int stride, int lane) {
  int row = (lane & 7) + ((lane >> 3) & 1) * 8;
  int col = (lane >> 4) * 16;
  ldmatrix_x4(a0, a1, a2, a3, smem_base + row * stride + col);
}

__device__ __forceinline__ int wfp8_row_xor(int row) { return row ^ (row >> 3); }

template <bool ROW_XOR>
__device__ __forceinline__ void ldmatrix_load_A_fp8_layout(uint32_t& a0, uint32_t& a1, uint32_t& a2,
                                                           uint32_t& a3, const uint8_t* smem_base,
                                                           int stride, int lane) {
  int row = (lane & 7) + ((lane >> 3) & 1) * 8;
  if constexpr (ROW_XOR) row = wfp8_row_xor(row);
  int col = (lane >> 4) * 16;
  ldmatrix_x4(a0, a1, a2, a3, smem_base + row * stride + col);
}

// FP8 B operand [8×32]
__device__ __forceinline__ void ldmatrix_load_B_fp8(uint32_t& b0, uint32_t& b1,
                                                    const uint8_t* smem_base, int stride,
                                                    int lane) {
  int row = lane & 7;
  int col = ((lane >> 3) & 1) * 16;
  ldmatrix_x2(b0, b1, smem_base + row * stride + col);
}

// BF16 A operand [16×16]
__device__ __forceinline__ void ldmatrix_load_A_bf16(uint32_t& a0, uint32_t& a1, uint32_t& a2,
                                                     uint32_t& a3, const bf16* smem_base,
                                                     int stride_elems, int lane) {
  int row = (lane & 7) + ((lane >> 3) & 1) * 8;
  int col = (lane >> 4) * 8;
  ldmatrix_x4(a0, a1, a2, a3, smem_base + row * stride_elems + col);
}
