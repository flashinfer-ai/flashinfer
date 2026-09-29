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

#include "../arch/common.cuh"

// D2: Direct B operand production for FP8 XV MMA (m16n8k32).
//
// Produces QMMA.16832 B operand registers directly from kv_smem[entry][dim],
// bypassing the V transpose buffer entirely.
//
// B register layout: b0 byte j = V[entry_base + tid*4+j][dim + gid]   (K=0..15)
//                    b1 byte j = V[entry_base+16 + tid*4+j][dim + gid] (K=16..31)
//
// Method: 4× LDS.32 (read 4 entries' 4-byte dim chunks) + 3× PRMT (extract
// the target dim byte from each and pack). Total: 8 LDS.32 + 6 PRMT per (b0,b1).
//
// Bank conflict: stride/4 mod 32 = {528/4%32=4(DSV3_2), 464/4%32=20(DSV4)}.
// Within a quad, 4 threads read at 4× stride apart → banks separated by
// 4×(stride/4%32) mod 32 = {16(DSV3_2), 16(DSV4)}. Max 2-way conflict.

template <int KV_STRIDE>
__device__ __forceinline__ void d2_load_b_fp8(uint32_t& b0, uint32_t& b1,
                                              const uint8_t* __restrict__ kv_smem, int entry_base,
                                              int dim, int lane) {
  const int gid = lane >> 2;
  const int tid = lane & 3;
  const int d = dim + gid;
  const int d_base = d & ~3;
  const int d_sel = d & 3;
  const uint32_t sel = ((4 + d_sel) << 4) | d_sel;

  // b0: entries entry_base + tid*4 + {0,1,2,3}
  {
    uint32_t r0 = *reinterpret_cast<const uint32_t*>(
        kv_smem + (entry_base + tid * 4 + 0) * KV_STRIDE + d_base);
    uint32_t r1 = *reinterpret_cast<const uint32_t*>(
        kv_smem + (entry_base + tid * 4 + 1) * KV_STRIDE + d_base);
    uint32_t r2 = *reinterpret_cast<const uint32_t*>(
        kv_smem + (entry_base + tid * 4 + 2) * KV_STRIDE + d_base);
    uint32_t r3 = *reinterpret_cast<const uint32_t*>(
        kv_smem + (entry_base + tid * 4 + 3) * KV_STRIDE + d_base);
    uint32_t t01, t23;
    asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(t01) : "r"(r0), "r"(r1), "r"(sel));
    asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(t23) : "r"(r2), "r"(r3), "r"(sel));
    asm volatile("prmt.b32 %0, %1, %2, 0x5410;\n" : "=r"(b0) : "r"(t01), "r"(t23));
  }

  // b1: entries entry_base+16 + tid*4 + {0,1,2,3}
  {
    uint32_t r0 = *reinterpret_cast<const uint32_t*>(
        kv_smem + (entry_base + 16 + tid * 4 + 0) * KV_STRIDE + d_base);
    uint32_t r1 = *reinterpret_cast<const uint32_t*>(
        kv_smem + (entry_base + 16 + tid * 4 + 1) * KV_STRIDE + d_base);
    uint32_t r2 = *reinterpret_cast<const uint32_t*>(
        kv_smem + (entry_base + 16 + tid * 4 + 2) * KV_STRIDE + d_base);
    uint32_t r3 = *reinterpret_cast<const uint32_t*>(
        kv_smem + (entry_base + 16 + tid * 4 + 3) * KV_STRIDE + d_base);
    uint32_t t01, t23;
    asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(t01) : "r"(r0), "r"(r1), "r"(sel));
    asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(t23) : "r"(r2), "r"(r3), "r"(sel));
    asm volatile("prmt.b32 %0, %1, %2, 0x5410;\n" : "=r"(b1) : "r"(t01), "r"(t23));
  }
}
