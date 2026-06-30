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
#include "../arch/ldmatrix_sm120.cuh"
#include "../arch/mma_sm120.cuh"
#include "../model/kv_cache_traits.cuh"

// Q rope: preload to registers and compute QK rope via BF16 MMA.
//
// Q rope is [HPB × D_ROPE] BF16 in smem (from quantize_q_to_smem).
// Preloaded to registers once before the main loop (survives across all tiles).
//
// KV rope B operands are prefetched from global into registers BEFORE QK nope
// MMA, so the ~300 cycle load latency overlaps with nope MMA compute.

struct QRopeRegs {
  uint32_t a[N_ROPE_CHUNKS][4];
};

struct KVRopePrefetch {
  uint32_t b[N_ROPE_CHUNKS][2];
};

__device__ __forceinline__ QRopeRegs preload_q_rope_regs(const bf16* q_rope_smem, int lane) {
  QRopeRegs regs;
#pragma unroll
  for (int ks = 0; ks < N_ROPE_CHUNKS; ks++)
    ldmatrix_load_A_bf16(regs.a[ks][0], regs.a[ks][1], regs.a[ks][2], regs.a[ks][3],
                         q_rope_smem + ks * 16, D_ROPE, lane);
  return regs;
}

__device__ __forceinline__ KVRopePrefetch prefetch_kv_rope(const bf16* kv_rope_ptr, int lane) {
  const int tid = lane & 3;
  KVRopePrefetch pf;
#pragma unroll
  for (int ks = 0; ks < N_ROPE_CHUNKS; ks++) {
    int ko = ks * 16;
    pf.b[ks][0] = *reinterpret_cast<const uint32_t*>(kv_rope_ptr + ko + tid * 2);
    pf.b[ks][1] = *reinterpret_cast<const uint32_t*>(kv_rope_ptr + ko + 8 + tid * 2);
  }
  return pf;
}

__device__ __forceinline__ void compute_qk_rope(float qk[4], const QRopeRegs& qr,
                                                const KVRopePrefetch& pf) {
  float ra[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int ks = 0; ks < N_ROPE_CHUNKS; ks++) {
    MmaBf16Result r = mma_bf16_m16n8k16(qr.a[ks][0], qr.a[ks][1], qr.a[ks][2], qr.a[ks][3],
                                        pf.b[ks][0], pf.b[ks][1], ra[0], ra[1], ra[2], ra[3]);
    ra[0] = r.d0;
    ra[1] = r.d1;
    ra[2] = r.d2;
    ra[3] = r.d3;
  }
  qk[0] += ra[0];
  qk[1] += ra[1];
  qk[2] += ra[2];
  qk[3] += ra[3];
}
