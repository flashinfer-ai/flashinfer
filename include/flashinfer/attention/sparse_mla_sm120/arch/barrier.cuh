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

// barrier.cta with immediate operands (17 ns vs 22 ns for register operands)
template <int ID, int CNT>
__device__ __forceinline__ void bar_arrive_t() {
  asm volatile("barrier.cta.arrive %0, %1;\n" ::"n"(ID), "n"(CNT) : "memory");
}

template <int ID, int CNT>
__device__ __forceinline__ void bar_sync_t() {
  asm volatile("barrier.cta.sync %0, %1;\n" ::"n"(ID), "n"(CNT) : "memory");
}

// mbarrier (SM90+) for async copy tracking
__device__ __forceinline__ void mbarrier_init(uint64_t* mbar, uint32_t count) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(addr), "r"(count));
}

__device__ __forceinline__ void mbarrier_inval(uint64_t* mbar) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" ::"r"(addr));
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t* mbar) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile(
      "{\n .reg .b64 state;\n"
      " mbarrier.arrive.shared::cta.b64 state, [%0];\n"
      "}\n" ::"r"(addr));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* mbar, uint32_t tx_bytes) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile(
      "{\n .reg .b64 state;\n"
      " mbarrier.arrive.expect_tx.shared::cta.b64 state, [%0], %1;\n"
      "}\n" ::"r"(addr),
      "r"(tx_bytes));
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* mbar, uint32_t phase) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  uint32_t done = 0;
  while (!done) {
    asm volatile(
        "{\n .reg .pred p;\n"
        " mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        " selp.u32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(done)
        : "r"(addr), "r"(phase));
  }
}
