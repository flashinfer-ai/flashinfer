/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cstdint>

// Bounded wait for the PCIe IPC peer-polling loops (#9108).
//
// A peer that dies mid-collective leaves the survivors spinning on its flag
// with nothing to break the loop: the polls are unbounded by design and the
// kernels are routinely captured in CUDA graphs, where no host-side watchdog
// can reach them. Compiling with FLASHINFER_PCIE_IPC_SPIN_TIMEOUT_NS > 0
// bounds every peer wait: once a poll passes the deadline, one report naming
// the kernel (and the peer, when the site knows it) is printed and the thread
// traps, turning the silent hang into a diagnosable CUDA error.
//
// The deadline is a compile-time constant on purpose. It reaches every
// kernel -- including the ones the tuner launches -- without touching any
// kernel signature, launch parameter, or workspace layout, and with the knob
// unset the guard compiles to nothing at all. The JIT layer derives it from
// FLASHINFER_PCIE_IPC_TIMEOUT_S and keys the build cache on the value.

#ifndef FLASHINFER_PCIE_IPC_SPIN_TIMEOUT_NS
#define FLASHINFER_PCIE_IPC_SPIN_TIMEOUT_NS 0ULL
#endif

namespace flashinfer::comm::pcie_ipc {

#if FLASHINFER_PCIE_IPC_SPIN_TIMEOUT_NS > 0ULL

namespace {
// One report per translation unit: the first thread to cross the deadline
// prints, everyone traps. Internal linkage keeps the symbol per-TU even when
// this header is included from several TUs of the same module.
__device__ unsigned int spin_timeout_reported = 0u;
__device__ unsigned int spin_timeout_report_done = 0u;
}  // namespace

// Bounded spin for one peer-flag poll site. A poll is a few hundred
// nanoseconds; the counter is register-allocated and the %globaltimer read
// happens once every 4096 polls, so the added latency sits far below the
// noise floor of the PCIe round trip the poll itself waits for. The first
// tick records the baseline, so the deadline measures time actually spent
// spinning, not time since launch.
struct SpinGuard {
  __device__ __forceinline__ explicit SpinGuard(const char* what) : what_(what) {}

  __device__ __forceinline__ void tick(int peer = -1) {
    if ((++n_ & 0xFFFu) != 0u) return;
    unsigned long long now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    if (t0_ == 0ULL) {
      t0_ = now;
      return;
    }
    if (now - t0_ > FLASHINFER_PCIE_IPC_SPIN_TIMEOUT_NS) {
      // The winner prints, then publishes the report; the losers wait for the
      // report to land before trapping, so a racing __trap() cannot eat it.
      if (atomicExch(&spin_timeout_reported, 1u) == 0u) {
        printf("%s: no progress after %llu ms waiting for peer %d's flag; trapping\n", what_,
               (now - t0_) / 1000000ULL, peer);
        __threadfence();
        atomicExch(&spin_timeout_report_done, 1u);
      } else {
        // Bounded by a generous window: the winner is between printf and the
        // flag store, normally microseconds. Plain loads only - no nanosleep,
        // so the guard compiles for every architecture the kernels build for.
        unsigned long long report_deadline = now + 1000000000ULL;
        while (atomicAdd(&spin_timeout_report_done, 0u) == 0u) {
          unsigned long long t;
          asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
          if (t > report_deadline) break;
        }
      }
      __trap();
    }
  }

 private:
  const char* what_;
  unsigned int n_ = 0u;
  unsigned long long t0_ = 0ULL;
};

#else

// Unbounded by default: today's behavior, compiled away entirely.
struct SpinGuard {
  __device__ __forceinline__ explicit SpinGuard(const char*) {}
  __device__ __forceinline__ void tick(int = -1) {}
};

#endif  // FLASHINFER_PCIE_IPC_SPIN_TIMEOUT_NS > 0ULL

}  // namespace flashinfer::comm::pcie_ipc
