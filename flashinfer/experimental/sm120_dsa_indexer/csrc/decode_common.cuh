/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Decode: what the logits scan and the selection share.
//
// A long row (more than kRowSelKeys keys) is selected by several CTAs from a
// histogram the scan builds. Its state (kLongWords words in long_ws) persists
// across calls, CUDA graph replays included, and is never reset by the host:
// the scan adds the row's histogram at the parity of the row's epoch, and the
// selection reads that parity, clears the other one for the row's next call
// and advances the epoch.
#pragma once

#include <cuda_fp16.h>

#include "sm120_dsa_indexer.cuh"

namespace flashinfer::sm120_dsa_indexer {

constexpr int kSelThreads = 1024;              // a selection CTA
constexpr int kRowSelKeys = kSelThreads * 16;  // the longest row one CTA selects (in registers)
constexpr int kSelList = 2048;                 // cut-bin keys kept for ranking
constexpr int kLongRounds = 4;                 // histogram rounds of a crowded cut bin
constexpr int kLongMaxCtas = 256;              // CTAs a long row may take

// A long row's state, in 32-bit words.
constexpr int kLongHist0 = 0;                        // [2 parities][4096] the scan's fp32 counts
constexpr int kLongHistR = 2 * 4096;                 // [kLongRounds - 1][4096] the rounds' counts
constexpr int kLongCtr = kLongHistR + 3 * 4096;      // counters (kCtr*)
constexpr int kLongList0 = kLongCtr + 32;            // [2 parities][keys, ids][kSelList]
constexpr int kLongTie = kLongList0 + 4 * kSelList;  // [kLongMaxCtas] the CTAs' tie counts
constexpr int kLongWords = kLongTie + kLongMaxCtas;
constexpr int kCtrCount = 0, kCtrGen = 1;  // the row's CTA barrier
constexpr int kCtrEpoch = 2;               // advanced once a call
constexpr int kCtrOut = 4, kCtrList = 6;   // [2 parities] output and list fill
constexpr int kCtrArrive = 8;              // the CTAs that read the epoch

// The histogram bin of a score: the top 12 bits of the order key of its fp16
// rounding (64 bins an octave, monotone in the score; -0.0 as +0.0).
__device__ __forceinline__ uint32_t rowsel_bin(float x) {
  const uint32_t h = static_cast<uint32_t>(__half_as_ushort(__float2half_rn(__fadd_rn(x, 0.0f))));
  return ((h & 0x8000u) ? (~h & 0xffffu) : (h | 0x8000u)) >> 4;
}

__device__ __forceinline__ void red_add_gpu(uint32_t* p, uint32_t v) {
  asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void red_add_v4_f32_gpu(float* p, float a, float b, float c, float d) {
  asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(p), "f"(a), "f"(b),
               "f"(c), "f"(d)
               : "memory");
}
__device__ __forceinline__ void red_release_add_gpu(int32_t* p, int v) {
  asm volatile("red.release.gpu.global.add.s32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ int atom_acq_rel_add_gpu(int32_t* p, int v) {
  int r;
  asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;"
               : "=r"(r)
               : "l"(p), "r"(v)
               : "memory");
  return r;
}
__device__ __forceinline__ uint32_t ld_acquire_gpu(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t ld_relaxed_gpu(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

}  // namespace flashinfer::sm120_dsa_indexer
