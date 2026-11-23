/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include <stddef.h>
#include <stdint.h>

// One serialized call sequence per workspace and operator family. All eight
// ranks must execute the same sequence. Counters and inboxes are initialized
// together, once, outside capture; neither may be reset independently.
namespace flashinfer::glm5_tp8::attention_q1 {
constexpr int kWorld = 8;
constexpr int kGrid = 148;
struct Q1Args {
  uint32_t n_rows;
  int32_t rank;
  const uint8_t* context;  // immutable operator descriptor, prepared before capture
  uint32_t* sync_words;    // workspace-owned, at least 256 bytes
  uint8_t* peers[kWorld];  // bases of this operator's own record regions
  uint32_t* launch_ctr;    // workspace-owned u32[148], initially zero
  uint32_t* reset_token;   // workspace-owned u32, initially zero
};
static_assert(sizeof(Q1Args) == 104, "q1 launch parameter layout");
static_assert(offsetof(Q1Args, peers) == 24 && offsetof(Q1Args, launch_ctr) == 88,
              "q1 launch field offsets");
constexpr size_t kQ1Words = 0;
constexpr size_t kQ1Counters = 256;
constexpr size_t kQ1ResetToken = kQ1Counters + kGrid * sizeof(uint32_t);
constexpr size_t kQ1StateBytes = 1024;
static_assert(kQ1ResetToken + sizeof(uint32_t) <= kQ1StateBytes, "q1 state capacity");

}  // namespace flashinfer::glm5_tp8::attention_q1
