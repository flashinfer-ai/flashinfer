/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "common.h"

namespace flashinfer::glm5_tp8 {

inline constexpr size_t kMoeQ4StateBytes = 0x500;
inline constexpr size_t kMoeQ4CounterOffset = 0x200;
inline constexpr size_t kMoeQ1StateBytes = 1024;
inline constexpr size_t kMoeQ1CounterOffset = 256;
inline constexpr size_t kMoeQ1ContextBytes = 640;
// State and record regions are initialized together, once: q4 counters start at
// one; q1 counters and all synchronization words start at zero. Calls advance
// counters on the GPU. Never clear either state or records independently.

// All addresses are owned by the caller. Peer pointers name this operator's
// record regions directly, not a larger runtime-specific inbox allocation.
struct MoeArguments {
  int32_t rank;
  int32_t q_len;
  uint8_t* peers[8];
  uint8_t* state;
  float* shared_down_scratch;
  const uint16_t* x;
  const float* gamma;
  const uint16_t* router;
  const uint8_t* gate_up;
  const float* bias;
  const uint8_t* down;
  const float* down_scale;
  float* scores;
  uint16_t* mid;
  uint16_t* x_norm;
  uint16_t* out;
  float* probabilities;
  int32_t* indices;
  const uint64_t* context;  // private q1 descriptor, prepared outside capture
};

// Initialization forces module loading and opts into shared memory. It neither
// allocates nor resets state, and runs before the first exchange/capture.
KernelResources initialize_moe(int q_len, int device);
cudaError_t launch_moe(const MoeArguments& args, cudaStream_t stream, bool enable_pdl,
                       bool cooperative);

}  // namespace flashinfer::glm5_tp8
