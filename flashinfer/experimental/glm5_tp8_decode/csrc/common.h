/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include <cstdint>

namespace flashinfer::glm5_tp8 {
struct KernelResources {
  int grid;
  int block;
  int dynamic_shared_bytes;
  int registers;
  int static_shared_bytes;
  int local_bytes;
  int blocks_per_sm;
};
}  // namespace flashinfer::glm5_tp8
