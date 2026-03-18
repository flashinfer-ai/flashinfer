/*
 * Copyright (c) 2025 by FlashInfer team.
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

// Shared definitions for the vertical and horizontal MTP kernels.

#pragma once

#include <cuda/barrier>

namespace flashinfer::mamba::mtp {

using barrier_t = cuda::barrier<cuda::thread_scope_block>;

enum class WarpRole { kCompute, kTMALoad, kEpilogue };

__device__ __forceinline__ WarpRole get_warp_role(int warp) {
  if (warp < 12) return WarpRole::kCompute;
  if (warp < 15) return WarpRole::kTMALoad;
  return WarpRole::kEpilogue;
}

// XOR-based bank-conflict-free swizzle for horizontal state traversal.
// Operates on flat byte addresses: XORs the bank index with the row (cycle) index.
// cycle_length = row stride in bytes, bank_size = sizeof(uint32_t).
template <int cycle_length, int bank_size>
__device__ __forceinline__ int xor_swizzle(int address) {
  int const cycle = address / cycle_length;
  int const delta = address % cycle_length;
  int const bank_idx = delta / bank_size;
  int const intra_bank = delta % bank_size;
  int const new_bank_idx = bank_idx ^ cycle;
  return cycle * cycle_length + new_bank_idx * bank_size + intra_bank;
}

}  // namespace flashinfer::mamba::mtp
