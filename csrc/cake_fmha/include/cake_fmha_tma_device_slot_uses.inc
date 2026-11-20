/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Completion record for the TMA descriptor slot leases of one launch: marks
// the leased slots busy until the launch stream passes the recorded event.
// Shared verbatim by the adapters that include
// ``cake_fmha_tma_device_arena.inc``.
//
// This is a textual fragment, not a self-contained header: each adapter
// includes it from inside its anonymous namespace after its own
// ``TmaDeviceSlot`` definition, at the exact position where it used to define
// the function inline, so the fragment has no includes and opens no namespace
// of its own and the adapter's preprocessed translation unit is unchanged.
// Requires the arena fragment's types, <initializer_list>, <mutex> and
// ``tvm_ffi_utils.h`` in scope.

#pragma once

void RecordTmaDeviceSlotUses(std::initializer_list<TmaDeviceSlotLease> leases,
                             cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(TmaDeviceSlotMutex());
  for (const auto& lease : leases) {
    if (!lease.track_completion) continue;
    auto arena_it = TmaDeviceArenas().find(lease.context);
    TVM_FFI_ICHECK(arena_it != TmaDeviceArenas().end())
        << "Cake FMHA TMA descriptor arena disappeared before completion";
    auto& arena = arena_it->second;
    TVM_FFI_ICHECK_LT(lease.slot_index, arena.slots.size())
        << "Cake FMHA TMA descriptor lease index is out of range";
    auto& slot = arena.slots[lease.slot_index];
    if (slot.pinned) {
      slot.reserved = false;
      continue;
    }
    cudaError_t status = cudaEventRecord(slot.completion, stream);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "failed to record Cake FMHA TMA descriptor completion: "
        << cudaGetErrorString(status);
    slot.has_completion = true;
    slot.last_stream = stream;
    slot.reserved = false;
  }
}
