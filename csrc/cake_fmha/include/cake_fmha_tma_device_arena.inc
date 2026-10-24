/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Completion-tracked device arena for the TMA descriptors (``CUtensorMap``)
// of the Cake FMHA launch adapters that stage their descriptors in device
// memory: the slot state, the per-context arena, the lease handed to a
// launch, the readiness query, chunk growth and the process-wide mutex and
// arena map.  Shared verbatim by the context_bf16 / context_fp8 /
// context_hd256 / decode_native_bf16 / decode_native_fp16_hd512 /
// decode_native_fp16_nhd / decode_quant_bf16q / decode_quant_fp8 adapters.
//
// This is a textual fragment, not a self-contained header: each adapter
// includes it from inside its anonymous namespace
// (``namespace flashinfer { namespace cake_fmha { namespace {``) at the exact
// position where it used to define these types and functions inline, so the
// fragment has no includes and opens no namespace of its own and the
// adapter's preprocessed translation unit is unchanged.  Requires <cuda.h>,
// <cuda_runtime.h>, <mutex>, <string>, <unordered_map>, <vector> and
// ``tvm_ffi_utils.h`` in scope.  The adapter defines the slot lookup
// (``TmaDeviceSlot``) itself and includes
// ``cake_fmha_tma_device_slot_uses.inc`` for the completion record.

#pragma once

struct TmaDeviceSlotState {
  CUdeviceptr pointer = 0;
  cudaEvent_t completion = nullptr;
  cudaStream_t last_stream = nullptr;
  std::string key;
  bool has_completion = false;
  bool reserved = false;
  bool pinned = false;
};

struct TmaDeviceArena {
  static constexpr size_t kSlotsPerChunk = 256;
  static constexpr size_t kMaxReusableSlots = 4096;
  static constexpr size_t kMaxPinnedSlots = 4096;
  std::vector<CUdeviceptr> chunks;
  std::vector<cudaEvent_t> events;
  std::vector<TmaDeviceSlotState> slots;
  std::unordered_map<std::string, size_t> pinned_slots;
  unsigned long long context_id = 0;
  size_t reusable_slots = 0;
  size_t pinned_count = 0;
  size_t cursor = 0;
};

struct TmaDeviceSlotLease {
  void* pointer;
  CUcontext context;
  size_t slot_index;
  bool track_completion;
};

bool TmaDeviceSlotReady(const TmaDeviceSlotState& slot) {
  if (!slot.has_completion) return true;
  cudaError_t status = cudaEventQuery(slot.completion);
  if (status == cudaSuccess) return true;
  TVM_FFI_ICHECK_EQ(status, cudaErrorNotReady)
      << "failed to query Cake FMHA TMA descriptor completion: "
      << cudaGetErrorString(status);
  return false;
}

void AddTmaDeviceSlotChunk(TmaDeviceArena& arena) {
  size_t count = std::min(TmaDeviceArena::kSlotsPerChunk,
                          TmaDeviceArena::kMaxReusableSlots - arena.reusable_slots);
  TVM_FFI_ICHECK_GT(count, 0);
  CUdeviceptr chunk = 0;
  CUresult result = cuMemAlloc(&chunk, count * sizeof(CUtensorMap));
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS)
      << "failed to allocate Cake FMHA TMA descriptor chunk";
  arena.chunks.push_back(chunk);
  for (size_t index = 0; index < count; ++index) {
    cudaEvent_t completion = nullptr;
    cudaError_t status = cudaEventCreateWithFlags(&completion, cudaEventDisableTiming);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "failed to create Cake FMHA TMA descriptor completion event: "
        << cudaGetErrorString(status);
    arena.events.push_back(completion);
    arena.slots.push_back(
        {chunk + index * sizeof(CUtensorMap), completion, nullptr, "", false, false, false});
  }
  arena.reusable_slots += count;
}

std::mutex& TmaDeviceSlotMutex() {
  static auto* mutex = new std::mutex();
  return *mutex;
}

std::unordered_map<CUcontext, TmaDeviceArena>& TmaDeviceArenas() {
  static auto* arenas = new std::unordered_map<CUcontext, TmaDeviceArena>();
  return *arenas;
}
