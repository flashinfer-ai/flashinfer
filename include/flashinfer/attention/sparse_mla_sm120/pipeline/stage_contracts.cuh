// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace flashinfer::sparse_mla_sm120::pipeline {

struct TwoSlotCursor {
  __host__ __device__ static constexpr int slot(int sequence) { return sequence & 1; }
  __host__ __device__ static constexpr int phase(int sequence) { return (sequence >> 1) & 1; }
  __host__ __device__ static constexpr bool reuses(int sequence) { return sequence >= 2; }
};

struct SelectiveSlotPhases {
  int phases[2] = {0, 0};
  __device__ int current(int slot) const { return phases[slot]; }
  __device__ void advance(int slot) { phases[slot] ^= 1; }
};

template <int Even, int Odd, int ProducerThreads, int ConsumerThreads>
struct StoreHandoff {
  static constexpr int PARTICIPANTS = ProducerThreads + ConsumerThreads;
  static_assert(Even != Odd && Even >= 0 && Odd < 16);
  static_assert(ProducerThreads % 32 == 0 && ConsumerThreads % 32 == 0);
  // Every producer arrives after its shared stores; every consumer calls wait.
  // wait is a collective rendezvous, not an independent ready-event query.
  __device__ static void publish(int slot);
  __device__ static void wait(int slot);
};

template <int Even, int Odd, int WriterThreads, int ReaderThreads>
struct SlotRelease {
  static constexpr int PARTICIPANTS = WriterThreads + ReaderThreads;
  static_assert(Even != Odd && Even >= 0 && Odd < 16);
  static_assert(WriterThreads % 32 == 0 && ReaderThreads % 32 == 0);
  // Each declared reader participates after its final read. The writer cannot
  // reuse the slot until every reader group has arrived.
  __device__ static void acquire(int slot);
  __device__ static void release(int slot);
};

// Arrivals counts publish calls, not necessarily threads. Callers own election,
// initialization visibility and phase; a free handoff follows the last reader.
template <int Arrivals>
struct CountedHandoff {
  __device__ static void init(uint64_t* barrier);
  template <int Slots>
  __device__ static void init_slots(uint64_t* barrier);
  __device__ static void publish(uint64_t* barrier);
  __device__ static void wait(uint64_t* barrier, uint32_t phase);
};

// One elected expect arrival adds transaction bytes; wait also needs completion
// of those bytes. This does not publish unrelated ordinary stores by itself.
struct BulkReady : CountedHandoff<1> {
  __device__ static void expect(uint64_t* barrier, uint32_t bytes);
};

// init sets counters only. The schedule makes initialization visible and primes
// free slots (or bypasses their first wait) before entering its own reuse cycle.
template <int Slots, int ReadyArrivals = 1, int FreeArrivals = 1>
struct AsyncRing {
  using Ready = CountedHandoff<ReadyArrivals>;
  using Free = CountedHandoff<FreeArrivals>;
  template <typename Phase>
  __device__ static void advance(int& slot, Phase& phase);
  __device__ static void init(uint64_t* ready, uint64_t* free);
};

template <int Id, int Threads>
struct RoleSync {
  __device__ static void wait();
};

}  // namespace flashinfer::sparse_mla_sm120::pipeline
