// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "../arch/barrier.cuh"
#include "stage_contracts.cuh"

namespace flashinfer::sparse_mla_sm120::pipeline {

template <int Even, int Odd, int ProducerThreads, int ConsumerThreads>
__device__ inline void StoreHandoff<Even, Odd, ProducerThreads, ConsumerThreads>::publish(
    int slot) {
  bar_arrive_alt<Even, Odd, PARTICIPANTS>(slot);
}
template <int Even, int Odd, int ProducerThreads, int ConsumerThreads>
__device__ inline void StoreHandoff<Even, Odd, ProducerThreads, ConsumerThreads>::wait(int slot) {
  bar_sync_alt<Even, Odd, PARTICIPANTS>(slot);
}

template <int Even, int Odd, int WriterThreads, int ReaderThreads>
__device__ inline void SlotRelease<Even, Odd, WriterThreads, ReaderThreads>::acquire(int slot) {
  bar_sync_alt<Even, Odd, PARTICIPANTS>(slot);
}
template <int Even, int Odd, int WriterThreads, int ReaderThreads>
__device__ inline void SlotRelease<Even, Odd, WriterThreads, ReaderThreads>::release(int slot) {
  bar_arrive_alt<Even, Odd, PARTICIPANTS>(slot);
}

template <int Arrivals>
__device__ inline void CountedHandoff<Arrivals>::init(uint64_t* barrier) {
  mbarrier_init(barrier, Arrivals);
}
template <int Arrivals>
template <int Slots>
__device__ inline void CountedHandoff<Arrivals>::init_slots(uint64_t* barrier) {
#pragma unroll
  for (int slot = 0; slot < Slots; ++slot) init(barrier + slot);
}
template <int Arrivals>
__device__ inline void CountedHandoff<Arrivals>::publish(uint64_t* barrier) {
  mbarrier_arrive(barrier);
}
template <int Arrivals>
__device__ inline void CountedHandoff<Arrivals>::wait(uint64_t* barrier, uint32_t phase) {
  mbarrier_wait_parity(barrier, phase);
}

__device__ inline void BulkReady::expect(uint64_t* barrier, uint32_t bytes) {
  mbarrier_arrive_expect_tx(barrier, bytes);
}

template <int Slots, int ReadyArrivals, int FreeArrivals>
template <typename Phase>
__device__ inline void AsyncRing<Slots, ReadyArrivals, FreeArrivals>::advance(int& slot,
                                                                              Phase& phase) {
  if (++slot == Slots) {
    slot = 0;
    phase ^= 1;
  }
}
template <int Slots, int ReadyArrivals, int FreeArrivals>
__device__ inline void AsyncRing<Slots, ReadyArrivals, FreeArrivals>::init(uint64_t* ready,
                                                                           uint64_t* free) {
#pragma unroll
  for (int slot = 0; slot < Slots; ++slot) {
    Ready::init(ready + slot);
    Free::init(free + slot);
  }
}

template <int Id, int Threads>
__device__ inline void RoleSync<Id, Threads>::wait() {
  bar_sync_t<Id, Threads>();
}

}  // namespace flashinfer::sparse_mla_sm120::pipeline
