/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 ******************************************************************************/
// Pipeline abstractions for fused attention kernel — Step 2 (BSA-aligned)
//
// Pipeline overview (BSA naming):
//   PipelineKV       — K/V multi-stage buffer (Load <-> MMA), TMA-based
//   PipelineSPO      — S/P/O TMEM coordination (MMA -> Softmax+Correction)
//   PipelineOAcc     — Final O accumulator ready (MMA -> Correction)
//   PipelineSmStats  — Softmax stats (Softmax -> Correction), PipelineAsync
//   PipelineOEpi     — sO SMEM staging (Correction -> Epilogue TMA store)
//   PipelineCLC      — Cluster Launch Control (CLC) scheduler pipeline
#pragma once

#ifndef CUTLASS_ARCH_CLC_ENABLED
#define CUTLASS_ARCH_CLC_ENABLED
#endif

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm100_pipeline.hpp"
#include "cutlass/pipeline/sm90_pipeline.hpp"

namespace flash {

static constexpr uint32_t kMBarTicks = 1;

// ============================================================================
// Barrier helpers
// ============================================================================

__device__ __forceinline__ void mbarrier_arrive(cute::uint64_t& bar) {
  uint32_t addr = cute::cast_smem_ptr_to_uint(&bar);
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" : : "r"(addr) : "memory");
}

// mbarrier arrive with .release for SMEM store visibility (pairs with .acquire on wait).
// Single-CTA kernel: .cta scope suffices (was .cluster → MEMBAR.ALL.GPU, now cheaper).
__device__ __forceinline__ void mbarrier_arrive_release(cute::uint64_t& bar) {
  uint32_t addr = cute::cast_smem_ptr_to_uint(&bar);
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n" : : "r"(addr) : "memory");
}

__device__ __forceinline__ void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\n");
}

// wait_barrier: drop-in replacement for cute::wait_barrier with ticks hint.
// The 3-operand form passes a nanosecond hint to suppress YIELD insertion by ptxas.
__device__ __forceinline__ void wait_barrier(cute::uint64_t& bar, int phase) {
  uint32_t addr = cute::cast_smem_ptr_to_uint(&bar);
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "WAIT_BAR_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n"
      "@P1 bra.uni DONE_BAR_%=;\n"
      "bra.uni WAIT_BAR_%=;\n"
      "DONE_BAR_%=:\n"
      "}\n"
      :
      : "r"(addr), "r"(phase), "r"(kMBarTicks)
      : "memory");
}

// SM100 wait_barrier with .acquire semantics (the cute::wait_barrier from SM90
// uses mbarrier.try_wait.parity WITHOUT .acquire, which doesn't guarantee
// SMEM store visibility on SM100).
__device__ __forceinline__ void wait_barrier_acquire(cute::uint64_t& bar, int phase) {
  uint32_t addr = cute::cast_smem_ptr_to_uint(&bar);
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "WAIT_ACQ_%=:\n"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n"
      "@P1 bra.uni DONE_ACQ_%=;\n"
      "bra.uni WAIT_ACQ_%=;\n"
      "DONE_ACQ_%=:\n"
      "}\n"
      :
      : "r"(addr), "r"(phase), "r"(kMBarTicks)
      : "memory");
}

// Address-based overloads (for pre-computed UR-promoted addresses).
__device__ __forceinline__ void mbarrier_arrive_addr(uint32_t addr) {
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" : : "r"(addr) : "memory");
}

__device__ __forceinline__ void wait_barrier_addr(uint32_t addr, int phase) {
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "WAIT_BAR_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n"
      "@P1 bra.uni DONE_BAR_%=;\n"
      "bra.uni WAIT_BAR_%=;\n"
      "DONE_BAR_%=:\n"
      "}\n"
      :
      : "r"(addr), "r"(phase), "r"(kMBarTicks)
      : "memory");
}

__device__ __forceinline__ void wait_barrier_acquire_addr(uint32_t addr, int phase) {
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "WAIT_ACQ_%=:\n"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n"
      "@P1 bra.uni DONE_ACQ_%=;\n"
      "bra.uni WAIT_ACQ_%=;\n"
      "DONE_ACQ_%=:\n"
      "}\n"
      :
      : "r"(addr), "r"(phase), "r"(kMBarTicks)
      : "memory");
}

// Broadcast a barrier SMEM address via __shfl_sync for UR promotion.
__device__ __forceinline__ uint32_t barrier_smem_addr(cute::uint64_t& bar) {
  return __shfl_sync(0xFFFFFFFF, cute::cast_smem_ptr_to_uint(&bar), 0);
}

// ============================================================================
// PipelineState: auto-tracks stage index and barrier phase
// ============================================================================

template <int Stages_>
struct PipelineState {
  int index_ = 0;
  uint32_t phase_ = 0;
  uint32_t count_ = 0;

  CUTLASS_DEVICE int index() const { return index_; }
  CUTLASS_DEVICE uint32_t phase() const { return phase_; }
  CUTLASS_DEVICE uint32_t count() const { return count_; }

  CUTLASS_DEVICE PipelineState& operator++() {
    if (index_ == Stages_ - 1) {
      index_ = 0;
      phase_ ^= 1;
    } else {
      ++index_;
    }
    ++count_;
    return *this;
  }
};

// ============================================================================
// PipelineKV: K/V multi-stage buffer (TMA-based, 3-stage)
//
//   Producer: Load warp (1 thread) — TMA loads K/V into alternating slots
//   Consumer: MMA warp (1 thread)  — reads K/V for QK/PV MMA
// ============================================================================

// PipelineKV: use CUTLASS PipelineTmaUmmaAsync directly (example 77 pattern).
// 1CTA cluster, 1x1x1 atom shape.
using PipelineKV = cutlass::PipelineTmaUmmaAsync<
    /*Stages=*/3,
    /*ClusterShape=*/cute::Shape<cute::_1, cute::_1, cute::_1>,
    /*AtomThrShape_MNK=*/cute::Shape<cute::_1, cute::_1, cute::_1>>;
using PipelineKVState = cutlass::PipelineState<3>;

// ============================================================================
// PipelineSPO: S/P/O TMEM coordination (indexed 2-stage)
// BSA: pipeline_s_p_o (PipelineUmmaAsync)
//
//   Producer: MMA warp (1 warp) — writes S via QK, signals full
//   Consumer: Softmax+Correction (256 threads) — read S, write P, rescale O, signal empty
//
//   full[s]:  MMA -> Softmax (S ready), arrive_count = 1 (UMMA arrive)
//   empty[s]: Softmax+Correction -> MMA (P ready + O free), arrive_count = 256
// ============================================================================

struct PipelineSPO {
  static constexpr int kStages = 2;
  static constexpr int kConsumerThreads = 256;  // 128 softmax + 128 correction

  struct SharedStorage {
    alignas(16) cute::uint64_t full[kStages];
    alignas(16) cute::uint64_t empty[kStages];
  };

  SharedStorage& storage_;
  CUTLASS_DEVICE PipelineSPO(SharedStorage& s) : storage_(s) {}
  CUTLASS_DEVICE void precompute_addrs() {}

  CUTLASS_DEVICE void init() {
    for (int i = 0; i < kStages; ++i) {
      cute::initialize_barrier(storage_.full[i], 1);
      cute::initialize_barrier(storage_.empty[i], kConsumerThreads);
    }
  }
  // Correction prefills empty (128 threads per stage)
  CUTLASS_DEVICE void prefill_correction() {
    for (int i = 0; i < kStages; ++i) mbarrier_arrive(storage_.empty[i]);
  }

  // MMA: wait O[stage] free before PV
  CUTLASS_DEVICE void producer_acquire_w_index_phase(int stage, int& phase) {
    wait_barrier(storage_.empty[stage], phase);
    phase ^= 1;
  }
  // MMA: get barrier for UMMA hardware arrive (S ready)
  CUTLASS_DEVICE cute::uint64_t& producer_get_barrier_w_index(int stage) {
    return storage_.full[stage];
  }

  // Softmax: wait S[stage] ready
  CUTLASS_DEVICE void consumer_wait_w_index_phase(int stage, int& phase) {
    wait_barrier(storage_.full[stage], phase);
    phase ^= 1;
  }
  // Softmax: signal P[stage] ready (128 threads)
  CUTLASS_DEVICE void consumer_release_w_index(int stage) {
    mbarrier_arrive(storage_.empty[stage]);
  }
};

// ============================================================================
// PipelineOAcc: Final O accumulator ready (indexed 2-stage)
// BSA: pipeline_o_acc (PipelineUmmaAsync)
//
//   Producer: MMA warp — signals final PV done (UMMA arrive on full)
//   Consumer: Correction warps — waits before combine
// ============================================================================

struct PipelineOAcc {
  static constexpr int kStages = 2;
  struct SharedStorage {
    alignas(16) cute::uint64_t full[kStages];
  };

  SharedStorage& storage_;
  CUTLASS_DEVICE PipelineOAcc(SharedStorage& s) : storage_(s) {}
  CUTLASS_DEVICE void precompute_addrs() {}

  CUTLASS_DEVICE void init() {
    for (int i = 0; i < kStages; ++i) cute::initialize_barrier(storage_.full[i], 1);
  }

  CUTLASS_DEVICE cute::uint64_t& producer_get_barrier_w_index(int stage) {
    return storage_.full[stage];
  }
  CUTLASS_DEVICE void consumer_wait_w_index_phase(int stage, int& phase) {
    wait_barrier(storage_.full[stage], phase);
    phase ^= 1;
  }
};

// ============================================================================
// PipelineSmStats: Softmax stats (PipelineAsync, mbarrier-based)
// BSA: pipeline_sm_stats
//
//   Producer: Softmax warps (128 threads per WG) — publish acc_scale/stats
//   Consumer: Correction warps (128 threads) — read stats for O rescaling
//
//   Both "full" (stats ready) and "empty" (stats consumed) barriers.
// ============================================================================

struct PipelineSmStats {
  static constexpr int kStages = 2;
  static constexpr int kProducerThreads = 128;  // per softmax WG
  static constexpr int kConsumerThreads = 128;  // correction WG

  struct SharedStorage {
    alignas(16) cute::uint64_t full[kStages];   // stats ready
    alignas(16) cute::uint64_t empty[kStages];  // stats consumed
  };

  SharedStorage& storage_;
  CUTLASS_DEVICE PipelineSmStats(SharedStorage& s) : storage_(s) {}
  CUTLASS_DEVICE void precompute_addrs() {}

  CUTLASS_DEVICE void init() {
    for (int i = 0; i < kStages; ++i) {
      cute::initialize_barrier(storage_.full[i], kProducerThreads);
      cute::initialize_barrier(storage_.empty[i], kConsumerThreads);
    }
  }
  // Pre-signal: correction prefills empty (stats initially free)
  CUTLASS_DEVICE void prefill_consumer() {
    for (int i = 0; i < kStages; ++i) mbarrier_arrive(storage_.empty[i]);
  }

  // Softmax: wait for previous stats consumed before overwriting
  CUTLASS_DEVICE void producer_acquire_w_index_phase(int stage, int& phase) {
    wait_barrier(storage_.empty[stage], phase);
    phase ^= 1;
  }
  // Softmax: stats published (128 threads arrive with .release for SMEM visibility)
  CUTLASS_DEVICE void producer_commit_w_index(int stage) {
    mbarrier_arrive_release(storage_.full[stage]);
  }
  // Correction: wait for stats ready (with .acquire for SM100 SMEM visibility)
  CUTLASS_DEVICE void consumer_wait_w_index_phase(int stage, int& phase) {
    wait_barrier_acquire(storage_.full[stage], phase);
    phase ^= 1;
  }
  // Correction: stats consumed, free for next produce
  CUTLASS_DEVICE void consumer_release_w_index(int stage) {
    mbarrier_arrive(storage_.empty[stage]);
  }
};

// ============================================================================
// PipelineOEpi: sO SMEM staging (Correction -> Epilogue TMA store)
// ============================================================================

struct PipelineOEpi {
  struct SharedStorage {
    alignas(16) cute::uint64_t notify;
    alignas(16) cute::uint64_t free;
  };

  SharedStorage& storage_;
  CUTLASS_DEVICE PipelineOEpi(SharedStorage& s) : storage_(s) {}
  CUTLASS_DEVICE void precompute_addrs() {}

  CUTLASS_DEVICE void init() {
    cute::initialize_barrier(storage_.notify, 128);
    cute::initialize_barrier(storage_.free, 1);
  }
  CUTLASS_DEVICE void prefill() { mbarrier_arrive(storage_.free); }

  CUTLASS_DEVICE void producer_commit() { mbarrier_arrive(storage_.notify); }
  CUTLASS_DEVICE void consumer_wait(int& phase) {
    wait_barrier(storage_.notify, phase);
    phase ^= 1;
  }
  CUTLASS_DEVICE void consumer_release() { mbarrier_arrive(storage_.free); }
};

// ============================================================================
// PipelinePLastSplit: last-split P ready (Softmax -> MMA, BSA split_P_arrive)
// Softmax writes P in fragments. After 3/4, softmax releases SPO_empty.
// After all P, softmax signals this barrier so MMA can issue remaining UTCHMMA.
//   Producer: Softmax (elect_one per warp) — arrive after last P fragment
//   Consumer: MMA warp (inline in UTCHMMA PTX sequence) — try_wait
// ============================================================================

struct PipelinePLastSplit {
  static constexpr int kStages = 2;
  static constexpr int kSoftmaxWarps = 4;  // warps per softmax WG

  struct SharedStorage {
    alignas(16) cute::uint64_t full[kStages];
  };

  SharedStorage& storage_;
  CUTLASS_DEVICE PipelinePLastSplit(SharedStorage& s) : storage_(s) {}
  CUTLASS_DEVICE void precompute_addrs() {}

  CUTLASS_DEVICE void init() {
    for (int i = 0; i < kStages; ++i) cute::initialize_barrier(storage_.full[i], kSoftmaxWarps);
  }

  // Softmax (elect_one per warp): signal last split of P is ready
  CUTLASS_DEVICE void producer_commit_w_index(int stage) { mbarrier_arrive(storage_.full[stage]); }

  // MMA: get barrier smem address for inline PTX try_wait
  CUTLASS_DEVICE uint32_t get_barrier_addr(int stage) {
    return cute::cast_smem_ptr_to_uint(&storage_.full[stage]);
  }
};

// ============================================================================
// CLC (Cluster Launch Control) infrastructure
// ============================================================================

// CLC response buffer: 128-bit (4x uint32) to hold the clusterlaunchcontrol
// query_cancel response from hardware.
struct CLCResponse {
  uint32_t data[4];
};

// Decoded work tile info from a CLC response.
struct WorkTileInfo {
  int row_tile;
  int head;
  int batch;
  bool is_valid;
};

// Inline helper: cast any smem pointer to uint32 address (uses same idiom as
// cute::cast_smem_ptr_to_uint but without the cute dependency at callsite).
__device__ __forceinline__ uint32_t smem_ptr_to_uint(void const* ptr) {
  uint32_t r;
  asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(r) : "l"(ptr));
  return r;
}

// decode_clc_response: read a CLCResponse from smem_addr and decode it.
// Returns WorkTileInfo with (row_tile, head, batch, is_valid).
// row_tile <- ctaidx.x, head <- ctaidx.y, batch <- ctaidx.z.
__device__ __forceinline__ WorkTileInfo decode_clc_response(uint32_t smem_addr) {
  WorkTileInfo info{0, 0, 0, false};
#if defined(CUTLASS_ARCH_CLC_ENABLED)
  uint32_t bx = 0, by = 0, bz = 0, valid_u32 = 0;
  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  .reg .b128 r;\n"
      "  ld.shared.b128 r, [%4];\n"
      "  clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, r;\n"
      "  selp.u32 %3, 1, 0, p;\n"
      "  @p clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0,%1,%2,_}, r;\n"
      "}\n"
      : "=r"(bx), "=r"(by), "=r"(bz), "=r"(valid_u32)
      : "r"(smem_addr)
      : "memory");
  asm volatile("fence.proxy.async;\n" ::: "memory");
  info.row_tile = static_cast<int>(bx);
  info.head = static_cast<int>(by);
  info.batch = static_cast<int>(bz);
  info.is_valid = (valid_u32 == 1);
#endif
  return info;
}

// issue_clc_query: issue a clusterlaunchcontrol.try_cancel async query.
// result_addr   — smem address of CLCResponse (128-bit aligned)
// mbarrier_addr — smem address of the transaction mbarrier (full barrier)
//                 which will be signaled when the response arrives (16 bytes tx).
__device__ __forceinline__ void issue_clc_query(uint32_t result_addr, uint32_t mbarrier_addr) {
#if defined(CUTLASS_ARCH_CLC_ENABLED)
  asm volatile(
      "clusterlaunchcontrol.try_cancel.async.shared::cta"
      ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];\n"
      :
      : "r"(result_addr), "r"(mbarrier_addr));
#endif
}

// ============================================================================
// PipelineCLC: CLC scheduler pipeline — CUTLASS PipelineCLCFetchAsync<1>
//
//   Producer: Sched warp (warp 15 / kSchedWarp) — issues CLC queries.
//   Consumer: Worker warps (kWorkerThreads = 480) — read decoded tile info.
//
//   CUTLASS manages full/empty barriers internally.
//   CLCResponse buffer is stored separately in kernel SharedStorage.
// ============================================================================

static constexpr int kCLCStages = 1;
using PipelineCLC = cutlass::PipelineCLCFetchAsync<kCLCStages>;
using PipelineCLCState = cutlass::PipelineState<kCLCStages>;

}  // namespace flash
