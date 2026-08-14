/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 *Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/fast_math.h"
#include "pipeline.hpp"
#include "utils.h"

namespace flash {

// Noinline fence between consumer_release and consumer_wait.
// Both the function call boundary (compiler barrier) and membar.gl (hardware fence)
// are required to prevent CLC persistent scheduling hang on SM100.
static __device__ __noinline__ void clc_transition_fence() {
  asm volatile("membar.gl;\n" ::: "memory");
}

// CLC persistent tile scheduler with batch support.
// Grid = (num_row_tiles * num_heads * batch, 1, 1) for CLC 1D flattening.
//
// Uses CUTLASS PipelineCLCFetchAsync<1>.
// Pattern: release-then-wait (workers release old, then wait for new).
// This gives the scheduler a window to reinit pipeline_o_epi between tiles.
//
// consumer_arv_count = kWorkerThreads (480): only worker warps release.
// Scheduler warp does NOT call consumer_release (only producer_acquire + consumer_wait).
struct CLCTileScheduler {
  struct Params {
    int num_row_tiles;
    int num_kv_blocks;
    int num_heads;
    int num_batch;
  };

  PipelineCLC& pipeline_clc;
  CLCResponse* clc_response;
  Params params;
  PipelineCLCState prod_state;  // {0, 0, 0} — blocks on first acquire until workers release
  PipelineCLCState cons_state;  // {0, 0, 0}

  CUTLASS_DEVICE
  CLCTileScheduler(PipelineCLC& clc, CLCResponse* resp, Params const& p)
      : pipeline_clc(clc), clc_response(resp), params(p), prod_state(), cons_state() {}

  CUTLASS_DEVICE
  static WorkTileInfo decode(int linear_idx, int num_row_tiles, int num_heads) {
    int tiles_per_batch = num_row_tiles * num_heads;
    int batch = linear_idx / tiles_per_batch;
    int rem = linear_idx % tiles_per_batch;
    int head = rem / num_row_tiles;
    int row_tile = rem % num_row_tiles;
    return {row_tile, head, batch, true};
  }

  CUTLASS_DEVICE
  WorkTileInfo initial_work_tile_info() {
    return decode(static_cast<int>(blockIdx.x), params.num_row_tiles, params.num_heads);
  }

  // ---- Sched warp: acquire(empty) → [reinit callback] → issue CLC query ----
  // producer_acquire: waits for kWorkerThreads releases on empty, then does
  // arrive_and_expect_tx on full (only lane 0, internally by CUTLASS).
  // Called by single thread (lane 0 of scheduler warp).
  template <typename ReinitFn>
  CUTLASS_DEVICE void advance_to_next_work(ReinitFn&& reinit_fn) {
    pipeline_clc.producer_acquire(prod_state);
    reinit_fn();
    uint32_t mbar = pipeline_clc.producer_get_barrier(prod_state);
    uint32_t resp_addr = smem_ptr_to_uint(&clc_response[prod_state.index()]);
    issue_clc_query(resp_addr, mbar);
    ++prod_state;
  }

  CUTLASS_DEVICE void advance_to_next_work() {
    pipeline_clc.producer_acquire(prod_state);
    uint32_t mbar = pipeline_clc.producer_get_barrier(prod_state);
    uint32_t resp_addr = smem_ptr_to_uint(&clc_response[prod_state.index()]);
    issue_clc_query(resp_addr, mbar);
    ++prod_state;
  }

  // ---- Sched warp: wait(full) → decode (NO consumer_release) ----
  CUTLASS_DEVICE WorkTileInfo fetch_next_work() {
    pipeline_clc.consumer_wait(cons_state);
    uint32_t resp_addr = smem_ptr_to_uint(&clc_response[cons_state.index()]);
    auto resp = decode_clc_response(resp_addr);
    // Scheduler does NOT consumer_release — only workers do (consumer_arv_count = kWorkerThreads).
    ++cons_state;
    if (!resp.is_valid) {
      return {0, 0, 0, false};
    }
    return decode(resp.row_tile, params.num_row_tiles, params.num_heads);
  }

  // ---- Worker consumer: fence → release(old) → wait(new) → decode ----
  CUTLASS_DEVICE WorkTileInfo consumer_advance() {
    flash::tcgen05_fence_before_sync();
    cutlass::arch::fence_view_async_tmem_store();
    cutlass::arch::fence_view_async_shared();
    // Release old CLC result (enables scheduler's next producer_acquire)
    pipeline_clc.consumer_release(cons_state);
    clc_transition_fence();
    // Wait for new CLC result (scheduler has issued query after our release)
    pipeline_clc.consumer_wait(cons_state);
    uint32_t resp_addr = smem_ptr_to_uint(&clc_response[cons_state.index()]);
    auto resp = decode_clc_response(resp_addr);
    ++cons_state;
    if (!resp.is_valid) {
      // Extra release so producer_tail can drain the pending empty barrier.
      pipeline_clc.consumer_release(cons_state);
      return {0, 0, 0, false};
    }
    flash::tcgen05_commit();
    return decode(resp.row_tile, params.num_row_tiles, params.num_heads);
  }

  // ---- Producer tail: drain pipeline before exit ----
  CUTLASS_DEVICE void producer_tail() { pipeline_clc.producer_tail(prod_state); }
};

}  // namespace flash
