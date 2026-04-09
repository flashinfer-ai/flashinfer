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
#ifndef FLASHINFER_MAMBA_SSU_INCREMENTAL_CUH_
#define FLASHINFER_MAMBA_SSU_INCREMENTAL_CUH_

#include <cstdint>

namespace flashinfer::mamba::incremental {

struct SsuIncrementalParams {
  uint32_t batch{}, nheads{}, dim{}, dstate{}, ngroups{};
  uint32_t state_cache_size{};
  uint32_t ntokens_mtp{};
  int32_t pad_slot_id{-1};

  bool dt_softplus{false};

  // ── Tensor pointers ──
  void* __restrict__ state{nullptr};    // (cache, nheads, dim, dstate)
  void* __restrict__ x{nullptr};        // (batch, T, nheads, dim)
  void* __restrict__ dt{nullptr};       // (batch, T, nheads, dim) tie_hdim
  void* __restrict__ A{nullptr};        // (nheads, dim, dstate) tie_hdim
  void* __restrict__ B{nullptr};        // (batch, T, ngroups, dstate)
  void* __restrict__ C{nullptr};        // (batch, T, ngroups, dstate)
  void* __restrict__ D{nullptr};        // (nheads, dim), optional
  void* __restrict__ z{nullptr};        // (batch, T, nheads, dim), optional
  void* __restrict__ dt_bias{nullptr};  // (nheads, dim) tie_hdim, optional
  void* __restrict__ output{nullptr};   // (batch, T, nheads, dim)

  // ── Cache tensors for incremental replay ──
  void* __restrict__ old_x{nullptr};              // (cache, T, nheads, dim) single-buffered
  void* __restrict__ old_B{nullptr};              // (cache, 2, T, ngroups, dstate) double-buffered
  void* __restrict__ old_dt_proc{nullptr};        // (cache, 2, nheads, T) double-buffered, f32
  void* __restrict__ old_cumAdt{nullptr};         // (cache, 2, nheads, T) double-buffered, f32
  void* __restrict__ cache_buf_idx{nullptr};      // (cache,) int32
  void* __restrict__ prev_num_accepted{nullptr};  // (cache,) int32

  // ── Index tensors ──
  void* __restrict__ state_batch_indices{nullptr};  // (batch,) optional

  // ── Block-scale decode factors for quantized state ──
  void* __restrict__ state_scale{nullptr};  // float32: (cache, nheads, dim)

  // ── Philox PRNG seed for stochastic rounding ──
  const int64_t* rand_seed{nullptr};

  // ── Strides ──
  // state: (cache, nheads, dim, dstate) — inner 3 dims contiguous
  int64_t state_stride_batch{};

  // x: (batch, T, nheads, dim)
  int64_t x_stride_batch{};
  int64_t x_stride_mtp{};

  // dt: (batch, T, nheads, dim) — tie_hdim (stride_dim=0)
  int64_t dt_stride_batch{};
  int64_t dt_stride_mtp{};

  // B: (batch, T, ngroups, dstate)
  int64_t B_stride_batch{};
  int64_t B_stride_mtp{};

  // C: (batch, T, ngroups, dstate)
  int64_t C_stride_batch{};
  int64_t C_stride_mtp{};

  // output: (batch, T, nheads, dim)
  int64_t out_stride_batch{};
  int64_t out_stride_mtp{};

  // z: (batch, T, nheads, dim)
  int64_t z_stride_batch{};
  int64_t z_stride_mtp{};

  // old_x: (cache, T, nheads, dim) — single-buffered
  int64_t old_x_stride_cache{};
  int64_t old_x_stride_mtp{};

  // old_B: (cache, 2, T, ngroups, dstate) — double-buffered
  int64_t old_B_stride_cache{};
  int64_t old_B_stride_dbuf{};
  int64_t old_B_stride_mtp{};

  // old_dt_proc: (cache, 2, nheads, T) — double-buffered, T contiguous
  int64_t old_dt_proc_stride_cache{};
  int64_t old_dt_proc_stride_dbuf{};
  int64_t old_dt_proc_stride_head{};

  // old_cumAdt: (cache, 2, nheads, T) — double-buffered, T contiguous
  int64_t old_cumAdt_stride_cache{};
  int64_t old_cumAdt_stride_dbuf{};
  int64_t old_cumAdt_stride_head{};

  // state_scale: (cache, nheads, dim)
  int64_t state_scale_stride_batch{};
};

// Forward declaration — defined in kernel_ssu_incremental.cuh
template <typename input_t, typename dt_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, typename state_scale_t>
void launchSsuIncremental(SsuIncrementalParams& params, cudaStream_t stream);

}  // namespace flashinfer::mamba::incremental

#endif  // FLASHINFER_MAMBA_SSU_INCREMENTAL_CUH_
