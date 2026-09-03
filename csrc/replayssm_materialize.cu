/*
 * Copyright (c) 2026 by FlashInfer team.
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
// clang-format off
// replayssm_materialize_config.inc MUST come before the FlashInfer Mamba
// headers: it defines input_t, state_t, and the compile-time kernel parameters.
#include "replayssm_materialize_config.inc"
#include <flashinfer/mamba/checkpointing_ssu.cuh>
#include <flashinfer/mamba/kernel_checkpointing_ssu.cuh>
#include <flashinfer/mamba/kernel_checkpointing_ssu_8bit.cuh>
// clang-format on

#include <cuda_runtime.h>
#include <flashinfer/exception.h>

#include <limits>

#include "tvm_ffi_utils.h"

namespace flashinfer::mamba {
using namespace checkpointing;

struct MaterializeParams {
  const int64_t* state_ptrs;
  const int64_t* state_slot_strides;
  const int64_t* x_ptrs;
  const int64_t* x_slot_strides;
  const int64_t* b_ptrs;
  const int64_t* b_slot_strides;
  const int64_t* dt_ptrs;
  const int64_t* dt_slot_strides;
  const int64_t* a_ptrs;
  const int64_t* scale_ptrs;
  const int64_t* scale_slot_strides;
  const int32_t* src_slots;
  const int32_t* dst_slots;
  const int32_t* ring_start;
  const int32_t* replay_prefix_len;
  const int32_t* active_request_indices;
  const int64_t* rand_seed;
  int batch, layers, heads, ring_buffer_len, pad_slot_id;
};

__device__ __forceinline__ void advance_persistent_coordinates(
    int64_t& work, int& virtual_request, int& layer, int& head, int work_stride,
    int virtual_request_delta, int layer_delta, int head_delta, int layers, int heads) {
  work += work_stride;
  head += head_delta;
  if (head >= heads) {
    head -= heads;
    ++layer;
  }
  layer += layer_delta;
  if (layer >= layers) {
    layer -= layers;
    ++virtual_request;
  }
  virtual_request += virtual_request_delta;
}

// Derive a per-layer Philox key with the SplitMix64 finalizer (Stafford mix13).
// The golden-ratio increment and avalanche multipliers are SplitMix64 constants.
__device__ __forceinline__ int64_t layer_philox_seed(int64_t base_seed, int layer) {
  uint64_t z = static_cast<uint64_t>(base_seed) ^ (0x9E3779B97F4A7C15ULL * (uint64_t(layer) + 1));
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return static_cast<int64_t>(z ^ (z >> 31));
}

// Reuse checkpointing_ssu's force-inlined tensor-core recurrence while a fixed
// CTA pool grid-strides over the logical (virtual request, layer, head) work.
template <typename T>
__global__ void materialize_replay_kernel(MaterializeParams p) {
  constexpr int NUM_WARPS = 4;
  static_assert(MAX_WINDOW > 0 && MAX_WINDOW <= 16,
                "ReplaySSM materialization supports max_window in [1, 16]");
  static_assert(sizeof(T) != 1 || (DIM == 64 && DSTATE == 128),
                "8-bit ReplaySSM materialization requires DIM=64 and DSTATE=128");
  using SmemT = std::conditional_t<
      sizeof(T) == 1, CheckpointingSsuStorage8bit<input_t, T, NPREDICTED, MAX_WINDOW, DIM, DSTATE>,
      CheckpointingSsuStorage<input_t, T, NPREDICTED, MAX_WINDOW, DIM, DSTATE>>;
  extern __shared__ __align__(128) char smem_buf[];
  auto& smem = *reinterpret_cast<SmemT*>(smem_buf);
  int const lane = threadIdx.x, warp = threadIdx.y, tid = warp * warpSize + lane;
  int64_t const work_items = int64_t(p.batch) * p.layers * p.heads;
  int64_t work = blockIdx.x;
  int head = work % p.heads;
  int64_t virtual_request_layer = work / p.heads;
  int layer = virtual_request_layer % p.layers;
  int virtual_request = virtual_request_layer / p.layers;
  int const work_stride = gridDim.x;
  int const head_delta = work_stride % p.heads;
  int64_t const virtual_request_layer_delta = work_stride / p.heads;
  int const layer_delta = virtual_request_layer_delta % p.layers;
  int const virtual_request_delta = virtual_request_layer_delta / p.layers;
  // Virtual requests index active_request_indices; physical requests index
  // the other batch-sized inputs.
  int last_virtual_request = -1, physical_request = -1, count = -1;
  while (work < work_items) {
    if (virtual_request != last_virtual_request) {
      physical_request = p.active_request_indices[virtual_request];
      if (physical_request < 0) break;
      count = p.replay_prefix_len[physical_request];
      last_virtual_request = virtual_request;
    }
    int64_t const table = int64_t(layer) * p.batch + physical_request;
    int const src_slot = p.src_slots[table], dst_slot = p.dst_slots[table];
    if (src_slot == p.pad_slot_id || dst_slot == p.pad_slot_id || count > MAX_WINDOW) {
      advance_persistent_coordinates(work, virtual_request, layer, head, work_stride,
                                     virtual_request_delta, layer_delta, head_delta, p.layers,
                                     p.heads);
      continue;
    }
    int const group = head / HEADS_PER_GROUP;
    auto const* state = reinterpret_cast<T const*>(p.state_ptrs[layer]);
    auto* state_dst = reinterpret_cast<T*>(p.state_ptrs[layer]);
    int64_t const state_src_base =
        int64_t(src_slot) * p.state_slot_strides[layer] + int64_t(head) * DIM * DSTATE;
    int64_t const state_dst_base =
        int64_t(dst_slot) * p.state_slot_strides[layer] + int64_t(head) * DIM * DSTATE;
    if (count == 0) {
      for (int i = tid; i < DIM * DSTATE; i += blockDim.x * blockDim.y)
        state_dst[state_dst_base + i] = state[state_src_base + i];
      if (p.scale_ptrs[layer] != 0 && tid < DIM) {
        auto const* scales = reinterpret_cast<float const*>(p.scale_ptrs[layer]);
        auto* dst_scales = reinterpret_cast<float*>(p.scale_ptrs[layer]);
        int64_t const scale_src =
            int64_t(src_slot) * p.scale_slot_strides[layer] + int64_t(head) * DIM;
        int64_t const scale_dst =
            int64_t(dst_slot) * p.scale_slot_strides[layer] + int64_t(head) * DIM;
        dst_scales[scale_dst + tid] = scales[scale_src + tid];
      }
      advance_persistent_coordinates(work, virtual_request, layer, head, work_stride,
                                     virtual_request_delta, layer_delta, head_delta, p.layers,
                                     p.heads);
      continue;
    }
    load_state_per_warp<T, DIM, DSTATE, NUM_WARPS>(smem, state, state_src_base, warp, lane);
    auto const* x = reinterpret_cast<input_t const*>(p.x_ptrs[layer]);
    auto const* b = reinterpret_cast<input_t const*>(p.b_ptrs[layer]);
    auto const* dt = reinterpret_cast<float const*>(p.dt_ptrs[layer]);
    int const start = p.ring_start[physical_request];
    CheckpointingSsuParams view{};
    view.dt_cache = const_cast<float*>(dt);
    view.dt_cache_stride_seq = p.dt_slot_strides[layer];
    view.dt_cache_stride_head = p.ring_buffer_len;
    view.ring_buffer_len = p.ring_buffer_len;
    using XShape = cute::Shape<cute::Int<SmemT::MAX_WINDOW_PAD_MMA_K>, cute::Int<DIM>>;
    using BShape = cute::Shape<cute::Int<SmemT::MAX_WINDOW_PAD_MMA_K>, cute::Int<DSTATE>>;
    // The barrier below makes these shared ring tiles visible CTA-wide, so
    // unlike checkpointing_ssu, each tile needs only one warp to load it.
    if (warp == 0) {
      load_ring_tile_async<XShape, MAX_WINDOW>(
          smem.old_x,
          x + int64_t(src_slot) * p.x_slot_strides[layer] + int64_t(head) * p.ring_buffer_len * DIM,
          DIM, lane, start, p.ring_buffer_len, count);
    }
    if (warp == 1) {
      load_ring_tile_async<BShape, MAX_WINDOW>(smem.old_B,
                                               b + int64_t(src_slot) * p.b_slot_strides[layer] +
                                                   int64_t(group) * p.ring_buffer_len * DSTATE,
                                               DSTATE, lane, start, p.ring_buffer_len, count);
    }
    float const a = reinterpret_cast<matrixA_t const*>(p.a_ptrs[layer])[head];
    // Scan dt then multiply by A, matching checkpointing_ssu's recurrence.
    if (warp == 2) {
      load_old_dt_cumAdt(view, lane, src_slot, start, head, count, a, smem.old_dt, smem.old_cumAdt);
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    int64_t const rand_seed = (PHILOX_ROUNDS > 0) ? layer_philox_seed(*p.rand_seed, layer) : 0;
    if constexpr (sizeof(T) == 1) {
      // view's state and scale are used below only for the
      // destination.  state is already in smem, source_scale is
      // loaded directly into registers below.
      view.state = state_dst;
      view.state_stride_seq = p.state_slot_strides[layer];
      view.state_scale = reinterpret_cast<void*>(p.scale_ptrs[layer]);
      view.state_scale_stride_seq = p.scale_slot_strides[layer];
      auto tiled_mma_chain =
          cute::make_tiled_mma(cute::MMA_Atom<cute::MMA_Traits<checkpointing::MMA_prop::AtomK16>>{},
                               cute::Layout<cute::Shape<cute::_4, cute::_1>>{});
      auto thr_mma_chain = tiled_mma_chain.get_slice(tid);
      // frag_y_dxt is used in replay_state_* below for output token results, which we ignore.
      auto id_dxt = cute::make_identity_tensor(
          cute::make_shape(cute::Int<DIM>{}, cute::Int<SmemT::NPREDICTED_PAD_MMA_M>{}));
      cute::Tensor frag_y_dxt = thr_mma_chain.partition_fragment_C(id_dxt);
      cute::clear(frag_y_dxt);
      // replay_state* matmuls to get state but doesn't store it; it's just to get the encode_scale,
      // whose reciprocal it also writes to dest scale's dram through view's fields above.  (And
      // make output tokens, that for us are junk).  encode repeats the state update, and now that
      // we have the scale we can encode it.  Motivation is that matmul repeat is cheaper than more
      // smem storage hurting occupancy.
      float encode_scale_per_row[2];
      float total_scale[2];
      auto const* source_scale = reinterpret_cast<float const*>(p.scale_ptrs[layer]) +
                                 int64_t(src_slot) * p.scale_slot_strides[layer] +
                                 int64_t(head) * DIM;
      replay_state_mma_8bit_chain<input_t, T, DIM, DIM, DSTATE, SmemT, decltype(frag_y_dxt), true>(
          smem, view, warp, lane, count, /*d_tile=*/0, dst_slot, head,
          /*must_checkpoint=*/true, frag_y_dxt, encode_scale_per_row, total_scale, source_scale);
      encode_state_replay_8bit<input_t, T, DIM, DIM, DSTATE, PHILOX_ROUNDS>(
          smem, view, warp, lane, count, /*d_tile=*/0, dst_slot, head, encode_scale_per_row,
          total_scale, rand_seed, state_dst_base);
      // Make sure shared memory results are visible for storage.
      __syncthreads();
      store_state<T, DIM, DIM, DSTATE, NUM_WARPS>(smem, view, warp, lane, /*d_tile=*/0, head,
                                                  dst_slot);
    } else {
      replay_state_mma<input_t, T, DIM, DIM, DSTATE, PHILOX_ROUNDS, NUM_WARPS>(
          smem, view, warp, lane, count, /*d_tile=*/0, state_dst_base, state_dst + state_dst_base,
          rand_seed, /*must_checkpoint=*/true);
      // Different warp-partitioning for storage vs replay, so need to sync.
      __syncthreads();
      if constexpr (!(PHILOX_ROUNDS > 0 && std::is_same_v<T, __half>)) {
        view.state = state_dst;
        view.state_stride_seq = p.state_slot_strides[layer];
        store_state<T, DIM, DIM, DSTATE, NUM_WARPS>(smem, view, warp, lane, /*d_tile=*/0, head,
                                                    dst_slot);
      }
    }
    // The next grid-stride item reuses shared memory. The preceding store is
    // partitioned across warps, so wait for every warp before overwriting it.
    __syncthreads();
    advance_persistent_coordinates(work, virtual_request, layer, head, work_stride,
                                   virtual_request_delta, layer_delta, head_delta, p.layers,
                                   p.heads);
  }
}

void replayssm_materialize(TensorView state_ptrs, TensorView state_slot_strides, TensorView x_ptrs,
                           TensorView x_slot_strides, TensorView b_ptrs, TensorView b_slot_strides,
                           TensorView dt_ptrs, TensorView dt_slot_strides, TensorView a_ptrs,
                           TensorView scale_ptrs, TensorView scale_slot_strides,
                           TensorView src_slots, TensorView dst_slots, TensorView ring_start,
                           TensorView replay_prefix_len, TensorView active_request_indices,
                           int64_t num_layers, int64_t num_heads, int64_t ring_buffer_len,
                           int64_t pad_slot_id, tvm::ffi::Optional<TensorView> rand_seed) {
  CHECK_CUDA(state_ptrs);
  CHECK_CUDA(state_slot_strides);
  CHECK_CUDA(x_ptrs);
  CHECK_CUDA(x_slot_strides);
  CHECK_CUDA(b_ptrs);
  CHECK_CUDA(b_slot_strides);
  CHECK_CUDA(dt_ptrs);
  CHECK_CUDA(dt_slot_strides);
  CHECK_CUDA(a_ptrs);
  CHECK_CUDA(scale_ptrs);
  CHECK_CUDA(scale_slot_strides);
  CHECK_CUDA(src_slots);
  CHECK_CUDA(dst_slots);
  CHECK_CUDA(ring_start);
  CHECK_CUDA(replay_prefix_len);
  CHECK_CUDA(active_request_indices);
  auto check_same_device = [&state_ptrs](TensorView const& table, char const* name) {
    FLASHINFER_CHECK(table.device().device_id == state_ptrs.device().device_id, name,
                     " must be on the same CUDA device as state_ptrs");
  };
  check_same_device(state_slot_strides, "state_slot_strides");
  check_same_device(x_ptrs, "x_ptrs");
  check_same_device(x_slot_strides, "x_slot_strides");
  check_same_device(b_ptrs, "b_ptrs");
  check_same_device(b_slot_strides, "b_slot_strides");
  check_same_device(dt_ptrs, "dt_ptrs");
  check_same_device(dt_slot_strides, "dt_slot_strides");
  check_same_device(a_ptrs, "a_ptrs");
  check_same_device(scale_ptrs, "scale_ptrs");
  check_same_device(scale_slot_strides, "scale_slot_strides");
  check_same_device(src_slots, "src_slots");
  check_same_device(dst_slots, "dst_slots");
  check_same_device(ring_start, "ring_start");
  check_same_device(replay_prefix_len, "replay_prefix_len");
  check_same_device(active_request_indices, "active_request_indices");
  CHECK_DIM(1, state_ptrs);
  CHECK_DIM(1, state_slot_strides);
  CHECK_DIM(1, x_ptrs);
  CHECK_DIM(1, x_slot_strides);
  CHECK_DIM(1, b_ptrs);
  CHECK_DIM(1, b_slot_strides);
  CHECK_DIM(1, dt_ptrs);
  CHECK_DIM(1, dt_slot_strides);
  CHECK_DIM(1, a_ptrs);
  CHECK_DIM(1, scale_ptrs);
  CHECK_DIM(1, scale_slot_strides);
  CHECK_DIM(2, src_slots);
  CHECK_DIM(2, dst_slots);
  CHECK_DIM(1, ring_start);
  CHECK_DIM(1, replay_prefix_len);
  CHECK_DIM(1, active_request_indices);
  CHECK_CONTIGUOUS(state_ptrs);
  CHECK_CONTIGUOUS(state_slot_strides);
  CHECK_CONTIGUOUS(x_ptrs);
  CHECK_CONTIGUOUS(x_slot_strides);
  CHECK_CONTIGUOUS(b_ptrs);
  CHECK_CONTIGUOUS(b_slot_strides);
  CHECK_CONTIGUOUS(dt_ptrs);
  CHECK_CONTIGUOUS(dt_slot_strides);
  CHECK_CONTIGUOUS(a_ptrs);
  CHECK_CONTIGUOUS(scale_ptrs);
  CHECK_CONTIGUOUS(scale_slot_strides);
  CHECK_CONTIGUOUS(src_slots);
  CHECK_CONTIGUOUS(dst_slots);
  CHECK_CONTIGUOUS(ring_start);
  CHECK_CONTIGUOUS(replay_prefix_len);
  CHECK_CONTIGUOUS(active_request_indices);
  auto check_int_table = [](TensorView const& table, char const* name) {
    FLASHINFER_CHECK(table.dtype().code == kDLInt && table.dtype().bits == 64, name,
                     " must be int64");
  };
  check_int_table(state_ptrs, "state_ptrs");
  check_int_table(state_slot_strides, "state_slot_strides");
  check_int_table(x_ptrs, "x_ptrs");
  check_int_table(x_slot_strides, "x_slot_strides");
  check_int_table(b_ptrs, "b_ptrs");
  check_int_table(b_slot_strides, "b_slot_strides");
  check_int_table(dt_ptrs, "dt_ptrs");
  check_int_table(dt_slot_strides, "dt_slot_strides");
  check_int_table(a_ptrs, "a_ptrs");
  check_int_table(scale_ptrs, "scale_ptrs");
  check_int_table(scale_slot_strides, "scale_slot_strides");
  FLASHINFER_CHECK(src_slots.dtype().code == kDLInt && src_slots.dtype().bits == 32,
                   "src_slots must be int32");
  FLASHINFER_CHECK(dst_slots.dtype().code == kDLInt && dst_slots.dtype().bits == 32,
                   "dst_slots must be int32");
  FLASHINFER_CHECK(ring_start.dtype().code == kDLInt && ring_start.dtype().bits == 32,
                   "ring_start must be int32");
  FLASHINFER_CHECK(replay_prefix_len.dtype().code == kDLInt && replay_prefix_len.dtype().bits == 32,
                   "replay_prefix_len must be int32");
  FLASHINFER_CHECK(
      active_request_indices.dtype().code == kDLInt && active_request_indices.dtype().bits == 32,
      "active_request_indices must be int32");
  FLASHINFER_CHECK(num_layers > 0, "num_layers must be positive");
  FLASHINFER_CHECK(num_heads > 0, "num_heads must be positive");
  FLASHINFER_CHECK(num_heads % HEADS_PER_GROUP == 0,
                   "num_heads must be divisible by HEADS_PER_GROUP");
  constexpr int64_t kIntMax = std::numeric_limits<int>::max();
  FLASHINFER_CHECK(replay_prefix_len.size(0) > 0 && replay_prefix_len.size(0) <= kIntMax,
                   "batch must be in [1, INT_MAX]");
  FLASHINFER_CHECK(num_layers <= kIntMax, "num_layers must fit in int");
  FLASHINFER_CHECK(ring_buffer_len <= kIntMax, "ring_buffer_len must fit in int");
  auto check_layer_table = [num_layers](TensorView const& table, char const* name) {
    FLASHINFER_CHECK(table.size(0) == num_layers, name, " size must equal num_layers");
  };
  FLASHINFER_CHECK(state_ptrs.size(0) == num_layers, "state_ptrs size must equal num_layers");
  check_layer_table(state_slot_strides, "state_slot_strides");
  check_layer_table(x_ptrs, "x_ptrs");
  check_layer_table(x_slot_strides, "x_slot_strides");
  check_layer_table(b_ptrs, "b_ptrs");
  check_layer_table(b_slot_strides, "b_slot_strides");
  check_layer_table(dt_ptrs, "dt_ptrs");
  check_layer_table(dt_slot_strides, "dt_slot_strides");
  check_layer_table(a_ptrs, "a_ptrs");
  check_layer_table(scale_ptrs, "scale_ptrs");
  check_layer_table(scale_slot_strides, "scale_slot_strides");
  FLASHINFER_CHECK(
      src_slots.size(0) == num_layers && src_slots.size(1) == replay_prefix_len.size(0),
      "src_slots must be [num_layers, batch]");
  FLASHINFER_CHECK(
      dst_slots.size(0) == num_layers && dst_slots.size(1) == replay_prefix_len.size(0),
      "dst_slots must be [num_layers, batch]");
  FLASHINFER_CHECK(ring_start.size(0) == replay_prefix_len.size(0),
                   "ring_start must have shape [batch]");
  FLASHINFER_CHECK(active_request_indices.size(0) == replay_prefix_len.size(0),
                   "active_request_indices must have shape [batch]");
  FLASHINFER_CHECK(ring_buffer_len > 0, "ring_buffer_len must be positive");
  if constexpr (PHILOX_ROUNDS > 0) {
    FLASHINFER_CHECK(rand_seed.has_value(), "rand_seed is required when PHILOX_ROUNDS > 0");
    auto const& seed = rand_seed.value();
    CHECK_CUDA(seed);
    check_same_device(seed, "rand_seed");
    CHECK_DIM(1, seed);
    CHECK_CONTIGUOUS(seed);
    FLASHINFER_CHECK(seed.numel() == 1 && seed.dtype().code == kDLInt && seed.dtype().bits == 64,
                     "rand_seed must be a one-element CUDA int64 tensor");
  }
  MaterializeParams p{
      static_cast<const int64_t*>(state_ptrs.data_ptr()),
      static_cast<const int64_t*>(state_slot_strides.data_ptr()),
      static_cast<const int64_t*>(x_ptrs.data_ptr()),
      static_cast<const int64_t*>(x_slot_strides.data_ptr()),
      static_cast<const int64_t*>(b_ptrs.data_ptr()),
      static_cast<const int64_t*>(b_slot_strides.data_ptr()),
      static_cast<const int64_t*>(dt_ptrs.data_ptr()),
      static_cast<const int64_t*>(dt_slot_strides.data_ptr()),
      static_cast<const int64_t*>(a_ptrs.data_ptr()),
      static_cast<const int64_t*>(scale_ptrs.data_ptr()),
      static_cast<const int64_t*>(scale_slot_strides.data_ptr()),
      static_cast<const int32_t*>(src_slots.data_ptr()),
      static_cast<const int32_t*>(dst_slots.data_ptr()),
      static_cast<const int32_t*>(ring_start.data_ptr()),
      static_cast<const int32_t*>(replay_prefix_len.data_ptr()),
      static_cast<const int32_t*>(active_request_indices.data_ptr()),
      rand_seed.has_value() ? static_cast<const int64_t*>(rand_seed.value().data_ptr()) : nullptr,
      int(replay_prefix_len.size(0)),
      int(num_layers),
      int(num_heads),
      int(ring_buffer_len),
      int(pad_slot_id)};
  ffi::CUDADeviceGuard device_guard(state_ptrs.device().device_id);
  const cudaStream_t stream = get_stream(state_ptrs.device());
  int64_t const work_items = int64_t(p.batch) * p.layers * p.heads;
  constexpr size_t smem =
      sizeof(std::conditional_t < sizeof(state_t) == 1,
             CheckpointingSsuStorage8bit<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, DSTATE>,
             CheckpointingSsuStorage < input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, DSTATE >>);
  FLASHINFER_CUDA_CHECK(cudaFuncSetAttribute(materialize_replay_kernel<state_t>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  constexpr int kThreadsPerCta = warpSize * 4;
  int device = 0, sm_count = 0, blocks_per_sm = 0;
  FLASHINFER_CUDA_CHECK(cudaGetDevice(&device));
  FLASHINFER_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  FLASHINFER_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, materialize_replay_kernel<state_t>, kThreadsPerCta, smem));
  int64_t const persistent_ctas = int64_t(sm_count) * blocks_per_sm;
  FLASHINFER_CHECK(persistent_ctas > 0, "ReplaySSM materialize has no resident CTA configuration");
  dim3 grid(work_items < persistent_ctas ? int(work_items) : int(persistent_ctas));
  materialize_replay_kernel<state_t><<<grid, dim3(warpSize, 4), smem, stream>>>(p);
  FLASHINFER_CUDA_CHECK(cudaGetLastError());
}
}  // namespace flashinfer::mamba
