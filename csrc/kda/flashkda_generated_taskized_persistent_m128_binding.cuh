/* Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0. */
#pragma once

#include "flashkda_generated_binding_common.cuh"

namespace flashinfer::flash_kda_generated {

struct TaskizedPersistentM128Args {
  void *q{}, *q_tma{}, *k{}, *k_tma{}, *v{}, *v_tma{}, *g{}, *g_tma{};
  void *beta{}, *beta_tma{}, *a_log{}, *dt_bias{}, *cu_seqlens{}, *seq_order{};
  void *task_ids{}, *task_offsets{}, *task_token_starts{}, *task_token_counts{};
  void *task_state_sources{}, *task_state_destinations{}, *mid_state{}, *mid_state_ready{};
  void *initial_state{}, *out{}, *out_tma{}, *final_state{};
  uint64_t state_indices_addr{};
  int64_t state_slot_stride{};
  int32_t use_state_indices{};
  void *initial_state_f32{}, *final_state_f32{};
  int32_t num_heads{}, use_initial_state{}, store_final_state{};
  float scale{}, lower_bound{};
};

inline void LaunchTaskizedPersistentM128(TaskizedPersistentM128Args args,
                                         const StatePointerSlots& state,
                                         dim3 grid, cudaStream_t stream) {
  args.initial_state = state.initial_state;
  args.final_state = state.final_state;
  args.initial_state_f32 = state.initial_state_f32;
  args.final_state_f32 = state.final_state_f32;
  void* kernel_args[] = {
      &args.q, &args.q_tma, &args.k, &args.k_tma, &args.v, &args.v_tma,
      &args.g, &args.g_tma, &args.beta, &args.beta_tma, &args.a_log,
      &args.dt_bias, &args.cu_seqlens, &args.seq_order, &args.task_ids,
      &args.task_offsets, &args.task_token_starts, &args.task_token_counts,
      &args.task_state_sources, &args.task_state_destinations, &args.mid_state,
      &args.mid_state_ready, &args.initial_state, &args.out, &args.out_tma,
      &args.final_state, &args.state_indices_addr, &args.state_slot_stride,
      &args.use_state_indices, &args.initial_state_f32, &args.final_state_f32,
      &args.num_heads, &args.use_initial_state, &args.store_final_state,
      &args.scale, &args.lower_bound};
  CheckArgumentCount<36>(kernel_args);
  ConfigureAndLaunch(FLASHKDA_GENERATED_KERNEL_ARGUMENT, grid,
                     stream, kernel_args,
                     "generated taskized-persistent M128 launch");
}

inline void RunTaskizedPersistentM128(
    TensorView q, TensorView k, TensorView v, TensorView g, TensorView beta,
    TensorView beta_tma, TensorView a_log, TensorView dt_bias,
    TensorView cu_seqlens, TensorView seq_order, TensorView task_ids,
    TensorView task_offsets, TensorView task_token_starts,
    TensorView task_token_counts, TensorView task_state_sources,
    TensorView task_state_destinations, TensorView mid_state,
    TensorView mid_state_ready, TensorView state_indices,
    TensorView initial_state, TensorView out, TensorView final_state,
    TensorView descriptor_storage, int64_t prepare_descriptors,
    int64_t num_heads, int64_t beta_token_stride,
    int64_t state_slot_stride, int64_t use_state_indices,
    int64_t use_initial_state, int64_t store_final_state, double scale,
    double lower_bound, int64_t grid_x, int64_t grid_y, int64_t grid_z,
    int64_t cuda_stream) {
  TVM_FFI_ICHECK(q.device().device_type == kDLCUDA);
  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  const auto prepared = PrepareCommonInputs<
      FLASHKDA_GENERATED_VALUE_ROWS, FLASHKDA_GENERATED_TMA_TILE_TOKENS,
      FLASHKDA_GENERATED_PAIR_PACKED_BETA != 0,
      FLASHKDA_GENERATED_VALUE_TMA_RANK == 4>(
      q, k, v, g, beta, beta_tma, a_log, dt_bias, cu_seqlens, seq_order,
      state_indices, initial_state, out, final_state, descriptor_storage,
      prepare_descriptors, num_heads, beta_token_stride, state_slot_stride,
      use_state_indices, use_initial_state, store_final_state, scale,
      lower_bound, cuda_stream);
  TaskizedPersistentM128Args args{};
  args.q = q.data_ptr(); args.q_tma = prepared.tma.q;
  args.k = k.data_ptr(); args.k_tma = prepared.tma.k;
  args.v = v.data_ptr(); args.v_tma = prepared.tma.v;
  args.g = g.data_ptr(); args.g_tma = prepared.tma.g;
  args.beta = beta.data_ptr(); args.beta_tma = prepared.tma.beta;
  args.a_log = a_log.data_ptr(); args.dt_bias = dt_bias.data_ptr();
  args.cu_seqlens = cu_seqlens.data_ptr(); args.seq_order = seq_order.data_ptr();
#define FLASHKDA_SET_I32_BUFFER(field)                                                \
  args.field = CheckedBufferPointer(field, #field, prepared.device_id, dl_int32)
  FLASHKDA_SET_I32_BUFFER(task_ids);
  FLASHKDA_SET_I32_BUFFER(task_offsets);
  FLASHKDA_SET_I32_BUFFER(task_token_starts);
  FLASHKDA_SET_I32_BUFFER(task_token_counts);
  FLASHKDA_SET_I32_BUFFER(task_state_sources);
  FLASHKDA_SET_I32_BUFFER(task_state_destinations);
#undef FLASHKDA_SET_I32_BUFFER
  args.mid_state = CheckedBufferPointer(mid_state, "mid_state", prepared.device_id,
                                        dl_bfloat16);
  args.mid_state_ready = CheckedBufferPointer(
      mid_state_ready, "mid_state_ready", prepared.device_id, dl_uint32);
  const int64_t entry_count = task_ids.numel();
  TVM_FFI_ICHECK(task_ids.ndim() == 1 && entry_count > 0);
  for (const TensorView* tensor : {&task_token_starts, &task_token_counts,
                                   &task_state_sources, &task_state_destinations}) {
    TVM_FFI_ICHECK(tensor->ndim() == 1 && tensor->numel() == entry_count)
        << "each task metadata tensor must have one entry per task";
  }
  TVM_FFI_ICHECK(task_offsets.ndim() == 1 && task_offsets.numel() >= 2);
  TVM_FFI_ICHECK(mid_state.ndim() == 3 && mid_state.size(1) == 128 &&
                 mid_state.size(2) == 128 && mid_state.size(0) == mid_state_ready.numel())
      << "mid_state and mid_state_ready shapes disagree";
  args.out = out.data_ptr(); args.out_tma = prepared.tma.out;
  args.state_indices_addr = reinterpret_cast<uintptr_t>(state_indices.data_ptr());
  args.state_slot_stride = state_slot_stride;
  args.use_state_indices = CheckedInt32(use_state_indices, "use_state_indices");
  args.num_heads = CheckedInt32(num_heads, "num_heads");
  args.use_initial_state = CheckedInt32(use_initial_state, "use_initial_state");
  args.store_final_state = CheckedInt32(store_final_state, "store_final_state");
  args.scale = static_cast<float>(scale); args.lower_bound = static_cast<float>(lower_bound);
  LaunchTaskizedPersistentM128(args, prepared.state,
                               CheckedGrid(grid_x, grid_y, grid_z), prepared.stream);
}

}  // namespace flashinfer::flash_kda_generated

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    run, flashinfer::flash_kda_generated::RunTaskizedPersistentM128);
