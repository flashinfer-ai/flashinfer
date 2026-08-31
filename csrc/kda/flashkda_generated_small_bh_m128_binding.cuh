/* Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0. */
#pragma once

#include "flashkda_generated_binding_common.cuh"

namespace flashinfer::flash_kda_generated {

constexpr int64_t kGeneratedSmallBhPacketRows = 123;
constexpr int64_t kGeneratedSmallBhPacketElements = 128;
constexpr int64_t kGeneratedSmallBhGroupSize = 8;
constexpr int64_t kGeneratedSmallBhMaxTasks = 8;
constexpr int64_t kGeneratedSmallBhRingStages = 35;
constexpr int64_t kGeneratedSmallBhMinSequenceLength = 2048;
constexpr size_t kGeneratedSmallBhDescriptorCount = 7;

inline CUtensorMap EncodeGeneratedSmallBhPacketTma(const TensorView& tensor) {
  TVM_FFI_ICHECK(tensor.ndim() == 2 &&
                 tensor.size(1) == kGeneratedSmallBhPacketElements)
      << "packet_workspace must have shape [rows, 128]";
  TVM_FFI_ICHECK(tensor.stride(1) == 1 &&
                 tensor.stride(0) == kGeneratedSmallBhPacketElements)
      << "packet_workspace must be contiguous";
  TVM_FFI_ICHECK(tensor.size(0) >= kGeneratedSmallBhPacketRows)
      << "packet_workspace must contain at least one 123-row packet";
  uint64_t global_dim[2] = {
      static_cast<uint64_t>(kGeneratedSmallBhPacketElements),
      static_cast<uint64_t>(tensor.size(0))};
  uint64_t global_strides[1] = {
      static_cast<uint64_t>(tensor.stride(0) * sizeof(__nv_bfloat16))};
  uint32_t box_dim[2] = {
      static_cast<uint32_t>(kGeneratedSmallBhPacketElements),
      static_cast<uint32_t>(kGeneratedSmallBhPacketRows)};
  uint32_t elem_strides[2] = {1, 1};
  CUtensorMap tensor_map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, tensor.data_ptr(),
      global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for packet_workspace with CUresult="
      << int(result);
  return tensor_map;
}

struct GeneratedSmallBhPacketTensorMapWords {
  uint64_t words[sizeof(CUtensorMap) / sizeof(uint64_t)];
};

static __global__ void PublishGeneratedSmallBhPacketTensorMap(
    uint64_t* destination, GeneratedSmallBhPacketTensorMapWords source) {
  const uint32_t index = threadIdx.x;
  if (index < sizeof(source.words) / sizeof(source.words[0])) {
    destination[index] = source.words[index];
  }
}

struct SmallBhM128Args {
  void *q{}, *q_tma{}, *k{}, *k_tma{}, *v{}, *v_tma{}, *g{}, *g_tma{};
  void *beta{}, *beta_tma{}, *a_log{}, *dt_bias{}, *cu_seqlens{}, *seq_order{};
  void *initial_state{}, *out{}, *out_tma{}, *final_state{};
  int32_t num_heads{}, use_initial_state{}, store_final_state{};
  float scale{}, lower_bound{};
  uint64_t state_indices_addr{}, state_checkpoints_addr{}, checkpoint_cu_starts_addr{};
  int64_t beta_token_stride{}, state_slot_stride{};
  int32_t use_state_indices{}, checkpoint_every_n_tokens{};
  void *packet_workspace_tma{}, *packet_ready{}, *packet_consumed{}, *helper_done{};
  void *initial_state_f32{}, *final_state_f32{};
};

inline void LaunchSmallBhM128(SmallBhM128Args args,
                              const StatePointerSlots& state, dim3 grid,
                              cudaStream_t stream) {
  args.initial_state = state.initial_state;
  args.final_state = state.final_state;
  args.initial_state_f32 = state.initial_state_f32;
  args.final_state_f32 = state.final_state_f32;
  void* kernel_args[] = {
      &args.q, &args.q_tma, &args.k, &args.k_tma, &args.v, &args.v_tma,
      &args.g, &args.g_tma, &args.beta, &args.beta_tma, &args.a_log,
      &args.dt_bias, &args.cu_seqlens, &args.seq_order, &args.initial_state,
      &args.out, &args.out_tma, &args.final_state, &args.num_heads,
      &args.use_initial_state, &args.store_final_state, &args.scale,
      &args.lower_bound, &args.state_indices_addr, &args.state_checkpoints_addr,
      &args.checkpoint_cu_starts_addr, &args.beta_token_stride,
      &args.state_slot_stride, &args.use_state_indices,
      &args.checkpoint_every_n_tokens, &args.packet_workspace_tma,
      &args.packet_ready, &args.packet_consumed, &args.helper_done,
      &args.initial_state_f32, &args.final_state_f32};
  CheckArgumentCount<36>(kernel_args);
  ConfigureAndLaunch(FLASHKDA_GENERATED_KERNEL_ARGUMENT, grid,
                     stream, kernel_args, "generated small-BH M128 launch");
}

inline void RunSmallBhM128(
    TensorView q, TensorView k, TensorView v, TensorView g, TensorView beta,
    TensorView beta_tma, TensorView a_log, TensorView dt_bias,
    TensorView cu_seqlens, TensorView seq_order, TensorView state_indices,
    TensorView initial_state, TensorView out, TensorView final_state,
    TensorView state_checkpoints, TensorView checkpoint_cu_starts,
    TensorView packet_workspace, TensorView packet_ready,
    TensorView packet_consumed, TensorView helper_done,
    TensorView descriptor_storage, int64_t prepare_descriptors,
    int64_t num_heads, int64_t beta_token_stride,
    int64_t state_slot_stride, int64_t use_state_indices,
    int64_t use_initial_state, int64_t store_final_state,
    int64_t checkpoint_every_n_tokens, double scale, double lower_bound,
    int64_t grid_x, int64_t grid_y, int64_t grid_z,
    int64_t cuda_stream) {
  TVM_FFI_ICHECK(q.device().device_type == kDLCUDA);
  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  const auto p = PrepareCommonInputs<
      FLASHKDA_GENERATED_VALUE_ROWS, FLASHKDA_GENERATED_TMA_TILE_TOKENS,
      FLASHKDA_GENERATED_PAIR_PACKED_BETA != 0,
      FLASHKDA_GENERATED_VALUE_TMA_RANK == 4>(
      q,k,v,g,beta,beta_tma,a_log,dt_bias,cu_seqlens,seq_order,state_indices,
      initial_state,out,final_state,descriptor_storage,prepare_descriptors,
      num_heads,beta_token_stride,state_slot_stride,use_state_indices,
      use_initial_state,store_final_state,scale,lower_bound,cuda_stream);
  TVM_FFI_ICHECK(q.ndim() == 4 && q.size(0) == p.num_sequences)
      << "small-BH FlashKDA requires fixed [B, T, H, 128] layout";
  TVM_FFI_ICHECK(q.size(1) >= kGeneratedSmallBhMinSequenceLength)
      << "small-BH FlashKDA requires at least 2048 tokens per fixed sequence";
  const int64_t total_tasks = p.num_sequences * num_heads;
  TVM_FFI_ICHECK(total_tasks > 0 &&
                 total_tasks <= kGeneratedSmallBhMaxTasks &&
                 num_heads <= kGeneratedSmallBhMaxTasks)
      << "small-BH FlashKDA requires 1..8 sequence/head tasks and at most 8 heads";
  TVM_FFI_ICHECK(grid_x == kGeneratedSmallBhGroupSize * total_tasks &&
                 grid_y == 1 && grid_z == 1)
      << "small-BH grid must contain one eight-CTA group per task";
  SmallBhM128Args a{};
  a.q=q.data_ptr();a.q_tma=p.tma.q;a.k=k.data_ptr();a.k_tma=p.tma.k;
  a.v=v.data_ptr();a.v_tma=p.tma.v;a.g=g.data_ptr();a.g_tma=p.tma.g;
  a.beta=beta.data_ptr();a.beta_tma=p.tma.beta;a.a_log=a_log.data_ptr();
  a.dt_bias=dt_bias.data_ptr();a.cu_seqlens=cu_seqlens.data_ptr();
  a.seq_order=seq_order.data_ptr();a.out=out.data_ptr();a.out_tma=p.tma.out;
  a.num_heads=CheckedInt32(num_heads,"num_heads");
  a.use_initial_state=CheckedInt32(use_initial_state,"use_initial_state");
  a.store_final_state=CheckedInt32(store_final_state,"store_final_state");
  a.scale=static_cast<float>(scale);a.lower_bound=static_cast<float>(lower_bound);
  a.state_indices_addr=reinterpret_cast<uintptr_t>(state_indices.data_ptr());
  a.state_checkpoints_addr=reinterpret_cast<uintptr_t>(CheckedBufferPointer(
      state_checkpoints,"state_checkpoints",p.device_id,GeneratedStateDtype(),true));
  a.checkpoint_cu_starts_addr=reinterpret_cast<uintptr_t>(CheckedBufferPointer(
      checkpoint_cu_starts,"checkpoint_cu_starts",p.device_id,dl_int64,true));
  a.beta_token_stride=beta_token_stride;a.state_slot_stride=state_slot_stride;
  a.use_state_indices=CheckedInt32(use_state_indices,"use_state_indices");
  a.checkpoint_every_n_tokens=CheckedInt32(checkpoint_every_n_tokens,"checkpoint_every_n_tokens");
  CheckCudaTensor(packet_workspace, "packet_workspace", p.device_id);
  CheckDtype(packet_workspace, "packet_workspace", dl_bfloat16);
  const int64_t packet_slots = total_tasks * kGeneratedSmallBhRingStages;
  TVM_FFI_ICHECK(
      packet_workspace.ndim() == 2 &&
      packet_workspace.size(0) == packet_slots * kGeneratedSmallBhPacketRows &&
      packet_workspace.size(1) == kGeneratedSmallBhPacketElements)
      << "packet_workspace has the wrong compact-ring shape";
  TVM_FFI_ICHECK(descriptor_storage.numel() >=
                 static_cast<int64_t>(kGeneratedSmallBhDescriptorCount *
                                      sizeof(CUtensorMap)))
      << "small-BH descriptor_storage must hold seven TensorMaps";
  auto* descriptor_bytes =
      static_cast<unsigned char*>(descriptor_storage.data_ptr());
  a.packet_workspace_tma =
      descriptor_bytes + flash_kda::kTensorMapCount * sizeof(CUtensorMap);
  if (prepare_descriptors != 0) {
    const CUtensorMap packet_map =
        EncodeGeneratedSmallBhPacketTma(packet_workspace);
    GeneratedSmallBhPacketTensorMapWords words{};
    std::memcpy(words.words, &packet_map, sizeof(packet_map));
    PublishGeneratedSmallBhPacketTensorMap<<<1, 32, 0, p.stream>>>(
        reinterpret_cast<uint64_t*>(a.packet_workspace_tma), words);
    CheckCuda(cudaGetLastError(),
              "PublishGeneratedSmallBhPacketTensorMap launch");
  }
  a.packet_ready=CheckedBufferPointer(packet_ready,"packet_ready",p.device_id,dl_uint32);
  a.packet_consumed=CheckedBufferPointer(packet_consumed,"packet_consumed",p.device_id,dl_uint32);
  a.helper_done=CheckedBufferPointer(helper_done,"helper_done",p.device_id,dl_uint32);
  TVM_FFI_ICHECK(packet_ready.ndim()==1 && packet_consumed.ndim()==1 && helper_done.ndim()==1);
  TVM_FFI_ICHECK(packet_ready.numel() == packet_slots &&
                 packet_consumed.numel() == packet_slots)
      << "packet generation counters must contain one entry per ring slot";
  TVM_FFI_ICHECK(helper_done.numel() == total_tasks)
      << "helper_done must contain one entry per sequence/head task";
  LaunchSmallBhM128(a,p.state,CheckedGrid(grid_x,grid_y,grid_z),p.stream);
}

}  // namespace flashinfer::flash_kda_generated

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    run, flashinfer::flash_kda_generated::RunSmallBhM128);
