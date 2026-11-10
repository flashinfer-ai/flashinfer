/* Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0. */
#pragma once

#include "flashkda_generated_binding_common.cuh"

namespace flashinfer::flash_kda_generated {

struct Bt16ChainArgs {
  void *ws_qd{}, *ws_qd_tma{}, *ws_kd{}, *ws_kd_tma{}, *ws_w{}, *ws_w_tma{};
  void *ws_qk{}, *ws_qk_tma{}, *ws_diag{}, *ws_diag_tma{}, *v{}, *v_tma{};
  void *cu_seqlens{}, *cu_chunks{}, *seq_order{}, *initial_state{}, *out{};
  void *out_tma{}, *final_state{};
  int32_t num_heads{}, use_initial_state{}, store_final_state{};
  float scale{};
  uint64_t state_indices_addr{};
  int64_t state_slot_stride{};
  int32_t use_state_indices{};
  void *initial_state_f32{}, *final_state_f32{};
};

inline void LaunchBt16Chain(Bt16ChainArgs args, const StatePointerSlots& state,
                            dim3 grid, cudaStream_t stream) {
  args.initial_state = state.initial_state;
  args.final_state = state.final_state;
  args.initial_state_f32 = state.initial_state_f32;
  args.final_state_f32 = state.final_state_f32;
  void* kernel_args[] = {
      &args.ws_qd, &args.ws_qd_tma, &args.ws_kd, &args.ws_kd_tma,
      &args.ws_w, &args.ws_w_tma, &args.ws_qk, &args.ws_qk_tma,
      &args.ws_diag, &args.ws_diag_tma, &args.v, &args.v_tma,
      &args.cu_seqlens, &args.cu_chunks, &args.seq_order, &args.initial_state,
      &args.out, &args.out_tma, &args.final_state, &args.num_heads,
      &args.use_initial_state, &args.store_final_state, &args.scale,
      &args.state_indices_addr, &args.state_slot_stride, &args.use_state_indices,
      &args.initial_state_f32, &args.final_state_f32};
  CheckArgumentCount<28>(kernel_args);
  ConfigureAndLaunch(FLASHKDA_GENERATED_KERNEL_ARGUMENT, grid,
                     stream, kernel_args, "generated BT16 chain launch");
}

inline void RunBt16Chain(
    TensorView ws_qd,TensorView ws_kd,TensorView ws_w,TensorView ws_qk,
    TensorView ws_diag,TensorView v,
    TensorView cu_seqlens,TensorView cu_chunks,TensorView seq_order,
    TensorView state_indices,TensorView initial_state,TensorView out,TensorView final_state,
    TensorView descriptor_storage,int64_t prepare_descriptors,
    int64_t num_heads,int64_t state_slot_stride,
    int64_t use_state_indices,int64_t use_initial_state,int64_t store_final_state,
    double scale,int64_t grid_x,int64_t grid_y,int64_t grid_z,int64_t cuda_stream) {
  TVM_FFI_ICHECK(v.device().device_type==kDLCUDA);const int32_t dev=v.device().device_id;
  ffi::CUDADeviceGuard guard(dev);CheckFlashKDATarget(dev);
  TVM_FFI_ICHECK(std::isfinite(scale)&&std::isfinite(static_cast<float>(scale)));
  for(const auto& x:{std::pair<TensorView*,const char*>{&ws_qd,"ws_qd"},{&ws_kd,"ws_kd"},
      {&ws_w,"ws_w"},{&ws_qk,"ws_qk"},{&v,"v"},{&out,"out"}})
    CheckedBufferPointer(*x.first,x.second,dev,dl_bfloat16);
  CheckedBufferPointer(ws_diag,"ws_diag",dev,dl_float32);
  CheckedBufferPointer(cu_seqlens,"cu_seqlens",dev,dl_int64);
  CheckedBufferPointer(cu_chunks,"cu_chunks",dev,dl_int32);
  CheckedBufferPointer(seq_order,"seq_order",dev,dl_int32);
  const int64_t n=cu_seqlens.numel()-1;
  StatePointerSlots state=ResolveStatePointerSlots(state_indices,initial_state,final_state,dev,n,
      num_heads,state_slot_stride,use_state_indices,use_initial_state,store_final_state);
  TVM_FFI_ICHECK(grid_y==1&&grid_z==1&&grid_x==2*n*num_heads)
      << "BT16 chain grid_x must equal 2 * num_sequences * num_heads";
  flash_kda::CheckBt16DescriptorStorage(descriptor_storage,dev,prepare_descriptors);
  const cudaStream_t stream=CheckedStream(cuda_stream);
  const flash_kda::Bt16ChainTmaPointers tma=flash_kda::EncodeBt16ChainTma(
      ws_qd,ws_kd,ws_w,ws_qk,ws_diag,v,out,descriptor_storage,
      prepare_descriptors,stream);
  Bt16ChainArgs a{};
  a.ws_qd=ws_qd.data_ptr();a.ws_qd_tma=tma.ws_qd;
  a.ws_kd=ws_kd.data_ptr();a.ws_kd_tma=tma.ws_kd;
  a.ws_w=ws_w.data_ptr();a.ws_w_tma=tma.ws_w;
  a.ws_qk=ws_qk.data_ptr();a.ws_qk_tma=tma.ws_qk;
  a.ws_diag=ws_diag.data_ptr();a.ws_diag_tma=tma.ws_diag;
  a.v=v.data_ptr();a.v_tma=tma.v;
  a.cu_seqlens=cu_seqlens.data_ptr();a.cu_chunks=cu_chunks.data_ptr();a.seq_order=seq_order.data_ptr();
  a.out=out.data_ptr();a.out_tma=tma.out;
  a.num_heads=CheckedInt32(num_heads,"num_heads");a.use_initial_state=CheckedInt32(use_initial_state,"use_initial_state");
  a.store_final_state=CheckedInt32(store_final_state,"store_final_state");a.scale=static_cast<float>(scale);
  a.state_indices_addr=reinterpret_cast<uintptr_t>(state_indices.data_ptr());a.state_slot_stride=state_slot_stride;
  a.use_state_indices=CheckedInt32(use_state_indices,"use_state_indices");
  LaunchBt16Chain(a,state,CheckedGrid(grid_x,grid_y,grid_z),stream);
}

}  // namespace flashinfer::flash_kda_generated

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run,flashinfer::flash_kda_generated::RunBt16Chain);
