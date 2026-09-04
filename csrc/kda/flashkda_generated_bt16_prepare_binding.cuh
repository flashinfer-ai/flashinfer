/* Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0. */
#pragma once

#include "flashkda_generated_binding_common.cuh"

namespace flashinfer::flash_kda_generated {

static_assert(FLASHKDA_GENERATED_STATE_MODE == FLASHKDA_GENERATED_STATE_NONE,
              "BT16 prepare must not acquire serving-state pointers");

struct Bt16PrepareArgs {
  void *q{}, *q_tma{}, *k{}, *k_tma{}, *raw_gate{}, *raw_gate_tma{};
  void *beta_logits{}, *beta_logits_tma{}, *a_log{}, *dt_bias{}, *cu_seqlens{};
  void *cu_chunks{}, *chunk_to_seq{}, *ws_qd{}, *ws_qd_tma{}, *ws_kd{};
  void *ws_kd_tma{}, *ws_w{}, *ws_w_tma{}, *ws_qk_t{}, *ws_diag{};
  int32_t total_chunks{}, num_heads{};
  float gate_lower_bound{};
};

inline void LaunchBt16Prepare(Bt16PrepareArgs args, dim3 grid,
                              cudaStream_t stream) {
  void* kernel_args[] = {
      &args.q, &args.q_tma, &args.k, &args.k_tma, &args.raw_gate,
      &args.raw_gate_tma, &args.beta_logits, &args.beta_logits_tma,
      &args.a_log, &args.dt_bias, &args.cu_seqlens, &args.cu_chunks,
      &args.chunk_to_seq, &args.ws_qd, &args.ws_qd_tma, &args.ws_kd,
      &args.ws_kd_tma, &args.ws_w, &args.ws_w_tma, &args.ws_qk_t,
      &args.ws_diag, &args.total_chunks, &args.num_heads,
      &args.gate_lower_bound};
  CheckArgumentCount<24>(kernel_args);
  ConfigureAndLaunch(FLASHKDA_GENERATED_KERNEL_ARGUMENT, grid,
                     stream, kernel_args, "generated BT16 prepare launch");
}

inline void RunBt16Prepare(
    TensorView q,TensorView k,TensorView raw_gate,TensorView beta_logits,
    TensorView a_log,TensorView dt_bias,
    TensorView cu_seqlens,TensorView cu_chunks,TensorView chunk_to_seq,
    TensorView ws_qd,TensorView ws_kd,TensorView ws_w,TensorView ws_qk_t,
    TensorView ws_diag,TensorView descriptor_storage,int64_t prepare_descriptors,
    int64_t total_chunks,int64_t num_heads,double gate_lower_bound,
    int64_t grid_x,int64_t grid_y,int64_t grid_z,int64_t cuda_stream) {
  TVM_FFI_ICHECK(q.device().device_type==kDLCUDA);const int32_t dev=q.device().device_id;
  ffi::CUDADeviceGuard guard(dev);CheckFlashKDATarget(dev);
  TVM_FFI_ICHECK(grid_y==1&&grid_z==1);
  flash_kda::CheckBt16PrepareInputs(q,k,raw_gate,beta_logits,a_log,dt_bias,
      cu_seqlens,cu_chunks,chunk_to_seq,ws_qd,ws_kd,ws_w,ws_qk_t,ws_diag,
      descriptor_storage,prepare_descriptors,grid_x,total_chunks,num_heads,
      gate_lower_bound);
  const cudaStream_t stream=CheckedStream(cuda_stream);
  const flash_kda::Bt16PrepareTmaPointers tma=
      flash_kda::EncodeBt16PrepareTma<false>(q,k,raw_gate,beta_logits,ws_qd,
          ws_kd,ws_w,descriptor_storage,prepare_descriptors,stream);
  Bt16PrepareArgs a{};
  a.q=q.data_ptr();a.q_tma=tma.q;a.k=k.data_ptr();a.k_tma=tma.k;
  a.raw_gate=raw_gate.data_ptr();a.raw_gate_tma=tma.raw_gate;
  a.beta_logits=beta_logits.data_ptr();a.beta_logits_tma=tma.beta_logits;
  a.ws_qd=ws_qd.data_ptr();a.ws_qd_tma=tma.ws_qd;
  a.ws_kd=ws_kd.data_ptr();a.ws_kd_tma=tma.ws_kd;
  a.ws_w=ws_w.data_ptr();a.ws_w_tma=tma.ws_w;
  a.a_log=a_log.data_ptr();a.dt_bias=dt_bias.data_ptr();a.cu_seqlens=cu_seqlens.data_ptr();
  a.cu_chunks=cu_chunks.data_ptr();a.chunk_to_seq=chunk_to_seq.data_ptr();a.ws_qk_t=ws_qk_t.data_ptr();
  a.ws_diag=ws_diag.data_ptr();a.total_chunks=CheckedInt32(total_chunks,"total_chunks");
  a.num_heads=CheckedInt32(num_heads,"num_heads");a.gate_lower_bound=static_cast<float>(gate_lower_bound);
  LaunchBt16Prepare(a,CheckedGrid(grid_x,grid_y,grid_z),stream);
}

}  // namespace flashinfer::flash_kda_generated

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run,flashinfer::flash_kda_generated::RunBt16Prepare);
