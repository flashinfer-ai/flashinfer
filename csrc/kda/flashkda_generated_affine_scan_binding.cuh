/* Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0. */
#pragma once

#include "flashkda_generated_binding_common.cuh"

namespace flashinfer::flash_kda_generated {

static_assert(FLASHKDA_GENERATED_STATE_MODE == FLASHKDA_GENERATED_STATE_NONE,
              "affine scan state is carried by explicit workspace pointers");
static_assert(FLASHKDA_GENERATED_USE_PDL == 1,
              "the audited affine scan requires PDL extended launch");

struct AffineScanArgs {
  void *split_state{}, *map_state_bf16{}, *carry{};
  int32_t num_heads{}, num_parts{};
};

inline void LaunchAffineScan(AffineScanArgs args, dim3 grid,
                             cudaStream_t stream) {
  void* kernel_args[] = {&args.split_state, &args.map_state_bf16, &args.carry,
                         &args.num_heads, &args.num_parts};
  CheckArgumentCount<5>(kernel_args);
  ConfigureAndLaunch(FLASHKDA_GENERATED_KERNEL_ARGUMENT, grid,
                     stream, kernel_args, "generated affine-scan launch");
}

inline void RunAffineScan(TensorView split_state, TensorView map_state_bf16,
                          TensorView carry, int64_t num_heads,
                          int64_t num_parts, int64_t grid_x,
                          int64_t grid_y, int64_t grid_z,
                          int64_t cuda_stream) {
  TVM_FFI_ICHECK(split_state.device().device_type == kDLCUDA);
  const int32_t device_id = split_state.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckFlashKDATarget(device_id);
  AffineScanArgs args{};
  args.split_state = CheckedBufferPointer(split_state, "split_state", device_id,
                                          dl_float32);
  args.map_state_bf16 = CheckedBufferPointer(
      map_state_bf16, "map_state_bf16", device_id, dl_bfloat16);
  args.carry = CheckedBufferPointer(carry, "carry", device_id, dl_float32);
  args.num_heads = CheckedInt32(num_heads, "num_heads");
  args.num_parts = CheckedInt32(num_parts, "num_parts");
  TVM_FFI_ICHECK(args.num_heads > 0 && args.num_parts > 0);
  LaunchAffineScan(args, CheckedGrid(grid_x, grid_y, grid_z),
                   CheckedStream(cuda_stream));
}

}  // namespace flashinfer::flash_kda_generated

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    run, flashinfer::flash_kda_generated::RunAffineScan);
