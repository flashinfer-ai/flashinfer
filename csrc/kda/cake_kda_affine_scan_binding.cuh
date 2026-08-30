/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#include "cake_kda_affine_binding_common.cuh"

static_assert(FLASHINFER_CAKE_KDA_AFFINE_ROLE ==
                  FLASHINFER_CAKE_KDA_AFFINE_ROLE_SCAN,
              "Cake KDA affine scan binding requires the scan role");
static_assert(FLASHINFER_CAKE_KDA_AFFINE_THREADS == 128,
              "sealed Cake KDA affine scan uses 128 threads");
static_assert(FLASHINFER_CAKE_KDA_AFFINE_SMEM_BYTES == 66560,
              "sealed Cake KDA affine scan uses 66560 bytes of shared memory");
static_assert(FLASHINFER_CAKE_KDA_AFFINE_USE_PDL == 1,
              "sealed Cake KDA affine scan requires PDL");

namespace flashinfer {
namespace cake_kda {

struct CakeKDAAffineScanKernelArgs {
  void* split_state{};
  void* map_state_bf16{};
  void* carry{};
  int32_t num_heads{};
  int32_t num_parts{};
};

inline void RunCakeKDAAffineScan(
    TensorView split_state, TensorView map_state_bf16, TensorView carry,
    int64_t num_heads, int64_t num_parts, int64_t grid_x,
    int64_t grid_y, int64_t grid_z, int64_t cuda_stream) {
  TVM_FFI_ICHECK(split_state.device().device_type == kDLCUDA)
      << "split_state must be a CUDA tensor";
  const int32_t device_id = split_state.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckCakeKDATarget(device_id);
  TVM_FFI_ICHECK(num_heads > 0 && num_heads <= 32)
      << "Cake KDA affine scan requires 1 <= num_heads <= 32";
  TVM_FFI_ICHECK(num_parts >= 2)
      << "Cake KDA affine scan requires at least two parts";
  CakeKDAAffineCheckCompactState(split_state, "split_state", device_id,
                                 dl_float32, num_parts, num_heads);
  CakeKDAAffineCheckCompactState(map_state_bf16, "map_state_bf16",
                                 device_id, dl_bfloat16, num_parts - 1,
                                 num_heads);
  CakeKDAAffineCheckCompactState(carry, "carry", device_id, dl_float32,
                                 num_parts - 1, num_heads);
  CheckNoOverlap(split_state, "split_state", map_state_bf16,
                 "map_state_bf16");
  CheckNoOverlap(split_state, "split_state", carry, "carry");
  CheckNoOverlap(map_state_bf16, "map_state_bf16", carry, "carry");
  TVM_FFI_ICHECK(grid_x == 32 * num_heads && grid_y == 1 && grid_z == 1)
      << "Cake KDA affine scan grid must be [32 * H, 1, 1]";

  CakeKDAAffineScanKernelArgs args{};
  args.split_state = split_state.data_ptr();
  args.map_state_bf16 = map_state_bf16.data_ptr();
  args.carry = carry.data_ptr();
  args.num_heads = CakeKDAAffineCheckedInt32(num_heads, "num_heads");
  args.num_parts = CakeKDAAffineCheckedInt32(num_parts, "num_parts");
  void* kernel_args[] = {&args.split_state, &args.map_state_bf16,
                         &args.carry, &args.num_heads, &args.num_parts};
  CakeKDAAffineCheckArgumentCount<5>(kernel_args);
  CakeKDAAffineConfigureAndLaunch(
      reinterpret_cast<const void*>(FLASHINFER_CAKE_KDA_AFFINE_KERNEL),
      CakeKDAAffineCheckedGrid(grid_x, grid_y, grid_z), device_id,
      CakeKDAAffineCheckedStream(cuda_stream), kernel_args,
      "Cake KDA affine scan launch");
}

}  // namespace cake_kda
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    run, flashinfer::cake_kda::RunCakeKDAAffineScan);
