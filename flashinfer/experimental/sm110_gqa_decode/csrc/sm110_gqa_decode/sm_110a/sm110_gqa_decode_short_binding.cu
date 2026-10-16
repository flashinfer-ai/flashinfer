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
// Direct-source tvm-ffi launcher for the SM110 GQA decode kernel 'kernel_sm110_gqa_decode_short'
// (256 threads, 50176 B dynamic shared memory), linked from the kernel
// translation unit next to this file. Argument checks, the device guard and the
// stream come from tvm_ffi_utils.h; the TMA encoders and the one-time dynamic
// shared memory opt-in from the family's shared launch header.

#include "../../sm110_gqa_decode_launch.cuh"

extern "C" __global__ void kernel_sm110_gqa_decode_short(
    const __grid_constant__ sm110_gqa_decode::TensorMap64 Q,
    const __grid_constant__ sm110_gqa_decode::TensorMap64 K,
    const __grid_constant__ sm110_gqa_decode::TensorMap64 V, __half* __restrict__ O,
    int* __restrict__ sequence_lengths, float softmax_scale_log2);

namespace {

using tvm::ffi::TensorView;

void Run(TensorView Q, TensorView K, TensorView V, TensorView O,
         TensorView sequence_lengths, double softmax_scale_log2, int64_t grid_x, int64_t grid_y,
         int64_t grid_z) {
  tvm::ffi::CUDADeviceGuard device_guard(Q.device().device_id);
  check_cuda_tensor(Q, "Q");
  check_dtype(Q, dl_float16, "Q");
  check_cuda_tensor(K, "K");
  check_dtype(K, dl_float16, "K");
  check_cuda_tensor(V, "V");
  check_dtype(V, dl_float16, "V");
  check_cuda_tensor(O, "O");
  check_dtype(O, dl_float16, "O");
  check_contiguous(O, "O");
  check_cuda_tensor(sequence_lengths, "sequence_lengths");
  check_dtype(sequence_lengths, dl_int32, "sequence_lengths");
  check_contiguous(sequence_lengths, "sequence_lengths");
  check_same_device(K, Q, "K", "Q");
  check_same_device(V, Q, "V", "Q");
  check_same_device(O, Q, "O", "Q");
  check_same_device(sequence_lengths, Q, "sequence_lengths", "Q");
  sm110_gqa_decode::CheckGrid(grid_x, grid_y, grid_z);

  cudaStream_t stream = get_stream(Q.device());
  CUtensorMap p_Q = sm110_gqa_decode::EncodeQ(Q);
  CUtensorMap p_K = sm110_gqa_decode::EncodeKV(K, "K", 64u);
  CUtensorMap p_V = sm110_gqa_decode::EncodeKV(V, "V", 64u);
  __half* p_O = static_cast<__half*>(O.data_ptr());
  int* p_sequence_lengths = static_cast<int*>(sequence_lengths.data_ptr());
  float v_softmax_scale_log2 = static_cast<float>(softmax_scale_log2);
  void* kargs[] = {&p_Q, &p_K, &p_V, &p_O, &p_sequence_lengths, &v_softmax_scale_log2};

  static const bool smem_ready = sm110_gqa_decode::SetMaxDynamicSharedMemory(
      reinterpret_cast<const void*>(kernel_sm110_gqa_decode_short), 50176);
  (void)smem_ready;
  sm110_gqa_decode::Launch(reinterpret_cast<const void*>(kernel_sm110_gqa_decode_short), "kernel_sm110_gqa_decode_short",
                           dim3(static_cast<uint32_t>(grid_x), static_cast<uint32_t>(grid_y),
                                static_cast<uint32_t>(grid_z)),
                           dim3(256u, 1u, 1u), 50176u, stream, kargs);
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_short, Run);
