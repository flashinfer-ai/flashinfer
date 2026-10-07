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
// Direct-source tvm-ffi launcher for the SM110 GQA decode kernel 'kernel_sm110_gqa_decode_n64_kvlast_s10'
// (128 threads, 50304 B dynamic shared memory), linked from the kernel
// translation unit next to this file. Argument checks, the device guard and the
// stream come from tvm_ffi_utils.h; the TMA encoders and the one-time dynamic
// shared memory opt-in from the family's shared launch header.

#include "../sm110_gqa_decode_launch.cuh"

extern "C" __global__ void kernel_sm110_gqa_decode_n64_kvlast_s10(
    const __grid_constant__ sm110_gqa_decode::TensorMap64 Q,
    const __grid_constant__ sm110_gqa_decode::TensorMap64 K,
    const __grid_constant__ sm110_gqa_decode::TensorMap64 V, float* __restrict__ partial_O, float* __restrict__ partial_max, float* __restrict__ partial_sum, unsigned int* __restrict__ completed, __half* __restrict__ O,
    int* __restrict__ sequence_lengths, float softmax_scale_log2);

namespace {

using tvm::ffi::TensorView;

void Run(TensorView Q, TensorView K, TensorView V, TensorView partial_O, TensorView partial_max, TensorView partial_sum, TensorView completed, TensorView O,
         TensorView sequence_lengths, double softmax_scale_log2, int64_t grid_x, int64_t grid_y,
         int64_t grid_z) {
  tvm::ffi::CUDADeviceGuard device_guard(Q.device().device_id);
  check_cuda_tensor(Q, "Q");
  check_dtype(Q, dl_float16, "Q");
  check_cuda_tensor(K, "K");
  check_dtype(K, dl_float16, "K");
  check_cuda_tensor(V, "V");
  check_dtype(V, dl_float16, "V");
  check_cuda_tensor(partial_O, "partial_O");
  check_dtype(partial_O, dl_float32, "partial_O");
  check_contiguous(partial_O, "partial_O");
  check_cuda_tensor(partial_max, "partial_max");
  check_dtype(partial_max, dl_float32, "partial_max");
  check_contiguous(partial_max, "partial_max");
  check_cuda_tensor(partial_sum, "partial_sum");
  check_dtype(partial_sum, dl_float32, "partial_sum");
  check_contiguous(partial_sum, "partial_sum");
  check_cuda_tensor(completed, "completed");
  check_dtype(completed, dl_uint32, "completed");
  check_contiguous(completed, "completed");
  check_cuda_tensor(O, "O");
  check_dtype(O, dl_float16, "O");
  check_contiguous(O, "O");
  check_cuda_tensor(sequence_lengths, "sequence_lengths");
  check_dtype(sequence_lengths, dl_int32, "sequence_lengths");
  check_contiguous(sequence_lengths, "sequence_lengths");
  check_same_device(K, Q, "K", "Q");
  check_same_device(V, Q, "V", "Q");
  check_same_device(partial_O, Q, "partial_O", "Q");
  check_same_device(partial_max, Q, "partial_max", "Q");
  check_same_device(partial_sum, Q, "partial_sum", "Q");
  check_same_device(completed, Q, "completed", "Q");
  check_same_device(O, Q, "O", "Q");
  check_same_device(sequence_lengths, Q, "sequence_lengths", "Q");
  sm110_gqa_decode::CheckGrid(grid_x, grid_y, grid_z);

  cudaStream_t stream = get_stream(Q.device());
  CUtensorMap p_Q = sm110_gqa_decode::EncodeQ(Q);
  CUtensorMap p_K = sm110_gqa_decode::EncodeKV(K, "K", 64u);
  CUtensorMap p_V = sm110_gqa_decode::EncodeKV(V, "V", 64u);
  float* p_partial_O = static_cast<float*>(partial_O.data_ptr());
  float* p_partial_max = static_cast<float*>(partial_max.data_ptr());
  float* p_partial_sum = static_cast<float*>(partial_sum.data_ptr());
  unsigned int* p_completed = static_cast<unsigned int*>(completed.data_ptr());
  __half* p_O = static_cast<__half*>(O.data_ptr());
  int* p_sequence_lengths = static_cast<int*>(sequence_lengths.data_ptr());
  float v_softmax_scale_log2 = static_cast<float>(softmax_scale_log2);
  void* kargs[] = {&p_Q, &p_K, &p_V, &p_partial_O, &p_partial_max, &p_partial_sum, &p_completed, &p_O, &p_sequence_lengths, &v_softmax_scale_log2};

  static const bool smem_ready = sm110_gqa_decode::SetMaxDynamicSharedMemory(
      reinterpret_cast<const void*>(kernel_sm110_gqa_decode_n64_kvlast_s10), 50304);
  (void)smem_ready;
  sm110_gqa_decode::Launch(reinterpret_cast<const void*>(kernel_sm110_gqa_decode_n64_kvlast_s10), "kernel_sm110_gqa_decode_n64_kvlast_s10",
                           dim3(static_cast<uint32_t>(grid_x), static_cast<uint32_t>(grid_y),
                                static_cast<uint32_t>(grid_z)),
                           dim3(128u, 1u, 1u), 50304u, stream, kargs);
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, Run);
