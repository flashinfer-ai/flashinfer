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

#ifndef FLASHINFER_BT16_CHAIN_KERNEL
#error "FLASHINFER_BT16_CHAIN_KERNEL must name the frozen chain kernel"
#endif
#ifndef FLASHINFER_BT16_CHAIN_SMEM_BYTES
#error "FLASHINFER_BT16_CHAIN_SMEM_BYTES must match the frozen chain schedule"
#endif

namespace flashinfer {
namespace flash_kda {

static_assert(THREADS == 512);
static_assert(SMEM_TOTAL == FLASHINFER_BT16_CHAIN_SMEM_BYTES);

void RunBt16Chain(TensorView ws_qd, TensorView ws_kd, TensorView ws_w, TensorView ws_qk,
                  TensorView ws_diag, TensorView v, TensorView cu_seqlens, TensorView cu_chunks,
                  TensorView seq_order, TensorView initial_state, TensorView out,
                  TensorView final_state, TensorView descriptor_storage,
                  int64_t prepare_descriptors, int64_t num_heads, int64_t use_initial_state,
                  int64_t store_final_state, double scale, int64_t grid_x, int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0) << "cuda_stream must be a non-negative stream handle";
  const int32_t device_id = v.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  const int64_t num_seqs =
      CheckBt16ChainInputs(ws_qd, ws_kd, ws_w, ws_qk, ws_diag, v, cu_seqlens, cu_chunks, seq_order,
                           initial_state, out, final_state, descriptor_storage, prepare_descriptors,
                           num_heads, use_initial_state, store_final_state, scale);

  constexpr int32_t kSmemBytes = FLASHINFER_BT16_CHAIN_SMEM_BYTES;
  CheckDynamicSmemCapacity(device_id, kSmemBytes);
  CheckCuda(cudaFuncSetAttribute(FLASHINFER_BT16_CHAIN_KERNEL,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes),
            "cudaFuncSetAttribute(BT16 chain)");

  const int64_t expected_grid_x = kBt16ValueSplits * num_seqs * num_heads;
  TVM_FFI_ICHECK(grid_x == expected_grid_x && grid_x <= std::numeric_limits<uint32_t>::max())
      << "BT16 chain grid_x must equal 2 * N * H (" << expected_grid_x << "), got " << grid_x;
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  const Bt16ChainTmaPointers tma = EncodeBt16ChainTma(
      ws_qd, ws_kd, ws_w, ws_qk, ws_diag, v, out, descriptor_storage, prepare_descriptors, stream);
  const dim3 grid(static_cast<uint32_t>(grid_x), 1, 1);
  const dim3 block(THREADS, 1, 1);
  FLASHINFER_BT16_CHAIN_KERNEL<<<grid, block, kSmemBytes, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(ws_qd.data_ptr()),
      reinterpret_cast<flashkda_generated_CakeTensorMap const*>(tma.ws_qd),
      reinterpret_cast<__nv_bfloat16*>(ws_kd.data_ptr()),
      reinterpret_cast<flashkda_generated_CakeTensorMap const*>(tma.ws_kd),
      reinterpret_cast<__nv_bfloat16*>(ws_w.data_ptr()),
      reinterpret_cast<flashkda_generated_CakeTensorMap const*>(tma.ws_w),
      reinterpret_cast<__nv_bfloat16*>(ws_qk.data_ptr()),
      reinterpret_cast<flashkda_generated_CakeTensorMap const*>(tma.ws_qk),
      reinterpret_cast<float*>(ws_diag.data_ptr()),
      reinterpret_cast<flashkda_generated_CakeTensorMap const*>(tma.ws_diag),
      reinterpret_cast<__nv_bfloat16*>(v.data_ptr()),
      reinterpret_cast<flashkda_generated_CakeTensorMap const*>(tma.v),
      reinterpret_cast<long long*>(cu_seqlens.data_ptr()),
      reinterpret_cast<int*>(cu_chunks.data_ptr()), reinterpret_cast<int*>(seq_order.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(initial_state.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      reinterpret_cast<flashkda_generated_CakeTensorMap const*>(tma.out),
      reinterpret_cast<__nv_bfloat16*>(final_state.data_ptr()), static_cast<int32_t>(num_heads),
      static_cast<int32_t>(use_initial_state), static_cast<int32_t>(store_final_state),
      static_cast<float>(scale));
  CheckCuda(cudaGetLastError(), "BT16 chain launch");
}

}  // namespace flash_kda
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::flash_kda::RunBt16Chain);

#undef FLASHINFER_BT16_CHAIN_SMEM_BYTES
#undef FLASHINFER_BT16_CHAIN_KERNEL
