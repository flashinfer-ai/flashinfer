/*
 * Copyright (c) 2025 by FlashInfer team.
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
#include <flashinfer/attention/scheduler.cuh>
#ifndef FLASHINFER_MLA_SM89_Q_TILE
#define FLASHINFER_MLA_SM89_Q_TILE 64
#endif

#ifndef FLASHINFER_MLA_SM89_CHUNK_QUANTUM
#define FLASHINFER_MLA_SM89_CHUNK_QUANTUM 0
#endif

#include "batch_mla_config.inc"
#include "tvm/ffi/container/array.h"
#include "tvm/ffi/container/tuple.h"
#include "tvm_ffi_utils.h"

using namespace flashinfer;

using tvm::ffi::Array;
using tvm::ffi::Tuple;

Tuple<Array<int64_t>, int64_t> BatchMLAPagedAttentionPlan(
    TensorView float_workspace_buffer, TensorView int_workspace_buffer,
    TensorView page_locked_int_workspace_buffer, TensorView qo_indptr, TensorView kv_indptr,
    TensorView kv_len, int64_t num_heads, int64_t head_dim_o, bool causal) {
  CHECK_INPUT_TYPE(qo_indptr, dl_int32);
  CHECK_INPUT_TYPE(kv_indptr, dl_int32);
  CHECK_INPUT_TYPE(kv_len, dl_int32);

  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * get_element_size(float_workspace_buffer);
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * get_element_size(int_workspace_buffer);

  MLAPlanInfo plan_info;
  size_t staged_int_workspace_bytes = 0;

  int batch_size = kv_len.size(0);

  ffi::CUDADeviceGuard device_guard(float_workspace_buffer.device().device_id);
  const cudaStream_t stream = get_stream(float_workspace_buffer.device());

  int major, minor;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                         float_workspace_buffer.device().device_id);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                         float_workspace_buffer.device().device_id);
  int tile_q = (major == 8 && minor == 9 && std::is_same_v<DTypeKV, __nv_fp8_e4m3> &&
                HEAD_DIM_CKV == 512 && HEAD_DIM_KPE == 64)
                   ? FLASHINFER_MLA_SM89_Q_TILE
                   : 64;
  int chunk_quantum = 0;
  if (major == 8 && minor == 9 && std::is_same_v<DTypeKV, __nv_fp8_e4m3>) {
    chunk_quantum = FLASHINFER_MLA_SM89_CHUNK_QUANTUM;
  }
  cudaError_t status =
      MLAPlan(float_workspace_buffer.data_ptr(), float_workspace_size_in_bytes,
              int_workspace_buffer.data_ptr(), page_locked_int_workspace_buffer.data_ptr(),
              int_workspace_size_in_bytes, plan_info, staged_int_workspace_bytes,
              static_cast<IdType*>(qo_indptr.data_ptr()),
              static_cast<IdType*>(kv_indptr.data_ptr()), static_cast<IdType*>(kv_len.data_ptr()),
              batch_size, num_heads, head_dim_o, causal, stream, tile_q, chunk_quantum);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "Failed to plan MLA, error: " << cudaGetErrorString(status);

  return Tuple<Array<int64_t>, int64_t>(Array(plan_info.ToVector()), staged_int_workspace_bytes);
}
