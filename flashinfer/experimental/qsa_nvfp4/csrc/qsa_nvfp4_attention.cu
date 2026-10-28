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
#include "qsa_nvfp4_attention.cuh"
#include "tvm_ffi_utils.h"

using namespace flashinfer;

// Inputs are validated by the Python API; this only forwards pointers, strides and sizes.
// partial_out / partial_lse are read only when num_splits > 1.
void run(TensorView q, TensorView k_data, TensorView v_data, TensorView k_sf, TensorView v_sf,
         TensorView indices, TensorView block_table, TensorView token_to_req, TensorView gate,
         TensorView out, TensorView partial_out, TensorView partial_lse, double score_scale,
         double output_scale, int64_t num_splits) {
  qsa_nvfp4::Params p;
  p.q = static_cast<const nv_bfloat16*>(q.data_ptr());
  p.k_data = static_cast<const uint8_t*>(k_data.data_ptr());
  p.v_data = static_cast<const uint8_t*>(v_data.data_ptr());
  p.k_sf = static_cast<const uint8_t*>(k_sf.data_ptr());
  p.v_sf = static_cast<const uint8_t*>(v_sf.data_ptr());
  p.indices = static_cast<const int32_t*>(indices.data_ptr());
  p.block_table = static_cast<const int32_t*>(block_table.data_ptr());
  p.token_to_req = static_cast<const int32_t*>(token_to_req.data_ptr());
  p.gate = static_cast<const nv_bfloat16*>(gate.data_ptr());
  p.out = static_cast<nv_bfloat16*>(out.data_ptr());
  p.partial_out = static_cast<float*>(partial_out.data_ptr());
  p.partial_lse = static_cast<float*>(partial_lse.data_ptr());
  p.q_stride_row = q.strides()[0];
  p.q_stride_head = q.strides()[1];
  p.data_stride_block = k_data.strides()[0];
  p.data_stride_token = k_data.strides()[1];
  p.data_stride_head = k_data.strides()[2];
  p.sf_stride_block = k_sf.strides()[0];
  p.sf_stride_token = k_sf.strides()[1];
  p.sf_stride_head = k_sf.strides()[2];
  p.indices_stride_row = indices.strides()[0];
  p.table_stride_req = block_table.strides()[0];
  p.out_stride_row = out.strides()[0];
  p.out_stride_head = out.strides()[1];
  p.score_scale = static_cast<float>(score_scale);
  p.output_scale = static_cast<float>(output_scale);
  p.num_rows = q.size(0);
  p.num_qo_heads = q.size(1);
  p.group_size = q.size(1) / k_data.size(2);
  p.num_blocks = k_data.size(0);
  p.page_size = k_data.size(1);
  p.num_requests = block_table.size(0);
  p.table_width = block_table.size(1);
  p.topk = indices.size(1) - 1;
  p.num_tiles = (p.topk + qsa_nvfp4::kTileTokens - 1) / qsa_nvfp4::kTileTokens;
  p.num_splits = static_cast<int32_t>(num_splits);

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  const cudaError_t status = qsa_nvfp4::SparsePagedAttention(p, get_stream(q.device()));
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "QSA NVFP4 sparse paged attention failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
