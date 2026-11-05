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
using qsa_nvfp4::KvFormat;
using tvm::ffi::Optional;

template <typename Fn>
cudaError_t dispatch_format(int64_t kv_format, Fn&& fn) {
  switch (kv_format) {
    case 0:
      return fn(std::integral_constant<KvFormat, KvFormat::kNVFP4>{});
    case 1:
      return fn(std::integral_constant<KvFormat, KvFormat::kFP8>{});
    case 2:
      return fn(std::integral_constant<KvFormat, KvFormat::kBF16>{});
    default:
      return cudaErrorInvalidValue;
  }
}

// Inputs are validated by the Python API; this only forwards pointers, strides and sizes.
// kv_format is 0 (NVFP4, with k_sf / v_sf), 1 (FP8-e4m3) or 2 (BF16).
void run(TensorView q, TensorView k_data, TensorView v_data, Optional<TensorView> k_sf,
         Optional<TensorView> v_sf, TensorView indices, TensorView block_table,
         TensorView token_to_req, TensorView gate, TensorView out, double score_scale,
         double output_scale, int64_t kv_format, int64_t num_splits) {
  qsa_nvfp4::Params p;
  const int64_t element_bytes = get_element_size(k_data);
  p.q = static_cast<const nv_bfloat16*>(q.data_ptr());
  p.k_data = static_cast<const uint8_t*>(k_data.data_ptr());
  p.v_data = static_cast<const uint8_t*>(v_data.data_ptr());
  p.k_sf = k_sf.has_value() ? static_cast<const uint8_t*>(k_sf.value().data_ptr()) : nullptr;
  p.v_sf = v_sf.has_value() ? static_cast<const uint8_t*>(v_sf.value().data_ptr()) : nullptr;
  p.indices = static_cast<const int32_t*>(indices.data_ptr());
  p.block_table = static_cast<const int32_t*>(block_table.data_ptr());
  p.token_to_req = static_cast<const int32_t*>(token_to_req.data_ptr());
  p.gate = static_cast<const nv_bfloat16*>(gate.data_ptr());
  p.out = static_cast<nv_bfloat16*>(out.data_ptr());
  p.q_stride_row = q.strides()[0];
  p.q_stride_head = q.strides()[1];
  p.data_stride_block = k_data.strides()[0] * element_bytes;
  p.data_stride_token = k_data.strides()[1] * element_bytes;
  p.data_stride_head = k_data.strides()[2] * element_bytes;
  if (k_sf.has_value()) {
    p.sf_stride_block = k_sf.value().strides()[0];
    p.sf_stride_token = k_sf.value().strides()[1];
    p.sf_stride_head = k_sf.value().strides()[2];
  }
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
  p.num_splits = static_cast<int32_t>(num_splits);

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  // Split partials, merged into out on the same stream; a single split writes out directly.
  Optional<ffi::Tensor> partial_out, partial_lse;
  p.partial_out = p.partial_lse = nullptr;
  if (num_splits > 1) {
    partial_out =
        alloc_tensor({num_splits, q.size(0), q.size(1), q.size(2)}, dl_float32, q.device());
    partial_lse = alloc_tensor({num_splits, q.size(0), q.size(1)}, dl_float32, q.device());
    p.partial_out = static_cast<float*>(partial_out.value().data_ptr());
    p.partial_lse = static_cast<float*>(partial_lse.value().data_ptr());
  }
  const cudaError_t status = dispatch_format(kv_format, [&](auto format) {
    constexpr uint32_t kTileTokens = qsa_nvfp4::KvTraits<decltype(format)::value>::kTileTokens;
    p.num_tiles = (p.topk + kTileTokens - 1) / kTileTokens;
    return qsa_nvfp4::SparsePagedAttention<decltype(format)::value>(p, get_stream(q.device()));
  });
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "QSA sparse paged attention failed: " << cudaGetErrorString(status);
}

// Resident attention CTAs per SM of the current device, for sizing splits.
int64_t ctas_per_sm(int64_t kv_format) {
  int num_ctas = 0;
  const cudaError_t status = dispatch_format(kv_format, [&](auto format) {
    return qsa_nvfp4::CtasPerSm<decltype(format)::value>(&num_ctas);
  });
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "QSA sparse paged attention occupancy failed: " << cudaGetErrorString(status);
  return num_ctas;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ctas_per_sm, ctas_per_sm);
