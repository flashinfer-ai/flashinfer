/*
 * Copyright (c) 2023 by FlashInfer team.
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
#include <flashinfer/decode_attention_decl.cuh>

#include "flashinfer_ops.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

torch::Tensor single_decode_with_kv_cache(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                          torch::Tensor tmp, unsigned int pos_encoding_mode,
                                          unsigned int layout, float sm_scale, float rope_scale,
                                          float rope_theta) {
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  CHECK_DIM(2, q);
  CHECK_DIM(3, k);
  CHECK_DIM(3, v);
  CHECK_SHAPE(k, v);
  CHECK_EQ(q.size(1), k.size(2));
  unsigned int num_qo_heads = q.size(0);
  unsigned int head_dim = q.size(1);
  unsigned int kv_len, num_kv_heads;
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  if (kv_layout == QKVLayout::kNHD) {
    kv_len = k.size(0);
    num_kv_heads = k.size(1);
  } else {
    num_kv_heads = k.size(0);
    kv_len = k.size(1);
  }
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream();
  auto o = torch::empty_like(
      q, q.options().dtype(is_float8_tensor(q) ? torch::kFloat16 : q.scalar_type()));

  bool success;
  if (is_float8_tensor(q)) {
    success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8(q.scalar_type(), c_type, [&] {
      cudaError_t status = SingleDecodeWithKVCache(
          static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k.data_ptr()),
          static_cast<c_type*>(v.data_ptr()), static_cast<nv_half*>(o.data_ptr()),
          static_cast<nv_half*>(tmp.data_ptr()), num_qo_heads, num_kv_heads, kv_len, head_dim,
          kv_layout, PosEncodingMode(pos_encoding_mode), sm_scale, rope_scale, rope_theta,
          torch_current_stream);
      TORCH_CHECK(status == cudaSuccess, "SingleDecodeWithKVCache kernel launch failed, error: " +
                                             std::string(cudaGetErrorString(status)));
      return true;
    });
  } else {
    success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&] {
      cudaError_t status = SingleDecodeWithKVCache(
          static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k.data_ptr()),
          static_cast<c_type*>(v.data_ptr()), static_cast<c_type*>(o.data_ptr()),
          static_cast<c_type*>(tmp.data_ptr()), num_qo_heads, num_kv_heads, kv_len, head_dim,
          kv_layout, PosEncodingMode(pos_encoding_mode), sm_scale, rope_scale, rope_theta,
          torch_current_stream);
      TORCH_CHECK(status == cudaSuccess, "SingleDecodeWithKVCache kernel launch failed, error: " +
                                             std::string(cudaGetErrorString(status)));
      return true;
    });
  }

  TORCH_CHECK(success, "SingleDecodeWithKVCache kernel launch failed, error: unsupported dtype");
  return o;
}
