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
  CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads);
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream();
  auto o = torch::empty_like(
      q, q.options().dtype(is_float8_tensor(q) ? torch::kFloat16 : q.scalar_type()));

  if (is_float8_tensor(q)) {
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8(q.scalar_type(), c_type, [&] {
      return DISPATCH_group_size(num_qo_heads / num_kv_heads, GROUP_SIZE, [&] {
        return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
          return DISPATCH_kv_layout(kv_layout, KV_LAYOUT, [&] {
            return DISPATCH_pos_encoding_mode(
                PosEncodingMode(pos_encoding_mode), POS_ENCODING_MODE, [&] {
                  cudaError_t status =
                      SingleDecodeWithKVCacheDispatched<GROUP_SIZE, HEAD_DIM, KV_LAYOUT,
                                                        POS_ENCODING_MODE>(
                          static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k.data_ptr()),
                          static_cast<c_type*>(v.data_ptr()), static_cast<nv_half*>(o.data_ptr()),
                          static_cast<nv_half*>(tmp.data_ptr()), num_kv_heads, kv_len, sm_scale,
                          rope_scale, rope_theta, torch_current_stream);
                  TORCH_CHECK(status == cudaSuccess,
                              "SingleDecodeWithKVCache kernel launch failed, error: " +
                                  std::string(cudaGetErrorString(status)));
                  return true;
                });
          });
        });
      });
    });
  } else {
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&] {
      return DISPATCH_group_size(num_qo_heads / num_kv_heads, GROUP_SIZE, [&] {
        return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
          return DISPATCH_kv_layout(kv_layout, KV_LAYOUT, [&] {
            return DISPATCH_pos_encoding_mode(
                PosEncodingMode(pos_encoding_mode), POS_ENCODING_MODE, [&] {
                  cudaError_t status =
                      SingleDecodeWithKVCacheDispatched<GROUP_SIZE, HEAD_DIM, KV_LAYOUT,
                                                        POS_ENCODING_MODE>(
                          static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k.data_ptr()),
                          static_cast<c_type*>(v.data_ptr()), static_cast<c_type*>(o.data_ptr()),
                          static_cast<c_type*>(tmp.data_ptr()), num_kv_heads, kv_len, sm_scale,
                          rope_scale, rope_theta, torch_current_stream);
                  TORCH_CHECK(status == cudaSuccess,
                              "SingleDecodeWithKVCache kernel launch failed, error: " +
                                  std::string(cudaGetErrorString(status)));
                  return true;
                });
          });
        });
      });
    });
  }

  return o;
}
