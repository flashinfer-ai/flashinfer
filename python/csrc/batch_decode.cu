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
#include <flashinfer.cuh>

#include "flashinfer_ops.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

torch::Tensor batch_decode_with_padded_kv_cache(torch::Tensor q, torch::Tensor k_padded,
                                                torch::Tensor v_padded, unsigned int layout,
                                                unsigned int rotary_mode, float sm_scale,
                                                float rope_scale, float rope_theta) {
  CHECK_INPUT(q);
  CHECK_INPUT(k_padded);
  CHECK_INPUT(v_padded);
  CHECK_DIM(3, q);
  CHECK_DIM(4, k_padded);
  CHECK_DIM(4, v_padded);
  CHECK_SHAPE(k_padded, v_padded);
  CHECK_EQ(q.size(0), k_padded.size(0));
  CHECK_EQ(q.size(2), k_padded.size(3));
  unsigned int batch_size = q.size(0);
  unsigned int num_qo_heads = q.size(1);
  unsigned int head_dim = q.size(2);
  unsigned int padded_kv_len, num_kv_heads;
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  if (kv_layout == QKVLayout::kNHD) {
    padded_kv_len = k_padded.size(1);
    num_kv_heads = k_padded.size(2);
  } else {
    padded_kv_len = k_padded.size(2);
    num_kv_heads = k_padded.size(1);
  }

  auto o = torch::empty_like(q, q.options());

  bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&] {
    cudaError_t status = BatchDecodeWithPaddedKVCache<c_type, c_type>(
        static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k_padded.data_ptr()),
        static_cast<c_type*>(v_padded.data_ptr()), static_cast<c_type*>(o.data_ptr()),
        /*tmp=*/nullptr,
        /*lse=*/nullptr, batch_size, padded_kv_len, num_qo_heads, num_kv_heads, head_dim, kv_layout,
        RotaryMode(rotary_mode), rope_scale, rope_theta);
    TORCH_CHECK(status == cudaSuccess, "BatchDecodeWithPaddedKVCache failed with error code ",
                status);
    return true;
  });
  TORCH_CHECK(success, "BatchDecodeWithPaddedKVCache kernel launch failed: supported data type");
  return o;
}

std::vector<torch::Tensor> batch_decode_with_padded_kv_cache_return_lse(
    torch::Tensor q, torch::Tensor k_padded, torch::Tensor v_padded, unsigned int layout,
    unsigned int rotary_mode, float sm_scale, float rope_scale, float rope_theta) {
  CHECK_INPUT(q);
  CHECK_INPUT(k_padded);
  CHECK_INPUT(v_padded);
  CHECK_DIM(3, q);
  CHECK_DIM(4, k_padded);
  CHECK_DIM(4, v_padded);
  CHECK_SHAPE(k_padded, v_padded);
  CHECK_EQ(q.size(0), k_padded.size(0));
  CHECK_EQ(q.size(2), k_padded.size(3));
  unsigned int batch_size = q.size(0);
  unsigned int num_qo_heads = q.size(1);
  unsigned int head_dim = q.size(2);
  unsigned int padded_kv_len, num_kv_heads;
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  if (kv_layout == QKVLayout::kNHD) {
    padded_kv_len = k_padded.size(1);
    num_kv_heads = k_padded.size(2);
  } else {
    padded_kv_len = k_padded.size(2);
    num_kv_heads = k_padded.size(1);
  }

  auto o = torch::empty_like(q, q.options());
  auto lse = torch::empty({batch_size, num_qo_heads}, q.options().dtype(torch::kFloat32));

  bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&] {
    cudaError_t status = BatchDecodeWithPaddedKVCache<c_type, c_type>(
        static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k_padded.data_ptr()),
        static_cast<c_type*>(v_padded.data_ptr()), static_cast<c_type*>(o.data_ptr()),
        /*tmp=*/nullptr,
        /*lse=*/static_cast<float*>(lse.data_ptr()), batch_size, padded_kv_len, num_qo_heads,
        num_kv_heads, head_dim, kv_layout, RotaryMode(rotary_mode), rope_scale, rope_theta);
    TORCH_CHECK(status == cudaSuccess, "BatchDecodeWithPaddedKVCache failed with error code ",
                status);
    return true;
  });
  TORCH_CHECK(success, "BatchDecodeWithPaddedKVCache kernel launch failed: supported data type");
  return {o, lse};
}
