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

#include "flashinfer_ops_decode.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

torch::Tensor single_decode_with_kv_cache(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                          torch::Tensor tmp, unsigned int pos_encoding_mode,
                                          unsigned int layout, int window_left,
                                          float logits_soft_cap, float sm_scale, float rope_scale,
                                          float rope_theta) {
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  CHECK_INPUT(tmp);
  auto device = q.device();
  CHECK_EQ(k.device(), device);
  CHECK_EQ(v.device(), device);
  CHECK_EQ(tmp.device(), device);
  CHECK_DIM(2, q);
  CHECK_DIM(3, k);
  CHECK_DIM(3, v);
  CHECK_SHAPE(k, v);
  CHECK_EQ(q.size(1), k.size(2));
  CHECK_EQ(v.scalar_type(), k.scalar_type());
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
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  auto o = torch::empty_like(q);

  TORCH_CHECK(logits_soft_cap >= 0.f, "logits_soft_cap must be non-negative");
  const LogitsPostHook logits_post_hook =
      logits_soft_cap > 0.f ? LogitsPostHook::kSoftCap : LogitsPostHook::kNone;

  auto q_scalar_type = q.scalar_type();
  auto kv_scalar_type = k.scalar_type();

  if (q_scalar_type == kv_scalar_type) {
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q_scalar_type, qkv_type, [&] {
      return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
        return DISPATCH_logits_post_hook(logits_post_hook, LOGITS_POST_HOOK, [&] {
          return DISPATCH_pos_encoding_mode(
              PosEncodingMode(pos_encoding_mode), POS_ENCODING_MODE, [&] {
                cudaError_t status = SingleDecodeWithKVCacheDispatched<HEAD_DIM, LOGITS_POST_HOOK,
                                                                       POS_ENCODING_MODE>(
                    static_cast<qkv_type*>(q.data_ptr()), static_cast<qkv_type*>(k.data_ptr()),
                    static_cast<qkv_type*>(v.data_ptr()), static_cast<qkv_type*>(o.data_ptr()),
                    static_cast<qkv_type*>(tmp.data_ptr()), num_qo_heads, num_kv_heads, kv_len,
                    kv_layout, window_left, logits_soft_cap, sm_scale, rope_scale, rope_theta,
                    torch_current_stream);
                TORCH_CHECK(status == cudaSuccess,
                            "SingleDecodeWithKVCache kernel launch failed, error: " +
                                std::string(cudaGetErrorString(status)));
                return true;
              });
        });
      });
    });
  } else {
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q_scalar_type, q_type, [&] {
      return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8(kv_scalar_type, kv_type, [&] {
        return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
          return DISPATCH_logits_post_hook(logits_post_hook, LOGITS_POST_HOOK, [&] {
            return DISPATCH_pos_encoding_mode(
                PosEncodingMode(pos_encoding_mode), POS_ENCODING_MODE, [&] {
                  cudaError_t status = SingleDecodeWithKVCacheDispatched<HEAD_DIM, LOGITS_POST_HOOK,
                                                                         POS_ENCODING_MODE>(
                      static_cast<q_type*>(q.data_ptr()), static_cast<kv_type*>(k.data_ptr()),
                      static_cast<kv_type*>(v.data_ptr()), static_cast<q_type*>(o.data_ptr()),
                      static_cast<q_type*>(tmp.data_ptr()), num_qo_heads, num_kv_heads, kv_len,
                      kv_layout, window_left, logits_soft_cap, sm_scale, rope_scale, rope_theta,
                      torch_current_stream);
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
