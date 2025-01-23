/*
 * Copyright (c) 2023-2025 by FlashInfer team.
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
#include <flashinfer/pos_enc.cuh>
#include <optional>

#include "pytorch_extension_utils.h"
#include "single_decode_config.inc"

namespace flashinfer {

template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant,
          typename Params>
cudaError_t SingleDecodeWithKVCacheDispatched(Params params, typename Params::DTypeO* tmp,
                                              cudaStream_t stream);
}  // namespace flashinfer

using namespace flashinfer;

void single_decode_with_kv_cache(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor tmp,
                                 at::Tensor o, unsigned int layout,
                                 int window_left ADDITIONAL_FUNC_PARAMS, int64_t cuda_stream) {
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

  auto q_scalar_type = q.scalar_type();
  auto kv_scalar_type = k.scalar_type();

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, HEAD_DIM, POS_ENCODING_MODE, USE_SLIDING_WINDOW,
      USE_LOGITS_SOFT_CAP, AttentionVariant, Params, [&] {
        Params params;

        params.q = static_cast<DTypeQ*>(q.data_ptr());
        params.k = static_cast<DTypeKV*>(k.data_ptr());
        params.v = static_cast<DTypeKV*>(v.data_ptr());
        params.o = static_cast<DTypeO*>(o.data_ptr());
        params.lse = nullptr;
        params.kv_len = kv_len;
        params.num_qo_heads = num_qo_heads;
        params.num_kv_heads = num_kv_heads;
        params.q_stride_n = num_qo_heads * head_dim;
        params.q_stride_h = head_dim;
        params.kv_stride_n = (kv_layout == QKVLayout::kNHD) ? num_kv_heads * head_dim : head_dim;
        params.kv_stride_h = (kv_layout == QKVLayout::kNHD) ? head_dim : kv_len * head_dim;
        params.head_dim = head_dim;
        params.window_left = window_left;
        params.kv_chunk_size = 0;

        ADDITIONAL_PARAMS_SETTER

        cudaError_t status =
            flashinfer::SingleDecodeWithKVCacheDispatched<HEAD_DIM, POS_ENCODING_MODE,
                                                          AttentionVariant>(
                params, static_cast<DTypeO*>(tmp.data_ptr()), stream);
        TORCH_CHECK(status == cudaSuccess, "SingleDecodeWithKVCache kernel launch failed, error: " +
                                               std::string(cudaGetErrorString(status)));
        return true;
      });
}
