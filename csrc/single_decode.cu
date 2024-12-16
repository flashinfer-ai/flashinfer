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
#include <c10/cuda/CUDAGuard.h>

#include <flashinfer/attention/decode_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/pos_enc.cuh>
#include <optional>

#include "aot_extension_utils.h"

namespace flashinfer {

template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant>
cudaError_t SingleDecodeWithKVCacheDispatched(typename AttentionVariant::ParamsT params,
                                              typename AttentionVariant::DTypeO* tmp,
                                              cudaStream_t stream);
}  // namespace flashinfer

using namespace flashinfer;

void single_decode_with_kv_cache(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor tmp,
                                 std::optional<at::Tensor> alibi_slopes, at::Tensor o,
                                 unsigned int layout, int window_left, float logits_soft_cap,
                                 float sm_scale, float rope_scale, float rope_theta,
                                 int64_t cuda_stream) {
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

  TORCH_CHECK(logits_soft_cap >= 0.f, "logits_soft_cap must be non-negative");

  auto q_scalar_type = q.scalar_type();
  auto kv_scalar_type = k.scalar_type();

  constexpr auto POS_ENCODING_MODE = PosEncodingMode::kNone;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  DISPATCH_PYTORCH_QKV_DTYPE_TO_CTYPE(q_scalar_type, kv_scalar_type, q_type, kv_type, [&] {
    using DTypeQ = q_type;
    using DTypeKV = kv_type;
    using DTypeO = DTypeQ;
    return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
      return DISPATCH_BOOL(logits_soft_cap > 0, USE_LOGITS_SOFT_CAP, [&] {
        using ParamsT = SingleDecodeParams<DTypeQ, DTypeKV, DTypeO>;
        using AttentionVariant =
            ComposedAttention<ParamsT,
                              get_variant_code(/*use_custom_mask=*/false,
                                               /*use_sliding_window=*/true, USE_LOGITS_SOFT_CAP,
                                               /*use_alibi=*/false)>;
        ParamsT params(static_cast<DTypeQ*>(q.data_ptr()), static_cast<DTypeKV*>(k.data_ptr()),
                       static_cast<DTypeKV*>(v.data_ptr()), static_cast<DTypeO*>(o.data_ptr()),
                       /*alibi_slopes=*/nullptr, kv_len, num_qo_heads, num_kv_heads, kv_layout,
                       head_dim, window_left, logits_soft_cap, sm_scale, rope_scale, rope_theta);
        cudaError_t status =
            flashinfer::SingleDecodeWithKVCacheDispatched<HEAD_DIM, POS_ENCODING_MODE,
                                                          AttentionVariant>(
                params, static_cast<DTypeO*>(tmp.data_ptr()), stream);
        TORCH_CHECK(status == cudaSuccess, "SingleDecodeWithKVCache kernel launch failed, error: " +
                                               std::string(cudaGetErrorString(status)));
        return true;
      });
    });
  });
}
