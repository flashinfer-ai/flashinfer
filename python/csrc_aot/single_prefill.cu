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
#include <torch/extension.h>

#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/prefill_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <optional>

#include "pytorch_extension_utils.h"

namespace flashinfer {

template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, bool ALLOW_FP16_QK_REDUCTION,
          MaskMode MASK_MODE, typename AttentionVariant>
cudaError_t SinglePrefillWithKVCacheDispatched(typename AttentionVariant::ParamsT params,
                                               typename AttentionVariant::DTypeO* tmp,
                                               cudaStream_t stream);

}  // namespace flashinfer

torch::Tensor single_prefill_with_kv_cache(
    unsigned int mask_mode_code, torch::Tensor q, torch::Tensor k, torch::Tensor v,
    std::optional<torch::Tensor> maybe_packed_custom_mask, torch::Tensor tmp,
    std::optional<torch::Tensor> maybe_alibi_slopes, unsigned int layout, int32_t window_left,
    float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta,
    std::optional<torch::Tensor> maybe_lse) {
  auto device = q.device();
  unsigned int head_dim = q.size(2);
  unsigned int kv_len, qo_len, num_kv_heads, num_qo_heads;
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  qo_len = q.size(0);
  num_qo_heads = q.size(1);
  uint32_t q_stride_n = q.stride(0), q_stride_h = q.stride(1), kv_stride_n, kv_stride_h;
  if (kv_layout == QKVLayout::kNHD) {
    kv_len = k.size(0);
    num_kv_heads = k.size(1);
    kv_stride_n = k.stride(0);
    kv_stride_h = k.stride(1);
  } else {
    kv_len = k.size(1);
    num_kv_heads = k.size(0);
    kv_stride_h = k.stride(0);
    kv_stride_n = k.stride(1);
  }
  const at::cuda::CUDAGuard device_guard(device);
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  auto o = torch::empty_like(q, q.options());
  if (maybe_lse) {
    const auto& lse = *maybe_lse;
    TORCH_CHECK(lse.size(0) == qo_len, lse.size(0), q.size(0));
    TORCH_CHECK(lse.size(1) == num_qo_heads, lse.size(1), q.size(1));
    TORCH_CHECK(lse.dtype() == torch::kFloat32, "lse must be float32");
  }

  constexpr auto POS_ENCODING_MODE = PosEncodingMode::kNone;
  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  bool use_logits_soft_cap = logits_soft_cap > 0.f;

  auto q_scalar_type = q.scalar_type();
  auto kv_scalar_type = k.scalar_type();

  DISPATCH_PYTORCH_QKV_DTYPE_TO_CTYPE(q_scalar_type, kv_scalar_type, q_type, kv_type, [&] {
    using DTypeQ = q_type;
    using DTypeKV = kv_type;
    using DTypeO = DTypeQ;
    return DISPATCH_mask_mode(mask_mode, MASK_MODE, [&] {
      return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
        return DISPATCH_LOGITS_SOFT_CAP(use_logits_soft_cap, USE_LOGITS_SOFT_CAP, [&] {
          using ParamsT = SinglePrefillParams<DTypeQ, DTypeKV, DTypeO>;
          using AttentionVariant =
              ComposedAttention<ParamsT, get_variant_code(
                                             /*use_custom_mask=*/MASK_MODE == MaskMode::kCustom,
                                             /*use_sliding_window=*/true, USE_LOGITS_SOFT_CAP,
                                             /*use_alibi_slopes=*/false)>;

          ParamsT params(static_cast<DTypeQ*>(q.data_ptr()), static_cast<DTypeKV*>(k.data_ptr()),
                         static_cast<DTypeKV*>(v.data_ptr()),
                         maybe_packed_custom_mask.has_value()
                             ? static_cast<uint8_t*>(maybe_packed_custom_mask->data_ptr())
                             : nullptr,
                         static_cast<DTypeO*>(o.data_ptr()),
                         /*lse=*/(maybe_lse ? static_cast<float*>(maybe_lse->data_ptr()) : nullptr),
                         /*alibi_slopes=*/nullptr, num_qo_heads, num_kv_heads, qo_len, kv_len,
                         q_stride_n, q_stride_h, kv_stride_n, kv_stride_h, head_dim, window_left,
                         logits_soft_cap, sm_scale, rope_scale, rope_theta);

          cudaError_t status =
              flashinfer::SinglePrefillWithKVCacheDispatched<HEAD_DIM, POS_ENCODING_MODE,
                                                             /*use_fp16_qk_reduction=*/false,
                                                             MASK_MODE, AttentionVariant>(
                  params, static_cast<DTypeO*>(tmp.data_ptr()), torch_current_stream);
          TORCH_CHECK(status == cudaSuccess,
                      "SinglePrefillWithKVCache kernel launch failed, error: " +
                          std::string(cudaGetErrorString(status)));
          return true;
        });
      });
    });
  });

  return o;
}
