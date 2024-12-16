/*
 * Copyright (c) 2024 by FlashInfer team.
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
#include <cutlass/numeric_types.h>

#include <flashinfer/attention/hopper/params.cuh>
#include <flashinfer/attention/hopper/variants.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/cutlass_utils.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/math.cuh>
#include <optional>

#include "aot_extension_utils.h"

namespace flashinfer {

template <uint32_t HEAD_DIM, MaskMode MASK_MODE, bool LEFT_SLINDING_WINDOW,
          typename AttentionVariant, typename DTypeQ, typename DTypeKV, typename DTypeO>
cudaError_t SinglePrefillWithKVCacheDispatched(SinglePrefillParams<DTypeQ, DTypeKV, DTypeO>& params,
                                               cudaStream_t stream);

}  // namespace flashinfer

using namespace flashinfer;

void single_prefill_with_kv_cache_sm90(unsigned int mask_mode_code, at::Tensor q, at::Tensor k,
                                       at::Tensor v,
                                       std::optional<at::Tensor> maybe_packed_custom_mask,
                                       std::optional<at::Tensor> maybe_alibi_slopes, at::Tensor o,
                                       unsigned int layout, int32_t window_left,
                                       float logits_soft_cap, float sm_scale, float rope_scale,
                                       float rope_theta, std::optional<at::Tensor> maybe_lse,
                                       int64_t cuda_stream) {
  unsigned int head_dim = q.size(2);
  unsigned int num_qo_heads = q.size(1);
  unsigned int qo_len = q.size(0);

  auto q_scalar_type = q.scalar_type();

  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  bool use_logits_soft_cap = logits_soft_cap > 0.0f;

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q_scalar_type, q_type, [&] {
    using DTypeQ = cutlass_dtype_t<q_type>;
    using DTypeKV = DTypeQ;
    using DTypeO = DTypeQ;
    SinglePrefillParams<DTypeQ, DTypeKV, DTypeO> params;
    params.q_ptr = static_cast<DTypeQ*>(q.data_ptr());
    params.k_ptr = static_cast<DTypeKV*>(k.data_ptr());
    params.v_ptr = static_cast<DTypeKV*>(v.data_ptr());
    params.o_ptr = static_cast<DTypeO*>(o.data_ptr());
    params.lse_ptr = maybe_lse ? (static_cast<float*>(maybe_lse->data_ptr())) : nullptr;
    params.q_stride_n = q.stride(0);
    params.q_stride_h = q.stride(1);
    params.o_stride_n = o.stride(0);
    params.o_stride_h = o.stride(1);
    if (kv_layout == QKVLayout::kNHD) {
      params.k_stride_n = k.stride(0);
      params.k_stride_h = k.stride(1);
      params.v_stride_n = v.stride(0);
      params.v_stride_h = v.stride(1);
    } else {
      params.k_stride_h = k.stride(0);
      params.k_stride_n = k.stride(1);
      params.v_stride_h = v.stride(0);
      params.v_stride_n = v.stride(1);
    }
    params.qo_len = q.size(0);
    params.kv_len = k.size(0);
    params.head_dim = head_dim;
    params.num_qo_heads = q.size(1);
    params.num_kv_heads = k.size(1);
    params.causal = mask_mode == MaskMode::kCausal;
    params.group_size = params.num_qo_heads / params.num_kv_heads;
    params.window_left = window_left;
    params.logits_soft_cap = logits_soft_cap;
    params.sm_scale_log2 = sm_scale * math::log2e;
    bool use_swa = window_left != -1;
    return DISPATCH_mask_mode(mask_mode, MASK_MODE, [&] {
      return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
        return DISPATCH_BOOL(use_logits_soft_cap, USE_LOGITS_SOFT_CAP, [&] {
          return DISPATCH_BOOL(use_swa, USE_SWA, [&] {
            using AttentionVariant =
                std::conditional_t<USE_LOGITS_SOFT_CAP, LogitsSoftCap, StandardAttention>;
            cudaError_t status =
                SinglePrefillWithKVCacheDispatched<HEAD_DIM, MASK_MODE, USE_SWA, AttentionVariant>(
                    params, stream);
            TORCH_CHECK(status == cudaSuccess,
                        "single_prefill_with_kv_cache_sm90 failed with error: " +
                            std::string(cudaGetErrorString(status)));
            return true;
          });
        });
      });
    });
  });
}
