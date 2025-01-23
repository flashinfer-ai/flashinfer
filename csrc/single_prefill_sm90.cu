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
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/math.cuh>
#include <optional>

#include "pytorch_extension_utils.h"
#include "single_prefill_sm90_config.inc"

namespace flashinfer {

template <uint32_t HEAD_DIM, MaskMode MASK_MODE, bool LEFT_SLINDING_WINDOW,
          typename AttentionVariant, typename Params>
cudaError_t SinglePrefillWithKVCacheDispatched(Params& params, cudaStream_t stream);

}  // namespace flashinfer

using namespace flashinfer;

void single_prefill_with_kv_cache_sm90(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor tmp,
                                       at::Tensor o, std::optional<at::Tensor> maybe_lse,
                                       unsigned int mask_mode_code, unsigned int layout,
                                       int32_t window_left ADDITIONAL_FUNC_PARAMS,
                                       int64_t cuda_stream) {
  unsigned int head_dim = q.size(2);
  unsigned int num_qo_heads = q.size(1);
  unsigned int qo_len = q.size(0);

  auto q_scalar_type = q.scalar_type();
  auto kv_scalar_type = k.scalar_type();

  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM, USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP,
      AttentionVariant, Params, [&] {
        Params params;
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

        ADDITIONAL_PARAMS_SETTER

        cudaError_t status =
            SinglePrefillWithKVCacheDispatched<HEAD_DIM, MASK_MODE, USE_SLIDING_WINDOW,
                                               AttentionVariant>(params, stream);
        TORCH_CHECK(status == cudaSuccess, "single_prefill_with_kv_cache_sm90 failed with error: " +
                                               std::string(cudaGetErrorString(status)));
        return true;
      });
}
