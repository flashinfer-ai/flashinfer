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
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/pos_enc.cuh>
#include <optional>

#include "pytorch_extension_utils.h"
#include "single_prefill_config.inc"

namespace flashinfer {

template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, bool USE_FP16_QK_REDUCTION,
          MaskMode MASK_MODE, typename AttentionVariant, typename Params>
cudaError_t SinglePrefillWithKVCacheDispatched(Params params, typename Params::DTypeO* tmp,
                                               cudaStream_t stream);

}  // namespace flashinfer

using namespace flashinfer;

void single_prefill_with_kv_cache(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor tmp,
                                  at::Tensor o, std::optional<at::Tensor> maybe_lse,
                                  unsigned int mask_mode_code, unsigned int layout,
                                  int32_t window_left ADDITIONAL_FUNC_PARAMS, int64_t cuda_stream) {
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
  if (maybe_lse) {
    const auto& lse = *maybe_lse;
    TORCH_CHECK(lse.size(0) == qo_len, lse.size(0), q.size(0));
    TORCH_CHECK(lse.size(1) == num_qo_heads, lse.size(1), q.size(1));
  }

  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);

  auto q_scalar_type = q.scalar_type();
  auto kv_scalar_type = k.scalar_type();

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM, POS_ENCODING_MODE, USE_SLIDING_WINDOW,
      USE_LOGITS_SOFT_CAP, USE_FP16_QK_REDUCTION, AttentionVariant, Params, [&] {
        Params params;

        params.q = static_cast<DTypeQ*>(q.data_ptr());
        params.k = static_cast<DTypeKV*>(k.data_ptr());
        params.v = static_cast<DTypeKV*>(v.data_ptr());
        params.o = static_cast<DTypeO*>(o.data_ptr());
        params.lse = maybe_lse ? static_cast<float*>(maybe_lse->data_ptr()) : nullptr;
        params.num_qo_heads = num_qo_heads;
        params.num_kv_heads = num_kv_heads;
        params.qo_len = qo_len;
        params.kv_len = kv_len;
        params.q_stride_n = q_stride_n;
        params.q_stride_h = q_stride_h;
        params.kv_stride_n = kv_stride_n;
        params.kv_stride_h = kv_stride_h;
        params.head_dim = head_dim;
        params.window_left = window_left;
        params.partition_kv = false;

        ADDITIONAL_PARAMS_SETTER

        cudaError_t status = flashinfer::SinglePrefillWithKVCacheDispatched<
            HEAD_DIM, POS_ENCODING_MODE,
            /*use_fp16_qk_reduction=*/USE_FP16_QK_REDUCTION, MASK_MODE, AttentionVariant>(
            params, static_cast<DTypeO*>(tmp.data_ptr()), stream);
        TORCH_CHECK(status == cudaSuccess,
                    "SinglePrefillWithKVCache kernel launch failed, error: " +
                        std::string(cudaGetErrorString(status)));
        return true;
      });
}
