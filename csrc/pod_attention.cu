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
#include <flashinfer/attention/pod.cuh>
#include <optional>
#include "pytorch_extension_utils.h"

#pragma once
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/utils.cuh>

#include "aot_default_additional_params.h"
#include "aot_extension_utils.h"
//#include "single_prefill_config.inc"

namespace flashinfer {

template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, PosEncodingMode POS_ENCODING_MODE,
          bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE, typename AttentionVariant,
          typename Params>
cudaError_t PODWithKVCacheDispatched(Params params, typename Params::DTypeO* tmp,
                                               cudaStream_t stream);

}  // namespace flashinfer

using namespace flashinfer;

void pod_with_kv_cache(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor tmp,
                       at::Tensor o, std::optional<at::Tensor> maybe_lse,
                       unsigned int mask_mode_code, unsigned int layout,
                       int32_t window_left SINGLE_PREFILL_ADDITIONAL_FUNC_PARAMS,
                       int64_t cuda_stream) {
  auto device = q.device();
  unsigned int head_dim_qk = q.size(2);
  unsigned int kv_len, qo_len, num_kv_heads, num_qo_heads;
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  qo_len = q.size(0);
  num_qo_heads = q.size(1);
  uint32_t q_stride_n = q.stride(0), q_stride_h = q.stride(1), k_stride_n, k_stride_h, v_stride_n,
           v_stride_h;
  if (kv_layout == QKVLayout::kNHD) {
    kv_len = k.size(0);
    num_kv_heads = k.size(1);
    k_stride_n = k.stride(0);
    k_stride_h = k.stride(1);
    v_stride_n = v.stride(0);
    v_stride_h = v.stride(1);
  } else {
    kv_len = k.size(1);
    num_kv_heads = k.size(0);
    k_stride_h = k.stride(0);
    k_stride_n = k.stride(1);
    v_stride_h = v.stride(0);
    v_stride_n = v.stride(1);
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

  DISPATCH_mask_mode(mask_mode, MASK_MODE, [&] {
    return DISPATCH_PYTORCH_QKV_DTYPE_TO_CTYPE(
      q_scalar_type, kv_scalar_type, DTypeQ, DTypeKV, [&] {
        using DTypeO = DTypeQ;
        constexpr auto POS_ENCODING_MODE = PosEncodingMode::kNone;
        constexpr bool USE_FP16_QK_REDUCTION = false;
        constexpr bool use_custom_mask = MASK_MODE == MaskMode::kCustom;
        return DISPATCH_head_dim(head_dim_qk, HEAD_DIM_QK, [&] {
          [[maybe_unused]] constexpr int HEAD_DIM_VO = HEAD_DIM_QK;
          return DISPATCH_BOOL(window_left > -1, USE_SLIDING_WINDOW, [&] {
            return DISPATCH_BOOL(logits_soft_cap > 0.f, USE_LOGITS_SOFT_CAP, [&] {
              using AttentionVariant =
                  DefaultAttention</*use_custom_mask=*/use_custom_mask, USE_SLIDING_WINDOW,
                                    USE_LOGITS_SOFT_CAP, /*use_alibi_bias=*/false>;
              using PrefillParams = SinglePrefillParams<DTypeQ, DTypeKV, DTypeO>;
              PrefillParams prefill_params;
              {
                // Make params a reference to prefill_params to set values
                PrefillParams &params = prefill_params;
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
                params.k_stride_n = k_stride_n;
                params.k_stride_h = k_stride_h;
                params.v_stride_n = v_stride_n;
                params.v_stride_h = v_stride_h;
        
                params.window_left = window_left;
                params.partition_kv = false;
        
                SINGLE_PREFILL_ADDITIONAL_PARAMS_SETTER
              }
              // TODO: Decode params here
              {

              }
              cudaError_t status = flashinfer::PODWithKVCacheDispatched<
                  HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
                  /*use_fp16_qk_reduction=*/USE_FP16_QK_REDUCTION, MASK_MODE, AttentionVariant>(
                    prefill_params, static_cast<DTypeO*>(tmp.data_ptr()), stream);
              TORCH_CHECK(status == cudaSuccess,
                          "PODWithKVCache kernel launch failed, error: " +
                              std::string(cudaGetErrorString(status)));
              return true;
            });
          });
        });
      });
  });
}
