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

#include "flashinfer/fastdiv.cuh"
#include "single_prefill_config.inc"
#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

namespace flashinfer {

template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, PosEncodingMode POS_ENCODING_MODE,
          bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE, typename AttentionVariant,
          typename Params>
cudaError_t SinglePrefillWithKVCacheDispatched(Params params, typename Params::DTypeO* tmp,
                                               cudaStream_t stream);

}  // namespace flashinfer

using namespace flashinfer;

void single_prefill_with_kv_cache(ffi::TensorView q, ffi::TensorView k, ffi::TensorView v,
                                  ffi::TensorView tmp, ffi::TensorView o,
                                  Optional<ffi::TensorView> maybe_lse, int64_t mask_mode_code,
                                  int64_t layout, int64_t window_left ADDITIONAL_FUNC_PARAMS) {
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
  if (maybe_lse.has_value()) {
    const auto& lse = maybe_lse.value();
    TVM_FFI_ICHECK_EQ(lse.size(0), qo_len);
    TVM_FFI_ICHECK_EQ(lse.size(1), num_qo_heads);
  }

  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, USE_FP16_QK_REDUCTION, AttentionVariant, Params,
      [&] {
        Params params;

        params.q = static_cast<DTypeQ*>(q.data_ptr());
        params.k = static_cast<DTypeKV*>(k.data_ptr());
        params.v = static_cast<DTypeKV*>(v.data_ptr());
        params.o = static_cast<DTypeO*>(o.data_ptr());
        params.lse =
            maybe_lse.has_value() ? static_cast<float*>(maybe_lse.value().data_ptr()) : nullptr;
        params.num_qo_heads = num_qo_heads;
        params.num_kv_heads = num_kv_heads;
        params.group_size = uint_fastdiv(num_qo_heads / num_kv_heads);
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

        ADDITIONAL_PARAMS_SETTER

        cudaError_t status = flashinfer::SinglePrefillWithKVCacheDispatched<
            HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
            /*use_fp16_qk_reduction=*/USE_FP16_QK_REDUCTION, MASK_MODE, AttentionVariant>(
            params, static_cast<DTypeO*>(tmp.data_ptr()), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "SinglePrefillWithKVCache kernel launch failed, error: "
            << cudaGetErrorString(status);
        return true;
      });
}
