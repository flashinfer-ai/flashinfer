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

#include "single_prefill_sm90_config.inc"
#include "tvm_ffi_utils.h"

namespace flashinfer {

template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, MaskMode MASK_MODE, bool LEFT_SLINDING_WINDOW,
          typename AttentionVariant, typename Params>
cudaError_t SinglePrefillWithKVCacheDispatched(Params& params, cudaStream_t stream);

}  // namespace flashinfer

using namespace flashinfer;

using tvm::ffi::Optional;

void single_prefill_with_kv_cache_sm90(ffi::TensorView q, ffi::TensorView k, ffi::TensorView v,
                                       ffi::TensorView tmp, ffi::TensorView o,
                                       Optional<ffi::TensorView> maybe_lse, int64_t mask_mode_code,
                                       int64_t layout, int64_t window_left ADDITIONAL_FUNC_PARAMS) {
  unsigned int head_dim_qk = q.size(2);
  unsigned int head_dim_vo = v.size(2);
  unsigned int num_qo_heads = q.size(1);
  unsigned int qo_len = q.size(0);

  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());
  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);

  DISPATCH_context(DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO,
                   USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, AttentionVariant, Params, [&] {
                     Params params;
                     params.q_ptr = static_cast<DTypeQ*>(q.data_ptr());
                     params.k_ptr = static_cast<DTypeKV*>(k.data_ptr());
                     params.v_ptr = static_cast<DTypeKV*>(v.data_ptr());
                     params.o_ptr = static_cast<DTypeO*>(o.data_ptr());
                     params.lse_ptr = maybe_lse.has_value()
                                          ? (static_cast<float*>(maybe_lse.value().data_ptr()))
                                          : nullptr;
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
                     params.num_qo_heads = q.size(1);
                     params.num_kv_heads = k.size(1);
                     params.causal = mask_mode == MaskMode::kCausal;
                     params.group_size = params.num_qo_heads / params.num_kv_heads;
                     params.window_left = window_left;

                     ADDITIONAL_PARAMS_SETTER

                     cudaError_t status =
                         SinglePrefillWithKVCacheDispatched<HEAD_DIM_QK, HEAD_DIM_VO, MASK_MODE,
                                                            USE_SLIDING_WINDOW, AttentionVariant>(
                             params, stream);
                     TVM_FFI_ICHECK(status == cudaSuccess)
                         << "single_prefill_with_kv_cache_sm90 failed with error: "
                         << cudaGetErrorString(status);
                     return true;
                   });
}
