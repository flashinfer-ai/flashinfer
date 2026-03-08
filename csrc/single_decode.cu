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

#include "single_decode_config.inc"
#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

namespace flashinfer {

template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant,
          typename Params>
cudaError_t SingleDecodeWithKVCacheDispatched(Params params, typename Params::DTypeO* tmp,
                                              cudaStream_t stream);
}  // namespace flashinfer

using namespace flashinfer;

void single_decode_with_kv_cache(TensorView q, TensorView k, TensorView v, TensorView tmp,
                                 TensorView o, Optional<TensorView> maybe_lse, int64_t layout,
                                 int64_t window_left ADDITIONAL_FUNC_PARAMS) {
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  CHECK_INPUT(tmp);
  CHECK_DEVICE(k, q);
  CHECK_DEVICE(v, q);
  CHECK_DEVICE(tmp, q);
  CHECK_DIM(2, q);
  CHECK_DIM(3, k);
  CHECK_DIM(3, v);
  CHECK_SHAPE(k, v);
  TVM_FFI_ICHECK_EQ(q.size(1), k.size(2));
  TVM_FFI_ICHECK_EQ(v.dtype(), k.dtype());
  unsigned int num_qo_heads = q.size(0);
  unsigned int head_dim_qk = q.size(1);
  unsigned int head_dim_vo = v.size(2);
  unsigned int kv_len, num_kv_heads;
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  if (kv_layout == QKVLayout::kNHD) {
    kv_len = k.size(0);
    num_kv_heads = k.size(1);
  } else {
    num_kv_heads = k.size(0);
    kv_len = k.size(1);
  }
  TVM_FFI_ICHECK_EQ(num_qo_heads % num_kv_heads, 0)
      << "num_qo_heads(" << num_qo_heads << ") must be divisible by num_kv_heads(" << num_kv_heads
      << ")";

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());

  TVM_FFI_ICHECK_EQ(head_dim_qk, head_dim_vo)
      << "CUDA cores template only supports equal head dim for QK and VO, please use tensor "
         "cores template for different head dim";

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, AttentionVariant, Params, [&] {
        Params params;

        params.q = static_cast<DTypeQ*>(q.data_ptr());
        params.k = static_cast<DTypeKV*>(k.data_ptr());
        params.v = static_cast<DTypeKV*>(v.data_ptr());
        params.o = static_cast<DTypeO*>(o.data_ptr());
        params.lse =
            maybe_lse.has_value() ? static_cast<float*>(maybe_lse.value().data_ptr()) : nullptr;
        params.kv_len = kv_len;
        params.num_qo_heads = num_qo_heads;
        params.num_kv_heads = num_kv_heads;
        params.q_stride_n = num_qo_heads * head_dim_qk;
        params.q_stride_h = head_dim_qk;
        params.kv_stride_n =
            (kv_layout == QKVLayout::kNHD) ? num_kv_heads * head_dim_vo : head_dim_vo;
        params.kv_stride_h = (kv_layout == QKVLayout::kNHD) ? head_dim_vo : kv_len * head_dim_vo;
        params.window_left = window_left;
        params.kv_chunk_size = 0;

        ADDITIONAL_PARAMS_SETTER

        cudaError_t status =
            flashinfer::SingleDecodeWithKVCacheDispatched<HEAD_DIM_QK, POS_ENCODING_MODE,
                                                          AttentionVariant>(
                params, static_cast<DTypeO*>(tmp.data_ptr()), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "SingleDecodeWithKVCache kernel launch failed, error: "
            << cudaGetErrorString(status);
        return true;
      });
}
