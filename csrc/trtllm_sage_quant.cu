/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <flashinfer/trtllm/common.h>
#include <flashinfer/trtllm/common/sageQuant.h>

#include "tvm_ffi_utils.h"

namespace flashinfer {

using trtllm::invokeSageQuant;
using trtllm::SageQuantParams;

void trtllm_sage_attention_quantize(TensorView q_quant, TensorView k_quant, TensorView v_quant,
                                    TensorView q_scale, TensorView k_scale, TensorView v_scale,
                                    TensorView query, TensorView key, TensorView value,
                                    int64_t q_block_size, int64_t k_block_size, int64_t sm_count) {
  TVM_FFI_ICHECK_EQ(query.ndim(), 3) << "query must have shape [tokens, heads, head_dim]";
  TVM_FFI_ICHECK_EQ(key.ndim(), 3) << "key must have shape [tokens, heads, head_dim]";
  TVM_FFI_ICHECK_EQ(value.ndim(), 3) << "value must have shape [tokens, heads, head_dim]";
  TVM_FFI_ICHECK_EQ(query.dtype(), key.dtype()) << "query and key must have the same dtype";
  TVM_FFI_ICHECK_EQ(query.dtype(), value.dtype()) << "query and value must have the same dtype";
  TVM_FFI_ICHECK(query.dtype() == dl_float16 || query.dtype() == dl_bfloat16)
      << "SageQuant inputs must be float16 or bfloat16";
  TVM_FFI_ICHECK_EQ(query.size(2), key.size(2))
      << "query and key must have the same head dimension";
  TVM_FFI_ICHECK_EQ(key.size(2), value.size(2))
      << "key and value must have the same head dimension";
  TVM_FFI_ICHECK_EQ(key.size(0), value.size(0)) << "key and value must have the same token count";
  TVM_FFI_ICHECK_EQ(key.size(1), value.size(1)) << "key and value must have the same head count";
  auto const is_contiguous_3d = [](TensorView tensor) {
    return tensor.stride(2) == 1 && tensor.stride(1) == tensor.size(2) &&
           tensor.stride(0) == tensor.size(1) * tensor.size(2);
  };
  TVM_FFI_ICHECK(is_contiguous_3d(query)) << "query must be contiguous";
  TVM_FFI_ICHECK(is_contiguous_3d(key)) << "key must be contiguous";
  TVM_FFI_ICHECK(is_contiguous_3d(value)) << "value must be contiguous";

  TVM_FFI_ICHECK_EQ(q_quant.ndim(), 3);
  TVM_FFI_ICHECK_EQ(k_quant.ndim(), 3);
  TVM_FFI_ICHECK_EQ(v_quant.ndim(), 3);
  TVM_FFI_ICHECK_EQ(q_quant.numel(), query.numel());
  TVM_FFI_ICHECK_EQ(k_quant.numel(), key.numel());
  TVM_FFI_ICHECK_EQ(v_quant.numel(), value.numel());
  TVM_FFI_ICHECK_EQ(q_quant.dtype(), k_quant.dtype());
  TVM_FFI_ICHECK(q_quant.dtype() == dl_int8 || q_quant.dtype() == dl_float8_e4m3fn)
      << "quantized Q/K tensors must be int8 or float8_e4m3fn";
  TVM_FFI_ICHECK_EQ(v_quant.dtype(), dl_float8_e4m3fn)
      << "quantized V tensor must be float8_e4m3fn";
  TVM_FFI_ICHECK(is_contiguous_3d(q_quant) && is_contiguous_3d(k_quant) &&
                 is_contiguous_3d(v_quant))
      << "quantized outputs must be contiguous";

  TVM_FFI_ICHECK(q_block_size == 1 || q_block_size == 4 || q_block_size == 16)
      << "q_block_size must be 1, 4, or 16";
  TVM_FFI_ICHECK(k_block_size == 1 || k_block_size == 4 || k_block_size == 16)
      << "k_block_size must be 1, 4, or 16";
  TVM_FFI_ICHECK_EQ(q_scale.dtype(), dl_float32);
  TVM_FFI_ICHECK_EQ(k_scale.dtype(), dl_float32);
  TVM_FFI_ICHECK_EQ(v_scale.dtype(), dl_float32);
  TVM_FFI_ICHECK_EQ(q_scale.numel(),
                    query.size(1) * ((query.size(0) + q_block_size - 1) / q_block_size));
  TVM_FFI_ICHECK_EQ(k_scale.numel(),
                    key.size(1) * ((key.size(0) + k_block_size - 1) / k_block_size));
  TVM_FFI_ICHECK_EQ(v_scale.numel(), value.size(1) * value.size(2));
  TVM_FFI_ICHECK_GT(sm_count, 0);

  ffi::CUDADeviceGuard device_guard(query.device().device_id);
  auto const stream = get_stream(query.device());
  auto const memset_status =
      cudaMemsetAsync(v_scale.data_ptr(), 0, v_scale.numel() * sizeof(float), stream);
  TVM_FFI_ICHECK_EQ(memset_status, cudaSuccess) << cudaGetErrorString(memset_status);

  SageQuantParams params{};
  params.headDim = query.size(2);
  params.inputType = query.dtype() == dl_float16 ? DATA_TYPE_FP16 : DATA_TYPE_BF16;
  params.quantType = q_quant.dtype() == dl_int8 ? DATA_TYPE_INT8 : DATA_TYPE_E4M3;
  params.sumSeqLensV = value.size(0);
  params.numHeadsV = value.size(1);
  params.ptrV = value.data_ptr();
  params.ptrVQuant = v_quant.data_ptr();
  params.ptrVScale = static_cast<float*>(v_scale.data_ptr());
  params.smCount = sm_count;
  params.stream = stream;

  params.sumSeqLensQk = query.size(0);
  params.numHeads = query.size(1);
  params.tokenBlockSize = q_block_size;
  params.ptrQk = query.data_ptr();
  params.ptrQkQuant = q_quant.data_ptr();
  params.ptrQkScale = static_cast<float*>(q_scale.data_ptr());
  params.vStage = 1;
  invokeSageQuant(params);

  params.sumSeqLensQk = key.size(0);
  params.numHeads = key.size(1);
  params.tokenBlockSize = k_block_size;
  params.ptrQk = key.data_ptr();
  params.ptrQkQuant = k_quant.data_ptr();
  params.ptrQkScale = static_cast<float*>(k_scale.data_ptr());
  params.vStage = 2;
  invokeSageQuant(params);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_sage_attention_quantize, trtllm_sage_attention_quantize);

}  // namespace flashinfer
