/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <flashinfer/allocator.h>
#include <flashinfer/exception.h>
#include <tvm_ffi_utils.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>

#include "flashinfer/flat/prefill/prefill_kernel.hpp"

using tvm::ffi::Optional;
using tvm::ffi::TensorView;
using tvm::ffi::Variant;

namespace flashinfer {

void gdn_prefill_launcher(void* output, void* output_state, void* q, void* k, void* v,
                          void* input_state, void* alpha, void* beta, int64_t* cu_seqlens,
                          uint8_t* workspace_buffer, int64_t num_seqs, int64_t num_q_heads,
                          int64_t num_k_heads, int64_t num_v_heads, int64_t num_o_heads,
                          int64_t head_size, int64_t packed_seq, float scale, int64_t sm_count,
                          DLDataType dtype, cudaStream_t stream) {
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(dtype, DType, [&] {
    int dev_id;
    cudaGetDevice(&dev_id);
    int device_major;
    cudaDeviceGetAttribute(&device_major, cudaDevAttrComputeCapabilityMajor, dev_id);

#if defined(FLAT_SM90A_ENABLED)
    if (device_major == 9) {
      flat::launch_delta_rule_prefill_kernel<cutlass::arch::Sm90, DType, DType, float>(
          stream, static_cast<DType*>(output), static_cast<float*>(output_state),
          static_cast<DType const*>(q), static_cast<DType const*>(k), static_cast<DType const*>(v),
          static_cast<float const*>(input_state), static_cast<float const*>(alpha),
          static_cast<float const*>(beta), cu_seqlens, workspace_buffer, num_seqs, num_q_heads,
          num_k_heads, num_v_heads, num_o_heads, head_size, packed_seq, scale, sm_count);
      return true;
    } else {
      std::ostringstream err_msg;
      err_msg << "delta rule kernel does not support this device major version: " << device_major;
      FLASHINFER_ERROR(err_msg.str());
      return false;
    }
#else
    FLASHINFER_ERROR("sm_90a is not enabled, delta rule kernel is not built");
    return false;
#endif
  });
}

void gdn_prefill(TensorView output, TensorView output_state, TensorView q, TensorView k,
                 TensorView v, TensorView cu_seqlens, Optional<TensorView> input_state,
                 Optional<TensorView> alpha, Optional<TensorView> beta, double scale,
                 TensorView workspace_buffer) {
  int64_t num_seqs = cu_seqlens.size(0) - 1;
  int64_t packed_seq = q.size(0);
  int64_t head_size = q.size(2);
  int64_t num_q_heads = q.size(1);
  int64_t num_k_heads = k.size(1);
  int64_t num_v_heads = v.size(1);

  // NOTE: Qwen3-next alpha and beta heads are 32, equal to v heads, we limit it to larger one
  int32_t num_sab_heads = std::max(num_q_heads, num_v_heads);

  if (num_q_heads >= num_v_heads) {  // GQA
    auto ratio = num_q_heads / num_v_heads;
    TVM_FFI_ICHECK_EQ(num_k_heads, num_v_heads);
    TVM_FFI_ICHECK_EQ(num_q_heads, ratio * num_k_heads);
    TVM_FFI_ICHECK_EQ(num_q_heads, ratio * num_v_heads);
  } else {  // GVA
    auto ratio = num_v_heads / num_q_heads;
    TVM_FFI_ICHECK_EQ(num_q_heads, num_k_heads);
    TVM_FFI_ICHECK_EQ(num_v_heads, ratio * num_q_heads);
    TVM_FFI_ICHECK_EQ(num_v_heads, ratio * num_k_heads);
  }

  int64_t num_o_heads = output.size(1);
  TVM_FFI_ICHECK_EQ(num_o_heads, num_sab_heads);

  void* input_state_ptr = nullptr;
  if (input_state.has_value()) {
    CHECK_SHAPE(input_state.value(), output_state);
    TVM_FFI_ICHECK_EQ(input_state.value().dtype(), dl_float32);
    input_state_ptr = input_state.value().data_ptr();
  }

  CHECK_INPUT(output);
  CHECK_INPUT(output_state);
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  CHECK_INPUT(cu_seqlens);
  CHECK_INPUT(workspace_buffer);

  TVM_FFI_ICHECK(output.dtype() == dl_float16 || output.dtype() == dl_bfloat16);
  TVM_FFI_ICHECK_EQ(output_state.dtype(), dl_float32);
  TVM_FFI_ICHECK_EQ(output.dtype(), q.dtype());
  TVM_FFI_ICHECK_EQ(output.dtype(), k.dtype());
  TVM_FFI_ICHECK_EQ(output.dtype(), v.dtype());
  TVM_FFI_ICHECK_EQ(cu_seqlens.dtype(), dl_int64);
  TVM_FFI_ICHECK_EQ(workspace_buffer.dtype(), dl_uint8);

  TVM_FFI_ICHECK_EQ(packed_seq, k.size(0));
  TVM_FFI_ICHECK_EQ(packed_seq, v.size(0));
  TVM_FFI_ICHECK_EQ(packed_seq, output.size(0));

  TVM_FFI_ICHECK_EQ(num_seqs, output_state.size(0));
  TVM_FFI_ICHECK_EQ(num_sab_heads, output_state.size(1));

  TVM_FFI_ICHECK_EQ(head_size, output.size(2));
  TVM_FFI_ICHECK_EQ(head_size, k.size(2));
  TVM_FFI_ICHECK_EQ(head_size, v.size(2));
  TVM_FFI_ICHECK_EQ(head_size, output_state.size(2));
  TVM_FFI_ICHECK_EQ(head_size, output_state.size(3));

  void* alpha_ptr = nullptr;
  if (alpha.has_value()) {
    TensorView alpha_tensor = alpha.value();
    TVM_FFI_ICHECK_EQ(alpha_tensor.dtype(), dl_float32);
    TVM_FFI_ICHECK_EQ(alpha_tensor.size(0), packed_seq);
    TVM_FFI_ICHECK_EQ(alpha_tensor.size(1), num_sab_heads);
    CHECK_INPUT(alpha_tensor);
    alpha_ptr = alpha_tensor.data_ptr();
  }

  void* beta_ptr = nullptr;
  if (beta.has_value()) {
    TensorView beta_tensor = beta.value();
    TVM_FFI_ICHECK_EQ(beta_tensor.dtype(), dl_float32);
    TVM_FFI_ICHECK_EQ(beta_tensor.size(0), packed_seq);
    TVM_FFI_ICHECK_EQ(beta_tensor.size(1), num_sab_heads);
    CHECK_INPUT(beta_tensor);
    beta_ptr = beta_tensor.data_ptr();
  }

  if (scale == 0.0) {
    scale = 1.0 / std::sqrt(head_size);
  }

  // Use cudaDeviceGetAttribute for sm_count (much faster than cudaGetDeviceProperties)
  int dev_id;
  cudaGetDevice(&dev_id);
  int sm_count;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);

  auto stream = get_stream(q.device());

  gdn_prefill_launcher(output.data_ptr(), output_state.data_ptr(), q.data_ptr(), k.data_ptr(),
                       v.data_ptr(), input_state_ptr, alpha_ptr, beta_ptr,
                       static_cast<int64_t*>(cu_seqlens.data_ptr()),
                       static_cast<uint8_t*>(workspace_buffer.data_ptr()), num_seqs, num_q_heads,
                       num_k_heads, num_v_heads, num_o_heads, head_size, packed_seq,
                       static_cast<float>(scale), sm_count, q.dtype(), stream);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gdn_prefill, gdn_prefill);

}  // namespace flashinfer
