/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "allreduce_mhc_post_prenorm.cuh"
#include "tvm_ffi_utils.h"

using namespace flashinfer::trtllm_mnnvl_ar_mhc_post;

// MNNVL one-shot all-reduce of `input` fused with the mHC post-map and the split-K pre-norm
// partials. Shapes (M = tokens): input [M, H] bf16, residual [M, HC, H] bf16, post [M, HC] fp32,
// comb [M, HC, HC] fp32, fn [MIX, HC * H] fp32 -> yp [NS, M, MIX] fp32, rp [NS, M] fp32,
// residual_out [M, HC, H] bf16. Supported: world size 8, H 4096, HC 4, 1 <= M <= 32.
void allreduce_mhc_post_prenorm(TensorView input, TensorView residual, TensorView post,
                                TensorView comb, TensorView fn, TensorView yp, TensorView rp,
                                TensorView residual_out, int64_t multicast_buffer_ptr,
                                int64_t buffer_ptr_local, TensorView buffer_flags_mnnvl,
                                int64_t nranks, int64_t rank, bool launch_with_pdl) {
  CHECK_INPUT_AND_TYPE(input, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(residual, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(post, dl_float32);
  CHECK_INPUT_AND_TYPE(comb, dl_float32);
  CHECK_INPUT_AND_TYPE(fn, dl_float32);
  CHECK_INPUT_AND_TYPE(yp, dl_float32);
  CHECK_INPUT_AND_TYPE(rp, dl_float32);
  CHECK_INPUT_AND_TYPE(residual_out, dl_bfloat16);
  CHECK_DEVICE(input, residual);
  CHECK_DEVICE(input, fn);
  CHECK_DEVICE(input, yp);
  CHECK_DEVICE(input, residual_out);
  CHECK_DIM(2, input);
  CHECK_DIM(3, residual);
  CHECK_DIM(2, post);
  CHECK_DIM(3, comb);
  CHECK_DIM(2, fn);
  CHECK_DIM(3, yp);
  CHECK_DIM(2, rp);
  CHECK_DIM(3, residual_out);

  int64_t const num_tokens = input.size(0);
  TVM_FFI_ICHECK_EQ(nranks, kWorldSize) << "world size must be " << kWorldSize;
  TVM_FFI_ICHECK(rank >= 0 && rank < nranks) << "rank out of range: " << rank;
  TVM_FFI_ICHECK_EQ(input.size(1), kHidden) << "hidden size must be " << kHidden;
  TVM_FFI_ICHECK(num_tokens >= 1 && num_tokens <= kMaxTokens)
      << "num_tokens must be in [1, " << kMaxTokens << "], got " << num_tokens;
  TVM_FFI_ICHECK(residual.size(0) == num_tokens && residual.size(1) == kHcMult &&
                 residual.size(2) == kHidden)
      << "residual must be [num_tokens, " << kHcMult << ", " << kHidden << "]";
  TVM_FFI_ICHECK(post.size(0) == num_tokens && post.size(1) == kHcMult)
      << "post_layer_mix must be [num_tokens, " << kHcMult << "]";
  TVM_FFI_ICHECK(comb.size(0) == num_tokens && comb.size(1) == kHcMult && comb.size(2) == kHcMult)
      << "comb_res_mix must be [num_tokens, " << kHcMult << ", " << kHcMult << "]";
  TVM_FFI_ICHECK(fn.size(0) == kMix && fn.size(1) == kHcMult * kHidden)
      << "fn must be [" << kMix << ", " << kHcMult * kHidden << "]";
  TVM_FFI_ICHECK(yp.size(0) == kNumSplits && yp.size(1) == num_tokens && yp.size(2) == kMix)
      << "yp must be [" << kNumSplits << ", num_tokens, " << kMix << "]";
  TVM_FFI_ICHECK(rp.size(0) == kNumSplits && rp.size(1) == num_tokens)
      << "rp must be [" << kNumSplits << ", num_tokens]";
  TVM_FFI_ICHECK(residual_out.size(0) == num_tokens && residual_out.size(1) == kHcMult &&
                 residual_out.size(2) == kHidden)
      << "residual_out must be [num_tokens, " << kHcMult << ", " << kHidden << "]";

  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  cudaStream_t stream = get_stream(input.device());

  ArMhcPostParams params;
  params.input = static_cast<__nv_bfloat16 const*>(input.data_ptr());
  params.residual = static_cast<__nv_bfloat16 const*>(residual.data_ptr());
  params.post = static_cast<float const*>(post.data_ptr());
  params.comb = static_cast<float const*>(comb.data_ptr());
  params.fn = static_cast<float const*>(fn.data_ptr());
  params.yp = static_cast<float*>(yp.data_ptr());
  params.rp = static_cast<float*>(rp.data_ptr());
  params.residualOut = static_cast<__nv_bfloat16*>(residual_out.data_ptr());
  params.multicastPtr = reinterpret_cast<void*>(multicast_buffer_ptr);
  params.bufferPtrLocal = reinterpret_cast<void*>(buffer_ptr_local);
  params.bufferFlags = static_cast<uint32_t*>(buffer_flags_mnnvl.data_ptr());
  params.rank = static_cast<int>(rank);
  params.numTokens = static_cast<int>(num_tokens);

  cudaError_t status = arMhcPostPrenormDispatch(params, stream, launch_with_pdl);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "allreduce_mhc_post_prenorm failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(allreduce_mhc_post_prenorm, allreduce_mhc_post_prenorm);
