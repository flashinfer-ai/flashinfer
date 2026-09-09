/*
 * Copyright (c) 2026 by FlashInfer team.
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
#include <cstdint>

#include "flashinfer/trtllm/fused_moe/DevKernel.h"
#include "tvm_ffi_utils.h"

namespace flashinfer::trtllm_gen_gather_activation {

namespace tg = batchedGemm::trtllm::gen;
using tvm::ffi::TensorView;

/*!
 * \brief TVM-FFI entry point for the permuted -> expanded gather of the trtllm-gen MoE
 * post-activation FC1 output.
 *
 * The routed MoE ops return that output in the kernels' permuted layout together with
 * expanded_idx_to_permuted_idx; every consumer of it (a LoRA down-projection, an expert-level
 * probe) first has to bring the rows back into (token, slot) order. This runs that gather as a
 * single copy kernel instead of the eager index_select + masked scatter it replaces.
 *
 *   activation_output           : [num_padded_rows, intermediate_size] bfloat16 or float16,
 *                                 permuted post-activation FC1 rows. Rows no expanded index
 *                                 points at (the routing padding) are never read.
 *   expanded_idx_to_permuted_idx: num_tokens * top_k int32 elements; a negative entry marks an
 *                                 inactive slot (an expert outside the local expert-parallel
 *                                 shard), whose output row is zero-filled.
 *   output                      : [num_tokens, top_k, intermediate_size], same dtype as
 *                                 activation_output.
 *
 * out[t, k] = activation_output[perm(t, k)] if perm(t, k) >= 0 else 0.
 *
 * The permuted indices are only bounds-checked by a device-side assert, so a release build with
 * an index at or past num_padded_rows reads out of bounds. Torch's advanced indexing raises
 * instead; this op trades that check away for the fused copy.
 */
void trtllm_gen_moe_gather_activation(TensorView activation_output,
                                      TensorView expanded_idx_to_permuted_idx, TensorView output,
                                      bool enable_pdl) {
  CHECK_INPUT(activation_output);
  CHECK_INPUT(expanded_idx_to_permuted_idx);
  CHECK_INPUT(output);
  CHECK_DEVICE(activation_output, output);
  CHECK_DEVICE(expanded_idx_to_permuted_idx, output);

  CHECK_DIM(2, activation_output);
  CHECK_DIM(3, output);

  TVM_FFI_ICHECK(output.dtype() == dl_bfloat16 || output.dtype() == dl_float16)
      << "output must be bfloat16 or float16";
  TVM_FFI_ICHECK(activation_output.dtype() == output.dtype())
      << "activation_output must have the same dtype as output";
  TVM_FFI_ICHECK(expanded_idx_to_permuted_idx.dtype() == dl_int32)
      << "expanded_idx_to_permuted_idx must be int32";

  int64_t const num_tokens = output.size(0);
  int64_t const top_k = output.size(1);
  int64_t const inner_dim = output.size(2);

  TVM_FFI_ICHECK_EQ(activation_output.size(1), inner_dim)
      << "activation_output and output must have the same innermost dimension";
  TVM_FFI_ICHECK_EQ(expanded_idx_to_permuted_idx.numel(), num_tokens * top_k)
      << "expanded_idx_to_permuted_idx must have num_tokens * top_k elements";

  // Empty problem: nothing to gather, and a zero grid dimension is a CUDA error.
  if (num_tokens == 0 || top_k == 0 || inner_dim == 0) {
    return;
  }

  moe::dev::gatherActivation::Data data;
  data.mDtypeElt = output.dtype() == dl_float16 ? tg::Dtype::Fp16 : tg::Dtype::Bfloat16;
  data.mUsePdl = enable_pdl;
  data.inPtr = activation_output.data_ptr();
  data.outPtr = output.data_ptr();
  data.expandedIdxToPermutedIdx =
      static_cast<int32_t const*>(expanded_idx_to_permuted_idx.data_ptr());
  data.innerDim = static_cast<int32_t>(inner_dim);
  data.numTokens = static_cast<int32_t>(num_tokens);
  data.topK = static_cast<int32_t>(top_k);
  data.numPaddedRows = static_cast<int32_t>(activation_output.size(0));

  ffi::CUDADeviceGuard device_guard(output.device().device_id);
  auto stream = get_stream(output.device());
  moe::dev::gatherActivation::run(data, stream);
}

}  // namespace flashinfer::trtllm_gen_gather_activation

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    trtllm_gen_moe_gather_activation,
    flashinfer::trtllm_gen_gather_activation::trtllm_gen_moe_gather_activation);
