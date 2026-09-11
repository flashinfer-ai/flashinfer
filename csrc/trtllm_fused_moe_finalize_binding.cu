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

namespace flashinfer::trtllm_gen_finalize {

namespace tg = batchedGemm::trtllm::gen;
using tvm::ffi::Optional;
using tvm::ffi::TensorView;

/*!
 * \brief Standalone TVM-FFI entry point for the trtllm-gen MoE finalize stage.
 *
 * Runs the same moe::dev::finalize kernels the fused MoE launchers invoke when
 * do_finalize=true, so a caller holding the do_finalize=false outputs
 * (gemm2_output, expert_weights, expanded_idx_to_permuted_idx) can complete
 * the token combine as a separate op — optionally fusing in a per-token delta
 * (e.g. a LoRA down-projection delta) that would otherwise cost an extra
 * full-output read-modify-write pass.
 *
 *   gemm2_output                : [num_padded, hidden_padded] bfloat16 or float16,
 *                                 permuted FC2 output rows
 *   expert_weights              : [num_tokens, top_k] bfloat16 or float32
 *   expanded_idx_to_permuted_idx: num_tokens * top_k int32 elements, -1 marks
 *                                 inactive slots (e.g. experts outside the local
 *                                 expert-parallel shard)
 *   output                      : [num_tokens, hidden] same dtype as gemm2_output
 *   lora_delta (optional)       : [num_tokens, top_k, hidden] per-slot rows, or
 *                                 [num_tokens, hidden] pre-combined; same dtype
 *                                 as output. Rows are addressed by the expanded
 *                                 (token, slot) index, NOT the permutation, and
 *                                 are accumulated unconditionally — the producer
 *                                 must zero-fill rows of inactive slots.
 *   lora_delta_scale            : scalar factor applied to the delta sum (e.g.
 *                                 routed_scaling_factor)
 *   lora_apply_expert_weights   : multiply each per-slot delta row by its
 *                                 expert weight (requires the 3D delta layout)
 *
 * out[t, h] = sum_k w[t, k] * gemm2_output[perm(t, k), h]      (skipping -1)
 *           + lora_delta_scale * sum_k s[t, k] * delta[t, k, h] (all k)
 * with s[t, k] = w[t, k] if lora_apply_expert_weights else 1.
 *
 * PDL note: with enable_pdl=true the vectorized kernel prefetches expert_weights
 * and expanded_idx_to_permuted_idx BEFORE its grid-dependency sync (same as the
 * fused pipeline, where routing completed several kernels earlier). If either
 * tensor is produced by the immediately preceding kernel on the stream, pass
 * enable_pdl=false.
 */
void trtllm_gen_moe_finalize(TensorView gemm2_output, TensorView expert_weights,
                             TensorView expanded_idx_to_permuted_idx, TensorView output,
                             Optional<TensorView> lora_delta, double lora_delta_scale,
                             bool lora_apply_expert_weights, bool enable_pdl) {
  CHECK_INPUT(gemm2_output);
  CHECK_INPUT(expert_weights);
  CHECK_INPUT(expanded_idx_to_permuted_idx);
  CHECK_INPUT(output);
  CHECK_DEVICE(gemm2_output, output);
  CHECK_DEVICE(expert_weights, output);
  CHECK_DEVICE(expanded_idx_to_permuted_idx, output);

  CHECK_DIM(2, gemm2_output);
  CHECK_DIM(2, expert_weights);
  CHECK_DIM(2, output);

  TVM_FFI_ICHECK(output.dtype() == dl_bfloat16 || output.dtype() == dl_float16)
      << "output must be bfloat16 or float16";
  TVM_FFI_ICHECK(gemm2_output.dtype() == output.dtype())
      << "gemm2_output must have the same dtype as output";
  TVM_FFI_ICHECK(expert_weights.dtype() == dl_bfloat16 || expert_weights.dtype() == dl_float32)
      << "expert_weights must be bfloat16 or float32";
  TVM_FFI_ICHECK(expanded_idx_to_permuted_idx.dtype() == dl_int32)
      << "expanded_idx_to_permuted_idx must be int32";

  int64_t const num_tokens = output.size(0);
  int64_t const hidden_size = output.size(1);
  int64_t const hidden_size_padded = gemm2_output.size(1);
  int64_t const top_k = expert_weights.size(1);

  TVM_FFI_ICHECK_EQ(expert_weights.size(0), num_tokens)
      << "expert_weights dim0 must equal num_tokens";
  // The vectorized kernel caches the per-token weights/indices in MaxTopK-sized
  // shared-memory arrays; enforce its bound upfront rather than letting the
  // dispatch in finalize::run() turn top_k > 64 into a batch-size-dependent
  // runtime failure (mirrors the routing binding's upfront top_k validation).
  TVM_FFI_ICHECK(top_k >= 1 && top_k <= 64) << "top_k must be between 1 and 64, got " << top_k;
  TVM_FFI_ICHECK_EQ(expanded_idx_to_permuted_idx.numel(), num_tokens * top_k)
      << "expanded_idx_to_permuted_idx must have num_tokens * top_k elements";
  TVM_FFI_ICHECK_GE(hidden_size_padded, hidden_size)
      << "gemm2_output hidden dimension must be >= output hidden dimension";
  // Both finalize kernels may be dispatched; the vectorized one loads 128 bits
  // per thread and requires the (padded) hidden extents to be multiples of it
  // and every base pointer to be 16-byte aligned (fresh torch allocations are;
  // contiguous-but-offset views may not be).
  constexpr int64_t elems_per_128b = 128 / 16;  // both supported dtypes are 16-bit
  TVM_FFI_ICHECK(hidden_size % elems_per_128b == 0 && hidden_size_padded % elems_per_128b == 0)
      << "hidden dimensions must be multiples of " << elems_per_128b << ", got " << hidden_size
      << " and " << hidden_size_padded;
  auto check_aligned = [](void const* ptr, char const* name) {
    TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(ptr) % 16 == 0)
        << name << " must be 16-byte aligned";
  };
  check_aligned(gemm2_output.data_ptr(), "gemm2_output");
  check_aligned(expert_weights.data_ptr(), "expert_weights");
  check_aligned(expanded_idx_to_permuted_idx.data_ptr(), "expanded_idx_to_permuted_idx");
  check_aligned(output.data_ptr(), "output");

  void const* lora_delta_ptr = nullptr;
  int64_t lora_top_k = 0;
  if (lora_delta.has_value()) {
    auto const& delta = lora_delta.value();
    CHECK_INPUT(delta);
    CHECK_DEVICE(delta, output);
    TVM_FFI_ICHECK(delta.dtype() == output.dtype())
        << "lora_delta must have the same dtype as output";
    TVM_FFI_ICHECK(delta.ndim() == 2 || delta.ndim() == 3)
        << "lora_delta must be [num_tokens, hidden] or [num_tokens, top_k, hidden]";
    TVM_FFI_ICHECK_EQ(delta.size(0), num_tokens) << "lora_delta dim0 must equal num_tokens";
    TVM_FFI_ICHECK_EQ(delta.size(delta.ndim() - 1), hidden_size)
        << "lora_delta last dim must equal the output hidden dimension";
    lora_top_k = delta.ndim() == 3 ? delta.size(1) : 1;
    if (delta.ndim() == 3) {
      TVM_FFI_ICHECK_EQ(delta.size(1), top_k) << "3D lora_delta dim1 must equal top_k";
    }
    check_aligned(delta.data_ptr(), "lora_delta");
    lora_delta_ptr = delta.data_ptr();
  }
  TVM_FFI_ICHECK(!lora_apply_expert_weights || lora_top_k == top_k)
      << "lora_apply_expert_weights requires the [num_tokens, top_k, hidden] delta layout";

  // Empty batch: nothing to combine, and a zero grid dimension is a CUDA error.
  if (num_tokens == 0) {
    return;
  }

  moe::dev::finalize::Data data;
  data.mDtypeElt = output.dtype() == dl_float16 ? tg::Dtype::Fp16 : tg::Dtype::Bfloat16;
  data.mDtypeExpW = expert_weights.dtype() == dl_float32 ? tg::Dtype::Fp32 : tg::Dtype::Bfloat16;
  data.mUsePdl = enable_pdl;
  data.mUseDeepSeekFp8 = false;
  data.inPtr = const_cast<void*>(gemm2_output.data_ptr());
  data.outPtr = output.data_ptr();
  data.inDqSfsPtr = nullptr;
  data.outDqSfsPtr = nullptr;
  data.expertWeightsPtr = const_cast<void*>(expert_weights.data_ptr());
  data.expandedIdxToPermutedIdx =
      static_cast<int32_t*>(const_cast<void*>(expanded_idx_to_permuted_idx.data_ptr()));
  data.numTokens = static_cast<int32_t>(num_tokens);
  data.numExperts = 0;  // unused by the non-DeepSeek finalize kernels
  data.topK = static_cast<int32_t>(top_k);
  data.hiddenDim = static_cast<int32_t>(hidden_size);
  data.hiddenDimPadded = static_cast<int32_t>(hidden_size_padded);
  data.totalNumPaddedTokens = nullptr;  // only read by the DeepSeek finalize kernel
  data.loraDeltaPtr = lora_delta_ptr;
  data.loraTopK = static_cast<int32_t>(lora_top_k);
  data.loraDeltaScale = static_cast<float>(lora_delta_scale);
  data.loraApplyExpertWeights = lora_apply_expert_weights;

  auto stream = get_stream(output.device());
  moe::dev::finalize::run(data, stream);
}

}  // namespace flashinfer::trtllm_gen_finalize

TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_gen_moe_finalize,
                              flashinfer::trtllm_gen_finalize::trtllm_gen_moe_finalize);
