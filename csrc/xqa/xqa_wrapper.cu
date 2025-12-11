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

#include "../tvm_ffi_utils.h"
#include "mha.h"

using tvm::ffi::Optional;

#if MLA_WRAPPER
void xqa_wrapper_mla(int64_t multiProcessorCount, double qScale, Optional<TensorView> qScaleTensor,
                     TensorView output, TensorView q, TensorView kCacheVLLM, TensorView vCacheVLLM,
                     TensorView kvCachePageList, int64_t maxSeqLen, TensorView seqLen,
                     int64_t batchSize, double kvCacheScale, Optional<TensorView> kvScaleTensor,
                     TensorView semaphores, TensorView scratch, bool enable_pdl) {
  auto stream = get_stream(output.device());
  float const* qScalePtr = qScaleTensor.has_value()
                               ? reinterpret_cast<float const*>(qScaleTensor.value().data_ptr())
                               : nullptr;
  float const* kvScalePtr = kvScaleTensor.has_value()
                                ? reinterpret_cast<float const*>(kvScaleTensor.value().data_ptr())
                                : nullptr;
  // Extract strides from TensorView (in elements, not bytes)
  uint64_t kv_stride_page = kCacheVLLM.stride(0);
  uint64_t kv_stride_token = kCacheVLLM.stride(-2);
  uint64_t kv_stride_head = kCacheVLLM.stride(-3);

  launchMLAFlashInfer(multiProcessorCount, 1, qScale, qScalePtr,
                      reinterpret_cast<OutputHead*>(output.data_ptr()),
                      reinterpret_cast<InputHead const*>(q.data_ptr()),
                      reinterpret_cast<GMemCacheHead*>(kCacheVLLM.data_ptr()),
                      reinterpret_cast<GMemCacheHead*>(vCacheVLLM.data_ptr()),
                      reinterpret_cast<KVCachePageIndex const*>(kvCachePageList.data_ptr()),
                      maxSeqLen, reinterpret_cast<uint32_t const*>(seqLen.data_ptr()), batchSize,
                      kvCacheScale, kvScalePtr, reinterpret_cast<uint32_t*>(semaphores.data_ptr()),
                      reinterpret_cast<void*>(scratch.data_ptr()), enable_pdl, kv_stride_page,
                      kv_stride_token, kv_stride_head, stream);
}
#else

void xqa_wrapper(bool run_sm90_fp8_mha, int64_t multiProcessorCount, int64_t nbKHeads,
                 int64_t slidingWinSize, double qScale, Optional<TensorView> qScaleTensor,
                 TensorView output, double rcpOutScale, TensorView q,
                 Optional<TensorView> attentionSinks, TensorView kCacheVLLM, TensorView vCacheVLLM,
                 TensorView kvCachePageList, int64_t maxSeqLen, TensorView seqLen,
                 int64_t batchSize, double kvCacheScale, Optional<TensorView> kvScaleTensor,
                 int64_t qSeqLen, Optional<TensorView> mask, TensorView semaphores,
                 TensorView scratch, bool enable_pdl) {
  auto stream = get_stream(output.device());
  float const* attentionSinksPtr =
      attentionSinks.has_value() ? reinterpret_cast<float const*>(attentionSinks.value().data_ptr())
                                 : nullptr;
  float const* qScalePtr = qScaleTensor.has_value()
                               ? reinterpret_cast<float const*>(qScaleTensor.value().data_ptr())
                               : nullptr;
  float const* kvScalePtr = kvScaleTensor.has_value()
                                ? reinterpret_cast<float const*>(kvScaleTensor.value().data_ptr())
                                : nullptr;
#if USE_SM90_MHA
  auto const mha_func = run_sm90_fp8_mha ? &launchHopperF8MHAFlashInfer : &launchMHAFlashInfer;
#else
  auto const mha_func = &launchMHAFlashInfer;
#endif

  // Extract strides from TensorView (in elements, not bytes)
  uint64_t kv_stride_page = kCacheVLLM.stride(0);
  uint64_t kv_stride_token = kCacheVLLM.stride(-3);
  uint64_t kv_stride_head = kCacheVLLM.stride(-2);

#if SPEC_DEC
  MaskType const* maskPtr =
      mask.has_value() ? reinterpret_cast<MaskType const*>(mask.value().data_ptr()) : nullptr;
#endif

  mha_func(multiProcessorCount, nbKHeads, slidingWinSize, qScale, qScalePtr,
           reinterpret_cast<OutputHead*>(output.data_ptr()),
#if LOW_PREC_OUTPUT
           rcpOutScale,
#endif
           reinterpret_cast<InputHead const*>(q.data_ptr()), attentionSinksPtr,
           reinterpret_cast<GMemCacheHead*>(kCacheVLLM.data_ptr()),
           reinterpret_cast<GMemCacheHead*>(vCacheVLLM.data_ptr()),
           reinterpret_cast<KVCachePageIndex const*>(kvCachePageList.data_ptr()), maxSeqLen,
           reinterpret_cast<uint32_t const*>(seqLen.data_ptr()), batchSize, kvCacheScale,
           kvScalePtr,
#if SPEC_DEC
           qSeqLen, nullptr, maskPtr,
#endif
           reinterpret_cast<uint32_t*>(semaphores.data_ptr()),
           reinterpret_cast<void*>(scratch.data_ptr()), enable_pdl, kv_stride_page, kv_stride_token,
           kv_stride_head, stream);
}
#endif
