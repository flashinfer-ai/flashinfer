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

#include "../pytorch_extension_utils.h"
#include "mha.h"

void xqa_wrapper(int64_t multiProcessorCount, int64_t nbKHeads, int64_t slidingWinSize,
                 double qScale, at::Tensor output,
#if LOW_PREC_OUTPUT
                 at::Tensor rcpOutScale,
#endif
                 at::Tensor q, at::Tensor attentionSinks, at::Tensor pool,
                 at::Tensor kvCachePageList, int64_t maxSeqLen, at::Tensor seqLen,
                 int64_t batchSize, at::Tensor kvCacheScale,
#if SPEC_DEC
                 int64_t qSeqLen, at::Tensor qCuSeqLens, at::Tensor mask,
#endif
                 at::Tensor semaphores, at::Tensor scratch) {
  auto stream = at::cuda::getCurrentCUDAStream();
  float const* attentionSinksPtr = attentionSinks.defined()
                                       ? reinterpret_cast<float const*>(attentionSinks.data_ptr())
                                       : nullptr;

  launchMHAFlashInfer(multiProcessorCount, nbKHeads, slidingWinSize, qScale,
                      reinterpret_cast<OutputHead*>(output.data_ptr()),
#if LOW_PREC_OUTPUT
                      reinterpret_cast<float const*>(rcpOutScale.data_ptr()),
#endif
                      reinterpret_cast<InputHead const*>(q.data_ptr()), attentionSinksPtr,
                      reinterpret_cast<GMemCacheHead*>(pool.data_ptr()),
                      reinterpret_cast<KVCachePageIndex const*>(kvCachePageList.data_ptr()),
                      maxSeqLen, reinterpret_cast<uint32_t const*>(seqLen.data_ptr()), batchSize,
                      reinterpret_cast<float const*>(kvCacheScale.data_ptr()),
#if SPEC_DEC
                      qSeqLen, reinterpret_cast<uint32_t const*>(qCuSeqLens.data_ptr()),
                      reinterpret_cast<MaskType const*>(mask.data_ptr()),
#endif
                      reinterpret_cast<uint32_t*>(semaphores.data_ptr()),
                      reinterpret_cast<void*>(scratch.data_ptr()), stream);
}
