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

void xqa_wrapper(int64_t multiProcessorCount, int64_t nbKHeads, int64_t slidingWinSize,
                 double qScale, Tensor output,
#if LOW_PREC_OUTPUT
                 Tensor rcpOutScale,
#endif
                 Tensor q, Tensor attentionSinks, Tensor pool, Tensor kvCachePageList,
                 int64_t maxSeqLen, Tensor seqLen, int64_t batchSize, Tensor kvCacheScale,
#if SPEC_DEC
                 int64_t qSeqLen, Tensor qCuSeqLens, Tensor mask,
#endif
                 Tensor semaphores, Tensor scratch) {
  auto stream = get_stream(output->device);
  float const* attentionSinksPtr =
      attentionSinks.defined() ? reinterpret_cast<float const*>(attentionSinks->data) : nullptr;

  launchMHAFlashInfer(multiProcessorCount, nbKHeads, slidingWinSize, qScale,
                      reinterpret_cast<OutputHead*>(output->data),
#if LOW_PREC_OUTPUT
                      reinterpret_cast<float const*>(rcpOutScale->data),
#endif
                      reinterpret_cast<InputHead const*>(q->data), attentionSinksPtr,
                      reinterpret_cast<GMemCacheHead*>(pool->data),
                      reinterpret_cast<KVCachePageIndex const*>(kvCachePageList->data), maxSeqLen,
                      reinterpret_cast<uint32_t const*>(seqLen->data), batchSize,
                      reinterpret_cast<float const*>(kvCacheScale->data),
#if SPEC_DEC
                      qSeqLen, reinterpret_cast<uint32_t const*>(qCuSeqLens->data),
                      reinterpret_cast<MaskType const*>(mask->data),
#endif
                      reinterpret_cast<uint32_t*>(semaphores->data),
                      reinterpret_cast<void*>(scratch->data), stream);
}
