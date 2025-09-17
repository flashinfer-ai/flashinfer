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

#include "pytorch_extension_utils.h"

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
                 at::Tensor semaphores, at::Tensor scratch);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  // "XQA Wrapper"
  m.def("xqa_wrapper", xqa_wrapper);
}
