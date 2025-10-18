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

#include "tvm_ffi_utils.h"

void xqa_wrapper(int64_t multiProcessorCount, int64_t nbKHeads, int64_t slidingWinSize,
                 double qScale, TensorView output,
#if LOW_PREC_OUTPUT
                 TensorView rcpOutScale,
#endif
                 TensorView q, TensorView attentionSinks, TensorView pool,
                 TensorView kvCachePageList, int64_t maxSeqLen, TensorView seqLen,
                 int64_t batchSize, TensorView kvCacheScale,
#if SPEC_DEC
                 int64_t qSeqLen, TensorView qCuSeqLens, TensorView mask,
#endif
                 TensorView semaphores, TensorView scratch);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(xqa_wrapper, xqa_wrapper);
