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

#if MLA_WRAPPER
void xqa_wrapper_mla(int64_t multiProcessorCount, double qScale,
                     tvm::ffi::Optional<TensorView> qScaleTensor, TensorView output, TensorView q,
                     TensorView kCacheVLLM, TensorView vCacheVLLM, TensorView kvCachePageList,
                     int64_t maxSeqLen, TensorView seqLen, int64_t batchSize, double kvCacheScale,
                     tvm::ffi::Optional<TensorView> kvScaleTensor, TensorView semaphores,
                     TensorView scratch, bool enable_pdl);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(xqa_wrapper_mla, xqa_wrapper_mla);

#else

void xqa_wrapper(bool run_sm90_fp8_mha, int64_t multiProcessorCount, int64_t nbKHeads,
                 int64_t slidingWinSize, double qScale, tvm::ffi::Optional<TensorView> qScaleTensor,
                 TensorView output, double rcpOutScale, TensorView q,
                 tvm::ffi::Optional<TensorView> attentionSinks, TensorView kCacheVLLM,
                 TensorView vCacheVLLM, TensorView kvCachePageList, int64_t maxSeqLen,
                 TensorView seqLen, int64_t batchSize, double kvCacheScale,
                 tvm::ffi::Optional<TensorView> kvScaleTensor, int64_t qSeqLen,
                 tvm::ffi::Optional<TensorView> mask, TensorView semaphores, TensorView scratch,
                 bool enable_pdl);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(xqa_wrapper, xqa_wrapper);

#endif
