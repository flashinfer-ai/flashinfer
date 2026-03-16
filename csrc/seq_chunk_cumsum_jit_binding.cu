/*
 * Copyright (c) 2025 by FlashInfer team.
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

using tvm::ffi::Optional;

void seq_chunk_cumsum(TensorView seq_idx, TensorView chunk_indices, TensorView chunk_offsets,
                      TensorView output, Optional<TensorView> tile_state, int64_t chunk_size,
                      int64_t num_logical_chunks, int64_t num_seqs);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(seq_chunk_cumsum, seq_chunk_cumsum);
