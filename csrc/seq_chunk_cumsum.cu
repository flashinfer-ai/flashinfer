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
#include "flashinfer/mamba/seq_chunk_cumsum.cuh"
#include "tvm_ffi_utils.h"

using namespace flashinfer::mamba;

void seq_chunk_cumsum(TensorView seq_idx, TensorView chunk_indices, TensorView chunk_offsets,
                      TensorView output, int64_t chunk_size, int64_t num_logical_chunks,
                      int64_t num_seqs) {
  CHECK_INPUT(seq_idx);
  CHECK_INPUT(chunk_indices);
  CHECK_INPUT(chunk_offsets);
  CHECK_INPUT(output);

  auto stream = get_stream(seq_idx.device());

  cudaError_t status;
  DISPATCH_DLPACK_IDTYPE_TO_CTYPE(seq_idx.dtype(), SeqIdxT, [&] {
    status = SeqChunkCumsumLauncher(
        static_cast<const SeqIdxT*>(seq_idx.data_ptr()),
        static_cast<const int32_t*>(chunk_indices.data_ptr()),
        static_cast<const int32_t*>(chunk_offsets.data_ptr()),
        static_cast<int32_t*>(output.data_ptr()), static_cast<int>(chunk_size),
        static_cast<int>(num_logical_chunks), static_cast<int>(num_seqs), stream);
    return true;
  });

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "SeqChunkCumsumLauncher failed: " << cudaGetErrorString(status);
}
