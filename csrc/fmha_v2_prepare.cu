/*
 * Copyright (c) 2023-2025 by FlashInfer team.
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

// GPU preparation kernel for fmha_v2.
// Replaces CPU-side D2H transfers, get_seq_lens(), Python for-loop block table
// construction, and H2D copies with a single GPU kernel + one 16-byte D2H.

#include <cuda_runtime.h>

#include "tvm_ffi_utils.h"

namespace ffi = tvm::ffi;

////////////////////////////////////////////////////////////////////////////////////////////////////
// fmha_v2_prepare_kernel
//
// Each thread handles one request in the batch.
// Computes:
//   1. kv_lens[i] = max(num_pages_i - 1, 0) * page_size + last_page_len[i]
//   2. q_len[i] for max reduction
//   3. block_tables[i, 0..num_pages_i)
//   4. metadata: [max_q_len, max_kv_len, total_num_rows, max_blocks_per_seq]
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void fmha_v2_prepare_kernel(
    const int32_t* __restrict__ qo_indptr,              // [B+1]
    const int32_t* __restrict__ paged_kv_indptr,         // [B+1]
    const int32_t* __restrict__ paged_kv_last_page_len,  // [B]
    const int32_t* __restrict__ paged_kv_indices,        // [total_pages]
    int32_t* __restrict__ kv_lens,                       // [B] output
    int32_t* __restrict__ block_tables,                  // [B * max_blocks_per_seq] output
    int32_t* __restrict__ metadata,                      // [4] output
    int page_size, int batch_size, int max_blocks_per_seq) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= batch_size) return;

  // 1. Compute kv_lens
  int num_pages_i = paged_kv_indptr[i + 1] - paged_kv_indptr[i];
  int kv_len_i = max(num_pages_i - 1, 0) * page_size + paged_kv_last_page_len[i];
  kv_lens[i] = kv_len_i;

  // 2. Compute q_len for max reduction
  int q_len_i = qo_indptr[i + 1] - qo_indptr[i];

  // 3. Fill block_tables row i
  int block_start = paged_kv_indptr[i];
  for (int j = 0; j < num_pages_i && j < max_blocks_per_seq; j++) {
    block_tables[i * max_blocks_per_seq + j] = paged_kv_indices[block_start + j];
  }

  // 4. Atomically reduce max_q_len, max_kv_len
  atomicMax(&metadata[0], q_len_i);
  atomicMax(&metadata[1], kv_len_i);

  // Thread 0 writes total_num_rows and max_blocks_per_seq
  if (i == 0) {
    metadata[2] = qo_indptr[batch_size];
    metadata[3] = max_blocks_per_seq;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// TVM FFI wrapper
////////////////////////////////////////////////////////////////////////////////////////////////////

void fmha_v2_prepare(ffi::TensorView qo_indptr, ffi::TensorView paged_kv_indptr,
                      ffi::TensorView paged_kv_last_page_len, ffi::TensorView paged_kv_indices,
                      ffi::TensorView kv_lens_out, ffi::TensorView block_tables_out,
                      ffi::TensorView metadata_out, int page_size, int batch_size,
                      int max_blocks_per_seq) {
  cudaStream_t stream = static_cast<cudaStream_t>(get_stream(qo_indptr.device()));

  // Zero metadata before atomicMax
  cudaMemsetAsync(metadata_out.data_ptr(), 0, 4 * sizeof(int32_t), stream);

  int threads = min(batch_size, 1024);
  int blocks = (batch_size + threads - 1) / threads;

  fmha_v2_prepare_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const int32_t*>(qo_indptr.data_ptr()),
      static_cast<const int32_t*>(paged_kv_indptr.data_ptr()),
      static_cast<const int32_t*>(paged_kv_last_page_len.data_ptr()),
      static_cast<const int32_t*>(paged_kv_indices.data_ptr()),
      static_cast<int32_t*>(kv_lens_out.data_ptr()),
      static_cast<int32_t*>(block_tables_out.data_ptr()),
      static_cast<int32_t*>(metadata_out.data_ptr()), page_size, batch_size, max_blocks_per_seq);
}
