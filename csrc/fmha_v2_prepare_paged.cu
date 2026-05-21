/*
 * Copyright (c) 2023-2026 by FlashInfer team.
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

// Paged KV prep: derive seq_lens_q, kv_lens and dense block_table from the
// sparse indptr/indices representation. Feeds into fmha_v2_prepare afterwards.

#include <cuda_runtime.h>

#include "tvm_ffi_utils.h"

namespace ffi = tvm::ffi;

namespace {

constexpr int kThreadsPerBlock = 256;

__global__ void fmha_v2_prepare_paged_kernel(
    const int32_t* __restrict__ qo_indptr,           // [B+1]
    const int32_t* __restrict__ paged_kv_indptr,     // [B+1]
    const int32_t* __restrict__ paged_kv_last_page_len,  // [B]
    const int32_t* __restrict__ paged_kv_indices,    // [total_pages]
    int32_t* __restrict__ seq_lens_q,                // [B] output
    int32_t* __restrict__ kv_lens,                   // [B] output
    int32_t* __restrict__ block_tables,              // [B, max_blocks_per_seq] output
    int page_size,
    int batch_size,
    int max_blocks_per_seq) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= batch_size) return;

  // kv_len = (num_pages - 1) * page_size + last_page_len
  int num_pages_i = paged_kv_indptr[i + 1] - paged_kv_indptr[i];
  int kv_len_i = max(num_pages_i - 1, 0) * page_size + paged_kv_last_page_len[i];
  kv_lens[i] = kv_len_i;

  seq_lens_q[i] = qo_indptr[i + 1] - qo_indptr[i];

  // scatter page indices into the dense block table row
  int block_start = paged_kv_indptr[i];
  int row_offset = i * max_blocks_per_seq;
  for (int j = 0; j < num_pages_i && j < max_blocks_per_seq; j++) {
    block_tables[row_offset + j] = paged_kv_indices[block_start + j];
  }
}

}  // namespace

// TVM-FFI entry point.
void fmha_v2_prepare_paged(
    ffi::TensorView qo_indptr,
    ffi::TensorView paged_kv_indptr,
    ffi::TensorView paged_kv_last_page_len,
    ffi::TensorView paged_kv_indices,
    ffi::TensorView seq_lens_q_out,
    ffi::TensorView kv_lens_out,
    ffi::TensorView block_tables_out,
    int page_size, int batch_size, int max_blocks_per_seq) {
  cudaStream_t stream = static_cast<cudaStream_t>(get_stream(qo_indptr.device()));

  int threads = min(batch_size, kThreadsPerBlock);
  int blocks = (batch_size + threads - 1) / threads;

  fmha_v2_prepare_paged_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const int32_t*>(qo_indptr.data_ptr()),
      static_cast<const int32_t*>(paged_kv_indptr.data_ptr()),
      static_cast<const int32_t*>(paged_kv_last_page_len.data_ptr()),
      static_cast<const int32_t*>(paged_kv_indices.data_ptr()),
      static_cast<int32_t*>(seq_lens_q_out.data_ptr()),
      static_cast<int32_t*>(kv_lens_out.data_ptr()),
      static_cast<int32_t*>(block_tables_out.data_ptr()),
      page_size, batch_size, max_blocks_per_seq);
}
