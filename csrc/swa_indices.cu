// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Build per-token paged slot-id lists + window lengths for the sliding-window
// layers (compress_ratio == 0) of DSv4-Flash / GLM-5.1. Feeds the sparse-MLA
// paged-attention API's `indices` tensor for those layers. One block per row.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace flashinfer::swa_indices {

namespace {

constexpr int SWA_INDICES_THREADS = 128;

__global__ void compute_swa_indices_and_lens_kernel(
    int32_t* __restrict__ swa_indices, int swa_indices_stride, int32_t* __restrict__ swa_lens,
    int window_size, const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ seq_lens, const int32_t* __restrict__ token_to_req_indices,
    const bool* __restrict__ is_valid_token, const int32_t* __restrict__ block_table,
    int block_table_stride, int block_size, int token_offset) {
  const int pid = blockIdx.x;
  const int tid = threadIdx.x;
  const int token_idx = pid + token_offset;

  __shared__ int s_swa_len;
  __shared__ int s_start_pos;
  __shared__ int s_req_idx;
  __shared__ bool s_is_valid;

  if (tid == 0) {
    s_is_valid = is_valid_token[token_idx];
    if (!s_is_valid) {
      swa_lens[pid] = 0;
      s_swa_len = 0;
      s_start_pos = 0;
      s_req_idx = 0;
    } else {
      const int req_idx = token_to_req_indices[token_idx];
      const int query_start = query_start_loc[req_idx];
      const int query_end = query_start_loc[req_idx + 1];
      const int query_len = query_end - query_start;
      const int seq_len = seq_lens[req_idx];
      const int prefix_len = seq_len - query_len;

      const int pos = prefix_len + token_idx - query_start;
      const int start_pos = max(pos - window_size + 1, 0);
      const int swa_len = (pos + 1) - start_pos;

      swa_lens[pid] = swa_len;
      s_req_idx = req_idx;
      s_start_pos = start_pos;
      s_swa_len = swa_len;
    }
  }
  __syncthreads();

  if (!s_is_valid) return;

  const int req_idx = s_req_idx;
  const int start_pos = s_start_pos;
  const int swa_len = s_swa_len;

  int32_t* out_row = swa_indices + static_cast<size_t>(pid) * swa_indices_stride;
  const int32_t* block_row = block_table + static_cast<size_t>(req_idx) * block_table_stride;

  for (int offset = tid; offset < window_size; offset += SWA_INDICES_THREADS) {
    const int pos_offset = start_pos + offset;
    int32_t slot;
    if (offset < swa_len) {
      const int block_idx = pos_offset / block_size;
      const int block_off = pos_offset % block_size;
      const int32_t block_number = block_row[block_idx];
      slot = block_number * block_size + block_off;
    } else {
      slot = -1;
    }
    out_row[offset] = slot;
  }
}

}  // namespace

// Tensor / dtype validation lives in the TVM-FFI binding.
void launch_compute_swa_indices_and_lens(int32_t* swa_indices, int swa_indices_stride,
                                         int32_t* swa_lens, int window_size,
                                         const int32_t* query_start_loc, const int32_t* seq_lens,
                                         const int32_t* token_to_req_indices,
                                         const bool* is_valid_token, const int32_t* block_table,
                                         int block_table_stride, int block_size, int token_offset,
                                         int num_tokens, cudaStream_t stream) {
  if (num_tokens <= 0) return;
  compute_swa_indices_and_lens_kernel<<<num_tokens, SWA_INDICES_THREADS, 0, stream>>>(
      swa_indices, swa_indices_stride, swa_lens, window_size, query_start_loc, seq_lens,
      token_to_req_indices, is_valid_token, block_table, block_table_stride, block_size,
      token_offset);
}

}  // namespace flashinfer::swa_indices
