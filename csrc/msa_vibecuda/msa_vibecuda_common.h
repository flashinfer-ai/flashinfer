/*
 * Copyright (c) 2026 by FlashInfer team.
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

// Shared host-side declarations for the VibeCUDA MSA backend translation
// units. The device kernels in this directory are pure CUDA (no framework
// headers); only this header's raw-pointer forward declarations cross a TU
// boundary. Output and every scratch buffer are caller-allocated so the
// binding performs no implicit allocations.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#include <stdexcept>
#include <string>

#define MSAV_CHECK(cond, ...)                                              \
  do {                                                                     \
    if (!(cond)) {                                                         \
      char msav_msg_[512];                                                 \
      snprintf(msav_msg_, sizeof(msav_msg_), __VA_ARGS__);                 \
      throw std::runtime_error(std::string("msa_vibecuda: ") + msav_msg_); \
    }                                                                      \
  } while (0)

namespace msa_vibecuda {

// General per-token / packed-pair kernel parameters (msa_vibecuda_core.cu).
// The persistent-work-queue fields (ws_next/ws_total/ws_ntiles) are a
// compile-time-disabled A/B mechanism (MSA_PERSIST_WAVES=0); every dispatch
// passes ws_next==nullptr.
struct CoreParams {
  const void* __restrict__ q;
  const int* __restrict__ q2k;
  const int* __restrict__ cu_q;
  const int* __restrict__ cu_k;
  const int* __restrict__ page_table;
  void* __restrict__ out;
  long pt_stride;
  long q_tok, q_head;
  long o_tok, o_head;
  long q2k_h, q2k_n;
  int total_q, num_q_heads, num_kv_heads, group, topk, nbatch;
  int seqlen_q, causal, pack_T;  // pack_T: tokens per packed CTA (1 = unpacked)
  int* ws_next;                  // PERSIST work-queue head (nullptr: disabled)
  int ws_total;                  // PERSIST work items = ntiles * num_kv_heads
  int ws_ntiles;                 // PERSIST tiles per kv head (work id -> (tile, head) map)
  float scale_log2e;             // head_dim**-0.5 * log2(e)
};

// KV layout descriptor for the host-side TMA maps in the core TU. K and V
// share one layout by contract; element strides are in elements.
struct KvLayout {
  long d0;      // tokens (flat) or pages (paged)
  long d1;      // num_kv_heads
  long s0, s1;  // strides of dim0/dim1
  long s2;      // paged only: stride of the 128-token page dimension
};

}  // namespace msa_vibecuda

// Fallback launcher for the general per-token / packed pair kernels.
// kv_kind: 0 = bf16, 1 = fp16, 2 = fp8-e4m3 (q always bf16 then).
namespace msa_vibecuda_core {

void core_forward(const msa_vibecuda::CoreParams& p, const msa_vibecuda::KvLayout& kv,
                  const void* k, const void* v, bool q_is_bf16, int kv_kind, bool paged,
                  cudaStream_t stream);

}  // namespace msa_vibecuda_core

// Warp-specialized UMMA/TMEM prefill path (group_size==16, flat dense KV).
namespace msa_umma_g16 {

bool umma_g16_eligible(int group, int seqlen_q, int topk, int kv_dtype_code, bool paged,
                       bool causal_supported);

// q_is_bf16 selects the BF16 or FP16 template instantiation. This route is
// flat-only: k/v are [total_k, num_kv_heads, 128] dense tensors.
void umma_g16_forward(const void* q, bool q_is_bf16, const void* k, const void* v, const int* q2k,
                      const int* cu_q, const int* cu_k, void* out, int total_q, int total_k,
                      int num_q_heads, int num_kv_heads, int topk, int nbatch, bool causal,
                      cudaStream_t stream);

}  // namespace msa_umma_g16

// Round-24 block-bucketed UMMA/TMEM path (group_size==4, paged KV only).
namespace msa_umma_g4 {

bool umma_g4_eligible(int group, bool paged, int kv_dtype_code, int topk, int nbatch, int max_pages,
                      int num_kv_heads, int total_q);

// Scratch layout (ints, then floats) is defined by the routing math in
// umma_g4_forward; the caller must provide at least
// msa_vibecuda_g4_workspace_ints/floats elements (see the Python wrapper,
// which mirrors those exact formulas).
void umma_g4_forward(const void* q, bool q_is_bf16, const void* k, const void* v, const int* q2k,
                     const int* cu_q, const int* cu_k, const int* page_table, void* out,
                     int total_q, int num_q_heads, int num_kv_heads, int topk, int nbatch,
                     int num_pages, int max_pages, long pt_stride, long q_tok, long q_head,
                     long o_tok, long o_head, int* ws_int, float* ws_float, int seqlen_q,
                     bool causal, cudaStream_t stream);

}  // namespace msa_umma_g4
