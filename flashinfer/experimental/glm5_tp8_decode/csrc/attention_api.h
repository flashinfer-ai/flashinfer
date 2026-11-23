/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "common.h"

namespace flashinfer::glm5_tp8 {

// Named q4 MLA inputs. q1 uses the prepared descriptor because its scratch
// layout differs. RoPE is the full [capacity+8,64] table, indexed on the device.
struct MlaArguments {
  const uint16_t* x;
  const float* x_gamma;
  const float* q_gamma;
  const float* kv_gamma;
  const float* rope;
  const uint8_t* qkv_a_proj;
  const uint8_t* q_b_proj;
  const uint8_t* w_uk;
  const uint8_t* w_uv;
  const uint8_t* o_proj;
  uint16_t* kv_cache;
  uint16_t* pe_cache;
  int32_t* selected_ids;
  int32_t* selected_lengths;
  const int32_t* cur_pos;
  int32_t cache_rows;
  uint16_t* post_q;
  uint16_t* post_kv;
  uint16_t* post_qnd;
  uint16_t* post_qpe;
  uint16_t* post_qnope;
  uint16_t* post_o;
  uint16_t* post_uv;
  uint16_t* out;
  const uint8_t* ll_recv;
  uint32_t ll_tag;
};

struct AttentionArguments {
  int32_t rank;
  int32_t q_len;
  bool shared_indexer;       // q4 rank0: full indexing=false, shared index=true; ignored for q1
  const uint64_t* context;   // q4 rank0 or q1: immutable 8 x 128-byte descriptor
  uint8_t* peer_arenas[8];   // q4 arena bases, rank0 16 MiB and others >=8 MiB
  uint8_t* peer_records[8];  // q1 operator-owned 0x38000-byte record regions
  uint32_t* epoch_ctr;       // q4: workspace-owned u32[148], initially zero
  uint8_t* q1_state;         // q1: 1024 bytes, initially zero; never reset alone
  MlaArguments mla;          // q4 ranks1..7 only
};

// Logical rank is independent of the local CUDA device ordinal.
KernelResources initialize_attention(int q_len, int rank, int device);
cudaError_t launch_attention(const AttentionArguments& args, cudaStream_t stream, bool enable_pdl,
                             bool cooperative);

// Encodes a BF16 2D TMA map into host storage, outside capture. The caller
// uploads the resulting 128 bytes into a 64-byte-aligned descriptor allocation.
cudaError_t encode_attention_tmap(const void* base, int64_t rows, int64_t cols, int32_t box_rows,
                                  int32_t box_cols, void* out128);

}  // namespace flashinfer::glm5_tp8
