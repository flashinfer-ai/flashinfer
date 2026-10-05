/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
#pragma once

// Kernel ABI of the Cake StepFun fused-MoE stages other than FC1 (routing, FC2, the NVFP4 per-token
// requantization and finalize). The generated launch manifest
// (generated/cake_stepfun_generated_manifest.cuh) declares the exported kernels and renders, for
// every stage it covers, one `Submit_<i>` thunk per kernel that unpacks the stage's `*Args`
// structure below into the kernel's parameter list, plus a per-architecture table of `*KernelSpec`
// entries. The hand-written stage runners (cake_stepfun_stages.cu) consume only these structures
// and tables, so the host code is independent of a unit's parameter order and of whether an operand
// arrives as a TMA descriptor or as a pointer. README.md in this directory states the contract in
// prose.
//
// The FC1 stage keeps its own `Fc1Args` / `Fc1KernelSpec` inside the generated manifest.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "generated/cake_stepfun_generated_manifest.cuh"

namespace flashinfer::cake_stepfun::generated {

// Block-scale layout of a block-scaled operand, as the trtllm-gen kernels name them.
enum class SfLayout : int {
  kNone = 0,    // operand is not block-scaled
  kLinear = 1,  // row-major [rows, cols / vec]
  kR8c4 = 2,    // 8x4 swizzled blocks
  kR128c4 = 3,  // 128x4 swizzled blocks
};

// ------------------------------------------------------------------------------------------------
// Routing (Renormalize top-k over the routing logits; writes every table the GEMM stages consume).
// ------------------------------------------------------------------------------------------------
struct RoutingArgs {
  const void*
      routing_logits;  // [num_tokens, num_experts], dtype per RoutingKernelSpec::logits_dtype
  int* topk_packed;    // [num_tokens, top_k] packed (bf16 score, int16 expert) as the native kernel
                       // writes
  void* topk_weights;  // [num_tokens, top_k] bf16 expert weights
  int* expert_count_histogram;        // max(2 * num_experts, 512) int32 scratch
  int* total_num_padded_tokens;       // [1]
  int* expanded_idx_to_permuted_idx;  // [num_tokens * top_k], -1 when the expert is not local
  int* permuted_idx_to_token_idx;     // [max_padded_tokens + 1], -1 in padded slots
  int* cta_idx_xy_to_batch_idx;       // [max_num_ctas] local expert per CTA tile
  int* cta_idx_xy_to_mn_limit;        // [max_num_ctas]
  int* num_non_exiting_ctas;          // [1]
  int* num_tokens_per_expert;         // [num_experts] tokens routed to each expert, or nullptr
  int num_tokens;
  int num_experts;
  int top_k;
  int local_expert_offset;
  int local_num_experts;
  int tile_tokens_dim;
  int max_num_ctas;  // Routing::getMaxNumCtasInBatchDim(num_tokens, top_k, num_experts,
                     // tile_tokens_dim)
};

// How the host derives the launch grid of a routing kernel from the problem.
enum class RoutingGrid : int {
  kFixed = 0,        // grid = RoutingKernelSpec::grid (a fixed CTA count, e.g. the cluster kernel)
  kTokenBlocks = 1,  // grid.x = ceil(num_tokens / RoutingKernelSpec::tokens_per_cta)
};

using RoutingSubmitFn = cudaError_t (*)(const cudaLaunchConfig_t*, const RoutingArgs&);

struct RoutingKernelSpec {
  const char* symbol;
  // Logits dtype the kernel reads: 0 = float32, 1 = bfloat16.
  int logits_dtype;
  // Token range the kernel variant serves ([min_tokens, max_tokens]; the host picks the first
  // match), mirroring the native dispatcher's per-token-count kernel selection.
  int min_tokens;
  int max_tokens;
  RoutingGrid grid_rule;
  uint32_t grid[3];
  int tokens_per_cta;
  uint32_t block[3];
  uint32_t cluster[3];
  bool cluster_attribute;
  // True when the kernel writes benign entries into the routing tail [num_non_exiting_ctas,
  // max_num_ctas) (expert 0, mn_limit = tile * tile_tokens_dim, permuted slots -1), so no padding
  // kernel is needed before GEMM units that acquire surplus tiles through cluster launch control.
  bool writes_benign_tail;
  size_t dynamic_smem_bytes;
  ConfigureFn configure;
  RoutingSubmitFn submit;
};

// ------------------------------------------------------------------------------------------------
// FC2 (grouped GEMM over the permuted FC1 output, bf16 output in permuted order).
// ------------------------------------------------------------------------------------------------
// FC2 kernel families; the values equal the FC1 family values (Fc1Family) so one family index names
// a precision in every stage (families[name].index of the inventory). them).
enum Fc2Family : int {
  kFc2Nvfp4 = 0,          // E2m1 x E2m1, activation block scales from the FC1 epilogue
  kFc2Nvfp4PerToken = 1,  // E2m1 x E2m1 on the requantized FC1 output + fp32 per-token scales
  kFc2Bf16 = 2,           // bf16 activations x bf16 weights (BlockMajorK)
  kFc2Fp8PerTensor = 3,   // E4m3 x E4m3, per-expert output scale
  kFc2MxFp8 = 4,          // MxE4m3 x MxE4m3 with UE8M0 block scales
};

// Complete FC2 launch arguments. Tensor-map and pointer members of one operand coexist because
// kernels take either form; the manifest's Submit thunk passes the one the kernel declares.
struct Fc2Args {
  CUtensorMap A;        // expert weights
  CUtensorMap B_map;    // permuted activations
  CUtensorMap SFA;      // weight block scales
  CUtensorMap SFB_map;  // activation block scales
  CUtensorMap C_map;    // output
  void* B_ptr;
  void* SFB_ptr;
  void* C_ptr;
  float* per_token_scale;        // fp32 [max_padded_tokens] (kFc2Nvfp4PerToken), else nullptr
  int* tile_expert;              // cta_idx_xy_to_batch_idx
  int* tile_mn_limit;            // cta_idx_xy_to_mn_limit
  int* total_tiles;              // num_non_exiting_ctas
  int* total_num_padded_tokens;  // [1]
  int* work_counter;             // nullptr (cluster launch control)
  float* scale_c;                // per-expert output scale (output2_scales_scalar), or nullptr
  int N_out;                     // hidden size (output columns)
  int K;                         // intermediate size
  int grid_m;                    // N_out / output_rows_per_cta
  int grid_n;                    // max_num_ctas (static worst case)
  int K_tiles;                   // K / block_k
};

using Fc2SubmitFn = cudaError_t (*)(const cudaLaunchConfig_t*, const Fc2Args&);

struct Fc2KernelSpec {
  const char* symbol;
  int family;
  int tile_n;
  int output_rows_per_cta;
  int block_k;
  // Split-K factor: the launch grid is (grid_m, grid_n, split_k); 1 when the kernel does not
  // split K across a cluster.
  int split_k;
  // Block-scale layout the kernel reads for its activation operand (the requantization stage must
  // write this layout for kFc2Nvfp4PerToken; the FC1 epilogue writes kR8c4 for kFc2Nvfp4).
  SfLayout sf_layout_a;
  uint32_t block[3];
  uint32_t cluster[3];
  bool cluster_attribute;
  // Same meaning as Fc1KernelSpec::bounds_acquired_tiles.
  bool bounds_acquired_tiles;
  size_t dynamic_smem_bytes;
  EncodeTensorMapFn encode_a;
  EncodeTensorMapFn encode_b;
  EncodeTensorMapFn encode_sfa;
  EncodeTensorMapFn encode_sfb;
  EncodeTensorMapFn encode_c;
  ConfigureFn configure;
  Fc2SubmitFn submit;
};

// ------------------------------------------------------------------------------------------------
// NVFP4 per-token requantization of the bf16 FC1 output (kFc1Nvfp4PerToken -> kFc2Nvfp4PerToken).
// ------------------------------------------------------------------------------------------------
struct RequantArgs {
  const __nv_bfloat16* input;               // [max_padded_tokens, inner_dim] bf16 FC1 output
  const int* expanded_idx_to_permuted_idx;  // [num_expanded]
  uint8_t* output;                          // [max_padded_tokens, inner_dim / 2] packed E2m1
  uint8_t* output_scale;                    // E4m3 block scales in RequantKernelSpec::sf_layout
  float* per_token_scale;                   // fp32 [max_padded_tokens]
  float global_scale_inv;                   // recipe global scale inverse
  float e4m3_max;                           // recipe E4M3 maximum (448 or the 4/6 variant)
  int num_expanded;                         // num_tokens * top_k
  int inner_dim;                            // intermediate size
};

using RequantSubmitFn = cudaError_t (*)(const cudaLaunchConfig_t*, const RequantArgs&);

struct RequantKernelSpec {
  const char* symbol;
  SfLayout sf_layout;
  // E4M3 maximum the unit is specialized on (448, or 256 for the 4/6 recipe); the host selects the
  // kernel whose value matches the recipe of the forward.
  int e4m3_max;
  int rows_per_cta;  // grid.x = ceil(num_expanded / rows_per_cta)
  uint32_t block[3];
  size_t dynamic_smem_bytes;
  ConfigureFn configure;
  RequantSubmitFn submit;
};

// ------------------------------------------------------------------------------------------------
// Finalize (unpermute + top-k weighted sum, bf16 in / bf16 expert weights / bf16 out).
// ------------------------------------------------------------------------------------------------
struct FinalizeArgs {
  const __nv_bfloat16* input;               // [max_padded_tokens, hidden_dim_padded]
  const __nv_bfloat16* expert_weights;      // [num_tokens, top_k]
  __nv_bfloat16* output;                    // [num_tokens, hidden_dim]
  const int* expanded_idx_to_permuted_idx;  // [num_tokens * top_k], -1 = skip
  const int* total_num_padded_tokens;       // [1]
  int hidden_dim;
  int hidden_dim_padded;
  int num_tokens;
  int num_experts;
  int top_k;
};

// The two native finalize kernels and their launch geometry.
enum class FinalizeVariant : int {
  kScalar = 0,  // grid (ceil(hidden_dim / 256), min(8192, num_tokens)), one element per thread
  kVector = 1,  // grid (num_tokens), 128-bit loads, smem-staged indices and weights
};

using FinalizeSubmitFn = cudaError_t (*)(const cudaLaunchConfig_t*, const FinalizeArgs&);

struct FinalizeKernelSpec {
  const char* symbol;
  FinalizeVariant variant;
  int max_top_k;  // kVector: largest top_k the unit supports; kScalar: 0 (unbounded)
  uint32_t block[3];
  size_t dynamic_smem_bytes;
  ConfigureFn configure;
  FinalizeSubmitFn submit;
};

}  // namespace flashinfer::cake_stepfun::generated
