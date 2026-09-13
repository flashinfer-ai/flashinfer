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

// Sparse-MLA SM120 prefill. Single raw-pointer entry point that launches the
// planner-selected PrefillVariant:
//   - SWAPAB (warp specialized, 64 heads/CTA): the V32 family (DSV3_2 /
//     GLM_NSA / GLM53_NOPE), num_heads 64 / 128, single cache
//   - SG (single-group, 16 heads/CTA): V32 family num_heads 8 / 16;
//     DOTS3_SWA num_heads {8, 16, 32, 64} — SG-only, its D_NOPE=1024 does not
//     fit the MG layout; DSV4_1 num_heads {8, 16, 32, 64} — also SG-only, its
//     32-wide quant groups floor the MG XV warp split to zero tiles
//   - MG (multi-group, 32 heads/CTA): V32 family num_heads >= 32; DSV4
//     num_heads {8..128}
//   - MG_DUAL: dual-cache MG variants (DSV4 only)
//
// FP8-prefill indices require whole 64-entry rows; DOTS3_SWA also requires
// its model window to fit. Full-BF16 DSV4_1 accepts positive runtime topk and
// pitched index rows. Python proposes policy; the C++ resolver validates and
// freezes the executable route. These launchers consume that resolved plan.

#include <cuda_runtime.h>
#include <flashinfer/attention/sparse_mla_sm120/model/model_type.h>

#pragma once

#include <flashinfer/attention/sparse_mla_sm120/arch/common.cuh>
#include <flashinfer/attention/sparse_mla_sm120/kernels/dsv41_bf16/prefill.cuh>
#include <flashinfer/attention/sparse_mla_sm120/kernels/dsv41_fp8/prefill_schedule.cuh>
#include <flashinfer/attention/sparse_mla_sm120/kernels/fp8_prefill/prefill_sg.cuh>
#include <flashinfer/attention/sparse_mla_sm120/kernels/fp8_prefill/prefill_mg.cuh>
#include <flashinfer/attention/sparse_mla_sm120/kernels/fp8_prefill/smem_layout.cuh>
#include <flashinfer/attention/sparse_mla_sm120/kernels/fp8_prefill/prefill_swapab.cuh>
#include <flashinfer/attention/sparse_mla_sm120/model/kv_cache_traits.cuh>

#include "attention_plan.h"
#include "launch_validation.cuh"
#include "prefill_result.h"

namespace flashinfer::sparse_mla_sm120 {

namespace {

constexpr int kMaxCachedCudaDevices = 32;

template <typename Kernel>
PrefillLaunchResult configure_dynamic_smem_per_device(Kernel kernel, size_t smem_bytes,
                                                      bool (&configured)[kMaxCachedCudaDevices]) {
  if (smem_bytes <= 48 * 1024) return true;

  int device = 0;
  const cudaError_t device_error = cudaGetDevice(&device);
  if (device_error != cudaSuccess) return {device_error, "cudaGetDevice"};
  const bool cacheable_device = device >= 0 && device < kMaxCachedCudaDevices;
  if (cacheable_device && configured[device]) return true;

  const cudaError_t rc = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              static_cast<int>(smem_bytes));
  if (rc == cudaSuccess && cacheable_device) configured[device] = true;
  return {rc, "cudaFuncSetAttribute"};
}

template <int Heads, bool ExtraFp4>
PrefillLaunchResult launch_dsv41_bf16_prefill(const execution::AttentionParams& params,
                                            cudaStream_t stream) {
  using namespace kernels::dsv41_bf16;
  auto kernel = sparse_mla_prefill_dsv41_bf16_kernel<Heads, ExtraFp4>;
  constexpr size_t bytes = sizeof(Dsv41Bf16Smem);
  static bool configured[kMaxCachedCudaDevices] = {};
  const auto result = configure_dynamic_smem_per_device(kernel, bytes, configured);
  if (result.error != cudaSuccess) return result;
  dim3 grid(params.num_tokens * ((params.num_heads + 15) / 16));
  cudaLaunchConfig_t config{grid, dim3(Dsv41Bf16Resources::BLOCK_THREADS), bytes, stream, nullptr, 0};
  void* args[] = {(void*)&params};
  return {cudaLaunchKernelExC(&config, (const void*)kernel, args), "cudaLaunchKernelExC"};
}

template <ModelType MT, QkComputeMode QkMode, int NUM_HEADS, int PAGE_BLOCK_SIZE,
          typename GatherSchedule = Dsv41PrefillGatherSchedule>
PrefillLaunchResult launch_prefill_sg(const bf16* Q, const uint8_t* KV_cache,
                                      const int32_t* indices, const float* attn_sink, bf16* output,
                                      float* out_lse, float sm_scale, int num_tokens, int topk,
                                      size_t page_stride_bytes, size_t out_lse_stride_elems,
                                      const int* topk_length_ptr, cudaStream_t stream,
                                      const PrefillColdParams* dual = nullptr) {
  using Cfg = PrefillTileCfg<MT>;
  using Resources = Fp8PrefillResources<MT, QkMode, GatherSchedule>;
  constexpr size_t smem_bytes = Resources::SHARED_BYTES;
  // Ceil-div so NUM_HEADS < HPB (small-TP shards) still launches 1 CTA per token.
  constexpr int REPLICATE_H = (NUM_HEADS + HPB - 1) / HPB;
  dim3 grid(num_tokens * REPLICATE_H);
  dim3 block(Resources::BLOCK_THREADS);

  auto kernel = sparse_mla_prefill_kernel<MT, QkMode, NUM_HEADS, PAGE_BLOCK_SIZE, GatherSchedule>;
  static bool configured[kMaxCachedCudaDevices] = {};
  if constexpr (GatherSchedule::RAW_PIPELINE) {
    const cudaError_t result = execution::validate_mixed_smem(kernel, smem_bytes);
    if (result != cudaSuccess) return {result, "validate_mixed_smem"};
  }
  const auto configured_result = configure_dynamic_smem_per_device(kernel, smem_bytes, configured);
  if (configured_result.error != cudaSuccess) return configured_result;

  PrefillColdParams cold{sm_scale,
                         num_tokens,
                         page_stride_bytes,
                         /*extra_page_stride_bytes=*/(size_t)0,
                         out_lse_stride_elems,
                         topk,
                         /*topk_extra=*/0,
                         attn_sink,
                         topk_length_ptr,
                         /*topk_length_extra=*/(const int*)nullptr};
  if (dual != nullptr) cold = *dual;
  cudaLaunchConfig_t config{grid, block, smem_bytes, stream, nullptr, 0};
  void* args[] = {(void*)&Q,      (void*)&KV_cache, (void*)&indices, (void*)&attn_sink,
                  (void*)&output, (void*)&out_lse,  (void*)&cold};
  return {cudaLaunchKernelExC(&config, (const void*)kernel, args), "cudaLaunchKernelExC"};
}

// Warp-specialized swapAB dispatcher (DSV3_2 family, 64 heads/CTA, single cache).
template <ModelType MT, int NUM_HEADS>
PrefillLaunchResult launch_prefill_swapab(const bf16* Q, const uint8_t* KV_cache,
                                          const int32_t* indices, const float* attn_sink,
                                          bf16* output, float* out_lse, float sm_scale,
                                          int num_tokens, int topk, size_t page_stride_bytes,
                                          size_t out_lse_stride_elems, const int* topk_length_ptr,
                                          cudaStream_t stream) {
  using CT = ComputeTraitsSwapAB<MT>;
  using L = SmemLayoutSwapAB<MT>;
  static_assert(KVCacheTraits<MT>::SCALE_IN_KV_SMEM && KVCacheTraits<MT>::D_NOPE == D_V,
                "swapAB prefill is DSV3_2 family only: inline scales, V spans the nope half");
  constexpr size_t smem_bytes = L::TOTAL;
  constexpr int REPLICATE_H = NUM_HEADS / CT::HEADS_PER_CTA;
  dim3 grid(num_tokens * REPLICATE_H);
  dim3 block(BLOCK_THREADS);

  auto kernel = sparse_mla_prefill_swapab_kernel<MT, NUM_HEADS>;
  static bool configured[kMaxCachedCudaDevices] = {};
  const auto configured_result = configure_dynamic_smem_per_device(kernel, smem_bytes, configured);
  if (configured_result.error != cudaSuccess) return configured_result;

  PrefillColdParams cold{sm_scale,
                         num_tokens,
                         page_stride_bytes,
                         /*extra_page_stride_bytes=*/(size_t)0,
                         out_lse_stride_elems,
                         topk,
                         /*topk_extra=*/0,
                         attn_sink,
                         topk_length_ptr,
                         /*topk_length_extra=*/(const int*)nullptr};
  cudaLaunchConfig_t config{grid, block, smem_bytes, stream, nullptr, 0};
  void* args[] = {(void*)&Q,      (void*)&KV_cache, (void*)&indices, (void*)&attn_sink,
                  (void*)&output, (void*)&out_lse,  (void*)&cold};
  return {cudaLaunchKernelExC(&config, (const void*)kernel, args), "cudaLaunchKernelExC"};
}

// Single-cache MG dispatcher. MG_N_HG_T: 1 lets NUM_HEADS=8/16 use a 16-head
// CTA (NH=8 is internally padded); 2 is the default for NH >= 32.
template <ModelType MT, QkComputeMode QkMode, int NUM_HEADS, int PAGE_BLOCK_SIZE,
          int MG_N_HG_T = MG_N_HG_DEFAULT>
PrefillLaunchResult launch_prefill_mg(const bf16* Q, const uint8_t* KV_cache,
                                      const int32_t* indices, const float* attn_sink, bf16* output,
                                      float* out_lse, float sm_scale, int num_tokens, int topk,
                                      size_t page_stride_bytes, size_t out_lse_stride_elems,
                                      const int* topk_length_ptr, cudaStream_t stream) {
  constexpr size_t smem_bytes = SmemLayoutMG<MT, QkMode>::TOTAL;
  constexpr int MG_HEADS_PER_CTA_LOCAL = MG_N_HG_T * HPB;
  static_assert(NUM_HEADS % MG_HEADS_PER_CTA_LOCAL == 0 || (MG_N_HG_T == 1 && NUM_HEADS < HPB),
                "NUM_HEADS must fill the MG head tile, except a padded NH=8 tile");
  constexpr int REPLICATE_H = (NUM_HEADS + MG_HEADS_PER_CTA_LOCAL - 1) / MG_HEADS_PER_CTA_LOCAL;
  dim3 grid(num_tokens * REPLICATE_H);
  dim3 block(BLOCK_THREADS);

  auto kernel = sparse_mla_prefill_mg_kernel<MT, QkMode, NUM_HEADS, PAGE_BLOCK_SIZE, MG_N_HG_T>;
  static bool configured[kMaxCachedCudaDevices] = {};
  const auto configured_result = configure_dynamic_smem_per_device(kernel, smem_bytes, configured);
  if (configured_result.error != cudaSuccess) return configured_result;

  PrefillColdParams cold{sm_scale,
                         num_tokens,
                         page_stride_bytes,
                         /*extra_page_stride_bytes=*/(size_t)0,
                         out_lse_stride_elems,
                         topk,
                         /*topk_extra=*/0,
                         attn_sink,
                         topk_length_ptr,
                         /*topk_length_extra=*/(const int*)nullptr};
  cudaLaunchConfig_t config{grid, block, smem_bytes, stream, nullptr, 0};
  void* args[] = {(void*)&Q,       (void*)&KV_cache,  (void*)&indices, (void*)&output,
                  (void*)&out_lse, (void*)&attn_sink, (void*)&cold};
  return {cudaLaunchKernelExC(&config, (const void*)kernel, args), "cudaLaunchKernelExC"};
}

// Dual-cache MG dispatcher. `topk` and `topk_extra` are runtime;
// PAGE_BLOCK_SIZE_EXTRA stays template because it changes the KV stride.
template <ModelType MT, int NUM_HEADS, int PAGE_BLOCK_SIZE, int PAGE_BLOCK_SIZE_EXTRA,
          int MG_N_HG_T = MG_N_HG_DEFAULT>
PrefillLaunchResult launch_prefill_mg_dual_fulltile(
    const bf16* Q, const uint8_t* KV_cache, const int32_t* indices, const uint8_t* KV_cache_extra,
    const int32_t* indices_extra, const float* attn_sink, bf16* output, float* out_lse,
    float sm_scale, int num_tokens, int topk, int topk_extra, size_t page_stride_bytes,
    size_t extra_page_stride_bytes, size_t out_lse_stride_elems, cudaStream_t stream) {
  constexpr size_t smem_bytes = SmemLayoutMG<MT, QkComputeMode::BF16>::TOTAL;
  constexpr int MG_HEADS_PER_CTA_LOCAL = MG_N_HG_T * HPB;
  static_assert(NUM_HEADS % MG_HEADS_PER_CTA_LOCAL == 0 || (MG_N_HG_T == 1 && NUM_HEADS < HPB),
                "NUM_HEADS must fill the MG head tile, except a padded NH=8 tile");
  constexpr int REPLICATE_H = (NUM_HEADS + MG_HEADS_PER_CTA_LOCAL - 1) / MG_HEADS_PER_CTA_LOCAL;
  dim3 grid(num_tokens * REPLICATE_H);
  dim3 block(BLOCK_THREADS);

  auto kernel = sparse_mla_prefill_mg_dual_fulltile_kernel<MT, NUM_HEADS, PAGE_BLOCK_SIZE,
                                                           PAGE_BLOCK_SIZE_EXTRA, MG_N_HG_T>;
  static bool configured[kMaxCachedCudaDevices] = {};
  const auto configured_result = configure_dynamic_smem_per_device(kernel, smem_bytes, configured);
  if (configured_result.error != cudaSuccess) return configured_result;

  PrefillColdParams cold{sm_scale,
                         num_tokens,
                         page_stride_bytes,
                         extra_page_stride_bytes,
                         out_lse_stride_elems,
                         topk,
                         topk_extra,
                         attn_sink,
                         /*topk_length=*/(const int*)nullptr,
                         /*topk_length_extra=*/(const int*)nullptr};
  cudaLaunchConfig_t config{grid, block, smem_bytes, stream, nullptr, 0};
  void* args[] = {(void*)&Q,
                  (void*)&KV_cache,
                  (void*)&indices,
                  (void*)&KV_cache_extra,
                  (void*)&indices_extra,
                  (void*)&output,
                  (void*)&out_lse,
                  (void*)&attn_sink,
                  (void*)&cold};
  return {cudaLaunchKernelExC(&config, (const void*)kernel, args), "cudaLaunchKernelExC"};
}

template <ModelType MT, QkComputeMode QkMode, int NUM_HEADS, int PAGE_BLOCK_SIZE,
          int PAGE_BLOCK_SIZE_EXTRA, int MG_N_HG_T = MG_N_HG_DEFAULT>
PrefillLaunchResult launch_prefill_mg_dual(const bf16* Q, const uint8_t* KV_cache,
                                           const int32_t* indices, const uint8_t* KV_cache_extra,
                                           const int32_t* indices_extra, const float* attn_sink,
                                           bf16* output, float* out_lse, float sm_scale,
                                           int num_tokens, int topk, int topk_extra,
                                           size_t page_stride_bytes, size_t extra_page_stride_bytes,
                                           size_t out_lse_stride_elems, const int* topk_length_ptr,
                                           const int* topk_length_extra_ptr, cudaStream_t stream) {
  constexpr size_t smem_bytes = SmemLayoutMG<MT, QkMode>::TOTAL;
  constexpr int MG_HEADS_PER_CTA_LOCAL = MG_N_HG_T * HPB;
  static_assert(NUM_HEADS % MG_HEADS_PER_CTA_LOCAL == 0 || (MG_N_HG_T == 1 && NUM_HEADS < HPB),
                "NUM_HEADS must fill the MG head tile, except a padded NH=8 tile");
  constexpr int REPLICATE_H = (NUM_HEADS + MG_HEADS_PER_CTA_LOCAL - 1) / MG_HEADS_PER_CTA_LOCAL;
  dim3 grid(num_tokens * REPLICATE_H);
  dim3 block(BLOCK_THREADS);

  auto kernel = sparse_mla_prefill_mg_dual_kernel<MT, QkMode, NUM_HEADS, PAGE_BLOCK_SIZE,
                                                  PAGE_BLOCK_SIZE_EXTRA, MG_N_HG_T>;
  static bool configured[kMaxCachedCudaDevices] = {};
  const auto configured_result = configure_dynamic_smem_per_device(kernel, smem_bytes, configured);
  if (configured_result.error != cudaSuccess) return configured_result;

  PrefillColdParams cold{
      sm_scale, num_tokens, page_stride_bytes, extra_page_stride_bytes, out_lse_stride_elems,
      topk,     topk_extra, attn_sink,         topk_length_ptr,         topk_length_extra_ptr};
  cudaLaunchConfig_t config{grid, block, smem_bytes, stream, nullptr, 0};
  void* args[] = {(void*)&Q,
                  (void*)&KV_cache,
                  (void*)&indices,
                  (void*)&KV_cache_extra,
                  (void*)&indices_extra,
                  (void*)&output,
                  (void*)&out_lse,
                  (void*)&attn_sink,
                  (void*)&cold};
  return {cudaLaunchKernelExC(&config, (const void*)kernel, args), "cudaLaunchKernelExC"};
}

// swapAB (warp specialized, 64 heads/CTA), DSV3_2 family. Any runtime topk
// width is served (GLM53_NOPE's 2176 folds the 128-token indexer tail into
// the 2048 sparse selection).
// swapAB takes no PBS template parameter (the KV stride is runtime), but the
// envelope is pbs=64 like every other instantiation.
template <ModelType MT>
inline PrefillLaunchResult dispatch_v32_swapab(int num_heads, int topk, int page_block_size,
                                               const bf16* Q, const uint8_t* KV,
                                               const int32_t* indices, const float* attn_sink,
                                               bf16* output, float* out_lse, float sm_scale,
                                               int num_tokens, size_t page_stride_bytes,
                                               size_t out_lse_stride_elems,
                                               const int* topk_length_ptr, cudaStream_t stream) {
  if (page_block_size != execution::FixedPageSize) return false;
  return execution::visit_prefill_heads<MT, 4>(num_heads, [&](auto head) {
    return launch_prefill_swapab<MT, decltype(head)::value>(
        Q, KV, indices, attn_sink, output, out_lse, sm_scale, num_tokens, topk, page_stride_bytes,
        out_lse_stride_elems, topk_length_ptr, stream);
  });
}

// SG (single-group, 16 heads/CTA). PBS=64 matches the V32 decode
// (`kernels/dsv32_fp8/decode.cuh`). NH=8 covers small-TP shards; the SG kernel
// zero-pads invalid head slots up to HPB=16 internally and gates write-back
// by VALID_HPB.
template <ModelType MT>
inline PrefillLaunchResult dispatch_v32_sg(int num_heads, int topk, int page_block_size,
                                           const bf16* Q, const uint8_t* KV, const int32_t* indices,
                                           const float* attn_sink, bf16* output, float* out_lse,
                                           float sm_scale, int num_tokens, size_t page_stride_bytes,
                                           size_t out_lse_stride_elems, const int* topk_length_ptr,
                                           cudaStream_t stream) {
  if (page_block_size != execution::FixedPageSize) return false;
  return execution::visit_prefill_heads<MT, 1>(num_heads, [&](auto head) {
    return launch_prefill_sg<MT, QkComputeMode::FP8, decltype(head)::value, 64>(
        Q, KV, indices, attn_sink, output, out_lse, sm_scale, num_tokens, topk, page_stride_bytes,
        out_lse_stride_elems, topk_length_ptr, stream);
  });
}

// Single-cache MG (multi-group, 32 heads/CTA).
template <ModelType MT>
inline PrefillLaunchResult dispatch_v32_mg(int num_heads, int topk, int page_block_size,
                                           const bf16* Q, const uint8_t* KV, const int32_t* indices,
                                           const float* attn_sink, bf16* output, float* out_lse,
                                           float sm_scale, int num_tokens, size_t page_stride_bytes,
                                           size_t out_lse_stride_elems, const int* topk_length_ptr,
                                           cudaStream_t stream) {
  if (page_block_size != execution::FixedPageSize) return false;

  return execution::visit_prefill_heads<MT, 2>(num_heads, [&](auto head) {
    return launch_prefill_mg<MT, QkComputeMode::FP8, decltype(head)::value, 64>(
        Q, KV, indices, attn_sink, output, out_lse, sm_scale, num_tokens, topk, page_stride_bytes,
        out_lse_stride_elems, topk_length_ptr, stream);
  });
}

// DOTS3_SWA is SG-only: its D_NOPE=1024 puts the MG per-group buffers over the
// sm120 smem cap at every BI that still satisfies the FP8 XV k=32 floor. SG
// covers num_heads > HPB by replicating one CTA per 16-head tile
// (`REPLICATE_H`), so TP1..TP8 shards of a 64-head layer are all reachable.
//
// Any runtime topk >= 513 (the binding's sliding-window floor) is served;
// 576 is the tightest multiple of the BI=32 tile that covers the window. The
// window itself is baked into PrefillTilePrimary, so a caller passing no
// topk_length still gets a correctly bounded scan.
inline PrefillLaunchResult dispatch_dots3_swa_sg(int num_heads, int topk, int page_block_size,
                                                 const bf16* Q, const uint8_t* KV,
                                                 const int32_t* indices, const float* attn_sink,
                                                 bf16* output, float* out_lse, float sm_scale,
                                                 int num_tokens, size_t page_stride_bytes,
                                                 size_t out_lse_stride_elems,
                                                 const int* topk_length_ptr, cudaStream_t stream) {
  if (page_block_size != execution::FixedPageSize) return false;

  return execution::visit_prefill_heads<ModelType::DOTS3_SWA, 1>(num_heads, [&](auto head) {
    return launch_prefill_sg<ModelType::DOTS3_SWA, QkComputeMode::FP8, decltype(head)::value, 64>(
        Q, KV, indices, attn_sink, output, out_lse, sm_scale, num_tokens, topk, page_stride_bytes,
        out_lse_stride_elems, topk_length_ptr, stream);
  });
}

// DSV4_1 is SG-only for the same structural reason as DOTS3_SWA, but with the
// warp split driven by its 32-wide quant groups (V_CHUNK=32 floors
// NT_PER_WARP_XV to 0 for an 8-warp XV) rather than by smem capacity. Any
// runtime topk made of whole index tiles is served; the binding enforces
// topk % 64 == 0. TP1..TP8 shards of the 64-head layer ride REPLICATE_H.
template <typename GatherSchedule>
inline PrefillLaunchResult dispatch_dsv41_sg(
    int num_heads, int topk, int page_block_size, const bf16* Q, const uint8_t* KV,
    const int32_t* indices, const float* attn_sink, bf16* output, float* out_lse, float sm_scale,
    int num_tokens, size_t page_stride_bytes, size_t out_lse_stride_elems,
    const int* topk_length_ptr, cudaStream_t stream, const PrefillColdParams& cold) {
  if (page_block_size <= 0) return false;
  if (cold.extra_kv != nullptr && cold.extra_page_block_size <= 0) return false;

  return execution::visit_prefill_heads<ModelType::DSV4_1, 1>(num_heads, [&](auto head) {
    return launch_prefill_sg<ModelType::DSV4_1, QkComputeMode::FP8, decltype(head)::value, 64,
                             GatherSchedule>(Q, KV, indices, attn_sink, output, out_lse, sm_scale,
                                             num_tokens, topk, page_stride_bytes,
                                             out_lse_stride_elems, topk_length_ptr, stream, &cold);
  });
}

inline PrefillLaunchResult dispatch_dsv4_single(
    const execution::ExecutionPlan& plan, int num_heads, int topk, int page_block_size,
    const bf16* Q, const uint8_t* KV, const int32_t* indices, const float* attn_sink, bf16* output,
    float* out_lse, float sm_scale, int num_tokens, size_t page_stride_bytes,
    size_t out_lse_stride_elems, const int* topk_length_ptr, cudaStream_t stream) {
  if (page_block_size != execution::FixedPageSize) return false;
  return execution::visit_prefill_heads<ModelType::DSV4, 2>(num_heads, [&](auto head) {
    constexpr int H = decltype(head)::value;
    constexpr int groups = H <= HPB ? 1 : MG_N_HG_DEFAULT;
    if (plan.numeric == execution::NumericRoute::QkBF16PvFP8)
      return launch_prefill_mg<ModelType::DSV4, QkComputeMode::BF16, H, 64, groups>(
          Q, KV, indices, attn_sink, output, out_lse, sm_scale, num_tokens, topk, page_stride_bytes,
          out_lse_stride_elems, topk_length_ptr, stream);
    return launch_prefill_mg<ModelType::DSV4, QkComputeMode::FP8, H, 64, groups>(
        Q, KV, indices, attn_sink, output, out_lse, sm_scale, num_tokens, topk, page_stride_bytes,
        out_lse_stride_elems, topk_length_ptr, stream);
  });
}

inline PrefillLaunchResult dispatch_dsv4_dual(
    const execution::ExecutionPlan& plan, int num_heads, int topk, int topk_extra,
    int page_block_size, int extra_page_block_size, const bf16* Q, const uint8_t* KV,
    const int32_t* indices, const uint8_t* KV_extra, const int32_t* idx_extra,
    const float* attn_sink, bf16* output, float* out_lse, float sm_scale, int num_tokens,
    size_t page_stride_bytes, size_t extra_page_stride_bytes, size_t out_lse_stride_elems,
    const int* topk_length_ptr, const int* topk_length_extra_ptr, cudaStream_t stream) {
  if (page_block_size != execution::FixedPageSize) return false;
  auto page = [&](auto pbs) {
    return execution::visit_prefill_heads<ModelType::DSV4, 3>(num_heads, [&](auto head) {
      constexpr int H = decltype(head)::value, P = decltype(pbs)::value;
      constexpr int groups = H <= HPB ? 1 : MG_N_HG_DEFAULT;
      if (plan.implementation == execution::Implementation::FullTile)
        return launch_prefill_mg_dual_fulltile<ModelType::DSV4, H, 64, P, groups>(
            Q, KV, indices, KV_extra, idx_extra, attn_sink, output, out_lse, sm_scale, num_tokens,
            topk, topk_extra, page_stride_bytes, extra_page_stride_bytes, out_lse_stride_elems,
            stream);
      return launch_prefill_mg_dual<ModelType::DSV4, QkComputeMode::BF16, H, 64, P, groups>(
          Q, KV, indices, KV_extra, idx_extra, attn_sink, output, out_lse, sm_scale, num_tokens,
          topk, topk_extra, page_stride_bytes, extra_page_stride_bytes, out_lse_stride_elems,
          topk_length_ptr, topk_length_extra_ptr, stream);
    });
  };
  return execution::visit_extra_page(extra_page_block_size, page);
}

}  // namespace

}  // namespace flashinfer::sparse_mla_sm120
