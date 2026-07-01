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

// Sparse-MLA SM120 prefill. Single raw-pointer entry point that dispatches:
//   - DSV3_2 / DSV4 model split
//   - SG (single-group, 16 heads/CTA) for num_heads <= 16
//   - MG (multi-group, 32 heads/CTA) for num_heads > 16
//   - Dual-cache MG variants (DSV4 only)
//
// Raw-pointer interface; framework-agnostic.

#include <cuda_runtime.h>
#include <flashinfer/attention/sparse_mla_sm120/model/model_type.h>

#include <flashinfer/attention/sparse_mla_sm120/arch/common.cuh>
#include <flashinfer/attention/sparse_mla_sm120/common/smem_layout.cuh>
#include <flashinfer/attention/sparse_mla_sm120/model/kv_cache_traits.cuh>
#include <flashinfer/attention/sparse_mla_sm120/prefill_kernel.cuh>

namespace flashinfer::sparse_mla_sm120 {

namespace {

constexpr int kMaxCachedCudaDevices = 32;

template <typename Kernel>
void configure_dynamic_smem_per_device(Kernel kernel, size_t smem_bytes,
                                       bool (&configured)[kMaxCachedCudaDevices]) {
  if (smem_bytes <= 48 * 1024) return;

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  const bool cacheable_device = device >= 0 && device < kMaxCachedCudaDevices;
  if (cacheable_device && configured[device]) return;

  const cudaError_t rc = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              static_cast<int>(smem_bytes));
  if (rc == cudaSuccess) {
    if (cacheable_device) configured[device] = true;
    return;
  }
  CUDA_CHECK(rc);
}

template <ModelType MT, ComputeMode CM, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE>
void launch_prefill_sg(const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
                       const float* attn_sink, bf16* output, float* out_lse, float sm_scale,
                       int num_tokens, size_t stride_kv_block, const int* topk_length_ptr,
                       cudaStream_t stream) {
  constexpr size_t smem_bytes = SmemLayout<MT, CM>::TOTAL;
  // Ceil-div so NUM_HEADS < HPB (small-TP shards) still launches 1 CTA per token.
  constexpr int REPLICATE_H = (NUM_HEADS + HPB - 1) / HPB;
  dim3 grid(num_tokens * REPLICATE_H);
  dim3 block(BLOCK_THREADS);

  auto kernel = sparse_mla_prefill_kernel<MT, CM, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE>;
  static bool configured[kMaxCachedCudaDevices] = {};
  configure_dynamic_smem_per_device(kernel, smem_bytes, configured);

  // SG is single-cache only.
  PrefillColdParams cold{sm_scale,
                         num_tokens,
                         stride_kv_block,
                         /*stride_kv_block_extra=*/(size_t)0,
                         /*topk_extra=*/0,
                         attn_sink,
                         topk_length_ptr,
                         /*topk_length_extra=*/(const int*)nullptr};
  cudaLaunchConfig_t config{grid, block, smem_bytes, stream, nullptr, 0};
  void* args[] = {(void*)&Q,      (void*)&KV_cache, (void*)&indices, (void*)&attn_sink,
                  (void*)&output, (void*)&out_lse,  (void*)&cold};
  CUDA_CHECK(cudaLaunchKernelExC(&config, (const void*)kernel, args));
}

// Single-cache MG dispatcher. MG_N_HG_T: 1 lets NUM_HEADS=16 through MG
// (HEADS_PER_CTA=16, same shape as SG); 2 is the default for NH >= 32.
template <ModelType MT, ComputeMode CM, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE,
          int MG_N_HG_T = MG_N_HG_DEFAULT>
void launch_prefill_mg(const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
                       const float* attn_sink, bf16* output, float* out_lse, float sm_scale,
                       int num_tokens, size_t stride_kv_block, const int* topk_length_ptr,
                       cudaStream_t stream) {
  constexpr size_t smem_bytes = SmemLayoutMG<MT, CM>::TOTAL;
  constexpr int MG_HEADS_PER_CTA_LOCAL = MG_N_HG_T * HPB;
  static_assert(NUM_HEADS % MG_HEADS_PER_CTA_LOCAL == 0,
                "NUM_HEADS must be a multiple of MG_N_HG_T * HPB");
  constexpr int REPLICATE_H = NUM_HEADS / MG_HEADS_PER_CTA_LOCAL;
  dim3 grid(num_tokens * REPLICATE_H);
  dim3 block(BLOCK_THREADS);

  auto kernel = sparse_mla_prefill_mg_kernel<MT, CM, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE, MG_N_HG_T>;
  static bool configured[kMaxCachedCudaDevices] = {};
  configure_dynamic_smem_per_device(kernel, smem_bytes, configured);

  PrefillColdParams cold{sm_scale,
                         num_tokens,
                         stride_kv_block,
                         /*stride_kv_block_extra=*/(size_t)0,
                         /*topk_extra=*/0,
                         attn_sink,
                         topk_length_ptr,
                         /*topk_length_extra=*/(const int*)nullptr};
  cudaLaunchConfig_t config{grid, block, smem_bytes, stream, nullptr, 0};
  void* args[] = {(void*)&Q,       (void*)&KV_cache,  (void*)&indices, (void*)&output,
                  (void*)&out_lse, (void*)&attn_sink, (void*)&cold};
  CUDA_CHECK(cudaLaunchKernelExC(&config, (const void*)kernel, args));
}

// Dual-cache MG dispatcher. `topk_extra` is runtime; PAGE_BLOCK_SIZE_EXTRA
// stays template because it changes the KV stride.
template <ModelType MT, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE, int PAGE_BLOCK_SIZE_EXTRA,
          int MG_N_HG_T = MG_N_HG_DEFAULT>
void launch_prefill_mg_dual_fulltile(const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
                                     const uint8_t* KV_cache_extra, const int32_t* indices_extra,
                                     const float* attn_sink, bf16* output, float* out_lse,
                                     float sm_scale, int num_tokens, int topk_extra,
                                     size_t stride_kv_block, size_t stride_kv_block_extra,
                                     cudaStream_t stream) {
  constexpr size_t smem_bytes = SmemLayoutMG<MT, ComputeMode::BF16>::TOTAL;
  constexpr int MG_HEADS_PER_CTA_LOCAL = MG_N_HG_T * HPB;
  static_assert(NUM_HEADS % MG_HEADS_PER_CTA_LOCAL == 0,
                "NUM_HEADS must be a multiple of MG_N_HG_T * HPB");
  constexpr int REPLICATE_H = NUM_HEADS / MG_HEADS_PER_CTA_LOCAL;
  dim3 grid(num_tokens * REPLICATE_H);
  dim3 block(BLOCK_THREADS);

  auto kernel = sparse_mla_prefill_mg_dual_fulltile_kernel<MT, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE,
                                                           PAGE_BLOCK_SIZE_EXTRA, MG_N_HG_T>;
  static bool configured[kMaxCachedCudaDevices] = {};
  configure_dynamic_smem_per_device(kernel, smem_bytes, configured);

  PrefillColdParams cold{sm_scale,
                         num_tokens,
                         stride_kv_block,
                         stride_kv_block_extra,
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
  CUDA_CHECK(cudaLaunchKernelExC(&config, (const void*)kernel, args));
}

template <ModelType MT, ComputeMode CM, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE,
          int PAGE_BLOCK_SIZE_EXTRA, int MG_N_HG_T = MG_N_HG_DEFAULT>
void launch_prefill_mg_dual(const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
                            const uint8_t* KV_cache_extra, const int32_t* indices_extra,
                            const float* attn_sink, bf16* output, float* out_lse, float sm_scale,
                            int num_tokens, int topk_extra, size_t stride_kv_block,
                            size_t stride_kv_block_extra, const int* topk_length_ptr,
                            const int* topk_length_extra_ptr, cudaStream_t stream) {
  constexpr size_t smem_bytes = SmemLayoutMG<MT, CM>::TOTAL;
  constexpr int MG_HEADS_PER_CTA_LOCAL = MG_N_HG_T * HPB;
  static_assert(NUM_HEADS % MG_HEADS_PER_CTA_LOCAL == 0,
                "NUM_HEADS must be a multiple of MG_N_HG_T * HPB");
  constexpr int REPLICATE_H = NUM_HEADS / MG_HEADS_PER_CTA_LOCAL;
  dim3 grid(num_tokens * REPLICATE_H);
  dim3 block(BLOCK_THREADS);

  auto kernel = sparse_mla_prefill_mg_dual_kernel<MT, CM, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE,
                                                  PAGE_BLOCK_SIZE_EXTRA, MG_N_HG_T>;
  static bool configured[kMaxCachedCudaDevices] = {};
  configure_dynamic_smem_per_device(kernel, smem_bytes, configured);

  PrefillColdParams cold{sm_scale,   num_tokens, stride_kv_block, stride_kv_block_extra,
                         topk_extra, attn_sink,  topk_length_ptr, topk_length_extra_ptr};
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
  CUDA_CHECK(cudaLaunchKernelExC(&config, (const void*)kernel, args));
}

template <ModelType MT>
inline bool dispatch_v32(int num_heads, int topk, const bf16* Q, const uint8_t* KV,
                         const int32_t* indices, const float* attn_sink, bf16* output,
                         float* out_lse, float sm_scale, int num_tokens, size_t stride_kv_block,
                         const int* topk_length_ptr, cudaStream_t stream) {
  static_assert(KVCacheTraits<MT>::D_QK == 576);
  if (topk != 2048) return false;

  // PBS=64 matches the V32 decode (`decode_dsv3_2_kernel.cuh`). NH=8 covers
  // small-TP shards; the SG kernel zero-pads invalid head slots up to HPB=16
  // internally and gates write-back by VALID_HPB.
  if (num_heads <= HPB) {
    if (num_heads == 8) {
      launch_prefill_sg<MT, ComputeMode::FP8, 8, 2048, 64>(
          Q, KV, indices, attn_sink, output, out_lse, sm_scale, num_tokens, stride_kv_block,
          topk_length_ptr, stream);
      return true;
    }
    if (num_heads != 16) return false;
    launch_prefill_sg<MT, ComputeMode::FP8, 16, 2048, 64>(Q, KV, indices, attn_sink, output,
                                                          out_lse, sm_scale, num_tokens,
                                                          stride_kv_block, topk_length_ptr, stream);
    return true;
  }

#define DISPATCH_DSV3_2_MG(NH)                                                             \
  launch_prefill_mg<MT, ComputeMode::FP8, NH, 2048, 64>(Q, KV, indices, attn_sink, output, \
                                                        out_lse, sm_scale, num_tokens,     \
                                                        stride_kv_block, topk_length_ptr, stream)

  switch (num_heads) {
    case 32:
      DISPATCH_DSV3_2_MG(32);
      return true;
    case 64:
      DISPATCH_DSV3_2_MG(64);
      return true;
    case 128:
      DISPATCH_DSV3_2_MG(128);
      return true;
    default:
      return false;
  }
#undef DISPATCH_DSV3_2_MG
}

inline bool dispatch_dsv4_single(int num_heads, int topk, const bf16* Q, const uint8_t* KV,
                                 const int32_t* indices, const float* attn_sink, bf16* output,
                                 float* out_lse, float sm_scale, int num_tokens,
                                 size_t stride_kv_block, const int* topk_length_ptr,
                                 cudaStream_t stream) {
#define DISPATCH_MG_CM(CM, NH, TK, NHG)                                                  \
  launch_prefill_mg<ModelType::DSV4, ComputeMode::CM, NH, TK, 64, NHG>(                  \
      Q, KV, indices, attn_sink, output, out_lse, sm_scale, num_tokens, stride_kv_block, \
      topk_length_ptr, stream)

// NH=16 routes through MG with MG_N_HG_T=1 to avoid SG BF16-QK smem aliasing
// between Q staging and FP8 weight staging at multi-wave launches.
#define DISPATCH_BY_NH_CM(CM, TK)       \
  do {                                  \
    switch (num_heads) {                \
      case 16:                          \
        DISPATCH_MG_CM(CM, 16, TK, 1);  \
        return true;                    \
      case 32:                          \
        DISPATCH_MG_CM(CM, 32, TK, 2);  \
        return true;                    \
      case 64:                          \
        DISPATCH_MG_CM(CM, 64, TK, 2);  \
        return true;                    \
      case 128:                         \
        DISPATCH_MG_CM(CM, 128, TK, 2); \
        return true;                    \
      default:                          \
        return false;                   \
    }                                   \
  } while (0)

  // Small K-loop: BF16 QK skips the FP8 Q-quantize prologue. Larger K
  // amortises FP8's higher Tensor-Core throughput.
  if (topk == 128)
    DISPATCH_BY_NH_CM(BF16, 128);
  else if (topk == 512)
    DISPATCH_BY_NH_CM(FP8, 512);
  else if (topk == 1024)
    DISPATCH_BY_NH_CM(FP8, 1024);
  else if (topk == 2048)
    DISPATCH_BY_NH_CM(FP8, 2048);
  else
    return false;

#undef DISPATCH_BY_NH_CM
#undef DISPATCH_MG_CM
  return false;  // unreachable
}

inline bool dispatch_dsv4_dual(int num_heads, int topk, int topk_extra, int extra_page_block_size,
                               const bf16* Q, const uint8_t* KV, const int32_t* indices,
                               const uint8_t* KV_extra, const int32_t* idx_extra,
                               const float* attn_sink, bf16* output, float* out_lse, float sm_scale,
                               int num_tokens, size_t stride_kv_block, size_t stride_kv_block_extra,
                               const int* topk_length_ptr, const int* topk_length_extra_ptr,
                               cudaStream_t stream) {
  if (topk == 128 && topk_length_ptr == nullptr && topk_length_extra_ptr == nullptr &&
      topk_extra % BI == 0 && (extra_page_block_size == 64 || extra_page_block_size == 2)) {
#define DISPATCH_DUAL_MG_FULLTILE(NH, TK, PBSX, NHG)                                         \
  launch_prefill_mg_dual_fulltile<ModelType::DSV4, NH, TK, 64, PBSX, NHG>(                   \
      Q, KV, indices, KV_extra, idx_extra, attn_sink, output, out_lse, sm_scale, num_tokens, \
      topk_extra, stride_kv_block, stride_kv_block_extra, stream)

#define DISPATCH_FULLTILE_BY_NH_PBSX(PBSX)            \
  do {                                                \
    switch (num_heads) {                              \
      case 16:                                        \
        DISPATCH_DUAL_MG_FULLTILE(16, 128, PBSX, 1);  \
        return true;                                  \
      case 32:                                        \
        DISPATCH_DUAL_MG_FULLTILE(32, 128, PBSX, 2);  \
        return true;                                  \
      case 64:                                        \
        DISPATCH_DUAL_MG_FULLTILE(64, 128, PBSX, 2);  \
        return true;                                  \
      case 128:                                       \
        DISPATCH_DUAL_MG_FULLTILE(128, 128, PBSX, 2); \
        return true;                                  \
      default:                                        \
        return false;                                 \
    }                                                 \
  } while (0)

    if (extra_page_block_size == 64) {
      DISPATCH_FULLTILE_BY_NH_PBSX(64);
    } else {
      DISPATCH_FULLTILE_BY_NH_PBSX(2);
    }
#undef DISPATCH_FULLTILE_BY_NH_PBSX
#undef DISPATCH_DUAL_MG_FULLTILE
  }

// topk_extra is runtime; extra_page_block_size stays template because it
// changes the KV stride. NH=16 uses MG_N_HG_T=1.
#define DISPATCH_DUAL_MG_CM(CM, NH, TK, PBSX, NHG)                                                \
  launch_prefill_mg_dual<ModelType::DSV4, ComputeMode::CM, NH, TK, 64, PBSX, NHG>(                \
      Q, KV, indices, KV_extra, idx_extra, attn_sink, output, out_lse, sm_scale, num_tokens,      \
      topk_extra, stride_kv_block, stride_kv_block_extra, topk_length_ptr, topk_length_extra_ptr, \
      stream)

#define DISPATCH_BY_NH_PBSX(PBSX)                     \
  do {                                                \
    switch (num_heads) {                              \
      case 16:                                        \
        DISPATCH_DUAL_MG_CM(BF16, 16, 128, PBSX, 1);  \
        return true;                                  \
      case 32:                                        \
        DISPATCH_DUAL_MG_CM(BF16, 32, 128, PBSX, 2);  \
        return true;                                  \
      case 64:                                        \
        DISPATCH_DUAL_MG_CM(BF16, 64, 128, PBSX, 2);  \
        return true;                                  \
      case 128:                                       \
        DISPATCH_DUAL_MG_CM(BF16, 128, 128, PBSX, 2); \
        return true;                                  \
      default:                                        \
        return false;                                 \
    }                                                 \
  } while (0)

  if (topk != 128) return false;
  if (extra_page_block_size == 64) {
    DISPATCH_BY_NH_PBSX(64);
  } else if (extra_page_block_size == 2) {
    DISPATCH_BY_NH_PBSX(2);
  }
  return false;
#undef DISPATCH_BY_NH_PBSX
#undef DISPATCH_DUAL_MG_CM
}

}  // namespace

// Public dispatcher. Returns false if no template variant matches; caller
// is responsible for surfacing the supported envelope.
// Dual-cache (extra_KV_cache != nullptr) is DSV4-only.
bool sparse_mla_prefill_dispatch(ModelType mt, int num_heads, int topk, int page_block_size,
                                 int topk_extra, int extra_page_block_size, const bf16* Q,
                                 const uint8_t* KV_cache, const int32_t* indices,
                                 const uint8_t* extra_KV_cache, const int32_t* extra_indices,
                                 bf16* output, float* out_lse, float sm_scale, int num_tokens,
                                 size_t stride_kv_block, size_t stride_kv_block_extra,
                                 const float* attn_sink, const int* topk_length,
                                 const int* extra_topk_length, cudaStream_t stream) {
  if (extra_KV_cache != nullptr) {
    if (mt != ModelType::DSV4) return false;
    return dispatch_dsv4_dual(num_heads, topk, topk_extra, extra_page_block_size, Q, KV_cache,
                              indices, extra_KV_cache, extra_indices, attn_sink, output, out_lse,
                              sm_scale, num_tokens, stride_kv_block, stride_kv_block_extra,
                              topk_length, extra_topk_length, stream);
  }

  switch (mt) {
    case ModelType::DSV3_2:
      return dispatch_v32<ModelType::DSV3_2>(num_heads, topk, Q, KV_cache, indices, attn_sink,
                                             output, out_lse, sm_scale, num_tokens, stride_kv_block,
                                             topk_length, stream);
    case ModelType::GLM_NSA:
      return dispatch_v32<ModelType::GLM_NSA>(num_heads, topk, Q, KV_cache, indices, attn_sink,
                                              output, out_lse, sm_scale, num_tokens,
                                              stride_kv_block, topk_length, stream);
    case ModelType::DSV4:
      return dispatch_dsv4_single(num_heads, topk, Q, KV_cache, indices, attn_sink, output, out_lse,
                                  sm_scale, num_tokens, stride_kv_block, topk_length, stream);
  }
  return false;
}

}  // namespace flashinfer::sparse_mla_sm120
