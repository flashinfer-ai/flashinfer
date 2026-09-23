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
#pragma once
// TVM-FFI binding for the frozen radix top-k / sparse top-p sampling bundle.
//
// The JIT module (flashinfer/jit/cake_sampling.py) generates a small .cu that defines:
//   CAKE_SAMPLING_BODY_FILE     "generated/cake_sampling_kernels.cu" (one source for every device)
//   CAKE_SAMPLING_MIN_MAJOR / CAKE_SAMPLING_MIN_MINOR   oldest compute capability the frozen body
//                                                      supports (manifest min_compute_capability)
//   CAKE_SAMPLING_SLAB          slab row stride (entries per row of the top-k slab)
//   CAKE_SAMPLING_STAGE1_TABLE(X)  X(symbol, cluster, ept, stream, threads, smem_bytes) ...
//   CAKE_SAMPLING_STAGE23_TABLE(X) X(symbol, threads, items, smem_bytes) ...
// and then includes this header.  Every table entry is taken verbatim from manifest.json.
#ifndef CAKE_SAMPLING_BODY_FILE
#error "CAKE_SAMPLING_BODY_FILE must name the frozen generated body"
#endif
#ifndef CAKE_SAMPLING_STAGE1_TABLE
#error "CAKE_SAMPLING_STAGE1_TABLE must list the frozen stage-1 variants"
#endif
#ifndef CAKE_SAMPLING_STAGE23_TABLE
#error "CAKE_SAMPLING_STAGE23_TABLE must list the frozen stage-2/3 variants"
#endif
#ifndef CAKE_SAMPLING_SLAB
#error "CAKE_SAMPLING_SLAB must give the slab row stride"
#endif
#if !defined(CAKE_SAMPLING_MIN_MAJOR) || !defined(CAKE_SAMPLING_MIN_MINOR)
#error "CAKE_SAMPLING_MIN_MAJOR / CAKE_SAMPLING_MIN_MINOR must give the oldest supported capability"
#endif
// The frozen body is a self-contained CUDA translation-unit fragment.  Keep its fixed-width
// types intact: the generated vector-load code refers to them by name.
#include CAKE_SAMPLING_BODY_FILE

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace cake_sampling {

constexpr int64_t kSlab = CAKE_SAMPLING_SLAB;
constexpr int32_t kTopKScalar = 1;
constexpr int32_t kTopKPerRow = 2;
constexpr int32_t kTopPScalar = 1;
constexpr int32_t kTopPPerRow = 2;

inline void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK(status == cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

// The module is one fatbin holding a cubin per target architecture selected by FlashInfer's
// CompilationContext (FLASHINFER_CUDA_ARCH_LIST, else the visible devices).  A device below the
// frozen body's minimum capability, or one whose architecture is not among the compiled targets
// (no cubin in the fatbin), is rejected here with a message naming the fix instead of failing
// later inside cudaFuncSetAttribute / cudaLaunchKernelExC.
inline void CheckTarget(int32_t device_id, const void* probe_kernel) {
  int major = 0;
  int minor = 0;
  CheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
            "cudaDeviceGetAttribute(major)");
  CheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
            "cudaDeviceGetAttribute(minor)");
  TVM_FFI_ICHECK(major > CAKE_SAMPLING_MIN_MAJOR ||
                 (major == CAKE_SAMPLING_MIN_MAJOR && minor >= CAKE_SAMPLING_MIN_MINOR))
      << "the frozen radix sampling kernels need compute capability " << CAKE_SAMPLING_MIN_MAJOR
      << "." << CAKE_SAMPLING_MIN_MINOR << " or newer, got " << major << "." << minor;
  cudaFuncAttributes attributes;
  const cudaError_t status = cudaFuncGetAttributes(&attributes, probe_kernel);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "the frozen radix sampling module has no kernel image for compute capability " << major
      << "." << minor << " (" << cudaGetErrorString(status)
      << "); rebuild with this architecture in FLASHINFER_CUDA_ARCH_LIST";
}

struct Stage1Variant {
  const void* kernel;
  int32_t cluster;
  int32_t ept;
  int32_t stream;  // 1: streaming variant (any vocab, runtime chunk count); 0: register-resident
  int32_t threads;
  int32_t smem_bytes;
};

struct Stage23Variant {
  const void* kernel;
  int32_t threads;
  int32_t items;
  int32_t smem_bytes;
};

#define CAKE_SAMPLING_STAGE1_ENTRY(symbol, cluster, ept, stream, threads, smem) \
  {reinterpret_cast<const void*>(&symbol), cluster, ept, stream, threads, smem},
#define CAKE_SAMPLING_STAGE23_ENTRY(symbol, threads, items, smem) \
  {reinterpret_cast<const void*>(&symbol), threads, items, smem},

inline const Stage1Variant* FindStage1(int32_t cluster, int32_t ept, int32_t stream) {
  static const Stage1Variant kTable[] = {CAKE_SAMPLING_STAGE1_TABLE(CAKE_SAMPLING_STAGE1_ENTRY)};
  for (const Stage1Variant& v : kTable) {
    if (v.cluster == cluster && v.ept == ept && v.stream == stream) return &v;
  }
  return nullptr;
}

inline const Stage23Variant* FindStage23(int32_t threads, int32_t items) {
  static const Stage23Variant kTable[] = {CAKE_SAMPLING_STAGE23_TABLE(CAKE_SAMPLING_STAGE23_ENTRY)};
  for (const Stage23Variant& v : kTable) {
    if (v.threads == threads && v.items == items) return &v;
  }
  return nullptr;
}

inline void EnsureSmemAttribute(const void* kernel, int32_t smem_bytes) {
  if (smem_bytes > 48 * 1024) {
    CheckCuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes),
              "cudaFuncSetAttribute(MaxDynamicSharedMemorySize)");
  }
}

inline void CheckSlab(const TensorView& vals, const TensorView& idx, const TensorView& count,
                      int64_t rows) {
  CHECK_CUDA(vals);
  CHECK_CUDA(idx);
  CHECK_CUDA(count);
  CHECK_INPUT_TYPE(vals, dl_float32);
  CHECK_INPUT_TYPE(idx, dl_int32);
  CHECK_INPUT_TYPE(count, dl_int32);
  CHECK_CONTIGUOUS(vals);
  CHECK_CONTIGUOUS(idx);
  CHECK_CONTIGUOUS(count);
  TVM_FFI_ICHECK(vals.ndim() == 2 && vals.size(0) >= rows && vals.size(1) == kSlab)
      << "slab values must have shape [>= batch, " << kSlab << "]";
  TVM_FFI_ICHECK(idx.ndim() == 2 && idx.size(0) >= rows && idx.size(1) == kSlab)
      << "slab indices must have shape [>= batch, " << kSlab << "]";
  TVM_FFI_ICHECK(count.ndim() == 1 && count.size(0) >= rows)
      << "slab counts must have shape [>= batch]";
}

// Stage 1: exact top-k of every probability row into the [batch, kSlab] slab.
//   probs      float32 [batch, vocab], row-major, unit inner stride
//   topk_arr   int32 [batch] (only read when topk_kind == kTopKPerRow; pass any int32 CUDA tensor
//   otherwise) cluster/ept/stream_variant select the frozen variant; a register-resident variant
//   (stream_variant == 0) needs cluster * ept * threads >= vocab, a streaming one covers any vocab.
void RadixTopK(TensorView probs, TensorView topk_arr, int64_t topk_scalar, int64_t topk_kind,
               TensorView out_vals, TensorView out_idx, TensorView out_count, int64_t cluster,
               int64_t ept, int64_t stream_variant, int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0) << "cuda_stream must be a non-negative stream handle";
  CHECK_CUDA(probs);
  const int32_t device_id = probs.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CHECK_INPUT_TYPE(probs, dl_float32);
  TVM_FFI_ICHECK(probs.ndim() == 2) << "probs must have shape [batch, vocab]";
  TVM_FFI_ICHECK(probs.stride(1) == 1 && probs.stride(0) == probs.size(1))
      << "probs must be a contiguous [batch, vocab] tensor";
  const int64_t batch = probs.size(0);
  const int64_t vocab = probs.size(1);
  TVM_FFI_ICHECK(vocab > 0 && vocab <= std::numeric_limits<int32_t>::max()) << "vocab out of range";
  CheckSlab(out_vals, out_idx, out_count, batch);
  CHECK_DEVICE(probs, out_vals);
  CHECK_DEVICE(probs, out_idx);
  CHECK_DEVICE(probs, out_count);
  CHECK_CUDA(topk_arr);
  CHECK_INPUT_TYPE(topk_arr, dl_int32);
  TVM_FFI_ICHECK(topk_kind == kTopKScalar || topk_kind == kTopKPerRow) << "invalid topk_kind";
  if (topk_kind == kTopKPerRow) {
    CHECK_DEVICE(probs, topk_arr);
    TVM_FFI_ICHECK(topk_arr.ndim() == 1 && topk_arr.size(0) >= batch)
        << "topk_arr must have shape [batch]";
  } else {
    TVM_FFI_ICHECK(topk_scalar >= 1 && topk_scalar <= kSlab)
        << "top_k must be in [1, " << kSlab << "]";
  }
  TVM_FFI_ICHECK(stream_variant == 0 || stream_variant == 1) << "stream_variant must be 0 or 1";
  const Stage1Variant* v = FindStage1(static_cast<int32_t>(cluster), static_cast<int32_t>(ept),
                                      static_cast<int32_t>(stream_variant));
  TVM_FFI_ICHECK(v != nullptr) << "no frozen stage-1 variant for cluster=" << cluster
                               << " ept=" << ept << " stream=" << stream_variant;
  CheckTarget(device_id, v->kernel);
  TVM_FFI_ICHECK(v->stream == 1 || static_cast<int64_t>(v->cluster) * v->ept * v->threads >= vocab)
      << "stage-1 variant cluster=" << cluster << " ept=" << ept
      << " does not cover vocab=" << vocab;
  if (batch == 0) return;
  EnsureSmemAttribute(v->kernel, v->smem_bytes);

  float* probs_ptr = static_cast<float*>(probs.data_ptr());
  int* topk_ptr = static_cast<int*>(topk_arr.data_ptr());
  float* vals_ptr = static_cast<float*>(out_vals.data_ptr());
  int* idx_ptr = static_cast<int*>(out_idx.data_ptr());
  int* count_ptr = static_cast<int*>(out_count.data_ptr());
  int vocab_i = static_cast<int>(vocab);
  int topk_i = static_cast<int>(topk_scalar);
  int kind_i = static_cast<int>(topk_kind);
  void* args[] = {&probs_ptr, &topk_ptr, &vals_ptr, &idx_ptr,
                  &count_ptr, &vocab_i,  &topk_i,   &kind_i};

  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(static_cast<uint32_t>(batch * v->cluster), 1, 1);
  config.blockDim = dim3(static_cast<uint32_t>(v->threads), 1, 1);
  config.dynamicSmemBytes = static_cast<size_t>(v->smem_bytes);
  config.stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  config.numAttrs = 0;
  // The cluster shape is a compile-time __cluster_dims__ attribute of the frozen kernel.
  CheckCuda(cudaLaunchKernelExC(&config, v->kernel, args), "frozen radix top-k launch");
}

// Stages 2/3: sort the slab (descending probability, ascending index), apply top-p on exact
//   fixed-point prefix sums, renormalize, draw one token per row, and rewrite the slab in sorted
//   order (out_renorm follows that order).  Strictly deterministic: no atomics decide an output.
//   seed/offset are curand Philox parameters: curand_init(seed, row, offset).
//   enable_pdl launches with programmatic stream serialization so the prologue overlaps stage 1.
void SparseTopPSample(TensorView vals, TensorView idx, TensorView count, TensorView topp_arr,
                      double topp_scalar, int64_t topp_kind, TensorView out_samples,
                      TensorView out_renorm, int64_t seed, int64_t offset, int64_t emit_renorm,
                      int64_t threads, int64_t items, int64_t enable_pdl, int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0) << "cuda_stream must be a non-negative stream handle";
  CHECK_CUDA(out_samples);
  const int32_t device_id = out_samples.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CHECK_INPUT_TYPE(out_samples, dl_int32);
  TVM_FFI_ICHECK(out_samples.ndim() == 1) << "out_samples must have shape [batch]";
  CHECK_CONTIGUOUS(out_samples);
  const int64_t batch = out_samples.size(0);
  CheckSlab(vals, idx, count, batch);
  CHECK_DEVICE(out_samples, vals);
  CHECK_DEVICE(out_samples, idx);
  CHECK_DEVICE(out_samples, count);
  CHECK_CUDA(topp_arr);
  CHECK_INPUT_TYPE(topp_arr, dl_float32);
  TVM_FFI_ICHECK(topp_kind == kTopPScalar || topp_kind == kTopPPerRow) << "invalid topp_kind";
  if (topp_kind == kTopPPerRow) {
    CHECK_DEVICE(out_samples, topp_arr);
    TVM_FFI_ICHECK(topp_arr.ndim() == 1 && topp_arr.size(0) >= batch)
        << "topp_arr must have shape [batch]";
  } else {
    TVM_FFI_ICHECK(topp_scalar > 0.0 && topp_scalar <= 1.0) << "top_p must be in (0, 1]";
  }
  CHECK_CUDA(out_renorm);
  CHECK_INPUT_TYPE(out_renorm, dl_float32);
  if (emit_renorm != 0) {
    CHECK_DEVICE(out_samples, out_renorm);
    CHECK_CONTIGUOUS(out_renorm);
    TVM_FFI_ICHECK(out_renorm.ndim() == 2 && out_renorm.size(0) >= batch &&
                   out_renorm.size(1) == kSlab)
        << "out_renorm must have shape [>= batch, " << kSlab << "]";
  }
  const Stage23Variant* v = FindStage23(static_cast<int32_t>(threads), static_cast<int32_t>(items));
  TVM_FFI_ICHECK(v != nullptr) << "no frozen stage-2/3 variant for threads=" << threads
                               << " items=" << items;
  CheckTarget(device_id, v->kernel);
  if (batch == 0) return;
  EnsureSmemAttribute(v->kernel, v->smem_bytes);

  float* vals_ptr = static_cast<float*>(vals.data_ptr());
  int* idx_ptr = static_cast<int*>(idx.data_ptr());
  int* count_ptr = static_cast<int*>(count.data_ptr());
  float* topp_ptr = static_cast<float*>(topp_arr.data_ptr());
  int* samples_ptr = static_cast<int*>(out_samples.data_ptr());
  float* renorm_ptr = static_cast<float*>(out_renorm.data_ptr());
  float topp_f = static_cast<float>(topp_scalar);
  int kind_i = static_cast<int>(topp_kind);
  const uint64_t seed_u = static_cast<uint64_t>(seed);
  const uint64_t offset_u = static_cast<uint64_t>(offset);
  unsigned int seed_lo = static_cast<unsigned int>(seed_u & 0xFFFFFFFFu);
  unsigned int seed_hi = static_cast<unsigned int>(seed_u >> 32);
  unsigned int offset_lo = static_cast<unsigned int>(offset_u & 0xFFFFFFFFu);
  unsigned int offset_hi = static_cast<unsigned int>(offset_u >> 32);
  int renorm_i = emit_renorm != 0 ? 1 : 0;
  void* args[] = {&vals_ptr, &idx_ptr, &count_ptr, &topp_ptr,  &samples_ptr, &renorm_ptr, &topp_f,
                  &kind_i,   &seed_lo, &seed_hi,   &offset_lo, &offset_hi,   &renorm_i};

  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(static_cast<uint32_t>(batch), 1, 1);
  config.blockDim = dim3(static_cast<uint32_t>(v->threads), 1, 1);
  config.dynamicSmemBytes = static_cast<size_t>(v->smem_bytes);
  config.stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl != 0 ? 1 : 0;
  config.attrs = attrs;
  config.numAttrs = 1;
  CheckCuda(cudaLaunchKernelExC(&config, v->kernel, args), "frozen sparse top-p sampling launch");
}

}  // namespace cake_sampling
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(radix_topk, flashinfer::cake_sampling::RadixTopK);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_topp_sample, flashinfer::cake_sampling::SparseTopPSample);
