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
#include <cstdint>

#include "tvm_ffi_utils.h"

void launch_mqa_kernel_metadata(int* seq_lens, int batch_size, int num_physical_sms,
                                int* sm_mapping, cudaStream_t stream);

void launch_mqa_v3_fused_epilogue(uint8_t* q_ptr, uint8_t* k_ptr, float* weights, int* seq_lens,
                                  int* block_table, float* logits, uint32_t* histogram,
                                  int4* sm_map, int max_num_pages, int num_pages, int batch_size,
                                  int num_sms, int sm_multiple, int logit_batch_stride,
                                  bool pdl_enabled, cudaStream_t stream);

void launch_fast_topk_clusters(const float* logits, int* indices, int* seq_lens, int* pre_hist,
                               int batch_size, int logit_stride, int indices_stride, int num_cached,
                               int num_clusters, bool pdl_enabled, cudaStream_t stream);

void launch_fast_topk_clusters_exact(const float* logits, int* indices, int* seq_lens,
                                     int* pre_hist, int* cached_overflow, int overflow_stride,
                                     int batch_size, int logit_stride, int indices_stride,
                                     int num_cached, int num_clusters, bool pdl_enabled,
                                     cudaStream_t stream);

using tvm::ffi::Optional;

// ---------------------------------------------------------------------------
// Tensor check helpers
// ---------------------------------------------------------------------------

static inline bool is_fp8_or_uint8(const TensorView& t) {
  return t.dtype() == dl_float8_e4m3fn || t.dtype() == dl_float8_e5m2 || t.dtype() == dl_uint8;
}

// q: [batch, 64, 128]  fp8/uint8
static void check_q(const TensorView& t, int64_t batch_size, const char* fn) {
  TVM_FFI_ICHECK_EQ(t.ndim(), 3) << fn << ": q must be 3D [batch, 64, 128]";
  TVM_FFI_ICHECK_EQ(t.size(0), batch_size)
      << fn << ": q.size(0) must match batch_size (" << batch_size << ")";
  TVM_FFI_ICHECK_EQ(t.size(1), 64) << fn << ": q.size(1) must be 64";
  TVM_FFI_ICHECK_EQ(t.size(2), 128) << fn << ": q.size(2) must be 128";
  TVM_FFI_ICHECK(is_fp8_or_uint8(t)) << fn << ": q must be float8 or uint8";
  CHECK_CUDA(t);
}

// k_cache: [num_pages, 64, 1, 132]  fp8/uint8
static void check_k_cache(const TensorView& t, const char* fn) {
  TVM_FFI_ICHECK_EQ(t.ndim(), 4) << fn << ": k_cache must be 4D [num_pages, 64, 1, 132]";
  TVM_FFI_ICHECK_EQ(t.size(1), 64) << fn << ": k_cache.size(1) must be 64";
  TVM_FFI_ICHECK_EQ(t.size(2), 1) << fn << ": k_cache.size(2) must be 1";
  TVM_FFI_ICHECK_EQ(t.size(3), 132) << fn << ": k_cache.size(3) must be 132";
  TVM_FFI_ICHECK(is_fp8_or_uint8(t)) << fn << ": k_cache must be float8 or uint8";
  CHECK_CUDA(t);
}

// weights: [batch, 64]  float32
static void check_weights(const TensorView& t, int64_t batch_size, const char* fn) {
  TVM_FFI_ICHECK_EQ(t.ndim(), 2) << fn << ": weights must be 2D [batch, 64]";
  TVM_FFI_ICHECK_EQ(t.size(0), batch_size)
      << fn << ": weights.size(0) must match batch_size (" << batch_size << ")";
  TVM_FFI_ICHECK_EQ(t.size(1), 64) << fn << ": weights.size(1) must be 64";
  CHECK_INPUT_AND_TYPE(t, dl_float32);
}

// seq_lens: [batch]  int32
// Pass batch_size = -1 to skip the size(0) check (used by get_mqa_metadata).
static void check_seq_lens(const TensorView& t, const char* fn, int64_t batch_size = -1) {
  TVM_FFI_ICHECK_EQ(t.ndim(), 1) << fn << ": seq_lens must be 1D [batch]";
  if (batch_size >= 0) {
    TVM_FFI_ICHECK_EQ(t.size(0), batch_size)
        << fn << ": seq_lens.size(0) must match batch_size (" << batch_size << ")";
  }
  CHECK_INPUT_AND_TYPE(t, dl_int32);
}

// block_table: [batch, max_num_pages]  int32
static void check_block_table(const TensorView& t, int64_t batch_size, const char* fn) {
  TVM_FFI_ICHECK_EQ(t.ndim(), 2) << fn << ": block_table must be 2D [batch, max_num_pages]";
  TVM_FFI_ICHECK_EQ(t.size(0), batch_size)
      << fn << ": block_table.size(0) must match batch_size (" << batch_size << ")";
  CHECK_INPUT_AND_TYPE(t, dl_int32);
}

// histogram: [batch, 256]  int32, contiguous
static void check_histogram(const TensorView& t, int64_t batch_size, const char* fn) {
  TVM_FFI_ICHECK_EQ(t.ndim(), 2) << fn << ": histogram must be 2D [batch, 256]";
  TVM_FFI_ICHECK_EQ(t.size(0), batch_size)
      << fn << ": histogram.size(0) must match batch_size (" << batch_size << ")";
  TVM_FFI_ICHECK_EQ(t.size(1), 256) << fn << ": histogram.size(1) must be 256";
  CHECK_INPUT_AND_TYPE(t, dl_int32);
}

// sm_map: [num_sms, 4]  int32
static void check_sm_map(const TensorView& t, const char* fn) {
  TVM_FFI_ICHECK_EQ(t.ndim(), 2) << fn << ": sm_map must be 2D [num_sms, 4]";
  TVM_FFI_ICHECK_EQ(t.size(1), 4) << fn << ": sm_map.size(1) must be 4";
  CHECK_INPUT_AND_TYPE(t, dl_int32);
}

// logits: [batch, ?]  float32
// Checks shape/dtype/device, 16-byte pointer alignment, and that
// stride(0) % 4 == 0.  The kernel-visible stride is stride(0), which may
// differ from size(1) when logits is a non-contiguous slice.
static void check_logits(const TensorView& t, int64_t batch_size, const char* fn) {
  TVM_FFI_ICHECK_EQ(t.ndim(), 2) << fn << ": logits must be 2D [batch, cols]";
  TVM_FFI_ICHECK_EQ(t.size(0), batch_size)
      << fn << ": logits.size(0) must match batch_size (" << batch_size << ")";
  CHECK_CUDA(t);
  CHECK_INPUT_TYPE(t, dl_float32);
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0)
      << fn << ": logits data pointer must be 16-byte aligned";
  TVM_FFI_ICHECK(t.stride(0) % 4 == 0)
      << fn << ": logits.stride(0) (" << t.stride(0) << ") must be divisible by 4";
}

// indices: [batch, 2048]  int32
static void check_indices(const TensorView& t, int64_t batch_size, const char* fn) {
  TVM_FFI_ICHECK_EQ(t.ndim(), 2) << fn << ": indices must be 2D [batch, 2048]";
  TVM_FFI_ICHECK_EQ(t.size(0), batch_size)
      << fn << ": indices.size(0) must match batch_size (" << batch_size << ")";
  TVM_FFI_ICHECK_EQ(t.size(1), 2048) << fn << ": indices.size(1) must be 2048";
  CHECK_INPUT_AND_TYPE(t, dl_int32);
}

// ---------------------------------------------------------------------------
// Exported functions (TVM-FFI)
// ---------------------------------------------------------------------------

// fast_topk_clusters:
//   logits:       [batch, >=logit_stride]  float32, may be non-contiguous
//   indices:      [batch, 2048]            int32
//   seq_lens:     [batch]                  int32
//   histogram:    Optional[batch, 256]     int32; if provided, used as the first-pass histogram
//                                          (pre_hist), skipping the initial histogram scan
//   num_cached:   int64_t
//   num_clusters: int64_t
//   pdl_enabled:  bool
// Fills indices in-place.
void fast_topk_clusters(TensorView logits, TensorView indices, TensorView seq_lens,
                        Optional<TensorView> histogram, int64_t num_cached, int64_t num_clusters,
                        bool pdl_enabled) {
  const int64_t batch_size = logits.size(0);
  check_logits(logits, batch_size, "fast_topk_clusters");
  check_indices(indices, batch_size, "fast_topk_clusters");
  check_seq_lens(seq_lens, "fast_topk_clusters", batch_size);

  const int* hist_ptr = nullptr;
  if (histogram.has_value()) {
    check_histogram(histogram.value(), batch_size, "fast_topk_clusters");
    hist_ptr = static_cast<const int*>(histogram.value().data_ptr());
  }

  const int logit_stride = static_cast<int>(logits.stride(0));
  const int indices_stride = static_cast<int>(indices.stride(0));
  cudaStream_t stream = get_current_stream();

  launch_fast_topk_clusters(
      static_cast<const float*>(logits.data_ptr()), static_cast<int*>(indices.data_ptr()),
      static_cast<int*>(seq_lens.data_ptr()), const_cast<int*>(hist_ptr),
      static_cast<int>(batch_size), logit_stride, indices_stride, static_cast<int>(num_cached),
      static_cast<int>(num_clusters), pdl_enabled, stream);
  auto err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "launch_fast_topk_clusters failed: " << cudaGetErrorString(err);
}

// fast_topk_clusters_exact:
//   logits:          [batch, >=logit_stride]  float32, may be non-contiguous
//   indices:         [batch, 2048]            int32
//   seq_lens:        [batch]                  int32
//   histogram:       Optional[batch, 256]     int32; pre-computed first-byte histogram
//   num_cached:      int64_t  (shared-cache capacity per CTA per phase)
//   num_clusters:    int64_t
//   overflow_stride: int64_t  (global overflow cache capacity per CTA per phase)
//   pdl_enabled:     bool
// Allocates the global overflow buffer internally and fills indices in-place.
void fast_topk_clusters_exact(TensorView logits, TensorView indices, TensorView seq_lens,
                              Optional<TensorView> histogram, int64_t num_cached,
                              int64_t num_clusters, int64_t overflow_stride, bool pdl_enabled) {
  const int64_t batch_size = logits.size(0);
  check_logits(logits, batch_size, "fast_topk_clusters_exact");
  check_indices(indices, batch_size, "fast_topk_clusters_exact");
  check_seq_lens(seq_lens, "fast_topk_clusters_exact", batch_size);

  const int* hist_ptr = nullptr;
  if (histogram.has_value()) {
    check_histogram(histogram.value(), batch_size, "fast_topk_clusters_exact");
    hist_ptr = static_cast<const int*>(histogram.value().data_ptr());
  }

  const int logit_stride = static_cast<int>(logits.stride(0));
  const int indices_stride = static_cast<int>(indices.stride(0));
  const int n_clusters = static_cast<int>(num_clusters);
  const int ovf_stride = static_cast<int>(overflow_stride);
  cudaStream_t stream = get_current_stream();

  // Global overflow buffer: overflow_stride * 4 * num_clusters int32 per batch entry.
  // Each entry is a PackedCachedData (uint32_t bits + int index = 8 bytes = 2 int32).
  auto cached_overflow =
      alloc_tensor({batch_size * ovf_stride * 4 * n_clusters}, dl_int32, logits.device());

  launch_fast_topk_clusters_exact(
      static_cast<const float*>(logits.data_ptr()), static_cast<int*>(indices.data_ptr()),
      static_cast<int*>(seq_lens.data_ptr()), const_cast<int*>(hist_ptr),
      static_cast<int*>(cached_overflow.data_ptr()), ovf_stride, static_cast<int>(batch_size),
      logit_stride, indices_stride, static_cast<int>(num_cached), n_clusters, pdl_enabled, stream);
  auto err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "launch_fast_topk_clusters_exact failed: " << cudaGetErrorString(err);
}

// mqa_topk_indexer:
//   q:           [batch, 64, 128]              fp8/uint8
//   k_cache:     [num_pages, 64, 1, 132]       fp8/uint8
//   weights:     [batch, 64]                   float32
//   seq_lens:    [batch]                       int32
//   block_table: [batch, max_num_pages]        int32
//   histogram:   [batch, 256]                  int32, contiguous, zeroed by caller
//   sm_map:      [num_sms, 4]                  int32, from get_mqa_metadata()
//   logits:      [batch, max_num_pages * 64]   float32, may be non-contiguous
//   indices:     [batch, 2048]                 int32, may be non-contiguous
//   pdl_enabled: bool
//   sm_multiple: int64_t (kernel tuning param)
//   num_cached:  int64_t  (cache budget per sequence for the radix top-K pass)
//   num_clusters: int64_t (number of cooperative thread block clusters)
//   global_topk_overflow: int64_t (number of cached items in global memory for topk)
// Fills logits and indices in-place.
void mqa_topk_indexer(TensorView q, TensorView k_cache, TensorView weights, TensorView seq_lens,
                      TensorView block_table, TensorView histogram, TensorView sm_map,
                      TensorView logits, TensorView indices, bool pdl_enabled, int64_t sm_multiple,
                      int64_t num_cached, int64_t num_clusters, int64_t global_topk_overflow) {
  const int64_t batch_size = q.size(0);
  check_q(q, batch_size, "mqa_topk_indexer");
  check_k_cache(k_cache, "mqa_topk_indexer");
  check_weights(weights, batch_size, "mqa_topk_indexer");
  check_seq_lens(seq_lens, "mqa_topk_indexer", batch_size);
  check_block_table(block_table, batch_size, "mqa_topk_indexer");
  check_histogram(histogram, batch_size, "mqa_topk_indexer");
  check_sm_map(sm_map, "mqa_topk_indexer");
  check_logits(logits, batch_size, "mqa_topk_indexer");
  check_indices(indices, batch_size, "mqa_topk_indexer");

  const int num_pages = static_cast<int>(k_cache.size(0));
  const int max_num_pages = static_cast<int>(block_table.size(1));
  const int num_sms = static_cast<int>(sm_map.size(0));
  const int logit_stride = static_cast<int>(logits.stride(0));
  const int indices_stride = static_cast<int>(indices.stride(0));

  cudaStream_t stream = get_current_stream();

  launch_mqa_v3_fused_epilogue(
      reinterpret_cast<uint8_t*>(q.data_ptr()), reinterpret_cast<uint8_t*>(k_cache.data_ptr()),
      static_cast<float*>(weights.data_ptr()), static_cast<int*>(seq_lens.data_ptr()),
      static_cast<int*>(block_table.data_ptr()), static_cast<float*>(logits.data_ptr()),
      reinterpret_cast<uint32_t*>(histogram.data_ptr()), reinterpret_cast<int4*>(sm_map.data_ptr()),
      max_num_pages, num_pages, static_cast<int>(batch_size), num_sms,
      static_cast<int>(sm_multiple), logit_stride, pdl_enabled, stream);
  {
    auto err = cudaGetLastError();
    TVM_FFI_ICHECK(err == cudaSuccess)
        << "launch_mqa_v3_fused_epilogue failed: " << cudaGetErrorString(err);
  }

  if (global_topk_overflow == 0) {
    fast_topk_clusters(logits, indices, seq_lens, histogram, num_cached, num_clusters, pdl_enabled);
  } else {
    fast_topk_clusters_exact(logits, indices, seq_lens, histogram, num_cached, num_clusters,
                             global_topk_overflow, pdl_enabled);
  }

  {
    auto err = cudaGetLastError();
    TVM_FFI_ICHECK(err == cudaSuccess)
        << "launch_fast_topk_clusters failed: " << cudaGetErrorString(err);
  }
}

// get_mqa_metadata:
//   seq_lens:         [batch]  int32
//   num_physical_sms: int64_t  (pass 0 to auto-detect)
// Returns: sm_map [num_sms, 4] int32
ffi::Tensor get_mqa_metadata(TensorView seq_lens, int64_t num_physical_sms) {
  check_seq_lens(seq_lens, "get_mqa_metadata");

  if (num_physical_sms <= 0) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, seq_lens.device().device_id);
    num_physical_sms = prop.multiProcessorCount;
  }

  const int batch_size = static_cast<int>(seq_lens.size(0));
  int num_logical_sms = (batch_size + num_physical_sms - 1) / num_physical_sms * num_physical_sms;

  auto sm_map = alloc_tensor({num_logical_sms, 4}, dl_int32, seq_lens.device());

  cudaStream_t stream = get_current_stream();
  launch_mqa_kernel_metadata(static_cast<int*>(seq_lens.data_ptr()), batch_size,
                             static_cast<int>(num_physical_sms),
                             static_cast<int*>(sm_map.data_ptr()), stream);
  auto err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "launch_mqa_kernel_metadata failed: " << cudaGetErrorString(err);
  return sm_map;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mqa_topk_indexer, mqa_topk_indexer);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_mqa_metadata, get_mqa_metadata);
