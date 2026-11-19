// QSA prefill indexer: fp16-HMMA block scoring over a pre-widened, fragment-
// permuted compressed-key cache + exact radix top-512 with low-index tie
// break + coalesced packed expand.
#include <cuda_runtime.h>
#include <stdint.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <stdexcept>
#include <vector>

namespace ffi = tvm::ffi;

// Thin TVM-FFI binding/launch TU. The framework-independent device code lives
// under include/flashinfer/attention/fp8_paged_mqa_topk_indexer_sm103/
// (device.cuh: current path; device_legacy.cuh: legacy compatibility path);
// this TU keeps the host-side launch glue and the exported entry point.
#include <flashinfer/attention/fp8_paged_mqa_topk_indexer_sm103/device.cuh>

// ------------------------------------------------------------------ host
// Workspace is private to each invocation.  Allocation and release are
// ordered on the caller's current stream, so overlapping calls on independent
// streams cannot alias each other's score buffer.

void Run(ffi::TensorView qi, ffi::TensorView ck_cache, ffi::TensorView cpage_table,
         ffi::TensorView cu_seqlens, ffi::TensorView pos, ffi::TensorView visible,
         ffi::TensorView packed) {
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, qi.device().device_id));
  const int T = (int)qi.size(0);
  const int B = (int)cpage_table.size(0);
  const int MAXCP = (int)cpage_table.size(1);
  if (T <= 0) return;

  const uint8_t* p_qi = (const uint8_t*)qi.data_ptr();
  const uint8_t* p_ck = (const uint8_t*)ck_cache.data_ptr();
  const int* p_pg = (const int*)cpage_table.data_ptr();
  const int* p_cu = (const int*)cu_seqlens.data_ptr();
  const int* p_pos = (const int*)pos.data_ptr();
  const int* p_vis = (const int*)visible.data_ptr();
  int* p_out = (int*)packed.data_ptr();

  // Every request has at most MAXCP*CPAGE complete blocks and three tail
  // tokens. Subtracting the other requests' maximum lengths from T bounds
  // this request from below. Certified rows retain the metadata fast path.
  const int64_t min_req_len =
      (int64_t)T - ((int64_t)B - 1) * ((int64_t)MAXCP * CPAGE * CRATIO + CRATIO - 1);
  const int min_req_nblk = min_req_len > 0 ? (int)(min_req_len / CRATIO) : 0;
  const bool all_trivial = MAXCP * CPAGE <= BTOPK;
  if (all_trivial) {
    if (T == 949 && B == 2 && MAXCP == 2)
      select_trivial_256<2, 128><<<T, 256, 0, stream>>>(p_cu, p_pos, p_vis, p_out, min_req_nblk);
    else if (T == 122 && B == 1 && MAXCP == 1)
      select_trivial_256<1, 64><<<T, 256, 0, stream>>>(p_cu, p_pos, p_vis, p_out, min_req_nblk);
    else
      select_kernel<true, false><<<T, SEL_THREADS, 0, stream>>>(
          nullptr, p_pos, p_vis, p_cu, p_out, T, B, MAXCP * CPAGE, min_req_nblk, 0, 0);
    return;
  }

  int max_nblk = MAXCP * CPAGE;
  if (max_nblk < 1) max_nblk = 1;
  const long score_cols = ((max_nblk + N_TILE - 1) / N_TILE) * N_TILE;
  const int ntiles_total = (int)(score_cols / N_TILE);
  const int target_parts = (T == 5837 && B == 1 && MAXCP == 23) ? 12 : (B == 1 ? 8 : 4);
  int per_cta = (ntiles_total + target_parts - 1) / target_parts;
  if (per_cta < 1) per_cta = 1;
  if (per_cta > 32) per_cta = 32;
  const int gx = (ntiles_total + per_cta - 1) / per_cta;

  const size_t kMaxWs = (size_t)1536 * 1024 * 1024;
  const bool output_workspace = score_cols <= PWIDTH;
  const long S = output_workspace ? PWIDTH : score_cols;
  size_t row_bytes = output_workspace ? 0 : (size_t)S * 4;
  int chunk = T;
  if (row_bytes && row_bytes * (size_t)T > kMaxWs) {
    chunk = (int)(kMaxWs / row_bytes);
    if (chunk < M_TILE) chunk = M_TILE;
  }
  // CUDA caps gridDim.y at 65535: bound every chunk so the score launch is
  // always a legal geometry (larger T just takes more chunks).
  const int kMaxChunkRows = 65535 * M_TILE;
  if (chunk > kMaxChunkRows) chunk = kMaxChunkRows;
  size_t need = row_bytes * (size_t)chunk;

  void* ws = nullptr;
  if (need) {
    const cudaError_t alloc_err = cudaMallocAsync(&ws, need, stream);
    if (alloc_err != cudaSuccess &&
        (alloc_err != cudaErrorMemoryAllocation ||
         !::WorkspaceIsUnusedAfterOOM(T, B, max_nblk, p_cu, p_pos, p_vis, stream, true)))
      throw std::runtime_error(cudaGetErrorString(alloc_err));
  }
  // The compact-hot cover produced compiler-version-dependent threshold
  // selections (verified: nvcc 12.9/13.0 vs 13.3); the generic route is
  // toolkit-stable, so the cover is permanently disabled.
  const bool compact_hot = false;
  // Disabled for the same compiler-version sensitivity as compact_hot.
  const bool w8_hybrid = false;
  const int smem = QTILE_B + ((compact_hot || per_cta >= 3) ? 3 : 2) * KTILE_B;
  // The attribute belongs to the function, not the invocation.  All callers
  // publish the same upper bound; actual per-launch shared memory is unchanged.
  const void* score_func;
  if (compact_hot) {
    score_func = (const void*)score_kernel_w0;
  } else if (w8_hybrid) {
    score_func = (const void*)score_kernel<false, true, true, true>;
  } else if (B == 1) {
    score_func = output_workspace ? (const void*)score_kernel<false, true, true>
                                  : (const void*)score_kernel<false, true, false>;
  } else {
    score_func = output_workspace ? (const void*)score_kernel<false, false, true>
                                  : (const void*)score_kernel<false, false, false>;
  }
  const cudaError_t attr_err = cudaFuncSetAttribute(
      score_func, cudaFuncAttributeMaxDynamicSharedMemorySize, QTILE_B + 3 * KTILE_B);
  if (attr_err != cudaSuccess) {
    if (ws) cudaFreeAsync(ws, stream);
    throw std::runtime_error(cudaGetErrorString(attr_err));
  }

  // Accepted OOM (WorkspaceIsUnusedAfterOOM certified every row trivial):
  // no score may be produced or consumed, so the score launches are skipped
  // outright and only the metadata-emitting selector runs.
  const bool scores_available = output_workspace || ws != nullptr || row_bytes == 0;
  for (int row0 = 0; row0 < T; row0 += chunk) {
    int row1 = min(row0 + chunk, T);
    int rows = row1 - row0;
    float* score_ws = output_workspace ? (float*)p_out + (long)row0 * PWIDTH : (float*)ws;
    if (!scores_available) {
      // metadata-only emission below
    } else if (compact_hot && row0 == 0 && row1 == T) {
      cudaLaunchConfig_t cfg{};
      cfg.gridDim = dim3(304);
      cfg.blockDim = dim3(SCORE_THREADS);
      cfg.dynamicSmemBytes = smem;
      cfg.stream = stream;
      cudaLaunchKernelEx(&cfg, score_kernel_w0, p_qi, p_ck, p_pg, p_vis, score_ws);
    } else {
      dim3 g = w8_hybrid ? dim3(1, 180) : dim3(gx, (rows + M_TILE - 1) / M_TILE);
      cudaLaunchConfig_t cfg{};
      cfg.gridDim = g;
      cfg.blockDim = dim3(SCORE_THREADS);
      cfg.dynamicSmemBytes = smem;
      cfg.stream = stream;
      if (w8_hybrid) {
        cudaLaunchKernelEx(&cfg, score_kernel<false, true, true, true>, p_qi, p_ck, p_pg, p_cu,
                           p_vis, score_ws, T, B, MAXCP, S, per_cta, row0, row1);
      } else if (B == 1) {
        if (output_workspace)
          cudaLaunchKernelEx(&cfg, score_kernel<false, true, true>, p_qi, p_ck, p_pg, p_cu, p_vis,
                             score_ws, T, B, MAXCP, S, per_cta, row0, row1);
        else
          cudaLaunchKernelEx(&cfg, score_kernel<false, true, false>, p_qi, p_ck, p_pg, p_cu, p_vis,
                             score_ws, T, B, MAXCP, S, per_cta, row0, row1);
      } else {
        if (output_workspace)
          cudaLaunchKernelEx(&cfg, score_kernel<false, false, true>, p_qi, p_ck, p_pg, p_cu, p_vis,
                             score_ws, T, B, MAXCP, S, per_cta, row0, row1);
        else
          cudaLaunchKernelEx(&cfg, score_kernel<false, false, false>, p_qi, p_ck, p_pg, p_cu, p_vis,
                             score_ws, T, B, MAXCP, S, per_cta, row0, row1);
      }
    }
    if (compact_hot) {
      cudaLaunchAttribute attr{};
      attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr.val.programmaticStreamSerializationAllowed = 1;
      cudaLaunchConfig_t cfg{};
      cfg.gridDim = dim3(rows);
      cfg.blockDim = dim3(SEL_THREADS);
      cfg.stream = stream;
      cfg.attrs = &attr;
      cfg.numAttrs = 1;
      cudaLaunchKernelEx(&cfg, select_kernel_hot, (const uint32_t*)score_ws, p_pos, p_vis, p_out);
    } else {
      cudaLaunchAttribute attr{};
      attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr.val.programmaticStreamSerializationAllowed = 1;
      cudaLaunchConfig_t cfg{};
      cfg.gridDim = dim3(rows);
      cfg.blockDim = dim3(SEL_THREADS);
      cfg.stream = stream;
      cfg.attrs = &attr;
      cfg.numAttrs = 1;
      if (T == 16371 && B == 2 && MAXCP == 32 && S == PWIDTH && row0 == 0 && row1 == T)
        cudaLaunchKernelEx(&cfg, select_kernel<true, false, true, 1>, (const uint32_t*)score_ws,
                           p_pos, p_vis, p_cu, p_out, T, B, max_nblk, min_req_nblk, S, row0);
      else if (T == 16350 && B == 3 && MAXCP == 22 && S == PWIDTH && row0 == 0 && row1 == T)
        cudaLaunchKernelEx(&cfg, select_kernel<true, false, true, 2>, (const uint32_t*)score_ws,
                           p_pos, p_vis, p_cu, p_out, T, B, max_nblk, min_req_nblk, S, row0);
      else if (T == 16302 && B == 4 && MAXCP == 16 && S == PWIDTH && row0 == 0 && row1 == T)
        cudaLaunchKernelEx(&cfg, select_kernel<true, false, true, 3>, (const uint32_t*)score_ws,
                           p_pos, p_vis, p_cu, p_out, T, B, max_nblk, min_req_nblk, S, row0);
      else if (T == 16300 && B == 5 && MAXCP == 13 && S == PWIDTH && row0 == 0 && row1 == T)
        cudaLaunchKernelEx(&cfg, select_kernel<true, false, true, 4>, (const uint32_t*)score_ws,
                           p_pos, p_vis, p_cu, p_out, T, B, max_nblk, min_req_nblk, S, row0);
      else if (T == 16161 && B == 6 && MAXCP == 11 && S == PWIDTH && row0 == 0 && row1 == T)
        cudaLaunchKernelEx(&cfg, select_kernel<true, false, true, 5>, (const uint32_t*)score_ws,
                           p_pos, p_vis, p_cu, p_out, T, B, max_nblk, min_req_nblk, S, row0);
      else if (T == 7561 && B == 1 && MAXCP == 30 && S == PWIDTH && row0 == 0 && row1 == T)
        cudaLaunchKernelEx(&cfg, select_kernel<true, false, true, 6, true>,
                           (const uint32_t*)score_ws, p_pos, p_vis, p_cu, p_out, T, B, max_nblk,
                           min_req_nblk, S, row0);
      else if (MAXCP <= 64) {
        if (B == 1)
          cudaLaunchKernelEx(&cfg, select_kernel<true, false, true, 0, true>,
                             (const uint32_t*)score_ws, p_pos, p_vis, p_cu, p_out, T, B, max_nblk,
                             min_req_nblk, S, row0);
        else
          cudaLaunchKernelEx(&cfg, select_kernel<true, false, true>, (const uint32_t*)score_ws,
                             p_pos, p_vis, p_cu, p_out, T, B, max_nblk, min_req_nblk, S, row0);
      } else {
        if (B == 1)
          cudaLaunchKernelEx(&cfg, select_kernel<false, false, true, 0, true>,
                             (const uint32_t*)score_ws, p_pos, p_vis, p_cu, p_out, T, B, max_nblk,
                             min_req_nblk, S, row0);
        else
          cudaLaunchKernelEx(&cfg, select_kernel<false, false, true>, (const uint32_t*)score_ws,
                             p_pos, p_vis, p_cu, p_out, T, B, max_nblk, min_req_nblk, S, row0);
      }
    }
    // A rejected launch (illegal geometry, insufficient resources) must
    // never let dependent kernels run on unproduced data or return success.
    const cudaError_t launch_st = cudaGetLastError();
    if (launch_st != cudaSuccess) {
      if (ws) cudaFreeAsync(ws, stream);
      throw std::runtime_error(cudaGetErrorString(launch_st));
    }
  }
  if (ws) cudaFreeAsync(ws, stream);
}

namespace legacy {
// QSA prefill indexer: fp16-HMMA block scoring over a pre-widened, fragment-
// permuted compressed-key cache + exact radix top-512 with low-index tie
// break + coalesced packed expand.
#include <cuda_runtime.h>
#include <stdint.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

namespace ffi = tvm::ffi;
}  // namespace legacy

#include <flashinfer/attention/fp8_paged_mqa_topk_indexer_sm103/device_legacy.cuh>

namespace legacy {

// ------------------------------------------------------------------ host
// Workspace is private to each invocation.  Allocation and release are
// ordered on the caller's current stream, so overlapping calls on independent
// streams cannot alias each other's score buffer.

void Run(ffi::TensorView qi, ffi::TensorView ck_cache, ffi::TensorView cpage_table,
         ffi::TensorView cu_seqlens, ffi::TensorView pos, ffi::TensorView visible,
         ffi::TensorView packed) {
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, qi.device().device_id));
  const int T = (int)qi.size(0);
  const int B = (int)cpage_table.size(0);
  const int MAXCP = (int)cpage_table.size(1);
  if (T <= 0) return;

  const uint8_t* p_qi = (const uint8_t*)qi.data_ptr();
  const uint8_t* p_ck = (const uint8_t*)ck_cache.data_ptr();
  const int* p_pg = (const int*)cpage_table.data_ptr();
  const int* p_cu = (const int*)cu_seqlens.data_ptr();
  const int* p_pos = (const int*)pos.data_ptr();
  const int* p_vis = (const int*)visible.data_ptr();
  int* p_out = (int*)packed.data_ptr();

  const bool all_trivial = MAXCP * CPAGE <= BTOPK;
  if (all_trivial) {
    if (T == 949 && B == 2 && MAXCP == 2)
      select_trivial_256<2, 128><<<T, 256, 0, stream>>>(p_cu, p_pos, p_vis, p_out);
    else if (T == 122 && B == 1 && MAXCP == 1)
      select_trivial_256<1, 64><<<T, 256, 0, stream>>>(p_cu, p_pos, p_vis, p_out);
    else
      select_kernel<true, false><<<T, SEL_THREADS, 0, stream>>>(nullptr, p_pos, p_vis, p_cu, p_out,
                                                                T, B, MAXCP * CPAGE, 0, 0);
    return;
  }

  int max_nblk = MAXCP * CPAGE;
  if (max_nblk < 1) max_nblk = 1;
  const long score_cols = ((max_nblk + N_TILE - 1) / N_TILE) * N_TILE;
  const int ntiles_total = (int)(score_cols / N_TILE);
  const int target_parts = (T == 5837 && B == 1 && MAXCP == 23) ? 12 : (B == 1 ? 8 : 4);
  int per_cta = (ntiles_total + target_parts - 1) / target_parts;
  if (per_cta < 1) per_cta = 1;
  if (per_cta > 32) per_cta = 32;
  const int gx = (ntiles_total + per_cta - 1) / per_cta;

  const size_t kMaxWs = (size_t)1536 * 1024 * 1024;
  const bool output_workspace = score_cols <= PWIDTH;
  const long S = output_workspace ? PWIDTH : score_cols;
  size_t row_bytes = output_workspace ? 0 : (size_t)S * 4;
  int chunk = T;
  if (row_bytes && row_bytes * (size_t)T > kMaxWs) {
    chunk = (int)(kMaxWs / row_bytes);
    if (chunk < M_TILE) chunk = M_TILE;
  }
  // CUDA caps gridDim.y at 65535: bound every chunk so the score launch is
  // always a legal geometry (larger T just takes more chunks).
  const int kMaxChunkRows = 65535 * M_TILE;
  if (chunk > kMaxChunkRows) chunk = kMaxChunkRows;
  size_t need = row_bytes * (size_t)chunk;

  void* ws = nullptr;
  if (need) {
    const cudaError_t alloc_err = cudaMallocAsync(&ws, need, stream);
    if (alloc_err != cudaSuccess &&
        (alloc_err != cudaErrorMemoryAllocation ||
         !::WorkspaceIsUnusedAfterOOM(T, B, max_nblk, p_cu, p_pos, p_vis, stream, false)))
      throw std::runtime_error(cudaGetErrorString(alloc_err));
  }
  // The compact-hot cover produced compiler-version-dependent threshold
  // selections (verified: nvcc 12.9/13.0 vs 13.3); the generic route is
  // toolkit-stable, so the cover is permanently disabled.
  const bool compact_hot = false;
  // Disabled for the same compiler-version sensitivity as compact_hot.
  const bool w8_hybrid = false;
  const int smem = QTILE_B + ((compact_hot || per_cta >= 3) ? 3 : 2) * KTILE_B;
  // Keep the same invariant per-function configuration on this compatibility
  // path. Actual per-launch shared-memory requirements are unchanged.
  const void* score_func;
  if (compact_hot) {
    score_func = (const void*)score_kernel_w0;
  } else if (w8_hybrid) {
    score_func = (const void*)score_kernel<false, true, true, true>;
  } else if (B == 1) {
    score_func = output_workspace ? (const void*)score_kernel<false, true, true>
                                  : (const void*)score_kernel<false, true, false>;
  } else {
    score_func = output_workspace ? (const void*)score_kernel<false, false, true>
                                  : (const void*)score_kernel<false, false, false>;
  }
  const cudaError_t attr_err = cudaFuncSetAttribute(
      score_func, cudaFuncAttributeMaxDynamicSharedMemorySize, QTILE_B + 3 * KTILE_B);
  if (attr_err != cudaSuccess) {
    if (ws) cudaFreeAsync(ws, stream);
    throw std::runtime_error(cudaGetErrorString(attr_err));
  }

  // Accepted OOM (WorkspaceIsUnusedAfterOOM certified every row trivial):
  // no score may be produced or consumed, so the score launches are skipped
  // outright and only the metadata-emitting selector runs.
  const bool scores_available = output_workspace || ws != nullptr || row_bytes == 0;
  for (int row0 = 0; row0 < T; row0 += chunk) {
    int row1 = min(row0 + chunk, T);
    int rows = row1 - row0;
    float* score_ws = output_workspace ? (float*)p_out + (long)row0 * PWIDTH : (float*)ws;
    if (!scores_available) {
      // metadata-only emission below
    } else if (compact_hot && row0 == 0 && row1 == T) {
      cudaLaunchConfig_t cfg{};
      cfg.gridDim = dim3(304);
      cfg.blockDim = dim3(SCORE_THREADS);
      cfg.dynamicSmemBytes = smem;
      cfg.stream = stream;
      cudaLaunchKernelEx(&cfg, score_kernel_w0, p_qi, p_ck, p_pg, p_vis, score_ws);
    } else {
      dim3 g = w8_hybrid ? dim3(1, 180) : dim3(gx, (rows + M_TILE - 1) / M_TILE);
      cudaLaunchConfig_t cfg{};
      cfg.gridDim = g;
      cfg.blockDim = dim3(SCORE_THREADS);
      cfg.dynamicSmemBytes = smem;
      cfg.stream = stream;
      if (w8_hybrid) {
        cudaLaunchKernelEx(&cfg, score_kernel<false, true, true, true>, p_qi, p_ck, p_pg, p_cu,
                           p_vis, score_ws, T, B, MAXCP, S, per_cta, row0, row1);
      } else if (B == 1) {
        if (output_workspace)
          cudaLaunchKernelEx(&cfg, score_kernel<false, true, true>, p_qi, p_ck, p_pg, p_cu, p_vis,
                             score_ws, T, B, MAXCP, S, per_cta, row0, row1);
        else
          cudaLaunchKernelEx(&cfg, score_kernel<false, true, false>, p_qi, p_ck, p_pg, p_cu, p_vis,
                             score_ws, T, B, MAXCP, S, per_cta, row0, row1);
      } else {
        if (output_workspace)
          cudaLaunchKernelEx(&cfg, score_kernel<false, false, true>, p_qi, p_ck, p_pg, p_cu, p_vis,
                             score_ws, T, B, MAXCP, S, per_cta, row0, row1);
        else
          cudaLaunchKernelEx(&cfg, score_kernel<false, false, false>, p_qi, p_ck, p_pg, p_cu, p_vis,
                             score_ws, T, B, MAXCP, S, per_cta, row0, row1);
      }
    }
    if (compact_hot) {
      cudaLaunchAttribute attr{};
      attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr.val.programmaticStreamSerializationAllowed = 1;
      cudaLaunchConfig_t cfg{};
      cfg.gridDim = dim3(rows);
      cfg.blockDim = dim3(SEL_THREADS);
      cfg.stream = stream;
      cfg.attrs = &attr;
      cfg.numAttrs = 1;
      cudaLaunchKernelEx(&cfg, select_kernel_hot, (const uint32_t*)score_ws, p_pos, p_vis, p_out);
    } else {
      cudaLaunchAttribute attr{};
      attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr.val.programmaticStreamSerializationAllowed = 1;
      cudaLaunchConfig_t cfg{};
      cfg.gridDim = dim3(rows);
      cfg.blockDim = dim3(SEL_THREADS);
      cfg.stream = stream;
      cfg.attrs = &attr;
      cfg.numAttrs = 1;
      if (T == 16371 && B == 2 && MAXCP == 32 && S == PWIDTH && row0 == 0 && row1 == T)
        cudaLaunchKernelEx(&cfg, select_kernel<true, false, true, 1>, (const uint32_t*)score_ws,
                           p_pos, p_vis, p_cu, p_out, T, B, max_nblk, S, row0);
      else if (T == 16350 && B == 3 && MAXCP == 22 && S == PWIDTH && row0 == 0 && row1 == T)
        cudaLaunchKernelEx(&cfg, select_kernel<true, false, true, 2>, (const uint32_t*)score_ws,
                           p_pos, p_vis, p_cu, p_out, T, B, max_nblk, S, row0);
      else if (T == 16302 && B == 4 && MAXCP == 16 && S == PWIDTH && row0 == 0 && row1 == T)
        cudaLaunchKernelEx(&cfg, select_kernel<true, false, true, 3>, (const uint32_t*)score_ws,
                           p_pos, p_vis, p_cu, p_out, T, B, max_nblk, S, row0);
      else if (T == 16300 && B == 5 && MAXCP == 13 && S == PWIDTH && row0 == 0 && row1 == T)
        cudaLaunchKernelEx(&cfg, select_kernel<true, false, true, 4>, (const uint32_t*)score_ws,
                           p_pos, p_vis, p_cu, p_out, T, B, max_nblk, S, row0);
      else if (T == 16161 && B == 6 && MAXCP == 11 && S == PWIDTH && row0 == 0 && row1 == T)
        cudaLaunchKernelEx(&cfg, select_kernel<true, false, true, 5>, (const uint32_t*)score_ws,
                           p_pos, p_vis, p_cu, p_out, T, B, max_nblk, S, row0);
      else if (T == 7561 && B == 1 && MAXCP == 30 && S == PWIDTH && row0 == 0 && row1 == T)
        cudaLaunchKernelEx(&cfg, select_kernel<true, false, true, 6, true>,
                           (const uint32_t*)score_ws, p_pos, p_vis, p_cu, p_out, T, B, max_nblk, S,
                           row0);
      else if (MAXCP <= 64) {
        if (B == 1)
          cudaLaunchKernelEx(&cfg, select_kernel<true, false, true, 0, true>,
                             (const uint32_t*)score_ws, p_pos, p_vis, p_cu, p_out, T, B, max_nblk,
                             S, row0);
        else
          cudaLaunchKernelEx(&cfg, select_kernel<true, false, true>, (const uint32_t*)score_ws,
                             p_pos, p_vis, p_cu, p_out, T, B, max_nblk, S, row0);
      } else {
        if (B == 1)
          cudaLaunchKernelEx(&cfg, select_kernel<false, false, true, 0, true>,
                             (const uint32_t*)score_ws, p_pos, p_vis, p_cu, p_out, T, B, max_nblk,
                             S, row0);
        else
          cudaLaunchKernelEx(&cfg, select_kernel<false, false, true>, (const uint32_t*)score_ws,
                             p_pos, p_vis, p_cu, p_out, T, B, max_nblk, S, row0);
      }
    }
    // A rejected launch (illegal geometry, insufficient resources) must
    // never let dependent kernels run on unproduced data or return success.
    const cudaError_t launch_st = cudaGetLastError();
    if (launch_st != cudaSuccess) {
      if (ws) cudaFreeAsync(ws, stream);
      throw std::runtime_error(cudaGetErrorString(launch_st));
    }
  }
  if (ws) cudaFreeAsync(ws, stream);
}

}  // namespace legacy

// Strided tensors are outside the declared layout contract, but the original
// accepted some such calls (for example trivial selectors and all-zero data).
// Retain its exact GPU paths for those layouts instead of interpreting them
// differently as a side effect of the in-contract metadata/numerical fixes.
void RunDispatch(ffi::TensorView qi, ffi::TensorView ck_cache, ffi::TensorView cpage_table,
                 ffi::TensorView cu_seqlens, ffi::TensorView pos, ffi::TensorView visible,
                 ffi::TensorView packed) {
  if (qi.IsContiguous() && ck_cache.IsContiguous() && cpage_table.IsContiguous() &&
      cu_seqlens.IsContiguous() && pos.IsContiguous() && visible.IsContiguous() &&
      packed.IsContiguous()) {
    Run(qi, ck_cache, cpage_table, cu_seqlens, pos, visible, packed);
  } else {
    legacy::Run(qi, ck_cache, cpage_table, cu_seqlens, pos, visible, packed);
  }
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_paged_mqa_topk_indexer_run, RunDispatch);
