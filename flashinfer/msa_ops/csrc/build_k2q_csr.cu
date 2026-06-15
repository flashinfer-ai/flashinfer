// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer team
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from MSA (Minimax Sparse Attention) for SM120/SM121 support.
// Original: MSA/python/fmha_sm100/cute/src/sm100/build_k2q_csr/build_k2q_csr.cu
//
// Changes vs. original:
//   - Replaced torch/pybind11 bindings with TVM-FFI (TensorView, TVM_FFI_ICHECK,
//     TVM_FFI_DLL_EXPORT_TYPED_FUNC, get_current_stream).
//   - Replaced internal torch::zeros/empty temporaries with RAII cudaMalloc.
//   - SMEM budget comment updated: SM120 has 256KB; 228KB cap is conservative.

// CUDA C++ q2k -> k2q CSR builder.
//
// Five-stage pipeline. q-ascending order within each CSR row is preserved
// by partitioning q across (CTA, warp_in_CTA) units; each unit owns a
// contiguous q-sub-range and reserves a contiguous slot range per row via
// a precomputed exclusive prefix scan.
//
//   M:  build_row_map      -- round-robin packing of rows across batches
//   H:  histogram + tile_counts
//   PR: row prefix         -- single block per head, row_counts -> row_ptr
//   PT: tile prefix        -- multi-block, scan tile_counts along (c, w) axis
//   S:  scatter (sorted)   -- per-warp slot range, q-sequential within warp

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>

#include "tvm_ffi_utils.h"

namespace {

constexpr int kWarpSize = 32;

// ---------------------------------------------------------------------------
// RAII wrapper for temporary CUDA device allocations.
// ---------------------------------------------------------------------------
struct CudaBuf {
  void* ptr = nullptr;
  CudaBuf() = default;
  explicit CudaBuf(size_t bytes) {
    if (bytes > 0) {
      cudaError_t err = cudaMalloc(&ptr, bytes);
      TVM_FFI_ICHECK(err == cudaSuccess)
          << "cudaMalloc(" << bytes << ") failed: " << cudaGetErrorString(err);
    }
  }
  ~CudaBuf() {
    if (ptr) cudaFree(ptr);
  }
  CudaBuf(const CudaBuf&) = delete;
  CudaBuf& operator=(const CudaBuf&) = delete;
  template <typename T>
  T* as() {
    return static_cast<T*>(ptr);
  }
};

__device__ __forceinline__ void advance_batch_only(int const* __restrict__ cu_q, int B, int q_abs,
                                                   int& bi) {
  while (bi < B && cu_q[bi + 1] <= q_abs) ++bi;
}

// Atomic increment of a 16-bit half within a 32-bit SMEM word; returns OLD 16-bit value.
__device__ __forceinline__ int atomic_inc_int16_packed(int* base_int32, int row) {
  int idx = row >> 1;
  int shift = (row & 1) << 4;
  int delta = 1 << shift;
  int old = atomicAdd(&base_int32[idx], delta);
  return (old >> shift) & 0xFFFF;
}

__device__ __forceinline__ int read_int16_packed(int const* base_int32, int row) {
  int v = base_int32[row >> 1];
  int shift = (row & 1) << 4;
  return (v >> shift) & 0xFFFF;
}

// ---------------------------------------------------------------------------
// M: round-robin row map.
// ---------------------------------------------------------------------------
template <int kBlockK>
__global__ void k2q_build_row_map_kernel(int const* __restrict__ cu_k, int* __restrict__ row_map,
                                         int* __restrict__ row_coords, int B, int max_kv_blocks) {
  int level = blockIdx.x;
  if (level >= max_kv_blocks) return;
  if (threadIdx.x != 0) return;
  int rows_before = 0;
  for (int b = 0; b < B; ++b) {
    int rb = (cu_k[b + 1] - cu_k[b] + kBlockK - 1) / kBlockK;
    rows_before += (rb < level ? rb : level);
  }
  int active_before = 0;
  for (int b = 0; b < B; ++b) {
    int rb = (cu_k[b + 1] - cu_k[b] + kBlockK - 1) / kBlockK;
    if (rb > level) {
      int row_linear = rows_before + active_before;
      row_map[(size_t)b * max_kv_blocks + level] = row_linear;
      if (row_coords != nullptr) {
        row_coords[(size_t)row_linear * 2] = b;
        row_coords[(size_t)row_linear * 2 + 1] = level;
      }
      ++active_before;
    } else {
      row_map[(size_t)b * max_kv_blocks + level] = -1;
    }
  }
}

// ---------------------------------------------------------------------------
// H: per-warp histogram + tile_counts.
// ---------------------------------------------------------------------------
template <int kTopK, int kBlockK, int kWarps>
__global__ void k2q_hist_kernel(int const* __restrict__ q2k, int const* __restrict__ cu_q,
                                int const* __restrict__ row_map, int* __restrict__ row_counts,
                                int* __restrict__ tile_counts, int H, int B, int S_Q,
                                int total_rows, int max_kv_blocks, int q_per_cta, int q_per_warp) {
  constexpr int kThreads = kWarps * kWarpSize;
  extern __shared__ int smem_hist_int[];
  int tid = threadIdx.x;
  int warp_id = tid >> 5;
  int lane = tid & 31;
  int c = blockIdx.x;
  int q_start_cta = c * q_per_cta;
  int q_end_cta = min(q_start_cta + q_per_cta, S_Q);
  int q_start_warp = min(q_start_cta + warp_id * q_per_warp, q_end_cta);
  int q_end_warp = min(q_start_warp + q_per_warp, q_end_cta);

  constexpr int kInt4PerToken = kTopK / 4;
  int packed_per_warp = (total_rows + 1) >> 1;
  int* my_hist = smem_hist_int + warp_id * packed_per_warp;

  for (int h = 0; h < H; ++h) {
    for (int i = lane; i < packed_per_warp; i += kWarpSize) my_hist[i] = 0;
    __syncthreads();

    if (q_start_warp < q_end_warp) {
      int bi = 0;
      int qi = q_start_warp + lane;
      advance_batch_only(cu_q, B, qi, bi);
      int4 const* head_topk4 = reinterpret_cast<int4 const*>(q2k + (size_t)h * S_Q * kTopK);
      for (; qi < q_end_warp; qi += kWarpSize) {
        advance_batch_only(cu_q, B, qi, bi);
        int const* my_row_map = row_map + (size_t)bi * max_kv_blocks;
        int4 buf[kInt4PerToken];
#pragma unroll
        for (int v = 0; v < kInt4PerToken; ++v) buf[v] = head_topk4[(size_t)qi * kInt4PerToken + v];
#pragma unroll
        for (int t = 0; t < kTopK; ++t) {
          int kvb = reinterpret_cast<int const*>(buf)[t];
          if (kvb >= 0 && kvb < max_kv_blocks) {
            int row = my_row_map[kvb];
            if (row >= 0 && row < total_rows) atomic_inc_int16_packed(my_hist, row);
          }
        }
      }
    }
    __syncthreads();

    int* head_row_counts = row_counts + (size_t)h * total_rows;
    int* my_tile = tile_counts + ((size_t)(c * kWarps + warp_id) * H + h) * total_rows;
    for (int i = lane; i < total_rows; i += kWarpSize) my_tile[i] = read_int16_packed(my_hist, i);
    __syncthreads();

    for (int i = tid; i < total_rows; i += kThreads) {
      int sum = 0;
#pragma unroll
      for (int w = 0; w < kWarps; ++w)
        sum += read_int16_packed(smem_hist_int + w * packed_per_warp, i);
      if (sum > 0) atomicAdd(&head_row_counts[i], sum);
    }
    if (h + 1 < H) __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// PR: row prefix. One block per head.
// ---------------------------------------------------------------------------
template <int kThreads>
__global__ void k2q_row_prefix_kernel(int const* __restrict__ row_counts, int* __restrict__ row_ptr,
                                      int const* __restrict__ row_coords,
                                      int* __restrict__ scheduler_metadata,
                                      int* __restrict__ work_count, int total_rows,
                                      int target_q_per_cta, int work_capacity) {
  int h = blockIdx.x;
  int tid = threadIdx.x;
  __shared__ int scan_buf[kThreads];

  int const* head_counts = row_counts + (size_t)h * total_rows;
  int* head_rowptr = row_ptr + (size_t)h * (total_rows + 1);
  int chunk = (total_rows + kThreads - 1) / kThreads;
  int lo = tid * chunk;
  int hi = min(lo + chunk, total_rows);

  int local_sum = 0;
  for (int i = lo; i < hi; ++i) local_sum += head_counts[i];
  scan_buf[tid] = local_sum;
  __syncthreads();

  for (int off = 1; off < kThreads; off <<= 1) {
    int add = (tid >= off) ? scan_buf[tid - off] : 0;
    __syncthreads();
    scan_buf[tid] += add;
    __syncthreads();
  }
  int running = scan_buf[tid] - local_sum;
  for (int i = lo; i < hi; ++i) {
    int row_count = head_counts[i];
    running += row_count;
    head_rowptr[i + 1] = running;
    if (scheduler_metadata != nullptr && work_count != nullptr && row_count > 0) {
      int num_chunks = (row_count + target_q_per_cta - 1) / target_q_per_cta;
      int base = atomicAdd(work_count, num_chunks);
      int batch_idx = row_coords[(size_t)i * 2];
      int kv_block_idx = row_coords[(size_t)i * 2 + 1];
      for (int c = 0; c < num_chunks; ++c) {
        int work_idx = base + c;
        if (work_idx < work_capacity) {
          int q_begin = c * target_q_per_cta;
          int q_count = min(target_q_per_cta, row_count - q_begin);
          int* meta = scheduler_metadata + (size_t)work_idx * 6;
          meta[0] = h;
          meta[1] = i;
          meta[2] = q_begin;
          meta[3] = q_count;
          meta[4] = batch_idx;
          meta[5] = kv_block_idx;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// PT_smem: SMEM-staged tile prefix scan.
// ---------------------------------------------------------------------------
template <int kThreads, int kRowsPerBlock>
__global__ void k2q_tile_prefix_smem_kernel(int* __restrict__ tile_counts,
                                            int const* __restrict__ row_ptr, int H, int total_rows,
                                            int G_total) {
  static_assert(kRowsPerBlock > 0);
  extern __shared__ int smem_tprefix[];
  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp_id = tid >> 5;

  int blocks_per_h = (total_rows + kRowsPerBlock - 1) / kRowsPerBlock;
  int h = blockIdx.x / blocks_per_h;
  int b_in_h = blockIdx.x - h * blocks_per_h;
  if (h >= H) return;
  int base_r = b_in_h * kRowsPerBlock;
  if (base_r >= total_rows) return;
  int actual_M = min(kRowsPerBlock, total_rows - base_r);

  size_t stride_g = (size_t)H * total_rows;
  int* base_ptr = tile_counts + (size_t)h * total_rows + base_r;
  int total_elems = G_total * actual_M;

  for (int i = tid; i < total_elems; i += kThreads) {
    int r_off = i % actual_M;
    int g = i / actual_M;
    smem_tprefix[r_off * G_total + g] = base_ptr[g * stride_g + r_off];
  }
  __syncthreads();

  if (warp_id < actual_M) {
    int abs_r = base_r + warp_id;
    int rp = row_ptr[(size_t)h * (total_rows + 1) + abs_r];
    int* my_sm = smem_tprefix + warp_id * G_total;
    int running = rp;
    for (int g0 = 0; g0 < G_total; g0 += kWarpSize) {
      int g = g0 + lane;
      int v = (g < G_total) ? my_sm[g] : 0;
      int x = v;
#pragma unroll
      for (int off = 1; off < kWarpSize; off <<= 1) {
        int nbr = __shfl_up_sync(0xFFFFFFFF, x, off);
        if (lane >= off) x += nbr;
      }
      int excl = running + x - v;
      if (g < G_total) my_sm[g] = excl;
      running += __shfl_sync(0xFFFFFFFF, x, 31);
    }
  }
  __syncthreads();

  for (int i = tid; i < total_elems; i += kThreads) {
    int r_off = i % actual_M;
    int g = i / actual_M;
    base_ptr[g * stride_g + r_off] = smem_tprefix[r_off * G_total + g];
  }
}

// ---------------------------------------------------------------------------
// S: scatter.
// ---------------------------------------------------------------------------
template <int kTopK, int kBlockK, int kWarps>
__global__ void k2q_scatter_kernel(int const* __restrict__ q2k, int const* __restrict__ cu_q,
                                   int const* __restrict__ row_map,
                                   int const* __restrict__ abs_base, int* __restrict__ q_idx,
                                   int* __restrict__ qsplit_idx, int* __restrict__ split_counts,
                                   int H, int B, int S_Q, int total_rows, int max_kv_blocks,
                                   int q_per_cta, int q_per_warp, int max_seqlen_q) {
  constexpr int kQPerIter = kWarpSize / kTopK > 0 ? kWarpSize / kTopK : 1;
  extern __shared__ int smem_cursor_int[];
  int tid = threadIdx.x;
  int warp_id = tid >> 5;
  int lane = tid & 31;
  int c = blockIdx.x;
  int q_start_cta = c * q_per_cta;
  int q_end_cta = min(q_start_cta + q_per_cta, S_Q);
  int q_start_warp = min(q_start_cta + warp_id * q_per_warp, q_end_cta);
  int q_end_warp = min(q_start_warp + q_per_warp, q_end_cta);

  int q_in_iter = lane / kTopK;
  int slot_in_q = lane % kTopK;
  bool lane_active = (lane < kQPerIter * kTopK);

  int packed_per_warp = (total_rows + 1) >> 1;
  int* my_cursor = smem_cursor_int + warp_id * packed_per_warp;

  for (int h = 0; h < H; ++h) {
    for (int i = lane; i < packed_per_warp; i += kWarpSize) my_cursor[i] = 0;
    __syncwarp();

    if (q_start_warp < q_end_warp) {
      int bi = 0;
      advance_batch_only(cu_q, B, q_start_warp, bi);
      int const* head_q2k = q2k + (size_t)h * S_Q * kTopK;
      int const* my_abs_base = abs_base + ((size_t)(c * kWarps + warp_id) * H + h) * total_rows;
      int* head_qidx = q_idx + (size_t)h * S_Q * kTopK;

      constexpr int kUnroll = 16;
      int qi_base = q_start_warp;
      for (; qi_base + kUnroll * kQPerIter <= q_end_warp; qi_base += kUnroll * kQPerIter) {
        int kvb[kUnroll], qloc[kUnroll], batch_u[kUnroll];
        int const* rmap[kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
          int qi_u = qi_base + u * kQPerIter + q_in_iter;
          kvb[u] = -1;
          qloc[u] = 0;
          batch_u[u] = 0;
          if (lane_active) {
            advance_batch_only(cu_q, B, qi_u, bi);
            qloc[u] = qi_u - cu_q[bi];
            batch_u[u] = bi;
            kvb[u] = head_q2k[(size_t)qi_u * kTopK + slot_in_q];
          }
          rmap[u] = row_map + (size_t)bi * max_kv_blocks;
        }
        int row[kUnroll], abs_v[kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
          row[u] = -1;
          if (lane_active && kvb[u] >= 0 && kvb[u] < max_kv_blocks) row[u] = rmap[u][kvb[u]];
          abs_v[u] = (row[u] >= 0 && row[u] < total_rows) ? my_abs_base[row[u]] : 0;
        }
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
          int r = row[u];
          bool valid = r >= 0 && r < total_rows;
          unsigned int vmask = __ballot_sync(0xFFFFFFFFu, valid);
          unsigned int gmask =
              (kTopK == 32) ? 0xFFFFFFFFu : (((1u << kTopK) - 1u) << (q_in_iter * kTopK));
          unsigned int llmask = lane == 0 ? 0u : ((1u << lane) - 1u);
          int split_slot = __popc(vmask & gmask & llmask);
          int valid_count = __popc(vmask & gmask);
          if (split_counts != nullptr && slot_in_q == 0) {
            int q_abs = cu_q[batch_u[u]] + qloc[u];
            split_counts[(size_t)q_abs * H + h] = valid_count;
          }
          if (valid) {
            int slot = atomic_inc_int16_packed(my_cursor, r);
            int out_pos = abs_v[u] + slot;
            head_qidx[out_pos] = qloc[u];
            if (qsplit_idx != nullptr)
              qsplit_idx[(size_t)h * S_Q * kTopK + out_pos] = qloc[u] | ((split_slot & 0xFF) << 24);
          }
        }
      }
      // Tail
      for (; qi_base < q_end_warp; qi_base += kQPerIter) {
        int my_qi = qi_base + q_in_iter;
        bool valid_q = (my_qi < q_end_warp) && lane_active;
        int kvb_local = -1, q_local = 0, batch_local = 0;
        if (valid_q) {
          advance_batch_only(cu_q, B, my_qi, bi);
          batch_local = bi;
          q_local = my_qi - cu_q[bi];
          kvb_local = head_q2k[(size_t)my_qi * kTopK + slot_in_q];
        }
        int const* my_row_map = row_map + (size_t)bi * max_kv_blocks;
        int row = -1;
        if (valid_q && kvb_local >= 0 && kvb_local < max_kv_blocks) row = my_row_map[kvb_local];
        bool valid = row >= 0 && row < total_rows;
        unsigned int vmask = __ballot_sync(0xFFFFFFFFu, valid);
        unsigned int gmask =
            (kTopK == 32) ? 0xFFFFFFFFu : (((1u << kTopK) - 1u) << (q_in_iter * kTopK));
        unsigned int llmask = lane == 0 ? 0u : ((1u << lane) - 1u);
        int split_slot = __popc(vmask & gmask & llmask);
        int valid_count = __popc(vmask & gmask);
        if (split_counts != nullptr && valid_q && slot_in_q == 0)
          split_counts[(size_t)my_qi * H + h] = valid_count;
        if (valid) {
          int slot = atomic_inc_int16_packed(my_cursor, row);
          int out_pos = my_abs_base[row] + slot;
          head_qidx[out_pos] = q_local;
          if (qsplit_idx != nullptr)
            qsplit_idx[(size_t)h * S_Q * kTopK + out_pos] = q_local | ((split_slot & 0xFF) << 24);
        }
      }
    }
    if (h + 1 < H) __syncthreads();
  }
}

}  // anonymous namespace

// ===========================================================================
// Host orchestration
// ===========================================================================

template <int kTopK, int kBlockK>
static void launch_pipeline(const int* q2k_ptr, const int* cu_q_ptr, const int* cu_k_ptr,
                            int* row_ptr_out, int* q_idx_out, int H, int S_Q, int B, int total_rows,
                            int max_kv_blocks, int* scheduler_metadata_ptr, int* work_count_ptr,
                            int* qsplit_idx_ptr, int* split_counts_ptr, int target_q_per_cta,
                            int work_capacity, int max_seqlen_q, cudaStream_t stream) {
  bool emit_schedule = (scheduler_metadata_ptr != nullptr);
  cudaError_t err;

  err = cudaMemsetAsync(row_ptr_out, 0, (size_t)H * (total_rows + 1) * sizeof(int), stream);
  TVM_FFI_ICHECK(err == cudaSuccess) << cudaGetErrorString(err);
  err = cudaMemsetAsync(q_idx_out, 0xFF, (size_t)H * S_Q * kTopK * sizeof(int), stream);
  TVM_FFI_ICHECK(err == cudaSuccess) << cudaGetErrorString(err);

  CudaBuf row_counts_buf((size_t)H * total_rows * sizeof(int));
  CudaBuf row_map_buf((size_t)B * max_kv_blocks * sizeof(int));
  CudaBuf row_coords_buf(emit_schedule ? (size_t)total_rows * 2 * sizeof(int) : 0);

  int* row_counts = row_counts_buf.as<int>();
  int* row_map = row_map_buf.as<int>();
  int* row_coords = emit_schedule ? row_coords_buf.as<int>() : nullptr;

  err = cudaMemsetAsync(row_counts, 0, (size_t)H * total_rows * sizeof(int), stream);
  TVM_FFI_ICHECK(err == cudaSuccess) << cudaGetErrorString(err);

  if (emit_schedule) {
    err = cudaMemsetAsync(work_count_ptr, 0, sizeof(int), stream);
    TVM_FFI_ICHECK(err == cudaSuccess) << cudaGetErrorString(err);
    err =
        cudaMemsetAsync(scheduler_metadata_ptr, 0, (size_t)work_capacity * 6 * sizeof(int), stream);
    TVM_FFI_ICHECK(err == cudaSuccess) << cudaGetErrorString(err);
  }

  int dev = 0, num_sms = 0;
  cudaGetDevice(&dev);
  err = cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
  TVM_FFI_ICHECK(err == cudaSuccess) << cudaGetErrorString(err);

  // SMEM budget, queried from the device (carveout-aware) rather than hardcoded.
  // The per-SM capacity is the occupancy budget; the per-block opt-in maximum is
  // the hard ceiling on what a single CTA may request via cudaFuncSetAttribute.
  // These differ a lot across Blackwell: datacenter SM100 has 228KB/SM, but the
  // consumer/Pro SM120/SM121 parts (RTX 5080, RTX PRO 6000, GB10) have only
  // ~100KB/SM and a ~99KB per-block opt-in cap. The old hardcoded 228KB launched
  // fine on SM100 but let per_cta_smem reach 114KB, which exceeds the consumer
  // per-block limit once the per-warp histogram grows large (total_rows ~50k+),
  // aborting cudaFuncSetAttribute below.
  int smem_per_sm = 0, smem_per_block_optin = 0;
  err = cudaDeviceGetAttribute(&smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
  TVM_FFI_ICHECK(err == cudaSuccess) << cudaGetErrorString(err);
  err = cudaDeviceGetAttribute(&smem_per_block_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  TVM_FFI_ICHECK(err == cudaSuccess) << cudaGetErrorString(err);

  int per_warp_smem = ((total_rows + 1) >> 1) * (int)sizeof(int);
  int kWarps_pick = 4;
  // keep >=2 CTAs/SM resident: per-CTA smem <= half the per-SM budget.
  while (kWarps_pick > 1 && (kWarps_pick * per_warp_smem) * 2 > smem_per_sm) kWarps_pick >>= 1;
  if (kWarps_pick < 1) kWarps_pick = 1;

  int per_cta_smem = kWarps_pick * per_warp_smem;
  // a single CTA cannot opt into more than the device per-block maximum.
  TVM_FFI_ICHECK(per_cta_smem <= smem_per_block_optin)
      << "build_k2q_csr: required shared memory " << per_cta_smem
      << " B exceeds the device per-block opt-in limit " << smem_per_block_optin
      << " B (total_rows=" << total_rows << "); context too large for this GPU.";
  int max_ctas_per_sm = std::max(1, smem_per_sm / std::max(1, per_cta_smem));
  if (max_ctas_per_sm > 8) max_ctas_per_sm = 8;

  constexpr int kMinQPerCta = 256;
  int target_g = num_sms * std::min(max_ctas_per_sm, 3);
  int max_g_for_q = (S_Q + kMinQPerCta - 1) / kMinQPerCta;
  int G = std::min({target_g, max_g_for_q, S_Q});
  if (G < 1) G = 1;
  int q_per_cta = (S_Q + G - 1) / G;
  G = (S_Q + q_per_cta - 1) / q_per_cta;
  int q_per_warp = (q_per_cta + kWarps_pick - 1) / kWarps_pick;
  int G_total = G * kWarps_pick;

  CudaBuf tile_counts_buf((size_t)G_total * H * total_rows * sizeof(int));
  int* tile_counts = tile_counts_buf.as<int>();

  constexpr int kPtRowsPerBlock = 8;
  constexpr int kPtThreads = 256;

  if (max_kv_blocks > 0) {
    k2q_build_row_map_kernel<kBlockK>
        <<<max_kv_blocks, 32, 0, stream>>>(cu_k_ptr, row_map, row_coords, B, max_kv_blocks);
  }

  auto launch_hist_scatter = [&](auto kWarps_const) {
    constexpr int W = decltype(kWarps_const)::value;
    size_t smem_bytes = (size_t)W * per_warp_smem;
    auto hist_fn = k2q_hist_kernel<kTopK, kBlockK, W>;
    auto scat_fn = k2q_scatter_kernel<kTopK, kBlockK, W>;
    auto tpfx_fn = k2q_tile_prefix_smem_kernel<kPtThreads, kPtRowsPerBlock>;
    auto rpfx_fn = k2q_row_prefix_kernel<1024>;
    cudaError_t e;

    e = cudaFuncSetAttribute(hist_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    TVM_FFI_ICHECK(e == cudaSuccess) << cudaGetErrorString(e);
    e = cudaFuncSetAttribute(scat_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    TVM_FFI_ICHECK(e == cudaSuccess) << cudaGetErrorString(e);

    hist_fn<<<G, W * kWarpSize, smem_bytes, stream>>>(q2k_ptr, cu_q_ptr, row_map, row_counts,
                                                      tile_counts, H, B, S_Q, total_rows,
                                                      max_kv_blocks, q_per_cta, q_per_warp);

    rpfx_fn<<<H, 1024, 0, stream>>>(row_counts, row_ptr_out, row_coords, scheduler_metadata_ptr,
                                    work_count_ptr, total_rows, target_q_per_cta, work_capacity);

    int blocks_per_h = (total_rows + kPtRowsPerBlock - 1) / kPtRowsPerBlock;
    int pt_grid = std::max(1, H * blocks_per_h);
    size_t pt_smem = (size_t)kPtRowsPerBlock * G_total * sizeof(int);
    e = cudaFuncSetAttribute(tpfx_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)pt_smem);
    TVM_FFI_ICHECK(e == cudaSuccess) << cudaGetErrorString(e);
    tpfx_fn<<<pt_grid, kPtThreads, pt_smem, stream>>>(tile_counts, row_ptr_out, H, total_rows,
                                                      G_total);

    scat_fn<<<G, W * kWarpSize, smem_bytes, stream>>>(
        q2k_ptr, cu_q_ptr, row_map, tile_counts, q_idx_out, qsplit_idx_ptr, split_counts_ptr, H, B,
        S_Q, total_rows, max_kv_blocks, q_per_cta, q_per_warp, max_seqlen_q);
  };

  if (kWarps_pick == 4)
    launch_hist_scatter(std::integral_constant<int, 4>{});
  else if (kWarps_pick == 2)
    launch_hist_scatter(std::integral_constant<int, 2>{});
  else
    launch_hist_scatter(std::integral_constant<int, 1>{});
}

// ---------------------------------------------------------------------------
// TVM-FFI exported function
// ---------------------------------------------------------------------------

void build_k2q_csr(TensorView q2k, TensorView cu_q, TensorView cu_k, TensorView row_ptr,
                   TensorView q_idx, int64_t topk, int64_t blk_kv, int64_t total_rows,
                   int64_t max_kv_blocks) {
  CHECK_INPUT(q2k);
  CHECK_INPUT(cu_q);
  CHECK_INPUT(cu_k);
  CHECK_INPUT(row_ptr);
  CHECK_INPUT(q_idx);
  CHECK_DIM(3, q2k);
  CHECK_DIM(1, cu_q);
  CHECK_DIM(1, cu_k);
  CHECK_DIM(2, row_ptr);
  CHECK_DIM(2, q_idx);
  TVM_FFI_ICHECK(encode_dlpack_dtype(q2k.dtype()) == int32_code) << "q2k must be int32";
  TVM_FFI_ICHECK(encode_dlpack_dtype(cu_q.dtype()) == int32_code) << "cu_q must be int32";
  TVM_FFI_ICHECK(encode_dlpack_dtype(cu_k.dtype()) == int32_code) << "cu_k must be int32";
  TVM_FFI_ICHECK(encode_dlpack_dtype(row_ptr.dtype()) == int32_code) << "row_ptr must be int32";
  TVM_FFI_ICHECK(encode_dlpack_dtype(q_idx.dtype()) == int32_code) << "q_idx must be int32";
  TVM_FFI_ICHECK(blk_kv == 128) << "build_k2q_csr only supports blk_kv == 128";

  int H = (int)q2k.size(0);
  int S_Q = (int)q2k.size(1);
  int B = (int)cu_q.size(0) - 1;
  int tr = (int)total_rows;
  int mkv = (int)max_kv_blocks;
  TVM_FFI_ICHECK(tr >= 0 && mkv >= 0) << "total_rows / max_kv_blocks must be non-negative";
  TVM_FFI_ICHECK(row_ptr.size(0) == H && row_ptr.size(1) == (int64_t)tr + 1)
      << "row_ptr shape mismatch: expected [" << H << ", " << tr + 1 << "]";
  TVM_FFI_ICHECK(q_idx.size(0) == H && q_idx.size(1) == (int64_t)S_Q * (int)topk)
      << "q_idx shape mismatch";

  cudaStream_t stream = get_current_stream();

  if (S_Q == 0 || tr == 0 || H == 0 || mkv == 0) {
    cudaMemsetAsync(row_ptr.data_ptr(), 0, (size_t)H * (tr + 1) * sizeof(int), stream);
    cudaMemsetAsync(q_idx.data_ptr(), 0xFF, (size_t)H * S_Q * (int)topk * sizeof(int), stream);
    return;
  }

  auto q2k_p = static_cast<const int*>(q2k.data_ptr());
  auto cu_q_p = static_cast<const int*>(cu_q.data_ptr());
  auto cu_k_p = static_cast<const int*>(cu_k.data_ptr());
  auto row_ptr_p = static_cast<int*>(row_ptr.data_ptr());
  auto q_idx_p = static_cast<int*>(q_idx.data_ptr());

  if (topk == 16)
    launch_pipeline<16, 128>(q2k_p, cu_q_p, cu_k_p, row_ptr_p, q_idx_p, H, S_Q, B, tr, mkv, nullptr,
                             nullptr, nullptr, nullptr, 1, 0, 0, stream);
  else if (topk == 8)
    launch_pipeline<8, 128>(q2k_p, cu_q_p, cu_k_p, row_ptr_p, q_idx_p, H, S_Q, B, tr, mkv, nullptr,
                            nullptr, nullptr, nullptr, 1, 0, 0, stream);
  else if (topk == 32)
    launch_pipeline<32, 128>(q2k_p, cu_q_p, cu_k_p, row_ptr_p, q_idx_p, H, S_Q, B, tr, mkv, nullptr,
                             nullptr, nullptr, nullptr, 1, 0, 0, stream);
  else if (topk == 4)
    launch_pipeline<4, 128>(q2k_p, cu_q_p, cu_k_p, row_ptr_p, q_idx_p, H, S_Q, B, tr, mkv, nullptr,
                            nullptr, nullptr, nullptr, 1, 0, 0, stream);
  else
    TVM_FFI_ICHECK(false) << "unsupported topK " << topk << " (expected 4, 8, 16, or 32)";
}

// Variant that additionally emits the flat work schedule consumed by the
// KV-major sparse forward kernel:
//   qsplit_idx        [H, S_Q*topk] int32: q_local | (split_slot << 24)
//   split_counts      [S_Q, H]      int32: valid splits per (q, kv-head)
//   scheduler_metadata [capacity, 6] int32: {h, row, q_begin, q_count,
//                                            batch_idx, kv_block_idx}
//   work_count        [1]           int32: number of valid work items
void build_k2q_csr_schedule(TensorView q2k, TensorView cu_q, TensorView cu_k, TensorView row_ptr,
                            TensorView q_idx, TensorView qsplit_idx, TensorView split_counts,
                            TensorView scheduler_metadata, TensorView work_count, int64_t topk,
                            int64_t blk_kv, int64_t total_rows, int64_t max_kv_blocks,
                            int64_t target_q_per_cta, int64_t max_seqlen_q) {
  CHECK_INPUT(q2k);
  CHECK_INPUT(cu_q);
  CHECK_INPUT(cu_k);
  CHECK_INPUT(row_ptr);
  CHECK_INPUT(q_idx);
  CHECK_INPUT(qsplit_idx);
  CHECK_INPUT(split_counts);
  CHECK_INPUT(scheduler_metadata);
  CHECK_INPUT(work_count);
  CHECK_DIM(3, q2k);
  CHECK_DIM(2, row_ptr);
  CHECK_DIM(2, q_idx);
  CHECK_DIM(2, qsplit_idx);
  CHECK_DIM(2, split_counts);
  CHECK_DIM(2, scheduler_metadata);
  CHECK_DIM(1, work_count);
  TVM_FFI_ICHECK(encode_dlpack_dtype(qsplit_idx.dtype()) == int32_code);
  TVM_FFI_ICHECK(encode_dlpack_dtype(split_counts.dtype()) == int32_code);
  TVM_FFI_ICHECK(encode_dlpack_dtype(scheduler_metadata.dtype()) == int32_code);
  TVM_FFI_ICHECK(encode_dlpack_dtype(work_count.dtype()) == int32_code);
  TVM_FFI_ICHECK(blk_kv == 128) << "build_k2q_csr only supports blk_kv == 128";
  TVM_FFI_ICHECK(topk <= 256) << "split slot is packed into 8 bits";
  TVM_FFI_ICHECK(target_q_per_cta >= 1);

  int H = (int)q2k.size(0);
  int S_Q = (int)q2k.size(1);
  int B = (int)cu_q.size(0) - 1;
  int tr = (int)total_rows;
  int mkv = (int)max_kv_blocks;
  int work_capacity = (int)scheduler_metadata.size(0);
  TVM_FFI_ICHECK(scheduler_metadata.size(1) == 6) << "scheduler_metadata must be [capacity, 6]";
  TVM_FFI_ICHECK(qsplit_idx.size(0) == H && qsplit_idx.size(1) == (int64_t)S_Q * (int)topk);
  TVM_FFI_ICHECK(split_counts.size(0) == S_Q && split_counts.size(1) == H);
  TVM_FFI_ICHECK(row_ptr.size(0) == H && row_ptr.size(1) == (int64_t)tr + 1);
  TVM_FFI_ICHECK(q_idx.size(0) == H && q_idx.size(1) == (int64_t)S_Q * (int)topk);

  cudaStream_t stream = get_current_stream();

  if (S_Q == 0 || tr == 0 || H == 0 || mkv == 0) {
    cudaMemsetAsync(row_ptr.data_ptr(), 0, (size_t)H * (tr + 1) * sizeof(int), stream);
    cudaMemsetAsync(q_idx.data_ptr(), 0xFF, (size_t)H * S_Q * (int)topk * sizeof(int), stream);
    cudaMemsetAsync(work_count.data_ptr(), 0, sizeof(int), stream);
    cudaMemsetAsync(split_counts.data_ptr(), 0, (size_t)S_Q * H * sizeof(int), stream);
    return;
  }

  auto q2k_p = static_cast<const int*>(q2k.data_ptr());
  auto cu_q_p = static_cast<const int*>(cu_q.data_ptr());
  auto cu_k_p = static_cast<const int*>(cu_k.data_ptr());
  auto row_ptr_p = static_cast<int*>(row_ptr.data_ptr());
  auto q_idx_p = static_cast<int*>(q_idx.data_ptr());
  auto qsplit_p = static_cast<int*>(qsplit_idx.data_ptr());
  auto scnt_p = static_cast<int*>(split_counts.data_ptr());
  auto sched_p = static_cast<int*>(scheduler_metadata.data_ptr());
  auto wcnt_p = static_cast<int*>(work_count.data_ptr());

  if (topk == 16)
    launch_pipeline<16, 128>(q2k_p, cu_q_p, cu_k_p, row_ptr_p, q_idx_p, H, S_Q, B, tr, mkv, sched_p,
                             wcnt_p, qsplit_p, scnt_p, (int)target_q_per_cta, work_capacity,
                             (int)max_seqlen_q, stream);
  else if (topk == 8)
    launch_pipeline<8, 128>(q2k_p, cu_q_p, cu_k_p, row_ptr_p, q_idx_p, H, S_Q, B, tr, mkv, sched_p,
                            wcnt_p, qsplit_p, scnt_p, (int)target_q_per_cta, work_capacity,
                            (int)max_seqlen_q, stream);
  else if (topk == 32)
    launch_pipeline<32, 128>(q2k_p, cu_q_p, cu_k_p, row_ptr_p, q_idx_p, H, S_Q, B, tr, mkv, sched_p,
                             wcnt_p, qsplit_p, scnt_p, (int)target_q_per_cta, work_capacity,
                             (int)max_seqlen_q, stream);
  else if (topk == 4)
    launch_pipeline<4, 128>(q2k_p, cu_q_p, cu_k_p, row_ptr_p, q_idx_p, H, S_Q, B, tr, mkv, sched_p,
                            wcnt_p, qsplit_p, scnt_p, (int)target_q_per_cta, work_capacity,
                            (int)max_seqlen_q, stream);
  else
    TVM_FFI_ICHECK(false) << "unsupported topK " << topk << " (expected 4, 8, 16, or 32)";
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(build_k2q_csr, build_k2q_csr);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(build_k2q_csr_schedule, build_k2q_csr_schedule);
