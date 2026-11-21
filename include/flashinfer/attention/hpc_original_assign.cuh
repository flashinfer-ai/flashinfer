// Copyright 2025 hpc-ops authors
// Copyright (C) 2026 Tencent. All rights reserved.
// SPDX-License-Identifier: MIT
// Vendored from Tencent/hpc-ops 1cd332980ed46bd0172091c1c35d55338fcae47a:
// src/attention/decode/assign_task.cu and sched_task_info.h.
// See HPC_OPS_LICENSE.txt next to this file.
// Changes: isolated namespace, local vector memory helpers, zero-initialized padding.
// The parallel scheduling kernel and cross-CTA protocol are otherwise retained.
#pragma once
#include <cuda_runtime.h>

#include <algorithm>
namespace flashinfer::hpc_original {
namespace dynamic {
struct alignas(16) TaskScheduleInfo {
  int ihead_kv, ibatch, ichunk, iseq_start;
  int num_seqkv, num_seqkvcache, num_tile_kv, num_tile_full;
  int is_casual_chunk;
  int pad[3];
};
constexpr int kTaskScheduleInfoStride = sizeof(TaskScheduleInfo) / sizeof(int);
}  // namespace dynamic
template <typename T, int N>
struct alignas(16) vec_t {
  T data[N];
  __device__ T& operator[](int i) { return data[i]; }
};
__device__ __forceinline__ void store(int* p, vec_t<int, 4> v) {
  *reinterpret_cast<int4*>(p) = make_int4(v[0], v[1], v[2], v[3]);
}
template <int Stride>
__device__ __forceinline__ void store_task(int* p, const dynamic::TaskScheduleInfo& t) {
  const auto* src = reinterpret_cast<const int4*>(&t);
#pragma unroll
  for (int i = 0; i < Stride / 4; ++i) reinterpret_cast<int4*>(p)[i] = src[i];
}
template <int Stride>
__device__ __forceinline__ dynamic::TaskScheduleInfo load_task(int* p) {
  dynamic::TaskScheduleInfo t;
#pragma unroll
  for (int i = 0; i < Stride / 4; ++i)
    reinterpret_cast<int4*>(&t)[i] = reinterpret_cast<const int4*>(p)[i];
  return t;
}
__device__ __forceinline__ int load_global_volatile(int* p) {
  int v;
  asm volatile("ld.volatile.global.s32 {%0}, [%1];\n" : "=r"(v) : "l"(p));
  return v;
}
// Scheduling loop order: outer over ihead_kv, inner over ibatch.
template <int kMaxNumBatch, int kTileN>
__global__ void assign_attention_decode_task_kernel(int* task_map_ptr, const int* num_seq_kvcache,
                                                    int num_batch, int num_head_kv, int num_seq_q,
                                                    bool new_kv_included, int min_process_len,
                                                    int num_total_ctas, int max_splitk) {
  __shared__ int num_seqkvs[kMaxNumBatch];
  __shared__ int num_tiles[kMaxNumBatch];
  __shared__ int smem_total_tiles[1];

  int icta = blockIdx.x;
  int idx = threadIdx.x;
  int* task_map_base = task_map_ptr;

  int total_tiles_per_head = 0;

  int max_num_batch = task_map_ptr[3];
  if (idx == 0) {
    smem_total_tiles[0] = 0;
  }
  __syncthreads();

  for (int ibatch = idx; ibatch < num_batch; ibatch += blockDim.x) {
    // Skip batches whose total KV length is zero — nothing to attend to.
    int num_seqkv =
        (new_kv_included ? num_seq_kvcache[ibatch] : num_seq_kvcache[ibatch] + num_seq_q);
    num_seqkvs[ibatch] = num_seqkv;
    num_tiles[ibatch] = (num_seqkv + kTileN - 1) / kTileN;
    total_tiles_per_head += num_tiles[ibatch];
  }

  atomicAdd(&smem_total_tiles[0], total_tiles_per_head);
  __syncthreads();

  int total_tiles_all_heads = smem_total_tiles[0] * num_head_kv;

  int num_tile_per_cta = std::max((total_tiles_all_heads + num_total_ctas - 1) / num_total_ctas,
                                  min_process_len / kTileN);

  if (icta == 0 && idx == 1) {
    task_map_ptr[0] = num_tile_per_cta + 1;
    task_map_ptr[1] = num_total_ctas;
  }

  constexpr int kTaskStride = dynamic::kTaskScheduleInfoStride;
  int num_chunks_entries = max_num_batch * num_head_kv;
  int max_num_batch_pad_local = (num_chunks_entries + kTaskStride - 1) / kTaskStride * kTaskStride;
  int num_cta_count_pad_local = (num_total_ctas + kTaskStride - 1) / kTaskStride * kTaskStride;
  int* chunk_clear_ptr = task_map_ptr + kTaskStride * ((num_tile_per_cta + 1) * num_total_ctas + 1);
  int* finish_flag_ptr_global = chunk_clear_ptr + max_num_batch_pad_local;
  int* clear_sentinel_ptr = finish_flag_ptr_global + (num_cta_count_pad_local - 1);

  if (icta == 0) {
    for (int i = idx; i < num_chunks_entries; i += blockDim.x) {
      chunk_clear_ptr[i] = 0;
    }
    if (idx == 0) {
      task_map_ptr[5] = 0;
    }
    __syncthreads();
    if (idx == 0) {
      __threadfence();
      *clear_sentinel_ptr = 1;
    }
  } else {
    if (idx == 0) {
      while (load_global_volatile(clear_sentinel_ptr) == 0) {
      }
      __threadfence();
    }
  }
  __syncthreads();

  if (idx != 0) {
    return;
  }

  // Fast-forward state to this CTA's starting (ihead_kv, ibatch, chunk)
  int ihead_kv = 0;
  int ibatch = 0;
  int num_chunks = 0;
  int start_tiles = 0;
  int num_tile = num_tiles[0];

  for (int i = 0; i < icta; i++) {
    int bucket = num_tile_per_cta;
    while (bucket > 0 && ihead_kv < num_head_kv) {
      // Skip (ihead_kv, ibatch) with no tiles (e.g. empty KV cache).
      if (num_tile <= 0) {
        ibatch++;
        if (ibatch >= num_batch) {
          ibatch = 0;
          ihead_kv++;
          if (ihead_kv >= num_head_kv) {
            break;
          }
        }
        num_tile = num_tiles[ibatch];
        continue;
      }

      int add_tiles = std::min(num_tile, bucket);
      if (num_chunks == max_splitk - 1) {
        add_tiles = num_tile;
      }

      num_chunks++;
      start_tiles += add_tiles;
      num_tile -= add_tiles;
      bucket -= add_tiles;
      // forward to next batch/head
      if (num_tile <= 0) {
        ibatch++;
        if (ibatch >= num_batch) {
          ibatch = 0;
          ihead_kv++;
          if (ihead_kv >= num_head_kv) {
            break;
          }
        }
        num_tile = num_tiles[ibatch];
        num_chunks = 0;
        start_tiles = 0;
      }
    }
  }

  int max_num_batch_with_head = max_num_batch * num_head_kv;
  int max_num_batch_with_head_pad =
      (max_num_batch_with_head + kTaskStride - 1) / kTaskStride * kTaskStride;
  int num_cta_count_pad = (num_total_ctas + kTaskStride - 1) / kTaskStride * kTaskStride;

  auto* task_map_chunk_ptr =
      task_map_ptr + kTaskStride * ((num_tile_per_cta + 1) * num_total_ctas + 1);
  auto* task_map_sm_finish_ptr = task_map_chunk_ptr + max_num_batch_with_head_pad;
  auto* num_task_map_ptr = task_map_sm_finish_ptr + num_cta_count_pad;
  task_map_ptr += ((num_tile_per_cta + 1) * icta + 1) * kTaskStride;

  int itask = 0;
  int bucket = num_tile_per_cta;

  // When the last chunk of a (head, batch) has num_seqkvcache < 0 after
  // the causal adjustment, Q tokens straddle the prev/cur boundary. We
  // keep cur (clamped to 0) and propagate the overflow to prev so it gets
  // is_casual_chunk=1 with a correctly reduced num_seqkvcache.
  // If prev is in the same CTA we fix it in-place; otherwise we defer
  // via wait_batch and fix it after the CTA finishes (cross-CTA sync).
  int wait_batch = -1;
  int wait_batch_num_seqkvcache_overflow = 0;

  while (bucket > 0 && ihead_kv < num_head_kv) {
    // Skip (ihead_kv, ibatch) with no tiles (e.g. empty KV cache).
    if (num_tile <= 0) {
      ibatch++;
      if (ibatch >= num_batch) {
        ibatch = 0;
        ihead_kv++;
      }
      num_tile = (ihead_kv < num_head_kv) ? num_tiles[ibatch] : 0;
      if (num_tile == 0 && ihead_kv < num_head_kv) {
        task_map_chunk_ptr[ihead_kv * num_batch + ibatch] = 0;
      }
      continue;
    }

    int add_tiles = std::min(num_tile, bucket);
    int num_seqkv = num_seqkvs[ibatch];

    if (num_chunks == max_splitk - 1) {
      add_tiles = num_tile;
    }

    dynamic::TaskScheduleInfo task_info{};
    task_info.ihead_kv = ihead_kv;
    task_info.ibatch = ibatch;
    task_info.ichunk = num_chunks;
    task_info.iseq_start = start_tiles * kTileN;
    task_info.num_seqkv = std::min(add_tiles * kTileN, num_seqkv - task_info.iseq_start);
    task_info.num_seqkvcache = task_info.num_seqkv;
    task_info.num_tile_kv = (task_info.num_seqkv + kTileN - 1) / kTileN;
    task_info.num_tile_full = task_info.num_seqkvcache / kTileN;
    task_info.is_casual_chunk = 0;

    num_chunks++;
    start_tiles += add_tiles;
    num_tile -= add_tiles;
    bucket -= add_tiles;

    if (num_tile <= 0) {
      task_info.is_casual_chunk = 1;
      // int raw_seqkvcache = task_info.num_seqkvcache - num_seq_q;
      // task_info.num_seqkvcache = std::max(raw_seqkvcache, 0);
      // task_info.num_tile_full = task_info.num_seqkvcache / kTileN;

      task_info.num_seqkvcache -= num_seq_q;
      task_info.num_tile_full = std::max(task_info.num_seqkvcache / kTileN, 0);
      task_map_chunk_ptr[ihead_kv * num_batch + ibatch] = num_chunks;
      atomicMax(&task_map_base[5], num_chunks);

      // When num_seqkvcache < 0, some Q tokens overflow into the previous
      // chunk. Mark prev as causal and reduce its num_seqkvcache accordingly.
      if (task_info.num_seqkvcache < 0) {
        if (itask != 0) {
          dynamic::TaskScheduleInfo last_task_info =
              load_task<kTaskStride>(task_map_ptr + (itask - 1) * kTaskStride);
          last_task_info.is_casual_chunk = 1;
          last_task_info.num_seqkvcache += task_info.num_seqkvcache;
          last_task_info.num_tile_full = std::max(last_task_info.num_seqkvcache / kTileN, 0);
          store_task<kTaskStride>(task_map_ptr + (itask - 1) * kTaskStride, last_task_info);
        } else {
          wait_batch = ibatch;
          wait_batch_num_seqkvcache_overflow = task_info.num_seqkvcache;
        }
      }

      // Advance (ihead_kv, ibatch).
      ibatch++;
      if (ibatch >= num_batch) {
        ibatch = 0;
        ihead_kv++;
      }
      num_tile = (ihead_kv < num_head_kv) ? num_tiles[ibatch] : 0;
      if (num_tile == 0 && ihead_kv < num_head_kv) {
        task_map_chunk_ptr[ihead_kv * num_batch + ibatch] = 0;
      }
      num_chunks = 0;
      start_tiles = 0;
    }

    store_task<kTaskStride>(task_map_ptr + itask * kTaskStride, task_info);
    itask++;
  }

  task_map_ptr[itask * kTaskStride] = -1;
  task_map_ptr[itask * kTaskStride + 1] = -1;
  num_task_map_ptr[icta] = itask;

  // Fill every remaining slot in this bin with a terminator too, so
  // stragglers / early-terminating walkers stay safe on zero-init memory.
  for (int slot = itask + 1; slot <= num_tile_per_cta; slot++) {
    task_map_ptr[slot * kTaskStride] = -1;
    task_map_ptr[slot * kTaskStride + 1] = -1;
  }

  if (wait_batch >= 0) {
    if (icta > 0) {
      task_map_ptr -= (num_tile_per_cta + 1) * kTaskStride;
      __threadfence();
      task_map_sm_finish_ptr[icta] = 1;
      while (load_global_volatile(task_map_sm_finish_ptr + (icta - 1)) < 1) {
      }
      __threadfence();
      int last_task_id = num_task_map_ptr[icta - 1] - 1;
      if (last_task_id >= 0) {
        dynamic::TaskScheduleInfo last_task_info =
            load_task<kTaskStride>(task_map_ptr + last_task_id * kTaskStride);
        last_task_info.is_casual_chunk = 1;
        last_task_info.num_seqkvcache += wait_batch_num_seqkvcache_overflow;
        last_task_info.num_tile_full = std::max(last_task_info.num_seqkvcache / kTileN, 0);
        store_task<kTaskStride>(task_map_ptr + last_task_id * kTaskStride, last_task_info);
      }
      __threadfence();
      task_map_sm_finish_ptr[icta] = 2;
    } else {
      __threadfence();
      task_map_sm_finish_ptr[icta] = 2;
    }
  } else {
    __threadfence();
    task_map_sm_finish_ptr[icta] = 2;
  }

  if (icta == 0) {
    vec_t<int, 4> zeros;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      zeros[i] = 0;
    }

    for (int i = 0; i < num_total_ctas; i++) {
      while (load_global_volatile(task_map_sm_finish_ptr + i) != 2) {
      }
    }

    for (int i = 0; i < (num_total_ctas + 3) / 4; i++) {
      store(task_map_sm_finish_ptr + 4 * i, zeros);
    }
    *clear_sentinel_ptr = 0;
  }
}
}  // namespace flashinfer::hpc_original
