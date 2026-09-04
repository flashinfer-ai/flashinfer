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

#pragma once

#include "../arch/barrier.cuh"
#include "../arch/cp_async.cuh"
#include "../model/kv_cache_traits.cuh"

// KV cache IO: gather BI entries from global KV pool to smem.
//
// FlashMLA ABI: stride_kv_row = bytes_per_token (DSV3_2: 656, DSV4: 584).
// The IO stride used for address calculation is the DATA stride:
//   DSV3_2:    656 (nope+scale+rope all contiguous, 656 % 16 = 0 ✓)
//   DSV4: 576 (nope+rope only, footer scales excluded)
//           576 % 16 = 0 ✓ for cp.async.bulk
//
// DSV3_2 uses flat addressing: kv_ptr + global_idx * 656.
// DSV4 uses block-structured addressing (footer layout):
//   data:  kv_ptr + block_idx * stride_kv_block + local_idx * 576
//   scale: kv_ptr + block_idx * stride_kv_block + page_block_size * 576 + local_idx * 8
//
// Reference: FlashMLA SM90 splitkv_mla.cuh / SM100 kernel.cuh.

template <ModelType MT>
struct KVIOTraits {
  using KV = KVCacheTraits<MT>;
  // DSV3_2: IO_STRIDE = KV_GMEM_STRIDE = 656 (inline, bulk copy includes scale)
  // DSV4: IO_STRIDE = D_NOPE + D_ROPE*2 = 576 (footer, data portion only)
  static constexpr int IO_STRIDE =
      KV::SCALE_IN_KV_SMEM ? KV::KV_GMEM_STRIDE : (KV::D_NOPE + KV::D_ROPE * sizeof(bf16));
  static_assert(IO_STRIDE % 16 == 0, "IO stride must be 16B aligned for cp.async.bulk");
};

// Bulk gather token nope data (and inline scales for DSV3_2) from global to smem.
// DSV3_2: flat addressing (idx * 656). DSV4: block-structured (footer layout).
//
// `idx` is this IO thread's candidate index, staged in a register by the caller
// one tile ahead of use so the LDG latency does not sit on the TMA issue chain.
// TILE_BI <= TILE_IO_THREADS gives each IO thread at most one candidate, so the
// thread's slot in the smem tile is io_tid.
template <ModelType MT, int PAGE_BLOCK_SIZE, bool USE_L2_HINT = false, int TILE_BI = BI,
          int TILE_IO_THREADS = IO_THREADS>
__device__ __forceinline__ void io_bulk_gather_tile(uint8_t* dst, int idx,
                                                    const uint8_t* __restrict__ kv_ptr,
                                                    uint64_t* mbar, int io_tid,
                                                    size_t stride_kv_block,
                                                    uint64_t cache_policy = 0) {
  using KV = KVCacheTraits<MT>;
  using IO = KVIOTraits<MT>;
  constexpr int COPY_BYTES = KV::KV_SMEM_COPY_BYTES;
  constexpr int SMEM_STRIDE = KV::KV_SMEM_STRIDE;
  static_assert(TILE_BI <= TILE_IO_THREADS,
                "per-thread index staging assumes at most one candidate per IO thread");

  if (io_tid == 0) mbarrier_arrive_expect_tx(mbar, TILE_BI * COPY_BYTES);
  if (io_tid >= TILE_BI) return;

  idx = (idx >= 0) ? idx : 0;

  const uint8_t* src;
  if constexpr (KV::SCALE_IN_KV_SMEM) {
    src = kv_ptr + (size_t)idx * IO::IO_STRIDE;
  } else {
    constexpr int pbs = PAGE_BLOCK_SIZE;
    int block_idx = idx / pbs;
    int local_idx = idx % pbs;
    src = kv_ptr + (size_t)block_idx * stride_kv_block + (size_t)local_idx * IO::IO_STRIDE;
  }
  if constexpr (USE_L2_HINT)
    cp_async_bulk_g2s_l2hint(dst + io_tid * SMEM_STRIDE, src, COPY_BYTES, mbar, cache_policy);
  else
    cp_async_bulk_g2s(dst + io_tid * SMEM_STRIDE, src, COPY_BYTES, mbar);
}

// Warm L2 for a candidate row ahead of its gather; issued by IO threads idle
// on the release handshake. Pure hint: padding indices are skipped, not
// clamped. Addressing mirrors io_bulk_gather_tile; footer models also warm the
// scale line, whose synchronous LDG sits on the gather issue path.
template <ModelType MT, int PAGE_BLOCK_SIZE, bool USE_L2_HINT = false, int TILE_BI = BI,
          int TILE_IO_THREADS = IO_THREADS>
__device__ __forceinline__ void io_bulk_prefetch_l2(int idx, const uint8_t* __restrict__ kv_ptr,
                                                    int io_tid, size_t stride_kv_block,
                                                    uint64_t cache_policy = 0) {
  using KV = KVCacheTraits<MT>;
  using IO = KVIOTraits<MT>;
  static_assert(TILE_BI <= TILE_IO_THREADS);
  if (io_tid >= TILE_BI || idx < 0) return;

  const uint8_t* src;
  if constexpr (KV::SCALE_IN_KV_SMEM) {
    src = kv_ptr + (size_t)idx * IO::IO_STRIDE;
  } else {
    constexpr int pbs = PAGE_BLOCK_SIZE;
    src = kv_ptr + (size_t)(idx / pbs) * stride_kv_block + (size_t)(idx % pbs) * IO::IO_STRIDE;
    const uint8_t* footer = kv_ptr + (size_t)(idx / pbs) * stride_kv_block +
                            (size_t)pbs * IO::IO_STRIDE +
                            (size_t)(idx % pbs) * KV::SCALE_BYTES_PER_TOKEN;
    prefetch_l2_line(footer);
  }
  if constexpr (USE_L2_HINT)
    cp_async_bulk_prefetch_l2_hint(src, IO::IO_STRIDE, cache_policy);
  else
    cp_async_bulk_prefetch_l2(src, IO::IO_STRIDE);
}

// `idx` is the same per-thread staged value passed to io_bulk_gather_tile, so
// the footer-scale model reads each index from gmem once per tile, not twice.
template <ModelType MT, int PAGE_BLOCK_SIZE, int TILE_BI = BI, int TILE_IO_THREADS = IO_THREADS>
__device__ __forceinline__ void io_gather_scales(uint8_t* scale_dst, int idx,
                                                 const uint8_t* __restrict__ kv_ptr, int io_tid,
                                                 size_t stride_kv_block) {
  using KV = KVCacheTraits<MT>;
  using IO = KVIOTraits<MT>;
  if constexpr (KV::SCALE_IN_KV_SMEM) return;

  constexpr int pbs = PAGE_BLOCK_SIZE;
  constexpr int SCALE_BYTES = KV::SCALE_BYTES_PER_TOKEN;
  // Only reachable for footer-scale models (the inline ones return above), so
  // the width check is disjoined rather than applied to every instantiation.
  static_assert(KV::SCALE_IN_KV_SMEM || SCALE_BYTES == sizeof(uint64_t),
                "the footer gather moves one uint64 per token; a different footer width needs a "
                "different load");
  static_assert(TILE_BI <= TILE_IO_THREADS,
                "per-thread index staging assumes at most one candidate per IO thread");
  if (io_tid >= TILE_BI) return;

  idx = (idx >= 0) ? idx : 0;

  int block_idx = idx / pbs;
  int local_idx = idx % pbs;
  const uint8_t* src = kv_ptr + (size_t)block_idx * stride_kv_block + (size_t)pbs * IO::IO_STRIDE +
                       (size_t)local_idx * SCALE_BYTES;
  *reinterpret_cast<uint64_t*>(scale_dst + io_tid * SCALE_BYTES) =
      __ldg(reinterpret_cast<const uint64_t*>(src));
}
