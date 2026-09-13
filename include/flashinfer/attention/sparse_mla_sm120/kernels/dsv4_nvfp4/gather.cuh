// Copyright (c) 2026 by FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../../arch/cp_async.cuh"
#include "../../common/zero_row.cuh"
#include "../../model/dsv4_nvfp4_layout.cuh"
#include "../../pipeline/staged_pipeline.cuh"

#include "resources.cuh"

namespace flashinfer::sparse_mla_sm120::nvfp4 {

template <int PageSize, bool Dual>
__device__ void page_location(int index, bool extra, int page_size, int& page, int& local) {
  if constexpr (Dual) {
    if (extra && page_size == 2) { page = index >> 1; local = index & 1; }
    else { page = index >> 6; local = index & 63; }
  } else { page = index / PageSize; local = index - page * PageSize; }
}

template <int PageSize, bool Dual, int Stride, int GatherThreads>
__device__ void gather_tile(const uint8_t* cache, const int32_t* indices, int chunk,
                            int length, bool extra, int page_size, size_t page_stride,
                            int entry, uint8_t* packed, bf16* rope, uint8_t* scales,
                            uint64_t* ready, int main_chunks, int extra_length,
                            const uint8_t* extra_cache, const int32_t* extra_indices,
                            int extra_page_size, size_t extra_stride) {
  if constexpr (Dual) {
    if (chunk >= main_chunks) {
      extra = true;
      chunk -= main_chunks;
      length = extra_length;
      cache = extra_cache;
      indices = extra_indices;
      page_size = extra_page_size;
      page_stride = extra_stride;
    }
  }
  using Layout = Dsv4Nvfp4Layout;
  constexpr int Candidates = 64;
  static_assert(Layout::DATA_BYTES_PER_TOKEN <= SPARSE_MLA_ZERO_ROW_BYTES);
  const int position = chunk * Candidates + entry;
  const int index = position < min((chunk + 1) * Candidates, length) ? indices[position] : -1;
  uint4 s0 = make_uint4(0, 0, 0, 0), s1 = s0;
  if (index >= 0) {
    int page, local;
    page_location<PageSize, Dual>(index, extra, page_size, page, local);
    const uint8_t* source = cache + size_t(page) * page_stride +
                            Layout::scale_offset(size_t(page_size), size_t(local));
    s0 = *reinterpret_cast<const uint4*>(source);
    s1 = *reinterpret_cast<const uint4*>(source + sizeof(uint4));
  }
  *reinterpret_cast<uint4*>(scales + size_t(entry) * Layout::SCALE_BYTES_PER_TOKEN) = s0;
  *reinterpret_cast<uint4*>(scales + size_t(entry) * Layout::SCALE_BYTES_PER_TOKEN + sizeof(uint4)) = s1;
  __threadfence_block();
  const int safe_index = index >= 0 ? index : 0;
  int page, local;
  page_location<PageSize, Dual>(safe_index, extra, page_size, page, local);
  const uint8_t* source = index >= 0 ? cache + size_t(page) * page_stride +
                                          Layout::data_offset(size_t(local)) : sparse_mla_zero_row;
  if (entry == 0) pipeline::BulkReady::expect(ready, Candidates * Layout::DATA_BYTES_PER_TOKEN);
  pipeline::RoleSync<Dsv4Nvfp4Sync::GATHER, GatherThreads>::wait();
  cp_async_bulk_g2s(packed + size_t(entry) * Stride, source, Layout::PACKED_NOPE_BYTES, ready);
  cp_async_bulk_g2s(rope + size_t(entry) * Layout::D_ROPE, source + Layout::PACKED_NOPE_BYTES,
                    Layout::ROPE_BYTES, ready);
}

}  // namespace flashinfer::sparse_mla_sm120::nvfp4
