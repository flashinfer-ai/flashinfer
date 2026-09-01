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
#ifndef FLASHINFER_ATTENTION_SPARSE_SCORES_CUH_
#define FLASHINFER_ATTENTION_SPARSE_SCORES_CUH_

#include <cuda_runtime.h>

#include <cstdint>
#include <sstream>

#include "../cp_async.cuh"
#include "../fastdiv.cuh"
#include "../mma.cuh"
#include "../utils.cuh"

namespace flashinfer {

namespace sparse_scores {

// One warp scores 16 columns against up to 16 query heads with one m16n16k16
// tile, walking the feature axis 16 at a time.
constexpr uint32_t kTileM = 16;
constexpr uint32_t kTileN = 16;
constexpr uint32_t kTileK = 16;
// Columns a block scores at once. The two shapes below cover the same width
// with different thread counts.
constexpr uint32_t kBlockN = 64;
// Staging is what bounds this kernel, and how much of it is in flight is set
// by how many copies a thread issues before the wait -- fewer threads over the
// same tile means more per thread. A launch that does not fill the device
// takes the deeper shape; one that does trades depth for resident warps, since
// there the latency is already covered by other blocks.
constexpr uint32_t kDeepWarps = 2;
constexpr uint32_t kWideWarps = 4;
// Query heads a block handles. More than this needs a second n-tile, which the
// caller currently never asks for.
constexpr uint32_t kMaxHeads = kTileN;
// Elements of padding on each staged row. ldmatrix reads 16 rows at once, and a
// head dimension that is a multiple of the 32 four-byte banks puts all of them
// in the same banks. One 16-byte unit of padding rotates the banks each row
// lands in while keeping every row aligned for the 16-byte loads that fill it.
// The bank geometry this rests on has been the same since sm_70.
constexpr uint32_t kSmemPad = 16 / sizeof(uint32_t) * 2;
// Feature slice staged at a time when a block walks several column tiles. Two
// buffers of a whole head dimension would be most of a block's shared memory,
// and shared memory is what bounds how many blocks an SM holds. A block that
// walks a single tile stages the whole head instead: it is launched precisely
// because the device is not full, so the barrier per slice costs more than the
// occupancy buys.
constexpr uint32_t kMultiTileSliceK = 64;

}  // namespace sparse_scores

/*!
 * \brief Score every visible KV entry of a paged cache against a multi-head query.
 *
 * The score a sparse-attention selector ranks by:
 *
 *   score(row, col) = sum_h max(0, dot(K[col], Q[row, h])) / divisor
 *
 * There is no softmax and no value aggregation -- this produces the logits a
 * top-k runs on, not an attention output.
 *
 * Entries on a page the block table does not map come out as -inf so a top-k
 * never selects them. Columns past what the query can see are left untouched
 * instead; the count of visible entries per row is written out, and the
 * selector bounds its own k by that rather than by the column width.
 *
 * The ReLU applies to a head's completed dot product, so the feature axis has to
 * be fully accumulated before the heads are summed.
 *
 * \tparam HEAD_DIM per-head feature width, a multiple of 16
 * \tparam TILES_PER_BLOCK column tiles one block walks, to amortize staging the
 *   query across more columns when there are many rows to score
 */
template <uint32_t HEAD_DIM, uint32_t TILES_PER_BLOCK, uint32_t SLICE_K, uint32_t WARPS,
          typename DType, typename IdType>
__global__ void __launch_bounds__(WARPS * 32) SparsePagedScoresKernel(
    const DType* __restrict__ q, const DType* __restrict__ k_cache,
    const IdType* __restrict__ page_table, const IdType* __restrict__ token_to_req,
    const IdType* __restrict__ query_positions, const IdType* __restrict__ sequence_lengths,
    IdType* __restrict__ visible_out, float* __restrict__ logits, uint32_t stride_q_row,
    uint32_t stride_q_head, uint32_t stride_cache_page, uint32_t stride_cache_entry,
    uint32_t stride_table_req, uint32_t stride_logits_row, uint32_t rows, uint32_t num_columns,
    uint32_t num_pages, uint32_t num_requests, uint32_t table_width, uint32_t num_heads,
    uint_fastdiv page_size, uint_fastdiv compress_ratio, float divisor) {
  using namespace sparse_scores;
  constexpr uint32_t kThreads = WARPS * 32;
  // Tiles a warp owns so the block still covers kBlockN columns.
  constexpr uint32_t kTilesPerWarp = kBlockN / (WARPS * kTileM);
  constexpr uint32_t kSmemRow = HEAD_DIM + kSmemPad;
  constexpr uint32_t kSliceRow = SLICE_K + kSmemPad;

  extern __shared__ uint8_t smem_raw[];
  // Query first: it is staged once and read by every column tile.
  DType* q_smem = reinterpret_cast<DType*>(smem_raw);
  DType* k_smem = q_smem + kTileN * kSmemRow;
  int32_t* pages_smem = reinterpret_cast<int32_t*>(k_smem + 2 * kBlockN * kSliceRow);
  uint32_t* entries_smem = reinterpret_cast<uint32_t*>(pages_smem + TILES_PER_BLOCK * kBlockN);

  // Rows on x so that blocks scoring the same columns run together: rows of one
  // request read the same keys, and consecutive blocks are what shares a cache.
  const uint32_t row = blockIdx.x;
  if (row >= rows) return;

  const int32_t request = static_cast<int32_t>(token_to_req[row]);
  const bool request_valid = request >= 0 && request < static_cast<int32_t>(num_requests);
  const int32_t safe_request = min(max(request, 0), static_cast<int32_t>(num_requests) - 1);
  const int32_t query_position = static_cast<int32_t>(query_positions[row]);
  const int32_t sequence_length =
      request_valid ? static_cast<int32_t>(sequence_lengths[safe_request]) : 0;

  // A query sees only the compressed entries whose tokens are all behind it.
  // Clamp before the unsigned divide: a row with no request carries a length of
  // zero, and a negative position would otherwise divide as a huge number.
  uint32_t q_blocks, k_blocks, ignored;
  compress_ratio.divmod(static_cast<uint32_t>(max(query_position + 1, 0)), q_blocks, ignored);
  compress_ratio.divmod(static_cast<uint32_t>(max(sequence_length, 0)), k_blocks, ignored);
  const uint32_t visible = min(min(q_blocks, k_blocks), num_columns);
  if (blockIdx.y == 0 && threadIdx.x == 0) {
    visible_out[row] = static_cast<IdType>(min(q_blocks, k_blocks));
  }

  const uint32_t first_column = blockIdx.y * (kBlockN * TILES_PER_BLOCK);
  if (first_column >= visible) return;

  // Stage the query once: [head][feature], which is the byte layout the
  // row-col mma wants for its column-major B operand.
  const uint32_t heads = min(num_heads, kMaxHeads);
  // Sixteen bytes at a time: element-wise this is the same bytes but eight
  // times the load instructions, and instruction count is what this kernel is
  // short of. Every address it will form has to be aligned to that, which the
  // strides alone do not settle -- a view onto the middle of a tensor has
  // whatever strides it started with and a base that is not.
  constexpr uint32_t kQVec = 16 / sizeof(DType);
  const bool q_vectorizable = HEAD_DIM % kQVec == 0 && stride_q_head % kQVec == 0 &&
                              stride_q_row % kQVec == 0 && reinterpret_cast<uintptr_t>(q) % 16 == 0;
  if (q_vectorizable) {
    constexpr uint32_t kQVecsPerHead = HEAD_DIM / kQVec;
    for (uint32_t i = threadIdx.x; i < kTileN * kQVecsPerHead; i += kThreads) {
      const uint32_t h = i / kQVecsPerHead;
      const uint32_t d = (i - h * kQVecsPerHead) * kQVec;
      float4* dst = reinterpret_cast<float4*>(q_smem + h * kSmemRow + d);
      if (h < heads) {
        *dst = *reinterpret_cast<const float4*>(q + row * stride_q_row + h * stride_q_head + d);
      } else {
        *dst = make_float4(0.f, 0.f, 0.f, 0.f);
      }
    }
  } else {
    for (uint32_t i = threadIdx.x; i < kTileN * HEAD_DIM; i += kThreads) {
      const uint32_t h = i / HEAD_DIM;
      const uint32_t d = i - h * HEAD_DIM;
      q_smem[h * kSmemRow + d] =
          h < heads ? q[row * stride_q_row + h * stride_q_head + d] : DType(0);
    }
  }
  __syncthreads();

  const uint32_t warp = threadIdx.x / 32;
  const uint32_t lane = threadIdx.x % 32;
  // ldmatrix quadrant addresses, per the m8n8.x4 fragment layout: A is read as
  // 16 rows of 16 features, B as 16 heads of the same, but their quadrants are
  // ordered differently.
  const uint32_t a_row = lane % 16;
  const uint32_t a_col = (lane / 16) * 8;
  const uint32_t b_row = (lane % 8) + 8 * (lane / 16);
  const uint32_t b_col = ((lane % 16) / 8) * 8;

  // Bytes one thread stages at a time. A wider request per thread spreads a
  // warp over more pages, and the pages are what the gather has to chase, so
  // this stays at the narrower width.
  constexpr uint32_t kPerVec = 16 / sizeof(DType);
  constexpr uint32_t kSlices = ceil_div(HEAD_DIM, SLICE_K);
  // Whether every slice is full. When it is, the multiply below has a
  // compile-time trip count and unrolls; leaving the bound as a runtime min()
  // costs that, which is most of the instruction issue in this kernel.
  constexpr bool kEvenSlices = HEAD_DIM % SLICE_K == 0;
  constexpr uint32_t kSteps = TILES_PER_BLOCK * kSlices;

  // Resolve every column this block will score, once. Doing it per tile would
  // put a barrier in the middle of the pipeline. Consecutive columns walk a page
  // in order, so a thread divides once and then counts.
  const uint32_t page_span = static_cast<uint32_t>(page_size);
  for (uint32_t i = threadIdx.x; i < TILES_PER_BLOCK * kBlockN; i += kThreads) {
    const uint32_t column = first_column + i;
    int32_t page = -1;
    uint32_t entry = 0;
    if (column < visible && request_valid) {
      uint32_t logical_page;
      page_size.divmod(column, logical_page, entry);
      if (logical_page < table_width) {
        const int32_t mapped =
            static_cast<int32_t>(page_table[safe_request * stride_table_req + logical_page]);
        if (mapped >= 0 && mapped < static_cast<int32_t>(num_pages)) page = mapped;
      }
    }
    pages_smem[i] = page;
    entries_smem[i] = entry;
  }
  (void)page_span;
  __syncthreads();

  // The staged loads are 128 bits and cp_async has no narrower form, so every
  // row start the strides can produce has to be aligned to that as well as the
  // base. When it is not, the same slice is staged an element at a time; the
  // group is still committed so the pipeline's waits stay paired, and the
  // barrier that follows the wait publishes the writes either way.
  const bool k_vectorizable = reinterpret_cast<uintptr_t>(k_cache) % 16 == 0 &&
                              (stride_cache_page * sizeof(DType)) % 16 == 0 &&
                              (stride_cache_entry * sizeof(DType)) % 16 == 0;

  // One step stages one slice of one tile, so the pipeline runs unbroken across
  // tile boundaries.
  auto stage = [&](uint32_t step) {
    const uint32_t tile = step / kSlices;
    // Columns the row cannot see are never scored, so staging them reads the
    // cache for nothing. The group is still committed so the waits stay paired.
    if (first_column + tile * kBlockN >= visible) {
      cp_async::commit_group();
      return;
    }
    const uint32_t k0 = (step - tile * kSlices) * SLICE_K;
    DType* dst_base = k_smem + (step & 1) * kBlockN * kSliceRow;
    constexpr uint32_t kEvenVecs = SLICE_K / kPerVec;
    const uint32_t vecs = kEvenSlices ? kEvenVecs : min(SLICE_K, HEAD_DIM - k0) / kPerVec;
    if (k_vectorizable) {
#pragma unroll 4
      for (uint32_t i = threadIdx.x; i < kBlockN * vecs; i += kThreads) {
        const uint32_t c = i / vecs;
        const uint32_t v = i - c * vecs;
        const int32_t page = pages_smem[tile * kBlockN + c];
        // A column with no page contributes nothing; zero-filling keeps the mma
        // clean and the -inf below keeps it unselectable.
        const DType* src = page >= 0 ? k_cache + static_cast<int64_t>(page) * stride_cache_page +
                                           entries_smem[tile * kBlockN + c] * stride_cache_entry +
                                           k0 + v * kPerVec
                                     : nullptr;
        cp_async::pred_load<128, cp_async::PrefetchMode::kNoPrefetch,
                            cp_async::SharedMemFillMode::kFillZero>(
            dst_base + c * kSliceRow + v * kPerVec, src, page >= 0);
      }
    } else {
      const uint32_t elems = vecs * kPerVec;
      for (uint32_t i = threadIdx.x; i < kBlockN * elems; i += kThreads) {
        const uint32_t c = i / elems;
        const uint32_t e = i - c * elems;
        const int32_t page = pages_smem[tile * kBlockN + c];
        dst_base[c * kSliceRow + e] =
            page >= 0 ? k_cache[static_cast<int64_t>(page) * stride_cache_page +
                                entries_smem[tile * kBlockN + c] * stride_cache_entry + k0 + e]
                      : DType(0);
      }
    }
    cp_async::commit_group();
  };

  float acc[kTilesPerWarp][8];
#pragma unroll
  for (uint32_t m = 0; m < kTilesPerWarp; ++m) {
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) acc[m][i] = 0.f;
  }

  stage(0);
  for (uint32_t step = 0; step < kSteps; ++step) {
    const uint32_t tile = step / kSlices;
    const uint32_t slice_idx = step - tile * kSlices;
    const uint32_t block_column = first_column + tile * kBlockN;
    if (block_column >= visible) break;
    const uint32_t k0 = slice_idx * SLICE_K;

    cp_async::wait_group<0>();
    __syncthreads();
    if (step + 1 < kSteps) stage(step + 1);

    {
      const DType* slice_keys = k_smem + (step & 1) * kBlockN * kSliceRow;
      // Split on the compile-time case so the multiply has a constant trip
      // count there: a runtime bound stops the unroll, and the issue rate of
      // this loop is what the kernel spends its time on.
      if constexpr (kEvenSlices) {
#pragma unroll
        for (uint32_t kk = 0; kk < SLICE_K; kk += kTileK) {
          // One query fragment feeds every tile this warp owns.
          uint32_t b_frag[4];
          mma::ldmatrix_m8n8x4(b_frag, q_smem + b_row * kSmemRow + k0 + kk + b_col);
#pragma unroll
          for (uint32_t m = 0; m < kTilesPerWarp; ++m) {
            uint32_t a_frag[4];
            const uint32_t tile_row = (warp * kTilesPerWarp + m) * kTileM + a_row;
            mma::ldmatrix_m8n8x4(a_frag, slice_keys + tile_row * kSliceRow + kk + a_col);
            mma::mma_sync_m16n16k16_row_col_f16f16f32<DType>(acc[m], a_frag, b_frag);
          }
        }
      } else {
        const uint32_t slice = min(SLICE_K, HEAD_DIM - k0);
        for (uint32_t kk = 0; kk < slice; kk += kTileK) {
          uint32_t b_frag[4];
          mma::ldmatrix_m8n8x4(b_frag, q_smem + b_row * kSmemRow + k0 + kk + b_col);
#pragma unroll
          for (uint32_t m = 0; m < kTilesPerWarp; ++m) {
            uint32_t a_frag[4];
            const uint32_t tile_row = (warp * kTilesPerWarp + m) * kTileM + a_row;
            mma::ldmatrix_m8n8x4(a_frag, slice_keys + tile_row * kSliceRow + kk + a_col);
            mma::mma_sync_m16n16k16_row_col_f16f16f32<DType>(acc[m], a_frag, b_frag);
          }
        }
      }
    }

    if (slice_idx + 1 < kSlices) continue;

    const int32_t* tile_pages = pages_smem + tile * kBlockN;

    // C fragment: with g = lane >> 2 and u = lane & 3, this thread holds
    // (g, 2u) (g, 2u+1) (g+8, 2u) (g+8, 2u+1) (g, 8+2u) (g, 9+2u)
    // (g+8, 8+2u) (g+8, 9+2u) -- two columns of the tile and four heads each.
    const uint32_t g = lane >> 2;
    const uint32_t u = lane & 3;
#pragma unroll
    for (uint32_t m = 0; m < kTilesPerWarp; ++m) {
      const float* a = acc[m];
      float s0 = fmaxf(a[0], 0.f) + fmaxf(a[1], 0.f) + fmaxf(a[4], 0.f) + fmaxf(a[5], 0.f);
      float s1 = fmaxf(a[2], 0.f) + fmaxf(a[3], 0.f) + fmaxf(a[6], 0.f) + fmaxf(a[7], 0.f);
      // The four lanes sharing a row hold the remaining heads.
      s0 += __shfl_xor_sync(0xffffffffu, s0, 1);
      s0 += __shfl_xor_sync(0xffffffffu, s0, 2);
      s1 += __shfl_xor_sync(0xffffffffu, s1, 1);
      s1 += __shfl_xor_sync(0xffffffffu, s1, 2);

      // A column on an unmapped page staged as zeros, which would score 0; it
      // has to be unselectable instead.
      if (u == 0) {
        const uint32_t base = (warp * kTilesPerWarp + m) * kTileM;
        const uint32_t c0 = block_column + base + g;
        const uint32_t c1 = block_column + base + g + 8;
        if (c0 < visible) {
          logits[row * stride_logits_row + c0] =
              tile_pages[base + g] >= 0 ? s0 / divisor : -INFINITY;
        }
        if (c1 < visible) {
          logits[row * stride_logits_row + c1] =
              tile_pages[base + g + 8] >= 0 ? s1 / divisor : -INFINITY;
        }
      }
    }
#pragma unroll
    for (uint32_t m = 0; m < kTilesPerWarp; ++m) {
#pragma unroll
      for (uint32_t i = 0; i < 8; ++i) acc[m][i] = 0.f;
    }
  }
}

template <uint32_t HEAD_DIM, typename DType, typename IdType>
cudaError_t SparsePagedScores(const DType* q, const DType* k_cache, const IdType* page_table,
                              const IdType* token_to_req, const IdType* query_positions,
                              const IdType* sequence_lengths, IdType* visible_out, float* logits,
                              uint32_t stride_q_row, uint32_t stride_q_head,
                              uint32_t stride_cache_page, uint32_t stride_cache_entry,
                              uint32_t stride_table_req, uint32_t stride_logits_row, uint32_t rows,
                              uint32_t num_columns, uint32_t num_pages, uint32_t num_requests,
                              uint32_t table_width, uint32_t num_heads, uint32_t page_size,
                              uint32_t compress_ratio, float divisor, cudaStream_t stream) {
  using namespace sparse_scores;
  if (rows == 0 || num_columns == 0) return cudaSuccess;
  if (num_heads > kMaxHeads) return cudaErrorInvalidValue;
  if (HEAD_DIM % kTileK != 0) return cudaErrorInvalidValue;

  constexpr uint32_t kSmemRow = HEAD_DIM + kSmemPad;
  // Keys are two slice-sized buffers; only the resolved page metadata grows
  // with the tiles a block walks.
  auto smem_for = [](uint32_t tiles, uint32_t slice_k) {
    return (kTileN * kSmemRow + 2 * kBlockN * (slice_k + kSmemPad)) * sizeof(DType) +
           tiles * kBlockN * (sizeof(int32_t) + sizeof(uint32_t));
  };
  constexpr uint32_t kSingleTileSliceK = HEAD_DIM;

  // Few rows leave the device idle unless every column tile is its own block;
  // many rows are better off amortizing the query staging across tiles. The
  // crossover is where the narrow choice stops filling the device, which
  // depends on how many blocks this GPU can hold.
  int dev_id = 0, num_sms = 0, max_smem_per_block_optin = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&max_smem_per_block_optin,
                                              cudaDevAttrMaxSharedMemoryPerBlockOptin, dev_id));
  const size_t largest = max(smem_for(8, kMultiTileSliceK), smem_for(1, kSingleTileSliceK));
  if (largest > static_cast<size_t>(max_smem_per_block_optin)) {
    std::ostringstream err_msg;
    err_msg << "Required shared memory (" << largest << " bytes) for head_dim=" << HEAD_DIM
            << " exceeds this GPU's per-block limit (" << max_smem_per_block_optin
            << " bytes); this configuration is not supported on this architecture.";
    FLASHINFER_ERROR(err_msg.str());
  }

  // One column tile per block gives the most blocks, which is what a handful of
  // rows needs; past a full wave of them the extra blocks only re-stage the
  // query, so a block walks several tiles instead.
  auto launch = [&](auto tiles_tag) -> cudaError_t {
    constexpr uint32_t TILES = decltype(tiles_tag)::value;
    constexpr uint32_t SLICE_K = TILES == 1 ? kSingleTileSliceK : kMultiTileSliceK;
    // A single-tile launch runs because the device is not full, so it takes the
    // deeper staging; a multi-tile one keeps more warps resident instead.
    constexpr uint32_t WARPS = TILES == 1 ? kDeepWarps : kWideWarps;
    constexpr uint32_t THREADS = WARPS * 32;
    const size_t smem_size = smem_for(TILES, SLICE_K);
    auto kernel = SparsePagedScoresKernel<HEAD_DIM, TILES, SLICE_K, WARPS, DType, IdType>;
    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    const dim3 grid(rows, ceil_div(num_columns, kBlockN * TILES));
    kernel<<<grid, THREADS, smem_size, stream>>>(
        q, k_cache, page_table, token_to_req, query_positions, sequence_lengths, visible_out,
        logits, stride_q_row, stride_q_head, stride_cache_page, stride_cache_entry,
        stride_table_req, stride_logits_row, rows, num_columns, num_pages, num_requests,
        table_width, num_heads, uint_fastdiv(page_size), uint_fastdiv(compress_ratio), divisor);
    return cudaGetLastError();
  };

  int blocks_per_sm = 0;
  FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm,
      SparsePagedScoresKernel<HEAD_DIM, 1, kSingleTileSliceK, kDeepWarps, DType, IdType>,
      kDeepWarps * 32, smem_for(1, kSingleTileSliceK)));
  const uint32_t narrow_blocks = rows * ceil_div(num_columns, kBlockN);
  if (blocks_per_sm == 0 ||
      narrow_blocks <= static_cast<uint32_t>(num_sms) * static_cast<uint32_t>(blocks_per_sm)) {
    return launch(std::integral_constant<uint32_t, 1>{});
  }
  return launch(std::integral_constant<uint32_t, 8>{});
}

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_SPARSE_SCORES_CUH_
