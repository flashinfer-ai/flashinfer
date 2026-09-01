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
// Columns a block scores at once.
constexpr uint32_t kBlockN = 64;
// Warps in a block. Four is one column tile per warp, which is what both
// launch shapes want: the deep one is short of copies in flight and a fifth
// warp would not add any, and the wide one is short of resident blocks, which
// more warps per block only costs. Named once so the two launches cannot drift
// apart silently.
constexpr uint32_t kWarpsPerBlock = 4;
// Query heads a block handles. More than this needs a second n-tile, which the
// caller currently never asks for. Fewer than half of it is served by the
// narrower multiply instead, since the wide one would spend half its work on
// the padding.
constexpr uint32_t kMaxHeads = kTileN;
constexpr uint32_t kNarrowTileN = kTileN / 2;
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
 * instead; the count of what it can see is written out, and the selector bounds
 * its own k by that count.
 *
 * The ReLU applies to a head's completed dot product, so the feature axis has to
 * be fully accumulated before the heads are summed.
 *
 * \tparam HEAD_DIM per-head feature width, a multiple of 16
 * \tparam TILES_PER_BLOCK column tiles one block walks, to amortize staging the
 *   query across more columns when there are many rows to score
 * \tparam TILE_N query heads the multiply covers at once, 8 or 16
 */
template <uint32_t HEAD_DIM, uint32_t TILES_PER_BLOCK, uint32_t SLICE_K, uint32_t WARPS,
          uint32_t TILE_N, typename DType, typename IdType>
__global__ void __launch_bounds__(WARPS * 32) SparsePagedScoresKernel(
    const DType* __restrict__ q, const DType* __restrict__ k_cache,
    const IdType* __restrict__ page_table, const IdType* __restrict__ token_to_req,
    const IdType* __restrict__ query_positions, const IdType* __restrict__ sequence_lengths,
    IdType* __restrict__ visible_out, float* __restrict__ logits, uint32_t stride_q_row,
    uint32_t stride_q_head, uint32_t stride_cache_page, uint32_t stride_cache_entry,
    uint32_t stride_table_req, uint32_t stride_logits_row, uint32_t rows, uint32_t num_columns,
    uint32_t num_pages, uint32_t num_requests, uint32_t table_width, uint32_t num_heads,
    uint_fastdiv page_size, uint_fastdiv compress_ratio, float inv_divisor) {
  using namespace sparse_scores;
  constexpr uint32_t kThreads = WARPS * 32;
  // Tiles a warp owns so the block still covers kBlockN columns.
  constexpr uint32_t kTilesPerWarp = kBlockN / (WARPS * kTileM);
  constexpr uint32_t kSmemRow = HEAD_DIM + kSmemPad;
  constexpr uint32_t kSliceRow = SLICE_K + kSmemPad;
  constexpr uint32_t kSlices = ceil_div(HEAD_DIM, SLICE_K);
  // A launch that stages the whole head in one step has nothing to overlap, so
  // it takes one buffer rather than two. On the shapes that pick that launch the
  // block is what the device is short of, and the buffer is most of a block.
  constexpr uint32_t kStages = TILES_PER_BLOCK * kSlices > 1 ? 2 : 1;

  extern __shared__ uint8_t smem_raw[];
  // Query first: it is staged once and read by every column tile.
  DType* q_smem = reinterpret_cast<DType*>(smem_raw);
  DType* k_smem = q_smem + TILE_N * kSmemRow;
  // One resolved cache address per column. Holding the address rather than the
  // page and the entry keeps the staging loop free of the 64-bit multiply that
  // resolving them costs, and it takes the same eight bytes a column already
  // spent on the pair. A column with no page carries the first page's address
  // so that the copy needs no predicate; the bitmask beside it is what marks
  // the column unscorable.
  uintptr_t* bases_smem = reinterpret_cast<uintptr_t*>(k_smem + kStages * kBlockN * kSliceRow);
  uint32_t* live_smem =
      reinterpret_cast<uint32_t*>(bases_smem + TILES_PER_BLOCK * kBlockN);

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
    // Bounded by what was scored, not by what the query could see: a top-k
    // takes this as its width, and anything past num_columns has no logit.
    visible_out[row] = static_cast<IdType>(visible);
  }

  const uint32_t first_column = blockIdx.y * (kBlockN * TILES_PER_BLOCK);
  if (first_column >= visible) return;

  // Stage the query once: [head][feature], which is the byte layout the
  // row-col mma wants for its column-major B operand.
  const uint32_t heads = min(num_heads, TILE_N);
  // Sixteen bytes at a time: element-wise this is the same bytes but eight
  // times the load instructions, and instruction count is what this kernel is
  // short of. The head is a multiple of the vector width and both the query
  // rows and the staged rows start aligned.
  constexpr uint32_t kQVec = 16 / sizeof(DType);
  // Every address it forms has to be aligned to that, which the strides alone
  // do not settle -- a view onto the middle of a tensor keeps whatever strides
  // it started with and carries a base that is not.
  const bool q_vectorizable = HEAD_DIM % kQVec == 0 && stride_q_head % kQVec == 0 &&
                              stride_q_row % kQVec == 0 &&
                              reinterpret_cast<uintptr_t>(q) % 16 == 0;
  if (q_vectorizable) {
    constexpr uint32_t kQVecsPerHead = HEAD_DIM / kQVec;
    for (uint32_t i = threadIdx.x; i < TILE_N * kQVecsPerHead; i += kThreads) {
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
    // Built from a float, not an int: converting an integer literal to a
    // half-width float is a runtime conversion, and it would sit in the loop.
    const DType zero = DType(0.f);
    for (uint32_t i = threadIdx.x; i < TILE_N * HEAD_DIM; i += kThreads) {
      const uint32_t h = i / HEAD_DIM;
      const uint32_t d = i - h * HEAD_DIM;
      q_smem[h * kSmemRow + d] =
          h < heads ? q[row * stride_q_row + h * stride_q_head + d] : zero;
    }
  }

  // Resolve every column this block will score, once. Doing it per tile would
  // put a barrier in the middle of the pipeline.
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
    bases_smem[i] =
        page >= 0 ? reinterpret_cast<uintptr_t>(k_cache + static_cast<int64_t>(page) *
                                                              stride_cache_page +
                                                entry * stride_cache_entry)
                  // Never read, since a block with no page stages nothing, but
                  // it keeps the staging loop free of a predicate.
                  : reinterpret_cast<uintptr_t>(k_cache);
    // Consecutive threads take consecutive columns, so a warp holds exactly the
    // thirty-two bits of one mask word.
    const uint32_t live = __ballot_sync(0xffffffffu, page >= 0);
    if ((i & 31u) == 0) live_smem[i >> 5] = live;
  }
  __syncthreads();

  const uint32_t warp = threadIdx.x / 32;
  const uint32_t lane = threadIdx.x % 32;
  // ldmatrix quadrant addresses, per the m8n8.x4 fragment layout: A is read as
  // 16 rows of 16 features, B as 16 heads of the same, but their quadrants are
  // ordered differently.
  const uint32_t a_row = lane % 16;
  const uint32_t a_col = (lane / 16) * 8;
  // The x2 form takes its addresses from the low half of the warp. Having the
  // high half repeat them keeps every lane inside the staged rows, which the
  // architectures that read all thirty-two require.
  const uint32_t b_row = TILE_N == kTileN ? (lane % 8) + 8 * (lane / 16) : lane % 8;
  const uint32_t b_col = ((lane % 16) / 8) * 8;

  // Bytes one thread stages at a time. A wider request per thread spreads a
  // warp over more pages, and the pages are what the gather has to chase, so
  // this stays at the narrower width.
  constexpr uint32_t kPerVec = 16 / sizeof(DType);
  // Whether every slice is full. When it is, the multiply below has a
  // compile-time trip count and unrolls; leaving the bound as a runtime min()
  // costs that, which is most of the instruction issue in this kernel.
  constexpr bool kEvenSlices = HEAD_DIM % SLICE_K == 0;


  // One step stages one slice of one tile, so the pipeline runs unbroken across
  // tile boundaries.
  // The caller passes the slice offset rather than a step number so that it
  // stays a constant: every address below is then a fixed distance from one the
  // thread already holds.
  // The staged loads are 128 bits wide and cp_async has no narrower form, so
  // every column address the strides can resolve to has to be aligned to that.
  // When it is not, the slice is staged an element at a time; the group is
  // still committed so the pipeline's waits stay paired, and the barrier after
  // the wait publishes the writes either way.
  const bool k_vectorizable = reinterpret_cast<uintptr_t>(k_cache) % 16 == 0 &&
                              (stride_cache_page * sizeof(DType)) % 16 == 0 &&
                              (stride_cache_entry * sizeof(DType)) % 16 == 0;

  auto stage = [&](uint32_t tile, uint32_t k0, uint32_t parity) {
    // Columns the row cannot see are never scored, so staging them reads the
    // cache for nothing. The group is still committed so the waits stay paired.
    // With no page in the cache every column is unmapped, so there is nothing
    // to read and nothing to fall back to; the mask below writes -inf for all
    // of them. The test is uniform across the block.
    if (num_pages == 0 || first_column + tile * kBlockN >= visible) {
      cp_async::commit_group();
      return;
    }
    DType* dst_base = k_smem + parity * kBlockN * kSliceRow;
    const uintptr_t* tile_bases = bases_smem + tile * kBlockN;
    if (!k_vectorizable) {
      const uint32_t elems = kEvenSlices ? SLICE_K : min(SLICE_K, HEAD_DIM - k0);
      for (uint32_t i = threadIdx.x; i < kBlockN * elems; i += kThreads) {
        const uint32_t c = i / elems;
        const uint32_t e = i - c * elems;
        dst_base[c * kSliceRow + e] =
            reinterpret_cast<const DType*>(tile_bases[c])[k0 + e];
      }
      cp_async::commit_group();
      return;
    }
    constexpr uint32_t kSliceVecs = SLICE_K / kPerVec;
    // A thread keeps its position inside the column across the whole slice when
    // the launch divides evenly, which makes every address in the loop below a
    // constant away from one the thread already holds. The general form is kept
    // for the shapes that do not divide.
    constexpr bool kUniformVec = kEvenSlices && kThreads % kSliceVecs == 0 &&
                                 (kBlockN * kSliceVecs) % kThreads == 0;
    if constexpr (kUniformVec) {
      constexpr uint32_t kIters = kBlockN * kSliceVecs / kThreads;
      constexpr uint32_t kColStep = kThreads / kSliceVecs;
      const uint32_t v = threadIdx.x % kSliceVecs;
      const uint32_t c0 = threadIdx.x / kSliceVecs;
      const uint32_t offset = (k0 + v * kPerVec) * sizeof(DType);
      DType* dst = dst_base + c0 * kSliceRow + v * kPerVec;
      // Every address first, then every copy. A copy whose address is still
      // being fetched cannot be batched with the one before it, and the batching
      // is what decides how many pipeline slots the group costs.
      uintptr_t src[kIters];
#pragma unroll
      for (uint32_t it = 0; it < kIters; ++it) src[it] = tile_bases[c0 + it * kColStep] + offset;
#pragma unroll
      for (uint32_t it = 0; it < kIters; ++it) {
        // Rows of one request read the same keys, and blocks for those rows run
        // together, so L1 is where most of these repeats belong.
        cp_async::load_128b<cp_async::PrefetchMode::kNoPrefetch, cp_async::CacheMode::kCacheAll>(
            dst + it * kColStep * kSliceRow, reinterpret_cast<const DType*>(src[it]));
      }
    } else {
      constexpr uint32_t kEvenVecs = SLICE_K / kPerVec;
      const uint32_t vecs = kEvenSlices ? kEvenVecs : min(SLICE_K, HEAD_DIM - k0) / kPerVec;
      for (uint32_t i = threadIdx.x; i < kBlockN * vecs; i += kThreads) {
        const uint32_t c = i / vecs;
        const uint32_t v = i - c * vecs;
        const uintptr_t base = tile_bases[c];
        cp_async::load_128b<cp_async::PrefetchMode::kNoPrefetch, cp_async::CacheMode::kCacheAll>(
            dst_base + c * kSliceRow + v * kPerVec,
            reinterpret_cast<const DType*>(base + (k0 + v * kPerVec) * sizeof(DType)));
      }
    }
    cp_async::commit_group();
  };

  stage(0, 0, 0);

  // Every column tile multiplies against the same query, so a warp reads its
  // query fragments once and keeps them. Reloading them per multiply is what
  // puts this kernel at one ldmatrix per mma instead of the half that the key
  // side alone needs; the bound below is the register file.
  constexpr uint32_t kQFrags = HEAD_DIM / kTileK;
  constexpr uint32_t kBRegs = TILE_N / 4;
  constexpr uint32_t kAccs = TILE_N / 2;
  constexpr bool kHoldQuery = kQFrags * kBRegs <= 32;
  uint32_t q_frag[kHoldQuery ? kQFrags : 1][kBRegs];
  if constexpr (kHoldQuery) {
#pragma unroll
    for (uint32_t f = 0; f < kQFrags; ++f) {
      if constexpr (TILE_N == kTileN) {
        mma::ldmatrix_m8n8x4(q_frag[f], q_smem + b_row * kSmemRow + f * kTileK + b_col);
      } else {
        mma::ldmatrix_m8n8x2(q_frag[f], q_smem + b_row * kSmemRow + f * kTileK + b_col);
      }
    }
  }

  float acc[kTilesPerWarp][kAccs];
#pragma unroll
  for (uint32_t m = 0; m < kTilesPerWarp; ++m) {
#pragma unroll
    for (uint32_t i = 0; i < kAccs; ++i) acc[m][i] = 0.f;
  }

  // Slices are the inner loop so that the slice offset is a constant: it picks
  // the query fragment, and a register array can only be indexed by one.
  for (uint32_t tile = 0; tile < TILES_PER_BLOCK; ++tile) {
    const uint32_t block_column = first_column + tile * kBlockN;
    if (block_column >= visible) break;

#pragma unroll
    for (uint32_t slice = 0; slice < kSlices; ++slice) {
      const uint32_t k0 = slice * SLICE_K;
      const uint32_t step = tile * kSlices + slice;

      cp_async::wait_group<0>();
      __syncthreads();
      if (slice + 1 < kSlices) {
        stage(tile, k0 + SLICE_K, (step + 1) & (kStages - 1));
      } else if (tile + 1 < TILES_PER_BLOCK) {
        stage(tile + 1, 0, (step + 1) & (kStages - 1));
      }

      const DType* slice_keys = k_smem + (step & (kStages - 1)) * kBlockN * kSliceRow;
      // A short tail slice folds to a constant here because the offset does.
      const uint32_t slice_len = min(SLICE_K, HEAD_DIM - k0);
#pragma unroll
      for (uint32_t kk = 0; kk < slice_len; kk += kTileK) {
        uint32_t b_reg[kBRegs];
        uint32_t* b_frag;
        if constexpr (kHoldQuery) {
          b_frag = q_frag[(k0 + kk) / kTileK];
        } else {
          if constexpr (TILE_N == kTileN) {
            mma::ldmatrix_m8n8x4(b_reg, q_smem + b_row * kSmemRow + k0 + kk + b_col);
          } else {
            mma::ldmatrix_m8n8x2(b_reg, q_smem + b_row * kSmemRow + k0 + kk + b_col);
          }
          b_frag = b_reg;
        }
#pragma unroll
        for (uint32_t m = 0; m < kTilesPerWarp; ++m) {
          uint32_t a_frag[4];
          const uint32_t tile_row = (warp * kTilesPerWarp + m) * kTileM + a_row;
          mma::ldmatrix_m8n8x4(a_frag, slice_keys + tile_row * kSliceRow + kk + a_col);
          if constexpr (TILE_N == kTileN) {
            mma::mma_sync_m16n16k16_row_col_f16f16f32<DType>(acc[m], a_frag, b_frag);
          } else {
            mma::mma_sync_m16n8k16_row_col_f16f16f32<DType>(acc[m], a_frag, b_frag);
          }
        }
      }
    }

    const uint32_t tile_first = tile * kBlockN;
    float* row_logits = logits + row * stride_logits_row;

    // C fragment: with g = lane >> 2 and u = lane & 3, this thread holds
    // (g, 2u) (g, 2u+1) (g+8, 2u) (g+8, 2u+1) (g, 8+2u) (g, 9+2u)
    // (g+8, 8+2u) (g+8, 9+2u) -- two columns of the tile and four heads each.
    const uint32_t g = lane >> 2;
    const uint32_t u = lane & 3;
#pragma unroll
    for (uint32_t m = 0; m < kTilesPerWarp; ++m) {
      const float* a = acc[m];
      float s0 = fmaxf(a[0], 0.f) + fmaxf(a[1], 0.f);
      float s1 = fmaxf(a[2], 0.f) + fmaxf(a[3], 0.f);
      if constexpr (TILE_N == kTileN) {
        s0 += fmaxf(a[4], 0.f) + fmaxf(a[5], 0.f);
        s1 += fmaxf(a[6], 0.f) + fmaxf(a[7], 0.f);
      }
      // The four lanes sharing a row hold the remaining heads.
      s0 += __shfl_xor_sync(0xffffffffu, s0, 1);
      s0 += __shfl_xor_sync(0xffffffffu, s0, 2);
      s1 += __shfl_xor_sync(0xffffffffu, s1, 1);
      s1 += __shfl_xor_sync(0xffffffffu, s1, 2);

      // A column on an unmapped page read the first page instead, so its score
      // is meaningless; it has to be unselectable.
      if (u == 0) {
        const uint32_t base = (warp * kTilesPerWarp + m) * kTileM;
        const uint32_t i0 = tile_first + base + g;
        const uint32_t i1 = i0 + 8;
        const uint32_t c0 = block_column + base + g;
        const uint32_t c1 = c0 + 8;
        if (c0 < visible) {
          row_logits[c0] = (live_smem[i0 >> 5] >> (i0 & 31u)) & 1u ? s0 * inv_divisor : -INFINITY;
        }
        if (c1 < visible) {
          row_logits[c1] = (live_smem[i1 >> 5] >> (i1 & 31u)) & 1u ? s1 * inv_divisor : -INFINITY;
        }
      }
    }
#pragma unroll
    for (uint32_t m = 0; m < kTilesPerWarp; ++m) {
#pragma unroll
      for (uint32_t i = 0; i < kAccs; ++i) acc[m][i] = 0.f;
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
  // Keys are two slice-sized buffers; only the resolved column addresses grow
  // with the tiles a block walks.
  const uint32_t tile_n = num_heads <= kNarrowTileN ? kNarrowTileN : kTileN;
  auto smem_for = [&](uint32_t tiles, uint32_t slice_k) {
    const uint32_t stages = tiles * ceil_div(HEAD_DIM, slice_k) > 1 ? 2 : 1;
    return (tile_n * kSmemRow + stages * kBlockN * (slice_k + kSmemPad)) * sizeof(DType) +
           tiles * kBlockN * sizeof(uintptr_t) + ceil_div(tiles * kBlockN, 32u) * sizeof(uint32_t);
  };

  // Few rows leave the device idle unless every column tile is its own block;
  // many rows are better off amortizing the query staging across tiles. The
  // crossover is where the narrow choice stops filling the device, which
  // depends on how many blocks this GPU can hold.
  //
  // None of this changes between calls on the same device, and a scorer launch
  // is short enough that asking the driver again each time is a measurable part
  // of it, so the answers are kept.
  int dev_id = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  static thread_local int cached_dev = -1;
  static thread_local int num_sms = 0, max_smem_per_block_optin = 0;
  if (cached_dev != dev_id) {
    FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));
    FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(
        &max_smem_per_block_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev_id));
    cached_dev = dev_id;
  }
  const size_t largest = max(smem_for(8, kMultiTileSliceK), smem_for(1, HEAD_DIM));
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
  auto launch = [&](auto tiles_tag, auto tile_n_tag) -> cudaError_t {
    constexpr uint32_t TILES = decltype(tiles_tag)::value;
    constexpr uint32_t TILE_N = decltype(tile_n_tag)::value;
    // A single-tile launch runs because the device is not full. What it is then
    // short of is bytes in flight, not shared memory, so it asks for the whole
    // head at once instead of pipelining halves of it.
    constexpr uint32_t SLICE_K = TILES == 1 ? HEAD_DIM : kMultiTileSliceK;
    constexpr uint32_t WARPS = kWarpsPerBlock;
    constexpr uint32_t THREADS = WARPS * 32;
    const size_t smem_size = smem_for(TILES, SLICE_K);
    auto kernel = SparsePagedScoresKernel<HEAD_DIM, TILES, SLICE_K, WARPS, TILE_N, DType, IdType>;
    // The opt-in is a property of the kernel on the device, not of the launch.
    static thread_local int opted_in_dev = -1;
    if (opted_in_dev != dev_id) {
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      opted_in_dev = dev_id;
    }
    const dim3 grid(rows, ceil_div(num_columns, kBlockN * TILES));
    kernel<<<grid, THREADS, smem_size, stream>>>(
        q, k_cache, page_table, token_to_req, query_positions, sequence_lengths, visible_out,
        logits, stride_q_row, stride_q_head, stride_cache_page, stride_cache_entry,
        stride_table_req, stride_logits_row, rows, num_columns, num_pages, num_requests,
        table_width, num_heads, uint_fastdiv(page_size), uint_fastdiv(compress_ratio),
        1.0f / divisor);
    return cudaGetLastError();
  };

  // One tile per block re-stages the query for every column tile, which only
  // pays while the extra blocks still buy something. It buys more than one wave
  // of them: the single-tile block stages the whole head in one step and has
  // nothing to overlap inside itself, so what hides its latency is other blocks
  // on the same SM, and it keeps winning until the query re-staging outweighs
  // the pipelining the eight-tile shape does instead.
  //
  // Measured on SM80, the turn is at four waves and not at one. Sweeping rows
  // at three shapes, the last row count the single-tile shape wins and the
  // first the eight-tile shape does, in blocks: 1536 then 2048 at sixteen heads
  // and head dim 128, 2048 then 4096 at four heads, and already lost by 2048 at
  // eight heads and head dim 256. Those blocks per SM are 7, 8 and 4, so one
  // wave would be 490, 560 and 280 -- and four waves splits every one of them
  // correctly where one wave puts the whole 8-to-24 row range on the wrong side
  // and costs up to 1.64x there.
  //
  // The answer depends on the tile width too, since that is part of the block's
  // shared memory.
  constexpr uint32_t kWavesBeforeWide = 4;
  static thread_local int blocks_per_sm = 0;
  static thread_local int occupancy_dev = -1;
  static thread_local uint32_t occupancy_tile_n = 0;
  if (occupancy_dev != dev_id || occupancy_tile_n != tile_n) {
    FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        tile_n == kNarrowTileN
            ? SparsePagedScoresKernel<HEAD_DIM, 1, HEAD_DIM, kWarpsPerBlock, kNarrowTileN, DType,
                                      IdType>
            : SparsePagedScoresKernel<HEAD_DIM, 1, HEAD_DIM, kWarpsPerBlock, kTileN, DType, IdType>,
        kWarpsPerBlock * 32, smem_for(1, HEAD_DIM)));
    occupancy_dev = dev_id;
    occupancy_tile_n = tile_n;
  }
  const uint32_t narrow_blocks = rows * ceil_div(num_columns, kBlockN);
  const bool narrow = blocks_per_sm == 0 ||
                      narrow_blocks <= kWavesBeforeWide * static_cast<uint32_t>(num_sms) *
                                           static_cast<uint32_t>(blocks_per_sm);
  if (tile_n == kNarrowTileN) {
    constexpr std::integral_constant<uint32_t, kNarrowTileN> n{};
    return narrow ? launch(std::integral_constant<uint32_t, 1>{}, n)
                  : launch(std::integral_constant<uint32_t, 8>{}, n);
  }
  constexpr std::integral_constant<uint32_t, kTileN> n{};
  return narrow ? launch(std::integral_constant<uint32_t, 1>{}, n)
                : launch(std::integral_constant<uint32_t, 8>{}, n);
}

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_SPARSE_SCORES_CUH_
