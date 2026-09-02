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
#ifndef FLASHINFER_SPARSE_ROUTE_CUH_
#define FLASHINFER_SPARSE_ROUTE_CUH_

#include <cuda_runtime.h>

#include <cstdint>

#include "fastdiv.cuh"
#include "utils.cuh"

namespace flashinfer {

namespace sparse_route {

// Columns one thread block writes, and the threads that write them. Each thread
// takes every kThreads-th column of the tile, so a warp always covers 32 adjacent
// columns; giving a thread adjacent columns instead splits each warp's store across
// more sectors. The route is thousands wide, so tiling it also keeps enough blocks
// in flight when only a few rows are expanded -- a decode step has one per request.
// Two shapes, picked by whether the launch fills the device.
//
// A wide tile with few threads gives each block more columns and each SM more
// blocks, which wins once there are thousands of rows. It starves a decode step,
// where a handful of rows produce only a few blocks -- there the narrow tile with
// more threads per block keeps more lanes busy.
constexpr uint32_t kWideTile = 512;
constexpr uint32_t kWideThreads = 128;
constexpr uint32_t kNarrowTile = 256;
constexpr uint32_t kNarrowThreads = 256;

// Whether the wide shape fills the device is a question about this kernel on
// this GPU, so ask the driver rather than carry a number measured on one
// architecture. Half a wave of blocks is where the narrow shape's extra blocks
// stop buying anything.
template <typename KernelFn>
inline cudaError_t choose_wide(KernelFn wide_kernel, uint32_t threads, uint32_t wide_blocks,
                               bool* wide) {
  int dev_id = 0, num_sms = 0, blocks_per_sm = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));
  FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, wide_kernel,
                                                                     static_cast<int>(threads), 0));
  *wide = num_sms == 0 || blocks_per_sm == 0 ||
          wide_blocks >= static_cast<uint32_t>(num_sms) * static_cast<uint32_t>(blocks_per_sm) / 2;
  return cudaSuccess;
}

}  // namespace sparse_route

/*!
 * \brief Expand a per-query list of selected blocks into the token route it stands for.
 *
 * A block-granular selector picks `block_topk` blocks of `COMPRESS_RATIO` tokens each. The
 * attention that consumes the choice works on tokens, so every selected block becomes its
 * `COMPRESS_RATIO` tokens, laid out in selection order.
 *
 * The block a query itself sits in is only partially in the past, so it is never selected as a
 * whole. Its already-seen tokens are appended after the expanded blocks instead -- at most
 * `COMPRESS_RATIO - 1` of them, which is why a route is `block_topk * COMPRESS_RATIO +
 * COMPRESS_RATIO - 1` wide.
 *
 * Positions that no token reaches are written as -1; the consumer masks them.
 *
 * \tparam COMPRESS_RATIO tokens per block, a compile-time constant so the divisions
 *   by it fold into shifts for the power-of-two ratios this is used with
 * \tparam IdType index type of both the selection and the route
 */
template <uint32_t COMPRESS_RATIO, bool CONTIGUOUS_COLUMNS, uint32_t TILE, uint32_t THREADS,
          typename IdType>
__global__ void __launch_bounds__(THREADS)
    ExpandBlockRouteKernel(const IdType* __restrict__ block_indices,
                           const IdType* __restrict__ query_positions,
                           const IdType* __restrict__ sequence_lengths,
                           const IdType* __restrict__ token_to_req, IdType* __restrict__ out,
                           uint32_t stride_blocks_row, uint32_t stride_blocks_col,
                           uint32_t stride_out_row, uint32_t stride_out_col, uint32_t rows,
                           uint32_t num_requests, uint32_t block_topk) {
  // blockIdx.x walks the columns of one row so consecutive blocks write consecutive
  // memory; putting the row on x instead interleaves unrelated rows in flight.
  // The width follows from the selection: every block expands to COMPRESS_RATIO
  // tokens, plus the tail of the query's own block.
  const uint32_t output_width = block_topk * COMPRESS_RATIO + COMPRESS_RATIO - 1;
  const uint32_t row = blockIdx.y;
  const uint32_t tile_base = blockIdx.x * TILE;
  if (row >= rows || tile_base >= output_width) return;

  // Every column of this row shares them, so they are read once per block.
  // The index tensors may be int64 and everything below is int32, so each value
  // is bounded in its own width before it is narrowed. The request is already
  // compared against its count in IdType, which rejects anything an int32 could
  // not hold; the position and the length are not bounded from above by
  // anything else, and a value of exactly 2^32 narrows to zero. The bound is
  // written out for all three so the next one cannot regress quietly.
  constexpr IdType kInt32Max = static_cast<IdType>(2147483647);
  const IdType position_raw = query_positions[row];
  const int32_t query_position = (position_raw >= IdType(0) && position_raw <= kInt32Max)
                                     ? static_cast<int32_t>(position_raw)
                                     : -1;
  const IdType request_raw = token_to_req[row];
  const bool request_valid = request_raw >= IdType(0) && request_raw <= kInt32Max &&
                             request_raw < static_cast<IdType>(num_requests);
  const int32_t request = request_valid ? static_cast<int32_t>(request_raw) : -1;
  const int32_t safe_request =
      request_valid ? min(request, static_cast<int32_t>(num_requests) - 1) : 0;
  IdType length_raw = request_valid ? sequence_lengths[safe_request] : IdType(0);
  if (length_raw < IdType(0) || length_raw > kInt32Max) length_raw = IdType(0);
  const int32_t sequence_length = static_cast<int32_t>(length_raw);

  // Blocks entirely in the past, capped by what the selector produced.
  // One past the last block the query has entirely behind it. A selection is
  // only expandable while it names one of these: the block the query sits in
  // is partly ahead of it, and the tail below is what supplies its seen half.
  // query_position + 1 in its own width: INT32_MAX is inside the range the cast
  // above accepts, and adding one to it in int32 is signed overflow.
  const int64_t query_end = static_cast<int64_t>(query_position) + 1;
  const int32_t past_blocks = static_cast<int32_t>(
      min(query_end / static_cast<int64_t>(COMPRESS_RATIO),
          static_cast<int64_t>(sequence_length) / static_cast<int64_t>(COMPRESS_RATIO)));
  const int32_t complete_blocks = min(past_blocks, static_cast<int32_t>(block_topk));
  const int32_t expanded_count = complete_blocks * static_cast<int32_t>(COMPRESS_RATIO);
  const int32_t tail_start = static_cast<int32_t>(
      (query_end / static_cast<int64_t>(COMPRESS_RATIO)) * static_cast<int64_t>(COMPRESS_RATIO));
  const int32_t tail_count = static_cast<int32_t>(query_end - static_cast<int64_t>(tail_start));

  const IdType* row_blocks = block_indices + row * stride_blocks_row;
  IdType* row_out = out + row * stride_out_row;

  const uint32_t blocks_stride = CONTIGUOUS_COLUMNS ? 1u : stride_blocks_col;
  const uint32_t out_stride = CONTIGUOUS_COLUMNS ? 1u : stride_out_col;

#pragma unroll
  for (uint32_t i = 0; i < TILE / THREADS; ++i) {
    const uint32_t col = tile_base + threadIdx.x + i * THREADS;
    if (col >= output_width) break;
    const int32_t column = static_cast<int32_t>(col);
    int32_t token;
    bool valid;
    if (column < expanded_count) {
      const int32_t rank = column / static_cast<int32_t>(COMPRESS_RATIO);
      const int32_t offset = column - rank * static_cast<int32_t>(COMPRESS_RATIO);
      // The whole block, not the token: keeping only the seen half of a block
      // the query sits in would repeat exactly what the tail appends. Decided
      // before the multiply, so a block that is not one of ours is never
      // scaled by the ratio at all.
      const IdType block_raw = row_blocks[rank * blocks_stride];
      valid = block_raw >= IdType(0) && block_raw <= kInt32Max &&
              block_raw < static_cast<IdType>(past_blocks);
      token = valid
                  ? static_cast<int32_t>(block_raw) * static_cast<int32_t>(COMPRESS_RATIO) + offset
                  : -1;
    } else {
      const int32_t tail_offset = column - expanded_count;
      token = tail_start + tail_offset;
      valid = tail_offset < tail_count && tail_offset < static_cast<int32_t>(COMPRESS_RATIO) - 1;
    }
    // A selected block is meant to be one the query has already passed, which
    // is what the selector's own visible count bounds it to. Nothing here had
    // been checking it, though: the block id is the caller's, and the only
    // bound it met was the sequence length. A block past the query would have
    // expanded into tokens the query cannot see -- with a ratio of four, a
    // query at position 3 selecting block 2 routes tokens 8 through 11. The
    // route drops them rather than carry them.
    valid = valid && token >= 0 && token <= query_position && token < sequence_length;
    row_out[static_cast<uint32_t>(column) * out_stride] =
        valid ? static_cast<IdType>(token) : IdType(-1);
  }
}

/*!
 * \brief Turn a per-query block selection straight into a paged attention route.
 *
 * Fuses three steps that would otherwise each read and write the whole route:
 * expanding the selected blocks into tokens (see ExpandBlockRouteKernel), mapping
 * each token through the block table into a physical KV slot, and packing the
 * validity of every entry into the bitmask the attention kernel reads.
 *
 * A route entry is valid when it names a real token: inside the request, on a
 * logical page the block table covers, on a page the table actually maps, and in a
 * slot the cache holds. Invalid entries route to slot 0 with their mask bit clear,
 * because an out-of-range slot would be read before the mask is applied.
 *
 * The logical route is written out as well: a speculative decoder reuses the
 * selection across its steps, so it outlives the physical route derived from it.
 *
 * \tparam MASK_BYTES_PER_ROW ceil(output_width / 8), the stride of one mask row
 */
template <uint32_t COMPRESS_RATIO, bool CONTIGUOUS_COLUMNS, uint32_t TILE, uint32_t THREADS,
          typename IdType>
__global__ void __launch_bounds__(THREADS) QSARouteFromBlocksKernel(
    const IdType* __restrict__ block_indices, const IdType* __restrict__ query_positions,
    const IdType* __restrict__ sequence_lengths, const IdType* __restrict__ token_to_req,
    const IdType* __restrict__ block_table, IdType* __restrict__ out_logical,
    IdType* __restrict__ out_route, uint8_t* __restrict__ out_mask, uint32_t stride_blocks_row,
    uint32_t stride_blocks_col, uint32_t stride_logical_row, uint32_t stride_table_row,
    uint32_t rows, uint32_t num_requests, uint32_t block_topk, uint32_t table_width,
    uint32_t page_size, uint32_t num_slots, uint32_t mask_bytes_per_row) {
  const uint32_t output_width = block_topk * COMPRESS_RATIO + COMPRESS_RATIO - 1;
  const uint32_t row = blockIdx.y;
  const uint32_t tile_base = blockIdx.x * TILE;
  if (row >= rows || tile_base >= output_width) return;

  // The index tensors may be int64 and everything below is int32, so each value
  // is bounded in its own width before it is narrowed. The request is already
  // compared against its count in IdType, which rejects anything an int32 could
  // not hold; the position and the length are not bounded from above by
  // anything else, and a value of exactly 2^32 narrows to zero. The bound is
  // written out for all three so the next one cannot regress quietly.
  constexpr IdType kInt32Max = static_cast<IdType>(2147483647);
  const IdType position_raw = query_positions[row];
  const int32_t query_position = (position_raw >= IdType(0) && position_raw <= kInt32Max)
                                     ? static_cast<int32_t>(position_raw)
                                     : -1;
  const IdType request_raw = token_to_req[row];
  const bool request_valid = request_raw >= IdType(0) && request_raw <= kInt32Max &&
                             request_raw < static_cast<IdType>(num_requests);
  const int32_t request = request_valid ? static_cast<int32_t>(request_raw) : -1;
  const int32_t safe_request =
      request_valid ? min(request, static_cast<int32_t>(num_requests) - 1) : 0;
  IdType length_raw = request_valid ? sequence_lengths[safe_request] : IdType(0);
  if (length_raw < IdType(0) || length_raw > kInt32Max) length_raw = IdType(0);
  const int32_t sequence_length = static_cast<int32_t>(length_raw);

  // One past the last block the query has entirely behind it. A selection is
  // only expandable while it names one of these: the block the query sits in
  // is partly ahead of it, and the tail below is what supplies its seen half.
  // query_position + 1 in its own width: INT32_MAX is inside the range the cast
  // above accepts, and adding one to it in int32 is signed overflow.
  const int64_t query_end = static_cast<int64_t>(query_position) + 1;
  const int32_t past_blocks = static_cast<int32_t>(
      min(query_end / static_cast<int64_t>(COMPRESS_RATIO),
          static_cast<int64_t>(sequence_length) / static_cast<int64_t>(COMPRESS_RATIO)));
  const int32_t complete_blocks = min(past_blocks, static_cast<int32_t>(block_topk));
  const int32_t expanded_count = complete_blocks * static_cast<int32_t>(COMPRESS_RATIO);
  const int32_t tail_start = static_cast<int32_t>(
      (query_end / static_cast<int64_t>(COMPRESS_RATIO)) * static_cast<int64_t>(COMPRESS_RATIO));
  const int32_t tail_count = static_cast<int32_t>(query_end - static_cast<int64_t>(tail_start));

  const uint32_t blocks_stride = CONTIGUOUS_COLUMNS ? 1u : stride_blocks_col;
  const IdType* row_blocks = block_indices + row * stride_blocks_row;
  const IdType* row_table = request_valid ? block_table + safe_request * stride_table_row : nullptr;
  IdType* row_logical = out_logical + row * stride_logical_row;
  IdType* row_route = out_route + row * output_width;
  uint8_t* row_mask = out_mask + row * mask_bytes_per_row;

  // A warp always covers 32 consecutive columns, so its ballot is exactly the four
  // mask bytes that cover them and no two warps write the same byte.
  const uint32_t lane = threadIdx.x & 31u;

#pragma unroll
  for (uint32_t i = 0; i < TILE / THREADS; ++i) {
    const uint32_t col = tile_base + threadIdx.x + i * THREADS;
    const bool in_row = col < output_width;

    int32_t token = -1;
    bool valid = false;
    if (in_row) {
      const int32_t column = static_cast<int32_t>(col);
      if (column < expanded_count) {
        const int32_t rank = column / static_cast<int32_t>(COMPRESS_RATIO);
        const int32_t offset = column - rank * static_cast<int32_t>(COMPRESS_RATIO);
        // Same whole-block rule as the standalone expansion above, decided
        // before the multiply for the same reason.
        const IdType block_raw = row_blocks[rank * blocks_stride];
        valid = block_raw >= IdType(0) && block_raw <= kInt32Max &&
                block_raw < static_cast<IdType>(past_blocks);
        token =
            valid ? static_cast<int32_t>(block_raw) * static_cast<int32_t>(COMPRESS_RATIO) + offset
                  : -1;
      } else {
        const int32_t tail_offset = column - expanded_count;
        token = tail_start + tail_offset;
        valid = tail_offset < tail_count && tail_offset < static_cast<int32_t>(COMPRESS_RATIO) - 1;
      }
      // Same causal bound as the standalone expansion above: a selected block
      // the query has not reached would expand into tokens it cannot see, and
      // the sequence length alone does not stop them.
      valid = valid && token >= 0 && token <= query_position && token < sequence_length;
      if (!valid) token = -1;
      row_logical[col] = static_cast<IdType>(token);
    }

    // Logical token -> physical slot, folding every bound into the same validity.
    uint32_t slot = 0;
    if (valid) {
      const uint32_t logical_page = static_cast<uint32_t>(token) / page_size;
      if (logical_page < table_width && row_table != nullptr) {
        // The page id keeps its own width until it is bounded, and the slot is
        // formed in 64 bits: page * page_size overflows a uint32 long before
        // either factor does, and a wrapped product can land under num_slots.
        const IdType page = row_table[logical_page];
        if (page >= IdType(0) && page <= kInt32Max) {
          const uint64_t candidate =
              static_cast<uint64_t>(page) * page_size + static_cast<uint32_t>(token) % page_size;
          if (candidate < static_cast<uint64_t>(num_slots)) {
            slot = static_cast<uint32_t>(candidate);
          } else {
            valid = false;
          }
        } else {
          valid = false;
        }
      } else {
        valid = false;
      }
    }
    if (in_row) row_route[col] = static_cast<IdType>(slot);

    const uint32_t bits = __ballot_sync(0xffffffffu, valid && in_row);
    if (lane == 0) {
      const uint32_t byte_base = (tile_base + (threadIdx.x & ~31u) + i * THREADS) >> 3;
#pragma unroll
      for (uint32_t b = 0; b < 4; ++b) {
        const uint32_t byte_index = byte_base + b;
        if (byte_index < mask_bytes_per_row) {
          row_mask[byte_index] = static_cast<uint8_t>((bits >> (b * 8)) & 0xffu);
        }
      }
    }
  }
}

/*!
 * \brief Map a logical token route through a block table into physical KV slots.
 *
 * The second half of QSARouteFromBlocksKernel, for callers whose logical route was
 * produced earlier and outlived the physical one -- a speculative decoder reuses a
 * selection across its steps.
 *
 * An entry is valid when it names a real token: non-negative, on a logical page the
 * block table covers, on a page the table maps, and in a slot the cache holds.
 * Invalid entries route to slot 0 with their mask bit clear, because an
 * out-of-range slot would be read before the mask is applied.
 */
template <uint32_t TILE, uint32_t THREADS, typename IdType>
__global__ void __launch_bounds__(THREADS)
    QSARouteFromLogicalKernel(const IdType* __restrict__ logical,
                              const IdType* __restrict__ token_to_req,
                              const IdType* __restrict__ block_table,
                              IdType* __restrict__ out_route, uint8_t* __restrict__ out_mask,
                              uint32_t stride_logical_row, uint32_t stride_table_row, uint32_t rows,
                              uint32_t valid_rows, uint32_t num_requests, uint32_t width,
                              uint32_t table_width, uint_fastdiv page_size, uint32_t num_slots,
                              uint32_t mask_bytes_per_row) {
  const uint32_t row = blockIdx.y;
  const uint32_t tile_base = blockIdx.x * TILE;
  if (row >= rows || tile_base >= width) return;

  // Rows past the caller's token count are padding: they carry no request and must
  // come out fully masked.
  // Same range contract as the expansion kernels: an index is bounded in its
  // own width before it is narrowed, or a value of 2^32 becomes request zero.
  constexpr IdType kInt32Max = static_cast<IdType>(2147483647);
  const bool row_live = row < valid_rows;
  const IdType request_raw = row_live ? token_to_req[row] : IdType(-1);
  const bool request_valid = request_raw >= IdType(0) && request_raw <= kInt32Max &&
                             request_raw < static_cast<IdType>(num_requests);
  const int32_t request = request_valid ? static_cast<int32_t>(request_raw) : -1;
  const IdType* row_table = request_valid ? block_table + request * stride_table_row : nullptr;
  // Only a live row reads the logical route, so it needs to cover those rows and
  // no more -- a short step can hand over its own tensor instead of padding one.
  const IdType* row_logical = row_live ? logical + row * stride_logical_row : nullptr;
  IdType* row_route = out_route + row * width;
  uint8_t* row_mask = out_mask + row * mask_bytes_per_row;

  const uint32_t lane = threadIdx.x & 31u;

#pragma unroll
  for (uint32_t i = 0; i < TILE / THREADS; ++i) {
    const uint32_t col = tile_base + threadIdx.x + i * THREADS;
    const bool in_row = col < width;

    uint32_t slot = 0;
    bool valid = false;
    if (in_row && row_table != nullptr && row_logical != nullptr) {
      const IdType token_raw = row_logical[col];
      if (token_raw >= IdType(0) && token_raw <= kInt32Max) {
        const int32_t token = static_cast<int32_t>(token_raw);
        uint32_t logical_page, entry;
        page_size.divmod(static_cast<uint32_t>(token), logical_page, entry);
        if (logical_page < table_width) {
          const IdType page = row_table[logical_page];
          if (page >= IdType(0) && page <= kInt32Max) {
            const uint64_t candidate =
                static_cast<uint64_t>(page) * static_cast<uint32_t>(page_size) + entry;
            if (candidate < static_cast<uint64_t>(num_slots)) {
              slot = static_cast<uint32_t>(candidate);
              valid = true;
            }
          }
        }
      }
    }
    if (in_row) row_route[col] = static_cast<IdType>(slot);

    const uint32_t bits = __ballot_sync(0xffffffffu, valid);
    if (lane == 0) {
      const uint32_t byte_base = (tile_base + (threadIdx.x & ~31u) + i * THREADS) >> 3;
#pragma unroll
      for (uint32_t b = 0; b < 4; ++b) {
        const uint32_t byte_index = byte_base + b;
        if (byte_index < mask_bytes_per_row) {
          row_mask[byte_index] = static_cast<uint8_t>((bits >> (b * 8)) & 0xffu);
        }
      }
    }
  }
}

template <typename IdType>
cudaError_t QSARouteFromLogical(const IdType* logical, const IdType* token_to_req,
                                const IdType* block_table, IdType* out_route, uint8_t* out_mask,
                                uint32_t stride_logical_row, uint32_t stride_table_row,
                                uint32_t rows, uint32_t valid_rows, uint32_t num_requests,
                                uint32_t width, uint32_t table_width, uint32_t page_size,
                                uint32_t num_slots, uint32_t mask_bytes_per_row,
                                cudaStream_t stream) {
  if (rows == 0 || width == 0) return cudaSuccess;
  const uint32_t wide_blocks = rows * ceil_div(width, sparse_route::kWideTile);
  bool wide = true;
  FLASHINFER_CUDA_CALL(sparse_route::choose_wide(
      QSARouteFromLogicalKernel<sparse_route::kWideTile, sparse_route::kWideThreads, IdType>,
      sparse_route::kWideThreads, wide_blocks, &wide));
  const uint32_t tile = wide ? sparse_route::kWideTile : sparse_route::kNarrowTile;
  const dim3 grid(ceil_div(width, tile), rows);
  const uint_fastdiv page_div(page_size);

  if (wide) {
    QSARouteFromLogicalKernel<sparse_route::kWideTile, sparse_route::kWideThreads, IdType>
        <<<grid, sparse_route::kWideThreads, 0, stream>>>(
            logical, token_to_req, block_table, out_route, out_mask, stride_logical_row,
            stride_table_row, rows, valid_rows, num_requests, width, table_width, page_div,
            num_slots, mask_bytes_per_row);
  } else {
    QSARouteFromLogicalKernel<sparse_route::kNarrowTile, sparse_route::kNarrowThreads, IdType>
        <<<grid, sparse_route::kNarrowThreads, 0, stream>>>(
            logical, token_to_req, block_table, out_route, out_mask, stride_logical_row,
            stride_table_row, rows, valid_rows, num_requests, width, table_width, page_div,
            num_slots, mask_bytes_per_row);
  }
  return cudaGetLastError();
}

template <typename IdType>
cudaError_t ExpandBlockRoute(const IdType* block_indices, const IdType* query_positions,
                             const IdType* sequence_lengths, const IdType* token_to_req,
                             IdType* out, uint32_t stride_blocks_row, uint32_t stride_blocks_col,
                             uint32_t stride_out_row, uint32_t stride_out_col, uint32_t rows,
                             uint32_t num_requests, uint32_t block_topk, uint32_t compress_ratio,
                             uint32_t output_width, cudaStream_t stream) {
  if (rows == 0 || output_width == 0) return cudaSuccess;
  const uint32_t wide_blocks = rows * ceil_div(output_width, sparse_route::kWideTile);
  bool wide = true;
  FLASHINFER_CUDA_CALL(sparse_route::choose_wide(
      ExpandBlockRouteKernel<1, true, sparse_route::kWideTile, sparse_route::kWideThreads, IdType>,
      sparse_route::kWideThreads, wide_blocks, &wide));
  const uint32_t tile = wide ? sparse_route::kWideTile : sparse_route::kNarrowTile;
  const dim3 grid(ceil_div(output_width, tile), rows);
  // Both strides are 1 for every caller that hands over a plain route; folding them
  // away turns the inner address into an add.
  const bool contiguous = stride_blocks_col == 1 && stride_out_col == 1;

#define _FI_LAUNCH_CFG(RATIO, CONTIGUOUS, TILE, THREADS)                                          \
  ExpandBlockRouteKernel<RATIO, CONTIGUOUS, TILE, THREADS, IdType><<<grid, THREADS, 0, stream>>>( \
      block_indices, query_positions, sequence_lengths, token_to_req, out, stride_blocks_row,     \
      stride_blocks_col, stride_out_row, stride_out_col, rows, num_requests, block_topk)

#define _FI_LAUNCH(RATIO, CONTIGUOUS)                                                             \
  do {                                                                                            \
    if (wide) {                                                                                   \
      _FI_LAUNCH_CFG(RATIO, CONTIGUOUS, sparse_route::kWideTile, sparse_route::kWideThreads);     \
    } else {                                                                                      \
      _FI_LAUNCH_CFG(RATIO, CONTIGUOUS, sparse_route::kNarrowTile, sparse_route::kNarrowThreads); \
    }                                                                                             \
  } while (0)

#define _FI_DISPATCH_COMPRESS_RATIO(RATIO) \
  case RATIO: {                            \
    if (contiguous) {                      \
      _FI_LAUNCH(RATIO, true);             \
    } else {                               \
      _FI_LAUNCH(RATIO, false);            \
    }                                      \
    break;                                 \
  }

  switch (compress_ratio) {
    _FI_DISPATCH_COMPRESS_RATIO(1)
    _FI_DISPATCH_COMPRESS_RATIO(2)
    _FI_DISPATCH_COMPRESS_RATIO(4)
    _FI_DISPATCH_COMPRESS_RATIO(8)
    _FI_DISPATCH_COMPRESS_RATIO(16)
    _FI_DISPATCH_COMPRESS_RATIO(32)
    default:
      return cudaErrorInvalidValue;
  }
#undef _FI_DISPATCH_COMPRESS_RATIO
#undef _FI_LAUNCH
#undef _FI_LAUNCH_CFG

  return cudaGetLastError();
}

template <typename IdType>
cudaError_t QSARouteFromBlocks(const IdType* block_indices, const IdType* query_positions,
                               const IdType* sequence_lengths, const IdType* token_to_req,
                               const IdType* block_table, IdType* out_logical, IdType* out_route,
                               uint8_t* out_mask, uint32_t stride_blocks_row,
                               uint32_t stride_blocks_col, uint32_t stride_logical_row,
                               uint32_t stride_table_row, uint32_t rows, uint32_t num_requests,
                               uint32_t block_topk, uint32_t table_width, uint32_t page_size,
                               uint32_t num_slots, uint32_t mask_bytes_per_row,
                               uint32_t compress_ratio, cudaStream_t stream) {
  const uint32_t output_width = block_topk * compress_ratio + compress_ratio - 1;
  if (rows == 0 || output_width == 0) return cudaSuccess;
  const uint32_t wide_blocks = rows * ceil_div(output_width, sparse_route::kWideTile);
  bool wide = true;
  FLASHINFER_CUDA_CALL(
      sparse_route::choose_wide(QSARouteFromBlocksKernel<1, true, sparse_route::kWideTile,
                                                         sparse_route::kWideThreads, IdType>,
                                sparse_route::kWideThreads, wide_blocks, &wide));
  const uint32_t tile = wide ? sparse_route::kWideTile : sparse_route::kNarrowTile;
  const dim3 grid(ceil_div(output_width, tile), rows);
  const bool contiguous = stride_blocks_col == 1;

#define _FI_ROUTE_CFG(RATIO, CONTIGUOUS, TILE, THREADS)                                           \
  QSARouteFromBlocksKernel<RATIO, CONTIGUOUS, TILE, THREADS, IdType>                              \
      <<<grid, THREADS, 0, stream>>>(block_indices, query_positions, sequence_lengths,            \
                                     token_to_req, block_table, out_logical, out_route, out_mask, \
                                     stride_blocks_row, stride_blocks_col, stride_logical_row,    \
                                     stride_table_row, rows, num_requests, block_topk,            \
                                     table_width, page_size, num_slots, mask_bytes_per_row)

#define _FI_ROUTE_LAUNCH(RATIO, CONTIGUOUS)                                                      \
  do {                                                                                           \
    if (wide) {                                                                                  \
      _FI_ROUTE_CFG(RATIO, CONTIGUOUS, sparse_route::kWideTile, sparse_route::kWideThreads);     \
    } else {                                                                                     \
      _FI_ROUTE_CFG(RATIO, CONTIGUOUS, sparse_route::kNarrowTile, sparse_route::kNarrowThreads); \
    }                                                                                            \
  } while (0)

#define _FI_ROUTE_DISPATCH(RATIO)     \
  case RATIO: {                       \
    if (contiguous) {                 \
      _FI_ROUTE_LAUNCH(RATIO, true);  \
    } else {                          \
      _FI_ROUTE_LAUNCH(RATIO, false); \
    }                                 \
    break;                            \
  }

  switch (compress_ratio) {
    _FI_ROUTE_DISPATCH(1)
    _FI_ROUTE_DISPATCH(2)
    _FI_ROUTE_DISPATCH(4)
    _FI_ROUTE_DISPATCH(8)
    _FI_ROUTE_DISPATCH(16)
    _FI_ROUTE_DISPATCH(32)
    default:
      return cudaErrorInvalidValue;
  }
#undef _FI_ROUTE_DISPATCH
#undef _FI_ROUTE_LAUNCH
#undef _FI_ROUTE_CFG

  return cudaGetLastError();
}

}  // namespace flashinfer

#endif  // FLASHINFER_SPARSE_ROUTE_CUH_
