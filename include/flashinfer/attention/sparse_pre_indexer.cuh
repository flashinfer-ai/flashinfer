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
#ifndef FLASHINFER_ATTENTION_SPARSE_PRE_INDEXER_CUH_
#define FLASHINFER_ATTENTION_SPARSE_PRE_INDEXER_CUH_

#include <cuda_runtime.h>

#include <cstdint>

#include "../utils.cuh"

// What a sparse-route indexer reads has to be built before it can be scored:
// the query rows normalised and rotated, and the key rows pooled into the
// compression the route addresses. This does both in one launch.
//
// Every query row is RMS-normalised and rotated into the form the scorer takes.
// Every completed compression group is pooled from its raw keys -- which may
// straddle the chunk boundary into the per-request ring -- normalised, rotated
// and written to the compressed cache; the first work item of each request also
// commits that request's raw-key suffix back to the ring.
//
// A warp owns one row of head_dim. That is the whole shape of this file: the
// row is 128 or 256 elements, a lane holds four of them, and the only thing
// that crosses lanes is the norm's sum -- the rotation's partner is a quarter
// row away, which this layout puts in the same lane.

namespace flashinfer {

namespace sparse_pre_indexer {

constexpr int kWarp = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kBlock = kWarp * kWarpsPerBlock;
// A row is spread over one warp, so a lane holds head_dim/32 of it.
template <int D>
constexpr int kPer = D / kWarp;
// Rotating pairs a lane owns: the row's first quarter, spread over the warp.
template <int D>
constexpr int kPairs = D / (4 * kWarp);

// A cosine and its sine, adjacent in the pair-major table, read as one access.
template <typename T>
struct __align__(4) CosSin {
  T c, s;
};

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, offset);
  }
  return v;
}

template <typename T>
__device__ __forceinline__ float to_float(T x);
template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162float(x);
}
template <>
__device__ __forceinline__ float to_float<half>(half x) {
  return __half2float(x);
}

template <typename T>
__device__ __forceinline__ T from_float(float x);
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}
template <>
__device__ __forceinline__ half from_float<half>(float x) {
  return __float2half(x);
}

// A value as the cache will hold it. The pooled group and the normalised row
// both round here so that a fused result matches an unfused one bit for bit.
template <typename T>
__device__ __forceinline__ float round_to(float v) {
  return to_float<T>(from_float<T>(v));
}

/*!
 * \brief Which rotary axis each pair a lane owns belongs to.
 *
 * The choice is a function of the pair index alone, and a lane's pair indices
 * never change, so this is settled once for the warp rather than again for
 * every token it goes on to handle.
 */
template <int D, bool MROPE>
__device__ __forceinline__ void lane_axes(int lane, int mrope_h, int mrope_w,
                                          int (&axis)[kPairs<D>]) {
#pragma unroll
  for (int r = 0; r < kPairs<D>; ++r) {
    if constexpr (!MROPE) {
      axis[r] = 0;
    } else {
      const int p = lane + r * kWarp;
      const int mod = p % 3;
      axis[r] = (mod == 1 && p <= 3 * mrope_h) ? 1 : (mod == 2 && p <= 3 * mrope_w) ? 2 : 0;
    }
  }
}

/*!
 * \brief The rotary factors a lane needs, read once for a position.
 *
 * A lane holds the elements thirty-two apart starting at its own index, so the
 * pairs it rotates are both in its registers and the factors it needs are one
 * per rotating pair. Across the warp those factors are contiguous, and they are
 * the same for every head of a token.
 *
 * The table is pair-major: a row is cosine and sine interleaved, so a pair is
 * four bytes in one place rather than two halves of the row apart. The rotary
 * axes sit on this kernel's dependency chain -- a position is read, then its
 * factors -- so halving the reads on it is worth a table of its own.
 */
template <int D, bool MROPE, typename DType>
__device__ __forceinline__ void load_cos_sin(const DType* __restrict__ cos_sin, int lane,
                                             int64_t pos_t, int64_t pos_h, int64_t pos_w,
                                             const int (&axis)[kPairs<D>], float (&c)[kPairs<D>],
                                             float (&s)[kPairs<D>]) {
  constexpr int kHalf = D / 2;
  constexpr int kP = kPairs<D>;

  if constexpr (!MROPE) {
    const CosSin<DType>* row = reinterpret_cast<const CosSin<DType>*>(cos_sin + pos_t * kHalf);
#pragma unroll
    for (int r = 0; r < kP; ++r) {
      const CosSin<DType> cs = row[lane + r * kWarp];
      c[r] = to_float<DType>(cs.c);
      s[r] = to_float<DType>(cs.s);
    }
    return;
  }
  // Interleaved temporal, height and width pairs; the three axes index the same
  // table. Selected, not indexed: an array the compiler cannot fold an index
  // for lands in local memory, and this is the query path's inner step.
#pragma unroll
  for (int r = 0; r < kP; ++r) {
    const int p = lane + r * kWarp;
    const int64_t pos = axis[r] == 1 ? pos_h : axis[r] == 2 ? pos_w : pos_t;
    const CosSin<DType> cs = reinterpret_cast<const CosSin<DType>*>(cos_sin + pos * kHalf)[p];
    c[r] = to_float<DType>(cs.c);
    s[r] = to_float<DType>(cs.s);
  }
}

template <int D>
__device__ __forceinline__ float row_sumsq(const float (&v)[kPer<D>]) {
  float sum = 0.f;
#pragma unroll
  for (int j = 0; j < kPer<D>; ++j) sum += v[j] * v[j];
  return sum;
}

/*!
 * \brief Finish the RMSNorm and apply partial NeoX RoPE to one row.
 *
 * The row's sum of squares arrives already reduced, so a caller holding several
 * rows can put all their reductions in flight before finishing any of them --
 * a reduction is a chain of shuffles, and one row at a time leaves the warp
 * waiting on that chain with nothing else to issue.
 *
 * Lane L holds elements L, L+32, L+64 and so on. The first half of the row
 * rotates and the second half passes through; an element pairs with the one a
 * quarter-row away, which this layout puts in the same lane.
 */
template <int D, typename DType>
__device__ __forceinline__ void apply_norm_rope(float (&v)[kPer<D>], float sum,
                                                const float (&weight)[kPer<D>],
                                                const float (&c)[kPairs<D>],
                                                const float (&s)[kPairs<D>], float eps) {
  constexpr int kN = kPer<D>;
  constexpr int kP = kPairs<D>;
  const float rrms = rsqrtf(sum / static_cast<float>(D) + eps);
  // The result rounds to the cache dtype before the rotation reads it.
#pragma unroll
  for (int j = 0; j < kN; ++j) v[j] = round_to<DType>(v[j] * rrms * weight[j]);
#pragma unroll
  for (int r = 0; r < kP; ++r) {
    const float low = v[r];
    const float high = v[r + kP];
    v[r] = low * c[r] - high * s[r];
    v[r + kP] = high * c[r] + low * s[r];
  }
}

template <int D, typename DType>
__device__ __forceinline__ void norm_rope(float (&v)[kPer<D>], const float (&weight)[kPer<D>],
                                          const float (&c)[kPairs<D>], const float (&s)[kPairs<D>],
                                          float eps) {
  apply_norm_rope<D, DType>(v, warp_sum(row_sumsq<D>(v)), weight, c, s, eps);
}

// A lane's elements are strided by the warp width, which keeps every one of the
// row's loads coalesced across the warp and the rotating pairs in registers.
template <int D, typename DType>
__device__ __forceinline__ void load_row(float (&v)[kPer<D>], const DType* __restrict__ src,
                                         int lane, bool valid) {
#pragma unroll
  for (int j = 0; j < kPer<D>; ++j) {
    v[j] = valid ? to_float<DType>(src[lane + j * kWarp]) : 0.f;
  }
}

template <int D, typename DType>
__device__ __forceinline__ void store_row(DType* __restrict__ dst, const float (&v)[kPer<D>],
                                          int lane) {
#pragma unroll
  for (int j = 0; j < kPer<D>; ++j) dst[lane + j * kWarp] = from_float<DType>(v[j]);
}

// The stored weight is the offset from one, as the Gemma checkpoints keep it.
template <int D, typename DType>
__device__ __forceinline__ void load_weight(float (&w)[kPer<D>], const DType* __restrict__ src,
                                            int lane) {
#pragma unroll
  for (int j = 0; j < kPer<D>; ++j) {
    w[j] = to_float<DType>(src[lane + j * kWarp]) + 1.f;
  }
}

}  // namespace sparse_pre_indexer

/*!
 * \brief Everything one launch of the pre-indexer reads and writes.
 *
 * The three shifts stand in for divisions that are otherwise sixty-four bit and
 * runtime; a negative one means that divisor is not a power of two and the
 * launch takes the general path.
 */
template <typename DType>
struct QSAPreIndexerParams {
  const DType* q;
  int64_t q_stride_token;
  const DType* k;
  int64_t k_stride_token;
  const int64_t* positions;
  int64_t pos_stride_axis;
  int64_t pos_stride_token;
  const DType* cos_sin;
  const DType* q_norm_weight;
  const DType* k_norm_weight;
  float eps;
  DType* q_out;
  int64_t q_out_stride_token;
  int64_t q_out_stride_head;
  DType* state_cache;
  int64_t state_stride_block;
  int64_t state_stride_token;
  const int64_t* state_slots;
  const int32_t* state_table;
  int64_t state_table_stride_req;
  const int32_t* query_start_loc;
  const int64_t* logical_positions;
  const int64_t* compressed_slots;
  const int32_t* work_metadata;
  DType* compressed_cache;
  int64_t compressed_stride_block;
  int64_t compressed_stride_token;
  int32_t num_tokens;
  int32_t num_state_blocks;
  int32_t num_compressed_blocks;
  int32_t num_k_work;
  int32_t num_q_heads;
  int32_t compress_ratio;
  int32_t state_size;
  int32_t comp_page_size;
  int32_t mrope_h;
  int32_t mrope_w;
  int32_t k_blocks;
  int32_t ratio_shift;
  int32_t state_shift;
  int32_t comp_shift;
  float inv_ratio;
};

namespace sparse_pre_indexer {

/*!
 * \tparam D head dimension
 * \tparam MROPE_Q whether the query's rotary axes are chosen per pair
 * \tparam MROPE_K same for the compressed key
 * \tparam POS_2D whether the position tensor carries three axes
 * \tparam CACHE_POS whether the ring stores each row's rotary coordinates
 * \tparam POW2 whether the compression ratio, the ring and the compressed page
 *   are all powers of two, which is what turns their divisions into shifts
 */
template <int D, bool MROPE_Q, bool MROPE_K, bool POS_2D, bool CACHE_POS, bool POW2, typename DType>
__global__ void __launch_bounds__(kBlock) QSAPreIndexerKernel(QSAPreIndexerParams<DType> a) {
  const int lane = threadIdx.x % kWarp;
  const int warp = threadIdx.x / kWarp;
  const int block = static_cast<int>(blockIdx.x);

  // A lane's slice of the norm weight is the same for every row it will ever
  // see, so it is read once.
  float weight[kPer<D>];
  load_weight<D, DType>(weight, a.q_norm_weight, lane);
  int q_axis[kPairs<D>];
  lane_axes<D, MROPE_Q>(lane, a.mrope_h, a.mrope_w, q_axis);

  if (block >= a.k_blocks) {
    // Two tokens per warp. Their eight rows are all in flight before any is
    // normalised, which is what keeps a warp with something to issue while a
    // row is still arriving; one row at a time leaves the warp waiting.
    constexpr int kTokens = 2;
    const int token0 = ((block - a.k_blocks) * kWarpsPerBlock + warp) * kTokens;
    if (token0 >= a.num_tokens) return;
    // Only the last warp of a step can run past the end, so the rest are spared
    // carrying a validity predicate through every load and address.
    const bool tail = token0 + kTokens > a.num_tokens;

    float c[kTokens][kPairs<D>], sn[kTokens][kPairs<D>];
#pragma unroll
    for (int t = 0; t < kTokens; ++t) {
      const int token = token0 + t;
      const int safe = tail && token >= a.num_tokens ? token0 : token;
      const int64_t pos_t = a.positions[safe * a.pos_stride_token];
      int64_t pos_h = pos_t, pos_w = pos_t;
      if constexpr (POS_2D) {
        pos_h = a.positions[a.pos_stride_axis + safe * a.pos_stride_token];
        pos_w = a.positions[2 * a.pos_stride_axis + safe * a.pos_stride_token];
      }
      load_cos_sin<D, MROPE_Q, DType>(a.cos_sin, lane, pos_t, pos_h, pos_w, q_axis, c[t], sn[t]);
    }

    constexpr int kHeadGroup = 4;
    for (int h0 = 0; h0 < a.num_q_heads; h0 += kHeadGroup) {
      const int n = min(kHeadGroup, a.num_q_heads - h0);
      float v[kTokens][kHeadGroup][kPer<D>];
#pragma unroll
      for (int t = 0; t < kTokens; ++t) {
        const int token = token0 + t;
        const int safe = tail && token >= a.num_tokens ? token0 : token;
        const DType* q_row = a.q + static_cast<int64_t>(safe) * a.q_stride_token;
#pragma unroll
        for (int i = 0; i < kHeadGroup; ++i) {
          load_row<D, DType>(v[t][i], q_row + static_cast<int64_t>(h0 + i) * D, lane, i < n);
        }
      }
#pragma unroll
      for (int t = 0; t < kTokens; ++t) {
#pragma unroll
        for (int i = 0; i < kHeadGroup; ++i) {
          // Predicated, not broken out of: a runtime trip count would stop the
          // unroll and put the staged rows in local memory.
          if (i < n && (!tail || token0 + t < a.num_tokens)) {
            norm_rope<D, DType>(v[t][i], weight, c[t], sn[t], a.eps);
            store_row<D, DType>(a.q_out + static_cast<int64_t>(token0 + t) * a.q_out_stride_token +
                                    static_cast<int64_t>(h0 + i) * a.q_out_stride_head,
                                v[t][i], lane);
          }
        }
      }
    }
    return;
  }

  // Compression work. One warp owns one completed group.
  const int pid = block * kWarpsPerBlock + warp;
  if (pid >= a.num_k_work) return;
  const int32_t request = a.work_metadata[2 * pid];
  const int32_t work_in_request = a.work_metadata[2 * pid + 1];
  if (request < 0) return;

  const int32_t query_start = a.query_start_loc[request];
  const int32_t query_end = a.query_start_loc[request + 1];
  const int32_t query_len = query_end - query_start;
  // A request with no tokens has nothing to pool and no last token to read the
  // chunk's end from; query_end - 1 would index behind the tensor.
  if (query_len <= 0) return;
  const int64_t chunk_end = a.logical_positions[query_end - 1];
  const int64_t chunk_start = chunk_end - query_len + 1;
  const int32_t ratio = a.compress_ratio;
  auto div_ratio = [&](int64_t v) -> int64_t { return POW2 ? (v >> a.ratio_shift) : v / ratio; };
  const int32_t num_groups =
      static_cast<int32_t>(div_ratio(chunk_end + 1) - div_ratio(chunk_start));

  if (work_in_request < num_groups) {
    const int64_t first_boundary = div_ratio(chunk_start + ratio) * ratio - 1;
    const int64_t end_position = first_boundary + static_cast<int64_t>(work_in_request) * ratio;
    const int64_t boundary_token = query_start + end_position - chunk_start;
    const bool valid_token = boundary_token >= query_start && boundary_token < query_end &&
                             boundary_token < a.num_tokens;
    const int64_t compressed_slot = valid_token ? a.compressed_slots[boundary_token] : -1;
    const bool valid =
        valid_token && compressed_slot >= 0 &&
        compressed_slot < static_cast<int64_t>(a.num_compressed_blocks) * a.comp_page_size;
    const int32_t state_block =
        a.state_table[static_cast<int64_t>(request) * a.state_table_stride_req];
    const bool state_block_valid = state_block >= 0 && state_block < a.num_state_blocks;
    const int64_t safe_state_block = state_block > 0 ? state_block : 0;

    float acc[kPer<D>] = {};
    // The group's source rows are fetched together. One at a time leaves a
    // single row in flight and the warp waits out every one of them in turn,
    // which is what the compression half spent its time on.
    constexpr int kGroupBatch = 4;
    for (int32_t g0 = 0; g0 < ratio; g0 += kGroupBatch) {
      float rows[kGroupBatch][kPer<D>];
#pragma unroll
      for (int i = 0; i < kGroupBatch; ++i) {
        const int32_t g = g0 + i;
        const int64_t source_position = end_position - (ratio - 1) + g;
        const bool in_chunk = source_position >= chunk_start;
        const int64_t source_token = query_start + source_position - chunk_start;
        const bool token_valid =
            source_token >= query_start && source_token < query_end && source_token < a.num_tokens;
        // Only the first completed group can reach behind the chunk, into the
        // ring the previous step left.
        const DType* src = in_chunk ? a.k + (source_token > 0 ? source_token : 0) * a.k_stride_token
                                    : a.state_cache + safe_state_block * a.state_stride_block +
                                          (POW2 ? (source_position & (a.state_size - 1))
                                                : source_position % a.state_size) *
                                              a.state_stride_token;
        const bool live = g < ratio && valid && (in_chunk ? token_valid : state_block_valid);
        load_row<D, DType>(rows[i], src, lane, live);
      }
#pragma unroll
      for (int i = 0; i < kGroupBatch; ++i) {
#pragma unroll
        for (int j = 0; j < kPer<D>; ++j) acc[j] += rows[i][j];
      }
    }
    // An unfused caller pools in the cache dtype before the norm sees it.
#pragma unroll
    for (int j = 0; j < kPer<D>; ++j) {
      acc[j] = round_to<DType>(acc[j] * a.inv_ratio);
    }

    const int64_t first_position = end_position - (ratio - 1);
    int64_t pos_t = first_position, pos_h = first_position, pos_w = first_position;
    if constexpr (CACHE_POS) {
      // The rotation uses the first token of the pooled group, whose exact
      // coordinates are either in this chunk or in the ring beside its row.
      const bool first_in_chunk = first_position >= chunk_start;
      const int64_t first_token = query_start + first_position - chunk_start;
      const bool first_token_valid =
          first_token >= query_start && first_token < query_end && first_token < a.num_tokens;
      if (first_in_chunk && first_token_valid) {
        pos_t = a.positions[first_token * a.pos_stride_token];
        if constexpr (POS_2D) {
          pos_h = a.positions[a.pos_stride_axis + first_token * a.pos_stride_token];
          pos_w = a.positions[2 * a.pos_stride_axis + first_token * a.pos_stride_token];
        } else {
          pos_h = pos_t;
          pos_w = pos_t;
        }
      } else if (!first_in_chunk && state_block_valid) {
        const int64_t* tail = reinterpret_cast<const int64_t*>(
            a.state_cache + safe_state_block * a.state_stride_block +
            (POW2 ? (first_position & (a.state_size - 1)) : first_position % a.state_size) *
                a.state_stride_token +
            D);
        pos_t = tail[0];
        pos_h = tail[1];
        pos_w = tail[2];
      } else {
        pos_t = 0;
        pos_h = 0;
        pos_w = 0;
      }
    }
    float kw[kPer<D>];
    load_weight<D, DType>(kw, a.k_norm_weight, lane);
    int k_axis[kPairs<D>];
    lane_axes<D, MROPE_K>(lane, a.mrope_h, a.mrope_w, k_axis);
    float kc[kPairs<D>], ks[kPairs<D>];
    load_cos_sin<D, MROPE_K, DType>(a.cos_sin, lane, pos_t, pos_h, pos_w, k_axis, kc, ks);
    norm_rope<D, DType>(acc, kw, kc, ks, a.eps);
    if (valid) {
      const int64_t comp_block =
          POW2 ? (compressed_slot >> a.comp_shift) : compressed_slot / a.comp_page_size;
      const int64_t comp_row =
          POW2 ? (compressed_slot & (a.comp_page_size - 1)) : compressed_slot % a.comp_page_size;
      store_row<D, DType>(a.compressed_cache + comp_block * a.compressed_stride_block +
                              comp_row * a.compressed_stride_token,
                          acc, lane);
    }
  }

  if (work_in_request == 0) {
    // This warp may have just read history out of the ring it is about to
    // overwrite, so it finishes those reads first.
    __syncwarp();
    const int32_t rows = min(query_len, a.state_size);
    for (int32_t offset = 0; offset < rows; ++offset) {
      const int32_t token = query_end - rows + offset;
      const bool token_valid = token >= query_start && token < query_end && token < a.num_tokens;
      const int64_t slot = token_valid ? a.state_slots[token] : -1;
      const bool live = token_valid && slot >= 0 &&
                        slot < static_cast<int64_t>(a.num_state_blocks) * a.state_size;
      if (!live) continue;
      const int64_t ring_block = POW2 ? (slot >> a.state_shift) : slot / a.state_size;
      const int64_t ring_row = POW2 ? (slot & (a.state_size - 1)) : slot % a.state_size;
      DType* row =
          a.state_cache + ring_block * a.state_stride_block + ring_row * a.state_stride_token;
      float v[kPer<D>];
      load_row<D, DType>(v, a.k + static_cast<int64_t>(token) * a.k_stride_token, lane, true);
      store_row<D, DType>(row, v, lane);
      if constexpr (CACHE_POS) {
        if (lane == 0) {
          int64_t* tail = reinterpret_cast<int64_t*>(row + D);
          const int64_t p_t = a.positions[token * a.pos_stride_token];
          tail[0] = p_t;
          if constexpr (POS_2D) {
            tail[1] = a.positions[a.pos_stride_axis + token * a.pos_stride_token];
            tail[2] = a.positions[2 * a.pos_stride_axis + token * a.pos_stride_token];
          } else {
            tail[1] = p_t;
            tail[2] = p_t;
          }
        }
      }
    }
  }
}

}  // namespace sparse_pre_indexer

/*!
 * \brief Build the query and compressed-key rows a sparse route is scored on.
 *
 * \tparam HEAD_DIM 128 or 256
 * \param mrope_k whether the compressed key picks its rotary axis per pair
 * \param pos_2d whether \p params.positions carries three axes
 * \param cache_pos whether the ring keeps each row's rotary coordinates after
 *   its head_dim elements
 *
 * The query picks its axis per pair exactly when the positions carry three of
 * them, so a two-dimensional position tensor with a single-axis key is not a
 * configuration this builds; it returns cudaErrorInvalidValue.
 *
 * The norm accumulates in float32, so a row whose RMS passes about 1.6e18
 * squares to infinity and comes out zero rather than normalised. That is far
 * outside anything an activation reaches, but it is silent, so it is stated
 * rather than left to be discovered.
 */
template <uint32_t HEAD_DIM, typename DType>
cudaError_t QSAPreIndexer(QSAPreIndexerParams<DType> params, bool mrope_k, bool pos_2d,
                          bool cache_pos, cudaStream_t stream = nullptr) {
  using namespace sparse_pre_indexer;
  static_assert(HEAD_DIM == 128 || HEAD_DIM == 256,
                "the pre-indexer builds a head dimension of 128 or 256");
  if (params.num_tokens <= 0) return cudaSuccess;
  if (params.compress_ratio <= 0) return cudaErrorInvalidValue;

  const bool pow2 = params.ratio_shift >= 0 && params.state_shift >= 0 && params.comp_shift >= 0;
  // Two tokens to a warp, so a block covers twice its warps.
  constexpr int kTokensPerBlock = 2 * kWarpsPerBlock;
  const int64_t q_blocks = (params.num_tokens + kTokensPerBlock - 1) / kTokensPerBlock;
  const int64_t k_blocks = (params.num_k_work + kWarpsPerBlock - 1) / kWarpsPerBlock;
  params.k_blocks = static_cast<int32_t>(k_blocks);
  const dim3 grid(static_cast<unsigned>(k_blocks + q_blocks));

  cudaError_t status = cudaErrorInvalidValue;
  auto launch = [&](auto mrope_q_tag, auto mrope_k_tag, auto pos_2d_tag, auto cache_pos_tag) {
    if (pow2) {
      QSAPreIndexerKernel<HEAD_DIM, decltype(mrope_q_tag)::value, decltype(mrope_k_tag)::value,
                          decltype(pos_2d_tag)::value, decltype(cache_pos_tag)::value, true, DType>
          <<<grid, kBlock, 0, stream>>>(params);
    } else {
      QSAPreIndexerKernel<HEAD_DIM, decltype(mrope_q_tag)::value, decltype(mrope_k_tag)::value,
                          decltype(pos_2d_tag)::value, decltype(cache_pos_tag)::value, false, DType>
          <<<grid, kBlock, 0, stream>>>(params);
    }
    status = cudaGetLastError();
  };

#define FI_QSA_PRE_INDEXER_CASE(MQ, MK, P2, CP)                                     \
  if (pos_2d == (P2) && mrope_k == (MK) && cache_pos == (CP)) {                     \
    launch(std::integral_constant<bool, MQ>{}, std::integral_constant<bool, MK>{},  \
           std::integral_constant<bool, P2>{}, std::integral_constant<bool, CP>{}); \
    return status;                                                                  \
  }
  FI_QSA_PRE_INDEXER_CASE(true, true, true, true)
  FI_QSA_PRE_INDEXER_CASE(true, true, true, false)
  FI_QSA_PRE_INDEXER_CASE(false, true, false, true)
  FI_QSA_PRE_INDEXER_CASE(false, true, false, false)
  FI_QSA_PRE_INDEXER_CASE(false, false, false, true)
  FI_QSA_PRE_INDEXER_CASE(false, false, false, false)
#undef FI_QSA_PRE_INDEXER_CASE
  return cudaErrorInvalidValue;
}

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_SPARSE_PRE_INDEXER_CUH_
