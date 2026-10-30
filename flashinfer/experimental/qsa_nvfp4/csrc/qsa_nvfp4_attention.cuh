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
#ifndef FLASHINFER_EXPERIMENTAL_QSA_NVFP4_ATTENTION_CUH_
#define FLASHINFER_EXPERIMENTAL_QSA_NVFP4_ATTENTION_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <flashinfer/cp_async.cuh>
#include <flashinfer/math.cuh>
#include <flashinfer/mma.cuh>
#include <flashinfer/vec_dtypes.cuh>

namespace flashinfer {
namespace qsa_nvfp4 {

// Sparse paged GQA attention over an NVFP4, FP8-e4m3 or BF16 KV cache (head_dim 256), with the
// semantics of vLLM's Triton qsa_sparse_paged_attention: one CTA per (row, KV head, split) walks
// tiles of the row's selection, the attention output is rounded to BF16 before the FP32 output
// gate, and split selections are merged through their log2-domain LSE.
//
// Within a tile, QK splits tokens across the four warps (warp w scores its tokens over all 256
// dims) and PV splits dims (warp w accumulates 64 output dims over all tokens), so a warp holds a
// 16x64 FP32 accumulator rather than 16x256. Each warp copies its own QK tokens' K and V rows, so
// it only waits for its own K copies; V and the softmax exchange share two CTA barriers per tile.
// Lane l of a warp owns that warp's token l: it reads the selection three tiles ahead and the
// block table two tiles ahead, so the copies never wait on index or page lookups.
//
// QK takes BF16 operands; K is decoded exactly to BF16 (E2M1 times its E4M3 block scale, or
// E4M3). PV takes BF16 operands for NVFP4 and BF16, which keeps small probabilities normal, and
// FP16 operands for FP8 like the Triton kernel. Both accumulate in FP32. K is decoded straight
// into MMA B fragments; a dot product and an output column do not care about the order of dims,
// so both are permuted (see qk_dim):
//  * QK: quad lane t only touches dims [64t, 64t + 64) of a token, and Q is staged with the same
//    permutation, already in A-fragment order. FP8 and BF16 rotate each lane's 16-byte reads so
//    the four lanes of a token hit different banks.
//  * PV: column n of warp w's n8 tile i is physical dim 64w + 8n + i, so a lane reads one dim
//    group per token, and its output dims per row end up contiguous: [64w + 16t, +16).
// Row strides of 144 (NVFP4) and 272 (FP8) bytes keep both read patterns free of bank conflicts.
// BF16 rows are unpadded 512 bytes and store 16-byte chunk c of stage token r at c ^ (r % 8)
// instead, so three BF16 CTAs fit in the 228 KB of an SM90 SM.

enum class KvFormat { kNVFP4, kFP8, kBF16 };

constexpr uint32_t kHeadDim = 256;
constexpr uint32_t kNumWarps = 4;
constexpr uint32_t kNumThreads = kNumWarps * 32;
// Merge warps at most. NVFP4 keeps its four; BF16 and FP8 merge up to 65 splits in one pass.
template <KvFormat F>
constexpr uint32_t kMergeWarps = F == KvFormat::kNVFP4 ? 4 : 16;
constexpr int kMergeDepth = 8;            // splits in flight per merge warp
constexpr float kMaskedScore = -1.0e20f;  // the Triton kernel's sentinel

template <KvFormat F>
struct KvTraits;
template <>
struct KvTraits<KvFormat::kNVFP4> {
  static constexpr uint32_t kChunks = kHeadDim / 32 + 1;  // 16-byte copies: E2M1 data, scales
  static constexpr uint32_t kRowBytes = 16 * kChunks;
  static constexpr uint32_t kTileTokens = 64;
  using PvType = nv_bfloat16;
};
template <>
struct KvTraits<KvFormat::kFP8> {
  static constexpr uint32_t kChunks = kHeadDim / 16;
  static constexpr uint32_t kRowBytes = 16 * kChunks + 16;
  static constexpr uint32_t kTileTokens = 64;
  using PvType = half;
};
template <>
struct KvTraits<KvFormat::kBF16> {
  static constexpr uint32_t kChunks = kHeadDim / 8;
  static constexpr uint32_t kRowBytes = 16 * kChunks;
  // Two 64-token stages would exceed the shared memory of SM86 and SM89.
  static constexpr uint32_t kTileTokens = 32;
  using PvType = nv_bfloat16;
};

struct Params {
  const nv_bfloat16* q;
  const uint8_t* k_data;
  const uint8_t* v_data;
  const uint8_t* k_sf;  // NVFP4 only
  const uint8_t* v_sf;
  const int32_t* indices;
  const int32_t* block_table;
  const int32_t* token_to_req;
  const nv_bfloat16* gate;  // contiguous [num_rows, num_qo_heads, kHeadDim]
  nv_bfloat16* out;
  float* partial_out;  // [num_splits, num_rows, num_qo_heads, kHeadDim], split > 1 only
  float* partial_lse;  // [num_splits, num_rows, num_qo_heads], split > 1 only
  int64_t q_stride_row, q_stride_head;
  int64_t data_stride_block, data_stride_token, data_stride_head;  // bytes, shared by K and V
  int64_t sf_stride_block, sf_stride_token, sf_stride_head;        // bytes, shared by K and V
  int64_t indices_stride_row, table_stride_req;
  int64_t out_stride_row, out_stride_head;
  float score_scale;   // log2(e) / sqrt(head_dim) * K scale
  float output_scale;  // V scale
  int32_t num_rows, num_requests, num_blocks, page_size, table_width;
  int32_t topk, group_size, num_qo_heads, num_tiles, num_splits;
};

template <KvFormat F>
struct SharedStorage {
  using Traits = KvTraits<F>;
  alignas(16) uint8_t k[2][Traits::kTileTokens][Traits::kRowBytes];  // [stage][token]
  alignas(16) uint8_t v[2][Traits::kTileTokens][Traits::kRowBytes];
  alignas(16) uint4 q[kHeadDim / 16][32];             // Q A fragments by k-step and lane
  alignas(16) uint4 p[Traits::kTileTokens / 16][32];  // P A fragments by k-step and lane
  float row_max[kNumWarps][16];
  float row_sum[kNumWarps][16];
};

// First of the four physical dims that quad lane t feeds k-step s of the QK MMA.
template <KvFormat F>
__device__ __forceinline__ uint32_t qk_dim(uint32_t s, uint32_t t) {
  if constexpr (F == KvFormat::kNVFP4) {
    return 64 * t + 4 * s;
  } else if constexpr (F == KvFormat::kFP8) {
    return 64 * t + 16 * ((s / 4 + 2 * (t / 2)) % 4) + 4 * (s % 4);
  } else {
    return 64 * t + 8 * ((s / 2 + 2 * t) % 8) + 4 * (s % 2);
  }
}

// Byte offset of 16-byte chunk c in a K/V row of stage token r. Only r % 8 matters, so a
// warp-local token index works too: each warp owns whole 8-token groups.
template <KvFormat F>
__device__ __forceinline__ uint32_t chunk_offset(uint32_t c, uint32_t r) {
  return 16 * (F == KvFormat::kBF16 ? c ^ (r % 8) : c);
}

__device__ __forceinline__ uint32_t mul_bf16x2(uint32_t a, uint32_t b) {
  __nv_bfloat162 r =
      __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}

// Broadcast 16-bit element i (0..3) of a decoded uint2 to both halves of a 16x2 register.
__device__ __forceinline__ uint32_t broadcast16(uint2 v, uint32_t i) {
  return __byte_perm(i < 2 ? v.x : v.y, 0u, i % 2 ? 0x3232u : 0x1010u);
}

__device__ __forceinline__ float sigmoid(float x) {
  return math::ptx_rcp(1.f + math::ptx_exp2(-x * math::log2e));
}

template <typename T>
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  if constexpr (std::is_same_v<T, half>) {
    __half2 r = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&r);
  } else {
    __nv_bfloat162 r = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&r);
  }
}

__device__ __forceinline__ void mma_m16n8k16_bf16(float* c, const uint4& a, uint32_t b0,
                                                  uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
      "{%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

// Copy this warp's rows of one tile (K or V). Lane l holds the byte offset of token l and bit l
// of `valid` says whether it exists; invalid rows are zero-filled, never read: a masked score
// times a non-finite row would still poison the accumulator.
template <KvFormat F>
__device__ __forceinline__ void copy_rows(uint8_t* rows, const uint8_t* data, const uint8_t* sf,
                                          int64_t data_offset, int64_t sf_offset, uint32_t valid,
                                          uint32_t lane) {
  using Traits = KvTraits<F>;
  constexpr uint32_t kCopies = Traits::kTileTokens / kNumWarps * Traits::kChunks;
#pragma unroll
  for (uint32_t e0 = 0; e0 < kCopies; e0 += 32) {
    const uint32_t e = e0 + lane, token = e / Traits::kChunks, chunk = e % Traits::kChunks;
    const uint8_t* src = data + __shfl_sync(0xffffffffu, data_offset, token) + 16 * chunk;
    if constexpr (F == KvFormat::kNVFP4) {
      const int64_t token_sf = __shfl_sync(0xffffffffu, sf_offset, token);
      if (chunk == Traits::kChunks - 1) src = sf + token_sf;
    }
    if (kCopies % 32 == 0 || e < kCopies) {
      cp_async::pred_load_128b<cp_async::PrefetchMode::kNoPrefetch,
                               cp_async::SharedMemFillMode::kFillZero>(
          rows + token * Traits::kRowBytes + chunk_offset<F>(chunk, token), src,
          (valid >> token) & 1u);
    }
  }
}

// NVFP4 copy sources of one tile: lane l copies 16-byte column l % 8 of warp tokens l / 8 + 4i,
// and lanes below 16 the scales of token l. Gathered once per tile, shared by K and V.
struct Nvfp4Sources {
  int64_t data[4];
  int64_t sf;
  uint32_t valid;  // bit i: data[i], bit 4: sf
};

__device__ __forceinline__ Nvfp4Sources gather_nvfp4(int64_t data_offset, int64_t sf_offset,
                                                     uint32_t valid, uint32_t lane) {
  Nvfp4Sources s;
  s.valid = 0;
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    const uint32_t token = lane / 8 + 4 * i;
    s.data[i] = __shfl_sync(0xffffffffu, data_offset, token) + 16 * (lane % 8);
    s.valid |= ((valid >> token) & 1u) << i;
  }
  s.sf = __shfl_sync(0xffffffffu, sf_offset, lane % 16);
  s.valid |= (lane < 16 ? (valid >> lane) & 1u : 0u) << 4;
  return s;
}

// cp.async of 16 bytes, zero-filled when !pred, with the L2 evict-first policy: KV rows are read
// once.
__device__ __forceinline__ void pred_load_128b_evict_first(void* smem, const void* gmem,
                                                           bool pred) {
  const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n" ::"r"(dst),
               "l"(gmem), "r"(pred ? 16 : 0), "l"(0x12F0000000000000ull));
}

__device__ __forceinline__ void copy_nvfp4(uint8_t* rows, const uint8_t* data, const uint8_t* sf,
                                           const Nvfp4Sources& s, uint32_t lane) {
  constexpr uint32_t kRowBytes = KvTraits<KvFormat::kNVFP4>::kRowBytes;
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    pred_load_128b_evict_first(rows + (lane / 8 + 4 * i) * kRowBytes + 16 * (lane % 8),
                               data + s.data[i], (s.valid >> i) & 1u);
  }
  if (lane < 16) {
    pred_load_128b_evict_first(rows + lane * kRowBytes + kHeadDim / 2, sf + s.sf,
                               (s.valid >> 4) & 1u);
  }
}

// E2M1 x8 to BF16 times 2^120, and E4M3 x4 block scales to BF16 times 2^-120: both exact, and so
// is their product, which skips the bias multiply of the E4M3 conversion.
__device__ __forceinline__ void e2m1x8_to_bf16x8_x2p120(uint32_t packed, uint32_t* d) {
  constexpr uint32_t kHi0 = 0x7B7B7B00u, kHi1 = 0x7C7C7C7Cu;
  constexpr uint32_t kLo0 = 0xC0800000u, kLo1 = 0xC0804000u;
  const uint32_t mag = packed & 0x77777777u;
  uint32_t hi_a = __byte_perm(kHi0, kHi1, mag);
  uint32_t hi_b = __byte_perm(kHi0, kHi1, mag >> 16);
  const uint32_t lo_a = __byte_perm(kLo0, kLo1, mag);
  const uint32_t lo_b = __byte_perm(kLo0, kLo1, mag >> 16);
  hi_a |= __byte_perm(packed << 4, packed, 0xD9C8u) & 0x80808080u;
  hi_b |= __byte_perm(packed << 4, packed, 0xFBEAu) & 0x80808080u;
  d[0] = __byte_perm(lo_a, hi_a, 0x5140u);
  d[1] = __byte_perm(lo_a, hi_a, 0x7362u);
  d[2] = __byte_perm(lo_b, hi_b, 0x5140u);
  d[3] = __byte_perm(lo_b, hi_b, 0x7362u);
}

// The same decode with nibbles 0, 2 / 1, 3 / 4, 6 / 5, 7 paired: given two tokens' bytes
// interleaved, each output holds one dim of both tokens, a PV B half.
__device__ __forceinline__ void e2m1x8_pairs_to_bf16x8_x2p120(uint32_t packed, uint32_t* d) {
  constexpr uint32_t kHi0 = 0x7B7B7B00u, kHi1 = 0x7C7C7C7Cu;
  constexpr uint32_t kLo0 = 0xC0800000u, kLo1 = 0xC0804000u;
  const uint32_t mag = packed & 0x77777777u;
  uint32_t hi_a = __byte_perm(kHi0, kHi1, mag);
  uint32_t hi_b = __byte_perm(kHi0, kHi1, mag >> 16);
  const uint32_t lo_a = __byte_perm(kLo0, kLo1, mag);
  const uint32_t lo_b = __byte_perm(kLo0, kLo1, mag >> 16);
  hi_a |= __byte_perm(packed << 4, packed, 0xD9C8u) & 0x80808080u;
  hi_b |= __byte_perm(packed << 4, packed, 0xFBEAu) & 0x80808080u;
  d[0] = __byte_perm(lo_a, hi_a, 0x6240u);
  d[1] = __byte_perm(lo_a, hi_a, 0x7351u);
  d[2] = __byte_perm(lo_b, hi_b, 0x6240u);
  d[3] = __byte_perm(lo_b, hi_b, 0x7351u);
}

__device__ __forceinline__ uint2 e4m3x4_to_bf16x4_x2m120(uint32_t q) {
  q = __byte_perm(q, q, 0x1302u);
  return make_uint2((q & 0x80008000u) | ((q & 0x7F007F00u) >> 4),
                    ((q << 8) & 0x80008000u) | (((q << 8) & 0x7F007F00u) >> 4));
}

template <KvFormat F>
__global__ void __launch_bounds__(kNumThreads) SparsePagedAttentionKernel(const Params p) {
  using Traits = KvTraits<F>;
  using PvType = typename Traits::PvType;
  constexpr uint32_t kTileTokens = Traits::kTileTokens, kRowBytes = Traits::kRowBytes;
  constexpr uint32_t kWarpTokens = kTileTokens / kNumWarps;  // QK tokens per warp: 16 or 8
  constexpr uint32_t kQkTiles = kWarpTokens / 8;             // n8 tiles per warp
  extern __shared__ __align__(16) uint8_t smem_buffer[];
  SharedStorage<F>& smem = *reinterpret_cast<SharedStorage<F>*>(smem_buffer);
  const int row = blockIdx.x, kv_head = blockIdx.y, split = blockIdx.z;
  const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
  const int stride = p.num_splits;
  const int64_t indices_row = int64_t(row) * p.indices_stride_row;

  // Lane l < kWarpTokens owns this warp's token l of every tile.
  const int owned_column = int(kWarpTokens * warp + lane);
  auto load_index = [&](int tile) {
    const int column = tile * int(kTileTokens) + owned_column;
    return lane < kWarpTokens && tile < p.num_tiles && column < p.topk
               ? p.indices[indices_row + column]
               : -1;
  };
  // Tiles past the row's valid count are read here but never copied.
  int index0 = load_index(split), index1 = load_index(split + stride);
  int index2 = load_index(split + 2 * stride), index3 = load_index(split + 3 * stride);
  const int request = p.token_to_req[row];
  // The trailing column holds the row's valid-entry count, never a token index.
  const int valid_count = p.indices[indices_row + p.topk];
  const bool request_ok = request >= 0 && request < p.num_requests;
  const int64_t table_row = int64_t(min(max(request, 0), p.num_requests - 1)) * p.table_stride_req;
  const int first_head = kv_head * p.group_size;

  // Q in A-fragment order: k-step s, lane (g, t) holds rows g and g + 8 at dims qk_dim(s, t) .. +3.
  for (uint32_t e = threadIdx.x; e < kHeadDim / 16 * 32; e += kNumThreads) {
    const uint32_t s = e / 32, eg = e % 32 / 4, et = e % 4;
    const nv_bfloat16* src = p.q + int64_t(row) * p.q_stride_row +
                             (first_head + int(eg)) * p.q_stride_head + qk_dim<F>(s, et);
    const uint2 lo =
        int(eg) < p.group_size ? *reinterpret_cast<const uint2*>(src) : make_uint2(0, 0);
    const uint2 hi = int(eg) + 8 < p.group_size
                         ? *reinterpret_cast<const uint2*>(src + 8 * p.q_stride_head)
                         : make_uint2(0, 0);
    smem.q[s][e % 32] = make_uint4(lo.x, hi.x, lo.y, hi.y);
  }

  // Selection index -> (physical page or -1, page offset), then -> byte offsets and valid bits.
  // NVFP4: index / page_size as (umulhi(index, mul) + index) >> shift, exact for 0 <= index < 2^31.
  uint32_t page_shift = 0, page_mul = 0;
  if constexpr (F == KvFormat::kNVFP4) {
    page_shift = 32 - __clz(uint32_t(p.page_size) - 1);
    page_mul = uint32_t(((uint64_t(1) << page_shift) - uint32_t(p.page_size)) *
                            (uint64_t(1) << 32) / uint32_t(p.page_size) +
                        1);
  }
  auto lookup = [&](int index, int& page, int& offset) {
    const int safe = max(index, 0);
    const int logical_page =
        F == KvFormat::kNVFP4
            ? int((__umulhi(uint32_t(safe), page_mul) + uint32_t(safe)) >> page_shift)
            : safe / p.page_size;
    offset = safe - logical_page * p.page_size;
    page = request_ok && index >= 0 && logical_page < p.table_width
               ? p.block_table[table_row + logical_page]
               : -1;
  };
  auto locate = [&](int page, int offset, int64_t& data_offset, int64_t& sf_offset) {
    const bool valid = page >= 0 && page < p.num_blocks;
    const int64_t block = valid ? page : 0;
    data_offset =
        block * p.data_stride_block + offset * p.data_stride_token + kv_head * p.data_stride_head;
    if constexpr (F == KvFormat::kNVFP4) {
      sf_offset =
          block * p.sf_stride_block + offset * p.sf_stride_token + kv_head * p.sf_stride_head;
    }
    return __ballot_sync(0xffffffffu, valid);
  };
  const int tile_end =
      min(p.num_tiles, (min(valid_count, p.topk) + int(kTileTokens) - 1) / int(kTileTokens));
  int page, offset, next_page, next_offset;
  int64_t data_offset, sf_offset = 0;
  uint8_t* const k_rows = &smem.k[0][kWarpTokens * warp][0];
  uint8_t* const v_rows = &smem.v[0][kWarpTokens * warp][0];
  constexpr uint32_t kStageBytes = kTileTokens * kRowBytes;
  Nvfp4Sources src, next_src;
  auto copy_tile = [&](uint8_t* rows, const uint8_t* data, const uint8_t* sf, uint32_t tile_valid,
                       const Nvfp4Sources& sources) {
    if constexpr (F == KvFormat::kNVFP4) {
      copy_nvfp4(rows, data, sf, sources, lane);
    } else {
      copy_rows<F>(rows, data, sf, data_offset, sf_offset, tile_valid, lane);
    }
  };
  lookup(index0, page, offset);
  lookup(index1, next_page, next_offset);
  uint32_t valid = locate(page, offset, data_offset, sf_offset);
  if constexpr (F == KvFormat::kNVFP4) src = gather_nvfp4(data_offset, sf_offset, valid, lane);
  if (split < tile_end) {
    copy_tile(k_rows, p.k_data, p.k_sf, valid, src);
  }
  cp_async::commit_group();
  if (split < tile_end) {
    copy_tile(v_rows, p.v_data, p.v_sf, valid, src);
  }
  cp_async::commit_group();
  uint32_t next_valid = locate(next_page, next_offset, data_offset, sf_offset);
  if constexpr (F == KvFormat::kNVFP4)
    next_src = gather_nvfp4(data_offset, sf_offset, next_valid, lane);
  lookup(index2, next_page, next_offset);
  int ahead_index = index3;

  float o[4][8] = {};  // n16 accumulators: tiles 2c (o[c][0..3]) and 2c + 1 (o[c][4..7])
  float m[2] = {kMaskedScore, kMaskedScore}, l[2] = {0.f, 0.f};  // rows g and g + 8
  __syncthreads();                                               // Q

  uint32_t stage = 0;
  for (int tile = split; tile < tile_end; tile += stride, stage ^= 1) {
    const bool has_next = tile + stride < tile_end;
    // This warp's K rows are its own copies; V needs everyone's, published by the P barrier.
    cp_async::wait_group<1>();
    __syncwarp();
    if (has_next) {
      copy_tile(k_rows + (stage ^ 1) * kStageBytes, p.k_data, p.k_sf, next_valid, next_src);
    }
    cp_async::commit_group();

    // QK, s[4 * i + j]: row g + 8 * (j / 2), warp-local token 8i + 2t + j % 2.
    float s[4 * kQkTiles] = {};
    const uint8_t* k0 = k_rows + stage * kStageBytes + g * kRowBytes;
    if constexpr (F == KvFormat::kNVFP4) {
      const uint8_t* k1 = k0 + 8 * kRowBytes;
      uint2 sf0, sf1;  // block scales 4t .. 4t + 3 of each token
      sf0 = e4m3x4_to_bf16x4_x2m120(*reinterpret_cast<const uint32_t*>(k0 + kHeadDim / 2 + 4 * t));
      sf1 = e4m3x4_to_bf16x4_x2m120(*reinterpret_cast<const uint32_t*>(k1 + kHeadDim / 2 + 4 * t));
#pragma unroll
      for (uint32_t h = 0; h < 2; ++h) {
        const uint4 raw0 = *reinterpret_cast<const uint4*>(k0 + 32 * t + 16 * h);
        const uint4 raw1 = *reinterpret_cast<const uint4*>(k1 + 32 * t + 16 * h);
        const uint32_t w0[4] = {raw0.x, raw0.y, raw0.z, raw0.w};
        const uint32_t w1[4] = {raw1.x, raw1.y, raw1.z, raw1.w};
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
          const uint32_t pair = 4 * h + i;  // k-steps 2 * pair and 2 * pair + 1
          uint32_t d0[4], d1[4];
          e2m1x8_to_bf16x8_x2p120(w0[i], d0);
          e2m1x8_to_bf16x8_x2p120(w1[i], d1);
          const uint32_t scale0 = broadcast16(sf0, pair / 2), scale1 = broadcast16(sf1, pair / 2);
#pragma unroll
          for (uint32_t k = 0; k < 4; ++k) {
            d0[k] = mul_bf16x2(d0[k], scale0);
            d1[k] = mul_bf16x2(d1[k], scale1);
          }
          uint4 a = smem.q[2 * pair][lane];
          uint32_t b[4] = {d0[0], d0[1], d1[0], d1[1]};
          mma::mma_sync_m16n16k16_row_col_f16f16f32<nv_bfloat16>(s, &a.x, b);
          a = smem.q[2 * pair + 1][lane];
          uint32_t b2[4] = {d0[2], d0[3], d1[2], d1[3]};
          mma::mma_sync_m16n16k16_row_col_f16f16f32<nv_bfloat16>(s, &a.x, b2);
        }
      }
    } else if constexpr (F == KvFormat::kFP8) {
      const uint8_t* k1 = k0 + 8 * kRowBytes;
#pragma unroll
      for (uint32_t c = 0; c < 4; ++c) {  // k-steps 4c .. 4c + 3
        const uint32_t dim = qk_dim<F>(4 * c, t);
        const uint4 raw0 = *reinterpret_cast<const uint4*>(k0 + dim);
        const uint4 raw1 = *reinterpret_cast<const uint4*>(k1 + dim);
        uint32_t w0[4] = {raw0.x, raw0.y, raw0.z, raw0.w};
        uint32_t w1[4] = {raw1.x, raw1.y, raw1.z, raw1.w};
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
          uint2 d0, d1;
          fast_dequant_f8f16x4<__nv_fp8_e4m3, nv_bfloat16>(&w0[i], &d0);
          fast_dequant_f8f16x4<__nv_fp8_e4m3, nv_bfloat16>(&w1[i], &d1);
          uint4 a = smem.q[4 * c + i][lane];
          uint32_t b[4] = {d0.x, d0.y, d1.x, d1.y};
          mma::mma_sync_m16n16k16_row_col_f16f16f32<nv_bfloat16>(s, &a.x, b);
        }
      }
    } else {
#pragma unroll
      for (uint32_t c = 0; c < 8; ++c) {  // k-steps 2c and 2c + 1
        const uint4 raw =
            *reinterpret_cast<const uint4*>(k0 + chunk_offset<F>(qk_dim<F>(2 * c, t) / 8, g));
        mma_m16n8k16_bf16(s, smem.q[2 * c][lane], raw.x, raw.y);
        mma_m16n8k16_bf16(s, smem.q[2 * c + 1][lane], raw.z, raw.w);
      }
    }

    bool ok[4 * kQkTiles];
    float row_max[2] = {kMaskedScore, kMaskedScore};
#pragma unroll
    for (uint32_t j = 0; j < 4 * kQkTiles; ++j) {
      ok[j] = (valid >> (8 * (j / 4) + 2 * t + j % 2)) & 1u;
      s[j] = ok[j] ? s[j] * p.score_scale : kMaskedScore;
      row_max[(j / 2) % 2] = fmaxf(row_max[(j / 2) % 2], s[j]);
    }
#pragma unroll
    for (uint32_t r = 0; r < 2; ++r) {
      row_max[r] = fmaxf(row_max[r], math::shfl_xor_sync(row_max[r], 1));
      row_max[r] = fmaxf(row_max[r], math::shfl_xor_sync(row_max[r], 2));
      if (t == 0) smem.row_max[warp][g + 8 * r] = row_max[r];
    }
    // Also retires every warp's previous PV, which frees the other V stage.
    __syncthreads();
    if (has_next) {
      copy_tile(v_rows + (stage ^ 1) * kStageBytes, p.v_data, p.v_sf, next_valid, next_src);
    }
    cp_async::commit_group();

    float m_new[2], row_sum[2] = {0.f, 0.f};
#pragma unroll
    for (uint32_t r = 0; r < 2; ++r) {
      m_new[r] = m[r];
#pragma unroll
      for (uint32_t w = 0; w < kNumWarps; ++w)
        m_new[r] = fmaxf(m_new[r], smem.row_max[w][g + 8 * r]);
    }
#pragma unroll
    for (uint32_t j = 0; j < 4 * kQkTiles; ++j) {
      s[j] = ok[j] ? math::ptx_exp2(s[j] - m_new[(j / 2) % 2]) : 0.f;
      row_sum[(j / 2) % 2] += s[j];
    }
#pragma unroll
    for (uint32_t r = 0; r < 2; ++r) {
      row_sum[r] += math::shfl_xor_sync(row_sum[r], 1);
      row_sum[r] += math::shfl_xor_sync(row_sum[r], 2);
      if (t == 0) smem.row_sum[warp][g + 8 * r] = row_sum[r];
    }
    // The S accumulator of each n8 tile is half of a PV A fragment: k-step token // 16, low or
    // high eight tokens.
#pragma unroll
    for (uint32_t i = 0; i < kQkTiles; ++i) {
      const uint32_t token = kWarpTokens * warp + 8 * i;
      reinterpret_cast<uint2*>(&smem.p[token / 16][lane])[token / 8 % 2] = make_uint2(
          pack2<PvType>(s[4 * i], s[4 * i + 1]), pack2<PvType>(s[4 * i + 2], s[4 * i + 3]));
    }
    cp_async::wait_group<2>();
    __syncthreads();

#pragma unroll
    for (uint32_t r = 0; r < 2; ++r) {
      const float alpha = math::ptx_exp2(m[r] - m_new[r]);
      float tile_sum = 0.f;
#pragma unroll
      for (uint32_t w = 0; w < kNumWarps; ++w) tile_sum += smem.row_sum[w][g + 8 * r];
      l[r] = l[r] * alpha + tile_sum;
      m[r] = m_new[r];
#pragma unroll
      for (uint32_t c = 0; c < 4; ++c) {
        o[c][2 * r] *= alpha;
        o[c][2 * r + 1] *= alpha;
        o[c][2 * r + 4] *= alpha;
        o[c][2 * r + 5] *= alpha;
      }
    }

    // PV: k-step kk is tokens 16kk .. 16kk + 15; this lane needs tokens 2t, 2t + 1, 2t + 8, 2t + 9
    // at physical dims 64 * warp + 8g .. + 7.
    const uint8_t* v_tile = &smem.v[stage][0][0];
#pragma unroll
    for (uint32_t kk = 0; kk < kTileTokens / 16; ++kk) {
      uint4 a = smem.p[kk][lane];
      uint32_t d[4][4];
      if constexpr (F == KvFormat::kNVFP4) {
        // d[2p + h][k]: dim 4h + k of tokens j = 2p and 2p + 1.
        uint32_t w[4], sf_packed = 0;
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
          const uint8_t* v_row = v_tile + (16 * kk + 2 * t + j % 2 + 8 * (j / 2)) * kRowBytes;
          w[j] = *reinterpret_cast<const uint32_t*>(v_row + 32 * warp + 4 * g);
          sf_packed |= uint32_t(v_row[kHeadDim / 2 + 4 * warp + g / 2]) << (8 * j);
        }
        const uint2 sf = e4m3x4_to_bf16x4_x2m120(sf_packed);
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
          e2m1x8_pairs_to_bf16x8_x2p120(
              __byte_perm(w[i / 2 * 2], w[i / 2 * 2 + 1], i % 2 ? 0x7362u : 0x5140u), d[i]);
#pragma unroll
          for (uint32_t k = 0; k < 4; ++k) d[i][k] = mul_bf16x2(d[i][k], i < 2 ? sf.x : sf.y);
        }
      } else {
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
          const uint32_t token = 16 * kk + 2 * t + j % 2 + 8 * (j / 2);
          const uint8_t* v_row = v_tile + token * kRowBytes;
          if constexpr (F == KvFormat::kFP8) {
            uint2 raw = *reinterpret_cast<const uint2*>(v_row + 64 * warp + 8 * g);
            fast_dequant_f8f16x4<__nv_fp8_e4m3, half>(&raw.x, reinterpret_cast<uint2*>(d[j]));
            fast_dequant_f8f16x4<__nv_fp8_e4m3, half>(&raw.y, reinterpret_cast<uint2*>(d[j] + 2));
          } else {
            const uint4 raw =
                *reinterpret_cast<const uint4*>(v_row + chunk_offset<F>(8 * warp + g, token));
            d[j][0] = raw.x, d[j][1] = raw.y, d[j][2] = raw.z, d[j][3] = raw.w;
          }
        }
      }
#pragma unroll
      for (uint32_t c = 0; c < 4; ++c) {
        // Pair one dim across two tokens: n8 tile 2c takes the low halves, 2c + 1 the high.
        uint32_t b[4];
        if constexpr (F == KvFormat::kNVFP4) {
          const uint32_t h = c / 2, k = 2 * (c % 2);
          b[0] = d[h][k], b[1] = d[2 + h][k], b[2] = d[h][k + 1], b[3] = d[2 + h][k + 1];
        } else {
          b[0] = __byte_perm(d[0][c], d[1][c], 0x5410u),
          b[1] = __byte_perm(d[2][c], d[3][c], 0x5410u);
          b[2] = __byte_perm(d[0][c], d[1][c], 0x7632u),
          b[3] = __byte_perm(d[2][c], d[3][c], 0x7632u);
        }
        mma::mma_sync_m16n16k16_row_col_f16f16f32<PvType>(o[c], &a.x, b);
      }
    }

    // Advance the lookups: offsets for the tile after next, pages and indices further ahead.
    valid = next_valid;
    next_valid = locate(next_page, next_offset, data_offset, sf_offset);
    if constexpr (F == KvFormat::kNVFP4)
      next_src = gather_nvfp4(data_offset, sf_offset, next_valid, lane);
    lookup(ahead_index, next_page, next_offset);
    ahead_index = load_index(tile + 4 * stride);
  }

  const uint32_t dim0 = 64 * warp + 16 * t;
#pragma unroll
  for (uint32_t r = 0; r < 2; ++r) {
    const int head = int(g + 8 * r);
    if (head >= p.group_size) continue;
    const bool has_values = l[r] > 0.f;
    const float inv = p.output_scale / fmaxf(l[r], 1.0e-20f);
    float v[16];
#pragma unroll
    for (uint32_t c = 0; c < 4; ++c) {
      v[2 * c] = o[c][2 * r];
      v[2 * c + 1] = o[c][2 * r + 4];
      v[8 + 2 * c] = o[c][2 * r + 1];
      v[9 + 2 * c] = o[c][2 * r + 5];
    }
#pragma unroll
    for (uint32_t j = 0; j < 16; ++j) v[j] = has_values ? v[j] * inv : 0.f;
    const int qo_head = first_head + head;
    if (p.num_splits == 1) {
      // Round to BF16 like the unfused path, then apply the gate in FP32.
      const nv_bfloat16* gate =
          p.gate + (int64_t(row) * p.num_qo_heads + qo_head) * kHeadDim + dim0;
      nv_bfloat16* out =
          p.out + int64_t(row) * p.out_stride_row + qo_head * p.out_stride_head + dim0;
#pragma unroll
      for (uint32_t h = 0; h < 2; ++h) {
        uint4 gate8 = reinterpret_cast<const uint4*>(gate)[h], out8;
        const nv_bfloat16* gv = reinterpret_cast<const nv_bfloat16*>(&gate8);
        nv_bfloat16* ov = reinterpret_cast<nv_bfloat16*>(&out8);
#pragma unroll
        for (uint32_t j = 0; j < 8; ++j) {
          const float attn = __bfloat162float(__float2bfloat16(v[8 * h + j]));
          ov[j] = __float2bfloat16(attn * sigmoid(__bfloat162float(gv[j])));
        }
        reinterpret_cast<uint4*>(out)[h] = out8;
      }
    } else {
      const int64_t partial_row = (int64_t(split) * p.num_rows + row) * p.num_qo_heads + qo_head;
      float4* partial = reinterpret_cast<float4*>(p.partial_out + partial_row * kHeadDim + dim0);
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        partial[j] = make_float4(v[4 * j], v[4 * j + 1], v[4 * j + 2], v[4 * j + 3]);
      }
      if (warp == 0 && t == 0) {
        p.partial_lse[partial_row] = has_values ? m[r] + math::ptx_log2(l[r]) : -math::inf;
      }
    }
  }
}

// One CTA per (row, query head): LSE-weighted sum of the split outputs, BF16 rounding, FP32 gate.
// Each warp merges a contiguous chunk of the splits, kMergeDepth at a time, issuing a group's
// loads before using any of them, and lane l owns dims [8l, 8l + 8); the warps' sums are then
// rescaled to a common max. The launch gives one warp per kMergeDepth splits, at most kMergeWarps.
template <KvFormat F>
__global__ void __launch_bounds__(32 * kMergeWarps<F>) MergeKernel(const Params p) {
  const int row = blockIdx.x, head = blockIdx.y;
  const uint32_t num_warps = blockDim.x / 32, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const uint32_t dim = 8 * lane;
  const int64_t row_head = int64_t(row) * p.num_qo_heads + head;
  const int64_t split_pitch = int64_t(p.num_rows) * p.num_qo_heads;
  const int chunk = (p.num_splits + int(num_warps) - 1) / int(num_warps);
  const int first = int(warp) * chunk, last = min(first + chunk, p.num_splits);
  uint4 gate8 = make_uint4(0, 0, 0, 0);
  if (warp == 0) gate8 = *reinterpret_cast<const uint4*>(p.gate + row_head * kHeadDim + dim);

  float m = -math::inf, den = 0.f, acc[8] = {};
  for (int base = first; base < last; base += kMergeDepth) {
    float lse[kMergeDepth];
    float4 part[kMergeDepth][2];
#pragma unroll
    for (int k = 0; k < kMergeDepth; ++k) {
      const int64_t split_row = (base + k) * split_pitch + row_head;
      const float4* src =
          reinterpret_cast<const float4*>(p.partial_out + split_row * kHeadDim + dim);
      const bool ok = base + k < last;
      lse[k] = ok ? p.partial_lse[split_row] : -math::inf;
      part[k][0] = ok ? src[0] : make_float4(0.f, 0.f, 0.f, 0.f);
      part[k][1] = ok ? src[1] : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float next_m = m;
#pragma unroll
    for (int k = 0; k < kMergeDepth; ++k) next_m = fmaxf(next_m, lse[k]);
    if (next_m == -math::inf) continue;  // no values yet
    const float alpha = math::ptx_exp2(m - next_m);
    den *= alpha;
#pragma unroll
    for (uint32_t j = 0; j < 8; ++j) acc[j] *= alpha;
    m = next_m;
#pragma unroll
    for (int k = 0; k < kMergeDepth; ++k) {
      const float w = math::ptx_exp2(lse[k] - m);
      const float* v = reinterpret_cast<const float*>(part[k]);
      den += w;
#pragma unroll
      for (uint32_t j = 0; j < 8; ++j) acc[j] += w * v[j];
    }
  }

  __shared__ float warp_m[kMergeWarps<F>], warp_den[kMergeWarps<F>];
  float4(*warp_acc)[kHeadDim / 4];
  if constexpr (F == KvFormat::kNVFP4) {
    __shared__ float4 acc[kMergeWarps<F>][kHeadDim / 4];
    warp_acc = acc;
  } else {
    // Sized by the launch to the warps it runs, so few-split merges keep their occupancy.
    extern __shared__ __align__(16) uint8_t smem_buffer[];
    warp_acc = reinterpret_cast<float4(*)[kHeadDim / 4]>(smem_buffer);
  }
  if (lane == 0) {
    warp_m[warp] = m;
    warp_den[warp] = den;
  }
  warp_acc[warp][2 * lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
  warp_acc[warp][2 * lane + 1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
  __syncthreads();
  if (warp != 0) return;

  float max_m = -math::inf;
  for (uint32_t w = 0; w < num_warps; ++w) max_m = fmaxf(max_m, warp_m[w]);
  float total = 0.f, merged[8] = {};
  if (max_m > -math::inf) {
    for (uint32_t w = 0; w < num_warps; ++w) {
      const float scale = math::ptx_exp2(warp_m[w] - max_m);
      const float* v = reinterpret_cast<const float*>(&warp_acc[w][2 * lane]);
      total += scale * warp_den[w];
#pragma unroll
      for (uint32_t j = 0; j < 8; ++j) merged[j] += scale * v[j];
    }
  }
  const float inv = total > 0.f ? 1.f / total : 0.f;
  const nv_bfloat16* gv = reinterpret_cast<const nv_bfloat16*>(&gate8);
  uint4 out8;
  nv_bfloat16* ov = reinterpret_cast<nv_bfloat16*>(&out8);
#pragma unroll
  for (uint32_t j = 0; j < 8; ++j) {
    const float attn = __bfloat162float(__float2bfloat16(merged[j] * inv));
    ov[j] = __float2bfloat16(attn * sigmoid(__bfloat162float(gv[j])));
  }
  *reinterpret_cast<uint4*>(p.out + int64_t(row) * p.out_stride_row + head * p.out_stride_head +
                            dim) = out8;
}

template <KvFormat F>
inline cudaError_t PrepareKernel() {
  constexpr int kSmemBytes = sizeof(SharedStorage<F>);
  return cudaFuncSetAttribute(SparsePagedAttentionKernel<F>,
                              cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
}

// Resident CTAs per SM, for sizing splits to the device.
template <KvFormat F>
inline cudaError_t CtasPerSm(int* num_ctas) {
  cudaError_t status = PrepareKernel<F>();
  if (status != cudaSuccess) return status;
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(num_ctas, SparsePagedAttentionKernel<F>,
                                                       kNumThreads, sizeof(SharedStorage<F>));
}

template <KvFormat F>
inline cudaError_t SparsePagedAttention(const Params& p, cudaStream_t stream) {
  cudaError_t status = PrepareKernel<F>();
  if (status != cudaSuccess) return status;
  const dim3 grid(p.num_rows, p.num_qo_heads / p.group_size, p.num_splits);
  SparsePagedAttentionKernel<F><<<grid, kNumThreads, sizeof(SharedStorage<F>), stream>>>(p);
  if (p.num_splits > 1) {
    const int warps = min(int(kMergeWarps<F>), (p.num_splits + kMergeDepth - 1) / kMergeDepth);
    const size_t acc_bytes = F == KvFormat::kNVFP4 ? 0 : warps * kHeadDim * sizeof(float);
    MergeKernel<F><<<dim3(p.num_rows, p.num_qo_heads), 32 * warps, acc_bytes, stream>>>(p);
  }
  return cudaGetLastError();
}

}  // namespace qsa_nvfp4
}  // namespace flashinfer

#endif  // FLASHINFER_EXPERIMENTAL_QSA_NVFP4_ATTENTION_CUH_
