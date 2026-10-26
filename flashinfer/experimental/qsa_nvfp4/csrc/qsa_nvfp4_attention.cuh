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

#include <cstdint>
#include <flashinfer/cp_async.cuh>
#include <flashinfer/math.cuh>
#include <flashinfer/mma.cuh>
#include <flashinfer/vec_dtypes.cuh>

namespace flashinfer {
namespace qsa_nvfp4 {

// Sparse paged GQA attention over an NVFP4 KV cache (head_dim 256), with the semantics of vLLM's
// Triton qsa_sparse_paged_attention: one CTA per (row, KV head, split) walks 64-token tiles of the
// row's selection, the attention output is rounded to BF16 before the FP32 output gate, and split
// selections are merged through their log2-domain LSE.
//
// Within a tile, QK splits tokens across the four warps (warp w scores tokens 16w..16w+15 over
// all 256 dims) and PV splits dims (warp w accumulates 64 output dims over all 64 tokens), so a
// warp holds a 16x64 FP32 accumulator rather than 16x256.
//
// Both MMAs take BF16 operands and accumulate in FP32: E2M1 times an E4M3 block scale is exact in
// BF16, and BF16 keeps small probabilities normal where FP16 would round them to subnormals.
// NVFP4 is decoded straight into MMA B fragments. A dot product and an output column do not care
// about the order of dims, so both are permuted:
//  * QK: column 2t + 8a + b of k-step s reads physical dim 64t + 4s + 2a + b. Quad lane t then only
//    touches dims [64t, 64t + 64), and one packed uint32 covers two k-steps of one token. Q is
//    staged with the same permutation, already in A-fragment order.
//  * PV: column n of warp w's n8 tile i is physical dim 64w + 8n + i. A lane reads one packed
//    uint32 per token, and its output dims per row end up contiguous: [64w + 16t, +16).

constexpr uint32_t kHeadDim = 256;
constexpr uint32_t kPackedBytes = kHeadDim / 2;             // E2M1 bytes per (token, head)
constexpr uint32_t kScaleBytes = kHeadDim / 16;             // E4M3 block scales per (token, head)
constexpr uint32_t kRowBytes = kPackedBytes + kScaleBytes;  // 144: bank-conflict free for both
constexpr uint32_t kTileTokens = 64;                        // the QK and PV read patterns
constexpr uint32_t kNumWarps = 4;
constexpr uint32_t kNumThreads = kNumWarps * 32;
constexpr uint32_t kMergeThreads = kHeadDim / 4;
constexpr float kMaskedScore = -1.0e20f;  // the Triton kernel's sentinel

struct Params {
  const nv_bfloat16* q;
  const uint8_t* k_data;
  const uint8_t* v_data;
  const uint8_t* k_sf;
  const uint8_t* v_sf;
  const int32_t* indices;
  const int32_t* block_table;
  const int32_t* token_to_req;
  const nv_bfloat16* gate;  // contiguous [num_rows, num_qo_heads, kHeadDim]
  nv_bfloat16* out;
  float* partial_out;  // [num_splits, num_rows, num_qo_heads, kHeadDim], split > 1 only
  float* partial_lse;  // [num_splits, num_rows, num_qo_heads], split > 1 only
  int64_t q_stride_row, q_stride_head;
  int64_t data_stride_block, data_stride_token, data_stride_head;  // shared by K and V
  int64_t sf_stride_block, sf_stride_token, sf_stride_head;        // shared by K and V
  int64_t indices_stride_row, table_stride_req;
  int64_t out_stride_row, out_stride_head;
  float score_scale;   // log2(e) / sqrt(head_dim) * K global scale
  float output_scale;  // V global scale
  int32_t num_rows, num_requests, num_blocks, page_size, table_width;
  int32_t topk, group_size, num_qo_heads, num_tiles, num_splits;
};

struct SharedStorage {
  alignas(16) uint8_t kv[2][2][kTileTokens][kRowBytes];  // [stage][K, V][token][data | scales]
  alignas(16) uint4 q[kHeadDim / 16][32];                // Q A fragments by k-step and lane
  alignas(16) uint4 p[kNumWarps][32];                    // P A fragments; k-step w from warp w
  float row_max[kNumWarps][16];
  float row_sum[kNumWarps][16];
  uint32_t valid[2][2];  // per stage: valid bits of tokens 0..31 and 32..63
};

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

__device__ __forceinline__ uint32_t f32x2_to_bf16x2(float lo, float hi) {
  __nv_bfloat162 r = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&r);
}

// Thread j copies token j % 64 of the tile's K (j < 64) or V row, data and block scales, and the
// K-side warps publish which tokens are valid. Invalid tokens are zero-filled, never read: a
// masked score times a non-finite row would still poison the accumulator.
__device__ __forceinline__ void issue_tile(const Params& p, SharedStorage& smem, uint32_t stage,
                                           int tile, int row, int kv_head, bool request_ok,
                                           int safe_request) {
  const uint32_t token = threadIdx.x % kTileTokens;
  const bool is_v = threadIdx.x >= kTileTokens;
  const int column = tile * int(kTileTokens) + int(token);
  const int logical =
      column < p.topk ? p.indices[int64_t(row) * p.indices_stride_row + column] : -1;
  const int safe_token = max(logical, 0);
  const int logical_page = safe_token / p.page_size;
  bool valid = request_ok && logical >= 0 && logical_page < p.table_width;
  const int physical_page = valid ? p.block_table[int64_t(safe_request) * p.table_stride_req +
                                                  min(logical_page, p.table_width - 1)]
                                  : -1;
  valid = valid && physical_page >= 0 && physical_page < p.num_blocks;
  const int64_t block = max(physical_page, 0);
  const int64_t offset = safe_token % p.page_size;
  const uint8_t* data = (is_v ? p.v_data : p.k_data) + block * p.data_stride_block +
                        offset * p.data_stride_token + kv_head * p.data_stride_head;
  const uint8_t* sf = (is_v ? p.v_sf : p.k_sf) + block * p.sf_stride_block +
                      offset * p.sf_stride_token + kv_head * p.sf_stride_head;
  uint8_t* dst = smem.kv[stage][is_v][token];
#pragma unroll
  for (uint32_t c = 0; c < kPackedBytes / 16; ++c) {
    cp_async::pred_load_128b<cp_async::PrefetchMode::kNoPrefetch,
                             cp_async::SharedMemFillMode::kFillZero>(dst + 16 * c, data + 16 * c,
                                                                     valid);
  }
  cp_async::pred_load_128b<cp_async::PrefetchMode::kNoPrefetch,
                           cp_async::SharedMemFillMode::kFillZero>(dst + kPackedBytes, sf, valid);
  const uint32_t bits = __ballot_sync(0xffffffffu, valid);
  if (threadIdx.x < kTileTokens && threadIdx.x % 32 == 0) {
    smem.valid[stage][threadIdx.x / 32] = bits;
  }
}

__global__ void __launch_bounds__(kNumThreads) SparsePagedAttentionKernel(const Params p) {
  __shared__ SharedStorage smem;
  const int row = blockIdx.x, kv_head = blockIdx.y, split = blockIdx.z;
  const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
  const int request = p.token_to_req[row];
  const bool request_ok = request >= 0 && request < p.num_requests;
  const int safe_request = min(max(request, 0), p.num_requests - 1);
  // The trailing column holds the row's valid-entry count, never a token index.
  const int valid_count = p.indices[int64_t(row) * p.indices_stride_row + p.topk];
  const int tile_end =
      min(p.num_tiles, (min(valid_count, p.topk) + int(kTileTokens) - 1) / int(kTileTokens));
  const int first_head = kv_head * p.group_size;

  // Q in A-fragment order: k-step s, lane (g, t) holds rows g and g + 8 at dims 64t + 4s .. +3.
  for (uint32_t e = threadIdx.x; e < kHeadDim / 16 * 32; e += kNumThreads) {
    const uint32_t s = e / 32, eg = e % 32 / 4, et = e % 4;
    const nv_bfloat16* src = p.q + int64_t(row) * p.q_stride_row +
                             (first_head + int(eg)) * p.q_stride_head + 64 * et + 4 * s;
    const uint2 lo =
        int(eg) < p.group_size ? *reinterpret_cast<const uint2*>(src) : make_uint2(0, 0);
    const uint2 hi = int(eg) + 8 < p.group_size
                         ? *reinterpret_cast<const uint2*>(src + 8 * p.q_stride_head)
                         : make_uint2(0, 0);
    smem.q[s][e % 32] = make_uint4(lo.x, hi.x, lo.y, hi.y);
  }

  float o[4][8] = {};  // n16 accumulators: tiles 2c (o[c][0..3]) and 2c + 1 (o[c][4..7])
  float m[2] = {kMaskedScore, kMaskedScore}, l[2] = {0.f, 0.f};  // rows g and g + 8

  if (split < tile_end) issue_tile(p, smem, 0, split, row, kv_head, request_ok, safe_request);
  cp_async::commit_group();
  uint32_t stage = 0;
  for (int tile = split; tile < tile_end; tile += p.num_splits, stage ^= 1) {
    // wait_group only completes this thread's copies; the barrier publishes everyone's, and also
    // retires the previous tile's PV, which frees the other stage and the P/exchange buffers.
    cp_async::wait_group<0>();
    __syncthreads();
    if (tile + p.num_splits < tile_end) {
      issue_tile(p, smem, stage ^ 1, tile + p.num_splits, row, kv_head, request_ok, safe_request);
    }
    cp_async::commit_group();

    // QK: this warp's tokens g and g + 8 are the two n8 tiles of one n16 MMA.
    const uint8_t* k0 = smem.kv[stage][0][16 * warp + g];
    const uint8_t* k1 = smem.kv[stage][0][16 * warp + g + 8];
    uint2 sf0, sf1;  // block scales 4t .. 4t + 3 of each token
    uint32_t sf_bytes = *reinterpret_cast<const uint32_t*>(k0 + kPackedBytes + 4 * t);
    fast_dequant_f8f16x4<__nv_fp8_e4m3, nv_bfloat16>(&sf_bytes, &sf0);
    sf_bytes = *reinterpret_cast<const uint32_t*>(k1 + kPackedBytes + 4 * t);
    fast_dequant_f8f16x4<__nv_fp8_e4m3, nv_bfloat16>(&sf_bytes, &sf1);
    float s[8] = {};
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
        e2m1x8_to_bf16x8(w0[i], d0);
        e2m1x8_to_bf16x8(w1[i], d1);
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

    // s[j]: row g + 8 * ((j / 2) % 2), warp-local token 8 * (j / 4) + 2t + j % 2.
    const uint32_t valid_bits = smem.valid[stage][warp / 2] >> (16 * (warp % 2));
    bool ok[8];
    float row_max[2] = {kMaskedScore, kMaskedScore};
#pragma unroll
    for (uint32_t j = 0; j < 8; ++j) {
      ok[j] = (valid_bits >> (8 * (j / 4) + 2 * t + j % 2)) & 1u;
      s[j] = ok[j] ? s[j] * p.score_scale : kMaskedScore;
      row_max[(j / 2) % 2] = fmaxf(row_max[(j / 2) % 2], s[j]);
    }
#pragma unroll
    for (uint32_t r = 0; r < 2; ++r) {
      row_max[r] = fmaxf(row_max[r], math::shfl_xor_sync(row_max[r], 1));
      row_max[r] = fmaxf(row_max[r], math::shfl_xor_sync(row_max[r], 2));
      if (t == 0) smem.row_max[warp][g + 8 * r] = row_max[r];
    }
    __syncthreads();

    float m_new[2], row_sum[2] = {0.f, 0.f};
#pragma unroll
    for (uint32_t r = 0; r < 2; ++r) {
      m_new[r] = m[r];
#pragma unroll
      for (uint32_t w = 0; w < kNumWarps; ++w)
        m_new[r] = fmaxf(m_new[r], smem.row_max[w][g + 8 * r]);
    }
#pragma unroll
    for (uint32_t j = 0; j < 8; ++j) {
      s[j] = ok[j] ? math::ptx_exp2(s[j] - m_new[(j / 2) % 2]) : 0.f;
      row_sum[(j / 2) % 2] += s[j];
    }
#pragma unroll
    for (uint32_t r = 0; r < 2; ++r) {
      row_sum[r] += math::shfl_xor_sync(row_sum[r], 1);
      row_sum[r] += math::shfl_xor_sync(row_sum[r], 2);
      if (t == 0) smem.row_sum[warp][g + 8 * r] = row_sum[r];
    }
    // The S accumulator layout is the PV A-fragment layout of this warp's 16 tokens.
    smem.p[warp][lane] = make_uint4(f32x2_to_bf16x2(s[0], s[1]), f32x2_to_bf16x2(s[2], s[3]),
                                    f32x2_to_bf16x2(s[4], s[5]), f32x2_to_bf16x2(s[6], s[7]));
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

    // PV: k-step kk is warp kk's tokens; this lane needs tokens 2t, 2t + 1, 2t + 8, 2t + 9 at
    // physical dims 64 * warp + 8g .. + 7.
#pragma unroll
    for (uint32_t kk = 0; kk < kNumWarps; ++kk) {
      uint4 a = smem.p[kk][lane];
      const uint8_t* v_rows = smem.kv[stage][1][16 * kk];
      uint32_t d[4][4], sf_packed = 0;
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        const uint8_t* v_row = v_rows + (2 * t + j % 2 + 8 * (j / 2)) * kRowBytes;
        e2m1x8_to_bf16x8(*reinterpret_cast<const uint32_t*>(v_row + 32 * warp + 4 * g), d[j]);
        sf_packed |= uint32_t(v_row[kPackedBytes + 4 * warp + g / 2]) << (8 * j);
      }
      uint2 sf;
      fast_dequant_f8f16x4<__nv_fp8_e4m3, nv_bfloat16>(&sf_packed, &sf);
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        const uint32_t scale = broadcast16(sf, j);
#pragma unroll
        for (uint32_t k = 0; k < 4; ++k) d[j][k] = mul_bf16x2(d[j][k], scale);
      }
#pragma unroll
      for (uint32_t c = 0; c < 4; ++c) {
        // Pair one dim across two tokens: n8 tile 2c takes the low halves, 2c + 1 the high.
        uint32_t b[4] = {
            __byte_perm(d[0][c], d[1][c], 0x5410u), __byte_perm(d[2][c], d[3][c], 0x5410u),
            __byte_perm(d[0][c], d[1][c], 0x7632u), __byte_perm(d[2][c], d[3][c], 0x7632u)};
        mma::mma_sync_m16n16k16_row_col_f16f16f32<nv_bfloat16>(o[c], &a.x, b);
      }
    }
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
__global__ void __launch_bounds__(kMergeThreads) MergeKernel(const Params p) {
  const int row = blockIdx.x, head = blockIdx.y;
  const uint32_t dim = 4 * threadIdx.x;
  auto partial_row = [&](int split) {
    return (int64_t(split) * p.num_rows + row) * p.num_qo_heads + head;
  };
  float lse_max = -math::inf;
  for (int split = 0; split < p.num_splits; ++split) {
    lse_max = fmaxf(lse_max, p.partial_lse[partial_row(split)]);
  }
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  float denominator = 0.f;
  if (lse_max > -math::inf) {
    for (int split = 0; split < p.num_splits; ++split) {
      const float weight = math::ptx_exp2(p.partial_lse[partial_row(split)] - lse_max);
      const float4 part =
          *reinterpret_cast<const float4*>(p.partial_out + partial_row(split) * kHeadDim + dim);
      denominator += weight;
      acc.x += weight * part.x;
      acc.y += weight * part.y;
      acc.z += weight * part.z;
      acc.w += weight * part.w;
    }
  }
  const float inv = denominator > 0.f ? 1.f / denominator : 0.f;
  const float merged[4] = {acc.x * inv, acc.y * inv, acc.z * inv, acc.w * inv};
  const uint2 gate4 = *reinterpret_cast<const uint2*>(
      p.gate + (int64_t(row) * p.num_qo_heads + head) * kHeadDim + dim);
  const nv_bfloat16* gv = reinterpret_cast<const nv_bfloat16*>(&gate4);
  uint2 out4;
  nv_bfloat16* ov = reinterpret_cast<nv_bfloat16*>(&out4);
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    const float attn = __bfloat162float(__float2bfloat16(merged[j]));
    ov[j] = __float2bfloat16(attn * sigmoid(__bfloat162float(gv[j])));
  }
  *reinterpret_cast<uint2*>(p.out + int64_t(row) * p.out_stride_row + head * p.out_stride_head +
                            dim) = out4;
}

inline cudaError_t SparsePagedAttention(const Params& p, cudaStream_t stream) {
  const dim3 grid(p.num_rows, p.num_qo_heads / p.group_size, p.num_splits);
  SparsePagedAttentionKernel<<<grid, kNumThreads, 0, stream>>>(p);
  if (p.num_splits > 1) {
    MergeKernel<<<dim3(p.num_rows, p.num_qo_heads), kMergeThreads, 0, stream>>>(p);
  }
  return cudaGetLastError();
}

}  // namespace qsa_nvfp4
}  // namespace flashinfer

#endif  // FLASHINFER_EXPERIMENTAL_QSA_NVFP4_ATTENTION_CUH_
