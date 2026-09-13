// Copyright (c) 2026 FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../../arch/matrix_memory.cuh"
#include "../../arch/mma_sm120.cuh"
#include "../../compute/packed_to_bf16.cuh"
#include "../../model/dsv41_layout.cuh"
#include "resources.cuh"

namespace flashinfer::sparse_mla_sm120::kernels::dsv41_bf16 {

template <int NUM_HEADS, bool EXTRA_FP4>
__global__ void __launch_bounds__(Dsv41Bf16Resources::BLOCK_THREADS, 1)
    sparse_mla_decode_dsv41_bf16_kernel(
    const bf16* __restrict__ Q, const uint8_t* __restrict__ KV_cache,
    const int32_t* __restrict__ indices, bf16* __restrict__ mid_out, float* __restrict__ mid_lse,
    const int* __restrict__ topk_length_ptr, const uint8_t* __restrict__ extra_KV_cache,
    const int32_t* __restrict__ extra_indices, const int* __restrict__ extra_topk_length_ptr,
    int extra_topk, int pbs_extra, size_t extra_page_stride_bytes, int num_tokens, int num_heads,
    int topk, int scratch_split_stride, int chunks_per_block, float sm_scale, size_t page_stride_bytes,
    size_t indices_stride_elems, size_t extra_indices_stride_elems, int main_page_block_size) {
  using R = Dsv41Bf16Resources;
  using Geometry = R::Geometry;
  const int t = blockIdx.x, hs = blockIdx.y * R::HEADS_PER_CTA, split = blockIdx.z;
  const int qh = NUM_HEADS ? NUM_HEADS : num_heads;
  const int mh = NUM_HEADS ? NUM_HEADS : int(gridDim.y) * R::HEADS_PER_CTA;
  const int vh = min(R::HEADS_PER_CTA, qh - hs);
  const int tidx = threadIdx.x, warp = tidx / R::WARP_THREADS, lane = tidx % R::WARP_THREADS;
  const int gid = lane / 4, tid = lane % 4;
  if (t >= num_tokens) return;
  int len = topk_length_ptr ? topk_length_ptr[t] : topk;
  len = max(0, min(topk, len));
  int elen = extra_KV_cache ? (extra_topk_length_ptr ? extra_topk_length_ptr[t] : extra_topk) : 0;
  elen = max(0, min(extra_topk, elen));
  const int main_chunks = (len + R::CANDIDATES - 1) / R::CANDIDATES;
  const int chunks = main_chunks + (elen + R::CANDIDATES - 1) / R::CANDIDATES;
  const int lo = split * chunks_per_block, hi = min(lo + chunks_per_block, chunks);
  if (lo >= chunks) {
    if (tidx < vh) mid_lse[((size_t)t * mh + hs + tidx) * scratch_split_stride + split] = -1e30f;
    return;
  }
  extern __shared__ __align__(16) unsigned char storage[];
  auto& sm = *reinterpret_cast<Dsv41Bf16Smem*>(storage);
  for (int i = tidx; i < R::HEADS_PER_CTA * R::QK_VECTORS; i += R::BLOCK_THREADS) {
    const int h = i / R::QK_VECTORS, d = i % R::QK_VECTORS * R::VECTOR_ELEMS;
    uint4 data = h < vh ? *reinterpret_cast<const uint4*>(Q + ((size_t)t * qh + hs + h) * Geometry::D_QK + d)
                        : make_uint4(0, 0, 0, 0);
    *reinterpret_cast<uint4*>(&sm.q[h][d]) = data;
  }
  float acc[R::PV_TILES_PER_WARP][4] = {};
  float m[2] = {-1e30f, -1e30f}, sum[2] = {0.f, 0.f};
  for (int chunk = lo; chunk < hi; ++chunk) {
    const bool extra = chunk >= main_chunks;
    const bool fp4 = EXTRA_FP4 && extra;
    const int start = (extra ? chunk - main_chunks : chunk) * R::CANDIDATES;
    const int n = extra ? elen : len, pbs = extra ? pbs_extra : main_page_block_size;
    const size_t stride = extra ? extra_page_stride_bytes : page_stride_bytes;
    const uint8_t* cache = extra ? extra_KV_cache : KV_cache;
    const int32_t* ix = extra ? extra_indices + (size_t)t * extra_indices_stride_elems
                              : indices + (size_t)t * indices_stride_elems;
    if (tidx < R::CANDIDATES) sm.valid[tidx] = start + tidx < n ? ix[start + tidx] : -1;
    __syncthreads();
    auto gather = [&](bool power_of_two) __attribute__((always_inline)) {
      for (int task = tidx; task < R::CANDIDATES * R::QK_VECTORS; task += R::BLOCK_THREADS) {
        const int row = task / R::QK_VECTORS, d = task % R::QK_VECTORS * R::VECTOR_ELEMS;
        const int index = sm.valid[row];
        uint64_t raw = 0;
        __nv_bfloat162 scale = __floats2bfloat162_rn(0.f, 0.f);
        if (index >= 0) {
          const int page = power_of_two ? (index >> (__ffs(pbs) - 1)) : (index / pbs);
          const int slot = index - page * pbs;
          const uint8_t* base = cache + (size_t)page * stride;
          const auto data_offset =
              Dsv41Fp4Layout::selected_data_offset<Dsv41Fp8Layout>(fp4, size_t(slot));
          const uint8_t* src = base + data_offset + (fp4 ? d / 2 : d);
          raw = fp4 ? uint64_t(*reinterpret_cast<const uint32_t*>(src))
                    : *reinterpret_cast<const uint64_t*>(src);
          const auto scale_offset =
              Dsv41Fp4Layout::selected_scale_offset<Dsv41Fp8Layout>(fp4, size_t(pbs), slot);
          const uint8_t sc = base[scale_offset + d / (fp4 ? Dsv41Fp4Layout::SCALE_GROUP
                                                          : Dsv41Fp8Layout::QUANT_TILE)];
          scale = decode_e4m3_or_ue8m0_scale_bf16(sc, fp4);
        }
        __nv_bfloat162 pairs[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
          pairs[j] = __hmul2(
              decode_e2m1_or_e4m3_pair_bf16(uint32_t(raw >> (j * (fp4 ? 8 : 16))), fp4), scale);
        *reinterpret_cast<uint4*>(&sm.kv[row][d]) = *reinterpret_cast<uint4*>(pairs);
      }
    };
    if ((pbs & (pbs - 1)) == 0) gather(true);
    else gather(false);
    __syncthreads();
    float qk[4] = {};
#pragma unroll
    for (int k = 0; k < Geometry::D_QK; k += R::MMA_K) {
      uint32_t a0, a1, a2, a3;
      ldmatrix_load_A_bf16(a0, a1, a2, a3, &sm.q[0][k], R::QK_STRIDE_ELEMS, lane);
      const uint32_t b0 = *reinterpret_cast<uint32_t*>(&sm.kv[warp * 8 + gid][k + tid * 2]);
      const uint32_t b1 = *reinterpret_cast<uint32_t*>(&sm.kv[warp * 8 + gid][k + tid * 2 + 8]);
      auto r = mma_bf16_m16n8k16(a0, a1, a2, a3, b0, b1, qk[0], qk[1], qk[2], qk[3]);
      qk[0] = r.d0;
      qk[1] = r.d1;
      qk[2] = r.d2;
      qk[3] = r.d3;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
      qk[j] = sm.valid[warp * 8 + tid * 2 + (j & 1)] >= 0 ? qk[j] * (sm_scale * LOG2E) : -1e30f;
    float mx[2] = {fmaxf(qk[0], qk[1]), fmaxf(qk[2], qk[3])};
#pragma unroll
    for (int s = 1; s <= 2; s *= 2) {
      mx[0] = fmaxf(mx[0], __shfl_xor_sync(0xffffffff, mx[0], s));
      mx[1] = fmaxf(mx[1], __shfl_xor_sync(0xffffffff, mx[1], s));
    }
    if (tid == 0) {
      sm.reduce[0][warp][gid] = mx[0];
      sm.reduce[0][warp][gid + 8] = mx[1];
    }
    __syncthreads();
    float alpha[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float next = m[h];
#pragma unroll
      for (int w = 0; w < R::WARPS; ++w) next = fmaxf(next, sm.reduce[0][w][gid + h * 8]);
      alpha[h] = exp2f(m[h] - next);
      m[h] = next;
    }
    float p[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      p[j] = sm.valid[warp * 8 + tid * 2 + (j & 1)] >= 0 ? exp2f(qk[j] - m[j / 2]) : 0.f;
      sm.p[gid + j / 2 * 8][warp * 8 + tid * 2 + (j & 1)] = __float2bfloat16_rn(p[j]);
    }
    float ps[2] = {p[0] + p[1], p[2] + p[3]};
#pragma unroll
    for (int s = 1; s <= 2; s *= 2) {
      ps[0] += __shfl_xor_sync(0xffffffff, ps[0], s);
      ps[1] += __shfl_xor_sync(0xffffffff, ps[1], s);
    }
    if (tid == 0) {
      sm.reduce[1][warp][gid] = ps[0];
      sm.reduce[1][warp][gid + 8] = ps[1];
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      sum[h] *= alpha[h];
#pragma unroll
      for (int w = 0; w < R::WARPS; ++w) sum[h] += sm.reduce[1][w][gid + h * 8];
    }
#pragma unroll
    for (int v = 0; v < R::PV_TILES_PER_WARP; ++v) {
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[v][j] *= alpha[j / 2];
    }
#pragma unroll
    for (int k = 0; k < R::CANDIDATES; k += R::MMA_K) {
      uint32_t a0, a1, a2, a3;
      ldmatrix_load_A_bf16(a0, a1, a2, a3, &sm.p[0][k], R::P_STRIDE_ELEMS, lane);
#pragma unroll
      for (int v = 0; v < R::PV_TILES_PER_WARP; ++v) {
        uint32_t b0, b1;
        const uint32_t addr =
            uint32_t(__cvta_generic_to_shared(&sm.kv[k + (lane & 15)][v * (R::WARPS * R::MMA_N) + warp * R::MMA_N]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];"
                     : "=r"(b0), "=r"(b1)
                     : "r"(addr));
        auto r =
            mma_bf16_m16n8k16(a0, a1, a2, a3, b0, b1, acc[v][0], acc[v][1], acc[v][2], acc[v][3]);
        acc[v][0] = r.d0;
        acc[v][1] = r.d1;
        acc[v][2] = r.d2;
        acc[v][3] = r.d3;
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int head = hs + gid + h * 8;
    if (head < qh) {
      const float inv = sum[h] > 0.f ? 1.f / sum[h] : 0.f;
      const size_t base = ((size_t)t * mh + head) * scratch_split_stride * Geometry::D_V + split * Geometry::D_V;
#pragma unroll
      for (int v = 0; v < R::PV_TILES_PER_WARP; ++v) {
        const int d = v * (R::WARPS * R::MMA_N) + warp * R::MMA_N + tid * 2;
        *reinterpret_cast<__nv_bfloat162*>(mid_out + base + d) =
            __floats2bfloat162_rn(acc[v][h * 2] * inv, acc[v][h * 2 + 1] * inv);
      }
      if (warp == 0 && tid == 0)
        mid_lse[((size_t)t * mh + head) * scratch_split_stride + split] =
            sum[h] > 0.f ? log2f(sum[h]) + m[h] : -1e30f;
    }
  }
}

}  // namespace flashinfer::sparse_mla_sm120::kernels::dsv41_bf16

namespace flashinfer::sparse_mla_sm120 {
using kernels::dsv41_bf16::Dsv41Bf16Resources;
using kernels::dsv41_bf16::Dsv41Bf16Smem;
using kernels::dsv41_bf16::sparse_mla_decode_dsv41_bf16_kernel;
}  // namespace flashinfer::sparse_mla_sm120
