// Copyright (c) 2026 FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../../arch/matrix_memory.cuh"
#include "../../arch/mma_sm120.cuh"
#include "../../compute/packed_to_bf16.cuh"
#include "../../model/dsv41_layout.cuh"
#include "../../execution/attention_params.cuh"
#include "resources.cuh"

namespace flashinfer::sparse_mla_sm120::kernels::dsv41_bf16 {

template <bool EXTRA_FP4>
__device__ __forceinline__ void
run_bf16_tiles(Dsv41Bf16Smem &sm, const uint8_t *__restrict__ KV_cache,
               const int32_t *__restrict__ indices,
               const uint8_t *__restrict__ extra_KV_cache,
               const int32_t *__restrict__ extra_indices, int t, int lo, int hi,
               int len, int elen, int main_chunks, int main_page_block_size,
               int pbs_extra, size_t page_stride_bytes,
               size_t extra_page_stride_bytes, size_t indices_stride_elems,
               size_t extra_indices_stride_elems, float sm_scale,
               float (&acc)[Dsv41Bf16Resources::PV_TILES_PER_WARP][4],
               float (&m)[2], float (&sum)[2]) {
  using R = Dsv41Bf16Resources;
  using Geometry = R::Geometry;
  const int tidx = threadIdx.x, warp = tidx / R::WARP_THREADS,
            lane = tidx % R::WARP_THREADS;
  const int gid = lane / 4, tid = lane % 4;
  for (int chunk = lo; chunk < hi; ++chunk) {
    const bool extra = chunk >= main_chunks;
    const bool fp4 = EXTRA_FP4 && extra;
    const int start = (extra ? chunk - main_chunks : chunk) * R::CANDIDATES;
    const int n = extra ? elen : len,
              pbs = extra ? pbs_extra : main_page_block_size;
    const size_t stride = extra ? extra_page_stride_bytes : page_stride_bytes;
    const uint8_t *cache = extra ? extra_KV_cache : KV_cache;
    const int32_t *ix =
        extra ? extra_indices + (size_t)t * extra_indices_stride_elems
              : indices + (size_t)t * indices_stride_elems;
    if (tidx < R::CANDIDATES)
      sm.valid[tidx] = start + tidx < n ? ix[start + tidx] : -1;
    __syncthreads();
    auto gather = [&](bool power_of_two) __attribute__((always_inline)) {
      static_assert((R::CANDIDATES * R::QK_VECTORS) % (2 * R::BLOCK_THREADS) ==
                    0);
      for (int first = tidx; first < R::CANDIDATES * R::QK_VECTORS;
           first += 2 * R::BLOCK_THREADS) {
        uint64_t raw[2];
        uint32_t scale_code[2];
        const uint8_t *data_addr[2] = {}, *scale_addr[2] = {};
        bool valid[2];
#pragma unroll
        for (int item = 0; item < 2; ++item) {
          const int task = first + item * R::BLOCK_THREADS;
          const int row = task / R::QK_VECTORS,
                    d = task % R::QK_VECTORS * R::VECTOR_ELEMS;
          const int index = sm.valid[row];
          valid[item] = index >= 0;
          if (valid[item]) {
            const int page =
                power_of_two ? (index >> (__ffs(pbs) - 1)) : (index / pbs);
            const int slot = index - page * pbs;
            const uint8_t *base = cache + (size_t)page * stride;
            const auto data_offset =
                Dsv41Fp4Layout::selected_data_offset<Dsv41Fp8Layout>(
                    fp4, size_t(slot));
            data_addr[item] = base + data_offset + (fp4 ? d / 2 : d);
            const auto scale_offset =
                Dsv41Fp4Layout::selected_scale_offset<Dsv41Fp8Layout>(
                    fp4, size_t(pbs), slot);
            constexpr int fp4_scale_shift = []() constexpr {
              int shift = 0;
              for (int group = Dsv41Fp4Layout::SCALE_GROUP; group > 1;
                   group >>= 1)
                ++shift;
              return shift;
            }();
            constexpr int fp8_scale_shift = []() constexpr {
              int shift = 0;
              for (int group = Dsv41Fp8Layout::QUANT_TILE; group > 1;
                   group >>= 1)
                ++shift;
              return shift;
            }();
            static_assert((1u << fp4_scale_shift) ==
                          Dsv41Fp4Layout::SCALE_GROUP);
            static_assert((1u << fp8_scale_shift) ==
                          Dsv41Fp8Layout::QUANT_TILE);
            scale_addr[item] =
                base + scale_offset +
                (unsigned(d) >> (fp4 ? fp4_scale_shift : fp8_scale_shift));
          }
        }
        asm volatile(
            "{\n"
            ".reg .pred v0, v1, f, p;\n"
            ".reg .b64 a0, a1;\n"
            ".reg .b32 lo0, hi0, lo1, hi1, s0, s1;\n"
            "mov.b32 lo0, 0; mov.b32 hi0, 0; mov.b32 lo1, 0; mov.b32 hi1, 0;\n"
            "mov.u32 s0, 0; mov.u32 s1, 0;\n"
            "setp.ne.u32 v0, %8, 0; setp.ne.u32 v1, %9, 0; setp.ne.u32 f, %10, "
            "0;\n"
            "and.pred p, v0, f; @p ld.global.u32 lo0, [%4];\n"
            "and.pred p, v1, f; @p ld.global.u32 lo1, [%6];\n"
            "not.pred f, f;\n"
            "and.pred p, v0, f; @p ld.global.v2.u32 {lo0, hi0}, [%4];\n"
            "and.pred p, v1, f; @p ld.global.v2.u32 {lo1, hi1}, [%6];\n"
            "@v0 ld.global.u8 s0, [%5]; @v1 ld.global.u8 s1, [%7];\n"
            "mov.b64 a0, {lo0, hi0}; mov.b64 a1, {lo1, hi1};\n"
            "mov.b64 %0, a0; mov.b64 %1, a1; mov.u32 %2, s0; mov.u32 %3, s1;\n"
            "}\n"
            : "=l"(raw[0]), "=l"(raw[1]), "=r"(scale_code[0]),
              "=r"(scale_code[1])
            : "l"(data_addr[0]), "l"(scale_addr[0]), "l"(data_addr[1]),
              "l"(scale_addr[1]), "r"(int(valid[0])), "r"(int(valid[1])),
              "r"(int(fp4))
            : "memory");
#pragma unroll
        for (int item = 0; item < 2; ++item) {
          const int task = first + item * R::BLOCK_THREADS;
          const int row = task / R::QK_VECTORS,
                    d = task % R::QK_VECTORS * R::VECTOR_ELEMS;
          const __nv_bfloat162 scale =
              valid[item]
                  ? decode_e4m3_or_ue8m0_scale_bf16(scale_code[item], fp4)
                  : __floats2bfloat162_rn(0.f, 0.f);
          __nv_bfloat162 pairs[4];
#pragma unroll
          for (int j = 0; j < 4; ++j)
            pairs[j] =
                __hmul2(decode_e2m1_or_e4m3_pair_bf16(
                            uint32_t(raw[item] >> (j * (fp4 ? 8 : 16))), fp4),
                        scale);
          *reinterpret_cast<uint4 *>(&sm.kv[row][d]) =
              *reinterpret_cast<uint4 *>(pairs);
        }
      }
    };
    if ((pbs & (pbs - 1)) == 0)
      gather(true);
    else
      gather(false);
    __syncthreads();
    float qk[4] = {};
#pragma unroll
    for (int k = 0; k < Geometry::D_QK; k += R::MMA_K) {
      uint32_t a0, a1, a2, a3;
      ldmatrix_load_A_bf16(a0, a1, a2, a3, &sm.q[0][k], R::QK_STRIDE_ELEMS,
                           lane);
      const uint32_t b0 =
          *reinterpret_cast<uint32_t *>(&sm.kv[warp * 8 + gid][k + tid * 2]);
      const uint32_t b1 = *reinterpret_cast<uint32_t *>(
          &sm.kv[warp * 8 + gid][k + tid * 2 + 8]);
      auto r =
          mma_bf16_m16n8k16(a0, a1, a2, a3, b0, b1, qk[0], qk[1], qk[2], qk[3]);
      qk[0] = r.d0;
      qk[1] = r.d1;
      qk[2] = r.d2;
      qk[3] = r.d3;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
      qk[j] = sm.valid[warp * 8 + tid * 2 + (j & 1)] >= 0
                  ? qk[j] * (sm_scale * LOG2E)
                  : -1e30f;
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
      for (int w = 0; w < R::WARPS; ++w)
        next = fmaxf(next, sm.reduce[0][w][gid + h * 8]);
      alpha[h] = exp2f(m[h] - next);
      m[h] = next;
    }
    float p[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      p[j] = sm.valid[warp * 8 + tid * 2 + (j & 1)] >= 0
                 ? exp2f(qk[j] - m[j / 2])
                 : 0.f;
      sm.p[gid + j / 2 * 8][warp * 8 + tid * 2 + (j & 1)] =
          __float2bfloat16_rn(p[j]);
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
      for (int w = 0; w < R::WARPS; ++w)
        sum[h] += sm.reduce[1][w][gid + h * 8];
    }
#pragma unroll
    for (int v = 0; v < R::PV_TILES_PER_WARP; ++v) {
#pragma unroll
      for (int j = 0; j < 4; ++j)
        acc[v][j] *= alpha[j / 2];
    }
#pragma unroll
    for (int k = 0; k < R::CANDIDATES; k += R::MMA_K) {
      uint32_t a0, a1, a2, a3;
      ldmatrix_load_A_bf16(a0, a1, a2, a3, &sm.p[0][k], R::P_STRIDE_ELEMS,
                           lane);
#pragma unroll
      for (int v = 0; v < R::PV_TILES_PER_WARP; ++v) {
        uint32_t b0, b1;
        const uint32_t addr = uint32_t(__cvta_generic_to_shared(
            &sm.kv[k + (lane & 15)]
                  [v * (R::WARPS * R::MMA_N) + warp * R::MMA_N]));
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];"
            : "=r"(b0), "=r"(b1)
            : "r"(addr));
        auto r = mma_bf16_m16n8k16(a0, a1, a2, a3, b0, b1, acc[v][0], acc[v][1],
                                   acc[v][2], acc[v][3]);
        acc[v][0] = r.d0;
        acc[v][1] = r.d1;
        acc[v][2] = r.d2;
        acc[v][3] = r.d3;
      }
    }
    __syncthreads();
  }
}

template <int NUM_HEADS, bool EXTRA_FP4>
__global__ void __launch_bounds__(Dsv41Bf16Resources::BLOCK_THREADS, 1)
    sparse_mla_prefill_dsv41_bf16_kernel(const execution::AttentionParams params) {
  using R = Dsv41Bf16Resources;
  const int heads = NUM_HEADS ? NUM_HEADS : params.num_heads;
  const int groups = (heads + R::HEADS_PER_CTA - 1) / R::HEADS_PER_CTA;
  const int t = blockIdx.x / groups,
            hs = blockIdx.x % groups * R::HEADS_PER_CTA;
  if (t >= params.num_tokens)
    return;
  const int vh = min(R::HEADS_PER_CTA, heads - hs);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, gid = lane / 4,
            tid = lane % 4;
  const int len =
      max(0, min(params.topk,
                 params.topk_length ? params.topk_length[t] : params.topk));
  const int elen =
      params.extra_kv
          ? max(0, min(params.extra_topk, params.extra_topk_length
                                              ? params.extra_topk_length[t]
                                              : params.extra_topk))
          : 0;
  const int main_chunks = (len + R::CANDIDATES - 1) / R::CANDIDATES;
  const int chunks = main_chunks + (elen + R::CANDIDATES - 1) / R::CANDIDATES;
  extern __shared__ __align__(16) unsigned char storage[];
  auto &sm = *reinterpret_cast<Dsv41Bf16Smem *>(storage);
  for (int i = threadIdx.x; i < R::HEADS_PER_CTA * R::QK_VECTORS;
       i += R::BLOCK_THREADS) {
    const int h = i / R::QK_VECTORS, d = i % R::QK_VECTORS * R::VECTOR_ELEMS;
    uint4 data = h < vh
                     ? *reinterpret_cast<const uint4 *>(
                           params.q +
                           ((size_t)t * heads + hs + h) * R::Geometry::D_QK + d)
                     : make_uint4(0, 0, 0, 0);
    *reinterpret_cast<uint4 *>(&sm.q[h][d]) = data;
  }
  float acc[R::PV_TILES_PER_WARP][4] = {};
  float m[2] = {-1e30f, -1e30f}, sum[2] = {0.f, 0.f};
  run_bf16_tiles<EXTRA_FP4>(
      sm, params.kv, params.indices, params.extra_kv, params.extra_indices, t,
      0, chunks, len, elen, main_chunks, params.page_size,
      params.extra_page_size, params.page_stride_bytes,
      params.extra_page_stride_bytes, params.indices_stride_elems,
      params.extra_indices_stride_elems, params.sm_scale, acc, m, sum);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int head = hs + gid + h * 8;
    if (head < heads) {
      float norm = sum[h] > 0.f ? 1.f / sum[h] : 0.f;
      float lse = sum[h] > 0.f ? log2f(sum[h]) + m[h] : -1e30f;
      if (params.attn_sink) {
        const float sink = params.attn_sink[head] * LOG2E;
        if (sum[h] > 0.f) {
          const float peak = fmaxf(m[h], sink);
          const float weight = exp2f(m[h] - peak);
          const float total = sum[h] * weight + exp2f(sink - peak);
          norm = weight / total;
          lse = log2f(total) + peak;
        } else {
          lse = sink;
        }
      }
      const size_t base = ((size_t)t * heads + head) * R::Geometry::D_V;
#pragma unroll
      for (int v = 0; v < R::PV_TILES_PER_WARP; ++v) {
        const int d = v * (R::WARPS * R::MMA_N) + warp * R::MMA_N + tid * 2;
        *reinterpret_cast<__nv_bfloat162 *>(params.output + base + d) =
            __floats2bfloat162_rn(acc[v][h * 2] * norm,
                                  acc[v][h * 2 + 1] * norm);
      }
      if (warp == 0 && tid == 0)
        params.out_lse[(size_t)t * params.out_lse_stride_elems + head] = lse;
    }
  }
}

} // namespace flashinfer::sparse_mla_sm120::kernels::dsv41_bf16
