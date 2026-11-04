// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "common.cuh"
namespace flashinfer::sparse_mla_sm120::cute_prefill::v2_micro {
struct alignas(128) Shared {
  uint8_t k[2][2][64 * 128];  // (stage, feature-half, candidate, feature)
  uint8_t v[2][64 * 128];     // (stage, candidate, feature)
  uint8_t k_sf[2][64 * 16];   // independent candidate-tile parity
  uint8_t v_sf[2][64 * 16];
  float scores[2][32 * 64];
};
static_assert(sizeof(Shared) == 69632);
// Swizzle 16-byte vectors across banks; preserves the 128-byte physical row size.
using MicroLayout = decltype(composition(Swizzle<3, 4, 3>{},
    Layout<Shape<_64, _128>, Stride<_128, _1>>{}));
using MicroTransposed = decltype(composition(Swizzle<3, 4, 3>{},
    Layout<Shape<_128, _64>, Stride<_1, _128>>{}));
// Named barriers: 1 QK-local; 2/3 S-ready; 4/5 S-free (whole-tile: KV-free).
// Micro loader: 6/7 K-ready, 8/9 K-free, 10/11 V-ready, 12/13 V-free.
// Loader halves use independent 64-producer/128-consumer handoffs, never a WG barrier.
__global__ __launch_bounds__(384, 1) void splitk_prefill(Params p) {
  extern __shared__ __align__(128) unsigned char raw[];
  auto& sm = *reinterpret_cast<Shared*>(raw);
  const int lane = threadIdx.x % 32;
  const int wg = threadIdx.x / 128;
  const int warp = (threadIdx.x % 128) / 32;
  const int token = blockIdx.x / ((p.heads + 31) / 32);
  const int head_base = blockIdx.x % ((p.heads + 31) / 32) * 32;
  const int main_len = max(0, min(p.topk, p.lengths ? p.lengths[token] : p.topk));
  const int extra_len = max(0, min(p.extra_topk, p.extra_lengths ? p.extra_lengths[token] : p.extra_topk));
  const int main_tiles = (main_len + 63) / 64;
  const int tiles = main_tiles + (extra_len + 63) / 64;
  const float* cs = p.cos_sin + size_t(p.positions[token]) * 64;
  Mma mma;
  auto thr_mma = mma.get_slice(lane);

  if (wg == 2) {
    // Loader role: local warps 0/1 gather K, local warps 2/3 gather V.
    // Keep register-allocation regions inside disjoint role branches.
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
    const bool load_k = warp < 2;
    const int loader_tid = threadIdx.x % 64;
    for (int tile = 0; tile < tiles; ++tile) {
      const bool extra = tile >= main_tiles;
      const int tile_start = (extra ? tile - main_tiles : tile) * 64;
      const int length = extra ? extra_len : main_len;
      const int width = extra ? p.extra_topk : p.topk;
      const int* ids = (extra ? p.extra_indices : p.indices) + size_t(token) * width;
      const uint8_t* cache = extra ? p.extra_kv : p.kv;
      const int stride = extra ? p.extra_stride : p.kv_stride;
      const int micro_steps = load_k ? 2 : 4;
      for (int micro = 0; micro < micro_steps; ++micro) {
        const int stage = micro & 1;
        const int iteration = tile * micro_steps + micro;
        if (iteration >= 2) {
          if (load_k) bar_sync_alt<8, 9, 192>(stage);
          else bar_sync_alt<12, 13, 192>(stage);
        }
        // SF slots are indexed by candidate tile, not feature micro-step.
        // At most two feature stages are outstanding; before tile+2 starts,
        // the consumer has necessarily finished all scale reads for tile.
        if (micro == 0) {
          int id = tile_start + loader_tid < length ? ids[tile_start + loader_tid] : -1;
          uint8_t* dst = (load_k ? sm.k_sf[tile & 1] : sm.v_sf[tile & 1]) + loader_tid * 16;
          if (id >= 0) cp_async_16B(dst, cache + size_t(id) * stride + 512);
          else *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        }
        const int workers = load_k ? 32 : 64;
        const int tid = load_k ? lane : loader_tid;
        uint8_t* dst = load_k ? sm.k[stage][warp] : sm.v[stage];
        int feature_base = micro * 128 + (load_k ? warp * 256 : 0);
        // Each 128-feature vector is cooperatively copied in 16-byte units.
        for (int i = tid; i < 64 * 8; i += workers) {
          int n = i / 8, d = (i % 8) * 16;
          int id = tile_start + n < length ? ids[tile_start + n] : -1;
          if (id >= 0) cp_async_16B(dst + MicroLayout{}(n, d), cache + size_t(id) * stride + feature_base + d);
          else *reinterpret_cast<uint4*>(dst + MicroLayout{}(n, d)) = make_uint4(0, 0, 0, 0);
        }
        cp_async_commit();
        cp_async_wait_all();
        if (load_k) bar_arrive_alt<6, 7, 192>(stage);
        else bar_arrive_alt<10, 11, 192>(stage);
      }
    }
    return;
  }
  if (wg == 0) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 160;" ::: "memory");
    // QK role (WG0). Ownership: (head-half, feature-half) = (warp % 2, warp / 2).
    auto cQ = make_identity_tensor(Shape<_16, _256>{});
    auto tQcQ = thr_mma.partition_A(cQ);
    auto tQrQ = make_tensor<E4>(shape(tQcQ));
    auto rQScale = make_tensor<float>(Shape<_2, _8>{});
    const int h0 = head_base + (warp % 2) * 16;
    const int d0 = (warp / 2) * 256;
    // No Q SMEM stage. Each A fragment has two rows, eight features/row/lane.
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
      float q[16];
      #pragma unroll
      for (int v = 0; v < 16; ++v) {
        auto coord = tQcQ(v, 0, k);
        int h = h0 + get<0>(coord), d = d0 + get<1>(coord);
        float x = h < p.heads ? to_float(p.q[(size_t(token) * p.heads + h) * 512 + d]) : 0.f;
        if (d >= 448) {
          float y = h < p.heads ? to_float(p.q[(size_t(token) * p.heads + h) * 512 + (d ^ 1)]) : 0.f;
          x = to_float(to_bf16(mega_rotate<false>(x, y, d, cs)));
        }
        q[v] = x;
      }
      #pragma unroll
      for (int row = 0; row < 2; ++row) {
        float amax = 1.e-4f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(q[(j / 4) * 8 + row * 4 + j % 4]));
        amax = row_max4(amax);
        uint32_t bits = (__float_as_uint(amax / 448.f) + 0x007fffff) & 0x7f800000;
        float scale = __uint_as_float(bits);
        rQScale(row, k) = scale;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
          int v = (j / 4) * 8 + row * 4 + j % 4;
          tQrQ(v, 0, k) = E4(q[v] / scale);
        }
      }
    }
    auto cS = make_identity_tensor(Shape<_16, _64>{});
    auto tScS = thr_mma.partition_C(cS);
    auto tSrS = thr_mma.partition_fragment_C(cS);

    for (int tile = 0; tile < tiles; ++tile) {
      const int slot = tile & 1;
      if (tile >= 2) bar_sync_alt<4, 5, 256>(slot);
      const bool extra = tile >= main_tiles;
      const int tile_start = (extra ? tile - main_tiles : tile) * 64;
      const int length = extra ? extra_len : main_len;
      const int width = extra ? p.extra_topk : p.topk;
      const int* ids = (extra ? p.extra_indices : p.indices) + size_t(token) * width;
      clear(tSrS);
      #pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int micro = k / 4;
        if (k % 4 == 0) bar_sync_alt<6, 7, 192>(micro & 1);
        auto sK = make_tensor(make_smem_ptr(reinterpret_cast<E4*>(sm.k[micro & 1][warp / 2])),
            MicroLayout{});
        auto tSsK = thr_mma.partition_B(sK);
        auto tSrK = make_tensor<E4>(Shape<Shape<_4, _2>, _8, _1>{});
        auto rPartial = thr_mma.partition_fragment_C(cS);
        clear(rPartial);
        copy(tSsK(_, _, k % 4), tSrK(_, _, 0));
        gemm(mma, tQrQ(_, _, k), tSrK(_, _, 0), rPartial);
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
          #pragma unroll
          for (int v = 0; v < 4; ++v) {
            auto coord = tScS(v, 0, n);
            int row = get<0>(coord) / 8;
            int candidate = get<1>(coord);
            const uint8_t* sf;
            sf = sm.k_sf[slot] + candidate * 16;
            float scale = rQScale(row, k) * ue8(sf[d0 / 32 + k]);
            tSrS(v, 0, n) += rPartial(v, 0, n) * scale;
          }
        }
        if (k % 4 == 3) bar_arrive_alt<8, 9, 192>(micro & 1);
      }
      auto sS = make_tensor(make_smem_ptr(sm.scores[slot] + (warp % 2) * 16 * 64),
          Layout<Shape<_16, _64>, Stride<_64, _1>>{});
      auto tSsS = thr_mma.partition_C(sS);
      if (warp < 2) copy(tSrS, tSsS);
      bar_sync_t<1, 128>();
      if (warp >= 2) {
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
          #pragma unroll
          for (int v = 0; v < 4; ++v) {
            int candidate = get<1>(tScS(v, 0, n));
            bool valid = tile_start + candidate < length && ids[tile_start + candidate] >= 0;
            tSsS(v, 0, n) = valid ? (tSsS(v, 0, n) + tSrS(v, 0, n)) * (p.sm_scale * LOG2E) : -INFINITY;
          }
        }
      }
      bar_sync_t<1, 128>();
      bar_arrive_alt<2, 3, 256>(slot);
    }
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 248;" ::: "memory");
    // PV role (WG1): C=(features,heads), A=V^T, B=P. Four lanes own one head's P.
    const int h0 = head_base + warp * 8;
    auto cP = make_identity_tensor(Shape<_8, _64>{});
    auto tPcP = thr_mma.partition_B(cP);
    auto rP = make_tensor<float>(shape(tPcP));
    auto rP8 = make_tensor<E4>(shape(tPcP));
    auto cO = make_identity_tensor(Shape<_512, _8>{});
    auto tOcO = thr_mma.partition_C(cO);
    auto rO = thr_mma.partition_fragment_C(cO);
    clear(rO);
    float row_m = -INFINITY, row_l = 0.f;
    const int own_head = lane / 4;
    for (int tile = 0; tile < tiles; ++tile) {
      int slot = tile & 1;
      bar_sync_alt<2, 3, 256>(slot);
      auto sS = make_tensor(make_smem_ptr(sm.scores[slot] + warp * 8 * 64),
          Layout<Shape<_8, _64>, Stride<_64, _1>>{});
      copy(thr_mma.partition_B(sS), rP);
      bar_arrive_alt<4, 5, 256>(slot);
      float tile_m = -INFINITY;
      #pragma unroll
      for (int i = 0; i < size(rP); ++i) tile_m = fmaxf(tile_m, rP(i));
      float new_m = fmaxf(row_m, row_max4(tile_m));
      float alpha = isfinite(row_m) ? exp2f(row_m - new_m) : 0.f;
      float tile_l = 0.f;
      #pragma unroll
      for (int i = 0; i < size(rP); ++i) {
        rP(i) = isfinite(rP(i)) ? exp2f(rP(i) - new_m) : 0.f;
        tile_l += rP(i);
      }
      row_l = row_l * alpha + row_sum4(tile_l);
      row_m = new_m;
      // A thread owns two head columns, repeated across every feature tile.
      // Hoist collectives explicitly: nvcc does not CSE unrolled __shfl_sync.
      float alpha0 = __shfl_sync(0xffffffff, alpha, (lane % 4) * 8);
      float alpha1 = __shfl_sync(0xffffffff, alpha, (lane % 4) * 8 + 4);
      #pragma unroll
      for (int m = 0; m < 32; ++m) {
        #pragma unroll
        for (int v = 0; v < 4; ++v) {
          rO(v, m, 0) *= (v % 2 == 0 ? alpha0 : alpha1);
        }
      }
      #pragma unroll
      for (int group = 0; group < 16; ++group) {
        const int micro = group / 4;
        if (group % 4 == 0) bar_sync_alt<10, 11, 192>(micro & 1);
        auto rScaled = make_tensor_like(rP);
        float amax = 1.e-10f;
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
          #pragma unroll
          for (int v = 0; v < 8; ++v) {
            int n = get<1>(tPcP(v, 0, k));
            rScaled(v, 0, k) = rP(v, 0, k) * ue8(sm.v_sf[slot][n * 16 + group]);
            amax = fmaxf(amax, fabsf(rScaled(v, 0, k)));
          }
        }
        float scale = row_max4(amax) / 448.f;
        #pragma unroll
        for (int i = 0; i < size(rP8); ++i) rP8(i) = E4(rScaled(i) / scale);
        auto sVStage = make_tensor(make_smem_ptr(reinterpret_cast<E4*>(sm.v[micro & 1])), MicroTransposed{});
        auto sV = local_tile(sVStage, Shape<_32, _64>{}, make_coord(group % 4, _0{}));
        auto rV = thr_mma.partition_fragment_A(sV);
        auto tiled_copy = make_tiled_copy_A(Copy_Atom<SM120_U8x16_LDSM_T_RAW, E4>{}, mma);
        auto thr_copy = tiled_copy.get_slice(lane);
        auto rV_cv = thr_copy.retile_D(rV);
        copy(tiled_copy, thr_copy.partition_S(sV), rV_cv);
        auto rPV = thr_mma.partition_fragment_C(make_identity_tensor(Shape<_32, _8>{}));
        clear(rPV);
        gemm(mma, rV, rP8, rPV);
        float scale0 = __shfl_sync(0xffffffff, scale, (lane % 4) * 8);
        float scale1 = __shfl_sync(0xffffffff, scale, (lane % 4) * 8 + 4);
        #pragma unroll
        for (int m = 0; m < 2; ++m) {
          #pragma unroll
          for (int v = 0; v < 4; ++v) {
            rO(v, group * 2 + m, 0) += rPV(v, m, 0) * (v % 2 == 0 ? scale0 : scale1);
          }
        }
        if (group % 4 == 3) bar_arrive_alt<12, 13, 192>(micro & 1);
      }
    }
    // Normalize including the sink; preserve the original BF16 context boundary.
    float denominator = row_l;
    float out_factor = 1.f;
    float sink = p.sink && h0 + own_head < p.heads ? p.sink[h0 + own_head] * LOG2E : -INFINITY;
    if (sink == INFINITY) {
      out_factor = 0.f;
      denominator = 1.f;
      if ((lane % 4) == 0) p.lse[size_t(token) * p.heads + h0 + own_head] = INFINITY;
    } else if (sink != -INFINITY) {
      float shift = fmaxf(row_m, sink);
      out_factor = isfinite(row_m) ? exp2f(row_m - shift) : 0.f;
      denominator = row_l * out_factor + exp2f(sink - shift);
      if ((lane % 4) == 0) p.lse[size_t(token) * p.heads + h0 + own_head] = (shift + log2f(denominator)) / LOG2E;
    } else if ((lane % 4) == 0 && h0 + own_head < p.heads) {
      p.lse[size_t(token) * p.heads + h0 + own_head] = row_l > 0.f ? (row_m + log2f(row_l)) / LOG2E : -INFINITY;
    }
    float norm = denominator > 0.f ? out_factor / denominator : 0.f;
    float norm0 = __shfl_sync(0xffffffff, norm, (lane % 4) * 8);
    float norm1 = __shfl_sync(0xffffffff, norm, (lane % 4) * 8 + 4);
    #pragma unroll
    for (int group = 0; group < 16; ++group) {
      float vals[8];
      #pragma unroll
      for (int m = 0; m < 2; ++m) {
        #pragma unroll
        for (int v = 0; v < 4; ++v) {
          auto coord = tOcO(v, group * 2 + m, 0);
          int d = get<0>(coord);
          float x = to_float(to_bf16(rO(v, group * 2 + m, 0) * (v % 2 == 0 ? norm0 : norm1)));
          float y = __shfl_xor_sync(0xffffffff, x, 4);
          vals[m * 4 + v] = mega_rotate<false>(x, y, d, cs, true);
        }
      }
      #pragma unroll
      for (int head_col = 0; head_col < 2; ++head_col) {
        float amax = 1.e-4f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) amax = fmaxf(amax, fabsf(vals[j * 2 + head_col]));
        #pragma unroll
        for (int delta = 4; delta <= 16; delta *= 2) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, delta));
        uint32_t bits = (__float_as_uint(amax / 448.f) + 0x007fffff) & 0x7f800000;
        float inv = __uint_as_float(0x7f000000u - bits);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          int m = j / 2, v = (j % 2) * 2 + head_col;
          auto coord = tOcO(v, group * 2 + m, 0);
          int d = get<0>(coord), h = h0 + get<1>(coord);
          if (h < p.heads) {
            size_t offset = (size_t(token) * p.heads + h) * 512 + d;
            p.output[offset] = quantize_e4m3_byte(vals[j * 2 + head_col] * inv);
            if ((d % 32) == 0) p.scales[offset / 32] = bits >> 23;
          }
        }
      }
    }
  }
}
}
