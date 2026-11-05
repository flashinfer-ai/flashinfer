// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "common.cuh"
#include <cute/atom/copy_traits_sm75.hpp>
#include <cutlass/pipeline/sm90_pipeline.hpp>
namespace flashinfer::sparse_mla_sm120::cute_prefill::integrated {
// 32-head x 64-candidate tile with compile-time prologue/epilogue fusion.
// Unresolved: racecheck reports hazards on the asynchronous shared-memory copies.
struct Params : cute_prefill::Params {
  bf16* bf16_output;
  size_t lse_stride;
  float lse_scale;
};
enum class Role { load_k, load_v, mma_qk, mma_pv };
using Pipeline = cutlass::PipelineAsync<3>;
struct alignas(128) Shared {
  uint8_t k[3][2][64 * 128];  // (stage, feature-half, candidate, feature)
  uint8_t v[3][64 * 128];     // (stage, candidate, feature)
  uint8_t k_sf[2][64 * 16];   // independent candidate-tile parity
  uint8_t v_sf[2][64 * 16];
  float scores[2][32 * 64];
  Pipeline::SharedStorage k_pipeline, v_pipeline;
};
static_assert(sizeof(Shared) == 94336);
// K and V use three ring slots independently of the feature-loop extent.
// K: (candidate=64, micro-feature=128, feature-half=2, ring-slot=3).
// V: (candidate=64, micro-feature=128, ring-slot=3).
using MicroLayout = decltype(composition(Swizzle<3, 4, 3>{},
    Layout<Shape<_64, _128>, Stride<_128, _1>>{}));
using MicroTransposed = decltype(composition(Swizzle<3, 4, 3>{},
    Layout<Shape<_128, _64>, Stride<_1, _128>>{}));

// Score swizzle (float elements): row bit0 -> column bit4, row bit1 -> column bit3.
// It preserves QK's 2-float stores and PV's 4-float loads.
// Named barriers: 1 QK-local; 2/3 S-ready; 4/5 S-free.
// PQuantGroup defaults to 32; development harnesses can instantiate 64.
// Both use the same 64-candidate load tile and shared-memory allocation.
// The softmax loop and reduction order differ from the original kernel, so
// a 32-candidate quantization group does not guarantee bitwise equality.
template <bool Mega, int PQuantGroup = 32>
__global__ __launch_bounds__(384, 1) void prefill(Params p) {
  static_assert(PQuantGroup == 32 || PQuantGroup == 64);
  extern __shared__ __align__(128) unsigned char raw[];
  auto& sm = *reinterpret_cast<Shared*>(raw);
  const int lane = threadIdx.x % 32;
  const int wg = threadIdx.x / 128;
  const int warp = (threadIdx.x % 128) / 32;
  const Role role = wg == 0 ? Role::mma_qk : wg == 1 ? Role::mma_pv
      : warp < 2 ? Role::load_k : Role::load_v;
  Pipeline::Params k_params, v_params;
  k_params.producer_arv_count = v_params.producer_arv_count = 64;
  k_params.consumer_arv_count = v_params.consumer_arv_count = 128;
  k_params.dst_blockid = v_params.dst_blockid = 0;
  k_params.role = role == Role::mma_qk ? Pipeline::ThreadCategory::Consumer
      : (role == Role::load_k ? Pipeline::ThreadCategory::Producer : Pipeline::ThreadCategory::NonParticipant);
  v_params.role = role == Role::mma_pv ? Pipeline::ThreadCategory::Consumer
      : (role == Role::load_v ? Pipeline::ThreadCategory::Producer : Pipeline::ThreadCategory::NonParticipant);
  Pipeline k_pipeline(sm.k_pipeline, k_params);
  Pipeline v_pipeline(sm.v_pipeline, v_params);
  __syncthreads();
  const int token = blockIdx.x / ((p.heads + 31) / 32);
  const int head_base = blockIdx.x % ((p.heads + 31) / 32) * 32;
  const int main_len = max(0, min(p.topk, p.lengths ? p.lengths[token] : p.topk));
  const int extra_len = max(0, min(p.extra_topk, p.extra_lengths ? p.extra_lengths[token] : p.extra_topk));
  const int main_tiles = (main_len + 63) / 64;
  const int tiles = main_tiles + (extra_len + 63) / 64;
  Mma mma;
  auto thr_mma = mma.get_slice(lane);

  // Register redistribution must execute uniformly across the loader warpgroup,
  // before its two-warp K and V roles diverge.
  if (wg == 2) asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");

  if (role == Role::load_k) {
    // WG2 warps 0/1 own feature halves 0:256 and 256:512.
    auto write_state = cutlass::make_producer_start_state<Pipeline>();
    const int loader_tid = threadIdx.x % 64;
    #pragma unroll 1
    for (int tile = 0; tile < tiles; ++tile) {
      const bool extra = tile >= main_tiles;
      const int tile_start = (extra ? tile - main_tiles : tile) * 64;
      const int length = extra ? extra_len : main_len;
      const int width = extra ? p.extra_topk : p.topk;
      const int* ids = (extra ? p.extra_indices : p.indices) + size_t(token) * width;
      const uint8_t* cache = extra ? p.extra_kv : p.kv;
      const int stride = extra ? p.extra_stride : p.kv_stride;
      // Cache all 64 sparse row IDs per loader warp, reused across feature slices.
      const int id_lo = tile_start + lane < length ? ids[tile_start + lane] : -1;
      const int id_hi = tile_start + lane + 32 < length ? ids[tile_start + lane + 32] : -1;
      #pragma unroll 1
      for (int micro = 0; micro < 2; ++micro) {
        k_pipeline.producer_acquire(write_state);
        const int stage = write_state.index();
        if (micro == 0) {
          int id = warp ? id_hi : id_lo;
          uint8_t* dst = sm.k_sf[tile & 1] + loader_tid * 16;
          uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
          const uint8_t* src = cache + size_t(max(id, 0)) * stride + 512;
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
              :: "r"(address), "l"(src), "r"(id >= 0 ? 16 : 0) : "memory");
        }
        uint8_t* dst = sm.k[stage][warp];
        const int feature_base = micro * 128 + warp * 256;
        #pragma unroll 1
        for (int i = lane; i < 64 * 8; i += 32) {
          int n = i / 8, d = (i % 8) * 16;
          int id = __shfl_sync(0xffffffff, n < 32 ? id_lo : id_hi, n % 32);
          uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(dst + MicroLayout{}(n, d)));
          const uint8_t* src = cache + size_t(max(id, 0)) * stride + feature_base + d;
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
              :: "r"(address), "l"(src), "r"(id >= 0 ? 16 : 0) : "memory");
        }
        k_pipeline.producer_commit(write_state, cutlass::arch::cpasync_barrier_arrive);
        ++write_state;
      }
    }
    k_pipeline.producer_tail(write_state);
  } else if (role == Role::load_v) {
    // WG2 warps 2/3 cooperatively gather the full feature range.
    auto write_state = cutlass::make_producer_start_state<Pipeline>();
    const int loader_tid = threadIdx.x % 64;
    #pragma unroll 1
    for (int tile = 0; tile < tiles; ++tile) {
      const bool extra = tile >= main_tiles;
      const int tile_start = (extra ? tile - main_tiles : tile) * 64;
      const int length = extra ? extra_len : main_len;
      const int width = extra ? p.extra_topk : p.topk;
      const int* ids = (extra ? p.extra_indices : p.indices) + size_t(token) * width;
      const uint8_t* cache = extra ? p.extra_kv : p.kv;
      const int stride = extra ? p.extra_stride : p.kv_stride;
      // Cache all 64 sparse row IDs per loader warp, reused across feature slices.
      const int id_lo = tile_start + lane < length ? ids[tile_start + lane] : -1;
      const int id_hi = tile_start + lane + 32 < length ? ids[tile_start + lane + 32] : -1;
      // Scale loads and transpose use registers only; shared stores wait for FREE.
      int sf_id = (warp % 2) ? id_hi : id_lo;
      uint4 packed = sf_id >= 0
          ? *reinterpret_cast<const uint4*>(cache + size_t(sf_id) * stride + 512)
          : make_uint4(0, 0, 0, 0);
      auto words = make_tensor(make_rmem_ptr(reinterpret_cast<uint32_t*>(&packed)), Layout<_4>{});
      auto rSF = make_tensor<uint32_t>(Layout<_4>{});
      #pragma unroll
      for (int g = 0; g < 4; ++g) {
        uint32_t w = words(g);
        uint32_t a = __shfl_sync(0xffffffff, w, lane & ~3);
        uint32_t b = __shfl_sync(0xffffffff, w, (lane & ~3) + 1);
        uint32_t c = __shfl_sync(0xffffffff, w, (lane & ~3) + 2);
        uint32_t d = __shfl_sync(0xffffffff, w, (lane & ~3) + 3);
        int shift = (lane & 3) * 8;
        rSF(g) = ((a >> shift) & 255) | (((b >> shift) & 255) << 8)
            | (((c >> shift) & 255) << 16) | (((d >> shift) & 255) << 24);
      }
      #pragma unroll 1
      for (int micro = 0; micro < 4; ++micro) {
        v_pipeline.producer_acquire(write_state);
        const int stage = write_state.index();
        if (micro == 0) {
          #pragma unroll
          for (int g = 0; g < 4; ++g) {
            *reinterpret_cast<uint32_t*>(sm.v_sf[tile & 1] + (g * 16 + loader_tid / 4) * 16 + (lane & 3) * 4) = rSF(g);
          }
        }
        uint8_t* dst = sm.v[stage];
        const int feature_base = micro * 128;
        #pragma unroll 1
        for (int i = loader_tid; i < 64 * 8; i += 64) {
          int n = i / 8, d = (i % 8) * 16;
          int id = __shfl_sync(0xffffffff, n < 32 ? id_lo : id_hi, n % 32);
          uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(dst + MicroLayout{}(n, d)));
          const uint8_t* src = cache + size_t(max(id, 0)) * stride + feature_base + d;
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
              :: "r"(address), "l"(src), "r"(id >= 0 ? 16 : 0) : "memory");
        }
        v_pipeline.producer_commit(write_state, cutlass::arch::cpasync_barrier_arrive);
        ++write_state;
      }
    }
    v_pipeline.producer_tail(write_state);
  } else if (role == Role::mma_qk) {
    const float* cs = nullptr;
    if constexpr (Mega) cs = p.cos_sin + size_t(p.positions[token]) * 64;
    asm volatile("setmaxnreg.inc.sync.aligned.u32 208;" ::: "memory");
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
      for (int v = 0; v < 16; v += 4) {
        auto coord = tQcQ(v, 0, k);
        int h = h0 + get<0>(coord), d = d0 + get<1>(coord);
        uint2 packed = h < p.heads
            ? *reinterpret_cast<const uint2*>(p.q + (size_t(token) * p.heads + h) * 512 + d)
            : make_uint2(0, 0);
        auto q4 = make_tensor(make_rmem_ptr(reinterpret_cast<bf16*>(&packed)), Layout<_4>{});
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          float x = to_float(q4(j));
          if (Mega && d + j >= 448) {
            float y = to_float(q4(j ^ 1));
            x = to_float(to_bf16(mega_rotate<false>(x, y, d + j, cs)));
          }
          q[v + j] = x;
        }
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
        auto rQ_packed = recast<uint32_t>(tQrQ);
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
          int v = part * 8 + row * 4;
          rQ_packed(part * 2 + row, 0, k) = cvt_e4m3x4(q[v] / scale, q[v + 1] / scale,
                                                    q[v + 2] / scale, q[v + 3] / scale);
        }
      }
    }
    auto cS = make_identity_tensor(Shape<_16, _64>{});
    auto tScS = thr_mma.partition_C(cS);
    auto tSrS = thr_mma.partition_fragment_C(cS);

    Pipeline::PipelineState read_state;
    for (int tile = 0; tile < tiles; ++tile) {
      const int slot = tile & 1;
      if (tile >= 2) bar_sync_alt<4, 5, 256>(slot);
      const bool extra = tile >= main_tiles;
      const int tile_start = (extra ? tile - main_tiles : tile) * 64;
      const int length = extra ? extra_len : main_len;
      const int width = extra ? p.extra_topk : p.topk;
      const int* ids = (extra ? p.extra_indices : p.indices) + size_t(token) * width;
      clear(tSrS);
      auto rKScale = make_tensor<uint32_t>(Shape<_2, _2, _8>{});  // (micro, candidate-pair, N tile)
      #pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int micro = k / 4;
        const int stage = read_state.index();
        if (k % 4 == 0) {
          k_pipeline.consumer_wait(read_state);
          if (k == 0) {
            #pragma unroll
            for (int n = 0; n < 8; ++n) {
              #pragma unroll
              for (int c = 0; c < 2; ++c) {
                int candidate = get<1>(tScS(c, 0, n));
                uint2 packed = *reinterpret_cast<const uint2*>(sm.k_sf[slot] + candidate * 16 + d0 / 32);
                rKScale(0, c, n) = packed.x;
                rKScale(1, c, n) = packed.y;
              }
            }
          }
        }
        auto sK = make_tensor(make_smem_ptr(reinterpret_cast<E4*>(sm.k[stage][warp / 2])),
            MicroLayout{});
        auto sKTile = local_tile(sK, Shape<_64, _32>{}, make_coord(_0{}, k % 4));
        auto tSrK = make_tensor<E4>(Shape<Shape<_4, _2>, _8, _1>{});
        auto rPartial = thr_mma.partition_fragment_C(cS);
        clear(rPartial);
        auto copy_mma = make_tiled_mma(SM89_16x8x32_F32E4M3E4M3F32_TN{},
            Layout<Shape<_1, _1, _1>>{}, Tile<_16, _16, _32>{});
        auto copy_k = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, E4>{}, copy_mma);
        auto thr_copy_k = copy_k.get_slice(lane);
        auto tSrK_cv = thr_copy_k.retile_D(tSrK);
        copy(copy_k, thr_copy_k.partition_S(sKTile), tSrK_cv);
        gemm(mma, tQrQ(_, _, k), tSrK(_, _, 0), rPartial);
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
          #pragma unroll
          for (int v = 0; v < 4; ++v) {
            auto coord = tScS(v, 0, n);
            int row = get<0>(coord) / 8;
            uint8_t sf = (rKScale(micro, v % 2, n) >> ((k % 4) * 8)) & 255;
            float scale = rQScale(row, k) * ue8(sf);
            tSrS(v, 0, n) += rPartial(v, 0, n) * scale;
          }
        }
        if (k % 4 == 3) {
          k_pipeline.consumer_release(read_state);
          ++read_state;
        }
      }
      auto sS = make_tensor(make_smem_ptr(sm.scores[slot] + (warp % 2) * 16 * 64),
          composition(Swizzle<2, 3, 3>{}, make_layout(
              Layout<Shape<_2, _2, _4>, Stride<_128, _64, _256>>{}, Layout<_64>{})));
      auto tSsS = thr_mma.partition_C(sS);
      if (warp < 2) copy_aligned(tSrS, tSsS);
      bar_sync_t<1, 128>();
      if (warp >= 2) {
        auto rOther = make_tensor_like(tSrS);
        copy_aligned(tSsS, rOther);
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
          #pragma unroll
          for (int v = 0; v < 4; ++v) {
            int candidate = get<1>(tScS(v, 0, n));
            bool valid = tile_start + candidate < length && ids[tile_start + candidate] >= 0;
            tSrS(v, 0, n) = valid ? (rOther(v, 0, n) + tSrS(v, 0, n)) * (p.sm_scale * LOG2E) : -INFINITY;
          }
        }
        copy_aligned(tSrS, tSsS);
      }
      bar_sync_t<1, 128>();
      bar_arrive_alt<2, 3, 256>(slot);
    }
  } else if (role == Role::mma_pv) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 256;" ::: "memory");
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
    Pipeline::PipelineState read_state;
    for (int tile = 0; tile < tiles; ++tile) {
      int slot = tile & 1;
      bar_sync_alt<2, 3, 256>(slot);
      auto sS = make_tensor(make_smem_ptr(sm.scores[slot] + warp * 8 * 64),
          composition(Swizzle<2, 3, 3>{}, make_layout(
              Layout<Shape<_2, _2, _2>, Stride<_128, _64, _256>>{}, Layout<_64>{})));
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
      // Each thread owns two head columns. Compute their rescaling factors
      // once to avoid repeating shuffles for every feature tile.
      float alpha0 = __shfl_sync(0xffffffff, alpha, (lane % 4) * 8);
      float alpha1 = __shfl_sync(0xffffffff, alpha, (lane % 4) * 8 + 4);
      #pragma unroll
      for (int m = 0; m < 32; ++m) {
        #pragma unroll
        for (int v = 0; v < 4; ++v) {
          rO(v, m, 0) *= (v % 2 == 0 ? alpha0 : alpha1);
        }
      }
      auto rVScale = make_tensor<uint32_t>(Shape<_2, _4>{});  // (group-in-micro, candidate-quartet)
      #pragma unroll
      for (int group = 0; group < 16; ++group) {
        const int micro = group / 4;
        const int stage = read_state.index();
        if (group % 4 == 0) {
          v_pipeline.consumer_wait(read_state);
        }
        if (group % 2 == 0) {
          #pragma unroll
          for (int pack = 0; pack < 4; ++pack) {
            int n = get<1>(tPcP((pack % 2) * 4, 0, pack / 2));
            auto sSF = make_tensor(make_smem_ptr(reinterpret_cast<const uint32_t*>(
                sm.v_sf[slot] + (micro * 16 + n / 4) * 16 + (group % 4 / 2) * 8)), Layout<_2>{});
            copy_aligned(sSF, rVScale(_, pack));
          }
        }
        auto rScaled = make_tensor_like(rP);
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
          #pragma unroll
          for (int quartet = 0; quartet < 2; ++quartet) {
            uint32_t scales = rVScale(group % 2, k * 2 + quartet);
            #pragma unroll
            for (int byte = 0; byte < 4; ++byte) {
              int v = quartet * 4 + byte;
              rScaled(v, 0, k) = rP(v, 0, k) * ue8((scales >> (byte * 8)) & 255);
            }
          }
        }
        // B's K mode is (candidate=32, candidate-group=2). Keep the
        // 64-candidate load tile independent of the P quantization extent.
        float p_scale[64 / PQuantGroup];
        #pragma unroll
        for (int g = 0; g < 64 / PQuantGroup; ++g) {
          float amax = 1.e-10f;
          #pragma unroll
          for (int k = 0; k < PQuantGroup / 32; ++k) {
            #pragma unroll
            for (int v = 0; v < 8; ++v)
              amax = fmaxf(amax, fabsf(rScaled(v, 0, g * (PQuantGroup / 32) + k)));
          }
          p_scale[g] = row_max4(amax) / 448.f;
        }
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
          auto rP8_packed = recast<uint32_t>(rP8(_, _, k));
          float scale = p_scale[k / (PQuantGroup / 32)];
          #pragma unroll
          for (int i = 0; i < size(rP8_packed); ++i)
            rP8_packed(i) = cvt_e4m3x4(rScaled(i * 4, 0, k) / scale,
                rScaled(i * 4 + 1, 0, k) / scale, rScaled(i * 4 + 2, 0, k) / scale,
                rScaled(i * 4 + 3, 0, k) / scale);
        }
        auto sVStage = make_tensor(make_smem_ptr(reinterpret_cast<E4*>(sm.v[stage])), MicroTransposed{});
        auto sV = local_tile(sVStage, Shape<_32, _64>{}, make_coord(group % 4, _0{}));
        auto rV = thr_mma.partition_fragment_A(sV);
        auto tiled_copy = make_tiled_copy_A(Copy_Atom<SM120_U8x16_LDSM_T_RAW, E4>{}, mma);
        auto thr_copy = tiled_copy.get_slice(lane);
        auto rV_cv = thr_copy.retile_D(rV);
        copy(tiled_copy, thr_copy.partition_S(sV), rV_cv);
        auto rPV = thr_mma.partition_fragment_C(make_identity_tensor(Shape<_32, _8>{}));
        #pragma unroll
        for (int g = 0; g < 64 / PQuantGroup; ++g) {
          clear(rPV);
          if constexpr (PQuantGroup == 64)
            gemm(mma, rV, rP8, rPV);
          else
            gemm(mma, rV(_, _, g), rP8(_, _, g), rPV);
          float scale0 = __shfl_sync(0xffffffff, p_scale[g], (lane % 4) * 8);
          float scale1 = __shfl_sync(0xffffffff, p_scale[g], (lane % 4) * 8 + 4);
          #pragma unroll
          for (int m = 0; m < 2; ++m) {
            #pragma unroll
            for (int v = 0; v < 4; ++v)
              rO(v, group * 2 + m, 0) += rPV(v, m, 0) * (v % 2 == 0 ? scale0 : scale1);
          }
        }
        if (group % 4 == 3) {
          v_pipeline.consumer_release(read_state);
          ++read_state;
        }
      }
    }
    const float* cs = nullptr;
    if constexpr (Mega) cs = p.cos_sin + size_t(p.positions[token]) * 64;
    // Normalize including the sink; preserve the original BF16 context boundary.
    float denominator = row_l;
    float out_factor = 1.f;
    float sink = p.sink && h0 + own_head < p.heads ? p.sink[h0 + own_head] * LOG2E : -INFINITY;
    if (sink == INFINITY) {
      out_factor = 0.f;
      denominator = 1.f;
      if ((lane % 4) == 0) p.lse[size_t(token) * p.lse_stride + h0 + own_head] = INFINITY;
    } else if (sink != -INFINITY) {
      float shift = fmaxf(row_m, sink);
      out_factor = isfinite(row_m) ? exp2f(row_m - shift) : 0.f;
      denominator = row_l * out_factor + exp2f(sink - shift);
      if ((lane % 4) == 0) p.lse[size_t(token) * p.lse_stride + h0 + own_head] = (shift + log2f(denominator)) * p.lse_scale;
    } else if ((lane % 4) == 0 && h0 + own_head < p.heads) {
      p.lse[size_t(token) * p.lse_stride + h0 + own_head] = row_l > 0.f ? (row_m + log2f(row_l)) * p.lse_scale : -INFINITY;
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
          if constexpr (Mega) {
            float y = __shfl_xor_sync(0xffffffff, x, 4);
            vals[m * 4 + v] = mega_rotate<false>(x, y, d, cs, true);
          } else {
            int h = h0 + get<1>(coord);
            if (h < p.heads)
              p.bf16_output[(size_t(token) * p.heads + h) * 512 + d] = to_bf16(x);
          }
        }
      }
      if constexpr (Mega) {
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
  } else {
    CUTE_INVALID_CONTROL_PATH("Unreachable prefill role");
  }
}
}
