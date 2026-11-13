// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cute/arch/copy_sm80.hpp>
#include <cute/atom/copy_traits_sm75.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <flashinfer/attention/sparse_mla_sm120/kernels/cute_prefill/common.cuh>
namespace flashinfer::sparse_mla_sm120::dsv4_fp8_splitkv_decode {
using namespace cute_prefill;
struct Params : cute_prefill::Params {
  bf16* bf16_output;
  size_t lse_stride;
  float lse_scale;
  bf16* partial;
  float* partial_lse;
  int splits;
};
using KVLayout =
    decltype(composition(Swizzle<3, 4, 5>{}, Layout<Shape<_32, _512>, Stride<_512, _1>>{}));
using VTLayout =
    decltype(composition(Swizzle<3, 4, 5>{}, Layout<Shape<_512, _32>, Stride<_1, _512>>{}));
using Pipeline = cutlass::PipelineAsync<2>;
template <int Warps>
struct alignas(128) Shared {
  Pipeline::SharedStorage pipeline;
  uint8_t kv[2][32 * 512], sf[2][32 * 16];
  E4 q[Warps][8 * 512];
  uint8_t qs[Warps][8 * 16];
  float scores[Warps][8 * 32];
  int ids[2][32];
};
// Mega follows the attention-side boundary of FlashMLA's decode(nonorm):
//   BF16 Q -> adjacent-pair RoPE on the last 64 features -> BF16 -> Q quantization
//   -> sparse QK/softmax/PV (including sink) -> BF16 context -> inverse RoPE
//   -> FP8 E4M3 output with one UE8M0 scale per 32 features.
// Q RMSNorm, learned projections, cache packing, and index selection are inputs
// to this launch. Mega=false keeps the same attention arithmetic with BF16 I/O.
// A task owns one query, 8 heads/warp, and its selected KV range.
// H=64 uses 24/24/16-head tasks. All warps, including the final inactive head
// warp, participate in the pipelines and cluster barriers; stores mask H tails.
// Split CTAs publish normalized BF16 context and base-2 LSE, rendezvous, and
// share the final merge/epilogue. Empty chunks publish zero context and -inf LSE.
// Scratch belongs to the invocation; no initialization kernel/counter is needed.
// Every warp executes QK -> online softmax -> PV. All warps share the sparse
// gather, so there are no separate QK/PV roles or register handoff barriers.
template <int Warps, bool Mega = true, int FixedSplits = 0>
__global__ __launch_bounds__((Warps + 1) * 32, 1) void decode(Params p) {
  static_assert(Warps >= 1 && Warps <= 4);
  static_assert(FixedSplits == 0 || FixedSplits == 8);
  const int splits = FixedSplits ? FixedSplits : p.splits;
  extern __shared__ __align__(128) unsigned char raw[];
  auto& sm = *reinterpret_cast<Shared<Warps>*>(raw);
  int lane = threadIdx.x % 32, warp = int(threadIdx.x) / 32 - 1;
  int token = blockIdx.x, h0 = blockIdx.y * (Warps * 8) + warp * 8;
  int own_head = lane / 4;
  Pipeline::Params pp;
  pp.producer_arv_count = 32;
  pp.consumer_arv_count = Warps * 32;
  pp.dst_blockid = cute::block_rank_in_cluster();
  pp.role = warp < 0 ? Pipeline::ThreadCategory::Producer : Pipeline::ThreadCategory::Consumer;
  Pipeline pipeline(sm.pipeline, pp);
  __syncthreads();
  int main_len = max(0, min(p.topk, p.lengths ? p.lengths[token] : p.topk));
  int extra_len =
      max(0, min(p.extra_topk, p.extra_lengths ? p.extra_lengths[token] : p.extra_topk));
  int main_tiles = (main_len + 31) / 32, tiles = main_tiles + (extra_len + 31) / 32;
  const int begin = tiles * int(blockIdx.z) / splits, end = tiles * (int(blockIdx.z) + 1) / splits;
  if (warp < 0) {
    auto write_state = cutlass::make_producer_start_state<Pipeline>();
    for (int tile = begin; tile < end; ++tile) {
      pipeline.producer_acquire(write_state);
      int slot = write_state.index();
      // Phase 2: gather packed KV once for every head group in this CTA.
      bool extra = tile >= main_tiles;
      int start = (extra ? tile - main_tiles : tile) * 32;
      int length = extra ? extra_len : main_len, width = extra ? p.extra_topk : p.topk;
      const int* ids = (extra ? p.extra_indices : p.indices) + size_t(token) * width;
      const uint8_t* cache = extra ? p.extra_kv : p.kv;
      int stride = extra ? p.extra_stride : p.kv_stride;
      sm.ids[slot][lane] = start + lane < length ? ids[start + lane] : -1;
      __syncwarp();
      for (int i = lane; i < 32 * 32; i += 32) {
        int n = i / 32, d = (i % 32) * 16, id = sm.ids[slot][n];
        auto* dst = reinterpret_cast<cute::uint128_t*>(sm.kv[slot] + KVLayout{}(n, d));
        const uint8_t* src = cache + size_t(max(id, 0)) * stride + d;
        SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>::copy(
            *reinterpret_cast<const cute::uint128_t*>(src), *dst, id >= 0);
      }
      {
        int id = sm.ids[slot][lane];
        uint4 sf = id >= 0
                       ? *reinterpret_cast<const uint4*>(cache + size_t(max(id, 0)) * stride + 512)
                       : uint4{};
        auto bytes =
            make_tensor(make_rmem_ptr(reinterpret_cast<const uint8_t*>(&sf)), Layout<_16>{});
#pragma unroll
        for (int g = 0; g < 16; ++g) sm.sf[slot][g * 32 + lane] = bytes(g);
        __syncwarp();
      }
      pipeline.producer_commit(write_state, [](uint64_t const* barrier) {
        unsigned address = __shfl_sync(0xffffffff, cute::cast_smem_ptr_to_uint(barrier), 0);
        cutlass::arch::cpasync_barrier_arrive(
            reinterpret_cast<uint64_t const*>(__cvta_shared_to_generic(address)));
      });
      ++write_state;
    }
    pipeline.producer_tail(write_state);
    // All threads participate in the cluster rendezvous. Loader and compute
    // warps arrive at distinct program counters, so do not use the .aligned
    // cluster barrier helpers here.
    if (splits > 1) {
      asm volatile("barrier.cluster.arrive.release; barrier.cluster.wait.acquire;" ::: "memory");
    }
    return;
  }
  Mma mma;
  using ScaledOp = SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<E4, E4, float, float_ue8m0_t, 32>;
  MMA_Atom<ScaledOp> scaled_mma;
  auto thr_mma = mma.get_slice(lane);
  auto sQ =
      make_tensor(make_smem_ptr(sm.q[warp]),
                  composition(Swizzle<3, 4, 5>{}, Layout<Shape<_8, _512>, Stride<_512, _1>>{}));
  auto cS = make_identity_tensor(Shape<_32, _8>{});  // (candidate, head)
  auto tScS = thr_mma.partition_C(cS);
  auto cP = make_identity_tensor(Shape<_8, _32>{});  // (head, candidate)
  auto tPcP = thr_mma.partition_B(cP);
  auto cO = make_identity_tensor(Shape<_512, _8>{});  // (feature, head)
  auto tOcO = thr_mma.partition_C(cO);
  auto rO = thr_mma.partition_fragment_C(cO);
  clear(rO);
  // Phase 1: each four-lane group rotates and quantizes one query head.
  const float* qcs = nullptr;
  if constexpr (Mega) qcs = p.cos_sin + size_t(p.positions[token]) * 64;
#pragma unroll
  for (int group = 0; group < 16; ++group) {
    float vals[8], amax = 1.e-4f;
    const int d0 = group * 32 + (lane % 4) * 8;
    const auto qbits = h0 + own_head < p.heads
                           ? *reinterpret_cast<const uint4*>(
                                 p.q + (size_t(token) * p.heads + h0 + own_head) * 512 + d0)
                           : uint4{};
    auto qvals = make_tensor(make_rmem_ptr(reinterpret_cast<const bf16*>(&qbits)), Layout<_8>{});
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float x = to_float(qvals(i));
      if constexpr (Mega) {
        if (group >= 14)
          x = to_float(to_bf16(mega_rotate<false>(x, to_float(qvals(i ^ 1)), d0 + i, qcs)));
      }
      vals[i] = x;
      amax = fmaxf(amax, fabsf(x));
    }
    amax = row_max4(amax);
    uint32_t bits = (__float_as_uint(amax / 448.f) + 0x007fffff) & 0x7f800000;
    float scale = __uint_as_float(bits);
    if (lane % 4 == 0) sm.qs[warp][own_head * 16 + group] = bits >> 23;
    uint2 packed;
    packed.x = cvt_e4m3x4(vals[0] / scale, vals[1] / scale, vals[2] / scale, vals[3] / scale);
    packed.y = cvt_e4m3x4(vals[4] / scale, vals[5] / scale, vals[6] / scale, vals[7] / scale);
    *reinterpret_cast<uint2*>(&sQ(own_head, group * 32 + (lane % 4) * 8)) = packed;
  }
  __syncwarp();
  float row_m = -INFINITY, row_l = 0.f;
  Pipeline::PipelineState read_state;
  for (int tile = begin; tile < end; ++tile) {
    pipeline.consumer_wait(read_state);
    int slot = read_state.index();
    // Phase 3: QK. GEMM modes M=candidates, N=heads, K=features.
    // Each 32-feature product uses the native cache and query scale factors.
    auto rS = thr_mma.partition_fragment_C(cS);
    clear(rS);
    auto sKV = make_tensor(make_smem_ptr(reinterpret_cast<E4*>(sm.kv[slot])), KVLayout{});
#pragma unroll
    for (int group = 0; group < 16; ++group) {
      auto sK = local_tile(sKV, Shape<_32, _32>{}, make_coord(_0{}, group));
      auto sQt = local_tile(sQ, Shape<_8, _32>{}, make_coord(_0{}, group));
      auto rK = thr_mma.partition_fragment_A(sK);
      auto rQ = thr_mma.partition_fragment_B(sQt);
      auto ck = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, E4>{}, mma);
      auto cq = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, E4>{}, mma);
      auto tk = ck.get_slice(lane);
      auto tq = cq.get_slice(lane);
      auto rkcv = tk.retile_D(rK);
      auto rqcv = tq.retile_D(rQ);
      copy(ck, tk.partition_S(sK), rkcv);
      copy(cq, tq.partition_S(sQt), rqcv);
      // Native MXFP8 QK consumes the cache/query UE8M0 scales directly.
      // SF layouts map lane/4 to a row, with A's lane bit0 selecting row+8.
      auto sfA = make_tensor<uint8_t>(Layout<_32, _0>{});
      auto sfB = make_tensor<uint8_t>(Layout<_32, _0>{});
      sfB(0) = sm.qs[warp][(lane / 4) * 16 + group];
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        int n = m * 16 + lane / 4 + (lane % 2) * 8;
        sfA(0) = sm.sf[slot][group * 32 + n];
        auto rScore = rS(_, m, 0);
        scaled_mma.call(rScore, make_zip_tensor(rK(_, m, 0), sfA),
                        make_zip_tensor(rQ(_, 0, 0), sfB), rScore);
      }
    }
    auto sP =
        make_tensor(make_smem_ptr(sm.scores[warp]), Layout<Shape<_8, _32>, Stride<_32, _1>>{});
#pragma unroll
    for (int i = 0; i < size(rS); ++i) {
      auto c = tScS(i);
      int n = get<0>(c), h = get<1>(c);
      sP(h, n) = sm.ids[slot][n] >= 0 ? rS(i) * (p.sm_scale * LOG2E) : -INFINITY;
    }
    __syncwarp();
    // Phase 4: warp-local online softmax. Four lanes cooperate on one head.
    auto rP = make_tensor<float>(shape(tPcP));
    copy(thr_mma.partition_B(sP), rP);
    float tile_m = -INFINITY;
#pragma unroll
    for (int i = 0; i < size(rP); ++i) tile_m = fmaxf(tile_m, rP(i));
    float new_m = fmaxf(row_m, row_max4(tile_m));
    float alpha = isfinite(row_m) ? exp2f(row_m - new_m) : 0.f, tile_l = 0.f;
#pragma unroll
    for (int i = 0; i < size(rP); ++i) {
      rP(i) = isfinite(rP(i)) ? exp2f(rP(i) - new_m) : 0.f;
      tile_l += rP(i);
    }
    row_l = row_l * alpha + row_sum4(tile_l);
    row_m = new_m;
    // The first tile starts from an all-zero accumulator.
    if (tile != begin) {
      float a0 = __shfl_sync(0xffffffff, alpha, (lane % 4) * 8),
            a1 = __shfl_sync(0xffffffff, alpha, (lane % 4) * 8 + 4);
#pragma unroll
      for (int m = 0; m < 32; ++m)
        for (int v = 0; v < 4; ++v) rO(v, m, 0) *= v % 2 ? a1 : a0;
    }
    // Phase 5: PV. Fold per-candidate V scales into P before FP8 MMA.
    auto sVT = make_tensor(make_smem_ptr(reinterpret_cast<E4*>(sm.kv[slot])), VTLayout{});
#pragma unroll
    for (int group = 0; group < 16; ++group) {
      auto rScaled = make_tensor_like(rP);
      float amax = 1.e-10f;
#pragma unroll
      for (int i = 0; i < size(rP); ++i) {
        int n = get<1>(tPcP(i));
        rScaled(i) = rP(i) * ue8(sm.sf[slot][group * 32 + n]);
        amax = fmaxf(amax, fabsf(rScaled(i)));
      }
      float ps = row_max4(amax) / 448.f;
      auto rP8 = make_tensor<E4>(shape(rP));
      auto packed = recast<uint32_t>(rP8);
#pragma unroll
      for (int i = 0; i < size(packed); ++i)
        packed(i) = cvt_e4m3x4(rScaled(i * 4) / ps, rScaled(i * 4 + 1) / ps,
                               rScaled(i * 4 + 2) / ps, rScaled(i * 4 + 3) / ps);
      auto sV = local_tile(sVT, Shape<_32, _32>{}, make_coord(group, _0{}));
      auto rV = thr_mma.partition_fragment_A(sV);
      auto tc = make_tiled_copy_A(Copy_Atom<SM120_U8x16_LDSM_T_RAW, E4>{}, mma);
      auto copy_thread = tc.get_slice(lane);
      auto rVcv = copy_thread.retile_D(rV);
      copy(tc, copy_thread.partition_S(sV), rVcv);
      auto rPV = thr_mma.partition_fragment_C(make_identity_tensor(Shape<_32, _8>{}));
      clear(rPV);
      gemm(mma, rV, rP8, rPV);
      float s0 = __shfl_sync(0xffffffff, ps, (lane % 4) * 8),
            s1 = __shfl_sync(0xffffffff, ps, (lane % 4) * 8 + 4);
#pragma unroll
      for (int m = 0; m < 2; ++m)
        for (int v = 0; v < 4; ++v) rO(v, group * 2 + m, 0) += rPV(v, m, 0) * (v % 2 ? s1 : s0);
    }
    // Last consumer releases this stage to the loader; other stages can
    // overlap.
    pipeline.consumer_release(read_state);
    ++read_state;
  }
  // Phase 6a: independent local-KV tasks publish normalized BF16 partials.
  // This cluster is one query/head tile, never distributed context parallelism.
  if (splits > 1) {
    // All compute warps finish KV consumption before recycling the payload
    // ring.
    cutlass::arch::NamedBarrier::sync(Warps * 32, 0);
    using PartialLayout = decltype(composition(
        Swizzle<3, 3, 6>{}, Layout<Shape<Int<Warps * 8>, _512>, Stride<_512, _1>>{}));
    auto staged = make_tensor(make_smem_ptr(reinterpret_cast<bf16*>(sm.kv)), PartialLayout{});
    float inv = row_l > 0.f ? 1.f / row_l : 0.f;
    float inv0 = __shfl_sync(0xffffffff, inv, (lane % 4) * 8),
          inv1 = __shfl_sync(0xffffffff, inv, (lane % 4) * 8 + 4);
#pragma unroll
    for (int m = 0; m < 32; ++m) {
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        auto c = tOcO(v, m, 0);
        int d = get<0>(c);
        staged(warp * 8 + get<1>(c), d) = to_bf16(rO(v, m, 0) * (v % 2 ? inv1 : inv0));
      }
    }
    cutlass::arch::NamedBarrier::sync(Warps * 32, 0);
    for (int vec = warp * 32 + lane; vec < Warps * 8 * 64; vec += Warps * 32) {
      int local_h = vec / 64, d = (vec % 64) * 8;
      int h = blockIdx.y * (Warps * 8) + local_h;
      if (h < p.heads)
        *reinterpret_cast<uint4*>(p.partial +
                                  ((size_t(token) * p.heads + h) * splits + blockIdx.z) * 512 + d) =
            *reinterpret_cast<uint4*>(&staged(local_h, d));
    }
    if (lane % 4 == 0 && h0 + own_head < p.heads)
      p.partial_lse[(size_t(token) * p.heads + h0 + own_head) * splits + blockIdx.z] =
          row_l > 0.f ? row_m + log2f(row_l) : -INFINITY;
    asm volatile("barrier.cluster.arrive.release; barrier.cluster.wait.acquire;" ::: "memory");
    // Phase 6b: all cluster CTAs share the final merge and fused epilogue.
    SparseMlaMegaParams mega{};
    mega.positions = p.positions;
    mega.cos_sin = p.cos_sin;
    mega.output = p.output;
    mega.scales = p.scales;
    int head_base = blockIdx.y * (Warps * 8);
    for (int vec = (int(threadIdx.x) - 32) + int(blockIdx.z) * Warps * 32; vec < Warps * 8 * 64;
         vec += Warps * 32 * splits) {
      int h = head_base + vec / 64, d = (vec % 64) * 8;
      if (h >= p.heads) continue;
      const float* stats = p.partial_lse + (size_t(token) * p.heads + h) * splits;
      // All lanes own different feature vectors of the same head. Load each
      // split statistic once per warp, then broadcast its weight during PV
      // merge.
      float sink = p.sink ? p.sink[h] * LOG2E : -INFINITY;
      float split_lse = lane < splits ? __ldcg(stats + lane) : -INFINITY;
      float peak = fmaxf(sink, split_lse);
#pragma unroll
      for (int delta = 16; delta > 0; delta /= 2)
        peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, delta));
      float weight = isfinite(split_lse) ? exp2f(split_lse - peak) : 0.f;
      float total = weight;
#pragma unroll
      for (int delta = 16; delta > 0; delta /= 2)
        total += __shfl_xor_sync(0xffffffff, total, delta);
      if (isfinite(sink)) total += exp2f(sink - peak);
      float acc[8] = {};
      if constexpr (FixedSplits == 8) {
        // Issue all split loads before consuming them to expose memory parallelism.
        uint4 loaded[8];
#pragma unroll
        for (int z = 0; z < splits; ++z)
          loaded[z] = __ldcg(reinterpret_cast<const uint4*>(
              p.partial + ((size_t(token) * p.heads + h) * splits + z) * 512 + d));
#pragma unroll
        for (int z = 0; z < splits; ++z) {
          float w = __shfl_sync(0xffffffff, weight, z);
          auto* v = reinterpret_cast<const bf16*>(&loaded[z]);
#pragma unroll
          for (int i = 0; i < 8; ++i) acc[i] += to_float(v[i]) * w;
        }
      } else {
#pragma unroll
        for (int z = 0; z < splits; ++z) {
          float w = __shfl_sync(0xffffffff, weight, z);
          uint4 bits = __ldcg(reinterpret_cast<const uint4*>(
              p.partial + ((size_t(token) * p.heads + h) * splits + z) * 512 + d));
          auto* v = reinterpret_cast<const bf16*>(&bits);
#pragma unroll
          for (int i = 0; i < 8; ++i) acc[i] += to_float(v[i]) * w;
        }
      }
      float norm = total > 0.f ? 1.f / total : 0.f;
      uint4 packed;
      auto* v = reinterpret_cast<bf16*>(&packed);
#pragma unroll
      for (int i = 0; i < 8; ++i) v[i] = to_bf16(acc[i] * norm);
      if constexpr (Mega) {
        mega_store8(packed, (size_t(token) * p.heads + h) * 512 + d, token, mega);
      } else {
        *reinterpret_cast<uint4*>(p.bf16_output + (size_t(token) * p.heads + h) * 512 + d) = packed;
      }
      if (d == 0)
        p.lse[size_t(token) * p.lse_stride + h] =
            sink == INFINITY ? INFINITY
                             : (total > 0.f ? (peak + log2f(total)) * p.lse_scale : -INFINITY);
    }
    return;
  }
  if (h0 >= p.heads) return;
  // Phase 6: sink normalization, BF16 context rounding, inverse RoPE and FP8.
  const float* cs = nullptr;
  if constexpr (Mega) cs = p.cos_sin + size_t(p.positions[token]) * 64;
  // The sink contributes to the softmax denominator but has no value vector.
  // Write LSE in the caller's units; an empty row without a sink gets -infinity
  // and zero output. Normalize at the BF16 context rounding boundary.
  float denominator = row_l;
  float out_factor = 1.f;
  float sink = p.sink ? p.sink[h0 + own_head] * LOG2E : -INFINITY;
  if (sink == INFINITY) {
    out_factor = 0.f;
    denominator = 1.f;
    if ((lane % 4) == 0) p.lse[size_t(token) * p.lse_stride + h0 + own_head] = INFINITY;
  } else if (sink != -INFINITY) {
    float shift = fmaxf(row_m, sink);
    out_factor = isfinite(row_m) ? exp2f(row_m - shift) : 0.f;
    denominator = row_l * out_factor + exp2f(sink - shift);
    if ((lane % 4) == 0)
      p.lse[size_t(token) * p.lse_stride + h0 + own_head] =
          (shift + log2f(denominator)) * p.lse_scale;
  } else if ((lane % 4) == 0) {
    p.lse[size_t(token) * p.lse_stride + h0 + own_head] =
        row_l > 0.f ? (row_m + log2f(row_l)) * p.lse_scale : -INFINITY;
  }
  // Mega=false writes BF16 context. Mega=true applies inverse RoPE to the
  // normalized BF16 context, then emits FP8 output and one scale per 32
  // features. Whole inactive head warps have exited; output stores need no
  // predicates.
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
        for (int delta = 4; delta <= 16; delta *= 2)
          amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, delta));
        uint32_t bits = (__float_as_uint(amax / 448.f) + 0x007fffff) & 0x7f800000;
        float inv = __uint_as_float(0x7f000000u - bits);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          int m = j / 2, v = (j % 2) * 2 + head_col;
          auto coord = tOcO(v, group * 2 + m, 0);
          int d = get<0>(coord), h = h0 + get<1>(coord);
          size_t offset = (size_t(token) * p.heads + h) * 512 + d;
          p.output[offset] = quantize_e4m3_byte(vals[j * 2 + head_col] * inv);
          if ((d % 32) == 0) p.scales[offset / 32] = bits >> 23;
        }
      }
    }
  }
}
}  // namespace flashinfer::sparse_mla_sm120::dsv4_fp8_splitkv_decode
