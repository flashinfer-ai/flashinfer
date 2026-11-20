// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/atom/copy_traits_sm75.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <flashinfer/attention/sparse_mla_sm120/common/zero_row.cuh>
#include <flashinfer/attention/sparse_mla_sm120/kernels/cute_prefill/common.cuh>
namespace flashinfer::sparse_mla_sm120::dsv4_fp8_splitkv_decode {
using namespace cute_prefill;
struct Params : PagedParams {
  bf16* partial;
  float* partial_lse;
  int splits, partial_heads, partial_stride, chunks_per_block;
  size_t indices_stride, extra_indices_stride;
};
using GatherPipeline = cutlass::PipelineTransactionAsync<2>;
template <int HeadGroups>
inline constexpr int Candidates = HeadGroups == 2 ? 64 : 32;
template <int HeadGroups>
struct alignas(128) Shared {
  static constexpr int Tile = Candidates<HeadGroups>;
  GatherPipeline::SharedStorage pipeline;
  alignas(128) uint8_t kv[2][Tile * 528];
  alignas(128) E4 q[HeadGroups][16 * 512];
  uint8_t qs[HeadGroups][16 * 16];
  float scores[HeadGroups][8 * Tile];
  int ids[2][Tile];
};
template <bool Mega, int FixedSplits = 0>
__device__ __forceinline__ void merge_vector(Params const& p, int token, int h, int d, int lane) {
  const int splits = FixedSplits ? FixedSplits : p.splits;
  SparseMlaMegaParams mega{};
  mega.positions = p.positions;
  mega.cos_sin = p.cos_sin;
  mega.output = p.output;
  mega.scales = p.scales;
  const float* stats = p.partial_lse + (size_t(token) * p.partial_heads + h) * p.partial_stride;
  // All lanes own different feature vectors of the same head. Load each
  // split statistic once per warp, then broadcast its weight during PV
  // merge.
  float sink = p.sink ? p.sink[h] * LOG2E : -INFINITY;
  float split_lse = lane < splits ? __ldcg(stats + lane) : -INFINITY;
  float peak = fmaxf(sink, split_lse);
  // Caller-managed schedules can contain more splits than a warp.
  for (int z = lane + 32; z < splits; z += 32) peak = fmaxf(peak, __ldcg(stats + z));
#pragma unroll
  for (int delta = 16; delta > 0; delta /= 2)
    peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, delta));
  float weight = isfinite(split_lse) ? exp2f(split_lse - peak) : 0.f;
  float total = weight;
  for (int z = lane + 32; z < splits; z += 32) {
    float lse = __ldcg(stats + z);
    total += isfinite(lse) ? exp2f(lse - peak) : 0.f;
  }
#pragma unroll
  for (int delta = 16; delta > 0; delta /= 2) total += __shfl_xor_sync(0xffffffff, total, delta);
  if (isfinite(sink)) total += exp2f(sink - peak);
  float acc[8] = {};
  if constexpr (FixedSplits == 8) {
    // Issue all split loads before consuming them to expose memory parallelism.
    uint4 loaded[8];
#pragma unroll
    for (int z = 0; z < splits; ++z)
      loaded[z] = __ldcg(reinterpret_cast<const uint4*>(
          p.partial + ((size_t(token) * p.partial_heads + h) * p.partial_stride + z) * 512 + d));
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
      float w;
      if (splits <= 32) {
        w = __shfl_sync(0xffffffff, weight, z);
      } else {
        float lse = __ldcg(stats + z);
        w = isfinite(lse) ? exp2f(lse - peak) : 0.f;
      }
      uint4 bits = __ldcg(reinterpret_cast<const uint4*>(
          p.partial + ((size_t(token) * p.partial_heads + h) * p.partial_stride + z) * 512 + d));
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

template <int FixedSplits = 0, bool Mega = false>
__global__ void merge(Params p) {
  int token = blockIdx.x, h = blockIdx.y;
  merge_vector<Mega, FixedSplits>(p, token, h, int(threadIdx.x) * 8, int(threadIdx.x) % 32);
}

// Mega follows the attention-side boundary of FlashMLA's decode(nonorm):
//   BF16 Q -> adjacent-pair RoPE on the last 64 features -> BF16 -> Q quantization
//   -> sparse QK/softmax/PV (including sink) -> BF16 context -> inverse RoPE
//   -> FP8 E4M3 output with one UE8M0 scale per 32 features.
// Q RMSNorm, learned projections, cache packing, and index selection are inputs
// to this launch. Mega=false keeps the same attention arithmetic with BF16 I/O.
// A task owns one query and one or more eight-head groups. Low-head tasks
// use four cooperating warps per group: QK partitions candidates and PV
// partitions output features. Sixteen-head tasks use 64-candidate tiles.
// All compute warps participate in pipeline and cluster barriers; stores mask
// inactive heads in the final task.
// Split CTAs publish normalized BF16 context and base-2 LSE, rendezvous, and
// share the final merge/epilogue. Empty chunks publish zero context and -inf LSE.
// Scratch belongs to the invocation; no initialization kernel/counter is needed.
// All compute warps share the sparse gather. BF16 attention publishes partials
// for a separate merge; mega performs cluster finalization in this launch.
// The caller-managed adapter preserves its original scratch strides and KV
// chunk boundaries through partial_heads/partial_stride/chunks_per_block.
template <int HeadGroups, bool Mega = true, int FixedSplits = 0, bool Inline = true,
          bool PartialOnly = false>
__global__ __launch_bounds__((HeadGroups * 4 + 1) * 32, 1) void decode(Params p) {
  constexpr int WarpsPerGroup = 4;
  static_assert(HeadGroups >= 1 && HeadGroups <= 2);
  static_assert(FixedSplits == 0 || FixedSplits == 8);
  constexpr int Tile = Candidates<HeadGroups>;
  const int splits = FixedSplits ? FixedSplits : p.splits;
  extern __shared__ __align__(128) unsigned char raw[];
  auto& sm = *reinterpret_cast<Shared<HeadGroups>*>(raw);
  int lane = threadIdx.x % 32, physical_warp = int(threadIdx.x) / 32 - 1;
  int warp = physical_warp < 0 ? -1 : physical_warp / WarpsPerGroup;
  int worker = physical_warp < 0 ? 0 : physical_warp % WarpsPerGroup;
  int token = blockIdx.x, h0 = blockIdx.y * (HeadGroups * 8) + warp * 8;
  int own_head = lane / 4;
  using Pipeline = GatherPipeline;
  using KVLayout = Layout<Shape<Int<Tile>, _512>, Stride<Int<528>, _1>>;
  using VTLayout = Layout<Shape<_512, Int<Tile>>, Stride<_1, Int<528>>>;
  auto scale_factor = [&](int slot, int group, int n) {
    return sm.kv[slot][n * 528 + 512 + group];
  };
  typename Pipeline::Params pp;
  pp.transaction_bytes = Tile * 528;
  pp.producer_arv_count = 32;
  pp.consumer_arv_count = HeadGroups * WarpsPerGroup * 32;
  pp.dst_blockid = cute::block_rank_in_cluster();
  pp.role = warp < 0 ? Pipeline::ThreadCategory::Producer : Pipeline::ThreadCategory::Consumer;
  Pipeline pipeline(sm.pipeline, pp);
  __syncthreads();
  int main_len = max(0, min(p.topk, p.lengths ? p.lengths[token] : p.topk));
  int extra_len =
      max(0, min(p.extra_topk, p.extra_lengths ? p.extra_lengths[token] : p.extra_topk));
  int main_tiles = (main_len + Tile - 1) / Tile, tiles = main_tiles + (extra_len + Tile - 1) / Tile;
  if (p.chunks_per_block) {
    main_tiles = (main_len + 63) / 64 * (64 / Tile);
    tiles = main_tiles + (extra_len + 63) / 64 * (64 / Tile);
  }
  const int begin = p.chunks_per_block ? int(blockIdx.z) * p.chunks_per_block * (64 / Tile)
                                       : tiles * int(blockIdx.z) / splits;
  const int end = p.chunks_per_block ? min(begin + p.chunks_per_block * (64 / Tile), tiles)
                                     : tiles * (int(blockIdx.z) + 1) / splits;
  if (warp < 0) {
    auto write_state = cutlass::make_producer_start_state<Pipeline>();
    for (int tile = begin; tile < end; ++tile) {
      pipeline.producer_acquire(write_state);
      int slot = write_state.index();
      // Phase 2: gather packed KV once for every head group in this CTA.
      bool extra = tile >= main_tiles;
      int start = (extra ? tile - main_tiles : tile) * Tile;
      int length = extra ? extra_len : main_len;
      size_t width = extra ? p.extra_indices_stride : p.indices_stride;
      const int* ids = (extra ? p.extra_indices : p.indices) + size_t(token) * width;
      for (int n = lane; n < Tile; n += 32)
        sm.ids[slot][n] = start + n < length ? ids[start + n] : -1;
      __syncwarp();

      if (lane == 0) pipeline.producer_expect_transaction(write_state);
      __syncwarp();
      for (int n = lane; n < Tile; n += 32) {
        int id = sm.ids[slot][n];
        auto* barrier = pipeline.producer_get_barrier(write_state);
        const uint8_t* data =
            id >= 0 ? cache_row<Inline, false>(p, extra, id) : sparse_mla_zero_row;
        if constexpr (Inline) {
          SM90_BULK_COPY_G2S::copy(data, barrier, sm.kv[slot] + n * 528, 528);
        } else {
          const uint8_t* sf = id >= 0 ? cache_row<Inline, true>(p, extra, id) : sparse_mla_zero_row;
          SM90_BULK_COPY_G2S::copy(data, barrier, sm.kv[slot] + n * 528, 512);
          SM90_BULK_COPY_G2S::copy(sf, barrier, sm.kv[slot] + n * 528 + 512, 16);
        }
      }
      pipeline.producer_commit(write_state);

      ++write_state;
    }
    pipeline.producer_tail(write_state);
    // All threads participate in the cluster rendezvous. Loader and compute
    // warps arrive at distinct program counters, so do not use the .aligned
    // cluster barrier helpers here.
    if (!PartialOnly && splits > 1) {
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
                  composition(Swizzle<3, 4, 5>{}, Layout<Shape<_16, _512>, Stride<_512, _1>>{}));
  auto cP = make_identity_tensor(Shape<_8, Int<Tile>>{});  // (head, candidate)
  auto tPcP = thr_mma.partition_B(cP);
  auto cO = make_identity_tensor(Shape<Int<512 / WarpsPerGroup>, _8>{});  // (feature, head)
  auto tOcO = thr_mma.partition_C(cO);
  auto rO = thr_mma.partition_fragment_C(cO);
  clear(rO);
  if (worker == 0) {
    // Phase 1: one warp per head group rotates and quantizes Q.
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

    for (int vec = lane; vec < 8 * 32; vec += 32)
      *reinterpret_cast<uint4*>(&sQ(8 + vec / 32, (vec % 32) * 16)) = uint4{};
    if (lane < 8) *reinterpret_cast<uint4*>(sm.qs[warp] + (8 + lane) * 16) = uint4{};

    __syncwarp();
  }
  cutlass::arch::NamedBarrier::sync(HeadGroups * WarpsPerGroup * 32, 0);
  float row_m = -INFINITY, row_l = 0.f;
  typename Pipeline::PipelineState read_state;
  for (int tile = begin; tile < end; ++tile) {
    pipeline.consumer_wait(read_state);
    int slot = read_state.index();

    // Four warps partition candidates. GEMM is (head, candidate, feature).

    auto cScore = make_identity_tensor(Shape<_16, Int<Tile / 4>>{});
    auto tScore = thr_mma.partition_C(cScore);
    auto rScore = thr_mma.partition_fragment_C(cScore);
    clear(rScore);
    auto sKV = make_tensor(make_smem_ptr(reinterpret_cast<E4*>(sm.kv[slot])), KVLayout{});
#pragma unroll
    for (int group = 0; group < 16; ++group) {
      auto sQt = local_tile(sQ, Shape<_16, _32>{}, make_coord(_0{}, group));
      auto sK = local_tile(sKV, Shape<Int<Tile / 4>, _32>{}, make_coord(worker, group));
      auto rQ = thr_mma.partition_fragment_A(sQt);
      auto rK = thr_mma.partition_fragment_B(sK);
      auto cq = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, E4>{}, mma);
      auto ck = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, E4>{}, mma);
      auto tq = cq.get_slice(lane);
      auto tk = ck.get_slice(lane);
      auto rqcv = tq.retile_D(rQ);
      auto rkcv = tk.retile_D(rK);
      copy(cq, tq.partition_S(sQt), rqcv);
      copy(ck, tk.partition_S(sK), rkcv);
      auto sfA = make_tensor<uint8_t>(Layout<_32, _0>{});
      auto sfB = make_tensor<uint8_t>(Layout<_32, _0>{});
      sfA(0) = sm.qs[warp][(lane / 4 + (lane % 2) * 8) * 16 + group];
#pragma unroll
      for (int n = 0; n < Tile / 32; ++n) {
        sfB(0) = scale_factor(slot, group, worker * (Tile / 4) + n * 8 + lane / 4);
        auto score = rScore(_, 0, n);
        scaled_mma.call(score, make_zip_tensor(rQ(_, 0, 0), sfA), make_zip_tensor(rK(_, n, 0), sfB),
                        score);
      }
    }
#pragma unroll
    for (int i = 0; i < size(rScore); ++i) {
      auto c = tScore(i);
      int h = get<0>(c), n = worker * (Tile / 4) + get<1>(c);
      if (h < 8)
        sm.scores[warp][h * Tile + n] =
            sm.ids[slot][n] >= 0 ? rScore(i) * (p.sm_scale * LOG2E) : -INFINITY;
    }

    cutlass::arch::NamedBarrier::sync(HeadGroups * WarpsPerGroup * 32, 0);
    auto sP = make_tensor(make_smem_ptr(sm.scores[warp]),
                          Layout<Shape<_8, Int<Tile>>, Stride<Int<Tile>, _1>>{});
    // Phase 4: warp-local online softmax. Four lanes cooperate on one head.
    auto rP = make_tensor<float>(shape(tPcP));
    copy(thr_mma.partition_B(sP), rP);
    // Scores have one shared buffer, unlike the two-stage KV ring. Every
    // consumer must finish reading before any warp publishes the next tile.
    if (tile + 1 < end) cutlass::arch::NamedBarrier::sync(HeadGroups * WarpsPerGroup * 32, 0);
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
      for (int m = 0; m < 32 / WarpsPerGroup; ++m)
        for (int v = 0; v < 4; ++v) rO(v, m, 0) *= v % 2 ? a1 : a0;
    }
    // Phase 5: PV. Fold per-candidate V scales into P before FP8 MMA.
    auto sVT = make_tensor(make_smem_ptr(reinterpret_cast<E4*>(sm.kv[slot])), VTLayout{});
#pragma unroll
    for (int local_group = 0; local_group < 16 / WarpsPerGroup; ++local_group) {
      int group = worker * (16 / WarpsPerGroup) + local_group;
      auto rScaled = make_tensor_like(rP);
      float amax = 1.e-10f;
#pragma unroll
      for (int i = 0; i < size(rP); ++i) {
        int n = get<1>(tPcP(i));
        rScaled(i) = rP(i) * ue8(scale_factor(slot, group, n));
        amax = fmaxf(amax, fabsf(rScaled(i)));
      }
      float ps = row_max4(amax) / 448.f;
      auto rP8 = make_tensor<E4>(shape(rP));
      auto packed = recast<uint32_t>(rP8);
#pragma unroll
      for (int i = 0; i < size(packed); ++i)
        packed(i) = cvt_e4m3x4(rScaled(i * 4) / ps, rScaled(i * 4 + 1) / ps,
                               rScaled(i * 4 + 2) / ps, rScaled(i * 4 + 3) / ps);
      auto sV = local_tile(sVT, Shape<_32, Int<Tile>>{}, make_coord(group, _0{}));
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
        for (int v = 0; v < 4; ++v)
          rO(v, local_group * 2 + m, 0) += rPV(v, m, 0) * (v % 2 ? s1 : s0);
    }
    // Last consumer releases this stage to the loader; other stages can
    // overlap.
    pipeline.consumer_release(read_state);
    ++read_state;
  }
  // Phase 6a: independent local-KV tasks publish normalized BF16 partials.
  // This cluster is one query/head tile, never distributed context parallelism.
  if (splits > 1 || (PartialOnly && p.partial_stride > 0)) {
    // All compute warps finish KV consumption before recycling the payload
    // ring.
    cutlass::arch::NamedBarrier::sync(HeadGroups * WarpsPerGroup * 32, 0);
    using PartialLayout = decltype(composition(
        Swizzle<3, 3, 6>{}, Layout<Shape<Int<HeadGroups * 8>, _512>, Stride<_512, _1>>{}));
    auto staged = make_tensor(make_smem_ptr(reinterpret_cast<bf16*>(sm.kv)), PartialLayout{});
    float inv = row_l > 0.f ? 1.f / row_l : 0.f;
    float inv0 = __shfl_sync(0xffffffff, inv, (lane % 4) * 8),
          inv1 = __shfl_sync(0xffffffff, inv, (lane % 4) * 8 + 4);
#pragma unroll
    for (int m = 0; m < 32 / WarpsPerGroup; ++m) {
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        auto c = tOcO(v, m, 0);
        int d = get<0>(c) + worker * (512 / WarpsPerGroup);
        staged(warp * 8 + get<1>(c), d) = to_bf16(rO(v, m, 0) * (v % 2 ? inv1 : inv0));
      }
    }
    cutlass::arch::NamedBarrier::sync(HeadGroups * WarpsPerGroup * 32, 0);
    for (int vec = physical_warp * 32 + lane; vec < HeadGroups * 8 * 64;
         vec += HeadGroups * WarpsPerGroup * 32) {
      int local_h = vec / 64, d = (vec % 64) * 8;
      int h = blockIdx.y * (HeadGroups * 8) + local_h;
      if (h < p.heads)
        *reinterpret_cast<uint4*>(
            p.partial +
            ((size_t(token) * p.partial_heads + h) * p.partial_stride + blockIdx.z) * 512 + d) =
            *reinterpret_cast<uint4*>(&staged(local_h, d));
    }
    if (worker == 0 && lane % 4 == 0 && h0 + own_head < p.heads)
      p.partial_lse[(size_t(token) * p.partial_heads + h0 + own_head) * p.partial_stride +
                    blockIdx.z] = row_l > 0.f ? row_m + log2f(row_l) : -INFINITY;
    if constexpr (PartialOnly) return;
    asm volatile("barrier.cluster.arrive.release; barrier.cluster.wait.acquire;" ::: "memory");
    // Phase 6b: the same CTAs divide final output vectors after publishing all
    // partials to global scratch. Each vector owner reads every split; the loader
    // warp exits after the rendezvous above. Ownership changes, for example,
    // for a 16-head task with four splits (256 compute threads per CTA):
    //
    //              Before barrier                 After barrier
    //   CTA 0: KV subset 0 -> all 16 heads    merge + epilogue for heads  0-3
    //   CTA 1: KV subset 1 -> all 16 heads    merge + epilogue for heads  4-7
    //   CTA 2: KV subset 2 -> all 16 heads    merge + epilogue for heads  8-11
    //   CTA 3: KV subset 3 -> all 16 heads    merge + epilogue for heads 12-15
    //
    // The loop below statically partitions flattened [head, feature/8] vectors;
    // other head-group/split counts use the same rule, not these fixed ranges.
    int head_base = blockIdx.y * (HeadGroups * 8);
    for (int vec = (int(threadIdx.x) - 32) + int(blockIdx.z) * HeadGroups * WarpsPerGroup * 32;
         vec < HeadGroups * 8 * 64; vec += HeadGroups * WarpsPerGroup * 32 * splits) {
      int h = head_base + vec / 64, d = (vec % 64) * 8;
      if (h >= p.heads) continue;
      merge_vector<Mega, FixedSplits>(p, token, h, d, lane);
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
    if (worker == 0 && (lane % 4) == 0)
      p.lse[size_t(token) * p.lse_stride + h0 + own_head] = INFINITY;
  } else if (sink != -INFINITY) {
    float shift = fmaxf(row_m, sink);
    out_factor = isfinite(row_m) ? exp2f(row_m - shift) : 0.f;
    denominator = row_l * out_factor + exp2f(sink - shift);
    if (worker == 0 && (lane % 4) == 0)
      p.lse[size_t(token) * p.lse_stride + h0 + own_head] =
          (shift + log2f(denominator)) * p.lse_scale;
  } else if (worker == 0 && (lane % 4) == 0) {
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
  for (int local_group = 0; local_group < 16 / WarpsPerGroup; ++local_group) {
    float vals[8];
#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        auto coord = tOcO(v, local_group * 2 + m, 0);
        int d = get<0>(coord) + worker * (512 / WarpsPerGroup);
        float x = to_float(to_bf16(rO(v, local_group * 2 + m, 0) * (v % 2 == 0 ? norm0 : norm1)));
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
          auto coord = tOcO(v, local_group * 2 + m, 0);
          int d = get<0>(coord) + worker * (512 / WarpsPerGroup), h = h0 + get<1>(coord);
          size_t offset = (size_t(token) * p.heads + h) * 512 + d;
          p.output[offset] = quantize_e4m3_byte(vals[j * 2 + head_col] * inv);
          if ((d % 32) == 0) p.scales[offset / 32] = bits >> 23;
        }
      }
    }
  }
}
}  // namespace flashinfer::sparse_mla_sm120::dsv4_fp8_splitkv_decode
