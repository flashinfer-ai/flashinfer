/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 ******************************************************************************/
// CollectiveMainloopFwd — load, mma, softmax for fused attention (Step 2: BSA-aligned)
#pragma once

#include <math_constants.h>

#include "cute/algorithm/cooperative_copy.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/detail/sm100_tmem_helper.hpp"
#include "pipeline.hpp"
#include "softmax.h"
#include "utils.h"

namespace flash {

namespace cute = ::cute;

template <bool HasVariableBlockNums_ = false, bool HasBlockSizes_ = true>
struct CollectiveMainloopFwd {
  static constexpr bool HasVariableBlockNums = HasVariableBlockNums_;
  static constexpr bool HasBlockSizes = HasBlockSizes_;
  // ---- Element types ----
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::bfloat16_t;
  using ElementAccumulator = float;

  // ---- Tile sizes ----
  static constexpr int kRows = 64;
  static constexpr int kQkN = 256;
  static constexpr int kQkK = 128;
  static constexpr int kOutputCols = 128;
  static constexpr int kDualCols = 256;
  static constexpr int kDualK = 128;

  // ---- Sparse block constants ----
  static constexpr int kSparseBlockSize = 64;  // tokens per sparse block
  static constexpr int kSparseBlocksPerKV = kDualCols / kSparseBlockSize;  // 4
  static constexpr int kDimHalf = kDualK / 2;  // 64 — swizzle 128B splits dim in half
  static constexpr int kDimHalves = 2;
  // SMEM offsets for K sub-tile: K layout (256,(64,2)):(64,(1,16384))
  static constexpr int kKSubStride = kSparseBlockSize * kDimHalf;  // 4096 (token group)
  static constexpr int kKHalfStride = kDualCols * kDimHalf;        // 16384 (dim half)
  // SMEM offsets for V sub-tile: V layout (128,(64,4)):(64,(1,8192))
  static constexpr int kVSubStride = kOutputCols * kDimHalf;        // 8192 (token group)
  static constexpr int kVHalfStride = kSparseBlockSize * kDimHalf;  // 4096 (dim half)

  // ---- MMA types ----
  // QK GEMM: ss mode (Q and K both in SMEM)
  using QkTiledMma = decltype(cute::make_tiled_mma(
      cute::SM100_MMA_F16BF16_WS_SS_NOELECT<ElementA, ElementB, ElementAccumulator, kRows,
                                            kDualCols, cute::UMMA::Major::K,
                                            cute::UMMA::Major::K>{}));

  // PV GEMM: ts mode (P in TMEM, V in SMEM) — dual pattern
  using PvTiledMma = decltype(cute::make_tiled_mma(
      cute::SM100_MMA_F16BF16_WS_TS_NOELECT<ElementA, ElementB, ElementAccumulator, kRows,
                                            kDualCols, cute::UMMA::Major::K,
                                            cute::UMMA::Major::K>{}));

  using ALogicalShape = cute::Shape<cute::Int<kRows>, cute::Int<kDualK>>;

  // ---- SMEM layouts ----
  using SmemLayoutQ = decltype(flash::make_umma_k_major_layout<kRows, kQkK, 128, ElementA>());
  using SmemLayoutB =
      decltype(flash::make_umma_k_major_layout<kOutputCols, kDualCols, 128, ElementB>());
  using SmemLayoutBDual =
      decltype(flash::make_umma_k_major_layout<kDualCols, kDualK, 128, ElementB>());

  // ---- KV pipeline ----
  static constexpr int kKVStages = 3;
  static constexpr int kKVElemsPerStage = cute::cosize_v<SmemLayoutBDual>;
  static_assert(kKVElemsPerStage == cute::cosize_v<SmemLayoutB>);
  static constexpr int kKVTotalElems = kKVElemsPerStage * kKVStages;

  // ---- TMEM constants (2 S stages + 2 O stages = 512 cols) ----
  static constexpr uint32_t kTmemS0 = 0;
  static constexpr uint32_t kTmemS1 = 128;
  static constexpr uint32_t kTmemO0 = 256;
  static constexpr uint32_t kTmemO1 = 384;

  static constexpr int kCSpan = 128;
  static constexpr int kPackedCols = 64;

  // ---- Warp counts ----
  static constexpr int kSoftmaxWarps = 4;  // per WG
  static constexpr int kCorrWarps = 4;
  // SmStatsNotify NamedBarrier: per-warp, 32 softmax + 32 correction = 64 threads
  // BSA pattern: arrive_w_index(stage*4+warp_idx), 8 barriers total
  static constexpr int kSmStatsNotifyThreads = 64;

  // ---- Byte constants ----
  static constexpr int kQBytes = kRows * kQkK * sizeof(ElementA);
  static constexpr int kKVBytes = kDualCols * kDualK * sizeof(ElementB);
  static constexpr int kSubTileBytes =
      kSparseBlockSize * kDimHalf * sizeof(ElementB);  // 8KB per TMA

  // ---- Compact (64,64) SMEM layout for sparse TMA ----
  // Each sparse block's dim-half: (64 tokens, 64 dim) with Sw<3,4,3>
  // compact(r, c) + i*4096 + h*16384 == canonical(i*64+r, h*64+c) — verified!
  using SmemLayoutSubTile = decltype(cute::coalesce(
      cute::tile_to_shape(cute::UMMA::Layout_K_SW128_Atom<ElementB>{},
                          cute::Shape<cute::Int<kSparseBlockSize>, cute::Int<kDimHalf>>{},
                          cute::Step<cute::_1, cute::_2>{}),
      cute::Shape<cute::_1, cute::_1>{}));

  // ---- TMA types ----
  using ShapeTensor = cute::Shape<int, int, int>;
  using StrideQ = cute::Stride<cute::Int<kQkK>, cute::_1, int>;
  // Sparse K dim-half: global (64 tokens, 64 dim) per entry, stride = (64, 1)
  using StrideKHalf = cute::Stride<cute::Int<kDimHalf>, cute::_1, int>;
  // Sparse V dim-half: global (64 dim, 64 tokens) per entry, stride = (64, 1)
  using StrideVHalf = cute::Stride<cute::Int<kSparseBlockSize>, cute::_1, int>;

  using TMA_Q = decltype(cute::make_tma_copy(
      cute::SM90_TMA_LOAD{},
      cute::make_tensor(cute::make_gmem_ptr(static_cast<ElementA const*>(nullptr)),
                        cute::make_layout(ShapeTensor{}, StrideQ{})),
      SmemLayoutQ{}));
  // Sparse K TMA: loads one dim-half (64, 64) per call
  using TMA_K = decltype(cute::make_tma_copy(
      cute::SM90_TMA_LOAD{},
      cute::make_tensor(cute::make_gmem_ptr(static_cast<ElementB const*>(nullptr)),
                        cute::make_layout(ShapeTensor{}, StrideKHalf{})),
      SmemLayoutSubTile{}));
  // Sparse V TMA: loads one dim-half (64, 64) per call
  using TMA_V = decltype(cute::make_tma_copy(
      cute::SM90_TMA_LOAD{},
      cute::make_tensor(cute::make_gmem_ptr(static_cast<ElementB const*>(nullptr)),
                        cute::make_layout(ShapeTensor{}, StrideVHalf{})),
      SmemLayoutSubTile{}));

  // ---- TensorStorage ----
  struct TensorStorage {
    alignas(128) cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutQ>> smem_q;
    alignas(128) cute::ArrayEngine<ElementB, kKVTotalElems> smem_kv;
    alignas(16) float smem_max[256];       // 128 per softmax WG
    alignas(16) float smem_sum[256];       // 128 per softmax WG
    alignas(16) float corr_scale[2][128];  // per-stage rescale factor
  };

  // ---- Arguments (host-side) ----
  struct Arguments {
    ElementA const* ptr_Q;
    ElementB const* ptr_K;
    ElementB const* ptr_V;
    float softmax_scale;
    int const* ptr_block_indices =
        nullptr;  // sparse: [heads, num_q_tiles, num_kv_iters * kSparseBlocksPerKV]
    int block_indices_stride = 0;  // stride per (head, q_tile) = num_kv_iters * kSparseBlocksPerKV
    int const* ptr_block_sizes =
        nullptr;  // sparse: [total_sparse_blocks], actual tokens per block (0-64)
    int const* ptr_q2k_block_nums =
        nullptr;  // variable: [batch * heads, num_q_blocks], per-tile sparse block count
    int raw_block_sparse_num = 0;  // original unpadded block_sparse_num (for phantom clamping)
  };

  // ---- Params (device-side) ----
  struct Params {
    TMA_Q tma_load_Q;
    TMA_K tma_load_K;
    TMA_V tma_load_V;
    ShapeTensor shape_Q;
    ShapeTensor shape_K;
    ShapeTensor shape_V;
    float sm_scale_log2;
    int const* ptr_block_indices = nullptr;
    int block_indices_stride = 0;
    int kv_tma_per_head = 0;
    int const* ptr_block_sizes = nullptr;
    int num_heads = 1;
    int num_row_tiles = 1;
    int const* ptr_q2k_block_nums = nullptr;
    int raw_block_sparse_num = 0;  // original unpadded block_sparse_num (for index clamping)
  };

  // ---- Convert Arguments -> Params ----
  // seq_kv: actual K/V sequence length for TMA descriptor (may differ from seq_padded in sparse
  // mode) batch: batch size (folded into TMA 3rd dimension)
  static Params to_underlying_arguments(Arguments const& args, int rows_padded, int seq_padded,
                                        int heads, int seq_kv = 0, int batch = 1) {
    using namespace cute;
    int const num_row_tiles = rows_padded / kRows;
    if (seq_kv == 0) seq_kv = seq_padded;

    // TMA 3rd dim = batch * heads * tiles_per_head (batch folded in)
    auto shape_q = make_shape(kRows, kQkK, batch * heads * num_row_tiles);
    auto stride_q = make_stride(Int<kQkK>{}, _1{}, kRows * kQkK);
    auto tma_q = make_tma_copy(
        SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(args.ptr_Q), make_layout(shape_q, stride_q)),
        SmemLayoutQ{});

    int const total_sparse_blocks = seq_kv / kSparseBlockSize;
    auto shape_k =
        make_shape(kSparseBlockSize, kDimHalf, batch * heads * total_sparse_blocks * kDimHalves);
    auto stride_k = make_stride(Int<kDimHalf>{}, _1{}, kSparseBlockSize * kDimHalf);
    auto tma_k = make_tma_copy(
        SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(args.ptr_K), make_layout(shape_k, stride_k)),
        SmemLayoutSubTile{});

    auto shape_v =
        make_shape(kDimHalf, kSparseBlockSize, batch * heads * total_sparse_blocks * kDimHalves);
    auto stride_v = make_stride(Int<kSparseBlockSize>{}, _1{}, kDimHalf * kSparseBlockSize);
    auto tma_v = make_tma_copy(
        SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(args.ptr_V), make_layout(shape_v, stride_v)),
        SmemLayoutSubTile{});

    float sm_scale_log2 = float(args.softmax_scale * M_LOG2E);
    int kv_tma_per_head = total_sparse_blocks * kDimHalves;
    return {tma_q,
            tma_k,
            tma_v,
            shape_q,
            shape_k,
            shape_v,
            sm_scale_log2,
            args.ptr_block_indices,
            args.block_indices_stride,
            kv_tma_per_head,
            args.ptr_block_sizes,
            heads,
            num_row_tiles,
            args.ptr_q2k_block_nums,
            args.raw_block_sparse_num};
  }

  // Helper: compute per-tile num_kv_blocks (pipeline iterations) and raw block count.
  // Phantom block padding: round up raw count to multiple of kSparseBlocksPerKV*2=8,
  // then divide by kSparseBlocksPerKV to get even kv_iters.
  // raw_count: actual sparse blocks (for index clamping and phantom detection).
  CUTLASS_DEVICE static int get_tile_num_kv_blocks(Params const& params, int batch, int head,
                                                   int row_tile, int global_num_kv_blocks) {
    if constexpr (HasVariableBlockNums) {
      int tile_flat = (batch * params.num_heads + head) * params.num_row_tiles + row_tile;
      int raw_count = params.ptr_q2k_block_nums[tile_flat];
      if (raw_count <= 0) return 0;  // empty tile
      // Round up to multiple of 8 (kSparseBlocksPerKV * 2), then /4 → even kv_iters
      constexpr int kAlign = kSparseBlocksPerKV * 2;  // 8
      int padded = (raw_count + kAlign - 1) & ~(kAlign - 1);
      return padded / kSparseBlocksPerKV;
    } else {
      return global_num_kv_blocks;
    }
  }

  // Get the raw (unpadded) block count for a tile — used for index clamping and phantom detection.
  CUTLASS_DEVICE static int get_tile_raw_block_count(Params const& params, int batch, int head,
                                                     int row_tile) {
    if constexpr (HasVariableBlockNums) {
      int tile_flat = (batch * params.num_heads + head) * params.num_row_tiles + row_tile;
      return params.ptr_q2k_block_nums[tile_flat];
    } else {
      return params.raw_block_sparse_num;
    }
  }

  // ===========================================================================
  // Load warp (BSA q_stage=1 reverse order)
  // KV order: K[N-1], Q, K[N-2], {V[N-1-i], K[N-3-i]}x(N-2), V[1], V[0]
  // N >= 2 and even (guaranteed by host padding)
  // ===========================================================================

  struct LoadState {
    // Producer start state: phase=1 (example 77 make_producer_start_state)
    PipelineKVState kv_state = cutlass::make_producer_start_state<PipelineKV>();
  };

  // Prefetch TMA descriptors (called once from kernel, not per-tile).
  CUTLASS_DEVICE static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
  }

  template <typename SharedStorage>
  CUTLASS_DEVICE LoadState
  load(Params const& params, PipelineKV& pipeline_kv, SharedStorage& shared_storage, int head,
       int row_tile, int batch, int num_row_tiles, int num_kv_blocks,
       int raw_block_count,  // actual sparse block count (for phantom clamping)
       LoadState state) {
    using namespace cute;
    auto& ml = shared_storage.tensors.mainloop;

    if (elect_one_sync()) {
      auto thr_tma_q = params.tma_load_Q.get_slice(Int<0>{});
      auto thr_tma_k = params.tma_load_K.get_slice(Int<0>{});
      auto thr_tma_v = params.tma_load_V.get_slice(Int<0>{});

      Tensor gQ_full = params.tma_load_Q.get_tma_tensor(params.shape_Q);
      Tensor gK_full = params.tma_load_K.get_tma_tensor(params.shape_K);
      Tensor gV_full = params.tma_load_V.get_tma_tensor(params.shape_V);

      // Batch-aware TMA indices: batch folds into outermost dimension
      int q_tile_idx = (batch * params.num_heads + head) * num_row_tiles + row_tile;
      int kv_tma_base = (batch * params.num_heads + head) * params.kv_tma_per_head;

      // Block indices for this (batch, head, row_tile)
      int const* tile_block_indices = nullptr;
      if (params.ptr_block_indices != nullptr) {
        int tile_idx_flat = (batch * params.num_heads + head) * num_row_tiles + row_tile;
        tile_block_indices = params.ptr_block_indices + tile_idx_flat * params.block_indices_stride;
      }

      // Load Q (one-shot TMA into persistent smem_q)
      {
        Tensor gQ_tile = gQ_full(_, _, q_tile_idx);
        auto sQ = make_tensor(make_smem_ptr(ml.smem_q.begin()), SmemLayoutQ{});
        cute::set_barrier_transaction_bytes(shared_storage.pipelines.bar_q_ready, kQBytes);
        cute::copy(params.tma_load_Q.with(
                       reinterpret_cast<uint64_t&>(shared_storage.pipelines.bar_q_ready)),
                   thr_tma_q.partition_S(gQ_tile), thr_tma_q.partition_D(sQ));
      }

      auto kv_state = state.kv_state;

      // Load K: 8 TMAs per KV block (4 sparse blocks × 2 dim halves)
      // Each TMA loads (64, 64) bf16 = 8KB. 8 × 8KB = 64KB per stage.
      // SMEM offset for sub-block i, dim-half h: K: i*kKSubStride+h*kKHalfStride, V:
      // i*kVSubStride+h*kVHalfStride K interleave map: sub-block → SMEM slot. Ensures balanced
      // warp-col distribution. {0→0, 1→2, 2→1, 3→3}: warp-col 0 gets slots 0,1 (sub 0,2), warp-col
      // 1 gets 2,3 (sub 1,3)
      static constexpr int kInterleavedSlot[4] = {0, 2, 1, 3};

      // Resolve sparse block index: indirect via block_indices or direct (dense)
      // Phantom block clamping: if logical_idx >= raw_block_count,
      // clamp to last valid index (phantom loads same data, masked in softmax).
      auto get_sparse_idx = [&](int kv_block_idx, int sub) -> int {
        int logical_idx = kv_block_idx * kSparseBlocksPerKV + sub;
        int clamped = (logical_idx < raw_block_count) ? logical_idx : max(raw_block_count - 1, 0);
        return (tile_block_indices != nullptr) ? tile_block_indices[clamped] : clamped;
      };

      // Example 77 pattern: producer_acquire (wait free + arrive_and_expect_tx)
      // → get_barrier → TMA copies with barrier → advance state.
      // No manual set_barrier_transaction_bytes, no fence, no nanosleep.
      // Pass kv_st by value to avoid [&] capturing kv_state by reference → local memory.
      auto load_K = [&](int kv_block_idx, PipelineKVState kv_st) -> PipelineKVState {
        pipeline_kv.producer_acquire(kv_st);
        auto* tma_bar = pipeline_kv.producer_get_barrier(kv_st);
        int stage_base = kv_st.index() * kKVElemsPerStage;
        CUTLASS_PRAGMA_UNROLL
        for (int sub = 0; sub < kSparseBlocksPerKV; ++sub) {
          int sparse_idx = get_sparse_idx(kv_block_idx, sub);
          int slot = kInterleavedSlot[sub];
          CUTLASS_PRAGMA_UNROLL
          for (int h = 0; h < kDimHalves; ++h) {
            int smem_offset = stage_base + slot * kKSubStride + h * kKHalfStride;
            auto sK_sub =
                make_tensor(make_smem_ptr(ml.smem_kv.begin() + smem_offset), SmemLayoutSubTile{});
            int tma_idx = kv_tma_base + sparse_idx * kDimHalves + h;
            cute::copy(params.tma_load_K.with(*tma_bar),
                       thr_tma_k.partition_S(gK_full(_, _, tma_idx)),
                       thr_tma_k.partition_D(sK_sub));
          }
        }
        ++kv_st;
        return kv_st;
      };

      auto load_V = [&](int kv_block_idx, PipelineKVState kv_st) -> PipelineKVState {
        pipeline_kv.producer_acquire(kv_st);
        auto* tma_bar = pipeline_kv.producer_get_barrier(kv_st);
        int stage_base = kv_st.index() * kKVElemsPerStage;
        CUTLASS_PRAGMA_UNROLL
        for (int sub = 0; sub < kSparseBlocksPerKV; ++sub) {
          int sparse_idx = get_sparse_idx(kv_block_idx, sub);
          int slot = sub;
          CUTLASS_PRAGMA_UNROLL
          for (int h = 0; h < kDimHalves; ++h) {
            int smem_offset = stage_base + slot * kVSubStride + h * kVHalfStride;
            auto sV_sub =
                make_tensor(make_smem_ptr(ml.smem_kv.begin() + smem_offset), SmemLayoutSubTile{});
            int tma_idx = kv_tma_base + sparse_idx * kDimHalves + h;
            cute::copy(params.tma_load_V.with(*tma_bar),
                       thr_tma_v.partition_S(gV_full(_, _, tma_idx)),
                       thr_tma_v.partition_D(sV_sub));
          }
        }
        ++kv_st;
        return kv_st;
      };

      // BSA reverse order: K[N-1], K[N-2], {V[N-1-i], K[N-3-i]}x(N-2), V[1], V[0]
      kv_state = load_K(num_kv_blocks - 1, kv_state);
      kv_state = load_K(num_kv_blocks - 2, kv_state);

      CUTE_NO_UNROLL
      for (int i = 0; i < num_kv_blocks - 2; ++i) {
        kv_state = load_V(num_kv_blocks - 1 - i, kv_state);
        kv_state = load_K(num_kv_blocks - 3 - i, kv_state);
      }

      kv_state = load_V(1, kv_state);
      kv_state = load_V(0, kv_state);

      state.kv_state = kv_state;
    }
    return state;
  }

  // ===========================================================================
  // MMA warp (BSA q_stage=1: alternating stage 0/1, PV before QK in main loop)
  // ===========================================================================

  struct MmaState {
    PipelineKVState kv_state;  // consumer starts at phase=0 (default)
    int phase_s0 = 0;
    int phase_s1 = 0;
    int q_phase = 0;
    int pls_phase0 = 0;  // p_lastsplit phase for stage 0
    int pls_phase1 = 0;  // p_lastsplit phase for stage 1
  };

  // Split PV GEMM: issue first half tiles, wait p_lastsplit, issue remaining half.
  // Earlier SPO release (1/2 instead of 3/4) gives MMA earlier P access,
  // reducing pipeline wait stalls at the cost of a longer p_lastsplit wait.
  static constexpr int kSplitNumer = 1;
  static constexpr int kSplitDenom = 2;

  template <typename TiledMma, typename TensorA, typename TensorB, typename TensorFragC>
  CUTE_DEVICE static void utcmma_ts_split(TiledMma& tiled_mma, TensorA tA_frag, TensorB sB,
                                          TensorFragC tC_frag, bool clear_accum,
                                          uint32_t p_lastsplit_addr, int p_lastsplit_phase) {
    using namespace cute;
    tiled_mma.accumulate_ = clear_accum ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
    auto thr_mma = tiled_mma.get_slice(_0{});
    auto sB_frag = thr_mma.partition_fragment_B(sB);
    constexpr int kTotal = decltype(size<2>(tA_frag))::value;
    constexpr int kSplitK = kTotal * kSplitNumer / kSplitDenom;

    CUTE_UNROLL
    for (int k = 0; k < kSplitK; ++k) {
      cute::gemm(tiled_mma, tA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
      tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }

    // Wait for last split of P
    wait_barrier_addr(p_lastsplit_addr, p_lastsplit_phase);

    CUTE_UNROLL
    for (int k = kSplitK; k < kTotal; ++k) {
      cute::gemm(tiled_mma, tA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
    }
  }

  template <typename SharedStorage>
  CUTLASS_DEVICE MmaState mma(PipelineKV& pipeline_kv, PipelineSPO& pipeline_s_p_o,
                              PipelineOAcc& pipeline_o_acc,
                              PipelinePLastSplit& pipeline_p_lastsplit,
                              SharedStorage& shared_storage, uint32_t tmem_base, int num_kv_blocks,
                              MmaState state) {
    using namespace cute;
    auto& ml = shared_storage.tensors.mainloop;

    const uint32_t tmem_s[2] = {tmem_base + kTmemS0, tmem_base + kTmemS1};
    const uint32_t tmem_o[2] = {tmem_base + kTmemO0, tmem_base + kTmemO1};

    QkTiledMma qk_mma;
    PvTiledMma pv_mma;
    Tensor tC_qk = partition_fragment_C(qk_mma, Shape<Int<kRows>, Int<kDualCols>>{});
    Tensor tC_pv = partition_fragment_C(pv_mma, Shape<Int<kRows>, Int<kDualCols>>{});
    Tensor tP = pv_mma.get_slice(_0{}).make_fragment_A(partition_shape_A(pv_mma, ALogicalShape{}));

    auto sQ = make_tensor(make_smem_ptr(ml.smem_q.begin()), SmemLayoutQ{});

    if (elect_one_sync()) {
      auto kv_state = state.kv_state;
      int phase_s0 = state.phase_s0;
      int phase_s1 = state.phase_s1;
      bool o_acc_s0 = false, o_acc_s1 = false;

      // Wait for Q TMA
      int q_phase = state.q_phase;
      wait_barrier(shared_storage.pipelines.bar_q_ready, q_phase);
      q_phase ^= 1;
      flash::tcgen05_commit();

      // QK GEMM: S[stage] = Q @ K, then signal S ready via UMMA arrive
      // Pass kv_st by value to avoid [&] capturing kv_state → local memory.
      auto mma_qk = [&](int stage, PipelineKVState kv_st) -> PipelineKVState {
        pipeline_kv.consumer_wait(kv_st);
        flash::tcgen05_commit();
        tC_qk.data() = tmem_s[stage];
        auto sK = make_tensor(make_smem_ptr(ml.smem_kv.begin() + kv_st.index() * kKVElemsPerStage),
                              SmemLayoutBDual{});
        flash::utcmma_ss(qk_mma, sQ, sK, tC_qk, true);
        flash::umma_arrive(pipeline_s_p_o.producer_get_barrier_w_index(stage));
        // consumer_release: umma_arrive on empty barrier.
        flash::umma_arrive(reinterpret_cast<cute::uint64_t&>(
            shared_storage.pipelines.kv.empty_barrier_[kv_st.index()]));
        ++kv_st;
        return kv_st;
      };

      // PV GEMM: O[stage] += P[stage] @ V
      // Uses split_P_arrive: first 3/4 UTCHMMA, wait p_lastsplit, last 1/4.
      int pls_phase0 = state.pls_phase0;
      int pls_phase1 = state.pls_phase1;

      auto mma_pv = [&](int stage, bool clear_accum, PipelineKVState kv_st) -> PipelineKVState {
        pipeline_kv.consumer_wait(kv_st);
        flash::tcgen05_commit();
        tC_pv.data() = tmem_o[stage];
        tP.data() = tmem_s[stage];
        auto sV = make_tensor(make_smem_ptr(ml.smem_kv.begin() + kv_st.index() * kKVElemsPerStage),
                              SmemLayoutBDual{});
        int& pls_phase = (stage == 0) ? pls_phase0 : pls_phase1;
        utcmma_ts_split(pv_mma, tP, sV, tC_pv, clear_accum,
                        pipeline_p_lastsplit.get_barrier_addr(stage), pls_phase);
        pls_phase ^= 1;
        // consumer_release: umma_arrive on empty barrier.
        flash::umma_arrive(reinterpret_cast<cute::uint64_t&>(
            shared_storage.pipelines.kv.empty_barrier_[kv_st.index()]));
        ++kv_st;
        return kv_st;
      };

      // ---- Prologue: S0 = Q@K[N-1], S1 = Q@K[N-2] (N>=2 guaranteed) ----
      kv_state = mma_qk(0, kv_state);
      kv_state = mma_qk(1, kv_state);

      // ---- Main loop: pairs of {PV[stage] + QK[stage]} alternating stage 0,1 ----
      int pair_count = (num_kv_blocks - 2) / 2;
      CUTE_NO_UNROLL
      for (int i = 0; i < pair_count; ++i) {
        pipeline_s_p_o.producer_acquire_w_index_phase(0, phase_s0);
        kv_state = mma_pv(0, !o_acc_s0, kv_state);
        kv_state = mma_qk(0, kv_state);
        o_acc_s0 = true;

        pipeline_s_p_o.producer_acquire_w_index_phase(1, phase_s1);
        kv_state = mma_pv(1, !o_acc_s1, kv_state);
        kv_state = mma_qk(1, kv_state);
        o_acc_s1 = true;
      }

      // ---- Epilogue: 2 final PV GEMMs, signal O_acc ----
      pipeline_s_p_o.producer_acquire_w_index_phase(0, phase_s0);
      kv_state = mma_pv(0, !o_acc_s0, kv_state);
      flash::umma_arrive(pipeline_o_acc.producer_get_barrier_w_index(0));

      pipeline_s_p_o.producer_acquire_w_index_phase(1, phase_s1);
      kv_state = mma_pv(1, !o_acc_s1, kv_state);
      flash::umma_arrive(pipeline_o_acc.producer_get_barrier_w_index(1));

      state.kv_state = kv_state;
      state.phase_s0 = phase_s0;
      state.phase_s1 = phase_s1;
      state.q_phase = q_phase;
      state.pls_phase0 = pls_phase0;
      state.pls_phase1 = pls_phase1;
    }
    return state;
  }

  static constexpr int kFrgTile = 32;
  static constexpr int kFrgCount = kCSpan / kFrgTile;

  // ===========================================================================
  // Softmax (parameterized by stage: 0 or 1)
  // Each softmax WG processes only its own stage (WG0 -> stage 0, WG1 -> stage 1).
  // Two-pass T2R: pass 1 for fmax only, pass 2 for scale+exp2+sum+bf16+R2T.
  // ===========================================================================

  struct SoftmaxState {
    int s_phase = 0;
    int sm_stats_phase = 0;
  };

  // R2P bitmask: keep positions < limit within 32-element chunk s.
  // Matches blk128 mask.py: r2p_bitmask_below(limit, s).
  CUTLASS_DEVICE static uint32_t r2p_bitmask_below(int limit, int s) {
    uint32_t shift = static_cast<uint32_t>(max(0, (s + 1) * 32 - limit));
    uint32_t result;
    asm("shr.b32 %0, 0xFFFFFFFF, %1;" : "=r"(result) : "r"(shift));
    return result;
  }

  // Apply R2P block-size mask to a sub-block at offset.
  // Matches blk128 mask.py: apply_block_size_mask + mask_r2p_lambda.
  template <typename Tensor>
  CUTLASS_DEVICE static void apply_block_size_mask(Tensor& tSrS, int block_size, int offset) {
    if (block_size < kSparseBlockSize) {
      constexpr int kChunks = kSparseBlockSize / 32;
      CUTLASS_PRAGMA_UNROLL
      for (int s = 0; s < kChunks; ++s) {
        uint32_t mask = r2p_bitmask_below(block_size, s);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 32; ++i) {
          float val = tSrS(offset + s * 32 + i);
          asm("{\n\t"
              "  .reg .pred p;\n\t"
              "  .reg .u32 tmp;\n\t"
              "  and.b32 tmp, %1, %2;\n\t"
              "  setp.eq.u32 p, tmp, 0;\n\t"
              "  @p mov.f32 %0, 0fFF800000;\n\t"
              "}"
              : "+f"(val)
              : "r"(mask), "r"(1u << i));
          tSrS(offset + s * 32 + i) = val;
        }
      }
    }
  }

  // Single softmax iteration — matches blk128 softmax_step exactly:
  //   1. T2R load + mask
  //   2. update_row_max
  //   3. scale_subtract_rowmax  (separate FMA pass over all 128 elements)
  //   4. apply_exp2_convert     (per-fragment: exp2 in-place → bf16 convert → TMEM store)
  //   5. update_row_sum
  template <int Stage, bool IsFirst, typename SharedStorage, typename NamedBarriers>
  CUTLASS_DEVICE void softmax_step(int sm_idx, int sm_stats_bar, PipelineSPO& pipeline_s_p_o,
                                   PipelineSmStats& pipeline_sm_stats,
                                   PipelinePLastSplit& pipeline_p_lastsplit,
                                   SharedStorage& shared_storage, uint32_t tmem_s_cur,
                                   float sm_scale, float& row_max, float& row_sum, int& s_phase,
                                   int& sm_stats_phase, int block_size_lo = kSparseBlockSize,
                                   int block_size_hi = kSparseBlockSize) {
    using namespace cute;
    using cutlass::arch::NamedBarrier;
    auto& ml = shared_storage.tensors.mainloop;

    // 1. Wait for S ready in TMEM
    pipeline_s_p_o.consumer_wait_w_index_phase(Stage, s_phase);

    // 2. T2R load
    auto tSrS_t2r = cute::make_tensor<float>(cute::Shape<cute::Int<kCSpan>>{});
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < kFrgCount; ++c) {
      flash::tmem_load<kFrgTile>(tmem_s_cur + c * kFrgTile, &tSrS_t2r(c * kFrgTile));
    }

    // 2b. Block-size mask (R2P bitmask pattern, matches blk128 mask.py)
    apply_block_size_mask(tSrS_t2r, block_size_lo, 0);
    apply_block_size_mask(tSrS_t2r, block_size_hi, kSparseBlockSize);

    // 3. update_row_max
    float acc_scale, m_new_safe;
    update_row_max<IsFirst>(tSrS_t2r, sm_scale, row_max, acc_scale, m_new_safe);
    if constexpr (!IsFirst) {
      ml.corr_scale[Stage][sm_idx] = acc_scale;
    }

    // 4. Notify correction
    NamedBarrier::arrive(kSmStatsNotifyThreads, sm_stats_bar);

    // 5. scale_subtract_rowmax (separate FMA pass, matches blk128)
    {
      float neg_m_scaled = -m_new_safe * sm_scale;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCSpan; i += 2) {
        ffma2(tSrS_t2r(i), tSrS_t2r(i + 1), sm_scale, neg_m_scaled);
      }
    }

    // 6. apply_exp2_convert (per-fragment: exp2 → bf16 → TMEM store, matches blk128)
    {
      constexpr int kEmuFreq = 12;
      constexpr int kEmuRes = 4;
      constexpr int kEmuStartFrg = 0;
      constexpr int kPairsPerFrg = kFrgTile / 2;

      auto tSrS_frg = cute::make_tensor(
          tSrS_t2r.data(), cute::make_shape(cute::Int<kFrgTile>{}, cute::Int<kFrgCount>{}));

      CUTLASS_PRAGMA_UNROLL
      for (int frag = 0; frag < kFrgCount; ++frag) {
        // exp2 in-place
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kFrgTile; i += 2) {
          if (frag < kEmuStartFrg || frag >= kFrgCount - 1 || i % kEmuFreq < kEmuFreq - kEmuRes) {
            tSrS_frg(i, frag) = exp2f(tSrS_frg(i, frag));
            tSrS_frg(i + 1, frag) = exp2f(tSrS_frg(i + 1, frag));
          } else {
            exp2_emu2(tSrS_frg(i, frag), tSrS_frg(i + 1, frag));
          }
        }

        // Convert to bf16 + TMEM store
        uint32_t p_frag[kPairsPerFrg];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPairsPerFrg; ++i) {
          nv_bfloat162 pair =
              __floats2bfloat162_rn(tSrS_frg(2 * i, frag), tSrS_frg(2 * i + 1, frag));
          p_frag[i] = reinterpret_cast<uint32_t const&>(pair);
        }
        [&]<size_t... Is>(cute::index_sequence<Is...>) {
          SM100_TMEM_STORE_32dp32b16x::copy(p_frag[Is]..., tmem_s_cur + frag * kPairsPerFrg);
        }(cute::make_index_sequence<kPairsPerFrg>{});

        // split_P_arrive
        if (frag + 1 == kFrgCount * kSplitNumer / kSplitDenom) {
          cutlass::arch::fence_view_async_tmem_store();
          pipeline_s_p_o.consumer_release_w_index(Stage);
        }
      }
    }

    // 7. All P written — signal p_lastsplit
    cutlass::arch::fence_view_async_tmem_store();
    __syncwarp();
    if (elect_one_sync()) {
      pipeline_p_lastsplit.producer_commit_w_index(Stage);
    }

    // ---- 8. pipeline_sm_stats.producer_acquire ----
    pipeline_sm_stats.producer_acquire_w_index_phase(Stage, sm_stats_phase);

    // ---- 9. update_row_sum (blk128: softmax.update_row_sum) ----
    update_row_sum<IsFirst>(tSrS_t2r, acc_scale, row_sum);
  }

  // Softmax outer loop (BSA: softmax_loop).
  // Stage is compile-time: enables constant folding for TMEM addresses, barrier IDs.
  template <int Stage, typename SharedStorage, typename NamedBarriers>
  CUTLASS_DEVICE SoftmaxState softmax(PipelineSPO& pipeline_s_p_o,
                                      PipelineSmStats& pipeline_sm_stats,
                                      PipelinePLastSplit& pipeline_p_lastsplit,
                                      SharedStorage& shared_storage, uint32_t tmem_base,
                                      float sm_scale, int num_kv_blocks, SoftmaxState state,
                                      int const* tile_block_indices = nullptr,
                                      int const* ptr_block_sizes = nullptr,
                                      int raw_block_count = 0) {
    using namespace cute;
    using cutlass::arch::NamedBarrier;
    auto& ml = shared_storage.tensors.mainloop;

    constexpr uint32_t kTmemS = (Stage == 0) ? kTmemS0 : kTmemS1;
    const uint32_t tmem_s_cur = tmem_base + kTmemS;
    const int sm_idx = threadIdx.x - Stage * kSoftmaxWarps * 32;
    const int warp_in_wg = sm_idx / 32;
    const int sm_stats_bar = NamedBarriers::SmStatsNotify + Stage * kSoftmaxWarps + warp_in_wg;

    float row_max = -CUDART_INF_F;
    float row_sum = 0.0f;
    int s_phase = state.s_phase;
    int sm_stats_phase = state.sm_stats_phase;
    int wg_count = num_kv_blocks / 2;

    // Block-size lookup: slot → sub mapping (inverse of kInterleavedSlot)
    // Warp-col 0 (warps 0,1): TMEM cols 0-127 → subs 0,2
    // Warp-col 1 (warps 2,3): TMEM cols 128-255 → subs 1,3
    // sub_lo = warp_col, sub_hi = warp_col + 2 (avoids array indexing → no LDL)
    const int warp_col = warp_in_wg / 2;

    auto get_block_sizes = [&](int k, int& bs_lo, int& bs_hi) {
      if constexpr (HasBlockSizes) {
        int kv_block = num_kv_blocks - 1 - (2 * k + Stage);
        int logical_lo = kv_block * kSparseBlocksPerKV + warp_col;
        int logical_hi = kv_block * kSparseBlocksPerKV + warp_col + 2;
        // Phantom block detection: index >= raw_block_count → block_size=0 (all masked)
        int clamped_lo = (logical_lo < raw_block_count) ? logical_lo : max(raw_block_count - 1, 0);
        int clamped_hi = (logical_hi < raw_block_count) ? logical_hi : max(raw_block_count - 1, 0);
        int bi_lo = tile_block_indices[clamped_lo];
        int bi_hi = tile_block_indices[clamped_hi];
        bs_lo = (logical_lo < raw_block_count) ? ptr_block_sizes[bi_lo] : 0;
        bs_hi = (logical_hi < raw_block_count) ? ptr_block_sizes[bi_hi] : 0;
      } else {
        bs_lo = kSparseBlockSize;
        bs_hi = kSparseBlockSize;
      }
    };

    // BSA: acquire before loop
    pipeline_sm_stats.producer_acquire_w_index_phase(Stage, sm_stats_phase);

    // BSA: 1st block peeled (IsFirst=true), remaining blocks in loop (IsFirst=false)
    {
      int bs_lo, bs_hi;
      get_block_sizes(0, bs_lo, bs_hi);
      softmax_step<Stage, /*IsFirst=*/true, SharedStorage, NamedBarriers>(
          sm_idx, sm_stats_bar, pipeline_s_p_o, pipeline_sm_stats, pipeline_p_lastsplit,
          shared_storage, tmem_s_cur, sm_scale, row_max, row_sum, s_phase, sm_stats_phase, bs_lo,
          bs_hi);
    }

    CUTE_NO_UNROLL
    for (int k = 1; k < wg_count; ++k) {
      int bs_lo, bs_hi;
      get_block_sizes(k, bs_lo, bs_hi);
      softmax_step<Stage, /*IsFirst=*/false, SharedStorage, NamedBarriers>(
          sm_idx, sm_stats_bar, pipeline_s_p_o, pipeline_sm_stats, pipeline_p_lastsplit,
          shared_storage, tmem_s_cur, sm_scale, row_max, row_sum, s_phase, sm_stats_phase, bs_lo,
          bs_hi);
    }

    // Write final stats for correction combine
    ml.smem_sum[Stage * 128 + sm_idx] = row_sum;
    ml.smem_max[Stage * 128 + sm_idx] = row_max;
    __threadfence_block();
    NamedBarrier::arrive(kSmStatsNotifyThreads, sm_stats_bar);
    state.s_phase = s_phase;
    state.sm_stats_phase = sm_stats_phase;
    return state;
  }

  // ===========================================================================
  // Correction: indexed rescale + final combine (Correction warps 8-11)
  // Moved from epilogue_fwd.hpp — accesses both ml.* and el.* via SharedStorage union.
  // ===========================================================================

  // ---- SMEM layout for O (used by correction_combine for sO writes) ----
  using SmemLayoutO = decltype(cute::coalesce(
      cute::tile_to_shape(cute::UMMA::Layout_K_SW128_Atom<ElementA>{},
                          cute::Shape<cute::Int<kRows>, cute::Int<kOutputCols>>{},
                          cute::Step<cute::_1, cute::_2>{}),
      cute::Shape<cute::_1, cute::_1>{}));

  // ---- Chunk size for TMEM loads in correction/combine ----
  static constexpr int kChunk = 32;
  static constexpr int kNumChunks = kCSpan / kChunk;

  struct CorrState {
    int o_acc_phase0 = 0;
    int o_acc_phase1 = 0;
  };

  CUTLASS_DEVICE static void correction_rescale(uint32_t tmem_o, float rescale, int corr_idx) {
    using namespace cute;
    unsigned should_rescale = __ballot_sync(0xFFFFFFFF, rescale < 1.0f);
    if (should_rescale) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < kNumChunks; ++c) {
        auto tOrO = cute::make_tensor<float>(cute::Shape<cute::Int<kChunk>>{});
        flash::tmem_load<kChunk>(tmem_o + c * kChunk, &tOrO(0));
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < kChunk; j += 2) fmul2(tOrO(j), tOrO(j + 1), rescale);
        [&]<size_t... Is>(cute::index_sequence<Is...>) {
          SM100_TMEM_STORE_32dp32b32x::copy(reinterpret_cast<uint32_t const&>(tOrO(Is))...,
                                            tmem_o + c * kChunk);
        }(cute::make_index_sequence<kChunk>{});
      }
    }
  }

  template <typename SmemTensorO, typename EpiStorage, typename NamedBarriers>
  CUTLASS_DEVICE static void correction_combine(uint32_t tmem_o0, uint32_t tmem_o1, float my_scale0,
                                                float my_scale1, int corr_warp, int lane_idx,
                                                int reduce_bar, EpiStorage& el, SmemTensorO& sO) {
    using namespace cute;

    // Pass 1: each warp computes weighted O and writes ALL chunks to OWN slot
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < kNumChunks; ++c) {
      auto tOrO0 = make_tensor<float>(Shape<Int<kChunk>>{});
      auto tOrO1 = make_tensor<float>(Shape<Int<kChunk>>{});
      flash::tmem_load<kChunk>(tmem_o0 + c * kChunk, &tOrO0(0));
      flash::tmem_load<kChunk>(tmem_o1 + c * kChunk, &tOrO1(0));

      auto tOrO_combined = make_tensor<float>(Shape<Int<kChunk>>{});
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < kChunk; j += 2) {
        float scaled_o0 = tOrO0(j), scaled_o1 = tOrO0(j + 1);
        fmul2(scaled_o0, scaled_o1, my_scale0);
        float partner_o0 = tOrO1(j), partner_o1 = tOrO1(j + 1);
        fmul2(partner_o0, partner_o1, my_scale1);
        fadd2(scaled_o0, scaled_o1, partner_o0, partner_o1);
        tOrO_combined(j) = scaled_o0;
        tOrO_combined(j + 1) = scaled_o1;
      }

      // Write to OWN exchange slot
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kChunk / 4; ++i) {
        flash::smem_store_float4(
            &el.o_exchange[corr_warp][c * 32 * kChunk + i * 32 * 4 + lane_idx * 4],
            *reinterpret_cast<float4*>(&tOrO_combined(i * 4)));
      }
    }

    // Single barrier: all warps' exchange writes visible
    cutlass::arch::NamedBarrier::arrive_and_wait(kRows, reduce_bar);

    // Pass 2: warps 0,1 read own + partner exchange data → add → bf16 → sO
    {
      const int out_row = (corr_warp & 1) * 32 + lane_idx;
      const int partner_warp = corr_warp ^ 2;

      if (corr_warp < 2 && out_row < kRows) {
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < kNumChunks; ++c) {
          auto tOrO_final = make_tensor<float>(Shape<Int<kChunk>>{});
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < kChunk / 4; ++i) {
            const int off = c * 32 * kChunk + i * 32 * 4 + lane_idx * 4;
            float4 own = flash::smem_load_float4(&el.o_exchange[corr_warp][off]);
            float4 partner = flash::smem_load_float4(&el.o_exchange[partner_warp][off]);
            float2* dst = reinterpret_cast<float2*>(&tOrO_final(i * 4));
            float2 const* a = reinterpret_cast<float2 const*>(&own);
            float2 const* b = reinterpret_cast<float2 const*>(&partner);
            dst[0] = flash::float2_add(a[0], b[0]);
            dst[1] = flash::float2_add(a[1], b[1]);
          }

          // Convert fp32 → bf16 and write to sO via STS.128
          const int out_col_base = c * kChunk;
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < kChunk; j += 8) {
            nv_bfloat162 p0 = __floats2bfloat162_rn(tOrO_final(j + 0), tOrO_final(j + 1));
            nv_bfloat162 p1 = __floats2bfloat162_rn(tOrO_final(j + 2), tOrO_final(j + 3));
            nv_bfloat162 p2 = __floats2bfloat162_rn(tOrO_final(j + 4), tOrO_final(j + 5));
            nv_bfloat162 p3 = __floats2bfloat162_rn(tOrO_final(j + 6), tOrO_final(j + 7));
            uint32_t addr = cute::cast_smem_ptr_to_uint(&sO(out_row, out_col_base + j));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};\n"
                         :
                         : "r"(addr), "r"(reinterpret_cast<uint32_t const&>(p0)),
                           "r"(reinterpret_cast<uint32_t const&>(p1)),
                           "r"(reinterpret_cast<uint32_t const&>(p2)),
                           "r"(reinterpret_cast<uint32_t const&>(p3)));
          }
        }
      }
    }
  }

  template <typename SharedStorage, typename NamedBarriers>
  CUTLASS_DEVICE CorrState correction(float sm_scale_log2, PipelineSPO& pipeline_s_p_o,
                                      PipelineSmStats& pipeline_sm_stats,
                                      PipelineOAcc& pipeline_o_acc, PipelineOEpi& pipeline_o_epi,
                                      SharedStorage& shared_storage, uint32_t tmem_base,
                                      int num_kv_blocks, CorrState corr_state,
                                      float* ptr_LSE = nullptr, int lse_tile_offset = 0) {
    using namespace cute;
    using cutlass::arch::NamedBarrier;
    auto& ml = shared_storage.tensors.mainloop;
    auto& el = shared_storage.tensors.epilogue;

    const uint32_t tmem_o0 = tmem_base + kTmemO0;
    const uint32_t tmem_o1 = tmem_base + kTmemO1;
    const int warp_idx = threadIdx.x / 32;
    const int lane_idx = threadIdx.x % 32;
    const int corr_warp = warp_idx - 8;         // 0..3
    const int corr_idx = threadIdx.x - 8 * 32;  // 0..127

    // ---- (a) Skip first pair: no rescale needed (BSA pattern) ----
    pipeline_s_p_o.consumer_release_w_index(0);
    pipeline_s_p_o.consumer_release_w_index(1);
    NamedBarrier::arrive_and_wait(kSmStatsNotifyThreads,
                                  NamedBarriers::SmStatsNotify + 0 * kCorrWarps + corr_warp);
    pipeline_sm_stats.consumer_release_w_index(0);
    NamedBarrier::arrive_and_wait(kSmStatsNotifyThreads,
                                  NamedBarriers::SmStatsNotify + 1 * kCorrWarps + corr_warp);

    // ---- (b) Paired rescale loop (BSA: seqlen_corr_loop_steps) ----
    int pair_count = (num_kv_blocks - 2) / 2;
    CUTE_NO_UNROLL
    for (int i = 0; i < pair_count; ++i) {
      // Stage 0: rescale O0
      {
        NamedBarrier::arrive_and_wait(kSmStatsNotifyThreads,
                                      NamedBarriers::SmStatsNotify + 0 * kCorrWarps + corr_warp);
        correction_rescale(tmem_o0, ml.corr_scale[0][corr_idx], corr_idx);
        pipeline_s_p_o.consumer_release_w_index(0);
        pipeline_sm_stats.consumer_release_w_index(1);
      }
      // Stage 1: rescale O1
      {
        NamedBarrier::arrive_and_wait(kSmStatsNotifyThreads,
                                      NamedBarriers::SmStatsNotify + 1 * kCorrWarps + corr_warp);
        correction_rescale(tmem_o1, ml.corr_scale[1][corr_idx], corr_idx);
        pipeline_s_p_o.consumer_release_w_index(1);
        pipeline_sm_stats.consumer_release_w_index(0);
      }
    }

    // BSA: post-loop release for stage 1
    pipeline_sm_stats.consumer_release_w_index(1);

    // ---- (c) Read final stats (BSA: sm_stats_barrier for both stages) ----
    NamedBarrier::arrive_and_wait(kSmStatsNotifyThreads,
                                  NamedBarriers::SmStatsNotify + 0 * kCorrWarps + corr_warp);
    float row_sum0 = ml.smem_sum[0 * 128 + corr_idx];
    float row_max0 = ml.smem_max[0 * 128 + corr_idx];
    pipeline_sm_stats.consumer_release_w_index(0);

    NamedBarrier::arrive_and_wait(kSmStatsNotifyThreads,
                                  NamedBarriers::SmStatsNotify + 1 * kCorrWarps + corr_warp);
    float row_sum1 = ml.smem_sum[1 * 128 + corr_idx];
    float row_max1 = ml.smem_max[1 * 128 + corr_idx];
    pipeline_sm_stats.consumer_release_w_index(1);

    // ---- (d) Compute cross-stage combine scales ----
    float rm0 = (row_sum0 > 0.0f) ? row_max0 : -CUDART_INF_F;
    float rm1 = (row_sum1 > 0.0f) ? row_max1 : -CUDART_INF_F;
    float max_combined = fmaxf(rm0, rm1);
    float max_safe = (max_combined > -CUDART_INF_F) ? max_combined : 0.0f;
    float scale0 = (row_sum0 > 0.0f) ? exp2f((rm0 - max_safe) * sm_scale_log2) : 0.0f;
    float scale1 = (row_sum1 > 0.0f) ? exp2f((rm1 - max_safe) * sm_scale_log2) : 0.0f;
    float my_sum = row_sum0 * scale0 + row_sum1 * scale1;
    float my_max = max_safe;

    // ---- (e) Wait for final O from MMA ----
    int o_acc_phase0 = corr_state.o_acc_phase0;
    int o_acc_phase1 = corr_state.o_acc_phase1;
    pipeline_o_acc.consumer_wait_w_index_phase(0, o_acc_phase0);
    flash::tcgen05_commit();
    pipeline_o_acc.consumer_wait_w_index_phase(1, o_acc_phase1);
    flash::tcgen05_commit();

    // ---- (f) Warp-pair stats exchange + combine weight ----
    auto sO = make_tensor(make_smem_ptr(el.sO.begin()), SmemLayoutO{});
    const int reduce_bar = (corr_warp & 1) ? NamedBarriers::Reduce_13 : NamedBarriers::Reduce_02;

    float my_weight;
    {
      el.o_staging[corr_warp ^ 2][lane_idx * 2 + 0] = my_sum;
      el.o_staging[corr_warp ^ 2][lane_idx * 2 + 1] = my_max;
      NamedBarrier::arrive_and_wait(kRows, reduce_bar);

      float partner_sum = el.o_staging[corr_warp][lane_idx * 2 + 0];
      float partner_max = el.o_staging[corr_warp][lane_idx * 2 + 1];

      float max_total = fmaxf(my_max, partner_max);
      float max_total_safe = (max_total > -CUDART_INF_F) ? max_total : 0.0f;
      float my_rescale = (my_sum > 0.0f) ? exp2f((my_max - max_total_safe) * sm_scale_log2) : 0.0f;
      float partner_rescale =
          (partner_sum > 0.0f) ? exp2f((partner_max - max_total_safe) * sm_scale_log2) : 0.0f;
      float sum_total = my_sum * my_rescale + partner_sum * partner_rescale;
      float inv_sum_total = (sum_total > 0.0f) ? __frcp_rn(sum_total) : 0.0f;  // BSA: rcp_approx
      my_weight = my_rescale * inv_sum_total;

      // ---- Write LSE to global memory (one warp per warp-pair) ----
      if (ptr_LSE != nullptr && corr_warp < 2) {
        int out_row = (corr_warp & 1) * 32 + lane_idx;
        if (out_row < kRows) {
          float lse;
          if (sum_total > 0.0f) {
            lse = (max_total_safe * sm_scale_log2 + log2f(sum_total)) * 0.6931471805599453f;  // LN2
          } else {
            lse = -CUDART_INF_F;
          }
          ptr_LSE[lse_tile_offset * kRows + out_row] = lse;
        }
      }
    }

    // ---- (g) 2-pass combine: correction_combine ----
    float my_scale0 = scale0 * my_weight;
    float my_scale1 = scale1 * my_weight;
    correction_combine<decltype(sO), decltype(el), NamedBarriers>(
        tmem_o0, tmem_o1, my_scale0, my_scale1, corr_warp, lane_idx, reduce_bar, el, sO);

    // ---- (h) Fence + signal epilogue warp ----
    cutlass::arch::fence_view_async_shared();
    pipeline_o_epi.producer_commit();
    corr_state.o_acc_phase0 = o_acc_phase0;
    corr_state.o_acc_phase1 = o_acc_phase1;
    return corr_state;
  }
};

}  // namespace flash
