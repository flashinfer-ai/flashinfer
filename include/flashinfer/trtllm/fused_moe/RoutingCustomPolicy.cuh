/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cuda/functional>  // cuda::maximum<> used by SoftmaxPreprocess::applyToSmem

#include "RoutingKernel.cuh"

namespace moe::dev::routing {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Preprocess policies: applied to all expert scores BEFORE topK selection.
//
// Each policy must provide:
//   - template <typename InputT> using BaseType
//       The data type used for intermediate score computation.
//   - template <typename OutputT> struct Params { void set(Data const&); }
//       Policy-specific runtime data, populated from the host-side Data struct.
//       Empty for policies that don't need extra data (zero register cost).
//   - template <typename DataType, int VecSize, typename ParamsT>
//     static void apply(warp, score[VecSize], idx[VecSize], numExperts, params)
//       Warp-per-token interface.  Transforms per-lane register scores in-place
//       before topK selection.  Used by the fused cluster / coop / dyn-block
//       kernels, where 32 lanes collectively process one token's experts.
//
// A policy MAY additionally provide the block-per-token interface used by
// routingIndicesBlockScoresKernel.  Opting in requires:
//   - static constexpr bool kSupportsBlockPerToken = true;
//   - template <typename SmemT, typename InputT, typename ParamsT>
//     static void applyToSmem(block, ptrScores, numExperts, smemBiased, smemAux, params)
//       Reads a token's raw scores from global memory, writes the per-expert
//       "biased" (topK key) and "aux" (postprocess input) values into smem.
//       `smemBiased` and `smemAux` may alias — callers that don't need aux
//       data pass the same pointer for both.
// The block-per-token kernel is enabled only for (pre, post) pairs where
// *both* policies set kSupportsBlockPerToken = true; see
// PolicyPairSupportsBlockPerToken below.
////////////////////////////////////////////////////////////////////////////////////////////////////

/// No-op: scores are passed through unchanged.
struct NoOpPreprocess {
  /// Opts into the block-per-token kernel (provides applyToSmem below).
  static constexpr bool kSupportsBlockPerToken = true;

  /// BaseType: when no preprocess is applied, use the input type directly.
  template <typename InputT>
  using BaseType = InputT;

  template <typename OutputT>
  struct Params {
    void set(routingCustom::Data const& /*data*/) {}
  };

  template <typename DataType, int VecSize, typename ParamsT>
  __forceinline__ __device__ static void apply(cg::thread_block_tile<WarpSize> const& /*warp*/,
                                               DataType (& /*score*/)[VecSize],
                                               int32_t const (& /*idx*/)[VecSize],
                                               int32_t /*numExperts*/, ParamsT const& /*params*/) {}

  /// Block-per-token interface: copy raw scores into smem.  smemAux aliases
  /// smemBiased because no postprocess needs separate aux data.
  template <typename SmemT, typename InputT, typename ParamsT>
  __forceinline__ __device__ static void applyToSmem(cg::thread_block const& block,
                                                     InputT const* ptrScores, int32_t numExperts,
                                                     SmemT* smemBiased, SmemT* /*smemAux*/,
                                                     ParamsT const& /*params*/) {
    for (int e = block.thread_rank(); e < numExperts; e += block.size()) {
      smemBiased[e] = static_cast<SmemT>(ptrScores[e]);
    }
  }
};

/// Softmax: applies softmax over all expert scores before topK selection.
struct SoftmaxPreprocess {
  /// Opts into the block-per-token kernel (provides applyToSmem below).
  static constexpr bool kSupportsBlockPerToken = true;

  /// BaseType: softmax is always computed in float for numerical stability.
  template <typename InputT>
  using BaseType = float;

  template <typename OutputT>
  struct Params {
    void set(routingCustom::Data const& /*data*/) {}
  };

  template <typename DataType, int VecSize, typename ParamsT>
  __forceinline__ __device__ static void apply(cg::thread_block_tile<WarpSize> const& warp,
                                               DataType (&score)[VecSize],
                                               int32_t const (& /*idx*/)[VecSize],
                                               int32_t /*numExperts*/, ParamsT const& /*params*/) {
    calcSoftmax(warp, score);
  }

  /// Block-per-token interface: three-pass softmax over all experts.
  ///   Pass 1: find block-wide max and stash raw scores in smemBiased.
  ///   Pass 2: compute exp(score - max) into smemBiased and reduce block sum.
  ///   Pass 3: normalize by the reduced sum.
  /// smemAux aliases smemBiased (postprocess doesn't need pre-softmax scores).
  /// Preconditions:
  ///   - kBlockDim must match the kernel's blockDim.x so BlockReduce uses the
  ///     correct temp storage layout.
  ///   - The caller must launch enough threads to cover expert slots; work is
  ///     striped across experts via e += block.size().
  template <int kBlockDim, typename SmemT, typename InputT, typename ParamsT>
  __forceinline__ __device__ static void applyToSmem(cg::thread_block const& block,
                                                     InputT const* ptrScores, int32_t numExperts,
                                                     SmemT* smemBiased, SmemT* /*smemAux*/,
                                                     ParamsT const& /*params*/) {
    using BlockReduce = cub::BlockReduce<float, kBlockDim>;
    __shared__ typename BlockReduce::TempStorage reduceStorage;
    __shared__ float smemBlockMax;
    __shared__ float smemBlockSum;

    // Pass 1: block-wide max.  `fmaxf` / `cuda::maximum<>` map to hardware
    // `MAX.F32` and follow IEEE 754 NaN handling — both are used elsewhere in
    // trtllm_backend (e.g. trtllm_fused_moe_dev_kernel.cu).
    float localMax = -INFINITY;
    for (int e = block.thread_rank(); e < numExperts; e += block.size()) {
      float s = static_cast<float>(ptrScores[e]);
      smemBiased[e] = static_cast<SmemT>(s);  // stash raw score for pass 2
      localMax = fmaxf(s, localMax);
    }
    float blockMax = BlockReduce(reduceStorage).Reduce(localMax, cuda::maximum<>{});
    if (block.thread_rank() == 0) smemBlockMax = blockMax;
    __syncthreads();
    float const mx = smemBlockMax;

    // Pass 2: compute exp(score - max) into smemBiased, accumulate sum.
    float localSum = 0.f;
    for (int e = block.thread_rank(); e < numExperts; e += block.size()) {
      float v = expf(static_cast<float>(smemBiased[e]) - mx);
      smemBiased[e] = static_cast<SmemT>(v);
      localSum += v;
    }
    float blockSum = BlockReduce(reduceStorage).Sum(localSum);
    if (block.thread_rank() == 0) smemBlockSum = blockSum;
    __syncthreads();
    float const invSum = 1.f / smemBlockSum;

    // Pass 3: normalize.
    for (int e = block.thread_rank(); e < numExperts; e += block.size()) {
      smemBiased[e] = static_cast<SmemT>(static_cast<float>(smemBiased[e]) * invSum);
    }
  }
};

/// Sigmoid: applies sigmoid(score) for topK selection (no bias).
struct SigmoidPreprocess {
  /// Opts into the block-per-token kernel (provides applyToSmem below).
  static constexpr bool kSupportsBlockPerToken = true;

  /// BaseType: sigmoid is computed in float for numerical stability.
  template <typename InputT>
  using BaseType = float;

  template <typename OutputT>
  struct Params {
    void set(routingCustom::Data const& /*data*/) {}
  };

  template <typename DataType, int VecSize, typename ParamsT>
  __forceinline__ __device__ static void apply(cg::thread_block_tile<WarpSize> const& /*warp*/,
                                               DataType (&score)[VecSize],
                                               int32_t const (&idx)[VecSize], int32_t numExperts,
                                               ParamsT const& /*params*/) {
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      float s = sigmoid_accurate(static_cast<float>(score[i]));
      score[i] = idx[i] < numExperts ? static_cast<DataType>(s) : DataType{-INFINITY};
    }
  }

  /// Block-per-token interface: compute sigmoid per expert and write to smem.
  /// smemAux aliases smemBiased (SumNormalize postprocess only uses the
  /// biased top-K values, so no separate aux data is needed).
  template <typename SmemT, typename InputT, typename ParamsT>
  __forceinline__ __device__ static void applyToSmem(cg::thread_block const& block,
                                                     InputT const* ptrScores, int32_t numExperts,
                                                     SmemT* smemBiased, SmemT* /*smemAux*/,
                                                     ParamsT const& /*params*/) {
    for (int e = block.thread_rank(); e < numExperts; e += block.size()) {
      float s = sigmoid_accurate(static_cast<float>(ptrScores[e]));
      smemBiased[e] = static_cast<SmemT>(s);
    }
  }
};

/// SigmoidBias: applies sigmoid(score) + bias[expertIdx] for topK selection.
/// Used by DeepSeek-style routing where expert selection is based on biased sigmoid scores.
struct SigmoidBiasPreprocess {
  /// Opts into the block-per-token kernel (provides applyToSmem below).
  static constexpr bool kSupportsBlockPerToken = true;

  /// BaseType: sigmoid is computed in float for numerical stability.
  template <typename InputT>
  using BaseType = float;

  template <typename OutputT>
  struct Params {
    // Store as void const* to support any bias dtype (float, bfloat16, etc.) without conversion.
    void const* ptrRoutingBias = nullptr;
    batchedGemm::trtllm::gen::Dtype dtypeBias = batchedGemm::trtllm::gen::Dtype::Bfloat16;

    void set(routingCustom::Data const& data) {
      ptrRoutingBias = data.mPtrRoutingBias;
      dtypeBias = data.mDtypeBias;
    }
  };

  template <typename DataType, int VecSize, typename ParamsT>
  __forceinline__ __device__ static void apply(cg::thread_block_tile<WarpSize> const& /*warp*/,
                                               DataType (&score)[VecSize],
                                               int32_t const (&idx)[VecSize], int32_t numExperts,
                                               ParamsT const& params) {
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      float s = sigmoid_accurate(static_cast<float>(score[i]));
      float bias = idx[i] < numExperts ? loadScalar(params.ptrRoutingBias, idx[i], params.dtypeBias)
                                       : float{-INFINITY};
      score[i] = static_cast<DataType>(s + bias);
    }
  }

  /// Block-per-token interface: compute (sigmoid, sigmoid+bias) per expert.
  /// `smemAux[e] = sigmoid(score[e])`         — read by ScaledSumNormalizePostprocess
  /// `smemBiased[e] = sigmoid(score[e]) + bias[e]` — used as the topK selection key
  /// The two arrays must be distinct (cannot alias).
  template <typename SmemT, typename InputT, typename ParamsT>
  __forceinline__ __device__ static void applyToSmem(cg::thread_block const& block,
                                                     InputT const* ptrScores, int32_t numExperts,
                                                     SmemT* smemBiased, SmemT* smemAux,
                                                     ParamsT const& params) {
    for (int e = block.thread_rank(); e < numExperts; e += block.size()) {
      float s = sigmoid_accurate(static_cast<float>(ptrScores[e]));
      float bias = loadScalar(params.ptrRoutingBias, e, params.dtypeBias);
      smemAux[e] = static_cast<SmemT>(s);
      smemBiased[e] = static_cast<SmemT>(s + bias);
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Postprocess policies: applied to the top-K scores AFTER topK selection.
//
// Each policy must provide:
//   - template <typename OutputT> struct Params { void set(Data const&); }
//       Policy-specific runtime data. Empty when not needed.
//   - template <typename DataType, int K, typename ParamsT>
//     static void apply(warp, warpTopKScore[K], warpTopKExpertIdx[K], laneIdx, topK, params)
//       Transforms top-K scores in-place after topK selection.  Used by the
//       warp-per-token kernels.
//
// A policy MAY additionally provide the block-per-token interface used by
// routingIndicesBlockScoresKernel.  Opting in requires:
//   - static constexpr bool kSupportsBlockPerToken = true;
//   - template <typename DataType, int K, typename SmemT, typename ParamsT>
//     static void applyWithAux(warp, warpTopKScore[K], warpTopKExpertIdx[K], laneIdx, topK,
//                               smemAux, params)
//       Same as `apply` but additionally receives `smemAux` — the per-expert
//       auxiliary data written by the preprocess policy.  Policies that do
//       not need aux data can default-forward to `apply(...)`; policies that
//       do (e.g. ScaledSumNormalize needs un-biased sigmoid) read from
//       smemAux[warpTopKExpertIdx[laneIdx]].
// Postprocess policies that read per-expert aux data must additionally set
//   - static constexpr bool kNeedsAux = true;
// so the block-per-token kernel allocates a separate smemAux array instead
// of aliasing it to smemBiased.  Default: false (smemAux aliases smemBiased,
// saving MaxNumExperts × 4B of smem).
////////////////////////////////////////////////////////////////////////////////////////////////////

/// No-op: top-K scores are left unchanged.
struct NoOpPostprocess {
  /// Opts into the block-per-token kernel (provides applyWithAux below).
  static constexpr bool kSupportsBlockPerToken = true;

  template <typename OutputT>
  struct Params {
    void set(routingCustom::Data const& /*data*/) {}
  };

  template <typename DataType, int K, typename ParamsT>
  __forceinline__ __device__ static void apply(cg::thread_block_tile<WarpSize> const& /*warp*/,
                                               DataType (& /*warpTopKScore*/)[K],
                                               int32_t const (& /*warpTopKExpertIdx*/)[K],
                                               int32_t /*laneIdx*/, int32_t /*topK*/,
                                               ParamsT const& /*params*/) {}

  template <typename DataType, int K, typename SmemT, typename ParamsT>
  __forceinline__ __device__ static void applyWithAux(cg::thread_block_tile<WarpSize> const& warp,
                                                      DataType (&warpTopKScore)[K],
                                                      int32_t const (&warpTopKExpertIdx)[K],
                                                      int32_t laneIdx, int32_t topK,
                                                      SmemT const* /*smemAux*/,
                                                      ParamsT const& params) {
    apply(warp, warpTopKScore, warpTopKExpertIdx, laneIdx, topK, params);
  }
};

/// Softmax: applies softmax over the top-K scores.
struct SoftmaxPostprocess {
  /// Opts into the block-per-token kernel (provides applyWithAux below).
  static constexpr bool kSupportsBlockPerToken = true;

  template <typename OutputT>
  struct Params {
    void set(routingCustom::Data const& /*data*/) {}
  };

  template <typename DataType, int K, typename ParamsT>
  __forceinline__ __device__ static void apply(cg::thread_block_tile<WarpSize> const& warp,
                                               DataType (&warpTopKScore)[K],
                                               int32_t const (& /*warpTopKExpertIdx*/)[K],
                                               int32_t laneIdx, int32_t topK,
                                               ParamsT const& /*params*/) {
    DataType minScore = DataType{-INFINITY};
    auto softmaxScore =
        calcSoftmax(warp, laneIdx < topK ? warpTopKScore[laneIdx] : minScore, laneIdx, topK);
    if (laneIdx < topK) {
      warpTopKScore[laneIdx] = softmaxScore;
    }
  }

  template <typename DataType, int K, typename SmemT, typename ParamsT>
  __forceinline__ __device__ static void applyWithAux(cg::thread_block_tile<WarpSize> const& warp,
                                                      DataType (&warpTopKScore)[K],
                                                      int32_t const (&warpTopKExpertIdx)[K],
                                                      int32_t laneIdx, int32_t topK,
                                                      SmemT const* /*smemAux*/,
                                                      ParamsT const& params) {
    apply(warp, warpTopKScore, warpTopKExpertIdx, laneIdx, topK, params);
  }
};

/// SumNormalize: divides each top-K score by the sum of all top-K scores.
/// Used when softmax has already been applied before topK selection.
struct SumNormalizePostprocess {
  /// Opts into the block-per-token kernel (provides applyWithAux below).
  static constexpr bool kSupportsBlockPerToken = true;

  template <typename OutputT>
  struct Params {
    bool normTopkProb = true;

    void set(routingCustom::Data const& data) { normTopkProb = data.mNormTopkProb; }
  };

  template <typename DataType, int K, typename ParamsT>
  __forceinline__ __device__ static void apply(cg::thread_block_tile<WarpSize> const& warp,
                                               DataType (&warpTopKScore)[K],
                                               int32_t const (& /*warpTopKExpertIdx*/)[K],
                                               int32_t laneIdx, int32_t topK,
                                               ParamsT const& params) {
    float sum = float{1.f};
    if (params.normTopkProb) {
      sum = static_cast<float>(laneIdx < topK ? warpTopKScore[laneIdx] : 0);
      sum = cg::reduce(warp, sum, cg::plus<float>());
    }
    if (laneIdx < topK) {
      float denom = params.normTopkProb ? fmaxf(sum, 1e-20f) : 1.f;
      warpTopKScore[laneIdx] = warpTopKScore[laneIdx] / denom;
    }
  }

  template <typename DataType, int K, typename SmemT, typename ParamsT>
  __forceinline__ __device__ static void applyWithAux(cg::thread_block_tile<WarpSize> const& warp,
                                                      DataType (&warpTopKScore)[K],
                                                      int32_t const (&warpTopKExpertIdx)[K],
                                                      int32_t laneIdx, int32_t topK,
                                                      SmemT const* /*smemAux*/,
                                                      ParamsT const& params) {
    apply(warp, warpTopKScore, warpTopKExpertIdx, laneIdx, topK, params);
  }
};

/// ScaledSumNormalize: recovers un-biased sigmoid scores by subtracting per-expert bias from the
/// selection scores (sigmoid + bias), then normalizes by sum and applies routeScale.
/// Used by DeepSeek-style routing: final_weight = sigmoid(raw) * routeScale / (sum + epsilon).
/// DeepSeek uses epsilon=0 (no guard); MiniMax2 uses epsilon=1e-20 to prevent division by zero.
struct ScaledSumNormalizePostprocess {
  /// Opts into the block-per-token kernel (provides applyWithAux below).
  static constexpr bool kSupportsBlockPerToken = true;

  /// Needs per-expert aux data (un-biased sigmoid) in the block-per-token
  /// kernel — paired with SigmoidBiasPreprocess, which writes the un-biased
  /// sigmoid to smemAux[e].  The kernel allocates a distinct smemAux array
  /// instead of aliasing it to smemBiased.
  static constexpr bool kNeedsAux = true;

  template <typename OutputT>
  struct Params {
    // Store as void const* to support any bias dtype (float, bfloat16, etc.) without conversion.
    void const* ptrRoutingBias = nullptr;
    batchedGemm::trtllm::gen::Dtype dtypeBias = batchedGemm::trtllm::gen::Dtype::Bfloat16;
    float routeScale = 1.0f;
    float sumEpsilon = 0.0f;

    void set(routingCustom::Data const& data) {
      ptrRoutingBias = data.mPtrRoutingBias;
      dtypeBias = data.mDtypeBias;
      routeScale = data.mRouteScale;
      sumEpsilon = data.mSumEpsilon;
    }
  };

  template <typename DataType, int K, typename ParamsT>
  __forceinline__ __device__ static void apply(cg::thread_block_tile<WarpSize> const& warp,
                                               DataType (&warpTopKScore)[K],
                                               int32_t const (&warpTopKExpertIdx)[K],
                                               int32_t laneIdx, int32_t topK,
                                               ParamsT const& params) {
    // Recover sigmoid score: selection_score = sigmoid(raw) + bias, so sigmoid = score - bias
    float biasVal = laneIdx < topK ? loadScalar(params.ptrRoutingBias, warpTopKExpertIdx[laneIdx],
                                                params.dtypeBias)
                                   : 0.f;
    float sigmoidScore =
        laneIdx < topK ? (static_cast<float>(warpTopKScore[laneIdx]) - biasVal) : 0.f;
    float sum = cg::reduce(warp, sigmoidScore, cg::plus<float>());
    if (laneIdx < topK) {
      warpTopKScore[laneIdx] =
          static_cast<DataType>(sigmoidScore * params.routeScale / (sum + params.sumEpsilon));
    }
  }

  /// Block-per-token variant: read un-biased sigmoid directly from smemAux
  /// (written by SigmoidBiasPreprocess::applyToSmem) instead of reloading bias
  /// and subtracting.  Saves one global memory read + one FMA per top-K lane.
  template <typename DataType, int K, typename SmemT, typename ParamsT>
  __forceinline__ __device__ static void applyWithAux(cg::thread_block_tile<WarpSize> const& warp,
                                                      DataType (&warpTopKScore)[K],
                                                      int32_t const (&warpTopKExpertIdx)[K],
                                                      int32_t laneIdx, int32_t topK,
                                                      SmemT const* smemAux, ParamsT const& params) {
    float sigmoidScore =
        laneIdx < topK ? static_cast<float>(smemAux[warpTopKExpertIdx[laneIdx]]) : 0.f;
    float sum = cg::reduce(warp, sigmoidScore, cg::plus<float>());
    if (laneIdx < topK) {
      warpTopKScore[laneIdx] =
          static_cast<DataType>(sigmoidScore * params.routeScale / (sum + params.sumEpsilon));
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// ExpertSelectPolicy: encapsulates the entire expert selection logic.
//
// Each policy must provide:
//   - template <typename InputT> using BaseType
//       The data type used for intermediate score computation.
//   - template <typename OutputT> struct Params { void set(Data const&); }
//       Policy-specific runtime data, populated from the host-side Data struct.
//       Empty for policies that don't need extra data (zero register cost).
//   - template <typename DataType, typename InputType, int VecSize, int K, typename KP>
//     static void apply(warp, warpTopKScore[K], warpTopKExpertIdx[K], laneIdx, numExperts, topK,
//                        ptrScores, params)
//       Selects the top-K experts and computes their weights.
//
// The default TopKExpertSelect wraps existing PreprocessPolicy + PostprocessPolicy,
// but users can write completely custom policies that bypass the preprocess+topK+postprocess
// pattern (e.g., lookup-table-based expert selection).
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Trait: does a (pre, post) policy pair need a separate `smemAux` array in
/// the block-per-token kernel?  This is purely a postprocess-side property:
/// true iff the postprocess policy reads per-expert auxiliary data that
/// differs from the topK selection key (via its `kNeedsAux` member).  Today
/// only `ScaledSumNormalize` sets `kNeedsAux = true`.  For every other
/// postprocess, `smemAux` aliases `smemBiased` and no extra smem is
/// allocated.  Postprocess policies that don't declare `kNeedsAux` are
/// treated as not needing it (safe default).
template <typename PostprocessPolicy_, typename = void>
struct PostprocessNeedsAux : std::false_type {};

template <typename PostprocessPolicy_>
struct PostprocessNeedsAux<PostprocessPolicy_, std::void_t<decltype(PostprocessPolicy_::kNeedsAux)>>
    : std::bool_constant<PostprocessPolicy_::kNeedsAux> {};

template <typename PreprocessPolicy_, typename PostprocessPolicy_>
struct PolicyPairNeedsAux : PostprocessNeedsAux<PostprocessPolicy_> {};

/// Trait: does a (pre, post) policy pair implement the block-per-token
/// interface (`PreProc::applyToSmem` + `PostProc::applyWithAux`) and
/// therefore opt in to `routingIndicesBlockScoresKernel`?  A pair opts in
/// iff *both* policies declare `kSupportsBlockPerToken = true`.
///
/// To opt a new policy in:
///   1. Implement the block-per-token method on the policy itself
///      (`applyToSmem` for preprocess, `applyWithAux` for postprocess) —
///      see the "Policy Interfaces" doc at the top of this file.
///   2. Add `static constexpr bool kSupportsBlockPerToken = true;` to the
///      policy struct.
/// Once both members of a pair opt in, every (this_policy, other_policy)
/// combination works with the block-per-token kernel automatically — no
/// per-pair trait specialisation needed.  Every preprocess / postprocess
/// policy defined above must declare `kSupportsBlockPerToken` (true or
/// false); missing it is a compile error, which intentionally forces new
/// policies to make an explicit choice.
template <typename PreprocessPolicy_, typename PostprocessPolicy_>
struct PolicyPairSupportsBlockPerToken
    : std::bool_constant<PreprocessPolicy_::kSupportsBlockPerToken &&
                         PostprocessPolicy_::kSupportsBlockPerToken> {};

/// Default ExpertSelectPolicy: preprocess + topK reduction + postprocess.
/// Wraps existing PreprocessPolicy and PostprocessPolicy as internal composition.
template <typename PreprocessPolicy_, typename PostprocessPolicy_>
struct TopKExpertSelect {
  /// Expose component policies so the block-per-token kernel can dispatch
  /// directly on preprocess / postprocess without going through `apply()`.
  using PreprocessPolicy = PreprocessPolicy_;
  using PostprocessPolicy = PostprocessPolicy_;

  /// BaseType: delegated to the preprocess policy.
  template <typename InputT>
  using BaseType = typename PreprocessPolicy_::template BaseType<InputT>;

  /// Params: combines preprocess and postprocess runtime parameters.
  template <typename OutputT>
  struct Params {
    typename PreprocessPolicy_::template Params<OutputT> mPreprocessParams;
    typename PostprocessPolicy_::template Params<OutputT> mPostprocessParams;

    void set(routingCustom::Data const& data) {
      mPreprocessParams.set(data);
      mPostprocessParams.set(data);
    }
  };

  /// Selects top-K experts using preprocess → topK reduction → postprocess.
  template <typename DataType, typename InputType, int VecSize, int K, typename KP>
  __forceinline__ __device__ static void apply(cg::thread_block_tile<WarpSize> const& warp,
                                               DataType (&warpTopKScore)[K],
                                               int32_t (&warpTopKExpertIdx)[K],
                                               int32_t const laneIdx, int32_t const numExperts,
                                               int32_t topK, InputType const* ptrScores,
                                               KP const& params) {
    DataType minScore = DataType{-INFINITY};
    DataType score[VecSize];
    int32_t idx[VecSize];

    for (int i = 0; i < VecSize; i++) {
      auto expertIdx = i * WarpSize + laneIdx;
      auto newScore =
          expertIdx < numExperts ? static_cast<DataType>(ptrScores[expertIdx]) : minScore;
      score[i] = newScore;
      idx[i] = expertIdx;
    }

    // Apply preprocess (e.g. softmax over all scores, sigmoid + bias, ...)
    PreprocessPolicy_::apply(warp, score, idx, numExperts,
                             params.mExpertSelectParams.mPreprocessParams);

    // Get the top-k scores and their corresponding expert indices
    topk::reduceTopK(warp, warpTopKScore, warpTopKExpertIdx, score, idx, minScore, topK);

    // Apply postprocess (e.g. renormalize, softmax over top-K, scaled renormalize, ...)
    PostprocessPolicy_::apply(warp, warpTopKScore, warpTopKExpertIdx, laneIdx, topK,
                              params.mExpertSelectParams.mPostprocessParams);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace routingCustom {
////////////////////////////////////////////////////////////////////////////////////////////////////

// Expert-count tiers (must be multiples of WarpSize=32 and of 4).
// Each tier covers all values ≤ the tier constant.
static constexpr int NumExperts128Experts = 128;
static constexpr int NumExperts160Experts = 160;
static constexpr int NumExperts256Experts = 256;
static constexpr int NumExperts384Experts = 384;
static constexpr int NumExperts512Experts = 512;
static constexpr int NumExperts576Experts = 576;
static constexpr int NumExperts1024Experts = 1024;
static constexpr int MaxSupportedExperts = 2048;

// TopK tiers (must be ≤ WarpSize=32).
static constexpr int NumTop4Experts = 4;
static constexpr int NumTop8Experts = 8;
static constexpr int NumTop16Experts = 16;
static constexpr int NumTop22Experts = 22;
static constexpr int MaxSupportedTopExperts = 32;

static constexpr int NumThreads = 1024;
static constexpr int NumWarps = NumThreads / WarpSize;

static constexpr int MaxNumTokensSingleCluster = NumBlocksPerCluster * NumThreads;
static constexpr int MaxNumTokensSingleClusterScores = NumBlocksPerCluster * NumWarps;

static constexpr int BlockKernelMaxNumTokens = 4;
static constexpr int DynBlockKernelMaxNumTokens = 16;
static constexpr int DynBlockKernelMaxNumExperts = 512;

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t getMaxNumExperts(int32_t numExperts) {
  if (numExperts <= NumExperts128Experts) {
    return NumExperts128Experts;
  } else if (numExperts <= NumExperts160Experts) {
    return NumExperts160Experts;
  } else if (numExperts <= NumExperts256Experts) {
    return NumExperts256Experts;
  } else if (numExperts <= NumExperts384Experts) {
    return NumExperts384Experts;
  } else if (numExperts <= NumExperts512Experts) {
    return NumExperts512Experts;
  } else if (numExperts <= NumExperts576Experts) {
    return NumExperts576Experts;
  } else if (numExperts <= NumExperts1024Experts) {
    return NumExperts1024Experts;
  } else if (numExperts <= MaxSupportedExperts) {
    return MaxSupportedExperts;
  } else {
    FLASHINFER_WARN("Unsupported numExperts");
    return 0;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// TIER PAIR TYPES — compile-time (MaxNumExperts, MaxNumTopExperts) configuration.
//
// Each Tier<E, K> declares a supported kernel instantiation.
// TierList<Tier<...>, ...> is an ordered list tried from first to last.
// The dispatch picks the FIRST pair where numExperts ≤ E AND topK ≤ K.
//
// Pairs must be sorted so that tighter tiers come first:
//   - Sort by E ascending, then by K ascending within equal E.
//   - A config (numExperts, topK) always matches the tightest available pair.
//   - If the tightest expert tier doesn't have a topK that covers the runtime topK,
//     the dispatch falls through to the next larger expert tier that does.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int E_, int K_>
struct Tier {
  static constexpr int kExperts = E_;
  static constexpr int kTopK = K_;
};

template <typename... Tiers>
struct TierList {};

// Recursive dispatch: try each tier in order, call `fn` with the first match.
// fn receives (integral_constant<int, E>, integral_constant<int, K>) as compile-time args.
// Base case: empty list — no match.
template <typename Fn, typename Data>
inline bool dispatchTierPairs(TierList<>*, Data const& /*data*/, Fn&& /*fn*/) {
  return false;
}

// Recursive case: check First, then recurse on Rest...
template <typename First, typename... Rest, typename Fn, typename Data>
inline bool dispatchTierPairs(TierList<First, Rest...>*, Data const& data, Fn&& fn) {
  if (data.mNumExperts <= First::kExperts && data.mTopK <= First::kTopK) {
    fn(std::integral_constant<int, First::kExperts>{}, std::integral_constant<int, First::kTopK>{});
    return true;
  }
  return dispatchTierPairs(static_cast<TierList<Rest...>*>(nullptr), data, std::forward<Fn>(fn));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// POLICY TIER CONFIGURATION
//
// PolicyTraits<PreProc, PostProc>::Pairs declares the supported (expert, topK) pairs.
// Only these pairs are compiled as kernel instantiations.
// To add support for a new model config, add a Tier<E, K> to the appropriate TierList.
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Default: fallback for new/unknown policies.
template <typename PreProc, typename PostProc>
struct PolicyTraits {
  using Pairs = TierList<Tier<128, 8>, Tier<128, 32>, Tier<256, 8>, Tier<256, 32>, Tier<512, 8>,
                         Tier<512, 32>, Tier<2048, 8>, Tier<2048, 32>>;
};

/// Softmax + None (Default: Softmax -> TopK).
/// NOTE: Currently only covers ≤256 experts. If a model requires more, add a larger Tier here.
template <>
struct PolicyTraits<SoftmaxPreprocess, NoOpPostprocess> {
  using Pairs = TierList<Tier<128, 8>,  // Small expert counts (≤128 experts)
                         Tier<256, 8>   // Medium expert counts (≤256 experts)
                         >;
};

/// None + Softmax (Renormalize): many model configs.
template <>
struct PolicyTraits<NoOpPreprocess, SoftmaxPostprocess> {
  using Pairs = TierList<Tier<128, 4>,   // Mixtral 8x7B (topK=2), Qwen2-MoE (topK=4), Arctic
                                         // (topK=2), DBRX (topK=4), GPT-OSS
                         Tier<128, 8>,   // DeepSeek-V2-Lite (topK=6), Mixtral 8x22B (topK=2)
                         Tier<160, 8>,   // Qwen3-Coder-480B
                         Tier<256, 8>,   // Mistral Large 3 (topK=8)
                         Tier<256, 16>,  // Models with 256 experts and topK 9..16
                         Tier<512, 8>,   // Various 512-expert models
                         Tier<512, 16>,  // Various 512-expert models with high topK
                         Tier<512, 22>,  // Nemotron Super V3 (512 experts, topK=22)
                         Tier<576, 8>,   // Customized model with 576 experts
                         Tier<2048, 32>  // Large-expert fallback
                         >;
};

/// Sigmoid + SumNormalize (SigmoidRenorm: Sigmoid -> TopK -> Renormalize,
///                          Sigmoid: Sigmoid -> TopK with normTopkProb=false).
/// NOTE: Currently only covers ≤256 experts. If a model requires more, add a larger Tier here.
template <>
struct PolicyTraits<SigmoidPreprocess, SumNormalizePostprocess> {
  using Pairs = TierList<Tier<128, 8>,  // Small expert counts (≤128 experts)
                         Tier<256, 8>   // Medium expert counts (≤256 experts)
                         >;
};

/// SigmoidBias + ScaledSumNormalize (DeepSeek nGroup≤1 / MiniMax2 / Kimi-K2 / Nemotron SuperV3).
template <>
struct PolicyTraits<SigmoidBiasPreprocess, ScaledSumNormalizePostprocess> {
  using Pairs = TierList<Tier<128, 8>,  // Small expert counts (≤128 experts, e.g. DeepSeek-V2-Lite)
                         Tier<256, 8>,  // MiniMax M2 (256 experts, topK=6)
                         Tier<384, 8>,  // Kimi K2 (384 experts)
                         Tier<512, 8>,  // DeepSeek nGroup≤1 (256 experts → E512 fallback)
                         Tier<512, 22>,  // Nemotron Super V3 (512 experts, topK=22, nGroup≤1)
                         Tier<1024, 32>  // Default fallback (expert count may grow beyond 512)
                         >;
};

/// None + None (TopK only: no softmax or renormalize).
/// NOTE: Currently only covers ≤256 experts. If a model requires more, add a larger Tier here.
template <>
struct PolicyTraits<NoOpPreprocess, NoOpPostprocess> {
  using Pairs = TierList<Tier<128, 8>,  // Small expert counts (≤128 experts)
                         Tier<256, 8>   // Medium expert counts (≤256 experts)
                         >;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// GENERIC DISPATCH MACROS
//
// These macros are fixed infrastructure — they never need editing when adding new
// policies or changing tier support.  All configuration lives in PolicyTraits above.
////////////////////////////////////////////////////////////////////////////////////////////////////

// Generic per-policy dispatch.  Iterates PolicyTraits<PreProc, PostProc>::Pairs,
// picking the first (expert, topK) pair that covers the runtime values.
//
// IMPORTANT: numThreads is clamped to at least min(MaxNumExperts, 1024) from the dispatched tier.
#define LAUNCH_ROUTING_FOR_POLICY(data, coopLaunch, kernel, numBlocks, numThreads, smemSize,     \
                                  stream, PreProc, PostProc)                                     \
  [&](auto pt_tag_) {                                                                            \
    using Pairs_ = typename decltype(pt_tag_)::Pairs;                                            \
    bool dispatched_ =                                                                           \
        dispatchTierPairs(static_cast<Pairs_*>(nullptr), data, [&](auto eTag_, auto kTag_) {     \
          constexpr int tierMaxExp_ = decltype(eTag_)::value;                                    \
          constexpr int tierThreads_ = tierMaxExp_ <= 1024 ? tierMaxExp_ : 1024;                 \
          int const effectiveThreads_ = std::max(static_cast<int>(numThreads), tierThreads_);    \
          LAUNCH_ROUTING_WITH_POLICIES(data, coopLaunch, kernel, numBlocks, effectiveThreads_,   \
                                       smemSize, stream, PreProc, PostProc,                      \
                                       decltype(eTag_)::value, decltype(kTag_)::value);          \
        });                                                                                      \
    if (!dispatched_) {                                                                          \
      FLASHINFER_WARN(                                                                           \
          "No compiled tier covers numExperts=%d topK=%d for policy %s+%s. "                     \
          "Add a Tier<%d, %d> to the corresponding PolicyTraits in RoutingCustomPolicy.cuh.",    \
          data.mNumExperts, data.mTopK, #PreProc, #PostProc, getMaxNumExperts(data.mNumExperts), \
          data.mTopK);                                                                           \
    }                                                                                            \
  }(PolicyTraits<PreProc, PostProc>{})

////////////////////////////////////////////////////////////////////////////////////////////////////
// CUSTOM EXPERT SELECT DISPATCH
////////////////////////////////////////////////////////////////////////////////////////////////////

// Generic dispatch for custom ExpertSelectPolicy. PolicyTraits key is <ExpertSelect, void>.
// Same numThreads clamping as LAUNCH_ROUTING_FOR_POLICY — see comment above.
#define LAUNCH_ROUTING_FOR_EXPERT_SELECT(data, coopLaunch, kernel, numBlocks, numThreads,      \
                                         smemSize, stream, ExpertSelect)                       \
  [&](auto pt_tag_) {                                                                          \
    using Pairs_ = typename decltype(pt_tag_)::Pairs;                                          \
    bool dispatched_ =                                                                         \
        dispatchTierPairs(static_cast<Pairs_*>(nullptr), data, [&](auto eTag_, auto kTag_) {   \
          constexpr int tierMaxExp_ = decltype(eTag_)::value;                                  \
          constexpr int tierThreads_ = tierMaxExp_ <= 1024 ? tierMaxExp_ : 1024;               \
          int const effectiveThreads_ = std::max(static_cast<int>(numThreads), tierThreads_);  \
          LAUNCH_ROUTING_WITH_EXPERT_SELECT(data, coopLaunch, kernel, numBlocks,               \
                                            effectiveThreads_, smemSize, stream, ExpertSelect, \
                                            decltype(eTag_)::value, decltype(kTag_)::value);   \
        });                                                                                    \
    if (!dispatched_) {                                                                        \
      FLASHINFER_WARN(                                                                         \
          "No compiled tier covers numExperts=%d topK=%d for ExpertSelect policy %s. "         \
          "Add a Tier<%d, %d> to PolicyTraits<%s, void> in RoutingCustomPolicy.cuh.",          \
          data.mNumExperts, data.mTopK, #ExpertSelect, getMaxNumExperts(data.mNumExperts),     \
          data.mTopK, #ExpertSelect);                                                          \
    }                                                                                          \
  }(PolicyTraits<ExpertSelect, void>{})

////////////////////////////////////////////////////////////////////////////////////////////////////
// PUBLIC DISPATCH MACROS
////////////////////////////////////////////////////////////////////////////////////////////////////

// Lightweight dispatch for utility kernels (histogram, init-counts, offsets) that do NOT use
// expert select policies, InputT, or MaxNumTopExperts.
// - Always uses NoOp expert select (no policy dispatch).
// - Always uses a fixed NumTop8Experts (no topK-tier dispatch).
// - Dispatches only on expert tiers.
#define LAUNCH_ROUTING_CUSTOM_NO_POLICY(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, \
                                        stream)                                                    \
  if (data.mNumExperts <= NumExperts128Experts) {                                                  \
    LAUNCH_ROUTING_WITH_POLICIES(data, coopLaunch, kernel, numBlocks, numThreads, smemSize,        \
                                 stream, NoOpPreprocess, NoOpPostprocess, NumExperts128Experts,    \
                                 NumTop8Experts);                                                  \
  } else if (data.mNumExperts <= NumExperts160Experts) {                                           \
    LAUNCH_ROUTING_WITH_POLICIES(data, coopLaunch, kernel, numBlocks, numThreads, smemSize,        \
                                 stream, NoOpPreprocess, NoOpPostprocess, NumExperts160Experts,    \
                                 NumTop8Experts);                                                  \
  } else if (data.mNumExperts <= NumExperts256Experts) {                                           \
    LAUNCH_ROUTING_WITH_POLICIES(data, coopLaunch, kernel, numBlocks, numThreads, smemSize,        \
                                 stream, NoOpPreprocess, NoOpPostprocess, NumExperts256Experts,    \
                                 NumTop8Experts);                                                  \
  } else if (data.mNumExperts <= NumExperts384Experts) {                                           \
    LAUNCH_ROUTING_WITH_POLICIES(data, coopLaunch, kernel, numBlocks, numThreads, smemSize,        \
                                 stream, NoOpPreprocess, NoOpPostprocess, NumExperts384Experts,    \
                                 NumTop8Experts);                                                  \
  } else if (data.mNumExperts <= NumExperts512Experts) {                                           \
    LAUNCH_ROUTING_WITH_POLICIES(data, coopLaunch, kernel, numBlocks, numThreads, smemSize,        \
                                 stream, NoOpPreprocess, NoOpPostprocess, NumExperts512Experts,    \
                                 NumTop8Experts);                                                  \
  } else if (data.mNumExperts <= NumExperts576Experts) {                                           \
    LAUNCH_ROUTING_WITH_POLICIES(data, coopLaunch, kernel, numBlocks, numThreads, smemSize,        \
                                 stream, NoOpPreprocess, NoOpPostprocess, NumExperts576Experts,    \
                                 NumTop8Experts);                                                  \
  } else if (data.mNumExperts <= NumExperts1024Experts) {                                          \
    LAUNCH_ROUTING_WITH_POLICIES(data, coopLaunch, kernel, numBlocks, numThreads, smemSize,        \
                                 stream, NoOpPreprocess, NoOpPostprocess, NumExperts1024Experts,   \
                                 NumTop8Experts);                                                  \
  } else if (data.mNumExperts <= MaxSupportedExperts) {                                            \
    LAUNCH_ROUTING_WITH_POLICIES(data, coopLaunch, kernel, numBlocks, numThreads, smemSize,        \
                                 stream, NoOpPreprocess, NoOpPostprocess, MaxSupportedExperts,     \
                                 NumTop8Experts);                                                  \
  } else {                                                                                         \
    FLASHINFER_WARN("Unsupported numExperts");                                                     \
  }

// Single source of truth for runtime → compile-time policy dispatch.
// Maps (mPreprocessType, mPostprocessType) to compile-time (PreProc, PostProc) policy types.
// LAUNCH_ROUTING_CUSTOM, queryDispatchedMaxExperts, and queryPolicySupportsBlockPerToken use this
// function, so they are always in sync.
// The callback receives (PreProc{}, PostProc{}, policyName) where policyName is a human-readable
// string for diagnostics.
template <typename Fn>
inline void dispatchRoutingPolicy(Data const& data, Fn&& fn) {
  if (data.mPreprocessType == RoutingPreprocessType::SigmoidBias)
    fn(SigmoidBiasPreprocess{}, ScaledSumNormalizePostprocess{},
       "SigmoidBiasPreprocess+ScaledSumNormalizePostprocess");
  else if (data.mPreprocessType == RoutingPreprocessType::Sigmoid)
    fn(SigmoidPreprocess{}, SumNormalizePostprocess{}, "SigmoidPreprocess+SumNormalizePostprocess");
  else if (data.mPreprocessType == RoutingPreprocessType::Softmax &&
           data.mPostprocessType == RoutingPostprocessType::None)
    fn(SoftmaxPreprocess{}, NoOpPostprocess{}, "SoftmaxPreprocess+NoOpPostprocess");
  else if (data.mPreprocessType == RoutingPreprocessType::Softmax)
    fn(SoftmaxPreprocess{}, SumNormalizePostprocess{}, "SoftmaxPreprocess+SumNormalizePostprocess");
  else if (data.mPostprocessType == RoutingPostprocessType::Softmax)
    fn(NoOpPreprocess{}, SoftmaxPostprocess{}, "NoOpPreprocess+SoftmaxPostprocess");
  else
    fn(NoOpPreprocess{}, NoOpPostprocess{}, "NoOpPreprocess+NoOpPostprocess");
}

// Query the MaxNumExperts that the policy tier dispatch would select for the given data.
inline int32_t queryDispatchedMaxExperts(Data const& data) {
  int32_t result = getMaxNumExperts(data.mNumExperts);
  dispatchRoutingPolicy(data, [&](auto preProc, auto postProc, char const* /*policyName*/) {
    using Pairs = typename PolicyTraits<decltype(preProc), decltype(postProc)>::Pairs;
    dispatchTierPairs(static_cast<Pairs*>(nullptr), data,
                      [&](auto eTag, auto /*kTag*/) { result = decltype(eTag)::value; });
  });
  return result;
}

// Whether the dispatched (pre, post) policy pair implements the block-per-token kernel interface
// (both policies opt in; see PolicyPairSupportsBlockPerToken).
inline bool queryPolicySupportsBlockPerToken(Data const& data) {
  bool supports = false;
  dispatchRoutingPolicy(data, [&](auto preProc_, auto postProc_, char const* /*policyName*/) {
    using PreProc_ = decltype(preProc_);
    using PostProc_ = decltype(postProc_);
    supports = PolicyPairSupportsBlockPerToken<PreProc_, PostProc_>::value;
  });
  return supports;
}

// Top-level dispatch: maps runtime preprocess/postprocess enums to compile-time policy types,
// then delegates to LAUNCH_ROUTING_FOR_POLICY which reads PolicyTraits for tier support.
#define LAUNCH_ROUTING_CUSTOM(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream) \
  dispatchRoutingPolicy(data, [&](auto preProc_, auto postProc_, char const* policyName_) {      \
    using PreProc_ = decltype(preProc_);                                                         \
    using PostProc_ = decltype(postProc_);                                                       \
    using Pairs_ = typename PolicyTraits<PreProc_, PostProc_>::Pairs;                            \
    bool dispatched_ =                                                                           \
        dispatchTierPairs(static_cast<Pairs_*>(nullptr), data, [&](auto eTag_, auto kTag_) {     \
          constexpr int tierMaxExp_ = decltype(eTag_)::value;                                    \
          constexpr int tierThreads_ = tierMaxExp_ <= 1024 ? tierMaxExp_ : 1024;                 \
          int const effectiveThreads_ = std::max(static_cast<int>(numThreads), tierThreads_);    \
          LAUNCH_ROUTING_WITH_POLICIES(data, coopLaunch, kernel, numBlocks, effectiveThreads_,   \
                                       smemSize, stream, PreProc_, PostProc_,                    \
                                       decltype(eTag_)::value, decltype(kTag_)::value);          \
        });                                                                                      \
    if (!dispatched_) {                                                                          \
      FLASHINFER_WARN(                                                                           \
          "No compiled tier covers numExperts=%d topK=%d for policy %s. "                        \
          "Add a Tier<%d, %d> to the corresponding PolicyTraits in RoutingCustomPolicy.cuh.",    \
          data.mNumExperts, data.mTopK, policyName_, getMaxNumExperts(data.mNumExperts),         \
          data.mTopK);                                                                           \
    }                                                                                            \
  })

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace routingCustom
}  // namespace moe::dev::routing
