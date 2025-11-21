#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cmath>

#include "flashinfer/trtllm/fused_moe/noAuxTcKernels.h"
#include "moeTopKFuncs.cuh"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/envUtils.h"
#include "tvm_ffi_utils.h"

namespace cg = cooperative_groups;
using namespace tensorrt_llm::common;

namespace tensorrt_llm::kernels {
static constexpr int WARP_SIZE = 32;
static constexpr int NumKimiK2Experts = 384;
static constexpr int NumDeepseekExperts = 256;
static constexpr int MaxNumExpertsUnit = 128;
static constexpr int NumTopGroupScores = 2;
static constexpr int MaxNumTopExperts = 8;
static constexpr int MaxNumTopGroups = 4;

static __device__ inline float sigmoid_accurate(float x) { return 0.5f * tanhf(0.5f * x) + 0.5f; }

template <typename InputT, typename BiasT, typename OutputT, typename IdxT, int MaxNumExperts,
          bool UseGroups>
__global__ void deepseek_v3_topk_kernel(InputT* scores, OutputT* topkValues, IdxT* topkIndices,
                                        BiasT* routingBias, int64_t const numTokens,
                                        int64_t const numGroup, int64_t const topkGroup,
                                        int64_t const topk, int64_t const numExperts,
                                        int64_t const numExpertsPerGroup,
                                        double const routedScalingFactor) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  // declare shared memory structure
  // number of experts is bounded by number of threads
  __shared__ float __attribute((aligned(128))) smemScoreSigmoid[MaxNumExperts];
  __shared__ float __attribute((aligned(128))) smemScoreBias[MaxNumExperts];
  // number of expert groups is bounded by number of warps
  int constexpr NumWarps = MaxNumExperts / WARP_SIZE;
  __shared__ float __attribute((aligned(128))) smemGroupScores[NumWarps];

  // needed for warp reduce
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  // for the final reduction of weight norm, only some lanes need to participate
  int32_t laneIdx = threadIdx.x % WARP_SIZE;
  int32_t warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WARP_SIZE, 0);

  if constexpr (UseGroups) {
    if (warpIdx >= numGroup) {
      return;
    }
  }

  // note that for invalid scores, we simply use a negative value:
  // they work well even with the compacted format used in topK, and
  // sigmoid / bias activated scores cannot be negative
  static constexpr float invalidScoreFloat = float{-INFINITY};
  const OutputT invalidScore = OutputT{invalidScoreFloat};

  // load bias already; each warp represents one expert group
  auto threadExpert = threadIdx.x;
  bool expertSelected = threadExpert < numExperts;
  if constexpr (UseGroups) {
    threadExpert = warpIdx * numExpertsPerGroup + laneIdx;
    expertSelected = laneIdx < numExpertsPerGroup;
  }

  auto scoreIdx = int64_t{blockIdx.x} * int64_t{numExperts} + threadExpert;
  auto biasVal = expertSelected ? static_cast<float>(routingBias[threadExpert]) : invalidScoreFloat;
  topkValues += blockIdx.x * topk;
  topkIndices += blockIdx.x * topk;

  // get our assigned thread score; each warp represents one expert group
  float score = expertSelected ? static_cast<float>(scores[scoreIdx]) : invalidScoreFloat;
  auto scoreSigmoid = sigmoid_accurate(score);
  // write the sigmoid score to shared for later use
  if (expertSelected) {
    smemScoreSigmoid[threadExpert] = scoreSigmoid;
  }

  // get the score with bias
  // note that with invalid values, because sigmoid is < 1 and bias is -1,
  // we must get a negative value, which is smaller than any valid value
  auto scoreBias = float{scoreSigmoid + float{biasVal}};

  if (expertSelected) {
    smemScoreBias[threadExpert] = scoreBias;
  }

  // registers for top group score reduction
  float topExpGroupScores[NumTopGroupScores];
  [[maybe_unused]] int32_t topExpGroupIdx[NumTopGroupScores];
  float topGroups[MaxNumTopGroups];  // bound of numGroup
  int32_t topGroupIdx[MaxNumTopGroups];
  float expertScoreGroup[MaxNumTopGroups];
  int32_t expertIdxGroup[MaxNumTopGroups];
  float topScores[MaxNumTopExperts];  // bound of topk
  int32_t topExperts[MaxNumTopExperts];

  if constexpr (UseGroups) {
    reduce_topk::reduceTopK(warp, topExpGroupScores, topExpGroupIdx, scoreBias, threadExpert,
                            /* minValue */ invalidScoreFloat);

    // get the final group score and write it to shared
    if (laneIdx == 0) {
      auto groupScore = topExpGroupScores[0] + topExpGroupScores[1];
      smemGroupScores[warpIdx] = groupScore;
    }
  }

  // make group scores available to all warps
  __syncthreads();

  if constexpr (UseGroups) {
    if (warpIdx == 0) {
      // a single warp performs the selection of top groups, and goes on to select the final experts
      float groupScore = laneIdx < numGroup ? smemGroupScores[laneIdx] : invalidScoreFloat;

      reduce_topk::reduceTopK(warp, topGroups, topGroupIdx, groupScore, laneIdx,
                              /* minValue */ invalidScoreFloat);

      // final expert selection: get relevant indexes and scores from shared

#pragma unroll
      for (int ii = 0; ii < MaxNumTopGroups; ++ii) {  // bound of numGroup
        // auto groupIdx = topGroupIdx[ii];
        auto groupIdx = (ii < topkGroup) ? topGroupIdx[ii] : 0;
        expertIdxGroup[ii] = groupIdx * numExpertsPerGroup + laneIdx;

        expertScoreGroup[ii] = (ii < topkGroup) && expertSelected
                                   ? smemScoreBias[expertIdxGroup[ii]]
                                   : invalidScoreFloat;
      }

      tensorrt_llm::kernels::reduce_topk::reduceTopK(warp, topScores, topExperts, expertScoreGroup,
                                                     expertIdxGroup,
                                                     /* minValue */ invalidScoreFloat, topk);
    }
  } else if constexpr (MaxNumExperts > MaxNumExpertsUnit) {
    // without groups, and the expert number is larger than MaxNumExpertsUnit,
    // we need to use multiple warps to calculate the intermediate topk results

    int constexpr NumExpertWarps = (MaxNumExperts - 1) / MaxNumExpertsUnit + 1;
    int constexpr NumInterTopK = NumExpertWarps * MaxNumTopExperts;
    __shared__ float __attribute((aligned(128))) smemInterTopScores[NumInterTopK];
    __shared__ int32_t __attribute((aligned(128))) smemInterTopExperts[NumInterTopK];
    if (warpIdx < NumExpertWarps) {
      int offset = warpIdx * WARP_SIZE * MaxNumTopGroups;
#pragma unroll
      for (int ii = 0; ii < MaxNumTopGroups; ++ii) {
        auto expertIdx = ii * WARP_SIZE + laneIdx;
        expertIdxGroup[ii] = offset + expertIdx;
        expertScoreGroup[ii] =
            offset + expertIdx < numExperts ? smemScoreBias[offset + expertIdx] : invalidScoreFloat;
      }
      reduce_topk::reduceTopK(warp, topScores, topExperts, expertScoreGroup, expertIdxGroup,
                              /* minValue */ invalidScoreFloat, topk);

      if (laneIdx < topk) {
        smemInterTopScores[warpIdx * MaxNumTopExperts + laneIdx] = topScores[laneIdx];
        smemInterTopExperts[warpIdx * MaxNumTopExperts + laneIdx] = topExperts[laneIdx];
      }
    }
    __syncthreads();
    if (warpIdx == 0) {
      int constexpr NumInterTopKPerThread = (NumInterTopK * NumExpertWarps - 1) / WARP_SIZE + 1;
      float intermidiateScore[NumInterTopKPerThread];
      int32_t intermidiateExpert[NumInterTopKPerThread];
      for (int i = laneIdx; i < NumInterTopKPerThread * WARP_SIZE; i += WARP_SIZE) {
        int ii = i / WARP_SIZE;
        if (i < NumInterTopK) {
          intermidiateScore[ii] = smemInterTopScores[i];
          intermidiateExpert[ii] = smemInterTopExperts[i];
        } else {
          intermidiateScore[ii] = invalidScoreFloat;
          intermidiateExpert[ii] = MaxNumExperts - 1;
        }
      }
      reduce_topk::reduceTopK(warp, topScores, topExperts, intermidiateScore, intermidiateExpert,
                              /* minValue */ invalidScoreFloat, topk);
    }
  } else {
    // without groups, and the expert number is smaller than MaxNumExpertsUnit
    // each thread just takes `MaxNumTopGroups` experts
    if (warpIdx == 0) {
#pragma unroll
      for (int ii = 0; ii < MaxNumTopGroups; ++ii) {
        auto expertIdx = ii * WARP_SIZE + laneIdx;
        expertIdxGroup[ii] = expertIdx;
        expertScoreGroup[ii] =
            expertIdx < numExperts ? smemScoreBias[expertIdx] : invalidScoreFloat;
      }
      reduce_topk::reduceTopK(warp, topScores, topExperts, expertScoreGroup, expertIdxGroup,
                              /* minValue */ invalidScoreFloat, topk);
    }
  }

  if (warpIdx == 0) {
    // determine our lane's expert index and write to output
    int32_t expertIdx = laneIdx < topk ? topExperts[laneIdx] : MaxNumExperts - 1;
    // norm the value
    float scoreNorm = laneIdx < topk ? smemScoreSigmoid[expertIdx] : 0.F;
    auto redNorm = cg::reduce(warp, scoreNorm, cg::plus<float>{});
    auto finalScore = static_cast<OutputT>(scoreNorm * routedScalingFactor / (redNorm + 1e-20));
    // store the topk scores and experts to output
    if (laneIdx < topk) {
      topkValues[laneIdx] = static_cast<OutputT>(finalScore);
      topkIndices[laneIdx] = expertIdx;
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename InputT, typename BiasT, typename OutputT, typename IdxT>
void invokeNoAuxTc(InputT* scores, BiasT* bias, OutputT* topk_values, IdxT* topk_indices,
                   int64_t const num_tokens, int64_t const num_experts, int64_t const n_group,
                   int64_t const topk_group, int64_t const topk, double const routed_scaling_factor,
                   bool const launch_with_pdl, cudaStream_t const stream) {
  // Check if we can use the optimized deepseek_v3_topk_kernel
  bool const is_single_group = (n_group == 1) && (num_experts <= NumKimiK2Experts);

  int64_t const experts_per_group = num_experts / n_group;
  bool const is_multi_group = (n_group != 1) && (num_experts <= NumDeepseekExperts) &&
                              (experts_per_group <= WARP_SIZE) &&
                              (experts_per_group * topk_group <= MaxNumExpertsUnit);

  if (is_single_group || is_multi_group) {
    cudaLaunchConfig_t config;
    auto* kernel_instance =
        &deepseek_v3_topk_kernel<InputT, BiasT, OutputT, IdxT, NumDeepseekExperts, true>;
    int num_threads = NumDeepseekExperts;
    if (is_single_group) {
      if (num_experts > MaxNumExpertsUnit) {
        kernel_instance =
            &deepseek_v3_topk_kernel<InputT, BiasT, OutputT, IdxT, NumKimiK2Experts, false>;
        num_threads = NumKimiK2Experts;
      } else {
        kernel_instance =
            &deepseek_v3_topk_kernel<InputT, BiasT, OutputT, IdxT, MaxNumExpertsUnit, false>;
        num_threads = MaxNumExpertsUnit;
      }
    }

    config.gridDim = num_tokens;
    config.blockDim = num_threads;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = launch_with_pdl;
    config.numAttrs = 1;
    config.attrs = attrs;

    cudaLaunchKernelEx(&config, kernel_instance, scores, topk_values, topk_indices, bias,
                       num_tokens, n_group, topk_group, topk, num_experts, num_experts / n_group,
                       routed_scaling_factor);
    sync_check_cuda_error(stream);
  } else {
    // TODO: call the generic path (previous implementation) or signal unsupported config.
    TLLM_CHECK_WITH_INFO(false,
                         "invokeNoAuxTc: unsupported configuration (n_group=%ld, num_experts=%ld, "
                         "topk_group=%ld). Please use "
                         "original pytorch implementation.",
                         n_group, num_experts, topk_group);
  }
}

#define INSTANTIATE_NOAUX_TC(InputT, BiasT, OutputT, IdxT)                              \
  template void invokeNoAuxTc<InputT, BiasT, OutputT, IdxT>(                            \
      InputT * scores, BiasT * bias, OutputT * topk_values, IdxT * topk_indices,        \
      int64_t const num_tokens, int64_t const num_experts, int64_t const n_group,       \
      int64_t const topk_group, int64_t const topk, double const routed_scaling_factor, \
      bool const launch_with_pdl, cudaStream_t const stream);

INSTANTIATE_NOAUX_TC(float, float, float, int32_t);
INSTANTIATE_NOAUX_TC(float, half, float, int32_t);

INSTANTIATE_NOAUX_TC(half, float, half, int32_t);
INSTANTIATE_NOAUX_TC(half, half, half, int32_t);

#ifdef ENABLE_BF16
INSTANTIATE_NOAUX_TC(float, __nv_bfloat16, float, int32_t);
INSTANTIATE_NOAUX_TC(half, __nv_bfloat16, half, int32_t);

INSTANTIATE_NOAUX_TC(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int32_t);
INSTANTIATE_NOAUX_TC(__nv_bfloat16, float, __nv_bfloat16, int32_t);
INSTANTIATE_NOAUX_TC(__nv_bfloat16, half, __nv_bfloat16, int32_t);
#endif

}  // namespace tensorrt_llm::kernels

namespace flashinfer::trtllm_dsv3_fused_routing {
// th::Tensor const& scores, th::Tensor const& bias, int64_t n_group,
// int64_t topk_group, int64_t topk, double routed_scaling_factor
// th::Tensor topk_values, th::Tensor topk_indices

void NoAuxTc(TensorView scores, TensorView bias, int64_t n_group, int64_t topk_group, int64_t topk,
             double routed_scaling_factor, TensorView topk_values, TensorView topk_indices,
             bool launch_with_pdl) {
  auto data_type = scores.dtype();
  auto bias_type = bias.dtype();

  auto input_size = scores.sizes();
  int64_t num_tokens = input_size[0];
  int64_t num_experts = input_size[1];

  TVM_FFI_ICHECK(input_size.size() == 2) << "scores must be a 2D Tensor";
  TVM_FFI_ICHECK((scores.device().device_type == kDLCUDA) && (bias.device().device_type == kDLCUDA))
      << "scores and bias must be CUDA tensors";
  TVM_FFI_ICHECK(scores.device().device_id == bias.device().device_id)
      << "scores and bias must be on the same device";
  TVM_FFI_ICHECK(bias.dim() == 1 && bias.numel() == num_experts)
      << "bias must be 1D with length == number of experts (%ld)";
  TVM_FFI_ICHECK(num_experts % n_group == 0) << "num_experts should be divisible by n_group";
  TVM_FFI_ICHECK(n_group <= 32)
      << "n_group should be smaller than or equal to 32 for now";  //@todo: remove this restriction
                                                                   // later
  TVM_FFI_ICHECK(topk <= 32)
      << "topk should be smaller than or equal to 32 for now";  //@todo: remove this restriction
                                                                // later
  TVM_FFI_ICHECK(topk_values.dim() == 2) << "topk_values must be a 2D Tensor";
  TVM_FFI_ICHECK(topk_indices.dim() == 2) << "topk_indices must be a 2D Tensor";
  TVM_FFI_ICHECK(topk_values.sizes()[0] == num_tokens)
      << "topk_values must have the same number of tokens as scores";
  TVM_FFI_ICHECK(topk_indices.sizes()[0] == num_tokens)
      << "topk_indices must have the same number of tokens as scores";
  TVM_FFI_ICHECK(topk_values.sizes()[1] == topk)
      << "topk_values must have the same number of topk as scores";
  TVM_FFI_ICHECK(topk_indices.sizes()[1] == topk)
      << "topk_indices must have the same number of topk as scores";
  TVM_FFI_ICHECK(topk_values.dtype() == data_type)
      << "topk_values must have the same dtype as scores";
  TVM_FFI_ICHECK(encode_dlpack_dtype(topk_indices.dtype()) == int32_code)
      << "topk_indices must have the same dtype as scores";

  auto stream = get_stream(scores.device());
  using namespace tensorrt_llm::kernels;
  switch (encode_dlpack_dtype(data_type)) {
    case float16_code:
      // Handle Float16
      switch (encode_dlpack_dtype(bias_type)) {
        case float16_code:
          invokeNoAuxTc<half, half, half, int32_t>(
              reinterpret_cast<half*>(scores.data_ptr()), reinterpret_cast<half*>(bias.data_ptr()),
              reinterpret_cast<half*>(topk_values.data_ptr()),
              reinterpret_cast<int32_t*>(topk_indices.data_ptr()), num_tokens, num_experts, n_group,
              topk_group, topk, routed_scaling_factor, launch_with_pdl, stream);
          break;
        case float32_code:
          invokeNoAuxTc<half, float, half, int32_t>(
              reinterpret_cast<half*>(scores.data_ptr()), reinterpret_cast<float*>(bias.data_ptr()),
              reinterpret_cast<half*>(topk_values.data_ptr()),
              reinterpret_cast<int32_t*>(topk_indices.data_ptr()), num_tokens, num_experts, n_group,
              topk_group, topk, routed_scaling_factor, launch_with_pdl, stream);
          break;
        case bfloat16_code:
          invokeNoAuxTc<half, __nv_bfloat16, half, int32_t>(
              reinterpret_cast<half*>(scores.data_ptr()),
              reinterpret_cast<__nv_bfloat16*>(bias.data_ptr()),
              reinterpret_cast<half*>(topk_values.data_ptr()),
              reinterpret_cast<int32_t*>(topk_indices.data_ptr()), num_tokens, num_experts, n_group,
              topk_group, topk, routed_scaling_factor, launch_with_pdl, stream);
          break;
        default:
          throw std::invalid_argument(
              "Invalid bias dtype, only supports float16, float32, and bfloat16");
          break;
      }
      break;
    case float32_code:
      switch (encode_dlpack_dtype(bias_type)) {
        case float32_code:
          invokeNoAuxTc<float, float, float, int32_t>(
              reinterpret_cast<float*>(scores.data_ptr()),
              reinterpret_cast<float*>(bias.data_ptr()),
              reinterpret_cast<float*>(topk_values.data_ptr()),
              reinterpret_cast<int32_t*>(topk_indices.data_ptr()), num_tokens, num_experts, n_group,
              topk_group, topk, routed_scaling_factor, launch_with_pdl, stream);
          break;
        case float16_code:
          invokeNoAuxTc<float, half, float, int32_t>(
              reinterpret_cast<float*>(scores.data_ptr()), reinterpret_cast<half*>(bias.data_ptr()),
              reinterpret_cast<float*>(topk_values.data_ptr()),
              reinterpret_cast<int32_t*>(topk_indices.data_ptr()), num_tokens, num_experts, n_group,
              topk_group, topk, routed_scaling_factor, launch_with_pdl, stream);
          break;
        case bfloat16_code:
          invokeNoAuxTc<float, __nv_bfloat16, float, int32_t>(
              reinterpret_cast<float*>(scores.data_ptr()),
              reinterpret_cast<__nv_bfloat16*>(bias.data_ptr()),
              reinterpret_cast<float*>(topk_values.data_ptr()),
              reinterpret_cast<int32_t*>(topk_indices.data_ptr()), num_tokens, num_experts, n_group,
              topk_group, topk, routed_scaling_factor, launch_with_pdl, stream);
          break;
        default:
          throw std::invalid_argument(
              "Invalid bias dtype, only supports float16, float32, and bfloat16");
          break;
      }
      break;
    case bfloat16_code:
      // Handle BFloat16
      switch (encode_dlpack_dtype(bias_type)) {
        case bfloat16_code:
          invokeNoAuxTc<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int32_t>(
              reinterpret_cast<__nv_bfloat16*>(scores.data_ptr()),
              reinterpret_cast<__nv_bfloat16*>(bias.data_ptr()),
              reinterpret_cast<__nv_bfloat16*>(topk_values.data_ptr()),
              reinterpret_cast<int32_t*>(topk_indices.data_ptr()), num_tokens, num_experts, n_group,
              topk_group, topk, routed_scaling_factor, launch_with_pdl, stream);
          break;
        case float16_code:
          invokeNoAuxTc<__nv_bfloat16, half, __nv_bfloat16, int32_t>(
              reinterpret_cast<__nv_bfloat16*>(scores.data_ptr()),
              reinterpret_cast<half*>(bias.data_ptr()),
              reinterpret_cast<__nv_bfloat16*>(topk_values.data_ptr()),
              reinterpret_cast<int32_t*>(topk_indices.data_ptr()), num_tokens, num_experts, n_group,
              topk_group, topk, routed_scaling_factor, launch_with_pdl, stream);
          break;
        case float32_code:
          invokeNoAuxTc<__nv_bfloat16, float, __nv_bfloat16, int32_t>(
              reinterpret_cast<__nv_bfloat16*>(scores.data_ptr()),
              reinterpret_cast<float*>(bias.data_ptr()),
              reinterpret_cast<__nv_bfloat16*>(topk_values.data_ptr()),
              reinterpret_cast<int32_t*>(topk_indices.data_ptr()), num_tokens, num_experts, n_group,
              topk_group, topk, routed_scaling_factor, launch_with_pdl, stream);
          break;
        default:
          throw std::invalid_argument(
              "Invalid bias dtype, only supports bfloat16, float16, and float32");
          break;
      }
      break;
    default:
      // Handle other data types
      throw std::invalid_argument("Invalid dtype, only supports float16, float32, and bfloat16");
      break;
  }
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(NoAuxTc, flashinfer::trtllm_dsv3_fused_routing::NoAuxTc);
}  // namespace flashinfer::trtllm_dsv3_fused_routing
