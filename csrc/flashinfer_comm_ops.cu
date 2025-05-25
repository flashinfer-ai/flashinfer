// flashinfer: adapted from sglang + vllm code
// refer to: https://github.com/vllm-project/vllm/blob/v0.8.2/csrc/common_extension.cc
/*
 * Copyright (c) 2023 by FlashInfer team.
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
#include "flashinfer/comm/allReduceFusionKernels.cuh"
#include "flashinfer/comm/moeAllReduceFusionKernels.cuh"
#include "pytorch_extension_utils.h"

// ============ start cpp/tensorrt_llm/thop/allreduceOp.cpp ============

#include "flashinfer/comm/trtllm/common/cudaUtils.h"
#include "flashinfer/comm/trtllm/common/customAllReduceUtils.h"
#include "flashinfer/comm/trtllm/common/dataType.h"
#include "pytorch_extension_utils.h"
// #include "flashinfer/comm/trtllm/common/opUtils.h"
#include "flashinfer/comm/allReduceFusionKernels.cuh"
#include "flashinfer/comm/moeAllReduceFusionKernels.cuh"
// #include "flashinfer/comm/customAllReduceKernels.cuh"
// #include "flashinfer/comm/trtllm/kernels/internal_cutlass_kernels/include/fp4_gemm.h"
#include "flashinfer/comm/trtllm/kernels/quantization.h"
// #include "flashinfer/comm/trtllm/kernels/userbuffers/ub_interface.h"
// #include "flashinfer/comm/trtllm/runtime/torchUtils.h"
// #include "flashinfer/comm/trtllm/runtime/utils/mpiUtils.h"
// #include "flashinfer/comm/trtllm/thop/fp4Quantize.h"
// #include "flashinfer/comm/trtllm/thop/fp8Op.h"
// #include "flashinfer/comm/trtllm/thop/thUtils.h"
// #include "flashinfer/comm/trtllm/thop/userbuffersTensor.h"

#define ENABLE_MULTI_DEVICE 1

#if ENABLE_MULTI_DEVICE
#endif  // ENABLE_MULTI_DEVICE
#include <nvml.h>

#include <cstddef>
#include <cstdint>
#include <unordered_set>

// using namespace nvinfer1;
using tensorrt_llm::kernels::AllReduceFusionOp;
using tensorrt_llm::kernels::AllReduceStrategyType;
using tensorrt_llm::mpi::MpiTag;

namespace torch_ext {

#if ENABLE_MULTI_DEVICE

namespace {

class NvmlManager {
 public:
  NvmlManager() { NVML_CHECK(nvmlInit()); }

  ~NvmlManager() { NVML_CHECK(nvmlShutdown()); }
};

std::set<int> getLocalGroup(std::set<int> const& group) {
  auto const myRank = COMM_SESSION.getRank();
  auto const myLocalRank = LOCAL_COMM_SESSION.getRank();
  auto const localSize = static_cast<uint32_t>(LOCAL_COMM_SESSION.getSize());

  std::vector<int32_t> ranks(localSize, 0);
  std::vector<int32_t> localRanks(localSize, 0);
  if (group.size() >= localSize) {
    LOCAL_COMM_SESSION.allgather(&myRank, ranks.data(), 1, tensorrt_llm::mpi::MpiType::kINT32);
    LOCAL_COMM_SESSION.allgather(&myLocalRank, localRanks.data(), 1,
                                 tensorrt_llm::mpi::MpiType::kINT32);
  } else {
    if (myRank == *group.begin()) {
      ranks.clear();
      int rank;
      ranks.push_back(myRank);
      for (auto it = std::next(std::begin(group), 1); it != group.end(); ++it) {
        LOCAL_COMM_SESSION.recvValue(rank, *it, MpiTag::kDefault);
        ranks.push_back(rank);
      }
      for (auto it = std::next(std::begin(group), 1); it != group.end(); ++it) {
        LOCAL_COMM_SESSION.send(ranks.data(), localSize, tensorrt_llm::mpi::MpiType::kINT32, *it,
                                MpiTag::kDefault);
      }

      localRanks.clear();
      localRanks.push_back(myLocalRank);
      for (auto it = std::next(std::begin(group), 1); it != group.end(); ++it) {
        LOCAL_COMM_SESSION.recvValue(rank, *it, MpiTag::kDefault);
        localRanks.push_back(rank);
      }
      for (auto it = std::next(std::begin(group), 1); it != group.end(); ++it) {
        LOCAL_COMM_SESSION.send(localRanks.data(), localSize, tensorrt_llm::mpi::MpiType::kINT32,
                                *it, MpiTag::kDefault);
      }
    } else {
      LOCAL_COMM_SESSION.sendValue(myRank, *group.begin(), MpiTag::kDefault);
      LOCAL_COMM_SESSION.recv(ranks.data(), localSize, tensorrt_llm::mpi::MpiType::kINT32,
                              *group.begin(), MpiTag::kDefault);

      LOCAL_COMM_SESSION.sendValue(myLocalRank, *group.begin(), MpiTag::kDefault);
      LOCAL_COMM_SESSION.recv(localRanks.data(), localSize, tensorrt_llm::mpi::MpiType::kINT32,
                              *group.begin(), MpiTag::kDefault);
    }
  }

  std::set<int> localGroup;
  for (size_t i = 0; i < ranks.size(); ++i) {
    auto rank = ranks[i];
    if (group.find(rank) != group.end()) {
      localGroup.insert(localRanks[i]);
    }
  }
  return localGroup;
}

class AllreduceOp {
 public:
  AllreduceOp(std::set<int> group, nvinfer1::DataType type, AllReduceStrategyType strategy,
              AllReduceFusionOp op, float eps)
      : mGroup(std::move(group)), mType(type), mStrategy(strategy), mOp(op), mEps(eps) {}

  ~AllreduceOp() = default;

  std::vector<at::Tensor> run(at::Tensor const& input, at::optional<at::Tensor> const& residual,
                              at::optional<at::Tensor> const& norm_weight,
                              at::optional<at::Tensor> const& scale,
                              at::optional<at::Tensor> const& bias,
                              at::optional<at::Tensor> workspace) noexcept {
    size_t size = input.numel();
    size_t seq_len = input.size(0);

    // If strategy is set to UB, UB must be used as UB impl output is special and cannot be used
    // by others.

    // Log runtime strategy
    auto const rank = COMM_SESSION.getRank();

    // Dispatch to different allreduce implementations
    return runFusionAllReduce(input, residual, norm_weight, scale, bias, workspace, mStrategy);
  }

  int initialize() noexcept {
    // TLLM_LOG_TRACE("%s start for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
    mNcclComm = getComm(mGroup);
    if (mStrategy != AllReduceStrategyType::NCCL && mStrategy != AllReduceStrategyType::UB) {
      initGroupTopology();
    }

    // TLLM_LOG_TRACE("%s stop for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
    return 0;
  }

 private:
  std::vector<at::Tensor> runFusionAllReduce(at::Tensor const& input,
                                             at::optional<at::Tensor> const& residual,
                                             at::optional<at::Tensor> const& norm_weight,
                                             at::optional<at::Tensor> const& scale,
                                             at::optional<at::Tensor> const& bias,
                                             at::optional<at::Tensor> workspace,
                                             AllReduceStrategyType strategy) noexcept {
    // Should handle only Lamport implementation
    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    int size = input.numel();
    int hidden_size = input.size(-1);
    int seq_len = input.size(0);

    auto const tp_size = mGroup.size();
    auto const cur_rank = COMM_SESSION.getRank();
    int tp_rank = 0;

    for (auto const& currentRank : mGroup) {
      if (cur_rank == currentRank) break;
      ++tp_rank;
    }

    // Use cleaner output assigning
    at::Tensor reduce_out;
    at::Tensor residual_out;
    at::Tensor norm_out;
    at::Tensor quant_out;
    at::Tensor scale_out;

    tensorrt_llm::kernels::ar_fusion::AllReduceFusionParams allreduce_fusion_params;

    allreduce_fusion_params.residual_in = nullptr;
    allreduce_fusion_params.rms_gamma = nullptr;

    allreduce_fusion_params.allreduce_out = nullptr;
    allreduce_fusion_params.quant_out = nullptr;
    allreduce_fusion_params.scale_out = nullptr;
    allreduce_fusion_params.residual_out = nullptr;
    allreduce_fusion_params.norm_out = nullptr;

    // Determine if using oneshot or twoshot allreduce kernel
    if (strategy == AllReduceStrategyType::MIN_LATENCY) {
      allreduce_fusion_params.use_oneshot =
          seq_len <= tensorrt_llm::kernels::ar_fusion::kOneShotMaxToken;
    } else {
      allreduce_fusion_params.use_oneshot = strategy == AllReduceStrategyType::ONESHOT;
    }

    // Check for some kernel constraints if using TWOSHOT kernel
    if (!allreduce_fusion_params.use_oneshot) {
      TORCH_CHECK(input.size(0) >= static_cast<int64_t>(tp_size),
                  "Sequence length must be greater than or equal to TP size");
    }

    // Handle no fusion allreduce here
    if (mOp == AllReduceFusionOp::NONE) {
      reduce_out = torch::empty_like(input);
      allreduce_fusion_params.allreduce_out = reduce_out.mutable_data_ptr();
      allreduce_fusion_params.pattern =
          tensorrt_llm::kernels::ar_fusion::AllReduceFusionPattern::kAllReduce;
    }
    // Handle allreduce fusion here
    // Prepare required output tensors for each fusion pattern
    else if (mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM) {
      norm_out = torch::empty_like(input);
      residual_out = torch::empty_like(residual.value());

      allreduce_fusion_params.norm_out = norm_out.mutable_data_ptr();
      allreduce_fusion_params.residual_out = residual_out.mutable_data_ptr();
      allreduce_fusion_params.pattern =
          tensorrt_llm::kernels::ar_fusion::AllReduceFusionPattern::kARResidualRMSNorm;
    } else if (mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_FP8 ||
               mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_FP8) {
      quant_out = at::detail::empty_cuda(input.sizes(), torch::kFloat8_e4m3fn, input.device(),
                                         std::nullopt);
      residual_out = torch::empty_like(residual.value());

      allreduce_fusion_params.quant_out = quant_out.mutable_data_ptr();
      allreduce_fusion_params.residual_out = residual_out.mutable_data_ptr();
      allreduce_fusion_params.pattern =
          tensorrt_llm::kernels::ar_fusion::AllReduceFusionPattern::kARResidualRMSNormFP8Quant;

      // norm out is required
      if (mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_FP8) {
        norm_out = torch::empty_like(input);
        allreduce_fusion_params.norm_out = norm_out.mutable_data_ptr();
        allreduce_fusion_params.pattern =
            tensorrt_llm::kernels::ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP8Quant;
      }
    } else if (mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_NVFP4 ||
               mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4) {
      // TODO: Better check for each pattern
      int64_t sf_vec_size = 16;
      int64_t m = 1;
      auto const& input_shape = input.sizes();
      auto const& r = input_shape.size();
      TORCH_CHECK(r >= 2, "Input should be >=2D tensor.");
      for (size_t i = 0; i < r - 1; i++) {
        m *= input_shape[i];
      }
      auto const k = input_shape[r - 1];
      TORCH_CHECK(k % sf_vec_size == 0, "Input should be divisible by sfVecSize.");
      std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
      output_shape[r - 1] = k / 2;

      quant_out = at::detail::empty_cuda(output_shape, FLOAT4_E2M1X2, input.device(), std::nullopt);
      scale_out =
          at::detail::empty_cuda({tensorrt_llm::computeFP4SwizzledLayoutSFSize(m, k / sf_vec_size)},
                                 SF_DTYPE, input.device(), std::nullopt);
      residual_out = torch::empty_like(residual.value());

      allreduce_fusion_params.quant_out = quant_out.mutable_data_ptr();
      allreduce_fusion_params.scale_out = scale_out.mutable_data_ptr();
      allreduce_fusion_params.residual_out = residual_out.mutable_data_ptr();
      allreduce_fusion_params.pattern =
          tensorrt_llm::kernels::ar_fusion::AllReduceFusionPattern::kARResidualRMSNormFP4Quant;

      // norm out is required
      if (mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4) {
        norm_out = torch::empty_like(input);
        allreduce_fusion_params.norm_out = norm_out.mutable_data_ptr();
        allreduce_fusion_params.pattern =
            tensorrt_llm::kernels::ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP4Quant;
      }
    } else {
      TORCH_CHECK(false, "Unsupported fusion operation: " + tensorrt_llm::kernels::toString(mOp));
      return {};
    }

    allreduce_fusion_params.nranks = tp_size;
    allreduce_fusion_params.rank = tp_rank;
    allreduce_fusion_params.dtype = mType;
    allreduce_fusion_params.size = size;
    allreduce_fusion_params.hidden_dim = hidden_size;
    allreduce_fusion_params.workspace =
        reinterpret_cast<void**>(workspace.value().mutable_data_ptr());
    allreduce_fusion_params.allreduce_in = input.data_ptr();

    if (mOp != AllReduceFusionOp::NONE) {
      allreduce_fusion_params.residual_in = residual.value().data_ptr();
      allreduce_fusion_params.rms_gamma = norm_weight.value().data_ptr();
      allreduce_fusion_params.rms_eps = mEps;
    }

    allreduce_fusion_params.stream = stream;

    bool const is_scale_factor_required =
        mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_FP8 ||
        mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_FP8 ||
        mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_NVFP4 ||
        mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4;

    allreduce_fusion_params.scale_factor =
        is_scale_factor_required ? static_cast<float*>(scale.value().data_ptr()) : nullptr;

    tensorrt_llm::kernels::ar_fusion::allreduce_fusion_op(allreduce_fusion_params);

    // Pack output tensors
    switch (mOp) {
      case AllReduceFusionOp::NONE:
        return {reduce_out};
      case AllReduceFusionOp::RESIDUAL_RMS_NORM:
        return {norm_out, residual_out};
      case AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_FP8:
        return {quant_out, residual_out};
      case AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_FP8:
        return {norm_out, quant_out, residual_out};
      case AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_NVFP4:
        return {quant_out, scale_out, residual_out};
      case AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4:
        return {norm_out, quant_out, scale_out, residual_out};
      default:
        TORCH_CHECK(false, "Unsupported fusion operation: " + tensorrt_llm::kernels::toString(mOp));
    }
    return {};
  }

  std::vector<at::Tensor> fallbackRunSubsequentOps(at::Tensor const& input,
                                                   torch::optional<at::Tensor> const& residual,
                                                   torch::optional<at::Tensor> const& norm_weight,
                                                   torch::optional<at::Tensor> const& scale,
                                                   torch::optional<at::Tensor> const& bias,
                                                   at::Tensor& reduce_output) noexcept {
    // If we reach here, it means the extra fallback operations are required.
    // All patterns are broken into ALlReduce + residual_rms_norm + following operations
    // (quantization, etc.)
    auto const size = input.numel();
    auto const hidden_size = input.size(-1);
    auto const stream = at::cuda::getCurrentCUDAStream(input.get_device());

    at::Tensor norm_out = torch::empty_like(input);

    tensorrt_llm::kernels::AllReduceParams params;
    params.fusion_params.bias_buffer = bias ? bias.value().data_ptr() : nullptr;
    params.fusion_params.residual_buffer = residual ? residual.value().data_ptr() : nullptr;
    params.fusion_params.weight_buffer = norm_weight ? norm_weight.value().data_ptr() : nullptr;
    params.local_output_buffer_ptr = norm_out.mutable_data_ptr();
    params.elts_total = size;

    params.fusion_params.hidden_size = hidden_size;
    params.fusion_params.eps = mEps;
    params.fusion_params.intermediate_buffer = reduce_output.mutable_data_ptr();
    tensorrt_llm::kernels::residualRmsNorm(params, mType, stream,
                                           AllReduceFusionOp::RESIDUAL_RMS_NORM);

    // If no quantization is needed, return the norm and residual outputs.
    if (mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM) {
      return {norm_out, reduce_output};
    }

    const int64_t sf_vecsize = 16;
    bool const sf_use_ue8m0 = false;
    bool const is_sf_swizzled_layout = true;
    TORCH_CHECK(scale, "scale is required for quantization ops");

    // Attach the subsequent operations after the residual RMS norm all-reduce and return the final
    // outputs.
    switch (mOp) {
      case AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_FP8: {
        auto [quant_out, scale_out] =
            torch_ext::symmetric_static_quantize_per_tensor(norm_out, scale.value());
        return {quant_out, reduce_output};
      }
      case AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_NVFP4: {
        auto [quant_out, scale_out] = torch_ext::fp4_quantize(norm_out, scale.value(), sf_vecsize,
                                                              sf_use_ue8m0, is_sf_swizzled_layout);
        return {quant_out, scale_out, reduce_output};
      }
      case AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_FP8: {
        auto [quant_out, scale_out] =
            torch_ext::symmetric_static_quantize_per_tensor(norm_out, scale.value());
        return {norm_out, quant_out, reduce_output};
      }
      case AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4: {
        auto [quant_out, scale_out] = torch_ext::fp4_quantize(norm_out, scale.value(), sf_vecsize,
                                                              sf_use_ue8m0, is_sf_swizzled_layout);
        return {norm_out, quant_out, scale_out, reduce_output};
      }
      default:
        break;
    }

    TORCH_CHECK(false, "Unsupported fusion operation: " + tensorrt_llm::kernels::toString(mOp));
    return {};
  }

  bool Fusable() noexcept { return mOp != AllReduceFusionOp::NONE; }

  void initGroupTopology() noexcept {
    static std::map<std::set<int>, std::tuple<bool, bool>> cache;
    if (cache.find(mGroup) != cache.end()) {
      auto [is_NVLINK_supported, is_P2P_supported] = cache[mGroup];
      mIsNVLINKSupported = is_NVLINK_supported;
      mIsP2PSupported = is_P2P_supported;
      return;
    }
    setGroupTopology();
    cache[mGroup] = {mIsNVLINKSupported, mIsP2PSupported};
  }

  void setGroupTopology() noexcept {
    auto const rank = COMM_SESSION.getRank();
    // TLLM_LOG_INFO("Detecting local TP group for rank %d", rank);
    std::set<int> local_group = getLocalGroup(mGroup);
    if (mGroup.size() != local_group.size()) {
      mIsP2PSupported = false;
      mIsNVLINKSupported = false;
      // TLLM_LOG_INFO("Found inter-node TP group for rank %d", rank);
      return;
    }
    // TLLM_LOG_INFO("TP group is intra-node for rank %d", rank);

    NvmlManager nvml_manager;
    std::unordered_set<int> visited_device;
    mIsP2PSupported = true;
    mIsNVLINKSupported = true;

    // Use cudaDeviceCanAccessPeer to determine whether p2p is supported,
    // and use nvml to determine whether there are nvlink links between ranks.
    for (int first_device_id : local_group) {
      for (int second_device_id : local_group) {
        if (first_device_id == second_device_id ||
            visited_device.find(second_device_id) != visited_device.end()) {
          continue;
        }

        int can_access_peer = 0;
        TLLM_CUDA_CHECK(
            cudaDeviceCanAccessPeer(&can_access_peer, first_device_id, second_device_id));

        if (!can_access_peer) {
          mIsP2PSupported = false;
          mIsNVLINKSupported = false;

          return;
        }

        nvmlDevice_t first_device;
        NVML_CHECK(nvmlDeviceGetHandleByIndex(first_device_id, &first_device));

        bool is_NVLINK = false;

        for (unsigned int link = 0; link < NVML_NVLINK_MAX_LINKS; link++) {
          nvmlPciInfo_t remote_pci_info;
          if (nvmlDeviceGetNvLinkRemotePciInfo_v2(first_device, link, &remote_pci_info) !=
              NVML_SUCCESS) {
            continue;
          }

          nvmlDevice_t remote_device;
          auto const result =
              nvmlDeviceGetHandleByPciBusId_v2(remote_pci_info.busId, &remote_device);

          if (result == NVML_SUCCESS) {
            // Two GPUs are connected directly through nvlink
            unsigned int remote_device_id;
            NVML_CHECK(nvmlDeviceGetIndex(remote_device, &remote_device_id));

            if (remote_device_id == static_cast<unsigned int>(second_device_id)) {
              is_NVLINK = true;
            }
          } else if (result == NVML_ERROR_NOT_FOUND) {
            // Maybe Two GPUs are connected via nvswitch,
            // now remotePciInfo represents the pci information of nvswitch,
            // determine whether nvlink is supported by whether two GPUs are connected to the same
            // nvswitch.
            nvmlDevice_t second_device;
            NVML_CHECK(nvmlDeviceGetHandleByIndex(second_device_id, &second_device));

            for (unsigned int second_link = 0; second_link < NVML_NVLINK_MAX_LINKS; second_link++) {
              nvmlPciInfo_t second_remote_pci_info;
              if (nvmlDeviceGetNvLinkRemotePciInfo_v2(second_device, second_link,
                                                      &second_remote_pci_info) != NVML_SUCCESS) {
                continue;
              }

              if (strcmp(remote_pci_info.busId, second_remote_pci_info.busId) == 0) {
                is_NVLINK = true;
                break;
              }
            }
          } else {
            NVML_CHECK(result);
          }

          if (is_NVLINK) {
            break;
          }
        }

        mIsNVLINKSupported &= is_NVLINK;
      }
      visited_device.insert(first_device_id);
    }
  }

  bool ifFallbackToNCCL(size_t seq_len, size_t message_size_bytes, size_t max_workspace_size,
                        bool is_auto) noexcept {
    // If messageSize is less than maxWorkspaceSize, use NCCL, regardless of the fusion type.
    if (message_size_bytes > max_workspace_size) {
      if (!is_auto) {
        TLLM_LOG_WARNING(
            "Since messageSize is greater than maxWorkspaceSize, fallback to AllReduceStrategy: "
            "NCCL");
      }
      return true;
    }

    // If Peer to Peer is not supported, fallback to NCCL.
    if (!mIsP2PSupported) {
      if (!is_auto) {
        TLLM_LOG_WARNING("Since Peer to Peer not supported, fallback to AllReduceStrategy: NCCL");
      }
      return true;
    }

    // If NVLINK is not supported, fallback to NCCL.
    if (!mIsNVLINKSupported) {
      if (!is_auto) {
        TLLM_LOG_WARNING("Since NVLINK not supported, fallback to AllReduceStrategy: NCCL");
      }
      return true;
    }
    return false;
  }

 private:
  std::set<int> mGroup;
  bool mIsNVLINKSupported;
  bool mIsP2PSupported;
  nvinfer1::DataType mType;
  AllReduceStrategyType mStrategy;
  AllReduceFusionOp mOp;
  float mEps;
  std::shared_ptr<ncclComm_t> mNcclComm;
};

}  // namespace

#endif  // ENABLE_MULTI_DEVICE

std::vector<at::Tensor> allreduce(at::Tensor const& input, at::optional<at::Tensor> const& residual,
                                  at::optional<at::Tensor> const& norm_weight,
                                  at::optional<at::Tensor> const& scale,
                                  at::optional<at::Tensor> const& bias,
                                  at::optional<at::Tensor> const& workspace,
                                  at::List<int64_t> const& group_, int64_t const strategy_,
                                  int64_t const fusion_op_, double const eps_) {
#if ENABLE_MULTI_DEVICE
  auto const dtype = tensorrt_llm::runtime::TorchUtils::dataType(input.scalar_type());
  auto const strategy = static_cast<AllReduceStrategyType>(int8_t(strategy_));
  auto const fusion_op = static_cast<AllReduceFusionOp>(int8_t(fusion_op_));
  float const eps = eps_;
  std::set<int> group;
  for (int64_t rank : group_) {
    group.insert(static_cast<int>(rank));
  }
  AllreduceOp op(group, dtype, strategy, fusion_op, eps);
  op.initialize();
  return op.run(input, residual, norm_weight, scale, bias, workspace);
#else
  return {input};
#endif  // ENABLE_MULTI_DEVICE
}

// residual [m, hidden_dim]
// norm_weight [hidden_dim]
// device_num_experts [1]
// scale_input [global_num_experts, m]
// active_experts_token_input [device_num_experts, m, hidden_dim]
// token_input [m, hidden_dim]
std::vector<at::Tensor> moe_allreduce(at::Tensor const& residual, at::Tensor const& norm_weight,
                                      at::Tensor const& device_num_experts,
                                      at::Tensor const& scale_input,
                                      at::Tensor const& active_experts_token_input,
                                      at::Tensor const& token_input, at::Tensor workspace,
                                      int64_t const rank, int64_t const nranks, double const eps) {
  auto allreduce_fusion_params =
      tensorrt_llm::kernels::ar_fusion::moe::MoeReductionAllReduceFusionParams();

  allreduce_fusion_params.quant_out = nullptr;
  allreduce_fusion_params.scale_out = nullptr;
  allreduce_fusion_params.residual_out = nullptr;
  allreduce_fusion_params.norm_out = nullptr;

  allreduce_fusion_params.nranks = static_cast<int>(nranks);
  allreduce_fusion_params.rank = static_cast<int>(rank);
  allreduce_fusion_params.dtype =
      tensorrt_llm::runtime::TorchUtils::dataType(token_input.scalar_type());
  // size: num_token * hidden_dim
  allreduce_fusion_params.size = static_cast<int>(token_input.numel());
  allreduce_fusion_params.hidden_dim = static_cast<int>(active_experts_token_input.size(-1));

  // workspace: AR scratch space
  allreduce_fusion_params.workspace = reinterpret_cast<void**>(workspace.mutable_data_ptr());

  allreduce_fusion_params.rms_gamma = norm_weight.data_ptr();
  allreduce_fusion_params.rms_eps = static_cast<float>(eps);
  allreduce_fusion_params.stream = at::cuda::getCurrentCUDAStream(norm_weight.get_device());

  allreduce_fusion_params.residual_in = residual.data_ptr();

  // MOE Reduction specific params
  allreduce_fusion_params.allreduce_in = nullptr;  // for safety, set nullptr
  allreduce_fusion_params.moe_reduction_device_num_experts =
      static_cast<int*>(device_num_experts.data_ptr());
  allreduce_fusion_params.moe_reduction_scale_input = static_cast<float*>(scale_input.data_ptr());
  allreduce_fusion_params.moe_reduction_active_experts_token_input =
      active_experts_token_input.data_ptr();
  allreduce_fusion_params.moe_reduction_token_input = token_input.data_ptr();

  // output tensors
  at::Tensor norm_out = torch::empty_like(token_input);
  at::Tensor residual_out = torch::empty_like(residual);

  allreduce_fusion_params.norm_out = norm_out.mutable_data_ptr();
  allreduce_fusion_params.residual_out = residual_out.mutable_data_ptr();

  tensorrt_llm::kernels::ar_fusion::moe::moereduction_allreduce_fusion_op(allreduce_fusion_params);

  return {norm_out, residual_out};
}

}  // namespace torch_ext

// ============ end of cpp/tensorrt_llm/thop/allreduceOp.cpp ============

using fptr_t = int64_t;

fptr_t init_custom_ar(const std::vector<fptr_t>& fake_ipc_ptrs, at::Tensor& rank_data, int64_t rank,
                      bool full_nvlink);
void dispose(fptr_t _fa);
int64_t meta_size();
void all_reduce(fptr_t _fa, at::Tensor& inp, at::Tensor& out, fptr_t _reg_buffer,
                int64_t reg_buffer_sz_bytes, int64_t num_ctas);
std::tuple<std::vector<int64_t>, std::vector<int64_t>> get_graph_buffer_ipc_meta(fptr_t _fa);
void register_buffer(fptr_t _fa, const std::vector<fptr_t>& fake_ipc_ptrs);
void register_graph_buffers(fptr_t _fa, const std::vector<std::vector<int64_t>>& handles,
                            const std::vector<std::vector<int64_t>>& offsets);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);
  m.def("register_graph_buffers", &register_graph_buffers);
  m.def("dispose", &dispose);
  m.def("meta_size", &meta_size);
  m.def("register_buffer", &register_buffer);
  m.def("init_custom_ar", &init_custom_ar);
  m.def("all_reduce", &all_reduce);  // vllm
  m.def("allreduce", &torch_ext::allreduce);
  m.def("moe_allreduce", &torch_ext::moe_allreduce);
}
