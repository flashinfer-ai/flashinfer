/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/ATen.h>

#include <vector>

#include "flashinfer/comm/trtllm_alltoall.cuh"
#include "flashinfer/comm/trtllm_alltoall_prepare.cuh"
#include "pytorch_extension_utils.h"

using namespace flashinfer::trtllm_alltoall;

void moeCommPrepareIndicesOp(at::Tensor gatheredTargetRankIds,
                             std::optional<at::Tensor> realRankTokenCountCumSum,
                             at::Tensor localGatherIndices, at::Tensor sendRankCountCumSum,
                             at::Tensor sendRankLocalIndices, at::Tensor recvRankCountCumSum,
                             at::Tensor recvRankLocalIndices,
                             at::Tensor backwardRecvRankLocalIndices, int64_t maxTokenCountPerRank,
                             int64_t expertCount, int64_t topK, int64_t epRank, int64_t epSize) {
  CHECK_INPUT_TYPE(gatheredTargetRankIds, at::ScalarType::Int);
  TORCH_CHECK(gatheredTargetRankIds.dim() == 2, "gatheredTargetRankIds must be a 2D tensor");
  TORCH_CHECK(gatheredTargetRankIds.size(1) == topK,
              "gatheredTargetRankIds must have topK columns");

  int const* realRankTokenCountCumSumPtr = nullptr;
  if (realRankTokenCountCumSum.has_value()) {
    TORCH_CHECK(realRankTokenCountCumSum.value().dim() == 1,
                "realRankTokenCountCumSum must be a 1D tensor");
    TORCH_CHECK(realRankTokenCountCumSum.value().dtype() == at::ScalarType::Int,
                "realRankTokenCountCumSum must be a int32 tensor");
    TORCH_CHECK(realRankTokenCountCumSum.value().size(0) == epSize,
                "realRankTokenCountCumSum must have epSize elements");
    realRankTokenCountCumSumPtr = realRankTokenCountCumSum.value().data_ptr<int>();
  } else {
    TORCH_CHECK(gatheredTargetRankIds.size(0) == epSize * maxTokenCountPerRank,
                "gatheredTargetRankIds should have shape (epSize * maxTokenCountPerRank, topK)");
  }
  TORCH_CHECK(maxTokenCountPerRank > 0, "maxTokenCountPerRank must be greater than 0");
  TORCH_CHECK(expertCount > 0, "expertCount must be greater than 0");
  TORCH_CHECK(topK > 0, "topK must be greater than 0");
  TORCH_CHECK(topK <= expertCount, "topK must be less than or equal to expertCount");
  TORCH_CHECK(epRank >= 0 && epRank < epSize, "epRank must be in the range [0, epSize)");

  auto stream = at::cuda::getCurrentCUDAStream();

  int maxSendRanksPerToken = std::max(static_cast<int>(epSize), static_cast<int>(topK));

  CHECK_INPUT_TYPE(localGatherIndices, at::ScalarType::Int);
  TORCH_CHECK(localGatherIndices.dim() == 1);
  TORCH_CHECK(localGatherIndices.size(0) == maxTokenCountPerRank * epSize);

  CHECK_INPUT_TYPE(sendRankCountCumSum, at::ScalarType::Int);
  TORCH_CHECK(sendRankCountCumSum.dim() == 1);
  TORCH_CHECK(sendRankCountCumSum.size(0) == epSize)

  CHECK_INPUT_TYPE(sendRankLocalIndices, at::ScalarType::Int);
  TORCH_CHECK(sendRankLocalIndices.dim() == 1);
  TORCH_CHECK(sendRankLocalIndices.size(0) == maxTokenCountPerRank * maxSendRanksPerToken);

  CHECK_INPUT_TYPE(recvRankCountCumSum, at::ScalarType::Int);
  TORCH_CHECK(recvRankCountCumSum.dim() == 1);
  TORCH_CHECK(recvRankCountCumSum.size(0) == epSize);

  CHECK_INPUT_TYPE(recvRankLocalIndices, at::ScalarType::Int);
  TORCH_CHECK(recvRankLocalIndices.dim() == 1);
  TORCH_CHECK(recvRankLocalIndices.size(0) == maxTokenCountPerRank * epSize);

  CHECK_INPUT_TYPE(backwardRecvRankLocalIndices, at::ScalarType::Int);
  TORCH_CHECK(backwardRecvRankLocalIndices.dim() == 1);
  TORCH_CHECK(backwardRecvRankLocalIndices.size(0) == maxTokenCountPerRank * maxSendRanksPerToken);

  flashinfer::trtllm_alltoall::MoeExpertParallelInfo expertParallelInfo;
  expertParallelInfo.expertCount = static_cast<int>(expertCount);
  expertParallelInfo.topK = static_cast<int>(topK);

  flashinfer::trtllm_alltoall::MoeEpWorldInfo worldInfo = {static_cast<int>(epSize),
                                                           static_cast<int>(epRank)};

  auto cudaResult = flashinfer::trtllm_alltoall::moeAllToAllPrepareIndices(
      worldInfo, expertParallelInfo, static_cast<int>(maxTokenCountPerRank),
      gatheredTargetRankIds.data_ptr<int>(), realRankTokenCountCumSumPtr,
      localGatherIndices.data_ptr<int>(), sendRankCountCumSum.data_ptr<int>(),
      sendRankLocalIndices.data_ptr<int>(), recvRankCountCumSum.data_ptr<int>(),
      recvRankLocalIndices.data_ptr<int>(), backwardRecvRankLocalIndices.data_ptr<int>(), stream);
  TORCH_CHECK(cudaResult == cudaSuccess,
              "CUDA error in moeAllToAllPrepareIndices: ", cudaGetErrorString(cudaResult));
}

void moeLocalGatherOp(at::Tensor recvRankCumSum, at::Tensor localGatherIndices,
                      at::Tensor gatheredExpertIds, at::Tensor gatheredScales,
                      at::Tensor localExpertIds, at::Tensor localScales,
                      int64_t maxTokenCountPerRank, int64_t expertCount, int64_t topK,
                      int64_t epRank, int64_t epSize) {
  CHECK_INPUT_TYPE(recvRankCumSum, at::ScalarType::Int);
  CHECK_INPUT_TYPE(localGatherIndices, at::ScalarType::Int);
  CHECK_INPUT_TYPE(gatheredExpertIds, at::ScalarType::Int);
  CHECK_INPUT_TYPE(gatheredScales, at::ScalarType::Float);
  CHECK_INPUT_TYPE(localExpertIds, at::ScalarType::Int);
  CHECK_INPUT_TYPE(localScales, at::ScalarType::Float);

  TORCH_CHECK(maxTokenCountPerRank > 0, "maxTokenCountPerRank must be greater than 0");
  TORCH_CHECK(expertCount > 0, "expertCount must be greater than 0");
  TORCH_CHECK(topK > 0, "topK must be greater than 0");
  TORCH_CHECK(topK <= expertCount, "topK must be less than or equal to expertCount");
  TORCH_CHECK(epRank >= 0 && epRank < epSize, "epRank must be in the range [0, epSize)");

  TORCH_CHECK(recvRankCumSum.dim() == 1, "recvRankCumSum must be a 1D tensor");
  TORCH_CHECK(recvRankCumSum.size(0) == epSize, "recvRankCumSum must have epSize elements");
  TORCH_CHECK(localGatherIndices.dim() == 1, "localGatherIndices must be a 1D tensor");
  TORCH_CHECK(gatheredExpertIds.dim() == 2, "gatheredExpertIds must be a 2D tensor");
  TORCH_CHECK(gatheredScales.dim() == 2, "gatheredScales must be a 2D tensor");
  TORCH_CHECK(localExpertIds.dim() == 2, "localExpertIds must be a 2D tensor");
  TORCH_CHECK(localScales.dim() == 2, "localScales must be a 2D tensor");
  TORCH_CHECK(gatheredExpertIds.size(1) == topK, "gatheredExpertIds must have topK columns");
  TORCH_CHECK(gatheredScales.size(1) == topK, "gatheredScales must have topK columns");
  TORCH_CHECK(localExpertIds.size(1) == topK, "localExpertIds must have topK columns");
  TORCH_CHECK(localScales.size(1) == topK, "localScales must have topK columns");

  int localMaxTokenCount = static_cast<int>(localGatherIndices.size(0));
  TORCH_CHECK(localExpertIds.size(0) == localMaxTokenCount,
              "localExpertIds must have localMaxTokenCount rows");
  TORCH_CHECK(localScales.size(0) == localMaxTokenCount,
              "localScales must have localMaxTokenCount rows");

  auto stream = at::cuda::getCurrentCUDAStream();

  flashinfer::trtllm_alltoall::MoeExpertParallelInfo expertParallelInfo;
  expertParallelInfo.expertCount = static_cast<int>(expertCount);
  expertParallelInfo.topK = static_cast<int>(topK);

  flashinfer::trtllm_alltoall::MoeEpWorldInfo worldInfo = {static_cast<int>(epSize),
                                                           static_cast<int>(epRank)};
  flashinfer::trtllm_alltoall::moeLocalGather(
      worldInfo, expertParallelInfo, static_cast<int>(maxTokenCountPerRank), localMaxTokenCount,
      recvRankCumSum.data_ptr<int>(), localGatherIndices.data_ptr<int>(),
      gatheredExpertIds.data_ptr<int>(), gatheredScales.data_ptr<float>(),
      localExpertIds.data_ptr<int>(), localScales.data_ptr<float>(), stream);
}

void moeCommOp(at::Tensor input, at::Tensor sendRankCumSum, at::Tensor sendIndices,
               at::Tensor output, at::Tensor recvRankCumSum, at::Tensor recvIndices,
               at::Tensor allWorkspaces, int64_t epRank, int64_t epSize) {
  CHECK_INPUT_TYPE(sendRankCumSum, at::ScalarType::Int);
  CHECK_INPUT_TYPE(sendIndices, at::ScalarType::Int);
  CHECK_INPUT_TYPE(recvRankCumSum, at::ScalarType::Int);
  CHECK_INPUT_TYPE(recvIndices, at::ScalarType::Int);
  // allWorkspaces is a uint64 tensor, but may not be contiguous
  TORCH_CHECK(allWorkspaces.dtype() == at::ScalarType::UInt64,
              "allWorkspaces must be a uint64 tensor");

  TORCH_CHECK(input.dim() == 2, "input must be a 2D tensor");
  TORCH_CHECK(output.dim() == 2, "output must be a 2D tensor");
  TORCH_CHECK(sendRankCumSum.dim() == 1, "sendRankCumSum must be a 1D tensor");
  TORCH_CHECK(sendIndices.dim() == 1, "sendIndices must be a 1D tensor");
  TORCH_CHECK(recvRankCumSum.dim() == 1, "recvRankCumSum must be a 1D tensor");
  TORCH_CHECK(recvIndices.dim() == 1, "recvIndices must be a 1D tensor");
  TORCH_CHECK(allWorkspaces.dim() == 2, "allWorkspaces must be a 2D tensor");

  TORCH_CHECK(input.size(1) == output.size(1),
              "input and output must have the same second dimension");
  TORCH_CHECK(sendRankCumSum.size(0) == epSize, "sendRankCumSum must have epSize elements");
  TORCH_CHECK(recvRankCumSum.size(0) == epSize, "recvRankCumSum must have epSize elements");
  TORCH_CHECK(allWorkspaces.size(0) == epSize, "allWorkspaces must have epSize elements");

  TORCH_CHECK(epRank >= 0 && epRank < epSize, "epRank must be in the range [0, epSize)");

  flashinfer::trtllm_alltoall::MoeEpWorldInfo worldInfo = {static_cast<int>(epSize),
                                                           static_cast<int>(epRank)};
  flashinfer::trtllm_alltoall::SendRecvDataInfo sendRecvDataInfo;

  size_t eltSize = input.dtype().itemsize();
  size_t eltCountPerU64 = sizeof(uint64_t) / eltSize;
  TORCH_CHECK(input.size(1) % (eltCountPerU64 * 2) == 0,
              "input.size(1) must be aligned to 16 bytes");
  sendRecvDataInfo.vectorSizeInU64 = input.size(1) / eltCountPerU64;
  sendRecvDataInfo.DoPreCompute();

  flashinfer::trtllm_alltoall::SendRecvDispls sendDispls, recvDispls;
  sendDispls.dataPtr = static_cast<uint64_t*>(input.data_ptr());
  sendDispls.rankCountCumSum = sendRankCumSum.data_ptr<int>();
  sendDispls.rankLocalIndices = sendIndices.data_ptr<int>();
  sendDispls.vectorStrideInU64 = input.stride(0) / eltCountPerU64;

  recvDispls.dataPtr = static_cast<uint64_t*>(output.data_ptr());
  recvDispls.rankCountCumSum = recvRankCumSum.data_ptr<int>();
  recvDispls.rankLocalIndices = recvIndices.data_ptr<int>();
  recvDispls.vectorStrideInU64 = output.stride(0) / eltCountPerU64;

  flashinfer::trtllm_alltoall::MoeCommWorkspace workspace;
  workspace.workspacePtr = allWorkspaces.data_ptr<uint64_t>();
  workspace.rankStrideInU64 = allWorkspaces.stride(0);

  auto stream = at::cuda::getCurrentCUDAStream();

  flashinfer::trtllm_alltoall::moeAllToAll(worldInfo, sendRecvDataInfo, sendDispls, recvDispls,
                                           workspace, stream);
}

int64_t getWorkspaceSizePerRank(int64_t epSize) {
  int epSize32 = static_cast<int>(epSize);
  return flashinfer::trtllm_alltoall::getMoeCommWorkspaceSize(epSize32);
}

void setMaxUsableSmCount(int64_t maxSmCount) {
  flashinfer::trtllm_alltoall::setMaxUsableSmCount(static_cast<int>(maxSmCount));
}

int64_t getPrepareWorkspaceSizePerRank(int64_t epSize) {
  int epSize32 = static_cast<int>(epSize);
  return flashinfer::trtllm_alltoall::moe_prepare::getMoePrepareWorkspaceSize(epSize32);
}

std::tuple<at::Tensor, c10::optional<at::Tensor>, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, c10::optional<at::Tensor>>
moePrepareOp(at::Tensor expertsIds, c10::optional<at::Tensor> scales,
             c10::optional<at::Tensor> expertsStatics, at::Tensor allWorkspaces,
             int64_t maxTokenCountPerRank, int64_t epRank, int64_t epSize, int64_t expertCount,
             int64_t slotCount, int64_t topK) {
  CHECK_INPUT_TYPE(expertsIds, at::ScalarType::Int);
  TORCH_CHECK(expertCount % 4 == 0, "expertCount must be divisible by 4");
  TORCH_CHECK(slotCount % 4 == 0, "slotCount must be divisible by 4");

  int64_t maxSendRanksPerToken = std::max(epSize, topK);
  int64_t tokenCount = expertsIds.size(0);

  at::Tensor preparedLocalExpertIds = at::empty({maxTokenCountPerRank * epSize, topK},
                                                expertsIds.options().dtype(at::ScalarType::Int));

  at::Tensor sendRankCountCumSum =
      at::empty({epSize}, expertsIds.options().dtype(at::ScalarType::Int));
  at::Tensor RecvRankCountCumSum =
      at::empty({epSize}, expertsIds.options().dtype(at::ScalarType::Int));

  at::Tensor gatherRecvRankIndices =
      at::empty({maxTokenCountPerRank * epSize}, expertsIds.options().dtype(at::ScalarType::Int));
  at::Tensor recvRankIndices =
      at::empty({maxTokenCountPerRank * epSize}, expertsIds.options().dtype(at::ScalarType::Int));

  at::Tensor gatherBackwardRecvRankIndices =
      at::empty({maxTokenCountPerRank * maxSendRanksPerToken},
                expertsIds.options().dtype(at::ScalarType::Int));
  at::Tensor backwardRecvRankIndices = at::empty({maxTokenCountPerRank * maxSendRanksPerToken},
                                                 expertsIds.options().dtype(at::ScalarType::Int));

  at::Tensor gatherSendRankIndices = at::empty({maxTokenCountPerRank * maxSendRanksPerToken},
                                               expertsIds.options().dtype(at::ScalarType::Int));
  at::Tensor sendRankIndices = at::empty({maxTokenCountPerRank * maxSendRanksPerToken},
                                         expertsIds.options().dtype(at::ScalarType::Int));

  c10::optional<at::Tensor> preparedLocalScales;
  float* scalesPtr = nullptr;
  float* preparedLocalScalesPtr = nullptr;
  if (scales.has_value()) {
    CHECK_INPUT_TYPE(scales.value(), at::ScalarType::Float);
    scalesPtr = scales->data_ptr<float>();
    preparedLocalScales = at::empty({maxTokenCountPerRank * epSize, topK},
                                    expertsIds.options().dtype(at::ScalarType::Float));
    preparedLocalScalesPtr = preparedLocalScales->data_ptr<float>();
  }

  int* localExpertStaticsPtr = nullptr;
  int* gatheredExpertStaticsPtr = nullptr;
  c10::optional<at::Tensor> gatheredExpertStatics;
  if (expertsStatics.has_value()) {
    localExpertStaticsPtr = expertsStatics.value().data_ptr<int>();
    gatheredExpertStatics =
        at::empty({epSize, expertCount}, expertsIds.options().dtype(at::ScalarType::Int));
    gatheredExpertStaticsPtr = gatheredExpertStatics.value().data_ptr<int>();
  }

  flashinfer::trtllm_alltoall::moe_prepare::MoeCommWorkspace workspace;
  workspace.workspacePtr = allWorkspaces.data_ptr<uint64_t>();
  workspace.rankStrideInU64 = allWorkspaces.stride(0);

  auto stream = at::cuda::getCurrentCUDAStream();

  flashinfer::trtllm_alltoall::moe_prepare::computeCountAndIndice(
      expertsIds.data_ptr<int>(), sendRankCountCumSum.data_ptr<int>(),
      RecvRankCountCumSum.data_ptr<int>(), sendRankIndices.data_ptr<int>(),
      backwardRecvRankIndices.data_ptr<int>(), recvRankIndices.data_ptr<int>(), workspace,
      tokenCount, maxTokenCountPerRank, topK, slotCount, epRank, epSize, stream);

  flashinfer::trtllm_alltoall::moe_prepare::computeCumsum(sendRankCountCumSum.data_ptr<int>(),
                                                          RecvRankCountCumSum.data_ptr<int>(),
                                                          epRank, epSize, stream);

  flashinfer::trtllm_alltoall::moe_prepare::moveIndice(
      sendRankCountCumSum.data_ptr<int>(), RecvRankCountCumSum.data_ptr<int>(),
      sendRankIndices.data_ptr<int>(), gatherSendRankIndices.data_ptr<int>(),
      backwardRecvRankIndices.data_ptr<int>(), gatherBackwardRecvRankIndices.data_ptr<int>(),
      recvRankIndices.data_ptr<int>(), gatherRecvRankIndices.data_ptr<int>(), epRank, epSize,
      maxTokenCountPerRank, stream);

  flashinfer::trtllm_alltoall::moe_prepare::allToAllMetadata(
      expertsIds.data_ptr<int>(), preparedLocalExpertIds.data_ptr<int>(), scalesPtr,
      preparedLocalScalesPtr, localExpertStaticsPtr, gatheredExpertStaticsPtr, workspace,
      sendRankCountCumSum.data_ptr<int>(), sendRankIndices.data_ptr<int>(),
      RecvRankCountCumSum.data_ptr<int>(), recvRankIndices.data_ptr<int>(), tokenCount,
      maxTokenCountPerRank, topK, expertCount, slotCount, epRank, epSize, stream);

  return std::make_tuple(preparedLocalExpertIds, preparedLocalScales, sendRankCountCumSum,
                         gatherSendRankIndices, RecvRankCountCumSum, gatherRecvRankIndices,
                         gatherBackwardRecvRankIndices, gatheredExpertStatics);
}

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("moe_comm_prepare_indices", &moeCommPrepareIndicesOp);
  m.def("moe_local_gather", &moeLocalGatherOp);
  m.def("moe_comm", &moeCommOp);
  m.def("set_moe_max_usable_sm_count", static_cast<void (*)(int64_t)>(&setMaxUsableSmCount));
  m.def("get_moe_commworkspace_size_per_rank", &getWorkspaceSizePerRank);
  m.def("get_moe_prepare_workspace_size_per_rank", &getPrepareWorkspaceSizePerRank);
  m.def("moe_prepare", &moePrepareOp);
}
