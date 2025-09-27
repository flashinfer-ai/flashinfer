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

#include <tvm/ffi/container/tuple.h>

#include "flashinfer/comm/trtllm_alltoall.cuh"
#include "flashinfer/comm/trtllm_alltoall_prepare.cuh"
#include "tvm_ffi_utils.h"

using namespace flashinfer::trtllm_alltoall;

using tvm::ffi::Optional;
using tvm::ffi::Tuple;

void moeCommPrepareIndicesOp(Tensor gatheredTargetRankIds,
                             Optional<Tensor> realRankTokenCountCumSum, Tensor localGatherIndices,
                             Tensor sendRankCountCumSum, Tensor sendRankLocalIndices,
                             Tensor recvRankCountCumSum, Tensor recvRankLocalIndices,
                             Tensor backwardRecvRankLocalIndices, int64_t maxTokenCountPerRank,
                             int64_t expertCount, int64_t topK, int64_t epRank, int64_t epSize) {
  CHECK_INPUT_TYPE(gatheredTargetRankIds, dl_int32);
  TVM_FFI_ICHECK_EQ(gatheredTargetRankIds->ndim, 2) << "gatheredTargetRankIds must be a 2D tensor";
  TVM_FFI_ICHECK_EQ(gatheredTargetRankIds->shape[1], topK)
      << "gatheredTargetRankIds must have topK columns";

  int const* realRankTokenCountCumSumPtr = nullptr;
  if (realRankTokenCountCumSum.has_value()) {
    TVM_FFI_ICHECK_EQ(realRankTokenCountCumSum.value()->ndim, 1)
        << "realRankTokenCountCumSum must be a 1D tensor";
    CHECK_INPUT_TYPE(realRankTokenCountCumSum.value(), dl_int32)
    TVM_FFI_ICHECK_EQ(realRankTokenCountCumSum.value()->shape[0], epSize)
        << "realRankTokenCountCumSum must have epSize elements";
    realRankTokenCountCumSumPtr = static_cast<int*>(realRankTokenCountCumSum.value()->data);
  } else {
    TVM_FFI_ICHECK_EQ(gatheredTargetRankIds->shape[0], epSize * maxTokenCountPerRank)
        << "gatheredTargetRankIds should have shape (epSize * maxTokenCountPerRank, topK)";
  }
  TVM_FFI_ICHECK_GT(maxTokenCountPerRank, 0) << "maxTokenCountPerRank must be greater than 0";
  TVM_FFI_ICHECK_GT(expertCount, 0) << "expertCount must be greater than 0";
  TVM_FFI_ICHECK_GT(topK, 0) << "topK must be greater than 0";
  TVM_FFI_ICHECK_LE(topK, expertCount) << "topK must be less than or equal to expertCount";
  TVM_FFI_ICHECK(epRank >= 0 && epRank < epSize) << "epRank must be in the range [0, epSize)";

  auto stream = get_current_stream();

  int maxSendRanksPerToken = std::max(static_cast<int>(epSize), static_cast<int>(topK));

  CHECK_INPUT_TYPE(localGatherIndices, dl_int32);
  TVM_FFI_ICHECK_EQ(localGatherIndices->ndim, 1);
  TVM_FFI_ICHECK_EQ(localGatherIndices->shape[0], maxTokenCountPerRank * epSize);

  CHECK_INPUT_TYPE(sendRankCountCumSum, dl_int32);
  TVM_FFI_ICHECK_EQ(sendRankCountCumSum->ndim, 1);
  TVM_FFI_ICHECK_EQ(sendRankCountCumSum->shape[0], epSize);

  CHECK_INPUT_TYPE(sendRankLocalIndices, dl_int32);
  TVM_FFI_ICHECK_EQ(sendRankLocalIndices->ndim, 1);
  TVM_FFI_ICHECK_EQ(sendRankLocalIndices->shape[0], maxTokenCountPerRank * maxSendRanksPerToken);

  CHECK_INPUT_TYPE(recvRankCountCumSum, dl_int32);
  TVM_FFI_ICHECK_EQ(recvRankCountCumSum->ndim, 1);
  TVM_FFI_ICHECK_EQ(recvRankCountCumSum->shape[0], epSize);

  CHECK_INPUT_TYPE(recvRankLocalIndices, dl_int32);
  TVM_FFI_ICHECK_EQ(recvRankLocalIndices->ndim, 1);
  TVM_FFI_ICHECK_EQ(recvRankLocalIndices->shape[0], maxTokenCountPerRank * epSize);

  CHECK_INPUT_TYPE(backwardRecvRankLocalIndices, dl_int32);
  TVM_FFI_ICHECK_EQ(backwardRecvRankLocalIndices->ndim, 1);
  TVM_FFI_ICHECK_EQ(backwardRecvRankLocalIndices->shape[0],
                    maxTokenCountPerRank * maxSendRanksPerToken);

  flashinfer::trtllm_alltoall::MoeExpertParallelInfo expertParallelInfo;
  expertParallelInfo.expertCount = static_cast<int>(expertCount);
  expertParallelInfo.topK = static_cast<int>(topK);

  flashinfer::trtllm_alltoall::MoeEpWorldInfo worldInfo = {static_cast<int>(epSize),
                                                           static_cast<int>(epRank)};

  auto cudaResult = flashinfer::trtllm_alltoall::moeAllToAllPrepareIndices(
      worldInfo, expertParallelInfo, static_cast<int>(maxTokenCountPerRank),
      static_cast<int*>(gatheredTargetRankIds->data), realRankTokenCountCumSumPtr,
      static_cast<int*>(localGatherIndices->data), static_cast<int*>(sendRankCountCumSum->data),
      static_cast<int*>(sendRankLocalIndices->data), static_cast<int*>(recvRankCountCumSum->data),
      static_cast<int*>(recvRankLocalIndices->data),
      static_cast<int*>(backwardRecvRankLocalIndices->data), stream);
  TVM_FFI_ICHECK(cudaResult == cudaSuccess)
      << "CUDA error in moeAllToAllPrepareIndices: " << cudaGetErrorString(cudaResult);
}

void moeLocalGatherOp(Tensor recvRankCumSum, Tensor localGatherIndices, Tensor gatheredExpertIds,
                      Tensor gatheredScales, Tensor localExpertIds, Tensor localScales,
                      int64_t maxTokenCountPerRank, int64_t expertCount, int64_t topK,
                      int64_t epRank, int64_t epSize) {
  CHECK_INPUT_TYPE(recvRankCumSum, dl_int32);
  CHECK_INPUT_TYPE(localGatherIndices, dl_int32);
  CHECK_INPUT_TYPE(gatheredExpertIds, dl_int32);
  CHECK_INPUT_TYPE(gatheredScales, dl_float32);
  CHECK_INPUT_TYPE(localExpertIds, dl_int32);
  CHECK_INPUT_TYPE(localScales, dl_float32);

  TVM_FFI_ICHECK_GT(maxTokenCountPerRank, 0) << "maxTokenCountPerRank must be greater than 0";
  TVM_FFI_ICHECK_GT(expertCount, 0) << "expertCount must be greater than 0";
  TVM_FFI_ICHECK_GT(topK, 0) << "topK must be greater than 0";
  TVM_FFI_ICHECK_LE(topK, expertCount) << "topK must be less than or equal to expertCount";
  TVM_FFI_ICHECK(epRank >= 0 && epRank < epSize) << "epRank must be in the range [0, epSize)";

  TVM_FFI_ICHECK_EQ(recvRankCumSum->ndim, 1) << "recvRankCumSum must be a 1D tensor";
  TVM_FFI_ICHECK_EQ(recvRankCumSum->shape[0], epSize) << "recvRankCumSum must have epSize elements";
  TVM_FFI_ICHECK_EQ(localGatherIndices->ndim, 1) << "localGatherIndices must be a 1D tensor";
  TVM_FFI_ICHECK_EQ(gatheredExpertIds->ndim, 2) << "gatheredExpertIds must be a 2D tensor";
  TVM_FFI_ICHECK_EQ(gatheredScales->ndim, 2) << "gatheredScales must be a 2D tensor";
  TVM_FFI_ICHECK_EQ(localExpertIds->ndim, 2) << "localExpertIds must be a 2D tensor";
  TVM_FFI_ICHECK_EQ(localScales->ndim, 2) << "localScales must be a 2D tensor";
  TVM_FFI_ICHECK_EQ(gatheredExpertIds->shape[1], topK)
      << "gatheredExpertIds must have topK columns";
  TVM_FFI_ICHECK_EQ(gatheredScales->shape[1], topK) << "gatheredScales must have topK columns";
  TVM_FFI_ICHECK_EQ(localExpertIds->shape[1], topK) << "localExpertIds must have topK columns";
  TVM_FFI_ICHECK_EQ(localScales->shape[1], topK) << "localScales must have topK columns";

  int localMaxTokenCount = static_cast<int>(localGatherIndices->shape[0]);
  TVM_FFI_ICHECK_EQ(localExpertIds->shape[0], localMaxTokenCount)
      << "localExpertIds must have localMaxTokenCount rows";
  TVM_FFI_ICHECK_EQ(localScales->shape[0], localMaxTokenCount)
      << "localScales must have localMaxTokenCount rows";

  auto stream = get_current_stream();

  flashinfer::trtllm_alltoall::MoeExpertParallelInfo expertParallelInfo;
  expertParallelInfo.expertCount = static_cast<int>(expertCount);
  expertParallelInfo.topK = static_cast<int>(topK);

  flashinfer::trtllm_alltoall::MoeEpWorldInfo worldInfo = {static_cast<int>(epSize),
                                                           static_cast<int>(epRank)};
  flashinfer::trtllm_alltoall::moeLocalGather(
      worldInfo, expertParallelInfo, static_cast<int>(maxTokenCountPerRank), localMaxTokenCount,
      static_cast<int*>(recvRankCumSum->data), static_cast<int*>(localGatherIndices->data),
      static_cast<int*>(gatheredExpertIds->data), static_cast<float*>(gatheredScales->data),
      static_cast<int*>(localExpertIds->data), static_cast<float*>(localScales->data), stream);
}

void moeCommOp(Tensor input, Tensor sendRankCumSum, Tensor sendIndices, Tensor output,
               Tensor recvRankCumSum, Tensor recvIndices, Tensor allWorkspaces, int64_t epRank,
               int64_t epSize) {
  CHECK_INPUT_TYPE(sendRankCumSum, dl_int32);
  CHECK_INPUT_TYPE(sendIndices, dl_int32);
  CHECK_INPUT_TYPE(recvRankCumSum, dl_int32);
  CHECK_INPUT_TYPE(recvIndices, dl_int32);
  // allWorkspaces is a uint64 tensor, but may not be contiguous
  CHECK_INPUT_TYPE(allWorkspaces, dl_uint64);

  TVM_FFI_ICHECK_EQ(input->ndim, 2) << "input must be a 2D tensor";
  TVM_FFI_ICHECK_EQ(output->ndim, 2) << "output must be a 2D tensor";
  TVM_FFI_ICHECK_EQ(sendRankCumSum->ndim, 1) << "sendRankCumSum must be a 1D tensor";
  TVM_FFI_ICHECK_EQ(sendIndices->ndim, 1) << "sendIndices must be a 1D tensor";
  TVM_FFI_ICHECK_EQ(recvRankCumSum->ndim, 1) << "recvRankCumSum must be a 1D tensor";
  TVM_FFI_ICHECK_EQ(recvIndices->ndim, 1) << "recvIndices must be a 1D tensor";
  TVM_FFI_ICHECK_EQ(allWorkspaces->ndim, 2) << "allWorkspaces must be a 2D tensor";

  TVM_FFI_ICHECK_EQ(input->shape[1], output->shape[1])
      << "input and output must have the same second dimension";
  TVM_FFI_ICHECK_EQ(sendRankCumSum->shape[0], epSize) << "sendRankCumSum must have epSize elements";
  TVM_FFI_ICHECK_EQ(recvRankCumSum->shape[0], epSize) << "recvRankCumSum must have epSize elements";
  TVM_FFI_ICHECK_EQ(allWorkspaces->shape[0], epSize) << "allWorkspaces must have epSize elements";

  TVM_FFI_ICHECK(epRank >= 0 && epRank < epSize) << "epRank must be in the range [0, epSize)";

  flashinfer::trtllm_alltoall::MoeEpWorldInfo worldInfo = {static_cast<int>(epSize),
                                                           static_cast<int>(epRank)};
  flashinfer::trtllm_alltoall::SendRecvDataInfo sendRecvDataInfo;

  size_t eltSize = get_element_size(input);
  size_t eltCountPerU64 = sizeof(uint64_t) / eltSize;
  TVM_FFI_ICHECK_EQ(input->shape[1] % (eltCountPerU64 * 2), 0)
      << "input->shape[1] must be aligned to 16 bytes";
  sendRecvDataInfo.vectorSizeInU64 = input->shape[1] / eltCountPerU64;
  sendRecvDataInfo.DoPreCompute();

  flashinfer::trtllm_alltoall::SendRecvDispls sendDispls, recvDispls;
  sendDispls.dataPtr = static_cast<uint64_t*>(input->data);
  sendDispls.rankCountCumSum = static_cast<int*>(sendRankCumSum->data);
  sendDispls.rankLocalIndices = static_cast<int*>(sendIndices->data);
  sendDispls.vectorStrideInU64 = input->strides[0] / eltCountPerU64;

  recvDispls.dataPtr = static_cast<uint64_t*>(output->data);
  recvDispls.rankCountCumSum = static_cast<int*>(recvRankCumSum->data);
  recvDispls.rankLocalIndices = static_cast<int*>(recvIndices->data);
  recvDispls.vectorStrideInU64 = output->strides[0] / eltCountPerU64;

  flashinfer::trtllm_alltoall::MoeCommWorkspace workspace;
  workspace.workspacePtr = static_cast<uint64_t*>(allWorkspaces->data);
  workspace.rankStrideInU64 = allWorkspaces->strides[0];

  auto stream = get_current_stream();

  flashinfer::trtllm_alltoall::moeAllToAll(worldInfo, sendRecvDataInfo, sendDispls, recvDispls,
                                           workspace, stream);
}

int64_t getWorkspaceSizePerRank(int64_t epSize) {
  int epSize32 = static_cast<int>(epSize);
  return flashinfer::trtllm_alltoall::getMoeCommWorkspaceSize(epSize32);
}

int64_t getPrepareWorkspaceSizePerRank(int64_t epSize) {
  int epSize32 = static_cast<int>(epSize);
  return flashinfer::trtllm_alltoall::moe_prepare::getMoePrepareWorkspaceSize(epSize32);
}

void moePrepareOp(Tensor expertsIds, Optional<Tensor> scales, Optional<Tensor> expertsStatics,
                  Tensor allWorkspaces, Tensor preparedLocalExpertIds, Tensor sendRankCountCumSum,
                  Tensor recvRankCountCumSum, Tensor gatherRecvRankIndices, Tensor recvRankIndices,
                  Tensor gatherBackwardRecvRankIndices, Tensor backwardRecvRankIndices,
                  Tensor gatherSendRankIndices, Tensor sendRankIndices,
                  Optional<Tensor> preparedLocalScales, Optional<Tensor> gatheredExpertStatics,
                  int64_t maxTokenCountPerRank, int64_t epRank, int64_t epSize, int64_t expertCount,
                  int64_t slotCount, int64_t topK) {
  CHECK_INPUT_TYPE(expertsIds, dl_int32);
  TVM_FFI_ICHECK_EQ(expertCount % 4, 0) << "expertCount must be divisible by 4";
  TVM_FFI_ICHECK_EQ(slotCount % 4, 0) << "slotCount must be divisible by 4";

  int64_t maxSendRanksPerToken = std::max(epSize, topK);
  int64_t tokenCount = expertsIds->shape[0];

  CHECK_DEVICE(preparedLocalExpertIds, expertsIds);
  CHECK_INPUT_TYPE(preparedLocalExpertIds, dl_int32);
  TVM_FFI_ICHECK_EQ(preparedLocalExpertIds->ndim, 2);
  TVM_FFI_ICHECK_EQ(preparedLocalExpertIds->shape[0], maxTokenCountPerRank * epSize);
  TVM_FFI_ICHECK_EQ(preparedLocalExpertIds->shape[1], topK);

  CHECK_DEVICE(sendRankCountCumSum, expertsIds);
  CHECK_INPUT_TYPE(sendRankCountCumSum, dl_int32);
  TVM_FFI_ICHECK_EQ(sendRankCountCumSum->ndim, 1);
  TVM_FFI_ICHECK_EQ(sendRankCountCumSum->shape[0], epSize);

  CHECK_DEVICE(recvRankCountCumSum, expertsIds);
  CHECK_INPUT_TYPE(recvRankCountCumSum, dl_int32);
  TVM_FFI_ICHECK_EQ(recvRankCountCumSum->ndim, 1);
  TVM_FFI_ICHECK_EQ(recvRankCountCumSum->shape[0], epSize);

  CHECK_DEVICE(gatherRecvRankIndices, expertsIds);
  CHECK_INPUT_TYPE(gatherRecvRankIndices, dl_int32);
  TVM_FFI_ICHECK_EQ(gatherRecvRankIndices->ndim, 1);
  TVM_FFI_ICHECK_EQ(gatherRecvRankIndices->shape[0], maxTokenCountPerRank * epSize);

  CHECK_DEVICE(recvRankIndices, expertsIds);
  CHECK_INPUT_TYPE(recvRankIndices, dl_int32);
  TVM_FFI_ICHECK_EQ(recvRankIndices->ndim, 1);
  TVM_FFI_ICHECK_EQ(recvRankIndices->shape[0], maxTokenCountPerRank * epSize);

  CHECK_DEVICE(gatherBackwardRecvRankIndices, expertsIds);
  CHECK_INPUT_TYPE(gatherBackwardRecvRankIndices, dl_int32);
  TVM_FFI_ICHECK_EQ(gatherBackwardRecvRankIndices->ndim, 1);
  TVM_FFI_ICHECK_EQ(gatherBackwardRecvRankIndices->shape[0],
                    maxTokenCountPerRank * maxSendRanksPerToken);

  CHECK_DEVICE(backwardRecvRankIndices, expertsIds);
  CHECK_INPUT_TYPE(backwardRecvRankIndices, dl_int32);
  TVM_FFI_ICHECK_EQ(backwardRecvRankIndices->ndim, 1);
  TVM_FFI_ICHECK_EQ(backwardRecvRankIndices->shape[0], maxTokenCountPerRank * maxSendRanksPerToken);

  CHECK_DEVICE(gatherSendRankIndices, expertsIds);
  CHECK_INPUT_TYPE(gatherSendRankIndices, dl_int32);
  TVM_FFI_ICHECK_EQ(gatherSendRankIndices->ndim, 1);
  TVM_FFI_ICHECK_EQ(gatherSendRankIndices->shape[0], maxTokenCountPerRank * maxSendRanksPerToken);

  CHECK_DEVICE(sendRankIndices, expertsIds);
  CHECK_INPUT_TYPE(sendRankIndices, dl_int32);
  TVM_FFI_ICHECK_EQ(sendRankIndices->ndim, 1);
  TVM_FFI_ICHECK_EQ(sendRankIndices->shape[0], maxTokenCountPerRank * maxSendRanksPerToken);

  float* scalesPtr = nullptr;
  float* preparedLocalScalesPtr = nullptr;
  if (scales.has_value()) {
    CHECK_INPUT_TYPE(scales.value(), dl_float32);
    scalesPtr = static_cast<float*>(scales.value()->data);
    CHECK_DEVICE(preparedLocalScales.value(), expertsIds);
    CHECK_INPUT_TYPE(preparedLocalScales.value(), dl_int32);
    TVM_FFI_ICHECK_EQ(preparedLocalScales.value()->ndim, 2);
    TVM_FFI_ICHECK_EQ(preparedLocalScales.value()->shape[0], maxTokenCountPerRank * epSize);
    TVM_FFI_ICHECK_EQ(preparedLocalScales.value()->shape[1], topK);
    preparedLocalScalesPtr = static_cast<float*>(preparedLocalScales.value()->data);
  }

  int* localExpertStaticsPtr = nullptr;
  int* gatheredExpertStaticsPtr = nullptr;
  if (expertsStatics.has_value()) {
    localExpertStaticsPtr = static_cast<int*>(expertsStatics.value()->data);
    CHECK_DEVICE(gatheredExpertStatics.value(), expertsIds);
    CHECK_INPUT_TYPE(gatheredExpertStatics.value(), dl_int32);
    TVM_FFI_ICHECK_EQ(gatheredExpertStatics.value()->ndim, 2);
    TVM_FFI_ICHECK_EQ(gatheredExpertStatics.value()->shape[0], epSize);
    TVM_FFI_ICHECK_EQ(gatheredExpertStatics.value()->shape[1], expertCount);
    gatheredExpertStaticsPtr = static_cast<int*>(gatheredExpertStatics.value()->data);
  }

  flashinfer::trtllm_alltoall::moe_prepare::MoeCommWorkspace workspace;
  workspace.workspacePtr = static_cast<uint64_t*>(allWorkspaces->data);
  workspace.rankStrideInU64 = allWorkspaces->strides[0];

  auto stream = get_current_stream();

  flashinfer::trtllm_alltoall::moe_prepare::computeCountAndIndice(
      static_cast<int*>(expertsIds->data), static_cast<int*>(sendRankCountCumSum->data),
      static_cast<int*>(recvRankCountCumSum->data), static_cast<int*>(sendRankIndices->data),
      static_cast<int*>(backwardRecvRankIndices->data), static_cast<int*>(recvRankIndices->data),
      workspace, tokenCount, maxTokenCountPerRank, topK, slotCount, epRank, epSize, stream);

  flashinfer::trtllm_alltoall::moe_prepare::computeCumsum(
      static_cast<int*>(sendRankCountCumSum->data), static_cast<int*>(recvRankCountCumSum->data),
      epRank, epSize, stream);

  flashinfer::trtllm_alltoall::moe_prepare::moveIndice(
      static_cast<int*>(sendRankCountCumSum->data), static_cast<int*>(recvRankCountCumSum->data),
      static_cast<int*>(sendRankIndices->data), static_cast<int*>(gatherSendRankIndices->data),
      static_cast<int*>(backwardRecvRankIndices->data),
      static_cast<int*>(gatherBackwardRecvRankIndices->data),
      static_cast<int*>(recvRankIndices->data), static_cast<int*>(gatherRecvRankIndices->data),
      epRank, epSize, maxTokenCountPerRank, stream);

  flashinfer::trtllm_alltoall::moe_prepare::allToAllMetadata(
      static_cast<int*>(expertsIds->data), static_cast<int*>(preparedLocalExpertIds->data),
      scalesPtr, preparedLocalScalesPtr, localExpertStaticsPtr, gatheredExpertStaticsPtr, workspace,
      static_cast<int*>(sendRankCountCumSum->data), static_cast<int*>(sendRankIndices->data),
      static_cast<int*>(recvRankCountCumSum->data), static_cast<int*>(recvRankIndices->data),
      tokenCount, maxTokenCountPerRank, topK, expertCount, slotCount, epRank, epSize, stream);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_comm_prepare_indices, moeCommPrepareIndicesOp);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_local_gather, moeLocalGatherOp);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_comm, moeCommOp);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(set_moe_max_usable_sm_count, setMaxUsableSmCount);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_moe_commworkspace_size_per_rank, getWorkspaceSizePerRank);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_moe_prepare_workspace_size_per_rank,
                              getPrepareWorkspaceSizePerRank);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_prepare, moePrepareOp);
