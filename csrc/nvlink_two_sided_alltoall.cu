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

#include "nvlink_two_sided_alltoall.cuh"
#include "nvlink_two_sided_prepare.cuh"
#include "tvm_ffi_utils.h"

using namespace flashinfer::nvlink_two_sided;

using tvm::ffi::Optional;
using tvm::ffi::Tuple;

void moeCommOp(TensorView input, TensorView sendRankCumSum, TensorView sendIndices,
               TensorView output, TensorView recvRankCumSum, TensorView recvIndices,
               TensorView allWorkspaces, int64_t epRank, int64_t epSize) {
  CHECK_INPUT_TYPE(sendRankCumSum, dl_int32);
  CHECK_INPUT_TYPE(sendIndices, dl_int32);
  CHECK_INPUT_TYPE(recvRankCumSum, dl_int32);
  CHECK_INPUT_TYPE(recvIndices, dl_int32);
  // allWorkspaces is a uint64 tensor, but may not be contiguous
  CHECK_INPUT_TYPE(allWorkspaces, dl_uint64);

  TVM_FFI_ICHECK_EQ(input.ndim(), 2) << "input must be a 2D tensor";
  TVM_FFI_ICHECK_EQ(output.ndim(), 2) << "output must be a 2D tensor";
  TVM_FFI_ICHECK_EQ(sendRankCumSum.ndim(), 1) << "sendRankCumSum must be a 1D tensor";
  TVM_FFI_ICHECK_EQ(sendIndices.ndim(), 1) << "sendIndices must be a 1D tensor";
  TVM_FFI_ICHECK_EQ(recvRankCumSum.ndim(), 1) << "recvRankCumSum must be a 1D tensor";
  TVM_FFI_ICHECK_EQ(recvIndices.ndim(), 1) << "recvIndices must be a 1D tensor";
  TVM_FFI_ICHECK_EQ(allWorkspaces.ndim(), 2) << "allWorkspaces must be a 2D tensor";

  TVM_FFI_ICHECK_EQ(input.size(1), output.size(1))
      << "input and output must have the same second dimension";
  TVM_FFI_ICHECK_EQ(sendRankCumSum.size(0), epSize) << "sendRankCumSum must have epSize elements";
  TVM_FFI_ICHECK_EQ(recvRankCumSum.size(0), epSize) << "recvRankCumSum must have epSize elements";
  TVM_FFI_ICHECK_EQ(allWorkspaces.size(0), epSize) << "allWorkspaces must have epSize elements";

  TVM_FFI_ICHECK(epRank >= 0 && epRank < epSize) << "epRank must be in the range [0, epSize)";

  flashinfer::nvlink_two_sided::MoeEpWorldInfo worldInfo = {static_cast<int>(epSize),
                                                            static_cast<int>(epRank)};
  flashinfer::nvlink_two_sided::SendRecvDataInfo sendRecvDataInfo;

  size_t eltSize = get_element_size(input);
  size_t eltCountPerU64 = sizeof(uint64_t) / eltSize;
  TVM_FFI_ICHECK_EQ(input.size(1) % (eltCountPerU64 * 2), 0)
      << "input.size(1) must be aligned to 16 bytes";
  sendRecvDataInfo.vectorSizeInU64 = input.size(1) / eltCountPerU64;
  sendRecvDataInfo.DoPreCompute();

  flashinfer::nvlink_two_sided::SendRecvDispls sendDispls, recvDispls;
  sendDispls.dataPtr = static_cast<uint64_t*>(input.data_ptr());
  sendDispls.rankCountCumSum = static_cast<int*>(sendRankCumSum.data_ptr());
  sendDispls.rankLocalIndices = static_cast<int*>(sendIndices.data_ptr());
  sendDispls.vectorStrideInU64 = input.stride(0) / eltCountPerU64;

  recvDispls.dataPtr = static_cast<uint64_t*>(output.data_ptr());
  recvDispls.rankCountCumSum = static_cast<int*>(recvRankCumSum.data_ptr());
  recvDispls.rankLocalIndices = static_cast<int*>(recvIndices.data_ptr());
  recvDispls.vectorStrideInU64 = output.stride(0) / eltCountPerU64;

  flashinfer::nvlink_two_sided::MoeCommWorkspace workspace;
  workspace.workspacePtr = static_cast<uint64_t*>(allWorkspaces.data_ptr());
  workspace.rankStrideInU64 = allWorkspaces.stride(0);

  auto stream = get_current_stream();

  flashinfer::nvlink_two_sided::moeAllToAll(worldInfo, sendRecvDataInfo, sendDispls, recvDispls,
                                            workspace, stream);
}

int64_t getWorkspaceSizePerRank(int64_t epSize) {
  int epSize32 = static_cast<int>(epSize);
  return flashinfer::nvlink_two_sided::getMoeCommWorkspaceSize(epSize32);
}

int64_t getPrepareWorkspaceSizePerRank(int64_t epSize) {
  int epSize32 = static_cast<int>(epSize);
  return flashinfer::nvlink_two_sided::moe_prepare::getMoePrepareWorkspaceSize(epSize32);
}

void moePrepareOp(TensorView expertsIds, Optional<TensorView> scales,
                  Optional<TensorView> expertsStatics, TensorView allWorkspaces,
                  TensorView preparedLocalExpertIds, TensorView sendRankCountCumSum,
                  TensorView recvRankCountCumSum, TensorView gatherRecvRankIndices,
                  TensorView recvRankIndices, TensorView gatherBackwardRecvRankIndices,
                  TensorView backwardRecvRankIndices, TensorView gatherSendRankIndices,
                  TensorView sendRankIndices, Optional<TensorView> preparedLocalScales,
                  Optional<TensorView> gatheredExpertStatics, int64_t maxTokenCountPerRank,
                  int64_t epRank, int64_t epSize, int64_t expertCount, int64_t slotCount,
                  int64_t topK) {
  CHECK_INPUT_TYPE(expertsIds, dl_int32);
  TVM_FFI_ICHECK_EQ(expertCount % 4, 0) << "expertCount must be divisible by 4";
  TVM_FFI_ICHECK_EQ(slotCount % 4, 0) << "slotCount must be divisible by 4";

  int64_t maxSendRanksPerToken = std::max(epSize, topK);
  int64_t tokenCount = expertsIds.size(0);

  CHECK_DEVICE(preparedLocalExpertIds, expertsIds);
  CHECK_INPUT_TYPE(preparedLocalExpertIds, dl_int32);
  TVM_FFI_ICHECK_EQ(preparedLocalExpertIds.ndim(), 2);
  TVM_FFI_ICHECK_EQ(preparedLocalExpertIds.size(0), maxTokenCountPerRank * epSize);
  TVM_FFI_ICHECK_EQ(preparedLocalExpertIds.size(1), topK);

  CHECK_DEVICE(sendRankCountCumSum, expertsIds);
  CHECK_INPUT_TYPE(sendRankCountCumSum, dl_int32);
  TVM_FFI_ICHECK_EQ(sendRankCountCumSum.ndim(), 1);
  TVM_FFI_ICHECK_EQ(sendRankCountCumSum.size(0), epSize);

  CHECK_DEVICE(recvRankCountCumSum, expertsIds);
  CHECK_INPUT_TYPE(recvRankCountCumSum, dl_int32);
  TVM_FFI_ICHECK_EQ(recvRankCountCumSum.ndim(), 1);
  TVM_FFI_ICHECK_EQ(recvRankCountCumSum.size(0), epSize);

  CHECK_DEVICE(gatherRecvRankIndices, expertsIds);
  CHECK_INPUT_TYPE(gatherRecvRankIndices, dl_int32);
  TVM_FFI_ICHECK_EQ(gatherRecvRankIndices.ndim(), 1);
  TVM_FFI_ICHECK_EQ(gatherRecvRankIndices.size(0), maxTokenCountPerRank * epSize);

  CHECK_DEVICE(recvRankIndices, expertsIds);
  CHECK_INPUT_TYPE(recvRankIndices, dl_int32);
  TVM_FFI_ICHECK_EQ(recvRankIndices.ndim(), 1);
  TVM_FFI_ICHECK_EQ(recvRankIndices.size(0), maxTokenCountPerRank * epSize);

  CHECK_DEVICE(gatherBackwardRecvRankIndices, expertsIds);
  CHECK_INPUT_TYPE(gatherBackwardRecvRankIndices, dl_int32);
  TVM_FFI_ICHECK_EQ(gatherBackwardRecvRankIndices.ndim(), 1);
  TVM_FFI_ICHECK_EQ(gatherBackwardRecvRankIndices.size(0),
                    maxTokenCountPerRank * maxSendRanksPerToken);

  CHECK_DEVICE(backwardRecvRankIndices, expertsIds);
  CHECK_INPUT_TYPE(backwardRecvRankIndices, dl_int32);
  TVM_FFI_ICHECK_EQ(backwardRecvRankIndices.ndim(), 1);
  TVM_FFI_ICHECK_EQ(backwardRecvRankIndices.size(0), maxTokenCountPerRank * maxSendRanksPerToken);

  CHECK_DEVICE(gatherSendRankIndices, expertsIds);
  CHECK_INPUT_TYPE(gatherSendRankIndices, dl_int32);
  TVM_FFI_ICHECK_EQ(gatherSendRankIndices.ndim(), 1);
  TVM_FFI_ICHECK_EQ(gatherSendRankIndices.size(0), maxTokenCountPerRank * maxSendRanksPerToken);

  CHECK_DEVICE(sendRankIndices, expertsIds);
  CHECK_INPUT_TYPE(sendRankIndices, dl_int32);
  TVM_FFI_ICHECK_EQ(sendRankIndices.ndim(), 1);
  TVM_FFI_ICHECK_EQ(sendRankIndices.size(0), maxTokenCountPerRank * maxSendRanksPerToken);

  float* scalesPtr = nullptr;
  float* preparedLocalScalesPtr = nullptr;
  if (scales.has_value()) {
    CHECK_INPUT_TYPE(scales.value(), dl_float32);
    scalesPtr = static_cast<float*>(scales.value().data_ptr());
    CHECK_DEVICE(preparedLocalScales.value(), expertsIds);
    CHECK_INPUT_TYPE(preparedLocalScales.value(), dl_float32);
    TVM_FFI_ICHECK_EQ(preparedLocalScales.value().ndim(), 2);
    TVM_FFI_ICHECK_EQ(preparedLocalScales.value().size(0), maxTokenCountPerRank * epSize);
    TVM_FFI_ICHECK_EQ(preparedLocalScales.value().size(1), topK);
    preparedLocalScalesPtr = static_cast<float*>(preparedLocalScales.value().data_ptr());
  }

  int* localExpertStaticsPtr = nullptr;
  int* gatheredExpertStaticsPtr = nullptr;
  if (expertsStatics.has_value()) {
    localExpertStaticsPtr = static_cast<int*>(expertsStatics.value().data_ptr());
    CHECK_DEVICE(gatheredExpertStatics.value(), expertsIds);
    CHECK_INPUT_TYPE(gatheredExpertStatics.value(), dl_int32);
    TVM_FFI_ICHECK_EQ(gatheredExpertStatics.value().ndim(), 2);
    TVM_FFI_ICHECK_EQ(gatheredExpertStatics.value().size(0), epSize);
    TVM_FFI_ICHECK_EQ(gatheredExpertStatics.value().size(1), expertCount);
    gatheredExpertStaticsPtr = static_cast<int*>(gatheredExpertStatics.value().data_ptr());
  }

  flashinfer::nvlink_two_sided::moe_prepare::MoeCommWorkspace workspace;
  workspace.workspacePtr = static_cast<uint64_t*>(allWorkspaces.data_ptr());
  workspace.rankStrideInU64 = allWorkspaces.stride(0);

  auto stream = get_current_stream();

  flashinfer::nvlink_two_sided::moe_prepare::computeCountAndIndice(
      static_cast<int*>(expertsIds.data_ptr()), static_cast<int*>(sendRankCountCumSum.data_ptr()),
      static_cast<int*>(recvRankCountCumSum.data_ptr()),
      static_cast<int*>(sendRankIndices.data_ptr()),
      static_cast<int*>(backwardRecvRankIndices.data_ptr()),
      static_cast<int*>(recvRankIndices.data_ptr()), workspace, tokenCount, maxTokenCountPerRank,
      topK, slotCount, epRank, epSize, stream);

  flashinfer::nvlink_two_sided::moe_prepare::computeCumsum(
      static_cast<int*>(sendRankCountCumSum.data_ptr()),
      static_cast<int*>(recvRankCountCumSum.data_ptr()), epRank, epSize, stream);

  flashinfer::nvlink_two_sided::moe_prepare::moveIndice(
      static_cast<int*>(sendRankCountCumSum.data_ptr()),
      static_cast<int*>(recvRankCountCumSum.data_ptr()),
      static_cast<int*>(sendRankIndices.data_ptr()),
      static_cast<int*>(gatherSendRankIndices.data_ptr()),
      static_cast<int*>(backwardRecvRankIndices.data_ptr()),
      static_cast<int*>(gatherBackwardRecvRankIndices.data_ptr()),
      static_cast<int*>(recvRankIndices.data_ptr()),
      static_cast<int*>(gatherRecvRankIndices.data_ptr()), epRank, epSize, maxTokenCountPerRank,
      stream);

  flashinfer::nvlink_two_sided::moe_prepare::allToAllMetadata(
      static_cast<int*>(expertsIds.data_ptr()),
      static_cast<int*>(preparedLocalExpertIds.data_ptr()), scalesPtr, preparedLocalScalesPtr,
      localExpertStaticsPtr, gatheredExpertStaticsPtr, workspace,
      static_cast<int*>(sendRankCountCumSum.data_ptr()),
      static_cast<int*>(sendRankIndices.data_ptr()),
      static_cast<int*>(recvRankCountCumSum.data_ptr()),
      static_cast<int*>(recvRankIndices.data_ptr()), tokenCount, maxTokenCountPerRank, topK,
      expertCount, slotCount, epRank, epSize, stream);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_comm, moeCommOp);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(set_moe_max_usable_sm_count, setMaxUsableSmCount);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_moe_commworkspace_size_per_rank, getWorkspaceSizePerRank);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_moe_prepare_workspace_size_per_rank,
                              getPrepareWorkspaceSizePerRank);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_prepare, moePrepareOp);
