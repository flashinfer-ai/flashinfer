/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include <string>

#include "flashinfer/comm/trtllm_allreduce.cuh"
#include "pytorch_extension_utils.h"

using namespace flashinfer::trtllm_allreduce;

void trtllm_lamport_initialize(at::Tensor& buffer) {
  // TODO
}

void trtllm_custom_all_reduce(at::Tensor& buffer, int64_t tp_size, int64_t tp_rank,
                              int64_t token_num, int64_t hidden_size, int64_t fusion_op_code,
                              int64_t strategy_code) {
  AllReduceFusionOp fusion_op = static_cast<AllReduceFusionOp>(fusion_op_code);
  const c10::cuda::OptionalCUDAGuard device_guard(buffer.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  auto params =
      AllReduceParams<half>::deserialize(static_cast<int64_t*>(buffer.data_ptr()), tp_size, tp_rank,
                                         token_num, hidden_size, fusion_op);

  // TODO: add fusion params

  auto strategy = static_cast<AllReduceStrategyType>(strategy_code);
  auto config = AllReduceStrategyConfig::USE_MEMCPY;

  auto status =
      customAllReduce(params, strategy, config, fusion_op, /*launch_with_pdl=*/true, stream);
  TORCH_CHECK(status == cudaSuccess,
              "customAllReduce failed with error code " + std::string(cudaGetErrorString(status)));
}
