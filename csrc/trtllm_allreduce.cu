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
  // TODO: add lamport initialization to lamportInitialize
  auto status =
      lamportInitialize(buffer.data_ptr(), buffer.numel(), at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == cudaSuccess, "lamportInitialize failed with error code " +
                                         std::string(cudaGetErrorString(status)));
}

void trtllm_custom_all_reduce(at::Tensor& buffer, at::Tensor& in, at::Tensor& out, int64_t tp_size,
                              int64_t tp_rank, int64_t token_num, int64_t hidden_size,
                              int64_t fusion_op_code, int64_t strategy_code, int64_t config_code,
                              int64_t elts_total, std::optional<at::Tensor> bias,
                              std::optional<at::Tensor> residual,
                              std::optional<int64_t> hidden_size, std::optional<at::Tensor> weight,
                              std::optional<at::Tensor> weight_pre_residual_norm,
                              std::optional<float> eps,
                              std::optional<at::Tensor> intermediate_buffer,
                              std::optional<at::Tensor> lamport_peer_comm_buffer_ptrs) {
  AllReduceFusionOp fusion_op = static_cast<AllReduceFusionOp>(fusion_op_code);
  const c10::cuda::OptionalCUDAGuard device_guard(buffer.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  // TODO(zihao): review dispatch
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_ALLREDUCE(in, out, scalar_t, [&] {
    auto params =
        AllReduceParams<scalar_t>::deserialize(static_cast<int64_t*>(buffer.data_ptr()), tp_size,
                                               tp_rank, token_num, hidden_size, fusion_op);

    // TODO(yingyi): complete params init
    params.elts_total = elts_total;
    params.local_input_buffer_ptr = in.data_ptr();
    params.local_output_buffer_ptr = out.data_ptr();

    // TODO(yingyi): add fusion params
    params.fusion_params.bias_buffer = bias.has_value() ? bias.value().data_ptr() : nullptr;
    params.fusion_params.residual_buffer =
        residual.has_value() ? residual.value().data_ptr() : nullptr;
    params.fusion_params.weight_buffer = weight.has_value() ? weight.value().data_ptr() : nullptr;
    params.fusion_params.weight_buffer_pre_residual_norm =
        weight_pre_residual_norm.has_value() ? weight_pre_residual_norm.value().data_ptr()
                                             : nullptr;
    params.fusion_params.eps = eps.has_value() ? eps.value() : 1e-5f;
    params.fusion_params.intermediate_buffer =
        intermediate_buffer.has_value() ? intermediate_buffer.value().data_ptr() : nullptr;
    params.fusion_params.lamport_peer_comm_buffer_ptrs =
        lamport_peer_comm_buffer_ptrs.has_value() ? lamport_peer_comm_buffer_ptrs.value().data_ptr()
                                                  : nullptr;

    auto strategy = static_cast<AllReduceStrategyType>(strategy_code);
    auto config = static_cast<AllReduceStrategyConfig>(config_code);

    // TODO(yingyi): default launch_with_pdl to true?
    auto status =
        customAllReduce(params, strategy, config, fusion_op, /*launch_with_pdl=*/true, stream);
    TORCH_CHECK(status == cudaSuccess, "customAllReduce failed with error code " +
                                           std::string(cudaGetErrorString(status)));
  });
}
