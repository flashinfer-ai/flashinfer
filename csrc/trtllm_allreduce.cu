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

#define DISPATCH_ALLREDUCE_DTYPE(TENSOR_SCALAR_TYPE, CTYPE_ALIAS, CODE_BLOCK)               \
  [&]() {                                                                                   \
    if (TENSOR_SCALAR_TYPE == at::ScalarType::Float) {                                      \
      using CTYPE_ALIAS = float;                                                            \
      CODE_BLOCK;                                                                           \
    } else if (TENSOR_SCALAR_TYPE == at::ScalarType::Half) {                                \
      using CTYPE_ALIAS = half;                                                             \
      CODE_BLOCK;                                                                           \
    } else if (TENSOR_SCALAR_TYPE == at::ScalarType::BFloat16) {                            \
      using CTYPE_ALIAS = __nv_bfloat16;                                                    \
      CODE_BLOCK;                                                                           \
    } else {                                                                                \
      TORCH_CHECK(false, "Unsupported DType for custom op dispatch: ", TENSOR_SCALAR_TYPE); \
    }                                                                                       \
  }()

#define DISPATCH_FLOATING_TYPES_FOR_ALLREDUCE(ctype, in, ...)                                     \
  [&] {                                                                                           \
    const auto& scalar_type = (in).scalar_type();                                                 \
    switch (scalar_type) {                                                                        \
      case at::ScalarType::Float: {                                                               \
        using ctype = float;                                                                      \
        return __VA_ARGS__();                                                                     \
      }                                                                                           \
      /* Requires nv_half to be defined somewhere */                                              \
      case at::ScalarType::Half: {                                                                \
        using ctype = half;                                                                       \
        return __VA_ARGS__();                                                                     \
      }                                                                                           \
      /* Requires nv_bfloat16 to be defined somewhere */                                          \
      case at::ScalarType::BFloat16: {                                                            \
        using ctype = __nv_bfloat16;                                                              \
        return __VA_ARGS__();                                                                     \
      }                                                                                           \
      default:                                                                                    \
        TORCH_CHECK(false,                                                                        \
                    "Unsupported dtype in DISPATCH_FLOATING_TYPES_FOR_ALLREDUCE: ", scalar_type); \
    }                                                                                             \
  }()

void trtllm_lamport_initialize(at::Tensor& buffer) {
  DISPATCH_ALLREDUCE_DTYPE(buffer.scalar_type(), c_type, {
    cudaStream_t raw_stream = at::cuda::getCurrentCUDAStream().stream();
    auto status = lamportInitialize<c_type>(static_cast<void*>(buffer.data_ptr()),
                                            static_cast<size_t>(buffer.numel()), raw_stream);
    TORCH_CHECK(status == cudaSuccess, "lamportInitialize failed with error code " +
                                           std::string(cudaGetErrorString(status)));
  });
}

// TODO(yingyi): update params list since 
// message_size = max_token_num * hidden_dim;
// in/out/residual_in/residual_out/norm_out.. sized of message_size
// refer to cpp/tests/unit_tests/kernels/allReduce/allReduceFusionTest.cu:L268
void trtllm_custom_all_reduce(at::Tensor& buffer, at::Tensor& in, at::Tensor& out, int64_t tp_size,
                              int64_t tp_rank, int64_t token_num, int64_t fusion_op_code,
                              int64_t strategy_code, int64_t config_code, int64_t elts_total,
                              bool launch_with_pdl, std::optional<at::Tensor> bias,
                              std::optional<at::Tensor> residual,
                              std::optional<int64_t> hidden_size, std::optional<at::Tensor> weight,
                              std::optional<at::Tensor> weight_pre_residual_norm,
                              std::optional<float> eps,
                              std::optional<at::Tensor> intermediate_buffer,
                              std::optional<at::Tensor> lamport_peer_comm_buffer_ptrs) {
  AllReduceFusionOp fusion_op = static_cast<AllReduceFusionOp>(fusion_op_code);
  const c10::cuda::OptionalCUDAGuard device_guard(buffer.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  // TODO(zihao): review dispatch type: support fp16, bf16, fp32
  DISPATCH_FLOATING_TYPES_FOR_ALLREDUCE(c_type, in, [&] {
    int64_t hidden_size_val = hidden_size.has_value() ? hidden_size.value() : 0;

    // TODO(yingyi): might move all initialization into deserialization?
    // TODO(yingyi): remove type template here (used to check if lamport is supported)
    auto params =
        AllReduceParams<c_type>::deserialize(static_cast<int64_t*>(buffer.data_ptr()), tp_size,
                                             tp_rank, token_num, hidden_size_val, fusion_op);

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
    if (lamport_peer_comm_buffer_ptrs.has_value()) {
      auto* src = static_cast<void**>(lamport_peer_comm_buffer_ptrs.value().data_ptr());
      for (int i = 0; i < MAX_RANKS_PER_NODE * 3; ++i) {
        params.fusion_params.lamport_peer_comm_buffer_ptrs[i] = src[i];
      }
    }

    auto strategy = static_cast<AllReduceStrategyType>(strategy_code);
    auto config = static_cast<AllReduceStrategyConfig>(config_code);

    auto status = customAllReduce(params, strategy, config, fusion_op, launch_with_pdl, stream);
    TORCH_CHECK(status == cudaSuccess, "customAllReduce failed with error code " +
                                           std::string(cudaGetErrorString(status)));
  });
}
