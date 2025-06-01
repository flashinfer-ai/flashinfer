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

void trtllm_lamport_initialize(int64_t buffer_ptr, int64_t size, at::ScalarType dtype) {
  DISPATCH_ALLREDUCE_DTYPE(dtype, c_type, {
    cudaStream_t raw_stream = at::cuda::getCurrentCUDAStream().stream();
    auto status = lamportInitialize<c_type>(reinterpret_cast<void*>(buffer_ptr),
                                            static_cast<size_t>(size), raw_stream);
    TORCH_CHECK(status == cudaSuccess, "lamportInitialize failed with error code " +
                                           std::string(cudaGetErrorString(status)));
  });
}

void trtllm_lamport_initialize_all(int64_t buffer_0_ptr, int64_t buffer_1_ptr, int64_t buffer_2_ptr,
                                   int64_t size, at::ScalarType dtype) {
  DISPATCH_ALLREDUCE_DTYPE(dtype, c_type, {
    cudaStream_t raw_stream = at::cuda::getCurrentCUDAStream().stream();
    auto status = lamportInitializeAll<c_type>(
        reinterpret_cast<void*>(buffer_0_ptr), reinterpret_cast<void*>(buffer_1_ptr),
        reinterpret_cast<void*>(buffer_2_ptr), static_cast<size_t>(size), raw_stream);
    TORCH_CHECK(status == cudaSuccess, "lamportInitializeAll failed with error code " +
                                           std::string(cudaGetErrorString(status)));
  });
}

// refer to cpp/tests/unit_tests/kernels/allReduce/allReduceFusionTest.cu:L268
void trtllm_custom_all_reduce(at::Tensor& in, at::Tensor& out, int64_t tp_size, int64_t tp_rank,
                              int64_t token_num, int64_t fusion_op_code, int64_t strategy_code,
                              int64_t config_code, bool launch_with_pdl, int64_t flag_value,
                              at::Tensor peer_comm_buffer_ptrs,  // std::vector<void*>
                              at::Tensor peer_barrier_ptrs_in,   // std::vector<void*>
                              at::Tensor peer_barrier_ptrs_out,  // std::vector<void*>
                              std::optional<at::Tensor> bias, std::optional<at::Tensor> residual,
                              std::optional<at::Tensor> weight,
                              std::optional<at::Tensor> weight_pre_residual_norm,
                              std::optional<double> eps,
                              std::optional<at::Tensor> intermediate_buffer,
                              std::optional<at::Tensor> lamport_peer_comm_buffer_ptrs_0,
                              std::optional<at::Tensor> lamport_peer_comm_buffer_ptrs_1,
                              std::optional<at::Tensor> lamport_peer_comm_buffer_ptrs_2) {
  AllReduceFusionOp fusion_op = static_cast<AllReduceFusionOp>(fusion_op_code);
  const c10::cuda::OptionalCUDAGuard device_guard(in.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  // TODO(zihao): review dispatch type - support fp16, bf16 only
  DISPATCH_FLOATING_TYPES_FOR_ALLREDUCE(c_type, in, [&] {
    // TODO(yingyi): remove type template here (used to check if lamport is supported)
    int64_t message_size = in.numel();
    int64_t hidden_size = in.numel() / token_num;

    AllReduceParams<c_type> params;
    params.elts_total = message_size;
    params.local_rank = tp_rank;
    params.ranks_per_node = tp_size;
    params.local_input_buffer_ptr = in.data_ptr();
    params.local_output_buffer_ptr = out.data_ptr();

    // NOTE(yingyi): review the barrier flag
    // int flag_offset;
    // if (fusion_op == AllReduceFusionOp::RESIDUAL_RMS_NORM &&
    //     is_lamport_supported<c_type>(token_num, hidden_size)) {
    //   flag_offset = 0;
    // } else {
    //   flag_offset = 1;
    // }

    // auto const flag_ptr = reinterpret_cast<int64_t*>(flag_buffer_ptr) + NUM_POINTERS_PER_RANK *
    // tp_size + flag_offset; *flag_ptr += 1; uint32_t flag_value = *flag_ptr;
    params.barrier_flag = flag_value;

    // add fusion params
    params.fusion_params.bias_buffer = bias.has_value() ? bias.value().data_ptr() : nullptr;
    params.fusion_params.residual_buffer =
        residual.has_value() ? residual.value().data_ptr() : nullptr;
    params.fusion_params.hidden_size = hidden_size;
    params.fusion_params.weight_buffer = weight.has_value() ? weight.value().data_ptr() : nullptr;
    params.fusion_params.weight_buffer_pre_residual_norm =
        weight_pre_residual_norm.has_value() ? weight_pre_residual_norm.value().data_ptr()
                                             : nullptr;
    params.fusion_params.eps = eps.has_value() ? eps.value() : 1e-5f;
    params.fusion_params.intermediate_buffer =
        intermediate_buffer.has_value() ? intermediate_buffer.value().data_ptr() : nullptr;

    // add ipc buffer pointers
    for (int i = 0; i < tp_size; ++i) {
      params.peer_comm_buffer_ptrs[i] =
          reinterpret_cast<void*>(peer_comm_buffer_ptrs.data_ptr<int64_t>()[i]);
      params.peer_barrier_ptrs_in[i] =
          reinterpret_cast<uint32_t*>(peer_barrier_ptrs_in.data_ptr<int64_t>()[i]);
      params.peer_barrier_ptrs_out[i] =
          reinterpret_cast<uint32_t*>(peer_barrier_ptrs_out.data_ptr<int64_t>()[i]);
    }

    if (lamport_peer_comm_buffer_ptrs_0.has_value()) {
      TORCH_CHECK(lamport_peer_comm_buffer_ptrs_1.has_value(),
                  "lamport_peer_comm_buffer_ptrs_1 is required if lamport_peer_comm_buffer_ptrs_0 "
                  "is provided");
      TORCH_CHECK(lamport_peer_comm_buffer_ptrs_2.has_value(),
                  "lamport_peer_comm_buffer_ptrs_2 is required if lamport_peer_comm_buffer_ptrs_0 "
                  "is provided");
      for (int i = 0; i < tp_size; ++i) {
        params.fusion_params.lamport_peer_comm_buffer_ptrs[i] =
            reinterpret_cast<void*>(lamport_peer_comm_buffer_ptrs_0.value().data_ptr<int64_t>()[i]);
        params.fusion_params.lamport_peer_comm_buffer_ptrs[i + tp_size] =
            reinterpret_cast<void*>(lamport_peer_comm_buffer_ptrs_1.value().data_ptr<int64_t>()[i]);
        params.fusion_params.lamport_peer_comm_buffer_ptrs[i + tp_size * 2] =
            reinterpret_cast<void*>(lamport_peer_comm_buffer_ptrs_2.value().data_ptr<int64_t>()[i]);
      }
    }

    auto strategy = static_cast<AllReduceStrategyType>(strategy_code);
    auto config = static_cast<AllReduceStrategyConfig>(config_code);

    auto status = customAllReduce(params, strategy, config, fusion_op, launch_with_pdl, stream);
    TORCH_CHECK(status == cudaSuccess, "customAllReduce failed with error code " +
                                           std::string(cudaGetErrorString(status)));
  });
}

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("trtllm_lamport_initialize", &trtllm_lamport_initialize);
  m.def("trtllm_lamport_initialize_all", &trtllm_lamport_initialize_all);
  m.def("trtllm_custom_all_reduce", &trtllm_custom_all_reduce);
}
