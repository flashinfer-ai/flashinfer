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
#include <cstdint>
#include <string>

#include "flashinfer/comm/trtllm_allreduce.cuh"
#include "tvm_ffi_utils.h"

using namespace flashinfer::trtllm_allreduce;
using tvm::ffi::Optional;

#define DISPATCH_ALLREDUCE_DTYPE(dtype, C_TYPE, ...)                                             \
  [&]() {                                                                                        \
    if (dtype == dl_float32) {                                                                   \
      using C_TYPE = float;                                                                      \
      __VA_ARGS__                                                                                \
    } else if (dtype == dl_float16) {                                                            \
      using C_TYPE = half;                                                                       \
      __VA_ARGS__                                                                                \
    } else if (dtype == dl_bfloat16) {                                                           \
      using C_TYPE = __nv_bfloat16;                                                              \
      __VA_ARGS__                                                                                \
    } else {                                                                                     \
      TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported DType for custom op dispatch."; \
    }                                                                                            \
  }()

#define DISPATCH_FLOATING_TYPES_FOR_ALLREDUCE(dtype, c_type, ...)             \
  [&] {                                                                       \
    switch (encode_dlpack_dtype(dtype)) {                                     \
      case float32_code: {                                                    \
        using c_type = float;                                                 \
        return __VA_ARGS__();                                                 \
      }                                                                       \
      /* Requires nv_half to be defined somewhere */                          \
      case float16_code: {                                                    \
        using c_type = half;                                                  \
        return __VA_ARGS__();                                                 \
      }                                                                       \
      /* Requires nv_bfloat16 to be defined somewhere */                      \
      case bfloat16_code: {                                                   \
        using c_type = __nv_bfloat16;                                         \
        return __VA_ARGS__();                                                 \
      }                                                                       \
      default:                                                                \
        TVM_FFI_LOG_AND_THROW(NotImplementedError)                            \
            << "Unsupported dtype in DISPATCH_FLOATING_TYPES_FOR_ALLREDUCE."; \
    }                                                                         \
  }()

void trtllm_lamport_initialize(int64_t buffer_ptr, int64_t size, DLDataType dtype) {
  DISPATCH_ALLREDUCE_DTYPE(dtype, c_type, {
    cudaStream_t raw_stream = get_current_stream();
    auto status = lamportInitialize<c_type>(reinterpret_cast<void*>(buffer_ptr),
                                            static_cast<size_t>(size), raw_stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "lamportInitialize failed with error code " << cudaGetErrorString(status);
  });
}

void trtllm_lamport_initialize_all(int64_t buffer_0_ptr, int64_t buffer_1_ptr, int64_t buffer_2_ptr,
                                   int64_t size, DLDataType dtype) {
  DISPATCH_ALLREDUCE_DTYPE(dtype, c_type, {
    cudaStream_t raw_stream = get_current_stream();
    auto status = lamportInitializeAll<c_type>(
        reinterpret_cast<void*>(buffer_0_ptr), reinterpret_cast<void*>(buffer_1_ptr),
        reinterpret_cast<void*>(buffer_2_ptr), static_cast<size_t>(size), raw_stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "lamportInitializeAll failed with error code " << cudaGetErrorString(status);
  });
}

// refer to cpp/tests/unit_tests/kernels/allReduce/allReduceFusionTest.cu:L268
void trtllm_custom_all_reduce(TensorView in, TensorView out, int64_t tp_size, int64_t tp_rank,
                              int64_t token_num, int64_t fusion_op_code, int64_t strategy_code,
                              int64_t config_code, bool launch_with_pdl, int64_t flag_value,
                              TensorView peer_comm_buffer_ptrs, TensorView peer_barrier_ptrs_in,
                              TensorView peer_barrier_ptrs_out, Optional<TensorView> bias,
                              Optional<TensorView> residual, Optional<TensorView> weight,
                              Optional<TensorView> weight_pre_residual_norm, Optional<double> eps,
                              Optional<TensorView> intermediate_buffer,
                              Optional<TensorView> lamport_peer_comm_buffer_ptrs_0,
                              Optional<TensorView> lamport_peer_comm_buffer_ptrs_1,
                              Optional<TensorView> lamport_peer_comm_buffer_ptrs_2) {
  AllReduceFusionOp fusion_op = static_cast<AllReduceFusionOp>(fusion_op_code);
  ffi::CUDADeviceGuard device_guard(in.device().device_id);
  auto stream = get_stream(in.device());

  // TODO(zihao): review dispatch type - support fp16, bf16 only
  DISPATCH_FLOATING_TYPES_FOR_ALLREDUCE(in.dtype(), c_type, [&] {
    // TODO(yingyi): remove type template here (used to check if lamport is supported)
    int64_t message_size = in.numel();
    int64_t hidden_size = in.numel() / token_num;

    AllReduceParams<c_type> params;
    params.elts_total = message_size;
    params.local_rank = tp_rank;
    params.ranks_per_node = tp_size;
    params.local_input_buffer_ptr = in.data_ptr();
    params.local_output_buffer_ptr = out.data_ptr();
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
          reinterpret_cast<void*>(static_cast<int64_t*>(peer_comm_buffer_ptrs.data_ptr())[i]);
      params.peer_barrier_ptrs_in[i] =
          reinterpret_cast<uint32_t*>(static_cast<int64_t*>(peer_barrier_ptrs_in.data_ptr())[i]);
      params.peer_barrier_ptrs_out[i] =
          reinterpret_cast<uint32_t*>(static_cast<int64_t*>(peer_barrier_ptrs_out.data_ptr())[i]);
    }

    if (lamport_peer_comm_buffer_ptrs_0.has_value()) {
      TVM_FFI_ICHECK(lamport_peer_comm_buffer_ptrs_1.has_value())
          << "lamport_peer_comm_buffer_ptrs_1 is required if lamport_peer_comm_buffer_ptrs_0 is "
             "provided";
      TVM_FFI_ICHECK(lamport_peer_comm_buffer_ptrs_2.has_value())
          << "lamport_peer_comm_buffer_ptrs_2 is required if lamport_peer_comm_buffer_ptrs_0 "
             "is provided";
      for (int i = 0; i < tp_size; ++i) {
        params.fusion_params.lamport_peer_comm_buffer_ptrs[i] = reinterpret_cast<void*>(
            static_cast<int64_t*>(lamport_peer_comm_buffer_ptrs_0.value().data_ptr())[i]);
        params.fusion_params.lamport_peer_comm_buffer_ptrs[i + tp_size] = reinterpret_cast<void*>(
            static_cast<int64_t*>(lamport_peer_comm_buffer_ptrs_1.value().data_ptr())[i]);
        params.fusion_params.lamport_peer_comm_buffer_ptrs[i + tp_size * 2] =
            reinterpret_cast<void*>(
                static_cast<int64_t*>(lamport_peer_comm_buffer_ptrs_2.value().data_ptr())[i]);
      }
    }

    auto strategy = static_cast<AllReduceStrategyType>(strategy_code);
    auto config = static_cast<AllReduceStrategyConfig>(config_code);

    auto status = customAllReduce(params, strategy, config, fusion_op, launch_with_pdl, stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "customAllReduce failed with error code " << cudaGetErrorString(status);
  });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_lamport_initialize, trtllm_lamport_initialize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_lamport_initialize_all, trtllm_lamport_initialize_all);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_custom_all_reduce, trtllm_custom_all_reduce);
