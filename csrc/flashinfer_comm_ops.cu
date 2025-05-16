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
#include "flashinfer/distributed/comm_ops.h"
#include "pytorch_extension_utils.h"

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);
  m.def("register_graph_buffers", &register_graph_buffers);
  m.def("dispose", &dispose);
  m.def("meta_size", &meta_size);
  m.def("register_buffer", &register_buffer);

  m.def(
      "init_custom_ar(int[] ipc_tensors, Tensor rank_data, "
      "int rank, bool full_nvlink) -> int");
  m.impl("init_custom_ar", torch::kCUDA, &init_custom_ar);

  m.def(
      "all_reduce(int fa, Tensor inp, Tensor! out, int reg_buffer, "
      "int reg_buffer_sz_bytes, int num_ctas) -> ()");
  m.impl("all_reduce", torch::kCUDA, &all_reduce);

  // TODO: add trtllm communication ops
  m.def(
      "allreduce_fusion_op",
      [](int nranks, int rank, int dtype, int size, int hidden_dim, int64_t workspace,
         int64_t allreduce_in, int64_t residual_in, int64_t allreduce_out, int64_t residual_out,
         int64_t norm_out, int64_t quant_out, int64_t scale_out, int64_t rms_gamma, float rms_eps,
         int64_t scale_factor, bool use_oneshot, int layout, int64_t stream, int pattern) {
        trtllm::kernels::ar_fusion::AllReduceFusionParams params;
        params.nranks = nranks;
        params.rank = rank;
        params.dtype = static_cast<DataType>(dtype);
        params.size = size;
        params.hidden_dim = hidden_dim;
        params.workspace = reinterpret_cast<void**>(workspace);
        params.allreduce_in = reinterpret_cast<void*>(allreduce_in);
        params.residual_in = reinterpret_cast<void*>(residual_in);
        params.allreduce_out = reinterpret_cast<void*>(allreduce_out);
        params.residual_out = reinterpret_cast<void*>(residual_out);
        params.norm_out = reinterpret_cast<void*>(norm_out);
        params.quant_out = reinterpret_cast<void*>(quant_out);
        params.scale_out = reinterpret_cast<void*>(scale_out);
        params.rms_gamma = reinterpret_cast<void*>(rms_gamma);
        params.rms_eps = rms_eps;
        params.scale_factor = reinterpret_cast<float*>(scale_factor);
        params.use_oneshot = use_oneshot;
        params.layout = static_cast<trtllm::FP4QuantizationSFLayout>(layout);
        params.stream = reinterpret_cast<cudaStream_t>(stream);
        params.pattern = static_cast<trtllm::kernels::ar_fusion::AllReduceFusionPattern>(pattern);

        trtllm::kernels::ar_fusion::allreduce_fusion_op(params);
      },
      "Performs an all-reduce operation with optional fusion patterns");
  m.impl("allreduce_fusion_op", torch::kCUDA, &allreduce_fusion_op);

  m.def(
      "moereduction_allreduce_fusion_op",
      [](int nranks, int rank, int dtype, int size, int hidden_dim, int64_t workspace,
         int64_t allreduce_in, int64_t residual_in, int64_t residual_out, int64_t norm_out,
         int64_t quant_out, int64_t scale_out, int64_t rms_gamma, float rms_eps,
         int64_t scale_factor, bool use_oneshot, int layout, int64_t stream,
         int64_t moe_reduction_device_num_experts, int64_t moe_reduction_scale_input,
         int64_t moe_reduction_active_experts_token_input, int64_t moe_reduction_token_input) {
        trtllm::kernels::ar_fusion::moe::MoeReductionAllReduceFusionParams params;
        // Base params
        params.nranks = nranks;
        params.rank = rank;
        params.dtype = static_cast<DataType>(dtype);
        params.size = size;
        params.hidden_dim = hidden_dim;
        params.workspace = reinterpret_cast<void**>(workspace);
        params.allreduce_in = reinterpret_cast<void*>(allreduce_in);
        params.residual_in = reinterpret_cast<void*>(residual_in);
        params.residual_out = reinterpret_cast<void*>(residual_out);
        params.norm_out = reinterpret_cast<void*>(norm_out);
        params.quant_out = reinterpret_cast<void*>(quant_out);
        params.scale_out = reinterpret_cast<void*>(scale_out);
        params.rms_gamma = reinterpret_cast<void*>(rms_gamma);
        params.rms_eps = rms_eps;
        params.scale_factor = reinterpret_cast<float*>(scale_factor);
        params.use_oneshot = use_oneshot;
        params.layout = static_cast<trtllm::FP4QuantizationSFLayout>(layout);
        params.stream = reinterpret_cast<cudaStream_t>(stream);

        // MoE specific params
        params.moe_reduction_device_num_experts =
            reinterpret_cast<int*>(moe_reduction_device_num_experts);
        params.moe_reduction_scale_input = reinterpret_cast<float*>(moe_reduction_scale_input);
        params.moe_reduction_active_experts_token_input =
            reinterpret_cast<void*>(moe_reduction_active_experts_token_input);
        params.moe_reduction_token_input = reinterpret_cast<void*>(moe_reduction_token_input);

        trtllm::kernels::ar_fusion::moe::moereduction_allreduce_fusion_op(params);
      },
      "Performs a MoE reduction + all-reduce operation with optional fusion patterns");
  m.impl("moereduction_allreduce_fusion_op", torch::kCUDA, &moereduction_allreduce_fusion_op);
}
