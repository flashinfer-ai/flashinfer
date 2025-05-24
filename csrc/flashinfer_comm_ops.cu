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
  m.def("all_reduce", &all_reduce);
  m.def("allreduce_fusion_op",
        [](int64_t nranks, int64_t rank, int64_t dtype, int64_t size, int64_t hidden_dim,
           int64_t workspace, int64_t allreduce_in, int64_t residual_in, int64_t allreduce_out,
           int64_t residual_out, int64_t norm_out, int64_t quant_out, int64_t scale_out,
           int64_t rms_gamma, double rms_eps, int64_t scale_factor, bool use_oneshot,
           int64_t layout, int64_t stream, int64_t pattern) {
          tensorrt_llm::kernels::ar_fusion::AllReduceFusionParams params;
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
          params.layout = static_cast<tensorrt_llm::FP4QuantizationSFLayout>(layout);
          params.stream = reinterpret_cast<cudaStream_t>(stream);
          params.pattern =
              static_cast<tensorrt_llm::kernels::ar_fusion::AllReduceFusionPattern>(pattern);

          tensorrt_llm::kernels::ar_fusion::allreduce_fusion_op(params);
        });

  m.def("moereduction_allreduce_fusion_op",
        [](int64_t nranks, int64_t rank, int64_t dtype, int64_t size, int64_t hidden_dim,
           int64_t workspace, int64_t allreduce_in, int64_t residual_in, int64_t residual_out,
           int64_t norm_out, int64_t quant_out, int64_t scale_out, int64_t rms_gamma,
           double rms_eps, int64_t scale_factor, int64_t layout, int64_t stream,
           int64_t moe_reduction_device_num_experts, int64_t moe_reduction_scale_input,
           int64_t moe_reduction_active_experts_token_input, int64_t moe_reduction_token_input) {
          tensorrt_llm::kernels::ar_fusion::moe::MoeReductionAllReduceFusionParams params;
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
          params.layout = static_cast<tensorrt_llm::FP4QuantizationSFLayout>(layout);
          params.stream = reinterpret_cast<cudaStream_t>(stream);

          // MoE specific params
          params.moe_reduction_device_num_experts =
              reinterpret_cast<int*>(moe_reduction_device_num_experts);
          params.moe_reduction_scale_input = reinterpret_cast<float*>(moe_reduction_scale_input);
          params.moe_reduction_active_experts_token_input =
              reinterpret_cast<void*>(moe_reduction_active_experts_token_input);
          params.moe_reduction_token_input = reinterpret_cast<void*>(moe_reduction_token_input);

          tensorrt_llm::kernels::ar_fusion::moe::moereduction_allreduce_fusion_op(params);
        });
}
