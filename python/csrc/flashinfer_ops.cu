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
#include <torch/extension.h>

#include "flashinfer_ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("append_paged_kv_cache", &append_paged_kv_cache, "Append paged KV-Cache operator");
  m.def("merge_state", &merge_state, "Merge two self-attention states");
  m.def("merge_state_in_place", &merge_state_in_place,
        "Merge another self-attention state in-place.");
  m.def("merge_states", &merge_states, "Merge multiple self-attention states");
  m.def("sampling_from_probs", &sampling_from_probs, "Sample from probabilities");
  m.def("top_k_sampling_from_probs", &top_k_sampling_from_probs,
        "Top-k sampling from probabilities");
  m.def("min_p_sampling_from_probs", &min_p_sampling_from_probs,
        "Min-p sampling from probabilities");
  m.def("top_p_sampling_from_probs", &top_p_sampling_from_probs,
        "Top-p sampling from probabilities");
  m.def("top_k_top_p_sampling_from_probs", &top_k_top_p_sampling_from_probs,
        "Top-k and top-p sampling from probabilities");
  m.def("top_k_renorm_prob", &top_k_renorm_prob, "Renormalize probabilities by top-k mask");
  m.def("top_p_renorm_prob", &top_p_renorm_prob, "Renormalize probabilities by top-p mask");
  m.def("top_k_mask_logits", &top_k_mask_logits, "Mask logits by top-k mask");
  m.def("chain_speculative_sampling", &chain_speculative_sampling,
        "Speculative sampling from sequence of probabilities");
  m.def("rmsnorm", &rmsnorm, "Root mean square normalization");
  m.def("fused_add_rmsnorm", &fused_add_rmsnorm, "Fused add root mean square normalization");
  m.def("silu_and_mul", &silu_and_mul, "Fused SiLU and Mul");
  m.def("gelu_tanh_and_mul", &gelu_tanh_and_mul, "Fused GeLU Tanh and Mul");
  m.def("apply_rope_inplace", &apply_rope_inplace, "Apply RoPE in-place");
  m.def("apply_llama31_rope_inplace", &apply_llama31_rope_inplace,
        "Apply Llama 3.1 style RoPE in-place");
  m.def("apply_rope", &apply_rope, "Apply RoPE");
  m.def("apply_llama31_rope", &apply_llama31_rope, "Apply Llama 3.1 style RoPE");
  m.def("packbits", &packbits, "GPU packbits operator");
  m.def("segment_packbits", &segment_packbits, "GPU segment packbits operator");
  py::class_<CutlassSegmentGEMMPyTorchWrapper>(m, "CutlassSegmentGEMMPyTorchWrapper")
      .def(py::init<torch::Tensor>())
      .def("register_workspace", &CutlassSegmentGEMMPyTorchWrapper::RegisterWorkspaceBuffer)
      .def("forward", &CutlassSegmentGEMMPyTorchWrapper::Forward);
}
