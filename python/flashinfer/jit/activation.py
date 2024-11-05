"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os

import jinja2

from .env import FLASHINFER_GEN_SRC_DIR
from .utils import write_if_different

activation_templ = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <flashinfer/activation.cuh>
#include <torch/extension.h>
#include "pytorch_extension_utils.h"

{% set func_name = act_func_name ~ '_and_mul' %}

using namespace flashinfer;

{{ act_func_def }}

void {{ func_name }}(torch::Tensor& out, torch::Tensor& input) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    dim3 block(std::min(d / vec_size, 1024U));
    flashinfer::activation::act_and_mul_kernel<c_type, {{ act_func_name }}>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()),
                                     static_cast<c_type*>(input.data_ptr()), d);

    return true;
  });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("{{ func_name }}", &{{ func_name }}, "Fused {{ act_func_name }} and Mul");
}
"""


def get_act_and_mul_cu_str(act_func_name: str, act_func_def: str) -> str:
    template = jinja2.Template(activation_templ)
    return template.render(act_func_name=act_func_name, act_func_def=act_func_def)


def gen_act_and_mul_cu(act_func_name: str, act_func_def: str) -> None:
    gen_directory = FLASHINFER_GEN_SRC_DIR
    if not os.path.exists(gen_directory):
        os.makedirs(gen_directory)
    write_if_different(
        gen_directory / f"{act_func_name}_and_mul.cu",
        get_act_and_mul_cu_str(act_func_name, act_func_def),
    )
