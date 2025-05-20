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

from . import env as jit_env
from .core import JitSpec, gen_jit_spec
from .utils import write_if_different

activation_templ = r"""
#include <flashinfer/activation.cuh>
#include "pytorch_extension_utils.h"
#include <cuda_runtime.h>

{% set func_name = act_func_name ~ '_and_mul' %}

using namespace flashinfer;

{{ act_func_def }}

void {{ func_name }}(at::Tensor& out, at::Tensor& input, bool enable_pdl) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);

  const c10::cuda::OptionalCUDAGuard device_guard(out.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    cudaLaunchConfig_t config;
    config.gridDim = num_tokens;
    config.blockDim = std::min(d / vec_size, 1024U);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
    config.numAttrs = 1;
    config.attrs = attrs;

    auto kernel = flashinfer::activation::act_and_mul_kernel<c_type, {{ act_func_name }}>;

    cudaLaunchKernelEx(&config, kernel, static_cast<c_type*>(out.data_ptr()),
                       static_cast<c_type*>(input.data_ptr()), d);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Failed to launch kernel: ", cudaGetErrorString(err));

    return true;
  });
}

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("{{ func_name }}", {{ func_name }});
}
"""


def get_act_and_mul_cu_str(act_func_name: str, act_func_def: str) -> str:
    template = jinja2.Template(activation_templ)
    return template.render(act_func_name=act_func_name, act_func_def=act_func_def)


def gen_act_and_mul_module(act_func_name: str, act_func_def: str) -> JitSpec:
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR
    os.makedirs(gen_directory, exist_ok=True)
    sources = [gen_directory / f"{act_func_name}_and_mul.cu"]
    write_if_different(
        sources[0],
        get_act_and_mul_cu_str(act_func_name, act_func_def),
    )
    return gen_jit_spec(
        f"{act_func_name}_and_mul",
        sources,
    )
