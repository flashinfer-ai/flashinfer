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
#include <cuda_runtime.h>
#include "tvm_ffi_utils.h"

{% set func_name = act_func_name ~ '_and_mul' %}

using namespace flashinfer;

{{ act_func_def }}

void {{ func_name }}(TensorView out, TensorView input, bool enable_pdl) {
  int d = input.size(input.ndim() -1) / 2;
  int64_t num_tokens = input.numel() / input.size(input.ndim() -1);
  dim3 grid(num_tokens);

  cudaSetDevice(out.device().device_id);
  const cudaStream_t stream = get_stream(out.device());
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
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
    TVM_FFI_ICHECK(err == cudaSuccess) << "Failed to launch kernel: " << cudaGetErrorString(err);

    return true;
  });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC({{ func_name }}, {{ func_name }});
"""


def get_act_and_mul_cu_str(act_func_name: str, act_func_def: str) -> str:
    template = jinja2.Template(activation_templ)
    return template.render(act_func_name=act_func_name, act_func_def=act_func_def)


silu_def_cu_str = r"""
__device__ __forceinline__ float silu(const float& val) {
  return val / (1.0f + __expf(-val));
}
"""

gelu_def_cu_str = r"""
__device__ __forceinline__ float gelu(const float& val) {
  constexpr float kAlpha = M_SQRT1_2;
  return val * 0.5f * (1.0f + ::erf(val * kAlpha));
}
"""

gelu_def_tanh_cu_str = r"""
__device__ __forceinline__ float gelu_tanh(const float& val) {
  const float cdf =
      0.5f * (1.0f + math::tanh((0.7978845608028654f * (val + 0.044715f * val * val * val))));
  return val * cdf;
}
"""

act_func_def_str = {
    "silu": silu_def_cu_str,
    "gelu": gelu_def_cu_str,
    "gelu_tanh": gelu_def_tanh_cu_str,
}


def gen_act_and_mul_module(act_func_name: str) -> JitSpec:
    act_func_def = act_func_def_str[act_func_name]
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
