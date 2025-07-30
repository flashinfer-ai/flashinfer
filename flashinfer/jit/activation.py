# SPDX - FileCopyrightText : 2023 - 2025 Flashinfer team
# SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
#
# SPDX - License - Identifier : Apache 2.0

import os

import jinja2

from .core import JitSpec, check_hip_availability, gen_jit_spec
from .env import FLASHINFER_GEN_SRC_DIR
from .utils import write_if_different

if check_hip_availability():
    activation_templ = r"""
  #include <gpu_iface/platform.h>
  #include <flashinfer/attention/generic/activation.hip.h>
  #include "pytorch_extension_utils.h"
  #include <hip/hip_runtime.h>

  {% set func_name = act_func_name ~ '_and_mul' %}

  using namespace flashinfer;

  {{ act_func_def }}

  void {{ func_name }}(at::Tensor& out, at::Tensor& input, bool enable_pdl) {
    int d = input.size(-1) / 2;
    int64_t num_tokens = input.numel() / input.size(-1);
    dim3 grid(num_tokens);

    const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(out.device());
    auto stream = at::hip::getCurrentHIPStream();
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
      uint32_t vec_size = 16 / sizeof(c_type);
      uint64_t gridDim = num_tokens;
      uint64_t blockDim = std::min(d / vec_size, 1024U);
      uint64_t dynamicSmemBytes = 0;
      hipStream_t stream = stream;

      auto kernel = flashinfer::activation::act_and_mul_kernel<c_type, {{ act_func_name }}>;

      flashinfer::activation::act_and_mul_kernel<c_type, {{ act_func_name }}><<<<gridDim, blockDim, dynamicSmemBytes, stream>>>(static_cast<c_type*>(out.data_ptr()),
                        static_cast<c_type*>(input.data_ptr()), d);

      hipError_t err = hipGetLastError();
      TORCH_CHECK(err == hipSuccess, "Failed to launch kernel: ", hipGetErrorString(err));

      return true;
    });
  }

  TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
    m.def("{{ func_name }}", {{ func_name }});
  }
  """

else:
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


def gen_act_and_mul_module(act_func_name: str, act_func_def: str) -> None:
    gen_directory = FLASHINFER_GEN_SRC_DIR
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
