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

import ctypes
import functools
import os

# Re-export
from . import cubin_loader
from . import env as env
from .activation import gen_act_and_mul_module as gen_act_and_mul_module
from .activation import get_act_and_mul_cu_str as get_act_and_mul_cu_str
from .attention import cudnn_fmha_gen_module as cudnn_fmha_gen_module
from .attention import gen_batch_attention_module as gen_batch_attention_module
from .attention import gen_batch_decode_mla_module as gen_batch_decode_mla_module
from .attention import gen_batch_decode_module as gen_batch_decode_module
from .attention import gen_batch_mla_module as gen_batch_mla_module
from .attention import gen_batch_mla_tvm_binding as gen_batch_mla_tvm_binding
from .attention import gen_batch_prefill_module as gen_batch_prefill_module
from .attention import (
    gen_customize_batch_decode_module as gen_customize_batch_decode_module,
)
from .attention import (
    gen_customize_batch_decode_tvm_binding as gen_customize_batch_decode_tvm_binding,
)
from .attention import (
    gen_customize_batch_prefill_module as gen_customize_batch_prefill_module,
)
from .attention import (
    gen_customize_batch_prefill_tvm_binding as gen_customize_batch_prefill_tvm_binding,
)
from .attention import (
    gen_customize_single_decode_module as gen_customize_single_decode_module,
)
from .attention import (
    gen_customize_single_prefill_module as gen_customize_single_prefill_module,
)
from .attention import gen_fmha_cutlass_sm100a_module as gen_fmha_cutlass_sm100a_module
from .attention import gen_pod_module as gen_pod_module
from .attention import gen_sampling_tvm_binding as gen_sampling_tvm_binding
from .attention import gen_single_decode_module as gen_single_decode_module
from .attention import gen_single_prefill_module as gen_single_prefill_module
from .attention import get_batch_attention_uri as get_batch_attention_uri
from .attention import get_batch_decode_mla_uri as get_batch_decode_mla_uri
from .attention import get_batch_decode_uri as get_batch_decode_uri
from .attention import get_batch_mla_uri as get_batch_mla_uri
from .attention import get_batch_prefill_uri as get_batch_prefill_uri
from .attention import get_pod_uri as get_pod_uri
from .attention import get_single_decode_uri as get_single_decode_uri
from .attention import get_single_prefill_uri as get_single_prefill_uri
from .attention import trtllm_gen_fmha_module as trtllm_gen_fmha_module
from .core import JitSpec as JitSpec
from .core import build_jit_specs as build_jit_specs
from .core import clear_cache_dir as clear_cache_dir
from .core import gen_jit_spec as gen_jit_spec
from .core import sm90a_nvcc_flags as sm90a_nvcc_flags
from .core import sm100a_nvcc_flags as sm100a_nvcc_flags
from .cubin_loader import setup_cubin_loader, setup_metainfo_loader


@functools.cache
def get_cudnn_fmha_gen_module():
    mod = cudnn_fmha_gen_module()
    op = mod.build_and_load()
    setup_cubin_loader(mod.get_library_path())
    return op


cuda_lib_path = os.environ.get(
    "CUDA_LIB_PATH", "/usr/local/cuda/targets/x86_64-linux/lib/"
)
if os.path.exists(f"{cuda_lib_path}/libcudart.so.12"):
    ctypes.CDLL(f"{cuda_lib_path}/libcudart.so.12", mode=ctypes.RTLD_GLOBAL)
