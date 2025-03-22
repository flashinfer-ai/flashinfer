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
import os
import platform

import torch

# Re-export
from .activation import gen_act_and_mul_module as gen_act_and_mul_module
from .activation import get_act_and_mul_cu_str as get_act_and_mul_cu_str
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
from .attention import gen_pod_module as gen_pod_module
from .attention import gen_single_decode_module as gen_single_decode_module
from .attention import gen_single_prefill_module as gen_single_prefill_module
from .attention import get_batch_decode_mla_uri as get_batch_decode_mla_uri
from .attention import get_batch_decode_uri as get_batch_decode_uri
from .attention import get_batch_mla_uri as get_batch_mla_uri
from .attention import get_batch_prefill_uri as get_batch_prefill_uri
from .attention import get_pod_uri as get_pod_uri
from .attention import get_single_decode_uri as get_single_decode_uri
from .attention import get_single_prefill_uri as get_single_prefill_uri
from .core import clear_cache_dir, load_cuda_ops  # noqa: F401
from .env import *
from .utils import parallel_load_modules as parallel_load_modules

if platform.system == "Windows":
    cuda_path = None
    if os.environ.get("CUDA_LIB_PATH"):
        cuda_path = os.environ.get("CUDA_LIB_PATH")
    elif os.environ.get("CUDA_HOME"):
        cuda_path = os.environ.get("CUDA_HOME")
    elif os.environ.get("CUDA_ROOT"):
        cuda_path = os.environ.get("CUDA_ROOT")
    elif os.environ.get("CUDA_PATH"):
        cuda_path = os.environ.get("CUDA_PATH")
    else:
        cuda_path = f"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v{torch.version.cuda}"

    if cuda_path and os.path.exists(cuda_path):
        cudart_version = torch.version.cuda.split(".")[0]
        if cudart_version < "12":
            cudart_version += "0"
        ctypes.CDLL(
            os.path.join(cuda_path, "bin", f"cudart64_{cudart_version}.dll"),
            mode=ctypes.RTLD_GLOBAL,
        )
    else:
        raise ValueError(
            "CUDA_LIB_PATH is not set. "
            "CUDA_LIB_PATH need to be set with the absolute path "
            "to CUDA root folder on Windows (for example, set "
            "CUDA_LIB_PATH=C:\\CUDA\\v12.4)"
        )
else:
    cuda_lib_path = os.environ.get(
        "CUDA_LIB_PATH", "/usr/local/cuda/targets/x86_64-linux/lib/"
    )
    if os.path.exists(f"{cuda_lib_path}/libcudart.so.12"):
        ctypes.CDLL(f"{cuda_lib_path}/libcudart.so.12", mode=ctypes.RTLD_GLOBAL)


try:
    from .. import flashinfer_kernels  # noqa: F401

    load_sm90 = False
    for device_index in range(torch.cuda.device_count()):
        if torch.cuda.get_device_properties(device_index).major > 8:
            load_sm90 = True
            break
    if load_sm90:
        from .. import flashinfer_kernels_sm90

    from .aot_config import prebuilt_ops_uri as prebuilt_ops_uri

    has_prebuilt_ops = True
except ImportError as e:
    if "undefined symbol" in str(e):
        raise ImportError("Loading prebuilt ops failed.") from e

    from .core import logger

    logger.info("Prebuilt kernels not found, using JIT backend")
    prebuilt_ops_uri = {}
    has_prebuilt_ops = False
