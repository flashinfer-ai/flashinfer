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

# Re-export
from .activation import gen_act_and_mul_module as gen_act_and_mul_module
from .activation import get_act_and_mul_cu_str as get_act_and_mul_cu_str
from .attention import gen_batch_decode_mla_module as gen_batch_decode_mla_module
from .attention import gen_batch_decode_module as gen_batch_decode_module
from .attention import gen_batch_prefill_module as gen_batch_prefill_module
from .attention import (
    gen_customize_batch_decode_module as gen_customize_batch_decode_module,
)
from .attention import (
    gen_customize_batch_prefill_module as gen_customize_batch_prefill_module,
)
from .attention import (
    gen_customize_single_decode_module as gen_customize_single_decode_module,
)
from .attention import (
    gen_customize_single_prefill_module as gen_customize_single_prefill_module,
)
from .attention import gen_single_decode_module as gen_single_decode_module
from .attention import gen_single_prefill_module as gen_single_prefill_module
from .attention import get_batch_decode_mla_uri as get_batch_decode_mla_uri
from .attention import get_batch_decode_uri as get_batch_decode_uri
from .attention import get_batch_prefill_uri as get_batch_prefill_uri
from .attention import get_single_decode_uri as get_single_decode_uri
from .attention import get_single_prefill_uri as get_single_prefill_uri
from .core import clear_cache_dir, load_cuda_ops  # noqa: F401
from .env import *
from .utils import parallel_load_modules as parallel_load_modules

try:
    from .. import _kernels, _kernels_sm90  # noqa: F401
    from .aot_config import prebuilt_ops_uri as prebuilt_ops_uri

    has_prebuilt_ops = True
except ImportError:
    prebuilt_ops_uri = {}
    has_prebuilt_ops = False
