"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

JIT spec of the kernels of the Cake FMHA routes ``decode_super_nvfp4`` and
``context_super_nvfp4`` (``flashinfer.cake_fmha_super_nvfp4``).
"""

import functools

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm100a_nvcc_flags


@functools.cache
def gen_cake_fmha_super_nvfp4_module() -> JitSpec:
    """Generate the JIT spec of the Nemotron 3.5 Super NVFP4-KV attention kernels."""

    csrc_dir = jit_env.FLASHINFER_CSRC_DIR / "cake_fmha_super_nvfp4"
    sources = ("binding.cu", "decode_mtp.cu", "decode_base.cu", "prefill.cu")
    # The kernels use tcgen05 and are tuned for sm_100a only, so the module is
    # built for that target whatever else the compilation context lists.
    return gen_jit_spec(
        "cake_fmha_super_nvfp4_sm100a",
        [csrc_dir / source for source in sources],
        extra_cuda_cflags=sm100a_nvcc_flags + ["--expt-relaxed-constexpr"],
    )


@functools.cache
def load_cake_fmha_super_nvfp4_module():
    """Build or load the module."""

    return gen_cake_fmha_super_nvfp4_module().build_and_load()


__all__ = [
    "gen_cake_fmha_super_nvfp4_module",
    "load_cake_fmha_super_nvfp4_module",
]
