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
"""

import functools
from typing import Literal

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

MSADecodeNVFP4Target = Literal["sm100a", "sm103a"]

_NVCC_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}

# ``gen_jit_spec`` adds ``-use_fast_math`` unconditionally, which turns on
# ``--prec-div=false`` and ``--ftz=true``.  This kernel divides by the softmax
# denominator and by the P prescale in FP32 and must stay a drop-in numerical
# peer of the path it specializes, so the precision-relevant components are
# switched back on explicitly (later nvcc options win).  ``--fmad=true`` is
# nvcc's default and is restated so the whole floating-point contract of this
# translation unit is readable in one place.
_IEEE_FP32_FLAGS = ["--prec-div=true", "--prec-sqrt=true", "--ftz=false", "--fmad=true"]


def get_msa_decode_nvfp4_specialized_uri(target: MSADecodeNVFP4Target) -> str:
    """Return the JIT/AOT key for one target of the specialized NVFP4 decode."""

    if target not in _NVCC_FLAGS:
        raise ValueError(
            f"unsupported target for specialized NVFP4 MSA decode: {target}"
        )
    return f"msa_decode_nvfp4_specialized_{target}"


@functools.cache
def gen_msa_decode_nvfp4_specialized_module(target: MSADecodeNVFP4Target) -> JitSpec:
    """Generate the JIT spec for one target of the specialized NVFP4 decode."""

    return gen_jit_spec(
        get_msa_decode_nvfp4_specialized_uri(target),
        [jit_env.FLASHINFER_CSRC_DIR / "msa_decode_nvfp4_specialized.cu"],
        extra_cuda_cflags=_NVCC_FLAGS[target] + _IEEE_FP32_FLAGS,
    )


@functools.cache
def load_msa_decode_nvfp4_specialized_module(target: MSADecodeNVFP4Target):
    """Build or load the physical module for one target."""

    return gen_msa_decode_nvfp4_specialized_module(target).build_and_load()


__all__ = [
    "MSADecodeNVFP4Target",
    "gen_msa_decode_nvfp4_specialized_module",
    "get_msa_decode_nvfp4_specialized_uri",
    "load_msa_decode_nvfp4_specialized_module",
]
