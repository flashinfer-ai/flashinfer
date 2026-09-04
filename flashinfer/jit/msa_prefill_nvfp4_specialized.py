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

MSAPrefillNVFP4Target = Literal["sm100a", "sm103a"]

_NVCC_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}

# ``gen_jit_spec`` adds ``-use_fast_math`` unconditionally, which turns on
# ``--prec-div=false`` and ``--ftz=true``.  Two of those components are switched
# back on here and one is deliberately NOT (later nvcc options win).
#
# ``--prec-div=true``: the epilogue divides by the softmax denominator in FP32
# once per row, and this route must stay a drop-in numerical peer of the BF16
# path it specializes.  Correctly rounding one divide per row costs 112 SASS
# instructions out of 6,849 and no measurable time.
#
# ``--prec-sqrt=true`` and ``--fmad=true``: both compile to exactly the same
# SASS as without them (this translation unit has no square root, and fmad=true
# is nvcc's default).  They are stated so the whole floating-point contract of
# the unit is readable in one place, at zero cost.
#
# ``--ftz=false`` is NOT set, and that is a decision, not an omission.  Setting
# it costs 2,096 SASS instructions -- it turns 6,849 into 8,945, a 31% larger
# kernel -- because every FADD/FFMA/FMUL/FMNMX in the softmax loses its .FTZ
# form and picks up a denormal fix-up path.  It buys nothing here, for a reason
# specific to this kernel: the range guard in
# ``csrc/msa_prefill_nvfp4_specialized.cu`` refuses to let a tile finish with a
# denominator below 2**-120, six binades above FP32's smallest normal, and
# replays it against a data-derived origin instead.  A probability small enough
# for flush-to-zero to be observable would have to be ~2**-130 while its own row
# sum is >= 2**-120, i.e. it contributes less than a thousandth of one ULP of
# that sum -- and it is rounded to BF16 before it is used regardless.  Keeping
# flush-to-zero also means this route is compiled the way it was benchmarked.
#
# The one deliberately approximate operation is the softmax exponential:
# ``-use_fast_math`` maps ``exp2f`` to ``ex2.approx.f32``, which is the same
# instruction the stock MSA prefill path uses, and none of the flags below
# change that.
_IEEE_FP32_FLAGS = ["--prec-div=true", "--prec-sqrt=true", "--fmad=true"]


def get_msa_prefill_nvfp4_specialized_uri(target: MSAPrefillNVFP4Target) -> str:
    """Return the JIT/AOT key for one target of the specialized NVFP4 prefill."""

    if target not in _NVCC_FLAGS:
        raise ValueError(
            f"unsupported target for specialized NVFP4 MSA prefill: {target}"
        )
    return f"msa_prefill_nvfp4_specialized_{target}"


@functools.cache
def gen_msa_prefill_nvfp4_specialized_module(target: MSAPrefillNVFP4Target) -> JitSpec:
    """Generate the JIT spec for one target of the specialized NVFP4 prefill."""

    return gen_jit_spec(
        get_msa_prefill_nvfp4_specialized_uri(target),
        [jit_env.FLASHINFER_CSRC_DIR / "msa_prefill_nvfp4_specialized.cu"],
        extra_cuda_cflags=_NVCC_FLAGS[target] + _IEEE_FP32_FLAGS,
    )


@functools.cache
def load_msa_prefill_nvfp4_specialized_module(target: MSAPrefillNVFP4Target):
    """Build or load the physical module for one target."""

    return gen_msa_prefill_nvfp4_specialized_module(target).build_and_load()


__all__ = [
    "MSAPrefillNVFP4Target",
    "gen_msa_prefill_nvfp4_specialized_module",
    "get_msa_prefill_nvfp4_specialized_uri",
    "load_msa_prefill_nvfp4_specialized_module",
]
