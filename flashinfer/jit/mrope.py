# SPDX-License-Identifier: Apache-2.0
"""JIT/AOT build for cached multimodal rotary embedding."""

from . import env as jit_env
from .core import JitSpec, gen_jit_spec


def gen_mrope_module() -> JitSpec:
    # Preserve separate cache-type products before the explicit round-to-nearest FMA.
    return gen_jit_spec(
        "mrope",
        [
            jit_env.FLASHINFER_CSRC_DIR / "mrope.cu",
            jit_env.FLASHINFER_CSRC_DIR / "mrope_jit_binding.cu",
        ],
        extra_cuda_cflags=["--fmad=false"],
        use_fast_math=False,
    )
