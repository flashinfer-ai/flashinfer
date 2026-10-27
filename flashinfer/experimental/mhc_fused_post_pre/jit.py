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

# JIT specification of the fused mHC post -> pre (+ input RMSNorm) SM103 kernels.

from pathlib import Path

from ...jit.core import JitSpec, gen_jit_spec, sm103a_nvcc_flags

_CSRC_DIR = Path(__file__).resolve().parent / "csrc"


def gen_mhc_fused_post_pre_module() -> JitSpec:
    """Fused mHC post -> pre (+ input RMSNorm) kernels for H=4096, HC=4 on SM103.

    One module for every supported token count: the token count is a runtime kernel
    argument, and the regime kernels (decode, stream, short-stream, crossover,
    long-stream) are all compiled into it.
    """
    return gen_jit_spec(
        "mhc_fused_post_pre_sm103",
        [
            _CSRC_DIR / "mhc_fused_post_pre.cu",
            _CSRC_DIR / "mhc_fused_post_pre_decode_stream.cu",
            _CSRC_DIR / "mhc_fused_post_pre_short_stream.cu",
            _CSRC_DIR / "mhc_fused_post_pre_crossover.cu",
            _CSRC_DIR / "mhc_fused_post_pre_long_stream.cu",
        ],
        # Thread-block clusters of 16 CTAs, TMA, setmaxnreg and packed f32x2 FMA:
        # architecture-specific SM103 code only.
        extra_cuda_cflags=list(sm103a_nvcc_flags),
        # Numerics-preserving build: -use_fast_math would turn the accurate expf into
        # __expf, flush denormals in every fp32 add / f32x2 FMA and relax '/' to an
        # approximate division. These kernels must round like the unfused path.
        use_fast_math=False,
    )


__all__ = ["gen_mhc_fused_post_pre_module"]
