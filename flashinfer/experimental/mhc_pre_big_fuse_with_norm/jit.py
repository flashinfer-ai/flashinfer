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

# JIT specification of the SM100-family mHC pre big-fuse + RMSNorm cluster kernel.

from pathlib import Path

from ...jit.core import JitSpec, current_compilation_context, gen_jit_spec


def gen_mhc_pre_big_fuse_with_norm_module() -> JitSpec:
    """SM100-family cluster kernel for mHC pre big-fuse with the fused output RMSNorm.

    The kernel relies on accurate libdevice ``expf``, IEEE fp32 rounding without flush-to-zero
    on add / mul / fma, an IEEE division and explicitly Newton-refined reciprocals, so the
    module is compiled WITHOUT ``-use_fast_math``: fast-math would turn ``expf`` into
    ``__expf``, flush denormals in every fp32 operation and make divisions approximate.
    """
    root = Path(__file__).resolve().parent
    return gen_jit_spec(
        "mhc_pre_big_fuse_with_norm_sm10x",
        [root / "csrc" / "mhc_pre_big_fuse_with_norm_cluster.cu"],
        extra_cuda_cflags=current_compilation_context.get_nvcc_flags_list(
            supported_major_versions=[10],
        ),
        use_fast_math=False,
    )


__all__ = ["gen_mhc_pre_big_fuse_with_norm_module"]
