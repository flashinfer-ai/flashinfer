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

from typing import Literal

from .. import env as jit_env
from ..core import JitSpec, gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags


BlackwellBf16BmmTarget = Literal["sm100a", "sm103a"]

_BLACKWELL_BF16_BMM_NVCC_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}
_BLACKWELL_BF16_BMM_TARGET_MINOR = {
    "sm100a": 0,
    "sm103a": 3,
}


def gen_blackwell_bf16_bmm_module(target: BlackwellBf16BmmTarget) -> JitSpec:
    """Build the frozen CAKE BF16 BMM dispatcher for one Blackwell target."""

    if target not in _BLACKWELL_BF16_BMM_NVCC_FLAGS:
        raise ValueError(f"unsupported CAKE BF16 BMM target: {target}")

    return gen_jit_spec(
        f"blackwell_bf16_bmm_cake_{target}",
        [
            jit_env.FLASHINFER_CSRC_DIR / "blackwell_bf16_bmm.cu",
            jit_env.FLASHINFER_CSRC_DIR / "blackwell_bf16_bmm_kernels.cu",
        ],
        extra_cuda_cflags=_BLACKWELL_BF16_BMM_NVCC_FLAGS[target]
        + [
            f"-DFLASHINFER_BLACKWELL_BF16_BMM_TARGET_MINOR={_BLACKWELL_BF16_BMM_TARGET_MINOR[target]}",
            "--use_fast_math",
        ],
    )


__all__ = ["gen_blackwell_bf16_bmm_module"]
