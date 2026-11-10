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

from pathlib import Path

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm120a_nvcc_flags

_CUDA_SOURCE_NAME = "cake_minimax_h3_sm120_quant_out_proj_sm120a.cu"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _cuda_source() -> Path:
    packaged = jit_env.FLASHINFER_CSRC_DIR / _CUDA_SOURCE_NAME
    if packaged.is_file():
        return packaged
    source_tree = _repo_root() / "csrc" / _CUDA_SOURCE_NAME
    if source_tree.is_file():
        return source_tree
    raise FileNotFoundError(
        "MiniMax-H3 SM120 quantized output projection CUDA source not found. Checked:\n"
        f"  - {packaged}\n  - {source_tree}"
    )


def gen_minimax_h3_sm120_quant_out_proj_module() -> JitSpec:
    """JIT spec for the SM120 (GB202) FP8 / NVFP4 fused MiniMax-H3 output projection kernels.

    The device code uses ``mma.sync`` ``kind::f8f6f4`` / ``kind::mxf4nvf4`` block-scaled MMAs and
    TMA as exposed by ``sm_120a`` (single-CTA persistent tiles, no multicast); it is built for that
    target only.
    """

    return gen_jit_spec(
        "minimax_h3_sm120_quant_out_proj_sm120a_v1",
        [_cuda_source()],
        extra_cuda_cflags=list(sm120a_nvcc_flags),
    )


__all__ = ["gen_minimax_h3_sm120_quant_out_proj_module"]
