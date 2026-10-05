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
from typing import Dict, Literal, Tuple

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags
from .cpp_ext import is_cuda_version_at_least

MiniMaxH3OutProjTarget = Literal["sm100a", "sm103a"]

# Exact-architecture tcgen05 payloads: one generated source per target, compiled with that
# target's flags only (no fatbin; the 2-CTA MMA / TMEM code is not portable across 10.x minors).
_TARGETS = {
    "sm100a": ("cake_minimax_h3_out_proj_sm100a.cu", sm100a_nvcc_flags),
    "sm103a": ("cake_minimax_h3_out_proj_sm103a.cu", sm103a_nvcc_flags),
}
_CAPABILITY_TO_TARGET: Dict[Tuple[int, int], MiniMaxH3OutProjTarget] = {
    (10, 0): "sm100a",
    (10, 3): "sm103a",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _minimax_h3_out_proj_cuda_source(target: MiniMaxH3OutProjTarget) -> Path:
    source_name = _TARGETS[target][0]
    packaged = jit_env.FLASHINFER_CSRC_DIR / source_name
    if packaged.is_file():
        return packaged
    source_tree = _repo_root() / "csrc" / source_name
    if source_tree.is_file():
        return source_tree
    raise FileNotFoundError(
        f"MiniMax-H3 out-proj CUDA source not found. Checked:\n  - {packaged}\n  - {source_tree}"
    )


def minimax_h3_out_proj_target(
    capability: Tuple[int, int],
) -> MiniMaxH3OutProjTarget:
    """Map an exact compute capability to the generated source that serves it."""

    try:
        return _CAPABILITY_TO_TARGET[(int(capability[0]), int(capability[1]))]
    except KeyError:
        raise RuntimeError(
            "MiniMax-H3 out-proj requires exact compute capability 10.0 (B200/GB200) or "
            f"10.3 (B300/GB300), got {capability[0]}.{capability[1]}"
        ) from None


def gen_minimax_h3_out_proj_module(target: MiniMaxH3OutProjTarget) -> JitSpec:
    """JIT spec for the MiniMax-H3 direct-layout attention output projection + indexed gate +
    residual operator (BF16, MXFP8 and NVFP4 variants) on one exact Blackwell target.

    The module exposes ``minimax_h3_out_proj``, ``minimax_h3_out_proj_mxfp8`` and
    ``minimax_h3_out_proj_nvfp4``.  The device code is tcgen05 + TMEM + 2-CTA MMA and does not
    compile for SM90 or SM120.
    """

    if target not in _TARGETS:
        raise ValueError(f"unsupported MiniMax-H3 out-proj target: {target!r}")
    if not is_cuda_version_at_least("12.9"):
        raise RuntimeError("SM100a/SM103a compilation requires CUDA 12.9 or newer")
    source = _minimax_h3_out_proj_cuda_source(target)
    return gen_jit_spec(
        f"minimax_h3_out_proj_{target}_v1",
        [source],
        extra_cuda_cflags=list(_TARGETS[target][1]),
    )


__all__ = [
    "MiniMaxH3OutProjTarget",
    "gen_minimax_h3_out_proj_module",
    "minimax_h3_out_proj_target",
]
