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
from pathlib import Path

from . import env as jit_env
from .core import JitSpec, current_compilation_context, gen_jit_spec


def _get_csrc_dir() -> Path:
    """Locate csrc sources in installed and source checkouts."""

    installed = jit_env.FLASHINFER_CSRC_DIR
    if installed.exists():
        return installed

    checkout = Path(__file__).resolve().parents[2] / "csrc"
    if checkout.exists():
        return checkout

    raise FileNotFoundError(
        "vibecuda softmax sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _get_include_dir() -> Path:
    """Locate FlashInfer headers in installed and source checkouts."""

    if jit_env.FLASHINFER_INCLUDE_DIR.exists():
        return jit_env.FLASHINFER_INCLUDE_DIR

    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.exists():
        return checkout

    raise FileNotFoundError(
        "FlashInfer headers were not found. Checked:\n"
        f"  - {jit_env.FLASHINFER_INCLUDE_DIR}\n"
        f"  - {checkout}"
    )


@functools.cache
def gen_vibecuda_softmax_module() -> JitSpec:
    """Generate the JIT spec for the vibecuda cluster-softmax backend.

    The tuned register/pipe cluster paths are SM100-class only (thread
    clusters over 8+ CTAs, DSM pair pools, packed f32x2 exp math), so this
    module restricts compilation to SM 10.x; restricting through the
    compilation context makes an explicit selection on an unsupported
    architecture fail loudly at build time.
    """
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10]  # SM100 family (incl. SM103 / B300)
    )
    csrc_dir = _get_csrc_dir()
    return gen_jit_spec(
        "vibecuda_softmax",
        [
            csrc_dir / "vibecuda_softmax.cu",
            csrc_dir / "flashinfer_vibecuda_softmax_binding.cu",
        ],
        extra_cuda_cflags=nvcc_flags,
        extra_include_paths=[csrc_dir, _get_include_dir()],
    )


__all__ = [
    "gen_vibecuda_softmax_module",
]
