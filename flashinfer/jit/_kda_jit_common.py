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

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from . import env as jit_env
from .core import (
    JitSpec,
    gen_jit_spec,
    sm100a_nvcc_flags,
    sm100f_nvcc_flags,
    sm103a_nvcc_flags,
)

KDAJITTarget = Literal["sm100a", "sm100f", "sm103a"]

_NVCC_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm100f": sm100f_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}


def get_kda_csrc_dir() -> Path:
    """Locate KDA CUDA sources in installed and source checkouts."""

    installed = jit_env.FLASHINFER_CSRC_DIR / "kda"
    if installed.exists():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "kda"
    if checkout.exists():
        return checkout
    raise FileNotFoundError(
        f"KDA CUDA sources were not found. Checked:\n  - {installed}\n  - {checkout}"
    )


def get_flashinfer_include_dir() -> Path:
    """Locate FlashInfer headers in installed and source checkouts."""

    if jit_env.FLASHINFER_INCLUDE_DIR.exists():
        return jit_env.FLASHINFER_INCLUDE_DIR
    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.exists():
        return checkout
    raise FileNotFoundError(
        "FlashInfer headers were not found. Checked:\n"
        f"  - {jit_env.FLASHINFER_INCLUDE_DIR}\n  - {checkout}"
    )


def render_kda_decode_binding(
    defines: Sequence[tuple[str, str | int]], binding_header: str
) -> str:
    """Render the common frozen-decode binding translation unit."""

    define_lines = "".join(f"#define {name} {value}\n" for name, value in defines)
    return f"""\
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

{define_lines}

#include "{binding_header}"
"""


def gen_kda_jit_spec(
    *,
    name: str,
    sources: Sequence[Path],
    target: KDAJITTarget,
    target_define: str,
    csrc_dir: Path,
    include_dir: Path,
    extra_cuda_cflags: Sequence[str] = (),
) -> JitSpec:
    """Build the shared KDA JIT source, flag, and include closure."""

    return gen_jit_spec(
        name=name,
        sources=list(sources),
        extra_cuda_cflags=[
            *_NVCC_FLAGS[target],
            target_define,
            *extra_cuda_cflags,
        ],
        extra_include_paths=[csrc_dir, csrc_dir.parent, include_dir],
    )
