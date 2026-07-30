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
from .core import JitSpec, gen_jit_spec, sm100a_nvcc_flags

_VARIANTS = ("stage8", "stage8_pdl", "stage4", "stage4_pdl")


def _get_tinygemm2_sm100_csrc_dir() -> Path:
    """Locate the frozen tinygemm2_sm100 sources in installed and source checkouts."""

    installed = jit_env.FLASHINFER_CSRC_DIR / "tinygemm2_sm100"
    if installed.exists():
        return installed

    checkout = Path(__file__).resolve().parents[2] / "csrc" / "tinygemm2_sm100"
    if checkout.exists():
        return checkout

    raise FileNotFoundError(
        "tinygemm2_sm100 CUDA sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def gen_tinygemm2_sm100_module() -> JitSpec:
    """Generate the JIT spec for the SM100/SM103 generated tinygemm2 variants.

    The four checked-in frozen device TUs (deep/shallow pipeline ring x PDL
    on/off) are generated Loom schedules that exactly port the TensorRT-LLM
    tinygemm2 kernel; each is compiled in its own binding translation unit
    because the generated sources intentionally retain their helper names and
    macros. All four link into one module.
    """

    csrc_dir = _get_tinygemm2_sm100_csrc_dir()
    sources = []
    for variant in _VARIANTS:
        binding = csrc_dir / f"tinygemm2_sm100_{variant}_binding.cu"
        if not binding.exists():
            raise FileNotFoundError(
                f"tinygemm2_sm100 binding source not found: {binding}"
            )
        sources.append(binding)

    return gen_jit_spec(
        "tinygemm2_sm100",
        sources,
        extra_cuda_cflags=sm100a_nvcc_flags
        + ["-gencode=arch=compute_103a,code=sm_103a"],
        extra_include_paths=[
            csrc_dir,
            csrc_dir.parent,
        ],
    )
