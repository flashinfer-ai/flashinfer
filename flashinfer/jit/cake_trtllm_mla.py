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
from .core import JitSpec, gen_jit_spec, logger, sm103a_nvcc_flags

_CAKE_TRTLLM_MLA_MODULE_IDENT = "b3c5c7464f"
_CAKE_TRTLLM_MLA_SOURCE = "cake_trtllm_mla_bf16_low_batch_single_launch.cu"
_CAKE_TRTLLM_MLA_BINDING = "cake_trtllm_mla_bf16_low_batch_binding.cu"


def _get_cake_trtllm_mla_csrc_dir() -> Path:
    """Locate the exported Cake sources in installs and source checkouts."""

    installed = jit_env.FLASHINFER_CSRC_DIR / "mla"
    if (installed / _CAKE_TRTLLM_MLA_SOURCE).exists():
        return installed

    checkout = Path(__file__).resolve().parents[2] / "csrc" / "mla"
    if (checkout / _CAKE_TRTLLM_MLA_SOURCE).exists():
        return checkout

    raise FileNotFoundError(
        "Cake TRT-LLM MLA sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _get_cake_trtllm_mla_include_dir() -> Path:
    """Locate FlashInfer headers in installs and source checkouts."""

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


def get_cake_trtllm_mla_uri() -> str:
    """Return the source-keyed exact-SM103a JIT module name."""

    return f"cake_trtllm_mla_bf16_low_batch_{_CAKE_TRTLLM_MLA_MODULE_IDENT}_sm103a"


@functools.cache
def gen_cake_trtllm_mla_module() -> JitSpec:
    """Generate the exact-SM103a source-level Cake MLA module."""

    csrc_dir = _get_cake_trtllm_mla_csrc_dir()
    include_dir = _get_cake_trtllm_mla_include_dir()
    binding = csrc_dir / _CAKE_TRTLLM_MLA_BINDING
    if not binding.exists():
        raise FileNotFoundError(f"Cake TRT-LLM MLA binding source not found: {binding}")

    spec = gen_jit_spec(
        name=get_cake_trtllm_mla_uri(),
        sources=[binding],
        extra_cuda_cflags=[*sm103a_nvcc_flags],
        extra_include_paths=[csrc_dir, csrc_dir.parent, include_dir],
    )
    logger.info("Generated Cake TRT-LLM MLA SM103a JIT spec: %s", spec.name)
    return spec


@functools.cache
def load_cake_trtllm_mla_module():
    """Build or load the exact-SM103a Cake MLA module."""

    module = gen_cake_trtllm_mla_module().build_and_load()
    logger.info("Loaded Cake TRT-LLM MLA SM103a module")
    return module


def get_cake_trtllm_mla_module():
    """Return the module used by the explicit Cake backend."""

    return load_cake_trtllm_mla_module()


__all__ = [
    "gen_cake_trtllm_mla_module",
    "get_cake_trtllm_mla_module",
    "get_cake_trtllm_mla_uri",
    "load_cake_trtllm_mla_module",
]
