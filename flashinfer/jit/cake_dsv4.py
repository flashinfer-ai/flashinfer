"""JIT loader for CAKE-generated DeepSeek V4 sparse MLA kernels."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, logger, sm103a_nvcc_flags


CakeDSV4Module = Literal["pointer", "pointer_uumn", "grid_constant"]

_CAKE_DSV4_VARIANTS = {
    "pointer": (
        "bf16_h8_h32",
        "bf16_h8_h32_reduce",
        "bf16_h64_compressed",
        "bf16_h64_compressed_reduce",
        "bf16_h64_fixed_q",
        "bf16_h64_fixed_q_reduce",
        "bf16_h128_swa128",
        "bf16_h128_topk128x",
        "bf16_h128_topk128x_reduce",
        "bf16_swa128_single_cta",
        "fp8_lowhead_decode",
        "fp8_lowhead_prefill",
    ),
    "pointer_uumn": ("bf16_h64_prefill",),
    "grid_constant": (
        "bf16_h128_prefill",
        "bf16_h128_topk4x",
        "fp8_h128",
        "split_reduce",
    ),
}


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "cake_dsv4"
    if installed.exists():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "cake_dsv4"
    if checkout.exists():
        return checkout
    raise FileNotFoundError(
        "CAKE DSv4 CUDA sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _get_include_dir() -> Path:
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
def gen_cake_dsv4_module(module: CakeDSV4Module) -> JitSpec:
    if module not in _CAKE_DSV4_VARIANTS:
        raise ValueError(f"unsupported CAKE DSv4 module: {module}")
    csrc_dir = _get_csrc_dir()
    sources = [
        csrc_dir / f"cake_dsv4_{variant}_binding.cu"
        for variant in _CAKE_DSV4_VARIANTS[module]
    ]
    missing = [path for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "CAKE DSv4 binding sources were not found: "
            + ", ".join(str(path) for path in missing)
        )
    flags = list(sm103a_nvcc_flags)
    flags.append("--use_fast_math")
    if module in ("pointer_uumn", "grid_constant"):
        flags.append("-Xptxas=-uumn")
    spec = gen_jit_spec(
        name=f"cake_dsv4_{module}_sm103a",
        sources=sources,
        extra_cuda_cflags=flags,
        extra_include_paths=[csrc_dir, csrc_dir.parent, _get_include_dir()],
    )
    logger.info(f"Generated CAKE DSv4 {module} JIT spec: {spec.name}")
    return spec


@functools.cache
def get_cake_dsv4_module(module: CakeDSV4Module):
    loaded = gen_cake_dsv4_module(module).build_and_load()
    logger.info(f"Loaded CAKE DSv4 {module} module")
    return loaded


__all__ = ["CakeDSV4Module", "gen_cake_dsv4_module", "get_cake_dsv4_module"]
